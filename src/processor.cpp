#include "namturbo/processor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "a2_dispatch.h"
#include "a2_engine.h"
#include "model_impl.h"
#include "monitor.h"
#include "namturbo/cpu.h"
#include "namturbo/error.h"
#include "resampler.h"

namespace namturbo
{
namespace
{
constexpr std::size_t kHistoryCapacity = 32768;  // samples of engine input kept (model rate); a power of two
constexpr std::uint64_t kCatchUpTarget = 512;    // prepare_swap() stops replaying when fewer samples than this have arrived since
constexpr std::uint64_t kCommitMaxGap = 1024;    // commit_swap() refuses a ticket whose engine is further behind than this ...
constexpr std::uint64_t kAdoptMaxGap = 4096;     // ... and the audio thread drops one that is further behind than this
constexpr int kRetireCapacity = 8;               // engines that can wait to be freed by service()
constexpr int kReplayExtra = 64;                 // samples replayed beyond the receptive field

// An engine together with what keeps it valid: the model whose weights it uses (the engine copies them, but a reference keeps the model
// alive for as long as the processor reports it) and, for a swap, how far it has been brought up to date.
struct Slot
{
  std::shared_ptr<const Model> model;
  Variant variant = Variant::Full;
  std::unique_ptr<A2Engine> engine;
  std::uint64_t history_end = 0; // number of model-rate samples of input the engine has been given
  std::uint64_t generation = 0;  // of the processor's prepare() it was made for
  const void* owner = nullptr;   // the processor it was made for
};
} // namespace

struct SwapTicket::Impl
{
  std::unique_ptr<Slot> slot;
};

SwapTicket::SwapTicket() = default;
SwapTicket::SwapTicket(SwapTicket&&) noexcept = default;
SwapTicket& SwapTicket::operator=(SwapTicket&&) noexcept = default;
SwapTicket::~SwapTicket() = default;
SwapTicket::operator bool() const noexcept
{
  return impl_ && impl_->slot;
}

struct Processor::Impl
{
  std::shared_ptr<const Model> model; // the model the processor was constructed with
  // The engine that makes the sound; null before prepare(). After prepare() only the audio thread touches this pointer: it replaces it when a
  // swap is taken. Control threads must not read it (it is momentarily null during the replacement); they use `prepared`.
  std::unique_ptr<Slot> active;
  std::atomic<bool> prepared{false};  // true once prepare() has succeeded; read by control threads instead of `active`
  Slot* incoming = nullptr;           // the engine being faded in (audio thread)
  std::atomic<Slot*> pending{nullptr}; // a committed engine that the audio thread has not taken yet
  std::atomic<bool> fading{false};

  // Engines taken out of use wait here for service() to free them: one writer (the audio thread), one reader (service()).
  Slot* retired[kRetireCapacity] = {};
  std::atomic<std::uint32_t> retire_head{0}, retire_tail{0};

  std::unique_ptr<RateConverter> converter; // null if the host rate equals the model rate
  int max_block = 0;                        // largest block at the host rate
  int engine_block = 0;                     // largest block given to the engine at once (at the model rate)
  int model_hz = 0;
  int latency = 0;
  std::atomic<int> variant{static_cast<int>(Variant::Full)};
  std::atomic<std::uint64_t> generation{0};
  std::atomic<std::uint64_t> completed{0}, dropped{0};

  std::vector<float> in_f, out_f;    // scratch for the double version
  std::vector<float> model_buf;      // the signal at the model rate (with a converter)
  std::vector<float> scratch_in, scratch_out; // engine_block each: for the second engine during a fade and for replays

  // The engine input of the last kHistoryCapacity samples, as a ring. The audio thread writes it while a control thread may read it (a
  // reader checks afterwards that the part it read was not overwritten), so the elements are relaxed atomics: plain reads of memory that is
  // being written would be a data race. On x86 a relaxed atomic float load or store is an ordinary move.
  std::vector<std::atomic<float>> history;
  std::atomic<std::uint64_t> history_pos{0}; // number of samples ever written; one release store per block

  std::vector<float> fade_table;
  int fade_len = 0, fade_pos = 0;

  Monitor monitor;
  bool monitoring = false;
  double ns_per_sample = 0.0;        // duration of one host sample

  // adaptive mode: the audio thread keeps the smoothed load, service() on the control thread decides
  std::atomic<bool> adaptive_on{false};
  std::atomic<double> load_ema{0.0};
  double ema_tau_ns = 300e6;
  AdaptiveConfig adaptive;                    // control thread from here on
  std::shared_ptr<const Model> control_model; // the model most recently committed (or the one the processor was prepared with)
  double cost_ratio = 0.0;                    // cost of A2-Full per sample divided by that of A2-Lite, measured; 0 = unknown
  bool policy_downgraded = false;             // the adaptive mode, not the program, chose the current variant
  std::optional<std::chrono::steady_clock::time_point> above_since, below_since;

  ~Impl() { discard_swaps(); }

  SwapTicket make_ticket(std::shared_ptr<const Model> model, Variant variant);
  bool commit_ticket(SwapTicket&& ticket);
  void run_policy();

  // ---- control thread ---------------------------------------------------------------------------------------------------------

  // Frees everything that belongs to swaps. Not while process() runs.
  void discard_swaps()
  {
    delete pending.exchange(nullptr);
    delete incoming;
    incoming = nullptr;
    fading.store(false);
    free_retired();
  }

  void free_retired()
  {
    std::uint32_t tail = retire_tail.load(std::memory_order_relaxed);
    const std::uint32_t head = retire_head.load(std::memory_order_acquire);
    while (tail != head)
    {
      delete retired[tail % kRetireCapacity];
      ++tail;
      retire_tail.store(tail, std::memory_order_release);
    }
  }

  // Copies samples [from, to) of the history into out (to - from at most kHistoryCapacity). Returns false if the audio thread may have
  // overwritten some of them while they were being copied.
  bool copy_history(std::uint64_t from, std::uint64_t to, float* out) const
  {
    const std::size_t n = static_cast<std::size_t>(to - from);
    const std::size_t start = static_cast<std::size_t>(from & (kHistoryCapacity - 1));
    const std::size_t first = std::min(n, kHistoryCapacity - start);
    for (std::size_t i = 0; i < first; ++i)
      out[i] = history[start + i].load(std::memory_order_relaxed);
    for (std::size_t i = first; i < n; ++i)
      out[i] = history[i - first].load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire); // the reads above are complete before the position is read again
    return history_pos.load(std::memory_order_relaxed) - from <= kHistoryCapacity;
  }

  // Gives an engine `n` samples as input and throws the output away (only its state matters).
  void feed(A2Engine& engine, const float* x, std::size_t n, float* sink) const
  {
    for (std::size_t done = 0; done < n;)
    {
      const int k = static_cast<int>(std::min<std::size_t>(n - done, static_cast<std::size_t>(engine_block)));
      engine.process(x + done, sink, k);
      done += static_cast<std::size_t>(k);
    }
  }

  // ---- audio thread -----------------------------------------------------------------------------------------------------------

  bool retire_full() const
  {
    return retire_head.load(std::memory_order_relaxed) - retire_tail.load(std::memory_order_acquire) >= kRetireCapacity;
  }

  void retire(Slot* s)
  {
    const std::uint32_t head = retire_head.load(std::memory_order_relaxed);
    retired[head % kRetireCapacity] = s;
    retire_head.store(head + 1, std::memory_order_release);
  }

  void record_history(const float* x, int m)
  {
    const std::uint64_t pos = history_pos.load(std::memory_order_relaxed);
    const std::size_t start = static_cast<std::size_t>(pos & (kHistoryCapacity - 1));
    const std::size_t count = static_cast<std::size_t>(m);
    const std::size_t first = std::min(count, kHistoryCapacity - start);
    for (std::size_t i = 0; i < first; ++i)
      history[start + i].store(x[i], std::memory_order_relaxed);
    for (std::size_t i = first; i < count; ++i)
      history[i - first].store(x[i], std::memory_order_relaxed);
    history_pos.store(pos + static_cast<std::uint64_t>(m), std::memory_order_release);
  }

  // Makes `s` the active engine and puts the old one in the retire queue (there is room: the caller checked).
  void make_active(Slot* s)
  {
    Slot* old = active.release();
    active.reset(s);
    retire(old);
    variant.store(static_cast<int>(s->variant), std::memory_order_relaxed);
    completed.fetch_add(1, std::memory_order_relaxed);
  }

  // Takes the engine that commit_swap() published, if there is one and it can be taken now.
  void try_adopt()
  {
    Slot* s = pending.load(std::memory_order_acquire);
    if (!s || retire_full())
      return;
    fading.store(true, std::memory_order_relaxed); // the swap stays "busy" from the moment it leaves `pending`
    if (!pending.compare_exchange_strong(s, nullptr, std::memory_order_acq_rel))
    {
      fading.store(false, std::memory_order_relaxed);
      return; // a newer ticket was committed at this moment; it is taken at the next block
    }
    const std::uint64_t pos = history_pos.load(std::memory_order_relaxed);
    if (s->owner != this || s->generation != generation.load(std::memory_order_relaxed) || s->history_end > pos || pos - s->history_end > kAdoptMaxGap)
    {
      retire(s); // out of date: not used
      dropped.fetch_add(1, std::memory_order_relaxed);
      fading.store(false, std::memory_order_relaxed);
      return;
    }
    // bring the new engine up to the present: the samples that arrived since prepare_swap() stopped replaying
    while (s->history_end < pos)
    {
      const std::size_t k = static_cast<std::size_t>(std::min<std::uint64_t>(pos - s->history_end, static_cast<std::uint64_t>(engine_block)));
      copy_history(s->history_end, s->history_end + k, scratch_in.data());
      s->engine->process(scratch_in.data(), scratch_out.data(), static_cast<int>(k));
      s->history_end += k;
    }
    if (fade_len == 0)
    {
      make_active(s);
      fading.store(false, std::memory_order_relaxed);
      return;
    }
    incoming = s;
    fade_pos = 0;
  }

  // Runs the engine(s) on m model-rate samples. x and y may be the same buffer.
  void process_model(const float* x, float* y, int m)
  {
    for (int done = 0; done < m;)
    {
      const int k = std::min(m - done, engine_block);
      const float* xs = x + done;
      float* ys = y + done;
      if (!incoming)
        active->engine->process(xs, ys, k);
      else
      {
        std::memcpy(scratch_in.data(), xs, static_cast<std::size_t>(k) * sizeof(float)); // the first engine may overwrite xs
        active->engine->process(xs, ys, k);
        incoming->engine->process(scratch_in.data(), scratch_out.data(), k);
        for (int i = 0; i < k; ++i)
        {
          const int index = fade_pos + i;
          const float g = index < fade_len ? fade_table[static_cast<std::size_t>(index)] : 1.0f;
          ys[i] = (1.0f - g) * ys[i] + g * scratch_out[static_cast<std::size_t>(i)];
        }
        fade_pos += k;
        if (fade_pos >= fade_len)
        {
          Slot* s = incoming;
          incoming = nullptr;
          make_active(s);
          fading.store(false, std::memory_order_relaxed);
        }
      }
      done += k;
    }
  }

  // Adds a call that began at `start` and handled n host samples to the statistics.
  void record(std::chrono::steady_clock::time_point start, int n) noexcept
  {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    const double elapsed_ns = elapsed < 0 ? 0.0 : static_cast<double>(elapsed);
    const double block_ns = ns_per_sample * n;
    if (monitoring)
      monitor.record(static_cast<std::uint64_t>(elapsed_ns), static_cast<std::uint64_t>(block_ns));
    // exponential average of the load: each call counts in proportion to the time it covers
    const double alpha = 1.0 - std::exp(-block_ns / ema_tau_ns);
    const double previous = load_ema.load(std::memory_order_relaxed);
    load_ema.store(previous + alpha * (elapsed_ns / block_ns - previous), std::memory_order_relaxed);
  }

  // Processes n <= max_block samples at the host rate. `in` and `out` can be the same buffer.
  void run(const float* in, float* out, int n)
  {
    const float* x = in;
    float* y = out;
    int m = n;
    if (converter)
    {
      m = converter->to_model(in, n, model_buf.data());
      x = y = model_buf.data();
    }
    if (m > 0)
    {
      if (!incoming)
        try_adopt();
      record_history(x, m); // before the engine can overwrite x
      process_model(x, y, m);
    }
    if (converter)
      converter->to_host(model_buf.data(), m, out, n);
  }
};

Processor::Processor(std::shared_ptr<const Model> model)
  : impl_(std::make_unique<Impl>())
{
  if (!model)
    throw Error(ErrorCode::InvalidArgument, "the model is null");
  impl_->model = std::move(model);
}

Processor::~Processor() = default;

namespace
{
// An engine for the weights of `variant`, ready to run blocks of up to `engine_block` samples.
std::unique_ptr<A2Engine> make_engine(const A2Weights& weights, int engine_block)
{
  try
  {
    std::unique_ptr<A2Engine> engine = make_a2_engine(weights, Backend::Avx2);
    engine->reset(engine_block);
    return engine;
  }
  catch (const std::invalid_argument& e)
  {
    throw Error(ErrorCode::InvalidArgument, e.what());
  }
  catch (const std::runtime_error& e)
  {
    throw Error(ErrorCode::CpuUnsupported, e.what());
  }
}

// The preferred variant if the model has it, otherwise the other one.
Variant usable_variant(const Model& model, Variant wanted)
{
  if (model.has(wanted))
    return wanted;
  return wanted == Variant::Full ? Variant::Lite : Variant::Full;
}

void validate_adaptive(const AdaptiveConfig& c)
{
  if (!(c.low_load > 0.0 && c.low_load < c.high_load))
    throw Error(ErrorCode::InvalidArgument, "the adaptive thresholds must satisfy 0 < low_load < high_load");
  if (!(c.smoothing_ms > 0.0) || !(c.down_after_ms >= 0.0) || !(c.up_after_ms >= 0.0))
    throw Error(ErrorCode::InvalidArgument, "smoothing_ms must be positive and the hold times must not be negative");
}

// How much more A2-Full costs than A2-Lite per sample, measured now on this machine with throw-away engines (a few milliseconds);
// 0 if the model lacks one of the variants.
double measure_cost_ratio(const A2Weights* full_weights, const A2Weights* lite_weights)
{
  if (!full_weights || !lite_weights)
    return 0.0;
  try
  {
    auto time_engine = [](const A2Weights& weights) {
      std::unique_ptr<A2Engine> engine = make_a2_engine(weights, Backend::Avx2);
      engine->reset(64);
      std::vector<float> in(64), out(64);
      for (int i = 0; i < 64; ++i)
        in[static_cast<std::size_t>(i)] = 0.1f * std::sin(0.31f * float(i)) + 0.02f * float((i * 7919) % 13 - 6) / 6.0f;
      double best = 1e300;
      for (int trial = 0; trial < 7; ++trial)
      {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 150; ++i)
          engine->process(in.data(), out.data(), 64);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      return best;
    };
    const double full = time_engine(*full_weights), lite = time_engine(*lite_weights);
    return lite > 0.0 ? full / lite : 0.0;
  }
  catch (const std::exception&)
  {
    return 0.0;
  }
}
} // namespace

void Processor::prepare(const ProcessorConfig& config)
{
  if (!(config.sample_rate >= 1000.0 && config.sample_rate <= 2.0e6))
    throw Error(ErrorCode::InvalidArgument, "the sample rate must be between 1 kHz and 2 MHz");
  if (config.max_block < 1 || config.max_block > kMaxEngineBlock)
    throw Error(ErrorCode::InvalidArgument, "max_block must be between 1 and " + std::to_string(kMaxEngineBlock));
  if (!(config.swap_fade_ms >= 0.0 && config.swap_fade_ms <= 100.0))
    throw Error(ErrorCode::InvalidArgument, "swap_fade_ms must be between 0 and 100");
  const CpuSupport cpu = check_cpu();
  if (!cpu.supported)
    throw Error(ErrorCode::CpuUnsupported, cpu.detail);
  validate_adaptive(config.adaptive);

  const Model::Impl& m = *impl_->model->impl_;
  const double host_hz = std::round(config.sample_rate);
  if (std::fabs(config.sample_rate - host_hz) > 1e-3)
    throw Error(ErrorCode::Unsupported, "the sample rate must be a whole number of Hz");
  const double model_hz = std::round(m.info.sample_rate);
  if (std::fabs(m.info.sample_rate - model_hz) > 1e-3)
    throw Error(ErrorCode::Unsupported, "the sample rate of the model is not a whole number of Hz");

  const Variant variant = usable_variant(*impl_->model, config.variant);
  const A2Weights& weights = (variant == Variant::Full) ? *m.full : *m.lite;

  // Everything that can fail is done before the state of the processor changes: if prepare() throws, the processor is as it was.
  std::unique_ptr<RateConverter> converter;
  if (host_hz != model_hz)
    converter = std::make_unique<RateConverter>(static_cast<int>(host_hz), static_cast<int>(model_hz), config.max_block, config.resampler);
  const int engine_block = std::min(converter ? converter->max_model_block() : config.max_block, kMaxEngineBlock);

  auto slot = std::make_unique<Slot>();
  slot->model = impl_->model;
  slot->variant = variant;
  slot->engine = make_engine(weights, engine_block);

  const int fade_len = static_cast<int>(std::lround(config.swap_fade_ms * 1e-3 * model_hz));
  std::vector<float> fade_table(static_cast<std::size_t>(fade_len));
  for (int i = 0; i < fade_len; ++i)
    fade_table[static_cast<std::size_t>(i)] = static_cast<float>(0.5 * (1.0 - std::cos(3.14159265358979323846 * (i + 0.5) / fade_len)));
  std::vector<float> in_f(static_cast<std::size_t>(config.max_block)), out_f(static_cast<std::size_t>(config.max_block));
  std::vector<float> model_buf(converter ? static_cast<std::size_t>(converter->max_model_block()) : 0u);
  std::vector<float> scratch_in(static_cast<std::size_t>(engine_block)), scratch_out(static_cast<std::size_t>(engine_block));
  std::vector<std::atomic<float>> history(kHistoryCapacity);
  for (auto& h : history)
    h.store(0.0f, std::memory_order_relaxed);
  const double cost_ratio = config.adaptive.enabled ? measure_cost_ratio(m.full ? &*m.full : nullptr, m.lite ? &*m.lite : nullptr) : 0.0;

  Impl& p = *impl_;
  p.discard_swaps();
  p.generation.fetch_add(1);
  p.active = std::move(slot);
  p.converter = std::move(converter);
  p.in_f = std::move(in_f);
  p.out_f = std::move(out_f);
  p.model_buf = std::move(model_buf);
  p.scratch_in = std::move(scratch_in);
  p.scratch_out = std::move(scratch_out);
  p.history = std::move(history);
  p.history_pos.store(0);
  p.fade_table = std::move(fade_table);
  p.fade_len = fade_len;
  p.fade_pos = 0;
  p.max_block = config.max_block;
  p.engine_block = engine_block;
  p.model_hz = static_cast<int>(model_hz);
  p.latency = p.converter ? p.converter->latency_samples() : 0;
  p.variant.store(static_cast<int>(variant));
  p.completed.store(0);
  p.dropped.store(0);
  p.monitoring = config.monitoring;
  p.ns_per_sample = 1e9 / config.sample_rate;
  p.monitor.reset();
  p.adaptive_on.store(false);
  p.adaptive = config.adaptive;
  p.ema_tau_ns = config.adaptive.smoothing_ms * 1e6;
  p.load_ema.store(0.0);
  p.control_model = impl_->model;
  p.cost_ratio = cost_ratio;
  p.policy_downgraded = false;
  p.above_since.reset();
  p.below_since.reset();
  p.adaptive_on.store(config.adaptive.enabled);
  p.prepared.store(true, std::memory_order_release);
}

void Processor::reset()
{
  Impl& p = *impl_;
  if (!p.active)
    return;
  p.discard_swaps();
  p.generation.fetch_add(1); // tickets made before the reset belong to the old history
  for (auto& h : p.history)
    h.store(0.0f, std::memory_order_relaxed);
  p.history_pos.store(0);
  p.active->engine->reset(p.engine_block);
  if (p.converter)
    p.converter->reset();
}

void Processor::process(const float* in, float* out, int n) noexcept
{
  if (n <= 0)
    return;
  if (!impl_->active)
  {
    std::fill(out, out + n, 0.0f);
    return;
  }
  const bool timed = impl_->monitoring || impl_->adaptive_on.load(std::memory_order_relaxed);
  const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
  try
  {
    for (int done = 0; done < n;)
    {
      const int m = std::min(n - done, impl_->max_block);
      impl_->run(in + done, out + done, m);
      done += m;
    }
  }
  catch (...)
  {
    std::fill(out, out + n, 0.0f); // not reachable with valid arguments; keeps the function noexcept
  }
  if (timed)
    impl_->record(start, n);
}

void Processor::process(const double* in, double* out, int n) noexcept
{
  if (n <= 0)
    return;
  if (!impl_->active)
  {
    std::fill(out, out + n, 0.0);
    return;
  }
  const bool timed = impl_->monitoring || impl_->adaptive_on.load(std::memory_order_relaxed);
  const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();
  try
  {
    float* fin = impl_->in_f.data();
    float* fout = impl_->out_f.data();
    for (int done = 0; done < n;)
    {
      const int m = std::min(n - done, impl_->max_block);
      for (int i = 0; i < m; ++i)
        fin[i] = static_cast<float>(in[done + i]);
      impl_->run(fin, fout, m);
      for (int i = 0; i < m; ++i)
        out[done + i] = static_cast<double>(fout[i]);
      done += m;
    }
  }
  catch (...)
  {
    std::fill(out, out + n, 0.0);
  }
  if (timed)
    impl_->record(start, n);
}

int Processor::latency_samples() const noexcept
{
  return impl_->latency;
}

Variant Processor::variant() const noexcept
{
  return static_cast<Variant>(impl_->variant.load(std::memory_order_relaxed));
}

const ModelInfo& Processor::model_info() const noexcept
{
  return impl_->model->info();
}

SwapTicket Processor::Impl::make_ticket(std::shared_ptr<const Model> target, Variant wanted)
{
  Impl& p = *this;
  if (!p.prepared.load(std::memory_order_acquire))
    throw Error(ErrorCode::InvalidArgument, "prepare() must be called before prepare_swap()");
  if (!target)
    throw Error(ErrorCode::InvalidArgument, "the model is null");
  const double rate = target->info().sample_rate;
  if (std::lround(rate) != p.model_hz || std::fabs(rate - std::round(rate)) > 1e-3)
    throw Error(ErrorCode::Unsupported, "the new model was trained at " + std::to_string(rate) + " Hz, not at the " + std::to_string(p.model_hz) +
                                          " Hz of the running model; models of different sample rates cannot be swapped");

  const std::uint64_t made_for = p.generation.load();
  const Variant use = usable_variant(*target, wanted);
  const Model::Impl& m = *target->impl_;
  auto slot = std::make_unique<Slot>();
  slot->model = target;
  slot->variant = use;
  slot->engine = make_engine(use == Variant::Full ? *m.full : *m.lite, p.engine_block);
  slot->generation = made_for;
  slot->owner = this;

  // Replay the recent input, so that the engine gets the state of the running one. The audio thread keeps writing, so a copy is checked
  // after it was made, and made again if the audio thread has overtaken it.
  const std::uint64_t replay = static_cast<std::uint64_t>(a2_receptive_field() + kReplayExtra);
  std::vector<float> local(kHistoryCapacity);
  std::vector<float> sink(static_cast<std::size_t>(p.engine_block));
  std::uint64_t reached = 0;
  bool replayed = false;
  for (int attempt = 0; attempt < 8 && !replayed; ++attempt)
  {
    const std::uint64_t end = p.history_pos.load(std::memory_order_acquire);
    const std::uint64_t start = end > replay ? end - replay : 0;
    if (!p.copy_history(start, end, local.data()))
      continue;
    p.feed(*slot->engine, local.data(), static_cast<std::size_t>(end - start), sink.data());
    reached = end;
    replayed = true;
  }
  if (!replayed)
    throw Error(ErrorCode::Unsupported, "could not get a consistent copy of the recent input; the audio thread is writing faster than it can be read");
  for (int round = 0; round < 16; ++round)
  {
    const std::uint64_t end = p.history_pos.load(std::memory_order_acquire);
    if (end - reached <= kCatchUpTarget)
      break;
    if (end - reached > kHistoryCapacity / 2 || !p.copy_history(reached, end, local.data()))
      continue;
    p.feed(*slot->engine, local.data(), static_cast<std::size_t>(end - reached), sink.data());
    reached = end;
  }
  slot->history_end = reached;

  SwapTicket ticket;
  ticket.impl_ = std::make_unique<SwapTicket::Impl>();
  ticket.impl_->slot = std::move(slot);
  return ticket;
}

bool Processor::Impl::commit_ticket(SwapTicket&& ticket)
{
  Impl& p = *this;
  if (!ticket.impl_ || !ticket.impl_->slot)
    return false;
  Slot* s = ticket.impl_->slot.get();
  if (s->owner != this || s->generation != p.generation.load())
    return false;
  const std::uint64_t now = p.history_pos.load(std::memory_order_acquire);
  if (now < s->history_end || now - s->history_end > kCommitMaxGap)
    return false;
  std::shared_ptr<const Model> committed = s->model;
  Slot* replaced = p.pending.exchange(ticket.impl_->slot.release(), std::memory_order_acq_rel);
  delete replaced; // a ticket that the audio thread never took
  control_model = std::move(committed);
  policy_downgraded = false; // a commit by the program; the policy sets this again for its own
  return true;
}


void Processor::Impl::run_policy()
{
  if (!adaptive_on.load() || !prepared.load(std::memory_order_acquire) || !control_model)
    return;
  if (!control_model->has(Variant::Lite) || !control_model->has(Variant::Full) || cost_ratio <= 0.0)
    return;
  if (pending.load(std::memory_order_acquire) || fading.load(std::memory_order_relaxed))
  {
    above_since.reset(); // the load of the blocks of a fade says nothing about either variant
    below_since.reset();
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const double load = load_ema.load(std::memory_order_relaxed);
  const Variant current = static_cast<Variant>(variant.load(std::memory_order_relaxed));
  // true if `condition` has held for at least `ms`
  auto held = [&](std::optional<std::chrono::steady_clock::time_point>& since, bool condition, double ms) {
    if (!condition)
    {
      since.reset();
      return false;
    }
    if (!since)
      since = now;
    return std::chrono::duration<double, std::milli>(now - *since).count() >= ms;
  };
  Variant target;
  if (current == Variant::Full)
  {
    below_since.reset();
    if (!held(above_since, load > adaptive.high_load, adaptive.down_after_ms))
      return;
    target = Variant::Lite;
  }
  else if (policy_downgraded)
  {
    above_since.reset();
    // the load that A2-Full would cause: the load now, scaled by the measured cost ratio (the share of the conversion, if there is one,
    // is not scaled by it in reality, so this overestimates and the switch up comes a little late rather than too early)
    if (!held(below_since, load * cost_ratio < adaptive.low_load, adaptive.up_after_ms))
      return;
    target = Variant::Full;
  }
  else
    return; // the program chose A2-Lite: it stays
  SwapTicket ticket = make_ticket(control_model, target);
  if (!commit_ticket(std::move(ticket)))
    return; // out of date: the next service() tries again
  policy_downgraded = (target == Variant::Lite);
  above_since.reset();
  below_since.reset();
  if (adaptive.on_switch)
    adaptive.on_switch(target);
}

SwapTicket Processor::prepare_swap(std::shared_ptr<const Model> model, Variant variant)
{
  return impl_->make_ticket(std::move(model), variant);
}

bool Processor::commit_swap(SwapTicket&& ticket)
{
  return impl_->commit_ticket(std::move(ticket));
}

void Processor::service()
{
  impl_->free_retired();
  try
  {
    impl_->run_policy();
  }
  catch (const Error&)
  {
    // a switch that could not be prepared is tried again at the next call
  }
}

void Processor::set_adaptive(const AdaptiveConfig& config)
{
  Impl& p = *impl_;
  if (!p.prepared.load(std::memory_order_acquire))
    throw Error(ErrorCode::InvalidArgument, "prepare() must be called before set_adaptive()");
  validate_adaptive(config);
  if (config.enabled)
  {
    const Model::Impl& m = *p.control_model->impl_;
    p.cost_ratio = measure_cost_ratio(m.full ? &*m.full : nullptr, m.lite ? &*m.lite : nullptr);
  }
  p.adaptive = config;
  p.ema_tau_ns = config.smoothing_ms * 1e6;
  p.above_since.reset();
  p.below_since.reset();
  p.adaptive_on.store(config.enabled);
}

double Processor::smoothed_load() const noexcept
{
  return impl_->load_ema.load(std::memory_order_relaxed);
}

bool Processor::swap_busy() const noexcept
{
  return impl_->pending.load(std::memory_order_acquire) != nullptr || impl_->fading.load(std::memory_order_relaxed);
}

std::uint64_t Processor::swaps_completed() const noexcept
{
  return impl_->completed.load(std::memory_order_relaxed);
}

std::uint64_t Processor::swaps_dropped() const noexcept
{
  return impl_->dropped.load(std::memory_order_relaxed);
}

PerfStats Processor::stats() const noexcept
{
  return impl_->monitor.snapshot();
}

void Processor::reset_stats() noexcept
{
  impl_->monitor.reset();
}
} // namespace namturbo
