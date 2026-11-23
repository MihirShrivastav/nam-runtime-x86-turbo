// Real-time safety test: process() must not allocate or free memory, whatever it is doing.
// Usage: test_realtime <slimmable A2 model.nam>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing, or the CPU lacks AVX2 and FMA).
//
// The global operator new and operator delete are replaced by versions that count the calls made by a thread while it is "armed"; the
// aligned buffers of the engines, which do not go through operator new, are counted by engine_util::aligned_buffer_events(). Every
// case arms the thread around the audio calls only. The test also checks itself: a deliberate allocation inside an armed region
// must be seen. (Locks and system calls cannot be detected this way; the design avoids them and the code review checks it.)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "engine_util.h"
#include "namturbo/namturbo.h"

using namespace namturbo;

namespace
{
thread_local bool t_armed = false;
thread_local unsigned long long t_calls = 0; // calls of operator new or delete made while armed

void* counted_alloc(std::size_t n, std::size_t align)
{
  if (t_armed)
    ++t_calls;
  if (n == 0)
    n = 1;
  void* p = nullptr;
#if defined(_MSC_VER)
  p = align > alignof(std::max_align_t) ? _aligned_malloc(n, align) : std::malloc(n);
#else
  if (align > alignof(std::max_align_t))
  {
    if (posix_memalign(&p, align, n) != 0)
      p = nullptr;
  }
  else
    p = std::malloc(n);
#endif
  if (!p)
    throw std::bad_alloc();
  return p;
}

void counted_free(void* p, std::size_t align) noexcept
{
  if (!p)
    return;
  if (t_armed)
    ++t_calls;
#if defined(_MSC_VER)
  if (align > alignof(std::max_align_t))
    _aligned_free(p);
  else
    std::free(p);
#else
  (void)align;
  std::free(p);
#endif
}
} // namespace

void* operator new(std::size_t n) { return counted_alloc(n, 0); }
void* operator new[](std::size_t n) { return counted_alloc(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) { return counted_alloc(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return counted_alloc(n, static_cast<std::size_t>(a)); }
void operator delete(void* p) noexcept { counted_free(p, 0); }
void operator delete[](void* p) noexcept { counted_free(p, 0); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p, 0); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p, 0); }
void operator delete(void* p, std::align_val_t a) noexcept { counted_free(p, static_cast<std::size_t>(a)); }
void operator delete[](void* p, std::align_val_t a) noexcept { counted_free(p, static_cast<std::size_t>(a)); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { counted_free(p, static_cast<std::size_t>(a)); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { counted_free(p, static_cast<std::size_t>(a)); }

namespace
{
int g_failures = 0;
#define CHECK(cond)                                                         \
  do                                                                        \
  {                                                                         \
    if (!(cond))                                                            \
    {                                                                       \
      std::printf("  CHECK FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

// Counts the allocation events of the code run while it exists.
class Armed
{
public:
  Armed() : calls_(t_calls), aligned_(engine_util::aligned_buffer_events())
  {
    t_armed = true;
  }
  ~Armed() { t_armed = false; }
  unsigned long long events() const { return (t_calls - calls_) + (engine_util::aligned_buffer_events() - aligned_); }

private:
  unsigned long long calls_, aligned_;
};

std::vector<float> make_signal(size_t n)
{
  std::vector<float> x(n);
  unsigned s = 12345;
  for (size_t i = 0; i < n; ++i)
  {
    s = s * 1664525u + 1013904223u;
    x[i] = 0.3f * (float(s >> 8) / float(1 << 24) - 0.5f);
  }
  return x;
}

// Runs `blocks` calls of the sizes in `sizes` (cycling) through the float and then the double interface.
unsigned long long run_audio(Processor& p, const std::vector<int>& sizes, int blocks)
{
  const int largest = *std::max_element(sizes.begin(), sizes.end());
  std::vector<float> fin = make_signal(size_t(largest)), fout(static_cast<size_t>(largest));
  std::vector<double> din(fin.begin(), fin.end()), dout(static_cast<size_t>(largest));
  Armed armed;
  for (int b = 0; b < blocks; ++b)
  {
    const int n = sizes[size_t(b) % sizes.size()];
    if (b % 2 == 0)
      p.process(fin.data(), fout.data(), n);
    else
      p.process(din.data(), dout.data(), n);
  }
  return armed.events();
}

void plain_processing(const std::shared_ptr<const Model>& model)
{
  std::printf("plain_processing\n");
  for (Variant v : {Variant::Full, Variant::Lite})
    for (double rate : {48000.0, 44100.0, 96000.0})
      for (Oversampling os : {Oversampling::Off, Oversampling::X2, Oversampling::X4})
      {
        if (os != Oversampling::Off && rate == 96000.0 && v == Variant::Full)
          continue; // covered by the other combinations
        Processor p(model);
        ProcessorConfig c;
        c.sample_rate = rate;
        c.max_block = 256;
        c.variant = v;
        c.oversampling = os;
        c.monitoring = true;
        p.prepare(c);
        const unsigned long long e = run_audio(p, {1, 7, 64, 256, 0, 33, 1000, 5000}, 160); // 1000 and 5000 exceed max_block
        std::printf("  %s %dx at %.0f Hz: %llu allocation events\n", v == Variant::Full ? "A2-Full" : "A2-Lite", int(os), rate, e);
        CHECK(e == 0);
      }
}

void swaps_and_adaptive_mode(const std::shared_ptr<const Model>& model)
{
  std::printf("swaps_and_adaptive_mode\n");
  for (double rate : {48000.0, 44100.0})
    for (Oversampling os : {Oversampling::Off, Oversampling::X2})
    {
      Processor p(model);
      ProcessorConfig c;
      c.sample_rate = rate;
      c.max_block = 128;
      c.oversampling = os;
      c.monitoring = true;
      c.swap_fade_ms = 5.0;
      c.adaptive.enabled = true;
      c.adaptive.high_load = 1e-6; // the policy switches down at once and then stays
      c.adaptive.low_load = 1e-9;
      c.adaptive.down_after_ms = 0.0;
      c.adaptive.up_after_ms = 0.0;
      c.adaptive.smoothing_ms = 20.0;
      p.prepare(c);
      unsigned long long events = 0;
      int taken = 0;
      for (int round = 0; round < 12; ++round)
      {
        events += run_audio(p, {64, 128}, 40); // the adaptive policy runs in service(), outside the armed region
        p.service();
        const Variant want = (round % 2 == 0) ? Variant::Lite : Variant::Full;
        SwapTicket t = p.prepare_swap(model, want);
        if (p.commit_swap(std::move(t)))
        {
          // The audio thread takes the ticket, runs the catch-up, fades and retires the old engine: all armed.
          events += run_audio(p, {16, 64, 128}, 60);
          ++taken;
        }
        p.service();
      }
      std::printf("  %.0f Hz %dx: %d swaps taken in 12 rounds, %llu swaps completed, %llu allocation events on the audio thread\n", rate, int(os),
                  taken, static_cast<unsigned long long>(p.swaps_completed()), events);
      CHECK(taken >= 8);
      CHECK(p.swaps_completed() >= 6);
      CHECK(events == 0);
    }
}

void the_test_sees_an_allocation()
{
  std::printf("the_test_sees_an_allocation\n");
  {
    Armed armed;
    static int* volatile escape; // a volatile pointer keeps the compiler from removing the pair of calls
    escape = new int[16];
    delete[] escape;
    CHECK(armed.events() == 2);
  }
  {
    Armed armed;
    engine_util::AlignedFloats f;
    f.assign_zero(64);
    CHECK(armed.events() >= 1);
  }
  {
    Armed armed;
    CHECK(armed.events() == 0); // and nothing is counted when nothing happens
  }
}
} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2 || !std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found\n");
    return 77;
  }
  if (!check_cpu().supported)
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2 and FMA\n");
    return 77;
  }
  try
  {
    auto model = Model::load(argv[1]);
    the_test_sees_an_allocation();
    plain_processing(model);
    swaps_and_adaptive_mode(model);
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION: %s\n", e.what());
    ++g_failures;
  }
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
