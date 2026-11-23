// Tests of the public API (include/namturbo). Usage: test_api <slimmable A2 model.nam> <data_dir>
// The data directory holds synth.raw and sweep10s.raw (written by the make_golden test).
// Exit codes: 0 pass, 1 failure, 77 skipped (model or data missing, or the CPU lacks AVX2 and FMA).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <string>
#include <vector>

#include "a2_dispatch.h"
#include "a2_weights.h"
#include "common.h"
#include "json.hpp"
#include "namturbo/namturbo.h"
#include "resampler.h"

using namespace namturbo;
using json = nlohmann::json;

static int g_failures = 0;
#define CHECK(cond)                                                         \
  do                                                                        \
  {                                                                         \
    if (!(cond))                                                            \
    {                                                                       \
      std::printf("  CHECK FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

static std::string g_model, g_data;
static bool exists(const std::string& p) { return std::ifstream(p).good(); }

static bool bit_equal(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The error code that f() throws, or nullopt if it does not throw an namturbo::Error.
template <class F>
static std::optional<ErrorCode> error_of(F&& f)
{
  try
  {
    f();
  }
  catch (const Error& e)
  {
    return e.code();
  }
  catch (...)
  {
    return std::nullopt;
  }
  return std::nullopt;
}

static std::string dump(const json& j) { return j.dump(); }

static std::vector<float> test_signal()
{
  std::vector<float> x = read_raw(g_data + "/synth.raw");
  const std::vector<float> s = read_raw(g_data + "/sweep10s.raw");
  x.insert(x.end(), s.begin(), s.begin() + 48000); // 1 s of the complex signal (the quiet noise ramp)
  return x;
}

// What the engines, used directly, produce for the whole signal in blocks of 64 after reset(512): the reference of these tests.
static std::vector<float> engine_output(double slim, const std::vector<float>& in)
{
  auto engine = make_a2_engine(load_a2_weights(g_model, slim), Backend::Avx2);
  engine->reset(512);
  std::vector<float> out(in.size());
  for (size_t pos = 0; pos < in.size(); pos += 64)
    engine->process(in.data() + pos, out.data() + pos, int(std::min<size_t>(64, in.size() - pos)));
  return out;
}

// A processor cannot be copied or moved, so it is constructed where it is used and then prepared with this.
static void prepare_to(Processor& p, Variant variant, int max_block)
{
  ProcessorConfig config;
  config.max_block = max_block;
  config.variant = variant;
  p.prepare(config);
}

static void model_info_and_variants()
{
  std::printf("model_info_and_variants\n");
  auto model = Model::load(g_model);
  const ModelInfo& info = model->info();
  CHECK(info.has_lite && info.has_full);
  CHECK(model->has(Variant::Lite) && model->has(Variant::Full));
  CHECK(info.sample_rate == 48000.0 && info.sample_rate_stated);

  // The expected metadata, read here from the file with a different code path than the library's.
  std::ifstream is(g_model);
  const json file = json::parse(is);
  const json& md = file.at("metadata");
  CHECK(info.loudness_db.has_value() == md.contains("loudness"));
  if (info.loudness_db && md.contains("loudness"))
    CHECK(*info.loudness_db == md["loudness"].get<double>());
  CHECK(info.input_level_dbu.has_value() == md.contains("input_level_dbu"));
  if (info.input_level_dbu && md.contains("input_level_dbu"))
    CHECK(*info.input_level_dbu == md["input_level_dbu"].get<double>());
  CHECK(info.output_level_dbu.has_value() == md.contains("output_level_dbu"));
  std::printf("  sample rate %.0f Hz, loudness %.2f dB, input level %.2f dBu, both variants present\n", info.sample_rate,
              info.loudness_db.value_or(0.0), info.input_level_dbu.value_or(0.0));

  // The same model read from memory is the same model.
  std::ifstream raw(g_model, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(raw)), std::istreambuf_iterator<char>());
  auto from_memory = Model::load(text.data(), text.size());
  CHECK(from_memory->has(Variant::Lite) && from_memory->has(Variant::Full));
  CHECK(from_memory->info().sample_rate == info.sample_rate);
}

static void model_load_errors()
{
  std::printf("model_load_errors\n");
  std::ifstream is(g_model);
  const json file = json::parse(is);
  const json& sub = file.at("config").at("submodels")[0].at("model"); // the A2-Lite WaveNet as a plain file

  CHECK(error_of([&] { Model::load(g_data + "/no-such-file.nam"); }) == ErrorCode::Io);
  CHECK(error_of([&] { Model::load("not json at all", 15); }) == ErrorCode::InvalidFile);
  CHECK(error_of([&] { Model::load("", 0); }) == ErrorCode::InvalidFile);
  CHECK(error_of([&] { Model::load(nullptr, 5); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { const std::string t = "[1, 2, 3]"; Model::load(t.data(), t.size()); }) == ErrorCode::InvalidFile);
  CHECK(error_of([&] { const std::string t = "{\"a\": 1}"; Model::load(t.data(), t.size()); }) == ErrorCode::NotA2);

  { // another architecture
    json j = sub;
    j["architecture"] = "LSTM";
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::NotA2);
  }
  { // a WaveNet that is not A2: a kernel size changed
    json j = sub;
    j["config"]["layers"][0]["kernel_sizes"][3] = 7;
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::NotA2);
  }
  { // an A2 model of a size that is not run
    json j = sub;
    j["config"]["layers"][0]["channels"] = 5;
    j["config"]["layers"][0]["bottleneck"] = 5;
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::UnsupportedShape);
  }
  { // too few weights
    json j = sub;
    j["weights"].erase(j["weights"].end() - 1);
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::InvalidFile);
  }
  { // too many weights
    json j = sub;
    j["weights"].push_back(0.0);
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::InvalidFile);
  }
  { // weights that are not numbers
    json j = sub;
    j["weights"][5] = "x";
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::InvalidFile);
  }
  { // a slimmable file with one good and one unusable submodel loads the good one
    json j = file;
    j["config"]["submodels"][1]["model"]["config"]["layers"][0]["channels"] = 5;
    j["config"]["submodels"][1]["model"]["config"]["layers"][0]["bottleneck"] = 5;
    const std::string t = dump(j);
    auto m = Model::load(t.data(), t.size());
    CHECK(m->has(Variant::Lite) && !m->has(Variant::Full));
  }
  { // a slimmable file with no usable submodel reports the first problem
    json j = file;
    j["config"]["submodels"][0]["model"]["config"]["layers"][0]["kernel_sizes"][0] = 9;
    j["config"]["submodels"][1]["model"]["config"]["layers"][0]["kernel_sizes"][0] = 9;
    const std::string t = dump(j);
    CHECK(error_of([&] { Model::load(t.data(), t.size()); }) == ErrorCode::NotA2);
  }
  { // an error message is a sentence a person can read
    try
    {
      const std::string t = "{\"architecture\": \"LSTM\"}";
      Model::load(t.data(), t.size());
    }
    catch (const Error& e)
    {
      std::printf("  example message: %s\n", e.what());
      CHECK(std::string(e.what()).find("LSTM") != std::string::npos);
    }
  }
}

static void processor_matches_the_engine()
{
  std::printf("processor_matches_the_engine\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();
  for (Variant v : {Variant::Full, Variant::Lite})
  {
    Processor p(model);
    prepare_to(p, v, 512);
    CHECK(p.variant() == v);
    std::vector<float> out(in.size());
    for (size_t pos = 0; pos < in.size(); pos += 64)
      p.process(in.data() + pos, out.data() + pos, int(std::min<size_t>(64, in.size() - pos)));
    const bool same = bit_equal(out, engine_output(v == Variant::Full ? 1.0 : 0.0, in));
    std::printf("  %s: output identical to the engine used directly: %s\n", v == Variant::Full ? "A2-Full" : "A2-Lite", same ? "yes" : "NO");
    CHECK(same);
  }
}

static void block_splitting()
{
  std::printf("block_splitting\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();
  for (Variant v : {Variant::Full, Variant::Lite})
  {
    Processor a(model);
    prepare_to(a, v, 256);
    std::vector<float> whole(in.size());
    a.process(in.data(), whole.data(), int(in.size())); // far larger than max_block: processed in pieces

    Processor b(model);

    prepare_to(b, v, 256);
    std::vector<float> split(in.size());
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 700); // includes 0 and sizes above max_block
    for (size_t pos = 0; pos < in.size();)
    {
      const int n = int(std::min<size_t>(size_t(dist(rng)), in.size() - pos));
      b.process(in.data() + pos, split.data() + pos, n);
      pos += n;
    }
    const bool same = bit_equal(whole, split);
    std::printf("  %s: one call of %zu samples against random blocks of 0 to 700: %s\n", v == Variant::Full ? "A2-Full" : "A2-Lite", in.size(),
                same ? "bit-identical" : "DIFFERENT");
    CHECK(same);
  }
}

static void double_version_equals_float_version()
{
  std::printf("double_version_equals_float_version\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();
  std::vector<double> in_d(in.begin(), in.end());
  Processor pf(model);
  prepare_to(pf, Variant::Full, 300);
  Processor pd(model);
  prepare_to(pd, Variant::Full, 300);
  std::vector<float> out_f(in.size());
  std::vector<double> out_d(in.size());
  pf.process(in.data(), out_f.data(), int(in.size()));
  pd.process(in_d.data(), out_d.data(), int(in.size()));
  bool same = true;
  for (size_t i = 0; i < in.size(); ++i)
    same = same && out_d[i] == double(out_f[i]);
  std::printf("  double output equals float output: %s\n", same ? "yes" : "NO");
  CHECK(same);

  // in place, double
  std::vector<double> inplace = in_d;
  Processor pi(model);
  prepare_to(pi, Variant::Full, 300);
  pi.process(inplace.data(), inplace.data(), int(inplace.size()));
  CHECK(inplace == out_d);
}

static void edge_cases()
{
  std::printf("edge_cases\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();

  { // before prepare(): zeros; n == 0 touches nothing
    Processor p(model);
    std::vector<float> out(100, 7.0f);
    p.process(in.data(), out.data(), 0);
    CHECK(std::all_of(out.begin(), out.end(), [](float v) { return v == 7.0f; }));
    p.process(in.data(), out.data(), 100);
    CHECK(std::all_of(out.begin(), out.end(), [](float v) { return v == 0.0f; }));
    std::vector<double> outd(100, 7.0);
    std::vector<double> ind(100, 0.5);
    p.process(ind.data(), outd.data(), 100);
    CHECK(std::all_of(outd.begin(), outd.end(), [](double v) { return v == 0.0; }));
    CHECK(p.latency_samples() == 0);
  }
  { // in place (float) gives the same result as separate buffers; negative n is ignored
    Processor a(model);
    prepare_to(a, Variant::Full, 512);
    Processor b(model);
    prepare_to(b, Variant::Full, 512);
    std::vector<float> separate(in.size()), inplace = in;
    a.process(in.data(), separate.data(), int(in.size()));
    b.process(inplace.data(), inplace.data(), int(inplace.size()));
    b.process(in.data(), inplace.data(), -5);
    CHECK(bit_equal(separate, inplace));
  }
  { // reset() restores the start state: the same input gives the same output again
    Processor p(model);
    prepare_to(p, Variant::Full, 512);
    std::vector<float> first(in.size()), second(in.size());
    p.process(in.data(), first.data(), int(in.size()));
    p.reset();
    p.process(in.data(), second.data(), int(in.size()));
    CHECK(bit_equal(first, second));
  }
  { // two processors share one model and keep separate state
    Processor a(model);
    prepare_to(a, Variant::Full, 512);
    Processor b(model);
    prepare_to(b, Variant::Full, 512);
    std::vector<float> in2(in.size());
    for (size_t i = 0; i < in.size(); ++i)
      in2[i] = in[in.size() - 1 - i];
    std::vector<float> oa(in.size()), ob(in.size()), ra(in.size()), rb(in.size());
    for (size_t pos = 0; pos < in.size(); pos += 100) // interleaved
    {
      const int n = int(std::min<size_t>(100, in.size() - pos));
      a.process(in.data() + pos, oa.data() + pos, n);
      b.process(in2.data() + pos, ob.data() + pos, n);
    }
    Processor c(model);
    prepare_to(c, Variant::Full, 512);
    Processor d(model);
    prepare_to(d, Variant::Full, 512);
    c.process(in.data(), ra.data(), int(in.size()));
    d.process(in2.data(), rb.data(), int(in.size()));
    CHECK(bit_equal(oa, ra) && bit_equal(ob, rb));
  }
  { // a model with one variant: the processor falls back to it
    std::ifstream is(g_model);
    const json file = json::parse(is);
    const std::string t = dump(file.at("config").at("submodels")[0].at("model")); // A2-Lite alone
    auto lite_only = Model::load(t.data(), t.size());
    CHECK(lite_only->has(Variant::Lite) && !lite_only->has(Variant::Full));
    Processor p(lite_only);
    prepare_to(p, Variant::Full, 128);
    CHECK(p.variant() == Variant::Lite);
    std::vector<float> out(in.size());
    p.process(in.data(), out.data(), int(in.size()));
    CHECK(bit_equal(out, engine_output(0.0, in)));
  }
}

static void prepare_errors_and_guarantee()
{
  std::printf("prepare_errors_and_guarantee\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();

  CHECK(error_of([&] { Processor p(nullptr); }) == ErrorCode::InvalidArgument);
  Processor p(model);
  auto prepare = [&](double rate, int block) {
    ProcessorConfig c;
    c.sample_rate = rate;
    c.max_block = block;
    p.prepare(c);
  };
  CHECK(error_of([&] { prepare(48000.0, 0); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(48000.0, -3); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(48000.0, 65537); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(0.0, 512); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(std::nan(""), 512); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(500.0, 512); }) == ErrorCode::InvalidArgument);   // outside 1 kHz to 2 MHz
  CHECK(error_of([&] { prepare(3.0e6, 512); }) == ErrorCode::InvalidArgument);
  CHECK(error_of([&] { prepare(44100.5, 512); }) == ErrorCode::Unsupported);      // not a whole number of Hz
  CHECK(error_of([&] { prepare(44056.0, 512); }) == ErrorCode::Unsupported);      // the ratio to 48000 Hz is 6000 : 5507
  CHECK(error_of([&] { prepare(44100.0, 512); }) == std::nullopt);                // converted
  CHECK(error_of([&] { prepare(96000.0, 512); }) == std::nullopt);
  CHECK(error_of([&] { prepare(48000.0, 65536); }) == std::nullopt);              // the largest block is accepted
  CHECK(error_of([&] { prepare(44100.0, 65536); }) == std::nullopt);              // also with conversion
  try
  {
    prepare(44056.0, 512);
  }
  catch (const Error& e)
  {
    std::printf("  example message: %s\n", e.what());
  }

  // If prepare() throws, the processor is as it was: a processor that survived a failed prepare() continues like one that did not.
  Processor a(model);
  prepare_to(a, Variant::Full, 512);
  Processor b(model);
  prepare_to(b, Variant::Full, 512);
  std::vector<float> oa(in.size()), ob(in.size());
  const size_t half = in.size() / 2;
  a.process(in.data(), oa.data(), int(half));
  b.process(in.data(), ob.data(), int(half));
  ProcessorConfig bad;
  bad.sample_rate = 44056.0;
  CHECK(error_of([&] { a.prepare(bad); }) == ErrorCode::Unsupported);
  bad.sample_rate = 48000.0;
  bad.max_block = 0;
  CHECK(error_of([&] { a.prepare(bad); }) == ErrorCode::InvalidArgument);
  a.process(in.data() + half, oa.data() + half, int(in.size() - half));
  b.process(in.data() + half, ob.data() + half, int(in.size() - half));
  const bool same = bit_equal(oa, ob);
  std::printf("  output after a failed prepare() identical to output without it: %s\n", same ? "yes" : "NO");
  CHECK(same);
}

// A deterministic test signal at the given rate: three sine waves and a little noise (made from the raw output of mt19937, which is
// the same everywhere).
static std::vector<float> rate_test_signal(double rate, double seconds)
{
  const size_t n = size_t(seconds * rate);
  std::vector<float> x(n);
  std::mt19937 rng(77);
  for (size_t i = 0; i < n; ++i)
  {
    const double t = double(i) / rate;
    const double noise = (double(rng() >> 8) / 16777216.0 - 0.5) * 0.04;
    x[i] = float(0.10 * std::sin(6.283185307179586 * 440.0 * t) + 0.06 * std::sin(6.283185307179586 * 1200.0 * t) +
                 0.04 * std::sin(6.283185307179586 * 3300.0 * t) + noise);
  }
  return x;
}

static void processor_at_other_sample_rates()
{
  std::printf("processor_at_other_sample_rates\n");
  auto model = Model::load(g_model);
  for (double rate : {44100.0, 88200.0, 96000.0, 192000.0, 32000.0, 22050.0, 47952.0, 48048.0})
    for (ResamplerQuality quality : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
    {
      const std::vector<float> in = rate_test_signal(rate, 0.6);
      auto make = [&](Processor& p) {
        ProcessorConfig c;
        c.sample_rate = rate;
        c.max_block = 512;
        c.resampler = quality;
        p.prepare(c);
      };
      Processor whole(model), split(model), inplace(model);
      make(whole);
      make(split);
      make(inplace);

      std::vector<float> a(in.size()), b(in.size()), c = in;
      whole.process(in.data(), a.data(), int(in.size())); // one call, much larger than max_block
      std::mt19937 rng(4);
      std::uniform_int_distribution<int> dist(1, 700);
      for (size_t pos = 0; pos < in.size();)
      {
        const int n = int(std::min<size_t>(size_t(dist(rng)), in.size() - pos));
        split.process(in.data() + pos, b.data() + pos, n);
        pos += size_t(n);
      }
      inplace.process(c.data(), c.data(), int(c.size()));
      const bool same = bit_equal(a, b) && bit_equal(a, c);
      CHECK(same);

      // the same signal one sample at a time: the extreme split (only for the first rates, it is slow)
      bool one_by_one = true;
      if (rate == 44100.0 || rate == 96000.0)
      {
        Processor single(model);
        make(single);
        std::vector<float> d(in.size());
        for (size_t i = 0; i < in.size(); ++i)
          single.process(in.data() + i, d.data() + i, 1);
        one_by_one = bit_equal(a, d);
        CHECK(one_by_one);
      }

      // the latency is the one of the converter; the reported value is the same after a second prepare()
      RateConverter reference(int(rate), 48000, 512, quality);
      CHECK(whole.latency_samples() == reference.latency_samples() && whole.latency_samples() > 0);
      std::printf("  %7.0f Hz %-10s: latency %4d samples (%.2f ms); blocks of 1..700, one call and in place %s%s\n", rate,
                  quality == ResamplerQuality::Standard ? "Standard" : "LowLatency", whole.latency_samples(), 1000.0 * whole.latency_samples() / rate,
                  same ? "bit-identical" : "DIFFERENT", one_by_one ? "" : ", but sample by sample DIFFERENT");
    }
  { // no conversion at the rate of the model
    Processor p(model);
    prepare_to(p, Variant::Full, 512);
    CHECK(p.latency_samples() == 0);
  }
}

static void converted_rate_agrees_with_the_direct_path()
{
  std::printf("converted_rate_agrees_with_the_direct_path\n");
  auto model = Model::load(g_model);
  // A 1 kHz sine through the processor: level of the fundamental and of the harmonics 2 to 5, measured over the second second (an
  // integer number of periods at each rate, so the projections are exact). `delay` is how much the sine is delayed, in seconds.
  auto levels = [&](double rate, ResamplerQuality quality, double delay) {
    Processor p(model);
    ProcessorConfig c;
    c.sample_rate = rate;
    c.max_block = 512;
    c.resampler = quality;
    p.prepare(c);
    const size_t n = size_t(2.0 * rate);
    std::vector<float> in(n), out(n);
    for (size_t i = 0; i < n; ++i)
      in[i] = float(0.2 * std::sin(6.283185307179586 * 1000.0 * (double(i) / rate - delay)));
    p.process(in.data(), out.data(), int(n));
    std::vector<double> amp(5);
    for (int k = 1; k <= 5; ++k)
    {
      double re = 0.0, im = 0.0;
      for (size_t i = size_t(rate); i < 2 * size_t(rate); ++i)
      {
        const double ph = 6.283185307179586 * 1000.0 * k * double(i) / rate;
        re += out[i] * std::cos(ph);
        im += out[i] * std::sin(ph);
      }
      amp[size_t(k - 1)] = 2.0 * std::sqrt(re * re + im * im) / rate;
    }
    return amp;
  };
  // The levels of the harmonics of a model that distorts strongly depend on the phase at which its input is sampled (the model aliases:
  // components above half the sample rate fold into the lines below it, with a phase that follows the sampling phase). For this model a
  // fraction of a sample changes the 2nd to 5th harmonic by about -50 dB relative to the fundamental. The converted path delays the sine
  // by the (fractional) delay of the first stage, so the right reference is the direct path with the sine delayed by the same amount.
  const std::vector<double> undelayed = levels(48000.0, ResamplerQuality::Standard, 0.0);
  for (double rate : {44100.0, 96000.0, 192000.0, 32000.0})
    for (ResamplerQuality quality : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
    {
      RateConverter reference(int(rate), 48000, 512, quality);
      const double delay = reference.stage1_delay_model_samples() / 48000.0;
      const std::vector<double> direct = levels(48000.0, ResamplerQuality::Standard, delay);
      const std::vector<double> converted = levels(rate, quality, 0.0);
      const double fundamental = std::fabs(20.0 * std::log10(converted[0] / direct[0]));
      double worst_difference = -300.0;
      for (size_t k = 1; k < 5; ++k)
        worst_difference = std::max(worst_difference, 20.0 * std::log10(std::fabs(converted[k] - direct[k]) / direct[0]));
      double worst_unmatched = -300.0; // for comparison: the same, against the direct path with the wrong phase
      for (size_t k = 1; k < 5; ++k)
        worst_unmatched = std::max(worst_unmatched, 20.0 * std::log10(std::fabs(converted[k] - undelayed[k]) / undelayed[0]));
      std::printf("  %6.0f Hz %-10s: stage-1 delay %.3f model samples; fundamental differs by %.4f dB, harmonics 2 to 5 by at most %.1f dB re the fundamental"
                  " (%.1f dB if the sampling phase is not matched)\n",
                  rate, quality == ResamplerQuality::Standard ? "Standard" : "LowLatency", reference.stage1_delay_model_samples(), fundamental, worst_difference, worst_unmatched);
      CHECK(fundamental < 0.01);
      CHECK(worst_difference < (quality == ResamplerQuality::Standard ? -110.0 : -78.0));
    }
}

static void monitoring_of_the_processor()
{
  std::printf("monitoring_of_the_processor\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = test_signal();
  auto configure = [&](Processor& p, bool monitoring, double rate = 48000.0) {
    ProcessorConfig c;
    c.sample_rate = rate;
    c.max_block = 512;
    c.monitoring = monitoring;
    p.prepare(c);
  };

  { // switched off: nothing is recorded
    Processor p(model);
    configure(p, false);
    std::vector<float> out(in.size());
    p.process(in.data(), out.data(), 64);
    const PerfStats s = p.stats();
    CHECK(s.blocks == 0 && s.mean_us == 0.0 && s.max_load == 0.0);
  }
  { // switched on: counts, times and loads that agree with a clock outside
    Processor p(model);
    configure(p, true);
    std::vector<float> out(in.size());
    constexpr int kCalls = 400, kBlock = 64;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kCalls; ++i)
      p.process(in.data() + size_t(i) * kBlock, out.data() + size_t(i) * kBlock, kBlock);
    const double outside_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    const PerfStats s = p.stats();
    CHECK(s.blocks == kCalls);
    CHECK(s.mean_us > 0.0 && s.last_us > 0.0 && s.max_us >= s.mean_us && s.max_load >= s.mean_load && s.p99_load > 0.0 && s.p99_load <= s.max_load * 1.0001);
    const double block_us = 1e6 * kBlock / 48000.0;
    CHECK(std::fabs(s.mean_load - s.mean_us / block_us) < 1e-6 * s.mean_load + 1e-9); // the two ways of stating it agree
    CHECK(s.mean_us * kCalls <= outside_us * 1.001);                                   // the processor cannot have measured more than passed
    CHECK(s.mean_us * kCalls >= outside_us * 0.5);                                     // and it measured most of it
    std::printf("  %d calls of %d samples: mean %.1f us (outside clock %.1f us), mean load %.2f%%, p99 load %.2f%%, max load %.2f%%, over the deadline %llu\n", kCalls,
                kBlock, s.mean_us, outside_us / kCalls, 100.0 * s.mean_load, 100.0 * s.p99_load, 100.0 * s.max_load, static_cast<unsigned long long>(s.over_deadline));
    p.reset_stats();
    CHECK(p.stats().blocks == 0 && p.stats().over_deadline == 0 && p.stats().mean_load == 0.0);
    p.process(in.data(), out.data(), 64);
    CHECK(p.stats().blocks == 1);
  }
  { // a host at 2 MHz gives one-sample blocks of half a microsecond: every call that reaches the engine misses its deadline
    Processor p(model);
    configure(p, true, 2.0e6);
    std::vector<float> x(20000, 0.01f), y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      p.process(x.data() + i, y.data() + i, 1);
    const PerfStats s = p.stats();
    std::printf("  one-sample blocks at 2 MHz: %llu calls, %llu over the deadline, max load %.0f\n", static_cast<unsigned long long>(s.blocks),
                static_cast<unsigned long long>(s.over_deadline), s.max_load);
    CHECK(s.blocks == x.size());
    // The calls that give the engine a sample (one in about 42 at this rate) must all be counted as late. The others are cheap, but whether
    // they also exceed half a microsecond depends on the cost of reading the clock on the machine, so no upper bound is checked.
    CHECK(s.over_deadline >= x.size() / 60);
    CHECK(s.max_load > 1.0);
  }
  { // a thread that polls the statistics while the audio runs
    Processor p(model);
    configure(p, true);
    std::atomic<bool> stop{false};
    bool monotonic = true;
    std::thread reader([&] {
      std::uint64_t previous = 0;
      while (!stop.load())
      {
        const PerfStats s = p.stats();
        monotonic = monotonic && s.blocks >= previous;
        previous = s.blocks;
      }
    });
    std::vector<float> out(in.size());
    for (int round = 0; round < 3; ++round)
      for (size_t pos = 0; pos + 64 <= in.size(); pos += 64)
        p.process(in.data() + pos, out.data() + pos, 64);
    stop.store(true);
    reader.join();
    CHECK(monotonic && p.stats().blocks == 3 * (in.size() / 64));
  }
  { // measuring does not change the signal
    Processor a(model), b(model);
    configure(a, false);
    configure(b, true);
    std::vector<float> oa(in.size()), ob(in.size());
    a.process(in.data(), oa.data(), int(in.size()));
    b.process(in.data(), ob.data(), int(in.size()));
    CHECK(bit_equal(oa, ob));
  }
}

static void cpu_and_version()
{
  std::printf("cpu_and_version\n");
  const CpuSupport c = check_cpu();
  CHECK(c.supported && c.detail.empty());
  CHECK(std::string(version_string()) == "1.1.0");
  CHECK(version().major == 1 && version().minor == 1 && version().patch == 0);
  CHECK(std::string(kPreRelease).empty());
}

static void guarded(const char* name, void (*fn)())
{
  try
  {
    fn();
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION in %s: %s\n", name, e.what());
    ++g_failures;
  }
  catch (...)
  {
    std::printf("  UNKNOWN EXCEPTION in %s\n", name);
    ++g_failures;
  }
}

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3 || !exists(argv[1]) || !exists(std::string(argv[2]) + "/synth.raw") || !exists(std::string(argv[2]) + "/sweep10s.raw"))
  {
    std::printf("SKIP: model or data files not found\n");
    return 77;
  }
  if (!check_cpu().supported)
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2 and FMA\n");
    return 77;
  }
  g_model = argv[1];
  g_data = argv[2];
  guarded("model_info_and_variants", model_info_and_variants);
  guarded("model_load_errors", model_load_errors);
  guarded("processor_matches_the_engine", processor_matches_the_engine);
  guarded("block_splitting", block_splitting);
  guarded("double_version_equals_float_version", double_version_equals_float_version);
  guarded("edge_cases", edge_cases);
  guarded("prepare_errors_and_guarantee", prepare_errors_and_guarantee);
  guarded("processor_at_other_sample_rates", processor_at_other_sample_rates);
  guarded("converted_rate_agrees_with_the_direct_path", converted_rate_agrees_with_the_direct_path);
  guarded("monitoring_of_the_processor", monitoring_of_the_processor);
  guarded("cpu_and_version", cpu_and_version);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
