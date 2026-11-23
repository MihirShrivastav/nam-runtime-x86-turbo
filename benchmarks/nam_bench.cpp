// nam-bench: the benchmark program of the library. It uses only the public API and prints a table in Markdown.
//
//   nam_bench --model <a2.nam> [--model <another.nam> ...] [--buffers 16,32,64,128,256,512] [--variants full,lite]
//             [--runs 3] [--seconds 2] [--host-rate 48000] [--core N] [--json <file>] [--input <file.raw>]
//
// For every model, variant and buffer size it times Processor::process() one block at a time and prints the median (p50) and
// the 99th percentile (p99) of the time of a block, and the real-time factor (how many times faster than the audio it runs).
// If the program was built with NAMTURBO_BASELINE=ON, NAM Core's A2 fast path runs the same input in the same way, and the table
// adds its times, the speed-up and a correctness check of our output against it.
//
// Method: both sides process 64-bit floating-point buffers (the interface of NAM Core); the signal is two sine waves; the
// first pass is not timed; ours and the baseline are timed in turns within each run, so that a change of clock speed during
// the measurement affects both; the speed-up is the median over the runs of (baseline p50) / (our p50). Pin the thread to a core
// with --core on a machine whose cores differ in speed. --json writes every number, with the processor and compiler names, to a file
// that benchmarks/make_report.py turns into tables.
// Exit code: 0 success, 1 a correctness check failed, 2 usage or load error.
#ifndef _CRT_SECURE_NO_WARNINGS
  #define _CRT_SECURE_NO_WARNINGS // std::fopen for the --json file
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "namturbo/namturbo.h"
#include "pin.h"
#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
  #include <cpuid.h>
#endif
#ifdef NAMTURBO_WITH_NAMCORE
  #include "nam_core.h"
#endif

using namespace namturbo;
using Clock = std::chrono::steady_clock;

namespace
{
constexpr double kPi = 3.14159265358979323846;

double percentile(std::vector<double> xs, double q)
{
  std::sort(xs.begin(), xs.end());
  const double idx = q * double(xs.size() - 1);
  const size_t i = static_cast<size_t>(idx);
  if (i + 1 >= xs.size())
    return xs[i];
  const double frac = idx - double(i);
  return xs[i] * (1.0 - frac) + xs[i + 1] * frac;
}

struct Times
{
  double p50 = 0.0, p99 = 0.0;
};

// Times fn(in, out, n) for each block of the signal: one pass that is not counted, then one that is.
template <class Fn>
Times time_blocks(Fn&& fn, const std::vector<double>& in, std::vector<double>& out, int buffer)
{
  std::vector<double> us;
  for (int pass = 0; pass < 2; ++pass)
    for (size_t pos = 0; pos + buffer <= in.size(); pos += buffer)
    {
      const auto t0 = Clock::now();
      fn(in.data() + pos, out.data() + pos, buffer);
      const auto t1 = Clock::now();
      if (pass == 1)
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  return {percentile(us, 0.5), percentile(us, 0.99)};
}

std::vector<double> make_signal(size_t n, double rate)
{
  std::vector<double> x(n);
  for (size_t i = 0; i < n; ++i)
  {
    const double t = double(i) / rate;
    x[i] = 0.25 * std::sin(2.0 * kPi * 220.0 * t) + 0.10 * std::sin(2.0 * kPi * 1230.0 * t);
  }
  return x;
}

std::vector<int> parse_ints(const std::string& s)
{
  std::vector<int> v;
  size_t at = 0;
  while (at < s.size())
  {
    size_t comma = s.find(',', at);
    if (comma == std::string::npos)
      comma = s.size();
    v.push_back(std::atoi(s.substr(at, comma - at).c_str()));
    at = comma + 1;
  }
  return v;
}

double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

std::string cpu_name()
{
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
  unsigned regs[12] = {};
  #if defined(_MSC_VER)
  int r[4];
  __cpuid(r, int(0x80000000));
  if (static_cast<unsigned>(r[0]) < 0x80000004u)
    return "unknown";
  for (int i = 0; i < 3; ++i)
  {
    __cpuid(r, int(0x80000002u + i));
    for (int k = 0; k < 4; ++k)
      regs[i * 4 + k] = static_cast<unsigned>(r[k]);
  }
  #else
  if (__get_cpuid_max(0x80000000, nullptr) < 0x80000004u)
    return "unknown";
  for (unsigned i = 0; i < 3; ++i)
    __cpuid(0x80000002u + i, regs[i * 4], regs[i * 4 + 1], regs[i * 4 + 2], regs[i * 4 + 3]);
  #endif
  char text[49] = {};
  std::memcpy(text, regs, 48);
  std::string name = text;
  const size_t first = name.find_first_not_of(' ');
  const size_t last = name.find_last_not_of(' ');
  return first == std::string::npos ? "unknown" : name.substr(first, last - first + 1);
#else
  return "unknown";
#endif
}

std::string compiler_name()
{
#if defined(__clang__) && defined(_MSC_VER)
  return std::string("clang-cl ") + __clang_version__;
#elif defined(__clang__)
  return std::string("clang ") + __clang_version__;
#elif defined(_MSC_VER)
  return "MSVC " + std::to_string(_MSC_VER);
#elif defined(__GNUC__)
  return std::string("GCC ") + __VERSION__;
#else
  return "unknown";
#endif
}

std::string json_text(const std::string& s)
{
  std::string o = "\"";
  for (char c : s)
  {
    if (c == '"' || c == '\\')
      o += '\\';
    if (static_cast<unsigned char>(c) >= 0x20)
      o += c;
  }
  return o + "\"";
}

struct Row
{
  std::string model, variant;
  int buffer = 0;
  double block_us = 0.0;
  double ours_p50 = 0.0, ours_p99 = 0.0;
  double base_p50 = 0.0, base_p99 = 0.0, speedup = 0.0;
};

#ifdef NAMTURBO_WITH_NAMCORE
// The error energy of `a` against `b` over the signal, relative to the energy of `b`, in dB (the ESR of NAM's own tests).
double esr_db(const std::vector<double>& a, const std::vector<double>& b)
{
  double e = 0.0, p = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    e += (a[i] - b[i]) * (a[i] - b[i]);
    p += b[i] * b[i];
  }
  return 10.0 * std::log10(std::max(e / std::max(p, 1e-300), 1e-300));
}
#endif

void usage()
{
  std::cerr << "usage: nam_bench --model <a2.nam> [--model ...] [--buffers 16,32,...] [--variants full,lite] [--runs N]\n"
               "                 [--seconds S] [--host-rate HZ] [--oversampling 1|2|4] [--resampler standard|low-latency]\n"
               "                 [--core N] [--json FILE]\n";
}
} // namespace

int main(int argc, char** argv)
{
  std::vector<std::string> models;
  std::vector<int> buffers = {16, 32, 64, 128, 256, 512};
  std::string variants = "full,lite";
  int runs = 3, core = -1, os_factor = 1;
  ResamplerQuality resampler_q = ResamplerQuality::Standard;
  std::string json_path, input_path;
  double seconds = 2.0, host_rate = 48000.0;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") models.push_back(next());
    else if (a == "--buffers") buffers = parse_ints(next());
    else if (a == "--variants") variants = next();
    else if (a == "--runs") runs = std::atoi(next().c_str());
    else if (a == "--seconds") seconds = std::atof(next().c_str());
    else if (a == "--host-rate") host_rate = std::atof(next().c_str());
    else if (a == "--oversampling") os_factor = std::atoi(next().c_str());
    else if (a == "--resampler")
    {
      const std::string q = next();
      if (q == "standard") resampler_q = ResamplerQuality::Standard;
      else if (q == "low-latency") resampler_q = ResamplerQuality::LowLatency;
      else
      {
        usage();
        return 2;
      }
    }
    else if (a == "--core") core = std::atoi(next().c_str());
    else if (a == "--json") json_path = next();
    else if (a == "--input") input_path = next();
    else
    {
      usage();
      return 2;
    }
  }
  if (models.empty() || buffers.empty() || runs < 1 || seconds <= 0.0 || (os_factor != 1 && os_factor != 2 && os_factor != 4))
  {
    usage();
    return 2;
  }
  const Oversampling os_mode = os_factor == 4 ? Oversampling::X4 : (os_factor == 2 ? Oversampling::X2 : Oversampling::Off);
  try
  {
    const CpuSupport cpu = check_cpu();
    if (!cpu.supported)
    {
      std::cerr << "nam_bench: " << cpu.detail << "\n";
      return 2;
    }
    if (core >= 0 && !pin_to_core(core))
      std::cerr << "nam_bench: could not pin to core " << core << "\n";
    const size_t total = static_cast<size_t>(seconds * host_rate);
    std::vector<double> in;
    if (input_path.empty())
      in = make_signal(total, host_rate);
    else
    {
      // A recording: raw 32-bit floating-point mono samples at the host rate; the first `seconds` of it are used.
      std::ifstream f(input_path, std::ios::binary | std::ios::ate);
      if (!f)
      {
        std::cerr << "nam_bench: cannot open " << input_path << "\n";
        return 2;
      }
      const size_t available = static_cast<size_t>(f.tellg()) / sizeof(float);
      std::vector<float> raw(std::min(available, total));
      f.seekg(0);
      f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size() * sizeof(float)));
      in.assign(raw.begin(), raw.end());
    }
    std::vector<double> out(in.size(), 0.0);
    std::vector<Row> rows;
    bool all_ok = true;
#ifdef NAMTURBO_WITH_NAMCORE
    std::vector<std::string> checks;
    std::vector<double> esr_values;
#endif

    for (const std::string& path : models)
    {
      auto model = Model::load(path);
      for (const std::string& vname : {std::string("full"), std::string("lite")})
      {
        if (variants.find(vname) == std::string::npos)
          continue;
        const Variant variant = vname == "full" ? Variant::Full : Variant::Lite;
        if (!model->has(variant))
          continue;
#ifdef NAMTURBO_WITH_NAMCORE
        const double slim = vname == "full" ? 1.0 : 0.0;
#endif
        std::string label = path.substr(path.find_last_of("/\\") == std::string::npos ? 0 : path.find_last_of("/\\") + 1);

#ifdef NAMTURBO_WITH_NAMCORE
        // Correctness gate: our output against NAM Core's fast path on one second of the signal, blocks of 64.
        if (host_rate == model->info().sample_rate && os_mode == Oversampling::Off)
        {
          Processor p(model);
          ProcessorConfig c;
          c.sample_rate = host_rate;
          c.max_block = 64;
          c.variant = variant;
          c.resampler = resampler_q;
          c.oversampling = os_mode;
          p.prepare(c);
          NamCoreModel ref = make_nam_core_dsp(path, slim, "fast");
          ref.dsp->Reset(ref.sample_rate, 64);
          const size_t n = std::min<size_t>(total, static_cast<size_t>(host_rate)) / 64 * 64;
          std::vector<double> a(n), b(n);
          for (size_t pos = 0; pos < n; pos += 64)
          {
            p.process(in.data() + pos, a.data() + pos, 64);
            NAM_SAMPLE* ip[] = {const_cast<NAM_SAMPLE*>(in.data() + pos)};
            NAM_SAMPLE* op[] = {b.data() + pos};
            ref.dsp->process(ip, op, 64);
          }
          const double e = esr_db(a, b);
          char line[200];
          std::snprintf(line, sizeof line, "%s %s: error against NAM Core fast path %.1f dB (ESR)%s", label.c_str(), vname.c_str(), e,
                        e < -60.0 ? "" : "  ** TOO LARGE **");
          checks.push_back(line);
          esr_values.push_back(e);
          all_ok = all_ok && e < -60.0;
        }
#endif
        for (int buffer : buffers)
        {
          Row row;
          row.model = label;
          row.variant = vname;
          row.buffer = buffer;
          row.block_us = 1e6 * buffer / host_rate;
          std::vector<double> o50, o99, b50, b99, ratio;
          for (int run = 0; run < runs; ++run)
          {
            Processor p(model);
            ProcessorConfig c;
            c.sample_rate = host_rate;
            c.max_block = buffer;
            c.variant = variant;
            c.resampler = resampler_q;
            c.oversampling = os_mode;
            p.prepare(c);
            const Times ours = time_blocks([&](const double* i, double* o, int n) { p.process(i, o, n); }, in, out, buffer);
            o50.push_back(ours.p50);
            o99.push_back(ours.p99);
#ifdef NAMTURBO_WITH_NAMCORE
            if (host_rate == model->info().sample_rate && os_mode == Oversampling::Off)
            {
              NamCoreModel m = make_nam_core_dsp(path, slim, "fast");
              m.dsp->Reset(m.sample_rate, buffer);
              const Times base = time_blocks(
                  [&](const double* i, double* o, int n) {
                    NAM_SAMPLE* ip[] = {const_cast<NAM_SAMPLE*>(i)};
                    NAM_SAMPLE* op[] = {o};
                    m.dsp->process(ip, op, n);
                  },
                  in, out, buffer);
              b50.push_back(base.p50);
              b99.push_back(base.p99);
              ratio.push_back(base.p50 / ours.p50);
            }
#endif
          }
          row.ours_p50 = median(o50);
          row.ours_p99 = median(o99);
          if (!ratio.empty())
          {
            row.base_p50 = median(b50);
            row.base_p99 = median(b99);
            row.speedup = median(ratio);
          }
          rows.push_back(row);
        }
      }
    }

    if (os_factor > 1)
      std::printf("namturbo %s, %.0f Hz host rate, %dx oversampling, %.1f s of audio, median of %d run%s\n\n", version_string(), host_rate, os_factor,
                  seconds, runs, runs == 1 ? "" : "s");
    else
      std::printf("namturbo %s, %.0f Hz host rate, %.1f s of audio, median of %d run%s\n\n", version_string(), host_rate, seconds, runs,
                  runs == 1 ? "" : "s");
    const bool base = std::any_of(rows.begin(), rows.end(), [](const Row& r) { return r.speedup > 0.0; });
    if (base)
      std::printf("| Model | Variant | Buffer | Ours p50 (us) | Ours p99 (us) | NAM Core p50 (us) | NAM Core p99 (us) | Speed-up | Real-time factor |\n"
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    else
      std::printf("| Model | Variant | Buffer | p50 (us) | p99 (us) | Load at p50 | Real-time factor |\n|---|---|---:|---:|---:|---:|---:|\n");
    for (const Row& r : rows)
    {
      if (base)
        std::printf("| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2fx | %.1fx |\n", r.model.c_str(), r.variant.c_str(), r.buffer, r.ours_p50,
                    r.ours_p99, r.base_p50, r.base_p99, r.speedup, r.block_us / r.ours_p50);
      else
        std::printf("| %s | %s | %d | %.2f | %.2f | %.1f%% | %.1fx |\n", r.model.c_str(), r.variant.c_str(), r.buffer, r.ours_p50, r.ours_p99,
                    100.0 * r.ours_p50 / r.block_us, r.block_us / r.ours_p50);
    }
    if (!json_path.empty())
    {
      std::FILE* f = std::fopen(json_path.c_str(), "w");
      if (!f)
      {
        std::cerr << "nam_bench: cannot write " << json_path << "\n";
        return 2;
      }
      std::fprintf(f, "{\n  \"namturbo_version\": %s,\n  \"cpu\": %s,\n  \"compiler\": %s,\n", json_text(version_string()).c_str(),
                   json_text(cpu_name()).c_str(), json_text(compiler_name()).c_str());
      std::fprintf(f, "  \"host_rate\": %.0f,\n  \"oversampling\": %d,\n  \"seconds\": %.2f,\n  \"runs\": %d,\n  \"pinned_core\": %d,\n  \"baseline\": %s,\n  \"rows\": [\n",
                   host_rate, os_factor, seconds, runs, core, base ? "true" : "false");
      for (size_t i = 0; i < rows.size(); ++i)
      {
        const Row& r = rows[i];
        std::fprintf(f,
                     "    {\"model\": %s, \"variant\": \"%s\", \"buffer\": %d, \"block_us\": %.3f, \"ours_p50_us\": %.3f, \"ours_p99_us\": %.3f, "
                     "\"base_p50_us\": %.3f, \"base_p99_us\": %.3f, \"speedup\": %.4f}%s\n",
                     json_text(r.model).c_str(), r.variant.c_str(), r.buffer, r.block_us, r.ours_p50, r.ours_p99, r.base_p50, r.base_p99, r.speedup,
                     i + 1 < rows.size() ? "," : "");
      }
      std::fprintf(f, "  ],\n  \"esr_db\": [");
#ifdef NAMTURBO_WITH_NAMCORE
      for (size_t i = 0; i < esr_values.size(); ++i)
        std::fprintf(f, "%s%.2f", i ? ", " : "", esr_values[i]);
#endif
      std::fprintf(f, "]\n}\n");
      std::fclose(f);
    }
#ifdef NAMTURBO_WITH_NAMCORE
    std::printf("\nCorrectness:\n");
    for (const std::string& c : checks)
      std::printf("- %s\n", c.c_str());
#endif
    return all_ok ? 0 : 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "nam_bench: " << e.what() << "\n";
    return 2;
  }
}
