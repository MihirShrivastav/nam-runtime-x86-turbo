// accuracy: error of our output against NAM Core, for real models and real recordings. Needs NAMTURBO_BASELINE=ON.
//
//   accuracy --model a.nam [--model b.nam ...] --input track_48k.raw [--input ...] [--csv file] [--dump folder]
// --dump writes the outputs (32-bit float raw files: ours and the NAM Core fast path) so that other tools can judge them.
//
// Inputs are raw files of 32-bit floating-point mono samples at the sample rate of the model (48 kHz). For every model, variant
// (A2-Full, A2-Lite) and input it processes the signal in blocks of 64 samples through
//   - ours (Processor),
//   - the NAM Core A2 fast path (the reference of this project),
//   - the NAM Core generic WaveNet (a second implementation, to show how far two correct implementations are apart),
// and prints the error energy relative to the signal energy (ESR) in dB, ours against fast and generic against fast, and the
// largest sample difference relative to the largest output sample. Lower is better; -100 dB or lower is far below audibility.
// Exit code 0, or 1 if the error of ours against the fast path is above the limit (--limit, default -100 dB).
#ifndef _CRT_SECURE_NO_WARNINGS
  #define _CRT_SECURE_NO_WARNINGS // std::fopen for the --csv file
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nam_core.h"
#include "namturbo/namturbo.h"

using namespace namturbo;

namespace
{
std::vector<float> read_raw(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  const size_t n = static_cast<size_t>(f.tellg()) / sizeof(float);
  std::vector<float> x(n);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(x.data()), std::streamsize(n * sizeof(float)));
  return x;
}

struct Diff
{
  double esr_db = 0.0;
  double peak_rel_db = 0.0; // largest difference relative to the largest reference sample
};

Diff compare(const std::vector<double>& a, const std::vector<double>& ref)
{
  double e = 0.0, p = 0.0, worst = 0.0, top = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const double d = a[i] - ref[i];
    e += d * d;
    p += ref[i] * ref[i];
    worst = std::max(worst, std::fabs(d));
    top = std::max(top, std::fabs(ref[i]));
  }
  Diff r;
  r.esr_db = 10.0 * std::log10(std::max(e / std::max(p, 1e-300), 1e-300));
  r.peak_rel_db = 20.0 * std::log10(std::max(worst / std::max(top, 1e-300), 1e-300));
  return r;
}

std::string base_name(const std::string& p)
{
  const size_t at = p.find_last_of("/\\");
  return at == std::string::npos ? p : p.substr(at + 1);
}
} // namespace

int main(int argc, char** argv)
{
  std::vector<std::string> models, inputs;
  std::string csv, dump;
  double limit = -100.0;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "accuracy: missing value after " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") models.push_back(next());
    else if (a == "--input") inputs.push_back(next());
    else if (a == "--csv") csv = next();
    else if (a == "--dump") dump = next();
    else if (a == "--limit") limit = std::atof(next().c_str());
    else
    {
      std::cerr << "usage: accuracy --model a.nam [...] --input track.raw [...] [--csv file] [--limit dB]\n";
      return 2;
    }
  }
  if (models.empty() || inputs.empty())
  {
    std::cerr << "usage: accuracy --model a.nam [...] --input track.raw [...] [--csv file] [--limit dB]\n";
    return 2;
  }
  try
  {
    if (!check_cpu().supported)
    {
      std::cerr << "accuracy: " << check_cpu().detail << "\n";
      return 2;
    }
    std::FILE* out_csv = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
    if (out_csv)
      std::fprintf(out_csv, "model,variant,input,seconds,ours_vs_fast_esr_db,ours_vs_fast_peak_db,generic_vs_fast_esr_db,generic_vs_fast_peak_db\n");
    std::printf("| Model | Variant | Input | Seconds | Ours vs NAM fast: ESR (dB) | largest difference (dB re peak) | NAM generic vs NAM fast: ESR (dB) |\n"
                "|---|---|---|---:|---:|---:|---:|\n");
    bool ok = true;
    double worst_ours = -1e9;
    for (const std::string& input_path : inputs)
    {
      const std::vector<float> in_f = read_raw(input_path);
      std::vector<double> in(in_f.begin(), in_f.end());
      const size_t n = in.size();
      for (const std::string& model_path : models)
      {
        auto model = Model::load(model_path);
        for (const char* vname : {"full", "lite"})
        {
          const bool full = std::string(vname) == "full";
          const Variant variant = full ? Variant::Full : Variant::Lite;
          if (!model->has(variant))
            continue;
          const double slim = full ? 1.0 : 0.0;

          Processor p(model);
          ProcessorConfig c;
          c.max_block = 64;
          c.variant = variant;
          p.prepare(c);
          NamCoreModel fast = make_nam_core_dsp(model_path, slim, "fast");
          NamCoreModel generic = make_nam_core_dsp(model_path, slim, "generic");
          fast.dsp->Reset(fast.sample_rate, 64);
          generic.dsp->Reset(generic.sample_rate, 64);

          std::vector<double> ours(n), ref(n), gen(n);
          for (size_t pos = 0; pos < n; pos += 64)
          {
            const int len = static_cast<int>(std::min<size_t>(64, n - pos));
            p.process(in.data() + pos, ours.data() + pos, len);
            NAM_SAMPLE* ip[] = {in.data() + pos};
            NAM_SAMPLE* op[] = {ref.data() + pos};
            fast.dsp->process(ip, op, len);
            NAM_SAMPLE* op2[] = {gen.data() + pos};
            generic.dsp->process(ip, op2, len);
          }
          const Diff e1 = compare(ours, ref), e2 = compare(gen, ref);
          worst_ours = std::max(worst_ours, e1.esr_db);
          ok = ok && e1.esr_db <= limit;
          if (!dump.empty())
          {
            const std::string stem = dump + "/" + base_name(model_path) + "_" + vname + "_" + base_name(input_path);
            for (int which = 0; which < 2; ++which)
            {
              const std::vector<double>& src = which == 0 ? ours : ref;
              std::vector<float> f(src.size());
              std::transform(src.begin(), src.end(), f.begin(), [](double v) { return static_cast<float>(v); });
              std::ofstream o(stem + (which == 0 ? ".ours.raw" : ".fast.raw"), std::ios::binary);
              o.write(reinterpret_cast<const char*>(f.data()), std::streamsize(f.size() * sizeof(float)));
            }
          }
          const double seconds = double(n) / 48000.0;
          std::printf("| %s | %s | %s | %.1f | %.1f | %.1f | %.1f |\n", base_name(model_path).c_str(), vname, base_name(input_path).c_str(), seconds,
                      e1.esr_db, e1.peak_rel_db, e2.esr_db);
          std::fflush(stdout);
          if (out_csv)
            std::fprintf(out_csv, "%s,%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f\n", base_name(model_path).c_str(), vname, base_name(input_path).c_str(), seconds,
                         e1.esr_db, e1.peak_rel_db, e2.esr_db, e2.peak_rel_db);
        }
      }
    }
    if (out_csv)
      std::fclose(out_csv);
    std::printf("\nWorst error of ours against the NAM Core fast path: %.1f dB (limit %.1f dB): %s\n", worst_ours, limit, ok ? "within the limit" : "ABOVE THE LIMIT");
    return ok ? 0 : 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "accuracy: " << e.what() << "\n";
    return 2;
  }
}
