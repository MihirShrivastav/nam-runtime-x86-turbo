// variant_compare: how far is the output of A2-Lite from the output of A2-Full of the same model?
//
//   variant_compare --model amp.nam [--model ...] --input signal.raw [--input ...] [--csv file]
//
// Inputs are raw files of 32-bit floating-point mono samples at the sample rate of the model. Both variants process the signal in blocks of
// 64 samples. For every model and input the program prints
//   - the error of A2-Lite against A2-Full as an error-to-signal ratio in dB (energy of the difference divided by the energy of A2-Full),
//   - the same after the level of A2-Lite has been matched to A2-Full with the best single gain (least squares), which removes a plain
//     difference of loudness, and
//   - the difference of the RMS levels in dB (A2-Lite minus A2-Full).
// A2-Lite is a smaller network, so its output is not meant to be equal to the output of A2-Full: the numbers say how different it sounds.
// Uses only the public interface; needs no NAM Core.
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

std::vector<float> run(const std::shared_ptr<const Model>& model, Variant variant, const std::vector<float>& in)
{
  Processor p(model);
  ProcessorConfig c;
  c.max_block = 64;
  c.variant = variant;
  p.prepare(c);
  std::vector<float> out(in.size());
  for (size_t pos = 0; pos < in.size(); pos += 64)
    p.process(in.data() + pos, out.data() + pos, static_cast<int>(std::min<size_t>(64, in.size() - pos)));
  return out;
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
  std::string csv;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "variant_compare: missing value after " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") models.push_back(next());
    else if (a == "--input") inputs.push_back(next());
    else if (a == "--csv") csv = next();
    else
    {
      std::cerr << "usage: variant_compare --model amp.nam [...] --input signal.raw [...] [--csv file]\n";
      return 2;
    }
  }
  if (models.empty() || inputs.empty())
  {
    std::cerr << "usage: variant_compare --model amp.nam [...] --input signal.raw [...] [--csv file]\n";
    return 2;
  }
  try
  {
    if (!check_cpu().supported)
    {
      std::cerr << "variant_compare: " << check_cpu().detail << "\n";
      return 2;
    }
    std::FILE* out_csv = csv.empty() ? nullptr : std::fopen(csv.c_str(), "w");
    if (out_csv)
      std::fprintf(out_csv, "model,input,esr_db,esr_gain_matched_db,level_difference_db\n");
    std::printf("| Model | Input | A2-Lite against A2-Full: ESR (dB) | after matching the level (dB) | Level of A2-Lite minus A2-Full (dB) |\n"
                "|---|---|---:|---:|---:|\n");
    for (const std::string& model_path : models)
    {
      auto model = Model::load(model_path);
      if (!model->has(Variant::Lite) || !model->has(Variant::Full))
      {
        std::cerr << "variant_compare: " << model_path << " does not contain both variants\n";
        continue;
      }
      for (const std::string& input_path : inputs)
      {
        const std::vector<float> in = read_raw(input_path);
        const std::vector<float> full = run(model, Variant::Full, in), lite = run(model, Variant::Lite, in);
        double e = 0.0, pf = 0.0, pl = 0.0, cross = 0.0;
        for (size_t i = 0; i < in.size(); ++i)
        {
          const double f = full[i], l = lite[i];
          e += (l - f) * (l - f);
          pf += f * f;
          pl += l * l;
          cross += l * f;
        }
        const double g = pl > 0.0 ? cross / pl : 1.0; // the gain that minimises the error of g * lite against full
        const double e_matched = pf - 2.0 * g * cross + g * g * pl;
        const double esr = 10.0 * std::log10(std::max(e / std::max(pf, 1e-300), 1e-300));
        const double esr_matched = 10.0 * std::log10(std::max(e_matched / std::max(pf, 1e-300), 1e-300));
        const double level = 10.0 * std::log10(std::max(pl, 1e-300) / std::max(pf, 1e-300));
        std::printf("| %s | %s | %.1f | %.1f | %+.1f |\n", base_name(model_path).c_str(), base_name(input_path).c_str(), esr, esr_matched, level);
        std::fflush(stdout);
        if (out_csv)
          std::fprintf(out_csv, "%s,%s,%.2f,%.2f,%.2f\n", base_name(model_path).c_str(), base_name(input_path).c_str(), esr, esr_matched, level);
      }
    }
    if (out_csv)
      std::fclose(out_csv);
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "variant_compare: " << e.what() << "\n";
    return 2;
  }
}
