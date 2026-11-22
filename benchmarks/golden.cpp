// golden: run NAM Core on a raw float32 input and write its output as raw float32.
//
//   golden --model <nam> --slim <0..1> --input <in.raw> --out <out.raw> --block <n> --path fast|generic
//
// Raw files are mono little-endian float32 at the model's sample rate (48 kHz). `--path fast` uses
// NAM Core's A2 fast path, `--path generic` its generic WaveNet. The model is Reset(sampleRate, block)
// once, so prewarm happens the way a host would start it, then the input is processed in blocks of
// `--block` frames (the last block may be shorter), converting float <-> NAM_SAMPLE.
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include "nam_core.h"

namespace
{
struct Options
{
  std::string model, input, out, path = "fast";
  double slim = 1.0;
  int block = 64;
};

[[noreturn]] void die(const std::string& msg)
{
  std::cerr << "golden: " << msg << "\n";
  std::exit(1);
}

Options parse(int argc, char** argv)
{
  Options o;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        die("missing value after " + a);
      return argv[++i];
    };
    if (a == "--model") o.model = next();
    else if (a == "--slim") o.slim = std::atof(next().c_str());
    else if (a == "--input") o.input = next();
    else if (a == "--out") o.out = next();
    else if (a == "--block") o.block = std::atoi(next().c_str());
    else if (a == "--path") o.path = next();
    else die("unknown argument " + a);
  }
  if (o.model.empty() || o.input.empty() || o.out.empty()) die("--model, --input and --out are required");
  if (o.block < 1) die("--block must be at least 1");
  if (o.path != "fast" && o.path != "generic") die("--path must be fast or generic");
  return o;
}
} // namespace

int main(int argc, char** argv)
{
  const Options o = parse(argc, argv);
  try
  {
    namturbo::NamCoreModel m = namturbo::make_nam_core_dsp(o.model, o.slim, o.path);
    m.dsp->Reset(m.sample_rate, o.block);

    const std::vector<float> in = namturbo::read_raw(o.input);
    std::vector<float> out(in.size());
    std::vector<NAM_SAMPLE> in_d(o.block), out_d(o.block);
    for (size_t pos = 0; pos < in.size(); pos += o.block)
    {
      const int n = static_cast<int>(std::min<size_t>(o.block, in.size() - pos));
      for (int i = 0; i < n; ++i)
        in_d[i] = static_cast<NAM_SAMPLE>(in[pos + i]);
      NAM_SAMPLE* in_ptr[] = {in_d.data()};
      NAM_SAMPLE* out_ptr[] = {out_d.data()};
      m.dsp->process(in_ptr, out_ptr, n);
      for (int i = 0; i < n; ++i)
        out[pos + i] = static_cast<float>(out_d[i]);
    }

    std::ofstream of(o.out, std::ios::binary);
    of.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return of ? 0 : 1;
  }
  catch (const std::exception& e)
  {
    die(e.what());
  }
}
