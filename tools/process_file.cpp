// process_file: run a raw audio file through a Processor at a chosen host rate and write the result.
//
//   process_file --model amp.nam --input in.raw --output out.raw [--host-rate 44100] [--variant full|lite]
//                [--resampler standard|low-latency] [--block 128]
//
// Raw files are mono little-endian 32-bit floating-point samples at the HOST rate. The output has the same length as the input and is
// delayed by the latency of the processor (printed). The program also prints the delay of the first conversion stage in samples of the
// model rate, which is what a comparison with a direct path needs (benchmarks/rate_check.py). Uses an internal header for that one number.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "namturbo/namturbo.h"
#include "resampler.h"

using namespace namturbo;

int main(int argc, char** argv)
{
  std::string model_path, in_path, out_path, variant = "full", quality = "standard";
  double host_rate = 48000.0;
  int block = 128;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "process_file: missing value after " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_path = next();
    else if (a == "--input") in_path = next();
    else if (a == "--output") out_path = next();
    else if (a == "--host-rate") host_rate = std::atof(next().c_str());
    else if (a == "--variant") variant = next();
    else if (a == "--resampler") quality = next();
    else if (a == "--block") block = std::atoi(next().c_str());
    else
    {
      std::cerr << "process_file: unknown option " << a << "\n";
      return 2;
    }
  }
  if (model_path.empty() || in_path.empty() || out_path.empty() || block < 1)
  {
    std::cerr << "usage: process_file --model amp.nam --input in.raw --output out.raw [--host-rate 44100] [--variant full|lite] "
                 "[--resampler standard|low-latency] [--block 128]\n";
    return 2;
  }
  try
  {
    std::ifstream f(in_path, std::ios::binary | std::ios::ate);
    if (!f)
      throw std::runtime_error("cannot open " + in_path);
    std::vector<float> in(static_cast<size_t>(f.tellg()) / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(in.data()), std::streamsize(in.size() * sizeof(float)));

    auto model = Model::load(model_path);
    Processor p(model);
    ProcessorConfig c;
    c.sample_rate = host_rate;
    c.max_block = block;
    c.variant = variant == "lite" ? Variant::Lite : Variant::Full;
    c.resampler = quality == "low-latency" ? ResamplerQuality::LowLatency : ResamplerQuality::Standard;
    p.prepare(c);

    std::vector<float> out(in.size());
    for (size_t pos = 0; pos < in.size(); pos += size_t(block))
      p.process(in.data() + pos, out.data() + pos, static_cast<int>(std::min<size_t>(size_t(block), in.size() - pos)));
    std::ofstream o(out_path, std::ios::binary);
    o.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * sizeof(float)));

    double stage1 = 0.0;
    if (host_rate != model->info().sample_rate)
    {
      RateConverter rc(static_cast<int>(host_rate), static_cast<int>(model->info().sample_rate), block, c.resampler);
      stage1 = rc.stage1_delay_model_samples();
    }
    std::printf("latency_samples=%d\nstage1_delay_model_samples=%.6f\nsamples=%zu\n", p.latency_samples(), stage1, in.size());
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "process_file: " << e.what() << "\n";
    return 2;
  }
}
