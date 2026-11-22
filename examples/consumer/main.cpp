// The smallest complete use of the library: load an A2 model, process one second of a sine wave, print what happened.
// Usage: consumer <model.nam> [host sample rate, default 48000]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "namturbo/namturbo.h"

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: consumer <model.nam> [sample rate]\n");
    return 2;
  }
  const double rate = argc > 2 ? std::atof(argv[2]) : 48000.0;
  try
  {
    const namturbo::CpuSupport cpu = namturbo::check_cpu();
    if (!cpu.supported)
    {
      std::fprintf(stderr, "%s\n", cpu.detail.c_str());
      return 3;
    }

    auto model = namturbo::Model::load(argv[1]);

    namturbo::Processor processor(model);
    namturbo::ProcessorConfig config;
    config.sample_rate = rate;
    config.max_block = 256;
    processor.prepare(config);

    std::vector<float> in(256), out(256);
    double peak = 0.0;
    const int blocks = static_cast<int>(rate / 256.0);
    for (int b = 0; b < blocks; ++b)
    {
      for (int i = 0; i < 256; ++i)
        in[i] = static_cast<float>(0.1 * std::sin(6.283185307179586 * 220.0 * double(b * 256 + i) / rate));
      processor.process(in.data(), out.data(), 256);
      for (float v : out)
        peak = std::max(peak, std::fabs(double(v)));
    }
    std::printf("namturbo %s: %.0f Hz, latency %d samples, output peak %.4f\n", namturbo::version_string(), rate, processor.latency_samples(), peak);
    return peak > 0.0 ? 0 : 1;
  }
  catch (const namturbo::Error& e)
  {
    std::fprintf(stderr, "namturbo error: %s\n", e.what());
    return 1;
  }
}
