// parity: run one of our engines over a raw float32 input and compare with a golden raw output.
//
//   parity --model <nam> --slim <0..1> --backend scalar --input <in.raw> --golden <golden.raw>
//          [--blocks 1,16,32,48,64,128,256,512] [--random-blocks] [--max-samples N] [--out <file.raw>]
//
// Prints ESR in dB per block size against the golden and exits non-zero if any is above the parity
// limit for the model's width (src/parity_limits.h). --random-blocks also runs block sizes drawn
// from 1..512 with a fixed seed. --out writes the first run's output.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "a2_weights.h"
#include "common.h"
#include "engines.h"
#include "parity_limits.h"

using namespace namturbo;

static std::vector<int> parse_list(const std::string& s)
{
  std::vector<int> v;
  std::stringstream ss(s);
  for (std::string tok; std::getline(ss, tok, ',');)
    v.push_back(std::atoi(tok.c_str()));
  return v;
}

int main(int argc, char** argv)
{
  std::string model, backend = "scalar", input, golden, out_path, blocks = "1,16,32,48,64,128,256,512";
  double slim = 1.0;
  bool random_blocks = false;
  size_t max_samples = 0;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "parity: missing value after " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--slim") slim = std::atof(next().c_str());
    else if (a == "--backend") backend = next();
    else if (a == "--input") input = next();
    else if (a == "--golden") golden = next();
    else if (a == "--blocks") blocks = next();
    else if (a == "--random-blocks") random_blocks = true;
    else if (a == "--max-samples") max_samples = static_cast<size_t>(std::atoll(next().c_str()));
    else if (a == "--out") out_path = next();
    else
    {
      std::cerr << "parity: unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model.empty() || input.empty() || golden.empty())
  {
    std::cerr << "parity: --model, --input and --golden are required\n";
    return 2;
  }

  try
  {
    const A2Weights w = load_a2_weights(model, slim);
    auto engine = make_engine(backend, w);
    std::vector<float> in = read_raw(input), ref = read_raw(golden);
    if (in.size() != ref.size())
      throw std::runtime_error("input and golden have different lengths");
    if (max_samples && max_samples < in.size())
    {
      in.resize(max_samples);
      ref.resize(max_samples);
    }
    const double limit = parity_limit_db(w.channels);
    constexpr int kMaxBlock = 512;

    auto run = [&](const std::vector<int>& sizes_in_order, bool cycle) {
      engine->reset(kMaxBlock);
      std::vector<float> out(in.size());
      size_t pos = 0, k = 0;
      while (pos < in.size())
      {
        const int want = cycle ? sizes_in_order[k++ % sizes_in_order.size()] : sizes_in_order[0];
        const int n = static_cast<int>(std::min<size_t>(want, in.size() - pos));
        engine->process(in.data() + pos, out.data() + pos, n);
        pos += n;
      }
      return out;
    };

    int failures = 0;
    bool wrote = false;
    auto report = [&](const std::string& label, const std::vector<float>& out) {
      const double e = esr_db(out, ref);
      const bool ok = e <= limit;
      std::printf("%-12s ESR %8.1f dB  (limit %.0f)  %s\n", label.c_str(), e, limit, ok ? "ok" : "FAIL");
      failures += ok ? 0 : 1;
      if (!out_path.empty() && !wrote)
      {
        std::ofstream(out_path, std::ios::binary).write(reinterpret_cast<const char*>(out.data()),
                                                        static_cast<std::streamsize>(out.size() * sizeof(float)));
        wrote = true;
      }
    };

    std::printf("backend %s, %d channels, %zu samples\n", backend.c_str(), w.channels, in.size());
    for (int b : parse_list(blocks))
      report("block " + std::to_string(b), run({b}, false));
    if (random_blocks)
    {
      std::mt19937 rng(7);
      std::uniform_int_distribution<int> dist(1, kMaxBlock);
      std::vector<int> sizes(64);
      for (int& s : sizes)
        s = dist(rng);
      report("random", run(sizes, true));
    }
    return failures ? 1 : 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "parity: " << e.what() << "\n";
    return 2;
  }
}
