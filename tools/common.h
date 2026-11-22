// Small helpers shared by the tools and tests: raw float32 I/O and ESR.
#pragma once

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace namturbo
{
inline std::vector<float> read_raw(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const size_t n = static_cast<size_t>(f.tellg()) / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(float)));
  return v;
}

// 10*log10( mean((y-ref)^2) / mean(ref^2) ). Exactly equal signals give -300 (instead of -inf).
inline double esr_db(const std::vector<float>& y, const std::vector<float>& ref)
{
  if (y.size() != ref.size() || ref.empty())
    throw std::runtime_error("esr_db: size mismatch or empty");
  double err = 0.0, pow = 0.0;
  for (size_t i = 0; i < ref.size(); ++i)
  {
    const double d = double(y[i]) - double(ref[i]);
    err += d * d;
    pow += double(ref[i]) * double(ref[i]);
  }
  if (err == 0.0)
    return -300.0;
  return 10.0 * std::log10(err / pow);
}
} // namespace namturbo
