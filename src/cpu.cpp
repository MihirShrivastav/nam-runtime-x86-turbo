#include "namturbo/cpu.h"

#include <string>

#include "cpu_features.h"
#include "namturbo/version.h"

namespace namturbo
{
CpuSupport check_cpu()
{
  const CpuFeatures f = detect_cpu();
  if (f.avx2 && f.fma)
    return {true, std::string()};
  std::string missing;
  if (!f.avx2)
    missing = "AVX2";
  if (!f.fma)
    missing += (missing.empty() ? "" : " and ") + std::string("FMA");
  return {false, "this processor, or its operating system, does not support " + missing + ", which namturbo requires"};
}

const char* version_string() noexcept
{
  static const std::string text = std::to_string(kVersion.major) + "." + std::to_string(kVersion.minor) + "." + std::to_string(kVersion.patch) +
                                  (kPreRelease[0] != '\0' ? std::string("-") + kPreRelease : std::string());
  return text.c_str();
}
} // namespace namturbo
