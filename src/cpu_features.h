// Run-time CPU feature detection. avx2 and fma are true only if the CPU supports them AND the operating
// system saves the 256-bit register state (XGETBV), so using the instructions is actually safe.
#pragma once

namespace namturbo
{
struct CpuFeatures
{
  bool avx2 = false;
  bool fma = false;
};

// Compiled without any AVX flags on purpose: it must be safe to call on a CPU that lacks them.
CpuFeatures detect_cpu();

// Makes detect_cpu() report the given features instead of the real ones, until cleared: for tests of what happens on a processor without
// AVX2 or FMA. Not part of the public interface; call it only while no processor is being prepared or processing.
void set_cpu_features_for_tests(CpuFeatures features);
void clear_cpu_features_for_tests();

inline bool cpu_supports_avx2_fma()
{
  const CpuFeatures f = detect_cpu();
  return f.avx2 && f.fma;
}
} // namespace namturbo
