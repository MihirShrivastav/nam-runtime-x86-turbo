#include "cpu_features.h"

#include <atomic>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define NAMTURBO_X86 1
  // GCC and clang (including clang-cl) via <cpuid.h> and an inline XGETBV: no dependence on the compiler
  // runtime's cpu-model symbols, which clang-cl does not link. MSVC via its intrinsics.
  #if defined(__clang__) || defined(__GNUC__)
    #include <cpuid.h>
  #elif defined(_MSC_VER)
    #include <intrin.h>
  #endif
#endif

namespace namturbo
{
namespace
{
// -1: no override; otherwise bit 0 is avx2 and bit 1 is fma. For tests of the failure path on a machine that has both.
std::atomic<int> g_override{-1};

CpuFeatures detect_hardware()
{
  CpuFeatures f;
#if defined(NAMTURBO_X86)
  bool osxsave = false, avx = false, fma = false, avx2 = false;
  #if defined(__clang__) || defined(__GNUC__)
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) || eax < 7)
    return f;
  __get_cpuid(1, &eax, &ebx, &ecx, &edx);
  osxsave = (ecx & (1u << 27)) != 0;
  avx = (ecx & (1u << 28)) != 0;
  fma = (ecx & (1u << 12)) != 0;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  avx2 = (ebx & (1u << 5)) != 0;
  bool os_ymm = false;
  if (osxsave && avx)
  {
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    os_ymm = (lo & 0x6u) == 0x6u; // XMM and YMM state enabled by the OS
  }
  #else
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7)
    return f;
  __cpuid(r, 1);
  osxsave = (r[2] & (1 << 27)) != 0;
  avx = (r[2] & (1 << 28)) != 0;
  fma = (r[2] & (1 << 12)) != 0;
  __cpuidex(r, 7, 0);
  avx2 = (r[1] & (1 << 5)) != 0;
  bool os_ymm = false;
  if (osxsave && avx)
    os_ymm = (_xgetbv(0) & 0x6) == 0x6;
  #endif
  f.avx2 = avx2 && avx && os_ymm;
  f.fma = fma && avx && os_ymm;
#endif
  return f; // non-x86 targets have neither
}
} // namespace

CpuFeatures detect_cpu()
{
  const int o = g_override.load(std::memory_order_relaxed);
  if (o >= 0)
  {
    CpuFeatures f;
    f.avx2 = (o & 1) != 0;
    f.fma = (o & 2) != 0;
    return f;
  }
  return detect_hardware();
}

void set_cpu_features_for_tests(CpuFeatures features)
{
  g_override.store((features.avx2 ? 1 : 0) | (features.fma ? 2 : 0), std::memory_order_relaxed);
}

void clear_cpu_features_for_tests() { g_override.store(-1, std::memory_order_relaxed); }
} // namespace namturbo
