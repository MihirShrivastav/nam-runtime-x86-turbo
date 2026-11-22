// Small helpers shared by the AVX2 engines (A2-Full and A2-Lite).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#if defined(_MSC_VER)
  #include <malloc.h>
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <xmmintrin.h>
  #define NAMTURBO_FTZ_X86 1
#endif

namespace namturbo
{
namespace engine_util
{
inline int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

// Counts, per thread, the allocations and frees of aligned buffers, which do not go through operator new. The real-time safety test
// (tests/test_realtime.cpp) reads it: the audio thread must not change it.
inline unsigned long long& aligned_buffer_events() noexcept
{
  static thread_local unsigned long long events = 0;
  return events;
}

// Zeroed, 32-byte aligned float storage.
class AlignedFloats
{
public:
  AlignedFloats() = default;
  AlignedFloats(const AlignedFloats&) = delete;
  AlignedFloats& operator=(const AlignedFloats&) = delete;
  AlignedFloats(AlignedFloats&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  AlignedFloats& operator=(AlignedFloats&& o) noexcept
  {
    if (this != &o)
    {
      release();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  ~AlignedFloats() { release(); }

  void assign_zero(size_t n)
  {
    release();
    ++aligned_buffer_events();
#if defined(_MSC_VER)
    p_ = static_cast<float*>(_aligned_malloc(n * sizeof(float), 32));
#else
    void* p = nullptr;
    if (posix_memalign(&p, 32, n * sizeof(float)) != 0)
      p = nullptr;
    p_ = static_cast<float*>(p);
#endif
    if (!p_)
      throw std::bad_alloc();
    std::memset(p_, 0, n * sizeof(float));
  }
  float* data() { return p_; }
  const float* data() const { return p_; }

private:
  void release()
  {
    if (p_)
    {
      ++aligned_buffer_events();
#if defined(_MSC_VER)
      _aligned_free(p_);
#else
      std::free(p_);
#endif
    }
    p_ = nullptr;
  }
  float* p_ = nullptr;
};

// Treat denormal floats as zero while processing, and put the caller's floating-point control state back
// afterwards. Denormal operands cost many cycles on some CPUs (notably Intel), a hostile input (a decaying
// tail from an earlier effect) can be all denormals, and 1e-38 is far below anything audible. On the AMD CPU
// used for testing denormals are not slow, so the benefit is not measurable here; the behaviour is
// still pinned by tests.
// MXCSR FTZ (bit 15) and DAZ (bit 6).
class ScopedNoDenormals
{
public:
#if defined(NAMTURBO_FTZ_X86)
  ScopedNoDenormals() : saved_(_mm_getcsr()) { _mm_setcsr(static_cast<unsigned>(saved_) | 0x8040u); }
  ~ScopedNoDenormals() { _mm_setcsr(static_cast<unsigned>(saved_)); }
#else
  ScopedNoDenormals() = default;
#endif
  ScopedNoDenormals(const ScopedNoDenormals&) = delete;
  ScopedNoDenormals& operator=(const ScopedNoDenormals&) = delete;

private:
  std::uint64_t saved_ = 0;
};
} // namespace engine_util
} // namespace namturbo
