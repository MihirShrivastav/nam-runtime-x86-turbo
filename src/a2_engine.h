// The interface every A2 engine implements (the scalar reference and the AVX2 engines).
#pragma once

namespace namturbo
{
// Largest block any engine accepts. Hosts use 32..8192; the cap keeps ring-buffer sizes and memory bounded.
constexpr int kMaxEngineBlock = 65536;

class A2Engine
{
public:
  virtual ~A2Engine() = default;

  // Clear all state, then prewarm the way NAM Core does: feed prewarm_samples() frames of silence.
  // max_block is the largest n that process() will be called with, 1..kMaxEngineBlock, otherwise
  // std::invalid_argument. May allocate.
  virtual void reset(int max_block) = 0;

  // Mono, 1 <= n <= max_block, otherwise std::invalid_argument (also before the first reset()). `in` and
  // `out` may be the same buffer (in-place) but must not otherwise overlap. Must not allocate or block.
  virtual void process(const float* in, float* out, int n) = 0;

  virtual int prewarm_samples() const = 0;
};
} // namespace namturbo
