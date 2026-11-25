// AVX2+FMA implementation of the A2-Full (8 channel) forward pass.
#pragma once

#include <cstdint>
#include <memory>

#include "a2_engine.h"
#include "a2_packed.h"
#include "a2_weights.h"

namespace namturbo
{
// Precondition: cpu_supports_avx2_fma(). Throws std::invalid_argument if the model is not 8 channels
// and std::runtime_error if the CPU lacks AVX2/FMA.
std::unique_ptr<A2Engine> make_a2_avx2(const A2Weights& weights);
std::unique_ptr<A2Engine> make_a2_avx2_packed(std::shared_ptr<const PackedA2Full> packed, uint32_t initial_position = 0);

// Test hook: start the running sample counter at `initial_position`, to exercise 32-bit wrap-around
// (the counter only ever indexes power-of-two rings, so output must not depend on it).
std::unique_ptr<A2Engine> make_a2_avx2_at_position(const A2Weights& weights, uint32_t initial_position);

// AVX2+FMA implementation of the A2-Lite (3 channel) forward pass, in a different layout from A2-Full: the 8 SIMD
// lanes hold 8 consecutive frames of one channel, because 3 channels would waste 5 of 8 lanes. Same preconditions
// and exceptions as make_a2_avx2, with std::invalid_argument for a model that is not 3 channels.
std::unique_ptr<A2Engine> make_a2_lite_avx2(const A2Weights& weights);
std::unique_ptr<A2Engine> make_a2_lite_avx2_packed(std::shared_ptr<const PackedA2Lite> packed, uint32_t initial_position = 0);
std::unique_ptr<A2Engine> make_a2_lite_avx2_at_position(const A2Weights& weights, uint32_t initial_position);

#ifdef NAMTURBO_PROFILE
// Opt-in profiling build only: print per-section time stamp counter totals per block.
void a2_avx2_profile_dump();
void a2_lite_avx2_profile_dump(); // prints nothing unless the Lite engine ran
#endif
} // namespace namturbo
