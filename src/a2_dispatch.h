// Run-time choice of A2 engine: the fastest one this CPU, this OS and this model can use.
#pragma once

#include <memory>

#include "a2_engine.h"
#include "a2_weights.h"
#include "cpu_features.h"

namespace namturbo
{
enum class Backend
{
  Auto,
  Scalar, // the reference: always works, far too slow for production
  Avx2
};

// Pure. Auto gives Avx2 when both AVX2 and FMA are usable, else Scalar. An explicit Avx2 request on a CPU
// without them throws std::runtime_error (asking for something that cannot run is an error, not a fallback).
Backend choose_backend(const CpuFeatures& cpu, Backend requested);

// True for backends meant for production. A caller that gets a non-accelerated backend should prefer NAM
// Core's own DSP for that model: it is faster than the reference.
inline bool is_accelerated(Backend b) { return b == Backend::Avx2; }

// Throws std::runtime_error as choose_backend does, and std::invalid_argument for an explicit Avx2 request
// on a model it cannot run (the AVX2 engines are A2-Full and A2-Lite). With Auto, such a model gets the Scalar engine.
// `chosen`, if given, receives the backend actually used.
std::unique_ptr<A2Engine> make_a2_engine(const A2Weights& weights, Backend requested, Backend* chosen = nullptr);
} // namespace namturbo
