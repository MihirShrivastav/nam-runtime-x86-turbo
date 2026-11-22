// Backend selection for the tools: the scalar reference, the float64 reference, or the AVX2 engine for the model's width.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "a2_avx2.h"
#include "a2_dispatch.h"
#include "a2_ref.h"

namespace namturbo
{
inline std::unique_ptr<A2Engine> make_engine(const std::string& backend, const A2Weights& w)
{
  if (backend == "scalar")
    return make_a2_ref(w);
  if (backend == "ref64")
    return make_a2_ref_f64(w);
  if (backend == "avx2") // the AVX2 engine for this model's width
    return make_a2_engine(w, Backend::Avx2);
  throw std::runtime_error("unknown backend: " + backend);
}
} // namespace namturbo
