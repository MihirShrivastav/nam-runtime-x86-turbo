#include "a2_dispatch.h"

#include <stdexcept>

#include "a2_avx2.h"
#include "a2_ref.h"

namespace namturbo
{
Backend choose_backend(const CpuFeatures& cpu, Backend requested)
{
  const bool avx2_ok = cpu.avx2 && cpu.fma;
  switch (requested)
  {
    case Backend::Scalar: return Backend::Scalar;
    case Backend::Avx2:
      if (!avx2_ok)
        throw std::runtime_error("AVX2 backend requested, but this CPU or OS does not support AVX2 and FMA");
      return Backend::Avx2;
    case Backend::Auto: break;
  }
  return avx2_ok ? Backend::Avx2 : Backend::Scalar;
}

std::unique_ptr<A2Engine> make_a2_engine(const A2Weights& weights, Backend requested, Backend* chosen)
{
  Backend b = choose_backend(detect_cpu(), requested);
  if (b == Backend::Avx2 && weights.channels != 8 && weights.channels != 3)
  {
    if (requested == Backend::Avx2)
      throw std::invalid_argument("the AVX2 backend runs A2-Full (8 channels) and A2-Lite (3 channels) only");
    b = Backend::Scalar;
  }
  if (chosen)
    *chosen = b;
  if (b == Backend::Scalar)
    return make_a2_ref(weights);
  return weights.channels == 8 ? make_a2_avx2(weights) : make_a2_lite_avx2(weights);
}
} // namespace namturbo
