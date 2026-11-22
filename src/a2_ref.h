// Scalar float reference implementation of the A2 forward pass: one sample at a time, no tiling.
// Slow on purpose. It defines what "correct" means for the faster engines.
#pragma once

#include <memory>

#include "a2_engine.h"
#include "a2_weights.h"

namespace namturbo
{
std::unique_ptr<A2Engine> make_a2_ref(const A2Weights& weights);

// The same engine with double-precision internal arithmetic (input and output stay float): the best available
// stand-in for the exact answer, to judge how precise a float engine is. Slow.
std::unique_ptr<A2Engine> make_a2_ref_f64(const A2Weights& weights);
}
