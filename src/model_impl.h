// The inside of namturbo::Model, shared by model.cpp and processor.cpp. Not a public header.
#pragma once

#include <optional>

#include "a2_weights.h"
#include "namturbo/model.h"

namespace namturbo
{
struct Model::Impl
{
  std::optional<A2Weights> lite, full;
  ModelInfo info;
};
} // namespace namturbo
