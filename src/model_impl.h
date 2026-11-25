// The inside of namturbo::Model, shared by model.cpp and processor.cpp. Not a public header.
#pragma once

#include <memory>

#include "a2_packed.h"
#include "namturbo/model.h"

namespace namturbo
{
struct Model::Impl
{
  std::shared_ptr<const PackedA2Lite> packed_lite;
  std::shared_ptr<const PackedA2Full> packed_full;
  ModelInfo info;
};
} // namespace namturbo
