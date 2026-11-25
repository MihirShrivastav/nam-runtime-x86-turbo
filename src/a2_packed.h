// Immutable, 32-byte aligned prepacked A2 weights shared across processors and oversampling phases.
// Header-only and free of AVX2 intrinsics so Model::load() stays safe on CPUs without AVX2/FMA.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "a2_weights.h"
#include "engine_util.h"

namespace namturbo
{
struct PackedA2FullLayer
{
  int K = 0, D = 0, lookback = 0;
  engine_util::AlignedFloats W; // K*64: [tap][in][out], 32-byte aligned
  float conv_b[8] = {}, mixin[8] = {}, l1x1[64] = {}, l1x1_b[8] = {};
};

struct PackedA2Full
{
  float rechannel[8] = {};
  std::array<PackedA2FullLayer, kLayers> layers{};
  float head_w[kHeadK][8] = {};
  float head_b = 0.0f, head_scale = 0.0f;
};

struct PackedA2LiteLayer
{
  static constexpr int C = 3;
  static constexpr int kMaxTaps = 15;
  int K = 0, D = 0, lookback = 0;
  float W[kMaxTaps * C * C] = {}; // [tap][in][out]
  float conv_b[C] = {}, mixin[C] = {}, l1x1[C * C] = {}, l1x1_b[C] = {};
};

struct PackedA2Lite
{
  static constexpr int C = 3;
  float rechannel[C] = {};
  std::array<PackedA2LiteLayer, kLayers> layers{};
  float head_w[kHeadK][C] = {};
  float head_b = 0.0f, head_scale = 0.0f;
};

inline std::shared_ptr<const PackedA2Full> pack_a2_full(const A2Weights& w)
{
  if (w.channels != 8)
    throw std::invalid_argument("pack_a2_full: A2-Full (8 channels) only");
  auto p = std::make_shared<PackedA2Full>();
  std::copy(w.rechannel.begin(), w.rechannel.begin() + 8, p->rechannel);
  for (int l = 0; l < kLayers; ++l)
  {
    const A2Layer& src = w.layers[l];
    PackedA2FullLayer& L = p->layers[l];
    L.K = src.k;
    L.D = src.d;
    L.lookback = (src.k - 1) * src.d;
    L.W.assign_zero(static_cast<std::size_t>(src.k) * 64);
    std::copy(src.conv.begin(), src.conv.end(), L.W.data());
    std::copy(src.conv_b.begin(), src.conv_b.begin() + 8, L.conv_b);
    std::copy(src.mixin.begin(), src.mixin.begin() + 8, L.mixin);
    std::copy(src.l1x1.begin(), src.l1x1.end(), L.l1x1);
    std::copy(src.l1x1_b.begin(), src.l1x1_b.begin() + 8, L.l1x1_b);
  }
  for (int k = 0; k < kHeadK; ++k)
    std::copy(w.head_w[k].begin(), w.head_w[k].begin() + 8, p->head_w[k]);
  p->head_b = w.head_b;
  p->head_scale = w.head_scale;
  return p;
}

inline std::shared_ptr<const PackedA2Lite> pack_a2_lite(const A2Weights& w)
{
  constexpr int C = PackedA2Lite::C;
  if (w.channels != C)
    throw std::invalid_argument("pack_a2_lite: A2-Lite (3 channels) only");
  auto p = std::make_shared<PackedA2Lite>();
  std::copy(w.rechannel.begin(), w.rechannel.begin() + C, p->rechannel);
  for (int l = 0; l < kLayers; ++l)
  {
    const A2Layer& src = w.layers[l];
    PackedA2LiteLayer& L = p->layers[l];
    if (src.k > PackedA2LiteLayer::kMaxTaps || src.conv.size() != static_cast<std::size_t>(src.k) * C * C)
      throw std::invalid_argument("pack_a2_lite: unexpected layer shape");
    L.K = src.k;
    L.D = src.d;
    L.lookback = (src.k - 1) * src.d;
    std::copy(src.conv.begin(), src.conv.end(), L.W);
    std::copy(src.conv_b.begin(), src.conv_b.begin() + C, L.conv_b);
    std::copy(src.mixin.begin(), src.mixin.begin() + C, L.mixin);
    std::copy(src.l1x1.begin(), src.l1x1.begin() + C * C, L.l1x1);
    std::copy(src.l1x1_b.begin(), src.l1x1_b.begin() + C, L.l1x1_b);
  }
  for (int k = 0; k < kHeadK; ++k)
    std::copy(w.head_w[k].begin(), w.head_w[k].begin() + C, p->head_w[k]);
  p->head_b = w.head_b;
  p->head_scale = w.head_scale;
  return p;
}
} // namespace namturbo
