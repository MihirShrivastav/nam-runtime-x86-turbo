// A2 WaveNet weights in a layout convenient for kernels.
//
// Shape (checked on load): one layer array, 23 layers, channels 3 or 8 with bottleneck == channels,
// kernel sizes and dilations of NAM Core's A2 fast path, LeakyReLU(0.01), no gating, no FiLM, no
// head1x1, layer1x1 active, head rechannel conv k=16 with bias, no condition DSP, no post-stack head.
//
// Weight stream order (NAM Core, a2_fast.cpp _load_weights): rechannel; per layer: conv
// (for out: for in: for tap), conv bias, input mixin, layer1x1 (for out: for in), its bias; then the
// head conv (for in: for tap), head bias, head scale.
#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace namturbo
{
constexpr int kLayers = 23;
constexpr int kHeadK = 16;
inline constexpr std::array<int, kLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};
inline constexpr std::array<int, kLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

// Samples of history one output depends on: 1 + sum((k-1)*d) + (head kernel - 1). Also the number of
// silent samples NAM Core feeds a model to prewarm it.
constexpr int a2_receptive_field()
{
  int r = 1;
  for (int i = 0; i < kLayers; ++i)
    r += (kKernelSizes[i] - 1) * kDilations[i];
  return r + (kHeadK - 1);
}
static_assert(a2_receptive_field() == 6347, "A2 receptive field is 6347 samples");

struct A2Layer
{
  int k = 0, d = 0;
  std::vector<float> conv; // [tap*C*C + in*C + out], k*C*C floats
  std::array<float, 8> conv_b{}, mixin{};
  std::array<float, 64> l1x1{}; // [in*C + out]
  std::array<float, 8> l1x1_b{};
};

struct A2Weights
{
  int channels = 0; // 3 or 8
  std::array<float, 8> rechannel{};
  std::array<A2Layer, kLayers> layers{};
  std::array<std::array<float, 8>, kHeadK> head_w{}; // [tap][in]
  float head_b = 0.0f, head_scale = 0.0f;
};

// What a model file states about itself, besides the weights.
struct A2Metadata
{
  double sample_rate = 48000.0; // the rate the model was trained at; 48000 if the file states none
  bool sample_rate_stated = false;
  std::optional<double> loudness_db, input_level_dbu, output_level_dbu;
};

// Both sizes of a model file: a slimmable file has both, a plain file has the one it contains.
struct A2File
{
  std::optional<A2Weights> lite, full; // 3 channels, 8 channels
  A2Metadata meta;
};

// The contents of a model file that is in memory. A slimmable file may hold sizes that are not A2-Lite or A2-Full; they are
// ignored if at least one size can be run. Throws namturbo::Error (a std::runtime_error): InvalidFile, NotA2, UnsupportedShape.
A2File parse_a2_file(const void* data, std::size_t size);

// The bytes of a file. Throws namturbo::Error(Io).
std::string read_model_file(const std::filesystem::path& path);

// Loads a plain WaveNet .nam, or a SlimmableContainer .nam. For a container, `slim` selects the
// submodel as NAM Core does: the first submodel whose max_value exceeds it, else the last.
// Throws namturbo::Error (a std::runtime_error) if the file is unreadable, the model is not A2-shaped, or the weight
// stream is not consumed exactly.
A2Weights load_a2_weights(const std::string& nam_path, double slim);
} // namespace namturbo
