#include "a2_weights.h"

#include <cmath>
#include <fstream>
#include <iterator>

#include "json.hpp"
#include "namturbo/error.h"

namespace namturbo
{
namespace
{
using json = nlohmann::json;

// The largest model file that is read: A2 files are well under 1 MB, so this only stops absurd input.
constexpr std::size_t kMaxFileBytes = 64u * 1024u * 1024u;

[[noreturn]] void not_a2(const std::string& what)
{
  throw Error(ErrorCode::NotA2, "this model is not an A2 model: " + what);
}

void require(bool ok, const std::string& what)
{
  if (!ok)
    not_a2(what);
}

// FiLM and head1x1 entries are absent, null, or {"active": false}.
bool inactive(const json& layer, const char* key)
{
  auto it = layer.find(key);
  return it == layer.end() || it->is_null() || !it->value("active", false);
}

// Returns the channel count (3 or 8) after checking every architectural knob.
int validate_a2_shape(const json& cfg)
{
  auto layers_it = cfg.find("layers");
  require(layers_it != cfg.end() && layers_it->is_array() && layers_it->size() == 1, "it does not have exactly one layer array");
  auto head_it = cfg.find("head");
  require(head_it == cfg.end() || head_it->is_null(), "it has a post-stack head");
  auto cond_it = cfg.find("condition_dsp");
  require(cond_it == cfg.end() || cond_it->is_null(), "it has a condition network");
  require(cfg.contains("head_scale") && cfg["head_scale"].is_number(), "head_scale is missing");

  const json& la = (*layers_it)[0];
  require(la.value("input_size", 0) == 1 && la.value("condition_size", 0) == 1, "input_size and condition_size are not 1");
  const int ch = la.value("channels", 0);
  require(ch == la.value("bottleneck", 0), "the bottleneck does not equal the channel count");

  auto ks = la.find("kernel_sizes");
  auto ds = la.find("dilations");
  require(ks != la.end() && ks->is_array() && ks->size() == kLayers, "it does not have 23 kernel sizes");
  require(ds != la.end() && ds->is_array() && ds->size() == kLayers, "it does not have 23 dilations");
  for (int i = 0; i < kLayers; ++i)
  {
    require((*ks)[i].is_number_integer() && (*ks)[i].get<int>() == kKernelSizes[i], "the kernel sizes do not match the A2 pattern");
    require((*ds)[i].is_number_integer() && (*ds)[i].get<int>() == kDilations[i], "the dilations do not match the A2 pattern");
  }

  auto act = la.find("activation");
  require(act != la.end() && act->is_array() && act->size() == kLayers, "it does not have 23 activations");
  for (const auto& a : *act)
    require(a.is_object() && a.value("type", std::string()) == "LeakyReLU" &&
              std::fabs(a.value("negative_slope", 0.0) - 0.01) < 1e-6,
            "an activation is not LeakyReLU with slope 0.01");

  auto gm = la.find("gating_mode");
  if (gm != la.end() && !gm->is_null())
    for (const auto& g : *gm)
      require(g.is_string() && g.get<std::string>() == "none", "it uses gating");

  for (const char* key : {"head1x1", "conv_pre_film", "conv_post_film", "input_mixin_pre_film",
                          "input_mixin_post_film", "activation_pre_film", "activation_post_film",
                          "layer1x1_post_film", "head1x1_post_film"})
    require(inactive(la, key), std::string(key) + " is active");

  auto l1 = la.find("layer1x1");
  require(l1 != la.end() && l1->value("active", false) && l1->value("groups", 1) == 1, "layer1x1 is not active and ungrouped");
  require(la.value("groups_input", 1) == 1 && la.value("groups_input_mixin", 1) == 1, "a convolution is grouped");
  auto sl = la.find("slimmable");
  require(sl == la.end() || sl->is_null(), "it slims by channels");

  auto hd = la.find("head");
  require(hd != la.end() && hd->value("out_channels", 0) == 1 && hd->value("kernel_size", 0) == kHeadK &&
            hd->value("bias", false),
          "the head is not a convolution with 1 output, kernel 16 and a bias");

  // Everything matches the A2 pattern; only the size can still be one this library does not run.
  if (ch != 3 && ch != 8)
    throw Error(ErrorCode::UnsupportedShape, "this is an A2 model with " + std::to_string(ch) +
                                               " channels; this library runs A2-Lite (3 channels) and A2-Full (8 channels)");
  return ch;
}

// One WaveNet object of a file -> weights. Throws Error.
A2Weights parse_wavenet(const json& model)
{
  const std::string arch = model.value("architecture", std::string());
  if (arch != "WaveNet")
    throw Error(ErrorCode::NotA2, "this model is not an A2 model: its architecture is \"" + arch + "\", not WaveNet");

  const json& cfg = model.at("config");
  A2Weights out;
  out.channels = validate_a2_shape(cfg);
  const int C = out.channels;

  const std::vector<float> flat = model.at("weights").get<std::vector<float>>();
  for (float w : flat)
    if (!std::isfinite(w))
      throw Error(ErrorCode::InvalidFile, "the model data contains a value that is not a finite number (or is too large for a float)");
  size_t pos = 0;
  auto take = [&]() -> float {
    if (pos >= flat.size())
      throw Error(ErrorCode::InvalidFile, "the model data is too short for an A2 model with " + std::to_string(C) + " channels");
    return flat[pos++];
  };

  for (int i = 0; i < C; ++i)
    out.rechannel[i] = take();
  for (int li = 0; li < kLayers; ++li)
  {
    A2Layer& L = out.layers[li];
    L.k = kKernelSizes[li];
    L.d = kDilations[li];
    L.conv.assign(size_t(L.k) * C * C, 0.0f);
    for (int o = 0; o < C; ++o)
      for (int in = 0; in < C; ++in)
        for (int t = 0; t < L.k; ++t)
          L.conv[size_t(t) * C * C + in * C + o] = take();
    for (int o = 0; o < C; ++o)
      L.conv_b[o] = take();
    for (int o = 0; o < C; ++o)
      L.mixin[o] = take();
    for (int o = 0; o < C; ++o)
      for (int in = 0; in < C; ++in)
        L.l1x1[in * C + o] = take();
    for (int o = 0; o < C; ++o)
      L.l1x1_b[o] = take();
  }
  for (int in = 0; in < C; ++in)
    for (int t = 0; t < kHeadK; ++t)
      out.head_w[t][in] = take();
  out.head_b = take();
  out.head_scale = take();

  if (pos != flat.size())
    throw Error(ErrorCode::InvalidFile, "the model data has " + std::to_string(flat.size() - pos) +
                                          " more values than an A2 model with " + std::to_string(C) + " channels needs");
  return out;
}

std::optional<double> number_of(const json& obj, const char* key)
{
  auto it = obj.find(key);
  if (it != obj.end() && it->is_number())
    return it->get<double>();
  return std::nullopt;
}

json parse_json(const void* data, std::size_t size)
{
  if (size == 0)
    throw Error(ErrorCode::InvalidFile, "the file is empty");
  if (size > kMaxFileBytes)
    throw Error(ErrorCode::InvalidFile, "the file is larger than " + std::to_string(kMaxFileBytes / (1024 * 1024)) + " MB, too large for a model");
  const char* first = static_cast<const char*>(data);
  try
  {
    return json::parse(first, first + size);
  }
  catch (const json::exception& e)
  {
    throw Error(ErrorCode::InvalidFile, std::string("the file is not valid JSON: ") + e.what());
  }
}

// The "metadata" object and the sample rate of a file (or of one of its submodels).
void read_meta(const json& file, A2Metadata& meta)
{
  if (auto sr = number_of(file, "sample_rate"); !meta.sample_rate_stated && sr && *sr > 0.0)
  {
    meta.sample_rate = *sr;
    meta.sample_rate_stated = true;
  }
  auto md = file.find("metadata");
  if (md != file.end() && md->is_object())
  {
    if (!meta.loudness_db)
      meta.loudness_db = number_of(*md, "loudness");
    if (!meta.input_level_dbu)
      meta.input_level_dbu = number_of(*md, "input_level_dbu");
    if (!meta.output_level_dbu)
      meta.output_level_dbu = number_of(*md, "output_level_dbu");
  }
}
} // namespace

std::string read_model_file(const std::filesystem::path& path)
{
  // A path that cannot be shown in the narrow character set of the system must not make the error itself fail.
  auto shown = [&] {
    try
    {
      return path.string();
    }
    catch (...)
    {
      return std::string("the model file");
    }
  };
  std::ifstream is(path, std::ios::binary);
  if (!is)
    throw Error(ErrorCode::Io, "cannot open " + shown());
  std::string text((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
  if (is.bad())
    throw Error(ErrorCode::Io, "cannot read " + shown());
  return text;
}

A2File parse_a2_file(const void* data, std::size_t size)
{
  const json file = parse_json(data, size);
  if (!file.is_object())
    throw Error(ErrorCode::InvalidFile, "the file is JSON but not a NAM model file");
  try
  {
    A2File out;
    read_meta(file, out.meta);
    const std::string arch = file.value("architecture", std::string());
    if (arch == "SlimmableContainer")
    {
      const json& subs = file.at("config").at("submodels");
      if (!subs.is_array() || subs.empty())
        throw Error(ErrorCode::InvalidFile, "the slimmable model has no submodels");
      std::optional<Error> first_error;
      for (const json& s : subs)
      {
        const json& m = s.at("model");
        try
        {
          A2Weights w = parse_wavenet(m);
          std::optional<A2Weights>& slot = (w.channels == 3) ? out.lite : out.full;
          if (!slot)
            slot = std::move(w);
          read_meta(m, out.meta); // fills only what the container itself did not state
        }
        catch (const Error& e)
        {
          if (!first_error)
            first_error = e;
        }
      }
      if (!out.lite && !out.full)
        throw *first_error;
    }
    else
    {
      A2Weights w = parse_wavenet(file);
      (w.channels == 3 ? out.lite : out.full) = std::move(w);
    }
    return out;
  }
  catch (const json::exception& e)
  {
    throw Error(ErrorCode::InvalidFile, std::string("the model file is malformed: ") + e.what());
  }
}

A2Weights load_a2_weights(const std::string& nam_path, double slim)
{
  const std::string text = read_model_file(nam_path);
  const json file = parse_json(text.data(), text.size());
  try
  {
    const json* model = &file;
    if (file.value("architecture", std::string()) == "SlimmableContainer")
    {
      const json& subs = file.at("config").at("submodels");
      if (!subs.is_array() || subs.empty())
        throw Error(ErrorCode::InvalidFile, "the slimmable model has no submodels");
      size_t idx = subs.size() - 1;
      for (size_t i = 0; i < subs.size(); ++i)
        if (slim < subs[i].at("max_value").get<double>())
        {
          idx = i;
          break;
        }
      model = &subs[idx].at("model");
    }
    return parse_wavenet(*model);
  }
  catch (const json::exception& e)
  {
    throw Error(ErrorCode::InvalidFile, std::string("the model file is malformed: ") + e.what());
  }
}
} // namespace namturbo
