// Load a model through NAM Core, either its A2 fast path or its generic WaveNet. Shared by the tools
// that need NAM Core itself (golden output, benchmark baselines). Built the way NAM Core's own
// tools/bench_a2_fast.cpp builds them.
#pragma once

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/dsp.h"
#include "NAM/wavenet/a2_fast.h"
#include "NAM/wavenet/model.h"

namespace namturbo
{
// Same selection rule as NAM Core's ContainerModel::SetSlimmableSize: the first submodel whose
// max_value exceeds `slim`, else the last. A plain WaveNet file is returned as is.
inline nlohmann::json select_nam_model(const nlohmann::json& file, double slim)
{
  if (file.value("architecture", std::string()) != "SlimmableContainer")
    return file;
  const auto& subs = file.at("config").at("submodels");
  size_t idx = subs.size() - 1;
  for (size_t i = 0; i < subs.size(); ++i)
    if (slim < subs[i].at("max_value").get<double>())
    {
      idx = i;
      break;
    }
  return subs[idx].at("model");
}

struct NamCoreModel
{
  std::unique_ptr<nam::DSP> dsp;
  double sample_rate = 48000.0;
  int channels = 0; // A2 width if the model is A2-shaped, else 0
};

// `which` is "fast" (A2 fast path; throws if the model is not A2-shaped) or "generic".
inline NamCoreModel make_nam_core_dsp(const std::string& path, double slim, const std::string& which)
{
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("cannot open model " + path);
  nlohmann::json file;
  f >> file;
  const nlohmann::json model = select_nam_model(file, slim);
  if (model.value("architecture", std::string()) != "WaveNet")
    throw std::runtime_error("selected model is not a WaveNet");
  const nlohmann::json& cfg = model.at("config");
  std::vector<float> weights = model.at("weights").get<std::vector<float>>();

  NamCoreModel m;
  if (model.contains("sample_rate") && !model["sample_rate"].is_null())
    m.sample_rate = model["sample_rate"].get<double>();
  if (!nam::wavenet::a2_fast::is_a2_shape(cfg, &m.channels))
    m.channels = 0;
  if (which == "fast")
  {
    if (m.channels == 0)
      throw std::runtime_error("model is not A2-shaped, so it has no fast path");
    m.dsp = nam::wavenet::a2_fast::create_a2_fast_config(cfg, m.sample_rate)->create(std::move(weights), m.sample_rate);
  }
  else if (which == "generic")
  {
    auto generic = nam::wavenet::parse_config_json(cfg, m.sample_rate);
    m.dsp = generic.create(std::move(weights), m.sample_rate);
  }
  else
    throw std::runtime_error("path must be fast or generic");
  return m;
}
} // namespace namturbo
