#include "namturbo/model.h"

#include <cmath>
#include <limits>

#include "model_impl.h"
#include "namturbo/error.h"

namespace namturbo
{
Model::Model(std::unique_ptr<Impl> impl)
  : impl_(std::move(impl))
{
}

Model::~Model() = default;

std::shared_ptr<const Model> Model::load(const std::filesystem::path& file)
{
  const std::string text = read_model_file(file);
  return load(text.data(), text.size());
}

std::shared_ptr<const Model> Model::load(const void* data, std::size_t size)
{
  if (!data && size != 0)
    throw Error(ErrorCode::InvalidArgument, "the model data pointer is null");
  A2File parsed = parse_a2_file(data, size);

  auto impl = std::make_unique<Impl>();
  if (parsed.lite)
    impl->packed_lite = pack_a2_lite(*parsed.lite);
  if (parsed.full)
    impl->packed_full = pack_a2_full(*parsed.full);
  impl->info.sample_rate = parsed.meta.sample_rate;
  impl->info.sample_rate_stated = parsed.meta.sample_rate_stated;
  impl->info.loudness_db = parsed.meta.loudness_db;
  impl->info.input_level_dbu = parsed.meta.input_level_dbu;
  impl->info.output_level_dbu = parsed.meta.output_level_dbu;
  impl->info.has_lite = impl->packed_lite != nullptr;
  impl->info.has_full = impl->packed_full != nullptr;
  return std::shared_ptr<const Model>(new Model(std::move(impl)));
}

const ModelInfo& Model::info() const noexcept
{
  return impl_->info;
}

bool Model::has(Variant variant) const noexcept
{
  return variant == Variant::Lite ? impl_->info.has_lite : impl_->info.has_full;
}

double db_to_linear(double db) noexcept
{
  return std::isfinite(db) ? std::pow(10.0, db / 20.0) : 0.0;
}

double linear_to_db(double linear) noexcept
{
  if (!(linear > 0.0) || !std::isfinite(linear))
    return -std::numeric_limits<double>::infinity();
  return 20.0 * std::log10(linear);
}

std::optional<double> input_calibration_db(const ModelInfo& info, double interface_input_dbu) noexcept
{
  if (!info.input_level_dbu || !std::isfinite(*info.input_level_dbu) || !std::isfinite(interface_input_dbu))
    return std::nullopt;
  return interface_input_dbu - *info.input_level_dbu;
}

std::optional<double> output_normalization_db(const ModelInfo& info, double target_loudness_db) noexcept
{
  if (!info.loudness_db || !std::isfinite(*info.loudness_db) || !std::isfinite(target_loudness_db))
    return std::nullopt;
  return target_loudness_db - *info.loudness_db;
}

std::optional<double> output_calibration_db(const ModelInfo& info, double interface_output_dbu) noexcept
{
  if (!info.output_level_dbu || !std::isfinite(*info.output_level_dbu) || !std::isfinite(interface_output_dbu))
    return std::nullopt;
  return *info.output_level_dbu - interface_output_dbu;
}
} // namespace namturbo
