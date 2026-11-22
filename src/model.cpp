#include "namturbo/model.h"

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
  impl->lite = std::move(parsed.lite);
  impl->full = std::move(parsed.full);
  impl->info.sample_rate = parsed.meta.sample_rate;
  impl->info.sample_rate_stated = parsed.meta.sample_rate_stated;
  impl->info.loudness_db = parsed.meta.loudness_db;
  impl->info.input_level_dbu = parsed.meta.input_level_dbu;
  impl->info.output_level_dbu = parsed.meta.output_level_dbu;
  impl->info.has_lite = impl->lite.has_value();
  impl->info.has_full = impl->full.has_value();
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
} // namespace namturbo
