// A loaded A2 model: the network weights, read-only, shareable between threads and processors.
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>

namespace namturbo
{
// The two sizes of an A2 model: A2-Lite has 3 channels and A2-Full has 8. A slimmable file contains both; a plain file
// contains one of them.
enum class Variant
{
  Lite,
  Full
};

struct ModelInfo
{
  double sample_rate = 48000.0;      // the rate the model was trained at; 48000 if the file does not state one
  bool sample_rate_stated = false;   // false if the file has no sample rate
  std::optional<double> loudness_db;      // from the metadata of the file, if present
  std::optional<double> input_level_dbu;  // idem
  std::optional<double> output_level_dbu; // idem
  bool has_lite = false;             // which variants the file contains
  bool has_full = false;
};

// Immutable after loading: every member function can be called from any thread at the same time.
class Model
{
public:
  // Reads a .nam file. Throws Error: Io, InvalidFile, NotA2 (another architecture, or a WaveNet that is not an A2 model) or
  // UnsupportedShape (an A2 model of a size that is not A2-Lite or A2-Full).
  static std::shared_ptr<const Model> load(const std::filesystem::path& file);
  // The same for a file that is already in memory.
  static std::shared_ptr<const Model> load(const void* data, std::size_t size);

  const ModelInfo& info() const noexcept;
  bool has(Variant variant) const noexcept;

  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Processor;
};
} // namespace namturbo
