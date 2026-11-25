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

// ---- gain-staging helpers (pure, allocation-free, noexcept; safe on any thread) ------------------------------------------

// Converts a gain in decibels to a linear amplitude multiplier (10^(db/20)), or 0 if db is not finite.
double db_to_linear(double db) noexcept;

// Converts a positive linear amplitude multiplier to decibels (20*log10(linear)), or -infinity if linear <= 0.
double linear_to_db(double linear) noexcept;

// Input gain in dB to match a model's capture level (ModelInfo::input_level_dbu) to an audio interface whose 0 dBFS peak
// sine corresponds to `interface_input_dbu` dBu RMS: returns interface_input_dbu - *info.input_level_dbu, or empty if the
// model does not state input_level_dbu (or either value is not finite).
std::optional<double> input_calibration_db(const ModelInfo& info, double interface_input_dbu) noexcept;

// Output gain in dB to normalize a model's stated loudness (ModelInfo::loudness_db) to `target_loudness_db` (default -18 dB):
// returns target_loudness_db - *info.loudness_db, or empty if the model does not state loudness_db (or either value is not finite).
std::optional<double> output_normalization_db(const ModelInfo& info, double target_loudness_db = -18.0) noexcept;

// Output gain in dB to match a model's capture output level (ModelInfo::output_level_dbu) to a DAC whose 0 dBFS peak sine
// corresponds to `interface_output_dbu` dBu RMS: returns *info.output_level_dbu - interface_output_dbu, or empty if the
// model does not state output_level_dbu (or either value is not finite).
std::optional<double> output_calibration_db(const ModelInfo& info, double interface_output_dbu) noexcept;
} // namespace namturbo
