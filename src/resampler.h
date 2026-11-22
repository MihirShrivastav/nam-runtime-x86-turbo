// Sample-rate conversion between the rate of the host and the rate of the model, by rational polyphase FIR filters.
// Internal; the public interface is Processor.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "engine_util.h"
#include "namturbo/processor.h"

namespace namturbo
{
// ---- filter design (exposed for the tests) ---------------------------------------------------------------------------

// Number of taps (odd) of a Kaiser-window low-pass with the given stop-band attenuation (dB) and transition width
// (in cycles per sample of the rate at which the filter is designed).
int kaiser_length(double attenuation_db, double transition_cycles_per_sample);

// Kaiser-window low-pass of odd length n, linear phase (symmetric about index (n - 1) / 2), cut-off in cycles per sample,
// scaled so that the taps add up to dc_gain.
std::vector<double> design_lowpass(int n, double cutoff_cycles_per_sample, double attenuation_db, double dc_gain);

// ---- one conversion stage ---------------------------------------------------------------------------------------------

// Upsampling by `up`, low-pass filtering with `taps` (designed at `up` times the input rate, already scaled by `up`), and
// downsampling by `down`, computed as a polyphase filter: only the output samples that are kept are calculated. An output sample is
// produced as soon as the input sample it depends on has arrived, so the output for a signal does not depend on how the signal is
// cut into blocks, and the arithmetic of an output sample is always the same.
class PolyphaseStage
{
public:
  PolyphaseStage(int up, int down, const std::vector<double>& taps, int max_input_block);

  // Consumes n input samples (any n >= 0) and writes the output samples that become available; `out` must have room for
  // max_output(n) samples. Returns the number written.
  int process(const float* in, int n, float* out);
  int max_output(int n_in) const { return static_cast<int>((static_cast<std::int64_t>(n_in) * up_ + down_ - 1) / down_) + 1; }
  void reset();
  int taps_per_phase() const { return taps_per_phase_; }

private:
  int process_piece(const float* in, int n, float* out);

  int up_, down_, taps_per_phase_;
  int step_whole_ = 0, step_fraction_ = 0; // down_ / up_ as a whole part and a remainder
  int max_input_block_;
  engine_util::AlignedFloats coefficients_; // up_ rows of taps_per_phase_ floats, in the order of the input window
  std::vector<float> window_;               // the last taps_per_phase_ - 1 input samples, then the current block
  std::int64_t next_input_ = 0;             // input index (relative to the start of the current block) of the next output
  int phase_ = 0;                           // phase of the next output
};

// ---- the pair of stages around the engine -----------------------------------------------------------------------------

// host rate -> model rate (stage 1), the engine, model rate -> host rate (stage 2), and a FIFO that makes the number of output samples
// of each block equal to the number of input samples. Mono. Not thread safe.
class RateConverter
{
public:
  // Throws Error(Unsupported) if the ratio is too complex or a rate is not a whole number of Hz.
  RateConverter(int host_rate, int model_rate, int max_block, ResamplerQuality quality);

  // Total delay in host samples, exact: the delay of the filters (a whole number by construction) plus the FIFO start.
  int latency_samples() const { return filter_delay_ + fifo_start_; }
  // Largest number of model-rate samples to_model() can return for a block of max_block host samples.
  int max_model_block() const { return max_model_block_; }

  // Converts n host samples (n <= max_block) to model rate and returns how many model samples were written (possibly 0).
  int to_model(const float* host, int n, float* model);
  // Takes m model samples back to host rate and writes exactly n host samples, the n of the matching to_model() call.
  void to_host(const float* model, int m, float* host, int n);
  void reset();

  // Delay of stage 1 in samples at the model rate (not a whole number in general): the signal that enters the engine is the host
  // signal delayed by this much. Used by the tests; useful to anyone who compares paths that are sampled at different phases.
  double stage1_delay_model_samples() const { return static_cast<double>(center1_) / down_; }

  // For the tests.
  PolyphaseStage& stage1() { return *stage1_; }
  int up() const { return up_; }
  int down() const { return down_; }
  int taps_per_phase_stage1() const { return stage1_->taps_per_phase(); }
  int taps_per_phase_stage2() const { return stage2_->taps_per_phase(); }

private:
  int up_ = 1, down_ = 1; // stage 1: model rate / host rate = up / down
  int max_model_block_ = 0;
  int filter_delay_ = 0, fifo_start_ = 0, center1_ = 0;
  std::unique_ptr<PolyphaseStage> stage1_, stage2_;
  std::vector<float> stage2_out_;
  std::vector<float> fifo_;      // ring buffer, power-of-two size
  std::size_t fifo_mask_ = 0, fifo_read_ = 0, fifo_count_ = 0;
};
} // namespace namturbo
