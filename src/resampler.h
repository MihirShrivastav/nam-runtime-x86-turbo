// Sample-rate conversion and oversampling by linear-phase polyphase and half-band FIR filters (AVX2+FMA).
// Internal; the public interface is Processor.
#pragma once

#include <cstddef>
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

// Kaiser-window half-band low-pass of length 4*M - 1 (center index 2*M - 1 is odd, center tap = 0.5, all other even-offset
// taps are identically 0, odd-offset taps are symmetric and sum to 0.5). pass_fraction is the flat pass-band edge as a
// fraction of the 1x Nyquist frequency (0 < pass_fraction < 1).
std::vector<double> design_halfband(double attenuation_db, double pass_fraction);

// ---- half-band 2x primitives (AVX2+FMA, symmetric) -------------------------------------------------------------------

// 1x -> 2x half-band upsampler using a symmetric half-band filter of length 4*M - 1 (exact delay 2*M - 1 at 2x rate).
// Even outputs y[2n] are the symmetric FIR interpolation; odd outputs y[2n+1] are a pure delay x[n - (M - 1)].
class HalfBandUp2x
{
public:
  HalfBandUp2x(const std::vector<double>& taps, int max_input_block);

  // Writes n interpolated even samples to even_out and n delayed odd samples to odd_out (no interleaving).
  void process_split(const float* in, int n, float* even_out, float* odd_out);
  // Writes 2*n interleaved samples (even, odd, even, odd, ...) to out2x.
  void process_interleaved(const float* in, int n, float* out2x);
  void reset();

  int half_taps() const { return M_; }
  int delay_2x_samples() const { return 2 * M_ - 1; }

private:
  void process_split_piece(const float* in, int n, float* even_out, float* odd_out);
  void process_interleaved_piece(const float* in, int n, float* out2x);
  void write_ring(const float* src, int n);

  int M_ = 0;               // number of unique symmetric odd-tap pairs
  int max_block_ = 0;
  int R_ = 0, mirror_ = 0;
  std::uint32_t mask_ = 0, pos_ = 0;
  engine_util::AlignedFloats coeffs2_; // M_ coefficients scaled by 2.0f for unity DC gain
  engine_util::AlignedFloats ring_;    // (R_ + mirror_) floats
};

// 2x -> 1x half-band downsampler using a symmetric half-band filter of length 4*M - 1 (exact delay 2*M - 1 at 2x rate).
class HalfBandDown2x
{
public:
  HalfBandDown2x(const std::vector<double>& taps, int max_output_block);

  // Consumes n pairs of (even, odd) samples and writes n downsampled outputs.
  void process_split(const float* even_in, const float* odd_in, int n, float* out);
  // Consumes n_in interleaved 2x samples (any n_in >= 0, even or odd) and writes floor((n_in + !phase) / 2) outputs.
  int process_interleaved(const float* in2x, int n_in, float* out);
  int max_output(int n_in) const { return (n_in + 1) / 2; }
  void reset();

  int half_taps() const { return M_; }
  int delay_2x_samples() const { return 2 * M_ - 1; }

private:
  void process_split_piece(const float* even_in, const float* odd_in, int n, float* out);
  void write_ring(float* ring, int R, std::uint32_t mask, int mirror, std::uint32_t pos, const float* src, int n);

  int M_ = 0;
  int max_out_block_ = 0;
  int R0_ = 0, mirror0_ = 0;
  int R1_ = 0, mirror1_ = 0;
  std::uint32_t mask0_ = 0, mask1_ = 0;
  std::uint32_t pos0_ = 0, pos1_ = 0;
  int phase_ = 0; // 0: next interleaved sample is even (p0); 1: odd (p1)
  engine_util::AlignedFloats coeffs_; // M_ unique symmetric odd-tap coefficients
  engine_util::AlignedFloats ring0_;  // even-phase history
  engine_util::AlignedFloats ring1_;  // odd-phase history
  std::vector<float> tmp_p0_, tmp_p1_;
};

// ---- one rational conversion stage -----------------------------------------------------------------------------------

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
  int R_ = 0, mirror_ = 0;
  std::uint32_t mask_ = 0, pos_ = 0;
  engine_util::AlignedFloats coefficients_; // up_ rows of taps_per_phase_ floats (multiple of 8), in the order of the input window
  engine_util::AlignedFloats ring_;         // mirrored power-of-two input history ring
  std::int64_t next_input_ = 0;             // input index (relative to the start of the current block) of the next output
  int phase_ = 0;                           // phase of the next output
};

// ---- 2x / 4x phase-interleaved oversampler around the engine ---------------------------------------------------------

// Upsamples a 1x model-rate signal to 2x or 4x via linear-phase half-band filters, splits it into `factor` phase-shifted
// 1x streams for parallel A2 engines, and decimates the `factor` engine outputs back to 1x with exact integer latency.
class Oversampler
{
public:
  Oversampler(int factor, int max_block, ResamplerQuality quality);

  int factor() const { return factor_; }
  int latency_samples() const { return latency_; } // exact integer delay in 1x (model-rate) samples
  void reset();

  // Splits n 1x samples into `factor_` phase streams (each of length n, written to phases[0..factor_-1]).
  void upsample(const float* in, int n, float* const* phases);
  // Combines `factor_` phase streams (each of length n) back into n 1x samples.
  void downsample(const float* const* phases, int n, float* out);

private:
  int factor_ = 2;
  int max_block_ = 0;
  int latency_ = 0;
  std::unique_ptr<HalfBandUp2x> up1_, up2_;
  std::unique_ptr<HalfBandDown2x> down1_, down2_;
  // Scratch buffers for 4x cascade (allocated in constructor, never in process)
  std::vector<float> mid0_, mid1_, mid96_, mid192_e_, mid192_o_, down96_;
  float delay96_ = 0.0f; // 1-sample alignment delay at 96 kHz so 4x total delay is an exact integer at 48 kHz
};

// ---- the pair of rate-conversion stages around the engine ------------------------------------------------------------

// host rate -> model rate (stage 1), the engine, model rate -> host rate (stage 2), and a FIFO that makes the number of output samples
// of each block equal to the number of input samples. Uses fast half-band cascades for 2x, 4x and 1/2x ratios, and rational polyphase
// FIR otherwise. Mono. Not thread safe.
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
  double stage1_delay_model_samples() const { return stage1_delay_model_; }

  // For the tests.
  PolyphaseStage& stage1() { return *stage1_; }
  int up() const { return up_; }
  int down() const { return down_; }
  int taps_per_phase_stage1() const { return taps_stage1_; }
  int taps_per_phase_stage2() const { return taps_stage2_; }

private:
  enum class Mode
  {
    Polyphase,
    HybridDown4 // host = 4 * model (e.g. 192 -> 48): outer HalfBand 4x<->2x + inner Polyphase 2x<->1x
  };

  Mode mode_ = Mode::Polyphase;
  int up_ = 1, down_ = 1; // stage 1: model rate / host rate = up / down
  int max_model_block_ = 0;
  int filter_delay_ = 0, fifo_start_ = 0;
  double stage1_delay_model_ = 0.0;
  int taps_stage1_ = 0, taps_stage2_ = 0;

  // Polyphase path
  std::unique_ptr<PolyphaseStage> stage1_, stage2_;
  // Half-band path
  std::unique_ptr<HalfBandDown2x> hb_down1_, hb_down2_;
  std::unique_ptr<HalfBandUp2x> hb_up1_, hb_up2_;
  std::vector<float> hb_mid_; // intermediate 2x buffer for 4x cascade

  std::vector<float> stage2_out_;
  std::vector<float> fifo_; // ring buffer, power-of-two size
  std::size_t fifo_mask_ = 0, fifo_read_ = 0, fifo_count_ = 0;
};
} // namespace namturbo
