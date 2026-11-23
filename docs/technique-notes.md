# Technique notes

Implementation notes for readers and maintainers of the code. They are not needed to use the library. File names refer to `src/`.

## 1. The network

An A2 model is a WaveNet with 23 dilated convolution layers and a head. For channel count C (3 or 8), kernel sizes K[l] and dilations D[l]
(see [model compatibility](model-compatibility.md)), with input signal `x`, residual state `s` and a head accumulator `h`, each output
sample is computed as follows:

    s = rechannel * x                                   (C numbers; the first weights of the file)
    h = 0
    for each layer l:
        z = conv_bias + mixin * x                       (mixin: C numbers; x is the input sample)
        for each tap k in 0..K-1:
            z += W[l][k] * s[t - (K-1-k) * D[l]]        (C x C matrix times the residual state of an earlier sample)
        a = LeakyReLU(z)                                (slope 0.01 below zero)
        h += a
        s += l1x1 * a + l1x1_bias                       (the residual update)
    y = head_scale * (head_bias + sum over 16 taps of head_weight[k] . h[t - (15-k)])

The network has no recurrence: the output at time `t` depends only on the last 6347 input samples. A block can therefore be computed layer by
layer, and the output does not depend on how the signal is partitioned into blocks.

The cost is dominated by the convolutions: A2-Full does about 8 x 8 x 6 = 384 multiply-adds per layer and sample for most layers (and
8 x 8 x 15 = 960 for two layers), A2-Lite about 9 x 6 = 54.

## 2. A2-Full engine (`a2_avx2.cpp`)

**Layout.** One time frame of 8 channels is 8 floats, one 256-bit register. The state of each layer is stored as a sequence of frames.

**Convolution.** For a tile of T frames, one accumulator register per frame holds the 8 outputs. For each tap and each input channel
`j`, the weights of that input channel (the 8 outputs, stored contiguously) are loaded once and used for all T frames:

    acc[frame] = fma(weights[k][j], broadcast(history[frame][j]), acc[frame])

A loaded weight vector is used T times, and the loop has T independent dependency chains, which hides the latency of the fused multiply-add.
The history window of a tap is a pointer into a ring buffer; the T frames of a tap are contiguous.

**History rings.** Each layer keeps the last frames of its input in a ring buffer whose length is a power of two (so the position wraps
with a mask). The ring has a mirrored tail: the first M columns are copied after the end, so that a window of M consecutive columns
never has to wrap and can be read with ordinary loads.

**Two-level sum.** The 8 products of one tap are first summed in their own register (the first product is a multiplication, the other
seven are fused multiply-adds) and the result is then added to the accumulator. The alternative, one chain of 8 x K fused
multiply-adds, rounds 8 x K times at the size of the final sum. The two-level sum rounds K times at that size, and rounds the small sums at
their own, smaller, size. Measured against a float64 reference on high-gain material, this removes 7 dB of error, which brings the engine to within about 2 dB of the
A2 fast path of NAM Core (see [accuracy](accuracy.md)). It costs about 3% of the time. The partial sums need
registers: a tile of 6 frames needs 6 accumulators and 6 partial sums, which fits in the 16 registers of AVX2 together with the weight and
the broadcast value. A tile of 8 frames would need all 16 registers for the accumulators and the partial sums alone. A tile of 4 frames
fits too, but it reuses each loaded weight less often and was measured to cost 12% instead of 3%.

**Activation, head and residual.** After the convolution, a block-wide pass applies LeakyReLU, adds the activation to the head accumulator,
and computes the residual update with the 1x1 convolution. At the end, the head convolution takes the 16 most recent frames of the head
accumulator, multiplies each by its weight vector lane by lane, and reduces the lanes to one sample.

**Independence of the tile.** The arithmetic of one output frame does not depend on the size of its tile or on its block. A frame does not read
values that depend on neighbouring frames of the tile. The output is therefore bit-identical for every partitioning of the signal into blocks.
The tests check block sizes from 1 to 700 samples.

## 3. A2-Lite engine (`a2_lite_avx2.cpp`)

With 3 channels, the A2-Full layout would use 3 of 8 lanes. The A2-Lite engine holds **8 consecutive frames of one channel** per register. All
lanes are used, a weight is a broadcast scalar, and the history of 8 frames of a tap is one unaligned load from the ring of that channel. A tile of
up to three registers (24 frames) computes the convolution, the activation, the head sum and the residual update together, so that only the
residual state is stored between the stages.

The last register of a block may be partly empty. Its unused lanes compute values from stale ring data that no valid lane reads (a lane reads
only positions at or before its own); only valid values are stored. Every lane executes the same arithmetic regardless of its position, so
the output is independent of the block size.

## 4. Denormal numbers

Denormal floating-point numbers can cost more than a hundred cycles per operation on some processors, and a decaying signal or silence can
produce them inside the network. During `process` the engines set the flush-to-zero and denormals-are-zero bits (`ScopedNoDenormals` in
`engine_util.h`) and restore the previous state on return, so the floating-point state of the host is unchanged. The block time does not
depend on the signal.

## 5. Sample-rate conversion and oversampling (`resampler.cpp`)

When the host rate differs from the rate of the model, two linear-phase FIR stages convert the signal: host rate to model rate before
the network, model rate to host rate after it.

- **4x host rates (192 kHz).** Uses a two-stage hybrid cascade: a short symmetric half-band AVX2 outer stage (`HalfBandUp2x`,
  `HalfBandDown2x`, 192 <-> 96 kHz, wide transition band) paired with a steep polyphase inner stage (96 <-> 48 kHz, stop-band at 24 kHz).
- **General rational ratios (e.g. 44.1, 88.2, 96, 32 kHz).** For a reduced ratio `a/b` (numerator and denominator at most 2000), a polyphase
  FIR stage computes only the retained output samples from a mirrored power-of-two input ring. Each phase is padded to a multiple of 8
  taps and evaluated with a 4-accumulator AVX2 dot product plus an 8-wide tail. Both stages share one design grid so the total filter
  delay is a whole number of host samples.
- **Presets.** `Standard` has 96 dB attenuation and a flat band to 90% of the lower Nyquist frequency; `LowLatency` has 72 dB and 80%.
- **Phase-interleaved oversampling (`Oversampler`).** Because an A2 WaveNet is a discrete-time filter bank trained at a fixed rate
  (48 kHz), feeding a 96 kHz stream directly into it would shift every filter up by one octave. Instead, `Oversampling::X2` and
  `Oversampling::X4` upsample the model-rate stream with half-band filters, split it into 2 or 4 phase-shifted 48 kHz streams, run an
  independent A2 engine on each phase (mathematically identical to dilating all layer taps by the oversampling factor), and decimate back
  to 48 kHz. In-band linear filter responses are preserved, while ultrasonic nonlinear harmonics reconstruct at their true frequencies
  across the interleaved phases and are attenuated by the decimator. For 4x, a 1-sample alignment delay at the intermediate 2x rate makes
  the round-trip latency an exact integer number of model-rate samples.

## 6. Model swap (`processor.cpp`)

The audio thread writes the signal that enters the engine (after the sample-rate conversion) into a history ring of 32768 samples, and
publishes the number of samples written with a release store. The elements of the ring are relaxed atomics, because a control
thread reads them while the audio thread writes: the reader copies a range, issues an acquire fence, and discards the copy if the published
position shows that the audio thread has overwritten part of it. Relaxed atomic floats are ordinary moves on x86. To swap:

1. `prepare_swap` (control thread) builds the new engine and runs it over the last 6347 + 64 samples of the history, so that its state matches
   the running engine, and then over the samples that arrived meanwhile, until the gap is below 512 samples. It checks that the audio
   thread did not overwrite the part of the history that it was reading.
2. `commit_swap` publishes the ticket with one atomic exchange. It refuses a ticket that is out of date (a different generation, a
   different processor, or an audio position more than 1024 samples ahead).
3. At the start of its next block, the audio thread takes the ticket, runs a short catch-up, and then runs both engines for the length of
   the fade, mixing them with a raised-cosine curve from a table.
4. When the fade ends, the old engine goes into a lock-free queue of 8 places. The audio thread never frees an engine; `service()` empties
   the queue.

After `prepare()`, only the audio thread reads or replaces the pointer to the active engine; control threads use a separate atomic flag to
know whether the processor is prepared. The audio thread therefore never allocates, never frees and never waits. The cost for it is the catch-up and the two engines during the
fade.

## 7. Monitoring and the adaptive mode

The monitor (`monitor.h`) keeps the sum, the maximum and a histogram of the block times in relaxed atomics, written by the audio thread
only. The histogram has 400 logarithmic buckets of 5% each, which gives the 99th percentile with an accuracy of 5%. A reader on another
thread gets a result from a moment a few calls ago.

In the adaptive mode, the audio thread keeps an exponential average of the load of each block in an atomic number. `service()` runs the
policy on a control thread: it compares the average with the two thresholds for the required time. To know whether the higher variant
fits, it multiplies the load by the ratio of the costs of A2-Full and A2-Lite, which is measured with two throw-away engines when the
mode is switched on. The switch is a swap.

## 8. Tests

| Test | What it proves |
|---|---|
| `test_ref_parity`, `test_avx2_parity`, `test_avx2_robust` | Each engine agrees with a scalar reference and with the output of NAM Core on stored signals, on the edge cases (extreme weights and inputs, silence, one-sample blocks), and gives the same output for all block divisions. |
| `test_resampler` | Filter design, accuracy of the conversion, exact latency, block independence. |
| `test_swap`, `test_adaptive` | No click at a swap (the fade equals the mix formula exactly), correct handling of stale and foreign tickets, the policy of the adaptive mode, many swaps while audio runs on another thread. |
| `test_realtime` | `process` does not allocate or free memory, also during swaps. |
| `test_monitor` | The statistics, with synthetic call times, and a concurrent reader. |
| `test_fuzz_loader` | Damaged and hostile files give an error or a model, nothing else. |
| `test_cpu_failure` | A processor without AVX2 or FMA gets a clear error, and a processor that is already running is not harmed. |
| `precision_vs_nam_*` (with the baseline) | The engines are as precise as NAM Core against a float64 reference. |
