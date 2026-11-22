## Two instances in real time

Two models run on one audio thread, for example a pedal capture and an amp capture. Every block must be finished before its deadline for the
whole run. The run is long (60 s per setting) so that rare delays occur; this kind of test is called a soak test.

Setup (`benchmarks/two_instance.cpp`):

- Two processors on one thread. Each block is processed by the first and then by the second. Pairs: A2-Full with A2-Full, and A2-Full with
  A2-Lite. Model: `test-amp-and-cab.nam`. Host rate: 48 kHz.
- The thread waits for the start of each block with a high-resolution timer, as an audio thread does, at raised priority on one pinned core.
  Between blocks the caches and the clock speed cool down, so a block takes longer than in the benchmark tables, by up to a factor of about two.
- 60 s of audio per configuration. namturbo runs first, then the NAM Core A2 fast path (built as described in [Method](#method)) with the same pair
  and signal.
- **Compute time**: the time of the pair for each block, compared with the block duration (buffer size divided by 48 kHz). "Blocks over the
  deadline" counts blocks whose compute time exceeded the duration.
- **Wake-up delay**: the delay between the timer expiry and the start of the thread. It is caused by the operating system and the machine, not by
  the library. It is reported for reference and is not counted in the blocks over the deadline.
- The signal is a generated test signal. The output is checked to be finite in every run.

Times are in microseconds (us) and milliseconds (ms). The share of the deadline is given for the median block (p50) and for the slowest block (max).
