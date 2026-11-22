# Model compatibility

namturbo runs one model architecture: the NAM **A2** network. This page specifies the accepted files, how they are interpreted, and what is rejected.

## Accepted files

A `.nam` file is a JSON file. namturbo accepts:

1. A **slimmable container** (`"architecture": "SlimmableContainer"`) whose submodels are A2 WaveNet models. It holds A2-Lite (3 channels,
   1871 weights) and A2-Full (8 channels, 12146 weights).
2. A **plain A2 WaveNet** (`"architecture": "WaveNet"`) of 3 or 8 channels. It holds one variant only.

The version number in the file is not checked. The files used in the tests have version 0.7.0.

### A2 architecture

The loader checks every architectural setting. A model is accepted only if all of the following hold.

| Setting | Required value |
|---|---|
| Layer arrays | exactly one |
| Layers | 23 |
| Channels, and bottleneck | 3 (A2-Lite) or 8 (A2-Full); the bottleneck equals the channel count |
| Input size and condition size | 1 and 1 |
| Kernel sizes | 6 for all layers except the 15th and 16th, which are 15 |
| Dilations | 1, 3, 7, 17, 41, 101, 239 (twice), then 1, 13, then 1, 3, 7, 17, 41, 101, 239 |
| Activation | LeakyReLU with slope 0.01, in every layer |
| Gating | none |
| Layer 1x1 convolution | active, ungrouped |
| Grouped convolutions | none |
| FiLM modulation, head 1x1 | none |
| Head | a convolution with 1 output, kernel size 16 and a bias |
| Condition network, post-stack head | none |
| Slimming by channels inside the model | none |
| `head_scale` | present |

The receptive field of this network is 6347 samples (132 ms at 48 kHz).

### Containers

The loader examines every submodel. The first A2 submodel with 3 channels becomes A2-Lite and the first with 8 channels becomes A2-Full. A
submodel that is not a valid A2 model is skipped if at least one other submodel is valid; otherwise the error of the first submodel is
reported. `Model::has(Variant)`, `ModelInfo::has_lite` and `ModelInfo::has_full` report the variants that were found.

The `max_value` entries of a container are not used. The variant is selected with `ProcessorConfig::variant` or `Processor::prepare_swap`.
If the requested variant is not in the file, the other variant is used.

## Rejected files

| What the file is | Error code | Message starts with |
|---|---|---|
| Not readable, or missing | `Io` | `cannot open ...` |
| Empty | `InvalidFile` | `the file is empty` |
| Larger than 64 MB | `InvalidFile` | `the file is larger than 64 MB ...` |
| Not valid JSON | `InvalidFile` | `the file is not valid JSON: ...` |
| JSON, but not an object | `InvalidFile` | `the file is JSON but not a NAM model file` |
| A container with no submodels | `InvalidFile` | `the slimmable model has no submodels` |
| Wrong number of weights (too few or too many) | `InvalidFile` | `the model data is too short ...` or `... has N more values ...` |
| A weight that is not a finite number, or too large for a 32-bit float | `InvalidFile` | `the model data contains a value that is not a finite number ...` |
| Another architecture (LSTM, standard WaveNet, ConvNet, linear) | `NotA2` | `this model is not an A2 model: ...` |
| A WaveNet that differs from the A2 pattern in any way listed above | `NotA2` | `this model is not an A2 model: ...` |
| An A2 pattern with a channel count other than 3 or 8 | `UnsupportedShape` | `this is an A2 model with N channels; ...` |

The messages are complete sentences. They are produced on a control thread, never on the audio thread. The loader is tested with damaged and
hostile files (random byte and structure changes, deep nesting, huge numbers, truncation) and returns either one of these errors or a
working model; see `tests/test_fuzz_loader.cpp`.

## Information read from the file

| Where in the file | Provided as |
|---|---|
| `sample_rate` (top level, or in a submodel) | `ModelInfo::sample_rate`; 48000 if absent, with `sample_rate_stated` set to `false` |
| `metadata.loudness` | `ModelInfo::loudness_db` |
| `metadata.input_level_dbu` | `ModelInfo::input_level_dbu` |
| `metadata.output_level_dbu` | `ModelInfo::output_level_dbu` |

Missing entries stay empty. Other metadata (name, gear, date, training data) is ignored. The loudness and level values are not applied to the
signal; see [integration guide, section 7](integration-guide.md#7-gain-staging).

## Sample rate

The network runs at the sample rate of the model (48 kHz for all models in the tests). For another host rate the processor converts the
signal before and after the network and reports the conversion delay with `latency_samples()`. The ratio of the two rates, reduced to lowest
terms, must have a numerator and a denominator of at most 2000; otherwise `prepare` throws an `Unsupported` error. Common rates are well inside
this limit (22.05, 32, 44.1, 48, 88.2, 96, 176.4 and 192 kHz).

`prepare_swap` requires the new model to have the same sample rate as the model of the processor.

## Unsupported models

- Any NAM architecture other than A2.
- A2-like models with other sizes than 3 or 8 channels.
- Stereo models. A model processes one mono signal.
- Models with a separate conditioning input. In A2 the condition size is 1 and the condition is the input signal itself.

## Checking a file

Build `examples/consumer` (see the [README](../README.md)) and run it with the file:

    consumer amp.nam

It prints the latency and the output peak for a short test signal, or the error message.
