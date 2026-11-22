// namturbo: a fast runtime for Neural Amp Modeler A2 models on x86-64 processors with AVX2 and FMA.
//
//   auto model = namturbo::Model::load("amp.nam");            // once, on a non-real-time thread
//   namturbo::Processor processor(model);                      // one per mono audio stream
//   processor.prepare({.sample_rate = 48000.0, .max_block = 512});   // when the host prepares to play
//   processor.process(in, out, n);                             // on the audio thread, for every block
#pragma once

#include "namturbo/cpu.h"
#include "namturbo/error.h"
#include "namturbo/model.h"
#include "namturbo/processor.h"
#include "namturbo/version.h"
