// Check of the processor, for a program that wants to tell its user before it loads a model.
#pragma once

#include <string>

namespace namturbo
{
struct CpuSupport
{
  bool supported;     // true if the engines can run on this processor and operating system
  std::string detail; // when not supported: which feature is missing
};

// The engines need AVX2 and FMA, and an operating system that saves the 256-bit registers. Does not throw.
CpuSupport check_cpu();
} // namespace namturbo
