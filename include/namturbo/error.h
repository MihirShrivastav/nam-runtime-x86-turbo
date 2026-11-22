// Errors reported by namturbo.
#pragma once

#include <stdexcept>
#include <string>

namespace namturbo
{
enum class ErrorCode
{
  Io,               // the file cannot be opened or read
  InvalidFile,      // the data is not valid JSON, or the model data is malformed
  NotA2,            // a valid NAM file, but not an A2 model
  UnsupportedShape, // an A2 model of a size this library does not run
  CpuUnsupported,   // the CPU or the operating system lacks AVX2 and FMA
  InvalidArgument,  // a parameter has a value that cannot be used
  Unsupported       // a valid request that this version does not implement
};

// Thrown by the functions that set things up (loading a model, preparing a processor). what() is a complete sentence
// that a program can show to its user. Processing audio never throws.
class Error : public std::runtime_error
{
public:
  Error(ErrorCode code, const std::string& message)
    : std::runtime_error(message)
    , code_(code)
  {
  }
  ErrorCode code() const noexcept { return code_; }

private:
  ErrorCode code_;
};
} // namespace namturbo
