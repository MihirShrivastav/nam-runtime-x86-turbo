// Library version (semantic versioning).
#pragma once

namespace namturbo
{
struct Version
{
  int major, minor, patch;
};

constexpr Version kVersion = {1, 0, 0};

// Pre-release identifier of Semantic Versioning, for example "rc.1" (version 1.0.0-rc.1, which precedes 1.0.0). Empty for a final release.
inline constexpr const char* kPreRelease = "rc.1";

constexpr Version version() noexcept
{
  return kVersion;
}

// "major.minor.patch", followed by "-" and kPreRelease for a pre-release: "1.0.0-rc.1".
const char* version_string() noexcept;
} // namespace namturbo
