// Pin the calling thread to one logical core, so that a benchmark does not move between cores of different speed.
#pragma once

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#elif defined(__linux__)
  #include <sched.h>
#endif

namespace namturbo
{
// Returns false if the platform has no pinning or the OS refuses (for example, the core does not exist).
inline bool pin_to_core(int core)
{
#if defined(_WIN32)
  if (core < 0 || core >= 64)
    return false;
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << core) != 0;
#elif defined(__linux__)
  if (core < 0 || core >= CPU_SETSIZE)
    return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return sched_setaffinity(0, sizeof set, &set) == 0;
#else
  (void)core;
  return false;
#endif
}
} // namespace namturbo
