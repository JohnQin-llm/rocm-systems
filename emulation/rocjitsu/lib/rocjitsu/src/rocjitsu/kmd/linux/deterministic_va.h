// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file deterministic_va.h
/// @brief Deterministic placement of the GPU runtime's address-space reservations.
///
/// @details On a GPU whose virtual address range covers the whole CPU range
/// (gfx9 and later, which is what rocjitsu reports), libhsakmt gives every GPU
/// allocation its virtual address by reserving it with
/// `mmap(NULL, size, PROT_NONE, MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE)`.
/// The kernel places that reservation wherever the host address space has room,
/// which depends on ASLR, on how long the environment and the paths are, and on
/// what every other thread has mapped by then. rocjitsu maps GPU addresses
/// directly to those host addresses, so device buffers, code objects and
/// kernel arguments land at different addresses from run to run -- and the
/// timing plane's cache sets, channels and DRAM rows, which hash those
/// addresses, change with them.
///
/// The arena hands those reservations out from a fixed base instead, in the
/// order the runtime asks for them, so two runs that make the same allocations
/// get the same addresses. The address space it uses is reserved and released
/// exactly as the runtime's own calls would have been; only the hint changes.
/// ROCJITSU_DETERMINISTIC_VA=0 turns it off; ROCJITSU_DETERMINISTIC_VA_BASE
/// (hex or decimal) moves it.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sys/types.h>

namespace rocjitsu::kmd {

class DeterministicVaArena {
public:
  using MapFn = void *(*)(void *, size_t, int, int, int, off_t);

  /// @brief 32 TiB: inside the GPU aperture rocjitsu reports, below the
  ///        top-down area where the kernel puts mmaps, libraries and stacks,
  ///        and above where executables and their heaps live.
  static constexpr std::uint64_t kDefaultBase = 0x200000000000ULL;
  /// @brief 96 TiB: the arena's end, well under the 128 TiB user limit.
  static constexpr std::uint64_t kDefaultLimit = 0x600000000000ULL;
  /// @brief Every reservation starts on a 2 MiB boundary, the runtime's largest
  ///        alignment, so its own alignment arithmetic is the same every run.
  static constexpr std::uint64_t kGranule = 2ULL << 20;

  explicit DeterministicVaArena(std::uint64_t base = kDefaultBase,
                                std::uint64_t limit = kDefaultLimit);

  /// @brief Whether a call has the shape of the runtime's address reservation:
  ///        no hint, no file, anonymous, private, never touched (PROT_NONE,
  ///        MAP_NORESERVE). Necessary, not sufficient; the caller also checks
  ///        who is asking (glibc's malloc reserves its heaps the same way).
  static bool is_reservation(const void *addr, size_t length, int prot, int flags, int fd);

  /// @brief Place a reservation at the next free granule of the arena with
  ///        @p map, which is called with MAP_FIXED_NOREPLACE.
  /// @details A granule something else already occupies is skipped (the
  ///        result then depends on that mapping, which is the one case this
  ///        cannot make deterministic). Returns MAP_FAILED, with errno from the
  ///        last attempt, when the arena is exhausted or @p map fails otherwise;
  ///        the caller then falls back to an ordinary placement.
  void *reserve(size_t length, int prot, int flags, MapFn map);

  /// @brief Where the next reservation will be tried.
  std::uint64_t cursor() const;
  std::uint64_t base() const { return base_; }

private:
  mutable std::mutex mutex_;
  std::uint64_t base_;
  std::uint64_t limit_;
  std::uint64_t cursor_;
};

/// @brief The process's arena, or null when ROCJITSU_DETERMINISTIC_VA=0.
DeterministicVaArena *runtime_va_arena();

/// @brief Record where the GPU runtime's code lives (libhsa-runtime64, and
///        libhsakmt when it is a library of its own), so reservations can be
///        told apart from everyone else's. Call when the runtime opens the KFD,
///        which it does before it allocates anything.
void note_gpu_runtime_text();

/// @brief Whether @p return_address is inside the code note_gpu_runtime_text()
///        recorded. Lock-free.
bool is_gpu_runtime_caller(const void *return_address);

/// @brief Whether @p name is the path of a GPU runtime library.
bool is_gpu_runtime_library(const char *name);

} // namespace rocjitsu::kmd
