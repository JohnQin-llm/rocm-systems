// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache_des.h
/// @brief A timed cache level, and the tag array behind it.

#pragma once

#include "rocjitsu/vm/timing/timed_component.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

namespace rocjitsu::timing {

/// @brief A set-associative tag array. Tags only; the data lives in the
///        functional simulator and the timing plane never reads it.
class TagArray {
public:
  /// @brief No line: what access() reports when its replacement wrote nothing back.
  static constexpr std::uint64_t kNoLine = ~0ULL;

  void configure(std::uint64_t sets, std::uint64_t ways, std::uint64_t line_bytes);
  /// @brief Probe and allocate, replacing least recently used.
  /// @param write The access is a store: the line is dirty from now on.
  /// @param cold Allocate a missing line as least recently used rather than most,
  ///        and leave a hit's recency alone. A streaming access does this so its
  ///        lines are the first to go and do not displace the working set.
  /// @param dirty_victim Set to the byte address of a dirty line the allocation
  ///        replaced, which now has to be written back; kNoLine when none was.
  /// @returns true when the line was already resident.
  bool access(std::uint64_t byte_address, std::uint64_t stamp, bool write = false,
              bool cold = false, std::uint64_t *dirty_victim = nullptr);
  /// @brief Write back every dirty line: mark it clean and report it.
  /// @returns How many lines were dirty.
  std::uint64_t flush();
  void invalidate();
  std::uint64_t line_bytes() const { return line_bytes_; }
  std::uint64_t line_shift() const { return line_shift_; }

private:
  std::uint64_t sets_ = 0;
  std::uint64_t ways_ = 0;
  std::uint64_t set_mask_ = 0;
  std::uint64_t line_bytes_ = 64;
  std::uint64_t line_shift_ = 6;
  std::vector<std::uint64_t> tags_;
  std::vector<std::uint64_t> stamps_;
  std::vector<std::uint8_t> dirty_;
  static constexpr std::uint64_t kEmpty = ~0ULL;
};

/// @brief One instance of a cache level: a first-level cache on one compute
///        unit, a second-level cache on one die, or one channel of the
///        memory-side cache.
///
/// @details Serves the lines it holds and forwards the rest to whatever is
/// downstream. The split is per request rather than per line, so one wave's
/// access that half hits produces one completion and one forwarded request
/// rather than sixty-four events.
class CacheDes final : public TimedComponent {
public:
  /// @brief What a level does with the requests it cannot serve.
  using Downstream = std::function<void(const MemoryRequest &)>;
  /// @brief What it does with the requests it can.
  using Completion = std::function<void(const MemoryRequest &, std::uint64_t tick)>;

  CacheDes(std::string name, const simdojo::ClockDomain &domain, TimingEngine &engine,
           const CacheTuning &tuning, std::vector<std::uint64_t> *line_pool);

  void set_downstream(Downstream downstream) { downstream_ = std::move(downstream); }
  void set_completion(Completion completion) { completion_ = std::move(completion); }
  /// @brief Bypass the tag array entirely; used for a level a request marks
  ///        non-temporal, which the hardware documents as not allocating.
  void set_allocates(bool allocates) { allocates_ = allocates; }

  std::uint64_t advance(std::uint64_t now) override;

  void invalidate();
  /// @brief Write back every dirty line, as a release at the end of a kernel
  ///        does. Nothing is sent downstream: the caller costs the flush.
  /// @returns The bytes written back.
  std::uint64_t flush() { return tags_.flush() * tags_.line_bytes(); }

  std::uint64_t hits() const { return hits_.load(std::memory_order_relaxed); }
  std::uint64_t misses() const { return misses_.load(std::memory_order_relaxed); }
  std::uint64_t bytes() const { return bytes_.load(std::memory_order_relaxed); }

private:
  /// @brief Send the dirty lines @p cause displaced to the level below.
  void write_back(const MemoryRequest &cause);

  const CacheTuning &tuning_;
  std::vector<std::uint64_t> *line_pool_;
  TagArray tags_;
  Downstream downstream_;
  Completion completion_;
  bool allocates_ = true;
  std::uint64_t stamp_ = 1;
  /// @brief Dirty lines one request's allocations displaced; reused.
  std::vector<std::uint64_t> victims_;
  std::atomic<std::uint64_t> hits_{0};
  std::atomic<std::uint64_t> misses_{0};
  std::atomic<std::uint64_t> bytes_{0};
};

} // namespace rocjitsu::timing
