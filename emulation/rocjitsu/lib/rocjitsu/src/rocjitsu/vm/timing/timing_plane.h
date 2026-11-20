// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_plane.h
/// @brief The timed machine: components, clocks, and the clock the guest reads.
///
/// @details Instantiated by the SoC when the architecture config carries a
/// `timing` block, and part of the machine rather than something loaded next to
/// it. It owns the clock domains, the compute units, the cache levels, the
/// fabric and the memory channels, wires them together, and runs them on its
/// own event loop over modelled hardware time.
///
/// The functional simulator drives it through three direct calls -- a dispatch
/// beginning, an instruction retiring, a dispatch ending. Those calls are what
/// a compute unit already knows and cost a null check when no plane is
/// installed, which is the same thing the execution-plugin group already costs
/// on the same path.
///
/// What comes out is one number: the tick the modelled machine has reached.
/// That is published into SimulatedClock, and from there it is what
/// hipEventElapsedTime subtracts, what the KFD clock-counters ioctl answers,
/// what a completion signal's timestamps carry, and what `s_memtime` returns
/// inside a kernel. A guest that times itself measures the modelled part rather
/// than the host it happens to be running on.

#pragma once

#include "rocjitsu/vm/timing/cache_des.h"
#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/dram_des.h"
#include "rocjitsu/vm/timing/engine.h"
#include "rocjitsu/vm/timing/request.h"
#include "rocjitsu/vm/timing/time_source.h"
#include "rocjitsu/vm/timing/tuning.h"

#include "simdojo/sim/clock_domain.h"

#include <cstring>
#include <unordered_set>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::timing {

/// @brief One record of the instruction-stream export.
///
/// @details A diagnostic, off unless ROCJITSU_TIMING_ISTREAM names a path
/// prefix. It carries what the compute unit computed for each instruction of
/// the sampled units -- port occupancy, result latency, wait thresholds and the
/// registers named -- so that a scheduling model can be replayed over the real
/// instruction streams offline. Fixed size and packed so a reader can map it.
#pragma pack(push, 1)
struct IStreamRecord {
  /// 0 instruction, 1 wave begin, 2 wave end, 3 barrier member, 4 dispatch
  /// acquire, 5 dispatch end, 8 measured-scheduler result of one unit (a =
  /// scheduled cycles, b = waves, pc = summed wavefront lifetimes).
  std::uint8_t type = 0;
  std::uint8_t inst_class = 0;
  std::uint8_t unit = 0;
  /// Wait counter an access posts to; 0xFF for none.
  std::uint8_t counter = 0xFF;
  std::uint16_t slot = 0;
  std::uint8_t reads = 0;
  std::uint8_t writes = 0;
  /// Folded compute unit.
  std::uint32_t cu = 0;
  /// Instruction: port occupancy. Wave begin: workgroup. Barrier: group
  /// sequence. Dispatch records: sequence.
  std::uint32_t a = 0;
  /// Instruction: cycles from issue until the result is readable. Wave begin:
  /// dispatch id.
  std::uint32_t b = 0;
  /// Wait thresholds, one byte per counter, 0xFF unconstrained.
  std::uint32_t wait = 0xFFFFFFFFu;
  /// Exposed instruction-fetch stall.
  std::uint32_t fetch = 0;
  std::uint8_t front_end = 0;
  /// Wave user priority (s_setprio) after the instruction.
  std::uint8_t priority = 0;
  std::uint16_t lines = 0;
  std::uint64_t pc = 0;
  /// Reads then writes: file << 24 | count << 16 | index.
  std::uint32_t regs[12] = {};
  /// Unfolded workgroup id: waves of different workgroups folded onto one unit share slot numbers.
  std::uint32_t wg = 0;
  std::uint32_t reserved = 0;
};
#pragma pack(pop)
static_assert(sizeof(IStreamRecord) == 96);

class DispatchDes;

/// @brief What the command processor reports when a dispatch starts.
struct DispatchShape {
  std::uint32_t dispatch_id = 0;
  std::uint32_t queue_id = 0;
  std::string kernel_name;
  std::uint32_t workgroup_count = 0;
  std::uint32_t waves_per_workgroup = 0;
  std::uint32_t vector_registers_per_wave = 0;
  std::uint32_t scalar_registers_per_wave = 0;
  std::uint32_t lds_bytes_per_workgroup = 0;
  std::uint32_t wave_size = 64;
};

/// @brief The timed machine.
class TimingPlane final : public TimeSource {
public:
  explicit TimingPlane(Tuning tuning);
  ~TimingPlane() override;

  bool enabled() const { return tuning_.enabled; }

  // -- Driven by the functional simulator ----------------------------------
  void dispatch_begin(const DispatchShape &shape);
  /// @brief A wavefront of @p dispatch_id started on @p compute_unit.
  ///
  /// @details Also where the launch acquire is applied, because this is the
  /// first moment the dispatch is known to be executing. The packet-processed
  /// callback is not: the command processor parses the next packet while the
  /// current one is still running.
  void wave_begin(std::uint32_t compute_unit, std::uint32_t slot, std::uint32_t dispatch_id,
                  std::uint32_t queue_id);
  void instruction(std::uint32_t compute_unit, std::uint32_t slot,
                   const RetiredInstruction &retired);
  void wave_end(std::uint32_t compute_unit, std::uint32_t slot);
  void dispatch_end(std::uint32_t dispatch_id, std::uint32_t queue_id);
  ComputeUnitDes &compute_unit(std::uint32_t index);

  /// @brief The timed compute unit a workgroup runs on: XCD-interleaved, as placed by the SPI.
  std::size_t fold_unit(std::uint64_t workgroup) const;

  // -- Used by the compute units -------------------------------------------
  /// @brief Hand one memory instruction's lines to the hierarchy.
  /// @returns The tick the access completes.
  std::uint64_t issue_memory(const MemoryRequest &request, const std::vector<std::uint64_t> &lines);
  /// @brief Apply a dispatch's launch acquire, exactly once, at the moment its
  ///        first wavefront issues. Not when its packet is parsed: the command
  ///        processor parses ahead, so a packet-driven invalidate lands at an
  ///        unpredictable point relative to execution.
  void acquire_once(std::uint32_t dispatch_id, std::uint32_t queue_id);

  // -- TimeSource ----------------------------------------------------------
  std::uint64_t current_cycles() const override { return cycles_.load(std::memory_order_relaxed); }
  double clock_ghz() const override { return tuning_.clock_mhz / 1000.0; }

  void write_report(std::string &out) const;

  /// @brief Whether the instruction-stream export samples folded unit @p unit.
  bool istream_wants(std::size_t unit) const {
    return istream_ != nullptr && unit % istream_stride_ == 0 && unit / istream_stride_ < istream_units_;
  }
  /// @brief Append one record. Callers hold the collector's plane lock.
  void istream_write(const IStreamRecord &record) {
    if (istream_ != nullptr)
      std::fwrite(&record, sizeof(record), 1, istream_);
  }
  std::uint32_t istream_next_barrier() { return istream_barriers_++; }
  /// @brief Unfolded workgroup of the call in progress; set by the plane and the collector.
  std::uint32_t istream_wg() const { return istream_wg_; }
  void istream_set_wg(std::uint32_t wg) { istream_wg_ = wg; }
  /// @brief True the first time @p pc is seen, so its mnemonic is written once.
  bool istream_first_pc(std::uint64_t pc) { return istream_pcs_.insert(pc).second; }

private:
  void publish(std::uint64_t cycles);
  /// @brief Record what every component had moved, so a dispatch's own traffic
  ///        is the difference rather than the total.
  /// @brief Split a second-level miss across the channels its lines select.
  void forward_to_channels(const MemoryRequest &request);
  void snapshot_bandwidth(std::uint64_t key);
  void charge_bandwidth(DispatchDes &dispatch, std::uint64_t key);
  /// @brief Append one dispatch's terms to the trace, when one is open.
  ///
  /// @details Off unless ROCJITSU_TIMING_TRACE names a file. What it writes is
  /// every term the composition used, not the total it produced: a tuning
  /// change that only recombines those terms can then be evaluated against a
  /// recorded corpus instead of by re-running it, which is the difference
  /// between a ten-minute experiment and a one-second one.
  void trace_dispatch(const DispatchDes &dispatch);

  Tuning tuning_;
  simdojo::ClockDomain shader_;
  simdojo::ClockDomain fabric_;
  simdojo::ClockDomain memory_;
  TimingEngine engine_;

  std::vector<std::unique_ptr<ComputeUnitDes>> compute_units_;
  std::vector<std::unique_ptr<CacheDes>> l1_vector_;
  std::vector<std::unique_ptr<CacheDes>> l1_scalar_;
  std::vector<std::unique_ptr<CacheDes>> l1_instruction_;
  std::vector<std::unique_ptr<CacheDes>> l2_;
  std::vector<std::unique_ptr<ChannelDes>> fabric_channels_;
  std::vector<std::unique_ptr<CacheDes>> mall_;
  std::vector<std::unique_ptr<ChannelDes>> dram_;

  /// @brief Line addresses in flight, referenced by MemoryRequest::line_base.
  std::vector<std::uint64_t> line_pool_;
  /// @brief Tick each compute unit's current dispatch began, and which that is.
  std::vector<std::uint64_t> cu_base_;
  std::vector<std::uint64_t> cu_dispatch_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> bandwidth_base_;
  std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> activation_base_;
  /// @brief Open trace stream, or null when no trace was asked for.
  std::FILE *trace_ = nullptr;
  /// @brief Instruction-stream export, or null; see IStreamRecord.
  std::FILE *istream_ = nullptr;
  std::uint32_t istream_units_ = 0;
  /// Sample every Nth folded unit, so workgroups with different work are all represented.
  std::uint32_t istream_stride_ = 1;
  std::uint32_t istream_barriers_ = 0;
  std::uint32_t istream_acquires_ = 0;
  std::uint32_t istream_ends_ = 0;
  std::uint32_t istream_wg_ = 0;
  std::unordered_set<std::uint64_t> istream_pcs_;
  /// @brief The request issue_memory() is waiting for, and its answer.
  std::uint64_t sequence_ = 0;
  std::uint32_t awaited_ = 0;
  std::uint32_t awaited_outstanding_ = 0;
  std::vector<std::uint64_t> channel_scratch_;
  std::vector<std::uint64_t> request_scratch_;
  /// @brief Distinct DRAM rows one request touches on one channel.
  ///
  /// @details A tiny insertion-ordered set reused across requests rather than
  /// allocated per request, because this runs inside the per-instruction path.
  struct RowSet {
    std::vector<std::uint64_t> rows;
    void clear() { rows.clear(); }
    bool insert(std::uint64_t row) {
      for (std::uint64_t seen : rows)
        if (seen == row)
          return false;
      rows.push_back(row);
      return true;
    }
  };
  RowSet row_scratch_;
  bool answered_ = false;
  std::uint64_t completion_tick_ = 0;

  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::unique_ptr<DispatchDes>> open_;
  std::vector<std::string> completed_;

  /// @brief Read from guest threads holding none of this plane's locks, so it
  ///        is published from a relaxed atomic and never allowed to retreat.
  std::atomic<std::uint64_t> cycles_{0};
};

} // namespace rocjitsu::timing
