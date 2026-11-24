// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file measured_scheduler.h
/// @brief Per-SIMD issue scheduler built from measured MI355X rules.
///
/// @details The alternative composition described in
/// docs/timing-measured-scheduler.md. A compute unit's wavefronts compete for
/// issue slots and execution pipes cycle by cycle instead of being composed
/// from closed-form terms:
///
/// - each wavefront issues in order, one instruction per issue interval;
/// - per-SIMD VALU, MFMA and SALU pipes and per-CU LDS, vector-memory and
///   scalar-memory ports carry the cycle they are next free;
/// - among the wavefronts ready earliest, the highest `s_setprio` wins, then
///   the oldest;
/// - a register scoreboard, wait counters with the hardware convention (the
///   wait issues, then holds the wavefront), and workgroup barriers that
///   release a fixed delay after the last arrival.
///
/// Every number comes from SchedulerTuning, which the config fills; each key
/// cites the microbenchmark it was measured with.
///
/// The functional plane still delivers instructions at retire, in whatever
/// order it executes wavefronts, and the per-wavefront timeline still issues
/// the memory accesses (so the caches and the bandwidth ledger see the same
/// traffic as before). ScheduleRecorder buffers each wavefront's stream with
/// the occupancy and latency computed there, and the scheduler replays the
/// buffered streams of one compute unit at dispatch end. This is the C++ port
/// of the offline replay (bench/aiterfa/istream_replay4..11.py, the frozen merged model), including its
/// memory model: per-pc dispatch-wide mean latency and miss lines, the
/// beyond-L2 share rescaled to the dispatch's fabric traffic, and a two-pass
/// per-unit memory queue at a demand-proportional bandwidth share, plus the
/// replay9 rules (loads complete with their slowest line, at most 118 L2
/// requests in flight per unit, slots freed at completion, a 180-cycle
/// relaunch), the merged rules of istream_replay11 (LDS permute / DMA / wide
/// reads, VALU pairing, miss merge, SIMD slots, cnd32 saturation, encoding
/// mix, conversion and f64 occupancies) and the queued or idle-gap kernel time
/// with the power-limited clock of kernel_eval5.

#pragma once

#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/tuning.h"
#include "rocjitsu/vm/timing/vocabulary.h"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu::timing {

/// @brief Execution resources the scheduler arbitrates.
enum class SchedPipe : std::uint8_t {
  None,  ///< Occupies nothing beyond the wavefront's own issue slot.
  Valu,  ///< Per SIMD.
  Mfma,  ///< Per SIMD.
  Salu,  ///< Per SIMD.
  Lds,   ///< Per CU.
  Vmem,  ///< Per CU.
  Smem,  ///< Per CU.
  Cnd32, ///< Per CU: the resource the 4-byte v_cndmask_b32 (reading VCC) goes through.
  ValuSlot, ///< Per SIMD: end of the VALU slot a double-rate op opened (valu_pair).
  Count,
};
inline constexpr std::size_t kNumSchedPipes = static_cast<std::size_t>(SchedPipe::Count);

/// @brief What one program counter costs, derived once from its mnemonic.
///
/// @details Fields marked "dynamic" are taken per execution from the
/// SchedRecord, because they depend on the access (bank conflicts, lines,
/// where the lines hit) rather than on the opcode.
struct SchedRule {
  std::uint64_t pc = 0;
  /// @brief What the rule was derived from, so a pc reused by other code
  ///        (another kernel loaded at the same address) is re-derived.
  std::string mnemonic;
  std::uint16_t immediate = 0;
  InstClass inst_class = InstClass::Unknown;
  std::uint8_t size_bytes = 4;
  SchedPipe pipe = SchedPipe::None;
  bool pipe_per_simd = false;
  /// @brief Pipe occupancy; taken from the record when occupancy_dynamic.
  bool occupancy_dynamic = false;
  double occupancy = 0.0;
  /// @brief Lower bound on a dynamic occupancy: the vector-memory address
  ///        unit's cost of the access (SchedulerTuning::vmem_address_unit).
  double occupancy_floor = 0.0;
  /// @brief A second resource the instruction holds at issue (MFMA on the
  ///        VALU pipe, v_cndmask_b32_e32 on the CU-wide resource).
  SchedPipe extra_pipe = SchedPipe::None;
  bool extra_per_simd = false;
  double extra_occupancy = 0.0;
  /// @brief The extra hold is occupancy + extra_occupancy (f32-input MFMA).
  bool extra_adds_occupancy = false;
  /// @brief Cycles before the wavefront may issue again.
  double interval = 4.0;
  /// @brief encoding_context: an 8-byte VALU encoding whose interval and
  ///        latency drop to short_interval when enough of the wave's recent
  ///        instructions are 4-byte.
  bool encoding_sensitive = false;
  double short_interval = 4.0;
  /// @brief A v_cndmask_b32 VOP2 form on the CU-wide cnd32 resource.
  bool cnd32 = false;
  /// @brief LDS cycles a buffer/global load to LDS also takes (lds_dma_cycles).
  double lds_dma = 0.0;
  /// @brief The access writes its payload to LDS, not to VGPRs
  ///        (RetiredInstruction::lds_destination); part of the rule's identity.
  bool lds_destination = false;
  /// @brief Cycles from issue until the destination registers are readable;
  ///        the record's latency when latency_dynamic, the pipe occupancy when
  ///        latency_is_occupancy.
  bool latency_dynamic = false;
  bool latency_is_occupancy = false;
  double latency = 4.0;
  /// @brief Control transfer whose taken / not-taken cost the branch rule decides.
  bool branch = false;
  /// @brief s_setpc and friends: the target is never "near".
  bool indirect = false;
  bool barrier = false;
  bool terminate = false;
  bool wait = false;
  std::array<std::uint32_t, kNumWaitCounters> wait_thresholds{};
  RegisterRanges reads;
  RegisterRanges writes;
};

/// @brief One executed instruction of one wavefront.
struct SchedRecord {
  std::uint32_t rule = 0;
  /// @brief Pipe occupancy the per-wavefront model computed (memory, MFMA).
  std::uint32_t occupancy = 0;
  /// @brief Cycles from issue to completion the memory model computed.
  std::uint32_t latency = 0;
  /// @brief Wait counter the access posts to, 0xFF for none.
  std::uint8_t counter = 0xFF;
  /// @brief s_setprio priority after this instruction.
  std::uint8_t priority = 0;
  /// @brief Lines of a memory access that missed the first level (0 when it
  ///        came back within the L1 hit latency), saturated.
  std::uint16_t lines = 0;
};
static_assert(sizeof(SchedRecord) == 16, "SchedRecord is buffered per executed instruction");

/// @brief One wavefront's buffered stream.
struct SchedWave {
  std::uint32_t workgroup = 0;
  std::uint32_t slot = 0;
  /// @brief Exposed instruction fetch before the first instruction.
  std::uint32_t fetch = 0;
  std::vector<SchedRecord> records;
};

/// @brief When one workgroup ran on its unit: its first wavefront's start
///        (launch delay and exposed fetch included) and its last wavefront's
///        end (completion: outstanding accesses and store acks included).
struct WorkgroupSpan {
  std::uint32_t workgroup = 0;
  double start = 0.0;
  double end = 0.0;
};

/// @brief What scheduling one compute unit produced.
struct ScheduleResult {
  /// @brief Each workgroup's residence on the unit, in workgroup-id order,
  ///        from the LAST schedule pass that ran: with the saturated-device
  ///        bisection that is the last probe, not necessarily the returned
  ///        one (istream_replay12 LAST_SPANS, read by kernel_eval8's
  ///        M2_DISP=busy).
  std::vector<WorkgroupSpan> workgroup_spans;
  /// @brief Cycle the last wavefront retired, from the first launch at 0.
  double end = 0.0;
  /// @brief Sum of wavefront lifetimes (end - start).
  double wave_cycles = 0.0;
  std::uint64_t waves = 0;
  std::uint64_t instructions = 0;
  /// @brief Each wavefront's retire cycle, in launch order.
  std::vector<double> wave_ends;
  /// @brief Stream-K flag spins: wavefronts that had one collapsed, and the
  ///        latest cycle one of them issued the first instruction of its
  ///        first spin's kept iteration (negative when none spun).
  std::uint64_t spinning_waves = 0;
  double spin_arrival = -1.0;
  /// @brief Bytes of the pc-averaged vector-memory accesses that went beyond
  ///        L2 / to L2, reads (atomics included) and writes (replay DEMAND).
  double far_read_bytes = 0.0;
  double far_write_bytes = 0.0;
  double l2_read_bytes = 0.0;
  double l2_write_bytes = 0.0;
  /// @brief Memory-queue pass 2: the utilisation pass 1 measured (negative
  ///        when there was no second pass) and the share divisor used.
  double memory_utilisation = -1.0;
  double memory_share = 0.0;
  /// @brief Pass 1's device busy time, cycles (negative without a second
  ///        pass), and the service scale the saturated-device rule settled on
  ///        (negative when it did not apply).
  double memory_device_cycles = -1.0;
  double memory_saturation_scale = -1.0;
};

/// @brief One executed spin loop found in a wavefront's stream.
struct SpinRange {
  /// @brief First record of the first counted iteration.
  std::size_t start = 0;
  /// @brief One past the last counted iteration.
  std::size_t end = 0;
  std::size_t iterations = 0;
};

/// @brief Find flag-spin loops (bench/aiterfa/blt_diag/s3_spin.py spin_ranges):
///        a taken backward branch whose body, from the last earlier record at
///        the target, is only scalar ALU / scalar memory / wait / branch / nop
///        / delay / message, followed by identical taken iterations, at least
///        @p min_iterations in all.
std::vector<SpinRange> find_spins(std::span<const SchedRule> rules,
                                  std::span<const SchedRecord> records,
                                  std::uint64_t min_iterations);

/// @brief One compute unit's scheduled end and, if it spun, its spin arrival.
struct UnitEnd {
  double end = 0.0;
  /// @brief Negative when the unit had no spin.
  double spin_arrival = -1.0;
};

/// @brief Hold every spinning unit until its partner's flag, then one poll.
///
/// @details kernel_eval5 (EVAL_SPIN): the partner's flag time is approximated
/// by the median end of the units that did not spin (the unit's own arrival
/// when none did); a spinning unit ends at max(arrival, partner) + poll + the
/// time it took from its arrival to its end. Units without a spin keep their end.
std::vector<double> hold_spinning_units(std::span<const UnitEnd> units, double poll_cycles);

/// @brief What the kernel-level rule needs to know about a dispatch.
struct KernelShape {
  std::uint64_t workgroups = 1;
  std::uint64_t waves_per_workgroup = 1;
  /// @brief Wavefronts resident per compute unit as the dispatch trace
  ///        reports it (min(waves on the unit, limit)): the teardown tail.
  std::uint64_t resident_waves = 1;
  /// @brief Wavefronts a compute unit can hold (occupancy_waves()): the
  ///        first round of the ramp. 0: resident_waves.
  std::uint64_t occupancy_waves = 0;
  std::uint64_t compute_units = 1;
  std::uint64_t xcds = 1;
  /// @brief The slowest compute unit's scheduled duration, cycles.
  double slowest_unit_cycles = 0.0;
  /// @brief Bytes the release writes back.
  double dirty_bytes = 0.0;
  /// @brief The configured clock: every term is converted to cycles at it.
  double clock_mhz = 1000.0;
  /// @brief Clock the slowest unit's cycles run at (power_clock_mhz());
  ///        0: clock_mhz.
  double body_clock_mhz = 0.0;
};

/// @brief Kernel time from the slowest unit (kernel_eval5 EVAL_KT), microseconds.
///
/// @details The body is the slowest unit's cycles at body_clock_mhz; the
/// total is converted back to cycles at the configured clock_mhz.
struct KernelTime {
  std::uint64_t xcds_used = 1;
  double head_us = 0.0;
  double stagger_us = 0.0;
  double ramp_us = 0.0;
  double body_us = 0.0;
  double tail_us = 0.0;
  double writeback_us = 0.0;
  /// @brief Minimum kernel time that applied.
  double floor_us = 0.0;
  double total_us = 0.0;
  /// @brief total_us at the configured clock, rounded to the nearest cycle.
  std::uint64_t cycles = 0;
};
KernelTime kernel_time(const SchedulerTuning &tuning, const KernelShape &shape);

/// @brief The kernel's XCD dynamic energy from its activity
///        (bench/aiterfa/power_apply/apply.py energy(); SchedulerTuning power_*).
///
/// @details add() counts the executed instructions of the buffered streams
/// per class and normalised mnemonic -- the mix within each class; joules()
/// scales each class's mix to the dispatch's class totals (extract.py) and
/// prices MFMA and VALU / transcendental instructions per instruction, LDS
/// reads and writes per byte, and the dispatch's L2 and fabric bytes per byte.
class KernelEnergy {
public:
  void add(std::span<const SchedRule> rules, std::span<const SchedWave> waves);
  double joules(const SchedulerTuning &tuning,
                const std::array<std::uint64_t, kNumInstClasses> &class_totals, double l2_bytes,
                double fabric_bytes) const;
  bool empty() const { return mix_.empty(); }

private:
  /// @brief (class, normalised mnemonic) -> executions in the buffered streams.
  std::unordered_map<std::string, double> mix_;
  std::array<double, kNumInstClasses> sampled_{};
};

/// @brief Energy of one wave64 MFMA, nanojoules (apply.py MFMA table, per-MAC rule).
double mfma_energy_nj(const SchedulerTuning &tuning, const std::string &mnemonic);
/// @brief Energy of one wave64 VALU or transcendental instruction, nanojoules (apply.py valu_e).
double valu_energy_nj(const SchedulerTuning &tuning, const std::string &mnemonic);
/// @brief Bytes one wave64 LDS instruction moves (apply.py lds_bytes).
double lds_instruction_bytes(const std::string &mnemonic);

/// @brief Clock the body runs at (kernel_eval5 EVAL_CLOCK=rule): with
///        D = joules / (slowest_unit_cycles / reference clock), the
///        configured clock when D <= limit, else
///        min(configured, reference x sqrt(limit / D)). The configured clock
///        when the rule is off or there is no energy or time.
double power_clock_mhz(const SchedulerTuning &tuning, double joules, double slowest_unit_cycles,
                       double clock_mhz);

/// @brief What a compute unit has, for occupancy_waves().
struct OccupancyLimits {
  std::uint64_t wave_slots = 32;
  std::uint64_t vector_registers = 2048;
  std::uint64_t scalar_registers = 3200;
  std::uint64_t lds_bytes = 163840;
  std::uint64_t simds = 4;
};

/// @brief Wavefronts one compute unit can hold for a dispatch (kernel_eval5
///        occ_limit): min(wave slots, SIMDs x (VGPRs per SIMD / VGPRs per
///        wave), SGPRs / SGPRs per wave, (LDS / LDS per workgroup) x waves per
///        workgroup), each at least 1, rounded down to whole workgroups and
///        at least one workgroup. A zero per-wave / per-workgroup demand
///        imposes no limit.
std::uint64_t occupancy_waves(const OccupancyLimits &limits, std::uint64_t vgprs,
                              std::uint64_t sgprs, std::uint64_t lds_bytes,
                              std::uint64_t waves_per_workgroup);

/// @brief Strip the encoding suffix (_e32, _e64, _sdwa, _dpp) the rules ignore.
std::string normalized_mnemonic(const std::string &mnemonic);

/// @brief Bytes one wave64 instruction moves, from its mnemonic (dwordxN, _bN, read2/write2).
double wave_bytes(const std::string &mnemonic);

/// @brief Derive the static rule of one instruction. @p lds_destination: a
///        vector-memory load that writes LDS instead of VGPRs (LDS-DMA).
SchedRule make_rule(const SchedulerTuning &tuning, const std::string &mnemonic,
                    InstClass inst_class, std::uint8_t size_bytes, std::uint16_t immediate,
                    const RegisterRanges &reads, const RegisterRanges &writes,
                    const std::array<std::uint32_t, kNumWaitCounters> &wait, std::uint64_t pc,
                    bool lds_destination = false);

/// @brief What the device-level workgroup dispatch simulation produced.
struct DispatchPlacement {
  /// @brief Latest workgroup end over every CU.
  double end = 0.0;
  /// @brief Workgroups each CU received, indexed
  ///        (xcd x shader engines + engine) x CUs per engine + CU.
  std::vector<std::uint64_t> workgroups_per_cu;
  /// @brief The largest entry of workgroups_per_cu.
  std::uint64_t busiest = 0;
};

/// @brief The measured MI355X workgroup dispatcher over a whole device
///        (bench/aiterfa/disp2/disp_sim.py sim_rr, ramp off; README section 1).
///
/// @details Workgroup i goes to XCD i % xcds; each XCD launches its own
/// workgroups strictly in id order. The k-th workgroup of an XCD goes to
/// shader engine order[k % engines] (dispatch_se_order; XCD 0 uses
/// dispatch_se_order_xcd0). Within the engine a round-robin pointer (engine s
/// starts at CU s) gives it the first CU, cyclically after the last one used,
/// that has room (fewer than @p cap of the dispatch's workgroups resident);
/// when no CU of that engine has room the whole XCD queue waits for the first
/// slot to free, plus @p launch cycles (the relaunch delay). A workgroup that
/// takes a freed slot holds the XCD's queue max(dispatch_issue_cycles,
/// dispatch_issue_cycles_per_wave x waves) cycles; first-round launches are
/// not throttled (the kernel-level ramp covers them). @p duration gives each
/// workgroup's residence (first wave start to last wave end) in cycles.
DispatchPlacement simulate_dispatch(const SchedulerTuning &tuning, std::uint64_t xcds,
                                    std::uint64_t compute_units, std::uint64_t workgroups,
                                    std::uint64_t cap, std::uint64_t waves_per_workgroup,
                                    double launch,
                                    const std::function<double(std::uint64_t)> &duration);

/// @brief Dispatch-wide mean memory behaviour of one vector-memory pc
///        (replay3..7 build_pc_avg, REPLAY_SYMM).
struct PcMemory {
  /// @brief Mean cycles from issue to completion.
  double latency = 0.0;
  /// @brief Mean L1-miss lines per execution.
  double lines = 0.0;
  /// @brief Shares of those lines that went beyond L2 / were served by L2.
  double far_share = 0.0;
  double l2_share = 0.0;
};
using PcMemoryTable = std::unordered_map<std::uint64_t, PcMemory>;

/// @brief Sums vector-memory executions per pc over every modelled wavefront
///        of a dispatch, then turns them into a PcMemoryTable.
class PcMemoryAverager {
public:
  void add(std::uint64_t pc, double latency, double lines, const SchedulerTuning &tuning);
  /// @brief Every vector-memory record of @p waves.
  void add(const SchedulerTuning &tuning, std::span<const SchedRule> rules,
           std::span<const SchedWave> waves);
  /// @brief The per-pc means. @p far_target, when not negative, is the
  ///        dispatch's fabric bytes / L2 bytes (replay5 FAR_TARGET): the
  ///        beyond-L2 lines are rescaled so that they are min(1, far_target)
  ///        of all missing lines, and each pc's latency moves by its moved
  ///        share x (far_latency - l2_latency).
  PcMemoryTable build(const SchedulerTuning &tuning, double far_target) const;
  bool empty() const { return sums_.empty(); }

private:
  struct Sum {
    std::uint64_t executions = 0;
    double latency = 0.0;
    double lines = 0.0;
    double far_lines = 0.0;
    double l2_lines = 0.0;
  };
  std::unordered_map<std::uint64_t, Sum> sums_;
  /// @brief pcs in first-seen order, so the table is built deterministically.
  std::vector<std::uint64_t> order_;
};

/// @brief How the scheduler costs vector-memory accesses.
struct ScheduleMemory {
  /// @brief Per-pc means; null: every access keeps its own latency.
  const PcMemoryTable *pcs = nullptr;
  /// @brief Queue the pc-averaged accesses (REPLAY_MEMQ).
  bool queue = false;
  /// @brief Bandwidth share divisor of this unit (share_divisor()).
  double share = 1.0;
  /// @brief Utilisation the first pass measured; the service is scaled by it
  ///        (floored by the CU link). Negative: unscaled.
  double scale = -1.0;
};

/// @brief What the memory model needs to know about the dispatch.
struct DispatchMemory {
  const PcMemoryTable *pcs = nullptr;
  /// @brief min(compute units, workgroups) (kernel_eval5 `act`).
  double active_cus = 1.0;
  std::uint64_t workgroups = 1;
};

/// @brief Bandwidth share divisor of a unit holding @p workgroups_on_unit of
///        the dispatch's workgroups (replay7 `_act`).
double share_divisor(const SchedulerTuning &tuning, double active_cus, std::uint64_t workgroups,
                     std::uint64_t workgroups_on_unit);

/// @brief Run the scheduler over one compute unit's wavefronts.
///
/// @param waves Any order; wavefronts are grouped by workgroup, workgroups
///        launch in id order, members of one are placed in slot order.
/// @param resident Wavefronts the unit holds at once; a workgroup launches
///        when it fits.
ScheduleResult schedule_compute_unit(const SchedulerTuning &tuning,
                                     std::span<const SchedRule> rules,
                                     std::span<const SchedWave> waves, std::uint64_t resident,
                                     const ScheduleMemory &memory = {});

/// @brief replay4 simulate2pass: pass 1 without the queue measures the unit's
///        demand; utilisation U = sum(bytes x active CUs / device bandwidth) /
///        pass-1 end; pass 2 queues with the service scaled by U when U < 1.
///        When U >= 1 and memory_saturated_device (replay7b/8 R7B_SAT): pass 2
///        at scale 1; if it ends after the busy time, the scale is bisected
///        over (0, 1] so that the unit ends at the busy time.
///        Without pcs (or with the queue off) a single pass.
ScheduleResult schedule_compute_unit(const SchedulerTuning &tuning,
                                     std::span<const SchedRule> rules,
                                     std::span<const SchedWave> waves, std::uint64_t resident,
                                     const DispatchMemory &memory);

/// @brief Buffers one compute unit's instruction streams for the scheduler.
///
/// @details Wavefronts are identified by (workgroup, slot), never by slot
/// alone: workgroups folded onto one modelled unit reuse slot numbers.
class ScheduleRecorder {
public:
  explicit ScheduleRecorder(const SchedulerTuning &tuning) : tuning_(tuning) {}

  void wave_begin(std::uint32_t workgroup, std::uint32_t slot);
  void instruction(std::uint32_t workgroup, std::uint32_t slot, const RetiredInstruction &retired,
                   std::uint64_t occupancy, std::uint64_t latency, std::uint8_t counter,
                   std::uint64_t exposed_fetch, std::uint64_t lines = 0);
  /// @brief Schedule everything buffered and forget it. With @p memory, the
  ///        pc-averaged memory model and its two-pass queue apply.
  ScheduleResult run(std::uint64_t resident, const DispatchMemory *memory = nullptr);
  /// @brief As run(), keeping the buffered streams (the dispatcher may
  ///        schedule the unit again with more workgroups).
  ScheduleResult schedule(std::uint64_t resident, const DispatchMemory *memory = nullptr) const;
  void clear();
  bool empty() const { return waves_.empty(); }

  const std::vector<SchedRule> &rules() const { return rules_; }
  const std::vector<SchedWave> &waves() const { return waves_; }

private:
  SchedWave &wave_for(std::uint32_t workgroup, std::uint32_t slot);
  std::uint32_t rule_for(const RetiredInstruction &retired);

  const SchedulerTuning &tuning_;
  std::vector<SchedRule> rules_;
  std::unordered_map<std::uint64_t, std::uint32_t> rule_index_;
  std::vector<SchedWave> waves_;
  /// @brief (workgroup << 32 | slot) to the live entry in waves_.
  std::unordered_map<std::uint64_t, std::uint32_t> wave_index_;
};

} // namespace rocjitsu::timing
