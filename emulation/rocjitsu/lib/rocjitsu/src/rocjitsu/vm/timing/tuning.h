// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tuning.h
/// @brief Every number the timing plane uses, read from the one config file.
///
/// @details The plane contains no constants. Each field below names a key under
/// the config's `timing.machine` object. There is no second file and no search
/// path: the architecture config rocjitsu is given is the whole description of
/// the part, and whoever produced that config -- mirage, or a person -- is
/// responsible for what is in it. That is the point of baking the numbers in
/// rather than referencing them: a run's timing is reproducible from the one
/// artefact it was given, and a config can be handed to someone without them
/// needing anything else to reproduce it.
///
/// Every value a config does not name resolves to the *slowest reasonable*
/// value for that parameter, not the typical one, and the run reports which
/// ones it fell back to. A forgotten parameter then makes a run read slow and
/// say so, rather than read fast and look like an answer.

#pragma once

#include "rocjitsu/vm/timing/inst_class.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rocjitsu::timing {

/// @brief How a cache level picks the set a line lives in.
enum class SetHash : std::uint8_t {
  /// @brief The line address XOR-folded onto the index bits (the default).
  Fold,
  /// @brief The measured MI355X vector L1 index: four sets of 128-byte lines,
  ///        chosen by a non-linear table over a 5-bit XOR fold of the line
  ///        address. Only defined for four 128-byte-line sets; any other
  ///        geometry falls back to Fold.
  Mi355x,
};

/// @brief Geometry and rates of one cache level.
struct CacheTuning {
  std::uint64_t line_bytes = 4;
  std::uint64_t sets = 1;
  std::uint64_t ways = 1;
  /// @brief Cycles from request to data when the line is resident.
  std::uint64_t hit_cycles = 10000;
  /// @brief Lines this level returns per cycle, per instance.
  double lines_per_cycle = 1.0;
  /// @brief A store allocates here dirty, unread, and reaches memory only when
  ///        the line is evicted or a release flushes it.
  bool write_back = false;
  /// @brief A store passes through without probing or allocating.
  bool write_through = false;
  /// @brief The set index function, `<level>.set_hash`: "fold" or "mi355x".
  ///
  /// @details Optional; absent is "fold". The distinction matters for capacity:
  /// MI355X's vector L1 index puts 9, 9, 7 and 7 of every 32 consecutive lines
  /// into its four sets, so a contiguous footprint overflows the fullest set at
  /// about 225 lines (28 KiB) rather than at the 256 lines the geometry holds.
  SetHash set_hash = SetHash::Fold;
};

/// @brief Which matrix-core rate an opcode's inputs select.
enum class MatrixType : std::uint8_t { Default, F64, F32, F16, BF16, Narrow, Integer, Count };
inline constexpr std::size_t kNumMatrixTypes = static_cast<std::size_t>(MatrixType::Count);

/// @brief Rules of the measured per-SIMD issue scheduler (measured_scheduler.h).
///
/// @details Read from `timing.machine.scheduler.*` only when
/// `scheduler.enabled` is true (or ROCJITSU_TIMING_SCHEDULER=1), so a config
/// that does not use the scheduler reports none of these. The defaults are the
/// MI355X measurements (docs/timing-measured-scheduler.md sections 4 and 7.4; bench/ubench
/// run1..run7); a config naming none of them still reports each as a fallback.
struct SchedulerTuning {
  /// @brief Replace the composed dispatch duration by the scheduled one.
  bool enabled = false;
  /// @brief The config's top-level `cpu_thread_budget` (0: not set). Caps the
  ///        host threads the units are scheduled on at dispatch end; not a
  ///        `timing` key, and it does not change the schedule.
  std::uint64_t host_thread_budget = 0;
  /// @brief SIMDs per CU; wavefronts are placed on them round-robin in launch order.
  std::uint64_t simds_per_cu = 4;
  /// @brief Wave issue interval, 4-byte encodings (run1).
  double issue_interval_cycles = 4.0;
  /// @brief Wave issue interval for 8-byte VALU encodings (VOP3/VOP3P/DPP) (run1).
  double long_encoding_interval_cycles = 4.6;
  /// @brief VALU pipe occupancy of the double-rate group, 4-byte / 8-byte forms (run1, run2).
  double valu_fast_cycles = 2.0;
  double valu_fast_long_encoding_cycles = 2.27;
  /// @brief VALU pipe occupancy of everything not named fast or slow (run1).
  double valu_cycles = 4.0;
  /// @brief Quarter-rate group, transcendentals included (run1, run2).
  double valu_slow_cycles = 8.0;
  std::vector<std::string> valu_fast_mnemonics = {
      "v_mov_b32", "v_add_u32", "v_lshrrev_b32", "v_and_b32", "v_mul_f32", "v_add_f32",
      "v_sub_f32", "v_or_b32",  "v_xor_b32",     "v_sub_u32", "v_fma_f32", "v_fmac_f32"};
  std::vector<std::string> valu_slow_mnemonics = {
      "v_cvt_pk_fp8_f32", "v_permlane32_swap_b32", "v_permlane16_swap_b32", "v_exp_f32",
      "v_rcp_f32",        "v_log_f32",             "v_sqrt_f32",            "v_rsq_f32"};
  /// @brief An SGPR source drops a double-rate VALU op to full occupancy (run7).
  bool sgpr_operand_loses_double_rate = true;
  /// @brief v_cndmask_b32_e32 occupies a CU-shared resource this long, and
  ///        holds its wavefront at least wave_interval (run7).
  double cndmask32_cu_cycles = 4.4;
  double cndmask32_wave_interval_cycles = 5.5;
  /// @brief Cycles an MFMA holds its SIMD's VALU pipe at issue (run4, run6):
  ///        bf16/f16/i8/f8 shapes; the legacy 32x32x8 bf16 / _1k shapes; and
  ///        f32-input shapes add this to their own occupancy.
  double mfma_valu_hold_cycles = 8.0;
  double mfma_valu_hold_legacy_cycles = 4.0;
  double mfma_valu_hold_f32_extra_cycles = 8.0;
  /// @brief Conversion and lane-swap VALU occupancy (istream_replay11 M_OCC,
  ///        branch F / r10 ubench): v_cvt_[scalef32_][sr_][pk_]{fp8,bf8}_f32,
  ///        v_cvt_[scalef32_]sr_[pk_]bf16_f32 and v_permlane{16,32}_swap_b32
  ///        hold the VALU valu_slow_cycles; v_cvt_[scalef32_][pk_]{f32,bf16,f16}_{fp8,bf8}
  ///        and v_cvt_pk_bf16_f32 valu_cycles. Ahead of the slow list.
  bool valu_conversion_table = true;
  /// @brief VALU co-issue within a SIMD's VALU slot (M_PAIR, branch B /
  ///        swln_diag): a double-rate op (occupancy < valu_cycles) opens a
  ///        slot of pair_slot_cycles that only another double-rate op of
  ///        another wave can share; any other VALU op or MFMA waits for the
  ///        slot to end.
  bool valu_pair = true;
  double valu_pair_slot_cycles = 4.0;
  /// @brief 8-byte VALU encodings cost long_encoding_interval only when fewer
  ///        than short_share of the wave's last window instructions (this one
  ///        included) are 4-byte; otherwise issue_interval (M_ENC, branch F,
  ///        r10 ubench: >= 50% 4-byte instructions hide the 8-byte cost).
  bool encoding_context = true;
  std::uint64_t encoding_window = 8;
  double encoding_short_share = 0.5;
  /// @brief cnd32 saturation (M_CND, branch F; ubench11 x_cnd_vs_add,
  ///        x4_cnd3_add): the VOP2 forms of v_cndmask_b32 (e32, SDWA, DPP)
  ///        share the CU's cnd32 resource; when one has to wait for it and the
  ///        resource was last taken from another SIMD, every other VALU op
  ///        issues at intervals of at least saturated_valu_cycles for
  ///        saturation_hold_cycles.
  bool cndmask32_saturation = true;
  double cndmask32_saturated_valu_cycles = 16.0;
  double cndmask32_saturation_hold_cycles = 8.8;
  /// @brief f64 MFMA (M_F64, branch F / r10): 16x16x4 occupies 64 cycles,
  ///        4x4x4 16 (other shapes keep the model's occupancy); the VALU hold
  ///        is the occupancy plus mfma_valu_hold_f32_extra_cycles.
  bool mfma_f64_rule = true;
  double mfma_f64_16x16x4_cycles = 64.0;
  double mfma_f64_4x4x4_cycles = 16.0;
  /// @brief SALU occupancy (run1).
  double salu_cycles = 4.0;
  /// @brief Wavefront hold of an LDS read; b128 reads (run3).
  double lds_read_hold_cycles = 4.0;
  double lds_read_b128_hold_cycles = 8.0;
  /// @brief LDS write hold = bytes per wave / this (thread trace of three kernels).
  double lds_write_bytes_per_cycle = 40.0;
  /// @brief Wave hold of ds_read_b64* and ds_read_b128* (M_DS8 / M_DS16,
  ///        branch A2, ubench f8lds_*: 4.8 and 8.2 cycles per instruction).
  ///        0: off.
  double lds_read_b64_wave_cycles = 4.8;
  double lds_read_b128_wave_cycles = 8.2;
  /// @brief ds_bpermute_b32 / ds_permute_b32 / ds_swizzle_b32 occupy the CU's
  ///        LDS at least permute_cycles (measured 6, address independent; the
  ///        emulator reports 1) and hold the wave permute_wave_cycles (M_PERM,
  ///        M_PERM_IV; branch A2, ubench f8lds_*). 0: off.
  double lds_permute_cycles = 6.0;
  double lds_permute_wave_cycles = 8.3;
  /// @brief A buffer_load / global_load ... lds (a vector-memory read with no
  ///        VGPR destination) also occupies the CU's LDS this long, without
  ///        gating its own issue (M_LDSDMA, branch A2, ubench f8lds_*:
  ///        ~1 cycle, dword or dwordx4). 0: off.
  double lds_dma_cycles = 1.0;
  /// @brief Vector memory hold = max(min, bytes per wave / rate) (run3).
  double vmem_min_hold_cycles = 4.0;
  double vmem_load_bytes_per_cycle = 85.0;
  double vmem_store_bytes_per_cycle = 51.0;
  /// @brief s_nop N holds the wave per_count * (min(N, max) + 1) (run2).
  double nop_cycles_per_count = 4.0;
  std::uint64_t nop_max_count = 15;
  /// @brief Branch costs (run5, run6): not taken; taken to the same or next
  ///        fetch line; taken to a line already fetched; instruction-cache miss.
  double branch_not_taken_cycles = 8.0;
  double branch_near_cycles = 22.0;
  double branch_cached_cycles = 46.0;
  double branch_miss_cycles = 350.0;
  std::uint64_t fetch_line_bytes = 64;
  /// @brief A barrier releases this long after the last member arrives (run2).
  double barrier_release_cycles = 4.0;
  /// @brief The scheduled duration is floored by the bandwidth ledger's bound.
  ///        Only without the memory queue (memory.queue and memory.pc_average)
  ///        unless bandwidth_floor_with_queue: the replay (kernel_eval5) has
  ///        only the queue. See applies_bandwidth_floor().
  bool bandwidth_floor = true;
  bool bandwidth_floor_with_queue = false;

  /// @brief The CU's vector-memory address unit (ubench ras,
  ///        TCP_TOTAL_CACHE_ACCESSES ~1 per clock; replay6 TA_RULE): a dense
  ///        access holds it bytes per wave / address_bytes_per_cycle, a
  ///        scratch (dword-swizzled private segment) dwordx1..x4 access the
  ///        listed cycles. Floors the vector-memory pipe occupancy.
  bool vmem_address_unit = true;
  double vmem_address_bytes_per_cycle = 64.0;
  std::array<double, 4> vmem_scratch_cycles = {4.0, 32.0, 48.0, 64.0};
  /// @brief A store completes (and its vmcnt entry retires) no earlier than
  ///        this long after issue; 0 turns the rule off (ubench ras/run8,
  ///        acked 285-300 cycles after issue when idle; replay6 STORE_ACK).
  double vmem_store_ack_cycles = 290.0;

  /// @brief Memory model of the replay (bench/aiterfa/istream_replay3..7, as
  ///        evaluated by kernel_eval5 in ke9.sh).
  ///
  /// pc_average (REPLAY_SYMM): every vector-memory access of one pc costs the
  /// dispatch-wide mean over all modelled wavefronts of its latency and of
  /// its L1-miss lines; a line counts as beyond L2 when the access took at
  /// least far_min_latency, as L2 when more than l2_min_latency.
  bool memory_pc_average = true;
  double memory_l2_min_latency_cycles = 150.0;
  double memory_far_min_latency_cycles = 450.0;
  /// @brief FAR_TARGET (replay5, EVAL_FARFIX=1): the beyond-L2 share of the
  ///        missing lines is rescaled to the dispatch's fabric bytes / L2
  ///        bytes; a pc's mean latency moves with its moved share by
  ///        far_latency - l2_latency (the emulator's beyond-L2 and L2 latencies).
  bool memory_far_target = true;
  double memory_far_latency_cycles = 535.0;
  double memory_l2_latency_cycles = 210.0;
  /// @brief REPLAY_MEMQ two-pass queue (replay4 simulate2pass): one queue
  ///        per CU for its L1-missing reads and writes; an access occupies it
  ///        lines x line_bytes x share x (far / far bandwidth + l2 / L2
  ///        bandwidth). Device bandwidths in bytes per cycle (ubench mb_bw;
  ///        the L2-hit read bandwidth from ubench l2r, istream_replay9
  ///        R9_L2BW=bw: 8 XCDs x ~14.5 requests/clk x 128 B; mb_bw gave 10000).
  ///        A first pass without the queue measures the CU's utilisation U of
  ///        the device; when U < 1 the second pass scales the service by U,
  ///        never below the CU's own link (cu_link_bytes_per_cycle, measured).
  ///        Needs pc_average.
  bool memory_queue = true;
  double memory_line_bytes = 128.0;
  double memory_l2_read_bytes_per_cycle = 14600.0;
  double memory_l2_write_bytes_per_cycle = 5200.0;
  double memory_far_read_bytes_per_cycle = 2700.0;
  double memory_far_write_bytes_per_cycle = 2400.0;
  double memory_cu_link_bytes_per_cycle = 64.0;
  /// @brief Bandwidth share divisor (replay7 R7_SHARE=demand, EVAL_ROUND=1):
  ///        min(active CUs, workgroups / workgroups on this CU), at least 1.
  ///        False: every active CU (min(compute units, workgroups)) gets an
  ///        equal share (replay4/5/6).
  bool memory_demand_share = true;
  /// @brief Saturated device (istream_replay7b/8 R7B_SAT, aiterfa/over_diag):
  ///        when pass 1's device busy time (sum of bytes x active CUs / device
  ///        bandwidth) is at least pass 1's end, the queued unit ends when the
  ///        device has delivered its share -- other units use its idle queue
  ///        time. The queue's service scale s in (0, 1] is bisected
  ///        (saturated_iterations steps) so that pass 2 ends at the busy time;
  ///        the result is the run at the smallest s that still ends after it.
  bool memory_saturated_device = true;
  std::uint64_t memory_saturated_iterations = 9;
  /// @brief A load completes with its slowest line (istream_replay9
  ///        R9_ANYFAR; aiterfa/under_diag d1/d2): a pc whose loads miss n
  ///        lines on average, a share f of them beyond L2, goes beyond L2 with
  ///        probability 1 - (1 - f)^max(1, n) per load rather than f, so its
  ///        mean latency is at least the no-far latency plus that probability
  ///        x (far_latency - l2_latency).
  bool memory_slowest_line = true;
  /// @brief Outstanding TCP-to-TCC read requests per CU (R9_OUTST; ubench
  ///        l2r: a CU's L2 read throughput saturates at ~118 lines in
  ///        flight). A load's missing lines (rounded half to even, at least
  ///        one, at most the cap) each take an entry; when there are not
  ///        enough, the load and every younger vector-memory access wait for
  ///        the oldest entries to free. An entry of a beyond-L2 line is held
  ///        until the load completes, of an L2 line until l2_latency after the
  ///        load's service starts. 0: no cap.
  std::uint64_t memory_outstanding_requests = 118;
  /// @brief In-flight miss merge (M_MERGE, branch B / swln_diag): a vector
  ///        load that came back within l2_min_latency (an L1 hit) where a
  ///        sibling wave of its workgroup missed at the same pc and occurrence
  ///        takes the mean latency of those misses, before the per-pc mean.
  bool memory_merge_misses = true;

  /// @brief A wavefront's slot frees when it has completed -- its
  ///        outstanding accesses (stores: acknowledged) done -- not when it
  ///        issues its last instruction (R9_SLOTEND; aiterfa/under_diag d7/d8,
  ///        hardware thread traces of the next workgroup's first issue).
  bool slot_release_at_completion = true;
  /// @brief A workgroup launched after the first starts this long after its
  ///        slots free (R9_LAUNCH; under_diag d7/d8: ~180 cycles from the
  ///        last wave's end to the next workgroup's first issue). 0: off.
  double workgroup_launch_cycles = 180.0;
  /// @brief Wave slots per SIMD (M_SIMDSLOT, branch B / swln_diag): each SIMD
  ///        holds resident / simds wavefronts; a workgroup's wavefronts spread
  ///        evenly over the least-loaded SIMDs and it launches only when they
  ///        have room. Off: round-robin placement.
  bool simd_slots = true;

  /// @brief Whether the bandwidth ledger's bound floors the scheduled body.
  bool applies_bandwidth_floor() const {
    return bandwidth_floor && (bandwidth_floor_with_queue || !(memory_queue && memory_pc_average));
  }

  /// @brief Stream-K flag spins (bench/aiterfa/blt_diag s3_spin, kernel_eval5
  ///        EVAL_SPIN): a backward-taken loop whose body is only SALU, SMEM,
  ///        waits, branches, nops and messages, repeated at least
  ///        min_iterations times in a row, is collapsed to its last
  ///        iteration; the unit is then held until its partner's flag
  ///        (approximated by the median end of the non-spinning units) plus
  ///        one poll (EVAL_SPIN_L, 1400 in ke7/ke8).
  bool spin_collapse = true;
  std::uint64_t spin_min_iterations = 4;
  double spin_poll_cycles = 1400.0;

  /// @brief Kernel-level time around the slowest unit (bench/ubench kt_*,
  ///        kernel_eval5 EVAL_KT, cold launch): head; the XCDs start
  ///        xcd_stagger apart; a ramp per workgroup and per wavefront of the
  ///        first round, per XCD; a teardown tail (single-XCD tail when one
  ///        XCD is used); release write-back of the dirty bytes, capped; and a
  ///        minimum with every XCD used / otherwise. Microseconds, converted
  ///        with the configured clock. Off falls back to the fixed launch
  ///        terms plus the flush.
  bool kernel_time = true;
  /// @brief Launch regime of the kernel time (docs section 5): queued behind
  ///        another kernel (back to back; the primary reference) or after an
  ///        idle gap. Queued: head + XCD skew queued_head_us, no stagger,
  ///        tail queued_tail_us, minimum queued_floor_us (ubench kt_* queued
  ///        sweeps; kernel_eval5 EVAL_KTREG=queued).
  bool kernel_queued = true;
  double kernel_queued_head_us = 0.149;
  double kernel_queued_tail_us = 1.25;
  double kernel_queued_floor_us = 3.9;
  double kernel_head_us = 0.12;
  double kernel_xcd_stagger_us = 0.18;
  double kernel_ramp_us_per_workgroup = 0.85e-3;
  double kernel_ramp_us_per_wave = 0.46e-3;
  double kernel_tail_us = 1.10;
  double kernel_tail_us_per_resident_wave = 0.006;
  double kernel_tail_single_xcd_us = 0.6;
  double kernel_writeback_us_per_mb = 0.115;
  double kernel_writeback_cap_us = 2.3;
  double kernel_floor_all_xcds_us = 3.4;
  double kernel_floor_us = 1.8;

  /// @brief Power-limited clock (kernel_eval5 EVAL_CLOCK=rule; bench/aiterfa
  ///        power, power_apply/apply.py). The kernel's XCD dynamic energy E is
  ///        its activity times measured per-operation energies (independent
  ///        of time); D = E / (slowest unit's cycles at reference_mhz); when
  ///        D > limit the body runs at reference_mhz x sqrt(limit / D), never
  ///        above the configured clock. Per-operation energies (duty-corrected,
  ///        random data) per wave64 instruction or per byte. MFMA: the four
  ///        measured shapes; any other shape at the per-MAC energy of the
  ///        measured 32x32 shape of its input type (f16 as bf16; f6/f4/i8 as
  ///        f8). VALU: v_fma_f32; v_pk_*_f32 (pkfma_nc12_rand throttled
  ///        point); f64 twice the FMA; other float arithmetic and conversions
  ///        the FMA; integer / logic / move / compare / select half the FMA
  ///        (an assumption of apply.py). LDS per byte moved (64 lanes); L2 per
  ///        byte served by L2 (L2 bytes - fabric bytes); fabric per byte
  ///        beyond L2 (the XCD's share of it).
  ///        limit_watts = 760 is IN-SAMPLE: set from the four throttled
  ///        development kernels (sustained synthetic loads give 525-550 W).
  bool power_clock = true;
  double power_limit_watts = 760.0;
  double power_reference_mhz = 2400.0;
  double power_mfma_f8_32x32x64_nj = 27.6;
  double power_mfma_f8_16x16x128_nj = 11.9;
  double power_mfma_bf16_32x32x16_nj = 16.3;
  double power_mfma_f32_32x32x2_nj = 15.4;
  double power_valu_fma_nj = 0.95;
  double power_valu_pk_f32_nj = 1.70;
  double power_lds_pj_per_byte = 6.4;
  double power_l2_pj_per_byte = 19.0;
  double power_fabric_pj_per_byte = 24.5;
};

/// @brief The part, as far as the timing plane is concerned.
struct Tuning {
  /// @brief The measured per-SIMD scheduler; off by default.
  SchedulerTuning scheduler;

  bool enabled = false;
  double clock_mhz = 1000.0;

  std::uint64_t compute_units = 1;
  std::uint64_t xcds = 1;
  std::uint64_t simd_lanes = 1;
  std::uint64_t wave_slots_per_cu = 1;
  std::uint64_t vector_registers_per_cu = 1;
  std::uint64_t scalar_registers_per_cu = 1;
  std::uint64_t lds_bytes_per_cu = 1;

  std::array<std::uint64_t, kNumInstClasses> issue_cycles{};
  /// @brief How many of a unit's operations retire per cycle.
  ///
  /// @details Fractional, and not because any unit has a fractional number of
  /// pipes. A class's issue occupancy is an integer count of cycles because
  /// that is what a pipeline stage is, but the *average* rate a unit sustains
  /// over a mixed instruction stream is not an integer, and this is the only
  /// place that average can be expressed. Writing it here rather than rounding
  /// a per-instruction cost keeps the per-instruction costs honest.
  std::array<double, kNumFunctionalUnits> ports{};

  /// @brief Cycles from an instruction's issue until a dependent instruction
  ///        can read the registers it wrote.
  ///
  /// @details Separate from issue_cycles because the two answer different
  /// questions: issue occupancy is how long the port is busy, this is how deep
  /// the pipeline behind it is. A model carrying only the first runs every
  /// compute unit at peak issue rate, which no pipeline does, and it does so
  /// worst on exactly the instruction-dense kernels a timing model is asked
  /// about. A memory access takes none of these: its destination is ready when
  /// the access completes, which the memory path computes.
  std::uint64_t vector_alu_result_cycles = 10000;
  std::uint64_t scalar_alu_result_cycles = 10000;
  std::uint64_t transcendental_result_cycles = 10000;
  std::uint64_t matrix_multiply_result_cycles = 10000;

  std::array<std::uint64_t, kNumMatrixTypes> matrix_macs_per_cycle{};
  double lane_addresses_per_cycle = 1.0;
  /// @brief Cycles before a wavefront may issue again, when its next
  ///        instruction goes to a different functional unit.
  ///
  /// @details A unit's issue occupancy is a throughput cost: how long that unit
  /// is unavailable to everyone. It is not how long the issuing wavefront must
  /// wait before its own next instruction, unless that instruction wants the
  /// same unit. Charging the full occupancy on the wavefront's critical path
  /// serialises a mixed instruction stream that the hardware pipelines, and it
  /// falls hardest on exactly the kernels with no parallelism to absorb it: a
  /// warp-shuffle softmax, which alternates transcendentals, shuffles and
  /// arithmetic, read 1.52 times its measured duration. Real dependencies are
  /// still enforced -- the register scoreboard holds an instruction until its
  /// operands are ready -- so this only removes a stall the hardware does not
  /// have. Defaults to the occupancy itself, which is the old behaviour.
  std::uint64_t wavefront_issue_cycles = 0;
  /// @brief Cycles one compute unit's front end spends on every instruction,
  ///        whatever functional unit it then goes to.
  ///
  /// @details A compute unit can start only so many instructions per cycle
  /// across all its units together, so a kernel whose work is spread thinly
  /// over many of them is limited by the front end rather than by any one of
  /// them. This is the term that binds instruction-dense kernels: at one cycle
  /// per instruction, which is the peak the part can do and not a rate it
  /// sustains, matrix and convolution kernels read 0.62 to 0.72 of measured
  /// while streaming kernels were unaffected, because only the dense ones reach
  /// the front end at all.
  ///
  /// Read from its own key rather than borrowed from the nop class, which is
  /// what it used to be: the value is load bearing and deserves a name that
  /// says what it is.
  std::uint64_t front_end_cycles = 0;
  /// @brief Front-end occupancy of one instruction of each class, in
  ///        sixteenths of a cycle.
  ///
  /// @details The front end is the one resource every instruction takes before
  /// it reaches any execution unit, and charging all of them the same is what
  /// made the model's accuracy a straight median-versus-maximum trade: at one
  /// cycle per instruction the instruction-dense matrix kernels read 0.62 to
  /// 0.72 of measured, and at two the memory-bound elementwise kernels read
  /// 2.37. Neither is a property of the kernels. A matrix instruction, a
  /// scalar load and a lane-parallel add do not occupy the same issue
  /// resources for the same time, and separating them removes the trade: the
  /// worst case fell from 46.9 to 34.7 per cent and the median from 21.8 to
  /// 12.9 with nothing else changed.
  ///
  /// Sixteenths because the useful values are fractions of a cycle and the
  /// accumulator they feed is an integer count of cycles; kFrontEndScale is
  /// divided back out where the total is used.
  std::array<std::uint64_t, kNumInstClasses> front_end_sixteenths{};

  /// @brief Cycles a dispatch's slowest wavefront runs beyond its average one,
  ///        per doubling of the wavefront count.
  ///
  /// @details A dispatch ends when its last wavefront ends. Every compute unit
  /// here runs the same instruction stream and finishes at the same modelled
  /// cycle, so the modelled maximum is the modelled mean and the straggler is
  /// missing entirely. The expected maximum of many independent noisy
  /// durations grows with the logarithm of how many there are, and the
  /// measurements agree: a launch-dominated kernel costs about the same extra
  /// per doubling of its workgroup count.
  ///
  /// Over workgroups, not wavefronts: the spread is between independent units,
  /// and the wavefronts of one workgroup share a compute unit and a barrier.
  std::uint64_t straggler_cycles = 0;

  /// @brief Calibration factors, not measurements.
  ///
  /// @details Each scales a term the model composes, and each is here because
  /// the term is right in shape and wrong in size for a reason this model
  /// cannot presently name. They are separated from the machine parameters
  /// deliberately: a reader has to be able to tell what is a property of the
  /// part from what is a number fitted to a corpus.
  /// @brief How much a barrier per wavefront removes from what other resident
  ///        wavefronts can hide.
  ///
  /// @details Hiding assumes the other resident wavefronts always have
  /// independent work ready the instant this one stalls. A barrier is exactly
  /// the instruction that makes that false: it puts the whole group in
  /// lockstep, so nothing on the far side of one can cover a stall on the near
  /// side. A software-pipelined matrix multiply staging tiles through the
  /// local data share barriers hundreds of times per wavefront and has almost
  /// no hiding available; the model was crediting it with all of it, and that
  /// family read 0.62 of measured. Zero disables it, which is the old
  /// behaviour.
  double barrier_lockstep = 0.0;

  /// @brief Cycles a workgroup barrier costs a wavefront, beyond the spread
  ///        between the wavefronts the model can see.
  ///
  /// @details A barrier costs the difference between the first wavefront to
  /// arrive and the last, and this model gives every wavefront in a group the
  /// same instruction stream, so the modelled difference is near zero and a
  /// barrier comes out nearly free. On hardware the wavefronts arrive skewed
  /// by whatever their memory accesses did, and a kernel that barriers
  /// thousands of times per wavefront -- a blocked matrix multiply staging
  /// tiles through the local data share does exactly that -- pays for all of
  /// them. Fitted, so it lives with the calibration; zero by default, which is
  /// the old behaviour.
  std::uint64_t barrier_cycles = 0;

  /// @brief Whether second-level caches keep their contents across dispatches.
  ///
  /// @details False (the default) drops every L2 at each dispatch's launch acquire. On MI355X a
  /// kernel launch does not: a pointer chase through a buffer the previous kernel touched costs an
  /// L2 hit (about 215 cycles a hop) in the next kernel, against about 895 right after the caches
  /// were flushed. Benchmark loops re-run a kernel on the same buffers, so dropping the L2 makes
  /// every iteration read as the cold first one. First-level caches are invalidated either way.
  bool l2_keep_across_dispatches = false;

  /// @brief How the issue term scales with wavefronts resident per compute
  ///        unit, as an exponent on that count.
  ///
  /// @details Zero is the linear model: a unit's queue is the work on it and
  /// nothing else. Measurement disagrees. The same kernel at four wavefronts
  /// per compute unit reaches a higher instruction rate than at sixteen --
  /// 1.35 against 0.86 instructions per cycle -- because fewer wavefronts
  /// contend for the same issue slot, and a linear term over-charges the
  /// sparse case and under-charges the dense one by exactly that ratio. This
  /// is the smallest correction of the right shape, and it is fitted rather
  /// than derived, which is why it lives with the calibration.
  double issue_occupancy_exponent = 0.0;
  double stall_exposed_fraction = 1.0;
  double latency_exposure_scale = 1.0;
  double fill_exposure_scale = 1.0;
  double fill_ramp_scale = 1.0;
  /// @brief How much of the release flush at a dispatch's end is exposed: the
  ///        dirty lines a write-back cache still holds when the last wavefront
  ///        finishes, written back at the memory's rate after it.
  double flush_exposure_scale = 1.0;

  CacheTuning l1_vector;
  CacheTuning l1_scalar;
  CacheTuning l1_instruction;
  CacheTuning l2;
  CacheTuning mall;

  std::uint64_t dram_latency_cycles = 10000;
  double dram_bytes_per_cycle = 1.0;
  double mall_bytes_per_cycle = 1.0;
  /// @brief Bytes per cycle across the link between the dies and memory,
  ///        charged on every second-level miss whether or not the memory-side
  ///        cache then serves it.
  ///
  /// @details The memory-side cache sits behind this link, so a hit in it saves
  /// a DRAM access and saves nothing on the way there. Leaving this out lets a
  /// working set that happens to fit in the memory-side cache read several
  /// times faster than the part has ever been measured to go: on an eight
  /// megabyte copy it produced 4.96 microseconds against a measured 10.88, and
  /// adding it produced 11.72.
  double fabric_bytes_per_cycle = 1.0;
  std::uint64_t memory_channels = 1;
  std::uint64_t fabric_request_bytes = 64;
  /// @brief Bytes in one DRAM row, per channel.
  std::uint64_t dram_row_bytes = 1024;
  /// @brief Extra cycles a channel spends closing one row and activating
  ///        another. Zero disables the row model.
  std::uint64_t dram_row_miss_cycles = 0;
  std::uint64_t miss_status_registers_per_cu = 1;
  /// @brief How many resident wavefronts' memory stalls genuinely overlap.
  ///
  /// @details Hiding a stall needs another wavefront with work that is ready,
  /// and in a kernel whose wavefronts all do the same thing they are not: they
  /// issue their loads together and wait together, so there is nothing ready to
  /// run. Crediting every resident wavefront's whole issue stream as available
  /// to overlap assumes perfect staggering, which is the best case and not the
  /// common one -- a strided copy with eight wavefronts per compute unit was
  /// credited with 4760 cycles of hiding against a 1320-cycle critical path and
  /// came out at 0.46 of its measured duration. One means the stalls do not
  /// overlap at all; the resident count means they overlap perfectly.
  std::uint64_t stall_overlap_wavefronts = 1;

  std::uint64_t lds_banks = 1;
  std::uint64_t lds_bank_bytes = 4;
  std::uint64_t lds_latency_cycles = 10000;
  std::uint64_t lds_lanes_per_phase = 1;
  /// @brief Granule model of the local data share (lds_granule_cycles());
  ///        zero keeps the phase model.
  std::uint64_t lds_granule_bytes = 0;
  std::uint64_t lds_max_ways = 64;
  double lds_cycles_per_way = 1.0;
  /// @brief Cost an LDS access from its bank multiplicity with the rule measured on MI355X.
  ///
  /// @details Multiplicity is the largest number of 4-byte accesses one instruction sends to a
  /// single bank. Reads cost max(2, multiplicity, x2 for 4-byte lanes) cycles of the CU's LDS;
  /// writes cost max(4 / 6 / 13 for b32 / b64 / b128, 2 x multiplicity); both capped at 64.
  /// Measured with strided and swizzled ds_read/ds_write patterns at 1, 4 and 8 wavefronts
  /// (tests/corpus/ubench LDS pattern sweep). False keeps the granule / conflict models.
  bool lds_measured_banks = false;

  std::uint64_t dispatch_start_cycles = 10000;
  std::uint64_t dispatch_end_cycles = 10000;
  std::uint64_t launch_invalidate_cycles = 10000;
  double workgroups_per_cycle = 0.001;

  /// @brief Keys the config named, and keys it did not, for the run's report.
  std::vector<std::string> resolved;
  std::vector<std::string> fell_back;

  /// @brief Read the `timing` block out of the raw config JSON.
  ///
  /// @details A second, schema-free pass over the same document, for the same
  /// reason the plugin block needs one: the typed load runs with unexpected
  /// fields skipped, so a `timing` block on the typed path would be dropped in
  /// silence and the plane would run entirely on fallbacks with nothing to say
  /// it had.
  static Tuning parse(const std::string &config_json);

  /// @brief Append what was in effect and what was not.
  void write_report(std::string &out) const;
};

} // namespace rocjitsu::timing
