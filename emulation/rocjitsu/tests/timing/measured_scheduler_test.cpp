// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file measured_scheduler_test.cpp
/// @brief Rule-level tests for the measured per-SIMD issue scheduler.
///
/// @details Unlike timing_model_test.cpp these pin exact cycle counts: the
/// scheduler's rules are measured facts (docs/timing-measured-scheduler.md
/// sections 4 and 7.4) carried as SchedulerTuning defaults, and each test builds the
/// shortest instruction stream that exercises one rule.

#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/measured_scheduler.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::timing {
namespace {

/// @brief Builds instruction streams for one compute unit.
class Stream {
public:
  explicit Stream(SchedulerTuning tuning = {}) : tuning_(std::move(tuning)), recorder_(tuning_) {}

  void begin(std::uint32_t workgroup, std::uint32_t slot) { recorder_.wave_begin(workgroup, slot); }

  /// @brief Append one instruction to (workgroup, slot).
  void add(std::uint32_t workgroup, std::uint32_t slot, const std::string &mnemonic,
           InstClass cls, std::uint64_t pc, std::uint8_t size = 4, std::uint64_t occupancy = 0,
           std::uint64_t latency = 0, std::uint8_t counter = 0xFF, std::uint8_t priority = 0,
           RegisterRanges reads = {}, RegisterRanges writes = {},
           std::array<std::uint32_t, kNumWaitCounters> wait = unconstrained(),
           std::uint16_t immediate = 0) {
    mnemonics_.push_back(std::make_unique<std::string>(mnemonic));
    RetiredInstruction retired;
    retired.pc = pc;
    retired.inst_class = cls;
    retired.mnemonic = mnemonics_.back().get();
    retired.size_bytes = size;
    retired.user_priority = priority;
    retired.reads = reads;
    retired.writes = writes;
    retired.wait = wait;
    retired.immediate = immediate;
    recorder_.instruction(workgroup, slot, retired, occupancy, latency, counter, 0);
  }

  /// @brief Append @p count copies of one ALU instruction at consecutive pcs.
  void repeat(std::uint32_t workgroup, std::uint32_t slot, const std::string &mnemonic,
              InstClass cls, std::uint32_t count, std::uint8_t size = 4) {
    for (std::uint32_t index = 0; index < count; ++index)
      add(workgroup, slot, mnemonic, cls, next_pc(workgroup, slot, size), size);
  }

  std::uint64_t next_pc(std::uint32_t workgroup, std::uint32_t slot, std::uint8_t size) {
    std::uint64_t &pc = pcs_[(static_cast<std::uint64_t>(workgroup) << 32) | slot];
    if (pc == 0)
      pc = 0x10000;
    const std::uint64_t here = pc;
    pc += size;
    return here;
  }

  ScheduleResult run(std::uint64_t resident = 32) { return recorder_.run(resident); }

  /// @brief Append one vector-memory access with @p lines L1-miss lines.
  void access(std::uint32_t workgroup, std::uint32_t slot, const std::string &mnemonic,
              InstClass cls, std::uint64_t pc, std::uint64_t latency, std::uint64_t lines,
              std::uint8_t counter) {
    mnemonics_.push_back(std::make_unique<std::string>(mnemonic));
    RetiredInstruction retired;
    retired.pc = pc;
    retired.inst_class = cls;
    retired.mnemonic = mnemonics_.back().get();
    retired.size_bytes = 8;
    retired.wait = unconstrained();
    recorder_.instruction(workgroup, slot, retired, 4, latency, counter, 0, lines);
  }

  /// @brief Schedule without consuming the buffered streams.
  ScheduleResult schedule(const ScheduleMemory &memory, std::uint64_t resident = 32) const {
    return schedule_compute_unit(tuning_, recorder_.rules(), recorder_.waves(), resident, memory);
  }
  ScheduleResult schedule(const DispatchMemory &memory, std::uint64_t resident = 32) const {
    return schedule_compute_unit(tuning_, recorder_.rules(), recorder_.waves(), resident, memory);
  }
  const ScheduleRecorder &recorder() const { return recorder_; }

  static std::array<std::uint32_t, kNumWaitCounters> unconstrained() {
    std::array<std::uint32_t, kNumWaitCounters> wait{};
    wait.fill(kUnconstrained);
    return wait;
  }

  SchedulerTuning tuning_;

private:
  ScheduleRecorder recorder_;
  std::vector<std::unique_ptr<std::string>> mnemonics_;
  std::unordered_map<std::uint64_t, std::uint64_t> pcs_;
};

RegisterRanges vgpr(std::uint16_t index, std::uint8_t count = 1) {
  RegisterRanges ranges;
  ranges.push({RegisterFile::Vector, index, count});
  return ranges;
}

RegisterRanges sgpr(std::uint16_t index) {
  RegisterRanges ranges;
  ranges.push({RegisterFile::Scalar, index, 1});
  return ranges;
}

constexpr std::uint8_t kVmcnt = static_cast<std::uint8_t>(WaitCounter::VectorLoad);

std::array<std::uint32_t, kNumWaitCounters> vmcnt(std::uint32_t threshold) {
  auto wait = Stream::unconstrained();
  wait[kVmcnt] = threshold;
  return wait;
}

// -- Issue interval ----------------------------------------------------------

TEST(MeasuredScheduler, AWaveIssuesOnceEveryFourCycles) {
  // v_mul_f32 holds the VALU pipe 2 cycles; the wave interval (4) binds.
  Stream stream;
  stream.begin(0, 0);
  stream.repeat(0, 0, "v_mul_f32_e32", InstClass::VectorAlu, 10);
  const ScheduleResult result = stream.run();
  EXPECT_DOUBLE_EQ(result.end, 40.0);
  EXPECT_EQ(result.instructions, 10u);
}

TEST(MeasuredScheduler, EightByteEncodingsIssueEveryFourPointSix) {
  Stream stream;
  stream.begin(0, 0);
  stream.repeat(0, 0, "v_mul_f32_e64", InstClass::VectorAlu, 10, 8);
  EXPECT_NEAR(stream.run().end, 46.0, 1e-9);
}

// -- Pipe occupancy ----------------------------------------------------------

TEST(MeasuredScheduler, QuarterRateOpsHoldTheWaveEightCycles) {
  Stream stream;
  stream.begin(0, 0);
  stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  EXPECT_DOUBLE_EQ(stream.run().end, 80.0);
}

TEST(MeasuredScheduler, WavesOnDifferentSimdsDoNotShareAValuPipe) {
  Stream stream;
  for (std::uint32_t slot = 0; slot < 4; ++slot) {
    stream.begin(0, slot);
    stream.repeat(0, slot, "v_exp_f32_e32", InstClass::Transcendental, 10);
  }
  const ScheduleResult result = stream.run();
  EXPECT_DOUBLE_EQ(result.end, 80.0);
}

TEST(MeasuredScheduler, WavesOnOneSimdShareItsValuPipe) {
  SchedulerTuning tuning;
  tuning.simds_per_cu = 1;
  Stream stream(tuning);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    stream.begin(0, slot);
    stream.repeat(0, slot, "v_exp_f32_e32", InstClass::Transcendental, 10);
  }
  EXPECT_DOUBLE_EQ(stream.run().end, 160.0);
}

TEST(MeasuredScheduler, DoubleRateOpsInterleaveTwoWavesOnOneSimd) {
  // Occupancy 2 against an interval of 4: two waves on one SIMD run at full
  // rate each, which is what makes the double-rate group double rate.
  SchedulerTuning tuning;
  tuning.simds_per_cu = 1;
  Stream stream(tuning);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    stream.begin(0, slot);
    stream.repeat(0, slot, "v_add_f32_e32", InstClass::VectorAlu, 10);
  }
  EXPECT_DOUBLE_EQ(stream.run().end, 42.0);
}

TEST(MeasuredScheduler, AnSgprOperandLosesDoubleRate) {
  const SchedulerTuning tuning;
  const auto wait = Stream::unconstrained();
  const SchedRule vector_only = make_rule(tuning, "v_add_f32_e32", InstClass::VectorAlu, 4, 0,
                                          vgpr(1), vgpr(2), wait, 0);
  const SchedRule with_scalar = make_rule(tuning, "v_add_f32_e32", InstClass::VectorAlu, 4, 0,
                                          sgpr(1), vgpr(2), wait, 0);
  EXPECT_DOUBLE_EQ(vector_only.occupancy, 2.0);
  EXPECT_DOUBLE_EQ(with_scalar.occupancy, 4.0);
}

TEST(MeasuredScheduler, MfmaHoldsTheValuPipeEightCycles) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "v_mfma_f32_16x16x32_bf16", InstClass::MatrixMultiply, 0x100, 8, 16);
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x108);
  // MFMA at 0; the independent v_mov waits for the VALU pipe (free at 8).
  EXPECT_DOUBLE_EQ(stream.run().end, 12.0);
}

TEST(MeasuredScheduler, F32MfmaHoldsTheValuPipeItsOccupancyPlusEight) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "v_mfma_f32_32x32x2_f32", InstClass::MatrixMultiply, 0x100, 8, 16);
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x108);
  EXPECT_DOUBLE_EQ(stream.run().end, 28.0);
}

TEST(MeasuredScheduler, ADependentReadWaitsForTheMfmaOccupancy) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "v_mfma_f32_32x32x16_bf16", InstClass::MatrixMultiply, 0x100, 8, 32, 0, 0xFF,
             0, {}, vgpr(0, 16));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x108, 4, 0, 0, 0xFF, 0, vgpr(3));
  EXPECT_DOUBLE_EQ(stream.run().end, 36.0);
}

// -- Arbitration -------------------------------------------------------------

TEST(MeasuredScheduler, TheOldestReadyWaveWins) {
  SchedulerTuning tuning;
  tuning.simds_per_cu = 1;
  Stream stream(tuning);
  for (std::uint32_t slot = 0; slot < 2; ++slot) {
    stream.begin(0, slot);
    stream.repeat(0, slot, "v_exp_f32_e32", InstClass::Transcendental, 10);
  }
  const ScheduleResult result = stream.run();
  // Strict age priority: the older wave takes every contested slot.
  ASSERT_EQ(result.wave_ends.size(), 2u);
  EXPECT_DOUBLE_EQ(result.wave_ends[0], 80.0);
  EXPECT_DOUBLE_EQ(result.wave_ends[1], 160.0);
}

TEST(MeasuredScheduler, SetprioOverridesAge) {
  SchedulerTuning tuning;
  tuning.simds_per_cu = 1;
  Stream stream(tuning);
  stream.begin(0, 0);
  stream.begin(0, 1);
  stream.add(0, 0, "s_setprio", InstClass::ScalarAlu, 0x100);
  stream.add(0, 1, "s_setprio", InstClass::ScalarAlu, 0x100, 4, 0, 0, 0xFF, /*priority=*/3);
  for (std::uint32_t index = 0; index < 10; ++index) {
    stream.add(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 0x104 + 4 * index);
    stream.add(0, 1, "v_exp_f32_e32", InstClass::Transcendental, 0x104 + 4 * index, 4, 0, 0, 0xFF, 3);
  }
  const ScheduleResult result = stream.run();
  ASSERT_EQ(result.wave_ends.size(), 2u);
  EXPECT_LT(result.wave_ends[1], result.wave_ends[0]);
}

// -- Wait counters -----------------------------------------------------------

TEST(MeasuredScheduler, AWaitIssuesThenHoldsUntilTheCounterDrains) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 8, 4, 500, kVmcnt);
  stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x108, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x10c);
  EXPECT_DOUBLE_EQ(stream.run().end, 504.0);
}

TEST(MeasuredScheduler, APartialWaitDrainsOnlyTheOldest) {
  const auto run = [](std::uint32_t first, std::uint32_t second) {
    Stream stream;
    stream.begin(0, 0);
    stream.add(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 8, 4, first, kVmcnt);
    stream.add(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 8, 4, second, kVmcnt);
    stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x110, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(1));
    stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 200);
    return stream.run().end;
  };
  // vmcnt(1) waits for the first load (issued at 0) only.
  EXPECT_DOUBLE_EQ(run(500, 100), 500.0 + 1600.0);
  EXPECT_DOUBLE_EQ(run(100, 500), 100.0 + 1600.0);
}

TEST(MeasuredScheduler, AWaveRetiresWhenItsLastAccessLands) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x100, 8, 4, 700, kVmcnt);
  stream.add(0, 0, "s_endpgm", InstClass::Terminate, 0x108);
  // The store holds the wave 256 / 51 cycles; the wave retires when it lands.
  EXPECT_DOUBLE_EQ(stream.run().end, 700.0);
}

// -- Barriers ----------------------------------------------------------------

TEST(MeasuredScheduler, ABarrierReleasesFourCyclesAfterTheLastArrival) {
  Stream stream;
  stream.begin(0, 0);
  stream.begin(0, 1);
  stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  stream.add(0, 0, "s_barrier", InstClass::Barrier, 0x200);
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x204);
  stream.add(0, 1, "s_barrier", InstClass::Barrier, 0x200);
  stream.add(0, 1, "v_mov_b32_e32", InstClass::VectorAlu, 0x204);
  const ScheduleResult result = stream.run();
  // The last arrival issues its barrier at 80; both leave at 84.
  ASSERT_EQ(result.wave_ends.size(), 2u);
  EXPECT_DOUBLE_EQ(result.wave_ends[0], 88.0);
  EXPECT_DOUBLE_EQ(result.wave_ends[1], 88.0);
}

TEST(MeasuredScheduler, BarriersAreLocalToTheirWorkgroup) {
  Stream stream;
  stream.begin(0, 0);
  stream.begin(1, 0);
  stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  stream.add(0, 0, "s_barrier", InstClass::Barrier, 0x200);
  stream.add(1, 0, "s_barrier", InstClass::Barrier, 0x200);
  stream.add(1, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x204);
  const ScheduleResult result = stream.run();
  ASSERT_EQ(result.wave_ends.size(), 2u);
  // Workgroup 1's only wave passes its own barrier without waiting for 0.
  EXPECT_DOUBLE_EQ(result.wave_ends[1], 8.0);
}

// -- Branches, s_nop, occupancy ----------------------------------------------

TEST(MeasuredScheduler, BranchCostsDependOnWhereTheyGo) {
  const auto run = [](std::uint64_t target) {
    Stream stream;
    stream.begin(0, 0);
    stream.add(0, 0, "s_cbranch_scc1", InstClass::Branch, 0x1000);
    stream.add(0, 0, "s_nop", InstClass::Nop, target);
    return stream.run().end - 4.0;
  };
  EXPECT_DOUBLE_EQ(run(0x1004), 8.0);    // not taken
  EXPECT_DOUBLE_EQ(run(0x1040), 22.0);   // taken, next fetch line
  EXPECT_DOUBLE_EQ(run(0x4000), 350.0);  // taken, never fetched
}

TEST(MeasuredScheduler, SNopHoldsFourCyclesPerCount) {
  Stream stream;
  stream.begin(0, 0);
  stream.add(0, 0, "s_nop", InstClass::Nop, 0x100, 4, 0, 0, 0xFF, 0, {}, {}, Stream::unconstrained(), 3);
  EXPECT_DOUBLE_EQ(stream.run().end, 16.0);
}

/// @brief The launch rules of istream_replay4..8: a slot frees when its wave
///        issues its last instruction, and a workgroup starts when it launches.
SchedulerTuning replay8_launch() {
  SchedulerTuning tuning;
  tuning.slot_release_at_completion = false;
  tuning.workgroup_launch_cycles = 0.0;
  tuning.simd_slots = false;
  return tuning;
}

/// @brief Round-robin SIMD placement (replay9): no per-SIMD slot limit.
SchedulerTuning round_robin() {
  SchedulerTuning tuning;
  tuning.simd_slots = false;
  return tuning;
}

TEST(MeasuredScheduler, AWorkgroupLaunchesWhenItFits) {
  Stream stream(replay8_launch());
  stream.begin(0, 0);
  stream.begin(1, 0);
  stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  stream.repeat(1, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  EXPECT_DOUBLE_EQ(stream.run(/*resident=*/2).end, 80.0);
  Stream serial(replay8_launch());
  serial.begin(0, 0);
  serial.begin(1, 0);
  serial.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  serial.repeat(1, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  // The second workgroup launches when the first issues its last instruction
  // (at 72), as in the offline replay.
  EXPECT_DOUBLE_EQ(serial.run(/*resident=*/1).end, 152.0);
}

TEST(MeasuredScheduler, WaveBytesFollowsTheMnemonic) {
  EXPECT_DOUBLE_EQ(wave_bytes("global_load_dwordx4"), 1024.0);
  EXPECT_DOUBLE_EQ(wave_bytes("ds_read_b128"), 1024.0);
  EXPECT_DOUBLE_EQ(wave_bytes("ds_write2st64_b64"), 1024.0);
  EXPECT_DOUBLE_EQ(wave_bytes("ds_write_b32"), 256.0);
}

// -- Vector-memory address unit (replay6 TA_RULE) ----------------------------

/// @brief Cycles the second of two waves (on different SIMDs) waits for the
///        CU's vector-memory pipe behind the first: the access's occupancy.
double address_unit_hold(const std::string &mnemonic, SchedulerTuning tuning = {}) {
  Stream stream(std::move(tuning));
  stream.begin(0, 0);
  stream.begin(0, 1);
  // The per-wavefront model reported 4 cycles of occupancy and no latency.
  stream.add(0, 0, mnemonic, InstClass::VectorMemoryRead, 0x100, 8, 4, 0);
  stream.add(0, 1, mnemonic, InstClass::VectorMemoryRead, 0x100, 8, 4, 0);
  const ScheduleResult result = stream.run();
  return result.wave_ends[1] - result.wave_ends[0];
}

TEST(MeasuredScheduler, ScratchAccessesHoldTheAddressUnitPerWidth) {
  EXPECT_DOUBLE_EQ(address_unit_hold("scratch_load_dword"), 4.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("scratch_load_dwordx2"), 32.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("scratch_load_dwordx3"), 48.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("scratch_load_dwordx4"), 64.0);
}

TEST(MeasuredScheduler, DenseAccessesHoldTheAddressUnitBytesOverSixtyFour) {
  EXPECT_DOUBLE_EQ(address_unit_hold("global_load_dwordx4"), 16.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("buffer_load_dwordx2"), 8.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("global_load_dword"), 4.0);
  SchedulerTuning off;
  off.vmem_address_unit = false;
  EXPECT_DOUBLE_EQ(address_unit_hold("scratch_load_dwordx4", off), 4.0);
  EXPECT_DOUBLE_EQ(address_unit_hold("global_load_dwordx4", off), 4.0);
}

// -- Store ack (replay6 STORE_ACK) -------------------------------------------

double store_then_end(std::uint32_t latency, double ack) {
  SchedulerTuning tuning;
  tuning.vmem_store_ack_cycles = ack;
  Stream stream(tuning);
  stream.begin(0, 0);
  stream.add(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x100, 8, 4, latency, kVmcnt);
  stream.add(0, 0, "s_endpgm", InstClass::Terminate, 0x108);
  return stream.run().end;
}

TEST(MeasuredScheduler, AStoreCompletesNoEarlierThanItsAck) {
  EXPECT_DOUBLE_EQ(store_then_end(100, 290.0), 290.0);
  EXPECT_DOUBLE_EQ(store_then_end(100, 0.0), 100.0);
  // A store the memory model already made slower keeps its own latency.
  EXPECT_DOUBLE_EQ(store_then_end(700, 290.0), 700.0);
}

TEST(MeasuredScheduler, AWaitRetiresTheStoreBeforeAYoungerLoad) {
  const auto run = [](double ack) {
    SchedulerTuning tuning;
    tuning.vmem_store_ack_cycles = ack;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.add(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x100, 8, 4, 100, kVmcnt);
    stream.add(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 8, 4, 10, kVmcnt);
    stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x110, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(1));
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x114);
    return stream.run().end;
  };
  // vmcnt(1) retires the older store even though the load landed at ~15: the
  // wave resumes at the store's ack (290), not at its 100-cycle latency.
  EXPECT_DOUBLE_EQ(run(290.0), 294.0);
  EXPECT_DOUBLE_EQ(run(0.0), 104.0);
}

// -- Stream-K spin collapse (kernel_eval5 EVAL_SPIN) -------------------------

constexpr std::uint8_t kLgkm = static_cast<std::uint8_t>(WaitCounter::LgkmCombined);

/// @brief Ten quarter-rate ops (80 cycles), then a flag poll executed
///        @p executions times (the last falls through), then s_endpgm.
void spin_stream(Stream &stream, std::uint32_t executions, bool with_valu = false) {
  stream.begin(0, 0);
  stream.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  auto lgkm0 = Stream::unconstrained();
  lgkm0[kLgkm] = 0;
  for (std::uint32_t iteration = 0; iteration < executions; ++iteration) {
    stream.add(0, 0, "s_load_dword", InstClass::ScalarMemory, 0x2000, 8, 4, 50, kLgkm);
    stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x2008, 4, 0, 0, 0xFF, 0, {}, {}, lgkm0);
    stream.add(0, 0, with_valu ? "v_mov_b32_e32" : "s_cmp_eq_u32",
               with_valu ? InstClass::VectorAlu : InstClass::ScalarAlu, 0x200c);
    stream.add(0, 0, "s_cbranch_scc0", InstClass::Branch, 0x2010);
  }
  stream.add(0, 0, "s_endpgm", InstClass::Terminate, 0x2014);
}

TEST(MeasuredScheduler, AFlagSpinCollapsesToItsLastIteration) {
  Stream spun;
  spin_stream(spun, 6);
  const ScheduleResult result = spun.run();
  // Six executions: five taken iterations count, the first four are dropped,
  // the fifth (taken) and sixth (falling through) are kept.
  EXPECT_EQ(result.instructions, 10u + 8u + 1u);
  EXPECT_EQ(result.spinning_waves, 1u);
  EXPECT_DOUBLE_EQ(result.spin_arrival, 80.0);
  SchedulerTuning off;
  off.spin_collapse = false;
  Stream two(off);
  spin_stream(two, 2);
  EXPECT_DOUBLE_EQ(result.end, two.run().end);
}

TEST(MeasuredScheduler, OnlyScalarLoopsOfFourOrMoreIterationsAreSpins) {
  Stream short_loop;
  spin_stream(short_loop, 4); // three taken iterations
  const ScheduleResult a = short_loop.run();
  EXPECT_EQ(a.instructions, 10u + 16u + 1u);
  EXPECT_EQ(a.spinning_waves, 0u);
  EXPECT_LT(a.spin_arrival, 0.0);
  Stream vector_loop;
  spin_stream(vector_loop, 6, /*with_valu=*/true);
  EXPECT_EQ(vector_loop.run().instructions, 10u + 24u + 1u);
}

TEST(MeasuredScheduler, ASpinningUnitIsHeldForItsPartnerThenOnePoll) {
  // Partner = median end of the units that did not spin (2000); the spinning
  // unit arrived at 500 and took 1000 more after its arrival.
  const std::vector<UnitEnd> units = {{1000.0, -1.0}, {3000.0, -1.0}, {2000.0, -1.0}, {1500.0, 500.0}};
  EXPECT_EQ(hold_spinning_units(units, 1400.0), (std::vector<double>{1000.0, 3000.0, 2000.0, 4400.0}));
  // An even count takes the mean of the middle two (1500).
  const std::vector<UnitEnd> even = {{1000.0, -1.0}, {2000.0, -1.0}, {1500.0, 500.0}};
  EXPECT_DOUBLE_EQ(hold_spinning_units(even, 1400.0)[2], 3900.0);
  // A partner that finished before the arrival costs only the poll.
  const std::vector<UnitEnd> early = {{100.0, -1.0}, {1500.0, 500.0}};
  EXPECT_DOUBLE_EQ(hold_spinning_units(early, 1400.0)[1], 2900.0);
  // Nobody to wait for: the unit's own arrival.
  const std::vector<UnitEnd> alone = {{1500.0, 500.0}};
  EXPECT_DOUBLE_EQ(hold_spinning_units(alone, 1400.0)[0], 2900.0);
}

// -- Kernel-level time (kernel_eval5 EVAL_KT) --------------------------------

/// @brief The idle-gap kernel-time rule (docs section 5.1).
SchedulerTuning idle_gap() {
  SchedulerTuning tuning;
  tuning.kernel_queued = false;
  return tuning;
}

TEST(MeasuredScheduler, KernelTimeAddsTheMeasuredLaunchAndTeardownTerms) {
  KernelShape shape;
  shape.workgroups = 512;
  shape.waves_per_workgroup = 4;
  shape.resident_waves = 8; // two workgroups per CU: the first round is 512
  shape.compute_units = 256;
  shape.xcds = 8;
  shape.slowest_unit_cycles = 10000.0;
  shape.dirty_bytes = 4e6;
  shape.clock_mhz = 2000.0;
  const KernelTime time = kernel_time(idle_gap(), shape);
  EXPECT_EQ(time.xcds_used, 8u);
  EXPECT_DOUBLE_EQ(time.head_us, 0.12);
  EXPECT_NEAR(time.stagger_us, 1.26, 1e-12);
  EXPECT_NEAR(time.ramp_us, (0.85e-3 * 512 + 0.46e-3 * 512 * 4) / 8, 1e-12); // 0.17216
  EXPECT_DOUBLE_EQ(time.body_us, 5.0);
  EXPECT_NEAR(time.tail_us, 1.148, 1e-12);
  EXPECT_NEAR(time.writeback_us, 0.46, 1e-12);
  EXPECT_NEAR(time.total_us, 8.16016, 1e-9);
  EXPECT_EQ(time.cycles, 16320u);
}

TEST(MeasuredScheduler, KernelTimeHasAMinimumAndACappedWriteBack) {
  KernelShape one;
  one.compute_units = 256;
  one.xcds = 8;
  one.slowest_unit_cycles = 100.0;
  one.clock_mhz = 2000.0;
  // One workgroup: one XCD, its 0.6 us tail; 0.77 us is under the 1.8 minimum.
  KernelTime time = kernel_time(idle_gap(), one);
  EXPECT_EQ(time.xcds_used, 1u);
  EXPECT_DOUBLE_EQ(time.tail_us, 0.6);
  EXPECT_DOUBLE_EQ(time.total_us, 1.8);
  EXPECT_EQ(time.cycles, 3600u);
  // Eight workgroups reach every XCD: 2.49 us is under the 3.4 minimum.
  KernelShape eight = one;
  eight.workgroups = 8;
  eight.slowest_unit_cycles = 0.0;
  time = kernel_time(idle_gap(), eight);
  EXPECT_EQ(time.xcds_used, 8u);
  EXPECT_DOUBLE_EQ(time.total_us, 3.4);
  // 100 MB of dirty lines would take 11.5 us; the write-back saturates at 2.3.
  eight.dirty_bytes = 100e6;
  EXPECT_DOUBLE_EQ(kernel_time(idle_gap(), eight).writeback_us, 2.3);
}

// -- Tuning ------------------------------------------------------------------

std::string scheduler_config(const std::string &extra) {
  return R"({"timing": {"enabled": true, "clock_mhz": 1000, "machine": {"compute_units": 1)" + extra +
         "}}}";
}

TEST(MeasuredScheduler, IsOffUnlessTheConfigTurnsItOn) {
  const Tuning off = Tuning::parse(scheduler_config(""));
  EXPECT_FALSE(off.scheduler.enabled);
  const Tuning on = Tuning::parse(scheduler_config(R"(, "scheduler.enabled": true,
      "scheduler.branch.miss_cycles": 300, "scheduler.valu.fast_mnemonics": "v_mov_b32, v_add_f32")"));
  EXPECT_TRUE(on.scheduler.enabled);
  EXPECT_DOUBLE_EQ(on.scheduler.branch_miss_cycles, 300.0);
  EXPECT_DOUBLE_EQ(on.scheduler.branch_near_cycles, 22.0);
  EXPECT_EQ(on.scheduler.valu_fast_mnemonics, (std::vector<std::string>{"v_mov_b32", "v_add_f32"}));
  // The rules ported from replay6 / kernel_eval5 default to their measured values.
  EXPECT_TRUE(on.scheduler.vmem_address_unit);
  EXPECT_DOUBLE_EQ(on.scheduler.vmem_scratch_cycles[3], 64.0);
  EXPECT_DOUBLE_EQ(on.scheduler.vmem_store_ack_cycles, 290.0);
  EXPECT_DOUBLE_EQ(on.scheduler.spin_poll_cycles, 1400.0);
  EXPECT_TRUE(on.scheduler.kernel_time);
  EXPECT_TRUE(on.scheduler.memory_pc_average);
  EXPECT_TRUE(on.scheduler.memory_far_target);
  EXPECT_TRUE(on.scheduler.memory_queue);
  EXPECT_TRUE(on.scheduler.memory_demand_share);
  EXPECT_DOUBLE_EQ(on.scheduler.memory_far_read_bytes_per_cycle, 2700.0);
  EXPECT_DOUBLE_EQ(on.scheduler.memory_cu_link_bytes_per_cycle, 64.0);
  const Tuning tuned = Tuning::parse(scheduler_config(R"(, "scheduler.enabled": true,
      "scheduler.vmem.scratch_x4_cycles": 60, "scheduler.vmem.store_ack_cycles": 0,
      "scheduler.spin.collapse": false, "scheduler.kernel.enabled": false,
      "scheduler.kernel.tail_us": 1.2, "scheduler.memory.queue": false,
      "scheduler.memory.l2_write_bytes_per_cycle": 5000)"));
  EXPECT_DOUBLE_EQ(tuned.scheduler.vmem_scratch_cycles[3], 60.0);
  EXPECT_DOUBLE_EQ(tuned.scheduler.vmem_store_ack_cycles, 0.0);
  EXPECT_FALSE(tuned.scheduler.spin_collapse);
  EXPECT_FALSE(tuned.scheduler.kernel_time);
  EXPECT_DOUBLE_EQ(tuned.scheduler.kernel_tail_us, 1.2);
  EXPECT_FALSE(tuned.scheduler.memory_queue);
  EXPECT_DOUBLE_EQ(tuned.scheduler.memory_l2_write_bytes_per_cycle, 5000.0);
}

// -- Memory model of the replay (REPLAY_SYMM, FAR_TARGET, REPLAY_MEMQ) --------

/// @brief Executions of one load pc: (latency, lines).
PcMemoryAverager averaged(const SchedulerTuning &tuning,
                          const std::vector<std::pair<double, double>> &executions,
                          std::uint64_t pc = 0x100) {
  PcMemoryAverager averager;
  for (const auto &[latency, lines] : executions)
    averager.add(pc, latency, lines, tuning);
  return averager;
}

TEST(MeasuredSchedulerMemory, APcCostsItsMeanOverTheDispatch) {
  // replay build_pc_avg: >= 450 is beyond L2, (150, 450) is L2, an access
  // that came back within 150 (or touched no missing line) is neither.
  SchedulerTuning tuning;
  tuning.memory_slowest_line = false;
  PcMemoryAverager averager = averaged(tuning, {{600, 4}, {200, 2}, {100, 0}});
  averager.add(0x200, 140, 3, tuning);
  const PcMemoryTable table = averager.build(tuning, -1.0);
  const PcMemory &load = table.at(0x100);
  EXPECT_DOUBLE_EQ(load.latency, 300.0);
  EXPECT_DOUBLE_EQ(load.lines, 2.0);
  EXPECT_DOUBLE_EQ(load.far_share, 4.0 / 6.0);
  EXPECT_DOUBLE_EQ(load.l2_share, 2.0 / 6.0);
  const PcMemory &near = table.at(0x200);
  EXPECT_DOUBLE_EQ(near.lines, 3.0);
  EXPECT_DOUBLE_EQ(near.far_share, 0.0);
  EXPECT_DOUBLE_EQ(near.l2_share, 0.0);
}

TEST(MeasuredSchedulerMemory, TheFarShareFollowsTheDispatchFabricBytes) {
  // replay5 FAR_TARGET: 6 missing lines, 4 beyond L2. A target of 0.5 wants
  // 3 beyond L2: one line moves to L2 and the mean latency drops by 1/6 of
  // (535 - 210).
  SchedulerTuning tuning;
  tuning.memory_slowest_line = false;
  const PcMemoryAverager averager = averaged(tuning, {{600, 4}, {200, 2}, {100, 0}});
  const PcMemory half = averager.build(tuning, 0.5).at(0x100);
  EXPECT_DOUBLE_EQ(half.far_share, 0.5);
  EXPECT_DOUBLE_EQ(half.l2_share, 0.5);
  EXPECT_DOUBLE_EQ(half.latency, 300.0 - 325.0 / 6.0);
  // Targets above 1 are capped at every missing line beyond L2.
  const PcMemory all = averager.build(tuning, 2.0).at(0x100);
  EXPECT_DOUBLE_EQ(all.far_share, 1.0);
  EXPECT_DOUBLE_EQ(all.l2_share, 0.0);
  EXPECT_DOUBLE_EQ(all.latency, 300.0 + 2.0 * 325.0 / 6.0);
  // With nothing beyond L2 to scale, the target is spread over every pc's
  // missing lines.
  const PcMemory spread = averaged(tuning, {{200, 4}}).build(tuning, 0.25).at(0x100);
  EXPECT_DOUBLE_EQ(spread.far_share, 0.25);
  EXPECT_DOUBLE_EQ(spread.l2_share, 0.75);
  EXPECT_DOUBLE_EQ(spread.latency, 200.0 + 0.25 * 325.0);
}

/// @brief Loads at 0x100 (and 0x108 if @p two), each recorded with a
///        100-cycle latency and 50 lines, then s_waitcnt vmcnt(0) and a v_mov.
void loads_then_wait(Stream &stream, bool two) {
  stream.begin(0, 0);
  stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 100, 50, kVmcnt);
  if (two)
    stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 100, 50, kVmcnt);
  stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x110, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x114);
}

/// @brief Every load costs 300 cycles and 50 lines, all beyond L2.
PcMemoryTable far_loads() {
  PcMemoryTable table;
  table[0x100] = {300.0, 50.0, 1.0, 0.0};
  table[0x108] = {300.0, 50.0, 1.0, 0.0};
  return table;
}

TEST(MeasuredSchedulerMemory, TheRecorderKeepsTheMissLines) {
  Stream stream;
  loads_then_wait(stream, false);
  EXPECT_EQ(stream.recorder().waves().front().records.front().lines, 50u);
  PcMemoryAverager averager;
  averager.add(stream.tuning_, stream.recorder().rules(), stream.recorder().waves());
  const PcMemoryTable table = averager.build(stream.tuning_, -1.0);
  ASSERT_EQ(table.size(), 1u); // only the vector-memory pc
  EXPECT_DOUBLE_EQ(table.at(0x100).latency, 100.0);
  EXPECT_DOUBLE_EQ(table.at(0x100).lines, 50.0);
}

TEST(MeasuredSchedulerMemory, AnAccessCostsItsPcMeanLatency) {
  Stream stream;
  loads_then_wait(stream, false);
  EXPECT_DOUBLE_EQ(stream.schedule(ScheduleMemory{}).end, 104.0);
  const PcMemoryTable table = far_loads();
  ScheduleMemory memory;
  memory.pcs = &table;
  const ScheduleResult result = stream.schedule(memory);
  EXPECT_DOUBLE_EQ(result.end, 304.0);
  // The unit's demand: 50 lines x 128 bytes beyond L2.
  EXPECT_DOUBLE_EQ(result.far_read_bytes, 6400.0);
  EXPECT_DOUBLE_EQ(result.l2_read_bytes, 0.0);
}

TEST(MeasuredSchedulerMemory, TheQueueServesAccessesAtTheUnitsShare) {
  // 6400 bytes beyond L2 at 1/27 of 2700 B/clk: 64 cycles each. The second
  // load waits for the first one's service: 64 + 64 + 300, then the v_mov.
  Stream stream;
  loads_then_wait(stream, true);
  const PcMemoryTable table = far_loads();
  ScheduleMemory memory;
  memory.pcs = &table;
  EXPECT_LT(stream.schedule(memory).end, 320.0);
  memory.queue = true;
  memory.share = 27.0;
  EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 432.0);
  // Pass 2 scales the service by the utilisation, never below the unit's
  // own 64 B/clk link (6400 / 64 = 100 cycles).
  memory.scale = 0.5;
  EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 504.0);
  memory.share = 270.0; // 640 x 0.5 = 320 per access
  EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 944.0);
}

TEST(MeasuredSchedulerMemory, WritesAndL2TrafficUseTheirOwnBandwidth) {
  // 6400 bytes at share 13: L2 write 6400 x 13 / 5200 = 16, then L2 read
  // 6400 x 13 / 14600 behind it (ubench l2r); the wait, then the v_mov.
  SchedulerTuning tuning;
  tuning.vmem_store_ack_cycles = 0.0;
  Stream stream(tuning);
  stream.begin(0, 0);
  stream.access(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x100, 100, 50, kVmcnt);
  stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 100, 50, kVmcnt);
  stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x110, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x114);
  PcMemoryTable table;
  table[0x100] = {300.0, 50.0, 0.0, 1.0};
  table[0x108] = {300.0, 50.0, 0.0, 1.0};
  ScheduleMemory memory;
  memory.pcs = &table;
  memory.queue = true;
  memory.share = 13.0;
  const ScheduleResult result = stream.schedule(memory);
  EXPECT_NEAR(result.end, 16.0 + 6400.0 * 13.0 / 14600.0 + 300.0 + 4.0, 1e-9);
  EXPECT_DOUBLE_EQ(result.l2_write_bytes, 6400.0);
  EXPECT_DOUBLE_EQ(result.l2_read_bytes, 6400.0);
}

TEST(MeasuredSchedulerMemory, TheShareFollowsTheUnitsWorkgroups) {
  // replay7 R7_SHARE=demand: min(active CUs, workgroups / on this unit), >= 1.
  SchedulerTuning tuning;
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 256.0, 1024, 4), 256.0);
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 256.0, 1024, 8), 128.0);
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 100.0, 100, 1), 100.0);
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 256.0, 300, 2), 150.0);
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 2.0, 2, 3), 1.0);
  // Equal shares: every active unit.
  tuning.memory_demand_share = false;
  EXPECT_DOUBLE_EQ(share_divisor(tuning, 256.0, 1024, 8), 256.0);
}

TEST(MeasuredSchedulerMemory, TwoPassesScaleTheQueueByTheUnitsUtilisation) {
  // Pass 1 (no queue) ends at 304 with 6400 bytes beyond L2; at 27 active
  // units the device would be busy 27 x 6400 / 2700 = 64 cycles of them,
  // U = 64 / 304. Pass 2 serves the access in max(100, 64 x U) = 100.
  Stream stream;
  loads_then_wait(stream, false);
  const PcMemoryTable table = far_loads();
  DispatchMemory memory;
  memory.pcs = &table;
  memory.active_cus = 27.0;
  memory.workgroups = 27;
  const ScheduleResult scaled = stream.schedule(memory);
  EXPECT_NEAR(scaled.memory_utilisation, 64.0 / 304.0, 1e-12);
  EXPECT_DOUBLE_EQ(scaled.memory_share, 27.0);
  EXPECT_DOUBLE_EQ(scaled.end, 404.0);
  // A unit that would saturate the device (U >= 1) keeps the full service
  // when the saturated-device rule is off.
  memory.active_cus = 256.0;
  memory.workgroups = 256;
  stream.tuning_.memory_saturated_device = false;
  const ScheduleResult saturated = stream.schedule(memory);
  EXPECT_GT(saturated.memory_utilisation, 1.0);
  EXPECT_NEAR(saturated.end, 256.0 * 6400.0 / 2700.0 + 304.0, 1e-9);
  EXPECT_LT(saturated.memory_saturation_scale, 0.0);
  stream.tuning_.memory_saturated_device = true;
  // The queue off: one pass, pc-averaged latencies only.
  stream.tuning_.memory_queue = false;
  const ScheduleResult single = stream.schedule(memory);
  EXPECT_DOUBLE_EQ(single.end, 304.0);
  EXPECT_LT(single.memory_utilisation, 0.0);
}

TEST(MeasuredSchedulerMemory, ASaturatedDeviceEndsTheUnitWhenItsShareIsDelivered) {
  // istream_replay7b R7B_SAT. 256 active units: the device is busy
  // 256 x 6400 / 2700 = 606.8 cycles for this unit's share, more than pass 1's
  // 304. At scale 1 pass 2 ends at 606.8 + 304; the scale is bisected (9
  // steps) for an end at 606.8: end(s) = 606.8 s + 304 exceeds it only for
  // s > 0.499, so the first midpoint 0.5 is the smallest scale that still ends
  // after it, and every later midpoint (0.25 ... 0.498) ends before it.
  Stream stream;
  loads_then_wait(stream, false);
  const PcMemoryTable table = far_loads();
  DispatchMemory memory;
  memory.pcs = &table;
  memory.active_cus = 256.0;
  memory.workgroups = 256;
  const double busy = 256.0 * 6400.0 / 2700.0;
  const ScheduleResult result = stream.schedule(memory);
  EXPECT_NEAR(result.memory_device_cycles, busy, 1e-9);
  EXPECT_NEAR(result.memory_utilisation, busy / 304.0, 1e-12);
  EXPECT_DOUBLE_EQ(result.memory_saturation_scale, 0.5);
  EXPECT_NEAR(result.end, 0.5 * busy + 304.0, 1e-9);
  // No bisection steps: the run at scale 1 (the CU link floor of 100 does
  // not bind).
  stream.tuning_.memory_saturated_iterations = 0;
  const ScheduleResult full = stream.schedule(memory);
  EXPECT_DOUBLE_EQ(full.memory_saturation_scale, 1.0);
  EXPECT_NEAR(full.end, busy + 304.0, 1e-9);
  // Two steps: 0.5 ends after the busy time, 0.25 before it.
  stream.tuning_.memory_saturated_iterations = 2;
  EXPECT_DOUBLE_EQ(stream.schedule(memory).memory_saturation_scale, 0.5);
  // A unit whose queue at scale 1 already ends by the busy time keeps it:
  // 64 workgroups on 256 active units, share 64, service 151.7.
  stream.tuning_.memory_saturated_iterations = 9;
  memory.workgroups = 64;
  const ScheduleResult early = stream.schedule(memory);
  EXPECT_DOUBLE_EQ(early.memory_saturation_scale, 1.0);
  EXPECT_NEAR(early.end, 64.0 * 6400.0 / 2700.0 + 304.0, 1e-9);
  // Below saturation the utilisation rule applies and this one does not.
  memory.active_cus = 27.0;
  memory.workgroups = 27;
  const ScheduleResult unsaturated = stream.schedule(memory);
  EXPECT_LT(unsaturated.memory_saturation_scale, 0.0);
  EXPECT_DOUBLE_EQ(unsaturated.end, 404.0);
}

TEST(MeasuredSchedulerMemory, TheBandwidthFloorIsOffWithTheMemoryQueue) {
  // kernel_eval5 has only the queue; the ledger's floor applies without it,
  // or with bandwidth_floor_with_queue.
  SchedulerTuning tuning;
  EXPECT_FALSE(tuning.applies_bandwidth_floor());
  tuning.bandwidth_floor_with_queue = true;
  EXPECT_TRUE(tuning.applies_bandwidth_floor());
  tuning.bandwidth_floor_with_queue = false;
  tuning.memory_queue = false;
  EXPECT_TRUE(tuning.applies_bandwidth_floor());
  tuning.memory_queue = true;
  tuning.memory_pc_average = false; // the queue needs the per-pc means
  EXPECT_TRUE(tuning.applies_bandwidth_floor());
  tuning.bandwidth_floor = false;
  EXPECT_FALSE(tuning.applies_bandwidth_floor());
  const Tuning parsed = Tuning::parse(scheduler_config(R"(, "scheduler.enabled": true)"));
  EXPECT_FALSE(parsed.scheduler.applies_bandwidth_floor());
  EXPECT_TRUE(parsed.scheduler.memory_saturated_device);
  EXPECT_EQ(parsed.scheduler.memory_saturated_iterations, 9u);
  const Tuning floored = Tuning::parse(scheduler_config(R"(, "scheduler.enabled": true,
      "scheduler.bandwidth_floor_with_queue": 1, "scheduler.memory.saturated_device": 0,
      "scheduler.memory.saturated_iterations": 5)"));
  EXPECT_TRUE(floored.scheduler.applies_bandwidth_floor());
  EXPECT_FALSE(floored.scheduler.memory_saturated_device);
  EXPECT_EQ(floored.scheduler.memory_saturated_iterations, 5u);
}

// -- Occupancy ---------------------------------------------------------------

TEST(MeasuredScheduler, OccupancyIsWholeWorkgroupsOfTheTightestLimit) {
  // kernel_eval5 occ_limit with the MI355X numbers (32 slots, 4 x 512 VGPRs,
  // 3200 SGPRs, 160 KiB LDS).
  const OccupancyLimits mi355x;
  EXPECT_EQ(occupancy_waves(mi355x, 0, 0, 0, 4), 32u);
  EXPECT_EQ(occupancy_waves(mi355x, 128, 0, 0, 4), 16u);
  // Per SIMD: 4 x (512 / 129) = 12, not 2048 / 129 = 15.
  EXPECT_EQ(occupancy_waves(mi355x, 129, 0, 0, 1), 12u);
  // 3200 / 128 = 25 SGPR-limited waves hold three eight-wave workgroups (24;
  // aiterfa/over_diag/occ_check: f8m4096n1024k1024 m32_n64_k64_w8_s2, trace
  // resident 25, hardware concurrency 24).
  EXPECT_EQ(occupancy_waves(mi355x, 0, 128, 0, 8), 24u);
  // LDS is per workgroup: two 64 KiB workgroups of four waves.
  EXPECT_EQ(occupancy_waves(mi355x, 0, 0, 65536, 4), 8u);
  // At least one workgroup, even when a limit is below its size.
  EXPECT_EQ(occupancy_waves(mi355x, 512, 0, 0, 8), 8u);
  EXPECT_EQ(occupancy_waves(mi355x, 0, 0, 200000, 2), 2u);
  OccupancyLimits ten = mi355x;
  ten.wave_slots = 10;
  EXPECT_EQ(occupancy_waves(ten, 0, 0, 0, 4), 8u);
}

/// @brief Workgroups 0..groups-1 of four waves; slot s runs s + 2 `s_nop 15`
///        (64 cycles each, no pipe), so only launch timing matters. A slot
///        frees when its wave issues its last instruction: at 64 (s + 1).
void staggered_groups(Stream &stream, std::uint32_t groups) {
  for (std::uint32_t wg = 0; wg < groups; ++wg)
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      stream.begin(wg, slot);
      for (std::uint32_t index = 0; index <= slot + 1; ++index)
        stream.add(wg, slot, "s_nop", InstClass::Nop, stream.next_pc(wg, slot, 4), 4, 0, 0, 0xFF,
                   0, {}, {}, Stream::unconstrained(), 15);
    }
}

TEST(MeasuredScheduler, AWorkgroupWaitsForRoomForAllItsWaves) {
  // Three four-wave workgroups whose waves last 128, 192, 256, 320. 12 hold
  // all three. With room for 10 the third launches when two waves have freed
  // their slots (64); whole workgroups (8) make it wait for four (128); with
  // 4 they run one at a time.
  const auto end = [](std::uint64_t resident) {
    Stream stream(replay8_launch());
    staggered_groups(stream, 3);
    return stream.run(resident).end;
  };
  const double twelve = end(12);
  const double ten = end(10);
  const double eight = end(8);
  const double four = end(4);
  EXPECT_LT(twelve, ten);
  EXPECT_LT(ten, eight);
  EXPECT_LT(eight, four);
  // Each later launch adds the 64 cycles between retirements.
  EXPECT_DOUBLE_EQ(ten - twelve, 64.0);
  EXPECT_DOUBLE_EQ(eight - twelve, 128.0);
  EXPECT_DOUBLE_EQ(twelve, 320.0);
  EXPECT_DOUBLE_EQ(four, 832.0);
}

// -- Plane level -------------------------------------------------------------

std::string plane_config(const std::string &extra = {}) {
  return R"({"timing": {"enabled": true, "clock_mhz": 1000, "machine": {
      "compute_units": 4, "xcds": 1, "simd_lanes": 16, "wave_slots_per_cu": 8,
      "vector_registers_per_cu": 512, "scalar_registers_per_cu": 800, "lds_bytes_per_cu": 65536,
      "memory_channels": 2, "vector_alu.issue_cycles": 4, "vector_alu.ports": 4,
      "l1_vector.hit_cycles": 10, "l1_vector.line_bytes": 64, "l1_vector.sets": 4, "l1_vector.ways": 4,
      "l1_instruction.hit_cycles": 10, "l1_instruction.line_bytes": 64,
      "l2.hit_cycles": 40, "l2.line_bytes": 64, "l2.sets": 16, "l2.ways": 4,
      "dispatch.start_cycles": 100, "dispatch.end_cycles": 50, "dispatch.invalidate_cycles": 10,
      "dispatch.workgroups_per_cycle": 1, "front_end.issue_cycles": 1,
      "vector_alu.result_cycles": 4, "wavefront_issue_cycles": 4,
      "mall.hit_cycles": 80, "mall.line_bytes": 64, "mall.sets": 64, "mall.ways": 4,
      "dram.latency_cycles": 200, "dram.bytes_per_cycle": 64, "mall.bytes_per_cycle": 128,
      "fabric.bytes_per_cycle": 128)" +
         extra + "}}}";
}

DispatchShape plane_shape(std::uint32_t workgroups) {
  DispatchShape shape;
  shape.dispatch_id = 1;
  shape.kernel_name = "test_kernel";
  shape.workgroup_count = workgroups;
  shape.waves_per_workgroup = 1;
  return shape;
}

RetiredInstruction plane_alu(std::uint64_t pc, const std::string *mnemonic) {
  RetiredInstruction retired;
  retired.pc = pc;
  retired.inst_class = InstClass::VectorAlu;
  retired.mnemonic = mnemonic;
  retired.active_lanes = 64;
  retired.wait.fill(kUnconstrained);
  return retired;
}

/// @brief Eight one-wave workgroups on four units: workgroups w and w + 4 fold
///        onto the same unit with the same slot (0).
std::uint64_t run_folded(TimingPlane &plane, bool interleaved, std::uint32_t count = 32,
                         std::uint32_t dispatch = 1) {
  static const std::string mnemonic = "v_exp_f32_e32";
  DispatchShape shape = plane_shape(8);
  shape.dispatch_id = dispatch;
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  if (interleaved) {
    for (std::uint32_t wg = 0; wg < 8; ++wg)
      plane.wave_begin(wg, 0, shape.dispatch_id, shape.queue_id);
    for (std::uint32_t index = 0; index < count; ++index)
      for (std::uint32_t wg = 0; wg < 8; ++wg)
        plane.instruction(wg, 0, plane_alu(0x1000 + 4 * index, &mnemonic));
    for (std::uint32_t wg = 0; wg < 8; ++wg)
      plane.wave_end(wg, 0);
  } else {
    for (std::uint32_t wg = 0; wg < 8; ++wg) {
      plane.wave_begin(wg, 0, shape.dispatch_id, shape.queue_id);
      for (std::uint32_t index = 0; index < count; ++index)
        plane.instruction(wg, 0, plane_alu(0x1000 + 4 * index, &mnemonic));
      plane.wave_end(wg, 0);
    }
  }
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  return plane.current_cycles() - before;
}

TEST(TimingWaveIdentity, FoldedWorkgroupsKeepTheirOwnTimelines) {
  // Workgroups folded onto one unit reuse slot numbers. Interleaving them --
  // which the functional plane does -- must not merge or reset their timelines.
  TimingPlane sequential{Tuning::parse(plane_config())};
  TimingPlane interleaved{Tuning::parse(plane_config())};
  EXPECT_EQ(run_folded(interleaved, true), run_folded(sequential, false));
}

TEST(TimingWaveIdentity, TheScheduledDispatchIsIndependentOfRetireOrder) {
  // Fixed launch terms around the slowest unit.
  const std::string on = R"(, "scheduler.enabled": true, "scheduler.kernel.enabled": false)";
  TimingPlane sequential{Tuning::parse(plane_config(on))};
  TimingPlane interleaved{Tuning::parse(plane_config(on))};
  const std::uint64_t a = run_folded(sequential, false);
  EXPECT_EQ(run_folded(interleaved, true), a);
  // Two one-wave workgroups per unit, on different SIMDs: 32 more quarter-rate
  // ops per wave cost exactly 32 x 8 more cycles.
  TimingPlane longer{Tuning::parse(plane_config(on))};
  EXPECT_EQ(run_folded(longer, false, 64), a + 256u);
}

TEST(TimingWaveIdentity, TheKernelTimeWrapsTheSlowestUnit) {
  // Measured idle-gap kernel-level terms. One XCD at 1000 MHz: head 120,
  // ramp 8 x (0.85 + 0.46) ns = 10.48, tail 600 cycles, minimum 1800. Long
  // enough streams to clear the minimum differ by the scheduled difference.
  const std::string on = R"(, "scheduler.enabled": true, "scheduler.kernel.regime": "idle_gap")";
  TimingPlane sequential{Tuning::parse(plane_config(on))};
  TimingPlane interleaved{Tuning::parse(plane_config(on))};
  const std::uint64_t a = run_folded(sequential, false, 200);
  EXPECT_EQ(run_folded(interleaved, true, 200), a);
  TimingPlane longer{Tuning::parse(plane_config(on))};
  EXPECT_EQ(run_folded(longer, false, 232), a + 256u);
  // Short streams sit on the single-XCD minimum.
  TimingPlane shorter{Tuning::parse(plane_config(on))};
  EXPECT_EQ(run_folded(shorter, false, 32), 1800u);
}

/// @brief @p groups four-wave workgroups; slot s runs s + 2 `s_nop 15`.
std::uint64_t run_groups(TimingPlane &plane, std::uint32_t groups) {
  static const std::string mnemonic = "s_nop";
  DispatchShape shape = plane_shape(groups);
  shape.waves_per_workgroup = 4;
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  for (std::uint32_t wg = 0; wg < groups; ++wg)
    for (std::uint32_t slot = 0; slot < 4; ++slot) {
      plane.wave_begin(wg, slot, shape.dispatch_id, shape.queue_id);
      for (std::uint32_t index = 0; index <= slot + 1; ++index) {
        RetiredInstruction nop = plane_alu(0x1000 + 4 * index, &mnemonic);
        nop.inst_class = InstClass::Nop;
        nop.immediate = 15;
        plane.instruction(wg, slot, nop);
      }
      plane.wave_end(wg, slot);
    }
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  return plane.current_cycles() - before;
}

std::string slots_config(const std::string &slots) {
  std::string config =
      plane_config(R"(, "scheduler.enabled": true, "scheduler.kernel.enabled": false)");
  const std::string from = R"("wave_slots_per_cu": 8)";
  config.replace(config.find(from), from.size(), R"("wave_slots_per_cu": )" + slots);
  return config;
}

TEST(TimingWaveIdentity, TheSchedulerHoldsWholeWorkgroupsOfTheOccupancyLimit) {
  // The scheduler's resident limit is the unit's occupancy (whole
  // workgroups), not the wavefronts the unit happened to hold.
  // Two four-wave workgroups per unit run at once under a limit of 8 or 10,
  // and one at a time under 4.
  TimingPlane eight{Tuning::parse(slots_config("8"))};
  TimingPlane ten{Tuning::parse(slots_config("10"))};
  TimingPlane four{Tuning::parse(slots_config("4"))};
  const std::uint64_t two_at_once = run_groups(eight, 8);
  EXPECT_EQ(run_groups(ten, 8), two_at_once);
  EXPECT_GT(run_groups(four, 8), two_at_once);
  // Three per unit: 10 slots hold two workgroups, exactly like 8.
  TimingPlane eight3{Tuning::parse(slots_config("8"))};
  TimingPlane ten3{Tuning::parse(slots_config("10"))};
  TimingPlane twelve3{Tuning::parse(slots_config("12"))};
  const std::uint64_t whole = run_groups(eight3, 12);
  EXPECT_EQ(run_groups(ten3, 12), whole);
  EXPECT_NE(run_groups(twelve3, 12), whole);
}

TEST(MeasuredScheduler, TheRampsFirstRoundIsWhatACuCanHold) {
  // 1024 four-wave workgroups on 256 CUs and 8 XCDs; the trace's resident is
  // 4 (one workgroup on the reported unit), the occupancy limit 12. First
  // round 768 workgroups: (0.85 + 4 x 0.46) ns x 768 / 8. The tail keeps the
  // reported resident: 1.10 + 0.006 x 4.
  const SchedulerTuning tuning = idle_gap();
  KernelShape shape;
  shape.workgroups = 1024;
  shape.waves_per_workgroup = 4;
  shape.resident_waves = 4;
  shape.occupancy_waves = 12;
  shape.compute_units = 256;
  shape.xcds = 8;
  shape.clock_mhz = 2400.0;
  const KernelTime time = kernel_time(tuning, shape);
  EXPECT_NEAR(time.ramp_us, (0.85e-3 + 4 * 0.46e-3) * 768.0 / 8.0, 1e-12);
  EXPECT_NEAR(time.tail_us, 1.10 + 0.006 * 4.0, 1e-12);
  shape.occupancy_waves = 0; // falls back to the reported resident
  EXPECT_NEAR(kernel_time(tuning, shape).ramp_us, (0.85e-3 + 4 * 0.46e-3) * 256.0 / 8.0, 1e-12);
}

// -- Rules of istream_replay9 --------------------------------------------------

TEST(MeasuredSchedulerMemory, ALoadCompletesWithItsSlowestLine) {
  // R9_ANYFAR. The pc misses 2 lines per load, 2/3 of them beyond L2: a load
  // goes beyond L2 with probability 1 - (1/3)^2 = 8/9, not 2/3. Its mean
  // latency without far lines is 300 - 2/3 x 325; with the rule it is that
  // plus 8/9 x 325.
  const SchedulerTuning tuning;
  EXPECT_TRUE(tuning.memory_slowest_line);
  const PcMemoryAverager averager = averaged(tuning, {{600, 4}, {200, 2}, {100, 0}});
  const PcMemory load = averager.build(tuning, -1.0).at(0x100);
  EXPECT_DOUBLE_EQ(load.latency, 300.0 - 2.0 / 3.0 * 325.0 + 8.0 / 9.0 * 325.0);
  EXPECT_DOUBLE_EQ(load.far_share, 4.0 / 6.0); // the shares do not change
  // After the FAR_TARGET rescale (half beyond L2): 1 - 0.5^2 = 3/4.
  const PcMemory half = averager.build(tuning, 0.5).at(0x100);
  EXPECT_DOUBLE_EQ(half.latency, 300.0 - 325.0 / 6.0 - 0.5 * 325.0 + 0.75 * 325.0);
  // One line per load (or fewer): the line share is the load's probability.
  const PcMemory one = averaged(tuning, {{600, 1}, {200, 1}}).build(tuning, -1.0).at(0x100);
  EXPECT_DOUBLE_EQ(one.latency, 400.0);
  const PcMemory part = averaged(tuning, {{600, 1}, {100, 0}}).build(tuning, -1.0).at(0x100);
  EXPECT_DOUBLE_EQ(part.latency, 350.0);
}

/// @brief @p count loads at 0x100, 0x108, ... (50 L1-miss lines each), then
///        s_waitcnt vmcnt(0) and a v_mov; the table gives each @p latency
///        and puts its lines beyond L2 (@p far) or in L2.
PcMemoryTable loads(Stream &stream, std::uint32_t count, double latency, bool far) {
  PcMemoryTable table;
  stream.begin(0, 0);
  for (std::uint32_t index = 0; index < count; ++index) {
    const std::uint64_t pc = 0x100 + 8 * index;
    stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, pc, 100, 50, kVmcnt);
    table[pc] = {latency, 50.0, far ? 1.0 : 0.0, far ? 0.0 : 1.0};
  }
  stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x400, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x404);
  return table;
}

TEST(MeasuredSchedulerMemory, ACuHoldsAtMost118RequestsToL2) {
  // R9_OUTST (ubench l2r). Loads issue every 4 cycles and take 50 entries
  // each. Two fit in 118: the second completes at 4 + 300, the v_mov issues
  // then. A third waits until the first one's entries free at 300.
  const SchedulerTuning tuning;
  EXPECT_EQ(tuning.memory_outstanding_requests, 118u);
  ScheduleMemory memory;
  {
    Stream stream;
    const PcMemoryTable table = loads(stream, 2, 300.0, true);
    memory.pcs = &table;
    EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 308.0);
  }
  {
    Stream stream;
    const PcMemoryTable table = loads(stream, 3, 300.0, true);
    memory.pcs = &table;
    EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 300.0 + 300.0 + 4.0);
  }
  {
    // L2 lines free their entries when they return (210 after the load's
    // service starts), not when the load completes.
    Stream stream;
    const PcMemoryTable table = loads(stream, 3, 300.0, false);
    memory.pcs = &table;
    EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 210.0 + 300.0 + 4.0);
  }
  {
    // No cap: the third load issues at 8.
    SchedulerTuning off;
    off.memory_outstanding_requests = 0;
    Stream stream(off);
    const PcMemoryTable table = loads(stream, 3, 300.0, true);
    memory.pcs = &table;
    EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 312.0);
  }
}

TEST(MeasuredSchedulerMemory, AWaitingLoadHoldsTheYoungerAccessesBehindIt) {
  // A cap of 60: the second load waits for the first one's entries (300);
  // the vector-memory path is in order, so the store behind it issues at
  // 300 too, and is acked 290 later. Stores take no entries.
  SchedulerTuning tuning;
  tuning.memory_outstanding_requests = 60;
  Stream stream(tuning);
  PcMemoryTable table;
  stream.begin(0, 0);
  stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 100, 50, kVmcnt);
  stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 100, 50, kVmcnt);
  stream.access(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x110, 100, 50, kVmcnt);
  stream.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x400, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x404);
  table[0x100] = {300.0, 50.0, 1.0, 0.0};
  table[0x108] = {300.0, 50.0, 1.0, 0.0};
  table[0x110] = {100.0, 50.0, 1.0, 0.0};
  ScheduleMemory memory;
  memory.pcs = &table;
  EXPECT_DOUBLE_EQ(stream.schedule(memory).end, 300.0 + 300.0 + 4.0);
  tuning.memory_outstanding_requests = 0;
  Stream uncapped(tuning);
  uncapped.begin(0, 0);
  uncapped.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 100, 50, kVmcnt);
  uncapped.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x108, 100, 50, kVmcnt);
  uncapped.access(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x110, 100, 50, kVmcnt);
  uncapped.add(0, 0, "s_waitcnt", InstClass::WaitCounter, 0x400, 4, 0, 0, 0xFF, 0, {}, {}, vmcnt(0));
  uncapped.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x404);
  // Store at 8, acked at 298; the second load completes at 304.
  EXPECT_DOUBLE_EQ(uncapped.schedule(memory).end, 308.0);
}

TEST(MeasuredScheduler, ASlotFreesWhenItsWaveCompletesThenTheNextWorkgroupLaunches180Later) {
  // R9_SLOTEND, R9_LAUNCH. One slot. Workgroup 0 stores and ends; its store
  // is acked at 290, so its slot frees then, and workgroup 1 (ten
  // quarter-rate ops, 80 cycles) starts 180 cycles later.
  const auto end = [](bool release, double delay) {
    SchedulerTuning tuning;
    tuning.slot_release_at_completion = release;
    tuning.workgroup_launch_cycles = delay;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(1, 0);
    stream.access(0, 0, "global_store_dword", InstClass::VectorMemoryWrite, 0x100, 100, 0, kVmcnt);
    stream.repeat(1, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
    return stream.run(/*resident=*/1).end;
  };
  EXPECT_DOUBLE_EQ(end(true, 180.0), 290.0 + 180.0 + 80.0);
  EXPECT_DOUBLE_EQ(end(true, 0.0), 290.0 + 80.0);
  // replay8: the slot frees when the store issues; nothing waits for the ack.
  EXPECT_DOUBLE_EQ(end(false, 0.0), 290.0);
  EXPECT_DOUBLE_EQ(end(false, 180.0), 290.0);
  // Workgroups that launch at the start do not wait: two slots, one store each.
  SchedulerTuning tuning;
  Stream both(tuning);
  both.begin(0, 0);
  both.begin(1, 0);
  both.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  both.repeat(1, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  EXPECT_DOUBLE_EQ(both.run(/*resident=*/2).end, 80.0);
  // The ALU-only serial case: the slot frees at 80, the second starts at 260.
  Stream serial(tuning);
  serial.begin(0, 0);
  serial.begin(1, 0);
  serial.repeat(0, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  serial.repeat(1, 0, "v_exp_f32_e32", InstClass::Transcendental, 10);
  EXPECT_DOUBLE_EQ(serial.run(/*resident=*/1).end, 80.0 + 180.0 + 80.0);
}

TEST(MeasuredScheduler, StaggeredWorkgroupsLaunchWhenTheirSlotsHaveCompleted) {
  // staggered_groups with the default launch rules: slot s frees when its
  // wave completes, at 64 (s + 2), and a workgroup starts 180 after it fits.
  // Under 8 the third workgroup fits when four waves of the first two have
  // completed (at 192); under 4 each one waits for the whole previous one
  // (320).
  const auto end = [](std::uint64_t resident) {
    Stream stream(round_robin());
    staggered_groups(stream, 3);
    return stream.run(resident).end;
  };
  EXPECT_DOUBLE_EQ(end(12), 320.0);
  EXPECT_DOUBLE_EQ(end(8), 192.0 + 180.0 + 320.0);
  EXPECT_DOUBLE_EQ(end(4), 3.0 * 320.0 + 2.0 * 180.0);
}

// -- Power-limited clock (kernel_eval5 EVAL_CLOCK=rule) ------------------------

TEST(MeasuredScheduler, MfmaEnergiesAreTheMeasuredShapesOrTheirPerMacRate) {
  // power_apply/apply.py MFMA table, nJ per wave64 instruction.
  const SchedulerTuning tuning;
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_32x32x64_f8f6f4"), 27.6);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_16x16x128_f8f6f4"), 11.9);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_32x32x16_bf16"), 16.3);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_16x16x32_bf16"), 16.3 / 2.0);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_32x32x16_f16"), 16.3);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_16x16x32_f16"), 16.3 / 2.0);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_32x32x2_f32"), 15.4);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_16x16x4_f32"), 15.4 / 2.0);
  // Shapes apply.py does not list: per MAC of the 32x32 shape of the type.
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_scale_f32_16x16x128_f8f6f4"), 11.9);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_i32_32x32x32_i8"), 27.6 / 2.0);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f32_32x32x8bf16_1k"), 16.3 / 2.0);
  EXPECT_DOUBLE_EQ(mfma_energy_nj(tuning, "v_mfma_f64_16x16x4_f64"), 15.4 / 2.0);
}

TEST(MeasuredScheduler, ValuAndLdsEnergiesFollowApplyPy) {
  const SchedulerTuning tuning;
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_fma_f32"), 0.95);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_exp_f32"), 0.95);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_cvt_pk_bf16_f32"), 0.95);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_pk_fma_f32"), 1.70);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_pk_mov_b32"), 0.475);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_add_f64"), 1.90);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_cmp_gt_f32"), 0.475);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_cndmask_b32"), 0.475);
  EXPECT_DOUBLE_EQ(valu_energy_nj(tuning, "v_add_u32"), 0.475);
  // LDS bytes per wave64 instruction.
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_read_b128"), 1024.0);
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_read_b64_tr_b16"), 512.0);
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_write2st64_b64"), 1024.0);
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_read_u16_d16_hi"), 128.0);
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_bpermute_b32"), 256.0);
  EXPECT_DOUBLE_EQ(lds_instruction_bytes("ds_swizzle"), 256.0);
}

TEST(MeasuredScheduler, KernelEnergyScalesTheMixToTheClassTotals) {
  // Recorded: 3 fp8 32x32x64 MFMAs and 1 bf16 32x32x16, 2 v_pk_fma_f32,
  // 1 ds_read_b128. The dispatch executed twice the MFMAs and VALU ops and
  // four times the LDS reads; 1 MB through L2, 0.25 MB of it beyond.
  Stream stream;
  stream.begin(0, 0);
  for (int index = 0; index < 3; ++index)
    stream.add(0, 0, "v_mfma_f32_32x32x64_f8f6f4", InstClass::MatrixMultiply, 0x100 + 8 * index, 8);
  stream.add(0, 0, "v_mfma_f32_32x32x16_bf16", InstClass::MatrixMultiply, 0x200, 8);
  stream.add(0, 0, "v_pk_fma_f32", InstClass::VectorAlu, 0x300, 8);
  stream.add(0, 0, "v_pk_fma_f32", InstClass::VectorAlu, 0x308, 8);
  stream.add(0, 0, "ds_read_b128", InstClass::LdsRead, 0x400, 8);
  KernelEnergy energy;
  energy.add(stream.recorder().rules(), stream.recorder().waves());
  std::array<std::uint64_t, kNumInstClasses> totals{};
  totals[static_cast<std::size_t>(InstClass::MatrixMultiply)] = 8;
  totals[static_cast<std::size_t>(InstClass::VectorAlu)] = 4;
  totals[static_cast<std::size_t>(InstClass::LdsRead)] = 4;
  const double joules = energy.joules(stream.tuning_, totals, 1e6, 0.25e6);
  const double expected = (6 * 27.6 + 2 * 16.3 + 4 * 1.70) * 1e-9 +
                          (4 * 1024 * 6.4 + 0.75e6 * 19.0 + 0.25e6 * 24.5) * 1e-12;
  EXPECT_NEAR(joules, expected, 1e-18);
}

TEST(MeasuredScheduler, AboveThePowerLimitTheClockDropsWithTheSquareRoot) {
  // D = E / (cycles at 2400 MHz). 24000 cycles = 10 us, so 760 W x 10 us =
  // 7.6 mJ is the limit. Four times that: 2400 x sqrt(1/4) = 1200.
  const SchedulerTuning tuning;
  EXPECT_DOUBLE_EQ(tuning.power_limit_watts, 760.0);
  EXPECT_NEAR(power_clock_mhz(tuning, 4.0 * 7.6e-3, 24000.0, 2390.0), 1200.0, 1e-9);
  // At or below the limit the configured clock.
  EXPECT_DOUBLE_EQ(power_clock_mhz(tuning, 7.6e-3, 24000.0, 2390.0), 2390.0);
  // Just above it: min(configured, 2400 sqrt(C / D)).
  EXPECT_DOUBLE_EQ(power_clock_mhz(tuning, 1.001 * 7.6e-3, 24000.0, 2390.0), 2390.0);
  EXPECT_NEAR(power_clock_mhz(tuning, 1.1 * 7.6e-3, 24000.0, 2390.0),
              2400.0 * std::sqrt(1.0 / 1.1), 1e-9);
  SchedulerTuning off = tuning;
  off.power_clock = false;
  EXPECT_DOUBLE_EQ(power_clock_mhz(off, 4.0 * 7.6e-3, 24000.0, 2390.0), 2390.0);
  // The kernel time runs the body at that clock and converts at the configured one.
  KernelShape shape;
  shape.workgroups = 1;
  shape.xcds = 1;
  shape.slowest_unit_cycles = 24000.0;
  shape.clock_mhz = 2400.0;
  const KernelTime full = kernel_time(tuning, shape);
  shape.body_clock_mhz = 1200.0;
  const KernelTime slow = kernel_time(tuning, shape);
  EXPECT_DOUBLE_EQ(slow.body_us, 20.0);
  EXPECT_NEAR(slow.total_us - full.total_us, 10.0, 1e-12);
  EXPECT_EQ(slow.cycles - full.cycles, 24000u);
}

TEST(TimingWaveIdentity, ThePowerLimitSlowsTheScheduledBody) {
  // 1600 v_exp_f32 (0.95 nJ each) and whatever the dispatch moved through
  // L2. A limit far above that power changes nothing; a 1 mW limit slows the
  // body by at least 1000 / (2400 sqrt(0.001 / D)) with D from the ALU
  // energy alone (the fixed terms, 160 cycles here, do not scale).
  const std::string base = R"(, "scheduler.enabled": true, "scheduler.kernel.enabled": false)";
  TimingPlane off{Tuning::parse(plane_config(base + R"(, "scheduler.power.enabled": false)"))};
  TimingPlane high{Tuning::parse(plane_config(base + R"(, "scheduler.power.limit_watts": 1e9)"))};
  TimingPlane low{Tuning::parse(plane_config(base + R"(, "scheduler.power.limit_watts": 0.001)"))};
  const std::uint64_t free_cycles = run_folded(off, false, 200);
  EXPECT_EQ(run_folded(high, false, 200), free_cycles);
  const double body = static_cast<double>(free_cycles - 160);
  const double watts = 1600 * 0.95e-9 / (body / 2400e6);
  const double clock = 2400.0 * std::sqrt(0.001 / watts);
  ASSERT_LT(clock, 1000.0);
  EXPECT_GE(static_cast<double>(run_folded(low, false, 200)), 160.0 + body * 1000.0 / clock - 1.0);
}

// -- Rules of the merged model (istream_replay11) -----------------------------

/// @brief One rule as make_rule derives it.
SchedRule rule_of(const SchedulerTuning &tuning, const std::string &mnemonic, InstClass cls,
                  std::uint8_t size = 4, RegisterRanges writes = {}) {
  return make_rule(tuning, mnemonic, cls, size, 0, {}, writes, Stream::unconstrained(), 0x100);
}

TEST(MeasuredSchedulerMerged, PermutesHoldTheLdsSixCyclesAndTheWave8Point3) {
  // M_PERM. The emulator reports 1 cycle; both waves' permutes (on two SIMDs)
  // share the CU's LDS: the second issues at 6 and holds its wave 8.3.
  const auto end = [](SchedulerTuning tuning) {
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(0, 1);
    stream.add(0, 0, "ds_bpermute_b32", InstClass::LdsRead, 0x100, 8, 1);
    stream.add(0, 1, "ds_bpermute_b32", InstClass::LdsRead, 0x100, 8, 1);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}), 6.0 + 8.3);
  SchedulerTuning off;
  off.lds_permute_cycles = 0.0;
  EXPECT_DOUBLE_EQ(end(off), 1.0 + 4.0);
}

TEST(MeasuredSchedulerMerged, ALoadToLdsAlsoTakesTheLdsOneCycle) {
  // M_LDSDMA. buffer_load ... lds (no VGPR destination) issues at 0 and takes
  // the LDS for a cycle; the other wave's ds_read waits for it.
  const auto end = [](SchedulerTuning tuning, bool to_lds) {
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(0, 1);
    stream.add(0, 0, "buffer_load_dword", InstClass::VectorMemoryRead, 0x100, 8, 4, 0, 0xFF, 0, {},
               to_lds ? RegisterRanges{} : vgpr(0));
    stream.add(0, 1, "ds_read_b32", InstClass::LdsRead, 0x200, 8, 4);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}, true), 1.0 + 4.0);
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}, false), 4.0);
  SchedulerTuning off;
  off.lds_dma_cycles = 0.0;
  EXPECT_DOUBLE_EQ(end(off, true), 4.0);
}

TEST(MeasuredSchedulerMerged, WideLdsReadsHoldTheWave4Point8And8Point2) {
  // M_DS8, M_DS16: two back-to-back reads of one wave.
  const auto end = [](SchedulerTuning tuning, const std::string &mnemonic) {
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.add(0, 0, mnemonic, InstClass::LdsRead, 0x100, 8);
    stream.add(0, 0, mnemonic, InstClass::LdsRead, 0x108, 8);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}, "ds_read_b64"), 2 * 4.8);
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}, "ds_read_b64_tr_b16"), 2 * 4.8);
  EXPECT_DOUBLE_EQ(end(SchedulerTuning{}, "ds_read_b128"), 2 * 8.2);
  SchedulerTuning off;
  off.lds_read_b64_wave_cycles = 0.0;
  off.lds_read_b128_wave_cycles = 0.0;
  EXPECT_DOUBLE_EQ(end(off, "ds_read_b64"), 2 * 4.0);
  EXPECT_DOUBLE_EQ(end(off, "ds_read_b128"), 2 * 8.0);
}

TEST(MeasuredSchedulerMerged, OnlyADoubleRateOpSharesTheValuSlot) {
  // M_PAIR, one SIMD. Wave 0's v_mov (2 cycles) opens a 4-cycle VALU slot at
  // 0. Wave 1's full-rate op waits for the slot (4) rather than for the pipe
  // (2); a second v_mov would co-issue at 2.
  const auto end = [](bool pair, const std::string &second) {
    SchedulerTuning tuning;
    tuning.simds_per_cu = 1;
    tuning.valu_pair = pair;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(0, 1);
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x100);
    stream.add(0, 1, second, InstClass::VectorAlu, 0x200);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(true, "v_cvt_f32_i32_e32"), 4.0 + 4.0);
  EXPECT_DOUBLE_EQ(end(false, "v_cvt_f32_i32_e32"), 2.0 + 4.0);
  EXPECT_DOUBLE_EQ(end(true, "v_mov_b32_e32"), 2.0 + 4.0);
}

TEST(MeasuredSchedulerMemory, AnL1HitTakesItsSiblingsMissLatency) {
  // M_MERGE: wave 1 of workgroup 0 hit L1 (100) where wave 0 missed (500) at
  // the same pc and occurrence: it costs 500. In another workgroup it keeps
  // its own latency.
  const auto mean = [](SchedulerTuning tuning, std::uint32_t other) {
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(other, 1);
    stream.access(0, 0, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 500, 2, kVmcnt);
    stream.access(other, 1, "global_load_dword", InstClass::VectorMemoryRead, 0x100, 100, 0, kVmcnt);
    PcMemoryAverager averager;
    averager.add(stream.tuning_, stream.recorder().rules(), stream.recorder().waves());
    return averager.build(stream.tuning_, -1.0).at(0x100).latency;
  };
  EXPECT_DOUBLE_EQ(mean(SchedulerTuning{}, 0), 500.0);
  EXPECT_DOUBLE_EQ(mean(SchedulerTuning{}, 1), 300.0);
  SchedulerTuning off;
  off.memory_merge_misses = false;
  EXPECT_DOUBLE_EQ(mean(off, 0), 300.0);
}

TEST(MeasuredScheduler, EachSimdHoldsAQuarterOfTheResidentWaves) {
  // M_SIMDSLOT, staggered_groups: under 8 each SIMD holds 2 and slot s of
  // both workgroups sits on SIMD s, so the third workgroup waits until slot 3
  // completes (320), then 180 + its own 320.
  Stream stream;
  staggered_groups(stream, 3);
  EXPECT_DOUBLE_EQ(stream.run(8).end, 320.0 + 180.0 + 320.0);
  Stream twelve;
  staggered_groups(twelve, 3);
  EXPECT_DOUBLE_EQ(twelve.run(12).end, 320.0);
}

TEST(MeasuredSchedulerMerged, ContendedCnd32SlowsOrdinaryValu) {
  // M_CND. Waves on SIMDs 0 and 1 issue v_cndmask_b32_e32; wave 1 waits for
  // the cnd32 resource (4.4) that SIMD 0 took, which saturates it until
  // 4.4 + 8.8. Wave 0's v_mov at 5.5 then holds it 16: its next op is at
  // 21.5 and ends at 25.5.
  const auto end = [](bool saturation) {
    SchedulerTuning tuning;
    tuning.cndmask32_saturation = saturation;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.begin(0, 1);
    stream.add(0, 0, "v_cndmask_b32_e32", InstClass::VectorAlu, 0x100);
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x104);
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x108);
    stream.add(0, 1, "v_cndmask_b32_e32", InstClass::VectorAlu, 0x100);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(true), 5.5 + 16.0 + 4.0);
  EXPECT_DOUBLE_EQ(end(false), 5.5 + 4.0 + 4.0);
  // The SDWA and DPP forms are VOP2 and use the resource too.
  EXPECT_TRUE(rule_of(SchedulerTuning{}, "v_cndmask_b32_sdwa", InstClass::VectorAlu, 8).cnd32);
  EXPECT_TRUE(rule_of(SchedulerTuning{}, "v_cndmask_b32_dpp", InstClass::VectorAlu, 8).cnd32);
  EXPECT_FALSE(rule_of(SchedulerTuning{}, "v_cndmask_b32_e64", InstClass::VectorAlu, 8).cnd32);
  SchedulerTuning off;
  off.cndmask32_saturation = false;
  EXPECT_FALSE(rule_of(off, "v_cndmask_b32_sdwa", InstClass::VectorAlu, 8).cnd32);
}

TEST(MeasuredSchedulerMerged, EightByteValuCosts4Point6OnlyAmongFewShortEncodings) {
  // M_ENC: a 4-byte v_mov, then three 8-byte ones. The first 8-byte op sees
  // 1 of 2 short encodings (>= 50%): 4.0; the next two 1/3 and 1/4: 4.6.
  const auto end = [](bool context) {
    SchedulerTuning tuning;
    tuning.encoding_context = context;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x100, 4);
    stream.add(0, 0, "v_mov_b32_e64", InstClass::VectorAlu, 0x104, 8);
    stream.add(0, 0, "v_mov_b32_e64", InstClass::VectorAlu, 0x10c, 8);
    stream.add(0, 0, "v_mov_b32_e64", InstClass::VectorAlu, 0x114, 8);
    return stream.run().end;
  };
  EXPECT_NEAR(end(true), 4.0 + 4.0 + 4.6 + 4.6, 1e-12);
  EXPECT_NEAR(end(false), 4.0 + 3 * 4.6, 1e-12);
}

TEST(MeasuredSchedulerMerged, ConversionsUseTheMeasuredOccupancyTable) {
  // M_OCC.
  const SchedulerTuning tuning;
  EXPECT_DOUBLE_EQ(rule_of(tuning, "v_cvt_scalef32_pk_fp8_f32", InstClass::VectorAlu, 8).occupancy, 8.0);
  EXPECT_DOUBLE_EQ(rule_of(tuning, "v_cvt_sr_bf16_f32", InstClass::VectorAlu, 8).occupancy, 8.0);
  EXPECT_DOUBLE_EQ(rule_of(tuning, "v_cvt_pk_f32_fp8_e32", InstClass::VectorAlu).occupancy, 4.0);
  EXPECT_DOUBLE_EQ(rule_of(tuning, "v_permlane16_swap_b32_e32", InstClass::VectorAlu).occupancy, 8.0);
  SchedulerTuning off;
  off.valu_conversion_table = false;
  EXPECT_DOUBLE_EQ(rule_of(off, "v_cvt_scalef32_pk_fp8_f32", InstClass::VectorAlu, 8).occupancy, 4.0);
}

TEST(MeasuredSchedulerMerged, F64MfmaOccupancyAndValuHold) {
  // M_F64: 16x16x4 f64 occupies 64 and holds the VALU 64 + 8; a VALU op
  // behind it on the same SIMD issues at 72.
  const auto end = [](bool rule) {
    SchedulerTuning tuning;
    tuning.mfma_f64_rule = rule;
    Stream stream(tuning);
    stream.begin(0, 0);
    stream.add(0, 0, "v_mfma_f64_16x16x4_f64", InstClass::MatrixMultiply, 0x100, 8, 16);
    stream.add(0, 0, "v_mov_b32_e32", InstClass::VectorAlu, 0x108);
    return stream.run().end;
  };
  EXPECT_DOUBLE_EQ(end(true), 72.0 + 4.0);
  EXPECT_DOUBLE_EQ(end(false), 8.0 + 4.0);
  const SchedRule small = rule_of(SchedulerTuning{}, "v_mfma_f64_4x4x4_4b_f64", InstClass::MatrixMultiply, 8);
  EXPECT_FALSE(small.occupancy_dynamic);
  EXPECT_DOUBLE_EQ(small.occupancy, 16.0);
}

TEST(MeasuredScheduler, QueuedKernelTimeHasItsOwnHeadTailAndMinimum) {
  // docs section 5.2, the default regime: head + skew 0.149, no stagger,
  // tail 1.25, minimum 3.9 us.
  KernelShape shape;
  shape.workgroups = 512;
  shape.waves_per_workgroup = 4;
  shape.resident_waves = 8;
  shape.compute_units = 256;
  shape.xcds = 8;
  shape.slowest_unit_cycles = 10000.0;
  shape.dirty_bytes = 4e6;
  shape.clock_mhz = 2000.0;
  const KernelTime time = kernel_time(SchedulerTuning{}, shape);
  EXPECT_DOUBLE_EQ(time.head_us, 0.149);
  EXPECT_DOUBLE_EQ(time.stagger_us, 0.0);
  EXPECT_DOUBLE_EQ(time.tail_us, 1.25);
  EXPECT_NEAR(time.total_us, 0.149 + 0.17216 + 5.0 + 1.25 + 0.46, 1e-9);
  EXPECT_EQ(time.cycles, 14062u);
  KernelShape one;
  one.compute_units = 256;
  one.xcds = 8;
  one.slowest_unit_cycles = 100.0;
  one.clock_mhz = 2000.0;
  EXPECT_DOUBLE_EQ(kernel_time(SchedulerTuning{}, one).total_us, 3.9);
}

} // namespace
} // namespace rocjitsu::timing
