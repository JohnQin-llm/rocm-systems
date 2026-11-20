// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file timing_model_test.cpp
/// @brief Behavioural tests for the timing plane.
///
/// @details Every assertion here is about a *relation* that has to hold
/// whatever the tuning says, never about a number the tuning supplies. That is
/// deliberate and it is the only way these can live in CI: the calibrated
/// values -- issue occupancies, cache latencies, launch costs -- are shipped
/// separately from this repository, and a test that pinned one would either
/// fail everywhere they are absent or leak one where they are not.
///
/// So the tests here ask questions like: does more work cost more? Does the
/// same input give the same answer? Does an unnamed parameter make the run
/// slower rather than faster? Is a divergent access more expensive than a
/// coalesced one? Those hold for any tuning that is internally consistent, and
/// they are what actually breaks when the model is changed carelessly.

#include "rocjitsu/vm/timing/cache_des.h"
#include "rocjitsu/vm/timing/classify.h"
#include "rocjitsu/vm/timing/coalesce.h"
#include "rocjitsu/vm/timing/cu_des.h"
#include "rocjitsu/vm/timing/engine.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/tuning.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

namespace rocjitsu::timing {
namespace {

/// @brief A small, entirely made-up machine.
///
/// @details Not a model of any part. Four compute units and round numbers, so
/// that a test runs in milliseconds and so that nothing here can be mistaken
/// for a measurement of hardware.
std::string tiny_config(const std::string &extra = {}) {
  return R"({
    "timing": {
      "enabled": true,
      "clock_mhz": 1000,
      "machine": {
        "compute_units": 4,
        "xcds": 1,
        "simd_lanes": 16,
        "wave_slots_per_cu": 8,
        "vector_registers_per_cu": 512,
        "scalar_registers_per_cu": 800,
        "lds_bytes_per_cu": 65536,
        "memory_channels": 2,
        "vector_alu.issue_cycles": 4,
        "vector_alu.ports": 4,
        "scalar_alu.issue_cycles": 1,
        "vector_memory_read.issue_cycles": 4,
        "vector_memory_write.issue_cycles": 4,
        "vector_memory.ports": 1,
        "l1_vector.hit_cycles": 10,
        "l1_vector.line_bytes": 64,
        "l1_vector.lines_per_cycle": 1,
        "l1_vector.sets": 4,
        "l1_vector.ways": 4,
        "l2.hit_cycles": 40,
        "l2.line_bytes": 64,
        "l2.lines_per_cycle": 4,
        "l2.sets": 16,
        "l2.ways": 4,
        "mall.hit_cycles": 80,
        "mall.line_bytes": 64,
        "mall.bytes_per_cycle": 128,
        "mall.sets": 64,
        "mall.ways": 4,
        "dram.latency_cycles": 200,
        "dram.bytes_per_cycle": 64,
        "fabric.bytes_per_cycle": 128,
        "fabric_request_bytes": 64,
        "dispatch.start_cycles": 100,
        "dispatch.end_cycles": 50,
        "dispatch.invalidate_cycles": 10,
        "dispatch.workgroups_per_cycle": 1,
        "front_end.issue_cycles": 1,
        "miss_status_registers_per_cu": 16,
        "wavefront_issue_cycles": 4)" +
         extra + R"(
      }
    }
  })";
}

DispatchShape shape_for(std::uint32_t workgroups, std::uint32_t waves_per_workgroup) {
  DispatchShape shape;
  shape.dispatch_id = 1;
  shape.queue_id = 0;
  shape.kernel_name = "test_kernel";
  shape.workgroup_count = workgroups;
  shape.waves_per_workgroup = waves_per_workgroup;
  shape.vector_registers_per_wave = 32;
  shape.scalar_registers_per_wave = 16;
  shape.lds_bytes_per_workgroup = 0;
  shape.wave_size = 64;
  return shape;
}

RetiredInstruction alu(std::uint64_t pc) {
  RetiredInstruction retired;
  retired.pc = pc;
  retired.inst_class = InstClass::VectorAlu;
  retired.active_lanes = 64;
  retired.wave_lanes = 64;
  retired.wait.fill(kUnconstrained);
  return retired;
}

RetiredInstruction load(std::uint64_t pc, const std::vector<std::uint64_t> &addresses) {
  RetiredInstruction retired;
  retired.pc = pc;
  retired.inst_class = InstClass::VectorMemoryRead;
  retired.active_lanes = static_cast<std::uint32_t>(addresses.size());
  retired.wave_lanes = 64;
  retired.wait.fill(kUnconstrained);
  retired.lane_addresses = &addresses;
  retired.bytes_per_lane = 4;
  retired.space = MemorySpace::Global;
  retired.is_load = true;
  return retired;
}

/// @brief Run one dispatch and report what it cost.
///
/// @param instructions Instructions each wavefront retires.
/// @param addresses When non-empty, every wavefront also performs one vector
///        load over these lane addresses.
std::uint64_t run_dispatch(TimingPlane &plane, const DispatchShape &shape,
                           std::uint32_t instructions,
                           const std::vector<std::uint64_t> &addresses = {}) {
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  const std::uint32_t waves =
      shape.workgroup_count * std::max<std::uint32_t>(1, shape.waves_per_workgroup);
  for (std::uint32_t wave = 0; wave < waves; ++wave) {
    const std::uint32_t unit = wave % 4;
    const std::uint32_t slot = wave / 4;
    plane.wave_begin(unit, slot, shape.dispatch_id, shape.queue_id);
    for (std::uint32_t index = 0; index < instructions; ++index)
      plane.instruction(unit, slot, alu(0x1000 + 4 * index));
    if (!addresses.empty())
      plane.instruction(unit, slot, load(0x2000, addresses));
    plane.wave_end(unit, slot);
  }
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  return plane.current_cycles() - before;
}

std::vector<std::uint64_t> contiguous_lanes(std::uint64_t base) {
  std::vector<std::uint64_t> addresses;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    addresses.push_back(base + lane * 4);
  return addresses;
}

std::vector<std::uint64_t> divergent_lanes(std::uint64_t base) {
  std::vector<std::uint64_t> addresses;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    addresses.push_back(base + lane * 4096);
  return addresses;
}

// -- Tuning ------------------------------------------------------------------

TEST(TimingTuning, ReadsWhatTheConfigNames) {
  const Tuning tuning = Tuning::parse(tiny_config());
  EXPECT_TRUE(tuning.enabled);
  EXPECT_EQ(tuning.compute_units, 4u);
  EXPECT_EQ(tuning.l1_vector.line_bytes, 64u);
  EXPECT_FALSE(tuning.resolved.empty());
}

TEST(TimingTuning, DisabledWithoutATimingBlock) {
  const Tuning tuning = Tuning::parse(R"({"soc": {"name": "whatever"}})");
  EXPECT_FALSE(tuning.enabled);
}

TEST(TimingTuning, RecordsEveryParameterItHadToGuess) {
  // The ledger is the whole fail-slow contract: a run whose config is missing
  // parameters has to be able to say which, or an incomplete config is
  // indistinguishable from a complete one.
  const Tuning tuning = Tuning::parse(tiny_config());
  EXPECT_FALSE(tuning.fell_back.empty());
  for (const std::string &name : tuning.fell_back)
    EXPECT_FALSE(name.empty());
}

TEST(TimingTuning, UnnamedInstructionClassCostsAsMuchAsTheDearestNamedOne) {
  // Fail slow. An opcode nobody classified has to make the run read slow and
  // look suspicious, never fast and look accurate.
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "matrix_multiply.issue_cycles": 32)"));
  const std::uint64_t unknown = tuning.issue_cycles[static_cast<std::size_t>(InstClass::Unknown)];
  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    EXPECT_GE(unknown, tuning.issue_cycles[index]) << "class " << index;
}

TEST(TimingTuning, PerClassFrontEndDefaultsToTheSharedFigure) {
  // A config from before the front end was split by class has to behave
  // exactly as it did.
  const Tuning tuning = Tuning::parse(tiny_config());
  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    EXPECT_EQ(tuning.front_end_sixteenths[index], tuning.front_end_cycles * kFrontEndScale)
        << inst_class_name(static_cast<InstClass>(index));
}

TEST(TimingTuning, PerClassFrontEndOverridesTheSharedFigure) {
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "front_end.matrix_multiply.issue_cycles": 2.5)"));
  EXPECT_EQ(tuning.front_end_sixteenths[static_cast<std::size_t>(InstClass::MatrixMultiply)],
            static_cast<std::uint64_t>(2.5 * kFrontEndScale));
  EXPECT_EQ(tuning.front_end_sixteenths[static_cast<std::size_t>(InstClass::VectorAlu)],
            tuning.front_end_cycles * kFrontEndScale);
}

TEST(TimingTuning, PortCountsMayBeFractional) {
  // A unit's sustained rate over a mixed instruction stream is not an integer
  // even though every pipeline stage is.
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "local_data_share.ports": 2.5)"));
  EXPECT_DOUBLE_EQ(tuning.ports[static_cast<std::size_t>(FunctionalUnit::LocalDataShare)], 2.5);
}

// -- Coalescing --------------------------------------------------------------

TEST(TimingCoalesce, ContiguousCostsFewerLinesThanDivergent) {
  std::vector<std::uint64_t> pool;
  const std::uint32_t together = coalesce_lines(contiguous_lanes(0x10000), 6, 4, pool);
  pool.clear();
  const std::uint32_t apart = coalesce_lines(divergent_lanes(0x10000), 6, 4, pool);
  EXPECT_LT(together, apart);
  EXPECT_EQ(apart, 64u) << "one line per lane is the worst case for a wave64";
}

TEST(TimingCoalesce, AnAccessStraddlingALineCountsBoth) {
  // Off by one byte from the line, so the last lane's four bytes run over.
  std::vector<std::uint64_t> pool;
  const std::uint32_t aligned = coalesce_lines({{0x1000}}, 6, 64, pool);
  pool.clear();
  const std::uint32_t straddling = coalesce_lines({{0x1001}}, 6, 64, pool);
  EXPECT_EQ(aligned, 1u);
  EXPECT_EQ(straddling, 2u);
}

TEST(TimingCoalesce, RecoalescingToACoarserLineNeverGrows) {
  std::vector<std::uint64_t> pool;
  const std::uint32_t fine = coalesce_lines(contiguous_lanes(0x20000), 6, 4, pool);
  const std::uint32_t coarse = recoalesce_lines(pool.data(), fine, 7);
  EXPECT_LE(coarse, fine);
  EXPECT_GE(coarse, 1u);
}

TEST(TimingLds, BroadcastIsFreeAndConflictsSerialise) {
  std::vector<std::uint64_t> same(64, 0x400);
  const std::uint64_t broadcast = lds_conflict_cycles(same, 4, 64, 4, 64);
  std::vector<std::uint64_t> strided;
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    strided.push_back(lane * 64 * 4); // every lane on bank zero, all distinct
  const std::uint64_t conflicting = lds_conflict_cycles(strided, 4, 64, 4, 64);
  EXPECT_EQ(broadcast, 1u);
  EXPECT_GT(conflicting, broadcast);
  EXPECT_LE(conflicting, 64u) << "a wave64 cannot conflict worse than one lane per pass";
}

TEST(TimingLds, EveryAccessCostsAtLeastOneCycle) {
  EXPECT_GE(lds_conflict_cycles({{0x100}}, 4, 64, 4, 64), 1u);
}

// -- Tag array ---------------------------------------------------------------

TEST(TimingCache, RepeatedAccessHitsAndCapacityEvicts) {
  TagArray tags;
  tags.configure(2, 2, 64);
  EXPECT_FALSE(tags.access(0x0000, 1)) << "cold";
  EXPECT_TRUE(tags.access(0x0000, 2)) << "warm";
  // Far more distinct lines than the array holds: the first is gone, whatever
  // set the index function puts each line in.
  for (std::uint64_t line = 1; line <= 16; ++line)
    tags.access(line * 64, 2 + line);
  EXPECT_FALSE(tags.access(0x0000, 100));
}

TEST(TimingCache, InvalidateLosesEverything) {
  TagArray tags;
  tags.configure(4, 4, 64);
  tags.access(0x0000, 1);
  ASSERT_TRUE(tags.access(0x0000, 2));
  tags.invalidate();
  EXPECT_FALSE(tags.access(0x0000, 3));
}

TEST(TimingCache, ADirtyLineIsWrittenBackWhenEvictedAndACleanOneIsNot) {
  TagArray tags;
  tags.configure(1, 2, 64);
  std::uint64_t victim = 0;
  tags.access(0x000, 1, /*write=*/true, /*cold=*/false, &victim);
  EXPECT_EQ(victim, TagArray::kNoLine) << "nothing was resident to displace";
  tags.access(0x040, 2, /*write=*/false, /*cold=*/false, &victim);
  EXPECT_EQ(victim, TagArray::kNoLine);
  tags.access(0x080, 3, /*write=*/false, /*cold=*/false, &victim);
  EXPECT_EQ(victim, 0x000u) << "the stored line leaves dirty";
  tags.access(0x0C0, 4, /*write=*/false, /*cold=*/false, &victim);
  EXPECT_EQ(victim, TagArray::kNoLine) << "a line that was only read leaves silently";
}

TEST(TimingCache, AFlushWritesBackEachDirtyLineOnceAndKeepsItResident) {
  TagArray tags;
  tags.configure(4, 4, 64);
  tags.access(0x000, 1, /*write=*/true);
  tags.access(0x040, 2, /*write=*/true);
  tags.access(0x080, 3, /*write=*/false);
  EXPECT_EQ(tags.flush(), 2u);
  EXPECT_EQ(tags.flush(), 0u) << "a flushed line is clean";
  EXPECT_TRUE(tags.access(0x000, 4)) << "and still resident";
}

TEST(TimingCache, AColdLineIsTheNextToGo) {
  // A streaming store's line is allocated as least recently used: it is the
  // one displaced next, written back early, and the temporal line it shares a
  // set with survives.
  TagArray tags;
  tags.configure(1, 2, 64);
  std::uint64_t victim = 0;
  tags.access(0x000, 1);
  tags.access(0x040, 2, /*write=*/true, /*cold=*/true);
  tags.access(0x080, 3, /*write=*/false, /*cold=*/false, &victim);
  EXPECT_EQ(victim, 0x040u);
  EXPECT_TRUE(tags.access(0x000, 4));
}

// -- MI355X vector L1 set index ------------------------------------------------

/// @brief The set each 128-byte line of a 128 KiB region was measured in on
///        MI355X (bench/ubench/l1g_sets.json; sets A, B, C, rest as 0..3).
constexpr const char *kMeasuredMi355xL1Sets =
    "0210312020301302312102131301203120301302021031201301203131210210"
    "3121021313012031021031202030130213012031312102132030130202103120"
    "2030130202103120130120313121021302103120203013023121021313012030"
    "1301203131210213203013020210312031210213130120310210312020301302"
    "1301203131210213203013020210312031210213130120310210312020301301"
    "2030130202103120130120313121021302103120203013023121021313012031"
    "3121021313012031021031202030130213012031312102132030130202103121"
    "0210312020301302312102131301203120301302021031201301203131210213"
    "2030130202103120130120313121021302103120203013023121021313012030"
    "1301203131210213203013020210312031210213130120310210312020301302"
    "0210312020301302312102131301203120301302021031201301203131210210"
    "3121021313012031021031202030130213012031312102132030130202103120"
    "3121021313012031021031202030130213012031312102132030130202103121"
    "0210312020301302312102131301203120301302021031201301203131210213"
    "1301203131210213203013020210312031210213130120310210312020301301"
    "2030130202103120130120313121021302103120203013023121021313012030";

TagArray mi355x_l1() {
  TagArray tags;
  tags.configure(4, 64, 128, SetHash::Mi355x);
  return tags;
}

/// @brief Lines that miss in one steady-state pass over @p lines consecutive
///        128-byte lines, after two warming passes.
std::uint64_t steady_misses(TagArray &tags, std::uint64_t base, std::uint64_t lines,
                            std::uint64_t stride = 128) {
  std::uint64_t stamp = 1;
  for (int pass = 0; pass < 2; ++pass)
    for (std::uint64_t line = 0; line < lines; ++line)
      tags.access(base + line * stride, stamp++);
  std::uint64_t misses = 0;
  for (std::uint64_t line = 0; line < lines; ++line)
    misses += tags.access(base + line * stride, stamp++) ? 0 : 1;
  return misses;
}

TEST(TimingCacheMi355x, SetIndexReproducesTheMeasuredTable) {
  const TagArray tags = mi355x_l1();
  for (std::uint64_t line = 0; line < 1024; ++line)
    EXPECT_EQ(tags.set_of(line * 128), static_cast<std::uint64_t>(kMeasuredMi355xL1Sets[line] - '0'))
        << "line " << line;
}

TEST(TimingCacheMi355x, SetIndexSpotChecks) {
  const TagArray tags = mi355x_l1();
  EXPECT_EQ(tags.set_of(0x00000), 0u);
  EXPECT_EQ(tags.set_of(0x0007f), 0u) << "the byte offset in the line does not matter";
  EXPECT_EQ(tags.set_of(0x00080), 2u) << "byte address bit 7 is in the hash";
  EXPECT_EQ(tags.set_of(0x00100), 1u);
  EXPECT_EQ(tags.set_of(0x00200), 3u);
  EXPECT_EQ(tags.set_of(0x01f80), 0u) << "line 63: one of the nine exceptions to the fold";
  EXPECT_EQ(tags.set_of(0x05f80), 0u) << "line 191: another";
  EXPECT_EQ(tags.set_of(0x03f80), 0u) << "line 127 follows the fold";
}

TEST(TimingCacheMi355x, ThePatternRepeatsSixteenMegabytesHigher) {
  const TagArray tags = mi355x_l1();
  for (std::uint64_t line = 0; line < 256; ++line)
    EXPECT_EQ(tags.set_of((16ULL << 20) + line * 128), tags.set_of(line * 128)) << "line " << line;
}

TEST(TimingCacheMi355x, HighAddressBitsEnterAsMeasured) {
  // bench/ubench/l1g_run8.jsonl: line bit 10 moves v by 01000, bit 11 by 11000,
  // bit 12 by 10000, bit 13 by 01000, bits 14 to 18 not at all; combinations
  // of bits add as XOR.
  const TagArray tags = mi355x_l1();
  EXPECT_EQ(tags.set_of(128ULL << 10), 1u);
  EXPECT_EQ(tags.set_of(128ULL << 11), 3u);
  EXPECT_EQ(tags.set_of(128ULL << 12), 2u);
  EXPECT_EQ(tags.set_of(128ULL << 13), 1u);
  EXPECT_EQ(tags.set_of((128ULL << 10) | (128ULL << 11)), 2u) << "01000 ^ 11000 = 10000";
  for (std::uint64_t line = 0; line < 1024; ++line)
    for (unsigned high = 14; high <= 30; ++high)
      ASSERT_EQ(tags.set_of((line | (1ULL << high)) * 128), tags.set_of(line * 128))
          << "line " << line << " bit " << high;
}

TEST(TimingCacheMi355x, ThirtyTwoConsecutiveLinesFillTheSetsNineNineSevenSeven) {
  const TagArray tags = mi355x_l1();
  std::array<int, 4> count{};
  for (std::uint64_t line = 0; line < 32; ++line)
    ++count[tags.set_of(line * 128)];
  std::sort(count.begin(), count.end());
  EXPECT_EQ(count, (std::array<int, 4>{7, 7, 9, 9}));
}

TEST(TimingCacheMi355x, AContiguousFootprintOverflowsTheFullestSetBeforeTheCapacity) {
  // Hardware (bench/ubench/l1g_run2.jsonl, per-access traces of a pointer chase
  // over consecutive lines): 224 lines leave 65 lines missing every pass, 240
  // leave 136, 256 leave 145. 223 lines fit.
  TagArray tags = mi355x_l1();
  EXPECT_EQ(steady_misses(tags, 0, 223), 0u);
  tags.invalidate();
  EXPECT_EQ(steady_misses(tags, 0, 224), 65u) << "one set holds 65 lines and thrashes";
  tags.invalidate();
  EXPECT_EQ(steady_misses(tags, 0, 240), 136u) << "two sets thrash";
  tags.invalidate();
  EXPECT_EQ(steady_misses(tags, 0, 256), 145u);
}

TEST(TimingCacheMi355x, TheFoldIndexHoldsTheSameFootprintsEvenly) {
  // What the default index does with the same footprint: every set gets 60 of
  // 240 lines and nothing misses, which is not what the hardware does.
  TagArray tags;
  tags.configure(4, 64, 128);
  EXPECT_EQ(steady_misses(tags, 0, 240), 0u);
}

TEST(TimingCacheMi355x, PowerOfTwoStridesSpreadOverAllFourSets) {
  // Hardware: 256 lines at any power-of-two stride from 256 B to 64 KiB all hit.
  for (std::uint64_t stride = 256; stride <= 65536; stride *= 2) {
    TagArray tags = mi355x_l1();
    EXPECT_EQ(steady_misses(tags, 0, 256, stride), 0u) << "stride " << stride;
  }
}

TEST(TimingTuning, SetHashDefaultsToFoldAndReadsMi355x) {
  EXPECT_EQ(Tuning::parse(tiny_config()).l1_vector.set_hash, SetHash::Fold);
  const Tuning tuning = Tuning::parse(tiny_config(R"(,
        "l1_vector.set_hash": "mi355x")"));
  EXPECT_EQ(tuning.l1_vector.set_hash, SetHash::Mi355x);
  EXPECT_EQ(tuning.l2.set_hash, SetHash::Fold);
  EXPECT_TRUE(std::none_of(tuning.fell_back.begin(), tuning.fell_back.end(),
                           [](const std::string &key) { return key.find("set_hash") != key.npos; }))
      << "an absent set_hash is a default, not a guess";
}

TEST(TimingCacheMi355x, OtherGeometriesKeepTheFoldIndex) {
  TagArray hashed;
  hashed.configure(8, 4, 64, SetHash::Mi355x);
  TagArray folded;
  folded.configure(8, 4, 64);
  for (std::uint64_t line = 0; line < 256; ++line)
    EXPECT_EQ(hashed.set_of(line * 64), folded.set_of(line * 64));
}

// -- Composition -------------------------------------------------------------

TEST(TimingDispatch, SameInputGivesTheSameAnswer) {
  TimingPlane first{Tuning::parse(tiny_config())};
  TimingPlane second{Tuning::parse(tiny_config())};
  const DispatchShape shape = shape_for(4, 1);
  EXPECT_EQ(run_dispatch(first, shape, 64), run_dispatch(second, shape, 64));
}

TEST(TimingDispatch, TimeOnlyMovesForward) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  std::uint64_t previous = plane.current_cycles();
  for (std::uint32_t round = 0; round < 4; ++round) {
    DispatchShape shape = shape_for(2, 1);
    shape.dispatch_id = round + 1;
    run_dispatch(plane, shape, 16);
    EXPECT_GE(plane.current_cycles(), previous);
    previous = plane.current_cycles();
  }
}

TEST(TimingDispatch, ADispatchThatRanNothingStillCostsItsLaunch) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  const DispatchShape shape = shape_for(1, 1);
  plane.dispatch_begin(shape);
  const std::uint64_t before = plane.current_cycles();
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
  EXPECT_GT(plane.current_cycles() - before, 0u);
}

TEST(TimingDispatch, MoreInstructionsNeverCostLess) {
  std::uint64_t previous = 0;
  for (std::uint32_t instructions : {8u, 64u, 512u, 4096u}) {
    // A fresh machine each time. Sharing one would compare a cold run against
    // a warm one and measure the caches rather than the work.
    TimingPlane plane{Tuning::parse(tiny_config())};
    DispatchShape shape = shape_for(4, 1);
    shape.dispatch_id = instructions;
    const std::uint64_t cost = run_dispatch(plane, shape, instructions);
    EXPECT_GE(cost, previous) << "at " << instructions << " instructions";
    previous = cost;
  }
}

TEST(TimingDispatch, MoreWorkgroupsNeverCostLess) {
  std::uint64_t previous = 0;
  for (std::uint32_t workgroups : {1u, 2u, 8u, 32u}) {
    TimingPlane plane{Tuning::parse(tiny_config())};
    DispatchShape shape = shape_for(workgroups, 1);
    shape.dispatch_id = 100 + workgroups;
    const std::uint64_t cost = run_dispatch(plane, shape, 32);
    EXPECT_GE(cost, previous) << "at " << workgroups << " workgroups";
    previous = cost;
  }
}

TEST(TimingDispatch, DivergentAccessCostsMoreThanCoalesced) {
  // The single relation the whole memory side exists to produce, and the one
  // that no static analysis of the source could recover.
  // A fresh plane each, so neither run finds the other's code or lines warm.
  TimingPlane coalesced{Tuning::parse(tiny_config())};
  TimingPlane divergent{Tuning::parse(tiny_config())};
  const std::uint64_t together =
      run_dispatch(coalesced, shape_for(4, 1), 8, contiguous_lanes(0x100000));
  const std::uint64_t apart = run_dispatch(divergent, shape_for(4, 1), 8, divergent_lanes(0x200000));
  EXPECT_GT(apart, together);
}

TEST(TimingDispatch, ReReadingWhatIsAlreadyResidentIsCheaper) {
  TimingPlane plane{Tuning::parse(tiny_config())};
  const std::vector<std::uint64_t> lanes = divergent_lanes(0x300000);
  DispatchShape shape = shape_for(1, 1);
  const std::uint64_t cold = run_dispatch(plane, shape, 4, lanes);
  shape.dispatch_id = 2;
  const std::uint64_t warm = run_dispatch(plane, shape, 4, lanes);
  EXPECT_LE(warm, cold);
}

TEST(TimingDispatch, ASlowerFrontEndNeverMakesADispatchFaster) {
  // Relations between tunings, not values: whatever the shipped numbers are,
  // charging the front end more cannot produce a shorter dispatch.
  const DispatchShape shape = shape_for(4, 2);
  TimingPlane quick{Tuning::parse(tiny_config())};
  TimingPlane slow{Tuning::parse(tiny_config(R"(,
        "front_end.vector_alu.issue_cycles": 4)"))};
  EXPECT_GE(run_dispatch(slow, shape, 256), run_dispatch(quick, shape, 256));
}

TEST(TimingDispatch, TheStragglerTermGrowsWithTheWorkgroupCountAndIsOffByDefault) {
  const std::string with = R"(,
        "straggler_cycles": 64)";
  TimingPlane without{Tuning::parse(tiny_config())};
  TimingPlane charged{Tuning::parse(tiny_config(with))};
  const DispatchShape small = shape_for(1, 1);
  DispatchShape large = shape_for(32, 4);
  large.dispatch_id = 2;

  const std::uint64_t plain_small = run_dispatch(without, small, 4);
  const std::uint64_t plain_large = run_dispatch(without, large, 4);
  const std::uint64_t straggling_small = run_dispatch(charged, small, 4);
  const std::uint64_t straggling_large = run_dispatch(charged, large, 4);

  // One workgroup has no straggler -- there is nothing for it to straggle
  // against -- and many workgroups do, and it grows with how many.
  EXPECT_EQ(plain_small, straggling_small);
  EXPECT_GT(straggling_large, plain_large);
  EXPECT_GT(straggling_large - plain_large, straggling_small - plain_small);

  // And it is the workgroup count that matters, not the wavefront count: one
  // workgroup of many wavefronts still has nothing to straggle against,
  // because they share a compute unit and a barrier.
  TimingPlane wide{Tuning::parse(tiny_config())};
  TimingPlane wide_charged{Tuning::parse(tiny_config(with))};
  DispatchShape one_group = shape_for(1, 8);
  one_group.dispatch_id = 3;
  EXPECT_EQ(run_dispatch(wide, one_group, 4), run_dispatch(wide_charged, one_group, 4));
}

TEST(TimingDispatch, ExposingLessOfAStallNeverCostsMore) {
  const DispatchShape shape = shape_for(2, 1);
  TimingPlane full{Tuning::parse(tiny_config())};
  TimingPlane halved{Tuning::parse(tiny_config(R"(,
        "stall_exposed_fraction": 0.5)"))};
  const std::vector<std::uint64_t> lanes = divergent_lanes(0x400000);
  EXPECT_LE(run_dispatch(halved, shape, 8, lanes), run_dispatch(full, shape, 8, lanes));
}

// -- Engine ------------------------------------------------------------------

TEST(TimingEngine, RunsToIdleAndStaysThere) {
  Tuning tuning = Tuning::parse(tiny_config());
  TimingPlane plane{std::move(tuning)};
  const DispatchShape shape = shape_for(2, 1);
  run_dispatch(plane, shape, 32);
  const std::uint64_t settled = plane.current_cycles();
  EXPECT_EQ(plane.current_cycles(), settled);
}

TEST(TimingMatrix, NarrowInputsSelectTheNarrowRate) {
  EXPECT_EQ(matrix_type_of("v_mfma_f32_32x32x16_bf16"), MatrixType::BF16);
  EXPECT_EQ(matrix_type_of("v_mfma_f32_32x32x16_fp8_fp8"), MatrixType::Narrow);
  EXPECT_EQ(matrix_type_of("v_mfma_f32_32x32x64_f8f6f4"), MatrixType::Narrow);
  EXPECT_EQ(matrix_type_of("v_mfma_scale_f32_16x16x128_f8f6f4"), MatrixType::Narrow);
  // Wave-size and target suffixes come after the type and do not change it.
  EXPECT_EQ(matrix_type_of("v_mfma_f32_32x32x64_f8f6f4_w64"), MatrixType::Narrow);
  EXPECT_EQ(matrix_type_of("v_mfma_f32_16x16x32_fp8_fp8_w64_gfx950"), MatrixType::Narrow);
  EXPECT_EQ(matrix_type_of("v_mfma_f32_16x16x32_bf16_w64"), MatrixType::BF16);
  EXPECT_EQ(matrix_macs("v_mfma_f32_32x32x64_f8f6f4"), 32u * 32u * 64u);
}

TEST(TimingLds, ATwoAddressAccessPaysForBothHalves) {
  // ds_write2st64_b64: each lane writes at its address and 64 rows further on.
  // The second half lands on the same banks as the first, so it is a second
  // pass, not a free one.
  const auto cost = [](bool second_half) {
    TimingPlane plane{Tuning::parse(tiny_config(R"(,
        "lds.banks": 64, "lds.bank_bytes": 4, "lds.lanes_per_phase": 64,
        "lds.latency_cycles": 20, "lds_write.issue_cycles": 1, "local_data_share.ports": 1)"))};
    const DispatchShape shape = shape_for(1, 1);
    std::vector<std::uint64_t> first, second;
    for (std::uint64_t lane = 0; lane < 64; ++lane) {
      first.push_back(lane * 8);
      second.push_back(lane * 8 + 64 * 64 * 8);
    }
    RetiredInstruction write;
    write.pc = 0x3000;
    write.inst_class = InstClass::LdsWrite;
    write.active_lanes = 64;
    write.wave_lanes = 64;
    write.wait.fill(kUnconstrained);
    write.lane_addresses = &first;
    write.second_lane_addresses = second_half ? &second : nullptr;
    write.bytes_per_lane = 8;
    write.space = MemorySpace::LocalDataShare;
    write.is_load = false;
    plane.dispatch_begin(shape);
    const std::uint64_t before = plane.current_cycles();
    plane.wave_begin(0, 0, shape.dispatch_id, shape.queue_id);
    for (int repeat = 0; repeat < 16; ++repeat)
      plane.instruction(0, 0, write);
    plane.wave_end(0, 0);
    plane.dispatch_end(shape.dispatch_id, shape.queue_id);
    return plane.current_cycles() - before;
  };
  EXPECT_GT(cost(true), cost(false));
}

TEST(TimingLds, GranuleModelCostsWideAccessesByEightByteStride) {
  const auto strided = [](std::uint64_t stride) {
    std::vector<std::uint64_t> addresses;
    for (std::uint64_t lane = 0; lane < 64; ++lane)
      addresses.push_back(lane * stride);
    return addresses;
  };
  // A dword at a 4-byte stride and a dwordx2 at an 8-byte stride are both
  // conflict free; doubling either stride is a two-way conflict.
  EXPECT_EQ(lds_granule_cycles(strided(4), 4, 64, 8, 32, 2.0), 2u);
  EXPECT_EQ(lds_granule_cycles(strided(8), 8, 64, 8, 32, 2.0), 2u);
  EXPECT_EQ(lds_granule_cycles(strided(8), 4, 64, 8, 32, 2.0), 4u);
  EXPECT_EQ(lds_granule_cycles(strided(16), 8, 64, 8, 32, 2.0),
            lds_granule_cycles(strided(16), 16, 64, 8, 32, 2.0));
  // Every lane on one bank is capped.
  EXPECT_EQ(lds_granule_cycles(strided(512), 4, 64, 8, 32, 2.0), 64u);
  // Broadcast stays free.
  EXPECT_EQ(lds_granule_cycles(std::vector<std::uint64_t>(64, 0x40), 4, 64, 8, 32, 2.0), 2u);
}

// Expected values are MI355X measurements (bench/ubench gen_ubench9/10, tr_*): CU-wide cycles per
// instruction with 4 waves on 4 SIMDs.
std::uint64_t measured_lds(std::uint32_t bytes, bool load, const std::vector<std::uint64_t> &addrs) {
  RetiredInstruction r;
  r.lane_addresses = &addrs;
  r.bytes_per_lane = bytes;
  r.is_load = load;
  r.space = MemorySpace::LocalDataShare;
  return lds_measured_cycles(r, 64, 4);
}

std::vector<std::uint64_t> lanes(std::uint64_t (*f)(std::uint64_t)) {
  std::vector<std::uint64_t> v(64);
  for (std::uint64_t lane = 0; lane < 64; ++lane)
    v[lane] = f(lane);
  return v;
}

TEST(TimingLdsMeasured, FourByteReadsCountDistinctWordsPerBank) {
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return l * 4; })), 2u);
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return l * 8; })), 4u);
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return l * 128; })), 64u);
  // broadcasts and shared words are free; distinct words in one bank are not
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t) { return std::uint64_t{0}; })), 2u);
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return (l >> 4) * 4; })), 2u);
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return (l & 1) * 512; })), 4u);
  EXPECT_EQ(measured_lds(4, true, lanes([](std::uint64_t l) { return (l & 3) * 512; })), 8u);
}

TEST(TimingLdsMeasured, WideReadsSumOverLaneGroups) {
  EXPECT_EQ(measured_lds(8, true, lanes([](std::uint64_t l) { return l * 8; })), 2u);
  EXPECT_EQ(measured_lds(8, true, lanes([](std::uint64_t l) { return l * 16; })), 4u);
  EXPECT_EQ(measured_lds(8, true, lanes([](std::uint64_t l) { return l * 128; })), 32u);
  EXPECT_EQ(measured_lds(8, true, lanes([](std::uint64_t) { return std::uint64_t{0}; })), 2u);
  EXPECT_EQ(measured_lds(16, true, lanes([](std::uint64_t l) { return l * 16; })), 4u);
  EXPECT_EQ(measured_lds(16, true, lanes([](std::uint64_t l) { return l * 32; })), 8u);
  EXPECT_EQ(measured_lds(16, true, lanes([](std::uint64_t l) { return l * 64; })), 16u);
  EXPECT_EQ(measured_lds(16, true, lanes([](std::uint64_t) { return std::uint64_t{0}; })), 4u);
}

TEST(TimingLdsMeasured, TritonTransposeReadPatternCostsEight) {
  // per-lane address of a Triton bf16 matmul's ds_read_b64_tr_b16 (mm320x2560x1280, v77)
  const auto v77 = lanes([](std::uint64_t t) {
    return ((t & 44) << 2) ^ (((t << 1) & 32) | ((t & 44) << 6) | ((t << 3) & 24));
  });
  EXPECT_EQ(measured_lds(8, true, v77), 8u);
}

TEST(TimingLdsMeasured, WritesKeepTheirWidthFloors) {
  EXPECT_EQ(measured_lds(4, false, lanes([](std::uint64_t l) { return l * 4; })), 4u);
  EXPECT_EQ(measured_lds(8, false, lanes([](std::uint64_t l) { return l * 8; })), 6u);
  EXPECT_EQ(measured_lds(16, false, lanes([](std::uint64_t l) { return l * 16; })), 13u);
  EXPECT_EQ(measured_lds(4, false, lanes([](std::uint64_t l) { return l * 16; })), 8u);
  EXPECT_EQ(measured_lds(4, false, lanes([](std::uint64_t l) { return l * 128; })), 64u);
}

} // namespace
} // namespace rocjitsu::timing
