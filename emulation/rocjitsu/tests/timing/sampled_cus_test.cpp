// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sampled_cus_test.cpp
/// @brief The timing plane's two opt-in run-time switches: ROCJITSU_TIMING_START
///        and ROCJITSU_SAMPLED_CUS.
///
/// @details Both are read from the environment when the plane and the collector
/// are built, so every test here sets what it needs through a scoped guard that
/// also clears every other switch first: a variable left over from the shell or
/// from another test in the same binary would otherwise change what is built.
///
/// What is asserted:
/// - with neither switch set, nothing a run writes changes: the trace line
///   carries no new key, and the collector drives the plane from the first
///   dispatch;
/// - with ROCJITSU_TIMING_START the plane sees nothing before the start kernel
///   -- no dispatch, wavefront, trace line or instruction-stream record -- and
///   a dispatch admitted before it stays untimed even when its wavefronts launch
///   afterwards;
/// - with ROCJITSU_SAMPLED_CUS the kept units, the per-dispatch decision and
///   the deny list are exactly what the variables say.

#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "rocjitsu/vm/timing/collector.h"
#include "rocjitsu/vm/timing/inst_class.h"
#include "rocjitsu/vm/timing/timing_plane.h"
#include "rocjitsu/vm/timing/tuning.h"

#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace rocjitsu::timing {
namespace {

/// @brief Every variable either switch reads, plus the outputs a test inspects.
constexpr std::array<const char *, 16> kSwitches = {
    "ROCJITSU_TIMING_START",
    "ROCJITSU_TIMING_TRACE",
    "ROCJITSU_TIMING_ISTREAM",
    "ROCJITSU_TIMING_ISTREAM_UNITS",
    "ROCJITSU_TIMING_ISTREAM_STRIDE",
    "ROCJITSU_SAMPLED_CUS",
    "ROCJITSU_SAMPLED_CUS_SPAN",
    "ROCJITSU_SAMPLED_CUS_MIN_WGS",
    "ROCJITSU_SAMPLED_CUS_PREFIX_WGS",
    "ROCJITSU_SAMPLED_CUS_KERNELS",
    "ROCJITSU_SAMPLED_CUS_EXCLUDE",
    "ROCJITSU_SAMPLED_CUS_DENY",
    "ROCJITSU_SAMPLED_CUS_ON_UNSAFE",
    "ROCJITSU_SAMPLED_CUS_EXIT_CODE",
    "ROCJITSU_SAMPLED_CUS_DENYLOG",
    "ROCJITSU_SAMPLED_CUS_SCALE_BW",
};

/// @brief Clears every switch for the scope, sets the ones asked for, and puts
///        the original environment back on exit.
class SwitchEnvironment {
public:
  SwitchEnvironment() {
    for (const char *name : kSwitches) {
      const char *value = std::getenv(name);
      saved_.emplace_back(name, value != nullptr ? std::optional<std::string>(value) : std::nullopt);
      unsetenv(name);
    }
  }
  ~SwitchEnvironment() {
    for (const auto &[name, value] : saved_) {
      if (value)
        setenv(name.c_str(), value->c_str(), 1);
      else
        unsetenv(name.c_str());
    }
  }
  SwitchEnvironment(const SwitchEnvironment &) = delete;
  SwitchEnvironment &operator=(const SwitchEnvironment &) = delete;

  void set(const char *name, const std::string &value) { setenv(name, value.c_str(), 1); }

private:
  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

/// @brief A small made-up machine, as in timing_model_test.cpp, with the
///        compute-unit and die counts the sampled-unit tests need.
std::string machine_config(unsigned compute_units, unsigned xcds) {
  return R"({
    "timing": {
      "enabled": true,
      "clock_mhz": 1000,
      "machine": {
        "compute_units": )" +
         std::to_string(compute_units) + R"(,
        "xcds": )" +
         std::to_string(xcds) + R"(,
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
        "wavefront_issue_cycles": 4
      }
    }
  })";
}

Tuning tiny_tuning() { return Tuning::parse(machine_config(4, 1)); }

DispatchShape shape_for(std::uint32_t dispatch_id, std::uint32_t workgroups) {
  DispatchShape shape;
  shape.dispatch_id = dispatch_id;
  shape.queue_id = 0;
  shape.kernel_name = "test_kernel";
  shape.workgroup_count = workgroups;
  shape.waves_per_workgroup = 1;
  shape.vector_registers_per_wave = 32;
  shape.scalar_registers_per_wave = 16;
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

/// @brief One dispatch straight into the plane, as timing_model_test.cpp does,
///        with an optional count of workgroups reported skipped while it is open.
void run_dispatch(TimingPlane &plane, const DispatchShape &shape, std::uint64_t skipped = 0) {
  plane.dispatch_begin(shape);
  for (std::uint32_t wg = 0; wg < shape.workgroup_count; ++wg) {
    plane.wave_begin(wg, 0, shape.dispatch_id, shape.queue_id);
    for (std::uint32_t index = 0; index < 8; ++index)
      plane.instruction(wg, 0, alu(0x1000 + 4 * index));
    plane.wave_end(wg, 0);
  }
  if (skipped != 0)
    plane.note_skipped(shape.dispatch_id, shape.queue_id, skipped);
  plane.dispatch_end(shape.dispatch_id, shape.queue_id);
}

std::vector<std::string> read_lines(const std::string &path) {
  std::vector<std::string> lines;
  std::ifstream input(path);
  for (std::string line; std::getline(input, line);)
    lines.push_back(line);
  return lines;
}

/// @brief The instruction-stream records this process wrote under @p prefix.
std::vector<IStreamRecord> read_istream(const std::string &prefix) {
  std::vector<IStreamRecord> records;
  const std::string path = prefix + "." + std::to_string(::getpid());
  std::ifstream input(path, std::ios::binary);
  IStreamRecord record;
  while (input.read(reinterpret_cast<char *>(&record), sizeof(record)))
    records.push_back(record);
  return records;
}

constexpr unsigned long long kAbsent = ~0ULL;

/// @brief The value of @p key in a trace line, or kAbsent.
unsigned long long trace_number(const std::string &line, const std::string &key) {
  const std::string needle = "\"" + key + "\":";
  const std::size_t at = line.find(needle);
  if (at == std::string::npos)
    return kAbsent;
  return std::strtoull(line.c_str() + at + needle.size(), nullptr, 10);
}

std::string trace_kernel(const std::string &line) {
  const std::string needle = "\"kernel\":\"";
  const std::size_t at = line.find(needle);
  if (at == std::string::npos)
    return {};
  const std::size_t begin = at + needle.size();
  return line.substr(begin, line.find('"', begin) - begin);
}

/// @brief Folded units a sampled dispatch executes, read through the public
///        predicate over enough workgroups to land on every unit.
std::set<std::size_t> kept_units(const TimingPlane &plane, std::size_t units) {
  std::set<std::size_t> kept;
  for (std::uint64_t wg = 0; wg < 4 * units; ++wg)
    if (plane.keeps_workgroup(wg, /*relative=*/~0ULL))
      kept.insert(plane.fold_unit(wg));
  return kept;
}

/// @brief Builds a plane on a 16-unit, two-die machine (8 units per die) with
///        sampling on and the given extra variables, and returns its kept units.
std::set<std::size_t> kept_with(const std::vector<std::pair<const char *, const char *>> &vars) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  for (const auto &[name, value] : vars)
    env.set(name, value);
  const TimingPlane plane{Tuning::parse(machine_config(16, 2))};
  return kept_units(plane, 16);
}

// -- Both switches off ---------------------------------------------------------

TEST(TimingSwitches, OffByDefault) {
  SwitchEnvironment env;
  TimingPlane plane{tiny_tuning()};
  EXPECT_FALSE(plane.sampling().enabled);
  EXPECT_TRUE(plane.sampling().deny.empty()) << "the deny list is only read with sampling on";
  for (std::uint64_t wg = 0; wg < 64; ++wg)
    EXPECT_TRUE(plane.keeps_workgroup(wg, wg));
  EXPECT_FALSE(plane.samples_dispatch("any_kernel", 1u << 20));
  EXPECT_FALSE(plane.sampling_denies("global_atomic_add_f32", true));

  TimingCollector collector{plane};
  EXPECT_FALSE(collector.sampling());
  EXPECT_FALSE(collector.sampling_wanted("any_kernel", 1u << 20));
  EXPECT_TRUE(collector.keeps_workgroup(7, 7));
}

TEST(TimingSwitches, TheTraceLineHasNoSkippedKeyWhenSamplingIsOff) {
  test::ScopedTempFile trace;
  ASSERT_TRUE(trace.initialize("rj_sampled_trace_"));
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    TimingPlane plane{tiny_tuning()};
    // A count reported with sampling off is dropped, not written.
    run_dispatch(plane, shape_for(1, 4), /*skipped=*/5);
  }
  const std::vector<std::string> lines = read_lines(trace.path());
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].find("skipped_workgroups"), std::string::npos) << lines[0];
  EXPECT_EQ(trace_number(lines[0], "workgroups"), 4u);
}

TEST(TimingSwitches, SamplingAddsOnlyTheSkippedKeyToTheTraceLine) {
  // The same dispatch on two fresh planes, one with sampling on. The executed
  // work is the same, so the lines must be the same apart from the one key.
  test::ScopedTempFile off_trace;
  test::ScopedTempFile on_trace;
  ASSERT_TRUE(off_trace.initialize("rj_sampled_off_"));
  ASSERT_TRUE(on_trace.initialize("rj_sampled_on_"));
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_TRACE", off_trace.path());
    TimingPlane plane{tiny_tuning()};
    run_dispatch(plane, shape_for(1, 4));
  }
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_TRACE", on_trace.path());
    env.set("ROCJITSU_SAMPLED_CUS", "1");
    TimingPlane plane{tiny_tuning()};
    run_dispatch(plane, shape_for(1, 4));
  }
  const std::vector<std::string> off = read_lines(off_trace.path());
  const std::vector<std::string> on = read_lines(on_trace.path());
  ASSERT_EQ(off.size(), 1u);
  ASSERT_EQ(on.size(), 1u);
  const std::string key = "\"skipped_workgroups\":0,";
  const std::size_t at = on[0].find(key);
  ASSERT_NE(at, std::string::npos) << on[0];
  EXPECT_EQ(on[0].find("\"workgroups\":4," + key), at - std::string("\"workgroups\":4,").size())
      << "the key follows workgroups";
  std::string stripped = on[0];
  stripped.erase(at, key.size());
  EXPECT_EQ(stripped, off[0]);
}

TEST(TimingSwitches, TheSkippedCountIsPerDispatchAndResetsWhenTheIdComesBack) {
  test::ScopedTempFile trace;
  ASSERT_TRUE(trace.initialize("rj_sampled_count_"));
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    env.set("ROCJITSU_SAMPLED_CUS", "1");
    TimingPlane plane{tiny_tuning()};
    plane.dispatch_begin(shape_for(1, 600));
    plane.note_skipped(1, 0, 3);
    plane.note_skipped(1, 0, 4); // one call per XCD share; they add up
    plane.note_skipped(1, 0, 0);
    plane.dispatch_end(1, 0);
    run_dispatch(plane, shape_for(1, 2)); // the id reused: starts from zero
  }
  const std::vector<std::string> lines = read_lines(trace.path());
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(trace_number(lines[0], "skipped_workgroups"), 7u);
  EXPECT_EQ(trace_number(lines[1], "skipped_workgroups"), 0u);
}

TEST(TimingSwitches, WithoutAStartKernelTheCollectorTimesFromTheFirstDispatch) {
  test::ScopedTempFile trace;
  ASSERT_TRUE(trace.initialize("rj_untgated_"));
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};
    KernelDispatchInfo info;
    info.dispatch_id = 1;
    info.kernel_name = "first_kernel";
    info.workgroup_count = 1;
    collector.dispatch_packet(info);
    collector.workgroups_skipped(1, 0, 9); // ignored: sampling is off
    collector.dispatch_execution_end(1);
  }
  const std::vector<std::string> lines = read_lines(trace.path());
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(trace_kernel(lines[0]), "first_kernel");
  EXPECT_EQ(lines[0].find("skipped_workgroups"), std::string::npos);
}

// -- ROCJITSU_TIMING_START -------------------------------------------------------

/// @brief A functional compute unit, only to own real wavefronts the collector
///        can be handed; nothing here executes.
class TimingCollectorWavesTest : public ::testing::Test {
protected:
  void SetUp() override {
    l2.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = ROCJITSU_CODE_ARCH_RDNA4;
    config.num_wf_slots = 8;
    config.sgprs_per_wf = 104;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("timing_switch_cu", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    for (std::uint64_t i = 0; i < 4; ++i)
      memory.write32(kProgram + i * 4, kEndpgm);
  }

  void TearDown() override {
    for (std::uint32_t slot = 0; cu && slot < cu->num_wf_slots(); ++slot)
      if (amdgpu::Wavefront *wave = cu->wf(slot); wave && !wave->is_halted())
        wave->halt();
  }

  /// @brief A wavefront of workgroup @p wg of dispatch @p dispatch_id, given its
  ///        collector state the way the compute unit does when one is installed.
  amdgpu::Wavefront *wave(TimingCollector &collector, std::uint32_t dispatch_id, std::uint32_t wg) {
    amdgpu::Wavefront *wf = cu->dispatch_wf(wg, kProgram, 104, 256);
    EXPECT_NE(wf, nullptr);
    if (wf == nullptr)
      return nullptr;
    wf->set_dispatch_id(dispatch_id);
    wf->set_queue_id(0);
    collector.attach(*wf);
    return wf;
  }

  static KernelDispatchInfo packet(std::uint32_t dispatch_id, const std::string &name,
                                   std::uint32_t workgroups = 1) {
    KernelDispatchInfo info;
    info.dispatch_id = dispatch_id;
    info.queue_id = 0;
    info.kernel_name = name;
    info.workgroup_count = workgroups;
    info.wfs_per_workgroup = 1;
    info.wave_size = 32;
    return info;
  }

  static constexpr std::uint64_t kProgram = 0x1000;
  static constexpr std::uint32_t kEndpgm = 0xBFB00000u; // s_endpgm (gfx12)

  amdgpu::GpuMemory memory{"timing_switch_memory"};
  amdgpu::L2Cache l2{"timing_switch_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
};

TEST_F(TimingCollectorWavesTest, TimingStartsAtTheStartKernelAndEarlierDispatchesStayUntimed) {
  test::ScopedTempFile trace;
  test::ScopedTempDirectory directory;
  ASSERT_TRUE(trace.initialize("rj_timing_start_"));
  ASSERT_TRUE(directory.initialize("rj_timing_start_is_"));
  const std::string istream = directory.path() + "/x.is";
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_START", "rj_timing_on");
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    env.set("ROCJITSU_TIMING_ISTREAM", istream);
    env.set("ROCJITSU_TIMING_ISTREAM_UNITS", "4"); // every unit of the tiny machine
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};

    // Warm-up: a dispatch that runs to completion before the start kernel.
    collector.dispatch_packet(packet(1, "warm_kernel"));
    amdgpu::Wavefront *warm = wave(collector, 1, 0);
    ASSERT_NE(warm, nullptr);
    collector.wave_dispatched(*warm);
    collector.wave_halted(*warm);
    collector.dispatch_execution_end(1);
    // Parsed ahead of the start kernel; its wavefront launches after it.
    collector.dispatch_packet(packet(2, "early_kernel"));
    EXPECT_EQ(plane.current_cycles(), 0u) << "the plane has not been called";
    EXPECT_TRUE(read_lines(trace.path()).empty());

    // The start kernel itself is the first dispatch the plane sees.
    collector.dispatch_packet(packet(3, "rj_timing_on_marker"));
    amdgpu::Wavefront *early = wave(collector, 2, 1);
    amdgpu::Wavefront *start = wave(collector, 3, 2);
    ASSERT_NE(early, nullptr);
    ASSERT_NE(start, nullptr);
    collector.wave_dispatched(*early);
    collector.wave_dispatched(*start);
    collector.wave_halted(*early);
    collector.wave_halted(*start);
    collector.dispatch_execution_end(2); // never announced: dropped
    collector.dispatch_execution_end(3);
    EXPECT_GT(plane.current_cycles(), 0u);

    // A later packet reusing the early dispatch's id is timed normally.
    collector.dispatch_packet(packet(2, "later_kernel"));
    amdgpu::Wavefront *later = wave(collector, 2, 3);
    ASSERT_NE(later, nullptr);
    collector.wave_dispatched(*later);
    collector.wave_halted(*later);
    collector.dispatch_execution_end(2);
  }

  std::vector<std::string> kernels;
  for (const std::string &line : read_lines(trace.path()))
    kernels.push_back(trace_kernel(line));
  EXPECT_EQ(kernels, (std::vector<std::string>{"rj_timing_on_marker", "later_kernel"}));

  // Header, then per timed dispatch: acquire (4), wave begin (1), wave end (2),
  // dispatch end (5). Nothing from the warm-up or the early dispatch.
  std::vector<int> types;
  std::vector<std::uint32_t> wave_dispatches;
  std::vector<std::uint32_t> wave_groups;
  for (const IStreamRecord &record : read_istream(istream)) {
    types.push_back(record.type);
    if (record.type == 1) {
      wave_dispatches.push_back(record.b);
      wave_groups.push_back(record.wg);
    }
  }
  EXPECT_EQ(types, (std::vector<int>{7, 4, 1, 2, 5, 4, 1, 2, 5}));
  EXPECT_EQ(wave_dispatches, (std::vector<std::uint32_t>{3, 2}));
  EXPECT_EQ(wave_groups, (std::vector<std::uint32_t>{2, 3}));
}

TEST_F(TimingCollectorWavesTest, WithoutTheStartKernelThePlaneIsNeverCalled) {
  test::ScopedTempFile trace;
  test::ScopedTempDirectory directory;
  ASSERT_TRUE(trace.initialize("rj_timing_never_"));
  ASSERT_TRUE(directory.initialize("rj_timing_never_is_"));
  const std::string istream = directory.path() + "/x.is";
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_START", "rj_timing_on");
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    env.set("ROCJITSU_TIMING_ISTREAM", istream);
    env.set("ROCJITSU_TIMING_ISTREAM_UNITS", "4");
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};
    for (std::uint32_t id = 1; id <= 3; ++id) {
      // A prefix match is on the start of the name, not anywhere in it.
      collector.dispatch_packet(packet(id, "not_rj_timing_on"));
      amdgpu::Wavefront *wf = wave(collector, id, id);
      ASSERT_NE(wf, nullptr);
      collector.wave_dispatched(*wf);
      collector.wave_halted(*wf);
      collector.dispatch_execution_end(id);
    }
    EXPECT_EQ(plane.current_cycles(), 0u);
  }
  EXPECT_TRUE(read_lines(trace.path()).empty());
  const std::vector<IStreamRecord> records = read_istream(istream);
  ASSERT_EQ(records.size(), 1u) << "only the header the plane writes when it is built";
  EXPECT_EQ(records[0].type, 7);
}

TEST_F(TimingCollectorWavesTest, SkippedWorkgroupsOfAnUntimedDispatchAreNotReported) {
  test::ScopedTempFile trace;
  ASSERT_TRUE(trace.initialize("rj_timing_skip_"));
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_TIMING_START", "rj_timing_on");
    env.set("ROCJITSU_TIMING_TRACE", trace.path());
    env.set("ROCJITSU_SAMPLED_CUS", "1");
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};
    collector.dispatch_packet(packet(1, "warm_kernel", 1024));
    collector.workgroups_skipped(1, 0, 1000);
    collector.dispatch_execution_end(1);
    collector.dispatch_packet(packet(2, "rj_timing_on", 1024));
    collector.workgroups_skipped(2, 0, 1000);
    collector.workgroups_skipped(2, 0, 20);
    collector.dispatch_execution_end(2);
  }
  const std::vector<std::string> lines = read_lines(trace.path());
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(trace_kernel(lines[0]), "rj_timing_on");
  EXPECT_EQ(trace_number(lines[0], "skipped_workgroups"), 1020u);
}

// -- ROCJITSU_SAMPLED_CUS: which units and which dispatches ----------------------

TEST(TimingSampledCus, FoldInterleavesWorkgroupsOverDiesThenUnits) {
  // The skip predicate folds with the plane's own fold, so a kept workgroup
  // runs on the unit it would have used in a full run.
  SwitchEnvironment env;
  const TimingPlane plane{Tuning::parse(machine_config(16, 2))};
  for (std::uint64_t wg = 0; wg < 256; ++wg)
    EXPECT_EQ(plane.fold_unit(wg), (wg % 2) * 8 + (wg / 2) % 8) << "workgroup " << wg;
}

TEST(TimingSampledCus, KeptUnitsAreTheExportedOnesWidenedBySpanWithinTheDie) {
  using Units = std::set<std::size_t>;
  // Defaults: one exported unit, no span.
  EXPECT_EQ(kept_with({}), (Units{0}));
  // UNITS / STRIDE are the instruction-stream export's variables.
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "2"}, {"ROCJITSU_TIMING_ISTREAM_STRIDE", "8"}}),
            (Units{0, 8}));
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "3"}, {"ROCJITSU_TIMING_ISTREAM_STRIDE", "5"}}),
            (Units{0, 5, 10}));
  // Exported units past the machine are dropped (the deployed 4 x 61 on 16 units).
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "4"}, {"ROCJITSU_TIMING_ISTREAM_STRIDE", "61"}}),
            (Units{0}));
  // SPAN keeps the next units on the same die.
  EXPECT_EQ(kept_with({{"ROCJITSU_SAMPLED_CUS_SPAN", "2"}}), (Units{0, 1, 2}));
  // ... wrapping within the die, never into the next one.
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "2"},
                       {"ROCJITSU_TIMING_ISTREAM_STRIDE", "6"},
                       {"ROCJITSU_SAMPLED_CUS_SPAN", "3"}}),
            (Units{0, 1, 2, 3, 6, 7}));
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "1"},
                       {"ROCJITSU_TIMING_ISTREAM_STRIDE", "1"},
                       {"ROCJITSU_SAMPLED_CUS_SPAN", "31"}}),
            (Units{0, 1, 2, 3, 4, 5, 6, 7}))
      << "a span of a die or more keeps the whole die";
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "2"},
                       {"ROCJITSU_TIMING_ISTREAM_STRIDE", "8"},
                       {"ROCJITSU_SAMPLED_CUS_SPAN", "31"}}),
            (Units{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  // Nothing exported still keeps one unit, so a sampled dispatch runs at all.
  EXPECT_EQ(kept_with({{"ROCJITSU_TIMING_ISTREAM_UNITS", "0"}}), (Units{0}));
}

TEST(TimingSampledCus, TheKeepPredicateFollowsTheFoldAndThePrefix) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_TIMING_ISTREAM_UNITS", "2");
  env.set("ROCJITSU_TIMING_ISTREAM_STRIDE", "8");
  env.set("ROCJITSU_SAMPLED_CUS_PREFIX_WGS", "10");
  const TimingPlane plane{Tuning::parse(machine_config(16, 2))};
  EXPECT_EQ(plane.sampling().prefix_workgroups, 10u);
  std::uint64_t kept = 0;
  for (std::uint64_t wg = 0; wg < 256; ++wg) {
    const std::size_t unit = plane.fold_unit(wg);
    const bool exported = unit == 0 || unit == 8;
    // Past the prefix: kept exactly when the workgroup folds onto a kept unit.
    EXPECT_EQ(plane.keeps_workgroup(wg, 1000), exported) << "workgroup " << wg;
    kept += exported ? 1 : 0;
    // The relative index decides the prefix, the grid-flat id the unit.
    EXPECT_TRUE(plane.keeps_workgroup(wg, 9));
    EXPECT_EQ(plane.keeps_workgroup(wg, 10), exported);
  }
  EXPECT_EQ(kept, 256u / 8u) << "two of sixteen units, every eighth workgroup";
}

TEST(TimingSampledCus, MinimumWorkgroupsDefaultsTo512) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  {
    const TimingPlane plane{tiny_tuning()};
    EXPECT_EQ(plane.sampling().min_workgroups, 512u);
    EXPECT_FALSE(plane.samples_dispatch("k", 511));
    EXPECT_TRUE(plane.samples_dispatch("k", 512));
  }
  env.set("ROCJITSU_SAMPLED_CUS_MIN_WGS", "4");
  const TimingPlane plane{tiny_tuning()};
  EXPECT_FALSE(plane.samples_dispatch("k", 3));
  EXPECT_TRUE(plane.samples_dispatch("k", 4));
}

TEST(TimingSampledCus, KernelListsMatchSubstringsAndExcludeWins) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_MIN_WGS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_KERNELS", ",_bwd,,_dkvm,");
  env.set("ROCJITSU_SAMPLED_CUS_EXCLUDE", "_bwd_dq,rj_timing_on");
  const TimingPlane plane{tiny_tuning()};
  EXPECT_EQ(plane.sampling().include, (std::vector<std::string>{"_bwd", "_dkvm"}))
      << "empty items are dropped";
  EXPECT_EQ(plane.sampling().exclude, (std::vector<std::string>{"_bwd_dq", "rj_timing_on"}));
  EXPECT_TRUE(plane.samples_dispatch("attn_bwd_dkdv", 1024));
  EXPECT_TRUE(plane.samples_dispatch("_dkvm", 1024));
  EXPECT_FALSE(plane.samples_dispatch("attn_bwd_dq", 1024)) << "excluded although included";
  EXPECT_FALSE(plane.samples_dispatch("attn_fwd", 1024)) << "not in the include list";
  EXPECT_FALSE(plane.samples_dispatch("", 1024)) << "an unnamed kernel is never sampled";
}

TEST(TimingSampledCus, AnyNamedKernelIsSampledWithoutAnIncludeList) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_EXCLUDE", "FillFunctor");
  TimingPlane plane{tiny_tuning()};
  TimingCollector collector{plane};
  EXPECT_TRUE(collector.sampling());
  EXPECT_TRUE(collector.sampling_wanted("anything", 512));
  EXPECT_FALSE(collector.sampling_wanted("at::FillFunctor<float>", 1u << 20));
  EXPECT_FALSE(collector.sampling_wanted("", 1u << 20));
}

TEST(TimingSampledCus, ZeroOrEmptyLeavesSamplingOff) {
  for (const char *value : {"0", ""}) {
    SwitchEnvironment env;
    env.set("ROCJITSU_SAMPLED_CUS", value);
    const TimingPlane plane{tiny_tuning()};
    EXPECT_FALSE(plane.sampling().enabled) << "'" << value << "'";
  }
}

// -- ROCJITSU_SAMPLED_CUS: the deny list ------------------------------------------

TEST(TimingSampledCusDeny, TheDefaultListRefusesAtomicsFencesGwsAndSleep) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  const TimingPlane plane{tiny_tuning()};
  ASSERT_EQ(plane.sampling().deny.size(), 9u);
  for (const auto &rule : plane.sampling().deny)
    EXPECT_FALSE(rule.returning_only) << rule.prefix;
  for (const char *mnemonic :
       {"global_atomic_add_f32", "buffer_atomic_add_f32", "flat_atomic_cmpswap", "s_atomic_add",
        "s_buffer_atomic_or", "ds_gws_init", "s_sleep", "buffer_wbl2", "buffer_inv"}) {
    EXPECT_TRUE(plane.sampling_denies(mnemonic, false)) << mnemonic;
    EXPECT_TRUE(plane.sampling_denies(mnemonic, true)) << mnemonic;
  }
  // Workgroup-local and ordinary memory operations stay allowed.
  for (const char *mnemonic : {"ds_add_u32", "ds_add_rtn_u32", "global_load_dword",
                               "buffer_store_dword", "s_barrier", "s_waitcnt"})
    EXPECT_FALSE(plane.sampling_denies(mnemonic, true)) << mnemonic;
  EXPECT_TRUE(plane.sampling().exit_on_unsafe);
  EXPECT_EQ(plane.sampling().unsafe_exit_code, 75);
  EXPECT_TRUE(plane.sampling().deny_log.empty());
}

TEST(TimingSampledCusDeny, ReturningQualifierMatchesOnlyInstructionsThatWriteARegister) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_DENY", "global_atomic+rtn,,buffer_atomic+rtn,s_sleep,buffer_wbl2");
  const TimingPlane plane{tiny_tuning()};
  const auto &deny = plane.sampling().deny;
  ASSERT_EQ(deny.size(), 4u) << "empty items are dropped";
  EXPECT_EQ(deny[0].prefix, "global_atomic") << "the qualifier is stripped from the prefix";
  EXPECT_TRUE(deny[0].returning_only);
  EXPECT_EQ(deny[1].prefix, "buffer_atomic");
  EXPECT_TRUE(deny[1].returning_only);
  EXPECT_EQ(deny[2].prefix, "s_sleep");
  EXPECT_FALSE(deny[2].returning_only);

  // A fire-and-forget atomic (no destination register) is allowed ...
  EXPECT_FALSE(plane.sampling_denies("global_atomic_add_f32", false));
  EXPECT_FALSE(plane.sampling_denies("buffer_atomic_add_f32", false));
  // ... one whose result comes back is not.
  EXPECT_TRUE(plane.sampling_denies("global_atomic_add_f32", true));
  EXPECT_TRUE(plane.sampling_denies("buffer_atomic_add_f32", true));
  // Unqualified rules match either way; prefixes not listed are allowed.
  EXPECT_TRUE(plane.sampling_denies("s_sleep", false));
  EXPECT_TRUE(plane.sampling_denies("buffer_wbl2", false));
  EXPECT_FALSE(plane.sampling_denies("buffer_inv", false));
  EXPECT_FALSE(plane.sampling_denies("flat_atomic_add", true));
}

TEST(TimingSampledCusDeny, AnEmptyListAllowsEverything) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_DENY", "");
  const TimingPlane plane{tiny_tuning()};
  EXPECT_TRUE(plane.sampling().deny.empty());
  EXPECT_FALSE(plane.sampling_denies("global_atomic_add_f32", true));
}

TEST(TimingSampledCusDeny, UnsafeActionExitCodeAndLogAreRead) {
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  env.set("ROCJITSU_SAMPLED_CUS_ON_UNSAFE", "warn");
  env.set("ROCJITSU_SAMPLED_CUS_EXIT_CODE", "3");
  env.set("ROCJITSU_SAMPLED_CUS_DENYLOG", "/tmp/deny.log");
  const TimingPlane plane{tiny_tuning()};
  EXPECT_FALSE(plane.sampling().exit_on_unsafe);
  EXPECT_EQ(plane.sampling().unsafe_exit_code, 3);
  EXPECT_EQ(plane.sampling().deny_log, "/tmp/deny.log");
}

// -- ROCJITSU_SAMPLED_CUS: the check on executed waves --------------------------

/// @brief Decoded RDNA4 instructions for the collector's per-pc check.
class TimingSampledCusCheckTest : public TimingCollectorWavesTest {
protected:
  std::unique_ptr<Instruction> decode(std::uint32_t word) {
    std::array<std::uint32_t, 4> encoding{word, 0, 0, 0};
    auto decoded = decoder->decode(encoding.data());
    EXPECT_TRUE(decoded.succeeded());
    if (!decoded.succeeded())
      return nullptr;
    return std::move(decoded.value());
  }

  /// @brief Run @p word once on a wave of a sampled (or not) dispatch.
  void execute_one(TimingCollector &collector, std::uint32_t dispatch_id, bool sampled,
                   const Instruction &inst, std::uint64_t pc) {
    collector.dispatch_packet(packet(dispatch_id, "kernel_" + std::to_string(dispatch_id), 1024));
    if (sampled)
      collector.dispatch_sampled(dispatch_id, 0, "kernel_" + std::to_string(dispatch_id));
    amdgpu::Wavefront *wf = wave(collector, dispatch_id, 0);
    ASSERT_NE(wf, nullptr);
    collector.wave_sampling(*wf, sampled);
    collector.wave_dispatched(*wf);
    collector.before_execute(pc, inst, *wf);
    collector.wave_halted(*wf);
    collector.dispatch_execution_end(dispatch_id);
  }

  static constexpr std::uint32_t kSleep = 0xBF830001u;   // s_sleep 1 (gfx12 SOPP)
  static constexpr std::uint32_t kMovS5 = 0xBE850087u;   // s_mov_b32 s5, 7

  std::unique_ptr<Decoder> decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
};

TEST_F(TimingSampledCusCheckTest, TheEncodingsAreWhatTheTestsAssume) {
  const auto sleep = decode(kSleep);
  const auto mov = decode(kMovS5);
  ASSERT_NE(sleep, nullptr);
  ASSERT_NE(mov, nullptr);
  EXPECT_EQ(sleep->mnemonic(), "s_sleep");
  EXPECT_EQ(sleep->num_dst_operands(), 0);
  EXPECT_EQ(mov->mnemonic(), "s_mov_b32");
  EXPECT_GE(mov->num_dst_operands(), 1);
}

TEST_F(TimingSampledCusCheckTest, ADeniedInstructionInASampledWaveIsLoggedOncePerKernel) {
  test::ScopedTempDirectory directory;
  ASSERT_TRUE(directory.initialize("rj_sampled_deny_"));
  const std::string log = directory.path() + "/deny.log";
  const auto sleep = decode(kSleep);
  const auto mov = decode(kMovS5);
  ASSERT_NE(sleep, nullptr);
  ASSERT_NE(mov, nullptr);
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_SAMPLED_CUS", "1");
    env.set("ROCJITSU_SAMPLED_CUS_ON_UNSAFE", "warn");
    env.set("ROCJITSU_SAMPLED_CUS_DENYLOG", log);
    env.set("ROCJITSU_SAMPLED_CUS_DENY", "s_sleep,s_mov+rtn");
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};
    execute_one(collector, 1, /*sampled=*/false, *sleep, 0x1000); // not sampled: not checked
    execute_one(collector, 2, /*sampled=*/true, *sleep, 0x1000);
    execute_one(collector, 2, /*sampled=*/true, *sleep, 0x1000); // same kernel: reported once
    // s_mov_b32 writes s5, so a "+rtn" rule matches it.
    execute_one(collector, 3, /*sampled=*/true, *mov, 0x2000);
  }
  EXPECT_EQ(read_lines(log), (std::vector<std::string>{"kernel_2\ts_sleep", "kernel_3\ts_mov_b32"}));
}

TEST_F(TimingSampledCusCheckTest, AReturningRuleIgnoresAnInstructionWithoutADestination) {
  test::ScopedTempDirectory directory;
  ASSERT_TRUE(directory.initialize("rj_sampled_rtn_"));
  const std::string log = directory.path() + "/deny.log";
  const auto sleep = decode(kSleep);
  ASSERT_NE(sleep, nullptr);
  {
    SwitchEnvironment env;
    env.set("ROCJITSU_SAMPLED_CUS", "1");
    env.set("ROCJITSU_SAMPLED_CUS_ON_UNSAFE", "warn");
    env.set("ROCJITSU_SAMPLED_CUS_DENYLOG", log);
    env.set("ROCJITSU_SAMPLED_CUS_DENY", "s_sleep+rtn");
    TimingPlane plane{tiny_tuning()};
    TimingCollector collector{plane};
    execute_one(collector, 1, /*sampled=*/true, *sleep, 0x1000);
  }
  EXPECT_TRUE(read_lines(log).empty());
}

TEST_F(TimingSampledCusCheckTest, ARefusedKernelExitsWithTheConfiguredStatus) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const auto sleep = decode(kSleep);
  ASSERT_NE(sleep, nullptr);
  SwitchEnvironment env;
  env.set("ROCJITSU_SAMPLED_CUS", "1");
  TimingPlane plane{tiny_tuning()};
  TimingCollector collector{plane};
  EXPECT_EXIT(execute_one(collector, 1, /*sampled=*/true, *sleep, 0x1000),
              ::testing::ExitedWithCode(75), "kernel 'kernel_1' executes s_sleep .* cannot be sampled; exiting");
}

} // namespace
} // namespace rocjitsu::timing
