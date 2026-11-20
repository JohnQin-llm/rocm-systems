// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/tuning.h"

#include "util/log.h"

#include <algorithm>
#include <cstdlib>
#include <flatbuffers/flexbuffers.h>
#include <flatbuffers/idl.h>
#include <map>

namespace rocjitsu::timing {
namespace {

/// @brief A latency nobody named, in cycles.
///
/// @details Roughly five microseconds at any plausible shader clock: longer
/// than most of the kernels this plane is asked about, so a run resting on it
/// is self-evidently not a measurement. That is the intent. An unconfigured
/// parameter has to be visible in the numbers as well as in the report, because
/// a plausible default is indistinguishable from a value someone chose.
constexpr std::uint64_t kUnnamedLatencyCycles = 10000;

/// @brief Re-read the config document with no schema.
///
/// @details Done here rather than through the plugin loader's copy of the same
/// two lines, so that the timing plane does not drag a dynamic-library loader
/// into every image that links it. The DBT translation tool and the HSA hook
/// library link the simulator and have no business gaining a dlopen dependency
/// because a config parser wanted a JSON reader.
bool flexbuffer_from_json(const std::string &json, flexbuffers::Builder &out) {
  flatbuffers::Parser parser;
  return parser.ParseFlexBuffer(json.c_str(), nullptr, &out);
}

/// @brief Flatten a nested map into dotted paths, so `l2: { ways: 16 }` and
///        `"l2.ways": 16` are the same key.
///
/// @details Both spellings occur in practice: a person writes the nested form
/// and a generator writes the flat one. Accepting only one of them turns a
/// perfectly clear config into silent fallbacks.
void flatten(const flexbuffers::Reference &value, const std::string &prefix,
             std::map<std::string, std::string> &out) {
  if (value.IsMap()) {
    const flexbuffers::Map map = value.AsMap();
    const flexbuffers::TypedVector keys = map.Keys();
    for (std::size_t index = 0; index < keys.size(); ++index) {
      const std::string key = keys[index].AsString().str();
      // A key beginning with two slashes is a comment. The config format has no
      // comment syntax of its own and the alternative is a config nobody can
      // read, so they are carried as data and skipped here.
      if (key.rfind("//", 0) == 0)
        continue;
      flatten(map[key.c_str()], prefix.empty() ? key : prefix + "." + key, out);
    }
    return;
  }
  if (value.IsVector() || value.IsNull())
    return;
  out[prefix] = value.ToString();
}

} // namespace

/// @brief Reads one key, recording whether the config named it.
class Resolver {
public:
  Resolver(const std::map<std::string, std::string> &machine, Tuning &tuning)
      : machine_(machine), tuning_(tuning) {}

  std::uint64_t integer(std::string_view key, std::uint64_t pessimistic) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    char *end = nullptr;
    const double parsed = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || !(parsed >= 0.0)) {
      util::Logger::warn("timing: machine." + name + " is not a number, using the slow default");
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return static_cast<std::uint64_t>(parsed);
  }

  double real(std::string_view key, double pessimistic) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    char *end = nullptr;
    const double parsed = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || !(parsed > 0.0)) {
      util::Logger::warn("timing: machine." + name +
                         " is not a positive rate, using the slow "
                         "default");
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return parsed;
  }

  /// @brief A structural switch. Off when absent, and absence is not a guess:
  ///        a config that does not name one describes a part without it.
  bool flag(std::string_view key) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end())
      return false;
    tuning_.resolved.push_back(name + " = " + found->second);
    return found->second == "true" || found->second == "1";
  }

  /// @brief An optional named choice. Absent is @p absent and is not a
  ///        fallback: the default is the documented behaviour, not a guess.
  std::string choice(std::string_view key, std::string_view absent) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end())
      return std::string(absent);
    tuning_.resolved.push_back(name + " = " + found->second);
    return found->second;
  }

  /// @brief A value with a default that is not a guess; reported either way.
  double measured(std::string_view key, double fallback) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name + " (measured default " + std::to_string(fallback) + ")");
      return fallback;
    }
    char *end = nullptr;
    const double parsed = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || !(parsed >= 0.0)) {
      util::Logger::warn("timing: machine." + name + " is not a number, using the measured default");
      tuning_.fell_back.push_back(name);
      return fallback;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return parsed;
  }

  bool measured_flag(std::string_view key, bool fallback) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name + (fallback ? " (default true)" : " (default false)"));
      return fallback;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return found->second == "true" || found->second == "1";
  }

  /// @brief A comma-separated list of names (the config's lists are flattened away).
  std::vector<std::string> names(std::string_view key, std::vector<std::string> fallback) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name + " (measured default list)");
      return fallback;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    std::vector<std::string> out;
    std::string current;
    for (const char character : found->second + ",") {
      if (character == ',') {
        if (!current.empty())
          out.push_back(current);
        current.clear();
      } else if (character != ' ' && character != '"') {
        current += character;
      }
    }
    return out;
  }

  /// @brief Cycles to charge one instruction of @p cls on its issue port.
  ///
  /// @details A class the config does not name resolves to the largest issue
  /// cost it gives any class. That is what makes InstClass::Unknown expensive
  /// without the config having to anticipate it: an opcode nobody classified is
  /// charged as much as the most expensive thing the part can do, so it makes a
  /// run read slow and look suspicious rather than fast and look accurate.
  std::uint64_t issue_cycles(InstClass cls) {
    const std::string key = std::string(inst_class_name(cls)) + ".issue_cycles";
    const auto found = machine_.find(key);
    if (found != machine_.end()) {
      tuning_.resolved.push_back(key + " = " + found->second);
      return std::max<std::uint64_t>(
          1, static_cast<std::uint64_t>(std::strtod(found->second.c_str(), nullptr)));
    }
    std::uint64_t widest = 1;
    for (const auto &entry : machine_) {
      if (entry.first.size() < 13 ||
          entry.first.compare(entry.first.size() - 13, 13, ".issue_cycles") != 0)
        continue;
      widest =
          std::max(widest, static_cast<std::uint64_t>(std::strtod(entry.second.c_str(), nullptr)));
    }
    tuning_.fell_back.push_back(
        key + " (resolved to the widest issue cost named: " + std::to_string(widest) + ")");
    return widest;
  }

private:
  const std::map<std::string, std::string> &machine_;
  Tuning &tuning_;
};

Tuning Tuning::parse(const std::string &config_json) {
  Tuning tuning;

  flexbuffers::Builder builder;
  if (!flexbuffer_from_json(config_json, builder))
    return tuning;
  const flexbuffers::Reference root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    return tuning;
  const flexbuffers::Reference block = root.AsMap()["timing"];
  if (!block.IsMap())
    return tuning;

  const flexbuffers::Map timing = block.AsMap();
  const flexbuffers::Reference enabled = timing["enabled"];
  // A block that is present but not enabled is a config that describes the part
  // and asks for it not to be modelled, which is a useful thing to be able to
  // say without deleting the numbers.
  tuning.enabled = !enabled.IsNull() ? enabled.AsBool() : true;
  if (!tuning.enabled)
    return tuning;

  const flexbuffers::Reference clock = timing["clock_mhz"];
  if (clock.IsNull()) {
    util::Logger::warn("timing: no clock_mhz, taking 1 GHz so cycles and nanoseconds coincide");
    tuning.fell_back.emplace_back("clock_mhz");
  } else {
    tuning.clock_mhz = clock.AsDouble();
  }

  std::map<std::string, std::string> machine;
  flatten(timing["machine"], "", machine);
  Resolver read(machine, tuning);

  tuning.compute_units = read.integer("compute_units", 1);
  tuning.xcds = read.integer("xcds", 1);
  tuning.simd_lanes = read.integer("simd_lanes", 1);
  tuning.wave_slots_per_cu = read.integer("wave_slots_per_cu", 1);
  // A register file with one register and a compute unit with one byte of group
  // memory: absurd, and deliberately so. Both resolve occupancy to a single
  // wavefront, which is the slowest the part can run and the only honest
  // reading of a config that does not describe it.
  tuning.vector_registers_per_cu = read.integer("vector_registers_per_cu", 1);
  tuning.scalar_registers_per_cu = read.integer("scalar_registers_per_cu", 1);
  tuning.lds_bytes_per_cu = read.integer("lds_bytes_per_cu", 1);

  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    tuning.issue_cycles[index] = read.issue_cycles(static_cast<InstClass>(index));
  for (std::size_t index = 0; index < kNumFunctionalUnits; ++index) {
    tuning.ports[index] = read.real(
        std::string(functional_unit_name(static_cast<FunctionalUnit>(index))) + ".ports", 1.0);
  }

  // A result latency nobody named resolves the same way every other latency
  // here does: to a figure long enough that a run resting on it is self-
  // evidently not a measurement. Resolving it to a plausible few cycles instead
  // would make a forgotten key indistinguishable from a chosen one.
  tuning.vector_alu_result_cycles = read.integer("vector_alu.result_cycles", kUnnamedLatencyCycles);
  tuning.scalar_alu_result_cycles = read.integer("scalar_alu.result_cycles", kUnnamedLatencyCycles);
  tuning.transcendental_result_cycles =
      read.integer("transcendental.result_cycles", kUnnamedLatencyCycles);
  tuning.matrix_multiply_result_cycles =
      read.integer("matrix_multiply.result_cycles", kUnnamedLatencyCycles);

  const std::uint64_t matrix_default = read.integer("matrix_multiply.macs_per_cycle", 1);
  static constexpr const char *kMatrixKeys[] = {
      "matrix_multiply.macs_per_cycle",        "matrix_multiply.macs_per_cycle.f64",
      "matrix_multiply.macs_per_cycle.f32",    "matrix_multiply.macs_per_cycle.f16",
      "matrix_multiply.macs_per_cycle.bf16",   "matrix_multiply.macs_per_cycle.narrow",
      "matrix_multiply.macs_per_cycle.integer"};
  for (std::size_t index = 0; index < kNumMatrixTypes; ++index)
    tuning.matrix_macs_per_cycle[index] = read.integer(kMatrixKeys[index], matrix_default);
  tuning.lane_addresses_per_cycle = read.real("vector_memory.lane_addresses_per_cycle", 1.0);
  tuning.wavefront_issue_cycles = read.integer("wavefront_issue_cycles", 0);
  // Falls back to the widest issue cost the config names, like any other
  // unnamed class cost: a front end nobody described should read slow.
  tuning.front_end_cycles = read.integer(
      "front_end.issue_cycles", tuning.issue_cycles[static_cast<std::size_t>(InstClass::Unknown)]);
  // Per class, defaulting to the one shared figure so that a config naming
  // only the shared one behaves exactly as it did.
  for (std::size_t index = 0; index < kNumInstClasses; ++index) {
    const std::string key = std::string("front_end.") +
                            inst_class_name(static_cast<InstClass>(index)) + ".issue_cycles";
    const double cycles = read.real(key, static_cast<double>(tuning.front_end_cycles));
    tuning.front_end_sixteenths[index] =
        static_cast<std::uint64_t>(cycles * static_cast<double>(kFrontEndScale) + 0.5);
  }
  tuning.straggler_cycles = read.integer("straggler_cycles", 0);
  tuning.barrier_cycles = read.integer("barrier_cycles", 0);
  tuning.l2_keep_across_dispatches = read.flag("l2.keep_across_dispatches");
  tuning.lds_measured_banks = read.flag("lds.measured_banks");
  tuning.barrier_lockstep = read.real("barrier_lockstep", 0.0);
  tuning.issue_occupancy_exponent = read.real("issue_occupancy_exponent", 0.0);
  tuning.stall_exposed_fraction = read.real("stall_exposed_fraction", 1.0);
  tuning.latency_exposure_scale = read.real("latency_exposure_scale", 1.0);
  tuning.fill_exposure_scale = read.real("fill_exposure_scale", 1.0);
  tuning.fill_ramp_scale = read.real("fill_ramp_scale", 1.0);
  tuning.flush_exposure_scale = read.real("flush_exposure_scale", 1.0);

  const auto cache = [&read](const std::string &prefix) {
    CacheTuning out;
    // Every pessimistic value here describes a cache that cannot help: one
    // line, four bytes wide, answering at the latency of a full miss. A level
    // the config forgot therefore turns every access into a divergent miss and
    // the run reads slow, rather than inventing a plausible cache.
    out.line_bytes = read.integer(prefix + ".line_bytes", 4);
    out.sets = read.integer(prefix + ".sets", 1);
    out.ways = read.integer(prefix + ".ways", 1);
    out.hit_cycles = read.integer(prefix + ".hit_cycles", kUnnamedLatencyCycles);
    out.lines_per_cycle = read.real(prefix + ".lines_per_cycle", 1.0);
    out.write_back = read.flag(prefix + ".write_back");
    out.write_through = read.flag(prefix + ".write_through");
    const std::string hash = read.choice(prefix + ".set_hash", "fold");
    if (hash == "mi355x") {
      out.set_hash = SetHash::Mi355x;
      if (out.sets != 4 || out.line_bytes != 128)
        util::Logger::warn("timing: machine." + prefix +
                           ".set_hash mi355x needs 4 sets of 128-byte lines; using fold");
    } else if (hash != "fold") {
      util::Logger::warn("timing: machine." + prefix + ".set_hash '" + hash +
                         "' is not fold or mi355x; using fold");
    }
    return out;
  };
  tuning.l1_vector = cache("l1_vector");
  tuning.l1_scalar = cache("l1_scalar");
  tuning.l1_instruction = cache("l1_instruction");
  tuning.l2 = cache("l2");
  tuning.mall = cache("mall");

  tuning.dram_latency_cycles = read.integer("dram.latency_cycles", kUnnamedLatencyCycles);
  tuning.dram_bytes_per_cycle = read.real("dram.bytes_per_cycle", 1.0);
  tuning.mall_bytes_per_cycle = read.real("mall.bytes_per_cycle", 1.0);
  tuning.fabric_bytes_per_cycle = read.real("fabric.bytes_per_cycle", 1.0);
  tuning.memory_channels = read.integer("memory_channels", 1);
  tuning.fabric_request_bytes = read.integer("fabric_request_bytes", 64);
  tuning.dram_row_bytes = read.integer("dram.row_bytes", 1024);
  tuning.dram_row_miss_cycles = read.integer("dram.row_miss_cycles", 0);
  tuning.miss_status_registers_per_cu = read.integer("miss_status_registers_per_cu", 1);
  tuning.stall_overlap_wavefronts = read.integer("stall_overlap_wavefronts", 1);

  tuning.lds_banks = read.integer("lds.banks", 1);
  tuning.lds_bank_bytes = read.integer("lds.bank_bytes", 4);
  tuning.lds_latency_cycles = read.integer("lds.latency_cycles", kUnnamedLatencyCycles);
  tuning.lds_lanes_per_phase = read.integer("lds.lanes_per_phase", 1);
  // Optional: absent, the phase model above stays in force and nothing here is
  // a guess, so these are looked up without being reported as fallbacks.
  if (machine.count("lds.granule_bytes") != 0) {
    tuning.lds_granule_bytes = read.integer("lds.granule_bytes", 0);
    tuning.lds_max_ways = read.integer("lds.max_ways", 64);
    tuning.lds_cycles_per_way = read.real("lds.cycles_per_way", 1.0);
  }

  tuning.dispatch_start_cycles = read.integer("dispatch.start_cycles", kUnnamedLatencyCycles);
  tuning.dispatch_end_cycles = read.integer("dispatch.end_cycles", kUnnamedLatencyCycles);
  tuning.launch_invalidate_cycles =
      read.integer("dispatch.invalidate_cycles", kUnnamedLatencyCycles);
  tuning.workgroups_per_cycle = read.real("dispatch.workgroups_per_cycle", 0.001);

  // The measured per-SIMD scheduler (measured_scheduler.h). Off unless the
  // config or ROCJITSU_TIMING_SCHEDULER turns it on; its keys are only read
  // (and reported) when it is on.
  {
    SchedulerTuning &s = tuning.scheduler;
    s.enabled = read.flag("scheduler.enabled");
    if (const char *env = std::getenv("ROCJITSU_TIMING_SCHEDULER"); env != nullptr && *env != '\0')
      s.enabled = env[0] == '1' || env[0] == 't' || env[0] == 'T';
    if (s.enabled) {
      const SchedulerTuning d;
      s.simds_per_cu = static_cast<std::uint64_t>(read.measured("scheduler.simds_per_cu", static_cast<double>(d.simds_per_cu)));
      s.issue_interval_cycles = read.measured("scheduler.issue_interval_cycles", d.issue_interval_cycles);
      s.long_encoding_interval_cycles = read.measured("scheduler.long_encoding_interval_cycles", d.long_encoding_interval_cycles);
      s.valu_fast_cycles = read.measured("scheduler.valu.fast_cycles", d.valu_fast_cycles);
      s.valu_fast_long_encoding_cycles = read.measured("scheduler.valu.fast_long_encoding_cycles", d.valu_fast_long_encoding_cycles);
      s.valu_cycles = read.measured("scheduler.valu.cycles", d.valu_cycles);
      s.valu_slow_cycles = read.measured("scheduler.valu.slow_cycles", d.valu_slow_cycles);
      s.valu_fast_mnemonics = read.names("scheduler.valu.fast_mnemonics", d.valu_fast_mnemonics);
      s.valu_slow_mnemonics = read.names("scheduler.valu.slow_mnemonics", d.valu_slow_mnemonics);
      s.sgpr_operand_loses_double_rate = read.measured_flag("scheduler.valu.sgpr_operand_loses_double_rate", d.sgpr_operand_loses_double_rate);
      s.cndmask32_cu_cycles = read.measured("scheduler.cndmask32.cu_cycles", d.cndmask32_cu_cycles);
      s.cndmask32_wave_interval_cycles = read.measured("scheduler.cndmask32.wave_interval_cycles", d.cndmask32_wave_interval_cycles);
      s.mfma_valu_hold_cycles = read.measured("scheduler.mfma.valu_hold_cycles", d.mfma_valu_hold_cycles);
      s.mfma_valu_hold_legacy_cycles = read.measured("scheduler.mfma.valu_hold_legacy_cycles", d.mfma_valu_hold_legacy_cycles);
      s.mfma_valu_hold_f32_extra_cycles = read.measured("scheduler.mfma.valu_hold_f32_extra_cycles", d.mfma_valu_hold_f32_extra_cycles);
      s.valu_conversion_table = read.measured_flag("scheduler.valu.conversion_table", d.valu_conversion_table);
      s.valu_pair = read.measured_flag("scheduler.valu.pair", d.valu_pair);
      s.valu_pair_slot_cycles = read.measured("scheduler.valu.pair_slot_cycles", d.valu_pair_slot_cycles);
      s.encoding_context = read.measured_flag("scheduler.encoding.context", d.encoding_context);
      s.encoding_window = static_cast<std::uint64_t>(read.measured("scheduler.encoding.window", static_cast<double>(d.encoding_window)));
      s.encoding_short_share = read.measured("scheduler.encoding.short_share", d.encoding_short_share);
      s.cndmask32_saturation = read.measured_flag("scheduler.cndmask32.saturation", d.cndmask32_saturation);
      s.cndmask32_saturated_valu_cycles = read.measured("scheduler.cndmask32.saturated_valu_cycles", d.cndmask32_saturated_valu_cycles);
      s.cndmask32_saturation_hold_cycles = read.measured("scheduler.cndmask32.saturation_hold_cycles", d.cndmask32_saturation_hold_cycles);
      s.mfma_f64_rule = read.measured_flag("scheduler.mfma.f64_rule", d.mfma_f64_rule);
      s.mfma_f64_16x16x4_cycles = read.measured("scheduler.mfma.f64_16x16x4_cycles", d.mfma_f64_16x16x4_cycles);
      s.mfma_f64_4x4x4_cycles = read.measured("scheduler.mfma.f64_4x4x4_cycles", d.mfma_f64_4x4x4_cycles);
      s.salu_cycles = read.measured("scheduler.salu_cycles", d.salu_cycles);
      s.lds_read_hold_cycles = read.measured("scheduler.lds.read_hold_cycles", d.lds_read_hold_cycles);
      s.lds_read_b128_hold_cycles = read.measured("scheduler.lds.read_b128_hold_cycles", d.lds_read_b128_hold_cycles);
      s.lds_write_bytes_per_cycle = read.measured("scheduler.lds.write_bytes_per_cycle", d.lds_write_bytes_per_cycle);
      s.lds_read_b64_wave_cycles = read.measured("scheduler.lds.read_b64_wave_cycles", d.lds_read_b64_wave_cycles);
      s.lds_read_b128_wave_cycles = read.measured("scheduler.lds.read_b128_wave_cycles", d.lds_read_b128_wave_cycles);
      s.lds_permute_cycles = read.measured("scheduler.lds.permute_cycles", d.lds_permute_cycles);
      s.lds_permute_wave_cycles = read.measured("scheduler.lds.permute_wave_cycles", d.lds_permute_wave_cycles);
      s.lds_dma_cycles = read.measured("scheduler.lds.dma_cycles", d.lds_dma_cycles);
      s.vmem_min_hold_cycles = read.measured("scheduler.vmem.min_hold_cycles", d.vmem_min_hold_cycles);
      s.vmem_load_bytes_per_cycle = read.measured("scheduler.vmem.load_bytes_per_cycle", d.vmem_load_bytes_per_cycle);
      s.vmem_store_bytes_per_cycle = read.measured("scheduler.vmem.store_bytes_per_cycle", d.vmem_store_bytes_per_cycle);
      s.nop_cycles_per_count = read.measured("scheduler.nop.cycles_per_count", d.nop_cycles_per_count);
      s.nop_max_count = static_cast<std::uint64_t>(read.measured("scheduler.nop.max_count", static_cast<double>(d.nop_max_count)));
      s.branch_not_taken_cycles = read.measured("scheduler.branch.not_taken_cycles", d.branch_not_taken_cycles);
      s.branch_near_cycles = read.measured("scheduler.branch.near_cycles", d.branch_near_cycles);
      s.branch_cached_cycles = read.measured("scheduler.branch.cached_cycles", d.branch_cached_cycles);
      s.branch_miss_cycles = read.measured("scheduler.branch.miss_cycles", d.branch_miss_cycles);
      s.fetch_line_bytes = static_cast<std::uint64_t>(read.measured("scheduler.branch.fetch_line_bytes", static_cast<double>(d.fetch_line_bytes)));
      s.barrier_release_cycles = read.measured("scheduler.barrier.release_cycles", d.barrier_release_cycles);
      s.bandwidth_floor = read.measured_flag("scheduler.bandwidth_floor", d.bandwidth_floor);
      s.bandwidth_floor_with_queue = read.measured_flag("scheduler.bandwidth_floor_with_queue", d.bandwidth_floor_with_queue);
      s.vmem_address_unit = read.measured_flag("scheduler.vmem.address_unit", d.vmem_address_unit);
      s.vmem_address_bytes_per_cycle = read.measured("scheduler.vmem.address_bytes_per_cycle", d.vmem_address_bytes_per_cycle);
      for (std::size_t width = 0; width < s.vmem_scratch_cycles.size(); ++width)
        s.vmem_scratch_cycles[width] = read.measured(
            "scheduler.vmem.scratch_x" + std::to_string(width + 1) + "_cycles", d.vmem_scratch_cycles[width]);
      s.vmem_store_ack_cycles = read.measured("scheduler.vmem.store_ack_cycles", d.vmem_store_ack_cycles);
      s.memory_pc_average = read.measured_flag("scheduler.memory.pc_average", d.memory_pc_average);
      s.memory_l2_min_latency_cycles = read.measured("scheduler.memory.l2_min_latency_cycles", d.memory_l2_min_latency_cycles);
      s.memory_far_min_latency_cycles = read.measured("scheduler.memory.far_min_latency_cycles", d.memory_far_min_latency_cycles);
      s.memory_far_target = read.measured_flag("scheduler.memory.far_target", d.memory_far_target);
      s.memory_far_latency_cycles = read.measured("scheduler.memory.far_latency_cycles", d.memory_far_latency_cycles);
      s.memory_l2_latency_cycles = read.measured("scheduler.memory.l2_latency_cycles", d.memory_l2_latency_cycles);
      s.memory_queue = read.measured_flag("scheduler.memory.queue", d.memory_queue);
      s.memory_line_bytes = read.measured("scheduler.memory.line_bytes", d.memory_line_bytes);
      s.memory_l2_read_bytes_per_cycle = read.measured("scheduler.memory.l2_read_bytes_per_cycle", d.memory_l2_read_bytes_per_cycle);
      s.memory_l2_write_bytes_per_cycle = read.measured("scheduler.memory.l2_write_bytes_per_cycle", d.memory_l2_write_bytes_per_cycle);
      s.memory_far_read_bytes_per_cycle = read.measured("scheduler.memory.far_read_bytes_per_cycle", d.memory_far_read_bytes_per_cycle);
      s.memory_far_write_bytes_per_cycle = read.measured("scheduler.memory.far_write_bytes_per_cycle", d.memory_far_write_bytes_per_cycle);
      s.memory_cu_link_bytes_per_cycle = read.measured("scheduler.memory.cu_link_bytes_per_cycle", d.memory_cu_link_bytes_per_cycle);
      s.memory_demand_share = read.measured_flag("scheduler.memory.demand_share", d.memory_demand_share);
      s.memory_saturated_device = read.measured_flag("scheduler.memory.saturated_device", d.memory_saturated_device);
      s.memory_saturated_iterations = static_cast<std::uint64_t>(read.measured("scheduler.memory.saturated_iterations", static_cast<double>(d.memory_saturated_iterations)));
      s.memory_slowest_line = read.measured_flag("scheduler.memory.slowest_line", d.memory_slowest_line);
      s.memory_outstanding_requests = static_cast<std::uint64_t>(read.measured("scheduler.memory.outstanding_requests", static_cast<double>(d.memory_outstanding_requests)));
      s.memory_merge_misses = read.measured_flag("scheduler.memory.merge_misses", d.memory_merge_misses);
      s.simd_slots = read.measured_flag("scheduler.launch.simd_slots", d.simd_slots);
      s.slot_release_at_completion = read.measured_flag("scheduler.launch.slot_release_at_completion", d.slot_release_at_completion);
      s.workgroup_launch_cycles = read.measured("scheduler.launch.workgroup_cycles", d.workgroup_launch_cycles);
      s.spin_collapse = read.measured_flag("scheduler.spin.collapse", d.spin_collapse);
      s.spin_min_iterations = static_cast<std::uint64_t>(read.measured("scheduler.spin.min_iterations", static_cast<double>(d.spin_min_iterations)));
      s.spin_poll_cycles = read.measured("scheduler.spin.poll_cycles", d.spin_poll_cycles);
      s.kernel_time = read.measured_flag("scheduler.kernel.enabled", d.kernel_time);
      s.kernel_queued = read.choice("scheduler.kernel.regime", "queued") != "idle_gap";
      s.kernel_queued_head_us = read.measured("scheduler.kernel.queued_head_us", d.kernel_queued_head_us);
      s.kernel_queued_tail_us = read.measured("scheduler.kernel.queued_tail_us", d.kernel_queued_tail_us);
      s.kernel_queued_floor_us = read.measured("scheduler.kernel.queued_floor_us", d.kernel_queued_floor_us);
      s.kernel_head_us = read.measured("scheduler.kernel.head_us", d.kernel_head_us);
      s.kernel_xcd_stagger_us = read.measured("scheduler.kernel.xcd_stagger_us", d.kernel_xcd_stagger_us);
      s.kernel_ramp_us_per_workgroup = read.measured("scheduler.kernel.ramp_us_per_workgroup", d.kernel_ramp_us_per_workgroup);
      s.kernel_ramp_us_per_wave = read.measured("scheduler.kernel.ramp_us_per_wave", d.kernel_ramp_us_per_wave);
      s.kernel_tail_us = read.measured("scheduler.kernel.tail_us", d.kernel_tail_us);
      s.kernel_tail_us_per_resident_wave = read.measured("scheduler.kernel.tail_us_per_resident_wave", d.kernel_tail_us_per_resident_wave);
      s.kernel_tail_single_xcd_us = read.measured("scheduler.kernel.tail_single_xcd_us", d.kernel_tail_single_xcd_us);
      s.kernel_writeback_us_per_mb = read.measured("scheduler.kernel.writeback_us_per_mb", d.kernel_writeback_us_per_mb);
      s.kernel_writeback_cap_us = read.measured("scheduler.kernel.writeback_cap_us", d.kernel_writeback_cap_us);
      s.kernel_floor_all_xcds_us = read.measured("scheduler.kernel.floor_all_xcds_us", d.kernel_floor_all_xcds_us);
      s.kernel_floor_us = read.measured("scheduler.kernel.floor_us", d.kernel_floor_us);
      s.power_clock = read.measured_flag("scheduler.power.enabled", d.power_clock);
      s.power_limit_watts = read.measured("scheduler.power.limit_watts", d.power_limit_watts);
      s.power_reference_mhz = read.measured("scheduler.power.reference_mhz", d.power_reference_mhz);
      s.power_mfma_f8_32x32x64_nj = read.measured("scheduler.power.mfma_f8_32x32x64_nj", d.power_mfma_f8_32x32x64_nj);
      s.power_mfma_f8_16x16x128_nj = read.measured("scheduler.power.mfma_f8_16x16x128_nj", d.power_mfma_f8_16x16x128_nj);
      s.power_mfma_bf16_32x32x16_nj = read.measured("scheduler.power.mfma_bf16_32x32x16_nj", d.power_mfma_bf16_32x32x16_nj);
      s.power_mfma_f32_32x32x2_nj = read.measured("scheduler.power.mfma_f32_32x32x2_nj", d.power_mfma_f32_32x32x2_nj);
      s.power_valu_fma_nj = read.measured("scheduler.power.valu_fma_nj", d.power_valu_fma_nj);
      s.power_valu_pk_f32_nj = read.measured("scheduler.power.valu_pk_f32_nj", d.power_valu_pk_f32_nj);
      s.power_lds_pj_per_byte = read.measured("scheduler.power.lds_pj_per_byte", d.power_lds_pj_per_byte);
      s.power_l2_pj_per_byte = read.measured("scheduler.power.l2_pj_per_byte", d.power_l2_pj_per_byte);
      s.power_fabric_pj_per_byte = read.measured("scheduler.power.fabric_pj_per_byte", d.power_fabric_pj_per_byte);
      s.spin_min_iterations = std::max<std::uint64_t>(2, s.spin_min_iterations);
      s.simds_per_cu = std::max<std::uint64_t>(1, s.simds_per_cu);
      s.fetch_line_bytes = std::max<std::uint64_t>(1, s.fetch_line_bytes);
    }
  }

  std::sort(tuning.resolved.begin(), tuning.resolved.end());
  std::sort(tuning.fell_back.begin(), tuning.fell_back.end());
  return tuning;
}

void Tuning::write_report(std::string &out) const {
  out += "timing plane: ";
  out += enabled ? "enabled" : "disabled";
  out += "\nshader clock: " + std::to_string(clock_mhz) + " MHz\n";
  out += std::string("composition: ") +
         (scheduler.enabled ? "measured per-SIMD scheduler" : "closed-form terms") + "\n";
  out += "parameters in effect (" + std::to_string(resolved.size()) + "):\n";
  for (const std::string &entry : resolved)
    out += "  machine." + entry + "\n";
  if (fell_back.empty())
    return;
  // Named, not counted. A parameter the config forgot changes what the run
  // computed, and a reader who is going to trust a number needs to see which
  // numbers were nobody's decision.
  out += "parameters the config did not name, resolved to their slowest value (" +
         std::to_string(fell_back.size()) + "):\n";
  for (const std::string &entry : fell_back)
    out += "  machine." + entry + "\n";
}

} // namespace rocjitsu::timing
