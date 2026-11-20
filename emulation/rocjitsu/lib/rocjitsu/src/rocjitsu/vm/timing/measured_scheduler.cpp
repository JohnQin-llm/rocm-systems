// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/measured_scheduler.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <regex>
#include <unordered_set>
#include <utility>

namespace rocjitsu::timing {
namespace {

bool ends_with(const std::string &text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(const std::string &text, std::string_view part) {
  return text.find(part) != std::string::npos;
}

bool listed(const std::vector<std::string> &names, const std::string &name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

bool reads_scalar(const RegisterRanges &reads) {
  for (const RegisterRange &range : reads)
    if (range.file == RegisterFile::Scalar)
      return true;
  return false;
}

/// @brief The classes the replay's memory model averages and queues.
bool is_vector_memory(InstClass inst_class) {
  return inst_class == InstClass::VectorMemoryRead || inst_class == InstClass::VectorMemoryWrite ||
         inst_class == InstClass::VectorMemoryAtomic;
}

/// @brief VALU hold of an MFMA on its SIMD (run4 / run6).
double mfma_valu_hold(const SchedulerTuning &tuning, const std::string &mnemonic,
                      bool &adds_occupancy) {
  adds_occupancy = false;
  if (ends_with(mnemonic, "_f32") || contains(mnemonic, "xf32")) {
    adds_occupancy = true;
    return tuning.mfma_valu_hold_f32_extra_cycles;
  }
  if (ends_with(mnemonic, "x8_bf16") || ends_with(mnemonic, "x8bf16_1k") || ends_with(mnemonic, "_1k"))
    return tuning.mfma_valu_hold_legacy_cycles;
  return tuning.mfma_valu_hold_cycles;
}

std::size_t pipe_slot(SchedPipe pipe, bool per_simd, std::uint32_t simd, std::size_t simds) {
  return static_cast<std::size_t>(pipe) * simds + (per_simd ? simd : 0);
}

} // namespace

std::string normalized_mnemonic(const std::string &mnemonic) {
  for (const std::string_view suffix : {"_e32", "_e64", "_sdwa", "_dpp", "_e64_dpp"})
    if (ends_with(mnemonic, suffix))
      return mnemonic.substr(0, mnemonic.size() - suffix.size());
  return mnemonic;
}

double wave_bytes(const std::string &mnemonic) {
  double per_lane = 4.0;
  if (const std::size_t at = mnemonic.find("dwordx");
      at != std::string::npos && at + 6 < mnemonic.size() &&
      std::isdigit(static_cast<unsigned char>(mnemonic[at + 6]))) {
    per_lane = 4.0 * (mnemonic[at + 6] - '0');
  } else if (contains(mnemonic, "dword")) {
    per_lane = 4.0;
  } else {
    // The first "_b<digits>" (ds_read_b128, ds_write_b64, ...).
    for (std::size_t at = mnemonic.find("_b"); at != std::string::npos;
         at = mnemonic.find("_b", at + 1)) {
      std::size_t end = at + 2;
      while (end < mnemonic.size() && std::isdigit(static_cast<unsigned char>(mnemonic[end])))
        ++end;
      if (end > at + 2) {
        per_lane = std::stod(mnemonic.substr(at + 2, end - at - 2)) / 8.0;
        break;
      }
    }
  }
  if (contains(mnemonic, "write2") || contains(mnemonic, "read2"))
    per_lane *= 2.0;
  return 64.0 * per_lane;
}

SchedRule make_rule(const SchedulerTuning &tuning, const std::string &raw_mnemonic,
                    InstClass inst_class, std::uint8_t size_bytes, std::uint16_t immediate,
                    const RegisterRanges &reads, const RegisterRanges &writes,
                    const std::array<std::uint32_t, kNumWaitCounters> &wait, std::uint64_t pc) {
  SchedRule rule;
  rule.pc = pc;
  rule.mnemonic = raw_mnemonic;
  rule.immediate = immediate;
  rule.inst_class = inst_class;
  rule.size_bytes = size_bytes;
  rule.reads = reads;
  rule.writes = writes;
  rule.wait_thresholds.fill(kUnconstrained);
  const std::string mnemonic = normalized_mnemonic(raw_mnemonic);
  const double base = tuning.issue_interval_cycles;
  const bool long_encoding = size_bytes >= 8;
  rule.interval = base;
  rule.latency = base;

  switch (inst_class) {
  case InstClass::MatrixMultiply: {
    rule.pipe = SchedPipe::Mfma;
    rule.pipe_per_simd = true;
    rule.occupancy_dynamic = true;
    rule.extra_pipe = SchedPipe::Valu;
    rule.extra_per_simd = true;
    rule.extra_occupancy = mfma_valu_hold(tuning, mnemonic, rule.extra_adds_occupancy);
    rule.latency_is_occupancy = true;
    if (tuning.mfma_f64_rule && contains(mnemonic, "_f64")) {
      // replay11 M_F64: measured f64 occupancies; the VALU hold is occupancy + 8.
      if (contains(mnemonic, "16x16x4")) {
        rule.occupancy_dynamic = false;
        rule.occupancy = tuning.mfma_f64_16x16x4_cycles;
      } else if (contains(mnemonic, "4x4x4")) {
        rule.occupancy_dynamic = false;
        rule.occupancy = tuning.mfma_f64_4x4x4_cycles;
      }
      rule.extra_occupancy = tuning.mfma_valu_hold_f32_extra_cycles;
      rule.extra_adds_occupancy = true;
    }
    break;
  }
  case InstClass::VectorAlu:
  case InstClass::Transcendental:
  case InstClass::Unknown: {
    // replay11 M_OCC: measured conversion / lane-swap occupancies (ahead of the slow list).
    static const std::regex kConversion8(
        "^v_cvt_(scalef32_)?(sr_)?(pk_)?(fp8|bf8)_f32$|^v_cvt_(scalef32_)?sr_(pk_)?bf16_f32$|"
        "^v_permlane(16|32)_swap_b32$");
    static const std::regex kConversion4(
        "^v_cvt_(scalef32_)?(pk_)?(f32|bf16|f16)_(fp8|bf8)$|^v_cvt_pk_bf16_f32$");
    double occupancy = tuning.valu_cycles;
    if (listed(tuning.valu_fast_mnemonics, mnemonic) &&
        !(tuning.sgpr_operand_loses_double_rate && reads_scalar(reads)))
      occupancy = long_encoding ? tuning.valu_fast_long_encoding_cycles : tuning.valu_fast_cycles;
    else if (tuning.valu_conversion_table && std::regex_match(mnemonic, kConversion8))
      occupancy = tuning.valu_slow_cycles;
    else if (tuning.valu_conversion_table && std::regex_match(mnemonic, kConversion4))
      occupancy = tuning.valu_cycles;
    else if (listed(tuning.valu_slow_mnemonics, mnemonic) || inst_class == InstClass::Transcendental)
      occupancy = tuning.valu_slow_cycles;
    const bool long_interval = long_encoding && inst_class != InstClass::Unknown;
    const double interval = long_interval ? tuning.long_encoding_interval_cycles : base;
    rule.pipe = SchedPipe::Valu;
    rule.pipe_per_simd = true;
    rule.occupancy = occupancy;
    rule.interval = std::max(interval, occupancy);
    rule.short_interval = std::max(base, occupancy);
    // replay11 M_CND: the SDWA and DPP forms are VOP2 too and share the resource.
    const bool vop2_form = tuning.cndmask32_saturation &&
                           (ends_with(raw_mnemonic, "_sdwa") || ends_with(raw_mnemonic, "_dpp"));
    if (mnemonic == "v_cndmask_b32" && (size_bytes == 4 || vop2_form)) {
      rule.extra_pipe = SchedPipe::Cnd32;
      rule.extra_per_simd = false;
      rule.extra_occupancy = tuning.cndmask32_cu_cycles;
      rule.cnd32 = true;
      rule.interval = std::max(interval, tuning.cndmask32_wave_interval_cycles);
      rule.short_interval = std::max(base, tuning.cndmask32_wave_interval_cycles);
    }
    rule.latency = rule.interval;
    rule.encoding_sensitive =
        tuning.encoding_context && long_interval && rule.short_interval != rule.interval;
    break;
  }
  case InstClass::ScalarAlu:
    if (mnemonic != "s_setprio") {
      rule.pipe = SchedPipe::Salu;
      rule.pipe_per_simd = true;
      rule.occupancy = tuning.salu_cycles;
    }
    break;
  case InstClass::LdsRead:
  case InstClass::LdsWrite: {
    double hold = mnemonic == "ds_read_b128" || contains(mnemonic, "b128")
                      ? tuning.lds_read_b128_hold_cycles
                      : tuning.lds_read_hold_cycles;
    if (inst_class == InstClass::LdsWrite)
      hold = std::max(hold, wave_bytes(mnemonic) / std::max(1e-9, tuning.lds_write_bytes_per_cycle));
    // replay11 M_DS8 / M_DS16: measured wave holds of wide LDS reads.
    if (inst_class == InstClass::LdsRead) {
      if (tuning.lds_read_b64_wave_cycles > 0.0 && mnemonic.rfind("ds_read_b64", 0) == 0)
        hold = std::max(hold, tuning.lds_read_b64_wave_cycles);
      else if (tuning.lds_read_b128_wave_cycles > 0.0 && mnemonic.rfind("ds_read_b128", 0) == 0)
        hold = std::max(hold, tuning.lds_read_b128_wave_cycles);
    }
    // replay11 M_PERM: cross-lane permutes occupy the LDS at least 6 cycles.
    if (tuning.lds_permute_cycles > 0.0 &&
        (mnemonic == "ds_bpermute_b32" || mnemonic == "ds_permute_b32" || mnemonic == "ds_swizzle_b32")) {
      rule.occupancy_floor = std::max(rule.occupancy_floor, tuning.lds_permute_cycles);
      hold = std::max(hold, tuning.lds_permute_wave_cycles);
    }
    rule.pipe = SchedPipe::Lds;
    rule.occupancy_dynamic = true;
    rule.interval = hold;
    rule.latency_dynamic = true;
    break;
  }
  case InstClass::VectorMemoryRead:
  case InstClass::VectorMemoryWrite:
  case InstClass::VectorMemoryAtomic:
  case InstClass::TensorMemory:
  case InstClass::Export: {
    const double rate = inst_class == InstClass::VectorMemoryWrite ? tuning.vmem_store_bytes_per_cycle
                                                                    : tuning.vmem_load_bytes_per_cycle;
    rule.pipe = SchedPipe::Vmem;
    rule.occupancy_dynamic = true;
    rule.interval = std::max(tuning.vmem_min_hold_cycles, wave_bytes(mnemonic) / std::max(1e-9, rate));
    rule.latency_dynamic = true;
    if (tuning.vmem_address_unit) {
      // The CU's address unit (replay6 TA_RULE): dense accesses go at
      // address_bytes_per_cycle, scratch is dword-swizzled and costs per width.
      if (mnemonic.rfind("scratch_", 0) == 0) {
        std::size_t width = 1;
        if (const std::size_t at = mnemonic.find("dwordx");
            at != std::string::npos && at + 6 < mnemonic.size() &&
            std::isdigit(static_cast<unsigned char>(mnemonic[at + 6])))
          width = std::clamp<std::size_t>(static_cast<std::size_t>(mnemonic[at + 6] - '0'), 1,
                                          tuning.vmem_scratch_cycles.size());
        rule.occupancy_floor = tuning.vmem_scratch_cycles[width - 1];
      } else {
        rule.occupancy_floor =
            wave_bytes(mnemonic) / std::max(1e-9, tuning.vmem_address_bytes_per_cycle);
      }
    }
    // replay11 M_LDSDMA: a load to LDS (no VGPR destination) also takes the LDS.
    if (tuning.lds_dma_cycles > 0.0 && inst_class == InstClass::VectorMemoryRead && writes.empty() &&
        (mnemonic.rfind("buffer_load", 0) == 0 || mnemonic.rfind("global_load", 0) == 0))
      rule.lds_dma = tuning.lds_dma_cycles;
    break;
  }
  case InstClass::ScalarMemory:
    rule.pipe = SchedPipe::Smem;
    rule.occupancy_dynamic = true;
    rule.latency_dynamic = true;
    break;
  case InstClass::Nop:
    if (mnemonic == "s_nop") {
      const double count = static_cast<double>(std::min<std::uint64_t>(immediate, tuning.nop_max_count));
      rule.interval = tuning.nop_cycles_per_count * (count + 1.0);
      rule.latency = 0.0;
    }
    break;
  case InstClass::Branch:
    rule.branch = true;
    rule.interval = tuning.branch_near_cycles;
    break;
  case InstClass::WaitCounter:
    rule.wait = true;
    rule.wait_thresholds = wait;
    break;
  case InstClass::Barrier:
    rule.barrier = true;
    break;
  case InstClass::Terminate:
    rule.terminate = true;
    break;
  default:
    break;
  }
  if (mnemonic.rfind("s_setpc", 0) == 0) {
    rule.branch = true;
    rule.indirect = true;
  }
  return rule;
}

namespace {

/// @brief A wavefront's state while it is being scheduled.
struct LiveWave {
  const SchedWave *in = nullptr;
  std::uint32_t group = 0;
  std::uint32_t simd = 0;
  std::uint64_t age = 0;
  std::size_t pc = 0;
  double next_at = 0.0;
  double start = 0.0;
  double end = 0.0;
  std::uint8_t priority = 0;
  bool blocked = false;
  bool ended = false;
  /// @brief Record index where the first collapsed spin's kept iteration starts.
  std::size_t spin_mark = std::numeric_limits<std::size_t>::max();
  std::array<std::vector<double>, kNumRegisterFiles> registers;
  std::array<std::deque<double>, kNumWaitCounters> outstanding;
  /// @brief encoding_context: whether each of the last instructions was 4-byte.
  std::deque<bool> recent_short;
  std::size_t recent_short_count = 0;
};

/// @brief What picking the next wavefront needs of one that has not ended,
///        kept compact so the per-issue scan touches little memory.
///
/// @details A wavefront's ready time is max(own, pipes_[slot[0..2]]): `own`
/// (next_at and its current instruction's source registers) and the slots
/// change only when it issues or a barrier releases it, the pipes whenever
/// anyone issues. Unused slots point at a pipe entry that stays -infinity.
struct Candidate {
  /// @brief +infinity while blocked at a barrier, else own.
  double key = 0.0;
  double own = 0.0;
  std::uint64_t age = 0;
  std::array<std::uint32_t, 3> slot{};
  std::uint32_t live = 0;
  std::uint8_t priority = 0;
};

class Scheduler {
public:
  Scheduler(const SchedulerTuning &tuning, std::span<const SchedRule> rules,
            std::span<const SchedWave> waves, std::uint64_t resident, const ScheduleMemory &memory)
      : tuning_(tuning), rules_(rules), resident_(std::max<std::uint64_t>(1, resident)),
        simds_(static_cast<std::size_t>(std::max<std::uint64_t>(1, tuning.simds_per_cu))),
        memory_(memory) {
    pipes_.assign(kNumSchedPipes * simds_ + 1, 0.0);
    unused_slot_ = static_cast<std::uint32_t>(kNumSchedPipes * simds_);
    pipes_[unused_slot_] = -std::numeric_limits<double>::infinity();
    rule_fetched_.assign(rules_.size(), 0);
    // The pc-averaged memory behaviour of each vector-memory rule, looked up once.
    if (memory_.pcs != nullptr) {
      rule_memory_.assign(rules_.size(), nullptr);
      for (std::size_t index = 0; index < rules_.size(); ++index) {
        if (!is_vector_memory(rules_[index].inst_class))
          continue;
        if (const auto found = memory_.pcs->find(rules_[index].pc); found != memory_.pcs->end())
          rule_memory_[index] = &found->second;
      }
    }
    // Workgroups launch in id order, as the dispatcher hands them out, and not
    // in the order the functional plane happened to start them: that order
    // depends on its threads, and the schedule must not. Members by slot.
    std::unordered_map<std::uint32_t, std::size_t> group_of;
    // Stream-K flag spins are collapsed to their last iteration first; the
    // collapsed copies live in collapsed_, everything else is used in place.
    std::vector<const SchedWave *> effective(waves.size());
    std::vector<std::vector<SpinRange>> spins(waves.size());
    std::size_t spinning = 0;
    for (std::size_t index = 0; index < waves.size(); ++index) {
      effective[index] = &waves[index];
      if (tuning.spin_collapse)
        spins[index] = find_spins(rules, waves[index].records, tuning.spin_min_iterations);
      spinning += spins[index].empty() ? 0 : 1;
    }
    collapsed_.reserve(spinning); // no reallocation: effective points into it
    for (std::size_t index = 0; index < waves.size(); ++index) {
      if (spins[index].empty())
        continue;
      const SchedWave &wave = waves[index];
      SchedWave copy;
      copy.workgroup = wave.workgroup;
      copy.slot = wave.slot;
      copy.fetch = wave.fetch;
      // Keep [next, spin.start) and the spin's last counted iteration onwards.
      std::size_t next = 0;
      for (const SpinRange &spin : spins[index]) {
        const std::size_t length = (spin.end - spin.start) / spin.iterations;
        copy.records.insert(copy.records.end(), wave.records.begin() + next,
                            wave.records.begin() + spin.start);
        next = spin.end - length;
      }
      copy.records.insert(copy.records.end(), wave.records.begin() + next, wave.records.end());
      collapsed_.push_back(std::move(copy));
      effective[index] = &collapsed_.back();
      // Nothing before the first spin was dropped, so its kept iteration
      // starts at the same index in the copy.
      spin_marks_[effective[index]] = spins[index].front().start;
    }
    for (const SchedWave *wave : effective) {
      if (wave->records.empty())
        continue;
      const auto [found, inserted] = group_of.emplace(wave->workgroup, groups_.size());
      if (inserted)
        groups_.emplace_back();
      groups_[found->second].push_back(wave);
    }
    for (auto &group : groups_)
      std::stable_sort(group.begin(), group.end(),
                       [](const SchedWave *a, const SchedWave *b) { return a->slot < b->slot; });
    std::stable_sort(groups_.begin(), groups_.end(),
                     [](const std::vector<const SchedWave *> &a, const std::vector<const SchedWave *> &b) {
                       return a.front()->workgroup < b.front()->workgroup;
                     });
    live_.reserve(waves.size());
  }

  ScheduleResult run() {
    ScheduleResult result;
    launch(0.0);
    double now = 0.0;
    while (true) {
      const Candidate *best = nullptr;
      // Only the wavefronts that have not ended compete; the order they are
      // visited in does not matter (ready time, priority, age is a total
      // order). First the earliest ready time (a blocked wavefront's is
      // +infinity), then the highest priority and oldest among those ready then.
      const double *pipes = pipes_.data();
      const std::size_t count = candidates_.size();
      ready_.resize(count);
      double best_time = std::numeric_limits<double>::infinity();
      for (std::size_t at = 0; at < count; ++at) {
        const Candidate &candidate = candidates_[at];
        const double ready = std::max(std::max(candidate.key, pipes[candidate.slot[0]]),
                                      std::max(pipes[candidate.slot[1]], pipes[candidate.slot[2]]));
        ready_[at] = ready;
        best_time = std::min(best_time, ready);
      }
      if (best_time != std::numeric_limits<double>::infinity()) {
        for (std::size_t at = 0; at < count; ++at) {
          if (ready_[at] != best_time)
            continue;
          const Candidate &candidate = candidates_[at];
          if (best == nullptr || candidate.priority > best->priority ||
              (candidate.priority == best->priority && candidate.age < best->age))
            best = &candidate;
        }
      }
      if (best == nullptr) {
        if (!candidates_.empty()) {
          // Everyone left is at a barrier nobody can release.
          for (Candidate &candidate : candidates_) {
            candidate.key = candidate.own;
            live_[candidate.live].blocked = false;
          }
          continue;
        }
        if (next_group_ < groups_.size()) {
          // Every wave has issued its last instruction; with slots freed at
          // completion the next workgroup fits once the last one completes.
          double at = now;
          if (tuning_.slot_release_at_completion)
            at = std::max(at, max_end_);
          launch(at);
          continue;
        }
        break;
      }
      now = std::max(now, best_time);
      floor_ = now;
      issue(live_[best->live], now);
      ++result.instructions;
    }
    for (const LiveWave &wave : live_) {
      result.end = std::max(result.end, wave.end);
      result.wave_cycles += wave.end - wave.start;
      result.wave_ends.push_back(wave.end);
      ++result.waves;
    }
    result.far_read_bytes = far_read_;
    result.far_write_bytes = far_write_;
    result.l2_read_bytes = l2_read_;
    result.l2_write_bytes = l2_write_;
    result.memory_share = memory_.pcs != nullptr && memory_.queue ? std::max(1.0, memory_.share) : 0.0;
    result.spinning_waves = spin_marks_.size();
    result.spin_arrival = spin_marks_.empty() ? -1.0 : std::max(0.0, spin_arrival_);
    return result;
  }

private:
  double ready_time(const LiveWave &wave) const {
    double core = 0.0;
    return ready_time(wave, core);
  }

  /// @brief @p core receives the ready time without the extra resource, so
  ///        the caller can tell whether the extra resource decided it.
  double ready_time(const LiveWave &wave, double &core) const {
    const SchedRecord &record = wave.in->records[wave.pc];
    const SchedRule &rule = rules_[record.rule];
    double ready = candidates_[candidate_of_[static_cast<std::size_t>(&wave - live_.data())]].own;
    if (rule.pipe != SchedPipe::None)
      ready = std::max(ready, pipes_[pipe_slot(rule.pipe, rule.pipe_per_simd, wave.simd, simds_)]);
    // valu_pair: a non-double-rate VALU op or an MFMA waits for the SIMD's open slot.
    if (tuning_.valu_pair && ((rule.pipe == SchedPipe::Valu && rule.occupancy >= tuning_.valu_cycles) ||
                              rule.pipe == SchedPipe::Mfma))
      ready = std::max(ready, pipes_[pipe_slot(SchedPipe::ValuSlot, true, wave.simd, simds_)]);
    core = ready;
    if (rule.extra_pipe != SchedPipe::None)
      ready = std::max(ready, pipes_[pipe_slot(rule.extra_pipe, rule.extra_per_simd, wave.simd, simds_)]);
    return ready;
  }

  void issue(LiveWave &wave, double now) {
    if (wave.spin_mark == wave.pc)
      spin_arrival_ = std::max(spin_arrival_, now);
    const SchedRecord &record = wave.in->records[wave.pc];
    const SchedRule &rule = rules_[record.rule];
    // Whether the cnd32 resource, not the wave or its pipes, decided the issue
    // cycle (replay11 M_CND's "pipe:cnd32(extra)" reason); before any update.
    double core_ready = now;
    const bool cnd_waited = rule.cnd32 && ready_time(wave, core_ready) > core_ready;
    const double occupancy = rule.occupancy_dynamic
                                 ? std::max(static_cast<double>(record.occupancy), rule.occupancy_floor)
                                 : rule.occupancy;
    if (rule.pipe != SchedPipe::None)
      pipes_[pipe_slot(rule.pipe, rule.pipe_per_simd, wave.simd, simds_)] = now + occupancy;
    if (rule.lds_dma > 0.0) {
      double &lds = pipes_[pipe_slot(SchedPipe::Lds, false, wave.simd, simds_)];
      lds = std::max(lds, now) + rule.lds_dma;
    }
    const double extra_hold = rule.extra_occupancy + (rule.extra_adds_occupancy ? occupancy : 0.0);
    if (rule.extra_pipe != SchedPipe::None)
      pipes_[pipe_slot(rule.extra_pipe, rule.extra_per_simd, wave.simd, simds_)] = now + extra_hold;
    if (tuning_.valu_pair && (rule.pipe == SchedPipe::Valu || rule.pipe == SchedPipe::Mfma)) {
      // replay11 M_PAIR: a double-rate op opens a VALU slot another wave's
      // double-rate op may co-issue into; anything else holds the slot.
      double &slot = pipes_[pipe_slot(SchedPipe::ValuSlot, true, wave.simd, simds_)];
      if (rule.pipe == SchedPipe::Valu && occupancy < tuning_.valu_cycles) {
        if (now < slot) {
          double &valu = pipes_[pipe_slot(SchedPipe::Valu, true, wave.simd, simds_)];
          valu = std::max(valu, slot);
        } else {
          slot = now + tuning_.valu_pair_slot_cycles;
        }
      } else {
        slot = std::max(slot, now + (rule.pipe == SchedPipe::Valu
                                         ? occupancy
                                         : (rule.extra_pipe != SchedPipe::None ? extra_hold : 0.0)));
      }
    }

    // replay11 M_ENC: the wave's recent encodings decide an 8-byte op's cost.
    bool short_context = false;
    if (tuning_.encoding_context) {
      wave.recent_short.push_back(rule.size_bytes == 4);
      wave.recent_short_count += rule.size_bytes == 4 ? 1 : 0;
      if (wave.recent_short.size() > std::max<std::uint64_t>(1, tuning_.encoding_window)) {
        wave.recent_short_count -= wave.recent_short.front() ? 1 : 0;
        wave.recent_short.pop_front();
      }
      short_context = static_cast<double>(wave.recent_short_count) >=
                      tuning_.encoding_short_share * static_cast<double>(wave.recent_short.size());
    }
    const bool short_form = rule.encoding_sensitive && short_context;
    double interval = short_form ? rule.short_interval : rule.interval;
    const std::uint64_t line_bytes = std::max<std::uint64_t>(1, tuning_.fetch_line_bytes);
    if (rule.branch) {
      const bool has_next = wave.pc + 1 < wave.in->records.size();
      const std::uint64_t target = has_next ? rules_[wave.in->records[wave.pc + 1].rule].pc : 0;
      const bool taken = has_next && target != rule.pc + rule.size_bytes;
      if (!taken) {
        // An indirect transfer that happens to fall through keeps its own cost.
        if (rule.inst_class == InstClass::Branch)
          interval = tuning_.branch_not_taken_cycles;
      } else {
        const std::uint64_t line = target / line_bytes;
        const std::uint64_t here = rule.pc / line_bytes;
        if (!rule.indirect && (line == here || line == here + 1)) {
          interval = tuning_.branch_near_cycles;
        } else if (fetched_.count(line) != 0) {
          interval = tuning_.branch_cached_cycles;
        } else {
          interval = tuning_.branch_miss_cycles;
          fetched_.insert(line);
        }
      }
    }
    if (!rule_fetched_[record.rule]) {
      // The set only grows, so a rule's own line needs inserting once.
      fetched_.insert(rule.pc / line_bytes);
      rule_fetched_[record.rule] = 1;
    }
    if (tuning_.cndmask32_saturation && rule.pipe == SchedPipe::Valu) {
      // replay11 M_CND: a cnd32 op that had to wait for the resource last taken
      // from another SIMD saturates it; ordinary VALU then issues slowly.
      if (rule.cnd32) {
        if (cnd_waited && cnd_simd_ >= 0 && cnd_simd_ != static_cast<int>(wave.simd))
          cnd_saturated_until_ = now + tuning_.cndmask32_saturation_hold_cycles;
        cnd_simd_ = static_cast<int>(wave.simd);
      } else if (cnd_saturated_until_ > now) {
        interval = std::max(interval, tuning_.cndmask32_saturated_valu_cycles);
      }
    }
    wave.next_at = now + interval;

    const double latency = rule.latency_dynamic ? static_cast<double>(record.latency)
                           : rule.latency_is_occupancy ? occupancy
                           : short_form                ? rule.short_interval
                                                       : rule.latency;
    double done = now + latency;
    if (const PcMemory *mean = rule_memory_.empty() ? nullptr : rule_memory_[record.rule]) {
      // REPLAY_SYMM: the pc's dispatch-wide mean replaces this access's own
      // latency; its missing lines are the unit's demand on the device.
      const bool write = rule.inst_class == InstClass::VectorMemoryWrite;
      const double bytes = mean->lines * tuning_.memory_line_bytes;
      done = now + mean->latency;
      // The load's requests to L2 (its missing lines) each need one of the
      // unit's outstanding-request entries (replay9 R9_OUTST); without
      // enough free it waits for the oldest, and so does every younger
      // vector-memory access (the path is in order).
      double start_at = now;
      const double requests = mean->lines * (mean->far_share + mean->l2_share);
      const std::uint64_t cap = tuning_.memory_outstanding_requests;
      std::uint64_t entries = 0;
      if (cap > 0 && requests > 0.0 && !write) {
        entries = std::min<std::uint64_t>(
            cap, std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::nearbyint(requests))));
        while (!entries_.empty() && entries_.front().release <= now) {
          entry_count_ -= entries_.front().count;
          std::pop_heap(entries_.begin(), entries_.end(), LaterRelease{});
          entries_.pop_back();
        }
        while (entry_count_ + entries > cap) {
          // Wait for the earliest-released entries, one at a time.
          EntryGroup &earliest = entries_.front();
          const std::uint64_t take = std::min(earliest.count, entry_count_ + entries - cap);
          start_at = std::max(start_at, earliest.release);
          earliest.count -= take;
          entry_count_ -= take;
          if (earliest.count == 0) {
            std::pop_heap(entries_.begin(), entries_.end(), LaterRelease{});
            entries_.pop_back();
          }
        }
        if (start_at > now) {
          double &path = pipes_[pipe_slot(SchedPipe::Vmem, false, wave.simd, simds_)];
          path = std::max(path, start_at);
        }
        done = start_at + mean->latency;
      }
      (write ? far_write_ : far_read_) += bytes * mean->far_share;
      (write ? l2_write_ : l2_read_) += bytes * mean->l2_share;
      if (memory_.queue && mean->lines > 0.0) {
        // REPLAY_MEMQ: one queue per unit; the access holds it for its share
        // of the device bandwidth of the level it goes to.
        const double far_bw = write ? tuning_.memory_far_write_bytes_per_cycle
                                    : tuning_.memory_far_read_bytes_per_cycle;
        const double l2_bw = write ? tuning_.memory_l2_write_bytes_per_cycle
                                   : tuning_.memory_l2_read_bytes_per_cycle;
        double service = bytes * std::max(1.0, memory_.share) *
                         (mean->far_share / std::max(1e-9, far_bw) +
                          mean->l2_share / std::max(1e-9, l2_bw));
        if (memory_.scale >= 0.0)
          service = std::max(bytes * (mean->far_share + mean->l2_share) /
                                 std::max(1e-9, tuning_.memory_cu_link_bytes_per_cycle),
                             service * memory_.scale);
        const double start = std::max(start_at, queue_free_);
        queue_free_ = start + service;
        done = queue_free_ + mean->latency;
      }
      if (entries != 0) {
        // A beyond-L2 line holds its entry until the load completes, an L2
        // line until it returns.
        const std::uint64_t far_entries = static_cast<std::uint64_t>(std::nearbyint(
            static_cast<double>(entries) * mean->far_share /
            std::max(mean->far_share + mean->l2_share, 1e-9)));
        const double hit_done =
            std::min(done, done - mean->latency + std::min(mean->latency, tuning_.memory_l2_latency_cycles));
        hold_entries(done, std::min(far_entries, entries));
        hold_entries(hit_done, entries - std::min(far_entries, entries));
      }
    }
    // A store is acked no earlier than store_ack after issue (replay6
    // STORE_ACK). The counter queues are FIFO, so a wait retires it in
    // program order: a younger load cannot satisfy a wait before it.
    if (rule.inst_class == InstClass::VectorMemoryWrite && tuning_.vmem_store_ack_cycles > 0.0)
      done = std::max(done, now + tuning_.vmem_store_ack_cycles);
    for (const RegisterRange &range : rule.writes) {
      std::vector<double> &file = wave.registers[static_cast<std::size_t>(range.file)];
      const std::size_t last = std::size_t{range.index} + std::max<std::uint8_t>(1, range.count);
      if (file.size() < last)
        file.resize(std::max(last, file.size() * 2), 0.0);
      for (std::size_t reg = range.index; reg < last; ++reg)
        file[reg] = done;
    }
    if (record.counter < kNumWaitCounters)
      wave.outstanding[record.counter].push_back(done);
    if (rule.wait) {
      // Hardware convention: the wait issues, then holds the wavefront until
      // the named counters have drained to their thresholds.
      for (std::size_t counter = 0; counter < kNumWaitCounters; ++counter) {
        const std::uint32_t threshold = rule.wait_thresholds[counter];
        if (threshold == kUnconstrained)
          continue;
        std::deque<double> &queue = wave.outstanding[counter];
        while (queue.size() > threshold) {
          wave.next_at = std::max(wave.next_at, queue.front());
          queue.pop_front();
        }
      }
    }
    wave.priority = record.priority;
    ++wave.pc;
    if (rule.barrier) {
      wave.blocked = true;
      release(wave.group, now);
    }
    const std::size_t self = static_cast<std::size_t>(&wave - live_.data());
    if (rule.terminate || wave.pc >= wave.in->records.size()) {
      finish(wave, now + interval);
      if (next_group_ < groups_.size())
        launch(tuning_.slot_release_at_completion ? live_[self].end : now);
    } else {
      refresh(self);
    }
  }

  void finish(LiveWave &wave, double at) {
    double end = at;
    for (const std::deque<double> &queue : wave.outstanding)
      for (const double completion : queue)
        end = std::max(end, completion);
    wave.end = end;
    wave.ended = true;
    drop(static_cast<std::size_t>(&wave - live_.data()));
    max_end_ = std::max(max_end_, end);
    if (tuning_.slot_release_at_completion)
      completing_.push_back(static_cast<std::size_t>(&wave - live_.data()));
    --running_;
    release(wave.group, at);
  }

  /// @brief Release the group's barrier if every live member has arrived.
  void release(std::uint32_t group, double now) {
    bool any = false;
    for (const std::size_t index : members_[group]) {
      const LiveWave &member = live_[index];
      if (member.ended)
        continue;
      if (!member.blocked)
        return;
      any = true;
    }
    if (!any)
      return;
    for (const std::size_t index : members_[group]) {
      LiveWave &member = live_[index];
      if (member.ended)
        continue;
      member.blocked = false;
      member.next_at = std::max(member.next_at, now + tuning_.barrier_release_cycles);
      refresh(index);
    }
  }

  /// @brief Launch the next workgroups, in id order, while they fit at @p at.
  ///
  /// @details A slot is held by a wavefront that has not issued its last
  /// instruction and, with slot_release_at_completion, by one whose
  /// accesses are still outstanding at @p at. Workgroups after the first
  /// start workgroup_launch_cycles after @p at.
  void launch(double at) {
    prune_completing(at);
    std::uint64_t held = running_;
    for (const std::size_t index : completing_)
      held += live_[index].end > at ? 1 : 0;
    const double delay = at > 0.0 ? tuning_.workgroup_launch_cycles : 0.0;
    while (next_group_ < groups_.size() &&
           (held + groups_[next_group_].size() <= resident_ || held == 0)) {
      std::vector<std::uint32_t> plan;
      if (tuning_.simd_slots && !simd_plan(at, groups_[next_group_].size(), plan) && held != 0)
        break;
      held += groups_[next_group_].size();
      const std::uint32_t group = static_cast<std::uint32_t>(members_.size());
      members_.emplace_back();
      std::size_t member = 0;
      for (const SchedWave *in : groups_[next_group_]) {
        LiveWave wave;
        wave.in = in;
        wave.group = group;
        wave.simd = plan.empty() ? static_cast<std::uint32_t>(launched_ % simds_) : plan[member];
        ++member;
        wave.age = launched_++;
        wave.next_at = at + delay + static_cast<double>(in->fetch);
        if (const auto mark = spin_marks_.find(in); mark != spin_marks_.end())
          wave.spin_mark = mark->second;
        wave.start = wave.next_at;
        members_.back().push_back(live_.size());
        live_.push_back(std::move(wave));
        candidate_of_.push_back(static_cast<std::uint32_t>(candidates_.size()));
        candidates_.emplace_back();
        candidates_.back().live = static_cast<std::uint32_t>(live_.size() - 1);
        refresh(live_.size() - 1);
        ++running_;
      }
      ++next_group_;
    }
  }

  /// @brief replay11 M_SIMDSLOT: SIMDs for a workgroup of @p count waves --
  ///        spread evenly, the least-loaded SIMDs (ties from the round-robin
  ///        position) taking the larger shares -- or false when one of them
  ///        lacks room (resident / SIMDs slots each). @p plan is round-robin
  ///        when false.
  bool simd_plan(double at, std::size_t count, std::vector<std::uint32_t> &plan) const {
    const std::size_t simds = simds_;
    const std::uint64_t cap = std::max<std::uint64_t>(1, resident_ / simds);
    std::vector<std::uint64_t> load(simds, 0);
    for (const Candidate &candidate : candidates_)
      ++load[live_[candidate.live].simd];
    for (const std::size_t index : completing_) // ended (slot_release_at_completion only)
      if (live_[index].end > at)
        ++load[live_[index].simd];
    std::vector<std::size_t> share(simds);
    for (std::size_t simd = 0; simd < simds; ++simd)
      share[simd] = count / simds + (simd < count % simds ? 1 : 0);
    std::sort(share.begin(), share.end(), std::greater<std::size_t>());
    std::vector<std::size_t> order(simds);
    for (std::size_t simd = 0; simd < simds; ++simd)
      order[simd] = simd;
    const std::size_t rotate = static_cast<std::size_t>(launched_ % simds);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      const std::size_t ra = (a + simds - rotate) % simds;
      const std::size_t rb = (b + simds - rotate) % simds;
      return load[a] != load[b] ? load[a] < load[b] : ra < rb;
    });
    plan.clear();
    for (std::size_t index = 0; index < simds; ++index) {
      const std::size_t need = share[index];
      if (need != 0 && load[order[index]] + need > cap) {
        plan.clear();
        for (std::size_t member = 0; member < count; ++member)
          plan.push_back(static_cast<std::uint32_t>((launched_ + member) % simds));
        return false;
      }
      plan.insert(plan.end(), need, static_cast<std::uint32_t>(order[index]));
    }
    return true;
  }

  /// @brief Recompute live_[@p index]'s candidate from its current state.
  void refresh(std::size_t index) {
    const LiveWave &wave = live_[index];
    if (wave.pc >= wave.in->records.size())
      return; // issued its last instruction (a barrier); finishes right away
    Candidate &candidate = candidates_[candidate_of_[index]];
    const SchedRule &rule = rules_[wave.in->records[wave.pc].rule];
    double own = wave.next_at;
    for (const RegisterRange &range : rule.reads) {
      const std::vector<double> &file = wave.registers[static_cast<std::size_t>(range.file)];
      const std::size_t last = std::min<std::size_t>(file.size(), std::size_t{range.index} +
                                                                      std::max<std::uint8_t>(1, range.count));
      for (std::size_t reg = range.index; reg < last; ++reg)
        own = std::max(own, file[reg]);
    }
    candidate.own = own;
    candidate.key = wave.blocked ? std::numeric_limits<double>::infinity() : own;
    candidate.age = wave.age;
    candidate.priority = wave.priority;
    candidate.slot.fill(unused_slot_);
    if (rule.pipe != SchedPipe::None)
      candidate.slot[0] = static_cast<std::uint32_t>(pipe_slot(rule.pipe, rule.pipe_per_simd, wave.simd, simds_));
    // valu_pair: a non-double-rate VALU op or an MFMA waits for the SIMD's open slot.
    if (tuning_.valu_pair && ((rule.pipe == SchedPipe::Valu && rule.occupancy >= tuning_.valu_cycles) ||
                              rule.pipe == SchedPipe::Mfma))
      candidate.slot[1] = static_cast<std::uint32_t>(pipe_slot(SchedPipe::ValuSlot, true, wave.simd, simds_));
    if (rule.extra_pipe != SchedPipe::None)
      candidate.slot[2] =
          static_cast<std::uint32_t>(pipe_slot(rule.extra_pipe, rule.extra_per_simd, wave.simd, simds_));
  }

  /// @brief Remove live_[@p index]'s candidate (it ended).
  void drop(std::size_t index) {
    const std::uint32_t at = candidate_of_[index];
    candidates_[at] = candidates_.back();
    candidate_of_[candidates_[at].live] = at;
    candidates_.pop_back();
  }

  /// @brief Hold @p count outstanding-request entries until @p release.
  void hold_entries(double release, std::uint64_t count) {
    if (count == 0)
      return;
    entries_.push_back({release, count});
    std::push_heap(entries_.begin(), entries_.end(), LaterRelease{});
    entry_count_ += count;
  }

  /// @brief Forget ended wavefronts that completed by min(@p at, floor_):
  ///        every later launch is at or after floor_ (the current issue cycle),
  ///        so they can never hold a slot again.
  void prune_completing(double at) {
    const double limit = std::min(at, floor_);
    for (std::size_t at_index = 0; at_index < completing_.size();) {
      if (live_[completing_[at_index]].end <= limit) {
        completing_[at_index] = completing_.back();
        completing_.pop_back();
      } else {
        ++at_index;
      }
    }
  }

  const SchedulerTuning &tuning_;
  std::span<const SchedRule> rules_;
  std::uint64_t resident_;
  std::size_t simds_;
  ScheduleMemory memory_;
  /// @brief Per rule: its pc's mean memory behaviour, or null (empty when
  ///        every access keeps its own latency).
  std::vector<const PcMemory *> rule_memory_;
  /// @brief Cycle the unit's memory queue is next free, and the demand.
  double queue_free_ = 0.0;
  /// @brief Release cycles of the unit's busy TCP-to-TCC read request
  ///        entries (memory_outstanding_requests), earliest on top.
  ///        Entries released at the same cycle are kept as one group.
  struct EntryGroup {
    double release = 0.0;
    std::uint64_t count = 0;
  };
  struct LaterRelease {
    bool operator()(const EntryGroup &a, const EntryGroup &b) const { return a.release > b.release; }
  };
  std::vector<EntryGroup> entries_; // a min-heap on release
  std::uint64_t entry_count_ = 0;
  /// @brief cndmask32_saturation: saturated until, and the SIMD that last took cnd32.
  double cnd_saturated_until_ = -1.0;
  int cnd_simd_ = -1;
  double far_read_ = 0.0;
  double far_write_ = 0.0;
  double l2_read_ = 0.0;
  double l2_write_ = 0.0;
  std::vector<double> pipes_;
  std::vector<std::vector<const SchedWave *>> groups_;
  std::size_t next_group_ = 0;
  std::vector<LiveWave> live_;
  /// @brief One per wavefront that has not ended, any order; candidate_of_
  ///        maps a live_ index to its position (stale once the wave ended).
  std::vector<Candidate> candidates_;
  std::vector<std::uint32_t> candidate_of_;
  /// @brief run()'s scratch: each candidate's ready time.
  std::vector<double> ready_;
  /// @brief The pipes_ entry that is always -infinity (an unused Candidate slot).
  std::uint32_t unused_slot_ = 0;
  /// @brief slot_release_at_completion: indices of ended wavefronts whose
  ///        accesses may still be outstanding at a later launch.
  std::vector<std::size_t> completing_;
  /// @brief Latest end of an ended wavefront, and the current issue cycle.
  double max_end_ = 0.0;
  double floor_ = 0.0;
  std::vector<std::vector<std::size_t>> members_;
  std::uint64_t launched_ = 0;
  std::uint64_t running_ = 0;
  std::unordered_set<std::uint64_t> fetched_;
  /// @brief Per rule: its own fetch line is already in fetched_.
  std::vector<std::uint8_t> rule_fetched_;
  /// @brief Streams with their flag spins collapsed, and where each one's
  ///        first kept spin iteration starts.
  std::vector<SchedWave> collapsed_;
  std::unordered_map<const SchedWave *, std::size_t> spin_marks_;
  double spin_arrival_ = -1.0;
};

} // namespace

std::vector<SpinRange> find_spins(std::span<const SchedRule> rules,
                                  std::span<const SchedRecord> records,
                                  std::uint64_t min_iterations) {
  // A line-for-line port of s3_spin.spin_ranges, on rule pcs and classes.
  const auto pc = [&](std::size_t index) { return rules[records[index].rule].pc; };
  const auto spin_class = [&](std::size_t index) {
    switch (rules[records[index].rule].inst_class) {
    case InstClass::ScalarAlu:
    case InstClass::ScalarMemory:
    case InstClass::WaitCounter:
    case InstClass::Branch:
    case InstClass::Nop:
    case InstClass::DelayAlu:
    case InstClass::Message:
      return true;
    default:
      return false;
    }
  };
  std::vector<SpinRange> out;
  const std::size_t n = records.size();
  std::size_t k = 0;
  while (k + 1 < n) {
    if (rules[records[k].rule].inst_class == InstClass::Branch && pc(k + 1) < pc(k)) {
      const std::uint64_t target = pc(k + 1);
      const std::uint64_t branch = pc(k);
      // The loop starts at the last record at the target, at or before k.
      std::size_t j = k + 1;
      while (j > 0 && pc(j - 1) != target)
        --j;
      if (j > 0) {
        const std::size_t start = j - 1;
        bool only_scalar = true;
        for (std::size_t index = start; index <= k && only_scalar; ++index)
          only_scalar = spin_class(index);
        if (only_scalar) {
          const std::size_t length = k + 1 - start;
          std::size_t iterations = 1;
          std::size_t end = k + 1;
          // Further iterations count while each ends at the branch and is
          // taken back to the target again.
          while (end + length < n && pc(end + length - 1) == branch && pc(end) == target &&
                 pc(end + length) == target) {
            ++iterations;
            end += length;
          }
          if (iterations >= min_iterations) {
            out.push_back({start, end, iterations});
            k = end;
            continue;
          }
        }
      }
    }
    ++k;
  }
  return out;
}

std::vector<double> hold_spinning_units(std::span<const UnitEnd> units, double poll_cycles) {
  std::vector<double> quiet;
  for (const UnitEnd &unit : units)
    if (unit.spin_arrival < 0.0)
      quiet.push_back(unit.end);
  double partner = -1.0;
  if (!quiet.empty()) {
    // statistics.median: the mean of the two middle values for an even count.
    std::sort(quiet.begin(), quiet.end());
    const std::size_t half = quiet.size() / 2;
    partner = quiet.size() % 2 != 0 ? quiet[half] : 0.5 * (quiet[half - 1] + quiet[half]);
  }
  std::vector<double> ends;
  ends.reserve(units.size());
  for (const UnitEnd &unit : units) {
    if (unit.spin_arrival < 0.0) {
      ends.push_back(unit.end);
      continue;
    }
    const double flag = partner < 0.0 ? unit.spin_arrival : partner;
    ends.push_back(std::max(unit.spin_arrival, flag) + poll_cycles + (unit.end - unit.spin_arrival));
  }
  return ends;
}

KernelTime kernel_time(const SchedulerTuning &tuning, const KernelShape &shape) {
  // kernel_eval5, EVAL_KT branch, term for term.
  KernelTime time;
  const std::uint64_t workgroups = std::max<std::uint64_t>(1, shape.workgroups);
  const std::uint64_t per_group = std::max<std::uint64_t>(1, shape.waves_per_workgroup);
  const std::uint64_t resident = std::max<std::uint64_t>(1, shape.resident_waves);
  // Workgroups per CU in the first round: what a CU can hold (the occupancy
  // limit), not what the reported unit happened to hold.
  const std::uint64_t holds = shape.occupancy_waves != 0 ? shape.occupancy_waves : resident;
  const std::uint64_t groups_per_cu = std::max<std::uint64_t>(1, holds / per_group);
  const std::uint64_t xcds = std::max<std::uint64_t>(1, shape.xcds);
  const double first_round = static_cast<double>(
      std::min(workgroups, std::max<std::uint64_t>(1, shape.compute_units) * groups_per_cu));
  time.xcds_used = std::min(xcds, workgroups);
  const double used = static_cast<double>(time.xcds_used);
  time.head_us = tuning.kernel_queued ? tuning.kernel_queued_head_us : tuning.kernel_head_us;
  time.stagger_us = !tuning.kernel_queued && time.xcds_used > 1
                        ? tuning.kernel_xcd_stagger_us * (used - 1.0)
                        : 0.0;
  time.ramp_us = (tuning.kernel_ramp_us_per_workgroup * first_round +
                  tuning.kernel_ramp_us_per_wave * first_round * static_cast<double>(per_group)) /
                 used;
  const double clock = std::max(1e-9, shape.clock_mhz);
  const double body_clock = shape.body_clock_mhz > 0.0 ? shape.body_clock_mhz : clock;
  time.body_us = shape.slowest_unit_cycles / body_clock;
  time.tail_us = tuning.kernel_queued ? tuning.kernel_queued_tail_us
                 : time.xcds_used > 1
                     ? tuning.kernel_tail_us +
                           tuning.kernel_tail_us_per_resident_wave * static_cast<double>(resident)
                     : tuning.kernel_tail_single_xcd_us;
  time.writeback_us = std::min(shape.dirty_bytes / 1e6 * tuning.kernel_writeback_us_per_mb,
                               tuning.kernel_writeback_cap_us);
  // The larger minimum needs every XCD of a multi-XCD part (8 of 8 on MI355X).
  time.floor_us = tuning.kernel_queued ? tuning.kernel_queued_floor_us
                  : (time.xcds_used > 1 && time.xcds_used == xcds) ? tuning.kernel_floor_all_xcds_us
                                                                    : tuning.kernel_floor_us;
  time.total_us = std::max(time.head_us + time.stagger_us + time.ramp_us + time.body_us +
                               time.tail_us + time.writeback_us,
                           time.floor_us);
  time.cycles = static_cast<std::uint64_t>(std::llround(time.total_us * clock));
  return time;
}

ScheduleResult schedule_compute_unit(const SchedulerTuning &tuning, std::span<const SchedRule> rules,
                                     std::span<const SchedWave> waves, std::uint64_t resident,
                                     const ScheduleMemory &memory) {
  Scheduler scheduler(tuning, rules, waves, resident, memory);
  return scheduler.run();
}

double share_divisor(const SchedulerTuning &tuning, double active_cus, std::uint64_t workgroups,
                     std::uint64_t workgroups_on_unit) {
  if (!tuning.memory_demand_share)
    return std::max(1.0, active_cus);
  // replay7 R7_SHARE=demand: a unit holding n of the dispatch's workgroups
  // gets n / workgroups of the device bandwidth.
  return std::max(1.0, std::min(active_cus, static_cast<double>(workgroups) /
                                                static_cast<double>(std::max<std::uint64_t>(1, workgroups_on_unit))));
}

ScheduleResult schedule_compute_unit(const SchedulerTuning &tuning, std::span<const SchedRule> rules,
                                     std::span<const SchedWave> waves, std::uint64_t resident,
                                     const DispatchMemory &memory) {
  ScheduleMemory schedule;
  schedule.pcs = memory.pcs;
  ScheduleResult first = schedule_compute_unit(tuning, rules, waves, resident, schedule);
  if (memory.pcs == nullptr || !tuning.memory_queue || first.end <= 0.0)
    return first;
  // Pass 1 measured the unit's demand; its utilisation of the device is what
  // it would take at an equal share of every active unit (simulate2pass).
  const double active = std::max(1.0, memory.active_cus);
  const double busy =
      active * (first.far_read_bytes / std::max(1e-9, tuning.memory_far_read_bytes_per_cycle) +
                first.far_write_bytes / std::max(1e-9, tuning.memory_far_write_bytes_per_cycle) +
                first.l2_read_bytes / std::max(1e-9, tuning.memory_l2_read_bytes_per_cycle) +
                first.l2_write_bytes / std::max(1e-9, tuning.memory_l2_write_bytes_per_cycle));
  const double utilisation = busy / first.end;
  std::unordered_set<std::uint32_t> groups;
  for (const SchedWave &wave : waves)
    groups.insert(wave.workgroup);
  schedule.queue = true;
  schedule.share = share_divisor(tuning, memory.active_cus, memory.workgroups, groups.size());
  const auto finish = [&](ScheduleResult result, double scale) {
    result.memory_utilisation = utilisation;
    result.memory_device_cycles = busy;
    result.memory_saturation_scale = scale;
    return result;
  };
  if (utilisation < 1.0 || !tuning.memory_saturated_device) {
    schedule.scale = utilisation < 1.0 ? utilisation : -1.0;
    return finish(schedule_compute_unit(tuning, rules, waves, resident, schedule), -1.0);
  }
  // Saturated device (istream_replay7b R7B_SAT): the device is busy for
  // longer than this unit runs without queueing, so the other units fill its
  // queue's idle phases; the unit ends when the device has delivered its
  // share. Bisect the service scale so that the queued end meets the busy
  // time; keep the run at the smallest scale that still ends after it.
  schedule.scale = 1.0;
  ScheduleResult best = schedule_compute_unit(tuning, rules, waves, resident, schedule);
  if (best.end <= busy)
    return finish(std::move(best), 1.0);
  double low = 0.0;
  double high = 1.0;
  for (std::uint64_t step = 0; step < tuning.memory_saturated_iterations; ++step) {
    schedule.scale = 0.5 * (low + high);
    ScheduleResult probe = schedule_compute_unit(tuning, rules, waves, resident, schedule);
    if (probe.end > busy) {
      high = schedule.scale;
      best = std::move(probe);
    } else {
      low = schedule.scale;
    }
  }
  return finish(std::move(best), high);
}

std::uint64_t occupancy_waves(const OccupancyLimits &limits, std::uint64_t vgprs,
                              std::uint64_t sgprs, std::uint64_t lds_bytes,
                              std::uint64_t waves_per_workgroup) {
  // kernel_eval5 occ_limit, with the MI355X numbers taken from the config.
  const std::uint64_t per_group = std::max<std::uint64_t>(1, waves_per_workgroup);
  std::uint64_t limit = std::max<std::uint64_t>(1, limits.wave_slots);
  if (vgprs > 0) {
    // Registers are allocated per SIMD: 4 x (512 // vgprs) on MI355X.
    const std::uint64_t simds = std::max<std::uint64_t>(1, limits.simds);
    limit = std::min(limit, std::max<std::uint64_t>(1, simds * (limits.vector_registers / simds / vgprs)));
  }
  if (sgprs > 0)
    limit = std::min(limit, std::max<std::uint64_t>(1, limits.scalar_registers / sgprs));
  if (lds_bytes > 0)
    limit = std::min(limit, std::max<std::uint64_t>(1, limits.lds_bytes / lds_bytes) * per_group);
  return std::max(per_group, (limit / per_group) * per_group);
}

// -- Power-limited clock -----------------------------------------------------

namespace {

/// @brief The per-op energy classes of apply.py and their (class, mnemonic) key.
bool priced(InstClass inst_class) {
  return inst_class == InstClass::MatrixMultiply || inst_class == InstClass::VectorAlu ||
         inst_class == InstClass::Transcendental || inst_class == InstClass::LdsRead ||
         inst_class == InstClass::LdsWrite;
}

/// @brief First "<prefix><digits>" in @p text (prefix '_' + one of @p letters), or 0.
std::uint64_t first_width(const std::string &text, std::string_view letters) {
  for (std::size_t at = 0; at + 2 < text.size(); ++at) {
    if (text[at] != '_' || letters.find(text[at + 1]) == std::string_view::npos ||
        std::isdigit(static_cast<unsigned char>(text[at + 2])) == 0)
      continue;
    std::uint64_t value = 0;
    for (std::size_t digit = at + 2; digit < text.size() &&
                                     std::isdigit(static_cast<unsigned char>(text[digit])) != 0;
         ++digit)
      value = value * 10 + static_cast<std::uint64_t>(text[digit] - '0');
    return value;
  }
  return 0;
}

} // namespace

double mfma_energy_nj(const SchedulerTuning &tuning, const std::string &mnemonic) {
  // The MxNxK shape and the input type after it.
  std::uint64_t dims[3] = {0, 0, 0};
  std::size_t rest = std::string::npos;
  for (std::size_t at = 0; at < mnemonic.size() && rest == std::string::npos; ++at) {
    if (std::isdigit(static_cast<unsigned char>(mnemonic[at])) == 0 || (at > 0 && mnemonic[at - 1] != '_'))
      continue;
    std::size_t cursor = at;
    std::size_t parsed = 0;
    for (; parsed < 3; ++parsed) {
      std::uint64_t value = 0;
      const std::size_t first = cursor;
      while (cursor < mnemonic.size() && std::isdigit(static_cast<unsigned char>(mnemonic[cursor])) != 0)
        value = value * 10 + static_cast<std::uint64_t>(mnemonic[cursor++] - '0');
      if (cursor == first)
        break;
      dims[parsed] = value;
      if (parsed < 2) {
        if (cursor >= mnemonic.size() || mnemonic[cursor] != 'x')
          break;
        ++cursor;
      }
    }
    if (parsed == 3)
      rest = cursor;
  }
  if (rest == std::string::npos)
    return tuning.power_mfma_bf16_32x32x16_nj; // no shape: priced as the bf16 reference
  const std::string type = mnemonic.substr(rest);
  const double macs = static_cast<double>(dims[0] * dims[1] * dims[2]);
  if (contains(type, "f64")) // not measured: the f32 per-MAC energy
    return tuning.power_mfma_f32_32x32x2_nj * macs / (32.0 * 32.0 * 2.0);
  const bool narrow = contains(type, "f8") || contains(type, "f6") || contains(type, "f4") ||
                      contains(type, "i8");
  if (narrow) {
    if (dims[0] == 32 && dims[1] == 32 && dims[2] == 64)
      return tuning.power_mfma_f8_32x32x64_nj;
    if (dims[0] == 16 && dims[1] == 16 && dims[2] == 128)
      return tuning.power_mfma_f8_16x16x128_nj;
    return tuning.power_mfma_f8_32x32x64_nj * macs / (32.0 * 32.0 * 64.0);
  }
  if (contains(type, "f16")) // bf16 and f16
    return tuning.power_mfma_bf16_32x32x16_nj * macs / (32.0 * 32.0 * 16.0);
  if (contains(type, "f32"))
    return tuning.power_mfma_f32_32x32x2_nj * macs / (32.0 * 32.0 * 2.0);
  return tuning.power_mfma_bf16_32x32x16_nj * macs / (32.0 * 32.0 * 16.0);
}

double valu_energy_nj(const SchedulerTuning &tuning, const std::string &mnemonic) {
  if (mnemonic.rfind("v_pk_", 0) == 0 && ends_with(mnemonic, "_f32") && !contains(mnemonic, "mov"))
    return tuning.power_valu_pk_f32_nj;
  if (contains(mnemonic, "f64"))
    return 2.0 * tuning.power_valu_fma_nj;
  if ((contains(mnemonic, "_f32") || contains(mnemonic, "_f16") || contains(mnemonic, "_bf16")) &&
      mnemonic.rfind("v_cmp", 0) != 0 && mnemonic.rfind("v_cndmask", 0) != 0)
    return tuning.power_valu_fma_nj;
  return 0.5 * tuning.power_valu_fma_nj;
}

double lds_instruction_bytes(const std::string &mnemonic) {
  if (contains(mnemonic, "permute"))
    return 256.0;
  std::uint64_t bits = first_width(mnemonic, "b");
  if (bits == 0)
    bits = first_width(mnemonic, "ui");
  double per_lane = bits != 0 ? static_cast<double>(bits / 8) : 4.0;
  if (contains(mnemonic, "write2") || contains(mnemonic, "read2"))
    per_lane *= 2.0;
  return 64.0 * per_lane;
}

void KernelEnergy::add(std::span<const SchedRule> rules, std::span<const SchedWave> waves) {
  std::vector<double> executions(rules.size(), 0.0);
  for (const SchedWave &wave : waves)
    for (const SchedRecord &record : wave.records)
      executions[record.rule] += 1.0;
  for (std::size_t index = 0; index < rules.size(); ++index) {
    const SchedRule &rule = rules[index];
    if (executions[index] == 0.0 || !priced(rule.inst_class))
      continue;
    const std::size_t cls = static_cast<std::size_t>(rule.inst_class);
    mix_[std::string(1, static_cast<char>('A' + cls)) + normalized_mnemonic(rule.mnemonic)] +=
        executions[index];
    sampled_[cls] += executions[index];
  }
}

double KernelEnergy::joules(const SchedulerTuning &tuning,
                            const std::array<std::uint64_t, kNumInstClasses> &class_totals,
                            double l2_bytes, double fabric_bytes) const {
  double nanojoules = 0.0;
  double lds_bytes = 0.0;
  for (const auto &[key, executions] : mix_) {
    const std::size_t cls = static_cast<std::size_t>(key[0] - 'A');
    const double count = executions / sampled_[cls] * static_cast<double>(class_totals[cls]);
    const std::string mnemonic = key.substr(1);
    switch (static_cast<InstClass>(cls)) {
    case InstClass::MatrixMultiply:
      nanojoules += count * mfma_energy_nj(tuning, mnemonic);
      break;
    case InstClass::VectorAlu:
    case InstClass::Transcendental:
      nanojoules += count * valu_energy_nj(tuning, mnemonic);
      break;
    default:
      lds_bytes += count * lds_instruction_bytes(mnemonic);
      break;
    }
  }
  const double picojoules = lds_bytes * tuning.power_lds_pj_per_byte +
                            std::max(l2_bytes - fabric_bytes, 0.0) * tuning.power_l2_pj_per_byte +
                            fabric_bytes * tuning.power_fabric_pj_per_byte;
  return nanojoules * 1e-9 + picojoules * 1e-12;
}

double power_clock_mhz(const SchedulerTuning &tuning, double joules, double slowest_unit_cycles,
                       double clock_mhz) {
  if (!tuning.power_clock || joules <= 0.0 || slowest_unit_cycles <= 0.0 ||
      tuning.power_reference_mhz <= 0.0)
    return clock_mhz;
  const double seconds = slowest_unit_cycles / (tuning.power_reference_mhz * 1e6);
  const double watts = joules / seconds;
  if (watts <= tuning.power_limit_watts)
    return clock_mhz;
  return std::min(clock_mhz, tuning.power_reference_mhz * std::sqrt(tuning.power_limit_watts / watts));
}

// -- PcMemoryAverager --------------------------------------------------------

void PcMemoryAverager::add(std::uint64_t pc, double latency, double lines,
                           const SchedulerTuning &tuning) {
  const auto [found, inserted] = sums_.try_emplace(pc);
  if (inserted)
    order_.push_back(pc);
  Sum &sum = found->second;
  ++sum.executions;
  sum.latency += latency;
  sum.lines += lines;
  if (lines > 0.0 && latency >= tuning.memory_far_min_latency_cycles)
    sum.far_lines += lines;
  else if (lines > 0.0 && latency > tuning.memory_l2_min_latency_cycles)
    sum.l2_lines += lines;
}

void PcMemoryAverager::add(const SchedulerTuning &tuning, std::span<const SchedRule> rules,
                           std::span<const SchedWave> waves) {
  // replay11 M_MERGE: an L1-hit load where a sibling wave of its workgroup
  // missed at the same pc and occurrence takes the mean of those misses.
  struct Key {
    std::uint32_t workgroup;
    std::uint64_t pc;
    std::uint64_t occurrence;
    bool operator==(const Key &other) const {
      return workgroup == other.workgroup && pc == other.pc && occurrence == other.occurrence;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key &key) const {
      return std::hash<std::uint64_t>()(key.pc * 0x9E3779B97F4A7C15ull ^ key.occurrence * 1315423911ull ^
                                        key.workgroup);
    }
  };
  const double hit = tuning.memory_l2_min_latency_cycles;
  std::unordered_map<Key, std::pair<double, std::uint64_t>, KeyHash> misses;
  if (tuning.memory_merge_misses) {
    for (const SchedWave &wave : waves) {
      std::unordered_map<std::uint64_t, std::uint64_t> seen;
      for (const SchedRecord &record : wave.records) {
        const SchedRule &rule = rules[record.rule];
        if (!is_vector_memory(rule.inst_class))
          continue;
        const std::uint64_t occurrence = seen[rule.pc]++;
        if (rule.inst_class == InstClass::VectorMemoryRead && static_cast<double>(record.latency) > hit) {
          auto &sum = misses[Key{wave.workgroup, rule.pc, occurrence}];
          sum.first += static_cast<double>(record.latency);
          ++sum.second;
        }
      }
    }
  }
  for (const SchedWave &wave : waves) {
    std::unordered_map<std::uint64_t, std::uint64_t> seen;
    for (const SchedRecord &record : wave.records) {
      const SchedRule &rule = rules[record.rule];
      if (!is_vector_memory(rule.inst_class))
        continue;
      double latency = static_cast<double>(record.latency);
      if (!misses.empty()) {
        const std::uint64_t occurrence = seen[rule.pc]++;
        if (rule.inst_class == InstClass::VectorMemoryRead && latency <= hit) {
          const auto found = misses.find(Key{wave.workgroup, rule.pc, occurrence});
          if (found != misses.end())
            latency = std::max(latency, found->second.first / static_cast<double>(found->second.second));
        }
      }
      add(rule.pc, latency, static_cast<double>(record.lines), tuning);
    }
  }
}

PcMemoryTable PcMemoryAverager::build(const SchedulerTuning &tuning, double far_target) const {
  // replay5 build_pc_avg. The emulator runs workgroups in order, so the unit
  // holding workgroup 0 takes every first-touch miss; the bytes beyond L2 are
  // right in total (fabric_bytes), only their split across units is biased,
  // so the beyond-L2 share is rescaled to the dispatch-wide figure.
  bool rescale = false;
  double scale = 0.0;
  double spread = 0.0;
  bool by_scale = false;
  if (far_target >= 0.0) {
    double missing = 0.0;
    double far_all = 0.0;
    for (const std::uint64_t pc : order_) {
      const Sum &sum = sums_.at(pc);
      missing += sum.far_lines + sum.l2_lines;
      far_all += sum.far_lines;
    }
    if (missing > 0.0) {
      rescale = true;
      const double want = std::min(1.0, far_target) * missing;
      by_scale = far_all > 0.0;
      if (by_scale)
        scale = want / far_all;
      else
        spread = want / missing;
    }
  }
  PcMemoryTable table;
  table.reserve(order_.size());
  for (const std::uint64_t pc : order_) {
    const Sum &sum = sums_.at(pc);
    const double total = std::max(sum.lines, 1e-9);
    PcMemory mean;
    mean.latency = sum.latency / static_cast<double>(sum.executions);
    mean.lines = sum.lines / static_cast<double>(sum.executions);
    mean.far_share = sum.far_lines / total;
    mean.l2_share = sum.l2_lines / total;
    const double missing = sum.far_lines + sum.l2_lines;
    if (rescale && missing > 0.0) {
      const double far = by_scale ? std::min(missing, sum.far_lines * scale) : missing * spread;
      mean.latency += (far - sum.far_lines) / total *
                      (tuning.memory_far_latency_cycles - tuning.memory_l2_latency_cycles);
      mean.far_share = far / total;
      mean.l2_share = (missing - far) / total;
    }
    const double reach = mean.far_share + mean.l2_share;
    if (tuning.memory_slowest_line && mean.lines > 0.0 && reach > 0.0) {
      // replay9 R9_ANYFAR: a load completes with its slowest line, so it
      // pays the beyond-L2 latency when any of its lines goes there.
      const double far_fraction = mean.far_share / reach;
      const double any_far = 1.0 - std::pow(1.0 - far_fraction, std::max(1.0, mean.lines));
      const double step = tuning.memory_far_latency_cycles - tuning.memory_l2_latency_cycles;
      const double without = mean.latency - far_fraction * step;
      mean.latency = std::max(mean.latency, without + any_far * step);
    }
    table.emplace(pc, mean);
  }
  return table;
}

// -- ScheduleRecorder --------------------------------------------------------

SchedWave &ScheduleRecorder::wave_for(std::uint32_t workgroup, std::uint32_t slot) {
  const std::uint64_t key = (static_cast<std::uint64_t>(workgroup) << 32) | slot;
  const auto found = wave_index_.find(key);
  if (found != wave_index_.end())
    return waves_[found->second];
  wave_index_[key] = static_cast<std::uint32_t>(waves_.size());
  waves_.emplace_back();
  waves_.back().workgroup = workgroup;
  waves_.back().slot = slot;
  return waves_.back();
}

void ScheduleRecorder::wave_begin(std::uint32_t workgroup, std::uint32_t slot) {
  // A new wavefront in a (workgroup, slot) seen before starts a new stream.
  const std::uint64_t key = (static_cast<std::uint64_t>(workgroup) << 32) | slot;
  wave_index_.erase(key);
  wave_for(workgroup, slot);
}

std::uint32_t ScheduleRecorder::rule_for(const RetiredInstruction &retired) {
  static const std::string kUnknown;
  const std::string &mnemonic = retired.mnemonic != nullptr ? *retired.mnemonic : kUnknown;
  const auto found = rule_index_.find(retired.pc);
  if (found != rule_index_.end()) {
    const SchedRule &rule = rules_[found->second];
    if (rule.inst_class == retired.inst_class && rule.size_bytes == retired.size_bytes &&
        rule.immediate == retired.immediate && rule.mnemonic == mnemonic &&
        (rule.inst_class != InstClass::WaitCounter || rule.wait_thresholds == retired.wait))
      return found->second;
  }
  const std::uint32_t index = static_cast<std::uint32_t>(rules_.size());
  rules_.push_back(make_rule(tuning_, mnemonic, retired.inst_class, retired.size_bytes,
                             retired.immediate, retired.reads, retired.writes, retired.wait,
                             retired.pc));
  rule_index_[retired.pc] = index;
  return index;
}

void ScheduleRecorder::instruction(std::uint32_t workgroup, std::uint32_t slot,
                                   const RetiredInstruction &retired, std::uint64_t occupancy,
                                   std::uint64_t latency, std::uint8_t counter,
                                   std::uint64_t exposed_fetch, std::uint64_t lines) {
  const std::uint32_t rule = rule_for(retired);
  SchedWave &wave = wave_for(workgroup, slot);
  if (wave.records.empty())
    wave.fetch = static_cast<std::uint32_t>(std::min<std::uint64_t>(exposed_fetch, 0xFFFFFFFFu));
  SchedRecord record;
  record.rule = rule;
  record.occupancy = static_cast<std::uint32_t>(std::min<std::uint64_t>(occupancy, 0xFFFFFFFFu));
  record.latency = static_cast<std::uint32_t>(std::min<std::uint64_t>(latency, 0xFFFFFFFFu));
  record.counter = counter;
  record.priority = retired.user_priority;
  record.lines = static_cast<std::uint16_t>(std::min<std::uint64_t>(lines, 0xFFFFu));
  wave.records.push_back(record);
}

ScheduleResult ScheduleRecorder::run(std::uint64_t resident, const DispatchMemory *memory) {
  ScheduleResult result = memory != nullptr
                              ? schedule_compute_unit(tuning_, rules_, waves_, resident, *memory)
                              : schedule_compute_unit(tuning_, rules_, waves_, resident, ScheduleMemory{});
  clear();
  return result;
}

void ScheduleRecorder::clear() {
  waves_.clear();
  wave_index_.clear();
  // The rule table outlives dispatches (a kernel launched again reuses it),
  // but a pc re-derived for other code appends; bound it.
  if (rules_.size() > (1u << 16)) {
    rules_.clear();
    rule_index_.clear();
  }
}

} // namespace rocjitsu::timing
