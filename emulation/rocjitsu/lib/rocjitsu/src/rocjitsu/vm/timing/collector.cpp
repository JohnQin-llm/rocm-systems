// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/collector.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wait_counters.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/timing/classify.h"
#include "rocjitsu/vm/timing/timing_plane.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <string_view>
#include <utility>

namespace rocjitsu::timing {
namespace {

bool has_prefix(std::string_view text, std::string_view prefix) { return text.starts_with(prefix); }

/// @brief A stand-in for a kernel whose ELF symbol could not be recovered.
///
/// @details Not a name and not pretending to be one; the prefix is there so it
/// cannot be mistaken for a demangled symbol. What it does provide is the two
/// properties a report needs: distinct kernels get distinct strings, and the
/// same kernel dispatched repeatedly gets the same one, so its rows group
/// instead of collapsing into an undifferentiated pile of "?". Falls back to
/// the entry pc when the kernel object handle is unavailable, and to the shared
/// unknown identity when neither is.
std::string unnamed_kernel_identity(const rocjitsu::KernelDispatchInfo &info) {
  const std::uint64_t handle = info.kernel_object != 0 ? info.kernel_object : info.entry_pc;
  if (handle == 0)
    return std::string(rocjitsu::kUnknownKernelIdentity);
  return std::format("unnamed@{:#x}", handle);
}

// -- Register ranges ---------------------------------------------------------

/// @brief Translate a decoded register reference into a dependency range.
/// @returns Whether the reference names a file the scoreboard tracks.
bool range_from(const RegisterRef &ref, RegisterRange &out) {
  switch (ref.cls) {
  case RegClass::VGPR:
    out.file = RegisterFile::Vector;
    break;
  case RegClass::ACC_VGPR:
    out.file = RegisterFile::Accumulator;
    break;
  case RegClass::SGPR:
    out.file = RegisterFile::Scalar;
    break;
  default:
    // The condition code, the exec mask, the mode and the program counter carry
    // dependencies the scalar unit forwards in a single cycle. Tracking them
    // would add bookkeeping on the hottest path for a stall the hardware does
    // not have.
    return false;
  }
  out.index = ref.index;
  out.count = ref.width == 0 ? 1u : ref.width;
  return true;
}

/// @brief Append @p operand's register range to @p ranges, if it names one.
void append_operand(const Operand *operand, RegisterRanges &ranges) {
  if (operand == nullptr)
    return;
  const std::optional<RegisterRef> ref = operand->to_register_ref();
  if (!ref.has_value())
    return;
  RegisterRange range;
  if (range_from(*ref, range))
    ranges.push(range);
}

// -- Waits -------------------------------------------------------------------

/// @brief Bring every counter to zero: the wavefront drains completely.
void set_full_drain(WaitThresholds &out) {
  for (std::size_t index = 0; index < kNumWaitCounters; ++index)
    out.set(static_cast<WaitCounter>(index), 0);
}

/// @brief Fill @p out with the thresholds *this* instruction established.
///
/// @param mnemonic Decides which fields the instruction wrote.
/// @param target The wavefront's decoded wait target, read after execution.
///
/// @details The wavefront's wait target is sticky: an instruction naming one
/// counter leaves every other field holding whatever an earlier wait put there.
/// Copying the whole target would therefore attribute stale thresholds to this
/// instruction, and the plane would insert stalls the hardware never had. The
/// mnemonic says exactly which fields were written, and rocjitsu has already
/// decoded the immediate into them, so the decoded value is used rather than a
/// per-family immediate layout duplicated here that would have to be kept in
/// step with the ISA generator forever.
void fill_wait(std::string_view mnemonic, const amdgpu::WaitTarget &target, WaitThresholds &out) {
  if (mnemonic == "s_waitcnt") {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    out.set(WaitCounter::LgkmCombined, target.lgkmcnt);
    out.set(WaitCounter::Export, target.expcnt);
    return;
  }
  // The GFX10 single-counter spellings. They are checked before the GFX11
  // `s_wait_<name>` family because they share no prefix with it, and leaving
  // them to fall through would produce a wait that constrains nothing, which is
  // the exact shape of a stall silently costing zero.
  if (mnemonic == "s_waitcnt_vmcnt") {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    return;
  }
  if (mnemonic == "s_waitcnt_lgkmcnt") {
    out.set(WaitCounter::LgkmCombined, target.lgkmcnt);
    return;
  }
  if (mnemonic == "s_waitcnt_expcnt") {
    out.set(WaitCounter::Export, target.expcnt);
    return;
  }
  if (has_prefix(mnemonic, "s_waitcnt_vscnt") || has_prefix(mnemonic, "s_wait_storecnt")) {
    out.set(WaitCounter::VectorStore, target.vscnt);
    if (mnemonic.ends_with("_dscnt"))
      out.set(WaitCounter::LdsAndGds, target.dscnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_loadcnt")) {
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    if (mnemonic.ends_with("_dscnt"))
      out.set(WaitCounter::LdsAndGds, target.dscnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_dscnt")) {
    out.set(WaitCounter::LdsAndGds, target.dscnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_kmcnt")) {
    out.set(WaitCounter::ScalarMemory, target.kmcnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_tensorcnt")) {
    out.set(WaitCounter::Tensor, target.tensorcnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_asynccnt")) {
    out.set(WaitCounter::Async, target.asynccnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_expcnt")) {
    out.set(WaitCounter::Export, target.expcnt);
    return;
  }
  if (has_prefix(mnemonic, "s_wait_samplecnt") || has_prefix(mnemonic, "s_wait_bvhcnt")) {
    // rocjitsu folds both onto the vector load counter.
    out.set(WaitCounter::VectorLoad, target.vmcnt);
    return;
  }
  // s_wait_xcnt and s_wait_event name things this vocabulary has no counter for
  // and the simulator keeps no threshold for, and anything reaching here from
  // an instruction flag rather than a name has unknown counters. Charging a
  // full drain is the slowest reasonable reading and is therefore the one
  // taken: it can only over-serialise, whereas leaving the thresholds
  // unconstrained makes the instruction free, and a wait that costs nothing is
  // the unbounded optimistic bias the plane exists to refuse.
  set_full_drain(out);
}

/// @brief Translate rocjitsu's wait-counter type into the plane's vocabulary.
///
/// @details Taken from the simulator rather than derived from the class,
/// because which counter an operation posts to is a per-target ISA decision.
/// Vector stores are the case that bites: the compute targets have no separate
/// store counter and post stores to the load counter, so a plane told the store
/// counter would park those completions where no wait instruction on that
/// target can name them, and the wait would cost nothing.
WaitCounter wait_counter_for(amdgpu::WaitCounterType type) {
  switch (type) {
  case amdgpu::WaitCounterType::VMCNT:
  case amdgpu::WaitCounterType::LOADCNT:
  // Image sample loads are vector reads with their own counter on gfx12.
  case amdgpu::WaitCounterType::SAMPLECNT:
    return WaitCounter::VectorLoad;
  case amdgpu::WaitCounterType::VSCNT:
  case amdgpu::WaitCounterType::STORECNT:
    return WaitCounter::VectorStore;
  case amdgpu::WaitCounterType::LGKMCNT:
    return WaitCounter::LgkmCombined;
  case amdgpu::WaitCounterType::DSCNT:
    return WaitCounter::LdsAndGds;
  case amdgpu::WaitCounterType::KMCNT:
    return WaitCounter::ScalarMemory;
  case amdgpu::WaitCounterType::EXPCNT:
    return WaitCounter::Export;
  case amdgpu::WaitCounterType::TENSORCNT:
    return WaitCounter::Tensor;
  case amdgpu::WaitCounterType::ASYNCCNT:
    return WaitCounter::Async;
  }
  return WaitCounter::Count;
}

// -- Memory ------------------------------------------------------------------

/// @brief Whether @p address falls in the wavefront's shared aperture.
///
/// @details Written to agree exactly with the test compute_unit.cpp uses to
/// route the same access, including the base-is-zero guard and the inclusive
/// limit. The two must not drift: the functional path decides where the data
/// went and this decides what it costs, and a disagreement charges local
/// data share latency for a trip to memory or the reverse.
bool in_shared_aperture(const amdgpu::Wavefront &wf, std::uint64_t address) {
  return wf.shared_aperture_base() != 0 && address >= wf.shared_aperture_base() &&
         address <= wf.shared_aperture_limit();
}

/// @brief Fill the memory fields from the state rocjitsu attached while
///        executing the instruction.
/// @returns Whether the instruction carried memory state at all.
bool fill_memory(const Instruction &inst, const amdgpu::Wavefront &wf, CollectedWave &wave,
                 RetiredInstruction &retired) {
  const DynamicInstState *state = inst.data();
  if (state == nullptr)
    return false;

  if (state->tag() == amdgpu::SCALAR_MEM) {
    const auto *scalar = inst.data_as<amdgpu::ScalarMemState>();
    retired.space = MemorySpace::Scalar;
    retired.is_load = scalar->is_load;
    retired.scalar_address = scalar->addr;
    retired.scalar_bytes = scalar->num_dwords * (scalar->elem_size == 0 ? 4u : scalar->elem_size);
    if (retired.scalar_bytes == 0)
      retired.scalar_bytes = 4;
    retired.wait_counter = wait_counter_for(scalar->wait_counter_type);
    return true;
  }

  if (state->tag() != amdgpu::GLOBAL_MEM && state->tag() != amdgpu::LOCAL_MEM)
    return false;

  const auto *vector = inst.data_as<amdgpu::VectorMemState>();
  retired.is_load = vector->is_load;
  retired.non_temporal = vector->non_temporal;
  // System scope: the store has to reach memory, so a write-back cache may not
  // keep it. The decoder folds the scope bits into the memory type.
  retired.write_through = !vector->is_load && vector->mtype == amdgpu::Mtype::UC;
  retired.wait_counter = wait_counter_for(vector->wait_counter_type);
  // A decoder that left the width unset gets four bytes, the same assumption
  // the compute unit's own bounds check makes for the same fields.
  const std::uint32_t element = vector->elem_size == 0 ? 4u : vector->elem_size;
  const std::uint32_t elements = vector->num_elems == 0 ? 1u : vector->num_elems;
  retired.bytes_per_lane = element * elements;

  const std::uint32_t lanes = std::min<std::uint32_t>(vector->wf_size, 64);
  wave.lane_addresses.clear();
  bool any_shared = false;
  bool any_global = false;
  for (std::uint32_t lane = 0; lane < lanes; ++lane) {
    if ((vector->lane_mask & (1ull << lane)) == 0)
      continue;
    const std::uint64_t address = vector->per_lane_addr[lane];
    wave.lane_addresses.push_back(address);
    if (in_shared_aperture(wf, address))
      any_shared = true;
    else
      any_global = true;
  }
  // A mixed access, some lanes in the aperture and some outside, is charged as
  // global. It is the pessimistic reading of the two and also the honest one:
  // the lanes that left the compute unit are the ones that set the latency.
  const bool local = state->tag() == amdgpu::LOCAL_MEM || (any_shared && !any_global);
  retired.space = local ? MemorySpace::LocalDataShare : MemorySpace::Global;
  // An empty list here means no lane participated, which is not the same as an
  // access whose addresses were lost: the plane tells the two apart by
  // addresses_known, and charges the first almost nothing and the second a
  // fully divergent miss.
  retired.lane_addresses = &wave.lane_addresses;
  retired.addresses_known = true;
  // A two-address access moves both halves through the same banks, one after
  // the other. Charging only the first set read ds_write2st64_b64, whose two
  // halves sit 64 rows apart, as half the work it is, and conflict free when
  // the hardware counts it among the most conflicted accesses of an attention
  // kernel.
  retired.second_lane_addresses = nullptr;
  if (local && vector->ds2_active) {
    wave.second_lane_addresses.clear();
    for (std::uint32_t lane = 0; lane < lanes; ++lane)
      if ((vector->lane_mask & (1ull << lane)) != 0)
        wave.second_lane_addresses.push_back(vector->ds2_per_lane_addr[lane]);
    retired.second_lane_addresses = &wave.second_lane_addresses;
  }
  return true;
}

} // namespace

// -- Construction ------------------------------------------------------------

TimingCollector::TimingCollector(TimingPlane &plane) : plane_(plane), enabled_(plane.enabled()) {
  if (!plane.enabled())
    return;
  // ROCJITSU_TIMING_START=<kernel-name prefix>: the plane stays inert -- no
  // dispatch, wavefront, trace line or istream record -- until the first
  // dispatch packet whose kernel name starts with the prefix. While it is
  // inert its clock does not advance, so guest-visible GPU time (event elapsed
  // times, signal timestamps, s_memtime) reads zero.
  if (const char *start = std::getenv("ROCJITSU_TIMING_START"); start != nullptr && *start != '\0') {
    gated_ = true;
    start_prefix_ = start;
    enabled_.store(false);
  }
  sampling_ = plane.sampling().enabled;
}

TimingCollector::~TimingCollector() = default;

void TimingCollector::attach(amdgpu::Wavefront &wf) {
  auto state = std::make_unique<CollectedWave>();
  state->timed = !gated_;
  wf.set_plugin_state(kWavefrontStateSlot, std::move(state));
}

CollectedWave *TimingCollector::wave_state(const amdgpu::Wavefront &wf) const {
  return static_cast<CollectedWave *>(wf.plugin_state(kWavefrontStateSlot));
}

// -- Static instruction properties -------------------------------------------

const StaticInstruction &TimingCollector::static_info(std::uint64_t pc, const Instruction &inst) {
  {
    std::shared_lock<std::shared_mutex> guard(info_mutex_);
    const auto found = info_cache_.find(pc);
    if (found != info_cache_.end() && found->second->mnemonic == inst.mnemonic())
      return *found->second;
  }

  auto entry = std::make_unique<StaticInstruction>();
  entry->mnemonic = std::string(inst.mnemonic());
  entry->inst_class = classify(inst);
  entry->size_bytes = static_cast<std::uint32_t>(inst.size() <= 0 ? 4 : inst.size());
  // The explicit operands only. An implicit use is either a file the scoreboard
  // does not track or an encoded field with no operand behind it, and inventing
  // a range for one would be a dependency the hardware never had.
  for (int index = 0; index < inst.num_src_operands(); ++index)
    append_operand(inst.src_operand(index), entry->reads);
  for (int index = 0; index < inst.num_dst_operands(); ++index)
    append_operand(inst.dst_operand(index), entry->writes);
  entry->denies_sampling =
      sampling_ && plane_.sampling_denies(entry->mnemonic, !entry->writes.empty());

  std::unique_lock<std::shared_mutex> guard(info_mutex_);
  std::unique_ptr<StaticInstruction> &slot = info_cache_[pc];
  // Another wavefront may have derived the same entry concurrently. Keeping
  // whichever landed first is fine, they describe the same instruction, but an
  // entry left by translated code that used to live here must be replaced.
  if (!slot || slot->mnemonic != entry->mnemonic) {
    // The superseded entry is retired, not freed: another wavefront is holding
    // it right now as its pending instruction. Freeing it here is a
    // use-after-free reachable only when translated code reuses an address,
    // which is exactly the case nothing routinely exercises. The list is
    // bounded by how often that reuse happens, not by instruction count.
    if (slot)
      retired_info_.push_back(std::move(slot));
    slot = std::move(entry);
  }
  return *slot;
}

// -- Dispatches --------------------------------------------------------------

void TimingCollector::dispatch_packet(const rocjitsu::KernelDispatchInfo &info) {
  if (!enabled_) {
    if (!gated_)
      return;
    const std::string name =
        info.kernel_name.empty() ? unnamed_kernel_identity(info) : info.kernel_name;
    if (!std::string_view(name).starts_with(start_prefix_)) {
      // Remembered so that its wavefronts stay untimed even when they launch
      // after timing has started.
      std::lock_guard<std::mutex> guard(dispatch_mutex_);
      untimed_.insert(packed(info.dispatch_id, info.queue_id));
      return;
    }
    // The start kernel. Timing begins here, at a dispatch boundary: this
    // dispatch is the first the plane sees, and the plane has not been called
    // before, so nothing in it is half open.
    enabled_.store(true);
  }
  if (gated_) {
    // A dispatch id can come back once its earlier use has retired.
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    untimed_.erase(packed(info.dispatch_id, info.queue_id));
  }

  DispatchShape shape;
  shape.dispatch_id = info.dispatch_id;
  shape.queue_id = info.queue_id;
  // The symbol is only recovered when the code object is findable in a mapped
  // host range. When it is not, every such dispatch would be called "?" and a
  // report full of them cannot be joined to a profile, compared between runs,
  // or even told apart from itself.
  shape.kernel_name = info.kernel_name.empty() ? unnamed_kernel_identity(info) : info.kernel_name;
  shape.workgroup_count = info.workgroup_count;
  shape.waves_per_workgroup = info.wfs_per_workgroup;
  shape.vector_registers_per_wave = info.vgprs_per_wf;
  shape.scalar_registers_per_wave = info.sgprs_per_wf;
  shape.lds_bytes_per_workgroup = info.lds_bytes_per_workgroup;
  shape.wave_size = info.wave_size;

  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    // The completion path carries only the id, so the queue half of the key has
    // to be remembered from here, where it is still available.
    dispatch_queue_[info.dispatch_id] = info.queue_id;
    if (!announced_.insert(packed(info.dispatch_id, info.queue_id)).second) {
      // Already announced. If that announcement was synthesised from a
      // wavefront it carries no kernel name and no workgroup count, and
      // returning here would make that guess final. Forwarding the real shape
      // replaces it.
      if (synthesized_.erase(packed(info.dispatch_id, info.queue_id)) == 0)
        return;
    }
  }

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.dispatch_begin(shape);
}

void TimingCollector::ensure_dispatch_announced(const amdgpu::Wavefront &wf,
                                                std::uint32_t dispatch_id, std::uint32_t queue_id) {
  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    dispatch_queue_[dispatch_id] = queue_id;
    if (!announced_.insert(packed(dispatch_id, queue_id)).second)
      return;
    // Marked as a guess so a real packet arriving later can replace it.
    synthesized_.insert(packed(dispatch_id, queue_id));
  }

  // Everything recoverable from a wavefront and nothing invented. A shape with
  // an unknown name still lands in the right report row; a wavefront with no
  // dispatch at all does not.
  DispatchShape shape;
  shape.dispatch_id = dispatch_id;
  shape.queue_id = queue_id;
  shape.kernel_name = rocjitsu::kUnknownKernelIdentity;
  shape.wave_size = wf.wf_size();
  shape.vector_registers_per_wave = wf.num_vgprs();
  shape.scalar_registers_per_wave = wf.num_sgprs();

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.dispatch_begin(shape);
}

void TimingCollector::dispatch_execution_end(std::uint32_t dispatch_id) {
  if (!enabled_)
    return;

  std::uint32_t queue_id = 0;
  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    const auto found = dispatch_queue_.find(dispatch_id);
    // The completion tracker retires every queue entry it holds, and not all of
    // them are kernel dispatches the collector announced. Delivering an end for
    // one of those would be worse than dropping it: with no queue recovered the
    // key would be (0, id), which on a multi-die part is a real dispatch some
    // other command processor announced, and the plane would close that one
    // instead. Nothing is open for a dispatch that was never announced, so
    // staying quiet leaves nothing dangling.
    if (found == dispatch_queue_.end())
      return;
    queue_id = found->second;
    dispatch_queue_.erase(found);
    announced_.erase(packed(dispatch_id, queue_id));
    synthesized_.erase(packed(dispatch_id, queue_id));
  }

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.dispatch_end(dispatch_id, queue_id);
}

// -- Wavefronts --------------------------------------------------------------

void TimingCollector::wave_dispatched(amdgpu::Wavefront &wf) {
  if (!enabled_) {
    if (gated_)
      mark_untimed(wf);
    return;
  }
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr) {
    attach(wf);
    wave = wave_state(wf);
  }
  if (gated_) {
    bool untimed = false;
    {
      std::lock_guard<std::mutex> guard(dispatch_mutex_);
      untimed = untimed_.count(packed(wf.dispatch_id(), wf.queue_id())) != 0;
    }
    wave->timed = !untimed;
    if (untimed) {
      wave->has_pending = false;
      wave->pending_info = nullptr;
      return;
    }
  }

  // Round-robin by workgroup, which is how a shader-processor input spreads
  // them: every wavefront of a workgroup lands on one compute unit, so a
  // workgroup's barrier stays local to it. The workgroup id is handed over
  // unreduced because the plane folds it into its own compute-unit count, and
  // reducing it here as well would mean two moduli that have to agree.
  wave->compute_unit = wf.wg_id();
  wave->wave_slot = wf.wf_id();
  wave->dispatch_id = wf.dispatch_id();
  wave->queue_id = wf.queue_id();
  // Plugin state survives Wavefront::reset(), so a recycled slot still holds
  // the previous wavefront's pending instruction until it is cleared here.
  wave->has_pending = false;
  wave->pending_info = nullptr;
  wave->user_priority = 0;

  ensure_dispatch_announced(wf, wave->dispatch_id, wave->queue_id);

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.wave_begin(wave->compute_unit, wave->wave_slot, wave->dispatch_id, wave->queue_id);
}

void TimingCollector::wave_halted(amdgpu::Wavefront &wf) {
  if (!enabled_)
    return;
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr || (gated_ && !wave->timed))
    return;

  // The instruction that terminated the wavefront never reaches the after hook:
  // an s_endpgm with no outstanding waits halts inside execute_instruction and
  // the compute unit returns on is_halted() before firing it. Reporting the
  // still-pending instruction here is what stops the plane losing exactly one
  // instruction, the terminator, from every wavefront in the run.
  if (wave->has_pending)
    emit_pending(*wave, wf, /*inst=*/nullptr);

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.wave_end(wave->compute_unit, wave->wave_slot);
}

void TimingCollector::barrier_resolved(std::span<amdgpu::Wavefront *> waves) {
  if (!enabled_ || waves.empty())
    return;

  // Gathered before the plane is locked, and as (unit, slot) pairs rather than
  // wavefronts, because the plane charges a barrier per compute unit: every
  // wavefront on one leaves at the cycle the slowest of them reached.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> members;
  members.reserve(waves.size());
  for (amdgpu::Wavefront *wf : waves) {
    if (wf == nullptr)
      continue;
    if (const CollectedWave *wave = wave_state(*wf); wave != nullptr && (!gated_ || wave->timed))
      members.emplace_back(wave->compute_unit, wave->wave_slot);
  }
  if (members.empty())
    return;
  // Usually one unit: every wavefront of a workgroup shares one. A cluster
  // barrier spans several, so they are grouped rather than assumed.
  std::sort(members.begin(), members.end());

  std::vector<std::uint32_t> slots;
  std::lock_guard<std::mutex> guard(plane_mutex_);
  for (std::size_t index = 0; index < members.size();) {
    const std::uint32_t unit = members[index].first;
    slots.clear();
    while (index < members.size() && members[index].first == unit)
      slots.push_back(members[index++].second);
    plane_.istream_set_wg(unit);
    plane_.compute_unit(unit).barrier(slots);
  }
}

// -- Instructions ------------------------------------------------------------

void TimingCollector::before_execute(std::uint64_t pc, const Instruction &inst,
                                     amdgpu::Wavefront &wf) {
  if (sampling_) [[unlikely]] {
    if (const CollectedWave *sampled = wave_state(wf); sampled != nullptr && sampled->sampled)
      check_sampled(pc, inst, wf);
  }
  if (!enabled_)
    return;
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr || (gated_ && !wave->timed))
    return;

  // A handful of paths retire an instruction without reaching the after hook:
  // an s_trap that enters the trap handler, and an s_trap with no configured
  // handler, both return early. Their instruction is still pending here, and
  // overwriting it would silently drop one instruction per occurrence, the same
  // class of loss as the terminator and just as invisible.
  if (wave->has_pending)
    emit_pending(*wave, wf, /*inst=*/nullptr);

  const StaticInstruction &info = static_info(pc, inst);
  wave->pending_pc = pc;
  wave->pending_info = &info;
  wave->pending_active_lanes = static_cast<std::uint32_t>(std::popcount(wf.exec()));
  wave->has_pending = true;
}

/// @details The program counter is deliberately unused. The compute unit fires
/// this hook *before* applying the common `pc += size` step, so for
/// straight-line code it is still the address the instruction issued at, and
/// only an instruction that wrote the program counter itself reads back
/// different. RetiredInstruction carries no branch-taken flag today; when it
/// does, the test is `pc != wave->pending_pc` and never a comparison against
/// `pending_pc + size_bytes`, which is true of every instruction and would
/// charge the whole kernel a taken-branch penalty per instruction.
void TimingCollector::after_execute(std::uint64_t /*pc*/, const Instruction &inst,
                                    amdgpu::Wavefront &wf) {
  if (!enabled_)
    return;
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr || !wave->has_pending || (gated_ && !wave->timed))
    return;
  emit_pending(*wave, wf, &inst);
}

void TimingCollector::emit_pending(CollectedWave &wave, const amdgpu::Wavefront &wf,
                                   const Instruction *inst) {
  wave.has_pending = false;
  const StaticInstruction *info = wave.pending_info;
  if (info == nullptr)
    return;

  RetiredInstruction retired;
  retired.pc = wave.pending_pc;
  retired.inst_class = info->inst_class;
  retired.mnemonic = &info->mnemonic;
  retired.active_lanes = wave.pending_active_lanes;
  retired.wave_lanes = wf.wf_size();
  retired.reads = info->reads;
  retired.writes = info->writes;
  if (inst != nullptr && info->mnemonic == "s_setprio" && inst->num_src_operands() > 0) {
    if (const Operand *op = inst->src_operand(0)) {
      if (const auto value = op->const_value())
        wave.user_priority = static_cast<std::uint8_t>(*value & 3u);
      else
        wave.user_priority = static_cast<std::uint8_t>(std::strtoul(op->name().c_str(), nullptr, 0) & 3u);
    }
  }
  retired.user_priority = wave.user_priority;
  retired.size_bytes = static_cast<std::uint8_t>(info->size_bytes);
  if (inst != nullptr && info->mnemonic == "s_nop" && inst->num_src_operands() > 0)
    if (const Operand *op = inst->src_operand(0))
      retired.immediate = static_cast<std::uint16_t>(op->const_value().value_or(
          std::strtoul(op->name().c_str(), nullptr, 0)));
  // Filled even for an instruction that is not a wait. The array's default is
  // zero and zero is the strictest threshold there is, so a stray read of it
  // would drain every counter the wavefront has.
  retired.wait.fill(kUnconstrained);

  if (info->inst_class == InstClass::WaitCounter) {
    WaitThresholds thresholds;
    fill_wait(info->mnemonic, wf.wait_target(), thresholds);
    retired.wait = thresholds.values;
  }

  if (inst != nullptr && fill_memory(*inst, wf, wave, retired)) {
    retired.inst_class = refine_with_memory_space(
        retired.inst_class, retired.space == MemorySpace::LocalDataShare, retired.is_load);
  } else if (class_is_memory(retired.inst_class)) {
    // The class says traffic was issued and no state came back to describe it:
    // a tensor transfer, whose pipeline state this target does not attach; an
    // instruction whose decoder never filled one; or the terminal path, where
    // the instruction object is already gone. The access is still declared,
    // with addresses_known false, so the plane charges it as an uncoalesced
    // miss to the farthest level it models. Dropping it instead is the single
    // most tempting and most wrong thing available here: it costs nothing, it
    // removes real traffic, and it leaves no trace in the numbers.
    switch (retired.inst_class) {
    case InstClass::LdsRead:
      retired.space = MemorySpace::LocalDataShare;
      break;
    case InstClass::LdsWrite:
      retired.space = MemorySpace::LocalDataShare;
      retired.is_load = false;
      break;
    case InstClass::ScalarMemory:
      retired.space = MemorySpace::Scalar;
      break;
    case InstClass::TensorMemory:
      retired.space = MemorySpace::Tensor;
      break;
    case InstClass::VectorMemoryWrite:
      retired.space = MemorySpace::Global;
      retired.is_load = false;
      break;
    case InstClass::Export:
      // Exports have no memory space in this vocabulary and no per-lane
      // addresses; the plane costs them from the class and the export port
      // alone, which leaves the bandwidth they consume unmodelled.
      break;
    default:
      retired.space = MemorySpace::Global;
      break;
    }
    retired.addresses_known = false;
  }

  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.instruction(wave.compute_unit, wave.wave_slot, retired);
}

// -- Timing start (ROCJITSU_TIMING_START) ------------------------------------

void TimingCollector::mark_untimed(amdgpu::Wavefront &wf) {
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr) {
    attach(wf);
    wave = wave_state(wf);
  }
  wave->timed = false;
  wave->has_pending = false;
  wave->pending_info = nullptr;
}

// -- Sampled-CU emulation (ROCJITSU_SAMPLED_CUS) -----------------------------

bool TimingCollector::sampling_wanted(std::string_view kernel_name,
                                      std::uint64_t workgroups) const {
  return sampling_ && plane_.samples_dispatch(kernel_name, workgroups);
}

bool TimingCollector::keeps_workgroup(std::uint64_t workgroup, std::uint64_t relative) const {
  return !sampling_ || plane_.keeps_workgroup(workgroup, relative);
}

void TimingCollector::dispatch_sampled(std::uint32_t dispatch_id, std::uint32_t queue_id,
                                       const std::string &kernel_name) {
  if (!sampling_)
    return;
  std::lock_guard<std::mutex> guard(dispatch_mutex_);
  sampled_names_[packed(dispatch_id, queue_id)] = kernel_name;
}

void TimingCollector::wave_sampling(amdgpu::Wavefront &wf, bool sampled) {
  if (!sampling_)
    return;
  CollectedWave *wave = wave_state(wf);
  if (wave == nullptr) {
    attach(wf);
    wave = wave_state(wf);
  }
  wave->sampled = sampled;
}

void TimingCollector::workgroups_skipped(std::uint32_t dispatch_id, std::uint32_t queue_id,
                                         std::uint64_t count) {
  if (!sampling_ || !enabled_ || count == 0)
    return;
  {
    // Only a dispatch the plane has open; one admitted before timing started
    // is not.
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    if (announced_.count(packed(dispatch_id, queue_id)) == 0)
      return;
  }
  std::lock_guard<std::mutex> guard(plane_mutex_);
  plane_.note_skipped(dispatch_id, queue_id, count);
}

void TimingCollector::check_sampled(std::uint64_t pc, const Instruction &inst,
                                    const amdgpu::Wavefront &wf) {
  const StaticInstruction &info = static_info(pc, inst);
  if (!info.denies_sampling)
    return;
  // Skipped workgroups have already been reported complete, so a sampled
  // workgroup that waits for one of them would wait forever, and one that reads
  // what another wrote reads data nobody produced. Neither can be undone here:
  // stop (the default) so the driver re-runs with this kernel excluded.
  std::string kernel = "?";
  {
    std::lock_guard<std::mutex> guard(dispatch_mutex_);
    const auto found = sampled_names_.find(packed(wf.dispatch_id(), wf.queue_id()));
    if (found != sampled_names_.end())
      kernel = found->second;
    if (!unsafe_reported_.insert(kernel).second)
      return;
  }
  const SamplingConfig &config = plane_.sampling();
  std::fprintf(stderr,
               "rocjitsu: ROCJITSU_SAMPLED_CUS: kernel '%s' executes %s (pc %#llx) and cannot be "
               "sampled%s\n",
               kernel.c_str(), info.mnemonic.c_str(), static_cast<unsigned long long>(pc),
               config.exit_on_unsafe ? "; exiting" : "");
  if (!config.deny_log.empty()) {
    if (std::FILE *log = std::fopen(config.deny_log.c_str(), "ae"); log != nullptr) {
      std::fprintf(log, "%s\t%s\n", kernel.c_str(), info.mnemonic.c_str());
      std::fclose(log);
    }
  }
  if (config.exit_on_unsafe) {
    std::fflush(nullptr);
    std::_Exit(config.unsafe_exit_code);
  }
}

} // namespace rocjitsu::timing
