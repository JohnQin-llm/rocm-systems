# Measured per-SIMD issue scheduler — design

Status: **prototype**. The scheduler runs as an offline replay over exported instruction streams and
predicts kernel time on the development data. It is not yet the shipped composition. Target
hardware: MI355X (gfx950).

This document describes:

- why the timing model is moving from a fitted closed-form composition to a scheduler whose every
  rule is a measured hardware fact;
- how that scheduler works and how it turns wavefront schedules into kernel time;
- how its parameters are obtained and how it is validated;
- what remains before it can replace `DispatchDes::end()`.

---

## 1. Why change the composition

The shipped model ([timing.md](timing.md)) keeps an independent timeline per wavefront slot and
composes a dispatch as

```text
fixed + max(issue, bandwidth, placement) + latency(exposed × rounds × latency_scale) + filling + flush
```

with about twenty knobs fitted by differential evolution against measured kernel times
([timing-variant-ranking.md](timing-variant-ranking.md)). Two structural problems showed up on the
development data:

1. **Throughput and latency are summed.** The slowest wavefront's own issue slots are counted in
   both terms. Kernels whose latency term is comparable to their throughput term are over-predicted
   by 16–45%. That covers most hipBLASLt GEMMs, Triton matmuls with software pipelining and
   long-loop attention. The error grows with loop length: on aiter flash attention it goes from 0 at
   S = 512 to +37% at S = 4096.
2. **Fitted knobs absorb structure, not physics.** Values such as `fixed_scale ≈ 0.03`,
   `unit.scalar_memory ≈ 0.002`, `level.mall ≈ 8.6e3` or `latency_scale ≈ 0.55` are not properties of
   the GPU. They make the fit work on the families it saw and give no reason to expect them to hold on
   the next one.

The alternative is to model what the hardware does — wavefronts competing for issue slots and
execution pipes, cycle by cycle — and to obtain each rule from a targeted measurement rather than from
end-to-end kernel time.

## 2. Model

### 2.1 Machine

Per compute unit (CU):

| Resource | Scope | Notes |
| --- | --- | --- |
| Instruction arbiter | per SIMD (4 per CU) | Each cycle each pipe may accept one instruction. Among ready wavefronts the highest `s_setprio` wins, then the oldest. |
| VALU pipe | per SIMD | Per-mnemonic occupancy of 2, 4 or 8 cycles. Transcendentals share this pipe. |
| MFMA pipe | per SIMD | Occupancy is the number of passes of the shape. Each MFMA also holds the VALU pipe at issue. |
| SALU | per SIMD | 4 cycles |
| `v_cndmask_b32` (VOP2 form) | per CU | A CU-shared resource, ≈4.4 cycles per instruction. |
| LDS | per CU | Occupancy from the bank rule (§2.2). |
| Vector memory address unit | per CU | Occupancy from accesses per instruction (§2.2). |
| Scalar memory | per CU | — |

Per wavefront:

- in-order issue, at most one instruction per *issue interval* (4 cycles; 4.6 for 8-byte encodings);
- a register scoreboard (VGPR / SGPR / AGPR): an instruction issues when its sources are ready;
- wait counters: `s_waitcnt` issues and then holds the wavefront until the named counters drain;
- wave-local holds for instructions that occupy the wavefront itself (branches, LDS writes,
  vector-memory issue, `s_nop`);
- workgroup barriers, which release 4 cycles after the last live member arrives.

Wavefronts of a workgroup are placed on SIMDs round-robin, and workgroups launch when enough wave
slots are free. Workgroup *i* runs on XCD *i* mod 8. Within an XCD, workgroups go round-robin over
shader engines and then over CUs, so *i* and *i* + 256 share a CU.

### 2.2 Rules (MI355X, measured)

All values come from microbenchmarks unless marked *trace*. The microbenchmarks use a single
workgroup of 1, 2 or 4 wavefronts per SIMD, `s_memtime` cycle counts, and `HW_ID` to verify SIMD
placement. *Trace* means a hardware thread trace of real kernels.

**Issue and VALU**

| Rule | Value |
| --- | --- |
| Wave issue interval | 4 cycles. A wave issuing only 8-byte encodings (VOP3/VOP3P/SDWA/DPP/literal) runs at ≈4.6; mixing in at least half 4-byte instructions hides this. |
| VALU occupancy 2 | `v_add/mul/fma/fmac_f32`, `v_mov_b32`, `v_add/sub_u32`, `v_and/or/xor_b32`, `v_lshrrev_b32` |
| VALU occupancy 4 | `v_pk_*`, `v_max3`, `v_cvt_pk_bf16_f32`, `v_cndmask`, `v_cmp`, `v_mov_b64`, `v_lshlrev`, `v_max_f32`, `v_div_*`, DPP |
| VALU occupancy 8 | `v_exp/rcp/log/sqrt/rsq_f32`; f32→fp8/bf8 conversions (packed, stochastic-rounding, scaled) and stochastic-rounding bf16; `v_permlane16/32_swap` |
| SGPR / VCC operand | Any VALU that reads or writes an SGPR or VCC loses double rate (occupancy 2 → 4). A literal does not. |
| `v_cndmask_b32` VOP2 (e32/SDWA/DPP) | Goes through a CU-shared resource, ≈4.4 cycles per instruction CU-wide (5.5 for a single wave). When saturated, it also slows plain VALU on the other SIMDs to ≈16. The e64 form is normal. |
| Dependent latency | Never longer than the producer's own occupancy, independent of distance. This holds for VALU→VALU, MFMA srcC/srcA chains, MFMA→VALU, VALU→MFMA and transcendental chains. |
| Arbitration | Strict oldest-first; `s_setprio` overrides. |
| `s_nop N` | 4·(N+1) cycles |

**MFMA**

| Rule | Value |
| --- | --- |
| MFMA occupancy | 32×32×16 bf16/f16/i8: 32; 16×16×32: 16; 32×32×64 f8f6f4: 64; 16×16×128 f8f6f4: 32; f32 32×32×2: 64, 16×16×4: 32; f64 16×16×4: 64, 4×4×4: 16 |
| MFMA hold on VALU pipe | 8 cycles. For f32 and f64 inputs it is occupancy + 8. (gfx950 has no xf32 MFMA.) |

**Control flow**

| Rule | Value |
| --- | --- |
| Taken branch | ≈22 cycles for a near target, ≈46 when the target is a new fetch line, ≈350 on an instruction-cache miss |
| Not-taken branch | ≈8 cycles (≈11 right after `s_cmp`) |
| `s_barrier` | 4 + 4 × wavefronts per SIMD in lockstep; release 4 cycles after the last arrival |

**LDS**

| Rule | Value |
| --- | --- |
| LDS bank cost | Let *m* be the largest number of **distinct** 4-byte words that any one of the 64 banks serves for one instruction. Lanes reading or writing the same word share a pass, so a broadcast is free. Reads cost max(2, 2*m*) for 4-byte lanes and max(2, *m*) for wider lanes; `b128` reads cost at least 4. Writes cost max(4 / 6 / 13, 2*m*) for b32 / b64 / b128. The cap is 64. |
| LDS wave hold | Writes hold ≈ bytes/40 (*trace*: 1 KiB writes cost 25–31 cycles in three kernels); b32 ≈ 8. `b128` reads hold ≈ 8. Latency ≈ 52–60. |

**Vector memory**

| Rule | Value |
| --- | --- |
| Vector-memory address unit | About one access per cycle per CU. A dense global `dwordx4` takes 16 cycles (64 B/clk). A scratch `dwordx4` takes 64, x2 takes 32 and x1 takes 4, because the private segment is dword-swizzled. Spill-heavy kernels are therefore limited by this unit. |
| Vector-memory wave hold | ≈ bytes/85 for loads (≈12 for `x4`), ≈ bytes/51 for stores |
| `vmcnt` order | Retires in program order across loads and stores. A wait after a store completes no earlier than the store's acknowledgement (≈290–300 cycles idle), even if the loads behind it hit L1. |
| Load after an in-flight store to the same address | No extra stall: the load returns the stored data without waiting for the acknowledgement. A store that writes a whole 64/128-byte unit keeps the L1 line valid. A partial-line store invalidates it, so the next load reads from L2. |

**Caches and synchronisation**

| Rule | Value |
| --- | --- |
| Cache retention across kernels | In back-to-back launches L2 and the memory-side cache (MALL) keep their contents. A pointer chase costs ≈895 cycles per hop cold, ≈215 when the next launch re-chases (L2 hit) and ≈540 from MALL. Under counter collection, L2 is emptied before every dispatch while MALL is kept. |
| Cross-workgroup flag | A scalar store with `glc` is seen by a polling `s_load … glc` ≈1.3–1.5k cycles (≈550 ns) later. Each poll goes past L2: ≈400 or ≈700 cycles depending on the flag's address, with little difference between same-XCD and cross-XCD. Without `glc` the flag may never be seen. |

**Clock**

| Rule | Value |
| --- | --- |
| Sustained clock | 2383–2400 MHz on 91 of the 95 development kernels. The clock drops only when board power exceeds ≈1270 W: three dense fp8 GEMMs ran at 1922–2260 MHz and one Triton matmul at 2142 MHz. |

These measurements imply the following corrections to the shipped tuning:

- VALU result latency goes from 9–13 cycles to at most the producer's occupancy.
- Transcendental cost goes from 12 cycles on its own pipe to 8 on the VALU pipe.
- MFMA latency goes from a fixed 44 to the MFMA's own occupancy.
- `s_nop` goes from 1 to 4·(N+1).
- Branches, which were not charged, now cost as in the table above.
- The LDS bank rule above replaces the earlier one.
- L2 is no longer invalidated between dispatches.

### 2.3 Kernel time

```text
kernel time = launch floor + (slowest sampled CU's completion + release write-back) / clock
```

- **Launch floor: 1.8 µs.** This is a near-empty kernel's duration minus its workgroup time, plus the
  XCD start skew (0.3–1.1 µs). Dispatch is capped at ≈1.56 workgroups per cycle.
- **Release write-back: min(bytes written, 14 MiB) / 2400 B/clk.** Kernels that write take 2–2.6 µs
  longer than kernels that only read, and this saturates near 14 MiB of dirty L2.
- **Memory bandwidth.**
  - Device-level bandwidth is shared among the active CUs.
  - Each CU has one queue for reads and writes, served at most at its 64 B/clk local link.
  - Bandwidth is scaled by the device utilisation measured in a first pass, so a CU may burst
    faster while the device is not saturated.
  - Device read rates: ≈10000 B/clk for L2 hits, ≈2700 B/clk beyond L2. Device write rates: ≈5200
    for L2 hits, ≈2400 beyond L2.

### 2.4 What is not modelled yet

- **Clock under load as a function of power.** It matters only for dense fp8 GEMMs and one matmul in
  the development data.
- **Co-resident workgroups sharing misses.** The emulator executes workgroups in order, so the first
  workgroup to touch a panel takes every compulsory miss. On hardware about 32 co-resident workgroups
  per XCD share them. Total bytes beyond L2 match hardware counters to about 1%; only the split
  across workgroups is skewed. The replay currently corrects this by rescaling each instruction's
  beyond-L2 share to the dispatch-wide total.
- **Bandwidth in the last round of a grid that only partly fills the CUs.**
- **Instruction fetch** beyond the branch rule.

## 3. Implementation

### 3.1 Instruction-stream export (diagnostic)

`ROCJITSU_TIMING_ISTREAM=<prefix>` makes the timing plane write `<prefix>.<pid>`, a stream of fixed
96-byte records (`IStreamRecord`, `vm/timing/timing_plane.h`) for a sample of folded CUs. The sample
is chosen by `ROCJITSU_TIMING_ISTREAM_UNITS` (how many) and `ROCJITSU_TIMING_ISTREAM_STRIDE` (unit
*u* is exported when *u* mod stride = 0). Kernels whose work is uneven across workgroups, such as
causal attention, need every CU sampled: their kernel time is set by the heaviest workgroup.

| Type | Contents |
| --- | --- |
| 0 instruction | class, unit, wait counter, slot, unfolded workgroup, port occupancy, result latency as the plane computed it, wait thresholds, register reads/writes, `s_setprio` priority, fetch stall, cache lines |
| 1 / 2 | wavefront begin / end |
| 3 | barrier member |
| 4 / 5 | dispatch acquire / end (aligns with `ROCJITSU_TIMING_TRACE` lines of the same pid) |
| 6 | first sighting of a pc: mnemonic, encoding size, `s_nop` immediate |
| 7 | file header (version) |

The export is off by default and changes no timing.

### 3.2 Emulator changes that come with it

- `l2.keep_across_dispatches` (default off; on in `gfx950_mi355x_measured.json`): do not invalidate
  L2 at dispatch acquire.
- `lds.measured_banks` (default off; on in `gfx950_mi355x_measured.json`): the LDS bank rule of §2.2.
  With it on, the per-pass issue floor no longer applies to LDS, because the rule already prices the
  whole wavefront.
- CDNA4 kernarg preload: when the kernel descriptor enables preload, AQL dispatch enters 256 bytes
  past the descriptor's entry. That skips the compatibility prologue the firmware jumps over; a
  hardware thread trace shows the first executed instruction at entry + 0x100.
  Test: `AmdgpuVmTest.Cdna4KernargPreloadEntersPastFirmwareWindow`.

### 3.3 Replay prototype

An offline replay (Python, kept with the measurement scripts, not in the repo) reads the export and
runs the scheduler of §2 on each sampled CU. It reports two things:

- wavefront lifetimes and the slowest CU's completion;
- for every cycle a wavefront does not issue, one cause: issue interval, register dependency (by
  producer class), wait counter, pipe, barrier or arbitration.

This keeps the rules cheap to change while they are being established.

One emulator artefact needs special handling. A spin loop waiting on another workgroup's flag (the
stream-K fix-up in hipBLASLt) is recorded with as many iterations as the emulator's own timing
produced, which is 34–213. Hardware runs 4–8. The replay therefore collapses such a loop to its last
iteration and holds the wavefront until the partner's flag, plus one poll.

### 3.4 Target C++ design

`ComputeUnitDes` becomes an event-driven scheduler over the CU's resident wavefronts:

- the functional plane still delivers instructions at retire; the timing plane buffers each
  wavefront's stream and advances the CU's scheduler to the earliest cycle any wavefront can issue;
- per-SIMD pipes and per-CU ports carry `free_at` cycles; the arbiter picks by priority, then age;
- memory requests are issued to the plane at the scheduled cycle, so latency reflects contention
  among wavefronts on the same clock;
- each (workgroup, slot) gets its own wavefront state, so workgroups folded onto one CU do not
  share a timeline;
- `DispatchDes::end()` takes the dispatch's duration from the scheduled CUs and the kernel-time
  terms of §2.3 instead of composing terms.

The rule tables of §2.2 become tuning data in the config, with each entry citing its measurement. The
shipped composition stays selectable.

## 4. How parameters are obtained

1. **Microbenchmarks.**
   - Setup: one workgroup of 1, 4, 8 or 16 wavefronts, an inline-assembly body repeated in a loop,
     `s_memtime` per wavefront, `HW_ID` for SIMD placement.
   - Per-SIMD occupancy is the slowest wavefront's cycles over all instructions issued on that SIMD.
   - CU-shared resources show up as per-wavefront cost growing with the number of SIMDs that run the
     instruction.
   - Each test runs on one GPU under the measurement guard.
2. **Hardware thread trace** (rocprofv3 `--att`) on real kernels, for costs that depend on access
   patterns: LDS writes, memory issue, branch-target fetch, spin-loop iteration counts.
   - Traces must be of a *warm* iteration (`--kernel-iteration-range`).
   - Trace timestamps are only comparable within one shader engine.
   - Writing the trace disturbs the memory system, so traces are used for per-wavefront and
     per-instruction shape, not for kernel time.
3. **Performance counters** for aggregate checks only: instruction counts, bytes beyond L2, average
   memory latency.
   - Counter collection runs every dispatch with an empty L2, so its cache counts describe a cold
     dispatch.
   - Counter-collection runs are not used as a timing reference.
4. **Clock and power:** sampled during a back-to-back loop of each kernel.

Nothing in §2.2 or §2.3 is fitted to end-to-end kernel time.

## 5. Validation

Development data only:

- **Mechanism level:** per-instruction comparison of replay against warm hardware thread traces,
  aligned by program order. Each mnemonic's issue-to-next-issue gap is compared, which shows which
  rule is wrong.
- **Kernel level:** predicted kernel time against `kernel_us` measured with the profiler's kernel
  timing (no tracing, no counters), back to back with warm caches.

The scheduler then goes through the existing protocol
([timing-variant-ranking.md](timing-variant-ranking.md)): development → pre-registration → one scoring
of a new held-out set.

## 6. Results so far (development data)

The development sample has 85 single-kernel items. Errors are median |predicted / measured − 1|.
Nothing in the scheduler is fitted. The shipped composition's numbers are in-sample, because its
knobs were fitted on this data.

| Family | n | Scheduler: median ratio | median error | Shipped composition: median error |
| --- | --- | --- | --- | --- |
| Triton softmax | 10 | 1.03 | 5.0% | 16.5% |
| RMSNorm | 6 | 1.04 | 5.3% | 10.3% |
| Triton flash attention | 6 | 0.92 | 8.4% | 18.2% |
| hipBLASLt GEMM | 20 | 0.99 | 10.1% | 16.1% |
| attention | 10 | 0.90 | 10.5% | 14.9% |
| Triton matmul | 15 | 1.04 | 11.8% | 18.0% |
| swiglu | 4 | 0.88 | 14.2% | 23.6% |
| LayerNorm | 4 | 1.01 | 18.5% | 15.3% |
| fp8 matmul | 10 | 0.83 | 27.3% | 11.2% |
| **all** | **85** | 0.96 | **11.1%** (79% within 25%) | 13.9% (69% within 25%) |

The table includes the stream-K spin handling of §3.3. Causal attention is sampled on every CU.
These are the remaining known errors:

| Family | Known error and status |
| --- | --- |
| fp8 matmul | Three items run power-throttled on hardware (§2.2, clock). Two items with small tiles are over-predicted: the hardware drives ≈2200 B/clk beyond L2, while the replay's bandwidth share for the last partial round is too low. |
| Spill-heavy attention | Needs the scratch address-unit cost and the in-order `vmcnt` store acknowledgement. Both are measured (§2.2) and being evaluated. |
| Small Triton matmuls | Need the beyond-L2 share corrected for co-resident workgroups (§2.4); being evaluated. |
| Multi-kernel hipBLASLt calls | Not yet included: only the main kernel is modelled. |

## 7. Plan

1. Finish the replay rules above (scratch address unit, store acknowledgement, beyond-L2 share,
   partial last round, clock under power) and re-evaluate the development sample.
2. Port the remaining replay rules into the C++ scheduler (§3.4), and model multi-kernel library
   calls.
3. Pre-register and score a new held-out set.
