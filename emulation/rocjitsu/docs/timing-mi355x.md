# Timing model on MI355X: bring-up notes

Notes from running the timing plane (`docs/timing.md`) against a real MI355X
(gfx950), calibrating it, and scoring it on *ranking* -- whether it orders
alternative implementations of the same operation the way hardware does. That
is the question an autotuner or a kernel-writing agent asks of a model.

Measured reference runs, fitted tunings and traces are **not** in the
repository, for the same reason `docs/timing.md` gives for calibrated values.

## Workflow

Everything follows `tests/corpus/README.md`; the additions are in **bold**.

1. Record the hardware reference on one MI355X GPU:
   `python3 tests/corpus/rocm-meter.py --tier standard --samples 5 --warmups 5 --output real-gfx950.json`
   (308 cases, about four minutes).
2. Build rocjitsu (`cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`).
3. Run the corpus under the emulator, uncalibrated, with per-dispatch traces:
   `METER_TRACE_DIR=traces tests/corpus/run-meter-emulated.sh --out emu --tuning ""`.
   The gemm shards exceed the default 1800 s shard timeout; run gemm one case
   per process instead (`--kernel gemm.<impl>.<dtype>.<shape>`), with the same
   `ROCJITSU_TIMING_TRACE` / `ROCM_METER_DEVICE_WINDOW` environment the script
   sets.
4. **Score ranking**: `python3 tests/corpus/meter_rank.py --real real-gfx950.json --emulated emu/*.json`.
   Variants are cases whose `case_id` differs only in the implementation
   segment (`eager` / `compiled` / `triton`); pairs closer than `--noise-pct`
   on hardware are ignored.
5. **Fit the composition knobs**:
   `python3 tests/corpus/meter_fit.py --shards emu --traces traces --real real-gfx950.json --config configs/gfx950_mi355x_kmd.json --out fit.json`.
   Differential evolution over `meter_refit.py`'s knobs in log space, on a
   per-category-averaged Huber loss of `log2(model / measured)`. Train and test
   are split by *variant group*, so the eager and compiled versions of a case
   land on the same side and every scored ranking pair in the test set is
   unseen. One fit of 23 knobs is about 184k recompositions and takes a few
   minutes on 32 cores.

## Results (gfx950, standard tier, gemm not yet included)

266 cases joined to traces.

| | median model / measured | within 2x | within 20% | variant pairs ordered correctly |
|---|---|---|---|---|
| uncalibrated | 386x slow | 0% | 0% | 82/96 = 85.4% |
| fitted, held-out test, 5 random splits pooled | x1.19 | 393/397 = 99% | 218/397 = 55% | 122/141 = 86.5% |

Calibration fixes the absolute scale -- the uncalibrated run is slow by design,
since every unnamed parameter resolves to its slowest value -- but barely moves
ranking. The knobs `meter_refit.py` exposes are global multipliers; they cannot
change which of two implementations of one case the model prefers. The
remaining mis-ordered pairs are structural:

- small fused (compiled) kernels predicted slower than the several eager
  kernels they replace (fused_elementwise and attention_softmax at 4096 / 128x128);
- strided copy, and compiled vs eager `atan` at 8M elements.

Each test split holds only about 30 pairs, so the ranking figure carries
roughly +-5 points of noise.

## Stream-K hang: `gemm.*.1x4096x4096`

All six `1x4096x4096` gemm cases spun for the full three-hour timeout. The
kernel is hipBLASLt's Stream-K GEMM
(`Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x16x128_..._SK3_SKFTR0_SKXCCM4_...`
for float32; the half-precision cases reach it through their float32
reference): 256 workgroups share 128 tiles, and a tile's finishing workgroup
polls a per-tile flag that its partner sets:

```
label_SK_Partials:                       ; producer
  buffer_store_dwordx4 ... sc0 sc1       ; partial tile
  s_waitcnt vmcnt(0) ; s_barrier
  s_store_dword s58(=1), s[34:35], s8 glc     ; flag[tile] = 1
label_SK_Fixup:                          ; consumer
  s_load_dword s75, s[34:35], s73 glc         ; poll flag[tile]
  s_waitcnt lgkmcnt(0)
  s_cmp_eq_u32 s75, 1
  s_cbranch_scc0 label_SK_Fixup               ; no s_sleep
  ...
  s_store_dword s75(=0), s[34:35], s73 glc    ; reset flag[tile]
```

This branch forked from develop on 2026-08-25 and is missing two scalar-memory
fixes that landed afterwards. Both are needed; either one alone still hangs.

| | Bug | Effect | Upstream fix |
|---|---|---|---|
| 1 | SMEM with IMM=0 and SOE=0 takes its offset from the SGPR named in the OFFSET field (`C00112D1 00000049` is `s[34:35] + s73`); the address calculation only read an SGPR offset when SOE=1 | every tile used `flag[0]`; a consumer's reset wiped a partner's set, and waiters missed their wakeup | 5cadd6a39c, #10250 (2026-08-27) |
| 2 | the scalar memory pipeline decoded GLC into `ScalarMemState::mtype` but never passed it to the K$, which only looked at the page MTYPE (RW) | a poll that first read 0 kept hitting the stale K$ line | 6515e70905, #12565 (2026-10-07) |

Bug 1 was found by logging every GLC scalar load and store: all 84 flag stores
of the hung dispatch went to one address. A smaller Stream-K grid
(`TENSILE_STREAMK_FIXED_GRID=128`, one workgroup per tile, no fixup) never
hung, and neither did the other gemm shapes, whose chosen kernels also assign
whole tiles.

The backport is minimal: `effective_mtype(instruction, page)` now decides K$
bypass in `L1ScalarCache::load/load_bytes/store`, and `smem_calculate_address`
reads `offset & 0x7f` as the SGPR when IMM=0. With it, the 3393 unit tests pass,
`tests/corpus/streamk_gemv_repro.py` finishes in about 50 s for float32, float16
and bfloat16 (the float32 max error matches the hardware run bit for bit), and
the six corpus cases finish in one to two minutes each.

## Open items

- Re-run the whole corpus with the fixed emulator. Bug 1 affected every kernel
  that uses the IMM=0 SMEM form, so traces recorded before the fix may carry
  wrong addresses even where the computed results passed the meter's
  correctness check.
- Add the 24 gemm cases to the fit, so the matrix-unit knobs are constrained.
- Work through the mis-ordered pairs; ranking will only move with changes to
  the model's structure, not to its tuning.
