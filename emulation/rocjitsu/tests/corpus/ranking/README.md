# Variant-ranking benchmark for the timing model

Does the timing model, run without a GPU, order alternative implementations of
one operation the way MI355X hardware does? Two benchmarks:

- **hipBLASLt**: for a GEMM shape, several hipBLASLt solutions, each forced
  through PyTorch TunableOp's results file (`gemm_solutions.py`).
- **Triton**: a matmul and a row softmax over a grid of launch configurations
  (`triton_bench.py`).

Scores are pairwise ordering accuracy on pairs hardware separates, and the
regret of the predicted-best pick, against baselines that need no timing model
(executed instructions, DRAM bytes, hipBLASLt's own `Default`, random). The
evaluation protocol and its success criteria were fixed before the data in
[PREREG.md](PREREG.md) and addenda [1](PREREG-addendum-1.md),
[2](PREREG-addendum-2.md), [3](PREREG-addendum-3.md), [4](PREREG-addendum-4.md) and
[5](PREREG-addendum-5.md).

Measurements are written under `$RANKING_WORK` (default: the current
directory), never into the tree. `ROCJITSU_BIN` names the emulator
(default `emulation/rocjitsu/build/tools/rocjitsu/rocjitsu`) and
`ROCJITSU_CONFIG` its config (default `configs/gfx950_mi355x_kmd_timing.json`).

## Hardware measurements must own their GPU

`gpu_guard.py` runs a measurement only when no other process has a KFD queue
on the target GPU and marks it contended if another process appears while it
runs; every hardware script retries contended measurements. Note that
`HIP_VISIBLE_DEVICES` numbering follows KFD node order and differs from
`amd-smi`'s PCI order; `gpu_guard.py status` prints the HIP index to gpu_id map.

## Workflow

```sh
export RANKING_WORK=$HOME/ranking
T=emulation/rocjitsu/tests/corpus/ranking
cd $RANKING_WORK
python3 $T/gemm_solutions.py enumerate --workdir tunable --out candidates.json --hip 5   # hardware
python3 $T/select.py                                                                     # 16 picks per shape
python3 $T/time_hw_runs.py $RANKING_WORK/hw 1:5 2:6 3:7                                  # 3 guarded runs
python3 $T/run_triton.py hw $RANKING_WORK/triton_hw 1:2 2:3 3:4
python3 $T/time_emu.py $RANKING_WORK/emu 32                                              # emulator
python3 $T/run_triton.py emu $RANKING_WORK/triton_emu 24
python3 $T/prereg_score.py gemm   --hw hw        --emu emu        --fits "fit-s*.json" --warm-only
python3 $T/prereg_score.py triton --hw triton_hw --emu triton_emu --fits "fit-s*.json" --warm-only
```

`fit-s*.json` are `meter_fit.py` outputs fitted on the meter corpus without its
gemm, quantized and concurrent cases (see `docs/timing-mi355x.md`).

Diagnostics: `counters.py collect|compare` puts the emulator's L1/L2/fabric
traffic next to hardware counters (rocprofv3) for the picks of a group;
`diag_terms.py` decomposes a fitted prediction into its terms per group.

## Results (MI355X, held-out shapes of PREREG-addendum-1)

| | ordering accuracy | best non-timing baseline | median pick regret | comparison |
|---|---|---|---|---|
| hipBLASLt, 8 shapes, 120 solutions | 76.1% [72.7, 79.3] | DRAM bytes 70.5% | 13.4% | hipBLASLt Default 9.4% |
| Triton, 4 shapes, 56 configs | 72.5% [67.3, 77.8] | instructions 75.0% | 3.2% | instructions 34.6% |

Primary criterion met for hipBLASLt, not for Triton; secondary met for Triton,
not for hipBLASLt. Single-row GEMV shapes and Triton matmul are the weak cases.
These scores predate the memory-request routing fix (PREREG-addendum-2.md),
which the model had wrong while they were measured.

## Results (MI355X, held-out set 2 of PREREG-addendum-2)

Model with the routing fix; COMBINED is `model^w * instructions^(1-w)` with w
chosen on the earlier sets only (hipBLASLt 0.9, Triton 0.7).

| | predictor | ordering accuracy | best non-timing baseline | median pick regret | comparison |
|---|---|---|---|---|---|
| hipBLASLt, 8 shapes, 120 solutions | combined | 85.1% [82.3, 87.8] | DRAM bytes 65.8% | 8.9% | hipBLASLt Default 12.7% |
| | model | 81.3% [78.2, 84.3] | | 17.5% | |
| Triton, 4 shapes, 56 configs | combined | 81.2% [76.5, 85.7] | instructions 67.2% | 0.4% | instructions 40.3% |
| | model | 72.0% [66.9, 77.1] | | 1.1% | |

The combined predictor meets both criteria for both libraries. The model alone
meets the accuracy criterion for hipBLASLt (not the regret one) and the regret
criterion for Triton (its accuracy CI lower bound, 66.9%, misses 67.2%).

## Results (MI355X, held-out set 3 of PREREG-addendum-3)

WB is the model with write-back caches and the release flush; V4 is the model
before it, scored on the same items. Both are the model alone.

| | WB accuracy | V4 accuracy | WB - V4 (paired) | best non-timing baseline | WB median regret | comparison |
|---|---|---|---|---|---|---|
| hipBLASLt, 8 shapes, 117 solutions | 83.4% [80.4, 86.2] | 80.5% | +2.9% [+0.3, +5.5] | DRAM bytes 65.9% | 17.8% | hipBLASLt Default 15.5% |
| Triton, 6 shapes, 84 configs | 80.0% [76.3, 83.8] | 67.5% | +12.5% [+7.9, +17.2] | instructions 60.6% | 5.5% | instructions 67.9% |

The accuracy criterion and the paired gain over V4 hold for both libraries;
the regret criterion holds for Triton and not for hipBLASLt, where WB's picks
(17.8%) are worse than Default's (15.5%) and than V4's (7.6%). The Triton gain
comes from matmul (81-84% against V4's 46-60%); on the softmax groups, whose
variants include the store policy, WB (69-89%) is not better than V4 (75-90%).
The DRAM-bytes baseline is taken from each emulation's own traces, and WB's
write-back memory-side cache changes them; against V4's (72.7%) WB still clears
it.

## Results (MI355X, held-out set 4 of PREREG-addendum-4)

R8 is the write-back model with the second calibration stage (`rank_fit.py`);
P7 is the same traces with the corpus-only knobs of addendum 3. Model alone.

| | R8 accuracy | P7 accuracy | R8 - P7 (paired) | best non-timing baseline | R8 median regret | comparison |
|---|---|---|---|---|---|---|
| hipBLASLt, 8 shapes, 119 solutions | 89.0% [86.4, 91.7] | 81.9% | +7.1 [+4.7, +9.7] | DRAM bytes 68.3% | 4.0% | hipBLASLt Default 27.9%, P7 8.1% |
| Triton, 6 shapes, 64 configs | 74.1% [69.1, 78.7] | 73.2% | +0.9 [-1.7, +3.5] | instructions 60.3% | 7.3% | instructions 102.9% |

All three criteria hold for hipBLASLt. For Triton, accuracy and regret hold and
the paired gain over P7 does not.

In the scored run, 20 Triton configs (one softmax group whole) had no clean
hardware measurement. Two of the operator's own measurement processes
overlapped on the same GPUs after a relaunch, and the guard rejected every
attempt. Re-measured afterwards and scored as a labelled supplement, with the
predictions unchanged: all 84 configs, R8 78.7% [74.9, 82.3] against P7 77.2%,
paired +1.6 [-0.7, +3.8], median regret 3.3%. The conclusion is the same.
