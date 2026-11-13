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
[PREREG.md](PREREG.md) and [PREREG-addendum-1.md](PREREG-addendum-1.md).

Measurements are written under `$RANKING_WORK` (default: the current
directory), never into the tree. `ROCJITSU_BIN` names the emulator
(default `emulation/rocjitsu/build/tools/rocjitsu/rocjitsu`).

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
