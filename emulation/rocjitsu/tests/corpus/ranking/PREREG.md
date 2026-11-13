# Pre-registered evaluation (written 2026-11-13, before any extended data exists)

## Question

Without a GPU, does the rocjitsu timing model rank alternative implementations
of one operation the way MI355X hardware does, better than signals that need no
timing model?

## Hardware ground truth

- Every hardware measurement runs through `gpu_guard.py`: it starts only when
  no other process has a KFD queue on the target GPU, and is discarded and
  re-measured if a second process appears on that GPU at any 0.2 s sample.
- Three runs, each on one GPU (HIP 5, 6, 7), every pick timed in every run.
  Truth is the per-pick median. Ceiling: one run vs the median of the others.
- The meter corpus reference used for calibration is re-recorded the same way
  (three guarded runs); the fit uses guarded run 1.

## Predictor (fixed before scoring)

- Knobs fitted by `meter_fit.py` on the meter corpus **with every gemm,
  quantized and concurrent case removed**, seeds 1-5, `--maxiter 400
  --popsize 20`.
- Prediction: per pick, the geometric mean over the five fitted tunings of the
  recomposed modelled time (`bench_refit.py`). No tuning sees benchmark data.

## Benchmarks

1. hipBLASLt solutions: the 6 existing bf16 shapes plus 11 new ones (bf16 GEMV
   and decode shapes, one large-K, fp16 and fp32), 16 picks per shape selected
   from TunableOp's log by the existing rule.
2. Triton: tutorial matmul over a grid of BLOCK_M/N/K, num_warps, num_stages,
   and a row softmax over BLOCK/num_warps, a few shapes each.

## Baselines

Random; fewer executed instructions; fewer DRAM bytes (both from the same
emulation, no timing model); for hipBLASLt, the library's `Default` pick.

## Metrics and success criteria

Primary (per benchmark): pairwise ordering accuracy on stable pairs (hardware
median difference > 10% and all runs agree), with bootstrap 95% CI.
**Success:** the CI lower bound of the predictor exceeds the point estimate of
the best non-timing baseline.

Secondary: per-shape pick regret (hardware time of the predicted best / best -
1). **Success:** median regret <= that of hipBLASLt `Default` (hipBLASLt only)
and <= the best non-timing baseline.

Reported regardless of outcome, including per-shape breakdowns and failures.
