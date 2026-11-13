# Pre-registration addendum 4 (written 2026-11-13, before any held-out-4 data exists)

## What changed after addendum 3 was scored

The diagnosis of held-out set 3, now development data, found the following:

- The hipBLASLt gain over V4, and WB's worse pick regret, followed the knob set, not the write-back traces.
- The knobs that decide which variant wins are not identified by the meter-corpus fit, which excludes its GEMM cases. Per-seed values spread 20x to 2000x (the LDS and matrix-unit weights, the stall/latency split, the L2 rate).

`rank_fit.py` adds a second calibration stage:

- It starts from each corpus fit and keeps every knob but nine.
- It refits those nine on every development ranking group: PREREG.md, held-out sets 1-3, 33+8 hipBLASLt shapes and 15+6 Triton cases. That is 907 items and 4713 stable pairs.
- The loss is logistic on stable pairs, with every group weighing the same, plus a pull toward the corpus fit's values.

Cross-validation by group (5 folds, test groups never fitted):

| | pairwise accuracy | median of fold median regrets |
|---|---|---|
| hipBLASLt | 84.8% -> 88.8% | 15.1% -> 5.3% |
| Triton | 78.7% -> 80.0% | 9.1% -> 7.9% |

On the meter corpus, the median absolute error of the five fits moves from 18.4-21.1% to 18.4-24.0% (non-GEMM cases).

## Predictors (fixed now)

Both predictors run the emulator at commit 121b699b1a with config `config_b1_wb.json` (sha256 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b). Runs use one thread and no address randomisation. Each prediction is the 5-fit geometric-mean ensemble of the model alone over warm calls (`--warm-only`).

- **R8 (under test):** knobs `prereg8-fit-s1..5.json`, the second stage on the PREREG-addendum-3 knobs.
- **P7 (reference):** knobs `prereg7c-fit-s1..5.json`, the WB predictor of addendum 3 (hashes there).

sha256 of the R8 knob files:

| file | sha256 |
|---|---|
| prereg8-fit-s1.json | 35761665a773b46469c59ce8f6f6caa512de0fb487e970f88bb7304a4d2545a5 |
| prereg8-fit-s2.json | 900749b2a0f9000505efb6c046d95c5e05ac35c462ca7258da0d687c8d900164 |
| prereg8-fit-s3.json | 8b703dc77e1f13e433abb1f72f3d8a854073e4cd66e32f1e5f1e98b4478b7a19 |
| prereg8-fit-s4.json | bc744aa7a794089ca1f8e7228eed25cfe887d169a28cf47e3149a4cbbc8c5c29 |
| prereg8-fit-s5.json | 134ced1513aa9d2fe6495c0a250dddc1705790b0f31819c0b3d2b399f7284578 |

## Held-out set 4 (never measured before this document)

- **hipBLASLt** (bf16 unless noted): m6_n4096_k6144, m12_n3072_k4096, m48_n5120_k1536, m112_n2048_k3072, m224_n1024_k2048, m448_n1792_k1024, m1_n6144_k2048_f16, m192_n2304_k2304_f16. Candidates come from TunableOp enumeration; picks follow the select.py rule.
- **Triton matmul:** mm320x2560x1280, mm896x896x1792, mm2048x256x1536. Variants follow the seeded rule.
- **Triton softmax:** sm2560x1792, sm12288x640, sm768x5120. Variants follow the addendum-3 rule: num_warps x rows-per-program x store policy.
- **Hardware truth** is taken exactly as in PREREG.md: gpu_guard, 3 runs, on HIP 5,6,7 for hipBLASLt and HIP 2,3,4 for Triton.

## Criteria

These apply to held-out set 4 only, to each library separately, scored once. Stable pairs, the bootstrap and regret are computed as in PREREG.md.

- **H1:** R8 pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline's point accuracy.
- **H2:** R8 median pick regret <= the best non-timing baseline's and, for hipBLASLt, hipBLASLt Default's.
- **H3:** R8 minus P7 pairwise accuracy, paired bootstrap 95% CI lower bound > 0.

Reported but not hypotheses: P7's H1/H2 numbers. R8's regret against P7's is the quantity the second stage was built to move.
