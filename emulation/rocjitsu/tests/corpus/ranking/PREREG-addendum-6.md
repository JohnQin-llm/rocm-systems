# Pre-registration addendum 6 (written 2026-11-13, before any held-out-6 data exists)

## What changed after addendum 5 was scored

Held-out set 5 is now development data. Its hipBLASLt median pick regret was 14.2% against Default's 11.2%, after 4.0% against 27.9% on held-out set 4. A median over 8 shapes is noisy. Pooled over every group held out in a 5-fold cross-validation by group (57 hipBLASLt shapes, 54 with Default), the current second stage (pairwise loss only, plain pick) has median 5.9% and mean 10.5% pick regret against Default's 14.8% and 15.6%, and is at least as good as Default on 67% of shapes.

Two changes aim at the pick itself:

- **Listwise term.** `rank_fit.py --listwise` adds the cross-entropy between the hardware's and the model's "which one is fastest" softmax over each group.
- **Deferring rule.** Take the model's pick only when it predicts it more than delta faster than hipBLASLt's Default; otherwise keep Default.

Cross-validated the same way, on the grid lambda in {0, 0.5, 2, 8, 32} x delta in {0, 3, 5, 10, 20}%, the lowest pooled mean regret is lambda = 8 with delta = 3%: median 3.0%, mean 7.7%, and at least as good as Default on 76% of shapes. That setting is fixed here.

## Predictors (fixed now)

Both predictors run the emulator at commit 121b699b1a with config `config_b1_wb.json` (sha256 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b). Runs use one thread and no address randomisation. Each prediction is the 5-fit geometric-mean ensemble of the model alone over warm calls.

Both start from the corpus knobs `prereg7c` and run the second stage on PREREG.md and held-out sets 1-5 (1334 items, 6802 stable pairs).

- **R10 (under test):** knobs `prereg10-fit-s1..5.json` (listwise weight 8). The pick follows the deferring rule with delta = 3% (`prereg_score.py --defer 0.03`).
- **R8c (reference):** knobs `prereg8c-fit-s1..5.json` (listwise weight 0, the addendum-4 recipe). The pick is the plain argmin.

sha256 of the knob files:

| file | sha256 |
|---|---|
| prereg10-fit-s1.json | 433dd4c0bd8e41c743bd3876048c261671b2af591d8ec8b4a8643059b21a1480 |
| prereg10-fit-s2.json | a07f470b26c3ccb4971a6bd324fa6df3ffb693b376d60fec77774022ea1473fc |
| prereg10-fit-s3.json | 3a38e8040758708e501ef348b682a612dec01bcb6ddcc46849f16227786354d4 |
| prereg10-fit-s4.json | 9874513547988d68f3af15dbc43e7a763ecab24eaab43622e3bdf9cdb0007243 |
| prereg10-fit-s5.json | c1f80d2f47553e34e27db1c2d2177fa8264372e0ab50c938bb7dcf388d76bd3c |
| prereg8c-fit-s1.json | a1bd63d841dfa07511cc2c8b9baf35c83eabe97e305f86f8f3f27f4481a6a98f |
| prereg8c-fit-s2.json | da53384a2c532f83ac4bccd17bdb5631c8407a2caaf9e550af5725c6edd6e4ca |
| prereg8c-fit-s3.json | 8a07f1df2420544346301020c39f2de8a8e85157ef1fb092717a305ca98924f9 |
| prereg8c-fit-s4.json | 69f57a58c73394823e19d03a8a45cc47de09a83ab441c1d1083c97a932995351 |
| prereg8c-fit-s5.json | 29ddcd2a869bf3301dc170ea9b26da9276cff60c67559fad04e59d73386ace41 |

## Held-out set 6 (never measured before this document)

hipBLASLt only, 12 shapes so that the regret tests rest on more than 8 groups (bf16 unless noted):

- m9_n4608_k4608, m16_n3072_k6144, m36_n6656_k1536, m72_n2560_k3072
- m144_n4096_k1024, m208_n1536_k2560, m384_n2048_k1536, m768_n1280_k1024
- m4_n7168_k3584_f16, m24_n2048_k8192_f16, m160_n3584_k1792_f16, m1024_n768_k2048_f16

Picks follow the select.py rule. Hardware truth is taken as in PREREG.md: gpu_guard, 3 runs on HIP 5,6,7, one measurement process at a time.

## Criteria

These are scored once. Regret is the hardware time of the pick over the group's best, minus 1. The tests are paired bootstraps over groups (4000 resamples) of the mean per-group regret difference, on the groups where Default has a clean measurement.

- **H1:** R10's deferring pick minus Default, mean regret: 95% CI upper bound < 0.
- **H2:** R10's deferring pick minus R8c's plain pick, mean regret: 95% CI upper bound < 0.
- **H3:** R10 pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline's point accuracy.

Reported, not hypotheses: median regrets, the share of groups at least as good as Default, and R10's plain pick.

The cross-validated difference behind H2 is about -3 points of mean regret, with group-to-group spread of the same order, so with 12 shapes H2 may well fail.
