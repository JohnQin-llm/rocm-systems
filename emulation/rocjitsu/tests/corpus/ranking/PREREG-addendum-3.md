# Pre-registration addendum 3 (written 2026-11-13, before any held-out-3 data exists)

## What changed after addendum 2 was scored

The timing model gained write-back caches (commits ceda5ac3b1, 121b699b1a):

- the second-level cache and the memory-side cache absorb stores dirty;
- evicted dirty lines are written back during the kernel;
- what is still dirty at a dispatch's end is flushed as a serial term.

Non-temporal stores are allocated cold, and system-scope stores pass the GPU caches.

The change was driven by MI355X counters on a softmax, not by ranking scores. The flush cost scale was calibrated on a separate copy microbenchmark (3 sizes x 3 store policies).

While building it, several model variants were scored on the groups of PREREG.md and addenda 1-2. Those scores are development data. The variant fixed below was chosen on mechanistic grounds: counters, the calibration microbenchmark, and the 9 agent softmax finals. It was not chosen as the best scorer, and on some of those groups it is not.

## Predictors (fixed now)

Both predictors use the emulator at commit 121b699b1a, run with one thread (`cpu_thread_budget` 1) and without address randomisation (`setarch -R`). Each prediction is the 5-fit geometric-mean ensemble of the model alone over the warm calls (`--warm-only`).

- **WB (under test):** config `config_b1_wb.json` (the gfx950 timing config with write-back caches), with knobs `prereg7c-fit-s1..5.json`. Those are the meter-corpus refit on this model plus `flush_scale` from the copy microbenchmark.
- **V4 (reference):** config `config_b1.json`, the same config without the write-back keys, which reproduces the earlier model's solution dispatches cycle for cycle. Knobs are `prereg3-fit-s1..5.json`, the predictor of addendum 2.

sha256 of those inputs:

| file | sha256 |
|---|---|
| config_b1_wb.json | 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b |
| config_b1.json | f2e4566c4a57ad5012d8519c2c9ede736a1bf2fea77b52ed01f22f25529670ec |
| prereg7c-fit-s1.json | cacf9b48fbcb6359a965f3e4e1046bad723674917f9887969baaa66619e4943c |
| prereg7c-fit-s2.json | 8c43f116f0b66561032b6d20c15bcb4ed1678c87dd345d5b6341162181657bb4 |
| prereg7c-fit-s3.json | 446b0693b82433ac595acdcfbaf1f9a2d8f99dee8bfd5241b5b49ff835b40873 |
| prereg7c-fit-s4.json | 5cc149861c66f7620e4e55dd2f7786ad30d37ee295a9501714660b6cf1125962 |
| prereg7c-fit-s5.json | 16e02149d167bbd6c4efd2c05d487a9c8a9ab53cf99845cc2ad3d33b0a1af253 |
| prereg3-fit-s1.json | e70e75ce517ae65f890f92ee1918680f61131033605d909b3ae2fe6141815421 |
| prereg3-fit-s2.json | e554c2abd72dd35460aefb17b704ed47cf00765450bd9ba16e0417ceee98d4d0 |
| prereg3-fit-s3.json | 589e867a621dfb0fddbbeb2ae4d52d62046279b782e5d1a2ad5c1545c8d6bb03 |
| prereg3-fit-s4.json | 0f49dbd396c040b7a2742cb359253431d6495aebdce890c6ae5ccebc72853c7b |
| prereg3-fit-s5.json | 2ad0c68d8f399eeff6add77c936a899fbeb4bce44cac41262dfa7a82a2b01408 |

## Held-out set 3 (never measured before this document)

- **hipBLASLt** (bf16 unless noted): m5_n5120_k4096, m40_n2560_k2048, m80_n6144_k1024, m160_n1024_k4096, m320_n3072_k768, m640_n640_k2560, m3_n3072_k3072_f16, m256_n1536_k1536_f16. Candidates come from TunableOp enumeration; picks follow the select.py rule.
- **Triton matmul:** mm384x1536x1536, mm1024x384x2048, mm192x4096x512. Variants follow the seeded rule.
- **Triton softmax:** sm3072x768, sm6144x1280, sm1536x2560. Variants are 12 seeded draws from num_warps x rows-per-program x output store policy (plain, .cs, .wt), using `softmax_store_kernel`.
- **Hardware truth** is taken exactly as in PREREG.md: gpu_guard, 3 runs, on HIP 5,6,7 for hipBLASLt and HIP 2,3,4 for Triton. Contended attempts are retried.

## Criteria

These apply to held-out set 3 only, to each library separately, scored once. Stable pairs, the bootstrap and regret are computed exactly as in PREREG.md. An item is scored only where both emulations ran the kernel hardware ran.

- **H1:** WB pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline's point accuracy (fewer instructions, fewer DRAM bytes).
- **H2:** WB median pick regret <= the best non-timing baseline's and, for hipBLASLt, hipBLASLt Default's.
- **H3:** WB minus V4 pairwise accuracy on the same stable pairs, paired bootstrap 95% CI lower bound > 0. This is the test of the model change.

Reported but not hypotheses: V4's own H1/H2 numbers, and the combined predictor with addendum 2's w (0.9 / 0.7) on WB.
