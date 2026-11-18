# Pre-registration addendum 7 (written 2026-11-18, before any hardware data of held-out set 7 exists)

## Why

Held-out sets 1-6 tested two operation families of Triton (matmul, softmax) and hipBLASLt GEMM. This set asks two questions about operation families the model has never been calibrated or developed on:

1. Does the variant ranking transfer to them?
2. How far off is the predicted absolute time?

For the second question, the development data gives a reference: median absolute error, warm calls, five-fit ensemble, in sample. Under the corpus knobs `prereg7c` it is 19.9% on hipBLASLt GEMM, 20.3% on Triton matmul and 25.0% on Triton softmax. Under `prereg8c` it is 11.8%, 24.5% and 38.4%.

## Held-out set 7 (never measured on hardware before this document)

`kernel_families.py` (sha256 93958fb8b9f83713294a7add62a129a9cfc8932a9d5c92b188802d62db4fc9fd) defines three families, 240 variants in 18 groups:

- **Attention:** flash-attention forward, bf16, online softmax. Groups fa1x16x1024x128, fa1x8x2048x64, fa2x32x512x128, and the causal fa1x8x2048x128c, fa1x32x1024x64c, fa4x16x512x64c. Each group has 12 variants of block M, block N, warps and stages.
- **RMSNorm:** bf16, with a weight per column. Groups rms4096x4096, rms2048x7168, rms8192x2048, rms1024x8192, rms16384x1024, rms512x5120. Each group has 12 variants of warps and rows per program.
- **fp8 matmul:** float8_e4m3fn operands, bf16 output, the Triton matmul kernel. Groups f8m1024n1024k2048, f8m256n4096k2048, f8m2048n512k4096, f8m512n2048k1024, f8m128n7168k2048, f8m4096n1024k1024. Each group has 16 variants of block M, N, K, warps and stages.

Before this document, the only runs were functional: one variant per family under the emulator, to check that the kernels compile and match a torch reference. No hardware time was taken.

Hardware truth is taken as in PREREG.md: gpu_guard, 3 runs, profiler kernel time, one measurement pipeline at a time. The node is pit2-p03-g40, HIP devices 2, 3 and 4.

## Predictors (fixed now)

The emulator runs at commit 121b699b1a; its timing library is unchanged in the commit that adds this file. The config is `config_b1_wb.json` (sha256 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b): one thread, run with `setarch -R`. Each prediction is the 5-fit geometric-mean ensemble of the model alone over warm calls.

- **R8c (ranking):** knobs `prereg8c-fit-s1..5.json`, with sha256 as in addendum 6.
- **P7c (absolute time):** the corpus knobs `prereg7c-fit-s1..5.json`, which have no second stage.

| file | sha256 |
|---|---|
| prereg7c-fit-s1.json | cacf9b48fbcb6359a965f3e4e1046bad723674917f9887969baaa66619e4943c |
| prereg7c-fit-s2.json | 8c43f116f0b66561032b6d20c15bcb4ed1678c87dd345d5b6341162181657bb4 |
| prereg7c-fit-s3.json | 446b0693b82433ac595acdcfbaf1f9a2d8f99dee8bfd5241b5b49ff835b40873 |
| prereg7c-fit-s4.json | 5cc149861c66f7620e4e55dd2f7786ad30d37ee295a9501714660b6cf1125962 |
| prereg7c-fit-s5.json | 16e02149d167bbd6c4efd2c05d487a9c8a9ab53cf99845cc2ad3d33b0a1af253 |

## Criteria (each family separately, scored once)

Stable pairs, the bootstrap, regret and the non-timing baselines (fewer instructions, fewer DRAM bytes) are as in PREREG.md.

- **H1:** R8c pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline point accuracy.
- **H2:** R8c median pick regret is no larger than that of the best non-timing baseline.
- **A1:** P7c median absolute error of the predicted kernel time is at most 25%. The error of an item is |predicted / measured - 1|, with the measured value the median of the 3 runs.

Reported, not hypotheses:

- the same absolute statistics for R8c;
- the median bias (predicted / measured);
- the share of items within 25%;
- the 90th percentile of the absolute error.

The development reference above puts A1 near its threshold, and fp8 matrix instructions have not been calibrated at all, so A1 may well fail for some families.
