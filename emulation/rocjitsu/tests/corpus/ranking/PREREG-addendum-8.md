# Pre-registration addendum 8 (written 2026-11-19, before any hardware data of held-out set 8 exists)

## What changed after addendum 7 was scored

Held-out set 7 is now development data. Diagnosing its attention failure showed that every earlier predictor ran without machine parameters. Every latency was 10 000 cycles, every rate 1 byte or line per cycle, and only the recompose knobs made the output usable.

Since then:

- `tests/corpus/ubench/mb.hip` measured the parameters on MI355X. The overlay `configs/tuning/gfx950_mi355x_measured.json` holds them (sha256 b2ef782d46c2e19f1bead2ae47804ba4702610e7dc9b6163fe29174369020203).
- The LDS model now charges both address sets of ds2 instructions, models banks by 8-byte granule and treats permutes as conflict free.
- The harness generates inputs and computes references on the host.

On all development data:

- Uncalibrated, the new emulator orders hipBLASLt variants 85.6% and Triton variants 88.0%. The median absolute error is 16.8% over 17 families.
- Cross-validated by group, with the absolute knobs below, Triton reaches 92.8%. With the ranking stage on top, hipBLASLt reaches 91.2%; that stage lowers Triton to 87.7%, so Triton keeps the absolute knobs.
- Leaving each family out of the absolute fit gives median errors of 5-18% on the family left out.

## Predictors (fixed now)

**New.** The emulator library is at commit 5b560b9e10, unchanged at 6775fd478a, which adds this file. The config is `config_hw4.json` (sha256 cf0cbff703723154d56ec5c39e9544fcb66f90d3cecbc3a6429d1c2f5b43149c). It is the shape config merged with the overlay, with one thread and `clock_mhz` 2390, and runs use `setarch -R`. Each prediction is the 5-fit geometric mean over warm calls.

- **N_G** (hipBLASLt ranking): knobs `hw4R-fit-s1..5.json`, the ranking stage on every development group.
- **N_T** (Triton ranking, including the kernel families): knobs `hw4A-fit-s1..5.json`.
- **N_A** (absolute time, every family): knobs `hw4A-fit-s1..5.json`, the absolute fit on every development family (`abs_fit` on the meter corpus and the ranking items).

**Old (reference).** The emulator library is at commit 439cfa9922, which includes the f8f6f4 fix and not the LDS changes. The config is `config_b1_wb.json` (sha256 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b) and the harness is the same.

- **O** (ranking): `prereg8c-fit-s1..5.json` (addendum 6).
- **O_A** (absolute time): `prereg7c-fit-s1..5.json` (addendum 7).

| file | sha256 |
|---|---|
| hw4A-fit-s1.json | 26660579fcf4f452d884a557afa42c7e692fd798640f34218e140dea309e03cf |
| hw4A-fit-s2.json | b86f6234500537f4fee58d08f5e3147c4817421b7124db149ff85cd7f4ba1dda |
| hw4A-fit-s3.json | 565fdb534aaad8f9f3edc51bab2f4306209a553e16ddf07ce2b2beec439c7582 |
| hw4A-fit-s4.json | facefe8b914098a565bfc9940decb790c4fbf228199a013c48d7199a91c6505e |
| hw4A-fit-s5.json | 2b34f5dcf2915741744176cd4f705d84dab5653b94531a87bddc0ecb9b7addfd |
| hw4R-fit-s1.json | 958f773989edf90b5868235420a113edb16f8bf556e7b50b38778433df17c2a2 |
| hw4R-fit-s2.json | 47f74609c97a44f9b556d655517a3b2d0f05141189303b7d2feb4ce03faef7b6 |
| hw4R-fit-s3.json | c91f7a99e9ab4f9556105de9fde30a3a900116eb2baa7695f6c77399906a455c |
| hw4R-fit-s4.json | 990dfaffd18640b1a2ead36cc96092da05ccb8ca7eede6e6c27109e308d98840 |
| hw4R-fit-s5.json | 7ef0e678cd299b645fae871a1c8812327489ebe329b78f7b2984ccc3d4935c17 |

## Held-out set 8 (never measured on hardware before this document)

The benchmark files are `gemm_solutions.py` (sha256 ad08ef2e84273f176d79d11a156c6827aa2367041973d62026a1a7357a61a843), `triton_bench.py` (217ae911a3ce10fffe73d2640fe52c184901bfe6bd68e7b1e89ac27b246683db) and `kernel_families.py` (574b57adb5b7a46fdda7b7434368b23c2e2c795375b73369afd012b9ca87f9e7).

- **hipBLASLt** (`HELDOUT8_GEMM`, 12 shapes): m11_n5632_k4096, m28_n3840_k3072, m52_n2048_k5120, m88_n6144_k2048, m176_n2560_k2560, m352_n1536_k3072, m704_n1024_k2048, m1536_n512_k3072, m6_n4096_k7168_f16, m40_n3072_k4096_f16, m120_n2048_k3584_f16, m896_n1536_k1024_f16. Picks follow the select.py rule.
- **Triton** (`HELDOUT8_TRITON`): matmul mm640x2304x1280, mm1792x768x2560, mm2560x1536x768; softmax with the store policy among the variants, sm4608x1536, sm1792x2816, sm9216x768.
- **Kernel families** (`HELDOUT8_FAMILIES`):
  - attention fa1x12x1536x64, fa2x8x1024x128c, fa1x24x768x128;
  - RMSNorm rms3072x5120, rms6144x2560;
  - fp8 matmul f8m768n1536k2048, f8m3072n768k1536, f8m64n5120k3072;
  - two families new to the model, LayerNorm (ln4096x2048, ln2048x5120, ln8192x1024, ln1024x7168) and SwiGLU (sw4096x4096, sw2048x7168, sw8192x2816, sw16384x1408).

Before this document, the only runs of the new families were functional: one variant each under the emulator. No hardware time was taken.

**Hardware truth.** gpu_guard, 3 runs, profiler kernel time, on node pit2-p03-g40. The units are spread over HIP 0-7 (`RUNS@GPU,...`): each (run, group) unit runs on one GPU, and the three runs of a group run on different GPUs. On the same five kernels, the eight GPUs agree within 2.6%. Only one measurement pipeline runs at a time.

## Criteria (scored once)

Stable pairs, the bootstrap, regret and the non-timing baselines are as in PREREG.md.

- **H1** (hipBLASLt): N_G pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline point accuracy.
- **H2** (hipBLASLt): N_G minus O pairwise accuracy, paired 95% CI lower bound, is above 0.
- **H3** (hipBLASLt): N_G mean pick regret minus that of hipBLASLt Default, paired bootstrap over groups, 95% CI upper bound below 0.
- **H4** (Triton, all 22 Triton groups of the set): N_T pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline point accuracy.
- **H5** (Triton): N_T minus O pairwise accuracy, paired 95% CI lower bound, is above 0.
- **A1** (absolute time, every item of the set): N_A pooled median of |predicted / measured - 1| is at most 20%.
- **A2** (absolute time): N_A minus O_A in mean |log2(predicted / measured)|, bootstrap over groups, 95% CI upper bound below 0.

Reported, not hypotheses:

- the absolute statistics per family, with LayerNorm and SwiGLU separately as families never seen;
- the 90th percentile;
- the share of items within 25%.

The scoring uses `prereg_score.py` (with `--ref-config` for O) and `abs_score.py`.

## Amendment (2026-11-19, before any held-out-8 kernel was timed for the benchmark)

TunableOp enumeration of m11_n5632_k4096 hung the GPU three times (`HW Exception ... GPU Hang`), on HIP 5 twice and on HIP 0 once, each time inside a different hipBLASLt solution. The shape cannot produce a candidate list, so it is replaced by m13_n4608_k4096, which has never been run. Everything else in this addendum is unchanged. The only hardware data seen so far is the partial TunableOp log of the dropped shape.
