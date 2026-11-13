# Pre-registration addendum 2 (written 2026-11-13, before any held-out-2 data exists)

## What changed after addendum 1 was scored

1. Model: addendum 1's placement fix also folded the index carried by a
   memory request, which is already the timed compute unit's own index, not
   a workgroup id. Requests were routed to the L1/L2 of a different unit
   (die (u / 8) % 8 instead of u / 32), so workgroups a kernel grouped onto
   one XCD were spread over several L2s. Found mechanistically: per-line
   fabric accounting on m1_n1024_k4096 showed every B line fetched by 2 or 4
   dies where hardware counters show one fetch. With requests routed by the
   unit's own index, the same picks fetch 1.01x. No ranking score was
   looked at for this fix.
2. Consequently the addendum-1 held-out scores were measured with a model
   that had this bug. They stand as reported; re-scoring that set with the
   fixed model is development data from now on, and is labelled post hoc.

## Predictors (fixed now)

- MODEL (v4): the rebuilt emulator (fix 1) re-runs the meter corpus; knobs
  refitted with the PREREG.md recipe (no gemm/quantized/concurrent cases,
  guarded run 1, seeds 1-5, output prereg3-fit-s*.json); prediction is the
  5-fit geometric-mean ensemble over warm calls (`--warm-only`).
- COMBINED: `MODEL^w * instructions^(1-w)`, where instructions is the
  emulator's executed-instruction count of the item's kernels (the "fewer
  instructions" baseline's input). Within a group only the ranking matters,
  so no scale term is needed.
- w is chosen per library (hipBLASLt, Triton) on development data only:
  every group not in held-out set 2 (PREREG.md's and addendum 1's), scored
  with MODEL v4 against the existing guarded hardware truth. Rule: w on the
  grid 0, 0.05, ..., 1 that maximises pooled pairwise accuracy over stable
  pairs; ties go to the w nearest 0.5, then to the larger w
  (`prereg_score.py --fit-w`). w = 0 or 1 is allowed and would mean the
  combination adds nothing.

## Held-out set 2 (never measured before this document)

- hipBLASLt (bf16 unless noted): m3_n6144_k4096, m24_n4096_k2048,
  m64_n1024_k8192, m192_n2560_k1536, m384_n1024_k3072, m1024_n2048_k768,
  m2_n1536_k6144_f16, m512_n512_k2048_f16. Candidates by TunableOp
  enumeration, picks by the select.py rule.
- Triton: matmul mm256x3072x1024, mm1536x640x512; softmax sm4096x512,
  sm1024x3072. Variants by the triton_bench.py seeded rule.
- Hardware truth exactly as in PREREG.md (gpu_guard, 3 runs on HIP 5,6,7
  for hipBLASLt, HIP 2,3,4 for Triton; contended attempts retried).

## Criteria

Applied to held-out set 2 only, each library separately, scored once:

- H1 (primary): COMBINED pairwise accuracy, 95% bootstrap CI lower bound,
  exceeds the best non-timing baseline's point accuracy (fewer
  instructions, fewer DRAM bytes).
- H2: COMBINED median pick regret <= the best non-timing baseline's and
  (hipBLASLt) hipBLASLt Default's.
- H3, H4: the same two criteria for MODEL v4 alone.

Stable pairs, bootstrap and regret exactly as in PREREG.md.
