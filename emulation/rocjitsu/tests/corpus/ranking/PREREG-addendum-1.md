# Pre-registration addendum 1 (written 2026-11-13, before any held-out data exists)

## What changed after PREREG.md was scored

Diagnosis on the PREREG benchmarks (so their scores are no longer clean tests):

1. Scoring: the recomposed prediction now uses only the timed (warm) calls,
   as the hardware and the emulator's event bracket do (`--warm-only`).
2. Model: the timing plane folds a workgroup onto the timed compute unit the
   way the SPI places it (XCD = wg % xcds, then round-robin within the XCD)
   instead of wg % units, which put 32 consecutive workgroups behind one L2.
3. Model: timing tag arrays use an XOR-hashed set index instead of
   line % sets, which piled power-of-two strides onto a few sets.

Fixes 2 and 3 were checked mechanistically against hardware counters
(L2 and fabric traffic), not against ranking scores.

## Held-out benchmark (never seen during diagnosis)

- hipBLASLt: m2_n4096_k4096, m4_n2048_k8192, m48_n3072_k1024,
  m96_n1536_k2048, m768_n768_k768, m1536_n512_k1024 (bf16),
  m1_n2048_k4096_f16, m128_n4096_k512_f16. Picks by the select.py rule.
- Triton: matmul mm768 (768^3), mm128x2048x2048; softmax sm2048x2048,
  sm16384x256. Variants by the triton_bench.py seeded rule.
- Hardware truth exactly as in PREREG.md (gpu_guard, 3 runs on HIP 5,6,7).

## Predictor (fixed now)

The rebuilt emulator (fixes 2 and 3) re-runs the meter corpus; knobs are
refitted with the PREREG.md recipe (no gemm/quantized/concurrent cases,
guarded run 1, seeds 1-5); prediction is the 5-fit geometric-mean ensemble
over warm calls.

## Criteria

Unchanged from PREREG.md, applied to the held-out benchmark only, each
library separately. The held-out set is scored once, with this predictor.
