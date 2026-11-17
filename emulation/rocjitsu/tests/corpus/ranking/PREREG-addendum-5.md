# Pre-registration addendum 5 (written 2026-11-13, before any held-out-5 data exists)

## What changed after addendum 4 was scored

Held-out set 4 is now development data.

Store-policy counters and a timing sweep of the copy microbenchmark (13 sizes x 3 store policies) showed that MI355X's store-policy cost is regime-dependent:

- below about 64 MB of working set, write-through and streaming stores beat plain stores by up to 1.6x;
- above about 80 MB, all three converge.

The model's single release-flush scale cannot express that. Calibrated on 3 sizes it was about 4 and over-penalised plain stores in small softmaxes.

Option A, taken here, recalibrates `flush_scale` on the whole sweep instead, which gives about 2.5 (the loss stays high). This is a compromise, not a mechanism.

Development evidence:

- On the 6 store-policy softmax groups of held-out sets 3-4, which neither flush calibration saw, the corpus knobs score +1.6 and +3.3 points with the sweep calibration. Both CIs span 0.
- With the second stage on top, cross-validated by group over all 76 development groups: Triton 79.3% vs 79.2%, hipBLASLt 88.1% vs 88.7%.

## Predictors (fixed now)

Both predictors run the emulator at commit 121b699b1a with config `config_b1_wb.json` (sha256 616c8498519014f30e505eb215bd77fbdcbecf3748ecf5fa3c03c9129b4bac0b). Runs use one thread and no address randomisation. Each prediction is the 5-fit geometric-mean ensemble of the model alone over warm calls. Both use the second stage (`rank_fit.py`) on the same development groups: PREREG.md and held-out sets 1-4, 1113 items, 5762 stable pairs.

- **R9 (under test):** knobs `prereg9-fit-s1..5.json`. These are `prereg7s`, i.e. the corpus knobs with `flush_scale` from the sweep, followed by the second stage.
- **R8b (reference):** knobs `prereg8b-fit-s1..5.json`. These are `prereg7c` (flush_scale from 3 sizes) followed by the same second stage.

sha256 of the knob files:

| file | sha256 |
|---|---|
| prereg9-fit-s1.json | 1aec33cefb2eb7aff2ad2d70696713e25befd0ecf86c96a676692f15a4500f34 |
| prereg9-fit-s2.json | 9e75ad1a365decfa0ae93764281d3a546c2b74eb346449b4e29706523d3ec58e |
| prereg9-fit-s3.json | b3b5a0d2ee274c764f122198b913fcf5374eb05f19e3f2edbcdc4b1004f777af |
| prereg9-fit-s4.json | 78d17359317a32e9015decf176169a8a8f48d65f52734be0f685f0ae2032c37e |
| prereg9-fit-s5.json | a9161ff25dead9d963d617ab6ea6d5fc36345d4b68f8bfcb77fed98dcd75a86c |
| prereg8b-fit-s1.json | e81caf49d84817246bf8c3e6a36f64944c708224d59c173d892686ab36f208db |
| prereg8b-fit-s2.json | 765cbb8b9f856ddbcea2e2e9eaf07ffee6d8f82511f535cc70a26757a8ca2093 |
| prereg8b-fit-s3.json | 3e0581010780444a4c1e13ef076aff22504b868d8db36f0c8530b90ff3431621 |
| prereg8b-fit-s4.json | 07ce568ef4df5850148fa13afb344663ed848b7e1463fba36ae9fb35bdd499fa |
| prereg8b-fit-s5.json | bea28d5a7d9610b35313b183011e7ebfd22fe846b53c4e6009c233abaaf27419 |

## Held-out set 5 (never measured before this document)

- **hipBLASLt** (bf16 unless noted): m7_n3584_k5120, m20_n7168_k2048, m60_n4608_k1024, m128_n2816_k2816, m272_n1280_k3584, m512_n1024_k1536, m2_n5120_k5120_f16, m96_n3584_k2048_f16. Picks follow the select.py rule.
- **Triton matmul:** mm448x1792x896, mm1280x512x3072.
- **Triton softmax** (variants include the store policy, addendum-3 rule): sm5120x1024, sm2048x3584, sm10240x384, sm3584x2048, sm1280x4608, sm6144x896.
- **Hardware truth** is taken as in PREREG.md: gpu_guard, 3 runs, on HIP 5,6,7 for hipBLASLt and HIP 2,3,4 for Triton. Only one measurement process runs at a time.

## Criteria

These apply to held-out set 5 only, scored once. Stable pairs, the bootstrap and regret are computed as in PREREG.md.

- **H1** (each library): R9 pairwise accuracy, 95% CI lower bound, exceeds the best non-timing baseline's point accuracy.
- **H2** (each library): R9 median pick regret <= the best non-timing baseline's and, for hipBLASLt, hipBLASLt Default's.
- **H3** (the target, the 6 softmax groups only): R9 minus R8b pairwise accuracy, paired bootstrap 95% CI lower bound > 0.
- **H4** (non-inferiority, hipBLASLt): R9 minus R8b paired 95% CI lower bound > -2 points.

The development effect behind H3 is small (about +2.5 points), so H3 may well fail. It is stated as the honest test of option A, not tuned for power.
