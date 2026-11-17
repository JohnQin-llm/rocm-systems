#!/usr/bin/env python3
"""Score PREREG.md's predictor on one benchmark and apply its success criteria.

  prereg_score.py gemm   --hw hw3         --emu emu
  prereg_score.py triton --hw triton_hw   --emu triton_emu
  prereg_score.py gemm ... --fit-w --groups DEV...              (PREREG-addendum-2.md: choose w)
  prereg_score.py gemm ... --combine-w W --primary comb --groups HELDOUT2...
  prereg_score.py gemm ... --ref-emu DIR --ref-fits GLOB --ref-name NAME   (PREREG-addendum-3.md: paired comparison)

Predictor: per item, the geometric mean over the prereg-fit-s*.json tunings of
the recomposed modelled cycles of the item's dispatches (bench_refit.py's
method). Hardware truth: per-item median over guarded, uncontended runs.
"""

import argparse
import glob
import itertools
import json
import os
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("bench", choices=("gemm", "triton"))
ap.add_argument("--hw", required=True)
ap.add_argument("--emu", required=True)
ap.add_argument("--fits", default="prereg-fit-s*.json")
ap.add_argument("--noise-pct", type=float, default=10.0)
ap.add_argument("--groups", nargs="*", help="score only these groups (shapes / cases)")
ap.add_argument("--warm-only", action="store_true",
                help="score only the timed (warm) calls, as hardware and the event bracket do")
ap.add_argument("--combine-w", type=float,
                help="also score pred^w * instructions^(1-w) (PREREG-addendum-2.md)")
ap.add_argument("--fit-w", action="store_true",
                help="print pairwise accuracy of the combined predictor over a w grid and the w the "
                     "PREREG-addendum-2.md rule selects, then stop (development data only)")
ap.add_argument("--primary", choices=("pred", "comb"), default="pred",
                help="predictor the pre-registered criteria apply to")
ap.add_argument("--ref-emu", help="a second emulation of the same items, to compare against (PREREG-addendum-3.md)")
ap.add_argument("--ref-fits", help="the reference's tunings (default: --fits)")
ap.add_argument("--ref-name", default="reference")
ap.add_argument("--defer", type=float,
                help="also pick with the deferring rule: the primary predictor's pick only when it predicts it "
                     "more than this fraction faster than Default, otherwise Default (PREREG-addendum-6.md)")
ap.add_argument("--regret-tests", action="store_true",
                help="paired bootstrap over groups of mean pick regret: the deferring (or primary) pick against "
                     "Default and against the reference (PREREG-addendum-6.md)")
ap.add_argument("--margin", type=float, default=0.0,
                help="paired comparison passes when its CI lower bound exceeds this (e.g. -0.02: non-inferiority)")
args = ap.parse_args()
if args.primary == "comb" and args.combine_w is None:
    ap.error("--primary comb needs --combine-w")
SAMPLES = 3


def per_call(e):
    """Calls made under the emulator: gemm = 1 warmup + 3 samples; triton adds the compile call."""
    return 4 if args.bench == "gemm" else 5
TOOLS = Path(__file__).resolve().parent                        # this directory
ROOT = TOOLS.parents[2]                                          # emulation/rocjitsu
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
B = WORK
sys.path.insert(0, str(TOOLS.parent))
import meter_refit as mr  # noqa: E402

base = mr.config_defaults(mr.load_config(str(ROOT / "configs/gfx950_mi355x_kmd_timing.json")))


def load_tunings(pattern):
    out = []
    for f in sorted(glob.glob(str(B / pattern))):
        t = dict(base)
        t.update(json.loads(Path(f).read_text())["knobs"])
        out.append(t)
    return out


tunings = load_tunings(args.fits)
G, I = ("shape", "solution") if args.bench == "gemm" else ("case", "variant")


def load_hw():
    runs = []
    for d in sorted(glob.glob(str(B / args.hw / "run*"))):
        run = {}
        for p in glob.glob(d + "/*.json"):
            if p.endswith(".guard.json"):
                continue
            g = Path(p[:-5] + ".guard.json")
            if not g.exists() or json.loads(g.read_text())["contended"]:
                continue
            r = json.loads(Path(p).read_text())
            if r.get("kernel_us"):
                run[(r[G], r[I])] = r
        runs.append(run)
    return runs


hw = load_hw()


def load_emu(emu_dir, tunings):
    """Per item: the ensemble prediction over warm calls, after the identity checks."""
    rows, dropped = {}, defaultdict(int)
    for p in glob.glob(str(B / emu_dir / "*__*.json")):
        if p.endswith(".raw.json"):
            continue
        e = json.loads(Path(p).read_text())
        key = (e[G], e[I])
        if args.groups and key[0] not in args.groups:
            continue
        if not all(key in h for h in hw):
            dropped["no clean hardware in every run"] += 1
            continue
        if args.bench == "gemm":
            same = set(e.get("kernels", [])) == {k for k in hw[0][key].get("kernels", []) if k.startswith("Cijk")}
            prefix = "Cijk"
        else:
            same = e.get("kernel_hash") and e.get("kernel_hash") == hw[0][key].get("kernel_hash") and \
                e.get("max_err", 1) < 0.05 and hw[0][key].get("max_err", 1) < 0.05
            prefix = e.get("kernel_name") or "?"
        if not same:
            dropped["kernel mismatch or wrong result"] += 1
            continue
        trace = Path(p[:-5] + ".trace.jsonl")
        disp = []
        for line in trace.read_text().splitlines() if trace.exists() else []:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("kernel", "")).startswith(prefix):
                disp.append(d)
        if not disp:
            dropped["no traced dispatch"] += 1
            continue
        # Hardware and the emulator's event bracket both time warm calls only; the
        # first call(s) here run on cold caches. Score the last SAMPLES calls, per call.
        per = max(1, len(disp) // per_call(e))       # dispatches per call
        warm = disp[-SAMPLES * per:] if args.warm_only else disp

        def call_cycles(t, f=lambda d, t: mr.recompose(d, t)):
            sums = [sum(f(d, t) for d in warm[i:i + per]) for i in range(0, len(warm), per)]
            return statistics.median(sums) if args.warm_only else sum(sums)
        pred = math.exp(statistics.fmean(math.log(call_cycles(t)) for t in tunings))
        runs = [h[key]["kernel_us"] for h in hw]
        rows[key] = dict(hw=statistics.median(runs), runs=runs, pred=pred,
                         uncal=call_cycles(None, lambda d, t: d["cycles"]),
                         insts=e.get("gemm_instructions") or e.get("instructions") or None,
                         dram=e.get("gemm_dram_bytes") or e.get("dram_bytes") or None)
    return rows, dropped


rows, dropped = load_emu(args.emu, tunings)
if args.ref_emu:
    # An item is scored only where both emulations ran the kernel hardware ran.
    ref_rows, _ = load_emu(args.ref_emu, load_tunings(args.ref_fits or args.fits))
    for key in list(rows):
        if key in ref_rows:
            rows[key]["ref"] = ref_rows[key]["pred"]
        else:
            del rows[key]
            dropped["no valid reference emulation"] += 1
print(f"{args.bench}: {len(rows)} items scored over {len({k[0] for k in rows})} groups; "
      f"{len(tunings)} tunings; hardware runs {len(hw)}; dropped {dict(dropped)}")

def combined(r, w):
    """PREREG-addendum-2.md: the model's prediction and the executed-instruction count, as a
    weighted geometric mean; within a group only the ranking matters, so no scale is needed."""
    if not r["insts"]:
        return None
    return math.exp(w * math.log(r["pred"]) + (1 - w) * math.log(r["insts"]))


PRED = {"PREDICTOR (5-fit ensemble)": "pred", "uncalibrated model": "uncal",
        "fewer instructions": "insts", "fewer DRAM bytes": "dram"}
if args.combine_w is not None:
    PRED[f"COMBINED (w={args.combine_w:g})"] = "comb"
    for r in rows.values():
        r["comb"] = combined(r, args.combine_w)
if args.ref_emu:
    PRED[f"REFERENCE ({args.ref_name})"] = "ref"
PRIMARY = next(n for n, f in PRED.items() if f == args.primary)
thr = math.log2(1 + args.noise_pct / 100)
stable, ceiling = [], []
for grp in sorted({k[0] for k in rows}):
    keys = [k for k in rows if k[0] == grp]
    for a, b in itertools.combinations(keys, 2):
        ra, rb = rows[a], rows[b]
        if abs(math.log2(ra["hw"] / rb["hw"])) < thr:
            continue
        for i in range(len(ra["runs"])):
            oa = statistics.median(ra["runs"][:i] + ra["runs"][i + 1:])
            ob = statistics.median(rb["runs"][:i] + rb["runs"][i + 1:])
            ceiling.append(float((ra["runs"][i] < rb["runs"][i]) == (oa < ob)))
        if len({x < y for x, y in zip(ra["runs"], rb["runs"])}) > 1:
            continue
        stable.append((grp, ra, rb, ra["hw"] < rb["hw"]))


def agree(x, y, truth):
    return 0.5 if x is None or y is None or x == y else float((x < y) == truth)


if args.fit_w:
    # The rule fixed in PREREG-addendum-2.md: w on a 0.05 grid maximising pooled pairwise
    # accuracy; ties go to the w nearest 0.5, then to the larger w.
    grid = [i / 20 for i in range(21)]
    acc_w = {w: statistics.fmean(agree(combined(ra, w), combined(rb, w), t) for _, ra, rb, t in stable)
             for w in grid}
    print(f"\nCOMBINED over {len(stable)} stable development pairs")
    for w in grid:
        print(f"  w={w:4.2f}  {acc_w[w]:6.1%}")
    best = max(acc_w.values())
    pick = min((w for w in grid if acc_w[w] == best), key=lambda w: (abs(w - 0.5), -w))
    print(f"\nSELECTED w={pick:g} ({best:.1%})")
    sys.exit(0)

pairs = []
for grp, ra, rb, truth in stable:
    sc = {n: agree(ra[f], rb[f], truth) for n, f in PRED.items()}
    sc["random"] = 0.5
    pairs.append((grp, sc))
rng = random.Random(0)


def ci(v):
    m = sorted(statistics.fmean(rng.choices(v, k=len(v))) for _ in range(4000))
    return m[100], m[3899]


acc = {}
print(f"\nPAIRWISE ({len(pairs)} stable pairs; hardware ceiling {statistics.fmean(ceiling):.1%})")
for n in list(PRED) + ["random"]:
    v = [p[1][n] for p in pairs]
    acc[n] = (statistics.fmean(v), *ci(v))
    print(f"  {n:28s} {acc[n][0]:6.1%}  95% CI [{acc[n][1]:.1%}, {acc[n][2]:.1%}]")
print(f"\n  {'group':22s} {'pairs':>5} " + " ".join(f"{f:>6s}" for f in PRED.values()))
for grp in sorted({p[0] for p in pairs}):
    ps = [p[1] for p in pairs if p[0] == grp]
    print(f"  {grp:22s} {len(ps):5d} " + " ".join(f"{statistics.fmean(p[n] for p in ps):6.0%}" for n in PRED))

print("\nDECISION (regret = hardware time of the pick / best - 1)")
by_group = defaultdict(dict)     # predictor -> group -> regret
DEFER = f"{PRIMARY.split(' (')[0]}, defer >{args.defer:.0%}" if args.defer is not None else None
for grp in sorted({k[0] for k in rows}):
    keys = [k for k in rows if k[0] == grp]
    best = min(rows[k]["hw"] for k in keys)
    for n, f in PRED.items():
        ks = [k for k in keys if rows[k][f]]
        by_group[n][grp] = rows[min(ks, key=lambda k: rows[k][f])]["hw"] / best - 1
    dkey = (grp, "Default")
    if args.bench == "gemm" and dkey in rows:
        by_group["hipBLASLt Default"][grp] = rows[dkey]["hw"] / best - 1
        if DEFER:
            f = PRED[PRIMARY]
            pick = min((k for k in keys if rows[k][f]), key=lambda k: rows[k][f])
            choice = pick if rows[dkey][f] / rows[pick][f] > 1 + args.defer else dkey
            by_group[DEFER][grp] = rows[choice]["hw"] / best - 1
    by_group["random (expected)"][grp] = statistics.fmean(rows[k]["hw"] / best - 1 for k in keys)
reg = {n: list(v.values()) for n, v in by_group.items()}
for n, v in reg.items():
    extra = ""
    if "hipBLASLt Default" in by_group and n != "hipBLASLt Default":
        common = [g for g in by_group[n] if g in by_group["hipBLASLt Default"]]
        if common:
            extra = (f"   <= Default in {sum(by_group[n][g] <= by_group['hipBLASLt Default'][g] + 1e-9 for g in common)}"
                     f"/{len(common)}")
    print(f"  {n:28s} median {statistics.median(v):6.1%}   mean {statistics.fmean(v):6.1%}   "
          f"worst {max(v):6.1%}   groups {len(v)}{extra}")

if args.regret_tests:
    # Paired over groups: resample groups, mean of the per-group regret differences.
    def paired(a, b):
        gs = [g for g in by_group[a] if g in by_group[b]]
        d = [by_group[a][g] - by_group[b][g] for g in gs]
        boot = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(4000))
        return statistics.fmean(d), boot[100], boot[3899]
    tested = DEFER or PRIMARY
    print("\nREGRET TESTS (mean pick regret difference, paired bootstrap over groups; criterion: CI upper bound < 0)")
    for other in ("hipBLASLt Default", f"REFERENCE ({args.ref_name})" if args.ref_emu else None):
        if other and other in by_group:
            m, lo, hi = paired(tested, other)
            print(f"  {tested} minus {other}: {m:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]: {'PASS' if hi < 0 else 'FAIL'}")

if args.ref_emu:
    # Both predictors on the same stable pairs: the difference is paired.
    ref_name = f"REFERENCE ({args.ref_name})"
    diffs = [p[1][PRIMARY] - p[1][ref_name] for p in pairs]
    lo, hi = ci(diffs)
    print(f"\nPAIRED: {PRIMARY} minus {ref_name}: {statistics.fmean(diffs):+.1%}  "
          f"95% CI [{lo:+.1%}, {hi:+.1%}]: {'PASS' if lo > args.margin else 'FAIL'} "
          f"(criterion: lower bound > {args.margin:+.1%})")

print(f"\nPRE-REGISTERED CRITERIA ({PRIMARY})")
best_base = max(("fewer instructions", "fewer DRAM bytes"), key=lambda n: acc[n][0])
ok1 = acc[PRIMARY][1] > acc[best_base][0]
print(f"  primary: CI lower {acc[PRIMARY][1]:.1%} > best non-timing baseline "
      f"({best_base}) {acc[best_base][0]:.1%}: {'PASS' if ok1 else 'FAIL'}")
med = {n: statistics.median(v) for n, v in reg.items()}
others = [med[best_base]] + ([med["hipBLASLt Default"]] if "hipBLASLt Default" in med else [])
ok2 = med[PRIMARY] <= min(others)
print(f"  secondary: median regret {med[PRIMARY]:.1%} <= "
      f"{', '.join(f'{x:.1%}' for x in others)}: {'PASS' if ok2 else 'FAIL'}")
