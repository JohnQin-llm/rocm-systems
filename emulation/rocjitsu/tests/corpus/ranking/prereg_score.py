#!/usr/bin/env python3
"""Score PREREG.md's predictor on one benchmark and apply its success criteria.

  prereg_score.py gemm   --hw hw3         --emu emu
  prereg_score.py triton --hw triton_hw   --emu triton_emu

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
args = ap.parse_args()
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
tunings = []
for f in sorted(glob.glob(str(B / args.fits))):
    t = dict(base)
    t.update(json.loads(Path(f).read_text())["knobs"])
    tunings.append(t)
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
rows, dropped = {}, defaultdict(int)
for p in glob.glob(str(B / args.emu / "*__*.json")):
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
print(f"{args.bench}: {len(rows)} items scored over {len({k[0] for k in rows})} groups; "
      f"{len(tunings)} tunings; hardware runs {len(hw)}; dropped {dict(dropped)}")

PRED = {"PREDICTOR (5-fit ensemble)": "pred", "uncalibrated model": "uncal",
        "fewer instructions": "insts", "fewer DRAM bytes": "dram"}
thr = math.log2(1 + args.noise_pct / 100)
pairs, ceiling = [], []
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
        truth = ra["hw"] < rb["hw"]
        sc = {n: (0.5 if ra[f] is None or rb[f] is None or ra[f] == rb[f] else float((ra[f] < rb[f]) == truth))
              for n, f in PRED.items()}
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
print(f"\n  {'group':22s} {'pairs':>5} {'pred':>6} {'uncal':>6} {'insts':>6} {'dram':>6}")
for grp in sorted({p[0] for p in pairs}):
    ps = [p[1] for p in pairs if p[0] == grp]
    print(f"  {grp:22s} {len(ps):5d} " + " ".join(f"{statistics.fmean(p[n] for p in ps):6.0%}" for n in PRED))

print("\nDECISION (regret = hardware time of the pick / best - 1)")
reg = defaultdict(list)
for grp in sorted({k[0] for k in rows}):
    keys = [k for k in rows if k[0] == grp]
    best = min(rows[k]["hw"] for k in keys)
    for n, f in PRED.items():
        ks = [k for k in keys if rows[k][f]]
        reg[n].append(rows[min(ks, key=lambda k: rows[k][f])]["hw"] / best - 1)
    if args.bench == "gemm" and (grp, "Default") in rows:
        reg["hipBLASLt Default"].append(rows[(grp, "Default")]["hw"] / best - 1)
    reg["random (expected)"].append(statistics.fmean(rows[k]["hw"] / best - 1 for k in keys))
for n, v in reg.items():
    print(f"  {n:28s} median {statistics.median(v):6.1%}   mean {statistics.fmean(v):6.1%}   "
          f"worst {max(v):6.1%}   groups {len(v)}")

print("\nPRE-REGISTERED CRITERIA")
best_base = max(("fewer instructions", "fewer DRAM bytes"), key=lambda n: acc[n][0])
ok1 = acc["PREDICTOR (5-fit ensemble)"][1] > acc[best_base][0]
print(f"  primary: CI lower {acc['PREDICTOR (5-fit ensemble)'][1]:.1%} > best non-timing baseline "
      f"({best_base}) {acc[best_base][0]:.1%}: {'PASS' if ok1 else 'FAIL'}")
med = {n: statistics.median(v) for n, v in reg.items()}
others = [med[best_base]] + ([med["hipBLASLt Default"]] if "hipBLASLt Default" in med else [])
ok2 = med["PREDICTOR (5-fit ensemble)"] <= min(others)
print(f"  secondary: median regret {med['PREDICTOR (5-fit ensemble)']:.1%} <= "
      f"{', '.join(f'{x:.1%}' for x in others)}: {'PASS' if ok2 else 'FAIL'}")
