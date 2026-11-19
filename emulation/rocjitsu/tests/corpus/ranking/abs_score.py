#!/usr/bin/env python3
"""Absolute-time accuracy of a calibrated emulator on ranking items (PREREG-addendum-8.md A1, A2).

  abs_score.py --set gemm:HW:EMU --set triton:HW:EMU [...] --fits GLOB --config CFG
               [--groups G ...] [--ref-set ... --ref-fits GLOB --ref-config CFG]

Prediction of an item: the 5-fit geometric mean of the median warm call, as in
prereg_score.py, divided by the config's clock. Measured: median of the clean
hardware runs. Error: |predicted / measured - 1|. Families come from the group
name. The paired test bootstraps over groups the difference in mean |log2
ratio| between this predictor and the reference.
"""

import argparse
import glob
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
import os

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))
import meter_refit as mr  # noqa: E402

WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))
FAMILY = [("mm", "triton matmul"), ("sm", "triton softmax"), ("fa", "attention"), ("rms", "rmsnorm"),
          ("f8", "fp8 matmul"), ("ln", "layernorm"), ("sw", "swiglu")]


def family(bench, group):
    if bench == "gemm":
        return "hipBLASLt GEMM"
    for prefix, name in FAMILY:
        if group.startswith(prefix):
            return name
    return "other"


def clean_hw(hw_dir, G, I):
    runs = []
    for d in sorted(glob.glob(str(WORK / hw_dir / "run*"))):
        run = {}
        for p in glob.glob(d + "/*.json"):
            if p.endswith(".guard.json"):
                continue
            g = Path(p[:-5] + ".guard.json")
            if g.exists() and not json.loads(g.read_text())["contended"]:
                r = json.loads(Path(p).read_text())
                if r.get("kernel_us"):
                    run[(r[G], r[I])] = r
        runs.append(run)
    return runs


def predictions(sets, fits, config, groups):
    cfg = mr.load_config(config)
    base = mr.config_defaults(cfg)
    clock = json.loads(re.sub(r"^\s*//.*$", "", open(config).read(), flags=re.M))["timing"]["clock_mhz"]
    tunings = []
    for f in sorted(glob.glob(str(WORK / fits))):
        t = dict(base)
        t.update(json.loads(Path(f).read_text())["knobs"])
        tunings.append(t)
    out = {}
    for spec in sets:
        bench, hw_dir, emu_dir = spec.split(":")
        G, I = ("shape", "solution") if bench == "gemm" else ("case", "variant")
        runs = clean_hw(hw_dir, G, I)
        for p in glob.glob(str(WORK / emu_dir / "*__*.json")):
            if p.endswith(".raw.json"):
                continue
            e = json.loads(Path(p).read_text())
            key = (e[G], e[I])
            if groups and key[0] not in groups:
                continue
            if not runs or not all(key in r for r in runs):
                continue
            if bench != "gemm" and e.get("kernel_hash") != runs[0][key].get("kernel_hash"):
                continue
            prefix = "Cijk" if bench == "gemm" else (e.get("kernel_name") or "?")
            disp = []
            for line in Path(p[:-5] + ".trace.jsonl").read_text().splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(d.get("kernel", "")).startswith(prefix):
                    disp.append(d)
            calls = 4 if bench == "gemm" else 5
            per = max(1, len(disp) // calls)
            warm = disp[-3 * per:]
            if not warm:
                continue

            def call(t):
                return statistics.median(sum(mr.recompose(d, t) for d in warm[i:i + per]) for i in range(0, len(warm), per))
            pred = math.exp(statistics.fmean(math.log(call(t)) for t in tunings)) / clock
            meas = statistics.median(r[key]["kernel_us"] for r in runs)
            out[(bench, key[0], key[1])] = (pred, meas, family(bench, key[0]))
    return out


def summary(rows, label):
    fam = defaultdict(list)
    for (pred, meas, f) in rows.values():
        fam[f].append(abs(math.log2(pred / meas)))
    print(f"-- {label}")
    pooled = []
    for f, v in sorted(fam.items()):
        pooled += v
        v.sort()
        print(f"  {f:16s} n={len(v):4d} median |err| {2 ** statistics.median(v) - 1:6.1%}  p90 {2 ** v[int(0.9 * (len(v) - 1))] - 1:6.1%}"
              f"  within 25% {sum(x < math.log2(1.25) for x in v) / len(v):5.1%}")
    pooled.sort()
    med = 2 ** statistics.median(pooled) - 1
    print(f"  {'pooled':16s} n={len(pooled):4d} median |err| {med:6.1%}  p90 {2 ** pooled[int(0.9 * (len(pooled) - 1))] - 1:6.1%}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--fits", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--ref-set", action="append")
    ap.add_argument("--ref-fits")
    ap.add_argument("--ref-config")
    ap.add_argument("--threshold", type=float, default=0.20)
    a = ap.parse_args()
    groups = set(a.groups or [])
    rows = predictions(a.set, a.fits, a.config, groups)
    med = summary(rows, "predictor")
    print(f"A1: pooled median |err| {med:.1%} <= {a.threshold:.0%}: {'PASS' if med <= a.threshold else 'FAIL'}")
    if a.ref_set:
        ref = predictions(a.ref_set, a.ref_fits, a.ref_config, groups)
        summary(ref, "reference")
        common = sorted(set(rows) & set(ref))
        by_group = defaultdict(list)
        for k in common:
            d = abs(math.log2(rows[k][0] / rows[k][1])) - abs(math.log2(ref[k][0] / ref[k][1]))
            by_group[(k[0], k[1])].append(d)
        gs = list(by_group)
        rng = random.Random(0)
        boot = []
        for _ in range(4000):
            sample = [rng.choice(gs) for _ in gs]
            boot.append(statistics.fmean(x for g in sample for x in by_group[g]))
        boot.sort()
        point = statistics.fmean(x for g in gs for x in by_group[g])
        lo, hi = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]
        print(f"A2: mean |log2 ratio| predictor - reference {point:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(gs)} groups, "
              f"{len(common)} items: {'PASS' if hi < 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
