#!/usr/bin/env python3
"""Second-stage calibration: refit the knobs that decide which variant wins, on variant rankings.

The meter corpus fixes how long a kernel takes; it does not pin down the knobs
that decide which of two variants of one operation is faster (the weights of
the local data share, the matrix and vector-memory units, how a memory stall is
split between exposed and hidden). Several of those wander by orders of
magnitude between seeds of one corpus fit. This stage keeps every other knob of
a corpus fit and refits those on the development ranking groups: a logistic
loss on the stable pairs of each group (every group weighs the same), plus a
pull back toward the corpus fit's values so the absolute calibration survives.

  rank_fit.py --dev gemm:hw3:emu_v7,emu_h3wb triton:triton_hw:triton_emu_v7m,triton_emu_h3wb \\
              --fits "prereg7c-fit-s*.json" --out-prefix prereg8-fit
  rank_fit.py ... --folds 5          (cross-validation by group: report, write nothing)
"""

import argparse
import glob
import itertools
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))
sys.path.insert(0, str(TOOLS.parent))
import meter_refit as mr  # noqa: E402

# name: (low, high), searched in log10.
KNOBS = {
    "unit.local_data_share": (1e-4, 1.0),
    "unit.matrix_multiply": (1e-3, 10.0),
    "unit.vector_memory": (1e-3, 10.0),
    "unit.vector_alu": (1e-4, 1.0),
    "stall_scale": (1e-3, 10.0),
    "latency_scale": (1e-4, 1.0),
    "stall_overlap": (1.0, 32.0),
    "level.l1": (1e-2, 100.0),
    "level.l2": (0.1, 1e4),
}
SAMPLES = 3
NOISE = math.log2(1.10)
TAU = 0.05          # logistic temperature on log2 predicted ratios
PULL = 0.02         # weight of the squared log10 distance from the corpus fit, per knob

_S: dict = {}


def load_items(bench, hw_dir, emu_dirs):
    """Per (group, item): hardware runs and the warm dispatches of each call, as prereg_score.py scores them."""
    G, I = ("shape", "solution") if bench == "gemm" else ("case", "variant")
    calls = 4 if bench == "gemm" else 5
    hw = []
    for d in sorted(glob.glob(str(WORK / hw_dir / "run*"))):
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
        hw.append(run)
    items = {}
    for emu in emu_dirs:
        for p in glob.glob(str(WORK / emu / "*__*.json")):
            if p.endswith(".raw.json"):
                continue
            e = json.loads(Path(p).read_text())
            key = (e[G], e[I])
            if key in items or not all(key in h for h in hw):
                continue
            if bench == "gemm":
                same = set(e.get("kernels", [])) == {k for k in hw[0][key].get("kernels", []) if k.startswith("Cijk")}
                prefix = "Cijk"
            else:
                same = e.get("kernel_hash") and e.get("kernel_hash") == hw[0][key].get("kernel_hash") and \
                    e.get("max_err", 1) < 0.05 and hw[0][key].get("max_err", 1) < 0.05
                prefix = e.get("kernel_name") or "?"
            if not same:
                continue
            disp = []
            trace = Path(p[:-5] + ".trace.jsonl")
            for line in trace.read_text().splitlines() if trace.exists() else []:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(d.get("kernel", "")).startswith(prefix):
                    disp.append(d)
            if not disp:
                continue
            per = max(1, len(disp) // calls)
            warm = disp[-SAMPLES * per:]
            runs = [h[key]["kernel_us"] for h in hw]
            items[(f"{bench}:{key[0]}", key[1])] = dict(
                calls=[warm[i:i + per] for i in range(0, len(warm), per)], runs=runs, hw=statistics.median(runs))
    return items


def stable_pairs(items):
    """Per group: (faster, slower) pairs hardware separates by more than the noise and every run agrees on."""
    groups = defaultdict(list)
    for key in items:
        groups[key[0]].append(key)
    out = {}
    for g, keys in groups.items():
        pairs = []
        for a, b in itertools.combinations(keys, 2):
            ra, rb = items[a], items[b]
            if abs(math.log2(ra["hw"] / rb["hw"])) < NOISE or len({x < y for x, y in zip(ra["runs"], rb["runs"])}) > 1:
                continue
            pairs.append((a, b) if ra["hw"] < rb["hw"] else (b, a))
        if pairs:
            out[g] = pairs
    return out


def predict(item, tuning):
    return statistics.median(sum(mr.recompose(d, tuning) for d in call) for call in item["calls"])


def decode(x):
    return {n: 10.0 ** v for n, v in zip(_S["names"], x)}


def loss(x):
    tuning = dict(_S["start"])
    tuning.update(decode(x))
    cache = {}
    per_group = []
    for g in _S["train_groups"]:
        total = 0.0
        for fast, slow in _S["pairs"][g]:
            for k in (fast, slow):
                if k not in cache:
                    p = predict(_S["items"][k], tuning)
                    if not p > 0:
                        return 1e9
                    cache[k] = math.log2(p)
            margin = (cache[fast] - cache[slow]) / TAU
            total += math.log1p(math.exp(margin)) if margin < 30 else margin
        per_group.append(total / len(_S["pairs"][g]))
    pull = sum((v - math.log10(_S["start"][n])) ** 2 for n, v in zip(_S["names"], x))
    return statistics.fmean(per_group) + PULL * pull


def init_worker(state):
    _S.update(state)


def fit(start, train_groups, items, pairs, seed, workers, maxiter, popsize):
    names = list(KNOBS)
    state = dict(start=start, names=names, train_groups=train_groups, items=items, pairs=pairs)
    _S.update(state)
    bounds = []
    for n in names:
        lo, hi = KNOBS[n]
        v = min(max(start.get(n, 1.0), lo), hi)
        start[n] = v
        bounds.append((math.log10(lo), math.log10(hi)))
    x0 = np.array([math.log10(start[n]) for n in names])
    with multiprocessing.get_context("fork").Pool(workers, initializer=init_worker, initargs=(state,)) as pool:
        res = differential_evolution(loss, bounds, maxiter=maxiter, popsize=popsize, seed=seed, tol=1e-6,
                                     polish=False, updating="deferred", workers=pool.map, x0=x0, disp=False)
    return decode(res.x), res.fun, loss(x0)


def score(groups, items, pairs, tunings):
    """Pairwise accuracy over the groups' stable pairs and median pick regret, for an ensemble of tunings."""
    preds = {}
    for g in groups:
        for k in {k for p in pairs[g] for k in p} | {k for k in items if k[0] == g}:
            preds[k] = math.exp(statistics.fmean(math.log(predict(items[k], t)) for t in tunings))
    ok = [float(preds[f] < preds[s]) for g in groups for f, s in pairs[g]]
    regrets = []
    for g in groups:
        keys = [k for k in items if k[0] == g]
        best = min(items[k]["hw"] for k in keys)
        regrets.append(items[min(keys, key=preds.get)]["hw"] / best - 1)
    return statistics.fmean(ok), statistics.median(regrets), len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True, help="BENCH:HWDIR:EMUDIR[,EMUDIR...]")
    ap.add_argument("--fits", required=True, help="corpus fits to start from (one second stage each)")
    ap.add_argument("--out-prefix")
    ap.add_argument("--folds", type=int, default=0, help="cross-validate by group instead of writing fits")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--maxiter", type=int, default=60)
    ap.add_argument("--popsize", type=int, default=10)
    args = ap.parse_args()

    base = mr.config_defaults(mr.load_config(str(ROOT / "configs/gfx950_mi355x_kmd_timing.json")))
    items = {}
    for spec in args.dev:
        bench, hw_dir, emus = spec.split(":")
        items.update(load_items(bench, hw_dir, emus.split(",")))
    pairs = stable_pairs(items)
    groups = sorted(pairs)
    print(f"{len(items)} items, {len(groups)} groups, {sum(len(v) for v in pairs.values())} stable pairs", flush=True)
    starts = []
    for f in sorted(glob.glob(str(WORK / args.fits))):
        t = dict(base)
        t.update(json.loads(Path(f).read_text())["knobs"])
        starts.append((f, t))

    if args.folds:
        rng = random.Random(0)
        order = groups[:]
        rng.shuffle(order)
        folds = [order[i::args.folds] for i in range(args.folds)]
        rows = []
        for i, test in enumerate(folds):
            train = [g for g in groups if g not in test]
            fitted = []
            for k, (f, start) in enumerate(starts):
                knobs, _, _ = fit(dict(start), train, items, pairs, seed=k + 1, workers=args.workers,
                                  maxiter=args.maxiter, popsize=args.popsize)
                t = dict(start)
                t.update(knobs)
                fitted.append(t)
            for lib in ("gemm", "triton"):
                tg = [g for g in test if g.startswith(lib)]
                if not tg:
                    continue
                before = score(tg, items, pairs, [s for _, s in starts])
                after = score(tg, items, pairs, fitted)
                rows.append((lib, before, after))
                print(f"fold {i} {lib:6s} groups {len(tg):2d}  corpus fit: acc {before[0]:.1%} regret {before[1]:.1%}"
                      f"  ->  second stage: acc {after[0]:.1%} regret {after[1]:.1%}  ({after[2]} pairs)", flush=True)
        for lib in ("gemm", "triton"):
            r = [x for x in rows if x[0] == lib]
            n = sum(x[2][2] for x in r)
            acc_b = sum(x[1][0] * x[1][2] for x in r) / n
            acc_a = sum(x[2][0] * x[2][2] for x in r) / n
            print(f"CV {lib:6s}: pairwise accuracy {acc_b:.1%} -> {acc_a:.1%}; median of fold median regrets "
                  f"{statistics.median(x[1][1] for x in r):.1%} -> {statistics.median(x[2][1] for x in r):.1%}")
        return

    for k, (f, start) in enumerate(starts, 1):
        knobs, fun, fun0 = fit(dict(start), groups, items, pairs, seed=k, workers=args.workers,
                               maxiter=args.maxiter, popsize=args.popsize)
        src = json.loads(Path(f).read_text())
        src["knobs"].update(knobs)
        src["rank_fit"] = {"from": os.path.basename(f), "loss": fun, "loss_at_start": fun0, "knobs": list(KNOBS),
                           "dev": args.dev, "tau": TAU, "pull": PULL}
        out = WORK / f"{args.out_prefix}-s{k}.json"
        out.write_text(json.dumps(src, indent=1))
        print(f"{out.name}: loss {fun0:.4f} -> {fun:.4f}  " + " ".join(f"{n}={v:.3g}" for n, v in knobs.items()), flush=True)


if __name__ == "__main__":
    main()
