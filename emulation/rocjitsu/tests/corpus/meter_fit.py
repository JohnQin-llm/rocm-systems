#!/usr/bin/env python3
"""Fit meter_refit's composition knobs against a hardware reference.

meter_refit.py recomposes every recorded dispatch under a candidate tuning in
about a second, but leaves the search to a human with --set.  This does the
search: differential evolution over the knobs in log space, scored on
log2(model / measured) per case.

Two things keep the fit honest:

* Cases are split into train and test by *variant group* (case_id with the
  implementation segment wildcarded), so the eager and compiled versions of
  one case always land on the same side and the ranking test pairs are never
  seen in training.
* The loss is averaged per category first, so the 50-odd elementwise cases
  do not outvote the six quantized ones.

Output: the fitted knobs (as meter_refit --set assignments) and a report of
absolute accuracy and variant-pair ordering on both splits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing
import os
import random
import statistics
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import differential_evolution

# name: (low, high, log-scaled?)  Multipliers act on the recorded,
# uncalibrated terms, so their bounds are wide on purpose.
KNOBS: dict[str, tuple[float, float, bool]] = {
    "fixed_scale": (1e-3, 10.0, True),
    "fixed_add": (0.0, 20000.0, False),
    "unit.none": (1e-3, 10.0, True),
    "unit.vector_alu": (1e-3, 10.0, True),
    "unit.scalar_alu": (1e-3, 10.0, True),
    "unit.transcendental": (1e-3, 10.0, True),
    "unit.matrix_multiply": (1e-3, 10.0, True),
    "unit.local_data_share": (1e-3, 10.0, True),
    "unit.vector_memory": (1e-3, 10.0, True),
    "unit.scalar_memory": (1e-3, 10.0, True),
    "stall_scale": (1e-3, 10.0, True),
    "latency_scale": (1e-3, 10.0, True),
    "stall_overlap": (1.0, 32.0, True),
    "level.l1": (0.1, 1e4, True),
    "level.l2": (0.1, 1e4, True),
    "level.fabric": (0.1, 1e4, True),
    "level.mall": (0.1, 1e4, True),
    "level.dram": (0.1, 1e4, True),
    "placement_scale": (1e-3, 10.0, True),
    "fill_scale": (1e-3, 10.0, True),
    "filling_scale": (1e-3, 10.0, True),
    "flush_scale": (1e-3, 10.0, True),
    "straggler_scale": (1e-3, 10.0, True),
    "issue_occupancy_exponent": (-1.0, 1.0, False),
}

_STATE: dict = {}


def variant_key(case_id: str, impl: str) -> str:
    token = f".{impl}."
    return case_id.replace(token, ".*.", 1) if token in case_id else case_id


def decode(x: np.ndarray, names: list[str]) -> dict[str, float]:
    out = {}
    for value, name in zip(x, names):
        lo, hi, logged = KNOBS[name]
        out[name] = 10.0 ** value if logged else value
    return out


def bounds_for(names: list[str]) -> list[tuple[float, float]]:
    out = []
    for name in names:
        lo, hi, logged = KNOBS[name]
        out.append((math.log10(lo), math.log10(hi)) if logged else (lo, hi))
    return out


def predict(case: dict, tuning: dict) -> float:
    refit = _STATE["refit"]
    return refit.case_us(case, refit.recompose, tuning, _STATE["clock_mhz"])


def loss(x: np.ndarray) -> float:
    names = _STATE["names"]
    tuning = dict(_STATE["base"])
    tuning.update(decode(x, names))
    per_cat: dict[str, list[float]] = defaultdict(list)
    delta = _STATE["huber"]
    for case, measured, category in _STATE["train"]:
        pred = predict(case, tuning)
        if not pred > 0:
            return 1e9
        r = abs(math.log2(pred / measured))
        per_cat[category].append(0.5 * r * r if r <= delta else delta * (r - 0.5 * delta))
    return statistics.fmean(statistics.fmean(v) for v in per_cat.values())


def init_worker(state: dict) -> None:
    _STATE.update(state)
    sys.path.insert(0, state["corpus_dir"])
    import meter_refit  # noqa: E402

    _STATE["refit"] = meter_refit


def evaluate(rows: list[tuple], tuning: dict, label: str, noise_pct: float) -> dict:
    preds = {}
    for case, measured, category in rows:
        preds[case["case_id"]] = (predict(case, tuning), measured, category)
    lr = [abs(math.log2(p / m)) for p, m, _ in preds.values()]
    lr.sort()
    within20 = sum(1 for p, m, _ in preds.values() if abs(p / m - 1) <= 0.2)
    within2x = sum(1 for v in lr if v <= 1.0)
    groups = defaultdict(list)
    for case_id in preds:
        groups[_STATE["vkey"][case_id]].append(case_id)
    thr = math.log2(1 + noise_pct / 100)
    n_pairs = ok = 0
    errs = []
    for members in groups.values():
        for a, b in itertools.combinations(sorted(members), 2):
            pa, ma, _ = preds[a]
            pb, mb, _ = preds[b]
            real_lr = math.log2(ma / mb)
            if abs(real_lr) < thr:
                continue
            model_lr = math.log2(pa / pb)
            n_pairs += 1
            ok += (real_lr > 0) == (model_lr > 0)
            errs.append(abs(model_lr - real_lr))
    out = {
        "split": label, "cases": len(lr),
        "median_abs_log2": statistics.median(lr) if lr else float("nan"),
        "p90_abs_log2": lr[int(0.9 * (len(lr) - 1))] if lr else float("nan"),
        "within_20pct": within20, "within_2x": within2x,
        "pairs": n_pairs, "pairs_correct": ok,
        "speedup_err_median": statistics.median(errs) if errs else float("nan"),
    }
    print(f"[{label:5}] cases {out['cases']:3d}  median|log2| {out['median_abs_log2']:.3f} "
          f"(x{2 ** out['median_abs_log2']:.2f})  p90 {out['p90_abs_log2']:.2f}  "
          f"within20% {within20}/{len(lr)}  within2x {within2x}/{len(lr)}  "
          f"pairs {ok}/{n_pairs}" + (f" = {ok / n_pairs:.1%}" if n_pairs else "")
          + (f"  speedup-err {out['speedup_err_median']:.2f} oct" if errs else ""))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="directory holding meter_refit.py (default: this script's)")
    parser.add_argument("--shards", action="append", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--clock-mhz", type=float, default=2100.0)
    parser.add_argument("--test-frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--huber", type=float, default=0.5)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--noise-pct", type=float, default=10.0)
    parser.add_argument("--knobs", help="comma list; default all")
    parser.add_argument("--out", required=True, help="JSON with fitted knobs and scores")
    args = parser.parse_args(argv)

    state = {"corpus_dir": os.path.abspath(args.corpus_dir), "clock_mhz": args.clock_mhz,
             "huber": args.huber}
    init_worker(state)
    refit = _STATE["refit"]
    ms = refit.meter_score

    base = refit.config_defaults(refit.load_config(args.config))
    cases = []
    for shard_dir in args.shards:
        cases.extend(refit.load_cases(shard_dir, args.traces, args.clock_mhz))
    real_by_id = ms.passed_results(ms.load(args.real))
    rows, vkey = [], {}
    seen = set()
    for case in cases:
        if case["case_id"] in seen:
            continue
        record = real_by_id.get(case["case_id"])
        measured = ms.kernel_us(record) if record else None
        if not measured:
            continue
        seen.add(case["case_id"])
        vkey[case["case_id"]] = variant_key(case["case_id"], record.get("implementation", ""))
        rows.append((case, measured, case["category"]))

    keys = sorted(set(vkey.values()))
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    test_keys = set(keys[: int(len(keys) * args.test_frac)])
    train = [r for r in rows if vkey[r[0]["case_id"]] not in test_keys]
    test = [r for r in rows if vkey[r[0]["case_id"]] in test_keys]
    names = args.knobs.split(",") if args.knobs else list(KNOBS)
    print(f"cases {len(rows)}  train {len(train)}  test {len(test)}  knobs {len(names)}")

    state.update({"base": base, "names": names, "train": train, "vkey": vkey})
    _STATE.update(state)
    evaluate(train, base, "train", args.noise_pct)
    evaluate(test, base, "test", args.noise_pct)

    # Fork explicitly: the default start method here is not fork, and the
    # workers need _STATE (the joined traces) without re-pickling it per call.
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        result = differential_evolution(
            loss, bounds_for(names), maxiter=args.maxiter, popsize=args.popsize,
            seed=args.seed, tol=1e-6, polish=False, updating="deferred",
            workers=pool.map, disp=False,
        )
    fitted = decode(result.x, names)
    tuning = dict(base)
    tuning.update(fitted)
    print(f"\nDE: loss {result.fun:.4f}  nfev {result.nfev}  {result.message}")
    for name in names:
        print(f"  {name:28s} {fitted[name]:.6g}")
    scores = [evaluate(train, tuning, "train", args.noise_pct),
              evaluate(test, tuning, "test", args.noise_pct),
              evaluate(rows, tuning, "all", args.noise_pct)]

    per_case = []
    for case, measured, category in rows:
        pred = predict(case, tuning)
        per_case.append({"case_id": case["case_id"], "category": category,
                         "split": "test" if vkey[case["case_id"]] in test_keys else "train",
                         "measured_us": measured, "model_us": pred,
                         "log2_ratio": math.log2(pred / measured)})
    with open(args.out, "w") as handle:
        json.dump({"knobs": fitted, "set_args": [f"{k}={v:.6g}" for k, v in fitted.items()],
                   "scores": scores, "cases": per_case, "seed": args.seed,
                   "test_frac": args.test_frac}, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
