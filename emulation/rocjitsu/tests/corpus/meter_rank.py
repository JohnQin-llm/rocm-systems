#!/usr/bin/env python3
"""Score a rocjitsu timing model on *ranking*, not absolute accuracy.

meter_score.py asks "how far is the modelled time from the measured one".  For
kernel selection (an agent or autotuner choosing between implementations of the
same op) the question is narrower: among variants of one case, does the model
put them in the same order as hardware, and does it get the speedup roughly
right?

Variants are cases whose case_id differs only in the implementation segment
(``swiglu.compiled.bfloat16.1x768`` vs ``swiglu.triton.bfloat16.1x768``).

Uses meter_score's own extraction (predicted = emulated event_us, measured =
real kernel_us) so the two scripts never disagree about what a case costs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
from collections import defaultdict


def load_meter_score(corpus_dir: str):
    sys.path.insert(0, corpus_dir)
    import meter_score  # noqa: E402

    return meter_score


def variant_key(record: dict) -> str:
    impl = record.get("implementation", "")
    case_id = record["case_id"]
    token = f".{impl}."
    return case_id.replace(token, ".*.", 1) if token in case_id else case_id


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = rankdata(xs), rankdata(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", required=True)
    parser.add_argument("--emulated", required=True, nargs="+")
    parser.add_argument("--corpus-dir", default=here,
                        help="directory holding meter_score.py")
    parser.add_argument("--noise-pct", type=float, default=10.0,
                        help="ignore pairs whose hardware times differ by less")
    parser.add_argument("--show", type=int, default=15,
                        help="worst mis-ordered pairs to print")
    parser.add_argument("--json", help="write the scored pairs here")
    args = parser.parse_args(argv)

    ms = load_meter_score(args.corpus_dir)
    real = ms.load(args.real)
    emul = ms.merge(args.emulated)
    real_by_id = ms.passed_results(real)
    emul_by_id = ms.passed_results(emul)

    cases = {}
    for case_id in sorted(set(real_by_id) & set(emul_by_id)):
        measured = ms.kernel_us(real_by_id[case_id])
        predicted = ms.event_us(emul_by_id[case_id])
        if measured and predicted and measured > 0 and predicted > 0:
            cases[case_id] = (measured, predicted, real_by_id[case_id])

    groups: dict[str, list[str]] = defaultdict(list)
    for case_id, (_, _, record) in cases.items():
        groups[variant_key(record)].append(case_id)

    threshold = math.log2(1.0 + args.noise_pct / 100.0)
    pairs = []
    for key, members in groups.items():
        for a, b in itertools.combinations(sorted(members), 2):
            ma, pa, _ = cases[a]
            mb, pb, _ = cases[b]
            real_lr = math.log2(ma / mb)
            pred_lr = math.log2(pa / pb)
            if abs(real_lr) < threshold:
                continue
            pairs.append({
                "group": key, "a": a, "b": b,
                "real_us": [ma, mb], "model_us": [pa, pb],
                "real_log2_ratio": real_lr, "model_log2_ratio": pred_lr,
                "correct": (real_lr > 0) == (pred_lr > 0),
                "speedup_err_octaves": abs(pred_lr - real_lr),
            })

    measured = [v[0] for v in cases.values()]
    predicted = [v[1] for v in cases.values()]
    print(f"scored cases: {len(cases)}  variant groups with >=2 members: "
          f"{sum(1 for m in groups.values() if len(m) > 1)}")
    print(f"global spearman (all cases, informative only): "
          f"{spearman(measured, predicted):.3f}")
    if not pairs:
        print("no variant pairs above the noise threshold")
        return 0
    correct = sum(p["correct"] for p in pairs)
    errs = sorted(p["speedup_err_octaves"] for p in pairs)
    big = [p for p in pairs if abs(p["real_log2_ratio"]) >= 1.0]
    print(f"variant pairs beyond {args.noise_pct:.0f}% hw difference: {len(pairs)}")
    print(f"  ordered correctly: {correct}/{len(pairs)} = {correct / len(pairs):.1%}")
    if big:
        bc = sum(p["correct"] for p in big)
        print(f"  of which hw differs >=2x: {bc}/{len(big)} = {bc / len(big):.1%}")
    print(f"  speedup error |log2 model/real ratio|: median {statistics.median(errs):.2f} "
          f"octaves, p90 {errs[int(0.9 * (len(errs) - 1))]:.2f}")

    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_kernel[p["group"].split(".", 1)[0]].append(p)
    print(f"\n{'kernel':<24} {'pairs':>5} {'correct':>8} {'med err':>8}")
    for kernel, ps in sorted(by_kernel.items()):
        ok = sum(p["correct"] for p in ps)
        med = statistics.median(p["speedup_err_octaves"] for p in ps)
        print(f"{kernel:<24} {len(ps):>5} {ok / len(ps):>8.0%} {med:>8.2f}")

    wrong = sorted((p for p in pairs if not p["correct"]),
                   key=lambda p: -abs(p["real_log2_ratio"]))
    if wrong and args.show:
        print(f"\nworst mis-ordered pairs (by hardware gap):")
        for p in wrong[: args.show]:
            print(f"  {p['a']}  vs  {p['b']}")
            print(f"     real {p['real_us'][0]:9.2f} / {p['real_us'][1]:9.2f} us   "
                  f"model {p['model_us'][0]:9.2f} / {p['model_us'][1]:9.2f} us")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"noise_pct": args.noise_pct, "pairs": pairs}, handle,
                      indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
