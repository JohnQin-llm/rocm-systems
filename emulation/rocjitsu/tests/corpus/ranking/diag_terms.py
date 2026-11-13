#!/usr/bin/env python3
"""Why does the fitted model mis-rank some groups? Decompose its prediction.

Re-implements meter_refit.recompose() term by term (and checks the sum matches)
under one fitted tuning, then for each group reports, per term: its median share
of the predicted total and its ordering accuracy on the group's stable pairs
when used alone. Also reports which term was the throughput maximum.
"""

import argparse
import glob
import itertools
import json
import os
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("bench", choices=("gemm", "triton"))
ap.add_argument("--hw", required=True)
ap.add_argument("--emu", required=True)
ap.add_argument("--fit", default="prereg-fit-s1.json")
ap.add_argument("--groups", nargs="*")
a = ap.parse_args()
TOOLS = Path(__file__).resolve().parent                        # this directory
ROOT = TOOLS.parents[2]                                          # emulation/rocjitsu
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
B = WORK
sys.path.insert(0, str(TOOLS.parent))
import meter_refit as mr  # noqa: E402

T = mr.config_defaults(mr.load_config(str(ROOT / "configs/gfx950_mi355x_kmd_timing.json")))
T.update(json.loads((B / a.fit).read_text())["knobs"])


def parts(d, t):
    """meter_refit.recompose(), returning its terms."""
    fixed = t.get("fixed_scale", 1.0) * d["fixed"] + t.get("fixed_add", 0.0)
    if d["waves"] == 0:
        return {"fixed": fixed}
    wt = max(1.0, float(d["waves"]))
    issue = mr._issue(d["unit_cycles"], t, mr._front_end(d, float(d["unit_waves"]) / wt, t))
    scaled = mr._issue(d["latency_unit_cycles"], t, mr._front_end(d, float(d["latency_waves"]) / wt, t))
    stall = float(d["latency_worst_stall"]) * t.get("stall_scale", 1.0)
    critical = max(0.0, float(d["latency_critical"]) - float(d["latency_worst_stall"])) + stall
    waves = max(1.0, float(d["latency_waves"]))
    resident = max(1.0, min(waves, float(d["resident"])))
    rounds = math.ceil(waves / resident)
    hidden = (resident - 1.0) * (scaled / waves)
    lock = t.get("barrier_lockstep", 0.0)
    if lock > 0.0:
        share = float(d["latency_waves"]) / max(1.0, float(d["waves"]))
        pwb = d["class_counts"].get("barrier", 0) * share / max(1.0, float(d["latency_waves"]))
        hidden /= 1.0 + lock * pwb
    exposed = max(0.0, critical - hidden)
    overlap = max(1.0, min(resident, t.get("stall_overlap", 1.0)))
    exposed = max(exposed, stall / overlap)
    latency = exposed * rounds * t.get("latency_scale", 1.0)
    bw = 0.0
    for lv in ("l1", "l2", "fabric", "mall", "dram"):
        bw = max(bw, d[f"{lv}_cycles"] / (t.get("bandwidth_scale", 1.0) * t.get(f"level.{lv}", 1.0)))
    work = t.get("work_scale", 1.0)
    issue *= work
    bw *= work
    placement = float(d["placement"]) * t.get("placement_scale", 1.0)
    g = t.get("issue_occupancy_exponent", 0.0)
    if g != 0.0:
        issue *= (max(1.0, float(d["resident"])) / 8.0) ** g
    fill = d["fill"] * t.get("fill_scale", 1.0)
    lat_total = max(latency, fill)
    if t.get("straggler_over_workgroups", 0.0) > 0.0:
        base = t.get("straggler_base", 0.0) * t.get("straggler_scale", 1.0)
        grp = max(1.0, float(d["workgroups"]))
        lat_total += base * math.log2(grp) if grp > 1 else 0.0
    else:
        lat_total += d.get("straggler", 0.0) * t.get("straggler_scale", 1.0)
    per_barrier = t.get("barrier_cycles", 0.0) - t.get("barrier_base", 0.0)
    if per_barrier != 0.0:
        wh = max(1.0, float(d["waves"]))
        sh = float(d["latency_waves"]) / wh
        lat_total += per_barrier * d["class_counts"].get("barrier", 0) * sh / max(1.0, float(d["latency_waves"]))
    out = {"fixed": fixed, "issue": issue, "bandwidth": bw, "placement": placement,
           "latency": lat_total, "filling": d["filling"] * t.get("filling_scale", 1.0)}
    out["_max"] = max(("issue", "bandwidth", "placement"), key=lambda k: out[k])
    return out


G, I = ("shape", "solution") if a.bench == "gemm" else ("case", "variant")
hw = []
for dd in sorted(glob.glob(str(B / a.hw / "run*"))):
    run = {}
    for p in glob.glob(dd + "/*.json"):
        if p.endswith(".guard.json"):
            continue
        g = Path(p[:-5] + ".guard.json")
        if g.exists() and not json.loads(g.read_text())["contended"]:
            r = json.loads(Path(p).read_text())
            if r.get("kernel_us"):
                run[(r[G], r[I])] = r
    hw.append(run)
rows = {}
for p in glob.glob(str(B / a.emu / "*__*.json")):
    if p.endswith(".raw.json"):
        continue
    e = json.loads(Path(p).read_text())
    key = (e[G], e[I])
    if a.groups and key[0] not in a.groups or not all(key in h for h in hw):
        continue
    prefix = "Cijk" if a.bench == "gemm" else (e.get("kernel_name") or "?")
    tr = Path(p[:-5] + ".trace.jsonl")
    agg, maxes, n = defaultdict(float), Counter(), 0
    for line in tr.read_text().splitlines() if tr.exists() else []:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(d.get("kernel", "")).startswith(prefix):
            pp = parts(d, T)
            tot = mr.recompose(d, T)
            assert abs(sum(v for k, v in pp.items() if not k.startswith("_") and k != "throughput")
                       - (pp["fixed"] + max(pp["issue"], pp["bandwidth"], pp["placement"]) + pp["latency"]
                          + pp["filling"])) >= 0
            calc = pp["fixed"] + max(pp["issue"], pp["bandwidth"], pp["placement"]) + pp["latency"] + pp["filling"]
            assert abs(calc - tot) <= 1e-6 * max(1.0, tot), (calc, tot)
            for k, v in pp.items():
                if not k.startswith("_"):
                    agg[k] += v
            agg["total"] += tot
            agg["waves"] += d["waves"]
            agg["workgroups"] += d["workgroups"]
            maxes[pp["_max"]] += 1
            n += 1
    if n:
        agg["throughput"] = max(agg["issue"], agg["bandwidth"], agg["placement"])
        rows[key] = {"hw": statistics.median(h[key]["kernel_us"] for h in hw),
                     "runs": [h[key]["kernel_us"] for h in hw], "t": dict(agg), "max": maxes}

TERMS = ["total", "fixed", "issue", "bandwidth", "placement", "latency", "filling", "workgroups"]
thr = math.log2(1.10)
for grp in sorted({k[0] for k in rows}):
    keys = [k for k in rows if k[0] == grp]
    sc, npairs = defaultdict(float), 0
    for x, y in itertools.combinations(keys, 2):
        rx, ry = rows[x], rows[y]
        if abs(math.log2(rx["hw"] / ry["hw"])) < thr or len({p < q for p, q in zip(rx["runs"], ry["runs"])}) > 1:
            continue
        npairs += 1
        truth = rx["hw"] < ry["hw"]
        for t in TERMS:
            vx, vy = rx["t"][t], ry["t"][t]
            sc[t] += 0.5 if vx == vy else float((vx < vy) == truth)
    share = {t: statistics.median(r["t"][t] / r["t"]["total"] for r in (rows[k] for k in keys))
             for t in TERMS[1:-1]}
    maxc = sum((rows[k]["max"] for k in keys), Counter())
    print(f"\n== {grp}: {len(keys)} items, {npairs} stable pairs, throughput term was max: {dict(maxc)}")
    print("   term        alone-acc  median share of predicted total")
    for t in TERMS:
        print(f"   {t:11s} {sc[t] / max(1, npairs):8.0%}   " + (f"{share[t]:6.1%}" if t in share else ""))
    # Hardware best vs model's pick, with their terms
    best = min(keys, key=lambda k: rows[k]["hw"])
    pick = min(keys, key=lambda k: rows[k]["t"]["total"])
    for tag, k in (("hw best", best), ("model pick", pick)):
        r = rows[k]
        print(f"   {tag:10s} {k[1][:24]:24s} hw {r['hw']:7.2f}us  pred {r['t']['total']:10.0f}  "
              + " ".join(f"{t}={r['t'][t]:.0f}" for t in ("fixed", "issue", "bandwidth", "latency", "workgroups")))
