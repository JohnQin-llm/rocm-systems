#!/usr/bin/env python3
"""Do the emulator's cache-traffic counts match hardware counters?

  counters.py collect GROUP [GROUP ...] --hips 5,6,7     (hardware, guarded)
  counters.py compare GROUP [GROUP ...]

collect: for every pick of the groups, run the bench harness under rocprofv3
with L1->L2, L2 hit/miss and L2->fabric read counters (two passes), and keep the
per-dispatch median over the timed (warm) dispatches.
compare: per pick, the emulator's warm per-call l1/l2/fabric bytes next to the
hardware counters, and how well each emulator quantity tracks its hardware
counterpart across the picks of a group (Spearman).
"""

import csv
import glob
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent                        # this directory
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
OUT = WORK / "counters"
PASSES = [["TCC_HIT_sum", "TCC_MISS_sum", "TCP_TCC_READ_REQ_sum"],
          ["TCC_EA0_RDREQ_sum", "TCC_EA0_RDREQ_32B_sum", "TCC_EA0_RDREQ_64B_sum", "TCC_EA0_RDREQ_128B_sum"]]


def items(groups):
    sel = json.loads((WORK / "selected.json").read_text())
    tri = json.loads(subprocess.run([sys.executable, str(TOOLS / "triton_bench.py"), "list"],
                                    capture_output=True, text=True, check=True).stdout)
    out = []
    for g in groups:
        if g in sel:
            out += [("gemm", g, s) for s in sel[g]]
        else:
            out += [("triton", g, v) for v in tri[g]]
    return out


def harness(kind, g, v, rec):
    if kind == "gemm":
        return [sys.executable, str(TOOLS / "gemm_solutions.py"), "time", "--meta", str(WORK / "candidates.json"),
                "--shape", g, "--solution", v, "--mode", "hw", "--warmup", "3", "--samples", "5",
                "--profile-iters", "1", "--workdir", "/tmp", "--out", rec]
    return [sys.executable, str(TOOLS / "triton_bench.py"), "time", "--case", g, "--variant", v, "--mode", "hw",
            "--warmup", "3", "--samples", "5", "--profile-iters", "1", "--out", rec]


def collect_one(args):
    (kind, g, v), hip = args
    dst = OUT / f"{g}__{v}.json"
    if dst.exists():
        return
    res = {}
    for i, pmc in enumerate(PASSES):
        d = OUT / "raw" / f"{g}__{v}" / f"p{i}"
        for _ in range(5):
            subprocess.run(["rm", "-rf", str(d)])
            cmd = [sys.executable, str(TOOLS / "gpu_guard.py"), "run", "--hip", str(hip), "--record",
                   str(d) + ".guard.json", "--wait", "1800", "--", os.environ.get("ROCPROFV3", "/opt/rocm/bin/rocprofv3"), "--pmc", *pmc,
                   "-d", str(d), "-o", "c", "--output-format", "csv", "--"] + harness(kind, g, v, f"/tmp/cnt_{hip}.json")
            d.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not json.loads(Path(str(d) + ".guard.json").read_text())["contended"]:
                break
        files = glob.glob(str(d) + "/**/*counter_collection.csv", recursive=True)
        if not files:
            return
        per = {}
        with open(files[0]) as fh:
            for r in csv.DictReader(fh):
                name = r.get("Kernel_Name", "")
                if not (name.startswith("Cijk") or "matmul" in name or "softmax" in name):
                    continue
                per.setdefault(int(r["Dispatch_Id"]), {"_name": name})[r["Counter_Name"]] = float(r["Counter_Value"])
        res.update(per_call(per, pmc))
    dst.write_text(json.dumps({"group": g, "item": v, "kind": kind, **res}))


def per_call(per, pmc):
    """Counters of one timed call: a call may launch several kernels (a GSU
    solution adds a reduction kernel), so take each kernel's median over its
    last 5 dispatches (the timed calls) and sum over kernels."""
    by_name = {}
    for k in sorted(per):
        by_name.setdefault(per[k]["_name"], []).append(per[k])
    out = {}
    for c in pmc:
        tot, seen = 0.0, False
        for ds in by_name.values():
            vals = [d[c] for d in ds[-5:] if c in d]
            if vals:
                tot += statistics.median(vals)
                seen = True
        if seen:
            out[c] = tot
    return out


def reparse():
    """Rebuild counters/*.json from the kept rocprofv3 CSVs."""
    for d in sorted((OUT / "raw").iterdir()):
        if not d.is_dir():
            continue
        g, v = d.name.split("__", 1)
        old = OUT / f"{d.name}.json"
        rec = json.loads(old.read_text()) if old.exists() else {"group": g, "item": v}
        for i, pmc in enumerate(PASSES):
            files = glob.glob(str(d / f"p{i}") + "/**/*counter_collection.csv", recursive=True)
            if not files:
                continue
            per = {}
            with open(files[0]) as fh:
                for r in csv.DictReader(fh):
                    name = r.get("Kernel_Name", "")
                    if name.startswith("Cijk") or "matmul" in name or "softmax" in name:
                        per.setdefault(int(r["Dispatch_Id"]), {"_name": name})[r["Counter_Name"]] = float(r["Counter_Value"])
            rec.update(per_call(per, pmc))
        old.write_text(json.dumps(rec))


def emu_traffic(kind, g, v):
    sfx = os.environ.get("EMU_SUFFIX", "")
    base = WORK / (("emu" if kind == "gemm" else "triton_emu") + sfx) / f"{g}__{v}"
    if not Path(str(base) + ".trace.jsonl").exists():
        return None
    calls = 4 if kind == "gemm" else 5
    e = json.loads(Path(str(base) + ".json").read_text())
    pre = "Cijk" if kind == "gemm" else (e.get("kernel_name") or "?")
    ds = [json.loads(l) for l in open(str(base) + ".trace.jsonl") if l.strip()]
    ds = [d for d in ds if str(d.get("kernel", "")).startswith(pre)]
    per = max(1, len(ds) // calls)
    warm = ds[-3 * per:]
    out = {}
    for k in ("l1_bytes", "l2_bytes", "fabric_bytes", "dram_bytes", "instructions"):
        out[k] = statistics.median(sum(d.get(k, 0) for d in warm[i:i + per]) for i in range(0, len(warm), per))
    return out


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for i, j in enumerate(o):
            r[j] = i
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def compare(groups):
    for g in groups:
        rows = []
        for p in sorted(OUT.glob(f"{g}__*.json")):
            h = json.loads(p.read_text())
            e = emu_traffic(h["kind"], g, h["item"])
            if not e or "TCC_HIT_sum" not in h or "TCC_EA0_RDREQ_sum" not in h:
                continue
            hw_l2_req = h["TCC_HIT_sum"] + h["TCC_MISS_sum"]
            hw_fabric = 32 * h.get("TCC_EA0_RDREQ_32B_sum", 0) + 64 * h.get("TCC_EA0_RDREQ_64B_sum", 0) + \
                128 * h.get("TCC_EA0_RDREQ_128B_sum", 0)
            rows.append((h["item"], e, hw_l2_req, h["TCP_TCC_READ_REQ_sum"], hw_fabric))
        if not rows:
            print(f"{g}: no data")
            continue
        print(f"\n== {g}: {len(rows)} picks")
        print(f"   {'item':26s} {'emu l2 MB':>9s} {'hw L2 req':>10s} {'emu fab MB':>10s} {'hw fab MB':>9s}  ratio fab")
        for item, e, l2r, tcp, fab in rows:
            print(f"   {item[:26]:26s} {e['l2_bytes'] / 1e6:9.2f} {l2r:10.0f} {e['fabric_bytes'] / 1e6:10.2f} "
                  f"{fab / 1e6:9.2f}  {e['fabric_bytes'] / fab if fab else float('nan'):6.2f}")
        print(f"   spearman emu l2_bytes vs hw L2 requests:      {spearman([r[1]['l2_bytes'] for r in rows], [r[2] for r in rows]):.2f}")
        print(f"   spearman emu l1->l2 (l2_bytes) vs TCP_TCC_READ_REQ: {spearman([r[1]['l2_bytes'] for r in rows], [r[3] for r in rows]):.2f}")
        print(f"   spearman emu fabric_bytes vs hw fabric bytes: {spearman([r[1]['fabric_bytes'] for r in rows], [r[4] for r in rows]):.2f}")


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    hips = [5]
    if "--hips" in rest:
        i = rest.index("--hips")
        hips = [int(x) for x in rest[i + 1].split(",")]
        rest = rest[:i] + rest[i + 2:]
    if mode == "collect":
        OUT.mkdir(parents=True, exist_ok=True)
        its = items(rest)
        with ThreadPoolExecutor(max_workers=len(hips)) as pool:
            # one item at a time per GPU
            list(pool.map(collect_one, [(it, hips[i % len(hips)]) for i, it in enumerate(its)]))
        print(f"collected {len(list(OUT.glob('*.json')))}")
    elif mode == "reparse":
        reparse()
    else:
        compare(rest)
