#!/usr/bin/env python3
"""Hardware ground truth, one GPU per run, every measurement guarded.

Each run times every selected (shape, solution) sequentially on its own GPU, so
whatever state that GPU is in applies to every pick of the run alike. Each
measurement goes through gpu_guard.py; a contended one is discarded and
re-measured (up to 5 attempts). Runs execute in parallel on different GPUs.

usage: time_hw_runs.py OUTDIR RUN:HIP [RUN:HIP ...]
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

TOOLS = Path(__file__).resolve().parent                        # this directory
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
out_root = Path(sys.argv[1])
runs = [tuple(int(x) for x in spec.split(":")) for spec in sys.argv[2:]]
selected = json.loads((WORK / "selected.json").read_text())
picks = [(s, sol) for s, sols in selected.items() for sol in sols]


def one_run(run_hip):
    run, hip = run_hip
    d = out_root / f"run{run}"
    d.mkdir(parents=True, exist_ok=True)
    retried = failed = 0
    for shape, sol in picks:
        out = d / f"{shape}__{sol}.json"
        guard = d / f"{shape}__{sol}.guard.json"
        if out.exists() and guard.exists() and not json.loads(guard.read_text())["contended"]:
            continue
        for attempt in range(5):
            out.unlink(missing_ok=True)
            cmd = [sys.executable, str(TOOLS / "gpu_guard.py"), "run", "--hip", str(hip), "--record", str(guard),
                   "--wait", "1800", "--", sys.executable, str(TOOLS / "gemm_solutions.py"), "time",
                   "--meta", str(WORK / "candidates.json"), "--shape", shape, "--solution", sol,
                   "--mode", "hw", "--workdir", "/tmp", "--out", str(out)]
            with open(str(out) + ".err", "w") as err:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err, check=False)
            if not json.loads(guard.read_text())["contended"]:
                break
            retried += 1
        else:
            failed += 1
            out.unlink(missing_ok=True)
    return run, len(list(d.glob("*__*[!d].json"))), retried, failed


with ThreadPoolExecutor(max_workers=len(runs)) as pool:
    for run, n, retried, failed in pool.map(one_run, runs):
        print(f"run{run}: {n}/{len(picks)} records, {retried} contended attempts re-measured, "
              f"{failed} picks never measured clean", flush=True)
print("DONE")
