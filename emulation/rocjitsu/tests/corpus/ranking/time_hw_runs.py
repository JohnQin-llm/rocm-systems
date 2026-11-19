#!/usr/bin/env python3
"""Hardware ground truth, one GPU per run, every measurement guarded.

Each run times every selected (shape, solution) sequentially on its own GPU, so
whatever state that GPU is in applies to every pick of the run alike. Each
measurement goes through gpu_guard.py; a contended one is discarded and
re-measured (up to 5 attempts). Runs execute in parallel on different GPUs.

usage: time_hw_runs.py OUTDIR RUN:HIP [RUN:HIP ...]
       time_hw_runs.py OUTDIR RUNS@GPU,GPU,... [SHAPE ...]   (shape units spread over the GPUs)
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
selected = json.loads((WORK / "selected.json").read_text())
picks = [(s, sol) for s, sols in selected.items() for sol in sols]


def spread(groups, runs, gpus):
    """RUNS@GPU,GPU,...: every (run, group) unit on one GPU, units spread over all GPUs.

    A group's variants are timed back to back on one GPU within a run, so the
    within-group comparison never mixes GPUs; a group's runs land on different
    GPUs (offsets 0, 3, 6, ... modulo the GPU count), which are distinct while
    the GPU count exceeds 3 * (runs - 1).
    """
    per_gpu = {g: [] for g in gpus}
    for run in range(1, runs + 1):
        for index, group in enumerate(groups):
            per_gpu[gpus[(index + 3 * (run - 1)) % len(gpus)]].append((run, group))
    return per_gpu


def one_run(run_hip, only_shape=None):
    run, hip = run_hip
    d = out_root / f"run{run}"
    d.mkdir(parents=True, exist_ok=True)
    retried = failed = 0
    for shape, sol in picks:
        if only_shape is not None and shape != only_shape:
            continue
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


if "@" in sys.argv[2]:
    n_runs, gpu_list = sys.argv[2].split("@")
    only = set(sys.argv[3:])
    plan = spread(sorted({s for s, _ in picks if not only or s in only}), int(n_runs),
                  [int(g) for g in gpu_list.split(",")])

    def gpu_worker(gpu):
        total = [0, 0]
        for run, shape in plan[gpu]:
            _, _, retried, failed = one_run((run, gpu), only_shape=shape)
            total[0] += retried
            total[1] += failed
        return gpu, total

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        for gpu, (retried, failed) in pool.map(gpu_worker, list(plan)):
            print(f"gpu{gpu}: {retried} contended attempts re-measured, {failed} picks never measured clean",
                  flush=True)
else:
    runs = [tuple(int(x) for x in spec.split(":")) for spec in sys.argv[2:]]
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        for run, n, retried, failed in pool.map(one_run, runs):
            print(f"run{run}: {n}/{len(picks)} records, {retried} contended attempts re-measured, "
                  f"{failed} picks never measured clean", flush=True)
print("DONE")
