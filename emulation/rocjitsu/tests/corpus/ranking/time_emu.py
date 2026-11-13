#!/usr/bin/env python3
"""Time every selected (shape, solution) under the rebased rocjitsu, uncalibrated.

One emulator process per pick. Records the modelled event time, the Cijk
kernels HIP actually launched (from AMD_LOG_LEVEL=3, to prove the forced
solution ran), the per-dispatch timing trace's executed-instruction and
DRAM-byte totals for the matmul dispatches (inputs to the non-timing
baselines), and the emulator wall time.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent                        # this directory
ROOT = TOOLS.parents[2]                                          # emulation/rocjitsu
WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
BIN = os.environ.get("ROCJITSU_BIN", str(ROOT / "build/tools/rocjitsu/rocjitsu"))
CFG = os.environ.get("ROCJITSU_CONFIG", str(ROOT / "configs/gfx950_mi355x_kmd_timing.json"))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else WORK / "emu"
JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
selected = json.loads((WORK / "selected.json").read_text())


def one(job):
    shape, sol = job
    rec_path = OUT / f"{shape}__{sol}.json"
    if rec_path.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"{shape}__{sol}.log"
    trace = OUT / f"{shape}__{sol}.trace.jsonl"
    raw = OUT / f"{shape}__{sol}.raw.json"
    start = time.time()
    cmd = ["timeout", "--kill-after=30", "7200", BIN, "--config", CFG, "--",
           sys.executable, str(TOOLS / "gemm_solutions.py"), "time",
           "--meta", str(WORK / "candidates.json"), "--shape", shape, "--solution", sol,
           "--mode", "emu", "--warmup", "1", "--samples", "3", "--workdir", "/tmp", "--out", str(raw)]
    env = dict(os.environ, AMD_LOG_LEVEL="3", ROCJITSU_TIMING_TRACE=str(trace))
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - start
    text = log.read_text(errors="replace")
    kernels = sorted(set(re.findall(r"ShaderName : (Cijk\S+)", text)))
    rec = {"shape": shape, "solution": sol, "rc": rc, "emu_wall_s": wall, "kernels": kernels}
    if raw.exists():
        rec.update({k: v for k, v in json.loads(raw.read_text()).items() if k not in rec})
    insts = dram = n = 0
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("kernel", "")).startswith("Cijk"):
                insts += d.get("instructions", 0)
                dram += d.get("dram_bytes", 0)
                n += 1
    rec.update({"gemm_dispatches": n, "gemm_instructions": insts, "gemm_dram_bytes": dram})
    rec_path.write_text(json.dumps(rec))
    log.write_text("\n".join(l for l in text.splitlines() if "ShaderName" in l or "rj " in l
                             or "Traceback" in l or "Error" in l)[-20000:])
    print(f"{shape} {sol} rc={rc} {wall:.0f}s event_us={rec.get('event_us')} kernels={len(kernels)}",
          flush=True)


only = set(sys.argv[3:])
jobs = [(s, sol) for s, sols in selected.items() for sol in sols if not only or s in only]
with ThreadPoolExecutor(max_workers=JOBS) as pool:
    list(pool.map(one, jobs))
print(f"DONE {len(list(OUT.glob('*__*[!w].json')))} records")
