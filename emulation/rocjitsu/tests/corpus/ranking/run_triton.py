#!/usr/bin/env python3
"""Run the Triton benchmark on hardware (guarded, one GPU per run) or under rocjitsu.

  run_triton.py hw  OUTDIR RUN:HIP [RUN:HIP ...]
  run_triton.py emu OUTDIR JOBS

Emulator records also carry the executed-instruction and DRAM-byte totals of
the Triton kernel's dispatches from the timing trace (non-timing baselines).
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent                        # this directory
ROOT = TOOLS.parents[2]                                          # emulation/rocjitsu
BIN = os.environ.get("ROCJITSU_BIN", str(ROOT / "build/tools/rocjitsu/rocjitsu"))
CFG = os.environ.get("ROCJITSU_CONFIG", str(ROOT / "configs/gfx950_mi355x_kmd_timing.json"))
cases = json.loads(subprocess.run([sys.executable, str(TOOLS / "triton_bench.py"), "list"],
                                  capture_output=True, text=True, check=True).stdout)
jobs = [(c, v) for c, vs in cases.items() for v in vs]
mode, out_root = sys.argv[1], Path(sys.argv[2])


def tb(case, variant, m, out):
    return [sys.executable, str(TOOLS / "triton_bench.py"), "time", "--case", case, "--variant", variant,
            "--mode", m, "--out", str(out)] + (["--warmup", "1", "--samples", "3"] if m == "emu" else [])


def hw_run(run_hip):
    run, hip = run_hip
    d = out_root / f"run{run}"
    d.mkdir(parents=True, exist_ok=True)
    retried = failed = 0
    for case, var in jobs:
        out, guard = d / f"{case}__{var}.json", d / f"{case}__{var}.guard.json"
        if out.exists() and guard.exists() and not json.loads(guard.read_text())["contended"]:
            continue
        for _ in range(5):
            out.unlink(missing_ok=True)
            cmd = [sys.executable, str(TOOLS / "gpu_guard.py"), "run", "--hip", str(hip), "--record", str(guard),
                   "--wait", "1800", "--"] + tb(case, var, "hw", out)
            with open(str(out) + ".err", "w") as err:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err, check=False)
            if not json.loads(guard.read_text())["contended"]:
                break
            retried += 1
        else:
            failed += 1
            out.unlink(missing_ok=True)
    return run, retried, failed


def emu_one(job):
    case, var = job
    out_root.mkdir(parents=True, exist_ok=True)
    rec_path = out_root / f"{case}__{var}.json"
    if rec_path.exists():
        return
    raw, trace, log = (out_root / f"{case}__{var}.{x}" for x in ("raw.json", "trace.jsonl", "log"))
    start = time.time()
    cmd = ["timeout", "--kill-after=30", "7200", BIN, "--config",
           CFG, "--"] + tb(case, var, "emu", raw)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, env=dict(os.environ, ROCJITSU_TIMING_TRACE=str(trace)), stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    rec = {"case": case, "variant": var, "rc": rc, "emu_wall_s": time.time() - start}
    if raw.exists():
        rec.update(json.loads(raw.read_text()))
    name = rec.get("kernel_name") or ""
    insts = dram = n = 0
    if trace.exists() and name:
        for line in trace.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(d.get("kernel", "")).startswith(name):
                insts += d.get("instructions", 0)
                dram += d.get("dram_bytes", 0)
                n += 1
    rec.update({"dispatches": n, "instructions": insts, "dram_bytes": dram})
    rec_path.write_text(json.dumps(rec))
    print(f"{case} {var} rc={rc} {rec['emu_wall_s']:.0f}s event_us={rec.get('event_us')}", flush=True)


if mode == "hw":
    runs = [tuple(int(x) for x in s.split(":")) for s in sys.argv[3:]]
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        for run, retried, failed in pool.map(hw_run, runs):
            print(f"run{run}: {retried} contended attempts re-measured, {failed} never clean", flush=True)
else:
    only = set(sys.argv[4:])
    jobs = [j for j in jobs if not only or j[0] in only]
    with ThreadPoolExecutor(max_workers=int(sys.argv[3])) as pool:
        list(pool.map(emu_one, jobs))
print("DONE")
