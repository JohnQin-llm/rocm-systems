#!/usr/bin/env python3
"""Run a hardware measurement only on a GPU nobody else is using, and prove it.

Ownership comes from KFD, not from utilisation: every process with a hardware
queue on a GPU appears under /sys/class/kfd/kfd/proc/<pid>/queues/*/gpuid, with
host PIDs from every container. HIP device N is the N-th KFD topology node with
a non-zero gpu_id, which is *not* amd-smi's (PCI-ordered) GPU index.

  gpu_guard.py run --hip N --record OUT.guard.json [--wait S] -- CMD ...

waits until no process has a queue on that GPU, runs CMD with
HIP_VISIBLE_DEVICES=N, samples queue owners every 0.2 s while it runs, and
writes {"gpu_id", "waited_s", "max_owners", "contended", "rc"}. Our own process
is one owner; a second owner at any sample means the measurement shared the GPU.
Exit status: CMD's, or 75 when the GPU never became free.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

KFD = Path("/sys/class/kfd/kfd")


def hip_gpu_ids():
    ids = []
    for node in sorted(KFD.glob("topology/nodes/*"), key=lambda p: int(p.name)):
        gid = int((node / "gpu_id").read_text().strip() or 0)
        if gid:
            ids.append(gid)
    return ids


def owners(gpu_id):
    pids = set()
    for q in glob.glob(str(KFD / "proc/*/queues/*/gpuid")):
        try:
            if int(Path(q).read_text().strip()) == gpu_id:
                pids.add(q.split("/")[6])
        except (OSError, ValueError):
            continue
    return pids


def free_hips():
    return [i for i, g in enumerate(hip_gpu_ids()) if not owners(g)]


def run(args):
    gpu_id = hip_gpu_ids()[args.hip]
    start = time.time()
    while owners(gpu_id):
        if time.time() - start > args.wait:
            Path(args.record).write_text(json.dumps({"gpu_id": gpu_id, "waited_s": time.time() - start,
                                                     "max_owners": None, "contended": True, "rc": 75}))
            return 75
        time.sleep(1.0)
    waited = time.time() - start
    peak = [0]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], len(owners(gpu_id)))
            stop.wait(0.2)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    rc = subprocess.run(args.cmd, env={**os.environ, "HIP_VISIBLE_DEVICES": str(args.hip)}).returncode
    stop.set()
    t.join()
    after = len(owners(gpu_id))
    rec = {"gpu_id": gpu_id, "hip": args.hip, "waited_s": waited, "max_owners": peak[0],
           "owners_after": after, "contended": peak[0] > 1 or after > 0, "rc": rc}
    Path(args.record).write_text(json.dumps(rec))
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd_name", required=True)
    r = sub.add_parser("run")
    r.add_argument("--hip", type=int, required=True)
    r.add_argument("--record", required=True)
    r.add_argument("--wait", type=float, default=600.0)
    r.add_argument("cmd", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd_name == "status":
        ids = hip_gpu_ids()
        for i, g in enumerate(ids):
            print(f"hip {i} gpu_id {g}: owners {sorted(owners(g)) or '-'}")
        sys.exit(0)
    if a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    sys.exit(run(a))
