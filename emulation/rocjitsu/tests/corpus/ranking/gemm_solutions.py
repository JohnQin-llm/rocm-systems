#!/usr/bin/env python3
"""hipBLASLt solution-ranking benchmark: enumerate and time GEMM solutions.

  enumerate  (hardware) run TunableOp tuning per shape and record every
             hipBLASLt solution it reports as supported, plus the op key.
  time       time one shape with one forced solution, on hardware or under
             rocjitsu, and write one JSON record.

Forcing a solution goes through TunableOp's results file: tuning off, one row
mapping the op key to the solution. The kernel names actually launched are
recorded so a fallback to another solution cannot pass unnoticed.

Hardware time is the profiler's per-kernel device time, because a host that
enqueues slower than an 8 us kernel runs would otherwise be what an event
bracket measures. Under rocjitsu the profiler has no device activity, and the
event bracket is the modelled kernel time (the modelled clock does not advance
while the device idles), as in tests/corpus/meter_score.py.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent

SHAPES = {  # name: (m, n, k[, dtype]) -- dtype defaults to bfloat16
    "sq512": (512, 512, 512),
    "sq1024": (1024, 1024, 1024),
    "m64_n4096_k1024": (64, 4096, 1024),
    "m256_n2048_k2048": (256, 2048, 2048),
    "m4096_n64_k1024": (4096, 64, 1024),
    "m1_n4096_k4096": (1, 4096, 4096),
    # Extension (PREREG.md): GEMV and decode shapes, large K, fp16, fp32.
    "m1_n1024_k4096": (1, 1024, 4096),
    "m1_n8192_k4096": (1, 8192, 4096),
    "m8_n4096_k4096": (8, 4096, 4096),
    "m16_n8192_k1024": (16, 8192, 1024),
    "m32_n4096_k4096": (32, 4096, 4096),
    "m2048_n2048_k512": (2048, 2048, 512),
    "m128_n128_k8192": (128, 128, 8192),
    "sq1024_f16": (1024, 1024, 1024, "float16"),
    "m64_n4096_k1024_f16": (64, 4096, 1024, "float16"),
    "sq512_f32": (512, 512, 512, "float32"),
    "m1_n4096_k4096_f32": (1, 4096, 4096, "float32"),
    # Held-out set (PREREG-addendum-1.md): never used for diagnosis.
    "m2_n4096_k4096": (2, 4096, 4096),
    "m4_n2048_k8192": (4, 2048, 8192),
    "m48_n3072_k1024": (48, 3072, 1024),
    "m96_n1536_k2048": (96, 1536, 2048),
    "m768_n768_k768": (768, 768, 768),
    "m1536_n512_k1024": (1536, 512, 1024),
    "m1_n2048_k4096_f16": (1, 2048, 4096, "float16"),
    "m128_n4096_k512_f16": (128, 4096, 512, "float16"),
    # Held-out set 2 (PREREG-addendum-2.md): chosen before any of it was measured.
    "m3_n6144_k4096": (3, 6144, 4096),
    "m24_n4096_k2048": (24, 4096, 2048),
    "m64_n1024_k8192": (64, 1024, 8192),
    "m192_n2560_k1536": (192, 2560, 1536),
    "m384_n1024_k3072": (384, 1024, 3072),
    "m1024_n2048_k768": (1024, 2048, 768),
    "m2_n1536_k6144_f16": (2, 1536, 6144, "float16"),
    "m512_n512_k2048_f16": (512, 512, 2048, "float16"),
    # Held-out set 3 (PREREG-addendum-3.md): chosen before any of it was measured.
    "m5_n5120_k4096": (5, 5120, 4096),
    "m40_n2560_k2048": (40, 2560, 2048),
    "m80_n6144_k1024": (80, 6144, 1024),
    "m160_n1024_k4096": (160, 1024, 4096),
    "m320_n3072_k768": (320, 3072, 768),
    "m640_n640_k2560": (640, 640, 2560),
    "m3_n3072_k3072_f16": (3, 3072, 3072, "float16"),
    "m256_n1536_k1536_f16": (256, 1536, 1536, "float16"),
    # Held-out set 4 (PREREG-addendum-4.md): chosen before any of it was measured.
    "m6_n4096_k6144": (6, 4096, 6144),
    "m12_n3072_k4096": (12, 3072, 4096),
    "m48_n5120_k1536": (48, 5120, 1536),
    "m112_n2048_k3072": (112, 2048, 3072),
    "m224_n1024_k2048": (224, 1024, 2048),
    "m448_n1792_k1024": (448, 1792, 1024),
    "m1_n6144_k2048_f16": (1, 6144, 2048, "float16"),
    "m192_n2304_k2304_f16": (192, 2304, 2304, "float16"),
}
HELDOUT_GEMM = ["m2_n4096_k4096", "m4_n2048_k8192", "m48_n3072_k1024", "m96_n1536_k2048",
                "m768_n768_k768", "m1536_n512_k1024", "m1_n2048_k4096_f16", "m128_n4096_k512_f16"]
HELDOUT4_GEMM = ["m6_n4096_k6144", "m12_n3072_k4096", "m48_n5120_k1536", "m112_n2048_k3072",
                 "m224_n1024_k2048", "m448_n1792_k1024", "m1_n6144_k2048_f16", "m192_n2304_k2304_f16"]
HELDOUT3_GEMM = ["m5_n5120_k4096", "m40_n2560_k2048", "m80_n6144_k1024", "m160_n1024_k4096",
                 "m320_n3072_k768", "m640_n640_k2560", "m3_n3072_k3072_f16", "m256_n1536_k1536_f16"]
HELDOUT2_GEMM = ["m3_n6144_k4096", "m24_n4096_k2048", "m64_n1024_k8192", "m192_n2560_k1536",
                 "m384_n1024_k3072", "m1024_n2048_k768", "m2_n1536_k6144_f16", "m512_n512_k2048_f16"]


def shape_of(name):
    s = SHAPES[name]
    return s[0], s[1], s[2], (s[3] if len(s) > 3 else "bfloat16")


def tensors(shape):
    import torch
    m, n, k, dtype = shape_of(shape)
    g = torch.Generator(device="cuda").manual_seed(0)
    dt = getattr(torch, dtype)
    a = torch.randn(m, k, device="cuda", dtype=dt, generator=g)
    b = torch.randn(k, n, device="cuda", dtype=dt, generator=g)
    return a, b


def enumerate_cmd(args):
    """Tune each shape in a fresh process; parse supported candidates."""
    out = {}
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    for shape in args.shapes:
        log = work / f"tune_{shape}.log"
        env = dict(os.environ, PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="1",
                   PYTORCH_TUNABLEOP_VERBOSE="3", PYTORCH_TUNABLEOP_ROCBLAS_ENABLED="0",
                   PYTORCH_TUNABLEOP_FILENAME=str(work / f"tune_{shape}.csv"))
        code = (f"import sys; sys.path.insert(0, {str(TOOLS)!r});"
                f"import gemm_solutions as g, torch; a, b = g.tensors({shape!r});"
                "c = a @ b; torch.cuda.synchronize()")
        if log.exists() and "tuning using" in log.read_text(errors="replace") and                 list(work.glob(f"tune_{shape}*.csv")):
            pass  # already enumerated (the original six shapes keep their logs)
        else:
            guard = [sys.executable, str(TOOLS / "gpu_guard.py"), "run", "--hip",
                     str(args.hip), "--record", str(work / f"tune_{shape}.guard.json"), "--"]
            for _ in range(5):
                # A results file left by an earlier attempt would be loaded instead of
                # tuned, and its log would list no timed candidates.
                for stale in work.glob(f"tune_{shape}*.csv"):
                    stale.unlink()
                with open(log, "w") as fh:
                    subprocess.run(guard + [sys.executable, "-c", code], env=env, stdout=fh,
                                   stderr=subprocess.STDOUT, check=False)
                if not json.loads((work / f"tune_{shape}.guard.json").read_text())["contended"]:
                    break
        text = log.read_text(errors="replace")
        names = sorted(set(re.findall(r"(?:found (?:better|slower) instance|skip slow instance)"
                                      r"[^\n]*?(Gemm_Hipblaslt_\d+)", text)))
        csvs = sorted(work.glob(f"tune_{shape}*.csv"))
        rows = [l for l in csvs[0].read_text().splitlines() if not l.startswith("Validator")] if csvs else []
        validators = [l for l in csvs[0].read_text().splitlines() if l.startswith("Validator")] if csvs else []
        assert rows, f"{shape}: no tuned row in {csvs}"
        op, key, best, _ = rows[0].split(",")
        out[shape] = {"shape": shape_of(shape), "op": op, "key": key, "tuned_best": best,
                      "candidates": names, "validators": validators}
        print(f"{shape}: {len(names)} supported candidates, tunableop best {best}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def force(meta, shape, solution, workdir):
    import torch
    m = meta[shape]
    path = Path(workdir) / f"force_{shape}_{solution}_{os.getpid()}.csv"
    path.write_text("\n".join(m["validators"] + [f"{m['op']},{m['key']},{solution},0"]) + "\n")
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(False)
    torch.cuda.tunable.record_untuned_enable(False)
    assert torch.cuda.tunable.read_file(str(path)), f"TunableOp rejected {path}"


def time_cmd(args):
    import torch
    meta = json.loads(Path(args.meta).read_text())
    if args.solution != "Default":
        force(meta, args.shape, args.solution, args.workdir)
    a, b = tensors(args.shape)
    for _ in range(args.warmup):
        a @ b
    torch.cuda.synchronize()
    rec = {"shape": args.shape, "solution": args.solution, "mode": args.mode}
    events = []
    for _ in range(args.samples):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        a @ b
        e.record()
        torch.cuda.synchronize()
        events.append(s.elapsed_time(e) * 1000.0)
    rec["event_us"] = statistics.median(events)
    rec["event_samples_us"] = events
    if args.mode == "hw":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_iters):
                a @ b
            torch.cuda.synchronize()
        kernels = {}
        for ev in prof.events():
            if ev.device_type.name == "CUDA" and ev.device_time_total > 0:
                kernels.setdefault(ev.name, []).append(ev.device_time_total)
        rec["kernels"] = sorted(kernels)
        per_call = sum(sum(v) for v in kernels.values()) / args.profile_iters
        rec["kernel_us"] = per_call
    rec["wall_s"] = time.time() - START
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec))
    print(json.dumps({k: rec[k] for k in rec if k != "event_samples_us"}), flush=True)


START = time.time()
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enumerate")
    e.add_argument("--shapes", nargs="+", default=list(SHAPES))
    e.add_argument("--workdir", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--hip", type=int, default=5, help="HIP device to tune on (via gpu_guard)")
    t = sub.add_parser("time")
    t.add_argument("--meta", required=True)
    t.add_argument("--shape", required=True)
    t.add_argument("--solution", required=True)
    t.add_argument("--mode", choices=("hw", "emu"), required=True)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--samples", type=int, default=20)
    t.add_argument("--profile-iters", type=int, default=100)
    t.add_argument("--workdir", default="/tmp")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    enumerate_cmd(a) if a.cmd == "enumerate" else time_cmd(a)
