#!/usr/bin/env python3
"""Triton variant-ranking benchmark: one operation, many launch configurations.

  list                     JSON of every (case, variant) in the benchmark
  time --case C --variant V --mode hw|emu --out OUT

Hardware time is the profiler's kernel device time; under rocjitsu it is the
event bracket (the modelled kernel time), as in gemm_solutions.py. Each record
carries the compiled kernel's hash, so hardware and emulator are known to have
run the same binary, and the max error against a torch reference.
"""

import argparse
import itertools
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % group) % size_m)
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        kmask = rk < K - k * BLOCK_K
        x = tl.load(a, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        y = tl.load(b, mask=kmask[:, None] & (rn[None, :] < N), other=0.0)
        acc += tl.dot(x, y)
        a += BLOCK_K * sak
        b += BLOCK_K * sbk
    c = c_ptr + rm[:, None] * scm + rn[None, :] * scn
    tl.store(c, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_rows, n_cols, stride, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for r in range(ROWS):
        row = pid * ROWS + r
        if row < n_rows:
            x = tl.load(in_ptr + row * stride + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
            x = x - tl.max(x, axis=0)
            e = tl.exp(x)
            y = e / tl.sum(e, axis=0)
            tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=cols < n_cols)


# The same kernel with the output store's cache policy as a variable (PREREG-addendum-3.md).
# A separate kernel, so the variants measured before keep their compiled code.
@triton.jit
def softmax_store_kernel(out_ptr, in_ptr, n_rows, n_cols, stride, ROWS: tl.constexpr, BLOCK: tl.constexpr,
                         STORE: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for r in range(ROWS):
        row = pid * ROWS + r
        if row < n_rows:
            x = tl.load(in_ptr + row * stride + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
            x = x - tl.max(x, axis=0)
            e = tl.exp(x)
            y = e / tl.sum(e, axis=0)
            tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=cols < n_cols,
                     cache_modifier=STORE)


STORE_POLICIES = ("", ".cs", ".wt")   # variant field c0, c1, c2


MATMUL_SHAPES = {"mm512": (512, 512, 512), "mm1024": (1024, 1024, 1024),
                 "mm64x4096x1024": (64, 4096, 1024), "mm2048x512x1024": (2048, 512, 1024),
                 # Held-out (PREREG-addendum-1.md)
                 "mm768": (768, 768, 768), "mm128x2048x2048": (128, 2048, 2048),
                 # Held-out set 2 (PREREG-addendum-2.md)
                 "mm256x3072x1024": (256, 3072, 1024), "mm1536x640x512": (1536, 640, 512),
                 # Held-out set 3 (PREREG-addendum-3.md)
                 "mm384x1536x1536": (384, 1536, 1536), "mm1024x384x2048": (1024, 384, 2048),
                 "mm192x4096x512": (192, 4096, 512),
                 # Held-out set 4 (PREREG-addendum-4.md)
                 "mm320x2560x1280": (320, 2560, 1280), "mm896x896x1792": (896, 896, 1792),
                 "mm2048x256x1536": (2048, 256, 1536),
                 # Held-out set 5 (PREREG-addendum-5.md)
                 "mm448x1792x896": (448, 1792, 896), "mm1280x512x3072": (1280, 512, 3072)}
SOFTMAX_SHAPES = {"sm4096x1024": (4096, 1024), "sm1024x4096": (1024, 4096), "sm8192x512": (8192, 512),
                  # Held-out (PREREG-addendum-1.md)
                  "sm2048x2048": (2048, 2048), "sm16384x256": (16384, 256),
                  # Held-out set 2 (PREREG-addendum-2.md)
                  "sm4096x512": (4096, 512), "sm1024x3072": (1024, 3072),
                  # Held-out set 3 (PREREG-addendum-3.md); variants also choose the store policy
                  "sm3072x768": (3072, 768), "sm6144x1280": (6144, 1280), "sm1536x2560": (1536, 2560),
                  # Held-out set 4 (PREREG-addendum-4.md); variants also choose the store policy
                  "sm2560x1792": (2560, 1792), "sm12288x640": (12288, 640), "sm768x5120": (768, 5120),
                  # Held-out set 5 (PREREG-addendum-5.md); variants also choose the store policy
                  "sm5120x1024": (5120, 1024), "sm2048x3584": (2048, 3584), "sm10240x384": (10240, 384),
                  "sm3584x2048": (3584, 2048), "sm1280x4608": (1280, 4608), "sm6144x896": (6144, 896)}
HELDOUT_TRITON = ["mm768", "mm128x2048x2048", "sm2048x2048", "sm16384x256"]
HELDOUT2_TRITON = ["mm256x3072x1024", "mm1536x640x512", "sm4096x512", "sm1024x3072"]
HELDOUT3_TRITON = ["mm384x1536x1536", "mm1024x384x2048", "mm192x4096x512", "sm3072x768", "sm6144x1280",
                   "sm1536x2560"]
HELDOUT4_TRITON = ["mm320x2560x1280", "mm896x896x1792", "mm2048x256x1536", "sm2560x1792", "sm12288x640",
                   "sm768x5120"]
HELDOUT5_TRITON = ["mm448x1792x896", "mm1280x512x3072", "sm5120x1024", "sm2048x3584", "sm10240x384",
                   "sm3584x2048", "sm1280x4608", "sm6144x896"]


def cases():
    out = {}
    grid = list(itertools.product((32, 64, 128), (32, 64, 128), (32, 64), (4, 8), (1, 2)))
    for name in MATMUL_SHAPES:
        rng = random.Random(f"triton-{name}")
        out[name] = [f"m{bm}_n{bn}_k{bk}_w{w}_s{s}" for bm, bn, bk, w, s in rng.sample(grid, 16)]
    sgrid = list(itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8)))
    cgrid = list(itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8), range(len(STORE_POLICIES))))
    for name in SOFTMAX_SHAPES:
        rng = random.Random(f"triton-{name}")
        if name in HELDOUT3_TRITON or name in HELDOUT4_TRITON or name in HELDOUT5_TRITON:
            out[name] = [f"w{w}_r{r}_c{c}" for w, r, c in rng.sample(cgrid, 12)]
        else:
            out[name] = [f"w{w}_r{r}" for w, r in rng.sample(sgrid, 12)]
    return out


def parse(variant):
    return {p[0]: int(p[1:]) for p in variant.split("_")}


def build(case, variant):
    g = torch.Generator(device="cuda").manual_seed(0)
    v = parse(variant)
    if case in MATMUL_SHAPES:
        m, n, k = MATMUL_SHAPES[case]
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16, generator=g)
        c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        grid = (triton.cdiv(m, v["m"]) * triton.cdiv(n, v["n"]),)

        def fn():
            return matmul_kernel[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                                       c.stride(0), c.stride(1), BLOCK_M=v["m"], BLOCK_N=v["n"],
                                       BLOCK_K=v["k"], GROUP_M=8, num_warps=v["w"], num_stages=v["s"])

        def err():
            return (c.float() - (a.float() @ b.float())).abs().max().item() / max(1.0, (a.float() @ b.float()).abs().max().item())
        return fn, err
    rows, cols = SOFTMAX_SHAPES[case]
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, generator=g)
    y = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    grid = (triton.cdiv(rows, v["r"]),)

    def fn():
        if "c" in v:
            return softmax_store_kernel[grid](y, x, rows, cols, x.stride(0), ROWS=v["r"], BLOCK=block,
                                              STORE=STORE_POLICIES[v["c"]], num_warps=v["w"])
        return softmax_kernel[grid](y, x, rows, cols, x.stride(0), ROWS=v["r"], BLOCK=block, num_warps=v["w"])

    def err():
        return (y.float() - torch.softmax(x.float(), dim=1)).abs().max().item()
    return fn, err


def time_cmd(a):
    start = time.time()
    fn, err = build(a.case, a.variant)
    handle = fn()
    torch.cuda.synchronize()
    rec = {"case": a.case, "variant": a.variant, "mode": a.mode, "kernel_hash": getattr(handle, "hash", None),
           "kernel_name": getattr(getattr(handle, "metadata", None), "name", None), "max_err": err()}
    for _ in range(a.warmup):
        fn()
    torch.cuda.synchronize()
    ev = []
    for _ in range(a.samples):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ev.append(s.elapsed_time(e) * 1000.0)
    rec["event_us"] = statistics.median(ev)
    if a.mode == "hw":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.profile_iters):
                fn()
            torch.cuda.synchronize()
        dev = [e.device_time_total for e in prof.events()
               if e.device_type.name == "CUDA" and e.device_time_total > 0 and "kernel" in e.name]
        rec["kernel_us"] = sum(dev) / a.profile_iters
    rec["wall_s"] = time.time() - start
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    t = sub.add_parser("time")
    t.add_argument("--case", required=True)
    t.add_argument("--variant", required=True)
    t.add_argument("--mode", choices=("hw", "emu"), required=True)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--samples", type=int, default=20)
    t.add_argument("--profile-iters", type=int, default=100)
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        print(json.dumps(cases(), indent=1))
    else:
        time_cmd(a)
