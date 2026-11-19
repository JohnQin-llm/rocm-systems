#!/usr/bin/env python3
"""More operation families for the variant-ranking benchmark: attention, RMSNorm, fp8 matmul.

  list                     JSON of every (case, variant) in these families
  time --case C --variant V --mode hw|emu --out OUT

Same contract as triton_bench.py, so run_triton.py drives it with
RANKING_BENCH=kernel_families.py, and prereg_score.py scores it in triton mode.

- fa*:  flash-attention forward, [batch, heads, sequence, head_dim] bf16, online
        softmax; a trailing c is causal. Variants: block M and N, warps, stages.
- rms*: RMSNorm over the last dimension, bf16, a weight per column. Variants:
        warps and rows per program.
- f8*:  matmul of two float8_e4m3fn operands into bf16, fp32 accumulation, the
        kernel of triton_bench.py. Variants: block M, N, K, warps, stages.
- ln*:  LayerNorm over the last dimension, bf16, weight and bias per column.
        Variants: warps and rows per program (held-out set 8).
- sw*:  SwiGLU, out[M, N] = silu(x[:, :N]) * x[:, N:] for x[M, 2N] bf16.
        Variants: block width and warps (held-out set 8).
"""

import argparse
import itertools
import json
import random
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from triton_bench import matmul_kernel, parse

LOG2E = 1.4426950408889634


@triton.jit
def attn_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, qk_scale, S, D: tl.constexpr, BM: tl.constexpr,
                    BN: tl.constexpr, CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    base = tl.program_id(1).to(tl.int64) * S * D
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + base + offs_m[:, None] * D + offs_d[None, :])
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    hi = (pid_m + 1) * BM if CAUSAL else S
    for start in range(0, hi, BN):
        offs_n = start + tl.arange(0, BN)
        k = tl.load(k_ptr + base + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_n[:, None] * D + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(o_ptr + base + offs_m[:, None] * D + offs_d[None, :], (acc / l_i[:, None]).to(o_ptr.dtype.element_ty))


@triton.jit
def rmsnorm_kernel(out_ptr, in_ptr, w_ptr, n_rows, n_cols, stride, eps, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    for r in range(ROWS):
        row = pid * ROWS + r
        if row < n_rows:
            x = tl.load(in_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            y = x * tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps) * w
            tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def layernorm_kernel(out_ptr, in_ptr, w_ptr, b_ptr, n_rows, n_cols, stride, eps, ROWS: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    for r in range(ROWS):
        row = pid * ROWS + r
        if row < n_rows:
            x = tl.load(in_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
            mean = tl.sum(x, axis=0) / n_cols
            d = tl.where(mask, x - mean, 0.0)
            y = d * tl.rsqrt(tl.sum(d * d, axis=0) / n_cols + eps) * w + b
            tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    a = tl.load(in_ptr + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(in_ptr + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    y = a * tl.sigmoid(a) * g
    tl.store(out_ptr + row * n_cols + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


# (batch, heads, sequence, head_dim, causal)
ATTN_SHAPES = {"fa1x16x1024x128": (1, 16, 1024, 128, False), "fa1x8x2048x64": (1, 8, 2048, 64, False),
               "fa2x32x512x128": (2, 32, 512, 128, False), "fa1x8x2048x128c": (1, 8, 2048, 128, True),
               "fa1x32x1024x64c": (1, 32, 1024, 64, True), "fa4x16x512x64c": (4, 16, 512, 64, True),
               # Held-out set 8 (PREREG-addendum-8.md)
               "fa1x12x1536x64": (1, 12, 1536, 64, False), "fa2x8x1024x128c": (2, 8, 1024, 128, True),
               "fa1x24x768x128": (1, 24, 768, 128, False)}
# (rows, columns)
RMS_SHAPES = {"rms4096x4096": (4096, 4096), "rms2048x7168": (2048, 7168), "rms8192x2048": (8192, 2048),
              "rms1024x8192": (1024, 8192), "rms16384x1024": (16384, 1024), "rms512x5120": (512, 5120),
              # Held-out set 8 (PREREG-addendum-8.md)
              "rms3072x5120": (3072, 5120), "rms6144x2560": (6144, 2560)}
# (M, N, K)
F8_SHAPES = {"f8m1024n1024k2048": (1024, 1024, 2048), "f8m256n4096k2048": (256, 4096, 2048),
             "f8m2048n512k4096": (2048, 512, 4096), "f8m512n2048k1024": (512, 2048, 1024),
             "f8m128n7168k2048": (128, 7168, 2048), "f8m4096n1024k1024": (4096, 1024, 1024),
             # Held-out set 8 (PREREG-addendum-8.md)
             "f8m768n1536k2048": (768, 1536, 2048), "f8m3072n768k1536": (3072, 768, 1536),
             "f8m64n5120k3072": (64, 5120, 3072)}
# Held-out set 8 (PREREG-addendum-8.md): two families new to the model. (rows, columns) and (rows, N).
LN_SHAPES = {"ln4096x2048": (4096, 2048), "ln2048x5120": (2048, 5120), "ln8192x1024": (8192, 1024),
             "ln1024x7168": (1024, 7168)}
SW_SHAPES = {"sw4096x4096": (4096, 4096), "sw2048x7168": (2048, 7168), "sw8192x2816": (8192, 2816),
             "sw16384x1408": (16384, 1408)}
HELDOUT8_FAMILIES = ["fa1x12x1536x64", "fa2x8x1024x128c", "fa1x24x768x128", "rms3072x5120", "rms6144x2560",
                     "f8m768n1536k2048", "f8m3072n768k1536", "f8m64n5120k3072"] + list(LN_SHAPES) + list(SW_SHAPES)


def cases():
    out = {}
    agrid = list(itertools.product((32, 64, 128), (32, 64, 128), (2, 4, 8), (1, 2)))
    for name in ATTN_SHAPES:
        rng = random.Random(f"families-{name}")
        out[name] = [f"m{bm}_n{bn}_w{w}_s{s}" for bm, bn, w, s in rng.sample(agrid, 12)]
    rgrid = list(itertools.product((1, 2, 4, 8, 16), (1, 2, 4, 8)))
    for name in RMS_SHAPES:
        rng = random.Random(f"families-{name}")
        out[name] = [f"w{w}_r{r}" for w, r in rng.sample(rgrid, 12)]
    fgrid = list(itertools.product((32, 64, 128), (32, 64, 128), (64, 128), (4, 8), (1, 2)))
    for name in F8_SHAPES:
        rng = random.Random(f"families-{name}")
        out[name] = [f"m{bm}_n{bn}_k{bk}_w{w}_s{s}" for bm, bn, bk, w, s in rng.sample(fgrid, 16)]
    for name in LN_SHAPES:
        rng = random.Random(f"families-{name}")
        out[name] = [f"w{w}_r{r}" for w, r in rng.sample(rgrid, 12)]
    sgrid = list(itertools.product((256, 512, 1024, 2048, 4096), (1, 2, 4, 8, 16)))
    for name in SW_SHAPES:
        rng = random.Random(f"families-{name}")
        out[name] = [f"b{b}_w{w}" for b, w in rng.sample(sgrid, 12)]
    return out


def build(case, variant):
    # Inputs and references on the host, as in triton_bench.build().
    g = torch.Generator().manual_seed(0)
    v = parse(variant)
    if case in ATTN_SHAPES:
        b, h, s, d, causal = ATTN_SHAPES[case]
        q, k, vv = (torch.randn(b, h, s, d, dtype=torch.bfloat16, generator=g).cuda() for _ in range(3))
        o = torch.empty_like(q)
        grid = (s // v["m"], b * h)

        def fn():
            return attn_fwd_kernel[grid](q, k, vv, o, d ** -0.5 * LOG2E, s, D=d, BM=v["m"], BN=v["n"],
                                         CAUSAL=causal, num_warps=v["w"], num_stages=v["s"])

        def err():
            qc, kc, vc = q.cpu().float(), k.cpu().float(), vv.cpu().float()
            sc = (qc @ kc.transpose(-1, -2)) * d ** -0.5
            if causal:
                sc = sc.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
            return (o.cpu().float() - torch.softmax(sc, -1) @ vc).abs().max().item()
        return fn, err
    if case in RMS_SHAPES:
        rows, cols = RMS_SHAPES[case]
        x = torch.randn(rows, cols, dtype=torch.bfloat16, generator=g).cuda()
        w = torch.randn(cols, dtype=torch.bfloat16, generator=g).cuda()
        y = torch.empty_like(x)
        grid = (triton.cdiv(rows, v["r"]),)

        def fn():
            return rmsnorm_kernel[grid](y, x, w, rows, cols, x.stride(0), 1e-6, ROWS=v["r"],
                                        BLOCK=triton.next_power_of_2(cols), num_warps=v["w"])

        def err():
            xf = x.cpu().float()
            ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.cpu().float()
            return ((y.cpu().float() - ref).abs().max() / ref.abs().max()).item()
        return fn, err
    if case in LN_SHAPES:
        rows, cols = LN_SHAPES[case]
        x = torch.randn(rows, cols, dtype=torch.bfloat16, generator=g).cuda()
        w = torch.randn(cols, dtype=torch.bfloat16, generator=g).cuda()
        bb = torch.randn(cols, dtype=torch.bfloat16, generator=g).cuda()
        y = torch.empty_like(x)
        grid = (triton.cdiv(rows, v["r"]),)

        def fn():
            return layernorm_kernel[grid](y, x, w, bb, rows, cols, x.stride(0), 1e-5, ROWS=v["r"],
                                          BLOCK=triton.next_power_of_2(cols), num_warps=v["w"])

        def err():
            ref = torch.nn.functional.layer_norm(x.cpu().float(), (cols,), w.cpu().float(), bb.cpu().float(), 1e-5)
            return ((y.cpu().float() - ref).abs().max() / ref.abs().max()).item()
        return fn, err
    if case in SW_SHAPES:
        rows, n = SW_SHAPES[case]
        x = torch.randn(rows, 2 * n, dtype=torch.bfloat16, generator=g).cuda()
        y = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
        grid = (rows, triton.cdiv(n, v["b"]))

        def fn():
            return swiglu_kernel[grid](y, x, n, BLOCK=v["b"], num_warps=v["w"])

        def err():
            xc = x.cpu().float()
            ref = torch.nn.functional.silu(xc[:, :n]) * xc[:, n:]
            return ((y.cpu().float() - ref).abs().max() / ref.abs().max()).item()
        return fn, err
    m, n, k = F8_SHAPES[case]
    f8 = torch.float8_e4m3fn
    a = (torch.randn(m, k, generator=g) / 4).to(f8).cuda()
    bt = (torch.randn(k, n, generator=g) / 4).to(f8).cuda()
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    grid = (triton.cdiv(m, v["m"]) * triton.cdiv(n, v["n"]),)

    def fn():
        return matmul_kernel[grid](a, bt, c, m, n, k, a.stride(0), a.stride(1), bt.stride(0), bt.stride(1),
                                   c.stride(0), c.stride(1), BLOCK_M=v["m"], BLOCK_N=v["n"], BLOCK_K=v["k"],
                                   GROUP_M=8, num_warps=v["w"], num_stages=v["s"])

    def err():
        ref = a.cpu().float() @ bt.cpu().float()
        return ((c.cpu().float() - ref).abs().max() / ref.abs().max()).item()
    return fn, err


def time_cmd(a):
    """As triton_bench.time_cmd: compile call, warmups, event-bracketed samples, profiler kernel time on hardware."""
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
