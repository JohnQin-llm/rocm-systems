#!/usr/bin/env python3
"""Minimal reproducer for the hipBLASLt Stream-K hang under rocjitsu.

A 1x4096 @ 4096x4096 matmul makes hipBLASLt pick a Stream-K kernel
(`..._MT32x16x128_..._SK3_...` for float32) whose 256 workgroups share 128
tiles and synchronise through per-tile flags polled with `s_load_dword ... glc`.
Without the SMEM SGPR-offset fix (#10250) and the scalar cache-policy fix
(#12565) the emulator spins forever; with both it finishes in about a minute.

    rocjitsu --config configs/gfx950_mi355x_kmd.json -- python3 streamk_gemv_repro.py float32

On hardware the same script finishes in about a second; the printed max error
should match the hardware run bit for bit.
"""

import sys
import time

import torch

m, n, k = 1, 4096, 4096
dtype = getattr(torch, sys.argv[1] if len(sys.argv) > 1 else "float32")
a = torch.randn(m, k, device="cuda", dtype=dtype)
b = torch.randn(k, n, device="cuda", dtype=dtype)
torch.cuda.synchronize()
print("inputs ready", flush=True)
start = time.time()
c = a @ b
torch.cuda.synchronize()
print(f"mm done in {time.time() - start:.1f}s wall", flush=True)
ref = (a.double() @ b.double()).to(dtype)
torch.cuda.synchronize()
print("max err", (c - ref).abs().max().item(), flush=True)
