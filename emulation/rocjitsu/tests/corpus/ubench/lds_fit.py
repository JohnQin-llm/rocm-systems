#!/usr/bin/env python3
"""Compare the local-data-share granule model with `mb lds` measurements.

  lds_fit.py mb_lds.out [--granule 8] [--banks 64] [--max-ways 32]

The model is coalesce.h lds_granule_cycles(): distinct addresses, aligned to
the granule, spread over `banks`; a lane wider than the granule covers several
consecutive banks; the access costs its worst bank multiplicity, capped. The
waited single-wavefront access costs that once on the port and once on the
return, so the measured cycles are fitted as base + slope * ways.
"""
import argparse
import re


def ways(width, stride, banks, granule, cap):
    g = max(4, min(max(4, width), granule))
    pieces = (max(4, width) + g - 1) // g
    mult = {}
    for addr in {(lane * stride) % 65536 // g * g for lane in range(64)}:
        for p in range(pieces):
            b = (addr // g + p) % banks
            mult[b] = mult.get(b, 0) + 1
    return min(max(mult.values()), cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("measurements")
    ap.add_argument("--granule", type=int, default=8)
    ap.add_argument("--banks", type=int, default=64)
    ap.add_argument("--max-ways", type=int, default=32)
    a = ap.parse_args()
    meas = {}
    for line in open(a.measurements):
        m = re.match(r"lds_b(\d+)_waited_stride(\d+) ([\d.]+)", line)
        if m:
            meas[(int(m.group(1)) // 8, int(m.group(2)))] = float(m.group(3))
    xs = [(ways(w, s, a.banks, a.granule, a.max_ways), t) for (w, s), t in sorted(meas.items())]
    n = len(xs)
    sx, sy = sum(x for x, _ in xs), sum(y for _, y in xs)
    sxx, sxy = sum(x * x for x, _ in xs), sum(x * y for x, y in xs)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    base = (sy - slope * sx) / n
    rmse = (sum((base + slope * x - y) ** 2 for x, y in xs) / n) ** 0.5
    print(f"base {base:.1f} cycles, {slope:.2f} cycles per way, rmse {rmse:.1f}")
    for ((w, s), t), (x, _) in zip(sorted(meas.items()), xs):
        print(f"  b{w * 8:<4d} stride {s:4d}  ways {x:3d}  measured {t:6.1f}  model {base + slope * x:6.1f}")


if __name__ == "__main__":
    main()
