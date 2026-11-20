#!/usr/bin/env python3
"""Draw the figures and diagrams of docs/timing-measured-scheduler.md.

  python3 make_figures.py                  # writes <name>.svg and <name>-dark.svg here
  python3 make_figures.py --preview DIR    # also writes PNG previews of each into DIR

Needs matplotlib only. Every number below is a summary value quoted in the document;
no measurement data is read. Each figure comes in a light and a dark variant; the
document selects one with <picture> and prefers-color-scheme.
"""

import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))

# Surfaces, ink and the categorical slots (fixed order: blue, orange, aqua).
LIGHT = dict(
    surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e8e7e3", axis="#bdbbb4",
    n_dark="#8f8d86", n_light="#d3d1ca", blue="#2a78d6", orange="#eb6834", aqua="#1baf7a",
    tint="#edf3fb", box="#f2f1ee", on_accent="#ffffff",
)
DARK = dict(
    surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#2e2d2b", axis="#55544f",
    n_dark="#8c8a82", n_light="#4b4a46", blue="#3987e5", orange="#d95926", aqua="#199e70",
    tint="#1e2836", box="#252523", on_accent="#ffffff",
)

PX = 100 / 72  # pixels per point at 100 dpi


def rc(t):
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.labelsize": 10,
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "xtick.color": t["axis"],
        "ytick.color": t["axis"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",          # text as outlines: renders identically everywhere
        "svg.hashsalt": "timing-measured-scheduler",
        "figure.facecolor": t["surface"],
        "axes.facecolor": t["surface"],
        "legend.labelcolor": t["ink"],
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------
LINE_PT = 2 * 72 / 100   # 2 px
RING_PT = 2 * 72 / 100   # 2 px marker ring
MARK_PT = 7              # ~9.7 px marker


def chart(t, title, sub, *, left=0.09, bottom=0.14, top=0.72, ncols=1, wspace=0.1,
          legend=None, legend_y=0.835, sharey=False, size=(8, 4.5), right=0.97):
    fig = plt.figure(figsize=size, dpi=100)
    gs = fig.add_gridspec(1, ncols, left=left, right=right, top=top, bottom=bottom, wspace=wspace)
    axes = []
    for i in range(ncols):
        axes.append(fig.add_subplot(gs[0, i], sharey=axes[0] if (sharey and axes) else None))
    fig.text(0.025, 0.955, title, fontsize=13, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.025, 0.893, sub, fontsize=9.5, color=t["ink2"], va="top", linespacing=1.35)
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.018, legend_y), ncol=len(legend),
                   frameon=False, fontsize=9.5, handlelength=1.4, handleheight=0.9, columnspacing=1.8,
                   borderaxespad=0, handletextpad=0.6)
    for ax in axes:
        ax.tick_params(length=0, pad=5)
        ax.set_axisbelow(True)
    return fig, axes


def grid(ax, t, axis):
    ax.grid(axis=axis, color=t["grid"], lw=0.8, zorder=0)


def px2data(ax, dx, dy):
    inv = ax.transData.inverted()
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((dx, dy))
    return abs(x1 - x0), abs(y1 - y0)


def rrect(ax, x0, x1, y0, y1, color, corners=(), r=4, z=3):
    """Filled rectangle in data coords; listed corners (ll, lr, ur, ul) get an r-px radius."""
    rx, ry = px2data(ax, r, r)
    rx = min(rx, (x1 - x0) / 2)
    ry = min(ry, (y1 - y0) / 2)

    def arc(cx, cy, a0, a1):
        return [(cx + rx * math.cos(a), cy + ry * math.sin(a))
                for a in [a0 + (a1 - a0) * k / 8 for k in range(9)]]

    pts = []
    pts += arc(x0 + rx, y0 + ry, math.pi, 1.5 * math.pi) if "ll" in corners else [(x0, y0)]
    pts += arc(x1 - rx, y0 + ry, 1.5 * math.pi, 2 * math.pi) if "lr" in corners else [(x1, y0)]
    pts += arc(x1 - rx, y1 - ry, 0, 0.5 * math.pi) if "ur" in corners else [(x1, y1)]
    pts += arc(x0 + rx, y1 - ry, 0.5 * math.pi, math.pi) if "ul" in corners else [(x0, y1)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))


def hbar(ax, y, x0, x1, h_px, color, *, end=True, gap=(0, 0)):
    """Horizontal bar of h_px thickness; gap=(left_px, right_px) trims it for 2 px spacers."""
    gx, hy = px2data(ax, 1, h_px / 2)
    x0, x1 = x0 + gap[0] * gx, x1 - gap[1] * gx
    rrect(ax, x0, x1, y - hy, y + hy, color, ("lr", "ur") if end else ())


def vbar(ax, x, y0, y1, w_px, color, corners=("ul", "ur")):
    wx, _ = px2data(ax, w_px / 2, 1)
    rrect(ax, x - wx, x + wx, y0, y1, color, corners)


def swatch(color, label):
    return Patch(facecolor=color, edgecolor="none", label=label)


def line_key(t, color, label, marker=True, dashed=False):
    return Line2D([], [], color=color, lw=LINE_PT if not dashed else 1.2, ls=(0, (4, 3)) if dashed else "-",
                  marker="o" if marker else None, ms=MARK_PT, mec=t["surface"], mew=RING_PT, label=label)


# ---------------------------------------------------------------------------
# Diagram helpers (pixel canvas, y grows downward)
# ---------------------------------------------------------------------------
def canvas(t, w, h):
    fig = plt.figure(figsize=(w, h), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 100)
    ax.set_ylim(h * 100, 0)
    ax.axis("off")
    return fig, ax


def kinds(t):
    return {
        "main": dict(fc=t["tint"], ec=t["blue"], lw=1.6, ls="-", title=t["ink"], sub=t["ink2"]),
        "out": dict(fc=t["blue"], ec=t["blue"], lw=1.6, ls="-", title=t["on_accent"], sub=t["on_accent"]),
        "plain": dict(fc=t["box"], ec=t["n_light"], lw=1.2, ls="-", title=t["ink"], sub=t["ink2"]),
        "side": dict(fc=t["box"], ec=t["n_dark"], lw=1.2, ls=(0, (4, 3)), title=t["ink"], sub=t["ink2"]),
        "group": dict(fc="none", ec=t["n_light"], lw=1.2, ls="-", title=t["ink"], sub=t["ink2"]),
    }


def node(ax, t, cx, cy, w, h, title, sub=None, kind="main", tsize=11, ssize=9.5, r=10):
    k = kinds(t)[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=k["fc"], ec=k["ec"], lw=k["lw"], ls=k["ls"], zorder=2))
    tl = title.count("\n") + 1
    th = tl * tsize * PX * 1.22
    if sub:
        sl = sub.count("\n") + 1
        sh = sl * ssize * PX * 1.32
        y = cy - (th + 5 + sh) / 2
        ax.text(cx, y, title, ha="center", va="top", fontsize=tsize, fontweight="bold", color=k["title"],
                linespacing=1.1, zorder=3)
        ax.text(cx, y + th + 5, sub, ha="center", va="top", fontsize=ssize, color=k["sub"],
                linespacing=1.32, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tsize, fontweight="bold",
                color=k["title"], linespacing=1.15, zorder=3)
    return dict(l=cx - w / 2, r=cx + w / 2, t=cy - h / 2, b=cy + h / 2, cx=cx, cy=cy)


def group(ax, t, x0, y0, x1, y1, label):
    k = kinds(t)["group"]
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=14",
                                fc="none", ec=k["ec"], lw=k["lw"], zorder=1))
    ax.text(x0 + 14, y0 + 12, label, ha="left", va="top", fontsize=10.5, fontweight="bold", color=t["ink2"])


def arrow(ax, t, pts, kind="main", dashed=False, lw=1.6):
    color = t["blue"] if kind == "main" else t["n_dark"]
    ls = (0, (4, 3)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=0.55,head_width=0.32",
                                 mutation_scale=14, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                 joinstyle="miter", zorder=1))


def label(ax, t, x, y, s, ha="center", va="center", size=9, color=None, bold=False, bg=True):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or t["ink2"],
            fontweight="bold" if bold else "normal", zorder=4, linespacing=1.25,
            bbox=dict(fc=t["surface"], ec="none", pad=1.5) if bg else None)


# ---------------------------------------------------------------------------
# 0. Overview: how the timing model works
# ---------------------------------------------------------------------------
def overview(t):
    fig, ax = canvas(t, 11, 4.05)
    ax.text(26, 22, "How the timing model works", fontsize=15, fontweight="bold", color=t["ink"], va="top")
    ax.text(26, 50, "Blue: the path from a GPU program to a predicted time.   "
                    "Grey: where the rules come from, and how the result is checked.",
            fontsize=10, color=t["ink2"], va="top")
    cy, w, h, step = 228, 184, 120, 216
    xs = [26 + w / 2 + i * step for i in range(5)]
    boxes = [
        ("Emulator", "runs the GPU program\nfunctionally, with its\ncaches and memory", "main"),
        ("Instruction stream", "per sampled CU: what\neach wave executed and\nwhere memory hit", "main"),
        ("Per-SIMD scheduler", "waves compete, cycle\nby cycle, for issue,\npipes, LDS, memory", "main"),
        ("Kernel terms", "slowest CU + launch,\nXCD stagger, teardown\ntail, write-back", "main"),
        ("Predicted\nkernel time", None, "out"),
    ]
    nb = [node(ax, t, x, cy, w, h, ti, su, kind=k, tsize=12 if k == "out" else 11)
          for x, (ti, su, k) in zip(xs, boxes)]
    for a, b in zip(nb[:-1], nb[1:]):
        arrow(ax, t, [(a["r"], cy), (b["l"] - 1, cy)])
    rules = node(ax, t, (xs[2] + xs[3]) / 2, 112, 450, 58, "Measured rules",
                 "microbenchmarks + thread traces on MI355X; nothing fitted", kind="side")
    for b in nb[2:4]:
        arrow(ax, t, [(b["cx"], rules["b"]), (b["cx"], b["t"] - 1)], kind="side")
    val = node(ax, t, 1074 - 320, 350, 640, 58, "Validated against measured kernel time",
               "held-out set 9, 95 calls scored once: 7.8% median error (queued), 91% within 25%",
               kind="side")
    arrow(ax, t, [(nb[4]["cx"], nb[4]["b"]), (nb[4]["cx"], val["t"] - 1)], kind="side")
    return fig


# ---------------------------------------------------------------------------
# Diagrams that replace the Mermaid sources
# ---------------------------------------------------------------------------
def architecture(t):
    fig, ax = canvas(t, 11.4, 4.0)
    cy, w, h, step = 178, 160, 100, 186
    xs = [25 + w / 2 + i * step for i in range(6)]
    spec = [
        ("Functional\nplane", "executes\nthe kernel", "main"),
        ("Timing\nplane", "caches, memory,\ninstruction costs", "main"),
        ("Instruction\nstream", "96-byte records,\nsampled CUs", "main"),
        ("Replay\nscheduler", "offline, Python;\nused for results", "main"),
        ("Kernel-level\ntime", "launch, stagger,\ntail, write-back", "main"),
        ("Predicted\nkernel time", None, "out"),
    ]
    nb = [node(ax, t, x, cy, w, h, ti, su, kind=k, tsize=10.5, ssize=9) for x, (ti, su, k) in zip(xs, spec)]
    group(ax, t, 12, 92, nb[1]["r"] + 13, nb[1]["b"] + 13, "rocjitsu emulator")
    for a, b in zip(nb[:-1], nb[1:]):
        arrow(ax, t, [(a["r"], cy), (b["l"] - 1, cy)])
    label(ax, t, nb[2]["cx"], nb[2]["b"] + 16, "ROCJITSU_TIMING_ISTREAM", size=8.5)
    label(ax, t, (nb[3]["r"] + nb[4]["l"]) / 2, nb[3]["b"] + 10, "per-CU completion,\nstall causes", size=8.5, va="top")
    rules = node(ax, t, nb[3]["cx"], 48, 290, 54, "Measured rule tables", "§4; each entry cites its measurement",
                 kind="side", tsize=10.5, ssize=9)
    arrow(ax, t, [(nb[3]["cx"], rules["b"]), (nb[3]["cx"], nb[3]["t"] - 1)], kind="side")
    s = node(ax, t, nb[3]["cx"], 330, 270, 70, "C++ port (not merged)",
             "same scheduler in ComputeUnitDes,\nbehind scheduler.enabled", kind="side", tsize=10.5, ssize=9)
    arrow(ax, t, [(nb[1]["cx"], nb[1]["b"] + 13), (nb[1]["cx"], s["cy"]), (s["l"] - 1, s["cy"])],
          kind="side", dashed=True)
    arrow(ax, t, [(s["r"], s["cy"]), (nb[4]["cx"], s["cy"]), (nb[4]["cx"], nb[4]["b"] + 1)],
          kind="side", dashed=True)
    # Key.
    ax.plot([25, 55], [355, 355], color=t["blue"], lw=1.6)
    ax.text(62, 355, "replay prototype (all results)", va="center", fontsize=9, color=t["ink2"])
    ax.plot([25, 55], [378, 378], color=t["n_dark"], lw=1.6, ls=(0, (4, 3)))
    ax.text(62, 378, "C++ port, separate branch", va="center", fontsize=9, color=t["ink2"])
    return fig


def cu_machine(t):
    fig, ax = canvas(t, 11, 4.6)
    group(ax, t, 14, 14, 872, 446, "Compute unit")
    # SIMD panel with two shadow cards: "x4".
    for d in (12, 6):
        ax.add_patch(FancyBboxPatch((36 + d, 58 + d), 552, 360, boxstyle="round,pad=0,rounding_size=12",
                                    fc=t["surface"], ec=t["n_light"], lw=1.0, zorder=1))
    ax.add_patch(FancyBboxPatch((36, 58), 552, 360, boxstyle="round,pad=0,rounding_size=12",
                                fc=t["surface"], ec=t["blue"], lw=1.2, zorder=1))
    ax.text(52, 72, "SIMD  (4 per CU, each with its own pipes)", fontsize=10.5, fontweight="bold",
            color=t["ink2"], va="top")
    cy = 250
    wv = node(ax, t, 136, cy, 168, 120, "Wavefronts", "in-order issue,\nregister scoreboard,\nwait counters")
    arb = node(ax, t, 312, cy, 120, 84, "Arbiter", "s_setprio,\nthen oldest")
    arrow(ax, t, [(wv["r"], cy), (arb["l"] - 1, cy)])
    pipes = [("VALU pipe", "2, 4 or 8 cycles", 150), ("MFMA pipe", "passes of the shape", 250),
             ("SALU", "4 cycles", 350)]
    for ti, su, py in pipes:
        p = node(ax, t, 496, py, 160, 64, ti, su, ssize=9)
        arrow(ax, t, [(arb["r"], cy), (arb["r"] + 16, cy), (arb["r"] + 16, py), (p["l"] - 1, py)])
    group(ax, t, 624, 58, 858, 418, "Shared by the 4 SIMDs")
    shared = [("LDS", "bank rule, §4.4", 140), ("Vector memory", "address unit", 216),
              ("Scalar memory", None, 292), ("v_cndmask_b32 VOP2", "≈4.4 cycles, CU-wide", 368)]
    sb = [node(ax, t, 741, sy, 206, 60, ti, su, kind="plain", tsize=10.5, ssize=9) for ti, su, sy in shared]
    arrow(ax, t, [(601, cy), (623, cy)])
    mem = node(ax, t, 990, 254, 190, 112, "L2 · MALL · HBM", "device bandwidth,\nshared by demand", kind="plain")
    arrow(ax, t, [(sb[1]["r"], 216), (mem["l"] - 1, 230)], kind="side")
    arrow(ax, t, [(sb[2]["r"], 292), (mem["l"] - 1, 278)], kind="side")
    return fig


def scheduling_step(t):
    fig, ax = canvas(t, 11, 3.55)
    cy, h = 165, 104
    a = node(ax, t, 108, cy, 176, h, "Advance", "to the earliest cycle\nany wave can issue")
    d1 = node(ax, t, 335, cy, 206, h, "Ready?", "issue interval elapsed,\nsources ready, s_waitcnt\nmet, no hold or barrier")
    d2 = node(ax, t, 553, cy, 160, h, "Port free?", "target pipe or\nCU-shared port")
    d3 = node(ax, t, 758, cy, 176, h, "Wins arbitration?", "highest s_setprio,\nthen oldest", tsize=10.5)
    iss = node(ax, t, 982, cy, 188, h, "Issue", "set pipe free_at,\nresult-ready cycle,\nhold, memory request",
               kind="out")
    for p, q in ((a, d1), (d1, d2), (d2, d3), (d3, iss)):
        arrow(ax, t, [(p["r"], cy), (q["l"] - 1, cy)])
        if p is not a:
            label(ax, t, (p["r"] + q["l"]) / 2, cy - 14, "yes", size=9)
    st = node(ax, t, 553, 300, 520, 46, "Charge the cycle to exactly one stall cause", kind="plain", tsize=10.5)
    for d, word in ((d1, "no"), (d2, "no"), (d3, "loses")):
        arrow(ax, t, [(d["cx"], d["b"]), (d["cx"], st["t"] - 1)], kind="side")
        label(ax, t, d["cx"] + 7, (d["b"] + st["t"]) / 2, word, ha="left", size=9, bg=False)
    arrow(ax, t, [(st["l"], 300), (a["cx"], 300), (a["cx"], a["b"] + 1)], kind="side")
    arrow(ax, t, [(iss["cx"], iss["t"]), (iss["cx"], 36), (a["cx"], 36), (a["cx"], a["t"] - 1)], kind="side")
    label(ax, t, (a["cx"] + iss["cx"]) / 2, 36, "next instruction", size=9)
    return fig


def spin_collapse(t):
    fig, ax = canvas(t, 10, 4.35)
    xp, xm, xw = 160, 500, 840
    for x, name in ((xp, "Partner workgroup"), (xm, "Memory past L2"), (xw, "Waiting workgroup")):
        node(ax, t, x, 40, 210, 44, name, kind="plain", tsize=10.5)
        ax.plot([x, x], [62, 420], color=t["n_light"], lw=1.2, ls=(0, (4, 3)), zorder=0)
    arrow(ax, t, [(xp, 108), (xm - 2, 108)])
    label(ax, t, (xp + xm) / 2, 94, "s_store_dword flag, glc", size=9.5, color=t["ink"])
    ax.add_patch(FancyBboxPatch((xm - 175, 131), 350, 34, boxstyle="round,pad=0,rounding_size=6",
                                fc=t["box"], ec=t["n_light"], lw=1.0, zorder=2))
    ax.text(xm, 148, "flag visible ≈1.3–1.5k cycles after the store", ha="center", va="center",
            fontsize=9.5, color=t["ink"], zorder=3)
    # Poll loop frame.
    ax.add_patch(FancyBboxPatch((412, 186), 500, 118, boxstyle="round,pad=0,rounding_size=8",
                                fc="none", ec=t["n_dark"], lw=1.1, zorder=1))
    label(ax, t, 424, 199, "poll loop: hardware runs it 4–8 times, the emulator records 34–213",
          ha="left", size=9, color=t["ink2"], bold=False)
    arrow(ax, t, [(xw, 240), (xm + 2, 240)])
    label(ax, t, (xm + xw) / 2, 227, "s_load_dword flag, glc", size=9.5, color=t["ink"])
    arrow(ax, t, [(xm, 284), (xw - 2, 284)], kind="side", dashed=True)
    label(ax, t, (xm + xw) / 2, 271, "≈400 or ≈700 cycles, by address", size=9.5)
    node(ax, t, 712, 362, 540, 70, "Replay rule",
         "keep one iteration and hold the wave until\nthe partner's flag, plus one poll (1400 cycles)",
         tsize=10.5, ssize=9.5)
    return fig


def rules_validation(t):
    fig, ax = canvas(t, 11, 3.95)
    src = [("Microbenchmarks", "one workgroup,\ns_memtime, HW_ID", "plain", 60),
           ("Thread traces", "warm iteration,\nreal kernels", "plain", 150),
           ("Clock and power", "back-to-back loop", "plain", 240),
           ("Perf counters", "aggregates only", "side", 330)]
    rule = node(ax, t, 368, 195, 170, 84, "Rule table\nentry", "cites its measurement", ssize=9)
    for i, (ti, su, k, y) in enumerate(src):
        b = node(ax, t, 116, y, 196, 72, ti, su, kind=k, tsize=10.5, ssize=9)
        if k == "side":
            arrow(ax, t, [(b["r"], y), (262, y), (262, 222), (rule["l"] - 1, 222)], kind="side", dashed=True)
        else:
            ax.plot([b["r"], 246], [y, y], color=t["n_dark"], lw=1.6, zorder=1)
    ax.plot([246, 246], [60, 240], color=t["n_dark"], lw=1.6, solid_capstyle="projecting", zorder=1)
    arrow(ax, t, [(246, 182), (rule["l"] - 1, 182)], kind="side")
    label(ax, t, 268, 300, "cross-check", ha="left", size=8.5, bg=False)
    sch = node(ax, t, 572, 195, 170, 84, "Scheduler", "replay or C++")
    arrow(ax, t, [(rule["r"], 195), (sch["l"] - 1, 195)])
    m = node(ax, t, 790, 110, 184, 80, "Mechanism check", "per-instruction gaps\nvs thread trace", tsize=10.5)
    k = node(ax, t, 790, 282, 184, 80, "Kernel check", "kernel time vs\nprofiler timing", tsize=10.5)
    for b in (m, k):
        arrow(ax, t, [(sch["r"], 195), (sch["r"] + 20, 195), (sch["r"] + 20, b["cy"]), (b["l"] - 1, b["cy"])])
    p = node(ax, t, 994, 282, 168, 100, "Scoring", "development →\npre-registration →\nheld-out set", kind="out")
    arrow(ax, t, [(k["r"], 282), (p["l"] - 1, 282)])
    arrow(ax, t, [(m["cx"], m["t"]), (m["cx"], 30), (rule["cx"], 30), (rule["cx"], rule["t"] - 1)],
          kind="side", dashed=True)
    label(ax, t, (m["cx"] + rule["cx"]) / 2, 30, "a gap that does not match points to the wrong rule", size=9)
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def progression(t):
    steps = [  # label, median error %, within 25 %, kind (start / rule / ref / total)
        ("scheduler\n+ LDS rule", 11.4, 74, "start"),
        ("+ spin\ncollapse", 11.1, 79, "rule"),
        ("+ beyond-L2\nrescale", 9.9, 80, "rule"),
        ("+ scratch,\nstore ack", 9.1, 81, "rule"),
        ("+ demand\nbandwidth", 9.0, 84, "rule"),
        ("+ kernel-\nlevel terms", 9.3, 86, "rule"),
        ("regime-\nmatched\nreference", 8.6, 87, "ref"),
        ("+ occupancy,\nsaturation", 8.8, 89, "rule"),
        ("+ slowest\nline, slot\nrelease, cap", 8.1, 91, "rule"),
        ("+ LDS lanes,\nL1 hash", 8.2, 92, "rule"),
        ("+ power\nclock, L2\nbandwidth", 7.9, 94, "rule"),
        ("+ frozen-\nmodel rules\n(§4.8)", 6.1, 95, "rule"),
        ("queued\nreference\n(primary)", 7.8, 94, "total"),
    ]
    fitted = 19.6  # shipped fitted composition, queued reference, in-sample (56.5 % within 25 %)
    xs = list(range(len(steps) - 1)) + [len(steps) - 0.6]   # set the queued total apart
    fig, (ax,) = chart(
        t, "Measured rules cut the development error to 7.8% queued (6.1% idle gap), vs 19.6% fitted",
        "85 development kernels (in-sample), measured rules added step by step; nothing fitted except the "
        "power limit",
        left=0.09, bottom=0.235, top=0.72, size=(12, 5.0), right=0.985,
        legend=[swatch(t["n_light"], "scheduler total"), swatch(t["blue"], "rule lowers error"),
                swatch(t["orange"], "rule raises error"), swatch(t["n_dark"], "reference change"),
                line_key(t, t["n_dark"], "fitted model (in-sample)", marker=False, dashed=True)])
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(0, 23)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_yticklabels(["0%", "5%", "10%", "15%", "20%"])
    ax.set_ylabel("median error")
    ax.spines["left"].set_visible(False)
    grid(ax, t, "y")
    ax.axhline(fitted, color=t["n_dark"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.text(xs[-1] + 0.55, fitted + 0.25, "fitted model, queued, in-sample: 19.6%", ha="right", va="bottom",
            fontsize=9, color=t["ink2"])
    # Reference change: mixed kernel_us before, regime-matched measurements after.
    ax.axvline(5.5, color=t["ink2"], lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(5.42, 22.6, "vs mixed kernel_us", ha="right", va="top", fontsize=9, color=t["ink2"])
    ax.text(5.58, 22.6, "vs regime-matched (idle gap)", ha="left", va="top", fontsize=9, color=t["ink2"])
    bw = 40
    prev = None
    for i, (x, (_, err, _, kind)) in enumerate(zip(xs, steps)):
        if kind in ("start", "total"):
            vbar(ax, x, 0, err, bw, t["n_light"])
            ax.text(x, err + 0.3, f"{err:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        else:
            lo, hi = sorted((prev, err))
            _, dy = px2data(ax, 1, 3)
            hi = max(hi, lo + dy)
            col = t["n_dark"] if kind == "ref" else t["blue"] if err < prev else t["orange"]
            vbar(ax, x, lo, hi, bw, col, corners=("ll", "lr", "ul", "ur"))
            wx, _ = px2data(ax, bw / 2, 1)
            ax.plot([xs[i - 1] + wx, x - wx], [prev, prev], color=t["n_dark"], lw=0.9, zorder=2)
            txt = f"{err - prev:+.1f}".replace("-", "−")
            if i == len(steps) - 2:
                ax.text(x, hi + 0.3, f"{txt}\n", ha="center", va="bottom", fontsize=9, color=t["ink2"],
                        linespacing=1.3)
                ax.text(x, hi + 0.3, f"= {err:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
            else:
                ax.text(x, hi + 0.3, txt, ha="center", va="bottom", fontsize=9, color=t["ink2"])
        prev = err
    ax.set_xticks(xs)
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.5, color=t["ink2"], linespacing=1.12)
    tr = ax.get_xaxis_transform()
    for x, (_, _, share, _) in zip(xs, steps):
        ax.text(x, -0.215, f"{share}%", transform=tr, ha="center", va="top", fontsize=9.5, color=t["ink"])
    ax.text(-0.6, -0.215, "within 25%:", transform=tr, ha="right", va="top", fontsize=9.5, color=t["ink2"])
    return fig


def dot_key(t, color, label):
    return Line2D([], [], ls="none", marker="o", ms=MARK_PT + 1, mfc=color, mec=t["surface"], mew=RING_PT,
                  label=label)


def families(t):
    rows = [  # family, n dev / held-out, fitted (dev, in-sample), frozen model dev, frozen model held-out 9
        ("RMSNorm", "6 / 6", 26.2, 2.6, 8.7),
        ("Triton softmax", "10 / 10", 24.4, 3.1, 5.3),
        ("attention", "10 / 10", 17.9, 4.8, 15.7),
        ("hipBLASLt GEMM", "20 / 30", 22.9, 6.3, 11.5),
        ("Triton flash attention", "6 / 6", 30.4, 8.8, 9.1),
        ("swiglu", "4 / 4", 34.1, 9.8, 5.1),
        ("fp8 matmul", "10 / 10", 14.1, 11.4, 10.9),
        ("Triton matmul", "15 / 15", 16.7, 12.0, 8.1),
        ("LayerNorm", "4 / 4", 24.3, 16.6, 2.2),
    ]
    rows = sorted(rows, key=lambda r: r[4])
    total = ("all", "85 / 95", 19.6, 7.8, 8.5)
    fig, (ax,) = chart(
        t, "Overall, held-out error is close to development and well below the fitted model",
        "Median error per family, queued reference, development-table metric; left is better.\n"
        "Development numbers are in-sample (fitted model and scheduler); held-out set 9 was scored once.",
        left=0.27, bottom=0.1, top=0.7, legend_y=0.79, size=(8.5, 5.4),
        legend=[dot_key(t, t["blue"], "scheduler, held-out 9"),
                dot_key(t, t["aqua"], "scheduler, development"),
                dot_key(t, t["orange"], "fitted model, development")])
    ys = [len(rows) + 0.5 - i for i in range(len(rows))] + [0]   # top to bottom, total set apart
    allrows = rows + [total]
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.6, ys[0] + 0.7)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0%", "10%", "20%", "30%"])
    ax.spines["left"].set_visible(False)
    grid(ax, t, "x")
    ax.axhline(0.75, color=t["n_light"], lw=0.8, zorder=1)
    for y, (fam, n, old, dev, ho) in zip(ys, allrows):
        lo, hi = min(old, dev, ho), max(old, dev, ho)
        ax.plot([lo, hi], [y, y], color=t["n_light"], lw=2, solid_capstyle="round", zorder=2)
        for v, col, z in ((old, t["orange"], 3), (dev, t["aqua"], 4), (ho, t["blue"], 5)):
            ax.plot([v], [y], ls="none", marker="o", ms=MARK_PT + 1, mfc=col, mec=t["surface"], mew=RING_PT,
                    zorder=z, clip_on=False)
        ax.text(ho, y + 0.27, f"{ho:.1f}", ha="center", va="bottom", fontsize=8.5, color=t["ink"], zorder=6,
                fontweight="bold" if fam == "all" else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{f}  (n={n})" for f, n, *_ in allrows], fontsize=9.5, color=t["ink"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.text(0.0, -0.075, "Numbers: held-out error. n: development / held-out (held-out hipBLASLt includes 9 "
            "two-kernel calls).", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=t["ink2"])
    return fig


def lds(t):
    ms = [1, 2, 4, 8, 16, 32, 64]
    cap = 64

    def cost(floor, k):
        return [min(cap, max(floor, k * m)) for m in ms]

    panels = [  # title, reads (floor, per-word), writes (floor, per-word)
        ("b32 lanes", (2, 2), (4, 2)),
        ("b64 lanes", (2, 1), (6, 2)),
        ("b128 lanes", (4, 1), (13, 2)),
    ]
    fig, axes = chart(
        t, "LDS cost grows with bank conflicts, above a fixed floor",
        "Measured on MI355X. m = most distinct 4-byte words one bank serves; broadcasts to one word are free.\n"
        "Wide reads: m is summed over lane groups (b64: two 32-lane halves, b128: four 16-lane quarters).",
        left=0.085, bottom=0.15, top=0.66, legend_y=0.78, ncols=3, wspace=0.08, sharey=True,
        legend=[line_key(t, t["blue"], "reads"), line_key(t, t["orange"], "writes"),
                line_key(t, t["n_dark"], "cap 64 cycles", marker=False, dashed=True)])
    for j, (ax, (name, rd, wr)) in enumerate(zip(axes, panels)):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xlim(0.8, 80)
        ax.set_ylim(1.6, 90)
        ax.axhline(cap, color=t["n_dark"], lw=1.1, ls=(0, (4, 3)), zorder=1)
        w, r = cost(*wr), cost(*rd)
        ax.plot(ms, w, color=t["orange"], lw=LINE_PT, marker="o", ms=MARK_PT, mec=t["surface"], mew=RING_PT,
                zorder=3, clip_on=False)
        ax.plot(ms, r, color=t["blue"], lw=LINE_PT, marker="o", ms=MARK_PT, mec=t["surface"], mew=RING_PT,
                zorder=4, clip_on=False)
        for v in (w[0], r[0]):
            ax.text(1, v * 1.16, f"{v}", va="bottom", ha="center", fontsize=9, color=t["ink"], zorder=5)
        ax.set_xticks([1, 4, 16, 64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_yticks([2, 4, 8, 16, 32, 64])
        ax.set_yticklabels(["2", "4", "8", "16", "32", "64"])
        ax.minorticks_off()
        ax.spines["left"].set_visible(j == 0)
        grid(ax, t, "y")
        ax.set_title(name, loc="left", fontsize=10.5, fontweight="bold", color=t["ink"], pad=6)
        if j == 0:
            ax.text(70, 2.3, "read = write\nfor m ≥ 2", ha="right", va="bottom", fontsize=9, color=t["ink2"])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("cycles per instruction")
    axes[1].set_xlabel("m, distinct words on the busiest bank (log scale)")
    return fig


def launch_timeline(t):
    head, gap = 0.12, 0.18
    work = 2.0                 # illustrative per-XCD work
    waves = 8                  # illustrative resident waves per CU
    tail = 1.10 + 0.006 * waves
    dirty_mb = 8               # illustrative dirty L2 at release
    wb = min(0.115 * dirty_mb, 2.3)
    starts = [head + k * gap for k in range(8)]
    end = max(s + work for s in starts)
    total = end + tail + wb
    fig, (ax,) = chart(
        t, "After an idle gap, the last XCD to start sets the kernel's end",
        f"Illustrative kernel: {work:.0f} µs of work per XCD, {waves} resident waves per CU, "
        f"{dirty_mb} MB of dirty L2 at release",
        left=0.13, bottom=0.13, top=0.73,
        legend=[swatch(t["n_dark"], "head"), swatch(t["n_light"], "XCD not started yet"),
                swatch(t["blue"], "work"), swatch(t["orange"], "teardown tail"),
                swatch(t["aqua"], "write-back")])
    ax.set_xlim(0, 6.2)
    ax.set_ylim(-1.25, 8.6)
    ax.spines["left"].set_visible(False)
    grid(ax, t, "x")
    bh = 13
    for k, s in enumerate(starts):
        y = 8 - k
        hbar(ax, y, 0, head, bh, t["n_dark"], end=False, gap=(0, 1))
        if s > head:
            hbar(ax, y, head, s, bh, t["n_light"], end=False, gap=(1, 1))
        hbar(ax, y, s, s + work, bh, t["blue"], gap=(1, 0))
    hbar(ax, 0, end, end + tail, bh, t["orange"], end=False, gap=(0, 1))
    hbar(ax, 0, end + tail, total, bh, t["aqua"], gap=(1, 0))
    ax.text(total + 0.06, 0, f"{total:.2f} µs", va="center", fontsize=10, fontweight="bold")
    ax.text(end + tail / 2, 0.42, f"1.10 + 0.006 × {waves}", ha="center", va="bottom", fontsize=8.5,
            color=t["ink2"])
    ax.text(end + tail + wb / 2, 0.42, f"0.115 × {dirty_mb} MB", ha="center", va="bottom", fontsize=8.5,
            color=t["ink2"])
    ax.plot([end, end], [-0.3, 8.4], color=t["ink2"], lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(end + 0.08, 7.5, f"last-started XCD\nsets the end ({end:.2f} µs)", va="center", fontsize=9.5,
            color=t["ink"], linespacing=1.3)
    ax.annotate("", xy=(starts[-1], -0.75), xytext=(starts[0], -0.75),
                arrowprops=dict(arrowstyle="<->", color=t["ink2"], lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(starts[-1] + 0.08, -0.75, "XCD stagger: 0.18 µs apart", ha="left", va="center",
            fontsize=9, color=t["ink2"])
    ax.set_yticks([8 - k for k in range(8)] + [0])
    ax.set_yticklabels([f"XCD {k + 1}" for k in range(8)] + ["kernel end"], color=t["ink"])
    ax.set_xlabel("time since launch (µs)")
    return fig


def launch_regimes(t):
    fig, (ax,) = chart(
        t, "Idle-gap launches pay the XCD stagger: 2.49 µs vs 1.40 µs",
        "Fixed time per kernel from bench microbenchmarks, before release write-back and the 0.006 µs/wave "
        "tail term.\nOn 95 real kernels, idle-gap launch is ≈0.96 µs slower than queued (median).",
        left=0.2, bottom=0.15, top=0.66, legend_y=0.78,
        legend=[swatch(t["n_dark"], "head (+ skew)"), swatch(t["n_light"], "XCD stagger"),
                swatch(t["orange"], "teardown tail")])
    bars = [
        ("after an idle gap", [("head", 0.12, "n_dark"), ("XCD stagger", 1.27, "n_light"),
                               ("tail", 1.10, "orange")]),
        ("queued behind\nanother kernel", [("head + skew", 0.15, "n_dark"), ("tail", 1.25, "orange")]),
    ]
    ax.set_xlim(0, 3.4)
    ax.set_ylim(-0.6, 1.6)
    ax.spines["left"].set_visible(False)
    grid(ax, t, "x")
    bh = 26
    _, hy = px2data(ax, 1, bh / 2)
    for i, (_, segs) in enumerate(bars):
        y = 1 - i
        left = 0
        for j, (name, w, col) in enumerate(segs):
            last = j == len(segs) - 1
            hbar(ax, y, left, left + w, bh, t[col], end=last, gap=(1 if j else 0, 0 if last else 1))
            if j == 0:
                ax.text(left, y - hy - 0.05, f"{name} {w:.2f}", ha="left", va="top", fontsize=9, color=t["ink2"])
            else:
                ax.text(left + w / 2, y + hy + 0.05, f"{name} {w:.2f}", ha="center", va="bottom", fontsize=9,
                        color=t["ink2"])
            left += w
        ax.text(left + 0.06, y, f"{left:.2f} µs + work", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks([1, 0])
    ax.set_yticklabels([b[0] for b in bars], color=t["ink"], fontsize=10)
    ax.set_xlabel("fixed time per kernel (µs)")
    return fig


FIGURES = [
    ("overview", overview),
    ("architecture", architecture),
    ("cu-machine", cu_machine),
    ("scheduling-step", scheduling_step),
    ("spin-collapse", spin_collapse),
    ("rules-validation", rules_validation),
    ("error-progression", progression),
    ("family-error", families),
    ("lds-bank-cost", lds),
    ("launch-timeline", launch_timeline),
    ("launch-regimes", launch_regimes),
]


def main():
    preview = None
    if len(sys.argv) == 3 and sys.argv[1] == "--preview":
        preview = sys.argv[2]
        os.makedirs(preview, exist_ok=True)
    for name, draw in FIGURES:
        for suffix, theme in (("", LIGHT), ("-dark", DARK)):
            with plt.rc_context(rc(theme)):
                fig = draw(theme)
                fig.savefig(os.path.join(OUT, f"{name}{suffix}.svg"), format="svg",
                            facecolor=theme["surface"], metadata={"Date": None})
                if preview:
                    fig.savefig(os.path.join(preview, f"{name}{suffix}.png"), dpi=100,
                                facecolor=theme["surface"])
                plt.close(fig)


if __name__ == "__main__":
    main()
