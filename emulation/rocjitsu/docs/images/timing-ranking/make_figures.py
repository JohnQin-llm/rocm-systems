#!/usr/bin/env python3
"""Draw the figures of docs/timing-variant-ranking.md as hand-laid-out SVG.

  python3 make_figures.py        # writes *.svg next to this file

No dependencies beyond the standard library, so the figures can be regenerated
anywhere. Each figure draws its own white card, so it reads the same on light
and dark pages.
"""

import os
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "Inter, Segoe UI, Helvetica Neue, Helvetica, Arial, sans-serif"
MONO = "JetBrains Mono, SFMono-Regular, Consolas, Menlo, monospace"

# name: (main, tint, deep)
PAL = {
    "slate": ("#64748b", "#f1f5f9", "#334155"),
    "red": ("#e11d48", "#fff1f2", "#9f1239"),
    "blue": ("#2563eb", "#eff6ff", "#1e3a8a"),
    "indigo": ("#4f46e5", "#eef2ff", "#312e81"),
    "teal": ("#0d9488", "#f0fdfa", "#134e4a"),
    "green": ("#059669", "#ecfdf5", "#064e3b"),
    "violet": ("#7c3aed", "#f5f3ff", "#4c1d95"),
    "amber": ("#d97706", "#fffbeb", "#78350f"),
    "ink": ("#0f172a", "#f8fafc", "#020617"),
}
MUTED = "#64748b"
LINE = "#94a3b8"


def tw(text, size, bold=False):
    """Rough rendered width of text, for layout checks."""
    return len(text) * size * (0.60 if bold else 0.55)


class Svg:
    def __init__(self, w, h, title, subtitle=""):
        self.w, self.h = w, h
        self.parts = []
        self.markers = set()
        self.grads = set()
        self.title, self.subtitle = title, subtitle

    def add(self, s):
        self.parts.append(s)

    def grad(self, color):
        self.grads.add(color)
        return f"url(#g-{color})"

    def marker(self, color):
        self.markers.add(color)
        return f"url(#m-{color.strip('#')})"

    def text(self, x, y, s, size=13, color="#0f172a", weight=400, anchor="start", family=FONT, italic=False):
        style = ' font-style="italic"' if italic else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}"{style}>{escape(s)}</text>')

    def panel(self, x, y, w, h, title, pal):
        main, tint, deep = PAL[pal]
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{tint}" '
                 f'stroke="{main}" stroke-opacity="0.35" stroke-width="1.5"/>')
        pw = tw(title, 12.5, True) + 28
        self.add(f'<rect x="{x + 16}" y="{y - 13}" width="{pw:.0f}" height="26" rx="13" fill="{self.grad(pal)}"/>')
        self.text(x + 16 + pw / 2, y + 4.5, title, 12.5, "#ffffff", 700, "middle")

    def node(self, x, y, w, h, title, sub="", pal="blue", sub2=""):
        main, tint, deep = PAL[pal]
        self.add(f'<g filter="url(#shadow)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                 f'fill="#ffffff" stroke="{main}" stroke-opacity="0.45" stroke-width="1.2"/></g>')
        self.add(f'<rect x="{x}" y="{y + 10}" width="5" height="{h - 20}" rx="2.5" fill="{main}"/>')
        lines = [s for s in (sub, sub2) if s]
        for s_, size, bold in [(title, 14.5, True)] + [(l, 11.5, False) for l in lines]:
            if tw(s_, size, bold) > w - 26:
                print(f"  overflow: {s_!r} needs {tw(s_, size, bold):.0f} of {w - 26}")
        top = y + h / 2 - (len(lines) * 8.5) + 4
        self.text(x + 18, top, title, 14.5, deep, 700)
        for i, s in enumerate(lines):
            self.text(x + 18, top + 19 + i * 16, s, 11.5, MUTED)

    def chip(self, x, y, s, pal, size=11.5, filled=True):
        main, tint, deep = PAL[pal]
        w = tw(s, size, True) + 18
        fill = main if filled else tint
        color = "#ffffff" if filled else deep
        self.add(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{size + 11}" rx="{(size + 11) / 2}" '
                 f'fill="{fill}" stroke="{main}" stroke-width="{0 if filled else 1}"/>')
        self.text(x + w / 2, y + size + 4.5, s, size, color, 700, "middle")
        return w

    def cylinder(self, x, y, w, h, title, sub, pal):
        main, tint, deep = PAL[pal]
        e = 10
        self.add(f'<g filter="url(#shadow)"><path d="M{x},{y + e} a{w / 2},{e} 0 0,1 {w},0 v{h - 2 * e} '
                 f'a{w / 2},{e} 0 0,1 {-w},0 z" fill="#ffffff" stroke="{main}" stroke-opacity="0.6" '
                 f'stroke-width="1.3"/></g>')
        self.add(f'<ellipse cx="{x + w / 2}" cy="{y + e}" rx="{w / 2}" ry="{e}" fill="{tint}" '
                 f'stroke="{main}" stroke-opacity="0.6" stroke-width="1.3"/>')
        self.text(x + w / 2, y + h / 2 + 8, title, 13.5, deep, 700, "middle")
        self.text(x + w / 2, y + h / 2 + 26, sub, 11.5, MUTED, 400, "middle")

    def arrow(self, pts, color=LINE, label=None, dashed=False, width=1.8, label_at=0.5, label_dy=-6):
        d = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} '
                 f'stroke-linejoin="round" marker-end="{self.marker(color)}"/>')
        if label:
            # Label sits on the longest segment, at label_at along it.
            segs = list(zip(pts, pts[1:]))
            (x1, y1), (x2, y2) = max(segs, key=lambda s: abs(s[1][0] - s[0][0]) + abs(s[1][1] - s[0][1]))
            lx, ly = x1 + (x2 - x1) * label_at, y1 + (y2 - y1) * label_at
            w = tw(label, 11) + 12
            self.add(f'<rect x="{lx - w / 2:.1f}" y="{ly + label_dy - 13:.1f}" width="{w:.0f}" height="18" rx="9" '
                     f'fill="#ffffff" stroke="{color}" stroke-opacity="0.5"/>')
            self.text(lx, ly + label_dy, label, 11, color if color != LINE else "#475569", 600, "middle")

    def render(self, name):
        defs = ['<filter id="shadow" x="-10%" y="-10%" width="120%" height="140%">'
                '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.10"/></filter>']
        for c in sorted(self.grads):
            main, _, deep = PAL[c]
            defs.append(f'<linearGradient id="g-{c}" x1="0" y1="0" x2="1" y2="1">'
                        f'<stop offset="0" stop-color="{main}"/><stop offset="1" stop-color="{deep}"/></linearGradient>')
        for c in sorted(self.markers):
            defs.append(f'<marker id="m-{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.title)}">'
                f'<defs>{"".join(defs)}</defs>'
                f'<rect x="1" y="1" width="{self.w - 2}" height="{self.h - 2}" rx="22" fill="#ffffff" stroke="#e2e8f0"/>')
        title = (f'<text x="34" y="44" font-family="{FONT}" font-size="21" font-weight="800" fill="#0f172a">'
                 f'{escape(self.title)}</text>')
        if self.subtitle:
            title += (f'<text x="34" y="66" font-family="{FONT}" font-size="13" fill="{MUTED}">'
                      f'{escape(self.subtitle)}</text>')
        body = head + title + "".join(self.parts) + "</svg>\n"
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(body)
        print("wrote", name, f"{self.w}x{self.h}")


def fig_architecture():
    s = Svg(1420, 610, "System architecture",
            "Unmodified programs run on the CPU; the timing plane costs every dispatch; calibrated knobs turn that into a prediction.")
    top, h = 112, 470
    s.panel(30, top, 190, h, "APPLICATION", "slate")
    s.panel(260, top, 240, h, "FUNCTIONAL EMULATION", "red")
    s.panel(540, top, 270, h, "TIMING PLANE", "blue")
    s.panel(850, top, 290, h, "CALIBRATION + PREDICTION", "green")
    s.panel(1180, top, 210, h, "USES", "violet")

    s.node(45, 300, 160, 96, "Your program", "PyTorch · Triton ·", "slate", "hipBLASLt programs")

    s.node(275, 150, 210, 96, "KFD / HSA emulation", "LD_PRELOAD · no driver,", "red", "no GPU needed")
    s.node(275, 290, 210, 96, "Command processor", "AQL packets → dispatches", "red")
    s.node(275, 430, 210, 96, "Functional VM", "8 XCD × 36 CU", "red", "every instruction executed")
    s.arrow([(205, 348), (240, 348), (240, 198), (273, 198)])
    s.arrow([(380, 246), (380, 288)])
    s.arrow([(380, 386), (380, 428)])

    ys = [140, 255, 370, 485]
    s.node(555, ys[0], 240, 82, "Collector", "instruction stream →", "blue", "timing events")
    s.node(555, ys[1], 240, 82, "CU model", "issue ports · scoreboard", "blue", "s_waitcnt counters")
    s.node(555, ys[2], 240, 82, "Memory hierarchy", "L1 → L2 → fabric", "blue", "→ MALL → DRAM")
    s.node(555, ys[3], 240, 82, "Dispatch model", "bandwidth ledger", "blue", "+ composition")
    s.arrow([(485, 478), (522, 478), (522, 181), (553, 181)], "#e11d48", "retired instructions", label_at=0.55)
    for a, b in zip(ys, ys[1:]):
        s.arrow([(675, a + 82), (675, b - 2)], "#2563eb")

    s.node(905, ys[0], 220, 82, "Timing trace", "one JSON line", "green", "per dispatch")
    s.cylinder(905, ys[1], 220, 82, "Calibrated knobs", "5 fits · two stages", "green")
    s.node(905, ys[2], 220, 82, "recompose()", "terms + knobs", "green", "→ cycles")
    s.node(905, ys[3], 220, 82, "Prediction", "geometric mean", "green", "of the 5 fits")
    s.arrow([(795, 526), (832, 526), (832, 172), (903, 172)], "#2563eb", "trace", label_at=0.5)
    s.arrow([(905, 200), (880, 200), (880, 411), (903, 411)], "#059669")
    s.arrow([(1015, ys[1] + 82), (1015, ys[2] - 2)], "#059669")
    s.arrow([(1015, ys[2] + 82), (1015, ys[3] - 2)], "#059669")

    s.node(1195, ys[0], 180, 82, "Diagnosis", "bytes + terms per level", "violet", "match hw counters")
    s.node(1195, 300, 180, 82, "Variant ranking", "GEMM 87–90% pairwise", "violet", "Triton 74–80%")
    s.node(1195, ys[3], 180, 82, "Agent feedback", "2.58× vs 2.69× with", "violet", "hardware feedback")
    s.arrow([(1125, 181), (1193, 181)], "#7c3aed")
    s.arrow([(1125, 500), (1158, 500), (1158, 341), (1193, 341)], "#7c3aed")
    s.arrow([(1125, 526), (1193, 526)], "#7c3aed")
    s.render("architecture.svg")


def fig_memory():
    s = Svg(1200, 740, "Memory hierarchy and write policy",
            "One request per memory instruction; each level serves its hits and passes its misses on.")
    x, w, cx = 50, 640, 300
    s.node(x, 100, w, 58, "Wavefront memory instruction", "one request carrying every line its lanes touch", "slate")
    s.node(x, 192, w, 70, "L1 vector cache", "loads allocate · stores write through, no allocate", "blue")
    s.chip(x + w - 185, 214, "×256 · one per CU", "blue", filled=False)
    s.node(x, 300, w, 124, "L2 cache", "write-back · dirty bit per line · LRU", "indigo")
    s.chip(x + w - 200, 316, "×8 · 4 MB per XCD", "indigo", filled=False)
    for i in range(8):
        bx = x + 20 + i * 76
        s.add(f'<rect x="{bx}" y="384" width="66" height="26" rx="7" fill="#eef2ff" stroke="#4f46e5" stroke-opacity="0.5"/>')
        s.text(bx + 33, 401, f"XCD {i}", 11, "#312e81", 700, "middle")
    s.node(x, 462, w, 58, "Fabric", "64 B interleave across the memory channels", "slate")
    s.node(x, 558, w, 58, "MALL · memory-side cache", "256 MB · write-back", "teal")
    s.node(x, 654, w, 58, "HBM DRAM", "row-buffer locality", "ink")
    s.arrow([(cx, 158), (cx, 190)])
    s.arrow([(cx, 262), (cx, 298)], LINE, "load miss · every store", label_dy=5)
    s.arrow([(cx, 424), (cx, 460)], LINE, "miss", label_dy=5)
    s.arrow([(cx, 520), (cx, 556)])
    s.arrow([(cx, 616), (cx, 652)], LINE, "miss · dirty eviction", label_dy=5)
    s.arrow([(560, 424), (560, 460)], "#d97706", dashed=True)
    s.text(572, 446, "dirty eviction → async write-back", 11, "#b45309", 600)

    rx, rw = 750, 410
    s.text(rx, 104, "Store policy (from the instruction bits)", 13, "#0f172a", 700)
    s.node(rx, 116, rw, 92, "Plain store", "allocates dirty in L2, never read; written", "indigo",
           "back on eviction or at the release flush")
    s.chip(rx + rw - 62, 128, "RW", "indigo")
    s.node(rx, 222, rw, 92, ".cs  streaming", "allocated least-recently-used: evicted", "teal",
           "early, little left at the end")
    s.chip(rx + rw - 88, 234, "sc0 nt", "teal")
    s.node(rx, 328, rw, 92, ".wt  write-through", "passes L1 and L2; absorbed by the", "slate",
           "memory-side cache")
    s.chip(rx + rw - 96, 340, "sc0 sc1", "slate")
    s.arrow([(rx, 162), (692, 330)], "#4f46e5")
    s.arrow([(rx, 268), (692, 352)], "#0d9488")
    s.arrow([(rx, 374), (692, 374)], "#64748b", dashed=True)

    s.node(rx, 470, rw, 104, "Release flush at kernel end", "dirty L2 lines left after the last wave", "amber",
           "are written back: a serial term in the cost")
    s.arrow([(690, 412), (718, 412), (718, 522), (748, 522)], "#d97706", dashed=True, width=2.2)
    s.render("memory-hierarchy.svg")


XCD_COLORS = ["#e11d48", "#ea580c", "#d97706", "#16a34a", "#0d9488", "#2563eb", "#7c3aed", "#db2777"]


def fig_placement():
    s = Svg(1100, 360, "Workgroup placement",
            "The timing plane places workgroups the way the SPI does: XCD = wg % 8, then round-robin over its CUs.")
    s.chip(760, 26, "XCD = wg % 8", "red")
    s.chip(890, 26, "CU = (wg / 8) % 32", "blue")
    size, gap, x0, y0 = 34, 8.6, 50, 104
    centers = []
    for wg in range(24):
        bx = x0 + wg * (size + gap)
        c = XCD_COLORS[wg % 8]
        s.add(f'<rect x="{bx:.1f}" y="{y0}" width="{size}" height="{size}" rx="8" fill="{c}"/>')
        s.text(bx + size / 2, y0 + 22, str(wg), 12, "#ffffff", 700, "middle")
        centers.append((bx + size / 2, y0 + size))
    s.text(x0, y0 - 10, "workgroup id", 11.5, MUTED, 600)
    cw, cg, cy = 116, 12.6, 236
    for i in range(8):
        bx = x0 + i * (cw + cg)
        c = XCD_COLORS[i]
        for wg in range(i, 24, 8):
            sx, sy = centers[wg]
            s.add(f'<path d="M{sx:.1f},{sy + 2} C{sx:.1f},{sy + 50} {bx + cw / 2:.1f},{cy - 50} {bx + cw / 2:.1f},{cy - 2}" '
                  f'fill="none" stroke="{c}" stroke-opacity="0.45" stroke-width="1.6"/>')
        s.add(f'<rect x="{bx:.1f}" y="{cy}" width="{cw}" height="78" rx="12" fill="{c}" fill-opacity="0.09" '
              f'stroke="{c}" stroke-width="1.4"/>')
        s.text(bx + cw / 2, cy + 26, f"XCD {i}", 14, c, 800, "middle")
        s.text(bx + cw / 2, cy + 46, f"wg {i}, {i + 8}, {i + 16}, …", 11, MUTED, 400, "middle")
        s.text(bx + cw / 2, cy + 63, "L2 · 4 MB", 10.5, MUTED, 600, "middle")
    s.text(50, 346, "hipBLASLt XCC remapping and Triton GROUP_M assume this placement: tiles that share data then share an L2.",
           12, "#334155", 500)
    s.render("placement.svg")


def fig_calibration():
    s = Svg(1300, 580, "Two-stage calibration and the cost of one dispatch",
            "Stage 1 fixes absolute time; stage 2 fixes which variant wins; the composition below is what both stages tune.")
    s.cylinder(40, 100, 250, 92, "Meter corpus", "260 cases · hardware times", "slate")
    s.cylinder(40, 218, 250, 92, "Copy microbenchmark", "3 store policies × 13 sizes", "amber")
    s.cylinder(40, 336, 250, 92, "Ranking development set", "1334 variants · 6802 pairs", "violet")
    s.node(360, 100, 340, 92, "Stage 1 · meter_fit", "absolute time, differential evolution", "blue", "5 seeds → 5 knob sets")
    s.node(360, 218, 340, 92, "flush_scale", "one scale for the release flush,", "amber", "fitted on the microbenchmark only")
    s.node(360, 336, 340, 92, "Stage 2 · rank_fit", "refits the 9 knobs that decide which variant wins", "violet",
           "pairwise logistic loss + pull toward stage 1")
    s.arrow([(290, 146), (358, 146)], "#64748b")
    s.arrow([(290, 264), (358, 264)], "#d97706")
    s.arrow([(290, 382), (358, 382)], "#7c3aed")
    s.arrow([(530, 192), (530, 216)], "#2563eb")
    s.arrow([(530, 310), (530, 334)], "#d97706")
    for i in range(5):
        ox, oy = 790 + i * 9, 190 + i * 13
        s.add(f'<g filter="url(#shadow)"><rect x="{ox}" y="{oy}" width="170" height="120" rx="12" fill="#ffffff" '
              f'stroke="#059669" stroke-opacity="0.6"/></g>')
    s.text(911, 280, "5 knob sets", 15, "#064e3b", 800, "middle")
    s.text(911, 300, "one per seed", 11.5, MUTED, 400, "middle")
    s.arrow([(700, 382), (744, 382), (744, 270), (788, 270)], "#059669")
    s.node(1040, 222, 220, 96, "Prediction", "geometric mean over the", "green", "5 knob sets, warm calls")
    s.arrow([(1006, 270), (1038, 270)], "#059669")

    y = 470
    s.add(f'<rect x="40" y="{y - 22}" width="1220" height="92" rx="16" fill="#f8fafc" stroke="#e2e8f0"/>')
    s.text(60, y, "cycles per dispatch", 11.5, MUTED, 700)
    x = 60
    parts = [("text", "cycles  ="), ("chip", "fixed", "slate"), ("text", "+  max("), ("chip", "issue", "blue"),
             ("text", ","), ("chip", "bandwidth", "indigo"), ("text", ","), ("chip", "placement", "teal"),
             ("text", ")  +"), ("chip", "latency", "violet"), ("text", "+"), ("chip", "filling", "green"),
             ("text", "+"), ("chip", "flush", "amber")]
    for p in parts:
        if p[0] == "text":
            s.text(x, y + 34, p[1], 19, "#0f172a", 700, family=MONO)
            x += tw(p[1], 19) * 1.08 + 12
        else:
            x += s.chip(x, y + 15, p[1], p[2], size=15) + 12
    s.render("calibration.svg")


def fig_validation():
    s = Svg(1300, 340, "How every claim is validated",
            "Each model or calibration change is pre-registered, tested on a held-out set nobody has seen, and scored once.")
    steps = [("1", "Develop", "fit and diagnose on", "seen data only", "slate"),
             ("2", "Pre-register", "model, knobs, rules;", "sha256, committed", "violet"),
             ("3", "Measure", "a new held-out set,", "gpu_guard, 3 runs", "blue"),
             ("4", "Score once", "prereg_score.py;", "pass or fail, reported", "green"),
             ("5", "Retire", "the set joins the", "development data", "amber")]
    w, gap, y = 220, 30, 110
    for i, (n, title, a, b, pal) in enumerate(steps):
        x = 40 + i * (w + gap)
        s.node(x, y, w, 110, title, a, pal, b)
        main = PAL[pal][0]
        s.add(f'<circle cx="{x + w - 26}" cy="{y + 26}" r="14" fill="{main}"/>')
        s.text(x + w - 26, y + 31, n, 13, "#ffffff", 800, "middle")
        if i < len(steps) - 1:
            s.arrow([(x + w + 2, y + 55), (x + w + gap - 2, y + 55)], main)
    s.add('<path d="M1190,222 C1190,300 150,300 150,224" fill="none" stroke="#94a3b8" stroke-width="1.8" '
          f'stroke-dasharray="6 5" marker-end="{s.marker(LINE)}"/>')
    s.text(670, 292, "next change starts from the larger development set", 12, "#475569", 600, "middle")
    s.render("validation.svg")


def fig_results():
    s = Svg(1060, 640, "Ranking accuracy on pre-registered held-out sets",
            "Share of variant pairs (hardware gap > 10%) the model orders correctly. Each set was scored once.")
    x0, x1, lo, hi = 330, 960, 50.0, 100.0

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    rows = [("GEMM (hipBLASLt)", None, None, None),
            ("held-out 4 · 8 shapes", 89.0, "blue", None),
            ("held-out 5 · 8 shapes", 87.4, "blue", None),
            ("held-out 6 · 12 shapes", 89.2, "blue", None),
            ("held-out 8 · 12 shapes", 94.2, "blue", None),
            ("best rule without a model", 68.3, "slate", 60.2),
            ("Triton", None, None, None),
            ("held-out 4 · 6 cases", 78.7, "violet", None),
            ("held-out 5 · 8 cases", 79.4, "violet", None),
            ("  store-policy softmaxes", 90.0, "violet", None),
            ("held-out 8 · 22 groups", 94.0, "violet", None),
            ("best rule without a model", 71.6, "slate", 57.7)]
    y = 112
    for v in range(50, 101, 10):
        s.add(f'<line x1="{X(v):.1f}" y1="100" x2="{X(v):.1f}" y2="{100 + 41 * len(rows)}" stroke="#e2e8f0"/>')
        s.text(X(v), 96, f"{v}%", 11, MUTED, 600, "middle")
    for label, v, pal, vmin in rows:
        if v is None:
            s.text(40, y + 24, label, 14, "#0f172a", 800)
            y += 41
            continue
        s.text(40, y + 21, label.strip(), 13, "#334155", 500, italic=label.startswith("  "))
        if vmin is not None:
            s.add(f'<rect x="{X(lo):.1f}" y="{y + 6}" width="{X(vmin) - X(lo):.1f}" height="22" rx="6" fill="#cbd5e1"/>')
            s.add(f'<rect x="{X(vmin):.1f}" y="{y + 6}" width="{X(v) - X(vmin):.1f}" height="22" rx="6" fill="#e2e8f0"/>')
            s.text(X(v) + 8, y + 22, f"{vmin:.0f}–{v:.0f}%", 12.5, MUTED, 700)
        else:
            s.add(f'<rect x="{X(lo):.1f}" y="{y + 6}" width="{X(v) - X(lo):.1f}" height="22" rx="6" fill="{s.grad(pal)}"/>')
            s.text(X(v) + 8, y + 22, f"{v:.1f}%", 12.5, PAL[pal][2], 800)
        y += 41
    s.add(f'<line x1="{X(98.5):.1f}" y1="100" x2="{X(98.5):.1f}" y2="{y}" stroke="#059669" stroke-width="2" '
          f'stroke-dasharray="5 4"/>')
    s.text(X(98.5), y + 16, "hardware ceiling", 11.5, "#059669", 700, "middle")
    s.text(40, y + 16, "axis starts at 50% (a coin flip)", 11, MUTED, 400, italic=True)
    s.render("results.svg")


# -- Examples ----------------------------------------------------------------
# Measured on MI355X (median of 3 guarded runs, us) and predicted by a model
# that had not seen the group: held-out set 6 with the second stage fitted on
# sets 1-5, and held-out set 5 with it fitted on sets 1-4. (name, tile, emulator
# estimate relative to its own first choice, hardware us)
GEMM_WIN = [("437661", "64x80x128", 1.000, 8.50), ("437659", "64x80x128", 1.034, 8.76),
            ("Default", "64x64x256", 1.077, 11.09), ("438214", "64x64x64", 1.081, 8.51),
            ("438694", "64x64x64", 1.087, 8.77), ("438645", "64x128x128", 1.241, 9.25),
            ("437752", "64x80x64", 1.277, 11.33), ("437602", "64x48x128", 1.317, 9.80),
            ("437614", "64x48x128", 1.363, 10.15), ("438662", "32x64x256", 1.432, 12.53),
            ("438360", "96x64x64", 1.479, 12.63), ("437626", "128x80x128", 1.482, 12.14),
            ("438545", "256x16x32", 1.488, 14.83), ("438162", "64x224x128", 1.749, 13.46),
            ("438068", "176x256x32", 4.381, 33.12)]
GEMM_MISS = [("438412", "32x128x128", 1.000, 14.89), ("Default", "64x16x512", 1.017, 12.39),
             ("437840", "32x32x256", 1.065, 10.64), ("437577", "32x32x256", 1.075, 14.09),
             ("438423", "32x32x256", 1.099, 13.07), ("438617", "32x32x64", 1.155, 12.67),
             ("438366", "64x128x64", 1.285, 16.15), ("438422", "32x64x256", 1.304, 14.25),
             ("438397", "128x32x128", 1.327, 13.99), ("437752", "64x80x64", 1.412, 16.69),
             ("437554", "128x128x32 +reduce", 1.416, 15.51), ("437575", "256x32x32 +reduce", 1.577, 19.11),
             ("438675", "32x16x512", 1.589, 13.84), ("437728", "32x16x64", 2.131, 24.23),
             ("437892", "256x224x64", 4.854, 56.88)]
SOFTMAX = [("w4_r1_c2", "4 warps · 1 row · .wt", 1.000, 8.96), ("w8_r1_c2", "8 warps · 1 row · .wt", 1.111, 9.69),
           ("w2_r1_c2", "2 warps · 1 row · .wt", 1.166, 8.46), ("w4_r2_c1", "4 warps · 2 rows · .cs", 1.255, 10.28),
           ("w8_r4_c2", "8 warps · 4 rows · .wt", 1.538, 10.27), ("w4_r1_c0", "4 warps · 1 row · plain", 1.680, 10.48),
           ("w2_r1_c0", "2 warps · 1 row · plain", 1.881, 10.27), ("w1_r2_c2", "1 warp · 2 rows · .wt", 2.122, 10.27),
           ("w8_r4_c0", "8 warps · 4 rows · plain", 2.269, 11.07), ("w4_r8_c1", "4 warps · 8 rows · .cs", 2.425, 12.19),
           ("w1_r4_c0", "1 warp · 4 rows · plain", 4.029, 15.33), ("w1_r8_c0", "1 warp · 8 rows · plain", 6.205, 26.46)]
AGENT_A7 = [0.833, 0.887, 0.908, 0.524, 0.524, 0.577, 0.489, 0.833, 0.489, 0.529, 0.482, 0.480, 0.482, 0.480]


def fig_ranks(name, title, subtitle, rows, footer, footer_pal):
    n = len(rows)
    s = Svg(1120, 150 + n * 34 + 70, title, subtitle)
    hw_order = sorted(range(n), key=lambda i: rows[i][3])
    hw_rank = {i: r for r, i in enumerate(hw_order)}
    best = min(r[3] for r in rows)
    lx, rx, y0, dy = 50, 690, 128, 34
    s.text(lx, y0 - 16, "Emulator ranking  (no GPU)", 14, "#1e3a8a", 800)
    s.text(rx, y0 - 16, "Hardware ranking  (MI355X, measured)", 14, "#064e3b", 800)
    for i, (nm, tile, pred, hw) in enumerate(rows):
        y = y0 + i * dy
        d = abs(hw_rank[i] - i)
        c = "#059669" if d <= 1 else ("#d97706" if d <= 3 else "#e11d48")
        ty = y0 + hw_rank[i] * dy + 12
        s.add(f"<path d=\"M{lx + 380},{y + 12} C{lx + 470},{y + 12} {rx - 90},{ty} {rx - 8},{ty}\" "
              f"fill=\"none\" stroke=\"{c}\" stroke-width=\"2\" stroke-opacity=\"0.75\"/>")
    for i, (nm, tile, pred, hw) in enumerate(rows):
        y = y0 + i * dy
        hl = "#ecfdf5" if i == 0 else ("#fff7ed" if nm == "Default" else "#ffffff")
        s.add(f"<rect x=\"{lx - 8}\" y=\"{y - 4}\" width=\"396\" height=\"30\" rx=\"8\" fill=\"{hl}\" stroke=\"#e2e8f0\"/>")
        s.text(lx + 6, y + 16, f"{i + 1}", 12, MUTED, 700)
        s.text(lx + 34, y + 16, nm, 13, "#0f172a", 700, family=MONO)
        s.text(lx + 140, y + 16, tile, 11.5, MUTED)
        s.text(lx + 372, y + 16, f"{pred:.2f}×", 12.5, "#1e3a8a", 700, "end")
    for r, i in enumerate(hw_order):
        nm, tile, pred, hw = rows[i]
        y = y0 + r * dy
        hl = "#ecfdf5" if r == 0 else ("#fff7ed" if nm == "Default" else "#ffffff")
        s.add(f"<rect x=\"{rx - 8}\" y=\"{y - 4}\" width=\"388\" height=\"30\" rx=\"8\" fill=\"{hl}\" stroke=\"#e2e8f0\"/>")
        s.text(rx + 6, y + 16, f"{r + 1}", 12, MUTED, 700)
        s.text(rx + 34, y + 16, nm, 13, "#0f172a", 700, family=MONO)
        s.text(rx + 250, y + 16, f"{hw:.2f} us", 12.5, "#064e3b", 700, "end")
        s.text(rx + 366, y + 16, "best" if hw == best else f"+{hw / best - 1:.0%}", 12, MUTED, 700, "end")
    yl = y0 + n * dy + 14
    lx2 = lx
    for label, c in (("same place ±1", "#059669"), ("±2–3", "#d97706"), ("off by more", "#e11d48")):
        s.add(f"<line x1=\"{lx2}\" y1=\"{yl}\" x2=\"{lx2 + 26}\" y2=\"{yl}\" stroke=\"{c}\" stroke-width=\"3\"/>")
        s.text(lx2 + 32, yl + 4, label, 11.5, MUTED, 600)
        lx2 += 150
    main, tint, deep = PAL[footer_pal]
    s.add(f"<rect x=\"{lx - 8}\" y=\"{yl + 18}\" width=\"1036\" height=\"38\" rx=\"10\" fill=\"{tint}\" "
          f"stroke=\"{main}\" stroke-opacity=\"0.5\"/>")
    s.text(lx + 8, yl + 42, footer, 13, deep, 700)
    s.render(name)


def fig_agent():
    s = Svg(1120, 460, "An agent optimizing softmax with emulator feedback only",
            "Run A7: estimated cost of each evaluated candidate relative to the baseline (lower is better). No GPU was used.")
    x0, x1, y0, y1, lo, hi = 90, 760, 100, 380, 0.4, 1.0

    def P(i, v):
        return x0 + (i - 1) / 13 * (x1 - x0), y1 - (v - lo) / (hi - lo) * (y1 - y0)

    for v in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        _, y = P(1, v)
        s.add(f"<line x1=\"{x0}\" y1=\"{y:.1f}\" x2=\"{x1}\" y2=\"{y:.1f}\" stroke=\"#e2e8f0\"/>")
        s.text(x0 - 10, y + 4, f"{v:.1f}", 11, MUTED, 600, "end")
    for i in range(1, 15):
        x, _ = P(i, lo)
        s.text(x, y1 + 20, str(i), 11, MUTED, 600, "middle")
    s.text((x0 + x1) / 2, y1 + 40, "evaluation", 11.5, MUTED, 600, "middle")
    best, pts = 1.0, []
    for i, v in enumerate(AGENT_A7, 1):
        best = min(best, v)
        pts.append(P(i, best))
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    s.add(f"<path d=\"{d}\" fill=\"none\" stroke=\"#059669\" stroke-width=\"3\" stroke-linejoin=\"round\"/>")
    for i, v in enumerate(AGENT_A7, 1):
        x, y = P(i, v)
        s.add(f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"6\" fill=\"#ffffff\" stroke=\"#2563eb\" stroke-width=\"2.2\"/>")
    notes = [(1, 0.833, "unmasked 3×512 chunks, one wave per row", -14, "start"),
             (4, 0.524, ".cs streaming stores", -16, "start"),
             (7, 0.489, ".wt write-through stores", 26, "start"),
             (12, 0.480, "reductions element-wise first", -16, "end")]
    for i, v, label, dy, anchor in notes:
        x, y = P(i, v)
        s.text(x + (8 if anchor == "start" else -8), y + dy, label, 11.5, "#1e3a8a", 700, anchor)
    s.add("<rect x=\"800\" y=\"110\" width=\"290\" height=\"250\" rx=\"16\" fill=\"#f0fdf4\" stroke=\"#059669\" stroke-opacity=\"0.5\"/>")
    s.text(822, 142, "Then measured on hardware", 13.5, "#064e3b", 800)
    s.text(822, 186, "15.64 us", 26, "#64748b", 800)
    s.text(950, 186, "baseline", 12, MUTED, 600)
    s.text(822, 236, "5.83 us", 26, "#059669", 800)
    s.text(950, 236, "A7 final", 12, MUTED, 600)
    s.text(822, 282, "2.68× faster", 20, "#064e3b", 800)
    s.text(822, 310, "hardware-feedback agents:", 12, MUTED, 600)
    s.text(822, 328, "2.69× on average", 12, MUTED, 600)
    s.render("example-agent.svg")


def fig_examples():
    fig_ranks("example-gemm-win.svg", "Example · hipBLASLt GEMM 768×1280×1024 (bf16)",
              "Held-out set 6, a shape the model had never seen. Lines join each solution across the two rankings.",
              GEMM_WIN, "Emulator pick = hardware best (8.50 us). hipBLASLt Default: 11.09 us, 30% slower.", "green")
    fig_ranks("example-gemm-miss.svg", "Example · hipBLASLt GEMM 72×2560×3072 (bf16): a miss",
              "Held-out set 6. The top five are predicted within 10% of each other; hardware spreads them 40%.",
              GEMM_MISS, "Emulator pick is 40% off the best (its 3rd choice). Default: +16%. The 5× outlier is still flagged.", "red")
    fig_ranks("example-softmax.svg", "Example · Triton softmax 2048×3584 with store-policy variants",
              "Held-out set 5, unseen by the model. Variants: num_warps × rows per program × output store policy.",
              SOFTMAX, "Top 3 predicted = the 3 fastest on hardware (all .wt, one row per program). Plain-store variants rank lower.", "green")
    fig_agent()


HATCH = ('<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<rect width="8" height="8" fill="#fffbeb"/><line x1="0" y1="0" x2="0" y2="8" stroke="#f59e0b" '
         'stroke-width="3" stroke-opacity="0.45"/></pattern></defs>')


def block(s, x0, x1, y, h, label, pal, size=12, hatch=False):
    main, tint, deep = PAL[pal]
    fill = "url(#hatch)" if hatch else tint
    s.add(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" rx="7" fill="{fill}" stroke="{main}" stroke-width="1.3"/>')
    if label:
        if tw(label, size, True) > x1 - x0 - 8:
            print(f"  overflow: {label!r} needs {tw(label, size, True):.0f} of {x1 - x0 - 8}")
        s.text((x0 + x1) / 2, y + h / 2 + size * 0.36, label, size, deep, 700, "middle")


def fig_wavefront():
    s = Svg(1200, 480, "Life of a wavefront",
            "Each wavefront has its own clock. A memory request runs in the background; s_waitcnt is where its latency shows.")
    s.add(HATCH)
    rows = {"A": 112, "M": 200, "B": 288}
    s.text(34, rows["A"] + 18, "wavefront A", 13.5, "#0f172a", 700)
    s.text(34, rows["A"] + 35, "one CU", 11.5, MUTED)
    s.text(34, rows["M"] + 18, "A's load request", 13.5, "#0f172a", 700)
    s.text(34, rows["M"] + 35, "through the hierarchy", 11.5, MUTED)
    s.text(34, rows["B"] + 18, "wavefront B", 13.5, "#0f172a", 700)
    s.text(34, rows["B"] + 35, "same CU, resident", 11.5, MUTED)
    for x0, x1, label, pal in ((190, 270, "v_mul", "blue"), (270, 370, "global_load", "teal"),
                               (370, 540, "v_mfma × 2", "violet"), (860, 960, "v_add", "blue"),
                               (960, 1080, "global_store", "teal")):
        block(s, x0, x1, rows["A"], 44, label, pal)
    block(s, 540, 860, rows["A"], 44, "s_waitcnt vmcnt(0) · stall", "amber", hatch=True)
    for x0, x1, label in ((370, 440, "L1 miss"), (440, 530, "L2 miss"), (530, 590, "fabric"),
                          (590, 680, "MALL miss"), (680, 860, "HBM")):
        block(s, x0, x1, rows["M"] + 6, 32, label, "slate", size=11)
    for x0, x1, label, pal in ((540, 640, "v_mfma", "violet"), (640, 730, "ds_read", "indigo"),
                               (730, 850, "v_mfma", "violet")):
        block(s, x0, x1, rows["B"], 44, label, pal)
    s.arrow([(320, rows["A"] + 46), (320, rows["M"] + 20), (366, rows["M"] + 20)], "#0d9488", width=1.6)
    s.arrow([(860, rows["M"] + 4), (860, rows["A"] + 48)], "#d97706", width=1.8)
    s.text(870, rows["M"] - 8, "data arrives", 11.5, "#b45309", 700)
    s.text(700, rows["M"] + 66, "each level: hit latency + queueing; misses go on", 11, MUTED, 500, "middle")
    s.text(695, rows["B"] + 64, "B issues while A waits: part of A's stall is hidden", 11.5, "#4c1d95", 600, "middle")
    by = 382
    s.add(f'<path d="M190,{by} v-8 M190,{by} H1080 M1080,{by} v-8" fill="none" stroke="#0f172a" stroke-width="1.4"/>')
    s.text(635, by + 20, "critical path of A = its issue time + its stalls", 12.5, "#0f172a", 700, "middle")
    s.text(34, 436, "issue term: work queued on each port of the busiest CU, over all its wavefronts.", 12.5, "#334155", 500)
    s.text(34, 458, "latency term: critical path − what the other resident wavefronts hide, × rounds of wavefronts.",
           12.5, "#334155", 500)
    s.render("wavefront-timeline.svg")


# (label, sub, fixed, throughput, which, latency, filling, flush, hardware)  fit s1, held-out set 5
COMPOSE = [("2 warps · 1 row/program · .wt", "w2_r1_c2", 1.19, 3.93, "bandwidth", 3.17, 0.65, 0.00, 8.46),
           ("2 warps · 1 row/program · plain", "w2_r1_c0", 1.19, 1.98, "bandwidth", 3.17, 0.65, 7.79, 10.27),
           ("1 warp · 8 rows/program · plain", "w1_r8_c0", 1.19, 1.98, "bandwidth", 35.96, 0.65, 7.77, 26.46)]


def fig_composition():
    s = Svg(1200, 470, "Composing one prediction: three softmax variants",
            "Held-out set 5, 2048×3584. Terms in µs under one knob set (fit s1); hardware is the median of 3 guarded runs.")
    terms = [("fixed", "slate"), ("max(issue, bandwidth, placement)", "blue"), ("latency", "violet"),
             ("filling", "teal"), ("flush", "amber")]
    x = 34
    for name, pal in terms:
        x += s.chip(x, 84, name, pal, 11) + 10
    s.chip(x + 10, 84, "hardware", "ink", 11)
    x0, k = 330, 16.0
    for i, (label, sub, fixed, thr, which, lat, fill, flush, hw) in enumerate(COMPOSE):
        y = 136 + i * 92
        s.text(34, y + 14, label, 13.5, "#0f172a", 700)
        s.text(34, y + 32, f"{sub} · rank: model {i + 1}, hardware {i + 1}", 11.5, MUTED)
        cx = x0
        for (name, pal), v in zip(terms, (fixed, thr, lat, fill, flush)):
            if v <= 0:
                continue
            w = v * k
            main, tint, deep = PAL[pal]
            s.add(f'<rect x="{cx:.1f}" y="{y}" width="{w:.1f}" height="30" fill="{main}" fill-opacity="0.85" '
                  f'stroke="#ffffff" stroke-width="1"/>')
            txt = f"{v:.2f}"
            if tw(txt, 11, True) < w - 6:
                s.text(cx + w / 2, y + 19.5, txt, 11, "#ffffff", 700, "middle")
            cx += w
        total = fixed + thr + lat + fill + flush
        s.text(cx + 8, y + 20, f"{total:.2f} µs", 13, "#0f172a", 800)
        s.add(f'<rect x="{x0}" y="{y + 38}" width="{hw * k:.1f}" height="12" rx="3" fill="#0f172a" fill-opacity="0.75"/>')
        s.text(x0 + hw * k + 8, y + 48, f"{hw:.2f} µs", 12, "#0f172a", 600)
    s.text(34, 428, "Same code except the store: .wt moves its output during the kernel (bandwidth up) and leaves nothing to flush.",
           12.5, "#334155", 500)
    s.text(34, 450, "8 rows per program: one wavefront per CU, nothing hides its stalls, so latency grows 11×.",
           12.5, "#334155", 500)
    s.render("composition-example.svg")


FIGURES = [fig_architecture, fig_memory, fig_placement, fig_wavefront, fig_composition, fig_calibration, fig_validation, fig_results, fig_examples]

if __name__ == "__main__":
    for f in FIGURES:
        f()
