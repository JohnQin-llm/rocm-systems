#!/usr/bin/env python3
"""Pick up to 16 solutions per shape from TunableOp's tuning log.

The rule (unchanged from select_and_time_hw.py, so the original six shapes keep
their picks): TunableOp's best, Default, 10 evenly spaced over the fully timed
instances, 4 seeded draws from the instances skipped as slow.
"""

import json
import os
import random
import re
from pathlib import Path

WORK = Path(os.environ.get("RANKING_WORK", os.getcwd()))         # measurements live here
meta = json.loads((WORK / "candidates.json").read_text())
selected = {}
for shape, m in meta.items():
    log = (WORK / "tunable" / f"tune_{shape}.log").read_text(errors="replace")
    timed = {name: float(ms) for ms, name in
             re.findall(r"found (?:better|slower) instance id=\d+\. ([0-9.e-]+)ms\. (\S+)", log)}
    skipped = sorted(set(re.findall(r"skip slow instance[^\n]*?(Gemm_Hipblaslt_\d+)", log)) - set(timed))
    order = sorted((n for n in timed if n != "Default"), key=timed.get)
    picks = [m["tuned_best"], "Default"]
    for i in range(10):
        cand = order[round(i * (len(order) - 1) / 9)]
        if cand not in picks:
            picks.append(cand)
    rng = random.Random(f"{shape}-skipped")
    for cand in rng.sample(skipped, min(4, len(skipped))):
        if cand not in picks:
            picks.append(cand)
    selected[shape] = picks
    print(f"{shape}: {len(timed)} timed, {len(skipped)} skipped, picked {len(picks)}")
(WORK / "selected.json").write_text(json.dumps(selected, indent=1))
