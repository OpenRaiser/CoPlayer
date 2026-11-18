#!/usr/bin/env python3
"""Take a small balanced smoke set (every game x every kind) from sft_train.json -> sft_smoke.json (256 samples).
usage: make_smoke.py [--dir $COPLAY_DATA/training]   (then add an "sft_smoke" entry to dataset_info.json)"""
import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.environ.get("COPLAY_DATA", "data/CoPlayBench"), "training"))
a = ap.parse_args()
data = json.load(open(os.path.join(a.dir, "sft_train.json")))
rng = random.Random(0)
groups = {}
for s in data:
    groups.setdefault((s["game"], s["kind"]), []).append(s)
per = max(1, 256 // len(groups))
pick = []
for k in sorted(groups):
    pick += rng.sample(groups[k], min(per, len(groups[k])))
rng.shuffle(pick)
json.dump(pick[:256], open(os.path.join(a.dir, "sft_smoke.json"), "w"), ensure_ascii=False)
print("smoke", len(pick[:256]), "from", len(groups), "game x kind groups")
