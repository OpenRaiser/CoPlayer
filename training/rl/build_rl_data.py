#!/usr/bin/env python3
"""Render the images of the final RL rows and write verl parquet files.

Input ($COPLAY_DATA/training/rl): {train,dev}_{reactive,proactive}.jsonl, one row per training sample in training order
(reactive rows repeat questions: 12,734 rows over 3,161 questions). Each row holds the prompt with <image> placeholders,
the reward metadata (ground_truth) and extra_info.label (reactive | must_speak | neg_after | neg_before | neg_far).
Images are rendered from the Train videos with the benchmark's own frame code:
  reactive  : 4 grids of 16 frames over [t-60, t]   -> <out>/images/reactive/<id>/grid{0..3}.jpg
  proactive : one 2x2 grid of step k (1 fps cache)  -> <out>/images/proactive/<vid>/k%05d.jpg
and embedded as bytes; the reward reads the same files again for the judge (COPLAY_RL_DATA=<out>).
Output: <out>/{train,dev}_{reactive,proactive}.parquet
usage: build_rl_data.py [--out $COPLAY_DATA/training/rl] [--jobs 24]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "coplaybench"))
from common import DATA  # noqa: E402
from frames import build_reactive, extract, grid_bytes  # noqa: E402

SPLITS = [(s, t) for s in ("train", "dev") for t in ("reactive", "proactive")]


def read_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def render_reactive(meta, out):
    it = {"id": meta["id"], "video": meta["video"],
          "context": {"grids": [os.path.join(out, p) for p in meta["context"]["grids"]], "frame_times": meta["context"]["frame_times"]}}
    build_reactive(it)


def render_proactive(vid, video, step, ks, out, tmp):
    """all steps of one video from one 1 fps extraction (deleted afterwards)"""
    targets = {k: os.path.join(out, "images", "proactive", vid, "k%05d.jpg" % k) for k in ks}
    if all(os.path.isfile(p) for p in targets.values()):
        return
    s = {"id": vid, "step_sec": step, "frames_dir": os.path.join(tmp, vid)}
    extract(s, os.path.join(DATA, video))
    try:
        os.makedirs(os.path.join(out, "images", "proactive", vid), exist_ok=True)
        for k, p in targets.items():
            if not os.path.isfile(p):
                img, _ = grid_bytes(s, k)
                with open(p + ".tmp", "wb") as fo:
                    fo.write(img)
                os.replace(p + ".tmp", p)
    finally:
        shutil.rmtree(s["frames_dir"], ignore_errors=True)


def image_paths(row):
    gt = row["ground_truth"]
    return gt["context"]["grids"] if row["extra_info"]["label"] == "reactive" else [gt["grid_path"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(DATA, "training", "rl"))
    ap.add_argument("--out", default=os.path.join(DATA, "training", "rl"))
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    tmp = tempfile.mkdtemp(prefix="coplay_rl_1fps_")
    rows = {st: read_rows(os.path.join(a.src, "%s_%s.jsonl" % st)) for st in SPLITS}
    reactive, proactive = {}, {}
    for (_, track), rs in rows.items():
        for r in rs:
            gt = r["ground_truth"]
            if track == "reactive":
                reactive[gt["id"]] = gt
            else:
                v = proactive.setdefault(gt["vid"], {"video": gt["video"], "step": None, "ks": set()})
                v["ks"].add(gt["k"])
                v["step"] = gt["t"] / gt["k"]
    print("rendering %d reactive questions and %d proactive steps over %d videos" % (
        len(reactive), sum(len(v["ks"]) for v in proactive.values()), len(proactive)), flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(render_proactive, vid, v["video"], v["step"], sorted(v["ks"]), out, tmp) for vid, v in proactive.items()]
        futs += [ex.submit(render_reactive, gt, out) for gt in reactive.values()]
        for k, f in enumerate(futs, 1):
            f.result()
            if k % 500 == 0:
                print("rendered %d/%d" % (k, len(futs)), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    for (split, track), rs in rows.items():
        path = os.path.join(out, "%s_%s.parquet" % (split, track))
        writer = None
        for i in range(0, len(rs), 512):                 # batches: the reactive train file holds ~6.5 GB of images
            data = []
            for r in rs[i:i + 512]:
                images = [{"bytes": open(os.path.join(out, p), "rb").read()} for p in image_paths(r)]
                if sum(m["content"].count("<image>") for m in r["prompt"]) != len(images):
                    raise ValueError("image placeholder mismatch in " + r["extra_info"]["id"])
                data.append({"data_source": r["data_source"], "prompt": r["prompt"], "images": images,
                             "reward_model": {"style": "rule", "ground_truth": json.dumps(r["ground_truth"], ensure_ascii=False, sort_keys=True)},
                             "extra_info": r["extra_info"]})
            table = pa.Table.from_pylist(data) if writer is None else pa.Table.from_pylist(data, schema=writer.schema)
            if writer is None:
                writer = pq.ParquetWriter(path + ".tmp", table.schema, compression="zstd")
            writer.write_table(table, row_group_size=128)
        writer.close()
        os.replace(path + ".tmp", path)
        print("wrote", path, len(rs), "rows", flush=True)


if __name__ == "__main__":
    main()
