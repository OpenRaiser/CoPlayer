#!/usr/bin/env python3
"""Reactive track, step 1: one streamed reply per (model, question).

For every item: 4 grids (16 frames over the last 60 s, all <= t) + the player's line -> one streamed reply.
TTFT = first visible token; chunk timestamps are kept for TTUA. Empty or truncated replies are kept and scored as failures.
Resumable: existing $COPLAY_OUT/reactive/runs/<model>/<id>.json are skipped.
usage: run_reactive.py --models a,b [--per-model-jobs 24] [--limit N] [--ids a,b]
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LIMITER, OUT, PAPER_MODELS, FatalError, load_config, load_reactive, stream_chat  # noqa: E402
from frames import build_reactive  # noqa: E402

SYSTEM = """You are the player's gaming buddy, watching their game screen together with them in real time.
Persona: a gaming buddy who is relaxed and cheerful, loves to hype the player up and joke around; never preachy, never sarcastic, no empty "you got this".
Real-time rules:
1. The game moves fast. Keep it short and lead with the conclusion: one or two sentences (usually no more than 25 words).
2. Only talk about what has already happened and can be seen on screen. Do not make up cards, pieces, heroes, numbers or events.
3. When the player needs guidance, give concrete, actionable advice (what to play, where to move, where to stand, what to do first). When the player needs emotional support, acknowledge their feelings first.
4. Output only what you would say to the player. Do not explain your reasoning, no quotes, no prefixes."""

Q_TEXT = {
    "in_play": 'At this moment (video second {t:.1f}) the player says to you: "{msg}"\nReply to the player now.',
    "aftermath": 'That play/fight has just ended. At this moment (video second {t:.1f}) the player says to you: "{msg}"\nReply to the player now.',
}


def build_messages(it):
    parts = [{"type": "text", "text": "Game: %s\nBelow are the last 60 seconds of the player's screen, as 4 images; each image is a 2x2 grid "
                                      "of 4 frames (top-left -> top-right -> bottom-left -> bottom-right). The last frame of the last "
                                      "image is the current moment." % it["game_name"]}]
    ts = it["context"]["frame_times"]
    for g, path in enumerate(it["context"]["grids"]):
        parts.append({"type": "text", "text": "[frame times " + ", ".join("%.1fs" % x for x in ts[4 * g:4 * g + 4]) + "]"})
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()}})
    parts.append({"type": "text", "text": Q_TEXT[it["question_type"]].format(t=it["t_sec"], msg=it["player_message"])})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]


def run_one(base, key, model, it, stop):
    p = os.path.join(OUT, "reactive", "runs", model, it["id"] + ".json")
    if os.path.isfile(p) or stop.is_set():
        return None
    build_reactive(it)
    rec = stream_chat(base, key, model, build_messages(it))
    rec.update({"id": it["id"], "model": model, "game": it["game"], "question_type": it["question_type"], "ts": time.strftime("%F %T")})
    if rec.get("error") is None:
        json.dump(rec, open(p + ".tmp", "w"), ensure_ascii=False)
        os.replace(p + ".tmp", p)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--bench", default=None, help="default $COPLAY_DATA/bench/reactive.jsonl")
    ap.add_argument("--per-model-jobs", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test)")
    ap.add_argument("--ids", help="comma-separated item ids")
    a = ap.parse_args()
    base, key = load_config()
    items = load_reactive(a.bench)
    if a.ids:
        want = set(a.ids.split(","))
        items = [it for it in items if it["id"] in want]
    if a.limit:
        items = items[:a.limit]
    models = a.models.split(",")
    stop = threading.Event()
    jobs = []
    for m in models:
        d = os.path.join(OUT, "reactive", "runs", m)
        os.makedirs(d, exist_ok=True)
        have = set(f[:-5] for f in os.listdir(d) if f.endswith(".json"))
        jobs += [(m, it) for it in items if it["id"] not in have]
    print(time.strftime("%T"), "%d requests (%d items x %d models, missing only)" % (len(jobs), len(items), len(models)), flush=True)
    pools = {m: ThreadPoolExecutor(a.per_model_jobs) for m in models}
    futs = {pools[m].submit(run_one, base, key, m, it, stop): (m, it["id"]) for m, it in jobs}
    done = err = 0
    errs = {}
    t0 = last = time.time()
    for f in as_completed(futs):
        m, iid = futs[f]
        done += 1
        try:
            rec = f.result()
            if rec and rec.get("error"):
                err += 1
                errs[m] = errs.get(m, 0) + 1
                if errs[m] <= 3:
                    print(time.strftime("%T"), "ERR", m, iid, rec["error"][:200], flush=True)
        except FatalError as e:
            stop.set()
            print(time.strftime("%T"), "FATAL", e, flush=True)
        except Exception:
            err += 1
            print(time.strftime("%T"), "CRASH", m, iid, traceback.format_exc()[-300:], flush=True)
        if time.time() - last > 60 or done == len(jobs):
            last = time.time()
            print(time.strftime("%T"), "[%d/%d] errors %d %s | 429s %d | %.1f min" % (
                done, len(jobs), err, errs, LIMITER.n429, (time.time() - t0) / 60), flush=True)
    for p in pools.values():
        p.shutdown(wait=False)
    if stop.is_set():
        sys.exit(2)


if __name__ == "__main__":
    main()
