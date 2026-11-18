#!/usr/bin/env python3
"""Proactive track, step 1: one streaming session per (model, video).

The model gets the player's opening line only, then every step_sec one 2x2 grid of the last step (4 frames from the
1 fps cache, last frame = now), its own memo and its last 8 lines, and answers JSON {speak, say, note}.
Real-time pacing: a call started at video second t that takes `total` s means the next look is the first step >= t+total
(skipped steps are not replayed; a failed call costs its wall time). Rate-limit waiting is not model latency.
Output $COPLAY_OUT/proactive/runs/<model>/<vid>.jsonl (one row per call). Resumable per session.
usage: run_proactive.py --models a,b [--sessions 300] [--streams vid1,vid2] [--limit N]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LIMITER, OUT, PAPER_MODELS, FatalError, load_config, load_proactive, stream_chat  # noqa: E402
from frames import extract, grid_bytes  # noqa: E402

SYSTEM = """You are the player's gaming buddy, watching their game screen together with them in real time.
Persona: a gaming buddy who is relaxed and cheerful, loves to hype the player up and joke around; never preachy, never sarcastic, no empty "you got this".
Every few seconds you receive a short clip of the latest screen and must decide yourself whether to speak now:
- At key moments (danger, a decision to make, a turning point, a great play, a mistake, winning or losing) speak up promptly with a concrete reminder, piece of advice or emotional support;
- When things are calm, stay quiet - do not talk every time. Rule: {QUIET};
- When you speak, keep it short and lead with the conclusion: one or two sentences (usually no more than 25 words). Only talk about what can be seen on screen or has already happened; do not make anything up.
Each time, output exactly one JSON object:
{"speak": true or false, "say": "what you say to the player (empty string when silent)", "note": "a memo of the game state for your next turn, at most 25 words"}"""

USER = ('Game: {game}\nAt the start the player told you: "{instr}" and has not spoken since.\n'
        "It is now video second {t:.0f}. Below is the screen from the last {step:.0f} seconds (a 2x2 grid, top-left -> top-right -> "
        "bottom-left -> bottom-right, frame times {times}; the last frame is now).\n"
        "Your memo from last time: {note}\nWhat you said recently:\n{hist}\nDecide whether to speak now. Output only the JSON.")
DEFAULT_QUIET = "outside key moments, at most one casual line every 2 minutes"
LOCKS = {}
LOCKS_GUARD = threading.Lock()


def parse_reply(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    say = str(d.get("say") or "").strip()
    return {"speak": bool(d.get("speak")) and bool(say), "say": say, "note": str(d.get("note") or "")[:300]}


def run_session(base, key, model, s, stop):
    out_p = os.path.join(OUT, "proactive", "runs", model, s["id"] + ".jsonl")
    with LOCKS_GUARD:                                       # one writer per session file
        lock = LOCKS.setdefault(out_p, threading.Lock())
    if not lock.acquire(blocking=False):
        return "busy"
    try:
        extract(s)
        os.makedirs(os.path.dirname(out_p), exist_ok=True)
        done = []
        if os.path.isfile(out_p):
            for l in open(out_p):
                try:
                    done.append(json.loads(l))
                except ValueError:
                    pass                                     # a torn last line from a killed process is ignored
        step = s["step_sec"]
        n_steps = int(s["duration_sec"] // step)
        system = SYSTEM.replace("{QUIET}", s.get("quiet_spans_policy") or DEFAULT_QUIET)
        note, said, next_k = "", [], 1
        for r in done:
            next_k = max(next_k, r.get("next_k") or r["k"] + 1)
            if r.get("parsed"):
                note = r["parsed"]["note"] or note
                if r["parsed"]["speak"]:
                    said.append((r["t"], r["parsed"]["say"]))
        with open(out_p, "a") as fo:
            k = next_k
            while k <= n_steps:
                if stop.is_set():
                    return "stopped"
                img, times = grid_bytes(s, k)
                t = round(k * step, 1)
                hist = "\n".join('[%.0fs] You said: %s' % (tt, x) for tt, x in said[-8:]) or "(you have not said anything yet)"
                text = USER.format(game=s["game_name"], instr=s["proactive_instruction"], t=t, step=step,
                                   times=", ".join("%.0fs" % x for x in times), note=note or "(none)", hist=hist)
                b64 = base64.b64encode(img).decode()
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": [{"type": "text", "text": text},
                                                     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]
                t_call = time.time()
                rec = stream_chat(base, key, model, msgs, retries=3, timeout=300)
                parsed = parse_reply(rec["text"])
                # real-time pacing; rate-limit waiting is inside stream_chat before t0, so `total` excludes it
                busy = rec["total"] if rec["total"] is not None else max(step, time.time() - t_call)
                nk = max(k + 1, int(-(-(t + busy) // step)))
                usage = rec.get("usage") or {}
                row = {"k": k, "t": t, "ttft": rec["ttft"], "total": rec["total"], "first_any": rec["first_any"],
                       "reasoning_chars": rec["reasoning_chars"], "finish_reason": rec["finish_reason"], "error": rec["error"],
                       "raw": (rec["text"] or "")[:600], "parsed": parsed, "next_k": nk, "skipped_after": nk - k - 1,
                       "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                       "completion_tokens": usage.get("completion_tokens")}
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                fo.flush()
                if parsed:
                    note = parsed["note"] or note
                    if parsed["speak"]:
                        said.append((t, parsed["say"]))
                k = nk
        return "ok"
    finally:
        lock.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--bench", default=None, help="default $COPLAY_DATA/bench/proactive.jsonl")
    ap.add_argument("--streams", help="comma-separated video ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke test)")
    ap.add_argument("--sessions", type=int, default=300, help="parallel (model, video) sessions")
    a = ap.parse_args()
    base, key = load_config()
    streams = load_proactive(a.bench)
    if a.streams:
        want = set(a.streams.split(","))
        streams = [s for s in streams if s["id"] in want]
    if a.limit:
        streams = streams[:a.limit]
    models = a.models.split(",")
    stop = threading.Event()
    # longest videos first (they bound the wall time), models interleaved
    jobs = [(m, s) for s in sorted(streams, key=lambda s: -s["duration_sec"]) for m in models]
    rng = random.Random(1)
    head, tail = jobs[:len(models) * 40], jobs[len(models) * 40:]
    rng.shuffle(tail)
    jobs = head + tail
    total_steps = sum(int(s["duration_sec"] // s["step_sec"]) for s in streams) * len(models)
    print(time.strftime("%T"), "%d sessions, %d video steps, parallel sessions %d, limiter %.0f rpm" % (
        len(jobs), total_steps, a.sessions, LIMITER.rpm), flush=True)
    stat = {}
    fatal = False

    def progress():
        while not stop.is_set():
            time.sleep(300)
            print(time.strftime("%T"), "sessions %s | 429s %d" % (dict(stat), LIMITER.n429), flush=True)
    threading.Thread(target=progress, daemon=True).start()
    with cf.ThreadPoolExecutor(a.sessions) as ex:
        futs = {ex.submit(run_session, base, key, m, s, stop): (m, s["id"]) for m, s in jobs}
        for f in cf.as_completed(futs):
            m, sid = futs[f]
            try:
                r = f.result()
            except FatalError as e:
                stop.set()
                fatal = True
                r = "fatal"
                print(time.strftime("%T"), "FATAL", e, flush=True)
            except Exception as e:
                r = "crash"
                print(time.strftime("%T"), "CRASH", m, sid, str(e)[:300], flush=True)
            stat[r] = stat.get(r, 0) + 1
    stop.set()
    print(time.strftime("%T"), "done", stat, flush=True)
    if fatal:
        sys.exit(2)


if __name__ == "__main__":
    main()
