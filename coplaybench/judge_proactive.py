#!/usr/bin/env python3
"""Proactive track, step 2: grade the utterances of every session with the judge (default qwen3.8-flash, thinking off).

Per (model, video) session:
  * in-window hits: for each GT event, the FIRST utterance whose ideal time t is in [window.start-1, window.end+1] is graded
    against that event's GT with the reactive rubric (judge_reactive.JUDGE_SYSTEM); needs_strategy = event needs_advice
  * off-window utterances: category useful / chat / wrong / noise (OFFWIN_SYSTEM), with the 3 previous lines for context
The judge sees the 2x2 grid the model saw at that step. Output $COPLAY_OUT/proactive/judge/<model>/<vid>.json
= {"events": {node_id: {...}}, "offwin": {k: {...}}}. Resumable; partial results are saved per session.
usage: judge_proactive.py --models a,b [--jobs 60]
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JUDGE_MODEL, LIMITER, OUT, PAPER_MODELS, FatalError, chat_once, load_judge_config, load_proactive  # noqa: E402
from frames import extract, grid_bytes  # noqa: E402
from judge_reactive import JUDGE_SYSTEM, parse  # noqa: E402

TOL = 1.0
INNER_JOBS = 8
OFFWIN_SYSTEM = """You are a strict grader for a "game companion" benchmark. The AI companion said one line on its own initiative while the player had not asked anything, and this moment is NOT one of the annotated key events.
Judge from the screen which category the line belongs to and output only JSON:
{"category": "useful|chat|wrong|noise", "reason": "one sentence"}
useful = specific, correct and genuinely helpful reminder / advice / praise (valuable even outside a key moment);
chat   = correct but low-information small talk, pleasantries, or restating the screen (e.g. "you played a card", "go go");
wrong  = contradicts the screen, made up, or bad advice;
noise  = repetitive, almost the same as something just said, or clearly disturbing the player."""


def utterances(rows):
    return [{"k": r["k"], "t": r["t"], "say": r["parsed"]["say"], "ttft": r.get("ttft")}
            for r in rows if r.get("parsed") and r["parsed"]["speak"]]


def gt_block(ev):
    return "\n".join([
        "[Question type] proactive (%s)   needs_strategy=%s" % (ev.get("response_kind") or "", str(ev.get("needs_advice", True)).lower()),
        "[Event] %s: %s" % (ev.get("type"), ev.get("objective") or ""),
        "[Situation at this moment (GT)] %s" % (ev.get("situation") or ""),
        "[Player message] (none - the player did not speak; this is a moment the AI should speak up)",
        "[GT player emotion] (none)",
        "[GT must_mention] " + "; ".join(ev["must_mention"]),
        "[GT must_not] " + "; ".join(ev["must_not"]),
        "[GT expected vibe] %s; emotion requirements: %s" % (ev.get("vibe"), "; ".join(ev.get("emotion_requirements") or [])),
        "[GT reference reply] %s" % (ev.get("reference_response") or "")])


def grid_part(s, k):
    """rebuild the exact grid the model saw at step k (same 1 fps frames, same ffmpeg settings)"""
    img, _ = grid_bytes(s, k)
    return [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}]


def event_messages(s, ev, u, grid=None):
    parts = [{"type": "text", "text": "Game: %s\n===== What the AI saw at this moment (2x2 grid; the last frame is now) =====" % s["game_name"]}]
    parts += grid or grid_part(s, u["k"])
    parts.append({"type": "text", "text": "\n===== GROUND TRUTH =====\n" + gt_block(ev) +
                  "\n===== REPLY UNDER TEST (said on its own at video second %.0f) =====\n%s\nOutput only the JSON according to the rules." % (u["t"], u["say"])})
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": parts}]


def offwin_messages(s, u, prev, grid=None):
    parts = [{"type": "text", "text": "Game: %s\nScreen (2x2 grid; the last frame is now):" % s["game_name"]}]
    parts += grid or grid_part(s, u["k"])
    parts.append({"type": "text", "text": "The AI said before: %s\nAt this moment (second %.0f) the AI says on its own: \"%s\"\nOutput only the JSON." % (
        " | ".join(prev[-3:]) or "(nothing)", u["t"], u["say"])})
    return [{"role": "system", "content": OFFWIN_SYSTEM}, {"role": "user", "content": parts}]


def judge_event(base, key, s, ev, u):
    text, usage = chat_once(base, key, JUDGE_MODEL, event_messages(s, ev, u), temperature=0.0, timeout=600)
    d = parse(text)
    d["_usage"] = usage
    return d


def judge_offwin(base, key, s, u, prev):
    text, usage = chat_once(base, key, JUDGE_MODEL, offwin_messages(s, u, prev), temperature=0.0, timeout=600)
    m = re.search(r"\{.*\}", text or "", re.S)
    d = json.loads(m.group(0)) if m else None
    if not d or d.get("category") not in ("useful", "chat", "wrong", "noise"):
        raise ValueError("bad offwin answer")
    return {"category": d["category"], "reason": d.get("reason", "")}


def retry(fn, *a):
    for attempt in range(3):
        try:
            return fn(*a)
        except FatalError:
            raise
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


def in_any_window(s, t):
    """dropped_events (timing audit): their windows were real key moments with a broken GT -> utterances there are
    neither graded hits nor off-window lines"""
    return any(e["window"]["start_sec"] - TOL <= t <= e["window"]["end_sec"] + TOL for e in s["events"] + (s.get("dropped_events") or []))


def judge_session(base, key, model, s, stop):
    out = os.path.join(OUT, "proactive", "judge", model, s["id"] + ".json")
    rp = os.path.join(OUT, "proactive", "runs", model, s["id"] + ".jsonl")
    if not os.path.isfile(rp) or stop.is_set():
        return "no-run"
    res = {"events": {}, "offwin": {}}
    if os.path.isfile(out):
        try:
            res = json.load(open(out))
        except ValueError:
            pass
    rows = []
    for l in open(rp):
        try:
            rows.append(json.loads(l))
        except ValueError:
            pass
    utts = utterances(rows)
    tasks = []
    for ev in s["events"]:
        if ev["node_id"] in res["events"] and (res["events"][ev["node_id"]].get("k") is None or "strategy_ok" in res["events"][ev["node_id"]]):
            continue
        w0, w1 = ev["window"]["start_sec"] - TOL, ev["window"]["end_sec"] + TOL
        hit = next((u for u in utts if w0 <= u["t"] <= w1), None)
        if hit is None:
            res["events"][ev["node_id"]] = {"k": None}
        else:
            tasks.append(("ev", ev, hit))
    prev = []
    for u in utts:
        if not in_any_window(s, u["t"]) and str(u["k"]) not in res["offwin"]:
            tasks.append(("off", u, list(prev)))
        prev.append(u["say"])
    if tasks:
        extract(s)

    def one(t):
        if t[0] == "ev":
            d = retry(judge_event, base, key, s, t[1], t[2])
            if d is not None:
                d["k"] = t[2]["k"]
            return "ev", t[1]["node_id"], d
        return "off", str(t[1]["k"]), retry(judge_offwin, base, key, s, t[1], t[2])

    with cf.ThreadPoolExecutor(INNER_JOBS) as ex:
        for kind, k_, d in ex.map(one, tasks):
            if d is not None:
                res["events" if kind == "ev" else "offwin"][k_] = d
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out + ".tmp", "w"), ensure_ascii=False)
    os.replace(out + ".tmp", out)
    missing = sum(1 for t in tasks if (t[0] == "ev" and t[1]["node_id"] not in res["events"]) or
                  (t[0] == "off" and str(t[1]["k"]) not in res["offwin"]))
    return "ok" if not missing else "partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--bench", default=None, help="default $COPLAY_DATA/bench/proactive.jsonl")
    ap.add_argument("--jobs", type=int, default=60)
    a = ap.parse_args()
    base, key = load_judge_config()
    streams = load_proactive(a.bench)
    jobs = [(m, s) for m in a.models.split(",") for s in streams]
    stop = threading.Event()
    print(time.strftime("%T"), "%d sessions to judge" % len(jobs), flush=True)
    stat = {}
    t0 = last = time.time()
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(judge_session, base, key, m, s, stop): (m, s["id"]) for m, s in jobs}
        for k, f in enumerate(cf.as_completed(futs), 1):
            try:
                r = f.result()
            except FatalError as e:
                stop.set()
                r = "fatal"
                print("FATAL", e, flush=True)
            except Exception as e:
                r = "crash"
                print("CRASH", futs[f], str(e)[:200], flush=True)
            stat[r] = stat.get(r, 0) + 1
            if time.time() - last > 120 or k == len(jobs):
                last = time.time()
                print(time.strftime("%T"), "[%d/%d] %s | 429s %d | %.1f min" % (k, len(jobs), stat, LIMITER.n429, (time.time() - t0) / 60), flush=True)
    if stop.is_set():
        sys.exit(2)


if __name__ == "__main__":
    main()
