#!/usr/bin/env python3
"""Reactive track, step 2: grade replies with the judge (default qwen3.8-flash, thinking off).

The judge sees the same 4 grids as the model under test + the item's GT + the reply.
  --models a,b    grade reactive/runs/<model>/<id>.json      -> reactive/judge/<model>/<id>.json
  --reference     grade the GT reference reply of every item -> reactive/judge/gt-reference/<id>.json
                  Run this first: it defines the advice set Adv (reference judged strategy_ok with a non-empty advice_end),
                  which decides needs_strategy for every model.
  --repeat N      re-grade N random (model, item) pairs independently -> reactive/judge_repeat/<model>/<id>.json
An empty / unparsable judge answer is retried (3 attempts).
usage: judge_reactive.py [--reference] [--models a,b] [--repeat N] [--limit N] [--jobs 160]
"""
import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JUDGE_MODEL, LIMITER, OUT, FatalError, chat_once, load_judge_config, load_reactive, models_in  # noqa: E402
from frames import build_reactive  # noqa: E402

JUDGE_SYSTEM = """You are a strict grader for a "game companion" benchmark. You will see the game information the player could see (screenshots up to the moment of the question), the annotated ground truth (GT), and one reply from the AI companion under test. Score each item objectively and output exactly one JSON object and nothing else.

Rules:
1. key_points: for every GT must_mention point, decide whether the reply expresses the same meaning (different wording with the same meaning counts as covered=true).
2. claims: extract every assertion in the reply about the game state / history / actions (exclude purely emotional words). Label each with status: supports_gt = matches a GT point; wrong = contradicts the screen or GT, or is made up; irrelevant = correct but unrelated to the GT points.
3. facts: extract every verifiable game fact in the reply (counts, positions, who played what, HP, wave number, cards left, who is where...). Label each verdict: correct (confirmed by the screen or GT), wrong (refuted or clearly made up), unverifiable (cannot be judged from the information given).
4. strategy_ok: judge only when needs_strategy=true (otherwise null). Is the main advice in the reply legal, actionable and a reasonable strategy (in line with the GT recommendation, or different but equally reasonable in that situation)? Giving no advice at all counts as false.
5. emotion_ok: judge only when the player spoke (otherwise null). Did the reply correctly perceive and appropriately respond to the player's emotion (GT emotion given below)?
6. support_score: emotional support quality 0-4.
   0 = wrong or inappropriate emotional response, or blaming / mocking the player; 1 = only mechanical, template-like comfort or pleasantries; 2 = responds to the emotion but not tied to the specific game situation; 3 = tied to the specific situation with appropriate encouragement / comfort / praise; 4 = natural, specific, well-judged emotional support whose tone fully matches the GT vibe, like a real friend.
   Note: when the GT vibe is "Hyped" or "Short & Fiery", the right emotional value is energetic hype, not comfort; for "Over-the-Top Praise" it is specific praise.
7. advice_end: if the reply contains one complete, correct piece of advice relevant to the current need, copy verbatim the last 4-10 words of the reply up to where that advice ends (must be a contiguous substring of the reply); otherwise null.
8. must_not_violations: which GT must_not items the reply violates (copy the item text), [] if none.

Output JSON format:
{"key_points":[{"point":"...","covered":true}],"claims":[{"text":"...","status":"supports_gt"}],"facts":[{"text":"...","verdict":"correct"}],"strategy_ok":true,"emotion_ok":true,"support_score":3,"advice_end":"...","must_not_violations":[],"brief":"one-sentence comment"}"""

# question-type label shown to the judge (the paper's runs used "passive" for in-play questions)
JUDGE_QTYPE = {"in_play": "passive", "aftermath": "aftermath"}
ADVICE = None


def advice_set():
    """Adv = items whose GT reference reply was judged to contain correct advice (fixed before grading models)"""
    global ADVICE
    if ADVICE is None:
        d = os.path.join(OUT, "reactive", "judge", "gt-reference")
        ADVICE = set()
        if os.path.isdir(d):
            def rd(f):
                try:
                    j = json.load(open(os.path.join(d, f)))
                except ValueError:
                    return None
                return f[:-5] if j.get("strategy_ok") is True and j.get("advice_end") else None
            with ThreadPoolExecutor(64) as ex:
                ADVICE = {x for x in ex.map(rd, os.listdir(d)) if x}
    return ADVICE


def needs_strategy(it):
    return it["id"] in advice_set()


def gt_block(it, ns):
    lines = ["[Question type] %s (%s)   needs_strategy=%s" % (JUDGE_QTYPE[it["question_type"]], it.get("response_kind") or "", str(ns).lower()),
             "[Event] %s: %s" % (it.get("event_type"), it.get("objective") or ""),
             "[Situation at the question moment (GT)] %s" % (it.get("situation") or "")]
    if it["question_type"] == "aftermath":
        lines.append("[Event outcome (GT, aftermath only)] %s" % (it.get("outcome") or ""))
    lines += ["[Player message] %s" % it["player_message"],
              "[GT player emotion] %s" % (it.get("gt_emotion") or "(none)"),
              "[GT must_mention] " + "; ".join(it["must_mention"]),
              "[GT must_not] " + "; ".join(it["must_not"]),
              "[GT expected vibe] %s; emotion requirements: %s" % (it.get("vibe"), "; ".join(it["emotion_requirements"])),
              "[GT reference reply] %s" % (it.get("reference_response") or "")]
    return "\n".join(lines)


def build_messages(it, reply, ns):
    parts = [{"type": "text", "text": "Game: %s\n===== The player's screen over the last minute (2x2 grids; the last frame is the question moment) =====" % it["game_name"]}]
    for path in it["context"]["grids"]:
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()}})
    parts.append({"type": "text", "text": "\n===== GROUND TRUTH =====\n" + gt_block(it, ns) + "\n===== REPLY UNDER TEST =====\n" +
                  (reply if reply.strip() else "(empty reply)") + "\nOutput only the JSON according to the rules."})
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": parts}]


def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON")
    d = json.loads(m.group(0))
    for k in ("key_points", "claims", "facts", "must_not_violations"):
        if not isinstance(d.get(k), list):
            d[k] = []
    s = d.get("support_score")
    d["support_score"] = max(0, min(4, int(s))) if isinstance(s, (int, float)) or (isinstance(s, str) and s.isdigit()) else None
    return d


def grade(base, key, it, reply, out, ns, stop):
    """reply: a run-file path (read here, in the worker thread) or {"text": ...} for the GT reference"""
    if os.path.isfile(out) or stop.is_set():
        return "skip"
    reply = reply["text"] if isinstance(reply, dict) else json.load(open(reply))["text"]
    build_reactive(it)
    msgs = build_messages(it, reply, ns)
    last = None
    for attempt in range(3):
        try:
            text, usage = chat_once(base, key, JUDGE_MODEL, msgs, temperature=0.0, timeout=600)
            d = parse(text)
            d["_usage"] = usage
            d["_needs_strategy"] = ns
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(d, open(out + ".tmp", "w"), ensure_ascii=False)
            os.replace(out + ".tmp", out)
            return "ok"
        except FatalError:
            raise
        except Exception as e:
            last = str(e)[:200]
    return "fail: " + (last or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--bench", default=None, help="default $COPLAY_DATA/bench/reactive.jsonl")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="first N items only (smoke test; use the same N for every step)")
    ap.add_argument("--jobs", type=int, default=160)
    a = ap.parse_args()
    base, key = load_judge_config()
    items = load_reactive(a.bench)
    items = {it["id"]: it for it in (items[:a.limit] if a.limit else items)}
    root = os.path.join(OUT, "reactive")
    jobs = []      # (item, reply, out_path, needs_strategy)
    if a.reference:
        for it in items.values():   # reference judged as an advice-needing item (needs_strategy=true) to define the set
            jobs.append((it, {"text": it.get("reference_response") or ""}, os.path.join(root, "judge", "gt-reference", it["id"] + ".json"), True))
    models = [x for x in a.models.split(",") if x]
    ref_dir = os.path.join(root, "judge", "gt-reference")
    if models and (a.reference or not os.path.isdir(ref_dir)):
        raise SystemExit("run judge_reactive.py --reference on its own first: it defines which items need advice")
    if models and len(os.listdir(ref_dir)) < len(items):
        print("WARNING: only %d / %d reference judgements; needs_strategy is incomplete" % (len(os.listdir(ref_dir)), len(items)), flush=True)
    for m in models:
        rd, jdir = os.path.join(root, "runs", m), os.path.join(root, "judge", m)
        have_run = set(os.listdir(rd)) if os.path.isdir(rd) else set()
        have_judge = set(os.listdir(jdir)) if os.path.isdir(jdir) else set()
        for it in items.values():
            f = it["id"] + ".json"
            if f in have_run and f not in have_judge:
                jobs.append((it, os.path.join(rd, f), os.path.join(jdir, f), needs_strategy(it)))
    if a.repeat:
        rng = random.Random(20261104)
        pool = []
        for m in [m for m in models_in(os.path.join(root, "judge")) if m != "gt-reference"]:
            have = set(os.listdir(os.path.join(root, "judge", m)))
            pool += [(m, iid) for iid in items if iid + ".json" in have]
        for m, iid in rng.sample(pool, min(a.repeat, len(pool))):
            out = os.path.join(root, "judge_repeat", m, iid + ".json")
            jobs.append((items[iid], os.path.join(root, "runs", m, iid + ".json"), out, needs_strategy(items[iid])))
    if a.repeat or a.reference:
        jobs = [j for j in jobs if not os.path.isfile(j[2])]
    random.Random(7).shuffle(jobs)          # interleave models: partial runs still cover every model evenly
    print(time.strftime("%T"), "%d judge requests" % len(jobs), flush=True)
    stop = threading.Event()
    stat, t0, last = {}, time.time(), time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(grade, base, key, *j, stop) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except FatalError as e:
                stop.set()
                r = "fatal"
                print(time.strftime("%T"), "FATAL", e, flush=True)
            except Exception as e:
                r = "crash"
                print(time.strftime("%T"), "CRASH", str(e)[:200], flush=True)
            k_ = r.split(":")[0]
            stat[k_] = stat.get(k_, 0) + 1
            if k_ == "fail" and stat["fail"] <= 5:
                print(time.strftime("%T"), "FAIL", r[:200], flush=True)
            if time.time() - last > 60 or k == len(jobs):
                last = time.time()
                print(time.strftime("%T"), "[%d/%d] %s | 429s %d | %.1f min" % (k, len(jobs), stat, LIMITER.n429, (time.time() - t0) / 60), flush=True)
    if stop.is_set():
        sys.exit(2)


if __name__ == "__main__":
    main()
