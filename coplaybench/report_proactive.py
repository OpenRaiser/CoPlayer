#!/usr/bin/env python3
"""Proactive track, step 3: metrics (paper names) -> $COPLAY_OUT/results/proactive_metrics.json + proactive_results.md

Event window W = [window.start - 1, window.end + 1] (1 fps granularity).
  Recall / Recall_S   share of Required events with an utterance at t (ideal) / at t + TTFT (strict) inside W
  RandBase            same number of utterances at uniform random times over the covered duration, 20 trials,
                      seed = sum of the session id's character codes
  TimingLift          Recall - RandBase
  SeenRate            share of Required windows the model looked at (some step t in W; steps are skipped while it is busy)
  Recall_O            Recall over the Required events whose window the model looked at
  OptRecall           Recall for Optional events
  HitStrat / HitSup / HitStratEmo / HitViolation   first in-window utterance graded against the event GT;
                      HitStrat and HitStratEmo only on events that need advice
  SpeakRate           utterances per minute      NoiseRate   off-window chat + wrong + noise per minute
  OffUseful / OffWrong  share of off-window lines judged useful / wrong
  QuotaOver           share of 2-min slots (1-min for Stardew Valley) with more than one off-window non-useful line
  TTFT                median over calls; SkipShare = skipped steps / (skipped + calls)
covered duration = min(video duration, (last next_k) * step).  Per model, per game, and self-annotation subsets;
95% bootstrap CI for Recall (events).
usage: report_proactive.py [--models a,b]   (default: every model under proactive/runs)
"""
import argparse
import collections
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DISPLAY, GAMES, OUT, PAPER_MODELS, load_proactive, models_in  # noqa: E402

TOL = 1.0


def lj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def load_rows(p):
    rows = []
    if os.path.isfile(p):
        for l in open(p):
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
    return rows


def session(m, s, ev_filter=lambda e: True):
    rows = load_rows(os.path.join(OUT, "proactive", "runs", m, s["id"] + ".jsonl"))
    jd = lj(os.path.join(OUT, "proactive", "judge", m, s["id"] + ".json"))
    if not rows or jd is None:
        return None
    step = s["step_sec"]
    covered = min(s["duration_sec"], max((r.get("next_k") or r["k"] + 1) for r in rows) * step)
    looks = [r["t"] for r in rows]
    utts = [r for r in rows if r.get("parsed") and r["parsed"]["speak"]]
    evs = [e for e in s["events"] if ev_filter(e)]
    c = collections.Counter()
    hits = {"A2": [], "B2": [], "B3": [], "mn": []}
    must_flags = []
    for e in evs:
        w0, w1 = e["window"]["start_sec"] - TOL, e["window"]["end_sec"] + TOL
        tier = "must" if e["tier"] == "Required" else "opt"
        obs = any(w0 <= t <= w1 for t in looks)
        ideal = any(w0 <= u["t"] <= w1 for u in utts)
        strict = any(w0 <= u["t"] + (u.get("ttft") or 0) <= w1 for u in utts)
        c[tier + "_n"] += 1
        c[tier + "_obs"] += obs
        c[tier + "_ideal"] += ideal
        c[tier + "_strict"] += strict
        c[tier + "_ideal_obs"] += ideal and obs
        if tier == "must":
            must_flags.append(ideal)
        je = jd["events"].get(e["node_id"])
        if je and je.get("k") is not None and "support_score" in je:
            ok = je.get("strategy_ok") is True
            sc = je.get("support_score")
            if sc is not None:
                hits["B2"].append(sc)
            if e.get("needs_advice", True):
                hits["A2"].append(ok)
                hits["B3"].append(ok and sc is not None and sc >= 3)
            hits["mn"].append(bool(je.get("must_not_violations")))
    rng = random.Random(sum(map(ord, s["id"])))
    for _ in range(20):
        ts = [rng.uniform(0, covered) for _ in utts]
        for e in evs:
            if e["tier"] == "Required":
                c["rand_n"] += 1
                c["rand_hit"] += any(e["window"]["start_sec"] - TOL <= x <= e["window"]["end_sec"] + TOL for x in ts)
    slot = 60.0 if s["game"] == "StardewValley" else 120.0
    slots = collections.Counter()
    for u in utts:
        if any(e["window"]["start_sec"] - TOL <= u["t"] <= e["window"]["end_sec"] + TOL
               for e in s["events"] + (s.get("dropped_events") or [])):
            continue                      # in an event window (or a dropped event's window): not an off-window line
        cat = (jd["offwin"].get(str(u["k"])) or {}).get("category")
        if cat is None:
            continue
        c["off_" + cat] += 1
        if cat != "useful":
            slots[int(u["t"] // slot)] += 1
    c["quiet_slots"] += max(1, int(-(-covered // slot)))      # ceil: the last partial slot can hold off-window lines too
    c["quiet_over"] += sum(1 for v in slots.values() if v > 1)
    c["speak"] += len(utts)
    c["calls"] += len(rows)
    c["skipped"] += sum(r.get("skipped_after") or 0 for r in rows)
    c["noparse"] += sum(1 for r in rows if not r.get("error") and r.get("parsed") is None)
    c["errors"] += sum(1 for r in rows if r.get("error"))
    c["think_calls"] += sum(1 for r in rows if (r.get("reasoning_tokens") or 0) > 0 or (r.get("reasoning_chars") or 0) > 0)
    c["think_tok"] += sum(r.get("reasoning_tokens") or 0 for r in rows)
    c["truncated"] += sum(1 for r in rows if r.get("finish_reason") == "length")
    return {"c": c, "hits": hits, "min": covered / 60, "ttft": [r["ttft"] for r in rows if r.get("ttft")], "must_flags": must_flags}


def combine(parts):
    parts = [p for p in parts if p]
    c = collections.Counter()
    hits = {"A2": [], "B2": [], "B3": [], "mn": []}
    mins, ttft, flags = 0.0, [], []
    for p in parts:
        c.update(p["c"])
        for k in hits:
            hits[k] += p["hits"][k]
        mins += p["min"]
        ttft += p["ttft"]
        flags += p["must_flags"]
    f = lambda a, b: c[a] / c[b] if c[b] else None
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    off = c["off_useful"] + c["off_chat"] + c["off_wrong"] + c["off_noise"]
    ri, rr = f("must_ideal", "must_n"), f("rand_hit", "rand_n")
    return {"sessions": len(parts), "video_min": round(mins, 1), "calls": c["calls"],
            "Recall": ri, "Recall_S": f("must_strict", "must_n"), "RandBase": rr,
            "TimingLift": (ri - rr) if ri is not None and rr is not None else None,
            "Recall_O": f("must_ideal_obs", "must_obs"), "SeenRate": f("must_obs", "must_n"),
            "OptRecall": f("opt_ideal", "opt_n"), "OptRecall_S": f("opt_strict", "opt_n"),
            "HitStrat": mean(hits["A2"]), "HitSup": sum(hits["B2"]) / (4 * len(hits["B2"])) if hits["B2"] else None,
            "HitStratEmo": mean(hits["B3"]), "HitViolation": mean(hits["mn"]), "hits_judged": len(hits["B2"]),
            "SpeakRate": c["speak"] / mins if mins else None,
            "NoiseRate": (c["off_chat"] + c["off_wrong"] + c["off_noise"]) / mins if mins else None,
            "OffUseful": c["off_useful"] / off if off else None, "OffWrong": c["off_wrong"] / off if off else None,
            "QuotaOver": f("quiet_over", "quiet_slots"), "TTFT": statistics.median(ttft) if ttft else None,
            "SkipShare": c["skipped"] / (c["skipped"] + c["calls"]) if c["calls"] else None,
            "noparse": c["noparse"], "errors": c["errors"], "_flags": flags,
            "think_share": c["think_calls"] / c["calls"] if c["calls"] else None,
            "think_tok_mean": c["think_tok"] / c["calls"] if c["calls"] else None, "truncated": c["truncated"]}


def boot(flags, seed=7, B=2000):
    if not flags:
        return None
    rng = random.Random(seed)
    n = len(flags)
    bs = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [bs[int(.025 * B)], bs[int(.975 * B)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="default: every model under $COPLAY_OUT/proactive/runs")
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    found = models_in(os.path.join(OUT, "proactive", "runs"))
    models = a.models.split(",") if a.models else [m for m in PAPER_MODELS if m in found] + [m for m in found if m not in PAPER_MODELS]
    streams = load_proactive(a.bench)
    subsets = {"all": (lambda s: True, lambda e: True),
               "gt_not_by_claude-opus-5.5": (lambda s: True, lambda e: e["gt_source"] != "claude-opus-5.5"),
               "gt_not_by_gpt-6.1-sol": (lambda s: True, lambda e: e["gt_source"] != "gpt-6.1-sol")}
    for g in GAMES:
        subsets["game_" + g] = ((lambda g_: lambda s: s["game"] == g_)(g), lambda e: True)

    def model_metrics(m):
        res = {}
        for name, (sf, ef) in subsets.items():
            res[name] = combine([session(m, s, ef) for s in streams if sf(s)])
        res["all"]["Recall_ci"] = boot(res["all"]["_flags"])
        for v in res.values():
            v.pop("_flags", None)
        return m, res
    with ThreadPoolExecutor(12) as ex:
        out = {"models": dict(ex.map(model_metrics, models)), "videos": len(streams)}
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "results", "proactive_metrics.json"), "w"), ensure_ascii=False, indent=1)
    p = lambda v: "-" if v is None else "%.1f" % (100 * v)
    num = lambda v, f: "-" if v is None else f % v
    ci = lambda c: "-" if not c else "[%.1f, %.1f]" % (100 * c[0], 100 * c[1])
    name = lambda m: DISPLAY.get(m, m)
    L = ["# CoPlayBench - proactive track", "",
         "%d full videos; after one opening line the model sees one 2x2 grid every 2/4/6 s and decides itself when to speak; "
         "steps are skipped in real time while it is busy. Values in %%, rates per minute, TTFT in seconds." % len(streams), "",
         "| Model | Sessions | Recall [95% CI] | Recall-S | RandBase | **TimingLift** | SeenRate | Recall-O | OptRecall | HitStrat | "
         "HitSup | **HitStratEmo** | HitViolation | SpeakRate | NoiseRate | OffUseful | OffWrong | QuotaOver | TTFT | SkipShare |",
         "|---|" + "---|" * 19]
    for m in models:
        r = out["models"][m]["all"]
        L.append("| %s | %d | %s %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name(m), r["sessions"], p(r["Recall"]), ci(r.get("Recall_ci")), p(r["Recall_S"]), p(r["RandBase"]),
            num(None if r["TimingLift"] is None else 100 * r["TimingLift"], "%+.1f"), p(r["SeenRate"]), p(r["Recall_O"]),
            p(r["OptRecall"]), p(r["HitStrat"]), p(r["HitSup"]), p(r["HitStratEmo"]), p(r["HitViolation"]),
            num(r["SpeakRate"], "%.1f"), num(r["NoiseRate"], "%.2f"), p(r["OffUseful"]), p(r["OffWrong"]), p(r["QuotaOver"]),
            num(r["TTFT"], "%.1f"), p(r["SkipShare"])))
    L += ["", "## Thinking and truncation check (per call)", "",
          "| Model | Calls | Calls with thinking | Mean thinking tokens | Truncated (1024 used up) | Unparsable | Errors |",
          "|---|---|---|---|---|---|---|"]
    for m in models:
        r = out["models"][m]["all"]
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (name(m), r["calls"], p(r.get("think_share")), num(r.get("think_tok_mean"), "%.1f"),
                                                      r.get("truncated"), r.get("noparse"), r.get("errors")))
    L += ["", "## Recall / Recall-S per game", "", "| Model | " + " | ".join(GAMES) + " |", "|---|" + "---|" * len(GAMES)]
    for m in models:
        L.append("| %s | " % name(m) + " | ".join("%s / %s" % (p(out["models"][m]["game_" + g]["Recall"]),
                                                              p(out["models"][m]["game_" + g]["Recall_S"])) for g in GAMES) + " |")
    ks = ["all", "gt_not_by_claude-opus-5.5", "gt_not_by_gpt-6.1-sol"]
    L += ["", "## Self-annotation check (claude-opus-5.5 and gpt-6.1-sol are GT annotators)", "",
          "| Model | " + " | ".join("%s %s" % (x, k) for x in ("Recall", "HitStratEmo") for k in ks) + " |", "|---|" + "---|" * 6]
    for m in models:
        r = out["models"][m]
        L.append("| %s | " % name(m) + " | ".join(p(r[k].get(x)) for x in ("Recall", "HitStratEmo") for k in ks) + " |")
    open(os.path.join(OUT, "results", "proactive_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:6 + len(models)]))


if __name__ == "__main__":
    main()
