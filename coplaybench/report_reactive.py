#!/usr/bin/env python3
"""Reactive track, step 3: metrics (paper names) -> $COPLAY_OUT/results/reactive_metrics.json + reactive_results.md

Advice set Adv = items whose GT reference reply was judged strategy_ok with a non-empty advice_end.
  GroundF1   key-point micro F1: TP covered must_mention, FN not covered, FP claims judged wrong
  StratAcc   mean(strategy_ok) on Adv            FactAcc   correct / (correct + wrong) facts
  EmoAcc     mean(emotion_ok)                    SupQual   sum(support) / (4 n)
  StratEmo   mean(strategy_ok and support >= 3) on Adv
  TTFT       median time to the first visible token (+ p90)
  TTUA       median time until the advice_end substring finished streaming, Adv items (+ p90; NoAdvice = share without one)
  OnTime     Adv -> TTUA <= D, other -> full reply end <= D;  D = min(scenario deadline, max(0.5, window left));
             deadlines urgent 2 s, strategic 5 s, emotional 10 s.   OnTime_F = same with D = scenario deadline only
  Violation  share of replies violating a must_not item       LenOK   English words <= max_words_en
Every metric: overall / board 4 / visual 5 / in-play / aftermath / per game, 95% bootstrap CI (2000 resamples) for the
headline ones, self-annotation subsets (GT not written by one annotator model) and translation-source subsets.
usage: report_reactive.py [--models a,b]   (default: every model under reactive/runs)
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DISPLAY, GAMES, OUT, PAPER_MODELS, load_reactive, models_in  # noqa: E402

DEADLINE = {"urgent": 2.0, "strategic": 5.0, "emotional": 10.0}
ANNOTATORS = ["claude-opus-5.5", "gpt-6.1-sol", "gemini-4-argon"]
B = 2000


def lj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def words(s):
    return len(re.findall(r"[A-Za-z0-9']+", s or ""))


def ttua(run, jd):
    end = jd.get("advice_end")
    text = run.get("text") or ""
    if not end or not run.get("chunks"):
        return None
    end = str(end).strip().strip('"\'')
    pos = text.find(end)
    if pos < 0:
        tail = " ".join(end.split()[-6:])
        pos = text.find(tail)
        end = tail
    if pos < 0:
        return None
    need = pos + len(end)
    for t, n in run["chunks"]:
        if n >= need:
            return t
    return run["chunks"][-1][0]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(q * (len(xs) - 1))] if xs else None


def per_item(it, run, jd, adv):
    """-> dict of per-item contributions (None when the reply or its judgement is missing)"""
    if not run or not jd:
        return None
    text = run.get("text") or ""
    kp = jd.get("key_points") or []
    cov = sum(1 for k in kp if k.get("covered") is True)
    fc = sum(1 for f in jd.get("facts") or [] if f.get("verdict") == "correct")
    fw = sum(1 for f in jd.get("facts") or [] if f.get("verdict") == "wrong")
    s = jd.get("support_score")
    ok = jd.get("strategy_ok") is True
    a = it["id"] in adv
    t_adv = ttua(run, jd) if a else None
    t_eff = t_adv if a else (run.get("total") if text.strip() else None)
    d = min(DEADLINE[it["scenario"]], max(0.5, it.get("window_len_sec") or 99))
    return {"tp": cov, "fn": len(kp) - cov, "fp": sum(1 for c in jd.get("claims") or [] if c.get("status") == "wrong"),
            "fc": fc, "fw": fw, "adv": a, "a2": ok if a else None, "b3": (ok and s is not None and s >= 3) if a else None,
            "b1": jd.get("emotion_ok") is True, "b2": s, "ttft": run.get("ttft"), "ttua": t_adv,
            "c3": t_eff is not None and t_eff <= d, "c3f": t_eff is not None and t_eff <= DEADLINE[it["scenario"]],
            "mn": bool(jd.get("must_not_violations")),
            "len": (words(text) <= it["max_words_en"]) if it.get("max_words_en") else None, "empty": not text.strip(),
            "rt": ((run.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            "trunc": run.get("finish_reason") == "length"}


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return {"n": 0}
    tp, fn, fp = sum(r["tp"] for r in rows), sum(r["fn"] for r in rows), sum(r["fp"] for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    m = lambda xs: sum(xs) / len(xs) if xs else None
    fc, fw = sum(r["fc"] for r in rows), sum(r["fw"] for r in rows)
    b2 = [r["b2"] for r in rows if r["b2"] is not None]
    adv = [r for r in rows if r["adv"]]
    tt = [r["ttua"] for r in adv]
    return {"n": len(rows), "n_adv": len(adv), "GroundF1": 2 * P * R / (P + R) if P + R else 0.0, "GroundP": P, "GroundR": R,
            "StratAcc": m([r["a2"] for r in adv]), "FactAcc": fc / (fc + fw) if fc + fw else None,
            "EmoAcc": m([r["b1"] for r in rows]), "SupQual": sum(b2) / (4 * len(b2)) if b2 else None,
            "StratEmo": m([r["b3"] for r in adv]),
            "TTFT": pct([r["ttft"] for r in rows], .5), "TTFT_p90": pct([r["ttft"] for r in rows], .9),
            "TTUA": pct(tt, .5), "TTUA_p90": pct(tt, .9),
            "NoAdvice": sum(1 for x in tt if x is None) / len(tt) if tt else None,
            "OnTime": m([r["c3"] for r in rows]), "OnTime_F": m([r["c3f"] for r in rows]),
            "Violation": m([r["mn"] for r in rows]), "LenOK": m([r["len"] for r in rows if r["len"] is not None]),
            "empty": sum(r["empty"] for r in rows),
            "think_share": m([r.get("rt", 0) > 0 for r in rows]), "think_tok_mean": m([r.get("rt", 0) for r in rows]),
            "truncated": sum(1 for r in rows if r.get("trunc"))}


def boot(vals, seed):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [bs[int(.025 * B)], bs[int(.975 * B)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="default: every model under $COPLAY_OUT/reactive/runs")
    ap.add_argument("--bench", default=None)
    a = ap.parse_args()
    root = os.path.join(OUT, "reactive")
    found = models_in(os.path.join(root, "runs"))
    models = a.models.split(",") if a.models else [m for m in PAPER_MODELS if m in found] + [m for m in found if m not in PAPER_MODELS]
    items = load_reactive(a.bench)
    adv = set()
    for it in items:
        j = lj(os.path.join(root, "judge", "gt-reference", it["id"] + ".json"))
        if j and j.get("strategy_ok") is True and j.get("advice_end"):
            adv.add(it["id"])

    def load_model(m):
        return m, [(it, per_item(it, lj(os.path.join(root, "runs", m, it["id"] + ".json")),
                                 lj(os.path.join(root, "judge", m, it["id"] + ".json")), adv)) for it in items]
    with ThreadPoolExecutor(12) as ex:
        data = dict(ex.map(load_model, models))
    data["gt-reference"] = [(it, per_item(it, {"text": it.get("reference_response") or "", "ttft": None, "total": None, "chunks": []},
                                          lj(os.path.join(root, "judge", "gt-reference", it["id"] + ".json")), adv)) for it in items]
    out = {"n_items": len(items), "n_adv": len(adv), "models": {}}
    subsets = {"all": lambda it: True, "board": lambda it: it["board"], "visual": lambda it: not it["board"],
               "in_play": lambda it: it["question_type"] == "in_play", "aftermath": lambda it: it["question_type"] == "aftermath"}
    for an in ANNOTATORS[:2]:
        subsets["gt_not_by_" + an] = (lambda an_: lambda it: it["gt_source"] != an_)(an)
    for tr in ("claude-opus-5.5", "gemini-4-argon"):
        subsets["translated_by_" + tr] = (lambda tr_: lambda it: it["translated_by"] == tr_)(tr)
    for g in GAMES:
        subsets["game_" + g] = (lambda g_: (lambda it: it["game"] == g_))(g)
    for m, rows in data.items():
        res = {k: agg([r for it, r in rows if f(it)]) for k, f in subsets.items()}
        allr = [r for _, r in rows if r]
        res["all"]["StratAcc_ci"] = boot([r["a2"] for r in allr if r["adv"]], 1)
        res["all"]["StratEmo_ci"] = boot([r["b3"] for r in allr if r["adv"]], 2)
        res["all"]["SupQual_ci"] = [x / 4 for x in boot([r["b2"] for r in allr if r["b2"] is not None], 3) or []] or None
        res["all"]["coverage"] = "%d/%d answered+judged" % (len(allr), len(items))
        out["models"][m] = res
    pairs = []
    for m in models:
        d = os.path.join(root, "judge_repeat", m)
        if os.path.isdir(d):
            for f in os.listdir(d):
                a_, b_ = lj(os.path.join(d, f)), lj(os.path.join(root, "judge", m, f))
                if a_ and b_:
                    pairs.append((a_, b_))
    if pairs:
        same = lambda k: sum(1 for x, y in pairs if x.get(k) == y.get(k)) / len(pairs)
        b2 = [(x.get("support_score"), y.get("support_score")) for x, y in pairs if x.get("support_score") is not None and y.get("support_score") is not None]
        out["judge_consistency"] = {"pairs": len(pairs), "strategy_same": same("strategy_ok"), "emotion_same": same("emotion_ok"),
                                    "support_exact": sum(1 for x, y in b2 if x == y) / len(b2) if b2 else None,
                                    "support_within1": sum(1 for x, y in b2 if abs(x - y) <= 1) / len(b2) if b2 else None}
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "results", "reactive_metrics.json"), "w"), ensure_ascii=False, indent=1)

    p = lambda v: "-" if v is None else "%.1f" % (100 * v)
    s_ = lambda v: "-" if v is None else "%.1f" % v
    ci = lambda c: "-" if not c else "[%.1f, %.1f]" % (100 * c[0], 100 * c[1])
    name = lambda m: DISPLAY.get(m, m)
    cols = ["GroundF1", "StratAcc", "FactAcc", "EmoAcc", "SupQual", "StratEmo", "TTFT", "TTUA", "OnTime", "OnTime_F",
            "Violation", "LenOK"]
    sec = {"TTFT", "TTUA"}
    row = lambda r: " | ".join(s_(r.get(k)) if k in sec else p(r.get(k)) for k in cols)
    L = ["# CoPlayBench - reactive track", "",
         "%d questions (in-play %d, aftermath %d), %d advice questions (Adv). Values in %%, TTFT / TTUA in seconds." % (
             len(items), sum(1 for i in items if i["question_type"] == "in_play"),
             sum(1 for i in items if i["question_type"] == "aftermath"), len(adv)), "",
         "## Overall", "", "| Model | N | " + " | ".join(cols) + " | StratAcc 95% CI | StratEmo 95% CI | Empty | Truncated |",
         "|---|---|" + "---|" * (len(cols) + 4)]
    for m in models + ["gt-reference"]:
        r = out["models"][m]["all"]
        L.append("| %s | %d | %s | %s | %s | %s | %s |" % (name(m), r["n"], row(r), ci(r.get("StratAcc_ci")), ci(r.get("StratEmo_ci")),
                                                       r.get("empty", "-"), r.get("truncated", "-")))
    for key, title in (("board", "Board games (4)"), ("visual", "Visual games (5)"), ("in_play", "In-play questions"),
                       ("aftermath", "Aftermath questions")):
        L += ["", "## " + title, "", "| Model | N | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for m in models:
            r = out["models"][m][key]
            L.append("| %s | %d | %s |" % (name(m), r["n"], row(r)))
    L += ["", "## Per game: StratEmo / StratAcc / OnTime", "", "| Model | " + " | ".join(GAMES) + " |", "|---|" + "---|" * len(GAMES)]
    for m in models:
        L.append("| %s | " % name(m) + " | ".join("%s / %s / %s" % (p(out["models"][m]["game_" + g].get("StratEmo")),
                                                                    p(out["models"][m]["game_" + g].get("StratAcc")),
                                                                    p(out["models"][m]["game_" + g].get("OnTime"))) for g in GAMES) + " |")
    ks = ["all"] + ["gt_not_by_" + an for an in ANNOTATORS[:2]]
    L += ["", "## Self-annotation check (claude-opus-5.5 and gpt-6.1-sol are GT annotators)", "",
          "| Model | " + " | ".join("%s %s" % (x, k) for x in ("StratAcc", "StratEmo", "SupQual") for k in ks) + " |", "|---|" + "---|" * 9]
    for m in models:
        r = out["models"][m]
        L.append("| %s | " % name(m) + " | ".join(p(r[k].get(x)) for x in ("StratAcc", "StratEmo", "SupQual") for k in ks) + " |")
    if out.get("judge_consistency"):
        c = out["judge_consistency"]
        L += ["", "## Judge consistency (%d replies graded twice)" % c["pairs"], "",
              "| strategy_ok same | emotion_ok same | support exact | support within 1 |", "|---|---|---|---|",
              "| %s | %s | %s | %s |" % (p(c["strategy_same"]), p(c["emotion_same"]), p(c["support_exact"]), p(c["support_within1"]))]
    open(os.path.join(OUT, "results", "reactive_results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:8 + len(models) + 1]))


if __name__ == "__main__":
    main()
