#!/usr/bin/env python3
"""Build the LlamaFactory SFT data (sharegpt + images) from the Train annotations, split by training/split.json.

Every sample is ONE call exactly as the benchmark makes it (coplaybench/run_reactive.py, run_proactive.py): same system
prompt, same user text, same images (frame times, 2x2 grid layout, width), so the train and test inputs match.

  in_play   : node.in_play (t, player line)          -> 4 grids of 16 frames over [t-60, t], width 1280 -> reference_response
  aftermath : node.aftermath.question                -> same input, aftermath wording                  -> reference_response
  proactive : one step k of the streaming session -> one 2x2 grid of the last step_sec + memo + last 8 lines of an *ideal*
              trajectory -> {"speak", "say", "note"} JSON
     ideal trajectory: speak reference_response at the first step inside each proactive window ([start-1, end+1], as the
     metric) at which the reference is already true (timing audit valid_from), silent elsewhere; memo always empty.
     steps sampled per video:  positive = first such step in each window
                               hard negatives = a later step in the same window after we already spoke (no repeat),
                                                1-2 steps before a window opens
                               easy negatives = random steps away from all windows, about as many as the positives
In-play / aftermath questions whose reference states something that only happens after the question time (timing audit)
are dropped. The prompt texts are imported from the benchmark code (not copied); their sha1 is stored in manifest.json.

Inputs ($COPLAY_DATA): annotations/train/<Game>.jsonl, training/split.json, training/gt_timing.jsonl, videos/train/
Outputs (--out, default $COPLAY_DATA/training): sft_train.json / sft_dev.json (image paths relative to --out),
  images/reactive/<id>/grid{0..3}.jpg, images/proactive/<vid>/k%05d.jpg, manifest.json
usage: build_sft_data.py [--jobs 32] [--limit-videos N] [--json-only]
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "coplaybench"))
import run_proactive as RS  # noqa: E402   (SYSTEM, USER, DEFAULT_QUIET)
import run_reactive as RP  # noqa: E402    (SYSTEM, Q_TEXT, build_messages)
from common import DATA, FFMPEG, GAMES, NO_ADVICE_TYPES, STEP  # noqa: E402
from frames import extract, frame_times, grid_inputs  # noqa: E402

PASSIVE_HEAD = ("Game: %s\nBelow are the last 60 seconds of the player's screen, as 4 images; each image is a 2x2 grid "
                "of 4 frames (top-left -> top-right -> bottom-left -> bottom-right). The last frame of the last "
                "image is the current moment.")
# force JPEG decoding of grid inputs: ffmpeg's probe misdetects some flat frames (e.g. a black Stardew frame) as CAVS
# video and the xstack then fails. Output bytes are identical to a plain "-i" on normal frames.
JPG_IN = ["-f", "image2", "-c:v", "mjpeg"]
TMP = os.path.join(tempfile.gettempdir(), "coplay_sft")


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def check_prompts():
    """the reactive user head is built inline in run_reactive.build_messages; make sure ours is still identical"""
    it = {"game_name": "G", "question_type": "in_play", "t_sec": 1.0, "player_message": "m",
          "context": {"frame_times": [0.0] * 16, "grids": [os.devnull] * 4}}
    msgs = RP.build_messages(it)
    assert msgs[1]["content"][0]["text"] == PASSIVE_HEAD % "G", "reactive prompt in run_reactive.py changed"
    assert msgs[0]["content"] == RP.SYSTEM


# --------------------------------------------------------------------------------------------------------- timing
def load_timing():
    """vid -> {node_id: audit row} from training/gt_timing.jsonl"""
    out = {}
    for l in open(os.path.join(DATA, "training", "gt_timing.jsonl")):
        d = json.loads(l)
        out[d["vid"]] = {r["node_id"]: r for r in d["nodes"]}
    return out


def ref_future(timing, node_id, kind):
    """True if the reference of this question states something that happens after the question time"""
    row = (timing or {}).get(node_id)
    if row is None:
        return False
    return bool(row.get("q_ref_future" if kind == "in_play" else "a_ref_future"))


# ------------------------------------------------------------------------------------------------- reactive items
def reactive_items(game, r):
    out = []
    for n in r["nodes"]:
        for kind in ("in_play", "aftermath"):
            q = n.get(kind)
            if not q:
                continue
            p = q if kind == "in_play" else (q.get("question") or {})
            t, msg = p.get("t_sec"), p.get("text")
            ref = (q.get("reference_response") or "").strip()
            if t is None or not msg or not ref:
                continue
            t = round(float(t), 1)
            if t > float(r["duration_sec"]):
                continue
            out.append({"id": "%s_%s_%s" % (r["vid"], n["node_id"], "inplay" if kind == "in_play" else kind), "kind": kind,
                        "node_id": n["node_id"], "game": game, "game_name": r["game_name"], "vid": r["vid"],
                        "video": os.path.join(DATA, r["video"]), "t": t, "player_message": msg, "reference": ref})
    return out


def render_reactive(it, img_dir):
    d = os.path.join(img_dir, it["id"])
    grids = [os.path.join(d, "grid%d.jpg" % g) for g in range(4)]
    ts = frame_times(it["t"])
    if all(os.path.isfile(g) and os.path.getsize(g) > 2000 for g in grids):
        return grids, ts
    tmp = os.path.join(TMP, "single", it["id"])                  # single frames on local disk, only the grids are kept
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(d, exist_ok=True)
    fps = []
    for k, tt in enumerate(ts):
        fp = os.path.join(tmp, "f%02d.jpg" % k)
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", "%.2f" % max(0.0, tt - 0.05),
                        "-i", it["video"], "-frames:v", "1", "-q:v", "4", fp], check=True)
        if not os.path.isfile(fp):                    # seek past the last frame: reuse the previous one
            shutil.copy(fps[-1], fp)
        fps.append(fp)
    for g in range(4):
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
        for fp in fps[4 * g:4 * g + 4]:
            cmd += JPG_IN + ["-i", fp]
        cmd += ["-filter_complex", "[0][1][2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,scale=1280:-2", "-q:v", "5",
                grids[g] + ".tmp.jpg"]
        subprocess.run(cmd, check=True)
        os.replace(grids[g] + ".tmp.jpg", grids[g])
    shutil.rmtree(tmp, ignore_errors=True)
    return grids, ts


def reactive_sample(it, grids, ts):
    content = PASSIVE_HEAD % it["game_name"]
    for g in range(4):
        content += "[frame times " + ", ".join("%.1fs" % x for x in ts[4 * g:4 * g + 4]) + "]<image>"
    content += RP.Q_TEXT[it["kind"]].format(t=it["t"], msg=it["player_message"])
    return {"id": it["id"], "kind": it["kind"], "game": it["game"],
            "messages": [{"role": "system", "content": RP.SYSTEM}, {"role": "user", "content": content},
                         {"role": "assistant", "content": it["reference"]}],
            "images": grids}


# ------------------------------------------------------------------------------------------------ proactive steps
def proactive_plan(game, r, rng, neg_ratio, timing=None):
    """ideal trajectory + the steps to train on.
    A positive (speak the proactive reference) is placed at the first step t inside the metric window with
    t >= valid_from (timing audit: the reference only becomes true once every fact it states has appeared on screen;
    frames of step t end at t). Events whose reference is never true inside the window get no positive; their whole
    window stays excluded from negatives (ambiguous)."""
    step = STEP[game]
    dur = float(r["duration_sec"])
    n_steps = int(dur // step)
    evs = []
    for n in r["nodes"]:
        p = n.get("proactive") or {}
        w = p.get("window")
        ref = (p.get("reference_response") or "").strip()
        if not w or not ref:
            continue
        lo, hi = float(w["start_sec"]) - 1, float(w["end_sec"]) + 1          # metric tolerance
        ks = [k for k in range(1, n_steps + 1) if lo <= k * step <= hi]
        if not ks:
            continue
        row = (timing or {}).get(n["node_id"])
        vf = float(row["p_valid_from"]) if row else float(w["start_sec"])
        ks_ok = [k for k in ks if k * step >= vf - 1e-6]                      # reference already true at step k
        evs.append({"node_id": n["node_id"], "k_first": ks_ok[0] if ks_ok else None, "ks": ks, "ks_ok": ks_ok,
                    "say": ref, "lo": lo, "hi": hi, "valid_from": vf, "tier": n.get("tier"), "type": n.get("type"),
                    "needs_advice": n.get("type") not in NO_ADVICE_TYPES})
    # ideal trajectory: one utterance per event at its first in-window step; two events on the same step -> keep the
    # Required / earlier one (one line per step is all the format allows)
    speak_at = {}
    for e in sorted([e for e in evs if e["ks_ok"]], key=lambda e: (e["k_first"], e["tier"] != "Required")):
        k = e["k_first"]
        while k in speak_at and k <= e["ks_ok"][-1]:
            k += 1
        if k <= e["ks_ok"][-1] and k not in speak_at:
            speak_at[k] = e
    in_win = set(k for e in evs for k in e["ks"])
    pick = {}
    for k, e in speak_at.items():
        pick[k] = ("pos", e)
    for k, e in speak_at.items():
        later = [kk for kk in e["ks_ok"] if kk > k and kk not in speak_at]
        if later:
            pick.setdefault(rng.choice(later), ("neg_after", e))
        before = [kk for kk in (k - 1, k - 2) if kk >= 1 and kk not in in_win]
        if before:
            pick.setdefault(rng.choice(before), ("neg_before", e))
    far = [k for k in range(1, n_steps + 1) if k not in in_win and k not in pick
           and all(abs(k - kk) > 2 for kk in speak_at)]
    rng.shuffle(far)
    for k in far[:int(round(neg_ratio * len(speak_at)))]:
        pick[k] = ("neg_far", None)
    if not speak_at and far:                        # videos with no usable window still teach silence a little
        for k in far[:2]:
            pick[k] = ("neg_far", None)
    return evs, speak_at, pick, n_steps


def stream_text(game_name, instr, k, step, note, said):
    t = round(k * step, 1)
    t0 = max(0.0, t - step)
    times = [round(t0 + (t - t0) * i / 3, 1) for i in range(4)]
    hist = "\n".join("[%.0fs] You said: %s" % (tt, x) for tt, x in said[-8:]) or "(you have not said anything yet)"
    return RS.USER.format(game=game_name, instr=instr, t=t, step=step, times=", ".join("%.0fs" % x for x in times),
                          note=note or "(none)", hist=hist), times


def render_video_steps(video, vid, step, ks, out_dir):
    """all sampled steps of one video: 1 fps cache in a local temp dir (same extraction as the benchmark), then the
    benchmark's own frame choice (frames.grid_inputs) and grid filter; cache deleted afterwards"""
    outs = [os.path.join(out_dir, vid, "k%05d.jpg" % k) for k in ks]
    if all(os.path.isfile(o) and os.path.getsize(o) > 2000 for o in outs):
        return
    s = {"id": vid, "step_sec": step, "frames_dir": os.path.join(TMP, "1fps", vid)}
    extract(s, video)
    os.makedirs(os.path.join(out_dir, vid), exist_ok=True)
    try:
        for k, out in zip(ks, outs):
            if os.path.isfile(out) and os.path.getsize(out) > 2000:
                continue
            paths, _ = grid_inputs(s, k)
            cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
            for p in paths:
                cmd += JPG_IN + ["-i", p]
            cmd += ["-filter_complex", "[0][1][2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0", "-q:v", "5",
                    out + ".tmp.jpg"]
            subprocess.run(cmd, check=True)
            os.replace(out + ".tmp.jpg", out)
    finally:
        shutil.rmtree(s["frames_dir"], ignore_errors=True)


def proactive_samples(game, r, rng, neg_ratio, img_dir, timing=None):
    """memo / note: always empty. The benchmark feeds back the model's own note; GT `situation` describes the whole event
    (and for board games comes from the engine timeline), so using it as the memo would leak later facts. With empty notes
    the benchmark input stays "(none)" too (run_proactive keeps the old note when the new one is empty)."""
    evs, speak_at, pick, n_steps = proactive_plan(game, r, rng, neg_ratio, timing)
    step = STEP[game]
    instr = r["proactive_instruction"]
    system = RS.SYSTEM.replace("{QUIET}", r.get("quiet_spans_policy") or RS.DEFAULT_QUIET)
    out, jobs = [], []
    note, said = "", []
    for k in range(1, n_steps + 1):
        if k in pick:
            label, e = pick[k]
            text, _ = stream_text(r["game_name"], instr, k, step, note, said)
            ans = {"speak": True, "say": e["say"], "note": ""} if label == "pos" else {"speak": False, "say": "", "note": ""}
            jobs.append(k)
            out.append({"id": "%s_k%05d" % (r["vid"], k), "kind": "proactive_" + label, "game": game,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": text + "<image>"},       # benchmark sends [text, image]
                                     {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)}],
                        "images": [os.path.join(img_dir, r["vid"], "k%05d.jpg" % k)]})
        if k in speak_at:                         # advance the ideal trajectory
            e = speak_at[k]
            said.append((round(k * step, 1), e["say"]))
    return out, (os.path.join(DATA, r["video"]), r["vid"], step, jobs, img_dir), evs


# ------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "training"))
    ap.add_argument("--limit-videos", type=int, default=0, help="per game, for a quick smoke build")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--neg-ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--json-only", action="store_true", help="write the json without rendering images")
    a = ap.parse_args()
    check_prompts()
    out_dir = os.path.abspath(a.out)
    split = json.load(open(os.path.join(DATA, "training", "split.json")))
    timing_all = load_timing()
    img_dir = os.path.join(out_dir, "images")
    per_split = {"sft": [], "dev": []}
    p_jobs, s_jobs = [], []
    tstat = Counter()
    for game in GAMES:
        n_vid = 0
        for l in open(os.path.join(DATA, "annotations", "train", game + ".jsonl")):
            r = json.loads(l)
            side = split[r["vid"]]
            if side not in per_split:                 # RL videos: see training/rl
                continue
            if a.limit_videos and n_vid >= a.limit_videos:
                continue
            n_vid += 1
            timing = timing_all.get(r["vid"])
            tstat["videos_with_timing" if timing else "videos_without_timing"] += 1
            for it in reactive_items(game, r):
                if ref_future(timing, it["node_id"], it["kind"]):     # the reference states something after the question time
                    tstat["qa_dropped_ref_future"] += 1
                    continue
                p_jobs.append((side, it))
            rng = random.Random("%d-%s" % (a.seed, r["vid"]))   # per video: picks do not depend on the split
            samples, vjob, evs = proactive_samples(game, r, rng, a.neg_ratio, os.path.join(img_dir, "proactive"), timing)
            tstat["events"] += len(evs)
            tstat["events_no_positive"] += sum(1 for e in evs if not e["ks_ok"])
            per_split[side] += samples
            if vjob[3]:
                s_jobs.append(vjob)
    print(time.strftime("%T"), "reactive items %d, proactive videos %d / steps %d" % (
        len(p_jobs), len(s_jobs), sum(len(j[3]) for j in s_jobs)), flush=True)

    fails = []
    if a.json_only:
        for side, it in p_jobs:
            d = os.path.join(img_dir, "reactive", it["id"])
            per_split[side].append(reactive_sample(it, [os.path.join(d, "grid%d.jpg" % g) for g in range(4)], frame_times(it["t"])))
    else:
        with ThreadPoolExecutor(a.jobs) as ex:
            # proactive videos first (long 1 fps extractions), reactive items after
            futs = {ex.submit(render_video_steps, va, vid, step, ks, od): ("proactive", vid)
                    for va, vid, step, ks, od in sorted(s_jobs, key=lambda j: -len(j[3]))}
            futs.update({ex.submit(render_reactive, it, os.path.join(img_dir, "reactive")): (side, it) for side, it in p_jobs})
            for done, (f, (side, it)) in enumerate(list(futs.items()), 1):
                try:
                    res = f.result()
                    if side != "proactive":
                        per_split[side].append(reactive_sample(it, *res))
                except Exception as e:
                    fails.append((it["id"] if isinstance(it, dict) else it, str(e)[:200]))
                if done % 1000 == 0:
                    print(time.strftime("%T"), "jobs %d/%d failed %d" % (done, len(futs), len(fails)), flush=True)
    bad = set(f[0] for f in fails)
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for side, rows in per_split.items():
        rows = [s for s in rows if s["id"] not in bad and s["id"].rsplit("_k", 1)[0] not in bad]
        if not a.json_only:
            with ThreadPoolExecutor(64) as ex:
                ok = list(ex.map(lambda s: all(os.path.isfile(p) for p in s["images"]), rows))
            rows = [s for s, k in zip(rows, ok) if k]
        random.Random(a.seed).shuffle(rows)
        counts[side] = dict(Counter(s["kind"] for s in rows))
        name = "sft_train.json" if side == "sft" else "sft_dev.json"
        with open(os.path.join(out_dir, name + ".tmp"), "w") as fo:
            json.dump([{"messages": s["messages"], "images": [os.path.relpath(p, out_dir) for p in s["images"]],
                        "id": s["id"], "kind": s["kind"], "game": s["game"]} for s in rows], fo, ensure_ascii=False)
        os.replace(os.path.join(out_dir, name + ".tmp"), os.path.join(out_dir, name))
    man = {"built": time.strftime("%F %T"), "args": vars(a), "counts": counts, "timing": dict(tstat),
           "image_failures": len(fails), "failures_sample": fails[:10],
           "prompt_sha1": {"reactive_system": sha(RP.SYSTEM), "reactive_head": sha(PASSIVE_HEAD),
                           "reactive_q": sha(json.dumps(RP.Q_TEXT, sort_keys=True)), "proactive_system": sha(RS.SYSTEM),
                           "proactive_user": sha(RS.USER)}}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(man, indent=1, ensure_ascii=False)[:2000])
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
