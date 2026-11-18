#!/usr/bin/env python3
"""Frame cache for both tracks (run once per video; run_* / judge_* also build missing frames on first use).

Reactive: 16 frames evenly spread over [max(0, t-60), t] (all <= t, no future), as 4 grids of 2x2, width 1280
          -> $COPLAY_FRAMES/reactive/<item id>/grid{0..3}.jpg
Proactive: every second of the video at 640 px -> $COPLAY_FRAMES/proactive/<vid>/s%05d.jpg. Frame s{N} shows the video
          at about N+0.95 s (the last frame of second N), so the newest frame at or before second `sec` is s{sec-1}.
          The 2x2 grid of one step (4 frames over the last step_sec) is built in memory by grid_bytes().
usage: frames.py reactive|proactive|all [--jobs 16] [--ids a,b]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FFMPEG, FRAMES, load_proactive, load_reactive, video_path  # noqa: E402

_LOCKS = {}
_GUARD = threading.Lock()


def _lock(name):
    with _GUARD:
        return _LOCKS.setdefault(name, threading.Lock())


# ---------------------------------------------------------------------------------------------------------- reactive
def frame_times(t):
    start = max(0.0, t - 60.0)
    return [round(start + (t - start) * i / 15, 2) for i in range(16)] if t > 0 else [0.0] * 16


def grids_ready(it):
    return all(os.path.isfile(g) and os.path.getsize(g) > 2000 for g in it["context"]["grids"])


def build_reactive(it, keep_frames=False):
    """-> (id, grid paths, frame times); it = an item from common.load_reactive()"""
    grids, ts = it["context"]["grids"], it["context"]["frame_times"]
    if grids_ready(it):
        return it["id"], grids, ts
    with _lock(it["id"]):
        if grids_ready(it):
            return it["id"], grids, ts
        d = os.path.dirname(grids[0])
        os.makedirs(d, exist_ok=True)
        fps = []
        for k, tt in enumerate(ts):
            fp = os.path.join(d, "f%02d.jpg" % k)
            if not (os.path.isfile(fp) and os.path.getsize(fp) > 1000):
                # -ss before -i: fast seek; frame at or just after tt is decoded -> clamp to <= t by seeking slightly earlier
                subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
                                "-ss", "%.2f" % max(0.0, tt - 0.05), "-i", video_path(it),
                                "-frames:v", "1", "-q:v", "4", "-threads", "1", fp], check=True)
            fps.append(fp)
        for g in range(4):
            cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-filter_complex_threads", "1"]
            for fp in fps[4 * g:4 * g + 4]:
                cmd += ["-f", "image2", "-c:v", "mjpeg", "-threads", "1", "-i", fp]
            cmd += ["-filter_complex", "[0][1][2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0,scale=1280:-2", "-q:v", "5",
                    grids[g] + ".tmp.jpg"]
            subprocess.run(cmd, check=True)
            os.replace(grids[g] + ".tmp.jpg", grids[g])
        if not keep_frames:
            for fp in fps:
                os.remove(fp)
    return it["id"], grids, ts


# --------------------------------------------------------------------------------------------------------- proactive
def frames_ready(s):
    return os.path.isfile(os.path.join(s["frames_dir"], "done"))


def extract(s, video=None):
    """1 fps / 640 px cache of a whole video; s needs frames_dir (+ video, or pass the absolute path)"""
    d = s["frames_dir"]
    if frames_ready(s):
        return s["id"], "skip"
    with _lock(d):
        if frames_ready(s):
            return s["id"], "skip"
        os.makedirs(d, exist_ok=True)
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-i", video or video_path(s),
                        "-vf", "fps=1:round=down,scale=640:-2", "-q:v", "6", os.path.join(d, "n%05d.jpg")], check=True)
        for f in os.listdir(d):
            if f.startswith("n"):
                os.replace(os.path.join(d, f), os.path.join(d, "s%05d.jpg" % (int(f[1:6]) - 1)))
        open(os.path.join(d, "done"), "w").write("ok")
    return s["id"], "%d frames" % len([f for f in os.listdir(d) if f.startswith("s")])


def grid_inputs(s, k):
    """the 4 cache frames of step k: evenly spread over [t_k - step, t_k], last frame at t_k"""
    t1 = k * s["step_sec"]
    t0 = max(0.0, t1 - s["step_sec"])
    secs = [int(round(t0 + (t1 - t0) * i / 3)) for i in range(4)]
    paths = []
    for sec in secs:
        # cache frame s{N} shows the video at N+0.93..N+0.97 s, so the newest frame at or before second `sec` is s{sec-1}
        sec = max(0, sec - 1)
        p = os.path.join(s["frames_dir"], "s%05d.jpg" % sec)
        while not os.path.isfile(p) and sec > 0:          # end of video: fall back to the last real frame
            sec -= 1
            p = os.path.join(s["frames_dir"], "s%05d.jpg" % sec)
        paths.append(p)
    return paths, [round(t0 + (t1 - t0) * i / 3, 1) for i in range(4)]


def grid_bytes(s, k):
    """2x2 grid of step k as JPEG bytes (ffmpeg -> stdout); the judge rebuilds the exact same image"""
    paths, times = grid_inputs(s, k)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]
    for p in paths:
        cmd += ["-f", "image2", "-c:v", "mjpeg", "-i", p]   # forced: some flat frames are probed as CAVS and xstack fails
    cmd += ["-filter_complex", "[0][1][2][3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0", "-q:v", "5", "-f", "image2pipe",
            "-vcodec", "mjpeg", "pipe:1"]
    return subprocess.run(cmd, check=True, capture_output=True).stdout, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=("reactive", "proactive", "all"))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--ids", help="comma-separated item ids (reactive) or video ids (proactive)")
    a = ap.parse_args()
    want = set(a.ids.split(",")) if a.ids else None
    if a.track in ("proactive", "all"):
        streams = [s for s in load_proactive() if want is None or s["id"] in want]
        with ThreadPoolExecutor(a.jobs) as ex:          # longest videos first so the tail is short
            for k, (sid, r) in enumerate(ex.map(extract, sorted(streams, key=lambda s: -s["duration_sec"])), 1):
                if k % 50 == 0 or k == len(streams):
                    print("proactive frames %d/%d (last %s: %s)" % (k, len(streams), sid, r), flush=True)
    if a.track in ("reactive", "all"):
        items = [it for it in load_reactive() if want is None or it["id"] in want or it["vid"] in want]
        fail = []
        with ThreadPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(build_reactive, it) for it in items]
            for k, f in enumerate(futs, 1):
                try:
                    f.result()
                except Exception as e:
                    fail.append(str(e)[:200])
                if k % 500 == 0 or k == len(items):
                    print("reactive grids %d/%d failed %d" % (k, len(items), len(fail)), flush=True)
        if fail:
            print("failures:", json.dumps(fail[:5]), file=sys.stderr)
            return 1
    print("frames ready under", FRAMES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
