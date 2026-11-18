"""Shared configuration and the streaming OpenAI-compatible client.

Every request is streamed so latency can be measured per token:
  t0            just before the HTTP request is sent (input fully prepared)
  first_any     first streamed chunk of any kind (incl. hidden reasoning)
  first_content first VISIBLE content token  -> TTFT (what the player waits for)
  chunks        [(seconds since t0, cumulative visible chars)] -> used for TTUA

Configuration (environment variables, see .env.example):
  COPLAY_DATA            local copy of the CoPlayBench dataset (default ./data/CoPlayBench)
  COPLAY_FRAMES          frame cache (default ./cache/frames, about 15 GB for the full benchmark)
  COPLAY_OUT             outputs: runs, judgements, reports (default ./outputs)
  COPLAY_BASE_URL        OpenAI-compatible endpoint of the model under test, e.g. https://api.openai.com/v1
  COPLAY_API_KEY         its key
  COPLAY_JUDGE_BASE_URL  endpoint of the judge (default: COPLAY_BASE_URL)
  COPLAY_JUDGE_API_KEY   its key (default: COPLAY_API_KEY)
  COPLAY_RPM             request budget per minute for this process (default 500)
  FFMPEG                 ffmpeg binary (default: ffmpeg on PATH)
"""
import json
import os
import random
import shutil
import threading
import time

import requests

DATA = os.path.abspath(os.environ.get("COPLAY_DATA", "data/CoPlayBench"))
FRAMES = os.path.abspath(os.environ.get("COPLAY_FRAMES", "cache/frames"))
OUT = os.path.abspath(os.environ.get("COPLAY_OUT", "outputs"))
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"

GAMES = ["HonorOfKings", "DouDiZhu", "Mahjong", "UNO", "Go", "PlantsvsZombies", "CarrotFantasy", "Hearthstone", "StardewValley"]
BOARD = {"DouDiZhu", "Mahjong", "UNO", "Go"}
# proactive step length per game (seconds between two looks at the screen)
STEP = {"DouDiZhu": 2.0, "Mahjong": 2.0, "UNO": 2.0, "Go": 2.0,
        "HonorOfKings": 4.0, "PlantsvsZombies": 4.0, "CarrotFantasy": 4.0, "Hearthstone": 4.0, "StardewValley": 6.0}
# emotion-only event types: no advice expected (needs_advice = false)
NO_ADVICE_TYPES = {"Endgame", "Game End", "Level Clear", "Stage Clear", "Highlight", "Game Decided", "Harvest/Big Profit",
                   "Social/Story Event", "Day Start", "Mine Milestone", "Fishing"}

# Protocol of the paper: thinking disabled for every model and for the judge, at most 1024 output tokens,
# temperature 0.3 for the models under test and 0.0 for the judge.
MAX_OUT = 1024
TEMPERATURE = 0.3
JUDGE_MODEL = "qwen3.8-flash"
# Request parameters that turn thinking off for the models evaluated in the paper. They were probed on an
# OpenAI-compatible gateway; a provider's own API may name the switch differently. Models that are not listed
# (e.g. a local vLLM server started with --default-chat-template-kwargs '{"enable_thinking": false}') get none.
NOTHINK_PARAMS = {
    "gpt-5.6-sol": {"reasoning_effort": "none"},
    "deepseek-v4.1-flash": {"thinking": {"type": "disabled"}},
    "qwen3.8-max": {"reasoning_effort": "none"},
    "kimi-k3": {"thinking": {"type": "disabled"}},             # reasoning_effort none still thought on 18/27 smoke items
    "claude-sonnet-5.5": {"reasoning_effort": "minimal"},      # mostly 0 thinking; `thinking: disabled` is worse on this route
    "gemini-3.8-flash": {"reasoning_effort": "none"},          # about half 0, rest 80-290 thinking tokens (cannot go lower)
    "hy4": {"thinking": {"type": "disabled"}},
    "doubao-seed-2.1-pro": {"thinking": {"type": "disabled"}},  # reasoning_effort has no effect on doubao
    "qwen3.8-flash": {"reasoning_effort": "none"},              # judge
}
PAPER_MODELS = ["gpt-5.6-sol", "deepseek-v4.1-flash", "qwen3.8-max", "kimi-k3", "claude-sonnet-5.5", "gemini-3.8-flash",
                "hy4", "doubao-seed-2.1-pro"]
DISPLAY = {"gpt-5.6-sol": "GPT-5.6-Sol", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "qwen3.8-max": "Qwen3.8-Max",
           "kimi-k3": "Kimi-K3", "claude-sonnet-5.5": "Claude-Sonnet-5.5", "gemini-3.8-flash": "Gemini-3.8-Flash",
           "hy4": "HY4", "doubao-seed-2.1-pro": "Doubao-Seed-2.1-Pro", "gt-reference": "Reference reply"}


def extra_params(model):
    return dict(NOTHINK_PARAMS.get(model, {}))


def load_reactive(path=None):
    """reactive items + where their 4 grids live in the frame cache (built by frames.py on first use)"""
    from frames import frame_times
    items = []
    for l in open(path or os.path.join(DATA, "bench", "reactive.jsonl")):
        it = json.loads(l)
        it["context"] = {"grids": [os.path.join(FRAMES, "reactive", it["id"], "grid%d.jpg" % g) for g in range(4)],
                         "frame_times": frame_times(it["t_sec"])}
        items.append(it)
    return items


def load_proactive(path=None):
    """proactive sessions + their 1 fps frame cache directory"""
    streams = []
    for l in open(path or os.path.join(DATA, "bench", "proactive.jsonl")):
        s = json.loads(l)
        s["frames_dir"] = os.path.join(FRAMES, "proactive", s["id"])
        streams.append(s)
    return streams


def video_path(rec):
    return os.path.join(DATA, rec["video"])


def models_in(d):
    return sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d, m))) if os.path.isdir(d) else []


def _endpoint(prefix):
    base = os.environ.get(prefix + "BASE_URL") or os.environ.get("COPLAY_BASE_URL")
    key = os.environ.get(prefix + "API_KEY") or os.environ.get("COPLAY_API_KEY") or "EMPTY"
    if not base:
        raise SystemExit("set COPLAY_BASE_URL (and COPLAY_API_KEY), see .env.example")
    return base.rstrip("/"), key


def load_config():
    """-> (base_url, api_key) of the models under test"""
    return _endpoint("COPLAY_")


def load_judge_config():
    """-> (base_url, api_key) of the judge"""
    return _endpoint("COPLAY_JUDGE_")


class FatalError(RuntimeError):
    pass


def _fatal(status, body):
    return status in (401, 403) or any(f in body for f in ("insufficient_quota", "Budget has been exceeded", "budget_exceeded"))


def stream_chat(base, key, model, messages, max_tokens=MAX_OUT, temperature=TEMPERATURE, timeout=300, retries=3):
    """-> dict(text, reasoning_chars, ttft, first_any, total, chunks, usage, attempts, error)"""
    last_err = None
    for attempt in range(1, retries + 1):
        LIMITER.wait()
        t0 = time.time()
        rec = {"text": "", "reasoning_chars": 0, "ttft": None, "first_any": None, "total": None,
               "chunks": [], "usage": {}, "attempts": attempt, "error": None, "finish_reason": None}
        try:
            r = requests.post(base + "/chat/completions", stream=True, timeout=(30, timeout),
                              headers={"Authorization": "Bearer " + key},
                              json={"model": model, "messages": messages, "max_tokens": min(max_tokens, MAX_OUT),
                                    "temperature": temperature, "stream": True,
                                    "stream_options": {"include_usage": True}, **extra_params(model)})
            if r.status_code != 200:
                body = r.text[:500].replace(key, "<KEY>")
                if _fatal(r.status_code, body):
                    raise FatalError("HTTP %d %s" % (r.status_code, body))
                if r.status_code == 429:
                    LIMITER.hit_429()
                raise RuntimeError("HTTP %d %s" % (r.status_code, body))
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                d = json.loads(data)
                now = time.time() - t0
                if d.get("usage"):
                    rec["usage"] = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
                    rc = delta.get("reasoning_content") or ""
                    c = delta.get("content") or ""
                    if (rc or c) and rec["first_any"] is None:
                        rec["first_any"] = round(now, 3)
                    rec["reasoning_chars"] += len(rc)
                    if c:
                        if rec["ttft"] is None:
                            rec["ttft"] = round(now, 3)
                        rec["text"] += c
                        rec["chunks"].append([round(now, 3), len(rec["text"])])
            rec["total"] = round(time.time() - t0, 3)
            return rec
        except FatalError:
            raise
        except Exception as e:                      # timeouts, 429, 5xx, broken streams
            last_err = str(e)[:500]
            time.sleep(min(60, 6 * 2 ** (attempt - 1)) * (0.5 + random.random()))
    rec = {"text": "", "reasoning_chars": 0, "ttft": None, "first_any": None, "total": None, "chunks": [],
           "usage": {}, "attempts": retries, "error": last_err, "finish_reason": None}
    return rec


def chat_once(base, key, model, messages, max_tokens=MAX_OUT, temperature=0.0, timeout=300, retries=3):
    """non-streaming call (judge) -> (text, usage)"""
    last = None
    for attempt in range(retries):
        try:
            LIMITER.wait()
            r = requests.post(base + "/chat/completions", timeout=timeout, headers={"Authorization": "Bearer " + key},
                              json={"model": model, "messages": messages, "max_tokens": min(max_tokens, MAX_OUT),
                                    "temperature": temperature, **extra_params(model)})
            if r.status_code != 200:
                body = r.text[:400].replace(key, "<KEY>")
                if _fatal(r.status_code, body):
                    raise FatalError(body)
                if r.status_code == 429:
                    LIMITER.hit_429()
                raise RuntimeError("HTTP %d %s" % (r.status_code, body))
            d = r.json()
            return d["choices"][0]["message"].get("content") or "", d.get("usage") or {}
        except FatalError:
            raise
        except Exception as e:
            last = str(e)[:400]
            time.sleep(min(60, 5 * 2 ** attempt) * (0.5 + random.random()))
    raise RuntimeError(last)


class RateLimiter:
    """sliding-window requests-per-minute budget shared by all threads of this process; an HTTP 429 means another
    client used the same key, so the caller pauses 5 s instead of the rate being lowered"""

    def __init__(self, rpm):
        self.rpm = float(rpm)
        self.lock, self.times, self.n429 = threading.Lock(), [], 0

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                self.times = [t for t in self.times if now - t < 60]
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
                wait = 60 - (now - self.times[0]) + 0.02
            time.sleep(max(0.02, min(wait, 1.0)))

    def hit_429(self):
        with self.lock:
            self.n429 += 1
        time.sleep(5)


LIMITER = RateLimiter(float(os.environ.get("COPLAY_RPM", "500")))
