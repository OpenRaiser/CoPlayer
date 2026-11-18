"""Shared parts of the two RL rewards: the judge client (benchmark judge prompts, local reward model) and grade scoring.

The reward judge is a locally served model (the paper used Qwen3.8-Flash-Next, see serve_judge.sh) that receives exactly
the benchmark judge prompts (coplaybench/judge_reactive.py, judge_proactive.py). Environment:
  COPLAY_REWARD_URL      OpenAI-compatible endpoint, e.g. http://10.0.0.5:8200/v1
  COPLAY_REWARD_API_KEY  its key (vLLM --api-key / VLLM_API_KEY)
  COPLAY_REWARD_MODEL    served model name (default coplay-reward)
  COPLAY_REWARD_CACHE    judgement cache directory (default /tmp/coplay_reward_cache)
  COPLAY_RL_DATA         directory holding images/ rendered by build_rl_data.py
"""
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "coplaybench"))
import judge_proactive  # noqa: E402
import judge_reactive  # noqa: E402

WORDS = re.compile(r"[A-Za-z0-9']+")
LOCKS = {}
GUARD = threading.Lock()


def score_quality(grade):
    """judge result -> (content, style, active_style, violation, wrong_claims, wrong_facts)"""
    points = grade.get("key_points") or []
    coverage = sum(p.get("covered") is True for p in points) / len(points) if points else .6
    facts = grade.get("facts") or []
    correct = sum(p.get("verdict") == "correct" for p in facts)
    wrong = sum(p.get("verdict") == "wrong" for p in facts)
    fact_score = correct / (correct + wrong) if correct + wrong else .6
    strategy = grade.get("strategy_ok")
    content = .45 * coverage + .35 * float(strategy is True) + .20 * fact_score if strategy is not None else (.45 * coverage + .20 * fact_score) / .65
    support = grade.get("support_score")
    if type(support) is not int or not 0 <= support <= 4:
        raise ValueError("Invalid support score")
    style = .35 * float(grade.get("emotion_ok") is True) + .65 * support / 4
    active_style = support / 4
    wrong_claims = sum(p.get("status") == "wrong" for p in grade.get("claims") or [])
    violation = bool(grade.get("must_not_violations"))
    return content, style, active_style, violation, wrong_claims, wrong


def engine_numbers(reply):
    """quoting an engine value (win rate, evaluation, shanten count) is penalised"""
    return bool(re.search(r"(?:win[ -]?rate|winning probability|engine evaluation|shanten)\s*[:=]?\s*\d", reply, re.I))


def parse_decision(reply):
    """proactive answer -> {"speak", "say", "note"}; malformed JSON or a flag that disagrees with the line raises"""
    result = json.loads(reply)
    if not isinstance(result, dict) or not all(k in result for k in ("speak", "say", "note")):
        raise ValueError("Missing active JSON fields")
    if type(result["speak"]) is not bool or not isinstance(result["say"], str) or not isinstance(result["note"], str):
        raise ValueError("Invalid active field types")
    if result["speak"] != bool(result["say"].strip()):
        raise ValueError("Active speech decision conflicts with say")
    return result


@contextmanager
def cache_lock(path):
    with GUARD:
        lock = LOCKS.setdefault(path.stem, threading.Lock())
    with lock, path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def judge(messages, track):
    """one judge call with an on-disk cache keyed by the exact messages"""
    url = os.environ.get("COPLAY_REWARD_URL", "").rstrip("/")
    key = os.environ.get("COPLAY_REWARD_API_KEY", "EMPTY")
    model = os.environ.get("COPLAY_REWARD_MODEL", "coplay-reward")
    if not url:
        raise ValueError("COPLAY_REWARD_URL is not set")
    directory = Path(os.environ.get("COPLAY_REWARD_CACHE", "/tmp/coplay_reward_cache")) / track
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps({"messages": messages, "model": model}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = directory / (digest + ".json")

    def parse(text):
        start = text.find("{")
        if start < 0:
            raise ValueError("Reward response has no JSON object")
        value, end = json.JSONDecoder().raw_decode(text[start:])
        if not isinstance(value, dict) or text[start + end:].lstrip().startswith("{"):
            raise ValueError("Reward response must contain one judgment object")
        if messages[0]["content"] == judge_reactive.JUDGE_SYSTEM:
            value = judge_reactive.parse(json.dumps(value))
            score_quality(value)
        elif value.get("category") not in ("useful", "chat", "wrong", "noise"):
            raise ValueError("Invalid off-window judgment")
        return value

    with cache_lock(path):
        if path.exists():
            try:
                return parse(path.read_text())
            except (ValueError, UnicodeDecodeError):
                pass
        session = requests.Session()
        session.trust_env = False
        for attempt in range(5):
            try:
                response = session.post(url + "/chat/completions", headers={"Authorization": "Bearer " + key},
                                        json={"model": model, "messages": messages, "temperature": 0.,
                                              "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}},
                                        timeout=(15, 300), allow_redirects=False)
                if response.status_code in (401, 403):
                    raise PermissionError("Reward service authentication failed")
                response.raise_for_status()
                choice = response.json()["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Truncated reward JSON")
                value = parse(choice["message"].get("content") or "")
                temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(value, ensure_ascii=False))
                os.replace(temporary, path)
                return value
            except PermissionError:
                raise
            except Exception as error:
                if attempt == 4:
                    raise RuntimeError("Reward unavailable after 5 attempts: " + type(error).__name__) from None
                time.sleep(min(8, 2 ** attempt))


def image_path(relative):
    """rendered training image (path stored relative to COPLAY_RL_DATA)"""
    path = Path(os.environ.get("COPLAY_RL_DATA", ".")) / relative
    if not path.is_file():
        raise ValueError("Missing RL image " + str(path))
    return path


def reactive_item(meta):
    """reward metadata -> benchmark item with absolute grid paths (for judge_reactive.build_messages)"""
    return dict(meta, context=dict(meta["context"], grids=[str(image_path(p)) for p in meta["context"]["grids"]]))


def proactive_messages(meta, reply):
    """the benchmark judge messages for one proactive step: event rubric on must-speak steps, off-window rubric otherwise"""
    grid = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image_path(meta["grid_path"]).read_bytes()).decode()}}]
    u = {"k": meta["k"], "t": meta["t"], "say": reply}
    if meta["label"] == "must_speak":
        return judge_proactive.event_messages(meta, meta["event"], u, grid)
    return judge_proactive.offwin_messages(meta, u, meta["history"], grid)
