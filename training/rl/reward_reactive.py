"""Reactive RL reward (verl custom_reward_function: compute_score).

r = clip(0.15 r_fmt + 0.55 r_con + 0.35 r_emo - p, -0.5, 1)
  r_con = 0.45 coverage + 0.35 sound advice + 0.20 correct-fact share   (coverage / facts default 0.6; without an advice
          verdict the coverage and fact weights are rescaled to sum to one)
  r_emo = 0.35 emotion perceived + 0.65 support / 4
  r_fmt = 0.4 well-formed + 0.35 word limit (1 / 0.5 up to 1.5x / 0) + 0.25 advice placed early
  p     = 0.5 must-not violation + 0.3 min(1, wrong claims / 2) + 0.2 quoted engine value
A violation, a wrong claim or fact, or an engine value caps the reward at -0.4; an empty reply gets -0.5.
needs_strategy for the graded reply comes from grading the reference reply first (as in the benchmark).
"""
import json
import re

from reward_common import WORDS, engine_numbers, judge, reactive_item, score_quality  # also puts coplaybench/ on sys.path
import judge_reactive  # noqa: E402


def position_score(reply, advice_end):
    if not advice_end:
        return .5
    offset = reply.find(advice_end)
    if offset < 0:
        return .5
    relative = (offset + len(advice_end)) / max(1, len(reply))
    return 1. if relative <= .5 else 1.5 - relative


def reactive_score(grade, reply, maximum):
    content, style, _, violation, wrong_claims, wrong_facts = score_quality(grade)
    words = len(WORDS.findall(reply))
    maximum = maximum or 25
    basic = bool(reply.strip()) and not re.match(r'^(AI:|Assistant:|As an AI|["\'])', reply.strip(), re.I)
    length = 1. if words <= maximum else .5 if words <= 1.5 * maximum else 0.
    position = position_score(reply, grade.get("advice_end"))
    formatting = .4 * float(basic) + .35 * length + .25 * position
    engine = engine_numbers(reply)
    penalty = .5 * violation + .3 * min(1., wrong_claims / 2) + .2 * engine
    score = .15 * formatting + .55 * content + .35 * style - penalty
    if violation or wrong_claims or wrong_facts or engine:
        score = min(score, -.4)
    return max(-.5, min(1., score)), position, violation, wrong_claims


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    meta = json.loads(ground_truth)
    if data_source not in ("coplay_in_play", "coplay_aftermath"):
        raise ValueError("The reactive reward only accepts reactive rows")
    reply = solution_str.strip()
    result = {"score": -.5, "acc": 0., "format_valid": 0., "speak": 0., "offwin_useful": 0., "position_score": 0.,
              "violation": 0., "wrong_claims": 0.}
    if not reply:
        return result
    item = reactive_item(meta)
    reference = judge(judge_reactive.build_messages(item, meta["reference_response"], True), "reactive")
    needs = reference.get("strategy_ok") is True and bool(reference.get("advice_end"))
    grade = judge(judge_reactive.build_messages(item, reply, needs), "reactive")
    score, position, violation, wrong = reactive_score(grade, reply, meta.get("max_words_en"))
    result.update(score=score, acc=float(grade.get("strategy_ok") is True), format_valid=1., position_score=position,
                  violation=float(violation), wrong_claims=float(wrong))
    return result
