"""Proactive RL reward (verl custom_reward_function: compute_score) - the decision-first "recall" profile of the paper.

The model answers {"speak", "say", "note"}; the reward scores the decision first and the line second:
  malformed JSON or a flag that disagrees with the line                -1
  inside a required window (label must_speak)
    silent                                                             -0.80
    advice wrong or missing                                            -0.55
    sound advice, support < 3                                          0.35 + 0.05 c
    sound advice, support >= 3                                         0.85 + 0.05 c + 0.05 [support = 4]
    event needs no advice                                              -0.10 + 0.25 c + 0.65 support / 4
  outside every window (neg_after / neg_before / neg_far)
    silent 0.10 | useful 0.35 | chat -0.35 | noise -0.5 | wrong -0.8 | verbatim repeat of an earlier line -0.6
Inside a required window, a must-not violation, a wrong claim or fact, or a quoted engine value caps the step at -0.8;
the reward is clipped to [-1, 1]. c = content term of the judge grade (reward_common.score_quality).
The other part of the recipe (a negative advantage on the "false" tokens of an all-silent must-speak group) lives in
trainer_hooks.py.
"""
import json

from reward_common import engine_numbers, judge, parse_decision, proactive_messages, score_quality

PROFILE = {"miss": -.80, "strategy_fail": -.55, "a2_base": .35, "joint_base": .85}
LABELS = ("must_speak", "neg_after", "neg_before", "neg_far")


def measurements(meta, parsed, grade=None):
    """diagnostics logged by verl + the off-window part of the reward"""
    positive = meta["label"] == "must_speak"
    needs = positive and bool(meta["event"].get("needs_advice", True))
    result = {"score": -.8, "acc": 0., "format_valid": 1., "speak": float(parsed["speak"]),
              "must": float(positive), "needs_advice": float(needs), "strategy_success": 0., "joint_success": 0.,
              "support_score": 0., "offwin_useful": 0., "offwin_annoy": 0., "violation": 0.}
    if not parsed["speak"]:
        result["score"] = -.25 if positive else .15
        return result
    say = parsed["say"].strip()
    if not positive:
        repeated = any(say.casefold() == old.strip().casefold() for old in meta["history"])
        if repeated:
            result.update(score=-.6, offwin_annoy=1.)
            return result
        category = (grade or {}).get("category")
        if category not in ("useful", "chat", "wrong", "noise"):
            raise ValueError("Speaking samples need a validated judgment")
        result.update(score={"useful": .35, "chat": -.35, "noise": -.5, "wrong": -.8}[category],
                      offwin_useful=float(category == "useful"), offwin_annoy=float(category != "useful"))
        return result
    content, _, _, violation, claims, wrong = score_quality(grade)
    support = grade["support_score"]
    strategy = grade.get("strategy_ok") is True
    joint = needs and strategy and support >= 3
    if needs:
        score = -.4 if not strategy else .30 + .05 * content if not joint else .85 + .10 * content + .05 * (support == 4)
    else:
        score = -.10 + .25 * content + .65 * support / 4
    unsafe = violation or claims or wrong or engine_numbers(say)
    if unsafe:
        score = min(score, -.6)
    result.update(score=max(-.8, min(1., score)), acc=float(joint if needs else support >= 3),
                  strategy_success=float(needs and strategy), joint_success=float(joint),
                  support_score=float(support), violation=float(bool(unsafe)))
    return result


def reward(meta, parsed, grade):
    result = measurements(meta, parsed, grade)
    if not parsed["speak"]:
        result["score"] = PROFILE["miss"] if meta["label"] == "must_speak" else .10
        return result
    if meta["label"] != "must_speak":
        return result
    needs = bool(meta["event"].get("needs_advice", True))
    content, _, _, violation, claims, wrong = score_quality(grade)
    support = grade["support_score"]
    if needs:
        if grade.get("strategy_ok") is not True:
            score = PROFILE["strategy_fail"]
        elif support < 3:
            score = PROFILE["a2_base"] + .05 * content
        else:
            score = PROFILE["joint_base"] + .05 * content + .05 * (support == 4)
    else:
        score = -.10 + .25 * content + .65 * support / 4
    if violation or claims or wrong or engine_numbers(parsed["say"]):
        score = min(score, -.8)
    result["score"] = max(-1., min(1., score))
    return result


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source != "coplay_proactive":
        raise ValueError("The proactive reward only accepts proactive rows")
    meta = json.loads(ground_truth)
    if meta.get("label") not in LABELS:
        raise ValueError("Proactive metadata without a step label")
    try:
        parsed = parse_decision(solution_str.strip())
    except (ValueError, TypeError):
        result = reward(meta, {"speak": False, "say": "", "note": ""}, None)
        result.update(score=-1., format_valid=0.)
        return result
    repeated = meta["label"] != "must_speak" and any(parsed["say"].strip().casefold() == old.strip().casefold() for old in meta["history"])
    grade = judge(proactive_messages(meta, parsed["say"]), "proactive") if parsed["speak"] and not repeated else None
    return reward(meta, parsed, grade)
