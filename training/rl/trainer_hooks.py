"""verl hooks: GRPO advantage handling of the two CoPlayer policies, installed by a custom TaskRunner.

Both tracks
  * groups (8 samples of one prompt) that did not come back complete are masked out of the update
  * groups whose rewards have a standard deviation below 0.01 carry no signal: their advantages are zeroed
Proactive track only
  * a must-speak group in which every sample is valid and silent would also be zeroed by GRPO, so the miss penalty
    would vanish exactly where it is needed. Such a group instead gets an advantage of -RESCUE on the tokens of the
    `false` decision and zero everywhere else.
Per-step statistics are appended to $COPLAY_RL_LOGS/health.jsonl.
"""
from collections import deque
import json
import os
from pathlib import Path
import re

import numpy as np
import ray
import torch
from tensordict import TensorDict

RESCUE = .30            # "recall" profile
GROUP = 8               # rollout.n


def unwrap(value):
    return getattr(value, "data", value)


def batch_info(data, track):
    labels, speaks, valid, useful = [], [], [], []
    for i in range(len(data)):
        item = data[i]
        meta = json.loads(unwrap(item["reward_model"])["ground_truth"])
        info = unwrap(item["extra_fields"]).get("reward_extra_info", {})
        label = meta.get("label", "reactive")
        if (track == "proactive") != (label != "reactive"):
            raise ValueError("Batch contains rows of the other track")
        if track == "proactive" and not all(name in info for name in ("speak", "format_valid")):
            raise ValueError("Missing proactive decision metadata")
        labels.append(label)
        speaks.append(bool(info.get("speak", 0)))
        valid.append(bool(info.get("format_valid", 0)))
        useful.append(bool(info.get("offwin_useful", 0)))
    padded = data.to_padded_tensor() if hasattr(data, "to_padded_tensor") else data
    return padded.get("uid").tolist(), labels, speaks, valid, useful


def plan_groups(uids, labels, speaks, valid, scores, track, keys=None, padding=None):
    """-> (complete groups, all-silent must-speak groups, low-variance groups, excluded sample indices)"""
    padding = padding if padding is not None else [False] * len(uids)
    groups, excluded = {}, {i for i, value in enumerate(padding) if value}
    for i, uid in enumerate(uids):
        if i not in excluded:
            groups.setdefault(uid, []).append(i)
    silent, low, incomplete = set(), set(), set()
    for uid, indices in groups.items():
        if len({labels[i] for i in indices}) != 1:
            raise ValueError("A group contains mixed labels")
        if keys is not None:
            sessions = [keys[i].rsplit("_", 2) for i in indices]
            if any(len(parts) != 3 or parts[0] != str(uid) or parts[2] != "0" for parts in sessions):
                raise ValueError("Expected single-turn rollouts")
            complete = len(indices) == GROUP and {int(parts[1]) for parts in sessions} == set(range(GROUP))
        else:
            complete = len(indices) == GROUP
        if not complete:
            incomplete.add(uid)
            excluded.update(indices)
            continue
        if track == "proactive" and labels[indices[0]] == "must_speak" and all(valid[i] and not speaks[i] for i in indices):
            silent.add(uid)
        elif np.std([scores[i] for i in indices]) < .01:
            low.add(uid)
    return {uid: indices for uid, indices in groups.items() if uid not in incomplete}, silent, low, excluded


def false_mask(tokenizer, tokens, response_mask):
    """mask of the tokens that spell the `false` of "speak": false in one response"""
    ids = tokens.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=False)
    span = re.search(r'"speak"\s*:\s*(false)', text)
    if span is None:
        raise ValueError("Silent response has no false decision field")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != ids:
        raise ValueError("False decision token mapping is not lossless")
    left, right = span.span(1)
    mask = torch.tensor([start < right and end > left for start, end in encoded["offset_mapping"]], device=response_mask.device)
    mask = mask & response_mask.bool()
    if not mask.any():
        raise ValueError("Silent decision has no valid tokens")
    return mask


def mask_excluded(tq, batch, data, excluded):
    masks = [row.clone() for row in data["response_mask"].unbind()]
    for i in excluded:
        masks[i].zero_()
    mask = torch.nested.as_nested_tensor(masks, layout=torch.jagged)
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id,
                    fields=TensorDict({"response_mask": mask, "loss_mask": mask.clone()}, batch_size=len(batch)))


def install(track):
    import transfer_queue as tq
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    original = PPOTrainer._compute_advantage
    if getattr(original, "coplay_hook", False):
        return
    directory = Path(os.environ.get("COPLAY_RL_LOGS", "logs/rl")) / track
    directory.mkdir(parents=True, exist_ok=True)
    recent = deque(maxlen=10)

    def compute(self, batch, metrics):
        if track == "reactive" and batch.partition_id != "train":
            return original(self, batch, metrics)
        fields = ["uid", "rm_scores", "reward_model", "extra_fields", "response_mask"] + (["responses"] if track == "proactive" else [])
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        uids, labels, speaks, valid, _ = batch_info(data, track)
        scores = [float(row.sum()) for row in data["rm_scores"].unbind()]
        padding = [tag.get("is_padding", False) for tag in batch.tags]
        groups, silent, low, excluded = plan_groups(uids, labels, speaks, valid, scores, track, batch.keys, padding)
        if not groups:
            raise RuntimeError("Batch has no complete groups; refusing an empty optimizer update")
        if excluded:
            mask_excluded(tq, batch, data, excluded)
        result = original(self, batch, metrics)
        decisions = {}
        if silent or low or excluded:
            state = tq.kv_batch_get(keys=result.keys, partition_id=result.partition_id,
                                    select_fields=["advantages", "returns", "response_mask"])
            masks = list(state["response_mask"].unbind())
            if track == "proactive":
                decisions = {i: false_mask(self.tokenizer, data["responses"][i], masks[i])
                             for i, uid in enumerate(uids) if uid in silent and i not in excluded}
            output = {}
            for name in ("advantages", "returns"):
                values = [row.clone() for row in state[name].unbind()]
                for i, uid in enumerate(uids):
                    if uid in low or uid in silent or i in excluded:
                        values[i].zero_()
                    if i in decisions:
                        values[i][decisions[i]] = -RESCUE
                output[name] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
            result = tq.kv_batch_put(keys=result.keys, partition_id=result.partition_id,
                                     fields=TensorDict(output, batch_size=len(result)))
        retained = [i for indices in groups.values() for i in indices]
        positive = [i for i in retained if labels[i] == "must_speak"]
        negative = [i for i in retained if labels[i] not in ("must_speak", "reactive")]
        record = {"step": self.global_steps, "track": track, "groups": len(groups), "low_variance_groups": len(low),
                  "excluded_samples": len(excluded) - sum(padding), "reward_mean": float(np.mean([scores[i] for i in retained])),
                  "format_valid": float(np.mean([valid[i] for i in retained]))}
        if track == "proactive":
            record.update(must_all_silent=len(silent) / max(1, len({uids[i] for i in positive})),
                          must_speak=float(np.mean([speaks[i] for i in positive])) if positive else 0.,
                          negative_speak=float(np.mean([speaks[i] for i in negative])) if negative else 0.,
                          decision_rescue_samples=len(decisions))
        recent.append(record)
        metrics.update({"coplay/" + k: v for k, v in record.items() if isinstance(v, (int, float))})
        with (directory / "health.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return result

    compute.coplay_hook = True
    PPOTrainer._compute_advantage = compute


@ray.remote
class CoPlayTaskRunner:
    """verl.trainer.main_ppo.run_ppo task runner with the CoPlayer hooks (track from COPLAY_RL_TRACK)"""

    def run(self, config):
        import transfer_queue as tq
        from omegaconf import OmegaConf
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ, get_trainer_cls
        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        install(os.environ["COPLAY_RL_TRACK"])
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer = None
        success = False
        try:
            trainer = get_trainer_cls(config.trainer.v1.trainer_mode)(config=config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(config=config, llm_client=trainer.get_llm_client(),
                                                teacher_client=trainer.get_teacher_client(),
                                                reward_loop_worker_handles=trainer.get_reward_handles())
            trainer.fit(manager)
            success = True
        finally:
            try:
                logger = getattr(trainer, "logger", None)
                if logger is not None:
                    logger.finish(exit_code=0 if success else 1)
            finally:
                tq.close()
