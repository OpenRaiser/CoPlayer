#!/usr/bin/env python3
"""Launch one CoPlayer GRPO run (verl) on an existing Ray cluster.

  python training/rl/launch.py --track reactive  --model saves/coplayer_sft [key=value ...]
  python training/rl/launch.py --track proactive --model saves/coplayer_sft [key=value ...]

Config = verl's ppo_trainer + configs/grpo_base.yaml + configs/<track>.yaml + command-line overrides (dotlist).
Needs: verl importable (pip install -e <verl> or VERL_DIR on PYTHONPATH), the parquet files from build_rl_data.py and a
reward judge server (serve_judge.sh). Environment passed to every Ray worker:
  COPLAY_REWARD_URL / COPLAY_REWARD_API_KEY / COPLAY_REWARD_MODEL   reward judge (required)
  COPLAY_DATA, COPLAY_RL_DATA (default $COPLAY_DATA/training/rl), COPLAY_REWARD_CACHE, COPLAY_RL_LOGS, FFMPEG
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "coplaybench"))


def verl_config_dir():
    spec = importlib.util.find_spec("verl")
    if spec is None:
        raise SystemExit("verl is not importable: pip install -e <verl checkout> (commit 5b79827b) or add it to PYTHONPATH")
    return str(Path(spec.submodule_search_locations[0]) / "trainer" / "config")


def build_config(track, model, data_dir, output_dir, overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=verl_config_dir(), version_base=None):
        base = compose(config_name="ppo_trainer")
    config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    config = OmegaConf.merge(config, OmegaConf.load(HERE / "configs" / "grpo_base.yaml"), OmegaConf.load(HERE / "configs" / (track + ".yaml")))
    output_dir = Path(output_dir).resolve()
    config.data.train_files = str(Path(data_dir) / ("train_%s.parquet" % track))
    config.data.val_files = str(Path(data_dir) / ("dev_%s.parquet" % track))
    config.actor_rollout_ref.model.path = str(Path(model).resolve()) if Path(model).exists() else model
    config.reward.custom_reward_function.path = str(HERE / ("reward_%s.py" % track))
    config.trainer.default_local_dir = str(output_dir / "checkpoints")
    config.trainer.rollout_data_dir = str(output_dir / "rollouts")
    config.trainer.validation_data_dir = str(output_dir / "validation")
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=("reactive", "proactive"), required=True)
    ap.add_argument("--model", required=True, help="the SFT checkpoint (HF format)")
    ap.add_argument("--data-dir", default=None, help="default $COPLAY_RL_DATA or $COPLAY_DATA/training/rl")
    ap.add_argument("--output-dir", default=None, help="default saves/coplayer_<track>")
    ap.add_argument("--print-config", action="store_true", help="print the composed config and exit")
    ap.add_argument("overrides", nargs="*", help="verl config overrides, e.g. trainer.nnodes=2")
    a = ap.parse_args()
    from omegaconf import OmegaConf

    data = os.path.abspath(os.environ.get("COPLAY_DATA", "data/CoPlayBench"))
    data_dir = os.path.abspath(a.data_dir or os.environ.get("COPLAY_RL_DATA") or os.path.join(data, "training", "rl"))
    output_dir = a.output_dir or "saves/coplayer_%s" % a.track
    config = build_config(a.track, a.model, data_dir, output_dir, a.overrides)
    if a.print_config:
        print(OmegaConf.to_yaml(config))
        return
    if not os.environ.get("COPLAY_REWARD_URL"):
        raise SystemExit("set COPLAY_REWARD_URL (and COPLAY_REWARD_API_KEY) to the reward judge server, see serve_judge.sh")

    import ray
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    from verl.trainer.main_ppo import run_ppo
    from verl.utils.config import validate_config
    from trainer_hooks import CoPlayTaskRunner

    validate_config(config, use_reference_policy=True, use_critic=False)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "resolved_config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=1))
    env = {"COPLAY_RL_TRACK": a.track, "COPLAY_DATA": data, "COPLAY_RL_DATA": data_dir,
           "COPLAY_RL_LOGS": str(Path(output_dir).resolve() / "logs"),
           "PYTHONPATH": os.pathsep.join([str(HERE), str(REPO / "coplaybench")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    for name in ("COPLAY_REWARD_URL", "COPLAY_REWARD_API_KEY", "COPLAY_REWARD_MODEL", "COPLAY_REWARD_CACHE", "FFMPEG"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    runtime = OmegaConf.to_container(OmegaConf.merge(get_ppo_ray_runtime_env(config), config.ray_kwargs.ray_init.runtime_env))
    runtime.setdefault("env_vars", {}).update(env)
    ray.init(address=config.ray_kwargs.ray_init.address, runtime_env=runtime)
    print("CoPlayer %s GRPO: %d nodes x %d GPUs, %d steps -> %s" % (a.track, config.trainer.nnodes, config.trainer.n_gpus_per_node,
                                                                 config.trainer.total_training_steps, output_dir), flush=True)
    try:
        run_ppo(config, CoPlayTaskRunner)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
