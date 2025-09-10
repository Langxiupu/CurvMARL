"""Generic MAPPO training script driven by JSON configuration files."""

import argparse
import json

from marl.env import EnvConfig, MultiSatEnv
from marl.mappo import MAPPO, MAPPOConfig

try:
    from curv.rewiring import CurvRewirer
except Exception:
    CurvRewirer = None


def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def build_env(cfg_dict):
    env_cfg = EnvConfig(**cfg_dict)
    return MultiSatEnv(env_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--updates", type=int, default=100000,
                        help="total training iterations")
    parser.add_argument("--rollout", type=int, default=200,
                        help="episode length in environment steps")
    args = parser.parse_args()

    cfg_all = load_config(args.config)
    cfg_root = cfg_all.get("mappo", cfg_all)
    env = build_env(cfg_root["env"])
    n_agents = cfg_root["env"]["num_sats"]
    algo = MAPPO(env, n_agents, MAPPOConfig())

    rew_cfg = cfg_root.get("rewiring", {"mode": "none"})
    rewirer = None
    if rew_cfg.get("mode") != "none" and CurvRewirer is not None:
        rewirer = CurvRewirer(**{k: v for k, v in rew_cfg.items() if k != "mode"})

    for _ in range(args.updates):
        buf, _ = algo.rollout(args.rollout, rewirer=rewirer)
        algo.update(buf)
    print("training completed")


if __name__ == "__main__":
    main()
