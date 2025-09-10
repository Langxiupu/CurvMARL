import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch

from gnn.gat import ObservationBuilder, GATBackbone, PolicyHead
from marl.env import EnvConfig, MultiSatEnv
from marl.mappo import MAPPO, MAPPOConfig, CentralisedCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    env_cfg = EnvConfig(
        num_sats=720,
        num_steps=args.steps,
        altitude_km=570.0,
        inclination_deg=70.0,
    )
    env = MultiSatEnv(env_cfg)
    n_agents = env_cfg.num_sats

    cfg = MAPPOConfig(lr_actor=1e-4, lr_critic=1e-4)
    algo = MAPPO(env, n_agents, cfg)

    # replicate network setup from training
    algo.ob_builder = ObservationBuilder(L=2)
    d_x, d_e = 6, 6
    algo.actor_backbone = GATBackbone(d_x=d_x, d_e=d_e, L=2)
    emb_dim = algo.actor_backbone.layers[-1].W.out_features
    algo.policy_head = PolicyHead(d_in=emb_dim, d_e=d_e)
    algo.critic = CentralisedCritic(d_emb=emb_dim, n_agents=n_agents)
    algo.opt_actor = torch.optim.Adam(algo.actor_backbone.parameters(), lr=cfg.lr_actor)
    algo.opt_critic = torch.optim.Adam(algo.critic.parameters(), lr=cfg.lr_critic)

    ckpt = torch.load(args.model, map_location="cpu")
    algo.actor_backbone.load_state_dict(ckpt["actor"])
    algo.policy_head.load_state_dict(ckpt["policy"])
    algo.critic.load_state_dict(ckpt.get("critic", {}))

    _, metrics = algo.rollout(args.steps)

    plrs = [m.get("packet_loss_rate", 0.0) for m in metrics]
    delays_ms = [m.get("avg_delivery_time_s", 0.0) * 1000.0 for m in metrics]
    throughputs = [m.get("system_throughput_Mbps", 0.0) for m in metrics]

    avg_plr = sum(plrs) / len(plrs) if plrs else 0.0
    avg_delay = sum(delays_ms) / len(delays_ms) if delays_ms else 0.0
    avg_thr = sum(throughputs) / len(throughputs) if throughputs else 0.0

    print(f"Average packet loss rate over {len(plrs)} steps: {avg_plr:.2f}%")
    print(f"Average end-to-end delay over {len(delays_ms)} steps: {avg_delay:.3f} ms")
    print(f"Average system throughput over {len(throughputs)} steps: {avg_thr:.3f} Mbps")


if __name__ == "__main__":
    main()
