"""Minimal MAPPO implementation for CurvMARL.

The training pipeline mirrors the three-stage design described in the
project documentation:

* **Graph preprocessing** – an optional rewiring step injects virtual
  edges into the physical topology.
* **Representation learning** – each agent builds an ``L``-hop ego
  network around itself and runs a GAT to obtain a local embedding.
* **Routing decision** – embeddings feed actor/critic heads under the
  CTDE (centralised training, decentralised execution) paradigm.

The code is intentionally lightweight and serves as a starting point for
further experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from gnn.gat import ObservationBuilder, GATBackbone, PolicyHead
from marl.env import MultiSatEnv

# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------


class RolloutBuffer:
    def __init__(self) -> None:
        self.embeddings: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    def add(
        self,
        emb: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        val: torch.Tensor,
        rew: float,
        done: bool,
    ) -> None:
        self.embeddings.append(emb)
        self.actions.append(act)
        self.logprobs.append(logp)
        self.values.append(val)
        self.rewards.append(torch.tensor(rew, dtype=torch.float32))
        self.dones.append(torch.tensor(float(done)))

    def clear(self) -> None:
        self.embeddings.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()


# ---------------------------------------------------------------------------
# GAE and PPO utilities
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation."""

    T, N = values.shape
    adv = torch.zeros_like(values)
    lastgaelam = torch.zeros(N, dtype=torch.float32)
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else torch.zeros(N)
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        lastgaelam = delta + gamma * lam * mask * lastgaelam
        adv[t] = lastgaelam
    ret = adv + values
    return adv, ret


# ---------------------------------------------------------------------------
# Actor / Critic networks
# ---------------------------------------------------------------------------


class CentralisedCritic(nn.Module):
    def __init__(self, d_emb: int, n_agents: int, d_global: int = 3, d_id: int = 16):
        super().__init__()
        self.id_emb = nn.Embedding(n_agents, d_id)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb + d_global + d_id, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, emb: torch.Tensor, global_feats: torch.Tensor) -> torch.Tensor:
        N = emb.size(0)
        graph_summary = emb.mean(dim=0)
        vals = []
        for i in range(N):
            id_vec = self.id_emb(torch.tensor(i, device=emb.device))
            x = torch.cat([graph_summary, global_feats, id_vec], dim=-1)
            vals.append(self.mlp(x))
        return torch.stack(vals).squeeze(-1)


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4
    epochs: int = 4
    minibatch: int = 1024


class MAPPO:
    def __init__(self, env: MultiSatEnv, n_agents: int, cfg: MAPPOConfig):
        self.env = env
        self.n_agents = n_agents
        self.cfg = cfg

        self.ob_builder = ObservationBuilder(L=3)
        d_x = 6
        d_e = 6
        self.actor_backbone = GATBackbone(d_x=d_x, d_e=d_e)
        self.policy_head = PolicyHead(d_in=self.actor_backbone.layers[-1].W.out_features // self.actor_backbone.layers[-1].heads, d_e=d_e)
        self.critic = CentralisedCritic(
            d_emb=self.actor_backbone.layers[-1].W.out_features
            * self.actor_backbone.layers[-1].heads,
            n_agents=n_agents,
        )
        self.opt_actor = torch.optim.Adam(self.actor_backbone.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    # ------------------------------------------------------------------
    def _forward(self, G_logic, G_phys) -> Tuple[torch.Tensor, Dict[int, List[int]], Dict[Tuple[int, int], torch.Tensor]]:
        """Run the actor backbone on each agent's local neighbourhood."""

        nodes = list(G_logic.nodes)
        idx_map = {n: i for i, n in enumerate(nodes)}

        embeddings: List[torch.Tensor] = []
        act_neighbors: Dict[int, List[int]] = {}
        edge_feats: Dict[Tuple[int, int], torch.Tensor] = {}

        for i in nodes:
            ego = self.ob_builder.ego_nodes(G_logic, i)
            X = self.ob_builder.node_features(G_logic, ego)
            edge_index, edge_attr = self.ob_builder.edge_arrays(G_logic, ego)
            emb_sub = self.actor_backbone(X, edge_index, edge_attr)
            center_idx = ego.index(i)
            embeddings.append(emb_sub[center_idx])

            nbrs = [
                j
                for j in G_phys.neighbors(i)
                if G_phys[i][j].get("cap_bps", 0.0) > 0.0
            ]
            act_neighbors[i] = nbrs
            for j in nbrs:
                e = G_phys[i][j]
                edge_feats[(idx_map[i], idx_map[j])] = torch.tensor(
                    [
                        e.get("cap_bps", 0.0) / 1e9,
                        e.get("R_tot_bps", 0.0)
                        / max(e.get("cap_bps", 1e-6), 1e-6),
                        e.get("p_loss", 0.0),
                        e.get("tprop_s", 0.0) * 1000.0,
                        0.0,
                        e.get("dist_km", 0.0) / 10000.0,
                    ],
                    dtype=torch.float32,
                )

        emb = torch.stack(embeddings)
        return emb, act_neighbors, edge_feats, idx_map

    # ------------------------------------------------------------------
    def sample_actions(
        self,
        emb: torch.Tensor,
        act_neighbors: Dict[int, List[int]],
        edge_feats: Dict[Tuple[int, int], torch.Tensor],
        idx_map: Dict[int, int],
    ) -> Tuple[Dict[int, int], torch.Tensor, torch.Tensor]:
        """Sample actions for all agents.

        Returns a dictionary mapping each agent to the chosen neighbour, a
        tensor of action indices (used for training) and a tensor of
        corresponding log-probabilities.
        """

        actions: Dict[int, int] = {}
        action_idx: List[int] = []
        logps: List[float] = []
        center_idx = [idx_map[i] for i in act_neighbors.keys()]
        nbrs = [[idx_map[j] for j in act_neighbors[i]] for i in act_neighbors.keys()]
        logits_list = self.policy_head(emb, center_idx, nbrs, edge_feats)
        for i, logits, nbr in zip(act_neighbors.keys(), logits_list, nbrs):
            if logits.numel() == 0:
                actions[i] = i
                action_idx.append(0)
                logps.append(0.0)
                continue
            dist = Categorical(logits=logits)
            a = dist.sample()
            actions[i] = act_neighbors[i][a.item()]
            action_idx.append(a.item())
            logps.append(dist.log_prob(a).item())
        return actions, torch.tensor(action_idx, dtype=torch.long), torch.tensor(logps, dtype=torch.float32)

    # ------------------------------------------------------------------
    def rollout(self, T: int, rewirer=None) -> RolloutBuffer:
        buf = RolloutBuffer()
        G_phys = self.env.reset()
        for t in range(T):
            if rewirer is None:
                G_logic = G_phys
            else:
                G_logic = rewirer.build_logic_topology(G_phys)
            emb, act_neighbors, edge_feats, idx_map = self._forward(G_logic, G_phys)
            act_dict, a_idx, logps = self.sample_actions(emb, act_neighbors, edge_feats, idx_map)
            global_feats = torch.tensor(
                [0.0, 0.0, 0.0], dtype=torch.float32
            )
            values = self.critic(emb, global_feats)
            G_phys, r, done, info = self.env.step(act_dict)
            buf.add(emb.detach(), a_idx.detach(), logps.detach(), values.detach(), r, done)
            if done:
                break
        return buf

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer) -> None:
        rewards = torch.stack(buf.rewards)
        values = torch.stack(buf.values)
        dones = torch.stack(buf.dones)
        adv, ret = compute_gae(rewards.unsqueeze(-1).expand_as(values), values, dones, self.cfg.gamma, self.cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T, N = adv.shape
        for epoch in range(self.cfg.epochs):
            idx = torch.randperm(T * N)
            for start in range(0, T * N, self.cfg.minibatch):
                end = start + self.cfg.minibatch
                mb_idx = idx[start:end]
                mb_adv = adv.view(-1)[mb_idx]
                mb_ret = ret.view(-1)[mb_idx]
                mb_logp_old = torch.stack(buf.logprobs).view(-1)[mb_idx]
                mb_actions = torch.stack(buf.actions).view(-1)[mb_idx]

                # For simplicity we reuse old logprobs as new ones (no recompute)
                ratio = torch.exp(mb_logp_old - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * mb_adv
                actor_loss = -(torch.min(surr1, surr2).mean())
                value_loss = nn.functional.mse_loss(mb_ret, mb_ret.detach())
                loss = actor_loss + self.cfg.vf_coef * value_loss

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_backbone.parameters(), 0.5)
                self.opt_actor.step()
                self.opt_critic.step()
