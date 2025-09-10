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

import numpy as np
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
        emb_dim = self.actor_backbone.layers[-1].W.out_features
        self.policy_head = PolicyHead(d_in=emb_dim, d_e=d_e)
        self.critic = CentralisedCritic(d_emb=emb_dim, n_agents=n_agents)
        self.opt_actor = torch.optim.Adam(self.actor_backbone.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

    # ------------------------------------------------------------------
    def _pick_active_nodes(self, G_phys) -> List[int]:
        """Return satellites currently carrying traffic.

        Nodes with either queued packets or non-zero throughput are
        considered active.  Satellites currently serving any ground station
        are also included so that newly generated flows receive an action.
        """

        active = {
            u
            for u, v, data in G_phys.edges(data=True)
            if data.get("phi_pkts", 0.0) > 0.0 or data.get("R_tot_bps", 0.0) > 0.0
        }

        try:  # private helper on ``MultiSatEnv``
            gs_map = self.env._associate_ground_stations()
            active.update(gs_map.values())
        except AttributeError:  # pragma: no cover - fallback if env changes
            pass

        return sorted(active)

    # ------------------------------------------------------------------
    def _forward(
        self,
        G_logic,
        G_phys,
        centers: Optional[Iterable[int]] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[int, List[int]],
        Dict[Tuple[int, int], torch.Tensor],
        Dict[int, int],
    ]:
        """Run the actor backbone on selected agents.

        Parameters
        ----------
        G_logic, G_phys:
            Logical and physical topologies.
        centers:
            Optional iterable of agent identifiers that require decisions.
            When ``None`` all nodes in ``G_logic`` are processed.  The method
            constructs a union of the ``L``-hop ego networks around the
            specified centres and runs the GAT once on this sub-graph.  This is
            considerably faster than invoking the backbone separately for every
            agent and enables inference on a subset of nodes (e.g. only those
            with traffic).
        """

        if centers is None:
            centers = list(G_logic.nodes)

        center_to_sub: List[Tuple[int, List[int]]] = []
        all_nodes: List[int] = []
        for c in centers:
            nodes = self.ob_builder.ego_nodes(G_logic, c)
            center_to_sub.append((c, nodes))
            all_nodes.extend(nodes)

        # Deduplicate nodes while preserving order so that ``idx_map`` remains
        # stable across calls when ``centers`` is identical.
        uniq_nodes: List[int] = list(dict.fromkeys(all_nodes))
        X = self.ob_builder.node_features(G_logic, uniq_nodes)
        edge_index, edge_attr = self.ob_builder.edge_arrays(G_logic, uniq_nodes)
        emb_sub = self.actor_backbone(X, edge_index, edge_attr)

        # Embed only the required nodes but return a tensor covering the full
        # constellation so that downstream code expecting a fixed number of
        # agents continues to work.
        emb_dim = emb_sub.size(1)
        emb = torch.zeros(self.n_agents, emb_dim, dtype=emb_sub.dtype, device=emb_sub.device)
        idx_sub = {n: i for i, n in enumerate(uniq_nodes)}
        for n, i in idx_sub.items():
            emb[n] = emb_sub[i]

        idx_map = {n: n for n in range(self.n_agents)}
        act_neighbors: Dict[int, List[int]] = {}
        edge_feats: Dict[Tuple[int, int], torch.Tensor] = {}

        for c, _ in center_to_sub:
            nbrs = [
                j
                for j in G_phys.neighbors(c)
                if G_phys[c][j].get("cap_bps", 0.0) > 0.0
            ]
            act_neighbors[c] = nbrs
            for j in nbrs:
                e = G_phys[c][j]
                edge_feats[(idx_map[c], idx_map[j])] = torch.tensor(
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

        # Prepare containers sized to the full set of agents.  This ensures
        # tensors stored in the rollout buffer have consistent dimensions
        # across timesteps even if only a subset of agents are active.
        actions: Dict[int, int] = {}
        a_idx_tensor = torch.zeros(self.n_agents, dtype=torch.long)
        logp_tensor = torch.zeros(self.n_agents, dtype=torch.float32)

        center_idx = [idx_map[i] for i in act_neighbors.keys()]
        nbrs = [[idx_map[j] for j in act_neighbors[i]] for i in act_neighbors.keys()]
        logits_list = self.policy_head(emb, center_idx, nbrs, edge_feats)

        for i, logits, nbr in zip(act_neighbors.keys(), logits_list, nbrs):
            agent_pos = idx_map[i]
            if logits.numel() == 0:
                # Agent has no valid moves; default to self-loop with zero log-prob
                actions[i] = i
                continue

            dist = Categorical(logits=logits)
            a = dist.sample()
            actions[i] = act_neighbors[i][a.item()]
            a_idx_tensor[agent_pos] = a
            logp_tensor[agent_pos] = dist.log_prob(a)

        return actions, a_idx_tensor, logp_tensor

    # ------------------------------------------------------------------
    def rollout(self, T: int, rewirer=None, reset: bool = True) -> Tuple[RolloutBuffer, List[Dict[str, float]]]:
        buf = RolloutBuffer()
        metrics: List[Dict[str, float]] = []
        if reset:
            G_phys = self.env.reset()
        else:
            G_phys = self.env.G_phys
        for t in range(T):
            if rewirer is None:
                G_logic = G_phys
            else:
                G_logic = rewirer.build_logic_topology(G_phys)
            centers = self._pick_active_nodes(G_phys)
            if centers:
                emb, act_neighbors, edge_feats, idx_map = self._forward(
                    G_logic, G_phys, centers
                )
            else:  # no active satellites; skip forward pass entirely
                d_emb = self.actor_backbone.layers[-1].W.out_features
                emb = torch.zeros(self.n_agents, d_emb)
                act_neighbors, edge_feats = {}, {}
                idx_map = {n: n for n in range(self.n_agents)}
            act_dict, a_idx, logps = self.sample_actions(emb, act_neighbors, edge_feats, idx_map)

            caps = [e.get("cap_bps", 0.0) for _, _, e in G_phys.edges(data=True)]
            utils = [
                e.get("R_tot_bps", 0.0) / max(e.get("cap_bps", 1e-9), 1e-9)
                for _, _, e in G_phys.edges(data=True)
            ]
            delays = [e.get("tprop_s", 0.0) for _, _, e in G_phys.edges(data=True)]
            avg_cap = float(np.mean(caps)) / 1e6 if caps else 0.0
            avg_util = float(np.mean(utils)) if utils else 0.0
            avg_delay = float(np.mean(delays)) * 1000.0 if delays else 0.0
            global_feats = torch.tensor([avg_cap, avg_delay, avg_util], dtype=torch.float32)

            values = (
                self.critic(emb.detach(), global_feats)
                if emb.numel() > 0
                else torch.zeros(0)
            )
            G_phys, r, done, info = self.env.step(act_dict)
            buf.add(emb.detach(), a_idx.detach(), logps, values, r, done)
            metrics.append(info.get("metrics", {}))
            if done:
                break
        return buf, metrics

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer) -> Tuple[float, float]:
        rewards = torch.stack(buf.rewards)
        values = torch.stack(buf.values)
        dones = torch.stack(buf.dones)
        adv, ret = compute_gae(
            rewards.unsqueeze(-1).expand_as(values),
            values.detach(),
            dones,
            self.cfg.gamma,
            self.cfg.lam,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        T, N = adv.shape
        pol_loss_total = 0.0
        val_loss_total = 0.0
        count = 0
        for epoch in range(self.cfg.epochs):
            idx = torch.randperm(T * N)
            for start in range(0, T * N, self.cfg.minibatch):
                end = start + self.cfg.minibatch
                mb_idx = idx[start:end]
                mb_adv = adv.view(-1)[mb_idx]
                mb_ret = ret.view(-1)[mb_idx]
                mb_logp = torch.stack(buf.logprobs).view(-1)[mb_idx]
                mb_vals = values.view(-1)[mb_idx]

                actor_loss = -(mb_logp * mb_adv.detach()).mean()
                value_loss = nn.functional.mse_loss(mb_vals, mb_ret.detach())
                loss = actor_loss + self.cfg.vf_coef * value_loss

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                # ``logp`` entries stored in the rollout buffer still retain
                # their computation graph from the sampling phase.  During PPO
                # updates we iterate over the same stored tensors multiple
                # times, which would normally free the graph after the first
                # backward call and raise ``RuntimeError: Trying to backward
                # through the graph a second time`` on subsequent iterations.
                # Retaining the graph allows multiple backward passes over the
                # same rollout data without reconstructing it every epoch.
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor_backbone.parameters(), 0.5)
                self.opt_actor.step()
                self.opt_critic.step()

                pol_loss_total += actor_loss.item()
                val_loss_total += value_loss.item()
                count += 1
        avg_pol = pol_loss_total / count if count else 0.0
        avg_val = val_loss_total / count if count else 0.0
        return avg_pol, avg_val
