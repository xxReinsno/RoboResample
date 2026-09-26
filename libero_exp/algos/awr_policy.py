"""Advantage-weighted supervised fine-tuning for RESample V2."""

import os

import torch

from calql.model import Critic, StateValueNetwork
from .bc_policy import BC_Policy
from ..models.resample_v2 import compute_awr_weights, lower_confidence_bound, mixed_supervised_loss
from ..models.value_guided_sampler import low_dim_state


class AWR_Policy(BC_Policy):
    """Keep the native DiT loss and reweight only executed rollout data."""

    def __init__(self, cfg, inference=False, device="cuda"):
        super().__init__(cfg, inference=inference, device=device)
        if inference:
            return
        awr = cfg.algo.awr
        action_dim = int(awr.action_dim) * int(awr.action_horizon)
        self.awr_critic = Critic(int(awr.state_dim), action_dim).to(self.device)
        critic_path = os.path.expanduser(awr.critic_checkpoint_path)
        self.awr_critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.awr_critic.eval().requires_grad_(False)

        self.value_model = None
        if awr.baseline == "value":
            self.value_model = StateValueNetwork(
                int(awr.state_dim), int(awr.value_hidden_dim)
            ).to(self.device)
            value_path = os.path.expanduser(awr.value_checkpoint_path)
            value_state = torch.load(value_path, map_location=self.device)
            if isinstance(value_state, dict) and "value" in value_state:
                value_state = value_state["value"]
            self.value_model.load_state_dict(value_state)
            self.value_model.eval().requires_grad_(False)
        elif awr.baseline != "batch_mean":
            raise ValueError(f"Unknown AWR baseline: {awr.baseline}")

    def _critic_values(self, data):
        awr = self.cfg.algo.awr
        # In future-chunk training the first ``frame_stack`` observations end
        # at s_t; later observations only exist because SequenceDataset also
        # fetches the future action window.
        state_index = (
            int(self.cfg.data.frame_stack) - 1
            if getattr(self.cfg.policy, "action_chunk_mode", "aligned") == "future"
            else -1
        )
        state = low_dim_state(data, time_index=state_index)
        action_chunk = data["actions"][:, : int(awr.action_horizon)]
        if action_chunk.shape[1] != int(awr.action_horizon):
            raise ValueError("Batch action sequence is shorter than action_horizon")
        action = action_chunk.flatten(start_dim=1)
        with torch.no_grad():
            q1, q2 = self.awr_critic(state, action)
            q_pair = torch.stack([q1.squeeze(-1), q2.squeeze(-1)], dim=0)
            q_mean = q_pair.min(dim=0).values
            q_std = q_pair.std(dim=0, unbiased=False)
            q_lcb = lower_confidence_bound(q_mean, q_std, awr.kappa)
            value = (
                self.value_model(state).squeeze(-1)
                if self.value_model is not None
                else q_lcb.mean().expand_as(q_lcb)
            )
        return q_lcb, value

    def compute_awr_loss(self, data, augmentation=None):
        if self.cfg.policy.policy_type != "BCDPPolicy":
            raise NotImplementedError(
                "AWR_Policy currently wires the DiT adapter only; other adapters must "
                "implement per_sample_loss before use."
            )
        per_sample, processed = self.model.per_sample_loss(data, augmentation=augmentation)
        is_rollout = processed.get(
            "is_rollout",
            torch.zeros(per_sample.shape[0], device=per_sample.device, dtype=torch.bool),
        ).bool()
        q_lcb, value = self._critic_values(processed)
        weighting = self.cfg.algo.awr.weighting
        if weighting == "bc":
            weights = torch.ones_like(q_lcb)
            advantage = (q_lcb - value).detach()
        else:
            weight_baseline = value if weighting == "awr" else torch.zeros_like(value)
            if weighting not in {"awr", "rwr"}:
                raise ValueError(f"Unknown weighting mode: {weighting}")
            weights, advantage = compute_awr_weights(
                q_lcb,
                weight_baseline,
                beta=self.cfg.algo.awr.beta,
                min_weight=self.cfg.algo.awr.min_weight,
                max_weight=self.cfg.algo.awr.max_weight,
            )
        # Demo weights are deliberately ignored by mixed_supervised_loss.
        total, demo_loss, rollout_loss = mixed_supervised_loss(
            per_sample,
            is_rollout,
            rollout_weights=weights,
            rollout_scale=self.cfg.algo.awr.rollout_scale,
        )
        rollout_weights = weights[is_rollout]
        if rollout_weights.numel():
            ess = rollout_weights.sum().square() / rollout_weights.square().sum().clamp_min(1e-8)
        else:
            ess = torch.zeros((), device=per_sample.device)
        stats = {
            "loss": total,
            "bc_loss": demo_loss,
            "rollout_loss": rollout_loss,
            "advantage_mean": advantage[is_rollout].mean() if is_rollout.any() else advantage.mean() * 0,
            "weight_mean": rollout_weights.mean() if rollout_weights.numel() else weights.mean() * 0,
            "weight_ess": ess,
            "weight_clip_low_fraction": (
                (rollout_weights <= self.cfg.algo.awr.min_weight).float().mean()
                if rollout_weights.numel() else weights.mean() * 0
            ),
            "weight_clip_high_fraction": (
                (rollout_weights >= self.cfg.algo.awr.max_weight).float().mean()
                if rollout_weights.numel() else weights.mean() * 0
            ),
            "rollout_fraction": is_rollout.float().mean(),
        }
        return stats

    def forward_backward(self, data):
        stats = self.compute_awr_loss(data)
        self.optimizer.zero_grad()
        self.fabric.backward(stats["loss"])
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.cfg.train.grad_clip
        )
        self.optimizer.step()
        return {key: value.detach().item() for key, value in stats.items()}

    def compute_loss(self, data, augmentation=None):
        # Validation data contains demonstrations only, so report native BC loss.
        per_sample, _ = self.model.per_sample_loss(data, augmentation=augmentation)
        return per_sample.mean()
