"""Value-guided candidate acquisition for the RoboResample DiT policy."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .resample_v2 import ActiveValueAcquisition, DoubleQValueModel


def low_dim_state(data: Dict[str, torch.Tensor], time_index: int = -1) -> torch.Tensor:
    """Build the Cal-QL state vector at a specified observation timestep."""
    obs = data["obs"]
    parts = []
    for key in ("gripper_states", "joint_states", "ee_states"):
        if key in obs:
            value = obs[key]
            parts.append(value[:, time_index] if value.ndim >= 3 else value)
    if not parts:
        raise KeyError("No low-dimensional state found in observation")
    return torch.cat(parts, dim=-1)


class ActionCoverageMemory:
    """Bounded action-chunk memory used to estimate behavioral novelty."""

    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self._chunks = []

    @torch.no_grad()
    def score(self, chunks: torch.Tensor) -> torch.Tensor:
        flat = chunks.flatten(start_dim=2)
        if not self._chunks:
            return torch.ones(flat.shape[:2], device=flat.device, dtype=flat.dtype)
        memory = torch.stack(self._chunks).to(device=flat.device, dtype=flat.dtype)
        return torch.cdist(flat, memory.unsqueeze(0).expand(flat.shape[0], -1, -1)).min(dim=-1).values

    def update(self, chunk: torch.Tensor) -> None:
        for item in chunk.detach().flatten(start_dim=1).cpu():
            self._chunks.append(item)
        if len(self._chunks) > self.capacity:
            del self._chunks[: len(self._chunks) - self.capacity]

    def reset(self) -> None:
        # Coverage is intentionally retained across episodes in a collection run.
        pass


class ValueGuidedActionSampler:
    """Sample DiT chunks, score them with a chunk critic, and choose one."""

    def __init__(
        self,
        policy,
        critics,
        acquisition: ActiveValueAcquisition,
        num_candidates: int = 8,
        execution_index: int = 0,
        coverage_memory: Optional[ActionCoverageMemory] = None,
    ):
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least one")
        self.policy = policy
        self.value_model = DoubleQValueModel(critics, action_index=None)
        self.acquisition = acquisition
        self.num_candidates = int(num_candidates)
        self.execution_index = int(execution_index)
        self.coverage_memory = coverage_memory or ActionCoverageMemory()
        self.last_selection_metadata = None
        self.policy.eval()
        for critic in self.value_model.critics:
            critic.eval()

    def reset(self):
        self.policy.reset()
        self.last_selection_metadata = None

    @torch.no_grad()
    def select_action(self, data, force_trigger=None):
        chunks = self.policy.sample_candidates(data, self.num_candidates)
        if chunks.shape[0] != 1:
            raise NotImplementedError(
                "Value-guided collection currently requires one environment per sampler"
            )
        state = low_dim_state(data)
        q_base, q_std = self.value_model.score(state, chunks)
        coverage = self.coverage_memory.score(chunks)
        if force_trigger is not None:
            force_trigger = torch.as_tensor(
                [force_trigger], device=chunks.device, dtype=torch.bool
            )
        result = self.acquisition.select(
            q_base, q_std, coverage=coverage, force_trigger=force_trigger
        )
        index = int(result.selected_index[0].item())
        selected_chunk = chunks[:, index]
        self.coverage_memory.update(selected_chunk)
        executed = selected_chunk[0, self.execution_index]

        self.last_selection_metadata = {
            "candidate_actions": chunks[0].detach().cpu().numpy(),
            "candidate_q_mean": self.value_model.last_head_mean[0].detach().cpu().numpy(),
            "candidate_q_base": q_base[0].detach().cpu().numpy(),
            "candidate_q_std": q_std[0].detach().cpu().numpy(),
            "candidate_q_lcb": result.q_lcb[0].detach().cpu().numpy(),
            "candidate_coverage": coverage[0].detach().cpu().numpy(),
            "candidate_acquisition": result.acquisition_score[0].detach().cpu().numpy(),
            "selected_index": np.int64(index),
            "triggered": np.bool_(result.triggered[0].item()),
            "trigger_code": np.int64(result.trigger_code[0].item()),
            "candidate_count": np.int64(self.num_candidates),
            "execution_index": np.int64(self.execution_index),
        }
        return executed.detach().cpu().numpy(), bool(result.triggered[0].item())
