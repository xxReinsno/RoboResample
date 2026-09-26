"""Policy-agnostic building blocks for RESample V2.

This module deliberately contains no LIBERO or OpenPI imports.  The acquisition
and AWR code can therefore be unit tested without a simulator and reused by
future policy adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch
from torch import nn
from calql.model import StateValueNetwork


class PolicyAdapter(Protocol):
    """Minimal interface required by value-guided acquisition and AWR."""

    def sample_candidates(self, observation, num_candidates: int) -> torch.Tensor:
        """Return action chunks shaped ``[B, K, H, A]``."""

    def per_sample_loss(self, batch) -> torch.Tensor:
        """Return the policy's native supervised loss shaped ``[B]``."""


class ValueModel(Protocol):
    """Minimal interface for scoring candidate action chunks."""

    def score(
        self, state: torch.Tensor, action_chunks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return conservative critic base and uncertainty shaped ``[B, K]``."""


class DoubleQValueModel:
    """Adapt a Double-Q critic (or an ensemble of them) to ``ValueModel``.

    The V2 critic consumes the complete flattened action chunk by default.
    ``action_index`` is retained only for single-step critic compatibility.
    Both heads of every critic are treated as ensemble members.
    """

    def __init__(self, critics, action_index: Optional[int] = None):
        if isinstance(critics, nn.Module):
            critics = [critics]
        if not critics:
            raise ValueError("At least one critic is required")
        self.critics = list(critics)
        self.action_index = action_index
        self.last_head_mean = None

    @torch.no_grad()
    def score(
        self, state: torch.Tensor, action_chunks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if action_chunks.ndim != 4:
            raise ValueError("action_chunks must have shape [B, K, H, A]")
        batch_size, num_candidates = action_chunks.shape[:2]
        if self.action_index is None:
            action = action_chunks.flatten(start_dim=2)
        else:
            action = action_chunks[:, :, self.action_index, :]
        expanded_state = state[:, None, :].expand(-1, num_candidates, -1)

        values = []
        for critic in self.critics:
            q1, q2 = critic(expanded_state, action)
            values.extend([q1.reshape(batch_size, num_candidates),
                           q2.reshape(batch_size, num_candidates)])
        stacked = torch.stack(values, dim=0)
        self.last_head_mean = stacked.mean(dim=0)
        # The first return is intentionally conservative despite the historical
        # ``q_mean`` name used by rollout metadata: clipped Double-Q supplies
        # the base value, while head/ensemble disagreement supplies uncertainty.
        return stacked.min(dim=0).values, stacked.std(dim=0, unbiased=False)


def lower_confidence_bound(
    q_mean: torch.Tensor, q_std: torch.Tensor, kappa: float
) -> torch.Tensor:
    if q_mean.shape != q_std.shape:
        raise ValueError("q_mean and q_std must have the same shape")
    return q_mean - float(kappa) * q_std


def _normalize_per_state(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if value.shape[-1] == 1:
        return torch.zeros_like(value)
    mean = value.mean(dim=-1, keepdim=True)
    std = value.std(dim=-1, keepdim=True, unbiased=False)
    return (value - mean) / (std + eps)


@dataclass(frozen=True)
class AcquisitionResult:
    selected_index: torch.Tensor
    q_lcb: torch.Tensor
    acquisition_score: torch.Tensor
    triggered: torch.Tensor
    trigger_code: torch.Tensor


class ActiveValueAcquisition:
    """Select candidates using value, critic uncertainty and coverage.

    Trigger codes are stable values written to rollout files:
    0=no trigger, 1=low value, 2=high uncertainty, 3=low coverage,
    4=forced/always.
    """

    MODES = {"active", "greedy_q", "random", "best_q"}

    def __init__(
        self,
        mode: str = "active",
        kappa: float = 1.0,
        uncertainty_weight: float = 0.5,
        coverage_weight: float = 0.5,
        min_safe_q: float = float("-inf"),
        trigger_q: float = float("-inf"),
        trigger_uncertainty: float = float("inf"),
        trigger_coverage: float = float("-inf"),
        always_trigger: bool = False,
    ):
        if mode not in self.MODES:
            raise ValueError(f"Unknown acquisition mode: {mode}")
        self.mode = mode
        self.kappa = kappa
        self.uncertainty_weight = uncertainty_weight
        self.coverage_weight = coverage_weight
        self.min_safe_q = min_safe_q
        self.trigger_q = trigger_q
        self.trigger_uncertainty = trigger_uncertainty
        self.trigger_coverage = trigger_coverage
        self.always_trigger = always_trigger

    def select(
        self,
        q_mean: torch.Tensor,
        q_std: torch.Tensor,
        coverage: Optional[torch.Tensor] = None,
        force_trigger: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> AcquisitionResult:
        if q_mean.ndim != 2 or q_mean.shape != q_std.shape:
            raise ValueError("critic scores must have shape [B, K]")
        batch_size, num_candidates = q_mean.shape
        if coverage is None:
            coverage = torch.zeros_like(q_mean)
        elif coverage.shape != q_mean.shape:
            raise ValueError("coverage must have shape [B, K]")

        q_lcb = lower_confidence_bound(q_mean, q_std, self.kappa)
        score = (
            _normalize_per_state(q_lcb)
            + self.uncertainty_weight * _normalize_per_state(q_std)
            + self.coverage_weight * _normalize_per_state(coverage)
        )

        safe = q_lcb >= self.min_safe_q
        safe_score = score.masked_fill(~safe, float("-inf"))
        no_safe_candidate = ~safe.any(dim=-1)
        active_index = safe_score.argmax(dim=-1)
        active_index = torch.where(no_safe_candidate, q_lcb.argmax(dim=-1), active_index)

        if self.mode in {"greedy_q", "best_q"}:
            selected = q_lcb.argmax(dim=-1)
        elif self.mode == "random":
            selected = torch.randint(
                num_candidates, (batch_size,), device=q_mean.device, generator=generator
            )
        else:
            selected = active_index

        trigger_code = torch.zeros(batch_size, dtype=torch.long, device=q_mean.device)
        triggered = torch.zeros(batch_size, dtype=torch.bool, device=q_mean.device)
        if self.always_trigger or self.mode != "active":
            triggered.fill_(True)
            trigger_code.fill_(4)
        else:
            low_q = q_lcb.max(dim=-1).values < self.trigger_q
            high_uncertainty = q_std.max(dim=-1).values > self.trigger_uncertainty
            low_coverage = coverage.max(dim=-1).values < self.trigger_coverage
            trigger_code = torch.where(low_q, torch.ones_like(trigger_code), trigger_code)
            trigger_code = torch.where(
                ~low_q & high_uncertainty,
                torch.full_like(trigger_code, 2),
                trigger_code,
            )
            trigger_code = torch.where(
                ~low_q & ~high_uncertainty & low_coverage,
                torch.full_like(trigger_code, 3),
                trigger_code,
            )
            triggered = trigger_code > 0

        if force_trigger is not None:
            force_trigger = force_trigger.to(device=q_mean.device, dtype=torch.bool)
            triggered = triggered | force_trigger
            trigger_code = torch.where(
                force_trigger, torch.full_like(trigger_code, 4), trigger_code
            )

        # Candidate zero is the unmodified policy fallback when no trigger fires.
        selected = torch.where(triggered, selected, torch.zeros_like(selected))
        return AcquisitionResult(selected, q_lcb, score, triggered, trigger_code)


def compute_awr_weights(
    q_lcb: torch.Tensor,
    value: torch.Tensor,
    beta: float = 1.0,
    min_weight: float = 0.05,
    max_weight: float = 20.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute detached, finite, clipped AWR weights.

    Normalization happens in :func:`mixed_supervised_loss` by dividing the
    weighted loss by the sum of weights. Keeping these values unnormalized
    preserves the configured clipping bounds for diagnostics.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    advantage = q_lcb.reshape(-1) - value.reshape(-1)
    raw = torch.exp(torch.clamp(advantage / beta, min=-30.0, max=30.0))
    clipped = raw.clamp(min=min_weight, max=max_weight)
    return clipped.detach(), advantage.detach()


def mixed_supervised_loss(
    per_sample_loss: torch.Tensor,
    is_rollout: torch.Tensor,
    rollout_weights: Optional[torch.Tensor] = None,
    rollout_scale: float = 1.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine unit-weight demonstrations with weighted rollout samples."""
    loss = per_sample_loss.reshape(-1)
    rollout_mask = is_rollout.reshape(-1).bool()
    demo_mask = ~rollout_mask
    zero = loss.sum() * 0.0
    demo_loss = loss[demo_mask].mean() if demo_mask.any() else zero
    if rollout_mask.any():
        if rollout_weights is None:
            weights = torch.ones_like(loss[rollout_mask])
        else:
            weights = rollout_weights.reshape(-1)[rollout_mask].detach()
        rollout_loss = (weights * loss[rollout_mask]).sum() / weights.sum().clamp_min(eps)
    else:
        rollout_loss = zero
    total = demo_loss + float(rollout_scale) * rollout_loss
    return total, demo_loss, rollout_loss
