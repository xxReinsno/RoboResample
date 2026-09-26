# model.py

import torch
import numpy as np
import torch.nn as nn
from typing import Tuple, List, Dict


class StateValueNetwork(nn.Module):
    """Small state-value baseline used by advantage-weighted regression."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class Actor(nn.Module):
    """
    A Soft Actor-Critic (SAC) style actor network.

    It maps a state to a probability distribution over actions, specifically a
    squashed Gaussian distribution to ensure actions are bounded within a
    [-max_action, max_action] range.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor, deterministic: bool = False,
                with_logprob: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass to get an action and its log probability.

        Args:
            state (torch.Tensor): The input state tensor.
            deterministic (bool): If True, returns the mean of the distribution
                                  (used for evaluation). Defaults to False.
            with_logprob (bool): If True, computes and returns the log probability
                                 of the action. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The action tensor and its
                                               log probability (or None).
        """
        hidden = self.net(state)
        mean = self.mean_layer(hidden)
        log_std = torch.clamp(self.log_std_layer(hidden), -20, 2)
        std = torch.exp(log_std)

        if deterministic:
            action = torch.tanh(mean) * self.max_action
            return action, None

        # Use reparameterization trick for differentiable sampling
        normal = torch.distributions.Normal(mean, std)
        action_raw = normal.rsample()
        action = torch.tanh(action_raw) * self.max_action

        # Compute log probability, accounting for the tanh squash
        if with_logprob:
            log_prob = normal.log_prob(action_raw).sum(axis=-1)
            # This correction term is crucial for SAC
            log_prob -= torch.log(
                self.max_action * (1 - (action / self.max_action).pow(2)) + 1e-6
            ).sum(axis=-1)
        else:
            log_prob = None

        return action, log_prob


class Critic(nn.Module):
    """
    A Clipped Double-Q Critic network for SAC and Cal-QL.

    It implements two independent Q-networks (Q1 and Q2) to mitigate Q-value
    overestimation. The forward pass can handle both standard (2D) and
    OOD-batched (3D) inputs.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        # Define the two Q-networks
        self.q1_net = self._build_net(state_dim, action_dim)
        self.q2_net = self._build_net(state_dim, action_dim)

    def _build_net(self, state_dim: int, action_dim: int) -> nn.Module:
        """Helper to construct a single Q-network."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the Q-values for given state-action pairs.

        This method robustly handles inputs of shape (batch, dim) for standard
        training and (batch, n_actions, dim) for OOD estimation in CQL.

        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Q1 and Q2 values.
        """
        batch_size = state.shape[0]
        # Flatten all dimensions except the last one for nn.Linear
        flat_state = state.reshape(-1, state.shape[-1])
        flat_action = action.reshape(-1, action.shape[-1])
        
        # Concatenate state and action for input to the Q-networks
        sa_input = torch.cat([flat_state, flat_action], dim=1)
        q1 = self.q1_net(sa_input)
        q2 = self.q2_net(sa_input)

        # Reshape the output back to match the original batch structure if needed
        if state.dim() == 3:
            # Reshape from (batch * n_actions, 1) to (batch, n_actions)
            q1 = q1.view(batch_size, -1)
            q2 = q2.view(batch_size, -1)
        
        return q1, q2
    

class MetaPolicy(nn.Module):
    """
    A high-level policy (gating network) that decides when to explore.

    This network takes the current state as input and outputs a probability
    distribution over two high-level actions: 'exploit' (use the main policy)
    or 'explore' (use the adversarial sampler). Its goal is to learn an
    optimal intervention strategy based on the principle of maximizing
    information gain, proxied by critic uncertainty.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        Initializes the MetaPolicy network.

        Args:
            state_dim (int): The dimension of the state space.
            hidden_dim (int): The size of the hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # 2 logits for 'exploit' (idx 0) and 'explore' (idx 1)
        )

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """
        Takes a state and returns a categorical distribution over high-level actions.

        Args:
            state (torch.Tensor): The input state tensor of shape (batch_size, state_dim).

        Returns:
            torch.distributions.Categorical: A distribution object from which you can
                                             sample actions and get log probabilities.
        """
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)


class MetaPolicyReplayBuffer:
    """
    A simple replay buffer to store transitions for training the MetaPolicy.

    It stores high-level decision events, which consist of the log-probability
    of the chosen high-level action and the delayed reward received after the
    action's duration.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Dict] = []
        self.position = 0

    def push(self, log_prob: torch.Tensor, reward: float):
        """Saves a transition, detaching tensors to prevent memory leaks."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = {
            "log_prob": log_prob.detach(),
            "reward": reward
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Dict]:
        """Samples a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
