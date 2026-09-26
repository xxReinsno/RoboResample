# core.py

import os
import torch
import logging
import argparse
import torch.nn as nn
from typing import Tuple, Dict
import torch.nn.functional as F

try:
    from .model import Actor, Critic, MetaPolicy, MetaPolicyReplayBuffer
except ImportError:  # Preserve direct ``python calql/core.py`` usage.
    from model import Actor, Critic, MetaPolicy, MetaPolicyReplayBuffer

class CalQLLearner:
    """
    The main class orchestrating the Cal-QL training process.

    This class encapsulates the models, optimizers, and the core training step
    which includes the TD-loss and the calibrated conservative loss.
    """
    def __init__(self, state_dim: int, action_dim: int, config: argparse.Namespace, device: torch.device):
        self.config = config
        self.device = device
        self.action_dim = action_dim
        
        # --- Initialize Models ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval() # Target network should always be in eval mode

        # --- Initialize Optimizers ---
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        
        # --- SAC Temperature (alpha) ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        """The temperature parameter alpha, learned automatically."""
        return self.log_alpha.exp()

    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """Performs a single gradient update step for all components."""
        state, action, reward, next_state, done, mc_return = [b.to(self.device) for b in batch]
        
        # ---------------------------------
        # 1. Update Critic (Q-network)
        # ---------------------------------
        
        # --- Compute Target Q-value (from SAC) ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q_intermediate = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = reward + (1. - done) * self.config.gamma * target_q_intermediate

        # --- Compute TD Loss ---
        current_q1, current_q2 = self.critic(state, action)

        critic_loss_td = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # --- Compute Cal-QL Conservative Loss ---
        # Sample Out-of-Distribution (OOD) actions from random and policy distributions
        batch_size = state.shape[0]
        ood_random_actions = torch.empty(
            batch_size, self.config.cql_n_actions, self.action_dim, device=self.device
        ).uniform_(-1.0, 1.0)
        
        repeated_states = state.unsqueeze(1).repeat(1, self.config.cql_n_actions, 1)
        with torch.no_grad():
            ood_pi_actions, _ = self.actor(repeated_states, with_logprob=False)

        # Get Q-values for these OOD actions
        q1_random, q2_random = self.critic(repeated_states, ood_random_actions)
        q1_pi, q2_pi = self.critic(repeated_states, ood_pi_actions)

        # *** THE CORE OF CAL-QL ***: Calibrate policy-based Q-values with MC lower bound
        mc_return_reshaped = mc_return.view(-1, 1)
        q1_pi_cal = torch.max(q1_pi, mc_return_reshaped)
        q2_pi_cal = torch.max(q2_pi, mc_return_reshaped)

        # Concatenate all Qs to compute the log-sum-exp term, which approximates max_a Q(s,a)
        cat_q1 = torch.cat([q1_random, q1_pi_cal, current_q1], dim=1)
        cat_q2 = torch.cat([q2_random, q2_pi_cal, current_q2], dim=1)
        
        # Final CQL loss: push down on OOD Qs (logsumexp) and push up on data Qs
        cql_loss_1 = (torch.logsumexp(cat_q1, dim=1) - current_q1.squeeze(-1)).mean()
        cql_loss_2 = (torch.logsumexp(cat_q2, dim=1) - current_q2.squeeze(-1)).mean()
        critic_loss_cql = self.config.cql_alpha * (cql_loss_1 + cql_loss_2)
        
        # Total critic loss is the sum of TD error and the conservative penalty
        critic_loss = critic_loss_td + critic_loss_cql
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------------------------
        # 2. Update Actor and Alpha (Standard SAC update)
        # ---------------------------------
        new_action, log_prob = self.actor(state)
        q1_new, q2_new = self.critic(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new).squeeze(-1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.alpha * (log_prob + self.config.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------------------------------
        # 3. Soft Update Target Network
        # ---------------------------------
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data
                )
                
        # --- Return metrics for logging ---
        return {
            "critic_loss_td": critic_loss_td.item(),
            "critic_loss_cql": critic_loss_cql.item(),
            "critic_total_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item()
        }

    def save_checkpoint(self, path: str):
        """Saves models and optimizers to a directory for resuming training."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, "critic_optimizer.pth"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, "actor_optimizer.pth"))
        torch.save(self.alpha_optimizer.state_dict(), os.path.join(path, "alpha_optimizer.pth"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pt"))

    def load_checkpoint(self, path: str):
        """Loads models and optimizers from a directory to resume training."""
        logging.info(f"Resuming training from checkpoint: {path}")
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, "critic_optimizer.pth")))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, "actor_optimizer.pth")))
        self.alpha_optimizer.load_state_dict(torch.load(os.path.join(path, "alpha_optimizer.pth")))
        self.log_alpha = torch.load(os.path.join(path, "log_alpha.pt"), map_location=self.device)


class CalQLLearnerWithHRL:
    """
    An enhanced learner that integrates a learnable high-level policy (MetaPolicy)
    for hierarchical exploration with the Cal-QL framework.
    """
    def __init__(self, state_dim: int, action_dim: int, config: argparse.Namespace, device: torch.device):
        self.config = config
        self.device = device
        self.action_dim = action_dim
        
        # --- 1. Initialize Low-Level Models (Actor-Critic) ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        
        # --- 2. Initialize Critic ENSEMBLE for Uncertainty Reward ---
        # The ensemble is crucial for providing a robust reward signal to the meta-policy.
        self.critics = nn.ModuleList(
            [Critic(state_dim, action_dim) for _ in range(config.num_critics)]
        ).to(device)
        self.critic_target_ensemble = nn.ModuleList(
            [Critic(state_dim, action_dim) for _ in range(config.num_critics)]
        ).to(device)
        
        for i in range(config.num_critics):
            self.critic_target_ensemble[i].load_state_dict(self.critics[i].state_dict())
            self.critic_target_ensemble[i].eval()

        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=config.learning_rate)
        
        # --- 3. Initialize SAC Temperature (alpha) ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)

        # --- 4. Initialize High-Level Meta-Policy ---
        self.meta_policy = MetaPolicy(state_dim).to(device)
        self.meta_policy_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(), lr=config.meta_policy_lr
        )
        self.meta_policy_buffer = MetaPolicyReplayBuffer(config.meta_buffer_capacity)

    @property
    def alpha(self) -> torch.Tensor:
        """The temperature parameter alpha, learned automatically."""
        return self.log_alpha.exp()

    def train_low_level_step(self, batch: Tuple) -> Dict[str, float]:
        """Performs a gradient update for the low-level Actor and Critic ensemble."""
        state, action, reward, next_state, done, mc_return = [b.to(self.device) for b in batch]
        
        # --- 1. Update Critic Ensemble ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            # Use the first two target critics for the target Q, a common practice for stability.
            target_q1, target_q2 = self.critic_target_ensemble[0](next_state, next_action), self.critic_target_ensemble[1](next_state, next_action)
            target_q_intermediate = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
            target_q = reward.unsqueeze(-1) + (1. - done.unsqueeze(-1)) * self.config.gamma * target_q_intermediate

        total_critic_loss = 0.0
        # Compute losses for EACH critic in the ensemble independently
        for i in range(self.config.num_critics):
            current_q1, current_q2 = self.critics[i](state, action)
            critic_loss_td = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Cal-QL Conservative Loss (identical logic, but applied to each critic)
            batch_size = state.shape[0]
            ood_random_actions = torch.empty(batch_size, self.config.cql_n_actions, self.action_dim, device=self.device).uniform_(-1.0, 1.0)
            repeated_states = state.unsqueeze(1).repeat(1, self.config.cql_n_actions, 1)
            with torch.no_grad(): ood_pi_actions, _ = self.actor(repeated_states, with_logprob=False)
            q1_random, q2_random = self.critics[i](repeated_states, ood_random_actions)
            q1_pi, q2_pi = self.critics[i](repeated_states, ood_pi_actions)
            mc_return_reshaped = mc_return.view(-1, 1)
            q1_pi_cal, q2_pi_cal = torch.max(q1_pi, mc_return_reshaped), torch.max(q2_pi, mc_return_reshaped)
            cat_q1 = torch.cat([q1_random, q1_pi_cal, current_q1], dim=1)
            cat_q2 = torch.cat([q2_random, q2_pi_cal, current_q2], dim=1)
            cql_loss_1 = (torch.logsumexp(cat_q1, dim=1) - current_q1.squeeze(-1)).mean()
            cql_loss_2 = (torch.logsumexp(cat_q2, dim=1) - current_q2.squeeze(-1)).mean()
            critic_loss_cql = self.config.cql_alpha * (cql_loss_1 + cql_loss_2)
            
            total_critic_loss += (critic_loss_td + critic_loss_cql)
        
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- 2. Update Actor ---
        new_action, log_prob = self.actor(state)
        # Actor update uses the first critic's Q-values for guidance
        q1_new, q2_new = self.critics[0](state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new).squeeze(-1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- 3. Update Alpha ---
        alpha_loss = -(self.alpha * (log_prob + self.config.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --- 4. Soft Update All Target Critic Networks ---
        with torch.no_grad():
            for i in range(self.config.num_critics):
                for param, target_param in zip(self.critics[i].parameters(), self.critic_target_ensemble[i].parameters()):
                    target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)
                
        return {"critic_loss": total_critic_loss.item() / self.config.num_critics, "actor_loss": actor_loss.item()}

    def train_meta_policy_step(self) -> Dict[str, float]:
        """Performs a gradient update for the high-level Meta-Policy using REINFORCE."""
        if len(self.meta_policy_buffer) < self.config.meta_policy_batch_size:
            return {"meta_loss": 0.0}

        batch = self.meta_policy_buffer.sample(self.config.meta_policy_batch_size)
        log_probs = torch.stack([item['log_prob'] for item in batch]).to(self.device)
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float32, device=self.device)

        # Normalize rewards for stable training (a crucial trick for policy gradients)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # REINFORCE loss: -log_prob * advantage (here, normalized reward is the advantage)
        loss = - (log_probs * rewards).mean()
        
        self.meta_policy_optimizer.zero_grad()
        loss.backward()
        self.meta_policy_optimizer.step()

        return {"meta_loss": loss.item()}

    def save_checkpoint(self, path: str):
        """Saves all models and optimizers."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.critics.state_dict(), os.path.join(path, "critics_ensemble.pth"))
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.meta_policy.state_dict(), os.path.join(path, "meta_policy.pth"))


    def load_checkpoint(self, path: str):
        """Loads all models and optimizers."""
        logging.info(f"Resuming training from checkpoint: {path}")
        self.critics.load_state_dict(torch.load(os.path.join(path, "critics_ensemble.pth"), map_location=self.device))
        for i in range(self.config.num_critics):
            self.critic_target_ensemble[i].load_state_dict(self.critics[i].state_dict())
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=self.device))
        self.meta_policy.load_state_dict(torch.load(os.path.join(path, "meta_policy.pth"), map_location=self.device))
