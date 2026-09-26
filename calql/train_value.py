"""Train the frozen state-value baseline used by RESample V2 AWR.

Example:
    python -m calql.train_value --dataset_path rollout/libero_spatial \
      --critic_path checkpoints/chunk_calql/critic.pth --action_horizon 10
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .dataset import HDF5CalQLDataset
from .model import Critic, StateValueNetwork


def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.square()).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a chunk-wise AWR value head")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--critic_path", required=True)
    parser.add_argument("--output_path", default="checkpoints/value.pth")
    parser.add_argument(
        "--obs_keys", nargs="+", default=["robot0_gripper_qpos", "robot0_joint_pos"]
    )
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.expectile < 1.0:
        raise ValueError("expectile must be in (0, 1)")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HDF5CalQLDataset(
        args.dataset_path, args.obs_keys, args.gamma, args.action_horizon
    )
    if not len(dataset):
        raise RuntimeError("No transitions found")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    state_dim = dataset.transitions[0]["state"].shape[-1]
    flat_action_dim = args.action_dim * args.action_horizon
    critic = Critic(state_dim, flat_action_dim).to(device)
    critic.load_state_dict(torch.load(args.critic_path, map_location=device))
    critic.eval().requires_grad_(False)
    value = StateValueNetwork(state_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(value.parameters(), lr=args.learning_rate)

    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            state, action, *_ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            state, action, *_ = next(iterator)
        state, action = state.to(device), action.to(device)
        with torch.no_grad():
            q1, q2 = critic(state, action)
            pair = torch.stack([q1, q2], dim=0)
            target = pair.min(dim=0).values - args.kappa * pair.std(dim=0, unbiased=False)
        prediction = value(state)
        loss = expectile_loss(target - prediction, args.expectile)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"step={step} value_loss={loss.item():.6f}")

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "value": value.state_dict(),
            "state_dim": state_dim,
            "hidden_dim": args.hidden_dim,
            "action_horizon": args.action_horizon,
            "expectile": args.expectile,
        },
        args.output_path,
    )


if __name__ == "__main__":
    main()
