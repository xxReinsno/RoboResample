"""Train Cal-QL on action chunks with an H-step TD target."""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .core import CalQLLearner
from .dataset import HDF5CalQLDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RESample V2 chunk critic")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/chunk_calql")
    parser.add_argument(
        "--obs_keys", nargs="+", default=["robot0_gripper_qpos", "robot0_joint_pos"]
    )
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--training_steps", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--cql_alpha", type=float, default=5.0)
    parser.add_argument("--cql_n_actions", type=int, default=10)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--save_interval", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action_horizon < 1:
        raise ValueError("action_horizon must be positive")
    torch.manual_seed(args.seed)
    base_gamma = args.gamma
    dataset = HDF5CalQLDataset(
        args.dataset_path, args.obs_keys, base_gamma, args.action_horizon
    )
    if not len(dataset):
        raise RuntimeError("No transitions found")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    state_dim = dataset.transitions[0]["state"].shape[-1]
    flat_action_dim = args.action_dim * args.action_horizon
    args.gamma = base_gamma ** args.action_horizon
    if args.target_entropy is None:
        args.target_entropy = -float(flat_action_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner = CalQLLearner(state_dim, flat_action_dim, args, device)

    iterator = iter(loader)
    for step in range(1, args.training_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        metrics = learner.train_step(batch)
        if step % 1000 == 0:
            message = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            print(f"step={step} {message}")
        if step % args.save_interval == 0:
            learner.save_checkpoint(os.path.join(args.output_dir, f"step_{step}"))
    learner.save_checkpoint(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    main()
