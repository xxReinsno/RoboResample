"""Convert OpenVLA-style LIBERO RLDS episodes to critic-only HDF5 files.

The VLA-Adapter data is stored as TFDS RLDS episodes.  The standalone chunk
critic intentionally consumes the same compact robomimic-style layout used by
the existing RoboResample loader, while keeping only proprioception (no image
copy).  We expose the conventional ``robot0_gripper_qpos`` (2D) and
``robot0_joint_pos`` (7D) keys, so the resulting state is 9D and remains
compatible with the current DiT sampler configuration.

Run this script from the VLA-Adapter environment because TensorFlow Datasets is
already installed there, e.g.::

  python -m calql.convert_rlds_libero \
      --rlds_dir /mnt/.../libero_spatial_no_noops/1.0.0 \
      --output_dir /mnt/.../critic_data/libero_spatial
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds


def _episode_arrays(episode):
    steps = list(episode["steps"].as_numpy_iterator())
    if not steps:
        return None
    state = np.stack([s["observation"]["state"] for s in steps]).astype(np.float32)
    joints = np.stack([s["observation"]["joint_state"] for s in steps]).astype(np.float32)
    actions = np.stack([s["action"] for s in steps]).astype(np.float32)
    rewards = np.asarray([s["reward"] for s in steps], dtype=np.float32)
    # ``is_last`` marks the final recorded step.  Treating it as done avoids
    # bootstrapping through an episode boundary in chunk-wise TD targets.
    dones = np.asarray([s["is_last"] or s["is_terminal"] for s in steps], dtype=np.bool_)
    if not dones[-1]:
        dones[-1] = True

    # RLDS state is [6D EEF pose, 2D gripper].  Keep the 2D gripper portion to
    # match the existing RoboResample 9D (gripper + joints) state interface.
    gripper = state[:, -2:]
    return gripper, joints, actions, rewards, dones


def convert(rlds_dir: Path, output_dir: Path, max_episodes: int | None = None) -> tuple[int, int]:
    builder = tfds.builder_from_directory(builder_dir=str(rlds_dir))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{builder.name}.hdf5"

    episodes = transitions = 0
    with h5py.File(output_path, "w") as h5:
        data = h5.create_group("data")
        # Iterate regular TF examples so the nested ``steps`` value remains a
        # Dataset and can be streamed without materialising all episodes first.
        for episode in dataset:
            arrays = _episode_arrays(episode)
            if arrays is None:
                continue
            gripper, joints, actions, rewards, dones = arrays
            group = data.create_group(f"demo_{episodes}")
            group.attrs["num_samples"] = int(len(actions))
            obs = group.create_group("obs")
            next_obs = group.create_group("next_obs")
            obs.create_dataset("robot0_gripper_qpos", data=gripper, compression="lzf")
            obs.create_dataset("robot0_joint_pos", data=joints, compression="lzf")
            # The critic only needs a bootstrap state; the final next state is
            # harmless because the corresponding transition is terminal.
            next_gripper = np.concatenate([gripper[1:], gripper[-1:]], axis=0)
            next_joints = np.concatenate([joints[1:], joints[-1:]], axis=0)
            next_obs.create_dataset("robot0_gripper_qpos", data=next_gripper, compression="lzf")
            next_obs.create_dataset("robot0_joint_pos", data=next_joints, compression="lzf")
            group.create_dataset("actions", data=actions, compression="lzf")
            group.create_dataset("rewards", data=rewards, compression="lzf")
            group.create_dataset("dones", data=dones, compression="lzf")
            transitions += len(actions)
            episodes += 1
            if max_episodes is not None and episodes >= max_episodes:
                break

    return episodes, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()
    episodes, transitions = convert(args.rlds_dir, args.output_dir, args.max_episodes)
    print(f"converted episodes={episodes} transitions={transitions} output={args.output_dir}")


if __name__ == "__main__":
    main()
