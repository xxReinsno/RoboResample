import os
import importlib.util
import tempfile
import unittest

import h5py
import numpy as np
import torch
from torch import nn

from calql.dataset import HDF5CalQLDataset
MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "libero_exp", "models", "resample_v2.py"
)
SPEC = importlib.util.spec_from_file_location("resample_v2", MODULE_PATH)
resample_v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resample_v2)
ActiveValueAcquisition = resample_v2.ActiveValueAcquisition
DoubleQValueModel = resample_v2.DoubleQValueModel
compute_awr_weights = resample_v2.compute_awr_weights
mixed_supervised_loss = resample_v2.mixed_supervised_loss


class FakeCritic(nn.Module):
    def forward(self, state, action):
        value = action.sum(dim=-1)
        return value, value - 2.0


class ResampleV2Tests(unittest.TestCase):
    def test_chunk_critic_uses_entire_chunk(self):
        chunks = torch.tensor([[[[1.0], [2.0]], [[4.0], [8.0]]]])
        mean, std = DoubleQValueModel(FakeCritic()).score(torch.zeros(1, 3), chunks)
        torch.testing.assert_close(mean, torch.tensor([[1.0, 10.0]]))
        torch.testing.assert_close(std, torch.ones_like(std))

    def test_k_one_degenerates_to_candidate_zero(self):
        selector = ActiveValueAcquisition(always_trigger=True)
        result = selector.select(torch.tensor([[3.0]]), torch.tensor([[0.2]]))
        self.assertEqual(result.selected_index.item(), 0)

    def test_active_selection_respects_safety_mask(self):
        selector = ActiveValueAcquisition(
            min_safe_q=0.0,
            uncertainty_weight=1.0,
            coverage_weight=0.0,
            always_trigger=True,
        )
        result = selector.select(
            q_mean=torch.tensor([[-2.0, 1.0, 0.5]]),
            q_std=torch.tensor([[5.0, 0.0, 1.0]]),
        )
        self.assertEqual(result.selected_index.item(), 1)

    def test_awr_is_detached_finite_and_clipped(self):
        q = torch.tensor([1.0, -1000.0], requires_grad=True)
        value = torch.zeros(2, requires_grad=True)
        weight, _ = compute_awr_weights(q, value, beta=0.5, min_weight=0.1, max_weight=10)
        self.assertFalse(weight.requires_grad)
        self.assertTrue(torch.isfinite(weight).all())
        self.assertGreaterEqual(weight.min().item(), 0.1)
        self.assertLessEqual(weight.max().item(), 10.0)

    def test_unit_weights_recover_grouped_bc(self):
        losses = torch.tensor([1.0, 3.0, 2.0, 4.0])
        is_rollout = torch.tensor([False, False, True, True])
        total, demo, rollout = mixed_supervised_loss(
            losses, is_rollout, torch.ones(4), rollout_scale=1.0
        )
        torch.testing.assert_close(demo, torch.tensor(2.0))
        torch.testing.assert_close(rollout, torch.tensor(3.0))
        torch.testing.assert_close(total, torch.tensor(5.0))

    def test_chunk_dataset_uses_h_step_return_and_next_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rollout.hdf5")
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("data").create_group("demo_0")
                episode.attrs["num_samples"] = 3
                obs = episode.create_group("obs")
                next_obs = episode.create_group("next_obs")
                obs.create_dataset("state", data=np.arange(3, dtype=np.float32)[:, None])
                next_obs.create_dataset("state", data=np.arange(1, 4, dtype=np.float32)[:, None])
                episode.create_dataset("actions", data=np.arange(3, dtype=np.float32)[:, None])
                episode.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.float32))
            dataset = HDF5CalQLDataset(path.rsplit("/", 1)[0], ["state"], 0.5, 2)
            state, action, reward, next_state, done, _ = dataset[1]
            torch.testing.assert_close(action, torch.tensor([1.0, 2.0]))
            torch.testing.assert_close(reward, torch.tensor([0.5]))
            torch.testing.assert_close(next_state, torch.tensor([3.0]))
            torch.testing.assert_close(done, torch.tensor([1.0]))

    def test_chunk_dataset_uses_recorded_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rollout.hdf5")
            with h5py.File(path, "w") as handle:
                episode = handle.create_group("data").create_group("demo_0")
                episode.attrs["num_samples"] = 3
                obs = episode.create_group("obs")
                next_obs = episode.create_group("next_obs")
                obs.create_dataset("state", data=np.arange(3, dtype=np.float32)[:, None])
                next_obs.create_dataset("state", data=np.arange(1, 4, dtype=np.float32)[:, None])
                episode.create_dataset("actions", data=np.arange(3, dtype=np.float32)[:, None])
                episode.create_dataset("rewards", data=np.array([0.2, 0.4, 0.8], dtype=np.float32))
                episode.create_dataset("dones", data=np.array([0, 0, 1], dtype=np.float32))
            dataset = HDF5CalQLDataset(directory, ["state"], 0.5, 2)
            _, _, reward, _, done, mc_return = dataset[0]
            torch.testing.assert_close(reward, torch.tensor([0.4]))
            torch.testing.assert_close(mc_return, torch.tensor([0.6]))
            torch.testing.assert_close(done, torch.tensor([0.0]))


if __name__ == "__main__":
    unittest.main()
