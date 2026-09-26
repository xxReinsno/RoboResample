import os
import glob
import h5py
import tqdm
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple

class HDF5CalQLDataset(Dataset):
    """
    A custom PyTorch Dataset that automatically discovers and loads data from all
    HDF5 files within a specified directory.

    It correctly reads the HDF5 structure where each file contains multiple '/data/demo_x'
    groups, with each group representing a single episode and containing its own
    'actions', 'dones', 'terminals', etc. datasets.
    """
    def __init__(
        self,
        dataset_path: str,
        obs_keys: List[str],
        gamma: float,
        action_horizon: int = 1,
    ):
        self.dataset_path = dataset_path
        self.obs_keys = obs_keys
        self.gamma = gamma
        if action_horizon < 1:
            raise ValueError("action_horizon must be at least one")
        self.action_horizon = action_horizon
        self.transitions = []
        self._load_and_process_data()

    def _discover_hdf5_files(self) -> List[str]:
        """Finds all HDF5 files in the specified directory."""
        if not os.path.isdir(self.dataset_path):
            logging.error(f"Dataset path is not a valid directory: {self.dataset_path}")
            return []
        
        search_pattern = os.path.join(self.dataset_path, "*.hdf5")
        hdf5_files = glob.glob(search_pattern)
        
        if not hdf5_files:
            logging.warning(f"No HDF5 files found in directory: {self.dataset_path}")
        
        return sorted(hdf5_files)

    def _load_and_process_data(self):
        """
        Internal method to read all HDF5 files by iterating through 'demo_x' groups
        and preparing all transitions.
        """
        hdf5_paths = self._discover_hdf5_files()
        
        if not hdf5_paths:
            return

        logging.info(f"Found {len(hdf5_paths)} HDF5 file(s) to process.")
        
        for hdf5_path in hdf5_paths:
            logging.info(f"Processing file: {hdf5_path}")
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    if 'data' not in f:
                        logging.warning(f"File {hdf5_path} does not contain a 'data' group. Skipping.")
                        continue
                        
                    demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))
                    
                    for demo_key in tqdm.tqdm(demo_keys, desc=f"Processing demos in {os.path.basename(hdf5_path)}"):
                        ep_data = f['data'][demo_key]
                        num_samples = ep_data.attrs['num_samples']
                        if num_samples < 1:
                            continue

                        # Correctly access datasets within the ep_data group
                        actions = ep_data['actions'][:]
                        dones = ep_data['dones'][:]
                        rewards = (
                            ep_data['rewards'][:].astype(np.float32)
                            if 'rewards' in ep_data
                            else np.zeros(num_samples, dtype=np.float32)
                        )
                        # Legacy files may omit rewards and encode success only
                        # through the terminal flag.
                        if 'rewards' not in ep_data and bool(dones[-1]):
                            rewards[-1] = 1.0
                        # # The 'terminals' key exists inside each demo group
                        # terminals = ep_data['terminals'][:]

                        state_keys = [key for key in self.obs_keys if 'image' not in key]
                        if state_keys:
                            states = np.concatenate([ep_data['obs'][key][:] for key in state_keys], axis=1)
                            next_states = np.concatenate([ep_data['next_obs'][key][:] for key in state_keys], axis=1)
                        else:
                            states = np.zeros((num_samples, 0), dtype=np.float32)
                            next_states = np.zeros((num_samples, 0), dtype=np.float32)

                        # Compute Monte-Carlo returns for this episode
                        mc_returns = np.zeros(num_samples, dtype=np.float32)
                        mc_return = 0.0
                        for i in reversed(range(num_samples)):
                            mc_return = rewards[i] + self.gamma * mc_return * (1.0 - float(dones[i]))
                            mc_returns[i] = mc_return

                        # Construct chunk-wise transitions. Near the end of an
                        # episode we repeat the final action only as padding;
                        # rewards and the bootstrap state still stop at the
                        # true terminal transition.
                        for i in range(num_samples):
                            end = min(i + self.action_horizon, num_samples)
                            action_chunk = actions[i:end]
                            if len(action_chunk) < self.action_horizon:
                                padding = np.repeat(
                                    action_chunk[-1:],
                                    self.action_horizon - len(action_chunk),
                                    axis=0,
                                )
                                action_chunk = np.concatenate([action_chunk, padding], axis=0)
                            chunk_return = 0.0
                            # A file boundary is a timeout/terminal for offline
                            # training even if the environment did not report
                            # success; never bootstrap beyond recorded data.
                            chunk_done = end == num_samples
                            for offset, step_idx in enumerate(range(i, end)):
                                chunk_return += (self.gamma ** offset) * float(rewards[step_idx])
                                chunk_done = chunk_done or bool(dones[step_idx])
                            self.transitions.append({
                                "state": states[i].astype(np.float32),
                                "action": action_chunk.reshape(-1).astype(np.float32),
                                "reward": np.array([chunk_return], dtype=np.float32),
                                "next_state": next_states[end - 1].astype(np.float32),
                                "done": np.array([chunk_done], dtype=np.float32),
                                "mc_return": np.array([mc_returns[i]], dtype=np.float32)
                            })
                
            except Exception as e:
                logging.error(f"Failed to load or process HDF5 file {hdf5_path}: {e}")
                continue
        
        logging.info(f"Finished processing all files. Total transitions loaded: {len(self.transitions)}")

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Tuple:
        trans = self.transitions[idx]
        return (
            torch.from_numpy(trans["state"]),
            torch.from_numpy(trans["action"]),
            torch.from_numpy(trans["reward"]),
            torch.from_numpy(trans["next_state"]),
            torch.from_numpy(trans["done"]),
            # Note: Added terminal to the output if you need it later
            # torch.from_numpy(trans["terminal"]), 
            torch.from_numpy(trans["mc_return"])
        )
