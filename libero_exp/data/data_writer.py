# data/data_writer.py

import os
import h5py
import numpy as np
import logging
import re
from typing import List, Dict

# Base class for data writer

class BaseHDF5Writer:
    """
    A base class for writing trajectory data to HDF5 files in a hierarchical,
    task-separated format that mimics the official Libero dataset.

    This class handles the core logic of creating files, managing demos,
    and writing data. Subclasses will define the specific save directory.
    """
    def __init__(self, output_dir: str, obs_keys: List[str]):
        """
        Initializes the BaseHDF5Writer.

        Args:
            output_dir (str): The specific directory where this writer will save files.
            obs_keys (list): A list of observation keys to be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.obs_keys = obs_keys
        
        # Internal state management for multiple HDF5 files
        self._file_handles: Dict[str, h5py.File] = {}
        self._demo_counts: Dict[str, int] = {}

        logging.info(f"Initialized HDF5 writer. Saving to: {self.output_dir}")

    def _sanitize_filename(self, name: str) -> str:
        """Cleans a task description to make it a valid filename."""
        name = name.lower()
        name = re.sub(r'[^\w\s-]', '_', name)
        name = re.sub(r'[-\s]+', '_', name)
        return f"{name}_demo.hdf5"

    def _get_or_create_file_handle(self, filename: str) -> h5py.File:
        """Lazily opens or creates an HDF5 file for a given task."""
        full_path = os.path.join(self.output_dir, filename)
        if full_path in self._file_handles:
            return self._file_handles[full_path]
        
        logging.info(f"Creating new HDF5 file: {full_path}")
        file_handle = h5py.File(full_path, 'w')
        file_handle.create_group("data")
        self._file_handles[full_path] = file_handle
        self._demo_counts[full_path] = 0
        return file_handle

    def write_episode(self, episode_data: Dict, task_description: str):
        """
        Writes a single episode to the correct HDF5 file based on the task.
        This method is intended to be called by subclasses.
        """
        hdf5_filename = self._sanitize_filename(task_description)
        file_handle = self._get_or_create_file_handle(hdf5_filename)
        
        demo_count = self._demo_counts[os.path.join(self.output_dir, hdf5_filename)]
        demo_key = f"demo_{demo_count}"
        
        # --- Write data (identical for all writers) ---
        data_grp = file_handle["data"]
        ep_data_grp = data_grp.create_group(demo_key)

        num_samples = len(episode_data['actions'])
        ep_data_grp.attrs["num_samples"] = num_samples

        obs_grp = ep_data_grp.create_group("obs")
        for key in self.obs_keys:
            if key in episode_data['obs']:
                obs_grp.create_dataset(key, data=np.array(episode_data['obs'][key]), compression='gzip')

        next_obs_grp = ep_data_grp.create_group("next_obs")
        for key in self.obs_keys:
            if key in episode_data['next_obs']:
                next_obs_grp.create_dataset(key, data=np.array(episode_data['next_obs'][key]), compression='gzip')
            
        ep_data_grp.create_dataset("actions", data=np.array(episode_data['actions']), compression='gzip')
        ep_data_grp.create_dataset("rewards", data=np.array(episode_data['rewards']), compression='gzip')
        ep_data_grp.create_dataset("dones", data=np.array(episode_data['dones']), compression='gzip')
        ep_data_grp.create_dataset("terminals", data=np.array(episode_data['dones']), compression='gzip')

        # RESample V2 metadata is optional so legacy rollout files remain valid.
        # Each value is time-aligned with ``actions``. Candidate tensors are
        # diagnostics only and are never loaded as actor supervision targets.
        if episode_data.get("acquisition"):
            acquisition_grp = ep_data_grp.create_group("acquisition")
            for key, value in episode_data["acquisition"].items():
                acquisition_grp.create_dataset(
                    key, data=np.asarray(value), compression="gzip"
                )

        self._demo_counts[os.path.join(self.output_dir, hdf5_filename)] += 1
        logging.info(f"Successfully wrote {demo_key} to {hdf5_filename} ({num_samples} steps).")

    def close(self):
        """Closes all opened HDF5 file handles."""
        logging.info(f"Closing HDF5 files for writer targeting: {self.output_dir}")
        for path, handle in self._file_handles.items():
            if handle:
                handle.close()
                logging.info(f"  - Closed {os.path.basename(path)}. Total demos: {self._demo_counts.get(path, 0)}")
        self._file_handles.clear()
        self._demo_counts.clear()

# all collect class

class HDF5Writer(BaseHDF5Writer):
    """
    A specific HDF5 writer that saves ALL collected trajectories.
    It determines its output directory by appending nothing to the env_name.
    """
    def __init__(self, base_output_dir: str, env_name: str, obs_keys: List[str]):
        output_dir = os.path.join(base_output_dir, env_name)
        super().__init__(output_dir, obs_keys)

# successful collect class

class HDF5WriterSucc(BaseHDF5Writer):
    """
    A specific HDF5 writer that ONLY saves successful trajectories.
    It determines its output directory by appending '_succ' to the env_name.
    """
    def __init__(self, base_output_dir: str, env_name: str, obs_keys: List[str]):
        output_dir = os.path.join(base_output_dir, f"{env_name}_succ")
        super().__init__(output_dir, obs_keys)

    def write_episode_if_successful(self, episode_data: Dict, task_description: str, is_success: bool):
        """
        A wrapper around the base write method that includes a success check.
        It will only write the episode if `is_success` is True.
        """
        if is_success:
            # If the trajectory was successful, call the parent's write method.
            super().write_episode(episode_data, task_description)
        # If not successful, this writer does nothing.
