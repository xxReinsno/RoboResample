import logging
import os
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from einops import rearrange
# data writer
from ..data.data_writer import HDF5Writer, HDF5WriterSucc

from .data_utils import raw_obs_to_tensor_obs
from .video_utils import video_pad_time, rearrange_videos, render_done_to_boundary
from ..models.adversarial_sampler import AdversarialActionSampler

@torch.no_grad()
def rollout(cfg, 
            env_dict, 
            policy, 
            num_env_rollouts, 
            horizon=None, 
            return_wandb_video=True,
            success_vid_first=False, 
            fail_vid_first=False, 
            video_writer=None, 
            device=None, 
            data_writer: HDF5Writer = None,
            data_writer_succ: HDF5WriterSucc = None,
            adversarial_sampler: AdversarialActionSampler = None,
            ):
    
    policy.eval()
    all_env_indices = []
    all_task_indices = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    all_env_descriptions = []

    for env_description, (env_idx, task_idx, env, init_states, task_emb) in env_dict.items():
        all_env_indices.append(env_idx)
        all_task_indices.append(task_idx)
        all_succ = []
        all_horizon = []
        vid = []

        if task_idx < 2:
            print(f"--- Skipping task {task_idx} as requested. ---")
            continue

        if video_writer != None:
            final_video_path = os.path.join(video_writer.video_path, str(task_idx) + '.' + env_description.split('/')[-1])
            if os.path.exists(final_video_path):
                files = os.listdir(final_video_path)
                if files != []:
                    scuccess_files = [file for file in files if file.endswith("True.mp4")]
                    all_succ += [len(scuccess_files) / cfg.env.env_num]
                    all_horizon += [0]
                    vid += [0]

                    all_env_succ.append(np.array(all_succ).astype(np.float32))
                    all_env_horizon.append(all_horizon)
                    env_vid.append(vid)  # [(b, t, c, h, w)]

                    print(f"***{task_idx} {env_description.split('/')[-1]}*** has done!!!")
                    continue
            
        # # Init the sampler
        # if adversarial_sampler:
        #     adversarial_sampler.reset()


        for num_env_rollout in range(num_env_rollouts):
            print(f"\nEnv: {env_idx}, Task_id: {task_idx}, Rollout: {num_env_rollout+1}/{num_env_rollouts} running...")
            env.reset()
            env.seed(cfg.train.seed)
            if adversarial_sampler is not None:
                adversarial_sampler.reset()
            else:
                policy.reset()
            if video_writer != None:
                video_writer.reset()

            obs = env.set_init_state(init_states)

            # Initialize the data writer
            if data_writer or data_writer_succ:
                active_writer = data_writer or data_writer_succ
                episode_data_collectors = [{
                    "obs": {key: [] for key in active_writer.obs_keys},
                    "next_obs": {key: [] for key in active_writer.obs_keys},
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "terminals": [],
                    "acquisition": {},
                } for _ in range(cfg.env.env_num)]

            if data_writer_succ:
                collect_succ = False

            if adversarial_sampler:
                inserted = False

            # simulate the physics without any actions
            # action_dim = cfg.policy.policy_head.network_kwargs.output_size
            # dummy_actions = np.zeros((cfg.env.env_num, action_dim))
            # for _ in range(5):
            #     env.step(dummy_actions)

            dones = [False] * cfg.env.env_num
            num_success = 0
            episode_frames = []
            
            for step_i in tqdm(range(horizon)):

                # for k in range(cfg.env.env_num):
                #     obs[k]["agentview_image"] = obs[k]["agentview_image"][::-1, ::-1]
                #     obs[k]["robot0_eye_in_hand_image"] = obs[k]["robot0_eye_in_hand_image"][::-1, ::-1]

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg, device)

                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                data[key][sub_key] = sub_value.to(device)
                    elif isinstance(value, torch.Tensor):
                        data[key] = value.to(device)

                if data_writer or data_writer_succ:
                    # Make a deep copy of the observation dict to prevent modification by env.step
                    current_obs = {i: {k: v.copy() for k, v in obs[i].items()} for i in range(cfg.env.env_num)}
                
                # --- SELECT ACTION: ADVERSARIAL OR STANDARD ---
                should_intervene = False
                was_inserted_this_step = False
                if adversarial_sampler is not None and hasattr(adversarial_sampler, "acquisition"):
                    a, was_inserted_this_step = adversarial_sampler.select_action(data)
                    inserted = inserted or was_inserted_this_step
                elif adversarial_sampler is not None:
                    # TODO: Add Timestep policy
                    # Check the intervention condition (e.g., gripper width)
                    # We only need to check for the first env in the parallel batch
                    gripper_qpos = data['obs']['gripper_states']
                    gripper_qpos_abs = torch.abs(gripper_qpos[0][-1]).item()


                    if gripper_qpos_abs < adversarial_sampler.intervention_threshold:
                        should_intervene = True
                        # logging.info(f"Intervening: {gripper_qpos_abs} < {adversarial_sampler.intervention_threshold} at timestep {step_i}")

                if adversarial_sampler is not None and hasattr(adversarial_sampler, "acquisition"):
                    pass
                elif should_intervene and not inserted:
                    a, was_inserted_this_step = adversarial_sampler.select_action(data)
                    if was_inserted_this_step:
                        inserted = True # Mark that an insertion happened in this episode
                else:
                    a = policy.get_action(cfg, data)

                # copy the step data before env.step()
                if data_writer or data_writer_succ:
                    current_actions = a.copy() # Make a copy of the action
                    selection_metadata = getattr(
                        adversarial_sampler, "last_selection_metadata", None
                    )

                obs, r, done, info = env.step(a)


                # Append data to collectors if data_writer is active
                if data_writer or data_writer_succ:
                    next_obs = obs.copy() # This is the next_obs for the previous state
                    # obs
                    # Loop through each parallel environment
                    for i in range(cfg.env.env_num):
                        # --- Append observations for the i-th environment ---
                        for key in active_writer.obs_keys:
                            # Append the state before the action was taken
                            episode_data_collectors[i]["obs"][key].append(current_obs[i][key])
                            # Append the state after the action was taken
                            episode_data_collectors[i]["next_obs"][key].append(next_obs[i][key])
                            
                        # --- Append other data for the i-th environment ---
                        action_i = current_actions[i] if (
                            np.ndim(current_actions) > 1
                            and len(current_actions) == cfg.env.env_num
                        ) else current_actions
                        reward_i = r[i] if np.ndim(r) > 0 else r
                        done_i = done[i] if np.ndim(done) > 0 else done
                        episode_data_collectors[i]["actions"].append(action_i)
                        episode_data_collectors[i]["rewards"].append(reward_i)
                        episode_data_collectors[i]["dones"].append(done_i)
                        episode_data_collectors[i]["terminals"].append(done_i)
                        if selection_metadata is not None:
                            for key, value in selection_metadata.items():
                                episode_data_collectors[i]["acquisition"].setdefault(key, []).append(value)

                video_img = []
                for k in range(cfg.env.env_num):
                    dones[k] = dones[k] or done[k]
                    video_img.append(obs[k]["agentview_image"][::-1])
                video_img = np.stack(video_img, axis=0)
                frame = rearrange(video_img, "b h w c -> b c h w")
                frame = render_done_to_boundary(frame, dones)

                # TASK 1: Add a mark at the inserted timestep
                if was_inserted_this_step:
                    # The intervention is based on gripper_qpos[0], so we mark the frame for the FIRST environment.
                    # The color needs to be shaped correctly for broadcasting over the image channels.

                    boundary = 5  # pixels
                    color = np.array([255, 0, 0], dtype=frame.dtype)[None, None, :] # Shape: (3, 1, 1)

                    render_frame = obs[k]["agentview_image"]
                    
                    # Apply border to at this step
                    render_frame[ :boundary, :,:] = color
                    render_frame[  -boundary:, :,:] = color
                    render_frame[  :, :boundary,:] = color
                    render_frame[  :, -boundary:,:] = color

                episode_frames.append(frame)

                if video_writer != None:
                    video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")
            
                # If the episode is successfule
                if all(dones):
                    collect_succ = True
                    break
            
            # write the collected episode to file
            if data_writer:
                # episode_data_collectors[0]["terminals"][-1] = np.array([True])
                # print(episode_data_collectors[0]["terminals"])
                for i in range(cfg.env.env_num):
                    # For each parallel env, write its collected trajectory as a demo
                    # Convert lists of dicts/arrays into dicts of lists of arrays first
                    episode_to_write = {
                        "obs": {key: np.array(episode_data_collectors[i]["obs"][key]) for key in data_writer.obs_keys},
                        "next_obs": {key: np.array(episode_data_collectors[i]["next_obs"][key]) for key in data_writer.obs_keys},
                        "actions": np.array(episode_data_collectors[i]["actions"]),
                        "rewards": np.array(episode_data_collectors[i]["rewards"]),
                        "dones": np.array(episode_data_collectors[i]["dones"]),
                        "acquisition": episode_data_collectors[i]["acquisition"],
                    }
                    data_writer.write_episode(episode_to_write, task_description=env_description)

            if data_writer_succ and collect_succ:
                for i in range(cfg.env.env_num):
                    # For each parallel env, write its collected trajectory as a demo
                    # Convert lists of dicts/arrays into dicts of lists of arrays first
                    episode_to_write = {
                        "obs": {key: np.array(episode_data_collectors[i]["obs"][key]) for key in data_writer_succ.obs_keys},
                        "next_obs": {key: np.array(episode_data_collectors[i]["next_obs"][key]) for key in data_writer_succ.obs_keys},
                        "actions": np.array(episode_data_collectors[i]["actions"]),
                        "rewards": np.array(episode_data_collectors[i]["rewards"]),
                        "dones": np.array(episode_data_collectors[i]["dones"]),
                        "acquisition": episode_data_collectors[i]["acquisition"],
                    }
                    data_writer_succ.write_episode(episode_to_write, task_description=env_description)

            if video_writer != None:
                # TASK 2: Mark the video of this rollout with 'inserted' tag when saving it
                # Create the video description with a tag if an action was inserted.
                video_description = env_description
                if adversarial_sampler:
                    if  inserted:
                        video_description += '_inserted'
                video_writer.get_last_info(num_env_rollout, dones, video_description, task_idx)
                video_writer.save()
            
            # log the success or failure
            print(f"Episode {num_env_rollout+1} finished with success: {dones[-1]}")

            for k in range(cfg.env.env_num):
                num_success += int(dones[k])
            all_succ += [num_success / cfg.env.env_num]
            all_horizon += [step_i+1]

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        all_env_descriptions.append(env_description)
    
    results = {}
    for idx, (env_idx, task_idx) in enumerate(zip(all_env_indices, all_task_indices)):
        results[f"rollout/horizon_env{env_idx}_task{task_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}_task{task_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}_task{task_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}_task{task_idx}"] = env_vid[idx]

    return results



def merge_results(results: List[dict], compute_avg=True):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v

    if compute_avg:
        merged_results["rollout/horizon_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/horizon_env" in k]).flatten())
        merged_results["rollout/success_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/success_env" in k]).flatten())
        
    return merged_results


def save_success_rate(epoch, success_metrics, summary_file_path):
    success_metrics = {k.replace("rollout/", ""): v for k, v in success_metrics.items()}
    success_heads = list(success_metrics.keys())
    success_heads.remove("success_env_avg")
    success_heads = sorted(success_heads, key=lambda x: int(x.split("success_env")[-1].split('task')[-1]))
    success_heads.append("success_env_avg")
    success_heads = ["epoch"] + success_heads

    success_metrics["epoch"] = str(epoch)
    df = pd.DataFrame(success_metrics, index=[0])

    if os.path.exists(summary_file_path):
        old_summary = pd.read_csv(summary_file_path)
        df = pd.concat([df, old_summary], ignore_index=True)

    df = df[success_heads]
    df.to_csv(summary_file_path)
