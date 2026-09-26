import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from .modules.rgb_modules import *
from .modules.language_modules import *
from .modules.transformer_modules import *
from .base_policy import BasePolicy
from .dp_modules.difussion_policy_head import DPHead


###############################################################################
#
# A model handling extra input modalities besides images at time t.
#
###############################################################################


class ExtraModalityTokens(nn.Module):
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_num_layers=0,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):
        """
        This is a class that maps all extra modality inputs into tokens of the same size
        """
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.num_extra = int(use_joint) + int(use_gripper) + int(use_ee)

        extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )

        assert extra_low_level_feature_dim > 0, "[error] no extra information"

        self.extra_encoders = {}

        def generate_proprio_mlp_fn(modality_name, extra_low_level_feature_dim):
            assert extra_low_level_feature_dim > 0  # we indeed have extra information
            if extra_num_layers > 0:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_hidden_size)]
                for i in range(1, extra_num_layers):
                    layers += [
                        nn.Linear(extra_hidden_size, extra_hidden_size),
                        nn.ReLU(inplace=True),
                    ]
                layers += [nn.Linear(extra_hidden_size, extra_embedding_size)]
            else:
                layers = [nn.Linear(extra_low_level_feature_dim, extra_embedding_size)]

            self.proprio_mlp = nn.Sequential(*layers)
            self.extra_encoders[modality_name] = {"encoder": self.proprio_mlp}

        for (proprio_dim, use_modality, modality_name) in [
            (joint_states_dim, self.use_joint, "joint_states"),
            (gripper_states_dim, self.use_gripper, "gripper_states"),
            (ee_dim, self.use_ee, "ee_states"),
        ]:

            if use_modality:
                generate_proprio_mlp_fn(modality_name, proprio_dim)

        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.extra_encoders.values()]
        )

    def forward(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []

        for (use_modality, modality_name) in [
            (self.use_joint, "joint_states"),
            (self.use_gripper, "gripper_states"),
            (self.use_ee, "ee_states"),
        ]:

            if use_modality:
                tensor_list.append(
                    self.extra_encoders[modality_name]["encoder"](
                        obs_dict[modality_name]
                    )
                )

        x = torch.stack(tensor_list, dim=-2)
        return x


class PerturbationAttention:
    """
    See https://arxiv.org/pdf/1711.00138.pdf for perturbation-based visualization
    for understanding a control agent.
    """

    def __init__(self, model, image_size=[128, 128], patch_size=[16, 16], device="cpu"):

        self.model = model
        self.patch_size = patch_size
        H, W = image_size
        num_patches = (H * W) // np.prod(patch_size)
        # pre-compute mask
        h, w = patch_size
        nh, nw = H // h, W // w
        mask = (
            torch.eye(num_patches)
            .view(num_patches, num_patches, 1, 1)
            .repeat(1, 1, patch_size[0], patch_size[1])
        )  # (np, np, h, w)
        mask = rearrange(
            mask.view(num_patches, nh, nw, h, w), "a b c d e -> a (b d) (c e)"
        )  # (np, H, W)
        self.mask = mask.to(device).view(1, num_patches, 1, H, W)
        self.num_patches = num_patches
        self.H, self.W = H, W
        self.nh, self.nw = nh, nw

    def __call__(self, data):
        rgb = data["obs"]["agentview_rgb"]  # (B, C, H, W)
        B, C, H, W = rgb.shape

        rgb_ = rgb.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)  # (B, np, C, H, W)
        rgb_mean = rgb.mean([2, 3], keepdims=True).unsqueeze(1)  # (B, 1, C, 1, 1)
        rgb_new = (rgb_mean * self.mask) + (1 - self.mask) * rgb_  # (B, np, C, H, W)
        rgb_stack = torch.cat([rgb.unsqueeze(1), rgb_new], 1)  # (B, 1+np, C, H, W)

        rgb_stack = rearrange(rgb_stack, "b n c h w -> (b n) c h w")
        res = self.model(rgb_stack).view(B, self.num_patches + 1, -1)  # (B, 1+np, E)
        base = res[:, 0].view(B, 1, -1)
        others = res[:, 1:].view(B, self.num_patches, -1)

        attn = F.softmax(1e5 * (others - base).pow(2).sum(-1), -1)  # (B, num_patches)
        attn_ = attn.view(B, 1, self.nh, self.nw)
        attn_ = (
            F.interpolate(attn_, size=(self.H, self.W), mode="bilinear")
            .detach()
            .cpu()
            .numpy()
        )
        return attn_


###############################################################################
#
# A Transformer Policy
#
###############################################################################


class BCDPPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        embed_size = policy_cfg.embed_size
        transformer_input_sizes = []
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }

        self.encoders = nn.ModuleList(      # 2.563904 M
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        policy_cfg.language_encoder.network_kwargs.output_size = embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(      # 0.049216 M
            **policy_cfg.language_encoder.network_kwargs
        )

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalityTokens(       # 0.000704 M
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
            extra_num_layers=policy_cfg.extra_state_encoder.extra_num_layers,
            extra_hidden_size=policy_cfg.extra_state_encoder.extra_hidden_size,
            extra_embedding_size=embed_size,
        )

        ### 4. define temporal transformer
        policy_cfg.temporal_position_encoding.network_kwargs.input_size = embed_size
        self.temporal_position_encoding_fn = eval(
            policy_cfg.temporal_position_encoding.network
        )(**policy_cfg.temporal_position_encoding.network_kwargs)

        self.temporal_transformer = TransformerDecoder(     # 0.526848 M
            input_size=embed_size,
            num_layers=policy_cfg.temporal_transformer.transformer_num_layers,
            num_heads=policy_cfg.temporal_transformer.transformer_num_heads,
            head_output_size=policy_cfg.temporal_transformer.transformer_head_output_size,
            mlp_hidden_size=policy_cfg.temporal_transformer.transformer_mlp_hidden_size,
            dropout=policy_cfg.temporal_transformer.transformer_dropout,
        )

        ### 5. diffusion policy head
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.token_size = embed_size
        policy_head_kwargs.in_channels = shape_meta["ac_dim"]

        # self.policy_head = eval(policy_cfg.policy_head.network)(    #  85.886 M
        #     **policy_cfg.policy_head.loss_kwargs,
        #     **policy_cfg.policy_head.network_kwargs
        # )
        self.policy_head = DPHead(    #  85.886 M
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )

        self.latent_queue = []
        self.max_seq_len = policy_cfg.temporal_transformer.transformer_max_seq_len

    def temporal_encode(self, x, return_latent=False):
        pos_emb = self.temporal_position_encoding_fn(x)
        x = x + pos_emb.unsqueeze(1)  # (B, T, num_modality, E)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (B, T*num_modality, E)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)

        if return_latent:
            return x, x[:, :, 0]
        
        return x[:, :, 0]  # (B, T, E)

    def spatial_encode(self, data):
        # 1. encode extra
        extra = self.extra_encoder(data["obs"])  # (B, T, num_extra, E)

        # 2. encode language, treat it as action token
        B, T = extra.shape[:2]
        text_encoded = self.language_encoder(data)  # (B, E)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(
            -1, T, -1, -1
        )  # (B, T, 1, E)
        encoded = [text_encoded, extra]

        # 3. encode image
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            img_encoded = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, 1, -1)
            encoded.append(img_encoded)
        encoded = torch.cat(encoded, -2)  # (B, T, num_modalities, E)
        return encoded

    def forward(self, data, return_latent=False):
        x = self.spatial_encode(data)
        if return_latent:
            z, z0 = self.temporal_encode(x, return_latent=return_latent)
        else:
            z0 = self.temporal_encode(x)

        if return_latent:
            return x, z, z0
        
        return z0

    def diffusion_features(self, data):
        """Build the condition for a future action chunk.

        With ``frame_stack=H`` and ``seq_len=H``, SequenceDataset returns
        observations ``[t-H+1, ..., t, ..., t+H-1]`` but actions
        ``[t, ..., t+H-1]``.  Only the first H observations may condition the
        policy; the remaining observations are future information and must not
        leak into training.
        """
        x = self.spatial_encode(data)
        if getattr(self.cfg.policy, "action_chunk_mode", "aligned") == "future":
            history_size = int(self.cfg.data.frame_stack)
            if history_size < 1 or x.shape[1] < history_size:
                raise ValueError("future action chunks require a valid observation history")
            x = x[:, :history_size]
        encoded = self.temporal_encode(x)
        return encoded.mean(dim=1, keepdim=True)

    def get_action(self, cfg, data):
        chunks, _ = self.get_sampled_actions(
            data,
            num_samples=1,
            return_action_chunks=True,
            advance_history=True,
        )
        execution_index = int(getattr(cfg.sampler, "execution_index", 0))
        actions = chunks[:, 0, execution_index]
        return actions.detach().squeeze(0).cpu().float().numpy()

    def reset(self):
        self.latent_queue = []



    def get_sampled_actions(
        self,
        data,
        num_samples=1,
        compute_log_prob=False,
        return_action_chunks=False,
        advance_history=False,
    ):
        """
        Generates a batch of action samples AND optionally their log probabilities.

        Args:
            data (dict): The observation data dictionary.
            num_samples (int): The number of action samples to generate.
            compute_log_prob (bool): If True, computes and returns the approximate
                                     log probability for each sampled action.

        Returns:
            torch.Tensor: Sampled actions (B, num_samples, action_dim).
            torch.Tensor or None: Approximate log probabilities (B, num_samples)
                                  or None if compute_log_prob is False.
        """
        self.eval()
        with torch.no_grad():
            # --- 1. Feature Extraction (same as before) ---
            # This part remains identical to the previous version
            data = self.preprocess_input(data, train_mode=False)
            x = self.spatial_encode(data)

            # The latent_queue is likely empty or has a short history when called from the sampler.
            # We must pad it to the model's expected sequence length.
            current_history = list(self.latent_queue)
            current_history.append(x) # Add the newest observation

            # If history is shorter than the max sequence length, pad from the front
            # by repeating the oldest available observation.
            padding_needed = self.max_seq_len - len(current_history)
            if padding_needed > 0:
                # Get the oldest observation we have to use for padding
                padding_tensor = current_history[0] 
                padding = [padding_tensor] * padding_needed
                padded_history = padding + current_history
            else:
                # If history is already full, just take the most recent items
                padded_history = current_history[-self.max_seq_len:]

            if advance_history:
                self.latent_queue.append(x)
                if len(self.latent_queue) > self.max_seq_len:
                    self.latent_queue.pop(0)

            # Candidate sampling advances this queue exactly once per environment step.
            x_temporal = torch.cat(padded_history, dim=1) # Shape: [B, max_seq_len, E]






            # self.latent_queue.append(x)
            # if len(self.latent_queue) > self.max_seq_len:
            #     self.latent_queue.pop(0)
            # x_temporal = torch.cat(self.latent_queue, dim=1)
            x_encoded = self.temporal_encode(x_temporal)
            features = x_encoded.mean(dim=1, keepdim=True)  # Shape: (B, 1, E)

            B = features.shape[0]
            if B != 1 and compute_log_prob:
                raise NotImplementedError("Log prob computation is only supported for B=1")

            # --- 2. Batched Sampling (same as before) ---
            # ... (The logic to generate `noise`, `expanded_features`, `flat_noise` is identical) ...
            noise_shape = (B * num_samples, self.cfg.policy.policy_head.network_kwargs.future_action_window_size + 1, 
                        self.cfg.policy.policy_head.network_kwargs.in_channels)
            noise = torch.randn(noise_shape, device=features.device)
            # Repeat the condition for each sample
            # `features` is (B, 1, E). After repeat, it's (B * num_samples, 1, E)
            condition_z = features.repeat_interleave(num_samples, dim=0)


            # expanded_features = features.unsqueeze(1).repeat(1, num_samples, 1, 1)
            model_kwargs = {"z": condition_z}

            # flat_noise = noise.view(B * num_samples, noise_shape[2], noise_shape[3])
            
            sample_fn = self.policy_head.net.forward
            if self.policy_head.ddim_diffusion is None:
                self.policy_head.create_ddim(ddim_step=10)

            # --- This call now just gets the final samples ---
            samples = self.policy_head.ddim_diffusion.ddim_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=features.device, eta=0.0
            ) # Output shape: (B * num_samples, T, action_dim)
            
            # Reshape and get the final action
            reshaped_samples = samples.view(B, num_samples, samples.shape[1], samples.shape[2])
            action_chunks = torch.clamp(reshaped_samples, -1.0, 1.0)
            execution_index = int(getattr(self.cfg.sampler, "execution_index", 0))
            actions = action_chunks[:, :, execution_index, :]


            log_probs = None
            if compute_log_prob:
                # --- 3. Compute Approximate Log Probability ---
                # To calculate the loss correctly, we must use the ENTIRE generated sequence,
                # not just the final action.
                
                # The `samples` tensor from the DDIM loop is already flat with shape
                # (B * num_samples, T, action_dim). This is the correct shape for x_start.
                # samples_for_loss = samples # Shape: (B * num_samples, T, action_dim)

                # We are calculating the negative ELBO, which is our loss.
                # The `sample_loss` function is now receiving the full sequence it expects.
                negative_elbos = self.policy_head.sample_loss(
                    x_start=samples, # <-- PASSES FULL T=10 SEQUENCE
                    z=condition_z,      # Use the same expanded features
                    reduction='none'          # Ensure it returns loss per sample
                ) # Expected output shape: (B * num_samples,)

                # The log probability is the negative of the loss
                log_probs = -negative_elbos

                # Reshape back to (B, num_samples)
                log_probs = log_probs.view(B, num_samples)

        return (action_chunks if return_action_chunks else actions), log_probs

    def sample_candidates(self, observation, num_candidates):
        """PolicyAdapter API: sample DiT chunks with shape [B, K, H, A]."""
        chunks, _ = self.get_sampled_actions(
            observation,
            num_samples=num_candidates,
            compute_log_prob=False,
            return_action_chunks=True,
            advance_history=True,
        )
        return chunks

    def per_sample_loss(self, batch, augmentation=None):
        """PolicyAdapter API: native DiT diffusion loss for each batch item."""
        data = self.preprocess_input(batch, augmentation=augmentation)
        features = self.diffusion_features(data)
        repeats = self.cfg.policy.policy_head.network_kwargs.repeated_diffusion_steps
        actions = data["actions"]
        batch_size = actions.shape[0]
        losses = self.policy_head.sample_loss(
            actions.repeat(repeats, 1, 1),
            features.repeat(repeats, 1, 1),
            reduction="none",
        )
        return losses.reshape(repeats, batch_size).mean(dim=0), data
