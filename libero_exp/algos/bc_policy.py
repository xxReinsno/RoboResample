import torch
from tqdm import tqdm

from .base import BaseAlgo


class BC_Policy(BaseAlgo):
    def __init__(self, cfg, inference=False, device='cuda'):
        super().__init__(cfg, inference, device)

    def forward_backward(self, data):
        bc_loss = self.compute_loss(data)
        loss = bc_loss

        self.optimizer.zero_grad()
        self.fabric.backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.train.grad_clip)
        self.optimizer.step()

        ret_dict = {
            "loss": loss.item(),
            "bc_loss": bc_loss.item(),
        }

        return ret_dict

    def compute_loss(self, data, augmentation=None):
        data = self.model.preprocess_input(data, augmentation=augmentation)
        if self.cfg.policy.policy_type == 'BCDPPolicy':
            repeated_diffusion_steps = self.cfg.policy.policy_head.network_kwargs.repeated_diffusion_steps
            actions = data["actions"]
            actions_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            features = self.model.diffusion_features(data)
            features_repeated = features.repeat(repeated_diffusion_steps, 1, 1)
            bc_loss = self.model.policy_head.loss(actions_repeated, features_repeated)
            return bc_loss

        x, z, dist = self.model(data, return_latent=True)
        if self.cfg.policy.policy_type == 'BCMLPPolicy':
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"][:, -1], reduction="mean")
        else:
            bc_loss = self.model.policy_head.loss_fn(dist, data["actions"], reduction="mean")

        return bc_loss
    
    @torch.no_grad()
    def evaluate(self, tag="val"):
        cfg = self.cfg
        tot_loss_dict, tot_items = {}, 0
        self.model.eval()

        print('Evaluating...')
        for data in tqdm(self.val_loader):
            out = self.compute_loss(data, augmentation=False)

            if isinstance(out, (tuple, list)) and len(out) == 3:
                bc_loss, mmd, kl_div = out
            else:
                bc_loss = out
                mmd = torch.tensor(0.0)
                kl_div = torch.tensor(0.0)

            loss = bc_loss

            ret_dict = {
                "loss": loss.item(),
                "bc_loss": bc_loss.item(),
                "mmd": mmd.item(),
                "kl_div": kl_div.item(),
            }

            for k, v in ret_dict.items():
                if k not in tot_loss_dict:
                    tot_loss_dict[k] = 0
                tot_loss_dict[k] += v
            tot_items += 1

            if cfg.train.debug:
                break

        # print('Evaluating...')
        # for data in tqdm(self.val_loader):
        #     bc_loss, mmd, kl_div = self.compute_loss(data, augmentation=False)
        #     loss = bc_loss

        #     ret_dict = {
        #         "loss": loss.item(),
        #         "bc_loss": bc_loss.item(),
        #         "mmd": mmd.item(),
        #         "kl_div": kl_div.item(),
        #     }

        #     for k, v in ret_dict.items():
        #         if k not in tot_loss_dict:
        #             tot_loss_dict[k] = 0
        #         tot_loss_dict[k] += v
        #     tot_items += 1

        #     if cfg.train.debug:
        #         break

        out_dict = {}
        for k, v in tot_loss_dict.items():
            out_dict[f"{tag}/{k}"] = tot_loss_dict[f"{k}"] / tot_items

        return out_dict
