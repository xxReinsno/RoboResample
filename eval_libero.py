import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'     
# os.environ['MUJOCO_GL'] = 'egl'             
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import datetime
torch.distributed.constants._DEFAULT_PG_TIMEOUT = datetime.timedelta(seconds=5000)

import hydra
from hydra.core.hydra_config import HydraConfig
import warnings
import lightning
from omegaconf import DictConfig

from libero_exp.algos import *


@hydra.main(config_path="libero_exp/configs/bc_policy", config_name="dp", version_base=None)
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    cfg.experiment_dir = work_dir
    warnings.simplefilter("ignore")
    lightning.seed_everything(cfg.train.seed)

    algo = get_algo_class(cfg.algo.algo_type)(cfg, inference=True, device='cuda')  # ['cpu', 'cuda']  
    algo.inference()
    

if __name__ == "__main__":
    main()
