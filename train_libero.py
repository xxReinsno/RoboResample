import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['MUJOCO_GL'] = 'osmesa'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' 
# os.environ['MUJOCO_GL'] = 'egl'             
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import hydra
from hydra.core.hydra_config import HydraConfig
import warnings
import lightning
from omegaconf import DictConfig, OmegaConf

from libero_exp.algos import *


@hydra.main(config_path="libero_exp/configs/bc_policy", config_name="dp", version_base=None)
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    cfg.experiment_dir = work_dir
    warnings.simplefilter("ignore")
    lightning.seed_everything(cfg.train.seed)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))
    
    algo = get_algo_class(cfg.algo.algo_type)(cfg)
    algo.train()


if __name__ == "__main__":
    main()
