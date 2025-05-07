from dataclasses import dataclass

from config.DatasetConfig import DataConfig

from omegaconf import MISSING


@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = "eso18-imperial-college-london"
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_project_name: str = "mvvae_clf"
    wandb_log_freq: int = 2
    wandb_offline: bool = False

    # logs
    dir_logs: str = "/rds/general/user/eso18/home/mmvmvae/clfs"


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 32 # 256
    batch_size_eval: int = 32 # 256
    lr: float = 1e-3
    epochs: int = 10


@dataclass
class MyClfConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/mean_acc" # was "val/loss/mean_metric" "val/loss/mean_acc" for CelebA
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
