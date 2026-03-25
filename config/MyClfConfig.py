from dataclasses import dataclass
from config.DatasetConfig import DataConfig
from config.UserVariables import folder_path, wandb_entity
from omegaconf import MISSING

@dataclass
class LogConfig:
    # wandb
    wandb_entity: str = wandb_entity
    wandb_group: str = ""
    wandb_run_name: str = ""
    wandb_project_name: str = "mvvae_clf"
    wandb_log_freq: int = 50
    wandb_offline: bool = False
    # logs
    dir_logs: str = folder_path + "clfs"

@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 256
    batch_size_eval: int = 256
    lr: float = 1e-3
    epochs: int = 50
    seed: int = 1


@dataclass
class MyClfConfig:
    seed: int = 0
    checkpoint_metric: str = "val/loss/loss"
    model: ModelConfig = MISSING
    log: LogConfig = MISSING
    dataset: DataConfig = MISSING
