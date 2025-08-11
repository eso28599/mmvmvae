from dataclasses import dataclass


@dataclass
class ModelConfig:
    device: str = "cuda"
    batch_size: int = 256 # 128
    batch_size_eval: int = 64
    lr: float = 5e-4 # 5e-4
    epochs: int = 350 #500
    temp_annealing: str = "cosine"
    
    latent_dim: int = 256

    # loss hyperparameters
    beta: float = 1.0

    # network architectures
    use_resnets: bool = True


@dataclass
class JointModelConfig(ModelConfig):
    name: str = "joint"
    aggregation: str = "poe"


@dataclass
class MixedPriorModelConfig(ModelConfig):
    name: str = "mixedprior"
    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True # True
    init_alpha_value: float = 1.0 # 1
    final_alpha_value: float = 0 # 0
    alpha_annealing_steps: int = 150000
    
@dataclass
class JointPriorModelConfig(ModelConfig):
    name: str = "jointprior"
    # weight on N(0,1) in mixed prior
    alpha_annealing: bool = True # True
    init_alpha_value: float = 1.0 # 1
    final_alpha_value: float = 0 # 0
    alpha_annealing_steps: int = 150000
    cov_scalar: float =  1 - 0.95 ** 2 # 1 - alpha^2
    alpha_scalar: float = 0.95


@dataclass
class UnimodalModelConfig(ModelConfig):
    name: str = "unimodal"


@dataclass
class SplitModelConfig(ModelConfig):
    name: str = "split"
    aggregation: str = "moe"
    mod_specific_dim: int = 256
    split_type: str = "simple"
