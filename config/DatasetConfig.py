from dataclasses import dataclass, field
from typing import List
from omegaconf import MISSING
from config.UserVariables import folder_path

@dataclass
class DataConfig:
    name: str = MISSING
    num_workers: int = 8
    # num views
    num_views: int = MISSING


@dataclass
class PolyMNISTDataConfig(DataConfig):
    num_views: int = 5
    modalities_order: int = 0
    dir_data_base: str = folder_path + "data"
    dir_clfs_base: str = (
        folder_path + "trained_classifiers/trained_clfs_polyMNIST"
    )
    n_clfs_outputs: int = 10
    num_labels: int = 1


@dataclass
class PMtranslatedData75Config(PolyMNISTDataConfig):
    name: str = "PM_translated75"
    suffix_data_train: str = "MMNIST/train" 
    suffix_data_test: str = "MMNIST/test" 
    suffix_clfs: str = "translated75_resnet"


@dataclass
class CelebADataConfig(DataConfig):
    name: str = "celeba"
    num_views: int = 2
    dir_data: str = folder_path + "data/CelebA"
    dir_alphabet: str = (
        folder_path + "utils"
    )
    dir_clf: str = (
        folder_path + "trained_classifiers/trained_clfs_celeba"
    )
    dir_clfs_base: str = (
        folder_path + "trained_classifiers/trained_clfs_celeba"
    )

    len_sequence: int = 256
    random_text_ordering: bool = False
    random_text_startindex: bool = True
    img_size: int = 64
    image_channels: int = 3
    crop_size_img: int = 148
    n_clfs_outputs: int = 40
    num_labels: int = 40

    num_features: int = 41  # len(alphabet)
    num_layers_img: int = 5
    num_layers_text: int = 7
    filter_dim_img: int = 64
    filter_dim_text: int = 64
    skip_connections_img_weight_a: float = 1.0
    skip_connections_img_weight_b: float = 1.0
    skip_connections_text_weight_a: float = 1.0
    skip_connections_text_weight_b: float = 1.0

    use_rec_weight: bool = True
    include_channels_rec_weight: bool = False
    
@dataclass
class scMNCDataConfig(DataConfig):
    name: str = "scMNC"
    num_views: int = 2
    dir_data: str = folder_path + "data/scMNC"
    num_labels: int = 6
    dir_clf: str = (
        folder_path + "trained_classifiers/trained_clfs_scMNC"
    )
    n_clfs_outputs: int = 6
    label_names: List[str] = field(
        default_factory=lambda: [
            "blue2red",
            "brown",
            "grey",
            "yellow",
            "black",
            "white",
        ]
    )
