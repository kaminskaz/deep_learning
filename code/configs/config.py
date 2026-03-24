from dataclasses import dataclass, field
from typing import Optional, List
import yaml


@dataclass
class AugmentorConfig:
    mode: str = 'different'
    p: float = 0.5
    seed: int = 42
    x_splits_number: int = 3
    y_splits_number: int = 3


@dataclass
class TrainingConfig:
    # model
    model: str = 'resnet50'
    num_classes: int = 10
    dropout_rate: float = 0.5

    # training
    num_epochs: int = 30
    batch_size: int = 64
    patience: int = 5
    seed: int = 42
    device: str = 'cpu'

    # optimizer
    optimizer: str = 'adam'
    learning_rate: float = 0.05
    momentum: float = 0.9

    # regularization
    regularization: str = 'l2' 
    lambda_param: float = 1e-4

    # scheduler
    scheduler: str = 'cosine'
    t_max: int = 30

    # data
    train_dir: str = './data/train'
    val_dir: str = './data/valid'

    # transforms
    preset: Optional[str] = None

    # augmentor
    augmentor_config: Optional[str] = None   # path to augmentor yaml

    #visualization
    show_augmentation_preview: bool = True
    preview_samples: int = 3

    # paths
    model_path: str = './checkpoints'


def load_config(path: str) -> TrainingConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return TrainingConfig(**raw)


def load_augmentor_config(path: str) -> AugmentorConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return AugmentorConfig(**raw)