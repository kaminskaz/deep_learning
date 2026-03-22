import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable

from augmentations.augmentor import Augmentor
from configs.config import AugmentorConfig
from augmentations.transforms import get_transform


class DatasetWrapper(Dataset):
    def __init__(
        self,
        dataset,
        split: str = 'train',
        preset: Optional[str] = None,
        augmentor_config: Optional[AugmentorConfig] = None,
        seed: int = 42
    ):
        self.dataset = dataset
        self.split = split
        self.config = augmentor_config
        self.augmentor = None
        if self.config is not None and preset is not None:
            print("Warning: Both preset and augmentor provided. Preset will be ignored in favor of custom augmentor.")
            self.augmentor = Augmentor(
                p=self.config.p,
                seed=seed
            )
            preset = None
        self.transform = get_transform(split, preset)

    def __len__(self):
        return len(self.dataset)

    def num_classes(self) -> int:
        return len(self.dataset.classes)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.augmentor is not None and self.split == 'train':
            image = self._apply_augmentor(image)
    
        image = self.transform(image)
        return image, label

    def _apply_augmentor(self, image) -> Image.Image:
        """Convert to numpy, apply Augmentor, return PIL Image."""
        image_np = np.array(image)
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3, 4]:
            image_np = np.transpose(image_np, (1, 2, 0))

        if self.config.mode not in ('same', 'different', 'combine'):
            raise ValueError(
                f"Invalid mode '{self.config.mode}'. "
                f"Choose from: 'same', 'different', 'combine'."
            )

        image_np = self.augmentor.augment_image(
            image_np,
            mode=self.config.mode,
            x_splits_number=self.config.x_splits_number,
            y_splits_number=self.config.y_splits_number
        )
        return Image.fromarray(image_np)