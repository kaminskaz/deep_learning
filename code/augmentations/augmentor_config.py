from dataclasses import dataclass, field
from typing import Optional
import albumentations as A

@dataclass
class AugmentorConfig:
    mode:            str        = 'different'
    p:               float      = 0.5
    x_splits_number: int        = 0
    y_splits_number: int        = 0
    aug:             Optional[A.Compose] = None