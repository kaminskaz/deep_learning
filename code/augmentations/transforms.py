from torchvision import transforms
from typing import Optional

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD  = [0.24205776, 0.23828046, 0.25874835]

BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CINIC_MEAN, CINIC_STD),
])

PRESETS = {
    'none': BASE_TRANSFORM,
    'light': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CINIC_MEAN, CINIC_STD),
    ]),
    'medium': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(CINIC_MEAN, CINIC_STD),
    ]),
    'heavy': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(CINIC_MEAN, CINIC_STD),
    ]),
}

def get_transform(
    split:            str,
    preset:           Optional[str]      = None
) -> transforms.Compose:
    """
    Val/test always return BASE_TRANSFORM.
    Train priority: preset > BASE_TRANSFORM.
    """
    if split in ('val', 'test'):
        return BASE_TRANSFORM

    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
        return PRESETS[preset]

    return BASE_TRANSFORM