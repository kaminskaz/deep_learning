import os
import random
import shutil
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from augmentations.augmentor import Augmentor 

def get_standard_augmentation(aug_name: str):
    augmentations = {
        'rotate': A.Rotate(limit=15, p=1.0),
        'blur': A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        'noise': A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
        'brightness': A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
        'flip': A.HorizontalFlip(p=1.0)
    }
    return A.Compose([augmentations[aug_name]])

def generate_isolated_dataset(
    source_dir: str, 
    target_dir: str, 
    k: int, 
    l: int, 
    method: str, 
    aug_name: str = 'random', 
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir) / f"{k}shot_{l}aug_{method}_{aug_name}"
    
    if target_path.exists():
        shutil.rmtree(target_path)
    
    classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])

    print(f"generating dataset: k={k}, l={l}, method={method}, aug={aug_name}")
    
    for cls in classes:
        cls_source_dir = source_path / cls
        cls_target_dir = target_path / cls
        cls_target_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = list(cls_source_dir.glob("*.png")) + list(cls_source_dir.glob("*.jpg"))
        all_images.sort() 
        
        if len(all_images) < k:
            continue

        k_images = random.sample(all_images, k)
        
        for img_path in k_images:
            shutil.copy(img_path, cls_target_dir / f"orig_{img_path.name}")
        
        
        if method == 'advanced':
            advanced_aug = Augmentor(p=1.0, seed=seed)
            
        aug_choices = ['rotate', 'blur', 'noise', 'brightness', 'flip']

        for i in range(l):
            base_img_path = random.choice(k_images)
            image = cv2.imread(str(base_img_path))
            if image is None: continue
                
            if method == 'standard':
                transform = get_standard_augmentation(aug_name)
                augmented_image = transform(image=image)['image']
                
            elif method == 'advanced':
                chosen_aug = random.choice(aug_choices)
                augmented_image = advanced_aug.augment_image(
                    image=image, 
                    mode='same',
                    x_splits_number=2, 
                    y_splits_number=2,
                    min_space_between_splits=5,
                    aug=chosen_aug 
                )
                
            save_name = f"aug_{i:04d}_{base_img_path.name}"
            cv2.imwrite(str(cls_target_dir / save_name), augmented_image)