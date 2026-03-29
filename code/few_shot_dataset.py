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
    if aug_name not in augmentations:
        raise ValueError(f"Unknown standard augmentation: {aug_name}")
    return A.Compose([augmentations[aug_name]])

def generate_isolated_dataset(
    source_dir: str, 
    target_dir: str, 
    k: int, 
    l: int, 
    method: str, 
    aug_name: str = 'random', # Default to random for advanced
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir) / f"{k}shot_{l}aug_{method}_{aug_name}"
    
    if target_path.exists():
        print(f"Warning: Target directory {target_path} already exists. Overwriting...")
        shutil.rmtree(target_path)
    
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    advanced_aug = Augmentor(p=1.0, seed=seed) if method == 'advanced' else None
    aug_choices = ['rotate', 'blur', 'noise', 'brightness', 'flip']

    print(f"\n--- Generating dataset: k={k}, l={l}, method={method}, aug={aug_name} ---")
    
    for cls in classes:
        cls_source_dir = source_path / cls
        cls_target_dir = target_path / cls
        cls_target_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = list(cls_source_dir.glob("*.png")) + list(cls_source_dir.glob("*.jpg"))
        if len(all_images) < k:
            continue # Skip if not enough images
            
        k_images = random.sample(all_images, k)
        
        # Save originals
        for img_path in k_images:
            shutil.copy(img_path, cls_target_dir / f"orig_{img_path.name}")
            
        # Generate augmented samples
        for i in range(l):
            base_img_path = random.choice(k_images)
            image = cv2.imread(str(base_img_path))
            if image is None: continue
                
            if method == 'standard':
                transform = get_standard_augmentation(aug_name)
                augmented_image = transform(image=image)['image']
                
            elif method == 'advanced':
                # Randomly pick ONE augmentation for this specific image
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

    print(f"Saved to: {target_path}")

if __name__ == "__main__":
    SOURCE_DATA_DIR = "./data/train" 
    TARGET_DATA_DIR = "./data/augmented_experiments"
    augmentation_types = ['rotate', 'blur', 'noise', 'brightness', 'flip']
    
    k_val = 10  
    l_val = 50  
    
    # 1. Standard Method: One dataset PER augmentation
    for aug_type in augmentation_types:
        generate_isolated_dataset(
            source_dir=SOURCE_DATA_DIR, target_dir=TARGET_DATA_DIR,
            k=k_val, l=l_val, method='standard', aug_name=aug_type
        )
        
    # 2. Advanced Method: ONE dataset total, randomly picking augmentations per image
    generate_isolated_dataset(
        source_dir=SOURCE_DATA_DIR, target_dir=TARGET_DATA_DIR,
        k=k_val, l=l_val, method='advanced', aug_name='mixed'
    )