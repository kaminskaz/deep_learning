import os
import random
import shutil
from pathlib import Path

def generate_50shot_baseline(source_dir: str, target_dir: str, seed: int = 42):
    random.seed(seed)
    source_path = Path(source_dir)
    target_path = Path(target_dir) / "50shot_0aug_baseline"
    
    if target_path.exists():
        shutil.rmtree(target_path)
    
    classes = sorted([d.name for d in source_path.iterdir() if d.is_dir()])
    
    for cls in classes:
        cls_source_dir = source_path / cls
        cls_target_dir = target_path / cls
        cls_target_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = list(cls_source_dir.glob("*.png")) + list(cls_source_dir.glob("*.jpg"))
        all_images.sort()
        
        if len(all_images) < 50:
            print(f"Skipping {cls}: Not enough images.")
            continue
            
        k_images = random.sample(all_images, 50)
        
        for img_path in k_images:
            shutil.copy(img_path, cls_target_dir / f"orig_{img_path.name}")

    print(f"Saved to: {target_path}")

if __name__ == "__main__":
    generate_50shot_baseline(source_dir="./data/train", target_dir="./data/augmented_experiments")