import os
import random
import shutil
from pathlib import Path

def create_mixed_from_existing(experiments_dir: str, k: int = 10, l: int = 50, seed: int = 42):
    random.seed(seed)
    base_path = Path(experiments_dir)
    
    # Define the source directories from Phase 1
    src_rotate = base_path / f"{k}shot_{l}aug_standard_rotate"
    src_flip = base_path / f"{k}shot_{l}aug_standard_flip"
    src_noise = base_path / f"{k}shot_{l}aug_standard_noise"
    
    # Define the target directory for Phase 2
    target_path = base_path / f"{k}shot_{l}aug_standard_mixed"
    
    # Validate that all source datasets exist
    for src in [src_rotate, src_flip, src_noise]:
        if not src.exists():
            raise FileNotFoundError(f"Missing source dataset: {src}. Did you run Phase 1 for this augmentation?")
            
    if target_path.exists():
        print(f"Warning: Target directory {target_path} already exists. Overwriting...")
        shutil.rmtree(target_path)
        
    classes = [d.name for d in src_rotate.iterdir() if d.is_dir()]
    print(f"\n--- Mixing datasets to create: {target_path.name} ---")
    
    for cls in classes:
        (target_path / cls).mkdir(parents=True, exist_ok=True)
        
        # 1. Copy the k original images (we can safely pull these from any source folder)
        orig_files = list((src_rotate / cls).glob("orig_*"))
        for f in orig_files:
            shutil.copy(f, target_path / cls / f.name)
            
        # 2. Mix the l augmented images
        # Since Phase 1 used a fixed seed, the filenames (e.g., aug_0000_img.jpg) are identical across all 3 source folders.
        aug_files = list((src_rotate / cls).glob("aug_*"))
        
        for f in aug_files:
            # Randomly select which augmentation type to use for this specific file
            chosen_src_dir = random.choice([src_rotate, src_flip, src_noise])
            src_file = chosen_src_dir / cls / f.name
            
            if src_file.exists():
                shutil.copy(src_file, target_path / cls / f.name)
            else:
                print(f"⚠️ Warning: {src_file} not found!")
                
    print(f"✅ Successfully created mixed dataset with {l} augmentations per class.")

if __name__ == "__main__":
    EXPERIMENTS_DIR = "./data/augmented_experiments"
    
    # Run the mixer
    create_mixed_from_existing(
        experiments_dir=EXPERIMENTS_DIR,
        k=10,
        l=50
    )