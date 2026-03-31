import os
import random
import shutil
from pathlib import Path

def create_mixed_from_existing(experiments_dir: str, k: int = 10, l: int = 50, seed: int = 42):
    random.seed(seed)
    base_path = Path(experiments_dir)
    
    src_rotate = base_path / f"{k}shot_{l}aug_standard_rotate"
    src_flip = base_path / f"{k}shot_{l}aug_standard_flip"
    src_noise = base_path / f"{k}shot_{l}aug_standard_noise"
    
    target_path = base_path / f"{k}shot_{l}aug_standard_mixed"
    
            
    if target_path.exists():
        shutil.rmtree(target_path)
        
    classes = [d.name for d in src_rotate.iterdir() if d.is_dir()]
    
    for cls in classes:
        (target_path / cls).mkdir(parents=True, exist_ok=True)
        
        orig_files = list((src_rotate / cls).glob("orig_*"))
        for f in orig_files:
            shutil.copy(f, target_path / cls / f.name)
            
        aug_files = list((src_rotate / cls).glob("aug_*"))
        
        for f in aug_files:
            chosen_src_dir = random.choice([src_rotate, src_flip, src_noise])
            src_file = chosen_src_dir / cls / f.name
            
            if src_file.exists():
                shutil.copy(src_file, target_path / cls / f.name)
            else:
                print(f"{src_file} not found")
                

if __name__ == "__main__":
    EXPERIMENTS_DIR = "./data/augmented_experiments"
    
    create_mixed_from_existing(
        experiments_dir=EXPERIMENTS_DIR,
        k=10,
        l=50
    )