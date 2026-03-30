import subprocess
import sys
from pathlib import Path

config_dir = Path("./code/configs")
script_to_run = Path("./code/train.py")

config_files = list(config_dir.glob("experiment_1*.yaml"))

if not config_files:
    print(f"No matching YAML files found in {config_dir}")
    sys.exit(1)

for yaml_path in config_files:
    version_name = yaml_path.stem
    
    print(f"\nStarting: {version_name}")
    print(f"Config: {yaml_path}")

    command = [
        sys.executable, 
        str(script_to_run), 
        "--config", str(yaml_path), 
        "--version", version_name
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Finished: {version_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error in {version_name}: {e}")

print("\nAll tasks completed.")