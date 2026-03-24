import subprocess
import sys
from pathlib import Path

# 1. Setup paths
# pathlib automatically handles the "//" or "/" slash differences
config_dir = Path("./code/configs")
script_to_run = Path("./code/train.py")

# 2. Find all matching files
# This looks for files starting with 'experiment_1' and ending in '.yaml'
config_files = list(config_dir.glob("experiment_1*.yaml"))

if not config_files:
    print(f"No matching YAML files found in {config_dir}")
    sys.exit(1)

for yaml_path in config_files:
    # .stem returns the filename without the extension (e.g., 'experiment_1_v1')
    version_name = yaml_path.stem
    
    print(f"\n🚀 Starting: {version_name}")
    print(f"Config: {yaml_path}")
    
    # 3. Construct and run the command
    # sys.executable ensures we use the same Python interpreter currently running
    command = [
        sys.executable, 
        str(script_to_run), 
        "--config", str(yaml_path), 
        "--version", version_name
    ]
    
    try:
        # check=True will raise an error if the script fails (non-zero exit code)
        subprocess.run(command, check=True)
        print(f"✅ Finished: {version_name}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {version_name}: {e}")
        # Optional: break or continue depending on if you want to stop on failure
        # break 

print("\nAll tasks completed.")