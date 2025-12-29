import os
import runpy
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../models/research/object_detection"))

if target_dir not in sys.path:
    sys.path.insert(0, target_dir)

if __name__ == "__main__":
    real_script_path = os.path.join(target_dir, "exporter_main_v2.py")

    if not os.path.exists(real_script_path):
        raise FileNotFoundError(f"Could not find the real script at: {real_script_path}")

    # Use runpy to execute the script in the current __main__ namespace
    runpy.run_path(real_script_path, run_name="__main__")
