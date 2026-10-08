import os
import sys

# Automatically locate ALL nvidia sub-library paths in this Conda environment
site_pkgs = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
nvidia_dir = os.path.join(site_pkgs, "nvidia")

nvidia_libs = []
if os.path.exists(nvidia_dir):
    for pkg in os.listdir(nvidia_dir):
        lib_path = os.path.join(nvidia_dir, pkg, "lib")
        if os.path.isdir(lib_path):
            nvidia_libs.append(lib_path)

# Append to LD_LIBRARY_PATH before any CUDA runtime loads
os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Suppress warnings & load bitsandbytes using CUDA 12
os.environ["BNB_CUDA_VERSION"] = "121"
# os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
# os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
# os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"

import shutil
from pathlib import Path

# 1. Target directory
TARGET_DIR = "/home/ghaith-mhamdi/ghaith_mjlab_project/src/ghaith_r1_wire/model/roboflow_cache"
os.makedirs(TARGET_DIR, exist_ok=True)

# 2. Tell Roboflow to download and store everything in your target directory
os.environ["MODEL_CACHE_DIR"] = TARGET_DIR

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "mhamdis-workspace/connector-hole-keypoints-4-rfdetr-keypoint-preview-t1"

if not ROBOFLOW_API_KEY:
    raise RuntimeError("Please set ROBOFLOW_API_KEY environment variable!")

print(f"Downloading model '{ROBOFLOW_MODEL_ID}' into {TARGET_DIR}...")

from inference import get_model

# Fetch model
model = get_model(model_id=ROBOFLOW_MODEL_ID, api_key=ROBOFLOW_API_KEY)

print("\nSUCCESS: Model successfully downloaded and cached locally!")
print(f"Location: {TARGET_DIR}")