# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import os
from pathlib import Path


def increment_path(path, exist_ok=False, sep="", mkdir=False):

    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def get_latest_run(search_dir="."):
    """Returns the path to the most recent 'last.pt' file in the specified directory for resuming training."""
    last_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ""
