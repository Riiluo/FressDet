# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.9"

import os

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.models import YOLO
from ultralytics.utils import ASSETS, SETTINGS


settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "checks",
    "settings",
)
