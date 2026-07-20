import os
import warnings
from pathlib import Path

from ultralytics import YOLO

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ultralytics/cfg/datasets/drmod.yaml"
WEIGHTS = ROOT / "best.pt"


if __name__ == "__main__":
    YOLO(str(WEIGHTS), task="obb").val(
        data=str(DATA),
        imgsz=1200,
        batch=12,
        workers=32,
        device="0",
        project="runs/fressdet_val",
        name="fressdet_val",
        split="val",
    )
