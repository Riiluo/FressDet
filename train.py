import warnings
from pathlib import Path

from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "ultralytics/cfg/models/FressDet.yaml"
DATA = ROOT / "ultralytics/cfg/datasets/drmod.yaml"

if __name__ == "__main__":
    YOLO(str(MODEL), task="obb").train(
        data=str(DATA),
        imgsz=1200,
        epochs=20,
        batch=12,
        workers=32,
        device="0,1",
        project="runs",
        name="fressdet",
    )
