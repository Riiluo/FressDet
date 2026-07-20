# Ultralytics YOLO, trimmed for AAA-FINALV4.2-DEMO train/val only.

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import OBBModel


class YOLO(Model):
    """Minimal YOLO wrapper used only for OBB training and validation."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        return {
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
            },
        }
