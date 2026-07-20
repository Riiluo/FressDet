from copy import copy
from typing import Dict

import torch

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel
from .val import OBBValidator


class OBBTrainer(BaseTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None):
        overrides = {} if overrides is None else overrides
        overrides["task"] = "obb"
        super().__init__(cfg, overrides)

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, mode == "train", rank)

    def preprocess_batch(self, batch: Dict) -> Dict:
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
        if batch["img"].shape[1] == 8:
            mean = torch.tensor(
                [81.59, 82.86, 81.93, 79.98, 82.54, 79.70, 81.44, 79.82],
                device=self.device,
                dtype=batch["img"].dtype,
            ).view(1, 8, 1, 1)
            std = torch.tensor(
                [48.81, 45.17, 41.96, 44.63, 47.32, 39.56, 43.19, 44.40],
                device=self.device,
                dtype=batch["img"].dtype,
            ).view(1, 8, 1, 1)
            batch["img"] = (batch["img"] - mean) / std
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        ch = self.data.get("channels", 8) if isinstance(self.data, dict) else 8
        return OBBModel(cfg, ch=ch, nc=self.data["nc"], verbose=verbose)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return OBBValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys

    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
