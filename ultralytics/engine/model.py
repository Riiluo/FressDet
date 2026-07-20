from pathlib import Path
from typing import Union

import torch

from ultralytics.cfg import TASK2DATA
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import DEFAULT_CFG_DICT, checks, emojis, yaml_load

class Model(nn.Module):

    def __init__(self, model: Union[str, Path] = "yolov8n.pt", task: str = None, verbose: bool = False) -> None:
        super().__init__()
        self.model = None
        self.trainer = None
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.task = task

        model = str(model).strip()
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("AAA-FINALV4.2-DEMO only keeps model.train() and model.val(); predict is removed.")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose)
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        if "://" in weights:
            raise NotImplementedError("Remote weights are removed from AAA-FINALV4.2-DEMO.")
        weights = checks.check_model_file_from_stem(weights)
        if Path(weights).suffix != ".pt":
            raise TypeError("AAA-FINALV4.2-DEMO only supports YAML construction or PyTorch .pt weights.")

        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = task or self.model.args["task"]
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("AAA-FINALV4.2-DEMO only supports PyTorch nn.Module models.")

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model":
        raise NotImplementedError("External weight loading is removed from AAA-FINALV4.2-DEMO.")

    def val(self, validator=None, **kwargs):
        args = {**self.overrides, **kwargs, "mode": "val"}
        validator = (validator or self._smart_load("validator"))(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def train(self, **kwargs):
        self._check_is_pytorch_model()
        overrides = self.overrides
        custom = {
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }
        args = {**overrides, **custom, **kwargs, "mode": "train"}

        self.trainer = self._smart_load("trainer")(overrides=args)
        self.trainer.model = self.trainer.get_model(weights=None, cfg=self.model.yaml)
        self.model = self.trainer.model

        self.trainer.train()
        self.model = self.trainer.model
        self.overrides = vars(self.trainer.args)
        self.metrics = getattr(self.trainer.validator, "metrics", None)
        return self.metrics

    def _apply(self, fn) -> "Model":
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.overrides["device"] = self.device
        return self

    @property
    def names(self) -> list:
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        raise AttributeError("Model names are unavailable before train/val model setup.")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, "transforms") else None

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        include = {"imgsz", "data", "task"}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key: str):
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            raise NotImplementedError(emojis(f"WARNING ⚠️ '{name}' does not support {key} for task '{self.task}'.")) from e

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task map for your model.")
