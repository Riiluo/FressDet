from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from ultralytics.utils import (
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    LOGGER,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    RUNS_DIR,
    ROOT,
    colorstr,
    yaml_load,
)

MODES = {"train", "val"}
TASKS = {"detect", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
    "obb": "yolo11n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

CFG_FLOAT_KEYS = {
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "batch",
}
CFG_BOUNDED_FLOAT_KEYS = {
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "conf",
    "iou",
}
CFG_INT_KEYS = {
    "epochs",
    "patience",
    "workers",
    "seed",
    "max_det",
    "nbs",
}
CFG_BOOL_KEYS = {
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "cos_lr",
    "val",
}

def cfg2dict(cfg):

    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg

def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):

    cfg = cfg2dict(cfg)

    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}

    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    check_cfg(cfg)

    return IterableSimpleNamespace(**cfg)

def check_cfg(cfg, hard=True):

    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_BOUNDED_FLOAT_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)

def get_save_dir(args, name=None):

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok)

    return Path(save_dir)

def _handle_deprecation(custom):

    return custom

def check_dict_alignment(base: Dict, custom: Dict, e=None):

    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e
