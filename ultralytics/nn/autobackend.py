# Minimal class-name checker kept for dataset validation.

from ultralytics.utils import ROOT, yaml_load


def check_class_names(names):
    """Normalize dataset class names to the dict[int, str] form used by train/val."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, "
                f"but invalid class indices {min(names.keys())}-{max(names.keys())} were defined."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):
            names_map = yaml_load(ROOT / "cfg/datasets/ImageNet.yaml")["map"]
            names = {k: names_map[v] for k, v in names.items()}
    return names
