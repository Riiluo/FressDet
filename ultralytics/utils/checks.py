import glob
import inspect
import math
import re
from importlib import metadata
from pathlib import Path
from typing import Optional

import torch

from ultralytics.utils import (
    LINUX,
    LOGGER,
    MACOS,
    PYTHON_VERSION,
    ROOT,
    WINDOWS,
    SimpleNamespace,
    colorstr,
    emojis,
)

def parse_requirements(file_path=ROOT.parent / "requirements.txt", package=""):

    if package:
        requires = [x for x in metadata.distribution(package).requires if "extra == " not in x]
    else:
        requires = Path(file_path).read_text().splitlines()

    requirements = []
    for line in requires:
        line = line.strip()
        if line and not line.startswith("#"):
            line = line.split("#")[0].strip()
            match = re.match(r"([a-zA-Z0-9-_]+)\s*([<>!=~]+.*)?", line)
            if match:
                requirements.append(SimpleNamespace(name=match[1], specifier=match[2].strip() if match[2] else ""))

    return requirements

def parse_version(version="0.0.0") -> tuple:

    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0, 0, 0

def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):

    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    elif isinstance(imgsz, str):
        imgsz = [int(imgsz)] if imgsz.isnumeric() else eval(imgsz)
    else:
        raise TypeError(
            f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
            f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'"
        )

    if len(imgsz) > max_dim:
        msg = (
            "'train' and 'val' imgsz must be an integer, while 'predict' and 'export' imgsz may be a [h, w] list "
            "or an integer, i.e. 'yolo export imgsz=640,480' or 'yolo export imgsz=640'"
        )
        if max_dim != 1:
            raise ValueError(f"imgsz={imgsz} is not a valid image size. {msg}")
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]

    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    if sz != imgsz:
        LOGGER.warning(f"WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}")

    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz

def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    name: str = "version",
    hard: bool = False,
    verbose: bool = False,
    msg: str = "",
) -> bool:

    if not current:
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False

    if not required:
        return True

    if "sys_platform" in required and (
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True

    op = ""
    version = ""
    result = True
    c = parse_version(current)
    for r in required.strip(",").split(","):
        op, version = re.match(r"([^0-9]*)([\d.]+)", r).groups()
        v = parse_version(version)
        if op == "==" and c != v:
            result = False
        elif op == "!=" and c == v:
            result = False
        elif op in {">=", ""} and not (c >= v):
            result = False
        elif op == "<=" and not (c <= v):
            result = False
        elif op == ">" and not (c > v):
            result = False
        elif op == "<" and not (c < v):
            result = False
    if not result:
        warning = f"WARNING ⚠️ {name}{op}{version} is required, but {name}=={current} is currently installed {msg}"
        if hard:
            raise ModuleNotFoundError(emojis(warning))
        if verbose:
            LOGGER.warning(warning)
    return result

def check_python(minimum: str = "3.8.0", hard: bool = True) -> bool:

    return check_version(PYTHON_VERSION, minimum, name="Python", hard=hard)

def check_requirements(requirements=ROOT.parent / "requirements.txt", exclude=(), install=True, cmds=""):

    _ = install, cmds
    prefix = colorstr("red", "bold", "requirements:")
    check_python()
    if isinstance(requirements, Path):
        file = requirements.resolve()
        assert file.exists(), f"{prefix} {file} not found, check failed."
        requirements = [f"{x.name}{x.specifier}" for x in parse_requirements(file) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    pkgs = []
    for r in requirements:
        r_stripped = r.split("/")[-1].replace(".git", "")
        match = re.match(r"([a-zA-Z0-9-_]+)([<>!=~]+.*)?", r_stripped)
        name, required = match[1], match[2].strip() if match[2] else ""
        try:
            assert check_version(metadata.version(name), required)
        except (AssertionError, metadata.PackageNotFoundError):
            pkgs.append(r)

    if pkgs:
        LOGGER.warning(f"{prefix} missing or incompatible package(s): {pkgs}")
        return False

    return True

def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):

    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"

def check_model_file_from_stem(model="yolov8n"):

    if model and not Path(model).suffix:

        model_str = str(model).lower()
        if any(x in model_str for x in ['yolo', 'sam', 'fastsam', 'mobile', 'nas']):
            return Path(model).with_suffix(".pt")
    return model

def check_file(file, suffix="", download=True, download_dir=".", hard=True):

    _ = download, download_dir
    check_suffix(file, suffix)
    file = str(file).strip()
    if not file or Path(file).exists():
        return file
    if "://" in file:
        if hard:
            raise FileNotFoundError(f"Remote paths are not supported in AAA-FINALV4.2-DEMO: {file}")
        return []

    files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(str(ROOT.parent / file))
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(files) > 1 and hard:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if len(files) else []

def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):

    return check_file(file, suffix, hard=hard)

def print_args(args: Optional[dict] = None, show_file=True, show_func=False):

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))

def check_font(font="Arial.ttf"):
    """Return a local font path/name for data utilities compatibility."""
    return Path(font)

def is_ascii(s) -> bool:
    """Check whether a string is composed of only ASCII characters."""
    return len(str(s).encode().decode("ascii", "ignore")) == len(str(s))


def check_amp(model):
    return False
