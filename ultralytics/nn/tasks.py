import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    Concat,
    Conv,
    OBB,
    OAHead,
)
from ultralytics.nn.neck.RECluster import ReCoW, ReCoWAAA
from ultralytics.nn.backbone.SpetialImplicitWarp import SpeIWMetaformerStage
from ultralytics.nn.redet.redet import (
    AREGUp,
    DS2GCBA,
    PatchMerging,
    RELiftGCBA,
    RESPPF,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8OBBLoss
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    model_info,
)

OBB_CLASS = (OBB,)
REDET_CLASS = (RELiftGCBA, RESPPF, DS2GCBA)


class BaseModel(nn.Module):
    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self._forward_once(x)

    def _forward_once(self, x):
        y = []  # outputs
        for idx, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if hasattr(m, 'backbone'):  # False
                x = m(x)
                for _ in range(5 - len(x)):
                    x.insert(0, None)
                for i_idx, i in enumerate(x):
                    if i_idx in self.save:
                        y.append(i)
                    else:
                        y.append(None)
                x = x[-1]
            else:
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

        return x

    def fuse(self, verbose=True):
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, Conv) and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, OBB_CLASS):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):

    def __init__(self, cfg="yolov8n.yaml", ch=8, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, OBB_CLASS):
            s = 640  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                output = self.forward(x)
                if isinstance(output, dict):
                    output = output.get("one2many", output)
                if isinstance(output, tuple):
                    output = output[0]
                return output

            try:
                m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(2, ch, s, s))])  # forward
            except (RuntimeError, ValueError) as e:
                if 'Not implemented on the CPU' in str(e) or 'Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)' in str(e) or \
                'CUDA tensor' in str(e) or 'is_cuda()' in str(e) or 'carafe_forward_impl' in str(e) or 'Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)' in str(e):
                    self.model.to(torch.device('cuda'))
                    m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(2, ch, s, s).to(torch.device('cuda')))])  # forward
                else:
                    raise e
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    def init_criterion(self):
        raise NotImplementedError("This demo only supports OBB training.")

    def net_update_temperature(self, temp):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)

class OBBModel(DetectionModel):

    def __init__(self, cfg="yolov8n-obb.yaml", ch=8, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8OBBLoss(self)


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    missing = object()
    module_state = {}
    attribute_state = []
    try:
        # Install module aliases first so attribute aliases can target renamed modules.
        for old, new in modules.items():
            module_state[old] = sys.modules.get(old, missing)
            sys.modules[old] = import_module(new)

        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            module = import_module(old_module)
            had_attr = hasattr(module, old_attr)
            previous = getattr(module, old_attr, None)
            attribute_state.append((module, old_attr, had_attr, previous))
            setattr(module, old_attr, getattr(import_module(new_module), new_attr))

        yield
    finally:
        for module, attr, had_attr, previous in reversed(attribute_state):
            if had_attr:
                setattr(module, attr, previous)
            else:
                delattr(module, attr)

        for old, previous in module_state.items():
            if previous is missing:
                sys.modules.pop(old, None)
            else:
                sys.modules[old] = previous


class SafeClass:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass




class Ensemble(nn.ModuleList):

    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        y = torch.cat(y, 2)
        return y, None


def torch_safe_load(weight, safe_only=False):
    check_suffix(file=weight, suffix=".pt")
    file = weight  # GitHub download functionality removed - use local file
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
                # Historical module names are used only while loading pre-RE checkpoints.
                "ultralytics.nn.r2det.r2det": "ultralytics.nn.redet.redet",
                "ultralytics.nn.r2det.r2gconv": "ultralytics.nn.redet.regconv",
                "ultralytics.nn.neck.R2Cluster": "ultralytics.nn.neck.RECluster",
            },
            attributes={
                "ultralytics.nn.r2det.r2det.R2LiftGCBA": "ultralytics.nn.redet.redet.RELiftGCBA",
                "ultralytics.nn.r2det.r2det.ER2GCBA": "ultralytics.nn.redet.redet.EREGCBA",
                "ultralytics.nn.r2det.r2det.DR2GCBA": "ultralytics.nn.redet.redet.DREGCBA",
                "ultralytics.nn.r2det.r2det.PR2GCBA": "ultralytics.nn.redet.redet.PREGCBA",
                "ultralytics.nn.r2det.r2det.R2SPPF": "ultralytics.nn.redet.redet.RESPPF",
                "ultralytics.nn.r2det.r2gconv.R2Lift": "ultralytics.nn.redet.regconv.RELift",
                "ultralytics.nn.r2det.r2gconv.PR2GConv": "ultralytics.nn.redet.regconv.PREGConv",
                "ultralytics.nn.r2det.r2gconv.DR2GConv": "ultralytics.nn.redet.regconv.DREGConv",
                "ultralytics.nn.r2det.r2gconv.ER2GConv": "ultralytics.nn.redet.regconv.EREGConv",
                "ultralytics.nn.r2det.r2gconv.AR2GUp": "ultralytics.nn.redet.regconv.AREGUp",
                (
                    "ultralytics.nn.modules.head."
                    "OBB_RE_CYCLIC_BOX_TIED_BIAS_READOUT_P5BOX100_P34BOX96_ANGLE40_P5DFLSBOCCA"
                ): "ultralytics.nn.modules.head.OAHead",
                "ultralytics.nn.modules.head.FressDetHead": "ultralytics.nn.modules.head.OAHead",
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch.load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch.load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using a compatible Ultralytics version."
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using a compatible Ultralytics version."
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """Build the single architecture described by FressDet.yaml."""
    import ast

    max_channels = float("inf")
    nc = d["nc"]
    scales = d.get("scales")
    reg_max = d.get("reg_max", 16)
    depth, width = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple"))
    scale = d.get("scale", "")
    if scales:
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        values = scales[scale]
        if len(values) != 3:
            raise ValueError("FressDet.yaml requires scales: [depth, width, max_channels].")
        depth, width, max_channels = values

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    selected_head = OAHead

    for i, (f, n, module_name, args) in enumerate(d["backbone"] + d["head"]):
        if not isinstance(module_name, str) or module_name not in globals():
            raise NotImplementedError(f"Unsupported FressDet module: {module_name}")
        m = globals()[module_name]
        t = module_name

        for j, value in enumerate(args):
            if isinstance(value, str):
                try:
                    args[j] = locals()[value] if value in locals() else ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass

        n = n_ = max(round(n * depth), 1) if n > 1 else n

        if m is selected_head:
            ch_in = [ch[x] for x in (f if isinstance(f, (list, tuple)) else [f])]
            base = [args[0], *(args[1:] if len(args) > 1 else [1])]
            args = [*base, ch_in, reg_max]
            c2 = sum(ch_in)
        elif m in (ReCoW, ReCoWAAA):
            c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            c2 = c1
            args = [c1]
        elif m is SpeIWMetaformerStage:
            c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            c2 = c1
            args = [c1]
        elif m is PatchMerging:
            c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            args = [c1, c2]
        elif m is AREGUp:
            c1 = ch[f] if isinstance(f, int) else sum(ch[x] for x in f)
            c2 = args[0]
            args = [c1, c2, *args[1:]] if len(args) > 1 else [c1, c2]
        elif m in REDET_CLASS:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        else:
            raise NotImplementedError(f"Module {module_name} is not used by FressDet.yaml.")

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_.np:10.0f}  {t:<45}{str(args):<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):  # False
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale. The function
    uses regular expression matching to find the pattern of the model scale in the YAML file name, which is denoted by
    n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    try:
        return re.search(r"yolo[v]?\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    except AttributeError:
        return ""




def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)
        args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()
        model.args = args
        model.pt_path = w
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    if len(ensemble) == 1:
        return ensemble[-1]
    LOGGER.info(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble

def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('obb').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg["head"][-1][-2].lower()  # output module name
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        try:
            return cfg2task(model)
        except:  # noqa E722
            pass

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            try:
                return eval(x)["task"]
            except:  # noqa E722
                pass
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            try:
                return cfg2task(eval(x))
            except:  # noqa E722
                pass

        for m in model.modules():
            if isinstance(m, OBB):
                return "obb"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-obb" in model.stem or "obb" in model.parts:
            return "obb"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING: Unable to automatically guess model task, assuming 'task=obb'. "
        "Explicitly define task='obb' for this demo."
    )
    return "obb"
