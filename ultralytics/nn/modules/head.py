# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Detection head required by FressDet.yaml."""

import math

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2rbox, make_anchors

from .block import DFL
from ..redet.redet import EREGCBA
from ..redet.regconv import TransferFlatten

__all__ = (
    "OBB",
    "OAHead",
)


def finite(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class Detect(nn.Module):
    """Shared base for the OBB heads kept in this demo."""

    dynamic = False  # force grid reconstruction
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=8, ch=(), reg_max=16):
        
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels 64 256
        self.cv2_re = nn.ModuleList(
            nn.Sequential(
                EREGCBA(x, c2, 3, 1, 1),
                EREGCBA(c2, c2, 3, 1, 1),
                TransferFlatten(),
                nn.Conv2d(c2 * 4, 4 * self.reg_max, 1),
            )
            for x in ch
        )
        self.cv3_re = (
            nn.ModuleList(
                nn.Sequential(
                    EREGCBA(x, c3, 3, 1, 1),
                    EREGCBA(c3, c3, 3, 1, 1),
                    TransferFlatten(),
                    nn.Conv2d(c3 * 4, self.nc, 1),
                )
                for x in ch
            )
        )
        self.cv2 = self.cv2_re
        self.cv3 = self.cv3_re
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def _forward_head(self, x, box_head, cls_head):
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((box_head[i](x[i]), cls_head[i](x[i])), 1))
        return outputs

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y, x

    def _inference(self, x):
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self  # self.model[-1]  # Detect() module
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            box_layer = None
            for layer in reversed(a):
                if hasattr(layer, "bias") and layer.bias is not None:
                    box_layer = layer
                    break
            if box_layer is not None:
                box_layer.bias.data[:] = 1.0  # box
            cls_layer = None
            for layer in reversed(b):
                if hasattr(layer, 'bias') and layer.bias is not None:
                    cls_layer = layer
                    break
            if cls_layer is not None:
                cls_layer.bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        raise NotImplementedError("Detect is retained only as the OBB head base.")


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=8, ne=1, ch=(), reg_max=16):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch, reg_max=reg_max)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)  # # (64,16) / (16,16) / (16,1)
        self.cv4_re = nn.ModuleList(
            nn.Sequential(
                EREGCBA(x, c4, 3, 1, 1),
                EREGCBA(c4, c4, 3, 1, 1),
                TransferFlatten(),
                nn.Conv2d(c4 * 4, self.ne, 1),
            )
            for x in ch
        )
        self.cv4 = self.cv4_re

    def _angle_from_head(self, x, angle_head):
        bs = x[0].shape[0]  # batch size
        angles = []
        for i in range(self.nl):
            logits = angle_head[i](x[i])
            angle = (logits.sigmoid() - 0.25) * math.pi
            angles.append(angle.view(bs, self.ne, -1))
        return torch.cat(angles, 2)

    def forward(self, x):
        angle = self._angle_from_head(x, self.cv4)
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x[0], angle], 1), (x[1], angle)

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class GroupSharedFlattenConv2d(nn.Module):
    """Group-tied 1x1 readout after TransferFlatten.

    It is equivalent to a Conv2d whose weights are shared across all flattened
    group slots. The output keeps the standard YOLO channel layout, so DFL and
    classification losses remain unchanged.
    """

    def __init__(self, c1, c2, group_order=4, bias=True):
        super().__init__()
        self.c1 = c1
        self.group_order = group_order
        self.conv = nn.Conv2d(c1, c2, 1, bias=bias)

    @property
    def bias(self):
        return self.conv.bias

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x):
        assert x.dim() == 4, f"GroupSharedFlattenConv2d expects flattened 4D input, got {x.shape}"
        b, cg, h, w = x.shape
        assert cg == self.c1 * self.group_order, (
            f"Expected {self.c1 * self.group_order} channels, got {cg}"
        )
        x = x.view(b, self.c1, self.group_order, h, w).mean(dim=2)
        return self.conv(x)


class GroupAngleFlattenConv2d(nn.Module):
    """Group-tied 1x1 angle readout with cyclic angle decoding."""

    def __init__(self, c1, ne=1, group_order=4, bias=True):
        super().__init__()
        self.c1 = c1
        self.ne = ne
        self.group_order = group_order
        self.conv = nn.Conv2d(c1, ne, 1, bias=bias)

    @property
    def bias(self):
        return self.conv.bias

    @property
    def weight(self):
        return self.conv.weight

    def _decode(self, logits):
        g = logits.shape[2]
        residual = (logits.sigmoid() - 0.5) * (2 * math.pi / g)
        offsets = torch.arange(g, device=logits.device, dtype=logits.dtype).view(1, 1, g, 1, 1)
        theta = residual + offsets * (2 * math.pi / g)
        weights = torch.softmax(logits, dim=2)
        cos = (weights * torch.cos(theta)).sum(dim=2)
        sin = (weights * torch.sin(theta)).sum(dim=2)
        angle = torch.atan2(sin, cos)
        return torch.remainder(angle + math.pi / 4, math.pi) - math.pi / 4

    def forward(self, x):
        assert x.dim() == 4, f"GroupAngleFlattenConv2d expects flattened 4D input, got {x.shape}"
        b, cg, h, w = x.shape
        assert cg == self.c1 * self.group_order, (
            f"Expected {self.c1 * self.group_order} channels, got {cg}"
        )
        x = x.view(b, self.c1, self.group_order, h, w).permute(0, 2, 1, 3, 4).reshape(
            b * self.group_order, self.c1, h, w
        )
        logits = self.conv(x).view(b, self.group_order, self.ne, h, w).permute(0, 2, 1, 3, 4)
        return self._decode(logits)


class GroupCyclicBoxTiedBiasFlattenConv2d(nn.Module):
    """Cyclic-tied box readout with side-shared bias.

    The original cyclic readout has equivariant weights, but independent side
    biases can break the C4 law after training. This variant ties the bias across
    the four side slots while keeping the same standard YOLO DFL output layout.
    """

    def __init__(self, c1, reg_max=16, group_order=4, bias=True):
        super().__init__()
        assert group_order == 4, "The cyclic box readout maps C4 group slots to four box sides."
        self.c1 = c1
        self.reg_max = reg_max
        self.group_order = group_order
        self.weight = nn.Parameter(torch.empty(group_order, reg_max, c1))
        self.bias = None
        nn.init.kaiming_uniform_(
            self.weight.view(group_order * reg_max, c1, 1, 1), a=math.sqrt(5)
        )
        self.bias = nn.Parameter(torch.zeros(reg_max)) if bias else None

    def forward(self, x):
        assert x.dim() == 4, f"GroupCyclicBoxTiedBiasFlattenConv2d expects flattened 4D input, got {x.shape}"
        b, cg, h, w = x.shape
        assert cg == self.c1 * self.group_order, (
            f"Expected {self.c1 * self.group_order} channels, got {cg}"
        )
        x = x.view(b, self.c1, self.group_order, h, w)
        outputs = []
        for side in range(self.group_order):
            side_logits = 0.0
            for offset in range(self.group_order):
                group_idx = (side + offset) % self.group_order
                side_logits = side_logits + torch.einsum("bchw,rc->brhw", x[:, :, group_idx], self.weight[offset])
            if self.bias is not None:
                side_logits = side_logits + self.bias.view(1, self.reg_max, 1, 1)
            outputs.append(side_logits)
        return torch.cat(outputs, dim=1)


class OAHead(OBB):
    """FressDet head with a bidirectional finite-only output boundary.

    ``torch.nan_to_num`` is an exact identity (including its VJP) for every
    finite value.  Non-finite box, class, or angle predictions are replaced by
    zero immediately before they leave the head. During training, the same
    operation is attached to gradients entering the shared features and head
    parameters, preventing a non-finite backward value from escaping the head.
    The official loss, assigner, decoder, optimizer, and trainer remain
    untouched.
    """

    def _set_group_readouts(self, ch, box_width, cls_width, angle_width, box_factory):
        """Build the three scale-specific readouts used by FressDet."""
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                EREGCBA(x, box_width, 3, 1, 1),
                EREGCBA(box_width, box_width, 3, 1, 1),
                TransferFlatten(),
                box_factory(box_width),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                EREGCBA(x, cls_width, 3, 1, 1),
                EREGCBA(cls_width, cls_width, 3, 1, 1),
                TransferFlatten(),
                GroupSharedFlattenConv2d(cls_width, self.nc, group_order=4),
            )
            for x in ch
        )
        self.cv4 = nn.ModuleList(
            nn.Sequential(
                EREGCBA(x, angle_width, 3, 1, 1),
                EREGCBA(angle_width, angle_width, 3, 1, 1),
                TransferFlatten(),
                GroupAngleFlattenConv2d(angle_width, self.ne, group_order=4),
            )
            for x in ch
        )

    def __init__(self, nc=8, ne=1, ch=(), reg_max=16):
        super().__init__(nc, ne=ne, ch=ch, reg_max=reg_max)

        # Keep the historical construction order so fresh seed-0 initialization
        # remains bitwise identical after flattening the former inheritance chain.
        self._set_group_readouts(
            ch,
            max((16, ch[0] // 4, self.reg_max * 4)),
            max(ch[0], min(self.nc, 100)),
            max(ch[0] // 4, self.ne),
            lambda width: GroupSharedFlattenConv2d(
                width, 4 * self.reg_max, group_order=4
            ),
        )
        for attr in ("cv2_re", "cv3_re", "cv4_re"):
            if hasattr(self, attr):
                delattr(self, attr)

        self._set_group_readouts(
            ch,
            max((96, ch[0] // 2, self.reg_max * 4)),
            max(80, ch[0], min(self.nc, 100)),
            max(40, ch[0] // 3, self.ne),
            lambda width: GroupCyclicBoxTiedBiasFlattenConv2d(
                width, self.reg_max, group_order=4
            ),
        )

        p5_width = max(100, ch[0] // 2, self.reg_max * 4)
        self.cv2[2] = nn.Sequential(
            EREGCBA(ch[2], p5_width, 3, 1, 1),
            EREGCBA(p5_width, p5_width, 3, 1, 1),
            TransferFlatten(),
            GroupCyclicBoxTiedBiasFlattenConv2d(
                p5_width, self.reg_max, group_order=4
            ),
        )

        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.register_hook(finite)

    def _angle_from_head(self, x, angle_head):
        bs = x[0].shape[0]
        angles = []
        for i in range(self.nl):
            angle = angle_head[i](x[i])
            angles.append(angle.view(bs, self.ne, -1))
        return torch.cat(angles, 2)

    def bias_init(self):
        super().bias_init()
        for angle_head in self.cv4:
            angle_head[-1].bias.data[:] = 0.0

    def forward(self, x):
        if self.training:
            for feature in x:
                if feature.requires_grad:
                    feature.register_hook(finite)

        angle = finite(self._angle_from_head(x, self.cv4))
        if not self.training:
            self.angle = angle

        outputs = []
        for i in range(self.nl):
            prediction = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            outputs.append(finite(prediction))

        if self.training:
            return outputs, angle

        prediction = self._inference(outputs)
        return torch.cat([prediction, angle], 1), (outputs, angle)
