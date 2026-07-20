# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import RotatedTaskAlignedAssigner, dist2rbox, make_anchors
from .metrics import probiou
class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left  # (230,4)
        tr = tl + 1  # target right  # (230,4)
        wl = tr - target  # weight left  # (230,4)
        wr = 1 - wl  # weight right  # (230,4)
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)  # torch.Size([230, 1])


class RotatedBboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])  # max:0.7574 min:0.0597 iou:(230,1)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:  # DFLoss()
            ap = anchor_points.unsqueeze(0)  # (1, HW, 2)
            cx, cy, w, h, theta = target_bboxes.split(1, dim=-1)
            dx = ap[..., 0:1] - cx
            dy = ap[..., 1:2] - cy
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            l = local_x + w / 2
            r = w / 2 - local_x
            t = local_y + h / 2
            b = h / 2 - local_y
            maxv = self.dfl_loss.reg_max - 1 - 1e-3
            target_ltrb = torch.cat((l, t, r, b), dim=-1).clamp_(0, maxv)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class v8OBBLoss:
    """Calculates losses for rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2=None, tal_stride=None):
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1  # True

        assigner_kwargs = dict(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        if tal_topk2 is not None:
            assigner_kwargs["topk2"] = tal_topk2
        if tal_stride is not None:
            assigner_kwargs["stride"] = tal_stride
        self.assigner = RotatedTaskAlignedAssigner(**assigner_kwargs)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)  # [0,1] [14,31]
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)  # [2,31,6]
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )  # [2,64,23142]; [2,8,23142]

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # [2,23142,8]
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # [2,23142,64]
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()  # [2,23142,1]

        dtype = pred_scores.dtype  # torch.float32
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # [928,1216]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)  # [45,7]
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()  #[45] [45] 
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("OBB dataset incorrectly formatted for rotated boxes.") from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4) [2,23142,5]

        bboxes_for_assigner = pred_bboxes.clone().detach()  # [2,23142,5]
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor  # [2,23142,5]
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),  # [2.23142.8]
            bboxes_for_assigner.type(gt_bboxes.dtype),  # [2,23142,5]
            anchor_points * stride_tensor,  # [23142,2]
            gt_labels,  # [2,31,1]
            gt_bboxes,  # [2,31,5]
            mask_gt,  # [2,31,1]
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():  # 230
            target_bboxes[..., :4] /= stride_tensor  # 转为dist域
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain  7.5
        loss[1] *= self.hyp.cls  # cls gain  0.5
        loss[2] *= self.hyp.dfl  # dfl gain  1.5

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)  # [2,23142,5]
