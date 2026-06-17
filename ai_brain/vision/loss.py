from __future__ import annotations

from .model import VisionConfig


def _anchor_iou(box_wh, anchors):
    """IoU between a box (w, h) and anchors, all in the same units, centered."""
    import torch

    bw, bh = box_wh
    inter = torch.min(bw, anchors[:, 0]) * torch.min(bh, anchors[:, 1])
    union = bw * bh + anchors[:, 0] * anchors[:, 1] - inter
    return inter / union.clamp(min=1e-9)


class YOLOLoss:
    """Single-scale YOLO loss: box regression + objectness + classification."""

    def __init__(
        self,
        config: VisionConfig,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        import torch

        self.config = config
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.anchors = torch.tensor(config.anchors, dtype=torch.float32)

    def build_targets(self, targets, device):
        """Turn per-image box lists into dense target tensors.

        Returns target boxes (B,A,G,G,4), an object mask (B,A,G,G),
        and class indices (B,A,G,G).
        """
        import torch

        batch = len(targets)
        grid = self.config.grid_size
        anchors = self.anchors.to(device)

        target_box = torch.zeros((batch, self.config.num_anchors, grid, grid, 4), device=device)
        obj_mask = torch.zeros((batch, self.config.num_anchors, grid, grid), device=device, dtype=torch.bool)
        class_idx = torch.zeros((batch, self.config.num_anchors, grid, grid), device=device, dtype=torch.long)

        for b, boxes in enumerate(targets):
            if boxes.numel() == 0:
                continue
            for cls, cx, cy, bw, bh in boxes.tolist():
                if bw <= 0 or bh <= 0:
                    continue
                gx, gy = cx * grid, cy * grid
                col = min(int(gx), grid - 1)
                row = min(int(gy), grid - 1)
                box_wh = (torch.tensor(bw * grid, device=device), torch.tensor(bh * grid, device=device))
                ious = _anchor_iou(box_wh, anchors)
                best = int(torch.argmax(ious).item())

                tx = gx - col
                ty = gy - row
                tw = torch.log(torch.tensor(bw * grid, device=device) / anchors[best, 0] + 1e-9)
                th = torch.log(torch.tensor(bh * grid, device=device) / anchors[best, 1] + 1e-9)

                target_box[b, best, row, col] = torch.tensor([tx, ty, tw, th], device=device)
                obj_mask[b, best, row, col] = True
                class_idx[b, best, row, col] = int(cls)

        return target_box, obj_mask, class_idx

    def __call__(self, prediction, targets):
        import torch
        import torch.nn.functional as F

        device = prediction.device
        target_box, obj_mask, class_idx = self.build_targets(targets, device)
        noobj_mask = ~obj_mask

        pred_xy = prediction[..., 0:2]
        pred_wh = prediction[..., 2:4]
        pred_obj = prediction[..., 4]
        pred_cls = prediction[..., 5:]

        # Localization loss only where an object is assigned.
        if obj_mask.any():
            xy_loss = F.binary_cross_entropy_with_logits(
                pred_xy[obj_mask], target_box[..., 0:2][obj_mask], reduction="sum"
            )
            wh_loss = F.mse_loss(
                pred_wh[obj_mask], target_box[..., 2:4][obj_mask], reduction="sum"
            )
            cls_loss = F.cross_entropy(
                pred_cls[obj_mask], class_idx[obj_mask], reduction="sum"
            )
        else:
            xy_loss = wh_loss = cls_loss = torch.zeros((), device=device)

        obj_target = obj_mask.float()
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[obj_mask], obj_target[obj_mask], reduction="sum"
        ) if obj_mask.any() else torch.zeros((), device=device)
        noobj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[noobj_mask], obj_target[noobj_mask], reduction="sum"
        )

        batch = prediction.size(0)
        total = (
            self.lambda_coord * (xy_loss + wh_loss)
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + cls_loss
        ) / batch

        parts = {
            "box": float((xy_loss + wh_loss).item()) / batch,
            "obj": float(obj_loss.item()) / batch,
            "noobj": float(noobj_loss.item()) / batch,
            "cls": float(cls_loss.item()) / batch,
        }
        return total, parts
