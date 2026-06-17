from __future__ import annotations

from dataclasses import dataclass

from .model import VisionConfig


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    # Normalized box in [0, 1] relative to the (square) network input.
    cx: float
    cy: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.cx - self.w / 2

    @property
    def y1(self) -> float:
        return self.cy - self.h / 2

    @property
    def x2(self) -> float:
        return self.cx + self.w / 2

    @property
    def y2(self) -> float:
        return self.cy + self.h / 2


def decode(prediction, config: VisionConfig, conf_threshold: float = 0.25):
    """Decode one image's raw grid output into a flat list of candidate boxes."""
    import torch

    grid = config.grid_size
    anchors = torch.tensor(config.anchors, device=prediction.device)

    pred = prediction  # (A, G, G, 5+C)
    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = torch.sigmoid(pred[..., 4])
    cls_logits = pred[..., 5:]
    cls_prob = torch.softmax(cls_logits, dim=-1)
    cls_conf, cls_id = cls_prob.max(dim=-1)
    confidence = obj * cls_conf

    cols = torch.arange(grid, device=prediction.device).view(1, 1, grid).float()
    rows = torch.arange(grid, device=prediction.device).view(1, grid, 1).float()
    anchor_w = anchors[:, 0].view(-1, 1, 1)
    anchor_h = anchors[:, 1].view(-1, 1, 1)

    bx = (tx + cols) / grid
    by = (ty + rows) / grid
    bw = (anchor_w * torch.exp(tw)) / grid
    bh = (anchor_h * torch.exp(th)) / grid

    mask = confidence >= conf_threshold
    if not mask.any():
        return []

    boxes = []
    for a, r, c in mask.nonzero(as_tuple=False).tolist():
        boxes.append(
            (
                float(bx[a, r, c]),
                float(by[a, r, c]),
                float(bw[a, r, c]),
                float(bh[a, r, c]),
                float(confidence[a, r, c]),
                int(cls_id[a, r, c]),
            )
        )
    return boxes


def _iou(a, b) -> float:
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(boxes, iou_threshold: float = 0.45):
    """Greedy per-class NMS over decoded boxes."""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [
            box
            for box in boxes
            if box[5] != best[5] or _iou(best, box) < iou_threshold
        ]
    return kept


def detections_from_output(
    prediction,
    config: VisionConfig,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Full decode + NMS for a single image, returning Detection objects."""
    raw = decode(prediction, config, conf_threshold)
    kept = non_max_suppression(raw, iou_threshold)
    results = []
    for cx, cy, w, h, conf, cls_id in kept:
        name = config.class_names[cls_id] if cls_id < len(config.class_names) else str(cls_id)
        results.append(
            Detection(class_id=cls_id, class_name=name, confidence=conf, cx=cx, cy=cy, w=w, h=h)
        )
    return results
