from __future__ import annotations

from .postprocess import Detection


def horizontal_zone(cx: float) -> str:
    if cx < 1 / 3:
        return "слева"
    if cx > 2 / 3:
        return "справа"
    return "по центру"


def vertical_zone(cy: float) -> str:
    if cy < 1 / 3:
        return "сверху"
    if cy > 2 / 3:
        return "снизу"
    return "посередине"


def size_label(area_fraction: float) -> str:
    if area_fraction < 0.02:
        return "крошечный"
    if area_fraction < 0.08:
        return "маленький"
    if area_fraction < 0.25:
        return "средний"
    return "крупный"


def describe_detection(det: Detection, image_w: int, image_h: int) -> str:
    """One human-readable line: what, where, how big, with pixel coordinates."""
    px_w = int(det.w * image_w)
    px_h = int(det.h * image_h)
    px_x = int(det.x1 * image_w)
    px_y = int(det.y1 * image_h)
    area = det.w * det.h
    zone = f"{vertical_zone(det.cy)} {horizontal_zone(det.cx)}".strip()
    return (
        f"{det.class_name} ({det.confidence:.0%}): {zone}, "
        f"{size_label(area)} размер {px_w}x{px_h}px, "
        f"bbox x={px_x} y={px_y} w={px_w} h={px_h}, "
        f"центр ({det.cx:.2f}, {det.cy:.2f})"
    )


def _center_relation(a: Detection, b: Detection) -> str:
    dx = b.cx - a.cx
    dy = b.cy - a.cy
    if abs(dx) >= abs(dy):
        return "правее" if dx > 0 else "левее"
    return "ниже" if dy > 0 else "выше"


def describe_relations(detections: list[Detection], max_pairs: int = 6) -> list[str]:
    """Pairwise spatial relations between detected objects."""
    lines: list[str] = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            relation = _center_relation(a, b)
            lines.append(f"{b.class_name} {relation} чем {a.class_name}")
            if len(lines) >= max_pairs:
                return lines
    return lines


def describe_scene(detections: list[Detection], image_w: int, image_h: int) -> str:
    """A compact scene summary: counts of each detected class."""
    if not detections:
        return "На изображении не найдено объектов."
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    parts = [f"{name} x{count}" if count > 1 else name for name, count in counts.items()]
    return f"Сцена {image_w}x{image_h}px, объектов {len(detections)}: " + ", ".join(parts)
