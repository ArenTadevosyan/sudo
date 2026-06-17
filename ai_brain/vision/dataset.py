from __future__ import annotations

import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_path(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir).with_suffix(".txt")
    return labels_dir / relative


def read_label_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read a YOLO label file: one ``class cx cy w h`` line per object."""
    if not path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        boxes.append((int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return boxes


def letterbox(image, size: int, fill: int = 114):
    """Resize keeping aspect ratio and pad to a square ``size`` x ``size``.

    Returns the padded image plus the scale and padding needed to map
    normalized boxes from the original image into the padded canvas.
    """
    from PIL import Image

    width, height = image.size
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y, new_w, new_h


def _remap_boxes(boxes, size, scale, pad_x, pad_y, new_w, new_h, orig_w, orig_h):
    """Map normalized boxes from the original image to the letterboxed canvas."""
    remapped = []
    for cls, cx, cy, bw, bh in boxes:
        abs_cx = cx * orig_w * scale + pad_x
        abs_cy = cy * orig_h * scale + pad_y
        abs_w = bw * orig_w * scale
        abs_h = bh * orig_h * scale
        remapped.append(
            (cls, abs_cx / size, abs_cy / size, abs_w / size, abs_h / size)
        )
    return remapped


class DetectionDataset:
    """YOLO-format detection dataset.

    Expects an ``images`` directory and a sibling ``labels`` directory with
    matching file stems. Each label line is ``class cx cy w h`` normalized.
    """

    def __init__(self, images_dir: str | Path, img_size: int, augment: bool = False):
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")
        self.labels_dir = self.images_dir.parent / "labels"
        self.img_size = img_size
        self.augment = augment
        self.samples = _list_images(self.images_dir)
        if not self.samples:
            raise RuntimeError(f"no images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import numpy as np
        import torch
        from PIL import Image

        image_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        boxes = read_label_file(_label_path(image_path, self.images_dir, self.labels_dir))

        canvas, scale, pad_x, pad_y, new_w, new_h = letterbox(image, self.img_size)
        boxes = _remap_boxes(
            boxes, self.img_size, scale, pad_x, pad_y, new_w, new_h, orig_w, orig_h
        )

        if self.augment and random.random() < 0.5:
            canvas = canvas.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [(cls, 1.0 - cx, cy, bw, bh) for cls, cx, cy, bw, bh in boxes]

        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        target = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return tensor, target


def collate(batch):
    """Stack images; keep targets as a list because counts differ per image."""
    import torch

    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
