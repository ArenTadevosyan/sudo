"""Prepare a vision dataset in the layout expected by train_vision.py.

Two modes:

1. ``init`` creates an empty dataset/vision/{train,val}/{images,labels} tree
   plus a classes.txt template. Useful as a starting scaffold.

2. ``from-coco`` converts a COCO-style ``annotations.json`` plus an images
   directory into the YOLO-format layout, with a configurable val split.

Run::

    python tools/prepare_vision_dataset.py init --classes person car dog
    python tools/prepare_vision_dataset.py from-coco \\
        --annotations path/to/coco.json \\
        --images path/to/images \\
        --val-fraction 0.1
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "vision"


def cmd_init(args: argparse.Namespace) -> None:
    base = args.data
    for split in ("train", "val"):
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
    classes_file = base / "classes.txt"
    if not classes_file.exists() or args.force:
        classes_file.write_text("\n".join(args.classes) + "\n", encoding="utf-8")
    print(f"Dataset scaffold ready in {base}")
    print(f"Classes ({len(args.classes)}): {', '.join(args.classes)}")
    print("Drop images into train/images and val/images, with matching label .txt files.")


def cmd_from_coco(args: argparse.Namespace) -> None:
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_index = {cat["id"]: idx for idx, cat in enumerate(categories)}
    class_names = [cat["name"] for cat in categories]

    images_by_id = {img["id"]: img for img in data["images"]}
    boxes_by_image: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        info = images_by_id.get(ann["image_id"])
        if info is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2) / info["width"]
        cy = (y + h / 2) / info["height"]
        nw = w / info["width"]
        nh = h / info["height"]
        cls = cat_index[ann["category_id"]]
        boxes_by_image.setdefault(info["id"], []).append((cls, cx, cy, nw, nh))

    image_ids = list(images_by_id.keys())
    random.Random(args.seed).shuffle(image_ids)
    val_count = max(1, int(len(image_ids) * args.val_fraction))
    val_set = set(image_ids[:val_count])

    base = args.data
    for split in ("train", "val"):
        (base / split / "images").mkdir(parents=True, exist_ok=True)
        (base / split / "labels").mkdir(parents=True, exist_ok=True)
    (base / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    copied = 0
    for image_id, info in images_by_id.items():
        split = "val" if image_id in val_set else "train"
        src = args.images / info["file_name"]
        if not src.exists():
            continue
        dst_image = base / split / "images" / Path(info["file_name"]).name
        if args.symlink:
            if dst_image.exists():
                dst_image.unlink()
            dst_image.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst_image)

        label_lines = [
            f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls, cx, cy, bw, bh in boxes_by_image.get(image_id, [])
        ]
        label_path = base / split / "labels" / (Path(info["file_name"]).stem + ".txt")
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
        copied += 1

    print(f"Converted {copied} images, {len(class_names)} classes, val={val_count}")
    print(f"Layout written to {base}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create empty dataset scaffold")
    init.add_argument("--data", type=Path, default=DEFAULT_DATA)
    init.add_argument("--classes", nargs="+", default=["object"])
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    coco = sub.add_parser("from-coco", help="convert COCO json into YOLO layout")
    coco.add_argument("--annotations", type=Path, required=True)
    coco.add_argument("--images", type=Path, required=True)
    coco.add_argument("--data", type=Path, default=DEFAULT_DATA)
    coco.add_argument("--val-fraction", type=float, default=0.1)
    coco.add_argument("--seed", type=int, default=42)
    coco.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    coco.set_defaults(func=cmd_from_coco)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
