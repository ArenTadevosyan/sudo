"""Run the trained detector on an image or a folder.

Outputs one line per detection (class, confidence, bbox, location, size) and
optionally writes results to the brain memory so the rest of the AI can recall
what it saw.

Run::

    python detect_vision.py --image path/to.jpg
    python detect_vision.py --image path/to.jpg --remember
    python detect_vision.py --folder path/to/dir --conf 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "vision"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--remember", action="store_true", help="store detections in brain memory")
    return parser.parse_args()


def collect_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [args.image]
    if args.folder:
        return sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise SystemExit("Pass --image or --folder.")


def detect_one(image_path: Path, model, config, args, device, vision_memory):
    import numpy as np
    import torch
    from PIL import Image

    from ai_brain.vision.dataset import letterbox
    from ai_brain.vision.describe import describe_detection, describe_relations, describe_scene
    from ai_brain.vision.postprocess import detections_from_output

    image = Image.open(image_path).convert("RGB")
    canvas, *_ = letterbox(image, config.img_size)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)
    detections = detections_from_output(
        prediction[0],
        config,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )

    print(f"\n=== {image_path} ===")
    print(describe_scene(detections, config.img_size, config.img_size))
    for det in detections:
        print(" -", describe_detection(det, config.img_size, config.img_size))
    for relation in describe_relations(detections):
        print("   ↳", relation)

    if args.remember and detections and vision_memory is not None:
        vision_memory.record_observation(
            detections,
            image_w=config.img_size,
            image_h=config.img_size,
            source=str(image_path),
        )
        print(f"   (записано в память: {len(detections)} объектов)")


def main() -> None:
    import torch

    from ai_brain.vision.memory_bridge import VisionMemory
    from ai_brain.vision.model import build_model, load_config

    args = parse_args()
    config_path = args.checkpoint_dir / "config.json"
    weights_path = args.checkpoint_dir / "model.pt"
    if not config_path.exists() or not weights_path.exists():
        raise SystemExit(f"Checkpoint not ready in {args.checkpoint_dir}")

    config = load_config(config_path)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    vision_memory = VisionMemory() if args.remember else None
    for path in collect_images(args):
        detect_one(path, model, config, args, device, vision_memory)


if __name__ == "__main__":
    main()
