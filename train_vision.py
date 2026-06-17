"""Train the vision detector from scratch.

Designed for a single T4 (16 GB). Uses pure PyTorch, random init, AMP, gradient
clipping, cosine LR schedule with linear warmup.

Dataset layout::

    dataset/vision/
        classes.txt          # one class name per line
        train/
            images/*.jpg
            labels/*.txt     # YOLO-format: class cx cy w h (normalized)
        val/
            images/*.jpg
            labels/*.txt

Run::

    python train_vision.py --device cuda --img-size 416 --batch-size 16 --epochs 80
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "dataset" / "vision"
DEFAULT_OUT = ROOT / "checkpoints" / "vision"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="enable mixed precision")
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_classes(data_dir: Path) -> list[str]:
    classes_file = data_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(
            f"Provide class names at {classes_file} (one per line)."
        )
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def lr_at(epoch: int, base_lr: float, warmup: int, total: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    from ai_brain.vision.dataset import DetectionDataset, collate
    from ai_brain.vision.loss import YOLOLoss
    from ai_brain.vision.model import VisionConfig, build_model, save_config

    args = parse_args()
    torch.manual_seed(args.seed)

    classes = read_classes(args.data)
    config = VisionConfig(
        num_classes=len(classes),
        img_size=args.img_size,
        class_names=classes,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    save_config(args.out / "config.json", config)

    train_set = DetectionDataset(args.data / "train" / "images", args.img_size, augment=True)
    val_dir = args.data / "val" / "images"
    val_set = DetectionDataset(val_dir, args.img_size, augment=False) if val_dir.exists() else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device == "cuda",
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=args.device == "cuda",
        )
        if val_set is not None
        else None
    )

    device = torch.device(args.device)
    model = build_model(config).to(device)
    if args.init_from is not None and (args.init_from / "model.pt").exists():
        state = torch.load(args.init_from / "model.pt", map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights from {args.init_from}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = YOLOLoss(config)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device == "cuda")
    best_val = float("inf")

    for epoch in range(args.epochs):
        lr = lr_at(epoch, args.lr, args.warmup_epochs, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        running = 0.0
        steps = 0
        start = time.time()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
                prediction = model(images)
                loss, parts = loss_fn(prediction, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            steps += 1

        train_loss = running / max(1, steps)
        elapsed = time.time() - start
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"lr={lr:.2e} train_loss={train_loss:.4f} "
            f"box={parts['box']:.3f} obj={parts['obj']:.3f} "
            f"noobj={parts['noobj']:.3f} cls={parts['cls']:.3f} "
            f"({elapsed:.1f}s)"
        )

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_steps = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, non_blocking=True)
                    targets = [t.to(device) for t in targets]
                    prediction = model(images)
                    loss, _ = loss_fn(prediction, targets)
                    val_running += float(loss.item())
                    val_steps += 1
            val_loss = val_running / max(1, val_steps)
            print(f"  val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), args.out / "model.pt")
                print(f"  saved best to {args.out / 'model.pt'}")
        else:
            torch.save(model.state_dict(), args.out / "model.pt")


if __name__ == "__main__":
    main()
