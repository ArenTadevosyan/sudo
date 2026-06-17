import argparse
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import trange

from ai_brain.tiny_lm import ModelConfig, build_model, load_config, save_config

ROOT = Path(__file__).resolve().parent

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a powerful coding agent from scratch.")
    parser.add_argument("--train", type=Path, default=ROOT / "dataset" / "code_data" / "train.bin")
    parser.add_argument("--val", type=Path, default=ROOT / "dataset" / "code_data" / "val.bin")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "coder_agent")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=12) # Reduced for larger vocab/model
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--init-from", type=Path, default=None, help="Checkpoint directory to fine-tune from.")
    parser.add_argument("--init-checkpoint", default="model.pt")
    args = parser.parse_args()

    vocab_size = 50304 # GPT-2 vocab padded for multiple of 64

    if args.init_from:
        config = load_config(args.init_from / "config.json")
    else:
        config = ModelConfig(
            vocab_size=vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )

    device = args.device

    train_data = np.memmap(args.train, dtype=np.uint16, mode='r') if args.train.exists() else None
    val_data = np.memmap(args.val, dtype=np.uint16, mode='r') if args.val.exists() else train_data

    if train_data is None:
        raise SystemExit(f"Dataset not found at {args.train}. Run prepare_code_data.py first.")

    model = build_model(config).to(device)
    if args.init_from:
        state = torch.load(args.init_from / args.init_checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        print(f"initialized from {args.init_from / args.init_checkpoint}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-1, betas=(0.9, 0.95))
    args.out.mkdir(parents=True, exist_ok=True)
    save_config(args.out / "config.json", config)

    print(f"vocab={config.vocab_size} block_size={config.block_size} params={count_params(model):,} device={device}")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - config.block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[i : i + config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + config.block_size + 1]).astype(np.int64)) for i in ix])
        if device == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss() -> tuple[float, float]:
        model.eval()
        losses = {}
        for split in ["train", "val"]:
            values = torch.zeros(args.eval_iters)
            for index in range(args.eval_iters):
                x, y = get_batch(split)
                _, loss = model(x, y)
                values[index] = loss.item()
            losses[split] = values.mean().item()
        model.train()
        return losses["train"], losses["val"]

    best_val = float("inf")
    started = time.time()
    for step in trange(1, args.steps + 1):
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            train_loss, val_loss = estimate_loss()
            print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.out / "model.pt", model, optimizer, step, best_val)

    save_checkpoint(args.out / "last.pt", model, optimizer, args.steps, best_val)
    print(f"done in {time.time() - started:.1f}s; best_val={best_val:.4f}; out={args.out}")


def count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
        },
        path,
    )


if __name__ == "__main__":
    main()
