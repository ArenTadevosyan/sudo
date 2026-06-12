from __future__ import annotations

import argparse
import time
from pathlib import Path

from ai_brain.tiny_lm import CharTokenizer, ModelConfig, build_model, save_config


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character language model from zero.")
    parser.add_argument("--train", type=Path, default=ROOT / "dataset" / "processed" / "train.txt")
    parser.add_argument("--val", type=Path, default=ROOT / "dataset" / "processed" / "val.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "tiny_lm")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=30)
    args = parser.parse_args()

    import torch
    from tqdm import trange

    train_text = args.train.read_text(encoding="utf-8")
    val_text = args.val.read_text(encoding="utf-8") if args.val.exists() else train_text
    tokenizer = CharTokenizer.train(train_text + val_text)
    config = ModelConfig(
        vocab_size=len(tokenizer.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --device cpu or install CUDA PyTorch.")

    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    if train_data.numel() <= args.block_size + 1:
        raise SystemExit("Dataset is too small. Add more memory or lower --block-size.")

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out / "tokenizer.json")
    save_config(args.out / "config.json", config)

    print(f"vocab={config.vocab_size} params={count_params(model):,} device={device}")

    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        if data.numel() <= args.block_size + 1:
            data = train_data
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i : i + args.block_size] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + args.block_size + 1] for i in ix]).to(device)
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
    import torch

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

