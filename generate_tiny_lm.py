from __future__ import annotations

import argparse
from pathlib import Path

from ai_brain.tiny_lm import CharTokenizer, build_model, load_config


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with the trained tiny LM.")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "tiny_lm")
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--prompt", default="<USER>привет\n<ASSISTANT>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--stop-at-end", action="store_true", default=True)
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Use --device cpu or install CUDA PyTorch.")

    tokenizer = CharTokenizer.load(args.checkpoint_dir / "tokenizer.json")
    config = load_config(args.checkpoint_dir / "config.json")
    model = build_model(config).to(args.device)
    state = torch.load(args.checkpoint_dir / args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    prompt = args.prompt.encode("utf-8").decode("unicode_escape")
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=args.device)
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    decoded = tokenizer.decode(out[0].tolist())
    if args.stop_at_end and "<END>" in decoded:
        decoded = decoded.split("<END>", 1)[0] + "<END>"
    print(decoded)


if __name__ == "__main__":
    main()

