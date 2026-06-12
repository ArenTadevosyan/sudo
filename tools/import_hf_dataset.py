from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dataset" / "raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import small Hugging Face dataset samples for local LM training.")
    parser.add_argument(
        "--source",
        choices=["oasst_ru", "wiki_ru", "c4_ru"],
        default="oasst_ru",
        help="Dataset preset to import.",
    )
    parser.add_argument("--limit", type=int, default=5000, help="Maximum samples/documents to export.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory under dataset/raw.")
    parser.add_argument("--min-chars", type=int, default=40, help="Skip shorter texts.")
    parser.add_argument("--max-chars", type=int, default=3000, help="Trim longer texts.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / f"{args.source}_{args.limit}.txt"

    if args.source == "oasst_ru":
        samples = import_oasst_ru(args.limit, args.min_chars, args.max_chars)
    elif args.source == "wiki_ru":
        samples = import_wiki_ru(args.limit, args.min_chars, args.max_chars)
    else:
        samples = import_c4_ru(args.limit, args.min_chars, args.max_chars)

    written = write_samples(output_path, samples)
    print(f"Imported {written} samples into {output_path}")
    print("Next: python tools/export_dataset.py")


def import_oasst_ru(limit: int, min_chars: int, max_chars: int) -> Iterable[str]:
    datasets = require_datasets()
    ds = datasets.load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    pending_prompts: dict[str, str] = {}
    count = 0

    for row in ds:
        if row.get("lang") != "ru" or row.get("deleted"):
            continue
        text = clean_text(row.get("text", ""), max_chars)
        if len(text) < min_chars:
            continue

        role = row.get("role")
        message_id = row.get("message_id")
        parent_id = row.get("parent_id")

        if role == "prompter" and message_id:
            pending_prompts[message_id] = text
            continue

        if role == "assistant" and parent_id in pending_prompts:
            prompt = pending_prompts[parent_id]
            yield "\n".join(["<TASK>dialogue", f"<USER>{prompt}", f"<ASSISTANT>{text}", "<END>"])
            count += 1
            if count >= limit:
                return


def import_wiki_ru(limit: int, min_chars: int, max_chars: int) -> Iterable[str]:
    datasets = require_datasets()
    ds = datasets.load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)
    count = 0

    for row in ds:
        title = clean_text(row.get("title", ""), 200)
        text = clean_text(row.get("text", ""), max_chars)
        if len(text) < min_chars:
            continue
        yield "\n".join([f"<SOURCE>wikipedia_ru:{title}", "<TEXT>", text, "<END>"])
        count += 1
        if count >= limit:
            return


def import_c4_ru(limit: int, min_chars: int, max_chars: int) -> Iterable[str]:
    datasets = require_datasets()
    ds = datasets.load_dataset("allenai/c4", "ru", split="train", streaming=True)
    count = 0

    for row in ds:
        text = clean_text(row.get("text", ""), max_chars)
        if len(text) < min_chars:
            continue
        url = clean_text(row.get("url", ""), 240)
        yield "\n".join([f"<SOURCE>c4_ru:{url}", "<TEXT>", text, "<END>"])
        count += 1
        if count >= limit:
            return


def write_samples(path: Path, samples: Iterable[str]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(sample.strip())
            handle.write("\n\n")
            count += 1
    return count


def clean_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).replace("\r", " ").split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip()
    return cleaned


def require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("Install dependency first: pip install datasets") from exc
    return datasets


if __name__ == "__main__":
    main()
