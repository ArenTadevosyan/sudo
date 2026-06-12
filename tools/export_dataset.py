from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY = ROOT / "data" / "memory.tsv"
DEFAULT_OUT = ROOT / "dataset" / "processed"
DEFAULT_RAW = ROOT / "dataset" / "raw"


@dataclass
class Memory:
    kind: str
    text: str
    tags: str
    importance: float
    strength: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Export brain memory into LM train/val text.")
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["all", "dialogue"], default="all")
    args = parser.parse_args()

    memories = read_memories(args.memory)
    samples = build_samples(memories)
    samples.extend(read_raw_texts(args.raw))
    if args.mode == "dialogue":
        samples = filter_dialogue_samples(samples)
    if not samples:
        raise SystemExit("No samples exported: memory is empty.")

    random.Random(args.seed).shuffle(samples)
    val_size = max(1, int(len(samples) * args.val_ratio)) if len(samples) > 1 else 0
    val = samples[:val_size]
    train = samples[val_size:] or samples

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train.txt").write_text("\n\n".join(train) + "\n", encoding="utf-8")
    (args.out / "val.txt").write_text("\n\n".join(val or train[:1]) + "\n", encoding="utf-8")

    print(f"Exported {len(train)} train and {len(val or train[:1])} val samples to {args.out}")


def read_memories(path: Path) -> list[Memory]:
    if not path.exists():
        return []

    memories: list[Memory] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 8:
            memories.append(
                Memory(
                    kind=unescape(parts[2]),
                    importance=float_or(parts[3], 0.5),
                    strength=float_or(parts[4], 0.5),
                    text=unescape(parts[6]),
                    tags=unescape(parts[7]),
                )
            )
        elif len(parts) == 5:
            text = unescape(parts[3])
            tags = unescape(parts[4])
            memories.append(
                Memory(
                    kind=infer_kind(text, tags),
                    importance=float_or(parts[2], 0.5),
                    strength=0.5,
                    text=text,
                    tags=tags,
                )
            )
    return memories


def build_samples(memories: list[Memory]) -> list[str]:
    samples: list[str] = []
    last_user: str | None = None

    for memory in memories:
        text = memory.text.strip()
        if not text:
            continue

        if text.startswith("Пользователь:"):
            last_user = text.removeprefix("Пользователь:").strip()
            samples.append(format_memory(memory))
            continue

        if text.startswith("ИИ:") and last_user:
            answer = text.removeprefix("ИИ:").strip()
            samples.append(
                "\n".join(
                    [
                        "<TASK>dialogue",
                        f"<USER>{last_user}",
                        f"<ASSISTANT>{answer}",
                        "<END>",
                    ]
                )
            )
            last_user = None
            continue

        samples.append(format_memory(memory))

    return samples


def read_raw_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    samples: list[str] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        samples.extend(split_raw_samples(raw, file_path.relative_to(path)))
    return samples


def split_raw_samples(raw: str, source: Path) -> list[str]:
    if "<END>" not in raw:
        return ["\n".join([f"<SOURCE>{source}", "<TEXT>", raw, "<END>"])]

    samples: list[str] = []
    for block in raw.split("<END>"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("<TASK>") or block.startswith("<KIND>") or block.startswith("<SOURCE>"):
            samples.append(block + "\n<END>")
        else:
            samples.append("\n".join([f"<SOURCE>{source}", block, "<END>"]))
    return samples



def filter_dialogue_samples(samples: list[str]) -> list[str]:
    return [
        sample
        for sample in samples
        if sample.startswith("<TASK>dialogue") and "<USER>" in sample and "<ASSISTANT>" in sample
    ]

def format_memory(memory: Memory) -> str:
    lines = [
        f"<KIND>{memory.kind}",
        f"<IMPORTANCE>{memory.importance:.2f}",
        f"<STRENGTH>{memory.strength:.2f}",
        f"<TEXT>{memory.text}",
    ]
    if memory.tags:
        lines.append(f"<TAGS>{memory.tags}")
    lines.append("<END>")
    return "\n".join(lines)


def infer_kind(text: str, tags: str) -> str:
    lowered = text.lower()
    tag_set = {tag.strip() for tag in tags.split(",")}
    if {"feedback", "positive", "negative"} & tag_set:
        return "feedback"
    if {"principle", "evolution"} & tag_set:
        return "rule"
    if lowered.startswith("цель:") or lowered.startswith("цель-кандидат"):
        return "goal"
    if lowered.startswith("факт:"):
        return "fact"
    return "dialogue"


def unescape(value: str) -> str:
    result: list[str] = []
    escaping = False
    for char in value:
        if not escaping and char == "\\":
            escaping = True
            continue
        if escaping:
            result.append({"t": "\t", "n": "\n", "r": "\r", "\\": "\\"}.get(char, char))
            escaping = False
        else:
            result.append(char)
    if escaping:
        result.append("\\")
    return "".join(result)


def float_or(value: str, fallback: float) -> float:
    try:
        return float(value)
    except ValueError:
        return fallback


if __name__ == "__main__":
    main()

