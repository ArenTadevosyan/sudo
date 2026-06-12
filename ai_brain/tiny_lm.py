from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CharTokenizer:
    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for index, char in enumerate(chars)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = self.stoi.get("\n", 0)
        return [self.stoi.get(char, unknown) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(index, "") for index in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(list(data["chars"]))


def save_config(path: Path, config: ModelConfig) -> None:
    path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: Path) -> ModelConfig:
    return ModelConfig(**json.loads(path.read_text(encoding="utf-8")))


def build_model(config: ModelConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if config.n_embd % config.n_head != 0:
                raise ValueError("n_embd must be divisible by n_head")
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            batch, time, channels = x.size()
            q, k, v = self.c_attn(x).split(config.n_embd, dim=2)
            head_size = channels // config.n_head
            q = q.view(batch, time, config.n_head, head_size).transpose(1, 2)
            k = k.view(batch, time, config.n_head, head_size).transpose(1, 2)
            v = v.view(batch, time, config.n_head, head_size).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (head_size**-0.5)
            att = att.masked_fill(self.mask[:, :, :time, :time] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(batch, time, channels)
            return self.dropout(self.c_proj(y))

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention()
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP()

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class TinyLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)
            self.blocks = nn.Sequential(*[Block() for _ in range(config.n_layer)])
            self.ln_f = nn.LayerNorm(config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        def forward(self, idx, targets=None):
            _, time = idx.shape
            if time > config.block_size:
                raise ValueError("sequence is longer than block_size")
            positions = torch.arange(0, time, device=idx.device)
            x = self.token_embedding(idx) + self.position_embedding(positions)
            x = self.dropout(x)
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens: int, temperature: float = 0.9, top_k: int = 50):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -config.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

    return TinyLanguageModel()

