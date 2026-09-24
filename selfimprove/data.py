"""Corpus loading and batch sampling."""

from __future__ import annotations

import random
from pathlib import Path

import torch

from .skills import Skill, format_example, sample_training


def load_corpus(corpus_dir: Path) -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(Path(corpus_dir).glob("**/*.txt"))]
    return "\n\n".join(parts)


def split_corpus(tokens: list[int], val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(len(tokens) * (1 - val_fraction))
    t = torch.as_tensor(tokens, dtype=torch.long)
    return t[:n], t[n:]


def task_stream(tok, skills: dict[str, Skill], weights: dict[str, float], n_examples: int,
                exam: dict[str, list], rng: random.Random) -> torch.Tensor:
    """Pack many skill examples into one token stream, sampled by per-skill weight."""
    names = list(skills)
    w = [max(0.0, float(weights.get(n, 1.0))) for n in names]
    if not names or sum(w) == 0:
        return torch.zeros(0, dtype=torch.long)
    counts = dict.fromkeys(names, 0)
    for n in rng.choices(names, weights=w, k=n_examples):
        counts[n] += 1
    examples = []
    for name, k in counts.items():
        exclude = {q for q, _ in exam.get(name, [])}
        examples += sample_training(skills[name], rng, k, exclude)
    rng.shuffle(examples)
    return torch.tensor(tok.encode("".join(format_example(q, a) for q, a in examples)), dtype=torch.long)


def examples_stream(tok, pairs: list[tuple[str, str]], repeats: int, rng: random.Random) -> torch.Tensor:
    items = pairs * repeats
    rng.shuffle(items)
    return torch.tensor(tok.encode("".join(format_example(q, a) for q, a in items)), dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator | None = None):
    if len(data) <= block_size + 1:  # tiny stream: tile it
        reps = (block_size + 2) // max(1, len(data)) + 1
        data = data.repeat(reps)
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def fixed_windows(data: torch.Tensor, block_size: int, max_windows: int = 32):
    """Deterministic evaluation windows so losses are comparable between versions."""
    if len(data) <= block_size + 1:
        data = data.repeat((block_size + 2) // max(1, len(data)) + 1)
    span = len(data) - block_size - 1
    n = min(max_windows, max(1, span))
    starts = [int(i * span / n) for i in range(n)]
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x, y
