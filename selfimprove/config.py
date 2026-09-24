"""Configuration dataclasses and workspace paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


ARCH_VERSION = 2  # bump when the network layout changes incompatibly


@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    arch: int = ARCH_VERSION


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 32
    grad_accum: int = 1            # micro-batches per optimizer step
    lr: float = 3e-3
    min_lr_ratio: float = 0.1
    warmup: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Share of each batch drawn from skill tasks (the rest is raw corpus).
    task_fraction: float = 0.5
    # Relative sampling weight per skill; missing skills default to 1.0.
    skill_weights: dict = field(default_factory=dict)


@dataclass
class ImproveConfig:
    exam_size: int = 30            # exam questions per skill
    target_accuracy: float = 0.9   # a skill below this is "weak"
    overfit_gap: float = 0.4       # val_loss - train_loss above this is overfitting
    underfit_loss: float = 2.0     # train_loss above this is underfitting
    repetition_limit: float = 0.6  # repeated 3-gram share above this is degenerate
    plateau_cycles: int = 3        # cycles without promotion before "plateau"
    min_delta: float = 0.002       # candidate must beat champion score by this
    regression_tolerance: float = 0.1  # max allowed per-skill accuracy drop
    max_layers: int = 48
    ucb_c: float = 0.05            # exploration strength of the strategy planner


# Size presets. Parameter counts are approximate and include embeddings.
PRESETS = {
    #          model                                                           training
    "tiny":   (dict(n_layer=4, n_head=4, n_embd=128, block_size=128, vocab_size=512),
               dict(steps=600, lr=3e-3, batch_size=32, grad_accum=1)),          # ~1M,   CPU minutes
    "small":  (dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=4096),
               dict(steps=2000, lr=1e-3, batch_size=16, grad_accum=2)),         # ~12M,  CPU hours / GPU minutes
    "medium": (dict(n_layer=12, n_head=12, n_embd=768, block_size=512, vocab_size=8192),
               dict(steps=5000, lr=6e-4, batch_size=16, grad_accum=4)),         # ~92M,  needs a GPU
    "large":  (dict(n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=16384),
               dict(steps=20000, lr=3e-4, batch_size=8, grad_accum=16)),        # ~320M, needs a strong GPU
}


def from_dict(cls, data: dict | None):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})


def to_dict(obj) -> dict:
    return asdict(obj)


class Workspace:
    """All on-disk state lives under two folders: data/ (inputs) and runs/ (outputs)."""

    def __init__(self, root: str | Path = "."):
        self.root = Path(root)
        self.data = self.root / "data"
        self.corpus = self.data / "corpus"
        self.knowledge = self.data / "knowledge.jsonl"
        self.runs = self.root / "runs"
        self.models = self.runs / "models"
        self.tokenizer = self.runs / "tokenizer.json"
        self.registry = self.runs / "registry.json"
        self.journal = self.runs / "journal.jsonl"
        self.strategies = self.runs / "strategies.json"
        self.settings = self.runs / "settings.json"

    def ensure(self) -> None:
        for d in (self.corpus, self.models):
            d.mkdir(parents=True, exist_ok=True)

    def improve_config(self) -> ImproveConfig:
        if self.settings.exists():
            return from_dict(ImproveConfig, json.loads(self.settings.read_text()))
        return ImproveConfig()
