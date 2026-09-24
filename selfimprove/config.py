"""Configuration dataclasses and workspace paths."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass
class TrainConfig:
    steps: int = 300
    batch_size: int = 32
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
    max_layers: int = 12
    ucb_c: float = 0.05            # exploration strength of the strategy planner


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
