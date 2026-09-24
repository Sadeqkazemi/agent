"""Everything the loop needs, loaded once from the workspace."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from .config import ImproveConfig, Workspace
from .data import load_corpus, split_corpus
from .skills import Skill, build_exam, builtin_skills
from .tokenizer import BPETokenizer


def encoded_corpus(ws: Workspace, tok: BPETokenizer) -> list[int]:
    """Tokenizing a big corpus is slow, so the result is cached until text or tokenizer change."""
    text = load_corpus(ws.corpus)
    key = hashlib.sha1(text.encode("utf-8") + repr(tok.merges).encode()).hexdigest()
    cache = ws.runs / "corpus_tokens.pt"
    if cache.exists():
        saved = torch.load(cache, weights_only=True)
        if saved.get("key") == key:
            return saved["tokens"].tolist()
    tokens = tok.encode(text)
    ws.runs.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "tokens": torch.tensor(tokens, dtype=torch.int32)}, cache)
    return tokens


@dataclass
class Context:
    ws: Workspace
    tok: BPETokenizer
    skills: dict[str, Skill]
    exam: dict[str, list[tuple[str, str]]]
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    icfg: ImproveConfig

    @classmethod
    def load(cls, ws: Workspace, tok: BPETokenizer | None = None) -> "Context":
        icfg = ws.improve_config()
        tok = tok or BPETokenizer.load(ws.tokenizer)
        skills = builtin_skills(ws.knowledge)
        exam = build_exam(skills, icfg.exam_size)
        train_t, val_t = split_corpus(encoded_corpus(ws, tok))
        return cls(ws, tok, skills, exam, train_t, val_t, icfg)
