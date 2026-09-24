"""Everything the loop needs, loaded once from the workspace."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ImproveConfig, Workspace
from .data import load_corpus, split_corpus
from .skills import Skill, build_exam, builtin_skills
from .tokenizer import BPETokenizer


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
        train_t, val_t = split_corpus(tok.encode(load_corpus(ws.corpus)))
        return cls(ws, tok, skills, exam, train_t, val_t, icfg)
