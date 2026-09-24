"""Verifiable skills: each can generate (question, answer) pairs and check a prediction.

Because answers can be checked automatically, the model can be examined and
can train on its own verified outputs without any human or external API.
Add a new skill by writing a generator and registering it in `builtin_skills`.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Skill:
    name: str
    generate: Callable[[random.Random], tuple[str, str]]
    # Fixed questions (e.g. knowledge facts) are both exam and training material.
    fixed: bool = False

    def check(self, prediction: str, answer: str) -> bool:
        return prediction.strip() == answer.strip()


def format_example(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}\n"


def format_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def _word(rng: random.Random, lo: int = 3, hi: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(lo, hi)))


def gen_add(rng):
    a, b = rng.randint(0, 99), rng.randint(0, 99)
    return f"{a}+{b}=", str(a + b)


def gen_reverse(rng):
    w = _word(rng)
    return f"reverse {w}", w[::-1]


def gen_sort(rng):
    ds = [rng.randint(0, 9) for _ in range(rng.randint(3, 6))]
    return "sort " + " ".join(map(str, ds)), " ".join(map(str, sorted(ds)))


def gen_copy(rng):
    ws = " ".join(_word(rng, 2, 4) for _ in range(rng.randint(1, 3)))
    return f"copy {ws}", ws


def load_knowledge(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    facts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            facts.append((d["q"], d["a"]))
    return facts


def add_knowledge(path: Path, question: str, answer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"q": question, "a": answer}, ensure_ascii=False) + "\n")


def builtin_skills(knowledge_path: Path | None = None) -> dict[str, Skill]:
    skills = {
        "add": Skill("add", gen_add),
        "reverse": Skill("reverse", gen_reverse),
        "sort": Skill("sort", gen_sort),
        "copy": Skill("copy", gen_copy),
    }
    facts = load_knowledge(knowledge_path) if knowledge_path else []
    if facts:
        skills["knowledge"] = Skill("knowledge", lambda rng: rng.choice(facts), fixed=True)
        skills["knowledge"].facts = facts  # type: ignore[attr-defined]
    return skills


def build_exam(skills: dict[str, Skill], size: int, seed: int = 1234) -> dict[str, list[tuple[str, str]]]:
    """A fixed, reproducible exam. Synthetic skills get held-out questions."""
    exam = {}
    for name, skill in skills.items():
        if skill.fixed:
            exam[name] = list(skill.facts)  # type: ignore[attr-defined]
            continue
        rng = random.Random(f"{seed}:{name}")
        seen, items = set(), []
        for _ in range(size * 20):
            q, a = skill.generate(rng)
            if q not in seen:
                seen.add(q)
                items.append((q, a))
            if len(items) == size:
                break
        exam[name] = items
    return exam


def sample_training(skill: Skill, rng: random.Random, n: int, exclude: set[str]) -> list[tuple[str, str]]:
    """Draw training pairs that never overlap the exam (except fixed facts)."""
    out = []
    for _ in range(n * 5):
        q, a = skill.generate(rng)
        if skill.fixed or q not in exclude:
            out.append((q, a))
        if len(out) == n:
            break
    return out
