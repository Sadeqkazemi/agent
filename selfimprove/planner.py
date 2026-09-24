"""Chooses which fix to try, learning from which strategies actually helped before."""

from __future__ import annotations

import json
import math

from .config import Workspace
from .diagnosis import Issue


class StrategyMemory:
    """Per-action track record: how often it was tried, won, and the average score gain."""

    def __init__(self, ws: Workspace):
        self.ws = ws
        self.stats: dict = json.loads(ws.strategies.read_text()) if ws.strategies.exists() else {}

    def update(self, action: str, gain: float, accepted: bool) -> None:
        s = self.stats.setdefault(action, {"tries": 0, "wins": 0, "gain_sum": 0.0})
        s["tries"] += 1
        s["wins"] += int(accepted)
        s["gain_sum"] += max(-0.1, min(0.1, gain))
        self.ws.strategies.write_text(json.dumps(self.stats, indent=2))

    def value(self, action: str, c: float) -> float:
        """Upper-confidence bound: proven gain + bonus for rarely tried strategies."""
        s = self.stats.get(action, {"tries": 0, "gain_sum": 0.0})
        total = sum(v["tries"] for v in self.stats.values())
        mean = s["gain_sum"] / s["tries"] if s["tries"] else 0.0
        return mean + c * math.sqrt(math.log(total + 2) / (s["tries"] + 1))


def choose(issues: list[Issue], memory: StrategyMemory, c: float) -> tuple[Issue, str, dict]:
    best, best_val = None, -float("inf")
    for issue in issues[:2]:  # focus on the two most severe problems
        for rank, (action, params) in enumerate(issue.actions):
            # severity and the diagnosis' own ordering act as a prior
            val = 0.1 * issue.severity + memory.value(action, c) - 0.005 * rank
            if val > best_val:
                best, best_val = (issue, action, params), val
    assert best is not None
    return best
