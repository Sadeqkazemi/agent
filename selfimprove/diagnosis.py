"""Turns an evaluation report into a list of concrete problems and candidate fixes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Issue:
    kind: str
    severity: float  # 0..1, higher is worse
    detail: str
    actions: list[tuple[str, dict]] = field(default_factory=list)


def diagnose(report: dict, icfg, journal: list[dict], n_layer: int) -> list[Issue]:
    issues: list[Issue] = []

    last = journal[-1] if journal else {}
    if not report["healthy"] or last.get("diverged"):
        issues.append(Issue("instability", 1.0,
                            "training diverged or weights are not finite",
                            [("lower_lr", {})]))

    weak = {k: v for k, v in report["skills"].items() if v < icfg.target_accuracy}
    if weak:
        worst = sorted(weak, key=weak.get)
        sev = 1 - min(weak.values())
        issues.append(Issue("weak_skills", sev,
                            "below target: " + ", ".join(f"{k}={weak[k]:.0%}" for k in worst),
                            [("targeted_training", {"skills": worst}),
                             ("self_training", {"skills": worst})]))

    gap = report["val_loss"] - report["train_loss"]
    if gap > icfg.overfit_gap:
        issues.append(Issue("overfitting", min(1.0, gap / 2),
                            f"val_loss exceeds train_loss by {gap:.2f}",
                            [("regularize", {}), ("targeted_training", {"skills": list(weak)})]))

    if report["train_loss"] > icfg.underfit_loss:
        grow = [("grow", {"layers": 1})] if n_layer < icfg.max_layers else []
        issues.append(Issue("underfitting", min(1.0, (report["train_loss"] - icfg.underfit_loss) / 3 + 0.3),
                            f"train_loss {report['train_loss']:.2f} is still high",
                            [("longer_training", {})] + grow))

    if report["repetition"] > icfg.repetition_limit:
        issues.append(Issue("repetition", report["repetition"] - icfg.repetition_limit + 0.2,
                            f"{report['repetition']:.0%} of generated 3-grams repeat",
                            [("continue_training", {}), ("regularize", {})]))

    stalled = 0
    for entry in reversed(journal):
        if entry.get("accepted"):
            break
        stalled += 1
    if stalled >= icfg.plateau_cycles:
        acts = [("grow", {"layers": 1})] if n_layer < icfg.max_layers else []
        acts += [("lower_lr", {}), ("longer_training", {})]
        issues.append(Issue("plateau", min(1.0, 0.3 + 0.1 * stalled),
                            f"{stalled} cycles without improvement", acts))

    if not issues:
        issues.append(Issue("polish", 0.1, "no known problem; keep refining",
                            [("continue_training", {}), ("self_training", {"skills": list(report["skills"])})]))
    return sorted(issues, key=lambda i: -i.severity)
