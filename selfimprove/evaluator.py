"""Self-examination: measures loss, per-skill accuracy, repetition and numerical health."""

from __future__ import annotations

import torch

from .data import fixed_windows
from .skills import format_prompt
from .trainer import device


@torch.no_grad()
def corpus_loss(model, tokens: torch.Tensor) -> float:
    if len(tokens) < 2:
        return float("nan")
    x, y = fixed_windows(tokens, model.cfg.block_size)
    dev = device()
    model.to(dev).eval()
    _, loss = model(x.to(dev), y.to(dev))
    return loss.item()


@torch.no_grad()
def answer(model, tok, question: str, max_new_tokens: int = 24, temperature: float = 0.0) -> str:
    dev = device()
    model.to(dev).eval()
    ids = torch.tensor([tok.encode(format_prompt(question))], dtype=torch.long, device=dev)
    out = model.generate(
        ids, max_new_tokens, temperature=temperature, top_k=20,
        stop=lambda new: "\n" in tok.decode(new),
    )
    text = tok.decode(out[0, ids.size(1):].tolist())
    return text.split("\n")[0].strip()


@torch.no_grad()
def repetition(model, tok, tokens: torch.Tensor, n_prompts: int = 4, length: int = 48) -> float:
    """Share of repeated 3-grams in free-running greedy text (0 = varied, 1 = stuck in a loop)."""
    if len(tokens) < 32:
        return 0.0
    dev = device()
    model.to(dev).eval()
    scores = []
    for i in range(n_prompts):
        s = int(i * (len(tokens) - 16) / n_prompts)
        out = model.generate(tokens[s:s + 16].unsqueeze(0).to(dev), length)[0, 16:].tolist()
        grams = list(zip(out, out[1:], out[2:]))
        if grams:
            scores.append(1 - len(set(grams)) / len(grams))
    return sum(scores) / len(scores) if scores else 0.0


def score_of(report: dict) -> float:
    """Single number used to rank versions: mostly skill accuracy, a bit of language modelling."""
    accs = list(report["skills"].values())
    mean_acc = sum(accs) / len(accs) if accs else 0.0
    val = report["val_loss"]
    lm = 0.0 if val != val else 1.0 / (1.0 + val)  # NaN-safe
    return round(0.85 * mean_acc + 0.15 * lm - 0.05 * report["repetition"], 5)


def evaluate(model, ctx, keep_failures: int = 3) -> dict:
    healthy = model.is_finite()
    report = {"healthy": healthy, "skills": {}, "failures": {}}
    if not healthy:
        report.update(val_loss=float("nan"), train_loss=float("nan"), repetition=1.0, score=-1.0)
        return report
    report["val_loss"] = round(corpus_loss(model, ctx.val_tokens), 4)
    report["train_loss"] = round(corpus_loss(model, ctx.train_tokens), 4)
    for name, items in ctx.exam.items():
        correct, fails = 0, []
        for q, a in items:
            budget = len(ctx.tok.encode(" " + a)) + 4
            pred = answer(model, ctx.tok, q, max_new_tokens=budget)
            if ctx.skills[name].check(pred, a):
                correct += 1
            elif len(fails) < keep_failures:
                fails.append({"q": q, "expected": a, "got": pred})
        report["skills"][name] = round(correct / max(1, len(items)), 4)
        if fails:
            report["failures"][name] = fails
    report["repetition"] = round(repetition(model, ctx.tok, ctx.val_tokens), 4)
    report["score"] = score_of(report)
    return report
