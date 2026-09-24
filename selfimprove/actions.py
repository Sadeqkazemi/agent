"""Repair strategies. Each takes the champion and returns a trained candidate copy."""

from __future__ import annotations

import copy
import random

import torch

from .data import examples_stream, task_stream
from .evaluator import answer
from .skills import sample_training
from .trainer import train

N_TASK_EXAMPLES = 4000


def _mix(ctx, tcfg, rng, emphasis: dict[str, float] | None = None, task_fraction: float | None = None):
    weights = {n: float(tcfg.skill_weights.get(n, 1.0)) for n in ctx.skills}
    for n, w in (emphasis or {}).items():
        if n in weights:
            weights[n] *= w
    tf = tcfg.task_fraction if task_fraction is None else task_fraction
    tasks = task_stream(ctx.tok, ctx.skills, weights, N_TASK_EXAMPLES, ctx.exam, rng)
    return [(ctx.train_tokens, 1 - tf), (tasks, tf)]


def _run(model, tcfg, sources, log, seed):
    stats = train(model, tcfg, sources, log=log, seed=seed)
    return model, tcfg, stats


def continue_training(ctx, model, tcfg, params, log, seed):
    rng = random.Random(seed)
    return _run(model, tcfg, _mix(ctx, tcfg, rng), log, seed)


def targeted_training(ctx, model, tcfg, params, log, seed):
    rng = random.Random(seed)
    emphasis = {s: 4.0 for s in params.get("skills", [])}
    tf = max(0.6, tcfg.task_fraction)
    return _run(model, tcfg, _mix(ctx, tcfg, rng, emphasis, tf), log, seed)


def self_training(ctx, model, tcfg, params, log, seed, questions: int = 40, attempts: int = 3):
    """STaR-style: the model answers new questions, a verifier keeps only correct answers,
    and the model trains on its own verified work."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    verified = []
    for name in params.get("skills", []):
        skill = ctx.skills[name]
        exclude = {q for q, _ in ctx.exam.get(name, [])}
        ok = 0
        for q, a in sample_training(skill, rng, questions, exclude):
            budget = len(ctx.tok.encode(" " + a)) + 4
            for t in range(attempts):
                pred = answer(model, ctx.tok, q, budget, temperature=0.0 if t == 0 else 0.8)
                if skill.check(pred, a):
                    verified.append((q, pred))
                    ok += 1
                    break
        log(f"  self-training {name}: {ok}/{questions} answers verified")
    sources = _mix(ctx, tcfg, rng)
    if verified:
        sources.append((examples_stream(ctx.tok, verified, 4, rng), 0.5))
    return _run(model, tcfg, sources, log, seed)


def regularize(ctx, model, tcfg, params, log, seed):
    tcfg.weight_decay = min(0.5, tcfg.weight_decay * 1.5)
    model.set_dropout(min(0.3, model.cfg.dropout + 0.05))
    log(f"  dropout -> {model.cfg.dropout:.2f}, weight_decay -> {tcfg.weight_decay:.3f}")
    return continue_training(ctx, model, tcfg, params, log, seed)


def lower_lr(ctx, model, tcfg, params, log, seed):
    tcfg.lr = max(1e-5, tcfg.lr * 0.5)
    log(f"  lr -> {tcfg.lr:.2e}")
    return continue_training(ctx, model, tcfg, params, log, seed)


def longer_training(ctx, model, tcfg, params, log, seed):
    tcfg.steps = min(3000, int(tcfg.steps * 1.5))
    log(f"  steps per cycle -> {tcfg.steps}")
    return continue_training(ctx, model, tcfg, params, log, seed)


def grow(ctx, model, tcfg, params, log, seed):
    """Upgrade capacity: add layers that start as identity, so nothing learned is lost."""
    bigger = model.grown(params.get("layers", 1))
    log(f"  grew model: {model.cfg.n_layer} -> {bigger.cfg.n_layer} layers, "
        f"{model.num_params():,} -> {bigger.num_params():,} params")
    return continue_training(ctx, bigger, tcfg, params, log, seed)


ACTIONS = {
    "continue_training": continue_training,
    "targeted_training": targeted_training,
    "self_training": self_training,
    "regularize": regularize,
    "lower_lr": lower_lr,
    "longer_training": longer_training,
    "grow": grow,
}


def apply(name, ctx, model, tcfg, params, log, seed):
    """Run an action on copies so the champion is never modified."""
    return ACTIONS[name](ctx, copy.deepcopy(model), copy.deepcopy(tcfg), params, log, seed)
