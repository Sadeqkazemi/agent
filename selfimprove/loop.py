"""The self-improvement cycle: examine -> diagnose -> fix -> re-examine -> promote or reject."""

from __future__ import annotations

import json
import random
import time

import torch

from . import actions
from .config import ModelConfig, TrainConfig, Workspace
from .context import Context
from .data import load_corpus
from .diagnosis import diagnose
from .evaluator import evaluate
from .model import GPT
from .planner import StrategyMemory, choose
from .registry import Registry
from .skills import builtin_skills, format_example, sample_training
from .tokenizer import BPETokenizer


def read_journal(ws: Workspace) -> list[dict]:
    if not ws.journal.exists():
        return []
    return [json.loads(l) for l in ws.journal.read_text().splitlines() if l.strip()]


def _append_journal(ws: Workspace, entry: dict) -> None:
    with ws.journal.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def bootstrap(ws: Workspace, mcfg: ModelConfig, tcfg: TrainConfig, log=print) -> dict:
    """Train the tokenizer and the first model (v0001) from local data only."""
    ws.ensure()
    corpus = load_corpus(ws.corpus)
    if len(corpus) < 1000:
        log("warning: corpus is very small; add .txt files to data/corpus for a better model")
    rng = random.Random(0)
    task_sample = "".join(
        format_example(q, a)
        for s in builtin_skills(ws.knowledge).values()
        for q, a in sample_training(s, rng, 300, set())
    )
    log("training tokenizer ...")
    tok = BPETokenizer.train(corpus + "\n" + task_sample, mcfg.vocab_size)
    tok.save(ws.tokenizer)
    mcfg.vocab_size = tok.vocab_size
    ctx = Context.load(ws, tok)
    torch.manual_seed(0)
    model = GPT(mcfg)
    log(f"new model: {model.num_params():,} parameters, {mcfg.n_layer} layers, vocab {tok.vocab_size}")
    model, tcfg, stats = actions.continue_training(ctx, model, tcfg, {}, log, seed=0)
    report = evaluate(model, ctx)
    reg = Registry(ws)
    vid = reg.next_id()
    reg.record(vid, None, "init", report, accepted=True, model=model, tcfg=tcfg)
    _append_journal(ws, {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": vid,
                         "action": "init", "accepted": True, "score": report["score"],
                         "diverged": stats["diverged"]})
    log(f"{vid} is the champion. score={report['score']:.4f} skills={report['skills']}")
    return report


def judge(old: dict, new: dict, icfg) -> tuple[bool, str]:
    if not new["healthy"]:
        return False, "candidate is numerically broken"
    for skill, acc in old["skills"].items():
        if skill in new["skills"] and acc - new["skills"][skill] > icfg.regression_tolerance:
            return False, f"regression on {skill}: {acc:.0%} -> {new['skills'][skill]:.0%}"
    delta = new["score"] - old["score"]
    if delta < icfg.min_delta:
        return False, f"not better enough (delta {delta:+.4f})"
    return True, f"improved (delta {delta:+.4f})"


def self_repair(ws: Workspace, reg: Registry, log=print):
    """Load the champion; if it is missing or corrupt, fall back to the last good version."""
    while True:
        try:
            model, tcfg = reg.load()
            if model.is_finite():
                return model, tcfg
            problem = "weights are not finite"
        except Exception as e:  # corrupt or missing checkpoint
            problem = f"{type(e).__name__}: {e}"
        bad = reg.champion
        log(f"self-repair: champion {bad} is broken ({problem})")
        if reg.rollback(reason=problem) is None:
            raise RuntimeError("no healthy version left to restore; run `init` again")
        log(f"self-repair: restored {reg.champion}")


def improve_cycle(ws: Workspace, log=print, seed: int | None = None) -> dict:
    reg = Registry(ws)
    ctx = Context.load(ws)
    model, tcfg = self_repair(ws, reg, log)
    champion = reg.champion
    journal = read_journal(ws)
    seed = seed if seed is not None else len(journal) + 1

    log(f"[examine] champion {champion} ({model.cfg.n_layer} layers, {model.num_params():,} params)")
    old = evaluate(model, ctx)
    log(f"  score {old['score']:.4f}  val_loss {old['val_loss']:.3f}  train_loss {old['train_loss']:.3f}  "
        f"repetition {old['repetition']:.2f}")
    log("  skills " + "  ".join(f"{k}={v:.0%}" for k, v in old["skills"].items()))

    issues = diagnose(old, ctx.icfg, journal, model.cfg.n_layer)
    log("[diagnose]")
    for i in issues:
        log(f"  - {i.kind} (severity {i.severity:.2f}): {i.detail}")

    memory = StrategyMemory(ws)
    issue, action, params = choose(issues, memory, ctx.icfg.ucb_c)
    log(f"[fix] {action} {params or ''} for '{issue.kind}'")
    cand, cand_tcfg, stats = actions.apply(action, ctx, model, tcfg, params, log, seed)

    new = evaluate(cand, ctx)
    accepted, reason = judge(old, new, ctx.icfg)
    gain = new["score"] - old["score"] if new["healthy"] else -0.1
    log(f"[verify] candidate score {new['score']:.4f}  skills "
        + "  ".join(f"{k}={v:.0%}" for k, v in new["skills"].items()))

    vid = reg.next_id()
    reg.record(vid, champion, action, new, accepted, cand if accepted else None, cand_tcfg, reason)
    memory.update(action, gain, accepted)
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "version": vid, "parent": champion,
             "issue": issue.kind, "action": action, "params": params, "accepted": accepted,
             "reason": reason, "old_score": old["score"], "score": new["score"],
             "diverged": stats["diverged"], "seconds": stats["seconds"], "failures": old["failures"]}
    _append_journal(ws, entry)
    log(f"[{'PROMOTED' if accepted else 'REJECTED'}] {vid}: {reason}. champion is now {reg.champion}")
    return entry


def doctor(ws: Workspace, log=print) -> bool:
    """Self-test of the whole system; repairs what it can."""
    ok = True

    def check(name, cond, fix=None):
        nonlocal ok
        log(f"  [{'ok' if cond else 'FAIL'}] {name}")
        if not cond:
            ok = False
            if fix:
                fix()

    tok = BPETokenizer.load(ws.tokenizer) if ws.tokenizer.exists() else None
    check("tokenizer exists", tok is not None)
    if tok:
        sample = "Hello 123 سلام دنیا! ✓"
        check("tokenizer round-trip (Persian, digits, symbols)", tok.decode(tok.encode(sample)) == sample)
    reg = Registry(ws)
    check("registry has a champion", reg.champion is not None)
    if reg.champion:
        missing = [v for v in reg.accepted_versions() if not reg.path(v).exists()]
        check("all accepted checkpoints present", not missing)
        try:
            model, _ = self_repair(ws, reg, log)
            check("champion loads and weights are finite", True)
            x = torch.randint(0, model.cfg.vocab_size, (1, 8))
            with torch.no_grad():
                a = model(x)[0]
                b = model.grown(1).eval()(x)[0]
            check("growth preserves behaviour", torch.allclose(a, b, atol=1e-5))
        except RuntimeError as e:
            check(f"champion usable ({e})", False)
    return ok
