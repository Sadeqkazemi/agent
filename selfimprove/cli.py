"""Command-line interface: python -m selfimprove <command>."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from .config import PRESETS, ModelConfig, TrainConfig, Workspace
from .context import Context
from .evaluator import answer, evaluate
from .loop import bootstrap, doctor, improve_cycle, read_journal, self_repair
from .registry import Registry
from .skills import add_knowledge
from .tokenizer import BPETokenizer


def cmd_init(ws, a):
    model_preset, train_preset = PRESETS[a.size]
    overrides = {"n_layer": a.layers, "n_embd": a.embd, "n_head": a.heads, "vocab_size": a.vocab, "block_size": a.block}
    mcfg = ModelConfig(**{**model_preset, **{k: v for k, v in overrides.items() if v is not None}})
    overrides = {"steps": a.steps, "lr": a.lr, "batch_size": a.batch, "grad_accum": a.accum}
    tcfg = TrainConfig(**{**train_preset, **{k: v for k, v in overrides.items() if v is not None}})
    if ws.registry.exists() and not a.force:
        sys.exit("already initialised (use --force to start over)")
    if a.force:
        for p in (ws.registry, ws.journal, ws.strategies):
            p.unlink(missing_ok=True)
        for p in ws.models.glob("*.pt"):
            p.unlink()
    bootstrap(ws, mcfg, tcfg)


def cmd_improve(ws, a):
    n = 0
    deadline = time.time() + a.hours * 3600 if a.hours else None
    while True:
        n += 1
        print(f"\n===== cycle {n} =====")
        improve_cycle(ws)
        if a.cycles and n >= a.cycles:
            break
        if deadline and time.time() > deadline:
            break


def cmd_eval(ws, a):
    reg = Registry(ws)
    model, _ = reg.load(a.version)
    report = evaluate(model, Context.load(ws))
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_chat(ws, a):
    reg = Registry(ws)
    model, _ = self_repair(ws, reg)
    tok = BPETokenizer.load(ws.tokenizer)
    print(f"model {reg.champion} ready. Commands: /teach question => answer, /quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/teach"):
            body = line[len("/teach"):]
            if "=>" not in body:
                print("usage: /teach question => answer")
                continue
            q, ans = (s.strip() for s in body.split("=>", 1))
            add_knowledge(ws.knowledge, q, ans)
            print("saved. It becomes part of the exam and training on the next `improve` cycle.")
            continue
        print("bot>", answer(model, tok, line, max_new_tokens=a.max_tokens, temperature=a.temperature))


def cmd_status(ws, a):
    reg = Registry(ws)
    print(f"champion: {reg.champion}")
    for vid, e in reg.versions.items():
        mark = "*" if vid == reg.champion else " "
        skills = " ".join(f"{k}={v:.0%}" for k, v in e["report"].get("skills", {}).items())
        print(f"{mark} {vid} {e['status']:<11} {e['action']:<18} score={e['score']}  {skills}  {e['note']}")
    if ws.strategies.exists():
        print("\nstrategy track record:")
        for k, s in json.loads(ws.strategies.read_text()).items():
            print(f"  {k:<18} tries={s['tries']} wins={s['wins']} avg_gain={s['gain_sum'] / s['tries']:+.4f}")
    j = read_journal(ws)
    if j and j[-1].get("failures"):
        print("\nlatest example mistakes:")
        for skill, fails in j[-1]["failures"].items():
            for f in fails[:2]:
                print(f"  [{skill}] {f['q']!r}: expected {f['expected']!r}, got {f['got']!r}")


def cmd_rollback(ws, a):
    reg = Registry(ws)
    to = reg.rollback(a.version, reason="manual")
    print(f"champion is now {to}" if to else "nothing to roll back to")


def cmd_teach(ws, a):
    add_knowledge(ws.knowledge, a.question, a.answer)
    print("saved to", ws.knowledge)


def cmd_doctor(ws, a):
    sys.exit(0 if doctor(ws) else 1)


def main(argv=None):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    p = argparse.ArgumentParser(prog="selfimprove", description="Offline self-improving language model")
    p.add_argument("--root", default=".", help="workspace folder containing data/ and runs/")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="train tokenizer and the first model")
    s.add_argument("--size", choices=list(PRESETS), default="tiny",
                   help="tiny ~1M, small ~12M, medium ~92M, large ~320M parameters")
    for flag in ("--layers", "--embd", "--heads", "--vocab", "--block", "--steps", "--batch", "--accum"):
        s.add_argument(flag, type=int, help="override the preset")
    s.add_argument("--lr", type=float, help="override the preset")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("improve", help="run self-improvement cycles")
    s.add_argument("--cycles", type=int, default=1, help="0 = run until --hours elapses or forever")
    s.add_argument("--hours", type=float, default=0)
    s.set_defaults(fn=cmd_improve)

    s = sub.add_parser("eval", help="evaluate a version (default: champion)")
    s.add_argument("--version")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("chat", help="talk to the champion")
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--max-tokens", type=int, default=48)
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("status", help="versions, strategy track record, recent mistakes")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("rollback", help="restore an earlier version")
    s.add_argument("--version")
    s.set_defaults(fn=cmd_rollback)

    s = sub.add_parser("teach", help="add a question/answer fact")
    s.add_argument("question")
    s.add_argument("answer")
    s.set_defaults(fn=cmd_teach)

    s = sub.add_parser("doctor", help="self-test and repair")
    s.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    ws = Workspace(a.root)
    ws.ensure()
    a.fn(ws, a)
