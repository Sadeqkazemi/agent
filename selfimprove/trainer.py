"""Training loop with warmup + cosine schedule and divergence detection."""

from __future__ import annotations

import math
import time

import torch

from .config import TrainConfig
from .data import get_batch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def train(model, cfg: TrainConfig, sources: list[tuple[torch.Tensor, float]], log=print, seed: int = 0) -> dict:
    """Train in place on a weighted mix of token streams. Returns training stats."""
    dev = device()
    model.to(dev).train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95),
    )
    sources = [(s, w) for s, w in sources if len(s) > 1 and w > 0]
    weights = torch.tensor([w for _, w in sources], dtype=torch.float)
    gen = torch.Generator().manual_seed(seed)
    bs, T = cfg.batch_size, model.cfg.block_size
    losses, diverged, t0 = [], False, time.time()
    use_amp = dev.type == "cuda"
    micro = max(1, cfg.grad_accum)
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        total = 0.0
        for _ in range(micro):
            # split the micro-batch across sources proportionally to their weights
            counts = torch.multinomial(weights, bs, replacement=True, generator=gen).bincount(minlength=len(sources))
            xs, ys = [], []
            for (data, _), c in zip(sources, counts.tolist()):
                if c:
                    x, y = get_batch(data, T, c, gen)
                    xs.append(x)
                    ys.append(y)
            x, y = torch.cat(xs).to(dev), torch.cat(ys).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                break
            (loss / micro).backward()
            total += loss.item() / micro
        if not torch.isfinite(loss):
            diverged = True
            log(f"  step {step}: loss is not finite, stopping")
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total)
        if step % max(1, cfg.steps // 5) == 0 or step == cfg.steps - 1:
            log(f"  step {step:5d}/{cfg.steps}  loss {total:.3f}  lr {lr_at(step, cfg):.2e}")
    model.eval()
    tail = losses[-max(1, len(losses) // 10):] if losses else [float("nan")]
    return {
        "steps": len(losses),
        "final_loss": sum(tail) / len(tail),
        "diverged": diverged or not model.is_finite(),
        "seconds": round(time.time() - t0, 1),
    }
