"""A modern decoder-only transformer (RoPE, RMSNorm, SwiGLU, KV cache)
that can grow deeper without forgetting what it already knows."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ARCH_VERSION, ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight


def rope_tables(head_dim: int, max_len: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache: dict | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=k.size(2) == T and T > 1,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = 64 * math.ceil(8 * cfg.n_embd / 3 / 64)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.arch = ARCH_VERSION
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches: list[dict] | None = None):
        T = idx.size(1)
        start = caches[0]["k"].size(2) if caches and "k" in caches[0] else 0
        assert start + T <= self.cfg.block_size, "sequence longer than block_size"
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches is not None else None)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, stop=None):
        """Sample tokens with a KV cache. temperature=0 means greedy.
        `stop(new_ids)` ends generation early."""
        start, out = idx.size(1), idx
        caches = [{} for _ in self.blocks]
        logits, _ = self(idx[:, -self.cfg.block_size:], caches=caches)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            out = torch.cat([out, nxt], dim=1)
            if stop is not None and stop(out[0, start:].tolist()):
                break
            if caches[0]["k"].size(2) >= self.cfg.block_size:
                # context window full: slide it and rebuild the cache
                caches = [{} for _ in self.blocks]
                logits, _ = self(out[:, -(self.cfg.block_size // 2):], caches=caches)
            else:
                logits, _ = self(nxt, caches=caches)
        return out

    def set_dropout(self, p: float) -> None:
        self.cfg.dropout = p
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
            if isinstance(m, Attention):
                m.dropout = p

    def grown(self, extra_layers: int = 1) -> "GPT":
        """Return a deeper copy that computes exactly the same function.

        New blocks have zeroed output projections, so they start as identity
        on the residual stream and only then learn something useful.
        """
        cfg = copy.deepcopy(self.cfg)
        cfg.n_layer += extra_layers
        new = GPT(cfg).to(self.wte.weight.device)
        missing, _ = new.load_state_dict(self.state_dict(), strict=False)
        for i in range(self.cfg.n_layer, cfg.n_layer):
            nn.init.zeros_(new.blocks[i].attn.proj.weight)
            nn.init.zeros_(new.blocks[i].mlp.proj.weight)
        assert all(k.startswith("blocks.") for k in missing)
        return new

    def is_finite(self) -> bool:
        return all(torch.isfinite(p).all() for p in self.parameters())
