"""Byte-level BPE tokenizer, trained locally. Handles any UTF-8 text (Persian included)."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Pre-tokenization: words (with optional leading space), single digits,
# punctuation runs, whitespace runs. Digits stay single so arithmetic is learnable.
PAT = re.compile(r" ?[^\W\d]+|\d| ?[^\w\s]+|\s+")


def _merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, max_chars: int = 50_000_000) -> "BPETokenizer":
        """Incremental BPE: only words containing the merged pair are touched each step,
        so training stays fast on large corpora."""
        words = Counter(PAT.findall(text[:max_chars]))
        seqs = [list(w.encode("utf-8")) for w in words]
        freq = list(words.values())
        pair_count: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, s in enumerate(seqs):
            for p in zip(s, s[1:]):
                pair_count[p] += freq[wi]
                where[p].add(wi)
        merges: list[tuple[int, int]] = []
        for i in range(max(0, vocab_size - 256)):
            if not pair_count:
                break
            best = max(pair_count, key=pair_count.__getitem__)
            if pair_count[best] < 2:
                break
            new_id = 256 + i
            merges.append(best)
            for wi in where.pop(best, ()):
                old, c = seqs[wi], freq[wi]
                new = _merge(old, best, new_id)
                if new == old:
                    continue
                for p in zip(old, old[1:]):
                    pair_count[p] -= c
                    if pair_count[p] <= 0:
                        del pair_count[p]
                for p in zip(new, new[1:]):
                    pair_count[p] += c
                    where[p].add(wi)
                seqs[wi] = new
            pair_count.pop(best, None)
        return cls(merges)

    def _encode_word(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        seq = list(word.encode("utf-8"))
        while len(seq) > 1:
            pair = min(zip(seq, seq[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            if pair not in self.ranks:
                break
            seq = _merge(seq, pair, 256 + self.ranks[pair])
        if len(self._cache) < 100_000:
            self._cache[word] = seq
        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in PAT.findall(text):
            ids.extend(self._encode_word(word))
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab.get(int(i), b"") for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"merges": self.merges}))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(json.loads(Path(path).read_text())["merges"])
