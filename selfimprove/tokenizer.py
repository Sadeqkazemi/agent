"""Byte-level BPE tokenizer, trained locally. Handles any UTF-8 text (Persian included)."""

from __future__ import annotations

import json
import re
from collections import Counter
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
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        words = Counter(PAT.findall(text))
        seqs = {w: list(w.encode("utf-8")) for w in words}
        merges: list[tuple[int, int]] = []
        for i in range(max(0, vocab_size - 256)):
            pairs: Counter = Counter()
            for w, count in words.items():
                s = seqs[w]
                for p in zip(s, s[1:]):
                    pairs[p] += count
            if not pairs:
                break
            best, count = pairs.most_common(1)[0]
            if count < 2:
                break
            merges.append(best)
            for w in words:
                seqs[w] = _merge(seqs[w], best, 256 + i)
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
