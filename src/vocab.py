"""Vocabulary shared by the N-gram and LSTM language models.

Both models use the *same* vocabulary object (built once from the training
split), which is what makes their perplexities comparable.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .config import BOS, EOS, PAD, SPECIALS, UNK


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad, self.unk = self.stoi[PAD], self.stoi[UNK]
        self.bos, self.eos = self.stoi[BOS], self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, sentences: Iterable[list[str]], min_count: int = 3,
              max_size: int | None = None) -> "Vocab":
        counts = Counter(w for s in sentences for w in s)
        words = [w for w, c in counts.most_common() if c >= min_count and w not in SPECIALS]
        if max_size:
            words = words[: max_size - len(SPECIALS)]
        return cls(SPECIALS + words)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.unk
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def read_corpus(path: Path) -> list[list[str]]:
    """One whitespace-tokenised sentence per line."""
    with open(path, encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]
