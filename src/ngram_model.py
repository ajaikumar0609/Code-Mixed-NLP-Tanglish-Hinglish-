"""Trigram language model with interpolated Kneser-Ney smoothing.

Implemented from scratch with sorted numpy arrays (no NLTK) so the maths in
the report can be mapped line by line onto the code:

    P3(w | u v) = max(c(uvw) - D, 0) / c(uv.)  +  D * N1+(uv.) / c(uv.) * P2(w | v)
    P2(w | v)   = max(N1+(.vw) - D, 0) / N1+(.v.) + D * N1+(v.) / N1+(.v.) * P1(w)
    P1(w)       = max(N1+(.w) - D, 0) / N1+(..)  +  D * |{w: N1+(.w)>0}| / N1+(..) * 1/|V|

where c() are raw counts, N1+(.vw) is the number of distinct words preceding
the bigram ``v w`` (continuation count), and the final 1/|V| term guarantees
that every vocabulary word (including ``<unk>``) has non-zero probability.

Unseen contexts back off to the next lower order. Keys ``(u, v, w)`` are packed
into one int64 (``V < 2**21``) so counting and lookup are vectorised.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import NON_SUGGESTIBLE
from .vocab import Vocab


def _lookup(sorted_keys: np.ndarray, values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """values[key == q] or 0 where q is absent (vectorised)."""
    if len(sorted_keys) == 0:
        return np.zeros(len(q), dtype=values.dtype)
    idx = np.searchsorted(sorted_keys, q)
    idx_c = np.minimum(idx, len(sorted_keys) - 1)
    hit = sorted_keys[idx_c] == q
    return np.where(hit, values[idx_c], 0)


class KNTrigramLM:
    def __init__(self, vocab: Vocab, discount: float = 0.75):
        self.vocab = vocab
        self.D = discount
        self.V = len(vocab)
        self._suggestible = np.array(
            [w not in NON_SUGGESTIBLE for w in vocab.itos], dtype=bool)

    # ------------------------------------------------------------------ fit
    def fit(self, sentences: list[list[int]]) -> "KNTrigramLM":
        V, bos, eos = self.V, self.vocab.bos, self.vocab.eos
        tri = []
        for s in sentences:
            seq = [bos, bos] + s + [eos]
            a = np.asarray(seq, dtype=np.int64)
            tri.append((a[:-2] * V + a[1:-1]) * V + a[2:])
        keys = np.concatenate(tri)
        self.k3, self.c3 = np.unique(keys, return_counts=True)       # trigram types / counts

        # context (u, v): total count and number of distinct followers
        ctx = self.k3 // V
        self.ctx3, first, inv = np.unique(ctx, return_index=True, return_inverse=True)
        self.ctx3_total = np.bincount(inv, weights=self.c3).astype(np.float64)
        self.ctx3_types = np.bincount(inv).astype(np.float64)

        # continuation counts for bigram level: N1+(. v w) = #trigram types ending in (v, w)
        suffix = self.k3 % (V * V)                                    # v * V + w
        self.k2, self.c2 = np.unique(suffix, return_counts=True)
        ctx2 = self.k2 // V
        self.ctx2, _, inv2 = np.unique(ctx2, return_index=True, return_inverse=True)
        self.ctx2_total = np.bincount(inv2, weights=self.c2).astype(np.float64)   # N1+(. v .)
        self.ctx2_types = np.bincount(inv2).astype(np.float64)                     # N1+(v .)

        # unigram level: N1+(. w) = #distinct preceding words over *bigram* types
        big = []
        for s in sentences:
            a = np.asarray([bos] + s + [eos], dtype=np.int64)
            big.append(a[:-1] * V + a[1:])
        bk = np.unique(np.concatenate(big))
        cont = np.bincount(bk % V, minlength=V).astype(np.float64)
        total = cont.sum()
        n_types = float((cont > 0).sum())
        self._cont, self._cont_total, self._cont_types = cont, total, n_types
        self._refresh_p1()
        return self

    def _refresh_p1(self) -> None:
        D = self.D
        self.p1 = (np.maximum(self._cont - D, 0) / self._cont_total
                   + D * self._cont_types / self._cont_total / self.V)
        # <pad>/<s> are never predicted; keep the distribution proper over the rest
        self.p1 = self.p1 / self.p1.sum()

    def set_discount(self, d: float) -> None:
        self.D = d
        self._refresh_p1()

    # ----------------------------------------------------------------- probs
    def prob_batch(self, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """P(w | u, v) for aligned arrays (vectorised)."""
        D, V = self.D, self.V
        u, v, w = (np.asarray(x, dtype=np.int64) for x in (u, v, w))
        p1 = self.p1[w]
        # bigram level
        tot2 = _lookup(self.ctx2, self.ctx2_total, v)
        typ2 = _lookup(self.ctx2, self.ctx2_types, v)
        c2 = _lookup(self.k2, self.c2, v * V + w).astype(np.float64)
        safe2 = np.where(tot2 > 0, tot2, 1.0)
        p2 = np.where(tot2 > 0, np.maximum(c2 - D, 0) / safe2 + D * typ2 / safe2 * p1, p1)
        # trigram level
        uv = u * V + v
        tot3 = _lookup(self.ctx3, self.ctx3_total, uv)
        typ3 = _lookup(self.ctx3, self.ctx3_types, uv)
        c3 = _lookup(self.k3, self.c3, uv * V + w).astype(np.float64)
        safe3 = np.where(tot3 > 0, tot3, 1.0)
        return np.where(tot3 > 0, np.maximum(c3 - D, 0) / safe3 + D * typ3 / safe3 * p2, p2)

    def _contexts(self, sentences: list[list[int]]):
        bos, eos = self.vocab.bos, self.vocab.eos
        U, Vv, W = [], [], []
        for s in sentences:
            a = np.asarray([bos, bos] + s + [eos], dtype=np.int64)
            U.append(a[:-2]); Vv.append(a[1:-1]); W.append(a[2:])
        return np.concatenate(U), np.concatenate(Vv), np.concatenate(W)

    def log_prob_sum(self, sentences: list[list[int]]) -> tuple[float, int]:
        U, Vv, W = self._contexts(sentences)
        p = self.prob_batch(U, Vv, W)
        return float(np.log(p).sum()), len(W)

    def perplexity(self, sentences: list[list[int]]) -> float:
        lp, n = self.log_prob_sum(sentences)
        return float(np.exp(-lp / n))

    def token_logprobs(self, ids: list[int]) -> list[float]:
        """Natural-log probability of every word in ``ids`` followed by ``</s>``."""
        U, Vv, W = self._contexts([ids])
        return np.log(self.prob_batch(U, Vv, W)).tolist()

    def tune_discount(self, val: list[list[int]], grid=(0.5, 0.6, 0.7, 0.75, 0.8, 0.9)) -> dict:
        scores = {}
        for d in grid:
            self.set_discount(d)
            scores[d] = self.perplexity(val)
        best = min(scores, key=scores.get)
        self.set_discount(best)
        return scores

    # ------------------------------------------------------------ prediction
    def _candidates(self, u: int, v: int, extra_unigram: int = 64) -> np.ndarray:
        V = self.V
        cands = [self._top_unigrams(extra_unigram)]
        i = np.searchsorted(self.k3, np.array([u * V + v, u * V + v + 1]) * V)
        cands.append(self.k3[i[0]:i[1]] % V)
        j = np.searchsorted(self.k2, np.array([v, v + 1]) * V)
        cands.append(self.k2[j[0]:j[1]] % V)
        c = np.unique(np.concatenate(cands))
        return c[self._suggestible[c]]

    def _top_unigrams(self, n: int) -> np.ndarray:
        if not hasattr(self, "_topuni") or len(self._topuni) < n:
            order = np.argsort(-np.where(self._suggestible, self.p1, -1.0))
            self._topuni = order[: max(n, 64)]
        return self._topuni[:n]

    def predict_next(self, context_ids: list[int], k: int = 5) -> list[tuple[str, float]]:
        """Exact top-k next words given the preceding word ids.

        Words unseen after the (bigram) context all share the backed-off
        unigram score order, so the top-k is guaranteed to lie within the
        union of trigram followers, bigram followers and top unigrams.
        """
        ctx = [self.vocab.bos, self.vocab.bos] + list(context_ids)
        u, v = ctx[-2], ctx[-1]
        cands = self._candidates(u, v, extra_unigram=k + 8)
        p = self.prob_batch(np.full(len(cands), u), np.full(len(cands), v), cands)
        order = np.argsort(-p)[:k]
        return [(self.vocab.itos[int(cands[i])], float(p[i])) for i in order]

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "KNTrigramLM":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj
