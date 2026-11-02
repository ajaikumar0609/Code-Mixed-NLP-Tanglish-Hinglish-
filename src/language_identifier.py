"""Token-level language identification.

Model A  CharNgramLID  : character n-gram TF-IDF + Logistic Regression (context-free baseline).
Model B  BiLSTMLID     : word embedding + character-CNN -> BiLSTM tagger (uses the whole sentence).

Both classify only *word* tokens into the labels of one variant
(``TA/EN/OTHER`` or ``HI/EN/OTHER``). Numbers, punctuation, URLs, mentions and
emoji get their labels from the deterministic rules in ``preprocessing``;
``identify_tokens`` merges the two so callers always see a label for every token.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import SEED
from .preprocessing import Token, rule_label


# --------------------------------------------------------------------------
# Model A
# --------------------------------------------------------------------------
def _case_feats(words: list[str]) -> sp.csr_matrix:
    rows = [[w[:1].isupper() and not w.isupper(), len(w) > 1 and w.isupper(),
             any(c.isdigit() for c in w), "'" in w] for w in words]
    return sp.csr_matrix(np.asarray(rows, dtype=np.float64))


class CharNgramLID:
    name = "Char n-gram + TF-IDF + LogReg"

    def __init__(self, labels: list[str], ngram_range=(1, 4), C: float = 10.0):
        self.labels = labels
        self.ngram_range, self.C = ngram_range, C

    def _x(self, words: list[str], fit: bool = False):
        low = [w.lower() for w in words]
        if fit:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=self.ngram_range,
                                       sublinear_tf=True, min_df=1, dtype=np.float64)
            X = self.vec.fit_transform(low)
        else:
            X = self.vec.transform(low)
        return sp.hstack([X, _case_feats(words)], format="csr")

    def fit(self, sentences: list[list[str]], labels: list[list[str]]) -> "CharNgramLID":
        # The model is context-free, so training on unique (token, label) types
        # weighted by their counts is mathematically identical to training on
        # every token occurrence, and much faster.
        cnt = Counter((w, l) for s, ls in zip(sentences, labels) for w, l in zip(s, ls))
        items = list(cnt.items())
        words = [w for (w, _), _ in items]
        y = [l for (_, l), _ in items]
        wts = np.asarray([c for _, c in items], dtype=np.float64)
        self.clf = LogisticRegression(C=self.C, max_iter=400, solver="lbfgs")
        self.clf.fit(self._x(words, fit=True), y, sample_weight=wts)
        return self

    def predict_proba_words(self, words: list[str]) -> np.ndarray:
        """(n_words, n_labels) probabilities, columns ordered as ``self.labels``."""
        if not words:
            return np.zeros((0, len(self.labels)))
        P = self.clf.predict_proba(self._x(words))
        out = np.zeros((len(words), len(self.labels)))
        for j, c in enumerate(self.clf.classes_):
            out[:, self.labels.index(c)] = P[:, j]
        return out

    def predict_sentence(self, words: list[str]) -> list[tuple[str, float]]:
        P = self.predict_proba_words(words)
        return [(self.labels[i], float(P[k, i])) for k, i in enumerate(P.argmax(1))]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "CharNgramLID":
        return joblib.load(path)


# --------------------------------------------------------------------------
# Model B
# --------------------------------------------------------------------------
MAX_CHARS = 24


class _Tagger(nn.Module):
    def __init__(self, n_words, n_chars, n_labels, w_dim=64, c_dim=32, c_filters=64,
                 hidden=128, dropout=0.3):
        super().__init__()
        self.cfg = dict(n_words=n_words, n_chars=n_chars, n_labels=n_labels, w_dim=w_dim,
                        c_dim=c_dim, c_filters=c_filters, hidden=hidden, dropout=dropout)
        self.wemb = nn.Embedding(n_words, w_dim, padding_idx=0)
        self.cemb = nn.Embedding(n_chars, c_dim, padding_idx=0)
        self.conv3 = nn.Conv1d(c_dim, c_filters, 3, padding=1)
        self.conv5 = nn.Conv1d(c_dim, c_filters, 5, padding=2)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(w_dim + 2 * c_filters, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_labels)

    def forward(self, words, chars):
        B, T, L = chars.shape
        c = self.cemb(chars.view(B * T, L)).transpose(1, 2)               # (BT, c_dim, L)
        mask = (chars.view(B * T, 1, L) != 0)
        c3 = F.relu(self.conv3(c)).masked_fill(~mask, -1e4).max(-1).values
        c5 = F.relu(self.conv5(c)).masked_fill(~mask, -1e4).max(-1).values
        c = torch.cat([c3, c5], -1).view(B, T, -1)
        c = torch.where(c < -1e3, torch.zeros_like(c), c)                 # padded tokens
        h = self.drop(torch.cat([self.wemb(words), c], -1))
        h, _ = self.lstm(h)
        return self.out(self.drop(h))


class BiLSTMLID:
    name = "BiLSTM (word + char-CNN) tagger"

    def __init__(self, labels: list[str]):
        self.labels = labels

    # -- encoding
    def _build_vocab(self, sentences, min_freq=2):
        wc = Counter(w.lower() for s in sentences for w in s)
        self.w2i = {"<pad>": 0, "<unk>": 1}
        for w, c in wc.items():
            if c >= min_freq:
                self.w2i[w] = len(self.w2i)
        cc = Counter(ch for s in sentences for w in s for ch in w)
        self.c2i = {"<pad>": 0, "<unk>": 1}
        for ch in cc:
            self.c2i[ch] = len(self.c2i)

    def _encode(self, sents: list[list[str]], word_dropout: float = 0.0, rng=None):
        T = max(len(s) for s in sents)
        W = np.zeros((len(sents), T), dtype=np.int64)
        C = np.zeros((len(sents), T, MAX_CHARS), dtype=np.int64)
        for i, s in enumerate(sents):
            for j, w in enumerate(s):
                wid = self.w2i.get(w.lower(), 1)
                if word_dropout and rng.random() < word_dropout:
                    wid = 1
                W[i, j] = wid
                for k, ch in enumerate(w[:MAX_CHARS]):
                    C[i, j, k] = self.c2i.get(ch, 1)
        return torch.from_numpy(W), torch.from_numpy(C)

    def fit(self, sentences, labels, val=None, epochs=10, batch=64, lr=2e-3, log=print,
            device=None):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        rng = random.Random(SEED)
        self.device = device or torch.device("cpu")
        self._build_vocab(sentences)
        self.model = _Tagger(len(self.w2i), len(self.c2i), len(self.labels)).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        l2i = {l: i for i, l in enumerate(self.labels)}
        pairs = [(s, [l2i[l] for l in ls]) for s, ls in zip(sentences, labels) if s]
        best, best_state = -1.0, None
        for ep in range(1, epochs + 1):
            self.model.train()
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + rng.random() * 4)
            chunks = [order[i:i + batch] for i in range(0, len(order), batch)]
            rng.shuffle(chunks)
            tot = 0.0
            for ch in chunks:
                sents = [pairs[i][0] for i in ch]
                W, C = self._encode(sents, word_dropout=0.1, rng=rng)
                Y = torch.full(W.shape, -100, dtype=torch.long)
                for r, i in enumerate(ch):
                    Y[r, : len(pairs[i][1])] = torch.tensor(pairs[i][1])
                logits = self.model(W.to(self.device), C.to(self.device))
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       Y.reshape(-1).to(self.device), ignore_index=-100)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                tot += loss.item()
            msg = f"  BiLSTM-LID epoch {ep}: loss {tot / len(chunks):.4f}"
            if val:
                acc = self.token_accuracy(*val)
                msg += f"  val acc {acc:.4f}"
                if acc > best:
                    best = acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            log(msg)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.device = torch.device("cpu")
        return self

    @torch.no_grad()
    def predict_proba_sentences(self, sents: list[list[str]], batch=128) -> list[np.ndarray]:
        self.model.eval()
        out = [None] * len(sents)
        idx = [i for i, s in enumerate(sents) if s]
        for i in range(len(sents)):
            if not sents[i]:
                out[i] = np.zeros((0, len(self.labels)))
        idx.sort(key=lambda i: len(sents[i]))
        for b in range(0, len(idx), batch):
            ch = idx[b:b + batch]
            W, C = self._encode([sents[i] for i in ch])
            P = F.softmax(self.model(W.to(self.device), C.to(self.device)), -1).cpu().numpy()
            for r, i in enumerate(ch):
                out[i] = P[r, : len(sents[i])]
        return out

    def predict_sentence(self, words: list[str]) -> list[tuple[str, float]]:
        P = self.predict_proba_sentences([words])[0]
        return [(self.labels[i], float(P[k, i])) for k, i in enumerate(P.argmax(1))]

    def token_accuracy(self, sentences, labels) -> float:
        probs = self.predict_proba_sentences(sentences)
        ok = n = 0
        for P, ls in zip(probs, labels):
            pred = [self.labels[i] for i in P.argmax(1)]
            ok += sum(p == g for p, g in zip(pred, ls)); n += len(ls)
        return ok / max(n, 1)

    def save(self, path: Path) -> None:
        torch.save({"labels": self.labels, "w2i": self.w2i, "c2i": self.c2i,
                    "cfg": self.model.cfg, "state": self.model.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> "BiLSTMLID":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = BiLSTMLID(ck["labels"])
        obj.w2i, obj.c2i = ck["w2i"], ck["c2i"]
        obj.model = _Tagger(**ck["cfg"])
        obj.model.load_state_dict(ck["state"])
        obj.model.eval()
        obj.device = torch.device("cpu")
        return obj


# --------------------------------------------------------------------------
# Merge rules + model for a full token sequence
# --------------------------------------------------------------------------
def identify_tokens(model, tokens: list[Token]) -> list[tuple[str, float]]:
    """Label every token: rules for NUM/PUNCT/URL/..., ``model`` for words.

    Returns ``(label, confidence)``; rule-based labels have confidence 1.0.
    """
    word_idx = [i for i, t in enumerate(tokens) if rule_label(t) is None]
    preds = model.predict_sentence([tokens[i].text for i in word_idx]) if word_idx else []
    out: list[tuple[str, float]] = [("OTHER", 1.0)] * len(tokens)
    for i, t in enumerate(tokens):
        rl = rule_label(t)
        if rl:
            out[i] = (rl, 1.0)
    for i, p in zip(word_idx, preds):
        out[i] = p
    return out
