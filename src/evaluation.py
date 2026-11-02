"""Metrics for the three modules. Everything the dashboard shows comes from here."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from .config import NUM_TOKEN, UNK, EOS
from .preprocessing import lm_tokens


# ----------------------------------------------------------------- language ID
def lid_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Token accuracy, macro P/R/F1 and per-class scores over ``labels``."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [i for i, n in enumerate(s) if n > 0]
    return {
        "n_tokens": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": float(np.mean(p[present])),
        "macro_recall": float(np.mean(r[present])),
        "macro_f1": float(np.mean(f[present])),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


# --------------------------------------------------------------- normalization
def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or token lists)."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    vocab = {x: i for i, x in enumerate(set(a) | set(b))}
    A = np.fromiter((vocab[x] for x in a), dtype=np.int64, count=len(a))
    B = np.fromiter((vocab[x] for x in b), dtype=np.int64, count=len(b))
    idx = np.arange(len(B) + 1)
    prev = idx.copy()
    for i in range(1, len(A) + 1):
        cost = (B != A[i - 1]).astype(np.int64)
        tmp = np.empty(len(B) + 1, dtype=np.int64)
        tmp[0] = i
        tmp[1:] = np.minimum(prev[1:] + 1, prev[:-1] + cost)
        prev = np.minimum.accumulate(tmp - idx) + idx
    return int(prev[-1])


def _canon(text: str) -> list[str]:
    """Case-folded word sequence with punctuation removed (the unit WER/CER use)."""
    return lm_tokens(text)


def wer_cer(hyp: str, ref: str) -> tuple[int, int, int, int]:
    """(word_errors, ref_words, char_errors, ref_chars)."""
    h, r = _canon(hyp), _canon(ref)
    hs, rs = " ".join(h), " ".join(r)
    return edit_distance(h, r), len(r), edit_distance(hs, rs), len(rs)


def normalization_scores(hyps: list[str], refs: list[list[str]]) -> dict:
    """Corpus WER/CER/exact-match against one or several references per sentence.

    ``best_of_refs`` picks, per sentence, the reference with the fewest word
    errors (several spellings can be acceptable); ``first_ref`` uses the first
    annotator only.
    """
    out = {}
    for mode in ("first_ref", "best_of_refs"):
        we = wn = ce = cn = ex = 0
        for h, rs in zip(hyps, refs):
            cand = [wer_cer(h, r) for r in (rs[:1] if mode == "first_ref" else rs)]
            best = min(cand, key=lambda x: (x[0], x[2]))
            we += best[0]; wn += best[1]; ce += best[2]; cn += best[3]
            ex += best[0] == 0
        out[mode] = {"wer": we / max(wn, 1), "cer": ce / max(cn, 1),
                     "exact_match": ex / max(len(hyps), 1)}
    out["n_sentences"] = len(hyps)
    return out


# ------------------------------------------------------------ language models
def topk_accuracy_ngram(lm, sentences: list[list[int]], ks=(1, 5), skip: set[int] = frozenset()):
    """Same protocol as ``LSTMPredictor.topk_accuracy``: next-word accuracy over positions
    whose target is a real word (``<unk>``, ``<num>`` and ``</s>`` targets are ignored)."""
    kmax = max(ks)
    hits = {k: 0 for k in ks}
    total = 0
    for s in sentences:
        ctx: list[int] = []
        for w in s:
            if w not in skip:
                top = [lm.vocab.stoi[x] for x, _ in lm.predict_next(ctx, k=kmax)]
                for k in ks:
                    hits[k] += w in top[:k]
                total += 1
            ctx.append(w)
    return {k: hits[k] / total for k in ks}, total
