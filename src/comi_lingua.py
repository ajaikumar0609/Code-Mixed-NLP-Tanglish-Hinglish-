"""Loaders for the COMI-LINGUA Hinglish benchmark (LID and TN subsets).

Raw files are the CC-BY-4.0 CSVs from huggingface.co/datasets/LingoIITGN/COMI-LINGUA.
The release stores annotations in two slightly different formats
(``{'key','value','index'}`` dicts and ``[token, tag-initial]`` pairs), so the
parser accepts both and maps them onto one label vocabulary.
"""
from __future__ import annotations

import ast
from collections import Counter

import pandas as pd

from .config import RAW
from .preprocessing import WORD, has_native_script, rule_label, tokenize

_TAG_MAP = {"hi": "HI", "h": "HI", "en": "EN", "e": "EN", "ot": "OTHER", "u": "OTHER", "o": "OTHER"}


def _parse_cell(cell: str) -> list[tuple[str, str]] | None:
    try:
        items = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return None
    out = []
    for it in items:
        if isinstance(it, dict):
            key, val = it.get("key"), it.get("value")
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            key, val = it
        else:
            return None
        val = _TAG_MAP.get(str(val).lower())
        if key is None or val is None:
            return None
        out.append((str(key), val))
    return out


def load_lid(split: str, roman_only: bool = True) -> list[dict]:
    """Return ``[{'sentence': str, 'tokens': [...], 'labels': [...]}]``.

    Gold labels are the majority vote of the three annotators (ties fall back
    to annotator 1). Sentences where the annotators disagree on tokenization
    are skipped, since votes cannot be aligned.
    """
    df = pd.read_csv(RAW / "comi_lingua" / f"LID_{split}.csv")
    ann_cols = [c for c in df.columns if c.startswith("Annotated by")]
    stats = Counter()
    out = []
    for _, row in df.iterrows():
        sent = row["Sentences"]
        if not isinstance(sent, str):
            stats["empty"] += 1
            continue
        if roman_only and has_native_script(sent):
            stats["native_script"] += 1
            continue
        anns = [_parse_cell(str(row[c])) for c in ann_cols]
        if any(a is None for a in anns):
            stats["unparsable"] += 1
            continue
        if len({tuple(k for k, _ in a) for a in anns}) != 1:
            stats["tokenization_mismatch"] += 1
            continue
        toks, labels = [], []
        for pos in range(len(anns[0])):
            key = anns[0][pos][0]
            votes = [a[pos][1] for a in anns]
            top, n = Counter(votes).most_common(1)[0]
            label = top if n >= 2 else votes[0]
            for t in tokenize(key):
                rl = rule_label(t)
                toks.append(t.text)
                labels.append(rl if rl else label)
        if toks:
            out.append({"sentence": sent, "tokens": toks, "labels": labels})
    load_lid.last_stats = dict(stats)
    return out


def load_tn(split: str, roman_only: bool = True) -> list[dict]:
    """Return ``[{'original': str, 'refs': [3 annotator normalisations]}]``."""
    df = pd.read_csv(RAW / "comi_lingua" / f"TN_{split}.csv")
    ann_cols = [c for c in df.columns if c.startswith("Annotated by")]
    out = []
    for _, row in df.iterrows():
        orig = row["Sentences"]
        refs = [row[c] for c in ann_cols]
        if not isinstance(orig, str) or any(not isinstance(r, str) for r in refs):
            continue
        if roman_only and (has_native_script(orig) or any(has_native_script(r) for r in refs)):
            continue
        out.append({"original": orig, "refs": refs})
    return out
