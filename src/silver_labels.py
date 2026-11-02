"""Weak ("silver") token-level language labels for Tanglish.

No public Tamil-English token-level LID corpus exists, so training labels for
the Tanglish identifier are produced by rules over curated lexicons plus
English word frequencies (``wordfreq``). The priority order is the annotation
convention of this project:

    1. non-ASCII / mojibake word          -> OTHER
    2. named entity / acronym lexicon     -> OTHER
    3. Tamil lexicon                      -> TA
    4. English chat-slang lexicon         -> EN
    5. single letters (a, i, u, r, k)     -> EN; other single letters -> OTHER
    6. common English word (zipf >= 3.0)  -> EN
    7. ALL-CAPS token not above           -> OTHER
    8. anything else (out of vocabulary)  -> TA

Elongated spellings (``hiii``, ``plsss``) are looked up through their reduced
candidates first. Labels are context-free by construction, which is why the
hand-reviewed gold set in ``data/gold/tanglish_lid_gold.tsv`` is used for the
honest evaluation of the learned models.
"""
from __future__ import annotations

import re
from .english import is_english as _is_english
from .preprocessing import Token, WORD, elongation_candidates, rule_label
from .tanglish_lexicon import EN_SLANG, NAMED_ENTITIES, TAMIL_WORDS

_SINGLE_EN = {"a", "i", "u", "r", "k"}
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def _lookup(w: str, original: str) -> str | None:
    if w in NAMED_ENTITIES:
        return "OTHER"
    if w in TAMIL_WORDS:
        return "TA"
    if w in EN_SLANG:
        return "EN"
    if len(w) == 1:
        return "EN" if w in _SINGLE_EN else "OTHER"
    if _is_english(w):
        return "EN"
    return None


def silver_label_word(text: str) -> str:
    """Label one WORD token (any casing) as TA / EN / OTHER."""
    if _NON_ASCII.search(text):
        return "OTHER"
    w = text.lower()
    lab = _lookup(w, text)
    if lab:
        return lab
    for cand in elongation_candidates(w):
        lab = _lookup(cand, text)
        if lab:
            return lab
    if len(text) >= 2 and text.isupper():
        return "OTHER"
    return "TA"


def silver_label_tokens(tokens: list[Token]) -> list[str]:
    labels = []
    for t in tokens:
        rl = rule_label(t)
        labels.append(rl if rl else silver_label_word(t.text))
    return labels
