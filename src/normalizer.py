"""Rule- and dictionary-based text normalizer for Tanglish / Hinglish.

Pipeline (per the project design):

    raw text
      -> Unicode / whitespace clean-up and tokenization
      -> punctuation and emoji run collapsing        (``!!!`` -> ``!``)
      -> elongation reduction against a known-word list  (``rombbba`` -> ``romba``)
      -> language-aware spelling dictionary           (``plz`` -> ``please``, ``iruku`` -> ``irukku``)
      -> normalized text

Normalization is *not* translation: Tamil/Hindi words are only mapped to a
preferred Romanized spelling, never to English. The preferred spellings are the
project's convention: for Hinglish they are learned from the aligned
COMI-LINGUA TN pairs (``learn_dictionary``); for Tanglish they come from the
curated groups in ``tanglish_lexicon``.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from .config import LEXICONS
from .preprocessing import (EMOJI, PUNCT, WORD, Token, detokenize,
                            elongation_candidates, tokenize, words_only)
from .english import is_english as _is_english
from .tanglish_lexicon import EN_SLANG_EXPANSIONS, VARIANT_GROUPS


@dataclass
class NormToken:
    original: str
    normalized: str
    rule: str | None = None          # which rule changed it (None = unchanged)
    tag: str | None = None

    @property
    def changed(self) -> bool:
        return self.original != self.normalized


@dataclass
class NormResult:
    original: str
    text: str
    tokens: list[NormToken] = field(default_factory=list)


def _match_case(src: str, new: str) -> str:
    if len(src) > 1 and src.isupper():
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


class Normalizer:
    def __init__(self, variant: str, known: Counter | None = None,
                 learned: dict[str, tuple[str, str | None]] | None = None):
        self.variant = variant
        self.known = known or Counter()
        self.learned = learned or {}
        self.variants: dict[str, str] = {}
        if variant == "tanglish":
            for pref, alts in VARIANT_GROUPS.items():
                for a in alts.split():
                    self.variants[a] = pref
        self.slang = dict(EN_SLANG_EXPANSIONS)

    # ------------------------------------------------------------ factory
    @classmethod
    def for_variant(cls, variant: str, lexicon_dir: Path = LEXICONS) -> "Normalizer":
        known, learned = Counter(), {}
        kp = lexicon_dir / f"known_words_{variant}.json"
        if kp.exists():
            known = Counter(json.loads(kp.read_text()))
        lp = lexicon_dir / f"learned_dict_{variant}.json"
        if lp.exists():
            learned = {k: tuple(v) for k, v in json.loads(lp.read_text()).items()}
        return cls(variant, known, learned)

    # ------------------------------------------------------------ words
    def _is_known(self, w: str) -> bool:
        return (self.known.get(w, 0) > 0 or w in self.variants or w in self.slang
                or w in self.variants.values() or w in self.learned)

    def _reduce_elongation(self, w: str) -> tuple[str, bool]:
        cands = elongation_candidates(w)
        if not cands:
            return w, False
        scored = [(self.known.get(c, 0) + (1 if self._is_known(c) else 0), -len(c), c) for c in cands]
        best = max(scored)
        if best[0] > 0:
            return best[2], True
        return max(cands, key=len), True      # conservative: keep a double letter

    def _lookup(self, w: str, tag: str | None) -> tuple[str, str] | None:
        if tag != "EN":
            tgt = self.variants.get(w)
            # variants that are also English words (e.g. "than") need a TA/HI tag
            if tgt and (tag is not None or not _is_english(w)):
                return tgt, "variant"
        if tag in ("EN", None):
            tgt = self.slang.get(w)
            if tgt:
                return tgt, "slang"
        ent = self.learned.get(w)
        if ent:
            tgt, lang = ent
            if tag is None or lang is None or lang == tag:
                return tgt, "learned"
        return None

    def normalize_word(self, text: str, tag: str | None = None) -> tuple[str, str | None]:
        if tag == "OTHER":
            return text, None
        lw = text.lower()
        rule, cur = None, lw
        cur, did = self._reduce_elongation(lw)
        rule = "elongation" if did and cur != lw else None
        hit = self._lookup(cur, tag)
        if hit:
            cur, rule = hit[0], hit[1]
        if cur == lw:
            return text, None
        return _match_case(text, cur), rule

    # ------------------------------------------------------------ sentences
    def normalize(self, text: str, tags: list[str] | None = None) -> NormResult:
        toks = tokenize(text)
        if tags is not None and len(tags) != len(toks):
            tags = None
        out: list[Token] = []
        items: list[NormToken] = []
        i = 0
        while i < len(toks):
            t = toks[i]
            tag = tags[i] if tags else None
            if t.kind == PUNCT:
                j = i
                while j + 1 < len(toks) and toks[j + 1].kind == PUNCT and toks[j + 1].text == t.text:
                    j += 1
                run = j - i + 1
                if run > 1:
                    new = "..." if t.text == "." else t.text
                    items.append(NormToken(t.text * run, new, "punct_run", "PUNCT"))
                    out.append(Token(new, PUNCT, t.space_before))
                else:
                    items.append(NormToken(t.text, t.text, None, "PUNCT"))
                    out.append(Token(t.text, PUNCT, t.space_before))
                i = j + 1
                continue
            if t.kind == EMOJI:
                new = re.sub(r"(.)\1+", r"\1", t.text)
                items.append(NormToken(t.text, new, "emoji_run" if new != t.text else None, "OTHER"))
                out.append(Token(new, EMOJI, t.space_before))
            elif t.kind == WORD:
                new, rule = self.normalize_word(t.text, tag)
                items.append(NormToken(t.text, new, rule, tag))
                parts = new.split(" ")
                for k, p in enumerate(parts):
                    out.append(Token(p, WORD, t.space_before if k == 0 else True))
            else:
                items.append(NormToken(t.text, t.text, None, tag))
                out.append(Token(t.text, t.kind, t.space_before))
            i += 1
        return NormResult(original=text, text=detokenize(out), tokens=items)

    def __call__(self, text: str, tags: list[str] | None = None) -> str:
        return self.normalize(text, tags).text


# ---------------------------------------------------------------- learning
def align_pairs(orig: str, ref: str) -> list[tuple[str, str]]:
    """1:1 word substitutions between a raw sentence and its reference."""
    a, b = words_only(orig), words_only(ref)
    pairs = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            pairs.extend(zip(a[i1:i2], b[j1:j2]))
    return pairs


class DictionaryLearner:
    """Alignment statistics from (original, 3 reference) pairs, computed once.

    ``select`` then derives a dictionary for any threshold setting cheaply, so the
    thresholds can be tuned on validation data.
    """

    def __init__(self, examples: list[dict], word_lang: dict[str, str] | None = None):
        self.word_lang = word_lang or {}
        self.occ = Counter()
        self.rate: dict[str, Counter] = defaultdict(Counter)       # avg fraction of annotators rewriting
        self.agreed: dict[str, Counter] = defaultdict(Counter)     # >= 2 annotators agree on the rewrite
        for ex in examples:
            self.occ.update(words_only(ex["original"]))
            n_ref = len(ex["refs"])
            votes = Counter()
            for ref in ex["refs"]:
                votes.update({(s, t) for s, t in align_pairs(ex["original"], ref)
                              if s != t and s.isalpha() and t.isalpha()})
            for (s, t), v in votes.items():
                self.rate[s][t] += v / n_ref
                if v >= 2:
                    self.agreed[s][t] += 1

    def select(self, min_count: int = 2, min_ratio: float = 0.5, consensus: bool = True):
        src = self.agreed if consensus else self.rate
        learned = {}
        for s, c in src.items():
            t, n = c.most_common(1)[0]
            if n >= min_count and n / self.occ[s] >= min_ratio:
                learned[s] = (t, self.word_lang.get(s))
        return learned


def learn_dictionary(examples: list[dict], word_lang: dict[str, str] | None = None,
                     min_count: int = 2, min_ratio: float = 0.5, consensus: bool = True):
    """Learn ``source -> preferred`` word mappings from aligned pairs.

    A mapping is kept when the source word is rewritten to the same target in at
    least ``min_ratio`` of its occurrences and at least ``min_count`` times.
    With ``consensus=True`` a rewrite only counts when >= 2 of the 3 annotators
    made it, which keeps individual annotators' spelling habits out of the
    dictionary; otherwise rewrites are weighted by the fraction of annotators.
    """
    return DictionaryLearner(examples, word_lang).select(min_count, min_ratio, consensus)
