"""Unicode handling, tokenization and detokenization for code-mixed text.

The tokenizer is lossless with respect to spacing: every token records whether
whitespace preceded it, so ``detokenize(tokenize(s))`` reproduces ``s`` (up to
whitespace collapsing). This lets the normalizer edit tokens and still rebuild
a readable sentence such as ``college-ku``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .config import NUM_TOKEN

# Token kinds
WORD, NUM, PUNCT, URL, MENTION, HASHTAG, EMOJI = (
    "WORD", "NUM", "PUNCT", "URL", "MENTION", "HASHTAG", "EMOJI",
)

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"), None)

_EMOJI_RANGES = (
    "\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿"
    "\U0001F900-\U0001F9FF️←-⇿⤀-⥿"
)

_TOKEN_RE = re.compile(
    rf"""
    (?P<URL>(?:https?://|www\.)\S+)
  | (?P<MENTION>@\w+)
  | (?P<HASHTAG>\#\w+)
  | (?P<NUM>\d+(?:[.,:/]\d+)*(?!\w))
  | (?P<WORD>\w+(?:['’]\w+)*)
  | (?P<EMOJI>[{_EMOJI_RANGES}]+)
  | (?P<PUNCT>[^\w\s])
    """,
    re.VERBOSE | re.UNICODE,
)

_NO_SPACE_BEFORE = set(",.!?;:%)]}…")
_NO_SPACE_AFTER = set("([{")


@dataclass
class Token:
    text: str
    kind: str
    space_before: bool = True

    def __repr__(self) -> str:  # compact for debugging
        return f"{self.text}/{self.kind}"


def clean_unicode(text: str) -> str:
    """NFKC-normalise, drop zero-width characters, unify quotes and collapse whitespace."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).translate(_ZERO_WIDTH)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[Token]:
    """Split text into typed tokens, remembering spacing."""
    text = clean_unicode(text)
    tokens: list[Token] = []
    prev_end = 0
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        tok = m.group()
        if kind == "WORD" and re.fullmatch(r"[\d_]+", tok):  # digits/underscore only
            kind = NUM
        # Pure-underscore or digit runs are not words.
        tokens.append(Token(tok, kind, space_before=m.start() > prev_end))
        prev_end = m.end()
    if tokens:
        tokens[0].space_before = False
    return tokens


def detokenize(tokens: list[Token]) -> str:
    out = []
    for i, t in enumerate(tokens):
        if i == 0:
            out.append(t.text)
            continue
        sp = t.space_before
        prev = tokens[i - 1]
        if t.kind == PUNCT and t.text[0] in _NO_SPACE_BEFORE:
            sp = False
        elif prev.kind == PUNCT and prev.text[-1] in _NO_SPACE_AFTER:
            sp = False
        elif prev.kind == PUNCT and prev.text[-1] in ",.!?;:" and t.kind != PUNCT:
            sp = True
        out.append((" " if sp else "") + t.text)
    return "".join(out)


def words_only(text: str) -> list[str]:
    """Lower-cased WORD tokens (used for dictionaries and quick checks)."""
    return [t.text.lower() for t in tokenize(text) if t.kind == WORD]


def lm_tokens(text: str) -> list[str]:
    """Token sequence used by the language models.

    Lower-cased words, numbers mapped to ``<num>``, hashtags reduced to their
    word, and punctuation / emoji / URLs / mentions dropped. Training and
    inference use this same function so the demo matches the experiments.
    """
    out: list[str] = []
    for t in tokenize(text):
        if t.kind == WORD:
            out.append(t.text.lower())
        elif t.kind == NUM:
            out.append(NUM_TOKEN)
        elif t.kind == HASHTAG:
            out.append(t.text[1:].lower())
    return out


def dedupe_key(text: str) -> str:
    """Key used to detect (near-)duplicate messages: lower-cased words only."""
    return " ".join(lm_tokens(text))


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_TAMIL = re.compile(r"[஀-௿]")


def has_native_script(text: str) -> bool:
    return bool(_DEVANAGARI.search(text) or _TAMIL.search(text))


def rule_label(tok: Token) -> str | None:
    """Deterministic label for non-word tokens; ``None`` for words."""
    if tok.kind == NUM:
        return "NUM"
    if tok.kind == PUNCT:
        return "PUNCT"
    if tok.kind in (URL, MENTION, HASHTAG, EMOJI):
        return "OTHER"
    if tok.kind == WORD and not tok.text.isascii():
        if _DEVANAGARI.search(tok.text):
            return "HI"
        if _TAMIL.search(tok.text):
            return "TA"
        return "OTHER"          # accented letters / mojibake noise
    return None


_RUN = re.compile(r"(.)\1{2,}")


def elongation_candidates(word: str) -> list[str]:
    """Candidate spellings of an elongated word (``hiii`` -> ``hi``, ``hii``).

    Only runs of 3+ identical characters are touched: each such run may shrink
    to one or two characters. Words without long runs return ``[]`` so that
    legitimate doubles (``irukku``, ``good``) are never altered.
    """
    runs = list(_RUN.finditer(word))
    if not runs or len(runs) > 3:
        return []
    cands = {""}
    last = 0
    for m in runs:
        seg = word[last:m.start()]
        ch = m.group(1)
        cands = {c + seg + ch * k for c in cands for k in (1, 2)}
        last = m.end()
    tail = word[last:]
    return sorted((c + tail for c in cands), key=lambda w: (len(w), w))
