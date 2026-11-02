"""English-word test shared by the silver labeler and the normalizer.

``wordfreq`` is imported lazily: it is only needed for Tanglish (to tell English
words from Romanized Tamil), so the Hinglish path works without touching it.
"""
from __future__ import annotations

from functools import lru_cache

EN_ZIPF_MIN = 3.0       # >= ~1 occurrence per million English words


@lru_cache(maxsize=None)
def is_english(word: str) -> bool:
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en") >= EN_ZIPF_MIN
