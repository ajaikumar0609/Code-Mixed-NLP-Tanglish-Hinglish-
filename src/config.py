"""Central paths, constants and label sets shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LEXICONS = DATA / "lexicons"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 42

# Language variants the system supports.
VARIANTS = {"tanglish": "TA", "hinglish": "HI"}

# Token-level label set. NUM / PUNCT / OTHER for non-words are assigned by
# deterministic rules in preprocessing; the classifiers decide only among
# the word-level labels of the selected variant.
WORD_LABELS = {
    "tanglish": ["TA", "EN", "OTHER"],
    "hinglish": ["HI", "EN", "OTHER"],
}
ALL_LABELS = ["TA", "HI", "EN", "NUM", "PUNCT", "OTHER"]

# Language-model special tokens.
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"
NUM_TOKEN = "<num>"
SPECIALS = [PAD, UNK, BOS, EOS]
# Never proposed as a "next word" suggestion.
NON_SUGGESTIBLE = {PAD, UNK, BOS, EOS, NUM_TOKEN}

for _d in (PROCESSED, LEXICONS, MODELS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
