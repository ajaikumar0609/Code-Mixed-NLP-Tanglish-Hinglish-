"""Build every dataset file the project trains and evaluates on.

Outputs
-------
data/language_id.csv                  token-level labels (hinglish: gold, tanglish: silver)
data/normalization.csv                aligned (original, normalized) sentence pairs
data/gold/tanglish_lid_gold.tsv       hand-reviewed Tanglish token labels (never trained on)
data/gold/tanglish_norm_gold.tsv      hand-written Tanglish normalization references
data/processed/<variant>_{train,val,test}.txt   normalised LM corpora (one sentence per line)
data/lexicons/known_words_<variant>.json, learned_dict_hinglish.json
data/processed/split_report.json      sizes, sources per split, duplicate counts

Splitting policy
----------------
* Tanglish: whole *sources* (a YouTube video id / subreddit / dataset) go to one split, so near-duplicate
  comments from one thread cannot straddle train and test.
* Hinglish: the official COMI-LINGUA train/test split is kept; validation is carved from train.
* Duplicates (after lower-casing and stripping punctuation) are removed globally, and any
  training sentence that also occurs in val/test is dropped.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict

import pandas as pd

from .comi_lingua import load_lid, load_tn
from .config import DATA, LEXICONS, PROCESSED, RAW, SEED
from .normalizer import Normalizer, learn_dictionary
from .preprocessing import dedupe_key, has_native_script, lm_tokens, rule_label, tokenize, words_only
from .silver_labels import silver_label_tokens

GOLD = DATA / "gold"
GOLD.mkdir(exist_ok=True)

TANGLISH_LID_TRAIN = 30000
TANGLISH_LID_VAL = 2000
TANGLISH_LID_TEST = 3000


def _write_lines(path, sentences):
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(" ".join(s) + "\n")


# ----------------------------------------------------------------------- Hinglish
def prepare_hinglish(rng: random.Random):
    lid_rows, report = [], {}
    train = load_lid("train"); stats_tr = dict(load_lid.last_stats)
    test = load_lid("test"); stats_te = dict(load_lid.last_stats)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_val = int(0.1 * len(train))
    val_ids = set(idx[:n_val])
    sid = 0
    for split, data in (("train", [(i, d) for i, d in enumerate(train) if i not in val_ids]),
                        ("val", [(i, d) for i, d in enumerate(train) if i in val_ids]),
                        ("test", list(enumerate(test)))):
        for _, d in data:
            sid += 1
            for t, l in zip(d["tokens"], d["labels"]):
                k = tokenize(t)[0].kind if tokenize(t) else "WORD"
                lid_rows.append(("hinglish", split, f"hi{sid:06d}", t, k, l, "gold"))
        report[f"lid_{split}_sentences"] = len(data)
    report["lid_skipped_train"] = stats_tr
    report["lid_skipped_test"] = stats_te

    # --- normalization pairs + LM corpus
    tn_train, tn_test = load_tn("train"), load_tn("test")
    rng.shuffle(tn_train)
    n_val = int(0.1 * len(tn_train))
    tn = {"val": tn_train[:n_val], "train": tn_train[n_val:], "test": tn_test}

    # word language from gold LID (majority) so learned dictionary entries are language-aware
    lang_counts = defaultdict(Counter)
    for _, split, _, tok, kind, lab, _ in lid_rows:
        if split == "train" and kind == "WORD":
            lang_counts[tok.lower()][lab] += 1
    word_lang = {w: c.most_common(1)[0][0] for w, c in lang_counts.items()
                 if c.most_common(1)[0][1] / sum(c.values()) >= 0.8 and sum(c.values()) >= 3}

    norm_rows, lm = [], {}
    seen_keys = set()
    for split in ("test", "val", "train"):         # test first so train never contains test text
        sents = []
        for i, ex in enumerate(tn[split]):
            toks = lm_tokens(ex["refs"][0])
            key = " ".join(toks)
            norm_rows.append(("hinglish", split, f"hi{split[:2]}{i:05d}", ex["original"], *ex["refs"]))
            if len(toks) < 3 or key in seen_keys:
                continue
            seen_keys.add(key)
            sents.append(toks)
        lm[split] = sents
    report["lm_hinglish"] = {k: len(v) for k, v in lm.items()}
    return lid_rows, norm_rows, lm, tn, word_lang, report


# ----------------------------------------------------------------------- Tanglish
def split_by_source(df: pd.DataFrame, rng: random.Random, fracs=(0.8, 0.1, 0.1)):
    """Assign whole sources to splits, greedily matching the target fractions."""
    sizes = df.groupby("source").size()
    sources = list(sizes.index)
    rng.shuffle(sources)
    n = len(df)
    targets = {"test": fracs[2] * n, "val": fracs[1] * n}
    assign, filled = {}, Counter()
    for s in sources:
        for name in ("test", "val"):
            if filled[name] < targets[name]:
                assign[s] = name
                filled[name] += sizes[s]
                break
        else:
            assign[s] = "train"
    return df["source"].map(assign)


def prepare_tanglish(rng: random.Random):
    df = pd.read_json(RAW / "tanglish" / "tanglish_corpus.jsonl", lines=True)
    report = {"raw_sentences": len(df)}
    df = df[~df.text.map(has_native_script)]
    df["key"] = df.text.map(dedupe_key)
    before = len(df)
    df = df[df.key.str.split().str.len() >= 3].drop_duplicates("key").reset_index(drop=True)
    report["after_dedupe_and_min_len"] = len(df)
    report["removed"] = before - len(df)
    df["split"] = split_by_source(df, rng)

    # known words for elongation reduction: raw frequent words (count >= 5)
    counts = Counter(w for t in df[df.split == "train"].text for w in words_only(t))
    known = {w: c for w, c in counts.items() if c >= 5}
    (LEXICONS / "known_words_tanglish.json").write_text(json.dumps(known, ensure_ascii=False))
    norm = Normalizer.for_variant("tanglish")

    def normalise(text):
        toks = tokenize(text)
        return norm.normalize(text, silver_label_tokens(toks)).text

    df["normalized"] = df.text.map(normalise)
    df["lm"] = df.normalized.map(lm_tokens)
    df["lm_key"] = df.lm.map(" ".join)
    # normalisation can create new duplicates; drop them (test/val take priority)
    seen, keep = set(), []
    for split in ("test", "val", "train"):
        for i in df.index[df.split == split]:
            k = df.at[i, "lm_key"]
            if len(df.at[i, "lm"]) >= 3 and k not in seen:
                seen.add(k); keep.append(i)
    df = df.loc[sorted(keep)]
    report["lm_tanglish"] = df.split.value_counts().to_dict()
    report["sources_per_split"] = df.groupby("split").source.nunique().to_dict()
    return df, report


def tanglish_lid_rows(df: pd.DataFrame, rng: random.Random):
    rows = []
    quota = {"train": TANGLISH_LID_TRAIN, "val": TANGLISH_LID_VAL, "test": TANGLISH_LID_TEST}
    sid = 0
    for split, n in quota.items():
        sub = df[(df.split == split) & df.text.str.split().str.len().between(4, 40)]
        sub = sub.sample(min(n, len(sub)), random_state=SEED)
        for text in sub.text:
            sid += 1
            toks = tokenize(text)
            for t, l in zip(toks, silver_label_tokens(toks)):
                rows.append(("tanglish", split, f"ta{sid:06d}", t.text, t.kind, l, "silver"))
    return rows


def write_gold_candidates(df: pd.DataFrame, n: int = 120, path=GOLD / "tanglish_lid_candidates.tsv"):
    """Draw held-out test-source sentences with silver labels for manual review."""
    sub = df[(df.split == "test") & df.text.str.split().str.len().between(6, 18)]
    sub = sub.sample(n, random_state=SEED + 1)
    with open(path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sub.text):
            toks = tokenize(text)
            labs = silver_label_tokens(toks)
            f.write(f"{i}\t" + " ".join(f"{t.text}/{l}" for t, l in zip(toks, labs)) + "\n")
    return path


# ----------------------------------------------------------------------- main
def _merge_csv(path, header, rows, variant):
    """Replace this variant's rows in a shared CSV, keeping the other variant's rows."""
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            old = [row for row in r if row and row[0] != variant]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(old)
        w.writerows(rows)


LID_HEADER = ["variant", "split", "sentence_id", "token", "kind", "language", "label_source"]
NORM_HEADER = ["variant", "split", "sentence_id", "original_text",
               "normalized_text", "normalized_text_annotator2", "normalized_text_annotator3"]


def main(variants=("hinglish", "tanglish")):
    rng = random.Random(SEED)
    report_path = PROCESSED / "split_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if "hinglish" in variants:
        lid_rows, norm_rows, lm_hi, _tn, _wl, rep_hi = prepare_hinglish(rng)
        _merge_csv(DATA / "language_id.csv", LID_HEADER, lid_rows, "hinglish")
        _merge_csv(DATA / "normalization.csv", NORM_HEADER, norm_rows, "hinglish")
        for split in ("train", "val", "test"):
            _write_lines(PROCESSED / f"hinglish_{split}.txt", lm_hi[split])
        (LEXICONS / "known_words_hinglish.json").write_text(json.dumps(
            {w: c for w, c in Counter(w for s in lm_hi["train"] for w in s).items() if c >= 2},
            ensure_ascii=False))
        report.update(rep_hi)

    if "tanglish" in variants:
        df, rep_ta = prepare_tanglish(rng)
        _merge_csv(DATA / "language_id.csv", LID_HEADER, tanglish_lid_rows(df, rng), "tanglish")
        for split in ("train", "val", "test"):
            _write_lines(PROCESSED / f"tanglish_{split}.txt", df[df.split == split].lm)
        write_gold_candidates(df)
        report.update(rep_ta)

    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("hinglish", "tanglish"))
