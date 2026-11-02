"""Learn the Hinglish spelling dictionary, then evaluate the normalizers.

    python -m src.train_normalizer

* Hinglish: dictionary learned from the COMI-LINGUA TN *train* pairs, thresholds tuned on *val*,
  final numbers on the official *test* pairs (3 annotator references).
* Tanglish: no parallel corpus exists, so a small hand-written reference set
  (``data/gold/tanglish_norm_gold.tsv``) acts as a sanity check, not a benchmark.
"""
from __future__ import annotations

import itertools
import json
import random
from collections import Counter, defaultdict

import pandas as pd

from .config import DATA, LEXICONS, MODELS, RESULTS, SEED
from .evaluation import normalization_scores, wer_cer
from .language_identifier import CharNgramLID, identify_tokens
from .evaluation import wer_cer
from .normalizer import DictionaryLearner, Normalizer, align_pairs
from .preprocessing import tokenize

GOLD = DATA / "gold"


def load_pairs(variant: str):
    df = pd.read_csv(DATA / "normalization.csv", keep_default_na=False)
    df = df[df.variant == variant]
    out = {}
    for split, g in df.groupby("split"):
        out[split] = [{"original": r.original_text,
                       "refs": [r.normalized_text, r.normalized_text_annotator2, r.normalized_text_annotator3]}
                      for r in g.itertuples()]
    return out


def hinglish_word_lang() -> dict[str, str]:
    df = pd.read_csv(DATA / "language_id.csv", keep_default_na=False)
    df = df[(df.variant == "hinglish") & (df.split == "train") & (df.kind == "WORD")]
    counts = defaultdict(Counter)
    for tok, lab in zip(df.token, df.language):
        counts[tok.lower()][lab] += 1
    out = {}
    for w, c in counts.items():
        lab, n = c.most_common(1)[0]
        if n >= 3 and n / sum(c.values()) >= 0.8 and lab in ("HI", "EN"):
            out[w] = lab
    return out


def predict_tags(lid, normalizer_input: str):
    toks = tokenize(normalizer_input)
    return [l for l, _ in identify_tokens(lid, toks)]


def run_system(norm: Normalizer | None, examples, lid=None):
    hyps = []
    for ex in examples:
        if norm is None:
            hyps.append(ex["original"])
        else:
            tags = predict_tags(lid, ex["original"]) if lid else None
            hyps.append(norm.normalize(ex["original"], tags).text)
    return hyps


def error_analysis(examples, hyps, top=25):
    """Most frequent word-level corrections the system still misses or gets wrong."""
    missed, wrong = Counter(), Counter()
    for ex, hyp in zip(examples, hyps):
        ref = ex["refs"][0]
        ref_map = dict(align_pairs(ex["original"], ref))
        hyp_map = dict(align_pairs(ex["original"], hyp))
        for s, t in ref_map.items():
            if s not in hyp_map:
                missed[(s, t)] += 1
            elif hyp_map[s] != t:
                wrong[(s, t, hyp_map[s])] += 1
        for s, t in hyp_map.items():
            if s not in ref_map:
                wrong[(s, "(unchanged in reference)", t)] += 1
    return {"missed_corrections": [[s, t, n] for (s, t), n in missed.most_common(top)],
            "wrong_corrections": [[s, t, h, n] for (s, t, h), n in wrong.most_common(top)]}


def inter_annotator(examples) -> dict:
    """Average word/char error between pairs of human annotators: the practical ceiling."""
    out = {}
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        we = wn = ce = cn = 0
        for e in examples:
            a, b, c, d = wer_cer(e["refs"][i], e["refs"][j])
            we += a; wn += b; ce += c; cn += d
        out[f"annotator{i + 1}_vs_annotator{j + 1}"] = {"wer": we / wn, "cer": ce / cn}
    out["mean_wer"] = sum(v["wer"] for v in out.values()) / 3
    out["mean_cer"] = sum(v["cer"] for v in list(out.values())[:3]) / 3
    return out


def run_hinglish(log=print) -> dict:
    pairs = load_pairs("hinglish")
    train, val, test = pairs["train"], pairs["val"], pairs["test"]
    word_lang = hinglish_word_lang()
    known = Counter(json.loads((LEXICONS / "known_words_hinglish.json").read_text()))

    learner = DictionaryLearner(train, word_lang)
    val_refs = [e["refs"] for e in val]
    grid, best = {}, None
    for consensus, min_count, min_ratio in itertools.product([False, True], [2, 3], [0.2, 0.3, 0.4, 0.5, 0.6]):
        learned = learner.select(min_count, min_ratio, consensus)
        hyps = run_system(Normalizer("hinglish", known, learned), val)
        sc = normalization_scores(hyps, val_refs)["best_of_refs"]
        grid[f"consensus={consensus},min_count={min_count},min_ratio={min_ratio}"] = {
            "val_wer": sc["wer"], "val_exact": sc["exact_match"], "entries": len(learned)}
        if best is None or sc["wer"] < best[0]:
            best = (sc["wer"], learned, min_count, min_ratio, consensus)
    log(f"[hinglish] best val WER {best[0]:.4f} (consensus={best[4]}, min_count={best[2]}, "
        f"min_ratio={best[3]}, {len(best[1])} dictionary entries)")
    learned = best[1]
    (LEXICONS / "learned_dict_hinglish.json").write_text(
        json.dumps({k: list(v) for k, v in sorted(learned.items())}, ensure_ascii=False, indent=0))

    lid = CharNgramLID.load(MODELS / "lid_char_hinglish.joblib") if (MODELS / "lid_char_hinglish.joblib").exists() else None
    refs = [e["refs"] for e in test]
    systems = {
        "no_normalization (baseline)": run_system(None, test),
        "rules_only (punctuation + elongation)": run_system(Normalizer("hinglish", known, {}), test),
        "rules + learned dictionary": run_system(Normalizer("hinglish", known, learned), test),
    }
    if lid:
        systems["rules + dictionary + predicted language tags"] = run_system(
            Normalizer("hinglish", known, learned), test, lid)
    results = {k: normalization_scores(v, refs) for k, v in systems.items()}
    final_key = list(systems)[-1]
    rng = random.Random(SEED)
    idx = rng.sample(range(len(test)), 40)
    return {
        "variant": "hinglish", "reference": "COMI-LINGUA TN test, 3 annotators",
        "tuning": grid, "chosen": {"consensus": best[4], "min_count": best[2], "min_ratio": best[3], "entries": len(learned)},
        "inter_annotator": inter_annotator(test),
        "systems": results,
        "review_sample": [{"original": test[i]["original"], "system": systems[final_key][i],
                           "reference": test[i]["refs"][0]} for i in idx],
        "error_analysis": error_analysis(test, systems[final_key]),
        "metric_note": ("WER/CER are computed on lower-cased words with punctuation removed, because the "
                        "references also add capitalisation and sentence punctuation that this normalizer "
                        "deliberately leaves alone."),
    }


def load_tanglish_gold_pairs():
    path = GOLD / "tanglish_norm_gold.tsv"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            o, n = line.split("\t")[:2]
            out.append({"original": o, "refs": [n]})
    return out


def run_tanglish(log=print) -> dict:
    gold = load_tanglish_gold_pairs()
    if not gold:
        return {"variant": "tanglish", "note": "no gold file"}
    norm = Normalizer.for_variant("tanglish")
    lid = CharNgramLID.load(MODELS / "lid_char_tanglish.joblib") if (MODELS / "lid_char_tanglish.joblib").exists() else None
    refs = [e["refs"] for e in gold]
    systems = {"no_normalization (baseline)": run_system(None, gold),
               "rules + curated dictionary": run_system(norm, gold)}
    if lid:
        systems["rules + dictionary + predicted language tags"] = run_system(norm, gold, lid)
    final = list(systems)[-1]
    return {
        "variant": "tanglish",
        "reference": f"{len(gold)} hand-written reference pairs (sanity check, not an independent benchmark)",
        "systems": {k: normalization_scores(v, refs) for k, v in systems.items()},
        "review_sample": [{"original": g["original"], "system": h, "reference": g["refs"][0]}
                          for g, h in zip(gold, systems[final])][:40],
        "error_analysis": error_analysis(gold, systems[final]),
    }


def main():
    out = {"hinglish": run_hinglish()}
    out["tanglish"] = run_tanglish()
    (RESULTS / "normalization.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for v, r in out.items():
        for k, s in r.get("systems", {}).items():
            b = s["best_of_refs"]
            print(f"[{v}] {k:48s} WER {b['wer']:.4f}  CER {b['cer']:.4f}  exact {b['exact_match']:.4f}")


if __name__ == "__main__":
    main()
