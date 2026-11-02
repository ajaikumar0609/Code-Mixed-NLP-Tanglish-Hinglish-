"""Train and evaluate the language identifiers (Model A and Model B) for both variants.

    python -m src.train_lid [--variant tanglish|hinglish] [--epochs N]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import pandas as pd

from .config import DATA, MODELS, RESULTS, WORD_LABELS
from .evaluation import lid_report
from .language_identifier import BiLSTMLID, CharNgramLID
from .preprocessing import Token, rule_label, tokenize

GOLD = DATA / "gold"


def _is_model_word(tok: str, kind: str) -> bool:
    return kind == "WORD" and rule_label(Token(tok, "WORD")) is None


def load_split(variant: str, split: str):
    """-> (word_sentences, word_labels, full_tokens, full_labels), grouped by sentence."""
    df = pd.read_csv(DATA / "language_id.csv", keep_default_na=False)
    df = df[(df.variant == variant) & (df.split == split)]
    ws, wl, ft, fl = [], [], [], []
    for _, g in df.groupby("sentence_id", sort=False):
        toks, kinds, labs = g.token.tolist(), g.kind.tolist(), g.language.tolist()
        ft.append(list(zip(toks, kinds))); fl.append(labs)
        keep = [i for i, (t, k) in enumerate(zip(toks, kinds)) if _is_model_word(t, k)]
        ws.append([toks[i] for i in keep]); wl.append([labs[i] for i in keep])
    return ws, wl, ft, fl


def load_tanglish_gold():
    """Hand-reviewed Tanglish sentences in the same 4-tuple format as ``load_split``."""
    path = GOLD / "tanglish_lid_gold.tsv"
    if not path.exists():
        return None
    ws, wl, ft, fl = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, body = line.split("\t", 1)
        pairs = [item.rsplit("/", 1) for item in body.split(" ")]
        toks, labs = [t for t, _ in pairs], [l for _, l in pairs]
        kinds = [(tokenize(t)[0].kind if len(tokenize(t)) == 1 else "WORD") for t in toks]
        ft.append(list(zip(toks, kinds))); fl.append(labs)
        keep = [i for i, (t, k) in enumerate(zip(toks, kinds)) if _is_model_word(t, k)]
        ws.append([toks[i] for i in keep]); wl.append([labs[i] for i in keep])
    return ws, wl, ft, fl


def predict_full(model, full_tokens):
    """Predict labels for every token of every sentence (rules + model)."""
    preds = []
    for sent in full_tokens:
        toks = [Token(t, k) for t, k in sent]
        word_idx = [i for i, t in enumerate(toks) if rule_label(t) is None]
        out = [rule_label(t) or "OTHER" for t in toks]
        if word_idx:
            for i, (lab, _) in zip(word_idx, model.predict_sentence([toks[i].text for i in word_idx])):
                out[i] = lab
        preds.append(out)
    return preds


def evaluate_model(model, variant, sets: dict):
    labels_w = WORD_LABELS[variant]
    labels_all = [labels_w[0], "EN", "OTHER", "NUM", "PUNCT"]
    res = {}
    for name, (ws, wl, ft, fl) in sets.items():
        # word-only: what the classifier itself is responsible for
        pw = []
        if hasattr(model, "predict_proba_sentences"):
            pw = [[model.labels[i] for i in P.argmax(1)] for P in model.predict_proba_sentences(ws)]
        else:
            pw = [[l for l, _ in model.predict_sentence(s)] if s else [] for s in ws]
        yt = [l for ls in wl for l in ls]; yp = [l for ls in pw for l in ls]
        res[name] = {"word_level": lid_report(yt, yp, labels_w)}
        # end-to-end: every token, rules + model
        if ft:
            pf = predict_full(model, ft)
            yt2 = [l for ls in fl for l in ls]; yp2 = [l for ls in pf for l in ls]
            res[name]["all_tokens"] = lid_report(yt2, yp2, labels_all)
    return res


def binary_native(y):
    return ["EN" if l == "EN" else "NATIVE" for l in y]


def cross_variant_check(variant: str, modelA=None, te=None, gold=None) -> dict:
    """EN-vs-native binary scores of this variant's and the *other* variant's Model A on one test set."""
    other = "hinglish" if variant == "tanglish" else "tanglish"
    other_path = MODELS / f"lid_char_{other}.joblib"
    if not other_path.exists():
        return {}
    modelA = modelA or CharNgramLID.load(MODELS / f"lid_char_{variant}.joblib")
    if variant == "hinglish":
        tgt, name = (te or load_split("hinglish", "test")), "hinglish test (gold)"
    else:
        tgt, name = (gold or load_tanglish_gold()), "tanglish hand-reviewed gold"
    if not tgt:
        return {}
    yt = binary_native([l for ls in tgt[1] for l in ls])
    out = {}
    for nm, mdl in ((f"{variant}_model", modelA), (f"{other}_model", CharNgramLID.load(other_path))):
        yp = binary_native([l for s in tgt[0] for l, _ in (mdl.predict_sentence(s) if s else [])])
        rep = lid_report(yt, yp, ["EN", "NATIVE"])
        out[nm] = {"accuracy": rep["accuracy"], "macro_f1": rep["macro_f1"]}
    out["evaluated_on"] = name
    return out


def run(variant: str, epochs: int, log=print) -> dict:
    t0 = time.time()
    labels = WORD_LABELS[variant]
    tr = load_split(variant, "train"); va = load_split(variant, "val"); te = load_split(variant, "test")
    log(f"[{variant}] train {len(tr[0])} / val {len(va[0])} / test {len(te[0])} sentences "
        f"({sum(map(len, tr[0]))} train word tokens)")
    out = {"variant": variant, "label_source": "gold (COMI-LINGUA, 3-annotator majority)"
           if variant == "hinglish" else "silver (lexicon rules; see src/silver_labels.py)",
           "n_train_sentences": len(tr[0]), "models": {}}

    # ---- Model A: small grid on validation accuracy / macro-F1
    grid, best = {}, None
    for rng, C in itertools.product([(1, 3), (1, 4), (2, 5)], [1.0, 10.0, 50.0]):
        m = CharNgramLID(labels, ngram_range=rng, C=C).fit(tr[0], tr[1])
        pv = [[l for l, _ in m.predict_sentence(s)] if s else [] for s in va[0]]
        f1 = lid_report([l for x in va[1] for l in x], [l for x in pv for l in x], labels)["macro_f1"]
        grid[f"ngram={rng},C={C}"] = f1
        if best is None or f1 > best[0]:
            best = (f1, m, rng, C)
    modelA = best[1]
    log(f"[{variant}] Model A best val macro-F1 {best[0]:.4f} with ngram={best[2]} C={best[3]}")
    modelA.save(MODELS / f"lid_char_{variant}.joblib")

    # ---- Model B
    modelB = BiLSTMLID(labels).fit(tr[0], tr[1], val=(va[0], va[1]), epochs=epochs, log=log)
    modelB.save(MODELS / f"lid_bilstm_{variant}.pt")

    sets = {"val": va, "test": te}
    gold = load_tanglish_gold() if variant == "tanglish" else None
    if gold:
        sets["hand_reviewed_gold"] = gold
    for key, model in (("char_ngram", modelA), ("bilstm", modelB)):
        out["models"][key] = {"name": model.name, "results": evaluate_model(model, variant, sets)}
    out["models"]["char_ngram"]["tuning_val_macro_f1"] = grid
    out["models"]["char_ngram"]["best"] = {"ngram_range": list(best[2]), "C": best[3]}

    out["cross_variant"] = cross_variant_check(variant, modelA, te, gold)
    out["seconds"] = time.time() - t0
    (RESULTS / f"lid_{variant}.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(WORD_LABELS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cross-only", action="store_true",
                    help="only refresh the cross-variant transfer check in existing result files")
    a = ap.parse_args()
    if a.cross_only:
        for v in WORD_LABELS:
            path = RESULTS / f"lid_{v}.json"
            res = json.loads(path.read_text())
            res["cross_variant"] = cross_variant_check(v)
            path.write_text(json.dumps(res, indent=2))
            print(v, res["cross_variant"])
        return
    for v in (list(WORD_LABELS) if a.variant == "all" else [a.variant]):
        r = run(v, a.epochs)
        for k, m in r["models"].items():
            t = m["results"]["test"]["word_level"]
            print(f"[{v}] {m['name']}: test word acc {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
