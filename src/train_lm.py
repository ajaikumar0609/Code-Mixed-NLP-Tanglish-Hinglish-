"""Train and compare the trigram (Kneser-Ney) and LSTM language models.

    python -m src.train_lm [--variant tanglish|hinglish|all] [--epochs N]

Both models share one ``Vocab`` built from the training split only, see the same
<s> ... </s> framing, and are scored on the same held-out test sentences with the
same metrics: perplexity over every token (including ``<unk>`` and ``</s>``) and
top-1 / top-5 next-word accuracy over real-word targets.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .config import MODELS, NUM_TOKEN, PROCESSED, RESULTS, WORD_LABELS
from .evaluation import topk_accuracy_ngram
from .lstm_model import LSTMPredictor, get_device, train_lstm
from .ngram_model import KNTrigramLM
from .vocab import Vocab, read_corpus

# Per-variant defaults: the Hinglish corpus is ~7x smaller, so the LSTM is smaller and
# more heavily regularised to limit over-fitting.
DEFAULTS = {
    "tanglish": dict(max_vocab=20000, min_count=3, emb=256, hidden=512, dropout=0.3, lr=2e-3, epochs=8),
    "hinglish": dict(max_vocab=30000, min_count=2, emb=200, hidden=400, dropout=0.4, lr=2e-3, epochs=15),
}

DEMO_PROMPTS = {
    "tanglish": ["naan inniku", "indha project romba", "enna panra", "super", "romba nalla"],
    "hinglish": ["kal meeting", "aap kaise", "main aaj", "please notes", "yeh bahut"],
}


def run(variant: str, epochs: int | None = None, time_budget_min: float | None = None,
        max_train: int | None = None, log=print, stage: str = "both") -> dict:
    cfg = dict(DEFAULTS[variant])
    if epochs:
        cfg["epochs"] = epochs
    tr_t, va_t, te_t = (read_corpus(PROCESSED / f"{variant}_{s}.txt") for s in ("train", "val", "test"))
    if max_train:
        tr_t = tr_t[:max_train]
    vocab_path = MODELS / f"vocab_{variant}.json"
    if stage == "lstm" and vocab_path.exists():
        vocab = Vocab.load(vocab_path)               # identical vocabulary to the N-gram stage
    else:
        vocab = Vocab.build(tr_t, min_count=cfg["min_count"], max_size=cfg["max_vocab"])
        vocab.save(vocab_path)
    tr, va, te = ([vocab.encode(s) for s in x] for x in (tr_t, va_t, te_t))
    n_test_tokens = sum(map(len, te))
    oov = sum(w == vocab.unk for s in te for w in s) / n_test_tokens
    log(f"[{variant}] sentences train/val/test = {len(tr)}/{len(va)}/{len(te)}; vocab {len(vocab)}; "
        f"test OOV rate {oov:.3%}")
    out = {"variant": variant, "vocab_size": len(vocab), "train_sentences": len(tr),
           "val_sentences": len(va), "test_sentences": len(te),
           "train_tokens": sum(map(len, tr)), "test_tokens": n_test_tokens,
           "test_oov_rate": oov, "config": cfg}
    skip = {vocab.unk, vocab.eos}      # accuracy is scored on real-word targets only
    if NUM_TOKEN in vocab.stoi:
        skip.add(vocab.stoi[NUM_TOKEN])

    part = RESULTS / f"lm_{variant}_ngram_part.json"
    if stage in ("both", "ngram"):
        t0 = time.time()
        ng = KNTrigramLM(vocab).fit(tr)
        disc = ng.tune_discount(va)
        ng.save(MODELS / f"ngram_{variant}.pkl")
        acc, n_pos = topk_accuracy_ngram(ng, te, (1, 5), skip)
        out["ngram"] = {"discount": ng.D, "discount_search_val_ppl": {str(k): v for k, v in disc.items()},
                        "val_perplexity": ng.perplexity(va), "test_perplexity": ng.perplexity(te),
                        "train_perplexity": ng.perplexity(tr[:20000]),
                        "top1": acc[1], "top5": acc[5], "n_eval_positions": n_pos,
                        "n_trigrams": int(len(ng.k3)), "seconds": time.time() - t0}
        log(f"[{variant}] trigram-KN: test ppl {out['ngram']['test_perplexity']:.1f}  "
            f"top1 {acc[1]:.3f}  top5 {acc[5]:.3f}  ({out['ngram']['seconds']:.0f}s)")

        part.write_text(json.dumps(out["ngram"], indent=2))
        if stage == "ngram":
            return out
    else:
        out["ngram"] = json.loads(part.read_text())
        ng = KNTrigramLM.load(MODELS / f"ngram_{variant}.pkl")
        acc, n_pos = {1: out["ngram"]["top1"], 5: out["ngram"]["top5"]}, out["ngram"]["n_eval_positions"]

    # --------------------------------------------------------------- LSTM
    t0 = time.time()
    model, hist = train_lstm(
        tr, va, vocab, emb=cfg["emb"], hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"],
        epochs=cfg["epochs"], log=log,
        time_budget_s=time_budget_min * 60 if time_budget_min else None)
    train_device = str(get_device())
    # Score on the CPU: a 16k-token x 20k-word logits tensor is unreliable on the MPS backend.
    pred = LSTMPredictor(model, vocab, torch.device("cpu"))
    pred.save(MODELS / f"lstm_{variant}.pt")          # checkpoint before the (slow) scoring
    acc_l, n_pos_l = pred.topk_accuracy(te, (1, 5), skip)
    out["lstm"] = {"val_perplexity": pred.perplexity(va), "test_perplexity": pred.perplexity(te),
                   "top1": acc_l[1], "top5": acc_l[5], "n_eval_positions": n_pos_l,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "history": hist, "seconds": time.time() - t0, "device": train_device}
    log(f"[{variant}] LSTM     : test ppl {out['lstm']['test_perplexity']:.1f}  "
        f"top1 {acc_l[1]:.3f}  top5 {acc_l[5]:.3f}  ({out['lstm']['seconds']:.0f}s)")
    assert n_pos == n_pos_l, "both models must be scored on identical positions"

    # ------------------------------------------------ side-by-side examples
    ex = []
    for prompt in DEMO_PROMPTS[variant]:
        ids = vocab.encode(prompt.split())
        ex.append({"prompt": prompt, "ngram": ng.predict_next(ids, 5), "lstm": pred.predict_next(ids, 5)})
    out["examples"] = ex
    (RESULTS / f"lm_{variant}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(WORD_LABELS) + ["all"], default="all")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--time-budget-min", type=float)
    ap.add_argument("--stage", choices=["both", "ngram", "lstm"], default="both",
                    help="run the stages in separate processes to keep memory use low")
    ap.add_argument("--max-train", type=int, help="debug: limit training sentences")
    a = ap.parse_args()
    for v in (list(WORD_LABELS) if a.variant == "all" else [a.variant]):
        run(v, a.epochs, a.time_budget_min, a.max_train, stage=a.stage)


if __name__ == "__main__":
    main()
