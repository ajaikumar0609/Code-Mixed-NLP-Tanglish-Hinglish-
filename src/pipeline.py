"""End-to-end inference: the single entry point used by the Streamlit app and the CLI.

    text -> tokenize -> language ID -> normalize -> LM tokens -> N-gram / LSTM next words

The same ``tokenize``, ``Normalizer``, ``lm_tokens`` and ``Vocab`` objects are used
in training and evaluation, so the demo cannot drift from the experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pathlib import Path

from .config import MODELS, WORD_LABELS
from .language_identifier import BiLSTMLID, CharNgramLID, identify_tokens
from .lstm_model import LSTMPredictor
from .ngram_model import KNTrigramLM
from .normalizer import Normalizer, NormResult
from .preprocessing import lm_tokens, tokenize
from .vocab import Vocab

MAX_INPUT_CHARS = 1000
LID_MODELS = {"Char n-gram + LogReg": "char_ngram", "BiLSTM": "bilstm"}


@dataclass
class Analysis:
    text: str
    variant: str
    tokens: list[str]
    labels: list[str]
    confidences: list[float]
    normalized: NormResult
    code_mixing_index: float
    summary: str
    lm_context: list[str]
    oov_words: list[str]
    ngram: list[tuple[str, float]] = field(default_factory=list)
    lstm: list[tuple[str, float]] = field(default_factory=list)
    # per-model {"perplexity": float, "words": [(word, log2-probability)]} for the normalized text
    fluency: dict = field(default_factory=dict)


def code_mixing_index(labels: list[str]) -> float:
    """Das & Gambäck CMI over language-tagged words: 0 = monolingual, higher = more mixed."""
    langs = [l for l in labels if l in ("TA", "HI", "EN")]
    if not langs:
        return 0.0
    top = max(langs.count(l) for l in set(langs))
    return 100.0 * (1 - top / len(langs))


def _describe(variant: str, labels: list[str]) -> str:
    present = {l for l in labels if l in ("TA", "HI", "EN")}
    native = "Tamil" if variant == "tanglish" else "Hindi"
    if not present:
        return "No language-tagged words found."
    if len(present) == 1:
        only = present.pop()
        return {"EN": "English only", "TA": "Tamil only (Romanized)", "HI": "Hindi only (Romanized)"}[only]
    return f"{native}-English code-mixed"


class CodeMixedPipeline:
    def __init__(self, variant: str, models_dir: Path = MODELS):
        if variant not in WORD_LABELS:
            raise ValueError(f"variant must be one of {list(WORD_LABELS)}")
        self.variant = variant
        self.models_dir = Path(models_dir)
        self.lid: dict[str, object] = {}
        self.normalizer = Normalizer.for_variant(variant)
        self.vocab = self.ngram = self.lstm = None
        self._load()

    def _load(self) -> None:
        d, v = self.models_dir, self.variant
        if (d / f"lid_char_{v}.joblib").exists():
            self.lid["char_ngram"] = CharNgramLID.load(d / f"lid_char_{v}.joblib")
        if (d / f"lid_bilstm_{v}.pt").exists():
            self.lid["bilstm"] = BiLSTMLID.load(d / f"lid_bilstm_{v}.pt")
        if (d / f"vocab_{v}.json").exists():
            self.vocab = Vocab.load(d / f"vocab_{v}.json")
        if (d / f"ngram_{v}.pkl").exists():
            self.ngram = KNTrigramLM.load(d / f"ngram_{v}.pkl")
        if (d / f"lstm_{v}.pt").exists():
            self.lstm = LSTMPredictor.load(d / f"lstm_{v}.pt")

    def available(self) -> dict[str, bool]:
        return {"lid_char_ngram": "char_ngram" in self.lid, "lid_bilstm": "bilstm" in self.lid,
                "ngram": self.ngram is not None, "lstm": self.lstm is not None}

    # ------------------------------------------------------------------
    def identify(self, text: str, lid_model: str = "char_ngram"):
        toks = tokenize(text)
        model = self.lid.get(lid_model) or next(iter(self.lid.values()))
        preds = identify_tokens(model, toks)
        return toks, [p[0] for p in preds], [p[1] for p in preds]

    def analyze(self, text: str, lid_model: str = "char_ngram", k: int = 5) -> Analysis:
        if text is None or not text.strip():
            raise ValueError("Please enter a sentence.")
        text = text.strip()[:MAX_INPUT_CHARS]
        toks, labels, conf = self.identify(text, lid_model)
        if not toks:
            raise ValueError("No tokens found in the input.")
        norm = self.normalizer.normalize(text, labels)
        ctx = lm_tokens(norm.text)
        ids = self.vocab.encode(ctx) if self.vocab else []
        oov = [w for w, i in zip(ctx, ids) if self.vocab and i == self.vocab.unk]
        res = Analysis(
            text=text, variant=self.variant, tokens=[t.text for t in toks], labels=labels,
            confidences=conf, normalized=norm,
            code_mixing_index=code_mixing_index(labels), summary=_describe(self.variant, labels),
            lm_context=ctx, oov_words=oov)
        if self.vocab and ids:
            ids = ids[-50:]          # long inputs: only the recent context matters
            if self.ngram:
                res.ngram = self.ngram.predict_next(ids, k)
            if self.lstm:
                res.lstm = self.lstm.predict_next(ids, k)
            ctx_words = ctx[-50:]
            for name, model in (("ngram", self.ngram), ("lstm", self.lstm)):
                if model:
                    lp = model.token_logprobs(ids)               # words + </s>
                    res.fluency[name] = {
                        "perplexity": float(np.exp(-np.mean(lp))),
                        "words": [(w, float(l / np.log(2))) for w, l in zip(ctx_words, lp[:-1])]}
        return res

    # ------------------------------------------------------------------
    def continue_text(self, text: str, model: str = "lstm", n_words: int = 8, top_k: int = 10,
                      temperature: float = 0.8, seed: int | None = None) -> list[str]:
        """Sample ``n_words`` continuations: draw from the model's top-k, re-weighted by temperature."""
        lm = self.lstm if model == "lstm" else self.ngram
        if lm is None or self.vocab is None:
            return []
        rng = np.random.default_rng(seed)
        norm = self.normalizer.normalize(text.strip()[:MAX_INPUT_CHARS],
                                         self.identify(text)[1]).text
        ids = self.vocab.encode(lm_tokens(norm))[-50:]
        out: list[str] = []
        for _ in range(n_words):
            cands = lm.predict_next(ids, top_k)
            w = np.array([p for _, p in cands]) ** (1.0 / max(temperature, 1e-3))
            word = cands[int(rng.choice(len(cands), p=w / w.sum()))][0]
            out.append(word)
            ids = (ids + [self.vocab.stoi[word]])[-50:]
        return out

    def analyze_batch(self, lines: list[str], lid_model: str = "char_ngram") -> list[dict]:
        rows = []
        for line in lines:
            try:
                a = self.analyze(line, lid_model, k=1)
            except ValueError:
                continue
            rows.append({"input": a.text,
                         "tagged": " ".join(f"{t}[{l}]" for t, l in zip(a.tokens, a.labels)),
                         "normalized": a.normalized.text, "type": a.summary,
                         "cmi": round(a.code_mixing_index, 1),
                         "ngram_next": a.ngram[0][0] if a.ngram else "",
                         "lstm_next": a.lstm[0][0] if a.lstm else "",
                         "oov_words": " ".join(sorted(set(a.oov_words)))})
        return rows

    def oov_rate(self, text: str) -> float:
        """Fraction of the text's words missing from this variant's LM vocabulary."""
        if not self.vocab:
            return 1.0
        w = lm_tokens(self.normalizer.normalize(text).text)      # clean spelling first (haiii -> hai)
        return sum(i == self.vocab.unk for i in self.vocab.encode(w)) / max(len(w), 1)


def detect_variant(text: str, pipelines: dict[str, "CodeMixedPipeline"]) -> tuple[str, dict[str, float]]:
    """Guess Tanglish vs Hinglish.

    The variant whose vocabulary covers more of the (normalized) words wins; when coverage ties,
    the variant whose trigram model finds the sentence less surprising wins.
    Returns the choice and each variant's share of unknown words.
    """
    scores = {v: p.oov_rate(text) for v, p in pipelines.items()}
    best_oov = min(scores.values())
    tied = [v for v, r in scores.items() if r - best_oov < 1e-6]
    if len(tied) == 1:
        return tied[0], scores

    def surprise(v: str) -> float:
        p = pipelines[v]
        ids = p.vocab.encode(lm_tokens(p.normalizer.normalize(text).text)) if p.vocab and p.ngram else []
        return -float(np.mean(p.ngram.token_logprobs(ids))) if ids else float("inf")

    return min(tied, key=lambda v: (surprise(v), v != "tanglish")), scores


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Analyse one code-mixed sentence.")
    ap.add_argument("text")
    ap.add_argument("--variant", choices=list(WORD_LABELS), default="tanglish")
    ap.add_argument("--lid", choices=["char_ngram", "bilstm"], default="char_ngram")
    a = ap.parse_args()
    r = CodeMixedPipeline(a.variant).analyze(a.text, a.lid)
    print("Tags      :", " ".join(f"{t}[{l}]" for t, l in zip(r.tokens, r.labels)))
    print("Normalized:", r.normalized.text)
    print("Summary   :", r.summary, f"(CMI {r.code_mixing_index:.1f})")
    print("N-gram    :", ", ".join(f"{w} ({p:.3f})" for w, p in r.ngram))
    print("LSTM      :", ", ".join(f"{w} ({p:.3f})" for w, p in r.lstm))


if __name__ == "__main__":
    main()
