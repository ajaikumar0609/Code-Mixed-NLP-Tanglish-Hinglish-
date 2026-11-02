"""Fast unit tests for the pieces that do not need trained models or the datasets."""
import numpy as np
import pytest

from src.evaluation import edit_distance, lid_report, normalization_scores, wer_cer
from src.ngram_model import KNTrigramLM
from src.normalizer import Normalizer, align_pairs, learn_dictionary
from src.preprocessing import (detokenize, elongation_candidates, lm_tokens, rule_label,
                               tokenize)
from src.vocab import Vocab


# ---------------------------------------------------------------- preprocessing
def test_tokenize_types_and_roundtrip():
    s = "naan college-ku late ah varen 😂 @bro #Tamil 10th 3.5!!!"
    toks = tokenize(s)
    assert [t.kind for t in toks][:5] == ["WORD", "WORD", "PUNCT", "WORD", "WORD"]
    assert detokenize(tokenize(s)) == s
    kinds = {t.text: t.kind for t in toks}
    assert kinds["@bro"] == "MENTION" and kinds["#Tamil"] == "HASHTAG"
    assert kinds["10th"] == "WORD" and kinds["3.5"] == "NUM"


def test_rule_labels_and_scripts():
    assert [rule_label(t) for t in tokenize("hi , 12 😂 @x")] == [None, "PUNCT", "NUM", "OTHER", "OTHER"]
    assert rule_label(tokenize("नमस्ते")[0]) == "HI"
    assert rule_label(tokenize("வணக்கம்")[0]) == "TA"
    assert rule_label(tokenize("â")[0]) == "OTHER"          # mojibake


def test_lm_tokens_drop_punct_and_map_numbers():
    assert lm_tokens("Meeting at 10:30, #Urgent!!! 😂") == ["meeting", "at", "<num>", "urgent"]
    assert lm_tokens("") == []


def test_elongation_candidates_only_touch_long_runs():
    assert elongation_candidates("irukku") == []
    assert elongation_candidates("good") == []
    assert set(elongation_candidates("hiii")) == {"hi", "hii"}


# ------------------------------------------------------------------ normalizer
def test_normalizer_collapses_punctuation_and_elongation():
    n = Normalizer("hinglish")
    assert n("wow!!!").endswith("wow!")
    assert n("kal meeting hai....").endswith("hai...")
    n2 = Normalizer("hinglish", known={"romba": 50, "hai": 100})
    assert n2("rombbba nalla!!!") == "romba nalla!"
    assert n2("haiii") == "hai"


def test_tanglish_normalizer_is_language_aware():
    from collections import Counter
    n = Normalizer("tanglish", known=Counter({"irukku": 10}))
    assert n("iruku").strip() == "irukku"
    assert n("plz send u", ["EN", "EN", "EN"]) == "please send you"
    # "than" is Tamil emphasis only when tagged TA; as English it must stay
    assert n("sari than", ["TA", "TA"]) == "seri thaan"
    assert n("better than", ["EN", "EN"]) == "better than"
    assert n("Vijay", ["OTHER"]) == "Vijay"                 # names untouched
    assert n("Plz") == "Please"                              # casing preserved


def test_normalizer_expands_to_multiple_words_and_keeps_urls():
    n = Normalizer("tanglish")
    assert n("tq bro http://x.co/a") == "thank you bro http://x.co/a"


def test_learn_dictionary_keeps_only_consistent_rewrites():
    ex = [{"original": "kal bnane ke liye", "refs": ["kal banane ke liye"] * 3}] * 3 + \
         [{"original": "hai kya", "refs": ["hain kya", "hai kya", "hai kya"]}] * 3
    d = learn_dictionary(ex, min_count=2, min_ratio=0.5)
    assert d["bnane"][0] == "banane"
    assert "hai" not in d                                     # rewritten in only 1/3 of references
    assert align_pairs("a b c", "a x c") == [("b", "x")]


# ------------------------------------------------------------------- evaluation
def test_edit_distance_and_wer_cer():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert edit_distance("", "abc") == 3
    we, wn, ce, cn = wer_cer("Hii FRNDS!!", "hii friends")
    assert (we, wn) == (1, 2) and ce > 0


def test_normalization_scores_best_of_refs():
    s = normalization_scores(["a b"], [["a c", "a b"]])
    assert s["best_of_refs"]["exact_match"] == 1.0 and s["first_ref"]["exact_match"] == 0.0


def test_lid_report_macro_f1():
    r = lid_report(["EN", "EN", "HI", "HI"], ["EN", "HI", "HI", "HI"], ["HI", "EN", "OTHER"])
    assert r["accuracy"] == 0.75
    assert r["per_class"]["OTHER"]["support"] == 0
    assert 0 < r["macro_f1"] < 1                              # unsupported classes are excluded


# -------------------------------------------------------------------- n-gram LM
@pytest.fixture(scope="module")
def toy_lm():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(40)]
    sents = [[words[i] for i in rng.integers(0, 12 if rng.random() < .6 else 40, rng.integers(3, 9))]
             for _ in range(600)]
    vocab = Vocab.build(sents, min_count=2)
    enc = [vocab.encode(s) for s in sents]
    return KNTrigramLM(vocab).fit(enc[:500]), enc[500:], vocab


def test_kn_distribution_sums_to_one_and_is_positive(toy_lm):
    lm, _, vocab = toy_lm
    V = len(vocab)
    for ctx in [(vocab.bos, vocab.bos), (vocab.stoi["w1"], vocab.stoi["w2"]), (vocab.unk, vocab.unk)]:
        p = lm.prob_batch(np.full(V, ctx[0]), np.full(V, ctx[1]), np.arange(V))
        assert p.sum() == pytest.approx(1.0, abs=1e-9) and p.min() > 0


def test_kn_topk_is_exact(toy_lm):
    lm, held, vocab = toy_lm
    V = len(vocab)
    for s in held[:30]:
        for i in range(len(s)):
            u, v = ([vocab.bos, vocab.bos] + s[:i])[-2:]
            p = lm.prob_batch(np.full(V, u), np.full(V, v), np.arange(V))
            p[~lm._suggestible] = -1
            fast = np.array([q for _, q in lm.predict_next(s[:i], 5)])
            assert np.allclose(np.sort(p)[::-1][:5], fast)


def test_kn_perplexity_beats_uniform(toy_lm):
    lm, held, vocab = toy_lm
    assert lm.perplexity(held) < len(vocab)
