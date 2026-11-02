"""Integration tests: need trained models (run `python -m src.run_all` first); skipped otherwise."""
import pytest

from src.config import MODELS
from src.pipeline import CodeMixedPipeline

pytestmark = pytest.mark.skipif(not (MODELS / "ngram_hinglish.pkl").exists(),
                                reason="models not trained")


@pytest.fixture(scope="module")
def pipe():
    return CodeMixedPipeline("hinglish")


def test_analyze_returns_aligned_outputs(pipe):
    a = pipe.analyze("kal meeting haiii yaar, please notes share kar dena")
    assert len(a.tokens) == len(a.labels) == len(a.confidences)
    assert a.labels[a.tokens.index(",")] == "PUNCT"
    assert a.normalized.text.startswith("kal meeting hai ")
    assert len(a.ngram) == 5 and len(a.lstm) == 5
    assert all(0 < p <= 1 for _, p in a.ngram + a.lstm)
    assert not {"<unk>", "<s>", "</s>", "<pad>"} & {w for w, _ in a.ngram + a.lstm}


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_empty_input_is_rejected_gracefully(pipe, bad):
    with pytest.raises(ValueError):
        pipe.analyze(bad)


def test_unknown_words_and_symbols_do_not_crash(pipe):
    a = pipe.analyze("zzqxv 😂😂 @someone http://x.co 12345 ???")
    assert a.labels[a.tokens.index("12345")] == "NUM"
    assert a.ngram and a.lstm


def test_very_long_input_is_truncated(pipe):
    a = pipe.analyze("kal meeting hai " * 500)
    assert len(a.text) <= 1000


@pytest.mark.skipif(not (MODELS / "ngram_tanglish.pkl").exists(), reason="tanglish models not trained")
@pytest.mark.parametrize("text,expected", [
    ("naan inniku college ku late ah varen", "tanglish"),
    ("plsss send pannunga brooo", "tanglish"),
    ("kal meeting haiii yaar", "hinglish"),          # coverage tie, broken by trigram surprise
    ("aap ka video bahut accha hai", "hinglish"),
])
def test_variant_detection(text, expected):
    from src.pipeline import detect_variant
    pipes = {v: CodeMixedPipeline(v) for v in ("tanglish", "hinglish")}
    assert detect_variant(text, pipes)[0] == expected


def test_continue_text_and_fluency(pipe):
    words = pipe.continue_text("kal meeting", "lstm", n_words=5, seed=0)
    assert len(words) == 5 and words == pipe.continue_text("kal meeting", "lstm", n_words=5, seed=0)
    a = pipe.analyze("kal meeting hai")
    assert set(a.fluency) == {"ngram", "lstm"} and all(f["perplexity"] > 1 for f in a.fluency.values())
