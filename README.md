# Code-Mixed NLP: Tanglish & Hinglish

Token-level **language identification**, **text normalization**, and **next-word prediction** (trigram vs LSTM)
for Romanized Tamil-English (Tanglish) and Hindi-English (Hinglish), with a Streamlit dashboard.

```
naan inniku college ku late ah varen   ->  naan[TA] inniku[TA] college[EN] ku[TA] late[EN] ah[TA] varen[TA]
plsss send pannunga brooo              ->  please send pannunga bro
kal meeting haiii yaar                 ->  kal meeting hai yaar
```

## Quick start

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_all          # download data, train, evaluate (about 1 hour on an Apple-silicon laptop)
.venv/bin/streamlit run app.py           # dashboard
.venv/bin/python -m pytest tests -q      # unit + integration tests
.venv/bin/python -m src.pipeline "kal meeting haiii yaar" --variant hinglish   # CLI
```

Stages can be run separately: `python -m src.run_all --only data|lid|norm|lm`. Trained models are written to
`models/`, metrics to `results/*.json` (the dashboard reads these; nothing is hard-coded).

## Results (held-out test data)

All numbers were produced by this repository's code. They describe these datasets and splits, not general accuracy.

**Language identification** (word tokens; numbers/punctuation/URLs are rule-labelled)

| | Char n-gram + TF-IDF + LogReg | BiLSTM (word + char-CNN) |
|---|---|---|
| Hinglish, gold COMI-LINGUA test: accuracy / macro-F1 | 95.8% / 0.889 | 96.3% / 0.906 |
| Tanglish, **hand-reviewed gold** (120 sentences): accuracy / macro-F1 | 95.3% / 0.857 | 95.1% / 0.850 |
| Tanglish, silver test (inflated, see below) | 99.0% / 0.989 | 98.8% / 0.987 |

Transfer check (English vs native-language only): the Hinglish model scores 96.3% on Hinglish but 63.1% on
Tanglish text; the Tanglish model scores 97.6% in-variant but the Hinglish model gets 85.6% on it. Evaluate the two mixes separately.

**Normalization** (COMI-LINGUA TN test, WER on lower-cased words without punctuation)

| System | WER vs best of 3 references | WER vs first reference | Exact match (best ref) |
|---|---|---|---|
| no normalization | 13.7% | 24.1% | 18.2% |
| rules only | 13.2% | 23.6% | 19.1% |
| rules + learned dictionary | **10.1%** | **18.2%** | **21.6%** |

The three human annotators disagree with each other at a mean WER of **16.2%** (no single correct spelling exists),
so the first-reference figure is the one to compare with that. Tanglish: 97% exact match on 68 hand-written pairs, which
only shows the rules do what the dictionary says (the references follow the same convention).

**Language models** (same corpus, same vocabulary, same test sentences and scoring for both models)

| | Hinglish trigram-KN | Hinglish LSTM | Tanglish trigram-KN | Tanglish LSTM |
|---|---|---|---|---|
| test perplexity (lower is better) | 192.6 | **141.7** | 415.7 | **259.4** |
| top-1 next-word accuracy | 17.8% | **19.0%** | 11.5% | **12.8%** |
| top-5 next-word accuracy | 34.9% | **37.0%** | 23.2% | **25.4%** |
| size | 284k trigram types | 3.1M params | 1.8M trigram types | 6.8M params |

The LSTM wins on perplexity by a wide margin and on next-word accuracy by a small one. Perplexity is computed over every token
including `</s>` and `<unk>`; accuracy only over positions whose true next word is a real vocabulary word.

## What was built

| Module | File | Notes |
|---|---|---|
| Tokenization | `src/preprocessing.py` | typed, lossless tokens (word / number / punctuation / URL / mention / hashtag / emoji) |
| Language ID | `src/language_identifier.py`, `src/train_lid.py` | Model A + Model B; small validation grid for Model A |
| Normalizer | `src/normalizer.py`, `src/tanglish_lexicon.py`, `src/train_normalizer.py` | elongation reduction checked against a known-word list, language-aware dictionary, slang expansion |
| Trigram LM | `src/ngram_model.py` | interpolated Kneser-Ney written from scratch (numpy); exact top-k; discount tuned on validation |
| LSTM LM | `src/lstm_model.py`, `src/train_lm.py` | embedding, LSTM, tied softmax, dropout, early stopping |
| Metrics | `src/evaluation.py` | accuracy / macro-F1, WER / CER (own edit distance), perplexity, top-k |
| Inference | `src/pipeline.py` | the one code path used by the app, CLI and evaluation |
| App | `app.py` | Analyze, Batch, Evaluation, About (see below) |

## The app

* **Analyze:** auto-detects Tanglish vs Hinglish (override any time); colour-coded word tags with a legend and low-confidence flags;
  normalization diff with the reason for each change; side-by-side next-word suggestions with "both models agree" badges and
  click-to-append; **Continue writing** (sample the next words from either model, with top-k and temperature controls);
  a **Fluency** heatmap showing which words each model finds surprising, plus sentence perplexity; recent-history sidebar; JSON export.
* **Batch:** paste lines or upload `.txt`/`.csv`, get tags, normalized text, code-mixing index and next-word guesses, download as CSV.
* **Evaluation:** headline cards, per-class scores, confusion matrices, transfer check, normalization errors, LM comparison charts.
* Everything runs locally; text you type is not sent anywhere.

## Data

| Task | Hinglish | Tanglish |
|---|---|---|
| Language ID | COMI-LINGUA LID (CC-BY-4.0), Roman-script sentences only (7,191 train / 799 val / 750 test), token labels = majority of 3 annotators | 30k / 2k / 3k sentences from Tanglish-Corpus-185k with **silver** labels (`src/silver_labels.py`) + 120 hand-reviewed sentences (`data/gold/`) |
| Normalization | COMI-LINGUA TN, 3 reference columns | curated variant groups + 68 hand-written pairs |
| Language modelling | COMI-LINGUA annotator-normalized text (21k / 2.3k / 4.9k) | Tanglish-Corpus-185k normalized by this project's normalizer (141k / 26k / 19k) |

Files you asked for: `data/language_id.csv`, `data/normalization.csv`, `data/processed/<variant>_{train,val,test}.txt`.
Label convention: `TA`/`HI` Romanized Tamil/Hindi; `EN` English incl. chat slang and loanwords like `bro`, `late`;
`OTHER` names, acronyms, handles, URLs, emoji, noise; `NUM`; `PUNCT`.

Splits: Tanglish is split **by source** (YouTube video / subreddit) after de-duplication, so one thread cannot straddle train and test.
Hinglish keeps the official train/test split, with validation carved from train.

## Limitations (please read)

* **Tanglish token labels are silver.** No public gold token-level Tamil-English data exists, so training labels come from a curated
  Tamil lexicon plus English word frequencies. Models can at best reproduce those rules, so the 99% silver-test score is misleading.
  The 95% hand-reviewed figure is the honest one, and the weak spot is names/acronyms (class `OTHER`, recall ~0.46).
* **The Tanglish gold set was reviewed by an AI assistant, not by independent native-speaker annotators.** About 4% of silver labels
  were corrected. It should be extended and re-checked by people before it is used for anything important.
* Tanglish normalization has no parallel corpus; its score is a sanity check, not a benchmark.
* Hinglish near-duplicate tweets can exist across the official train/test split, which flatters all Hinglish models slightly.
* Perplexity does not mean generated sentences are meaningful; the corpora are noisy social-media text.
* Tanglish vocabulary is capped at 20,000 words (test OOV 10.9%). A 30,000-word softmax froze LSTM training on the Mac GPU (MPS)
  backend, so the cap is a practical limit, applied identically to both models.
* The Tanglish LSTM trained for 10 epochs (validation perplexity was still falling slowly); the Hinglish LSTM early-stopped.
  Trigram discount and LID hyperparameters were tuned on validation data only.
* Out of scope: translation, transliteration into native script, sentiment, speech, transformers.

## Layout

```
app.py  requirements.txt  README.md
src/        preprocessing, language_identifier, normalizer, ngram_model, lstm_model, evaluation, pipeline, train_*.py, run_all.py
data/       raw/ (downloaded), processed/, lexicons/, gold/, language_id.csv, normalization.csv
models/     trained models (joblib / pickle / torch)
results/    metrics JSON + dashboard screenshots
notebooks/  dataset_exploration.ipynb
tests/      unit + integration tests
```
