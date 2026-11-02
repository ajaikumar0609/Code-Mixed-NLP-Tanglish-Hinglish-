"""Architecture diagrams and result charts for the report (values read from results/*.json)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/Users/ajaikumarn/Desktop/shally")
OUT = ROOT / "deliverables/build/assets"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "results"

NAVY, RED, TEAL, GREY, LIGHT, INK = "#14284B", "#D72639", "#2A9D8F", "#5C677D", "#EEF2F8", "#1F2937"
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 10


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, tc=INK, fs=8, bold=False, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q, color=GREY, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def canvas(w, h):
    """1 data unit = 1 inch, so font sizes are true sizes when the image is placed at 100% width."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def vstack_arrow(ax, x, y_from, y_to, color=GREY):
    arrow(ax, (x, y_from), (x, y_to), color=color)


# --------------------------------------------------------------- Fig: system architecture (7 x 7.6 in)
fig, ax = canvas(7, 7.7)
box(ax, 1.6, 6.95, 3.8, 0.6, "Raw code-mixed text (Roman script)\nTanglish or Hinglish", fc="white", bold=True, fs=8.5)
box(ax, 0.6, 5.95, 5.8, 0.65, "Preprocessing: NFKC + whitespace clean-up, lossless typed tokenizer\n(word, number, punctuation, URL, @mention, #hashtag, emoji)", fs=8)
vstack_arrow(ax, 3.5, 6.93, 6.62); 
# module 1
ax.add_patch(FancyBboxPatch((0.2, 3.95), 6.6, 1.65, boxstyle="round,pad=0,rounding_size=0.07", fc="white", ec=RED, lw=1.4))
ax.text(3.5, 5.42, "Module 1 - Language identification (token level)", ha="center", fontsize=8.8, fontweight="bold", color=RED)
box(ax, 0.4, 4.1, 1.95, 1.0, "Deterministic rules\nNUM, PUNCT, URL,\nmention, hashtag,\nemoji", fc=LIGHT, ec=GREY, fs=7.6)
box(ax, 2.55, 4.1, 1.95, 1.0, "Model A\nchar n-gram TF-IDF\n+ Logistic\nRegression", fc=LIGHT, ec=TEAL, fs=7.6)
box(ax, 4.65, 4.1, 1.95, 1.0, "Model B\nword emb. + char-CNN\n+ BiLSTM tagger", fc=LIGHT, ec=TEAL, fs=7.6)
vstack_arrow(ax, 3.5, 5.93, 5.62)
# module 2
box(ax, 0.6, 2.95, 5.8, 0.65, "Module 2 - Text normalization (uses the predicted tags)\nrule-based + language-aware dictionaries; not translation", ec=RED, fs=8, bold=False)
vstack_arrow(ax, 3.5, 3.93, 3.62)
# modules 3, 4
box(ax, 0.2, 1.5, 3.15, 0.95, "Module 3 - Trigram LM\ninterpolated Kneser-Ney (numpy)\nperplexity, top-1 / top-5", ec=RED, fs=7.8)
box(ax, 3.65, 1.5, 3.15, 0.95, "Module 4 - LSTM LM (PyTorch)\nembedding > LSTM > linear > softmax\nperplexity, top-1 / top-5", ec=RED, fs=7.8)
arrow(ax, (3.2, 2.93), (1.8, 2.47)); arrow(ax, (3.8, 2.93), (5.2, 2.47))
ax.text(3.5, 2.68, "normalized LM tokens", ha="center", fontsize=7.2, color=GREY, style="italic", bbox=dict(fc="white", ec="none", pad=1))
box(ax, 0.2, 0.2, 6.6, 0.75, "CodeMixedPipeline (src/pipeline.py): the single code path used by the\nStreamlit app, the command line and the evaluation scripts", fc=NAVY, ec=NAVY, tc="white", fs=8, bold=True)
arrow(ax, (1.8, 1.48), (1.8, 0.97), color=NAVY); arrow(ax, (5.2, 1.48), (5.2, 0.97), color=NAVY)
save(fig, "fig_system.png")

# --------------------------------------------------------------- Fig: LID models (7 x 4.4 in)
fig, ax = canvas(7, 4.4)
ax.text(3.5, 4.2, "Token-level language identification: two models, one label set", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
ax.text(0.05, 3.78, "Model A - context-free baseline", fontsize=8.8, fontweight="bold", color=TEAL)
bw, gap, x0 = 1.24, 0.1, 0.05
labsA = ["word token\n(e.g. inniku)", "lower-case,\nchar_wb n-grams\n(1-4 / 2-5)", "TF-IDF\n(sublinear tf)\n+ case flags", "Logistic\nRegression\n(lbfgs, L2)", "P(label | word)\nTA/HI, EN,\nOTHER"]
for i, t in enumerate(labsA):
    box(ax, x0 + i * (bw + gap), 2.82, bw, 0.82, t, fc="white" if i in (0, 4) else LIGHT, ec=TEAL, fs=7.4)
    if i: arrow(ax, (x0 + i * (bw + gap) - gap + 0.01, 3.23), (x0 + i * (bw + gap) - 0.01, 3.23))
ax.text(0.05, 2.48, "Model B - sentence-context tagger", fontsize=8.8, fontweight="bold", color=RED)
labsB = ["sentence\nw1 w2 ... wn", "word emb. (64)\n+ char-CNN\n(32-d, k=3,5)", "concatenate\n(64 + 128)\n+ dropout 0.3", "BiLSTM\n128 hidden\nper direction", "linear + softmax\nlabel per token"]
for i, t in enumerate(labsB):
    box(ax, x0 + i * (bw + gap), 1.52, bw, 0.82, t, fc="white" if i in (0, 4) else LIGHT, ec=RED, fs=7.4)
    if i: arrow(ax, (x0 + i * (bw + gap) - gap + 0.01, 1.93), (x0 + i * (bw + gap) - 0.01, 1.93))
box(ax, 0.05, 0.1, 6.9, 1.0, "Words: labelled TA/HI, EN or OTHER by the model.\nNumbers, punctuation, URLs, mentions, hashtags and emoji: fixed rule labels (confidence 1.0).\nModel B trained with cross-entropy and Adam; best validation epoch kept.", fc=NAVY, ec=NAVY, tc="white", fs=7.8, bold=False)
save(fig, "fig_lid.png")

# --------------------------------------------------------------- Fig: normalization (7 x 3.2 in)
fig, ax = canvas(7, 3.2)
ax.text(0.05, 2.95, "input:   plsss send pannunga brooo", fontsize=8, color=INK, family="DejaVu Sans Mono")
ax.text(3.6, 2.95, "output:  please send pannunga bro", fontsize=8, color=TEAL, family="DejaVu Sans Mono", fontweight="bold")
stages = ["Unicode and\nwhitespace\nclean-up", "Typed\ntokenization", "Punctuation /\nemoji run\ncollapsing", "Elongation\nreduction vs.\nknown words", "Dictionary\nlookup and\nslang expansion"]
for i, t in enumerate(stages):
    box(ax, x0 + i * (bw + gap), 1.55, bw, 1.0, t, fc=LIGHT, ec=NAVY, fs=7.6)
    if i: arrow(ax, (x0 + i * (bw + gap) - gap + 0.01, 2.05), (x0 + i * (bw + gap) - 0.01, 2.05))
ax.text(0.05, 1.28, "TA/HI-tagged words: variant and learned dictionaries.   EN-tagged words: chat-slang expansion (plz > please).   OTHER: unchanged.", fontsize=7.2, color=GREY)
box(ax, 0.05, 0.2, 6.9, 0.7, "Normalization = one preferred Romanized spelling. It is NOT translation and NOT transliteration.", fc=NAVY, ec=NAVY, tc="white", fs=8, bold=True)
save(fig, "fig_norm.png")

# --------------------------------------------------------------- Fig: language models (7 x 5.2 in)
fig, ax = canvas(7, 5.2)
ax.text(1.75, 5.0, "Trigram LM: interpolated Kneser-Ney", ha="center", fontsize=9, fontweight="bold", color=NAVY)
ax.text(5.25, 5.0, "LSTM LM (PyTorch)", ha="center", fontsize=9, fontweight="bold", color=NAVY)
box(ax, 0.05, 3.9, 3.4, 0.8, "context (u, v); trigram counts c(uvw);\ncontinuation counts N1+(. v w)", fc="white", ec=NAVY, fs=7.6)
box(ax, 0.05, 2.75, 3.4, 0.8, "P3(w|uv) = discounted trigram count\n+ gamma(uv) * P2(w|v)", fc=LIGHT, ec=NAVY, fs=7.6)
box(ax, 0.05, 1.6, 3.4, 0.8, "P2(w|v) = discounted continuation count\n+ gamma(v) * P1(w)", fc=LIGHT, ec=NAVY, fs=7.6)
box(ax, 0.05, 0.5, 3.4, 0.8, "P1(w) = continuation unigram + uniform\n1/|V| floor; D tuned on validation", fc="white", ec=NAVY, fs=7.6)
for a, b in ((3.88, 3.57), (2.73, 2.42), (1.58, 1.32)): arrow(ax, (1.75, a), (1.75, b), color=NAVY)
ax.text(1.75, 0.2, "exact top-1 / top-5 next word; perplexity", ha="center", fontsize=7.4, color=GREY)
lay = ["token ids  <s> w1 ... wt", "embedding (256 / 200-d)", "dropout", "LSTM, 1 layer (512 / 400 hidden)", "linear projection to embedding size", "tied output layer + softmax"]
ys = [4.05, 3.4, 2.75, 2.1, 1.45, 0.8]
for t, y in zip(lay, ys):
    box(ax, 3.75, y, 3.2, 0.5, t, fc="white" if y in (4.0, 2.76, 1.52) else LIGHT, ec=RED, fs=7.6)
for i in range(len(ys) - 1): arrow(ax, (5.35, ys[i] - 0.01), (5.35, ys[i + 1] + 0.51), color=RED)
ax.text(5.35, 0.68, "cross-entropy loss, Adam, BPTT, clip-norm 1.0;\nbest-validation checkpoint kept", ha="center", va="top", fontsize=7.2, color=GREY)
save(fig, "fig_lm.png")

# --------------------------------------------------------------- Fig: data flow (7 x 4.2 in)
fig, ax = canvas(7, 4.2)
ax.text(1.75, 4.0, "Hinglish: COMI-LINGUA", ha="center", fontsize=9, fontweight="bold", color=NAVY)
ax.text(5.25, 4.0, "Tanglish: Tanglish-Corpus-185k", ha="center", fontsize=9, fontweight="bold", color=RED)
Lh = ["LID + TN CSVs; 3 annotators per sentence", "Roman-script sentences only; token label =\n3-annotator majority  (GOLD)", "official train/test split kept;\n10% of train held out as validation", "LM text = annotator-normalized reference\n(de-duplicated)"]
Lt = ["185,973 sentences: YouTube, Reddit,\nDravidianCodeMix", "de-duplicate (303 removed) > 185,670", "split BY SOURCE; lexicon rules give\nSILVER token labels (weak supervision)", "120 hand-reviewed sentences: GOLD\nevaluation set, never trained on"]
for i, (a, b) in enumerate(zip(Lh, Lt)):
    y = 3.0 - i * 0.95
    box(ax, 0.05, y, 3.4, 0.7, a, fc="white" if i == 0 else LIGHT, ec=NAVY, fs=7.4)
    box(ax, 3.55, y, 3.4, 0.7, b, fc="white" if i == 0 else LIGHT, ec=RED, fs=7.4)
    if i < 3:
        arrow(ax, (1.75, y - 0.01), (1.75, y - 0.25)); arrow(ax, (5.25, y - 0.01), (5.25, y - 0.25))
save(fig, "fig_data.png")

# --------------------------------------------------------------- Charts
lh = json.load(open(R / "lid_hinglish.json")); lt = json.load(open(R / "lid_tanglish.json"))
mh = json.load(open(R / "lm_hinglish.json")); mt = json.load(open(R / "lm_tanglish.json"))
nm = json.load(open(R / "normalization.json"))

a_hi = lh["models"]["char_ngram"]["results"]["test"]["word_level"]["accuracy"] * 100
b_hi = lh["models"]["bilstm"]["results"]["test"]["word_level"]["accuracy"] * 100
a_ta = lt["models"]["char_ngram"]["results"]["hand_reviewed_gold"]["word_level"]["accuracy"] * 100
b_ta = lt["models"]["bilstm"]["results"]["hand_reviewed_gold"]["word_level"]["accuracy"] * 100
a_ta_s = lt["models"]["char_ngram"]["results"]["test"]["word_level"]["accuracy"] * 100
b_ta_s = lt["models"]["bilstm"]["results"]["test"]["word_level"]["accuracy"] * 100
wer0 = nm["hinglish"]["systems"]["no_normalization (baseline)"]["best_of_refs"]["wer"] * 100
wer1 = nm["hinglish"]["systems"]["rules + learned dictionary"]["best_of_refs"]["wer"] * 100
wer_ia = nm["hinglish"]["inter_annotator"]["mean_wer"] * 100
print("LID", a_hi, b_hi, a_ta, b_ta, a_ta_s, b_ta_s, "WER", wer0, wer1, wer_ia)


def bars(ax, groups, series, colors, ylabel, title, fmt="{:.1f}", ylim=None, legend_loc="upper left"):
    import numpy as np
    n = len(series); width = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        b = ax.bar(xs, vals, width * 0.92, label=name, color=colors[i], edgecolor="white")
        for x, v in zip(xs, vals):
            ax.text(x, v + (ylim[1] - ylim[0]) * 0.012 if ylim else v, fmt.format(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=8.8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY)
    if ylim: ax.set_ylim(*ylim)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc=legend_loc, ncol=2 if legend_loc=="upper center" else 1)


fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
bars(axs[0, 0], ["Hinglish\n(gold test)", "Tanglish\n(hand-reviewed gold)"],
     {"Char n-gram + LogReg": [a_hi, a_ta], "BiLSTM": [b_hi, b_ta]}, [GREY, RED], "word accuracy (%)",
     "(a) Language identification", ylim=(0, 125), legend_loc="upper center")
bars(axs[0, 1], ["Hinglish", "Tanglish"], {"Trigram KN": [mh["ngram"]["test_perplexity"], mt["ngram"]["test_perplexity"]],
     "LSTM": [mh["lstm"]["test_perplexity"], mt["lstm"]["test_perplexity"]]}, [GREY, RED], "test perplexity (lower is better)",
     "(b) Language-model perplexity", fmt="{:.0f}", ylim=(0, 520))
bars(axs[1, 0], ["Hinglish\ntop-1", "Hinglish\ntop-5", "Tanglish\ntop-1", "Tanglish\ntop-5"],
     {"Trigram KN": [mh["ngram"]["top1"] * 100, mh["ngram"]["top5"] * 100, mt["ngram"]["top1"] * 100, mt["ngram"]["top5"] * 100],
      "LSTM": [mh["lstm"]["top1"] * 100, mh["lstm"]["top5"] * 100, mt["lstm"]["top1"] * 100, mt["lstm"]["top5"] * 100]},
     [GREY, RED], "next-word accuracy (%)", "(c) Next-word prediction", ylim=(0, 48))
bars(axs[1, 1], ["No normalization", "Rules + learned\ndictionary", "Annotator\ndisagreement"],
     {"WER (%)": [wer0, wer1, wer_ia]}, [TEAL], "WER (%), lower is better", "(d) Hinglish normalization WER", ylim=(0, 21))
axs[1, 1].legend_.remove()
axs[1, 1].patches[2].set_color(GREY)
fig.tight_layout(w_pad=2.0, h_pad=2.2)
save(fig, "fig_results.png")
print("figures ok")
