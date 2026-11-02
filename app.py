"""Streamlit app: Analyze, Batch, Evaluation, About.

    streamlit run app.py

Every prediction comes from `src.pipeline.CodeMixedPipeline`, the same code the evaluation scripts use.
Metrics on the Evaluation page are read from `results/*.json` written by the training scripts.
"""
from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import RESULTS
from src.pipeline import LID_MODELS, CodeMixedPipeline, detect_variant

st.set_page_config(page_title="Code-Mixed NLP", page_icon="💬", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- style
st.markdown("""
<style>
.block-container {padding-top: 3.6rem; max-width: 1100px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(120deg, rgba(91,91,214,.16), rgba(30,136,229,.10) 55%, rgba(255,140,0,.12));
       border: 1px solid rgba(128,128,128,.22);}
.hero h1 {margin: 0 0 .15rem 0; font-size: 1.9rem;}
.hero p  {margin: 0; opacity: .78; font-size: .98rem;}
.chip {display:inline-flex; align-items:center; gap:5px; margin:3px 5px 3px 0; padding:4px 11px;
       border-radius:16px; font-size:1.0rem; line-height:1.3; border:1.5px solid transparent;}
.chip sup {font-size:.62rem; font-weight:700; opacity:.8; margin-left:1px}
.chip.low {border:1.5px dashed rgba(220,60,60,.85);}
.legend span {display:inline-block; margin-right:12px; font-size:.82rem; opacity:.9}
.legend i {display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:5px; vertical-align:-1px}
.diff {background:rgba(46,160,67,.22); border-radius:6px; padding:1px 6px; margin-right:3px}
.sugg {display:flex; align-items:center; gap:8px; margin:3px 0; min-height:28px; white-space:nowrap}
.sugg .bar {height:9px; border-radius:5px; background:rgba(91,91,214,.75);}
.sugg .w {min-width:90px; font-weight:600}
.sugg .p {font-size:.8rem; opacity:.75; min-width:48px}
.badge {white-space:nowrap; font-size:.68rem; padding:1px 7px; border-radius:9px; background:rgba(46,160,67,.25); margin-left:4px}
.gen b {background:rgba(91,91,214,.22); border-radius:6px; padding:1px 5px; margin-right:2px; font-weight:600}
.card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem}
.muted {opacity:.7; font-size:.88rem}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.22); border-radius:12px; padding:.55rem .9rem}
</style>
""", unsafe_allow_html=True)

TAGS = {  # label -> (background, display name); semi-transparent so they work in light and dark themes
    "TA": ("rgba(255,140,0,.24)", "Tamil"),
    "HI": ("rgba(0,150,136,.24)", "Hindi"),
    "EN": ("rgba(30,136,229,.24)", "English"),
    "OTHER": ("rgba(156,39,176,.22)", "Name / URL / other"),
    "NUM": ("rgba(120,120,120,.26)", "Number"),
    "PUNCT": ("rgba(120,120,120,.14)", "Punctuation"),
}
LOW_CONF = 0.70

EXAMPLES = {
    "Tanglish": [
        ("Code-mixed", "naan inniku college ku late ah varen"),
        ("Code-mixed", "indha project romba useful ah irukku"),
        ("Noisy spelling", "plsss send pannunga brooo"),
        ("Noisy spelling", "rombbba nalla irukku!!!"),
        ("Short", "enaku onnum theriyala bro"),
    ],
    "Hinglish": [
        ("Code-mixed", "kal meeting hai, please notes share kar dena"),
        ("Noisy spelling", "kal meeting haiii yaar"),
        ("Noisy spelling", "plz call me bro aaj shaam ko"),
        ("Short", "aap ka video bahut accha hai"),
        ("Short", "main kal office nahi aaunga"),
    ],
}
SETTINGS_HELP = ("Top-k limits sampling to the k most likely words; lower temperature makes the output "
                 "more predictable, higher makes it more varied.")


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading models…")
def get_pipeline(variant: str) -> CodeMixedPipeline:
    return CodeMixedPipeline(variant)


@st.cache_data
def load_results(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def legend(labels=("TA", "HI", "EN", "OTHER", "NUM", "PUNCT")) -> str:
    items = "".join(f"<span><i style='background:{TAGS[l][0].replace('.2', '.8').replace('.1', '.6')}'></i>"
                    f"{TAGS[l][1]}</span>" for l in labels)
    return f"<div class='legend'>{items}<span><i style='border:1.5px dashed rgb(220,60,60)'></i>low confidence</span></div>"


def chips(tokens, labels, confs) -> str:
    out = []
    for t, l, c in zip(tokens, labels, confs):
        bg, nice = TAGS.get(l, TAGS["OTHER"])
        low = " low" if (c < LOW_CONF and l in ("TA", "HI", "EN", "OTHER")) else ""
        out.append(f"<span class='chip{low}' style='background:{bg}' title='{nice} · confidence {c:.0%}'>"
                   f"{html.escape(t)}<sup>{l}</sup></span>")
    return "".join(out)


def diff_html(norm) -> str:
    parts = []
    for it in norm.tokens:
        if it.changed:
            parts.append(f"<span class='diff' title='rule: {it.rule}'><s style='opacity:.55'>"
                         f"{html.escape(it.original)}</s> → <b>{html.escape(it.normalized)}</b></span>")
        else:
            parts.append(html.escape(it.original))
    return " ".join(parts)


def suggestion_bars(preds, other_words=()) -> str:
    if not preds:
        return "<span class='muted'>Model not available.</span>"
    top = max(p for _, p in preds)
    rows = []
    for w, p in preds:
        badge = "<span class='badge'>both models</span>" if w in other_words else ""
        rows.append(f"<div class='sugg'><span class='w'>{html.escape(w)}</span>"
                    f"<div class='bar' style='width:{max(4, 150 * p / top):.0f}px'></div>"
                    f"<span class='p'>{p:.1%}</span>{badge}</div>")
    return "".join(rows)


def surprise_html(words) -> str:
    """Colour each word by how surprising it was (bits); darker = more surprising."""
    out = []
    for w, bits in words:
        b = -bits
        a = min(0.65, max(0.04, (b - 3) / 14))                  # ~3 bits: unremarkable, 17+ bits: very surprising
        out.append(f"<span class='chip' style='background:rgba(220,60,60,{a:.2f})' "
                   f"title='{b:.1f} bits (p = {2 ** bits:.1e})'>{html.escape(w)}</span>")
    return "".join(out)


def set_text(t: str) -> None:
    st.session_state["text"] = t


def append_word(word: str) -> None:
    st.session_state["text"] = (st.session_state.get("text", "").rstrip() + " " + word).strip()


def remember(text: str, variant: str) -> None:
    hist = st.session_state.setdefault("history", [])
    item = (text, variant)
    if item in hist:
        hist.remove(item)
    hist.insert(0, item)
    del hist[8:]


# ----------------------------------------------------------------------------- analyze page
def page_analyze():
    st.markdown("<div class='hero'><h1>💬 Code-Mixed Text Studio</h1>"
                "<p>Type a Tanglish or Hinglish message. See each word's language, a cleaned-up spelling, "
                "what two language models expect next, and how natural your sentence sounds.</p></div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1.35, 1])
    choice = c1.segmented_control("Language mix", ["Auto-detect", "Tanglish", "Hinglish"],
                                  default="Auto-detect", key="variant_choice",
                                  help="Auto-detect picks the variant whose vocabulary covers more of your words.")
    lid_name = c2.segmented_control("Language-ID model", list(LID_MODELS), default=list(LID_MODELS)[0],
                                    key="lid_choice",
                                    help="Model A reads each word alone; the BiLSTM also uses the surrounding words.")
    choice, lid_name = choice or "Auto-detect", lid_name or list(LID_MODELS)[0]

    with st.expander("Try an example", expanded="text" not in st.session_state):
        for variant, items in EXAMPLES.items():
            st.caption(variant)
            cols = st.columns(len(items))
            for col, (kind, ex) in zip(cols, items):
                col.button(ex, key=f"ex_{ex}", on_click=set_text, args=(ex,), use_container_width=True,
                           help=kind)

    st.session_state.setdefault("text", EXAMPLES["Tanglish"][0][1])
    text = st.text_area("Your sentence", key="text", height=90, max_chars=1000,
                        placeholder="e.g. naan inniku college ku late ah varen   (Ctrl/⌘+Enter to analyze)")

    if not text.strip():
        st.info("Type a sentence above, or open **Try an example**. Romanized Tamil and Hindi mixed with English "
                "work best; informal spelling is fine.")
        return

    pipes = {v: get_pipeline(v) for v in ("tanglish", "hinglish")}
    if choice == "Auto-detect":
        variant, scores = detect_variant(text, pipes)
        st.caption(f"🔎 Detected **{variant.capitalize()}** (share of words unknown to each vocabulary: "
                   f"Tanglish {scores['tanglish']:.0%} · Hinglish {scores['hinglish']:.0%}). "
                   f"Override it with the selector above.")
    else:
        variant = choice.lower()
    pipe = pipes[variant]
    lid_key = LID_MODELS[lid_name]
    if lid_key == "bilstm" and not pipe.available()["lid_bilstm"]:
        lid_key = "char_ngram"

    try:
        a = pipe.analyze(text, lid_key, k=5)
    except ValueError as e:
        st.warning(str(e))
        return
    remember(a.text, variant)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='card' style='padding:.55rem .9rem;margin:0'><div class='muted'>Sentence</div>"
                f"<div style='font-size:1.1rem;font-weight:600;line-height:1.35;padding-top:.15rem'>"
                f"{html.escape(a.summary.replace(' (Romanized)', ''))}</div></div>", unsafe_allow_html=True)
    m2.metric("Code-mixing index", f"{a.code_mixing_index:.0f}",
              help="0 = a single language; higher = more evenly mixed (Das & Gambäck CMI).")
    changed = sum(i.changed for i in a.normalized.tokens)
    m3.metric("Spelling fixes", changed, help="Words or punctuation runs the normalizer changed.")
    uncertain = sum(c < LOW_CONF and l in ("TA", "HI", "EN", "OTHER") for l, c in zip(a.labels, a.confidences))
    m4.metric("Uncertain words", uncertain, help=f"Words tagged with confidence below {LOW_CONF:.0%}.")

    t_tags, t_norm, t_next, t_gen, t_flu = st.tabs(
        ["🏷️ Language tags", "✨ Normalized", "🔮 Next word", "✍️ Continue writing", "📈 Fluency"])

    with t_tags:
        st.markdown(chips(a.tokens, a.labels, a.confidences), unsafe_allow_html=True)
        st.markdown(legend(), unsafe_allow_html=True)
        st.caption("Hover a word for its confidence. Numbers, punctuation, links and emoji are tagged by rules; "
                   "words by the selected model.")
        with st.expander("Token table"):
            st.dataframe(pd.DataFrame({"token": a.tokens, "language": a.labels,
                                       "confidence": [round(c, 3) for c in a.confidences]}),
                         hide_index=True, use_container_width=True)

    with t_norm:
        st.markdown(f"<div class='card'>{diff_html(a.normalized)}</div>", unsafe_allow_html=True)
        st.markdown("**Result**")
        st.code(a.normalized.text, language=None)
        if changed:
            with st.expander("What changed and why"):
                rules = {"elongation": "stretched letters shortened", "punct_run": "repeated punctuation collapsed",
                         "emoji_run": "repeated emoji collapsed", "variant": "spelling variant → preferred spelling",
                         "slang": "chat abbreviation expanded", "learned": "spelling learned from annotator corrections"}
                st.dataframe(pd.DataFrame([{"before": i.original, "after": i.normalized,
                                            "rule": rules.get(i.rule, i.rule)}
                                           for i in a.normalized.tokens if i.changed]),
                             hide_index=True, use_container_width=True)
        else:
            st.success("Already in the preferred spelling. Nothing to change.")
        st.caption("Normalization cleans spelling only. It does not translate or transliterate.")

    with t_next:
        if a.oov_words:
            st.caption(f"⚠️ Words the models haven't seen (treated as unknown): {', '.join(sorted(set(a.oov_words)))}")
        left, right = st.columns(2)
        ng_words, ls_words = {w for w, _ in a.ngram}, {w for w, _ in a.lstm}
        with left:
            st.markdown("**Trigram** · counts + Kneser-Ney smoothing")
            st.markdown(suggestion_bars(a.ngram, ls_words), unsafe_allow_html=True)
        with right:
            st.markdown("**LSTM** · neural network")
            st.markdown(suggestion_bars(a.lstm, ng_words), unsafe_allow_html=True)
        st.markdown("**Add a word to your sentence**")
        for title, preds, key in (("Trigram", a.ngram, "ng"), ("LSTM", a.lstm, "ls")):
            if preds:
                cols = st.columns([1.2] + [1] * len(preds))
                cols[0].markdown(f"<div class='muted' style='padding-top:.45rem'>{title}</div>", unsafe_allow_html=True)
                for i, (w, p) in enumerate(preds):
                    cols[i + 1].button(w, key=f"{key}_{i}_{w}", on_click=append_word, args=(w,),
                                       help=f"{p:.1%} · click to append", use_container_width=True)
        st.caption("Percentages are each model's probability over the whole vocabulary; they are small because many words are plausible.")

    with t_gen:
        s1, s2, s3 = st.columns(3)
        n_words = s1.slider("Words to add", 3, 20, 8)
        top_k = s2.slider("Top-k", 3, 30, 10)
        temp = s3.slider("Temperature", 0.3, 1.5, 0.8, 0.1)
        st.caption(SETTINGS_HELP)
        seed = st.session_state.setdefault("gen_seed", 1)
        if st.button("🎲 Regenerate", key="regen"):
            st.session_state["gen_seed"] = seed = seed + 1
        g1, g2 = st.columns(2)
        for col, title, key in ((g1, "Trigram", "ngram"), (g2, "LSTM", "lstm")):
            with col:
                st.markdown(f"**{title}**")
                words = pipe.continue_text(a.text, key, n_words, top_k, temp, seed=seed)
                if words:
                    st.markdown(f"<div class='card gen'>{html.escape(a.text)} "
                                f"{''.join(f'<b>{html.escape(w)}</b> ' for w in words)}</div>", unsafe_allow_html=True)
                    st.button("Use this", key=f"use_{key}", on_click=set_text,
                              args=(a.text + " " + " ".join(words),))
                else:
                    st.caption("Model not available.")
        st.caption("Sampled text is a demo of what each model has learned from social-media comments, not advice on what to write.")

    with t_flu:
        if not a.fluency:
            st.info("Language models are not available.")
        else:
            cols = st.columns(len(a.fluency))
            names = {"ngram": "Trigram", "lstm": "LSTM"}
            for col, (k, f) in zip(cols, a.fluency.items()):
                col.metric(f"{names[k]} perplexity", f"{f['perplexity']:.0f}",
                           help="How surprised the model is by the whole (normalized) sentence. Lower = more typical.")
            for k, f in a.fluency.items():
                st.markdown(f"**{names[k]}: surprise per word**")
                st.markdown(surprise_html(f["words"]), unsafe_allow_html=True)
            st.caption("Darker red = the model found that word less expected given the words before it. "
                       "This measures how typical the wording is for the training text, not grammar or correctness.")

    st.download_button("⬇️ Download this analysis (JSON)", data=json.dumps({
        "input": a.text, "variant": variant, "language_id_model": lid_name,
        "tokens": [{"token": t, "language": l, "confidence": round(c, 3)}
                   for t, l, c in zip(a.tokens, a.labels, a.confidences)],
        "normalized": a.normalized.text, "summary": a.summary, "code_mixing_index": a.code_mixing_index,
        "next_word": {"trigram": a.ngram, "lstm": a.lstm},
        "perplexity": {k: f["perplexity"] for k, f in a.fluency.items()}}, ensure_ascii=False, indent=2),
        file_name="analysis.json", mime="application/json")


# ----------------------------------------------------------------------------- batch page
def page_batch():
    st.markdown("<div class='hero'><h1>📚 Batch analysis</h1><p>Paste many messages or upload a file. "
                "Get tags, normalized text and next-word guesses for each, ready to download.</p></div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    choice = c1.segmented_control("Language mix", ["Auto-detect", "Tanglish", "Hinglish"], default="Auto-detect",
                                  key="batch_variant") or "Auto-detect"
    lid_name = c2.segmented_control("Language-ID model", list(LID_MODELS), default=list(LID_MODELS)[0],
                                    key="batch_lid") or list(LID_MODELS)[0]
    up = st.file_uploader("Upload .txt (one message per line) or .csv (first column)", type=["txt", "csv"])
    pasted = st.text_area("…or paste messages, one per line", height=140,
                          placeholder="naan inniku college ku late ah varen\nkal meeting haiii yaar")
    lines: list[str] = []
    if up is not None:
        raw = up.getvalue().decode("utf-8", errors="ignore")
        lines = (pd.read_csv(io.StringIO(raw)).iloc[:, 0].astype(str).tolist() if up.name.endswith(".csv")
                 else raw.splitlines())
    else:
        lines = pasted.splitlines()
    lines = [l.strip() for l in lines if l.strip()]
    if not lines:
        st.info("Add some messages to begin. Example: paste the two lines shown in the box above.")
        return
    if len(lines) > 500:
        st.warning(f"Only the first 500 of {len(lines)} lines are analyzed.")
        lines = lines[:500]

    pipes = {v: get_pipeline(v) for v in ("tanglish", "hinglish")}
    rows = []
    with st.spinner(f"Analyzing {len(lines)} messages…"):
        for line in lines:
            v = detect_variant(line, pipes)[0] if choice == "Auto-detect" else choice.lower()
            lid_key = LID_MODELS[lid_name] if pipes[v].available()["lid_bilstm"] else "char_ngram"
            r = pipes[v].analyze_batch([line], lid_key)
            if r:
                rows.append({"variant": v, **r[0]})
    if not rows:
        st.warning("No analyzable text found.")
        return
    df = pd.DataFrame(rows)
    k1, k2, k3 = st.columns(3)
    k1.metric("Messages", len(df))
    k2.metric("Code-mixed", pct((df["type"].str.contains("code-mixed")).mean()))
    k3.metric("Average code-mixing index", f"{df['cmi'].mean():.1f}")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="code_mixed_analysis.csv", mime="text/csv")


# ----------------------------------------------------------------------------- evaluation page
def page_eval():
    st.markdown("<div class='hero'><h1>📊 Evaluation</h1><p>Measured on held-out data and read from "
                "<code>results/*.json</code>. These describe the datasets used here, not accuracy in general.</p></div>",
                unsafe_allow_html=True)
    lm = {v: load_results(f"lm_{v}.json") for v in ("hinglish", "tanglish")}
    lid = {v: load_results(f"lid_{v}.json") for v in ("hinglish", "tanglish")}
    norm = load_results("normalization.json")

    if all(lm.values()) and all(lid.values()) and norm:
        st.subheader("Headline results")
        h = st.columns(4)
        hb = lid["hinglish"]["models"]["bilstm"]["results"]["test"]["word_level"]["accuracy"]
        tg = lid["tanglish"]["models"]["char_ngram"]["results"]["hand_reviewed_gold"]["word_level"]["accuracy"]
        nb, nd = (norm["hinglish"]["systems"][k]["best_of_refs"]["wer"]
                  for k in ("no_normalization (baseline)", "rules + learned dictionary"))
        h[0].metric("Hinglish language ID", pct(hb), help="BiLSTM, word accuracy on COMI-LINGUA test (gold).")
        h[1].metric("Tanglish language ID", pct(tg), help="Hand-reviewed gold set, 120 sentences.")
        h[2].metric("Normalization WER (HI)", pct(nd), delta=f"{100 * (nd - nb):.1f} pts vs raw",
                    delta_color="inverse", help="Word error rate against the best of 3 annotator references.")
        d = lm["hinglish"]["lstm"]["test_perplexity"] / lm["hinglish"]["ngram"]["test_perplexity"] - 1
        h[3].metric("LSTM perplexity (HI)", f"{lm['hinglish']['lstm']['test_perplexity']:.0f}",
                    delta=f"{100 * d:.0f}% vs trigram", delta_color="inverse", help="Lower is better.")

    tabs = st.tabs(["Language ID", "Normalization", "Language models", "Limitations"])

    with tabs[0]:
        for v in ("hinglish", "tanglish"):
            r = lid[v]
            if not r:
                st.warning(f"No LID results for {v} yet.")
                continue
            st.subheader(v.capitalize())
            st.caption(f"Training labels: {r['label_source']}")
            rows = []
            for key, m in r["models"].items():
                for setname, res in m["results"].items():
                    w, a = res["word_level"], res.get("all_tokens")
                    rows.append({"model": m["name"], "evaluated on": setname.replace("_", " "),
                                 "word accuracy": pct(w["accuracy"]), "word macro-F1": f"{w['macro_f1']:.3f}",
                                 "all-token accuracy": pct(a["accuracy"]) if a else "–", "tokens": w["n_tokens"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            k1, k2, k3 = st.columns([1.2, 1.2, 1])
            choice = k1.selectbox("Model", list(r["models"]), key=f"cls_{v}", format_func=lambda k: r["models"][k]["name"])
            sets = list(r["models"][choice]["results"])
            default = "hand_reviewed_gold" if "hand_reviewed_gold" in sets else "test"
            sset = k2.selectbox("Data split", sets, index=sets.index(default), key=f"set_{v}")
            res = r["models"][choice]["results"][sset]["word_level"]
            c1, c2 = st.columns(2)
            c1.markdown("**Per-class scores**")
            c1.dataframe(pd.DataFrame(res["per_class"]).T.round(3), use_container_width=True)
            cm = res["confusion_matrix"]
            c2.markdown("**Confusion matrix** (rows = truth)")
            c2.dataframe(pd.DataFrame(cm["matrix"], index=[f"true {l}" for l in cm["labels"]],
                                      columns=[f"pred {l}" for l in cm["labels"]])
                         .style.background_gradient(cmap="Blues", axis=None), use_container_width=True)
            if r.get("cross_variant"):
                cv = r["cross_variant"]
                st.markdown(f"**Transfer check** on {cv.get('evaluated_on')} (English vs native language only)")
                st.dataframe(pd.DataFrame({k.replace("_", " "): {"accuracy": pct(x["accuracy"]),
                                                                 "macro-F1": f"{x['macro_f1']:.3f}"}
                                           for k, x in cv.items() if isinstance(x, dict)}).T,
                             use_container_width=True)
            st.divider()

    with tabs[1]:
        if not norm:
            st.warning("No normalization results yet.")
        else:
            for v, blk in norm.items():
                st.subheader(v.capitalize())
                st.caption(blk.get("reference", ""))
                if "systems" not in blk:
                    continue
                rows = [{"system": name, "scored against": lab, "WER": pct(s[mode]["wer"]),
                         "CER": pct(s[mode]["cer"]), "exact sentence match": pct(s[mode]["exact_match"])}
                        for name, s in blk["systems"].items()
                        for mode, lab in (("best_of_refs", "best of references"), ("first_ref", "first reference"))]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                ia = blk.get("inter_annotator")
                if ia:
                    st.info(f"**Human yardstick:** the three annotators disagree with each other at a mean WER of "
                            f"{pct(ia['mean_wer'])} (CER {pct(ia['mean_cer'])}), so there is no single correct spelling. "
                            f"Compare with the *first reference* rows; *best of references* is more lenient.")
                if blk.get("metric_note"):
                    st.caption(blk["metric_note"])
                with st.expander("Human-review sample (original → system vs reference)"):
                    st.dataframe(pd.DataFrame(blk["review_sample"]), hide_index=True, use_container_width=True)
                ea = blk.get("error_analysis")
                if ea:
                    with st.expander("Most frequent remaining errors"):
                        a, b = st.columns(2)
                        a.markdown("**Corrections the system missed**")
                        a.dataframe(pd.DataFrame(ea["missed_corrections"], columns=["raw", "reference", "count"]),
                                    hide_index=True)
                        b.markdown("**Wrong or unwanted rewrites**")
                        b.dataframe(pd.DataFrame(ea["wrong_corrections"], columns=["raw", "reference", "system", "count"]),
                                    hide_index=True)

    with tabs[2]:
        for v in ("tanglish", "hinglish"):
            r = lm[v]
            if not r:
                st.warning(f"No language-model results for {v} yet.")
                continue
            st.subheader(v.capitalize())
            ng, ls = r["ngram"], r["lstm"]
            st.caption(f"{r['train_sentences']:,} train / {r['val_sentences']:,} val / {r['test_sentences']:,} test "
                       f"sentences · vocabulary {r['vocab_size']:,} · test OOV rate {pct(r['test_oov_rate'])}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test perplexity: trigram", f"{ng['test_perplexity']:.0f}")
            c2.metric("Test perplexity: LSTM", f"{ls['test_perplexity']:.0f}",
                      delta=f"{100 * (ls['test_perplexity'] / ng['test_perplexity'] - 1):.0f}%", delta_color="inverse")
            c3.metric("Top-5 accuracy: LSTM", pct(ls["top5"]), delta=f"{100 * (ls['top5'] - ng['top5']):+.1f} pts")
            cmp = pd.DataFrame({"Trigram": {"top-1 accuracy": ng["top1"], "top-5 accuracy": ng["top5"]},
                                "LSTM": {"top-1 accuracy": ls["top1"], "top-5 accuracy": ls["top5"]}})
            d1, d2 = st.columns(2)
            d1.markdown("**Next-word accuracy**")
            d1.bar_chart(cmp, height=230)
            hist = pd.DataFrame(ls["history"]).set_index("epoch")[["train_ppl", "val_ppl"]]
            d2.markdown("**LSTM perplexity per epoch**")
            d2.line_chart(hist, height=230)
            with st.expander("Side-by-side predictions for fixed prompts"):
                st.dataframe(pd.DataFrame([{"prompt": e["prompt"], "trigram top-5": ", ".join(w for w, _ in e["ngram"]),
                                            "LSTM top-5": ", ".join(w for w, _ in e["lstm"])} for e in r["examples"]]),
                             hide_index=True, use_container_width=True)
            st.divider()
        st.caption("Perplexity is computed over every token including `</s>` and `<unk>`, with one shared vocabulary, "
                   "so the two models are directly comparable (lower is better). Accuracy counts only positions whose "
                   "true next word is a real vocabulary word.")

    with tabs[3]:
        st.markdown(LIMITATIONS)


LIMITATIONS = """
**What these numbers do and do not show**

* **Hinglish** language ID and normalization use the *gold* COMI-LINGUA annotations (3 annotators; token labels are the
  majority vote). Annotators agree on roughly 89% of tokens, a practical ceiling.
* **Tanglish** has no public token-level gold data. Training labels are *silver*, produced by a curated Tamil lexicon plus
  English word frequencies, so the silver test score overstates real accuracy. The small *hand-reviewed gold* set is the
  honest check; it was reviewed by an AI assistant, not independent native speakers, and should be extended and re-checked by people.
* **Tanglish normalization** has no parallel corpus. Its score comes from ~70 hand-written pairs that follow the dictionary's
  own convention: a sanity check, not a benchmark.
* **Language models** are trained on noisy social-media text. Perplexity says how well a model predicts that kind of text, not
  whether generated sentences are meaningful. Tanglish training text was normalized by this project's normalizer; Hinglish uses
  the annotators' normalized sentences. Tanglish vocabulary is capped at 20,000 words.
* **Splits:** Tanglish is split by source after de-duplication; Hinglish keeps the official split. Near-duplicate tweets can
  remain in Hinglish, which flatters all models slightly.
* A model trained on one language mix should not be assumed to work on the other (see the transfer check).
"""


# ----------------------------------------------------------------------------- about page
def page_about():
    st.markdown("<div class='hero'><h1>ℹ️ About</h1><p>How the pieces fit together.</p></div>", unsafe_allow_html=True)
    st.markdown("""
**What it does.** People often type Tamil or Hindi in English letters and mix in English words. This app handles that:

1. **Language ID**: tags every word (Tamil/Hindi, English, name/other); numbers, punctuation, links and emoji by rules.
2. **Normalization**: cleans spelling (`plsss` → `please`, `rombbba` → `romba`) without translating.
3. **Language models**: a trigram (Kneser-Ney) and an LSTM predict the next word and rate how typical a sentence is.

**Pipeline.** input → tokenize → language ID → normalize → next-word models. The app, the command line and the evaluation
scripts all call the same `CodeMixedPipeline`, so the demo cannot drift from the experiments.

**Tag convention.** `TA`/`HI` Romanized Tamil/Hindi · `EN` English incl. chat slang and loanwords (`bro`, `late`) ·
`OTHER` names, acronyms, handles, URLs, emoji, noise · `NUM` · `PUNCT`.

**Data.** COMI-LINGUA (Hinglish, CC-BY-4.0) and Tanglish-Corpus-185k (CC-BY-4.0). See the README for sizes and splits.

**Privacy.** Everything runs locally; text you enter is not sent anywhere.
""")
    p = Path(__file__).parent / "data" / "processed" / "split_report.json"
    if p.exists():
        with st.expander("Dataset split report"):
            st.json(json.loads(p.read_text()))


# ----------------------------------------------------------------------------- navigation
PAGES = {"🔍 Analyze": page_analyze, "📚 Batch": page_batch, "📊 Evaluation": page_eval, "ℹ️ About": page_about}
with st.sidebar:
    st.markdown("### Code-Mixed NLP")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    hist = st.session_state.get("history", [])
    if hist and page == "🔍 Analyze":
        st.markdown("**Recent**")
        for i, (t, v) in enumerate(hist):
            st.button(t if len(t) < 34 else t[:32] + "…", key=f"h{i}", on_click=set_text, args=(t,),
                      use_container_width=True, help=v.capitalize())
    st.caption("Tanglish = Tamil + English · Hinglish = Hindi + English")
PAGES[page]()
