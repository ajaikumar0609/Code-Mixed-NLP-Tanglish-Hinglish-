"""Builds the 11-slide 25AI2008 skill-based evaluation review deck (PPTX) from the college template geometry."""
import copy
import json
import re
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Pt, Emu

ROOT = Path("/Users/ajaikumarn/Desktop/shally")
A = ROOT / "deliverables/build/assets"
OUT = ROOT / "deliverables/25AI2008_REGNO_Review.pptx"

NAVY, RED, TEAL, GREY, INK, PANEL, ORANGE = "14284B", "D7263D", "2A9D8F", "5C677D", "1F2937", "EEF2F8", "F4A261"
HEAD, BODY, TBL = "Cambria", "Calibri", "Arial"

# ------------------------------------------------------------------ results (read from saved files)
RES = ROOT / "results"
lh, lt = json.load(open(RES / "lid_hinglish.json")), json.load(open(RES / "lid_tanglish.json"))
mh, mt = json.load(open(RES / "lm_hinglish.json")), json.load(open(RES / "lm_tanglish.json"))
nm = json.load(open(RES / "normalization.json"))
acc = lambda d: d["accuracy"] * 100
hi_a, hi_b = acc(lh["models"]["char_ngram"]["results"]["test"]["word_level"]), acc(lh["models"]["bilstm"]["results"]["test"]["word_level"])
ta_a, ta_b = acc(lt["models"]["char_ngram"]["results"]["hand_reviewed_gold"]["word_level"]), acc(lt["models"]["bilstm"]["results"]["hand_reviewed_gold"]["word_level"])
NS = nm["hinglish"]["systems"]
wer0, wer1 = NS["no_normalization (baseline)"]["best_of_refs"]["wer"] * 100, NS["rules + learned dictionary"]["best_of_refs"]["wer"] * 100
wer_ia = nm["hinglish"]["inter_annotator"]["mean_wer"] * 100
split = json.load(open(ROOT / "data/processed/split_report.json"))
tl, hl = split["lm_tanglish"], split["lm_hinglish"]
tot_t, tot_h = sum(tl.values()), sum(hl.values())
pct = lambda d, tot: " / ".join(f"{d[k] / tot * 100:.0f}" for k in ("train", "val", "test"))

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)
BLANK = prs.slide_layouts[6]
pt = lambda v: Emu(int(round(v * 12700)))
rgb = lambda h: RGBColor.from_string(h)


def tb(slide, x, y, w, h, paras, size=14, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=None, space_after=0, italic=False):
    """paras: str | list of str | list of list of (text, {opts}) runs."""
    s = slide.shapes.add_textbox(pt(x), pt(y), pt(w), pt(h))
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if line: para.line_spacing = line
        para.space_after = Pt(space_after)
        runs = p if isinstance(p, list) else [(p, {})]
        for text, o in runs:
            r = para.add_run(); r.text = text
            f = r.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold); f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
            if o.get("hl"):
                rPr = r._r.get_or_add_rPr()
                h = etree.SubElement(rPr, qn("a:highlight")); c = etree.SubElement(h, qn("a:srgbClr")); c.set("val", "FFFF00")
                # a:highlight must follow solidFill and precede latin
                lat = rPr.find(qn("a:latin"))
                if lat is not None:
                    rPr.remove(h); lat.addprevious(h)
    return s


def bullets(slide, x, y, w, items, size=14, step=None, gap=4, color=INK, bullet_w=18):
    """Bullet list as one text box with hanging indent (editable)."""
    s = slide.shapes.add_textbox(pt(x), pt(y), pt(w), pt(10))
    tf = s.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(gap)
        pPr = para._p.get_or_add_pPr()
        pPr.set("marL", str(int(bullet_w * 12700))); pPr.set("indent", str(-int(bullet_w * 12700)))
        bu = etree.SubElement(pPr, qn("a:buChar")); bu.set("char", "•")
        runs = it if isinstance(it, list) else [(it, {})]
        for text, o in runs:
            r = para.add_run(); r.text = text
            r.font.name = BODY; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", False)
            r.font.color.rgb = rgb(o.get("color", color))
            if o.get("hl"):
                rPr = r._r.get_or_add_rPr()
                h = etree.SubElement(rPr, qn("a:highlight")); c = etree.SubElement(h, qn("a:srgbClr")); c.set("val", "FFFF00")
                lat = rPr.find(qn("a:latin"))
                if lat is not None:
                    rPr.remove(h); lat.addprevious(h)
    return s


def rect(slide, x, y, w, h, fill=PANEL, line=None, rounded=True, dash=False, radius=0.05, lw=1.0):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, pt(x), pt(y), pt(w), pt(h))
    if rounded:
        shp.adjustments[0] = radius
    if fill: shp.fill.solid(); shp.fill.fore_color.rgb = rgb(fill)
    else: shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line); shp.line.width = Pt(lw)
        if dash:
            ln = shp.line._get_or_add_ln(); d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.margin_left = shp.text_frame.margin_right = Emu(36000)
    return shp


def label_box(slide, x, y, w, h, text, fill="FFFFFF", line=NAVY, color=INK, size=10.5, bold=False, lw=1.0):
    shp = rect(slide, x, y, w, h, fill, line, True, False, 0.12, lw)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(30000); tf.margin_top = tf.margin_bottom = Emu(12000)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = text.split("\n")
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = t
        r.font.name = BODY; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(color)
    return shp


def arrow(slide, x1, y1, x2, y2, color=GREY, w=1.5):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, pt(x1), pt(y1), pt(x2), pt(y2))
    c.line.color.rgb = rgb(color); c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c


def circle(slide, x, y, d, n, fill=RED, size=16):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, pt(x), pt(y), pt(d), pt(d))
    c.fill.solid(); c.fill.fore_color.rgb = rgb(fill); c.line.fill.background(); c.shadow.inherit = False
    tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = str(n); r.font.name = BODY; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb("FFFFFF")
    return c


def base(title, n):
    s = prs.slides.add_slide(BLANK)
    tb(s, 43.2, 43, 780, 44, title, 34, NAVY, HEAD, True)
    s.shapes.add_picture(str(A / "ppt_logo.png"), pt(864), pt(29), pt(52), pt(49))
    tb(s, 43.2, 511, 400, 14, "25AI2008  |  Multilingual AI  |  Skill Based Evaluation", 10, GREY)
    tb(s, 856, 509, 52, 14, str(n), 10, GREY, align=PP_ALIGN.RIGHT)
    return s


def pic_fit(slide, path, x, y, w, h):
    from PIL import Image
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    return slide.shapes.add_picture(str(path), pt(x + (w - pw) / 2), pt(y + (h - ph) / 2), pt(pw), pt(ph))


H = lambda t: [(t, {"hl": True})]

# =================================================================== Slide 1: title
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(str(A / "ppt_banner.png"), pt(40), pt(26), pt(392), pt(65))
tb(s, 43.2, 140, 500, 22, "SKILL BASED EVALUATION REVIEW", 16, RED, BODY, True)
tb(s, 43.2, 178, 873, 100, "Code-Mixed (Tanglish/Hinglish) Text: Language Identification, Normalization and N-gram/LSTM Language Modelling", 29, NAVY, HEAD, True, line=0.95)
tb(s, 43.2, 293, 500, 30, "25AI2008  Multilingual AI", 22, INK, BODY)
rect(s, 43.2, 352.8, 873.4, 122.4, PANEL)
tb(s, 65, 364, 200, 16, "Team Members", 12, GREY, BODY, True)
tb(s, 65, 381, 540, 20, "Angel Sweety P (5051)   |   Shally S (5078)", 15, INK)
tb(s, 65, 401, 540, 20, "K Kokila Dharshini (5037)   |   Selvarani S (5088)", 15, INK)
tb(s, 65, 428, 200, 16, "Course Faculty", 12, GREY, BODY, True)
tb(s, 65, 445, 540, 20, [H("<<Faculty Name, Designation>>")], 15, INK)
tb(s, 560, 374, 345, 60, ["Division of Artificial Intelligence and Machine Learning", "School of Computer Science and Technology", "October 2026"],
   13, GREY, align=PP_ALIGN.RIGHT, space_after=2)

# =================================================================== Slide 2: problem statement and objectives
s = base("Problem Statement & Objectives", 2)
rect(s, 43.2, 108, 403.2, 374.4, PANEL)
tb(s, 65, 124, 360, 28, "Problem Statement", 20, NAVY, HEAD, True)
tb(s, 65, 160, 362, 310, [
    [("What: ", {"bold": True}), ("Tanglish (Tamil-English) and Hinglish (Hindi-English) are typed in Roman letters with English words mixed in. NLP tools built for one language and one script fail on this text.", {})],
    [("Why it is hard: ", {"bold": True}), ("no standard spelling (iruku / irukku, hai / haiii), ambiguous words and language boundaries, names and acronyms, very few gold-annotated corpora and many unseen word combinations.", {})],
    [("Who benefits: ", {"bold": True}), ("search, moderation, keyboard / chat prediction and analytics for users of Romanized Indian languages.", {})],
], 16, INK, space_after=10)
tb(s, 482.4, 115, 300, 28, "Objectives", 20, NAVY, HEAD, True)
objs = [("Token-level language identification", "Success: word accuracy and macro-F1 on held-out gold data; char n-gram + LogReg vs BiLSTM"),
        ("Consistent code-mixed normalization", "Success: WER vs three human references (Hinglish); spelling only, no translation"),
        ("Statistical vs neural language modelling", "Success: perplexity and top-1 / top-5 next-word accuracy, same corpus and test positions")]
for i, (t, sub) in enumerate(objs):
    y = 158.4 + i * 108
    circle(s, 482.4, y, 43.2, i + 1)
    tb(s, 540, y + 15, 376, 22, t, 16, INK, BODY, True)
    tb(s, 540, y + 49, 376, 44, sub, 14, GREY)

# =================================================================== Slide 3: literature survey
s = base("Literature Survey", 3)
cols = [86.4, 216.0, 302.4, 268.6]
rows = [("Ref", "Authors & Year", "Method / Model", "Gap / Limitation"),
        ("[1]", ("Sheth, Beniwal & Singh, 2025", "Findings of EMNLP 2025"), "COMI-LINGUA: expert-annotated Hindi-English code-mixed dataset (3 annotators per instance); 5 tasks incl. token-level LID and benchmarks with fine-tuned LLMs",
         "Hindi-English only: no Tamil-English counterpart; no lightweight from-scratch baselines"),
        ("[2]", ("Barman, Das, Wagner & Foster, 2014", "CALCS workshop, pp. 13-23"), "Word-level language identification on Bengali-English-Hindi Facebook comments; preliminary experiments",
         "Work in progress; no Tamil; LID only (no normalization or language model)"),
        ("[3]", ("Han & Baldwin, 2011", "ACL 2011, pp. 368-378"), "Lexical normalisation of ill-formed out-of-vocabulary words in Twitter / SMS text (dictionary lookup combined with other methods)",
         "Targets English short text; not designed for Romanized Tamil / Hindi mixed with English"),
        ("[4]", ("Pratapa et al., 2018", "ACL 2018, pp. 1543-1553"), "RNN language model for code-mixing trained with synthetic data from Equivalence Constraint Theory",
         "Focus on synthetic-data curriculum for perplexity; no LID or normalization of noisy Romanized text")]
tbl = s.shapes.add_table(5, 4, pt(43.2), pt(115.2), pt(sum(cols)), pt(39.6 + 4 * 61.2)).table
tblPr = tbl._tbl.tblPr
for a in ("firstRow", "bandRow"): tblPr.set(a, "0")
sid = tblPr.find(qn("a:tableStyleId"))
if sid is not None: tblPr.remove(sid)
for j, w in enumerate(cols): tbl.columns[j].width = pt(w)
tbl.rows[0].height = pt(39.6)
for i in range(1, 5): tbl.rows[i].height = pt(61.2)


def cell_border(cell, color=PANEL, w=1.0):
    tcPr = cell._tc.get_or_add_tcPr()
    for side in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(side)); ln.set("w", str(int(w * 12700)))
        sf = etree.SubElement(ln, qn("a:solidFill")); c = etree.SubElement(sf, qn("a:srgbClr")); c.set("val", color)


for i, row in enumerate(rows):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.margin_left = c.margin_right = pt(8); c.margin_top = c.margin_bottom = pt(4)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = c.text_frame; tf.word_wrap = True
        items = val if isinstance(val, tuple) else (val,)
        for k, t in enumerate(items):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            r = p.add_run(); r.text = t
            r.font.name = TBL
            if i == 0:
                r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = rgb("FFFFFF")
            else:
                r.font.size = Pt(11 if k == 0 else 9); r.font.color.rgb = rgb(INK if k == 0 else GREY)
                r.font.bold = (j == 1 and k == 0)
        c.fill.solid(); c.fill.fore_color.rgb = rgb(NAVY if i == 0 else (PANEL if i % 2 == 0 else "FFFFFF"))
        cell_border(c)
rect(s, 43.2, 421.2, 873.4, 61.2, PANEL)
tb(s, 65, 429, 830, 48, [[("Research gap:  ", {"bold": True, "color": RED, "size": 15}),
                          ("The surveyed work handles one task or one language pair at a time. This project covers Tanglish and Hinglish in one pipeline (language ID, normalization, language models) and compares N-gram vs LSTM under one protocol, with explicit gold / silver label provenance.", {"size": 13})]], 13, INK)

# =================================================================== Slide 4: methodology
s = base("Proposed Methodology", 4)
steps = [("Data Collection", "COMI-LINGUA (Hinglish) and Tanglish-Corpus-185k, both CC-BY-4.0"),
         ("Preprocessing", "NFKC clean-up, typed tokenizer, de-duplication, split by source"),
         ("Tokenization &\nEmbeddings", "Char n-grams + TF-IDF; word and char-CNN embeddings; LM vocabulary"),
         ("Model Training", "LogReg and BiLSTM taggers; KN trigram; LSTM LM, all from scratch"),
         ("Evaluation", "Accuracy / macro-F1, WER, perplexity, top-1 / top-5 on held-out data")]
xs = [43.2, 221.8, 400.3, 578.9, 757.4]
for i, ((t, cap), x) in enumerate(zip(steps, xs)):
    rect(s, x, 129.6, 158.4, 208.8, PANEL)
    circle(s, x + 14.4, 144, 43.2, i + 1, TEAL if i == 4 else RED)
    tb(s, x + 14.4, 205, 135, 44, t.split("\n"), 18, NAVY, HEAD, True)
    tb(s, x + 14.4, 258, 132, 76, cap, 12.5, GREY)
    if i < 4:
        tri = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, pt(x + 159.8), pt(223.2), pt(17.3), pt(21.6))
        tri.fill.solid(); tri.fill.fore_color.rgb = rgb(GREY); tri.line.fill.background(); tri.shadow.inherit = False
tb(s, 43.2, 354, 400, 28, "Workflow Summary", 20, NAVY, HEAD, True)
tb(s, 43.2, 388, 873, 100, [
    "Romanized Tanglish / Hinglish sentence → typed tokenizer → token-level language ID (rules for numbers, punctuation, URLs, emoji + char n-gram LogReg or BiLSTM) → normalizer (rules + language-aware dictionaries) → LM tokens → trigram-KN and LSTM next-word suggestions and perplexity.",
    [("One pipeline class serves the Streamlit app, CLI and evaluation.  Module owners: ", {}), ("<<to be filled by the team>>", {"hl": True})]],
   14, INK, space_after=6)

# =================================================================== Slide 5: dataset
s = base("Dataset & Languages", 5)
cards = [("2", "Languages covered: Tanglish (Tamil-English) and Hinglish (Hindi-English)"),
         (f"{tot_t + tot_h:,}", f"Sentences in the LM corpora: {tot_t:,} Tanglish + {tot_h:,} Hinglish"),
         (pct(tl, tot_t), f"Train / Val / Test (%), Tanglish LM corpus. Hinglish LM corpus: {pct(hl, tot_h)}")]
for (big, lab), x in zip(cards, (43.2, 338.4, 633.6)):
    rect(s, x, 115.2, 283, 165.6, PANEL)
    tb(s, x + 21.6, 138, 245, 52, big, 38 if len(big) < 10 else 34, RED, HEAD, True)
    tb(s, x + 21.6, 205, 245, 66, lab, 14, INK)
tb(s, 43.2, 297, 400, 28, "Source & Preparation", 20, NAVY, HEAD, True)
bullets(s, 43.2, 332, 873, [
    [("Hinglish: ", {"bold": True}), ("COMI-LINGUA (LingoIITGN, CC-BY-4.0, huggingface.co/datasets/LingoIITGN/COMI-LINGUA); Roman-script sentences only; 3 annotators per sentence give token tags and normalized references. LID: 7,191 / 799 / 750 sentences.", {})],
    [("Tanglish: ", {"bold": True}), ("Tanglish-Corpus-185k (vishnu-n, CC-BY-4.0): 185,973 sentences from YouTube, Reddit and DravidianCodeMix; 185,670 after de-duplication; split by source (video / subreddit).", {})],
    [("Labels: ", {"bold": True}), ("Hinglish tokens are GOLD (3-annotator majority). Tanglish tokens are SILVER (lexicon rules, not human); evaluated on 120 hand-reviewed sentences, not independently validated.", {})],
    [("Conventions: ", {"bold": True}), ("TA / HI, EN, OTHER (names, acronyms, handles, noise); NUM and PUNCT by rules; numbers → <num>; native-script sentences removed.", {})],
], size=13, gap=5)

# =================================================================== Slide 6: model architecture (native shapes)
s = base("Model Architecture", 6)
rect(s, 43.2, 115.2, 518.4, 367.2, "FFFFFF", GREY, True, True, 0.02, 1.2)
LBL = dict(size=10, color=GREY)
tb(s, 52, 120, 400, 13, "1  LANGUAGE IDENTIFICATION (per word)", 10, GREY, BODY, True)
label_box(s, 52, 135, 82, 76, "Tokens\n(rules label NUM, PUNCT, URL, emoji)", PANEL, GREY, size=9.5)
label_box(s, 148, 135, 122, 34, "Char n-grams\n+ TF-IDF", "FFFFFF", TEAL, size=10)
label_box(s, 284, 135, 100, 34, "Logistic\nRegression", "FFFFFF", TEAL, size=10)
label_box(s, 148, 177, 122, 34, "Word emb. +\nchar-CNN", "FFFFFF", RED, size=10)
label_box(s, 284, 177, 100, 34, "BiLSTM\ntagger", "FFFFFF", RED, size=10)
label_box(s, 410, 135, 142, 76, "TA / HI · EN · OTHER\n(Model A or Model B)", PANEL, NAVY, size=10.5, bold=True)
for y in (152, 194):
    arrow(s, 135, (135 + 211) / 2, 147, y)
    arrow(s, 271, y, 283, y); arrow(s, 385, y, 409, y)
tb(s, 52, 223, 400, 13, "2  NORMALIZATION (rule- and dictionary-based)", 10, GREY, BODY, True)
label_box(s, 52, 239, 82, 44, "Tagged\ntokens", PANEL, GREY, size=10)
label_box(s, 148, 239, 270, 44, "Unicode clean-up · punctuation runs · elongation\nslang expansion · language-aware dictionaries", "FFFFFF", NAVY, size=10)
label_box(s, 432, 239, 120, 44, "Normalized text\n(not translation)", PANEL, NAVY, size=10, bold=True)
arrow(s, 135, 261, 147, 261); arrow(s, 419, 261, 431, 261)
tb(s, 52, 296, 480, 13, "3  LANGUAGE MODELLING (same corpus, vocabulary, test positions)", 10, GREY, BODY, True)
label_box(s, 52, 312, 92, 96, "Normalized\nLM tokens +\nshared\nvocabulary", PANEL, GREY, size=10)
label_box(s, 158, 312, 120, 40, "Trigram counts +\ncontinuation counts", "FFFFFF", TEAL, size=10)
label_box(s, 290, 312, 130, 40, "Interpolated\nKneser-Ney", "FFFFFF", TEAL, size=10)
for x, w, t in ((158, 62, "Embedding"), (228, 56, "LSTM"), (292, 56, "Linear"), (356, 64, "Softmax")):
    label_box(s, x, 368, w, 40, t, "FFFFFF", RED, size=10)
label_box(s, 440, 312, 112, 96, "Perplexity\ntop-1 / top-5\nnext word", PANEL, NAVY, size=10.5, bold=True)
arrow(s, 145, 332, 157, 332); arrow(s, 279, 332, 289, 332); arrow(s, 421, 332, 439, 332)
arrow(s, 145, 388, 157, 388); arrow(s, 221, 388, 227, 388); arrow(s, 285, 388, 291, 388); arrow(s, 349, 388, 355, 388); arrow(s, 421, 388, 439, 388)
tb(s, 158, 355, 100, 10, "N-gram (statistical)", 8, TEAL, BODY, True)
tb(s, 158, 411, 130, 10, "LSTM (neural, PyTorch)", 8, RED, BODY, True)
tb(s, 52, 432, 500, 44, "The BiLSTM tagger (lane 1) and the LSTM language model (lane 3) are different models for different tasks. Everything is trained from scratch; no pretrained or transformer model is used.",
   10.5, GREY, italic=True)
tb(s, 590, 119, 300, 26, "Key Components", 20, NAVY, HEAD, True)
bullets(s, 590, 155, 326, [
    "LID: char n-gram TF-IDF + LogReg (context-free) vs word + char-CNN + BiLSTM (sentence context)",
    "Normalization: rules + learned (Hinglish) / curated (Tanglish) dictionaries",
    "LM: KN trigram (NumPy) vs embedding → LSTM → linear → tied softmax",
    "No transformers, no pretrained weights",
], size=13, gap=5, bullet_w=16)
rect(s, 590.4, 316.8, 326.2, 165.6, PANEL)
tb(s, 612, 328, 280, 26, "Hyperparameters", 20, NAVY, HEAD, True)
tb(s, 612, 360, 290, 118, [
    "BiLSTM tagger: word emb 64, char-CNN 32-d (k = 3, 5), 128 hidden x 2, dropout 0.3, Adam 2e-3, 10 epochs",
    "LSTM LM: Tanglish 256 / 512, dropout 0.3, 10 epochs; Hinglish 200 / 400, dropout 0.4, early stop at epoch 25; Adam 2e-3, clip 1.0",
    "KN trigram: discount D = 0.9 (tuned on validation)"], 11, INK, space_after=3)

# =================================================================== Slide 7: implementation
s = base("Implementation", 7)
cards = [("Language & Libraries", "Python 3.14.3 · PyTorch 2.14.0 · scikit-learn 1.9.1 · NumPy 2.4.0 · pandas 3.0.3 · wordfreq · joblib. KN trigram model, tokenizer and metrics written from scratch (no NLTK).", RED),
         ("Models & Frameworks", "No pretrained models: all models trained from scratch. Char n-gram + LogReg, BiLSTM tagger, KN trigram, LSTM LM (tied softmax). Streamlit 1.62.0 dashboard (Analyze, Batch, Evaluation, About); pytest: 25 tests passed.", TEAL),
         ("Environment & Hardware", "Local development on an Apple M4 laptop, 16 GB, macOS; PyTorch MPS (Apple GPU) backend. Data, models and results stored locally. LSTM training: 233 s (Hinglish), 1,206 s (Tanglish).", RED),
         ("Modules Developed", None, TEAL)]
pos = [(43.2, 115.2), (487.4, 115.2), (43.2, 309.6), (487.4, 309.6)]
for i, ((t, body, col), (x, y)) in enumerate(zip(cards, pos)):
    rect(s, x, y, 429.1, 180, PANEL)
    circle(s, x + 21.6, y + 21.6, 39.6, i + 1, col)
    tb(s, x + 76, y + 31, 340, 26, t, 20, NAVY, HEAD, True)
    if body:
        tb(s, x + 21.6, y + 78, 388, 98, body, 15, INK)
    else:
        tb(s, x + 21.6, y + 78, 388, 98, [
            "M1 language ID · M2 normalization · M3 trigram LM · M4 LSTM LM · pipeline.py + Streamlit app.py · evaluation.py",
            [("Owner per module: ", {}), ("<<to be filled by the team>>", {"hl": True})]], 15, INK, space_after=6)

# =================================================================== Slide 8: results (native charts)
s = base("Results & Evaluation", 8)


def chart(slide, x, y, w, h, title, cats, series, colors, maxv, fmt='0.0', legend=True, unit=None, hide_axis=False):
    cd = CategoryChartData(); cd.categories = cats
    for n, v in series: cd.add_series(n, v)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, pt(x), pt(y), pt(w), pt(h), cd)
    ch = gf.chart
    ch.has_title = True; ch.chart_title.text_frame.text = title
    r = ch.chart_title.text_frame.paragraphs[0].runs[0]; r.font.size = Pt(11); r.font.bold = True; r.font.name = BODY; r.font.color.rgb = rgb(NAVY)
    ch.has_legend = legend and len(series) > 1
    if ch.has_legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(9); ch.legend.font.name = BODY
    va = ch.value_axis; va.maximum_scale = maxv; va.minimum_scale = 0
    if unit: va.major_unit = unit
    if hide_axis: va.visible = False
    va.has_major_gridlines = True; va.major_gridlines.format.line.color.rgb = rgb("D9DEE8")
    va.tick_labels.font.size = Pt(9); va.tick_labels.font.name = BODY; va.format.line.fill.background()
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(9); ca.tick_labels.font.name = BODY
    pl = ch.plots[0]; pl.gap_width = 60; pl.overlap = -5
    pl.has_data_labels = True
    dl = pl.data_labels; dl.font.size = Pt(9); dl.font.name = BODY; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    for ser, col in zip(pl.series, colors):
        ser.format.fill.solid(); ser.format.fill.fore_color.rgb = rgb(col)
    return ch


chart(s, 43.2, 106, 270, 172, "Language ID: word accuracy (%)", ["Hinglish\n(gold test)", "Tanglish\n(hand-reviewed gold)"],
      [("Char n-gram + LogReg", (round(hi_a, 1), round(ta_a, 1))), ("BiLSTM", (round(hi_b, 1), round(ta_b, 1)))], [GREY, RED], 125, unit=25)
chart(s, 320, 106, 270, 172, "Test perplexity (lower is better)", ["Hinglish", "Tanglish"],
      [("Trigram KN", (round(mh["ngram"]["test_perplexity"]), round(mt["ngram"]["test_perplexity"]))), ("LSTM", (round(mh["lstm"]["test_perplexity"]), round(mt["lstm"]["test_perplexity"])))], [GREY, RED], 500, '0', unit=100)
chart(s, 43.2, 278, 270, 172, "Next-word accuracy (%)", ["Hi top-1", "Hi top-5", "Ta top-1", "Ta top-5"],
      [("Trigram KN", [round(v * 100, 1) for v in (mh["ngram"]["top1"], mh["ngram"]["top5"], mt["ngram"]["top1"], mt["ngram"]["top5"])]),
       ("LSTM", [round(v * 100, 1) for v in (mh["lstm"]["top1"], mh["lstm"]["top5"], mt["lstm"]["top1"], mt["lstm"]["top5"])])], [GREY, RED], 50, unit=10)
chart(s, 320, 278, 270, 172, "Hinglish normalization: WER (%)", ["Raw text", "Rules + dict.", "Annotators differ"],
      [("WER", (round(wer0, 1), round(wer1, 1), round(wer_ia, 1)))], [TEAL], 20, unit=5)
tb(s, 43.2, 456, 548, 40, "Takeaway: the LSTM lowers perplexity most (Hinglish 193 → 142, Tanglish 416 → 259); the two language-ID models score about the same. Tanglish language ID uses the 120-sentence hand-reviewed gold set.", 12, GREY)
rect(s, 604.8, 108, 311.8, 374.4, PANEL)
tb(s, 626, 118, 260, 26, "Observations", 20, NAVY, HEAD, True)
bullets(s, 626, 152, 275, [
    f"LID: char n-gram ≈ BiLSTM ({hi_a:.1f} vs {hi_b:.1f} Hinglish; {ta_a:.1f} vs {ta_b:.1f} Tanglish gold), so sentence context adds little.",
    f"LSTM top-5 is +{(mh['lstm']['top5'] - mh['ngram']['top5']) * 100:.1f} / +{(mt['lstm']['top5'] - mt['ngram']['top5']) * 100:.1f} pts (Hi / Ta); top-1 only +1.2 / +1.3.",
    f"Normalization WER {wer0:.1f}% → {wer1:.1f}%; the 3 annotators differ by {wer_ia:.1f}%, so scoring is subjective.",
    "Metrics: accuracy, macro-F1, WER, perplexity, top-1 / top-5.",
    [("Limits: ", {"bold": True, "color": RED}), ("Tanglish gold = 120 sentences, not independently validated; 99% silver score is inflated; names / acronyms recall about 46%; Tanglish normalization 97% uses 68 self-authored pairs.", {})],
], size=12, gap=6, bullet_w=12)

# =================================================================== Slide 9: demo screenshots
s = base("Demo & Output Screenshots", 9)
for x in (43.2, 487.4):
    rect(s, x, 115.2, 429.1, 302.4, PANEL, GREY, True, True, 0.02, 1.0)
pic_fit(s, A / "slide9_left.png", 49, 121, 417, 290)
pic_fit(s, A / "slide9_right.png", 493, 121, 417, 290)
tb(s, 43.2, 428, 429.1, 40, "Input: “plsss send pannunga brooo” → token-level language ID (plsss[EN] send[EN] pannunga[TA] brooo[EN]) → normalized output “please send pannunga bro”", 12, INK, align=PP_ALIGN.CENTER)
tb(s, 487.4, 428, 429.1, 40, "Next-word top-5, side by side: trigram (Kneser-Ney) vs LSTM; both models rank “please” first (9.1% vs 13.4%)", 12, INK, align=PP_ALIGN.CENTER)
tb(s, 43.2, 480, 873, 16, "Genuine screenshots of the project's Streamlit dashboard (project results/ folder and captures from the running app); cropped to the relevant panels, not edited.", 10.5, GREY, italic=True, align=PP_ALIGN.CENTER)

# =================================================================== Slide 10: conclusion & future work
s = base("Conclusion & Future Work", 10)
for x, n, t, col in ((43.2, 1, "Conclusion", RED), (487.4, 2, "Future Work", TEAL)):
    rect(s, x, 115.2, 429.1, 367.2, PANEL)
    circle(s, x + 25.2, 140.4, 39.6, n, col)
    tb(s, x + 79, 146, 300, 28, t, 20, NAVY, HEAD, True)
bullets(s, 72, 198, 380, [
    "Built language ID, normalization and two language models (KN trigram, LSTM) from scratch in one tested pipeline and dashboard.",
    f"Key numbers: LID {hi_a:.1f}% / {hi_b:.1f}% (Hinglish), {ta_a:.1f}% / {ta_b:.1f}% (Tanglish gold); WER {wer0:.1f}% → {wer1:.1f}%; perplexity 193 → 142 and 416 → 259.",
    "LSTM beats the trigram on every LM metric, mainly in perplexity.",
    [("Limitations: ", {"bold": True, "color": RED}), ("Tanglish labels are weak (silver); 120-sentence gold set not independently validated; vocabulary capped at 20k; no cross-variant transfer; one-command rebuild (run_all) not verified end to end.", {})],
], size=14, gap=8, bullet_w=16)
bullets(s, 516, 198, 380, [
    "Native-speaker validation and a larger Tanglish gold set",
    "Independent Tanglish normalization benchmark",
    "Better handling of names and acronyms",
    "Larger vocabulary and more robust GPU training",
    "Cross-language generalization; more Indic languages",
    "Context-aware, language-aware normalization; verify the full rebuild",
], size=15, gap=10, bullet_w=16)

# =================================================================== Slide 11: thank you
s = prs.slides.add_slide(BLANK)
bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb(NAVY)
tb(s, 0, 190, 960, 70, "Thank You", 60, "FFFFFF", HEAD, True, PP_ALIGN.CENTER)
tb(s, 0, 282, 960, 36, "Questions & Discussion", 24, ORANGE, BODY, False, PP_ALIGN.CENTER)
tb(s, 0, 462, 960, 20, "25AI2008  Multilingual AI  |  Division of Artificial Intelligence and Machine Learning", 14, PANEL, BODY, False, PP_ALIGN.CENTER)

prs.core_properties.title = "Code-Mixed (Tanglish/Hinglish) Text: Language Identification, Normalization and N-gram/LSTM Language Modelling"
prs.core_properties.author = "Angel Sweety P, Shally S, K Kokila Dharshini, Selvarani S"
prs.core_properties.subject = "25AI2008 Multilingual AI - Skill Based Evaluation Review"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
