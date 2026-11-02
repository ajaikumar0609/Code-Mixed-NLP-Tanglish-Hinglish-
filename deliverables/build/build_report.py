"""Builds the 25AI2008 skill-based evaluation report (DOCX) following the college template."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dx import *  # noqa

OUT = ROOT / "deliverables" / "25AI2008_REGNO.docx"
PAGES = json.loads((ROOT / "deliverables/build/pages.json").read_text()) if (ROOT / "deliverables/build/pages.json").exists() else {}

TITLE = "Code-Mixed (Tanglish/Hinglish) Text: Language Identification, Normalization and N-gram/LSTM Language Modelling"
TEAM = [("Angel Sweety P", "5051"), ("Shally S", "5078"), ("K Kokila Dharshini", "5037"), ("Selvarani S", "5088")]

# ----------------------------------------------------------------------------- results (read, not typed)
RES = ROOT / "results"
lh, lt = json.load(open(RES / "lid_hinglish.json")), json.load(open(RES / "lid_tanglish.json"))
mh, mt = json.load(open(RES / "lm_hinglish.json")), json.load(open(RES / "lm_tanglish.json"))
nm = json.load(open(RES / "normalization.json"))
pc = lambda x, d=1: f"{x * 100:.{d}f}%"
LID = {
    "hi_a": lh["models"]["char_ngram"]["results"]["test"]["word_level"], "hi_b": lh["models"]["bilstm"]["results"]["test"]["word_level"],
    "ta_a": lt["models"]["char_ngram"]["results"]["hand_reviewed_gold"]["word_level"], "ta_b": lt["models"]["bilstm"]["results"]["hand_reviewed_gold"]["word_level"],
    "ts_a": lt["models"]["char_ngram"]["results"]["test"]["word_level"], "ts_b": lt["models"]["bilstm"]["results"]["test"]["word_level"],
}
NS = nm["hinglish"]["systems"]
N0, N1, N2 = NS["no_normalization (baseline)"], NS["rules_only (punctuation + elongation)"], NS["rules + learned dictionary"]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, m, Inches(1))
sec.footer_distance = Inches(0.5)

st = doc.styles["Normal"]
st.font.name, st.font.size = TNR, Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), TNR)
st.paragraph_format.line_spacing = 1.5
st.paragraph_format.space_after = Pt(0)
st.paragraph_format.space_before = Pt(0)


# ----------------------------------------------------------------------------- helpers
def P(text="", align=J, size=12, name=TNR, bold=False, italic=False, before=0, after=6, line=1.5, **kw):
    p = doc.add_paragraph()
    fmt(p, align, before, after, line, **kw)
    if text:
        add_runs(p, text, name, size, bold, italic)
    return p


def H1(text, bm=None, page_before=True):
    p = doc.add_paragraph()
    fmt(p, C, 0, 10, 1.5, keep_next=True, page_before=page_before)
    add_runs(p, text, TNR, 14, True)
    _set_outline(p, 0)
    if bm: bookmark(p, bm)
    return p


def _set_outline(p, lvl):
    e = OxmlElement("w:outlineLvl"); e.set(qn("w:val"), str(lvl))
    p._p.get_or_add_pPr().insert_element_before(e, "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")


def H2(text, bm=None, page_before=False):
    p = doc.add_paragraph()
    fmt(p, L, 8, 4, 1.5, keep_next=True, page_before=page_before)
    add_runs(p, text, TNR, 13, True)
    _set_outline(p, 1)
    if bm: bookmark(p, bm)
    return p


def H3(text):
    p = doc.add_paragraph()
    fmt(p, L, 4, 2, 1.5, keep_next=True)
    add_runs(p, text, TNR, 12, True, True)
    return p


def BUL(items, style="bullet", size=12, line=1.5, after=2):
    for i, t in enumerate(items, 1):
        p = doc.add_paragraph()
        fmt(p, J, 0, after, line, left=0.35, first=-0.25)
        mark = "•\t" if style == "bullet" else f"{i}.\t"
        r = p.add_run(mark); set_font(r, TNR, size)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.35))
        add_runs(p, t, TNR, size)


def CAP(text, before=2, after=10):
    p = doc.add_paragraph()
    fmt(p, C, before, after, 1.0, keep_lines=True)
    add_runs(p, text, TNR, 11, False, True)
    return p


def TCAP(text):
    p = doc.add_paragraph()
    fmt(p, C, 6, 4, 1.0, keep_next=True)
    add_runs(p, text, TNR, 11, True)


def FIG(img, width, cap, keep_next=True):
    p = doc.add_paragraph()
    fmt(p, C, 4, 2, 1.0, keep_next=keep_next)
    p.add_run().add_picture(str(A / img), width=Inches(width))
    CAP(cap)


def EQ(img, num, scale=1.0, maxw=5.6):
    from PIL import Image
    w, h = Image.open(A / img).size
    wi = min(w / 300 * scale, maxw)
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(t, [5.7, 0.57])
    c0, c1 = t.rows[0].cells
    c0.vertical_alignment = c1.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = c0.paragraphs[0]; fmt(p, C, 3, 3, 1.0)
    p.add_run().add_picture(str(A / img), width=Inches(wi))
    q = c1.paragraphs[0]; fmt(q, R, 0, 0, 1.0)
    add_runs(q, f"({num})", TNR, 12)


def TABLE(rows, widths, header=True, size=10.5, align=None, shade_first_col=False, bold_last=False):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_col_widths(t, widths)
    for i, row in enumerate(rows):
        no_split(t.rows[i])
        for j, val in enumerate(row):
            c = t.cell(i, j)
            cell_borders(c, 6, "444444"); cell_margins(c)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = c.paragraphs[0]
            fmt(p, (align[j] if align else L), 0, 0, 1.0)
            hd = header and i == 0
            add_runs(p, str(val), TNR, size, hd or (bold_last and i == len(rows) - 1))
            if hd: cell_shade(c, "DCE3EE")
            elif shade_first_col and j == 0: cell_shade(c, "F4F5F7")
    if header: repeat_header(t.rows[0])
    sp = doc.add_paragraph(); fmt(sp, L, 0, 4, 1.0)
    return t


def CODE(src, size=7.5):
    lines = src.rstrip("\n").split("\n")
    long = [l for l in lines if len(l) > 98]
    if long:
        print("WARNING long code lines:", [l[:40] for l in long])
    p = doc.add_paragraph()
    fmt(p, L, 0, 5, 1.0)
    para_shade(p)
    for i, ln in enumerate(lines):
        indent = len(ln) - len(ln.lstrip(" "))
        r = p.add_run(" " * indent + ln.lstrip(" "))
        set_font(r, MONO, size)
        if i < len(lines) - 1:
            r.add_break()


def CODE_CAP(text):
    p = doc.add_paragraph()
    fmt(p, L, 6, 2, 1.0, keep_next=True)
    add_runs(p, text, TNR, 11, True)


def sig_box(doc_or_cell, lines_top, lines_bottom, align_right=True, width=2.7):
    t = doc.add_table(rows=2, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.RIGHT if align_right else WD_TABLE_ALIGNMENT.LEFT
    set_col_widths(t, [width])
    for i, lines in enumerate((lines_top, lines_bottom)):
        c = t.cell(i, 0); cell_borders(c, 6); cell_margins(c, 80, 80, 100, 100)
        c.paragraphs[0].text = ""
        for k, (txt, bold) in enumerate(lines):
            p = c.paragraphs[0] if k == 0 else c.add_paragraph()
            fmt(p, C, 3 if k else 2, 3, 1.0)
            add_runs(p, txt, ARIAL, 10, bold)
    return t


def banner():
    p = doc.add_paragraph(); fmt(p, C, 0, 0, 1.0)
    p.add_run().add_picture(str(A / "logo_banner.png"), width=Inches(6.2))


def AR(text, size=14, bold=True, align=C, before=0, after=6, line=1.15, **kw):
    p = doc.add_paragraph(); fmt(p, align, before, after, line, **kw)
    add_runs(p, text, ARIAL, size, bold)
    return p


def team_lines(size=16, upper=True, name=TNR, after=2):
    for n, r in TEAM:
        p = doc.add_paragraph(); fmt(p, C, 0, after, 1.15)
        add_runs(p, f"{n.upper() if upper else n} (Reg.No: {r})", name, size, True)


# ============================================================================= FRONT MATTER
# --- Page 1: cover
banner()
AR("SKILL BASED EVALUATION REPORT", 17, before=38, after=30)
AR("SUBMITTED BY", 14, after=6)
team_lines(15)
AR("25AI2008", 14, before=24, after=2)
AR("MULTILINGUAL AI", 14, after=0)
AR("DIVISION OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING", 14, before=60, after=6)
AR("SCHOOL OF COMPUTER SCIENCE AND TECHNOLOGY", 14, after=16)
p = doc.add_paragraph(); fmt(p, C, 0, 10, 1.0); p.add_run().add_picture(str(A / "logo_big.png"), width=Inches(1.15))
AR("OCTOBER 2026", 14, before=6)

# --- Page 2: title page
p = doc.add_paragraph(); fmt(p, C, 0, 0, 1.5, page_before=True)
add_runs(p, TITLE, TNR, 18, True)
p = doc.add_paragraph(); fmt(p, C, 70, 8, 1.5)
add_runs(p, "Submitted by", TNR, 14, True, True)
team_lines(16, True, TNR, 4)
p = doc.add_paragraph(); fmt(p, C, 50, 30, 1.0); p.add_run().add_picture(str(A / "logo_small.png"), width=Inches(1.0))
AR("DIVISION OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING", 14, after=30)
for t_ in ("KARUNYA INSTITUTE OF TECHNOLOGY AND SCIENCES", "(Declared as Deemed-to-be-under Sec-3 of the UGC Act, 1956)", "Karunya Nagar, Coimbatore - 641 114. INDIA"):
    p = doc.add_paragraph(); fmt(p, C, 0, 2, 1.15); add_runs(p, t_, TNR, 14, True)
AR("OCTOBER 2026", 14, before=44)

# --- Page 3: bonafide certificate
p = doc.add_paragraph(); fmt(p, C, 0, 0, 1.0, page_before=True); p.add_run().add_picture(str(A / "logo_banner.png"), width=Inches(6.2))
AR("DIVISION OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING", 14, before=6, after=2)
AR("SCHOOL OF COMPUTER SCIENCE AND TECHNOLOGY", 14, after=26)
AR("BONAFIDE CERTIFICATE", 12, after=18)
p = doc.add_paragraph(); fmt(p, J, 0, 10, 1.4)
add_runs(p, "This is to certify that the skill-based evaluation report for the course 25AI2008 Multilingual AI is a bonafide work done during the Odd semester of the academic year 2026-2027 by", ARIAL, 12)
for n, r in TEAM:
    q = doc.add_paragraph(); fmt(q, C, 0, 3, 1.3); add_runs(q, f"{n} (Reg.No: {r})", ARIAL, 12, True)
p = doc.add_paragraph(); fmt(p, J, 8, 10, 1.4)
add_runs(p, "in partial fulfilment of the requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering ([[AI/AIML]]) of Karunya Institute of Technology and Sciences.", ARIAL, 12)
p = doc.add_paragraph(); fmt(p, L, 4, 6, 1.0); para_border(p, ("bottom",), 8)
p = doc.add_paragraph(); fmt(p, L, 6, 24, 1.4)
add_runs(p, "Submitted for the assessment held on ____________________________", ARIAL, 12)
sig_box(None, [("Signature of the Class Teacher", True)], [("[[Class Teacher]]", False), ("[[Designation]]", False), ("Division of Artificial Intelligence and Machine Learning", False)])

# --- Page 4: declaration
AR("DECLARATION", 14, after=26, page_before=True)
decl = [
    "We hereby declare that the skill-based evaluation report for the course 25AI2008 Multilingual AI submitted by us to Karunya Institute of Technology and Sciences is an authentic work carried out by us under the guidance of [[Course Teacher]]. All the information presented in this report is original and has not been submitted for any other academic purpose.",
    "We acknowledge that the sources of information used in this report have been appropriately cited and referenced. Any contributions from other individuals or organizations have been duly acknowledged.",
    "We understand that any act of plagiarism or academic dishonesty is strictly prohibited and may result in disciplinary action as per the university's regulations.",
    "We take full responsibility for the content and integrity of this report, and we are willing to provide further clarification or information if required.",
]
for t_ in decl:
    p = doc.add_paragraph(); fmt(p, J, 0, 10, 1.9); add_runs(p, t_, ARIAL, 12)
p = doc.add_paragraph(); fmt(p, L, 10, 14, 1.0); para_border(p, ("bottom",), 12)
sig_box(None, [("Signature of student(s)", True)], [(f"{n} ({r})", False) for n, r in TEAM], width=2.7)

# --- Page 5: evaluation report
AR("DIVISION OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING", 14, after=2, page_before=True)
AR("SCHOOL OF COMPUTER SCIENCE AND TECHNOLOGY", 14, after=22)
AR("EVALUATION REPORT", 12, after=16)
rows = [["S.No", "Evaluation Parameter", "Max. Marks", "Marks Secured"],
        ["1", "Technical Proficiency", "10", ""], ["2", "Problem-solving Ability", "5", ""], ["3", "Creativity and Innovation", "5", ""],
        ["4", "Communication Skills", "5", ""], ["5", "Teamwork and Presentation", "10", ""], ["Total", "", "40", ""]]
t = doc.add_table(rows=len(rows), cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER
set_col_widths(t, [0.75, 2.85, 1.2, 1.45])
for i, row in enumerate(rows):
    t.rows[i].height = Inches(0.48 if i else 0.55)
    for j, v in enumerate(row):
        c = t.cell(i, j); cell_borders(c, 6); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        pp = c.paragraphs[0]; fmt(pp, C, 0, 0, 1.0); add_runs(pp, v, ARIAL, 10.5, i == 0)
t.cell(6, 0).merge(t.cell(6, 1))
t.cell(6, 0).paragraphs[0].text = ""
add_runs(t.cell(6, 0).paragraphs[0], "Total", ARIAL, 10.5)
fmt(t.cell(6, 0).paragraphs[0], C, 0, 0, 1.0)
sp = doc.add_paragraph(); fmt(sp, L, 0, 18, 1.0)
sig_box(None, [("Signature of the Class Teacher", True)], [("[[Class Teacher]]", False), ("[[Designation]]", False), ("Division of Artificial Intelligence and Machine Learning", False)])

# --- Page 6: table of contents
_pb = doc.add_paragraph(); fmt(_pb, L, 0, 0, 1.0); _pb.add_run().add_break(WD_BREAK.PAGE)
AR("TABLE OF CONTENTS", 12, after=14)
toc = [("1", "", "Abstract", "toc_abstract"), ("2", "", "Problem Statement", "toc_problem"), ("3", "", "Methodology/Architecture", "toc_method"),
       ("", "4.1", "Implementation", "toc_impl"), ("", "4.2", "Coding", "toc_code"), ("", "4.3", "Output Screenshot", "toc_shots"),
       ("5", "", "Conclusion", "toc_conc"), ("6", "", "References", "toc_refs")]
t = doc.add_table(rows=len(toc) + 1, cols=4); t.alignment = WD_TABLE_ALIGNMENT.CENTER
set_col_widths(t, [0.7, 0.7, 3.2, 1.2])
hdr = ["S.No", "", "Content Title", "Page No"]
for j, v in enumerate(hdr):
    c = t.cell(0, j); cell_borders(c, 6); c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    pp = c.paragraphs[0]; fmt(pp, C, 0, 0, 1.0); add_runs(pp, v, ARIAL, 10.5, True)
t.rows[0].height = Inches(0.5)
t.cell(0, 0).merge(t.cell(0, 1)); t.cell(0, 0).paragraphs[0].text = ""; add_runs(t.cell(0, 0).paragraphs[0], "S.No", ARIAL, 10.5, True); fmt(t.cell(0, 0).paragraphs[0], C, 0, 0, 1.0)
for i, (sno, sub, title, bm) in enumerate(toc, 1):
    t.rows[i].height = Inches(0.46)
    vals = [sno, sub, title, None]
    for j, v in enumerate(vals):
        c = t.cell(i, j)
        if j == 0 and i in (4, 5, 6):          # the '4' cell spans 4.1-4.3: draw only its outer borders
            sides = ["left", "right"] + (["top"] if i == 4 else []) + (["bottom"] if i == 6 else [])
            cell_borders(c, 6, "000000", tuple(sides))
        else:
            cell_borders(c, 6)
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        pp = c.paragraphs[0]; fmt(pp, C, 0, 0, 1.0)
        if v is None:
            if os.environ.get("TOC_FIELDS", "1") == "1":
                add_field(pp, f"PAGEREF {bm} \\h", str(PAGES.get(bm, "0")), ARIAL, 10.5)
            else:
                add_runs(pp, str(PAGES.get(bm, "0")), ARIAL, 10.5)
        elif v:
            add_runs(pp, v, ARIAL, 10.5)
        elif i == 5 and j == 0:
            pass
for i in (1, 2, 3, 7, 8):
    # rows without a sub-number: merge the S.No and sub-number cells
    t.cell(i, 0).merge(t.cell(i, 1))
    c = t.cell(i, 0)
    for pp in c.paragraphs[1:]:
        pp._p.getparent().remove(pp._p)
    c.paragraphs[0].text = ""
    add_runs(c.paragraphs[0], toc[i - 1][0], ARIAL, 10.5); fmt(c.paragraphs[0], C, 0, 0, 1.0)
# vertically centre the '4' in the middle row of its block
c = t.cell(5, 0); c.paragraphs[0].text = ""; add_runs(c.paragraphs[0], "4", ARIAL, 10.5); fmt(c.paragraphs[0], C, 0, 0, 1.0)
t.cell(4, 0).paragraphs[0].text = ""

# ============================================================================= BODY (new section, numbering restarts)
sp = doc.add_paragraph(); fmt(sp, L, 0, 0, 1.0)
new = doc.add_section(WD_SECTION.NEW_PAGE)
pg = OxmlElement("w:pgNumType"); pg.set(qn("w:start"), "1")
new._sectPr.insert_element_before(pg, "w:cols", "w:formProt", "w:vAlign", "w:noEndnote", "w:titlePg", "w:textDirection", "w:bidi", "w:rtlGutter", "w:docGrid", "w:printerSettings")
new.footer.is_linked_to_previous = False
fp = new.footer.paragraphs[0]; fp.text = ""; fmt(fp, C, 0, 0, 1.0)
add_field(fp, "PAGE", "1", TNR, 11)

# ----------------------------------------------------------------------------- 1. ABSTRACT
H1("ABSTRACT", "toc_abstract", page_before=False)
ABS = [
    "Social-media users in India routinely write Tamil and Hindi in the Roman alphabet and mix English words into the same sentence. The result, called Tanglish (Tamil-English) and Hinglish (Hindi-English), has no standard spelling, no clear word-level language boundaries and very few annotated resources, so tools built for monolingual text perform poorly on it. This project implements and evaluates three NLP tasks on Romanized Tanglish and Hinglish: token-level language identification, text normalization and next-word language modelling.",
    "Language identification uses two models, a character n-gram TF-IDF classifier with Logistic Regression and a BiLSTM tagger over word embeddings and character-CNN features, plus deterministic rules for numbers, punctuation, URLs, mentions, hashtags and emoji. Normalization is rule- and dictionary-based (Unicode clean-up, repeated-character reduction, slang expansion, a Hinglish dictionary learned from annotator corrections and a curated Tanglish lexicon); it is neither translation nor transliteration. For language modelling, a trigram model with interpolated Kneser-Ney smoothing, written from scratch, is compared with a PyTorch LSTM on the same corpus, vocabulary and evaluation protocol.",
    "Hinglish data come from COMI-LINGUA (CC-BY-4.0; three-annotator labels and references). Tanglish data come from Tanglish-Corpus-185k (CC-BY-4.0), whose token labels are weak (silver) lexicon-based labels, so the Tanglish identifiers are evaluated on a separate 120-sentence hand-reviewed gold set.",
    f"Word-level language-identification accuracy is {pc(LID['hi_a']['accuracy'])} (character n-gram) and {pc(LID['hi_b']['accuracy'])} (BiLSTM) for Hinglish, and {pc(LID['ta_a']['accuracy'])} and {pc(LID['ta_b']['accuracy'])} for Tanglish on the hand-reviewed set. Hinglish normalization lowers WER against the best of three references from {pc(N0['best_of_refs']['wer'])} to {pc(N2['best_of_refs']['wer'])}, while the annotators disagree with each other at {pc(nm['hinglish']['inter_annotator']['mean_wer'])} mean WER. The LSTM lowers test perplexity from {mh['ngram']['test_perplexity']:.0f} to {mh['lstm']['test_perplexity']:.0f} (Hinglish) and from {mt['ngram']['test_perplexity']:.0f} to {mt['lstm']['test_perplexity']:.0f} (Tanglish). The limitations are stated explicitly: weak Tanglish labels, a small Tanglish gold set without independent native-speaker validation, and a one-command rebuild that has not been verified end to end.",
]
for t_ in ABS:
    P(t_)
P("**Keywords:** code-mixing, Tanglish, Hinglish, language identification, text normalization, Kneser-Ney, LSTM language model.", after=0)

# ----------------------------------------------------------------------------- 2. PROBLEM STATEMENT
H1("PROBLEM STATEMENT", "toc_problem")
H2("2.1 The code-mixing challenge")
P("Millions of Indian users type Tamil or Hindi with English letters and switch to English in the middle of a sentence, for example *naan inniku college ku late ah varen* (Tanglish) or *kal meeting hai, please notes share kar dena* (Hinglish). Such text is the dominant form of informal communication in comments, chats and posts, yet it breaks the assumptions of standard NLP pipelines, which expect one language, one script and consistent spelling. The main difficulties are:")
BUL([
    "**Mixing of English with Tamil and Hindi** inside one sentence, often with Tamil or Hindi grammatical suffixes attached to English words (*college-ku*).",
    "**Romanized writing instead of native scripts.** Tamil and Hindi are written in Latin letters, so script can no longer separate the languages.",
    "**Informal spelling variation.** The same word appears as *iruku / irukku / erukku* or *hai / haiii / hain*, and chat slang (*plz, brooo*) is common.",
    "**Ambiguous words and unclear language boundaries.** Many tokens are valid in more than one language (*than, ah, to, hi*), and names, acronyms and handles belong to no language.",
    "**Limited gold-standard annotated data.** Public token-level data exist for Hindi-English, but no public gold token-level language-identification corpus was available for Romanized Tamil-English, which forces the use of weak labels.",
    "**Unseen word combinations.** Because spelling and mixing are so varied, many word sequences at test time never occur in training, which makes language modelling and smoothing difficult.",
])
H2("2.2 Objectives")
P("The project has three measurable objectives, each tied to a metric that is reported in this document:")
BUL([
    "**Token-level language identification.** Label every token of a Tanglish or Hinglish sentence as TA/HI, EN or OTHER (with NUM and PUNCT handled by rules), and compare a context-free character n-gram model with a context-aware BiLSTM using word-level accuracy and macro-F1 on held-out data.",
    "**Consistent code-mixed text normalization.** Map spelling variants to one preferred Romanized spelling without translating, and measure word error rate (WER) against three human references for Hinglish.",
    "**Statistical versus neural language modelling.** Compare an interpolated Kneser-Ney trigram model with an LSTM on the same corpus, vocabulary and evaluation positions using perplexity and top-1 / top-5 next-word accuracy.",
], style="num")
H2("2.3 Scope")
P("**In scope:** Romanized Tanglish and Hinglish; the three tasks above; training of all models from scratch on the stated corpora; an interactive Streamlit dashboard that runs locally; honest reporting of evaluation limits.")
P("**Out of scope:** translation, transliteration into native script, sentiment analysis, speech recognition, transformer or pretrained multilingual-model fine-tuning, and any claim of cross-language transfer between Tanglish and Hinglish models.", after=0)

# ----------------------------------------------------------------------------- 3. METHODOLOGY
H1("METHODOLOGY / ARCHITECTURE", "toc_method")
H2("3.1 System overview")
P("The system is organised as four modules that share one preprocessing layer and one inference entry point (Figure 1). Raw text is cleaned and tokenized; each token receives a language label (Module 1); the labels guide the normalizer (Module 2); the normalized words are then scored and extended by two language models (Modules 3 and 4). The same `CodeMixedPipeline` class is used by the Streamlit application, the command-line tool and the evaluation scripts, so the demonstration cannot drift from the reported experiments.")
FIG("fig_system.png", 5.1, "Figure 1. End-to-end architecture of the code-mixed text processing system.")

H2("3.2 Datasets and annotation")
FIG("fig_data.png", 6.0, "Figure 2. Data sources, splits and label provenance for the two language variants.", keep_next=True)
H3("Hinglish: COMI-LINGUA")
P("COMI-LINGUA (Sheth et al., Findings of EMNLP 2025) is a manually annotated Hindi-English code-mixed dataset published under CC-BY-4.0 (https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) [1]. Every instance is annotated by three annotators. The project uses two subsets: *Language Identification (LID)*, where each token carries a Hindi, English or other tag, and *Text Normalization (TN)*, where each sentence has three annotator-written normalized references. Only Roman-script sentences are used, because this project concerns Romanized text: after removing native-script sentences (22,166 training and 4,199 test) and sentences on which the annotators' tokenizations disagree (331 and 51), 7,191 training, 799 validation and 750 test sentences remain for language identification. The token label is the majority of the three annotators (ties fall back to annotator 1). The official train/test split is kept, and validation is a 10% hold-out of train. For language modelling, the annotator-normalized text is used (21,022 / 2,336 / 4,933 sentences after de-duplication).")
H3("Tanglish: Tanglish-Corpus-185k")
P("Tanglish-Corpus-185k (https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k, CC-BY-4.0) [12] contains 185,973 Romanized Tamil-English sentences collected from YouTube comments, Reddit and the DravidianCodeMix corpus; each row carries the text and its source. After lower-cased, punctuation-stripped de-duplication and a minimum-length filter, 185,670 sentences remain (303 removed). The corpus has no token labels, so it is used as follows:")
BUL([
    "**Language-model corpus.** The text is normalized by this project's Tanglish normalizer and split *by source* (a whole YouTube video or subreddit goes to a single split, so a comment thread cannot straddle training and test) into 140,682 / 26,297 / 18,622 sentences.",
    "**Silver token labels for language identification.** 30,000 / 2,000 / 3,000 sentences (train / validation / test) receive weak labels from rules over a curated Tamil lexicon, an English chat-slang lexicon, a named-entity/acronym lexicon and English word frequencies (`wordfreq`), applied in a fixed priority order (Table 1).",
    "**Gold evaluation set.** 120 sentences (1,327 word tokens) were hand-reviewed and corrected token by token. These sentences are never used for training or tuning, and they are the only honest estimate of Tanglish language-identification accuracy.",
])
TCAP("Table 1. Priority order of the Tanglish silver-labelling rules (src/silver_labels.py).")
TABLE([["Priority", "Rule", "Label"],
       ["1", "Non-ASCII / mojibake word", "OTHER"], ["2", "Named-entity / acronym lexicon", "OTHER"], ["3", "Curated Tamil lexicon", "TA"],
       ["4", "English chat-slang lexicon", "EN"], ["5", "Single letters (a, i, u, r, k are English)", "EN / OTHER"],
       ["6", "Common English word (word-frequency zipf >= 3.0)", "EN"], ["7", "ALL-CAPS token not matched above", "OTHER"],
       ["8", "Anything else (out of vocabulary)", "TA"]], [0.8, 4.3, 1.1], align=[C, L, C])
P("**Gold versus silver.** Silver labels are *not* human annotations: they are the output of the rules above, so a model trained on them can at best reproduce those rules. The 99% score on the silver test split (Table 6) therefore measures agreement with the rules and must not be read as the real accuracy; the hand-reviewed gold figure is the one reported as the Tanglish result. The gold set was reviewed by the project team with an AI assistant and was not validated by independent native-speaker annotators (see Section 5.3).")
H3("Label convention")
P("**TA** Romanized Tamil; **HI** Romanized Hindi; **EN** English, including chat slang and loanwords such as *bro* and *late*; **OTHER** names, acronyms, handles, URLs, emoji and noise (the annotation rule for non-linguistic or other tokens); **NUM** numbers; **PUNCT** punctuation. The classifiers decide only among the word-level labels of the selected variant (TA/EN/OTHER or HI/EN/OTHER); NUM, PUNCT and the non-word kinds of OTHER come from rules.")

H2("3.3 Module 1: Language identification")
P("The tokenizer is lossless and typed: every token records whether whitespace preceded it and its kind (WORD, NUM, PUNCT, URL, MENTION, HASHTAG, EMOJI), so a sentence can be rebuilt after editing. Tokens that are not words receive their label from `rule_label` (number, punctuation, URL, @mention, #hashtag and emoji map to NUM, PUNCT or OTHER; a word written in Devanagari or Tamil script maps to HI or TA). Only WORD tokens are passed to a classifier (Figure 3).")
FIG("fig_lid.png", 6.2, "Figure 3. The two language-identification models.")
H3("Model A: character n-gram TF-IDF + Logistic Regression")
P("Each word is lower-cased and converted into character n-grams within word boundaries (`char_wb`), weighted by sublinear TF-IDF, and extended with four case features (initial capital, all capitals, contains a digit, contains an apostrophe). A multinomial Logistic Regression predicts the label. Because the model is context-free, training on unique (word, label) types weighted by their counts is mathematically identical to training on every token occurrence and is much faster. The n-gram range and regularization strength C were chosen on the validation set from a small grid (ranges 1-3, 1-4, 2-5; C in 1, 10, 50): the selected setting was (1,4) with C = 10 for Hinglish and (2,5) with C = 50 for Tanglish.")
H3("Model B: word embedding + character CNN + BiLSTM")
P("Model B sees the whole sentence. Each word is represented by a 64-dimensional word embedding (out-of-vocabulary words map to `<unk>`, and 10% of words are randomly replaced by `<unk>` during training so the model relies on spelling as well as identity) concatenated with the output of a character CNN (32-dimensional character embeddings, two convolutions of width 3 and 5 with 64 filters each, max-pooled over characters). The concatenated vector (64 + 128 dimensions) passes through dropout (0.3) into a bidirectional LSTM with 128 hidden units per direction, and a linear layer with softmax assigns a label to every token. Training uses cross-entropy loss, the Adam optimizer, gradient clipping at 5.0 and keeps the epoch with the best validation accuracy.")
P("**Important distinction.** The BiLSTM above is a *sequence tagger* that predicts a language label for each word. It is a different model, solving a different task, from the LSTM *language model* of Module 4, which predicts the next word.")

H2("3.4 Module 2: Text normalization")
P("Normalization rewrites noisy spellings into one preferred Romanized spelling so that the same word is represented consistently. It is **not translation** (Tamil and Hindi words are never replaced by English) and **not transliteration** (nothing is converted to Tamil or Devanagari script). The pipeline (Figure 4) applies the following steps in order:")
FIG("fig_norm.png", 6.2, "Figure 4. The rule- and dictionary-based normalization pipeline.")
BUL([
    "**Unicode and whitespace handling:** NFKC normalization, removal of zero-width characters, unification of quotation marks and collapsing of whitespace.",
    "**Tokenization and detokenization:** the typed tokenizer splits text; `detokenize` rebuilds it with correct spacing around punctuation.",
    "**Punctuation normalization:** runs of the same punctuation or emoji are collapsed (`!!!` becomes `!`, `....` becomes `...`).",
    "**Repeated-character reduction:** only runs of three or more identical letters are touched, and each run may shrink to one or two characters; the candidate that is a known word is chosen (`rombbba` to `romba`, `haiii` to `hai`), so legitimate double letters (`irukku`, `good`) are never altered.",
    "**English chat-slang expansion:** applied to English-tagged or untagged words (`plz` to `please`).",
    "**Language-aware spelling dictionaries:** for Tanglish, a curated lexicon of variant groups maps spellings such as `iruku` and `erukku` to `irukku`; for Hinglish, a dictionary is learned automatically from the COMI-LINGUA annotator corrections by aligning each original sentence with its references and keeping rewrites that are frequent enough (minimum count 2 and minimum rewrite ratio 0.4, chosen on validation data; 1,683 entries). Dictionary entries respect the predicted language tag, so a Tamil-tagged word is not rewritten by an English rule.",
])
P("Words tagged OTHER (names, handles, acronyms) are left unchanged. The Hinglish dictionary is derived only from the training split's annotator references; the Tanglish lexicon is hand-curated from the most frequent tokens of Tanglish-Corpus-185k.")

H2("3.5 Module 3: Trigram language model")
P("A language model assigns a probability to a word sequence. Under the trigram assumption, the probability of a word *w* depends only on the two previous words *u v*, P(w | u v), and is estimated from n-gram frequency counts in the training corpus: unigram counts c(w), bigram counts c(v w) and trigram counts c(u v w). Raw relative frequencies assign zero probability to unseen trigrams, which are very common in noisy code-mixed text, so smoothing is essential. The model implemented here is **interpolated Kneser-Ney (KN)** [7], written from scratch with sorted NumPy arrays (no NLTK or external language-model library). The highest order uses discounted counts, and the probability mass removed by the absolute discount D is redistributed to a lower-order model:")
EQ("eq1.png", 1)
EQ("eq1b.png", "1a", maxw=2.0)
P("Lower orders use *continuation counts* instead of raw counts. N1+(• v w) is the number of distinct words that precede the bigram *v w*, which measures how freely a word can start new contexts (the classic example is that *Francisco* is frequent but follows almost only *San*):")
EQ("eq2.png", 2)
EQ("eq3.png", 3)
P("The final uniform term 1/|V| guarantees that every vocabulary word, including `<unk>`, has non-zero probability. Unseen contexts back off to the next lower order. The discount D is tuned on validation perplexity over the grid {0.5, 0.6, 0.7, 0.75, 0.8, 0.9}; the best value was 0.9 for both variants, which is the upper edge of the grid, so a larger value was not tried. **Prediction:** the top-1 and top-5 next words are exact: the candidates are the trigram followers, the bigram followers and the most probable unigrams, which provably contain the top-k under this back-off scheme. **Evaluation:** perplexity over all test tokens, including `</s>` and `<unk>`:")
EQ("eq4.png", 4)

H2("3.6 Module 4: LSTM language model")
P("The neural language model reads the same word sequence and predicts the next word with a recurrent network (Figure 5). Its components are:")
BUL([
    "**Token vocabulary.** The vocabulary is built once from the training split (words with at least `min_count` occurrences, capped at a maximum size) plus the special tokens `<pad>`, `<unk>`, `<s>` and `</s>`; numbers are mapped to `<num>`.",
    "**Embedding layer and dropout.** Word indices are mapped to dense vectors (256 dimensions for Tanglish, 200 for Hinglish) and regularized with dropout (0.3 and 0.4).",
    "**LSTM layer** [8]: one layer with 512 (Tanglish) or 400 (Hinglish) hidden units.",
    "**Linear projection and softmax.** The hidden state is projected back to the embedding size and multiplied by the transposed embedding matrix (weight tying [10]), then a softmax gives the probability of every vocabulary word.",
    "**Training.** Cross-entropy loss with teacher forcing, the Adam optimizer [11] (learning rate 0.002), backpropagation through time over each padded sentence batch, gradient-norm clipping at 1.0, and a learning-rate reduction on plateau.",
    "**Validation and checkpoint selection.** Validation perplexity is computed after every epoch; the weights from the epoch with the lowest validation perplexity are restored and saved, and training stops early if validation perplexity stops improving.",
])
FIG("fig_lm.png", 6.2, "Figure 5. Architecture of the Kneser-Ney trigram model (left) and the LSTM language model (right).")
EQ("eq5.png", 5, maxw=6.2)
EQ("eq5b.png", "5a", maxw=6.2)
EQ("eq6.png", 6, maxw=6.2)
P("**Fair comparison.** Within each language variant the trigram model and the LSTM use the *same* training corpus, the *same* vocabulary object (built once from the training split), the same `<s> ... </s>` sentence framing, the same test sentences, and the same scoring: perplexity over every token, and top-1/top-5 accuracy over positions whose true next word is a real vocabulary word (`<unk>`, `<num>` and `</s>` targets are skipped). The training script asserts that both models are scored on an identical number of positions (101,053 for Hinglish and 268,856 for Tanglish).")

H2("3.7 Evaluation protocol and metrics")
BUL([
    "**Language identification:** word-level accuracy and macro-F1 on WORD tokens (also reported over all tokens), with per-class precision/recall and confusion matrices; Hinglish on the gold COMI-LINGUA test split; Tanglish on the hand-reviewed gold set (primary) and on the silver test split (reported separately and flagged as inflated).",
    "**Normalization:** corpus WER (and CER) on lower-cased words without punctuation (WER = (S + D + I) / N, where S, D and I count substituted, deleted and inserted words and N is the number of reference words), against the first reference and against the best of the three references; exact-match rate; inter-annotator WER as a ceiling reference.",
    "**Language modelling:** perplexity, top-1 and top-5 next-word accuracy.",
])
H2("3.8 Individual contribution of team members")
P("The template asks each team to highlight individual contributions. The repository does not record who wrote which module, so this table is intentionally left for the team to complete; no contributions have been assumed.", after=4)
TABLE([["Team member", "Reg.No", "Contribution (to be filled by the team)"]] +
      [[n, r, "[[to be filled]]"] for n, r in TEAM], [1.9, 0.9, 3.4], align=[L, C, L])

# ----------------------------------------------------------------------------- 4. IMPLEMENTATION
H1("IMPLEMENTATION – CODING AND OUTPUT SCREENSHOT", "toc_impl")
H2("4.1 Implementation")
H3("Software stack")
P("The software environment was read from `requirements.txt` and from the virtual environment of the project (Table 2). The n-gram model, the evaluation metrics (WER/CER, perplexity, top-k) and the tokenizer are written from scratch with NumPy; NLTK is not used. Matplotlib is listed in `requirements.txt` as optional (used for the exploration notebook only).")
TCAP("Table 2. Software stack.")
TABLE([["Component", "Version", "Used for"],
       ["Python", "3.14.3", "implementation language"],
       ["PyTorch", "2.14.0", "BiLSTM tagger, LSTM language model (MPS backend for training)"],
       ["scikit-learn", "1.9.1", "TF-IDF, Logistic Regression, metrics"],
       ["NumPy / pandas", "2.4.0 / 3.0.3", "KN trigram model, data handling"],
       ["wordfreq", "(requirements: >= 3.1)", "English word frequencies for silver labels and the normalizer"],
       ["joblib", "(requirements: >= 1.4)", "saving scikit-learn models"],
       ["Streamlit", "1.62.0", "interactive dashboard (app.py)"],
       ["pytest", ">= 8", "25 unit and integration tests (run: 25 passed)"]], [1.5, 1.6, 3.1], align=[L, L, L])
H3("Project structure")
TABLE([["Path", "Contents"],
       ["app.py", "Streamlit dashboard: Analyze, Batch, Evaluation and About pages"],
       ["src/preprocessing.py", "typed lossless tokenizer, rule labels, elongation candidates"],
       ["src/language_identifier.py, train_lid.py", "Model A (char n-gram + LogReg) and Model B (BiLSTM tagger); training"],
       ["src/normalizer.py, tanglish_lexicon.py, train_normalizer.py", "rule/dictionary normalizer, curated Tanglish lexicon, dictionary learning"],
       ["src/ngram_model.py", "interpolated Kneser-Ney trigram model (NumPy)"],
       ["src/lstm_model.py, train_lm.py, vocab.py", "LSTM language model, shared vocabulary, joint LM training and evaluation"],
       ["src/evaluation.py, pipeline.py, run_all.py", "metrics; single inference entry point; stage runner"],
       ["data/", "raw/ (downloaded), processed/ (splits), lexicons/, gold/ (Tanglish gold files)"],
       ["models/, results/, tests/", "trained models; metric JSON files and screenshots; 25 tests"]], [2.6, 3.6], align=[L, L], size=10)
H3("Hardware and training configuration")
P("Training and evaluation were run on an Apple M4 laptop with 16 GB of memory under macOS (hardware read from the development machine); the neural models were trained on PyTorch's MPS (Apple GPU) backend, as recorded in the saved results. The BiLSTM taggers were trained for 10 epochs (batch 64, Adam, learning rate 0.002). The LSTM language models used the settings in Table 3. The Tanglish LSTM vocabulary is capped at 20,000 words (10.9% of test tokens are out of vocabulary) because a 30,000-word softmax froze training on the MPS backend; the cap is applied identically to the trigram model. The Tanglish LSTM trained for 10 epochs (validation perplexity was still falling slowly), the Hinglish LSTM stopped early after 25 epochs.")
TCAP("Table 3. Language-model configuration and size (from results/lm_*.json).")
TABLE([["", "Hinglish", "Tanglish"],
       ["Train / val / test sentences", f"{mh['train_sentences']:,} / {mh['val_sentences']:,} / {mh['test_sentences']:,}", f"{mt['train_sentences']:,} / {mt['val_sentences']:,} / {mt['test_sentences']:,}"],
       ["Vocabulary (min count)", f"{mh['vocab_size']:,} (2)", f"{mt['vocab_size']:,} (3, capped at 20,000)"],
       ["Test out-of-vocabulary rate", pc(mh["test_oov_rate"]), pc(mt["test_oov_rate"])],
       ["Trigram types / discount D", f"{mh['ngram']['n_trigrams']:,} / {mh['ngram']['discount']}", f"{mt['ngram']['n_trigrams']:,} / {mt['ngram']['discount']}"],
       ["LSTM embedding / hidden / dropout", "200 / 400 / 0.4", "256 / 512 / 0.3"],
       ["LSTM parameters", f"{mh['lstm']['parameters']:,}", f"{mt['lstm']['parameters']:,}"],
       ["LSTM epochs run / training time", f"{len(mh['lstm']['history'])} (early stop) / {mh['lstm']['seconds']:.0f} s", f"{len(mt['lstm']['history'])} / {mt['lstm']['seconds']:.0f} s"]],
      [2.4, 1.9, 1.9], align=[L, C, C])

H3("Experimental results")
P("All numbers below were read from the saved files `results/lid_*.json`, `results/normalization.json` and `results/lm_*.json`. They describe these datasets and splits and are not general accuracy claims.")
TCAP("Table 4. Language identification: word-level accuracy and macro-F1 on held-out data.")
TABLE([["Model", "Hinglish gold test\nacc. / macro-F1", "Tanglish hand-reviewed gold (120 sent.)\nacc. / macro-F1", "Tanglish silver test (inflated)\nacc. / macro-F1"],
       ["Character n-gram + TF-IDF + LogReg", f"{pc(LID['hi_a']['accuracy'])} / {LID['hi_a']['macro_f1']:.3f}", f"{pc(LID['ta_a']['accuracy'])} / {LID['ta_a']['macro_f1']:.3f}", f"{pc(LID['ts_a']['accuracy'])} / {LID['ts_a']['macro_f1']:.3f}"],
       ["BiLSTM (word + char-CNN)", f"{pc(LID['hi_b']['accuracy'])} / {LID['hi_b']['macro_f1']:.3f}", f"{pc(LID['ta_b']['accuracy'])} / {LID['ta_b']['macro_f1']:.3f}", f"{pc(LID['ts_b']['accuracy'])} / {LID['ts_b']['macro_f1']:.3f}"]],
      [2.0, 1.4, 1.5, 1.3], align=[L, C, C, C], size=10)
P("The Tanglish gold figures come from a hand-reviewed set, not from an independently annotated public benchmark. The silver-test figures (about 99%) measure agreement with the labelling rules and are shown only to make that inflation visible. Per-class results show where errors concentrate: on the Tanglish gold set the OTHER class (names and acronyms) has recall of only 47.5% (character n-gram) and 45.0% (BiLSTM) on 80 tokens, i.e. roughly 46%, whereas TA and EN recall are above 97% and 99%; on Hinglish the OTHER recall is 66.3% and 73.0%.")
TCAP("Table 5. Hinglish normalization on the COMI-LINGUA TN test split (4,934 sentences).")
TABLE([["System", "WER vs best of\n3 references", "WER vs first\nreference", "Exact match\n(best reference)"],
       ["No normalization (baseline)", pc(N0["best_of_refs"]["wer"]), pc(N0["first_ref"]["wer"]), pc(N0["best_of_refs"]["exact_match"])],
       ["Rules only (punctuation + elongation)", pc(N1["best_of_refs"]["wer"]), pc(N1["first_ref"]["wer"]), pc(N1["best_of_refs"]["exact_match"])],
       ["Rules + learned dictionary (final)", pc(N2["best_of_refs"]["wer"]), pc(N2["first_ref"]["wer"]), pc(N2["best_of_refs"]["exact_match"])],
       ["Mean inter-annotator WER (reference ceiling)", pc(nm["hinglish"]["inter_annotator"]["mean_wer"]), "", ""]],
      [2.9, 1.1, 1.1, 1.1], align=[L, C, C, C], size=10)
P(f"Normalization WER falls from {pc(N0['best_of_refs']['wer'])} to {pc(N2['best_of_refs']['wer'])}. The three human annotators disagree with one another at a mean WER of {pc(nm['hinglish']['inter_annotator']['mean_wer'])}, which shows that no single correct spelling exists; normalization evaluation is therefore partly subjective, and scores should be read against this disagreement level and not against zero.")
TCAP("Table 6. Language modelling on the held-out test sentences (same corpus, vocabulary and positions for both models).")
TABLE([["Metric", "Hinglish\nTrigram KN", "Hinglish\nLSTM", "Tanglish\nTrigram KN", "Tanglish\nLSTM"],
       ["Test perplexity (lower is better)", f"{mh['ngram']['test_perplexity']:.0f}", f"{mh['lstm']['test_perplexity']:.0f}", f"{mt['ngram']['test_perplexity']:.0f}", f"{mt['lstm']['test_perplexity']:.0f}"],
       ["Top-1 next-word accuracy", pc(mh["ngram"]["top1"]), pc(mh["lstm"]["top1"]), pc(mt["ngram"]["top1"]), pc(mt["lstm"]["top1"])],
       ["Top-5 next-word accuracy", pc(mh["ngram"]["top5"]), pc(mh["lstm"]["top5"]), pc(mt["ngram"]["top5"]), pc(mt["lstm"]["top5"])]],
      [2.4, 0.95, 0.95, 0.95, 0.95], align=[L, C, C, C, C], size=10)
FIG("fig_results.png", 5.3, "Figure 6. Summary of results: (a) language identification, (b) perplexity, (c) next-word accuracy, (d) Hinglish normalization WER. Tanglish language identification uses the hand-reviewed gold set.")
P(f"The LSTM reduces perplexity by {(1 - mh['lstm']['test_perplexity'] / mh['ngram']['test_perplexity']) * 100:.0f}% for Hinglish and {(1 - mt['lstm']['test_perplexity'] / mt['ngram']['test_perplexity']) * 100:.0f}% for Tanglish, but improves next-word accuracy by only about one to two points: the distributions are better calibrated, while the single most likely word is often the same. Perplexity does not mean that generated sentences are meaningful; the corpora are noisy social-media text. Tanglish normalization was checked against 68 hand-written reference pairs and reached 97.1% exact match (WER 1.4%), but those references were written with the same dictionary conventions as the system, so this is a consistency check and **not** independent evidence of generalization. Cross-variant checks show that the models do not transfer reliably: the Hinglish identifier reaches {pc(lh['cross_variant']['hinglish_model']['accuracy'])} on Hinglish but {pc(lh['cross_variant']['tanglish_model']['accuracy'])} on Tanglish text, and the Hinglish model scores {pc(lt['cross_variant']['hinglish_model']['accuracy'])} on the Tanglish gold set (English versus native-language words only).")
H3("Testing and reproducibility status")
P("The 25 unit and integration tests in `tests/` pass (`python -m pytest tests -q`). The stages of the pipeline (data preparation, language identification, normalization, language models) were each run and their results saved separately. The one-command rebuild `python -m src.run_all` has **not** been verified end to end in a single clean run, and the training pipeline was modified after some stages had been run; the reported numbers come from the saved result files, not from a fresh full rebuild. No model was retrained while preparing this report.")

# ---- 4.2 coding
H2("4.2 Coding", "toc_code", page_before=True)
P("Short, representative excerpts of the actual source files are shown below (the complete code is not reproduced). Each excerpt is copied unchanged from the repository.", after=2, line=1.15)
CODE_CAP("Listing 1. Typed tokenizer and deterministic labels for non-word tokens (src/preprocessing.py).")
import re as _re
pre = (ROOT / "src/preprocessing.py").read_text()
m = _re.search(r"_TOKEN_RE = re\.compile\(.*?re\.VERBOSE \| re\.UNICODE,\n\)", pre, _re.S)
CODE(m.group(0) + "\n\n" + source_of("src/preprocessing.py", "rule_label"))
CODE_CAP("Listing 2. Model A: character n-gram TF-IDF + Logistic Regression (src/language_identifier.py).")
_ind = lambda s_: "\n".join("    " + l if l else l for l in s_.split("\n"))
CODE("class CharNgramLID:\n" + _ind(source_of("src/language_identifier.py", "CharNgramLID._x")) + "\n\n" + _ind(source_of("src/language_identifier.py", "CharNgramLID.fit")))
CODE_CAP("Listing 3. Model B: word embedding + character CNN + BiLSTM tagger (src/language_identifier.py).")
CODE(source_of("src/language_identifier.py", "_Tagger"))
CODE_CAP("Listing 4. Merging rule labels with model predictions (src/language_identifier.py).")
CODE(source_of("src/language_identifier.py", "identify_tokens"))
CODE_CAP("Listing 5. Weak (silver) labelling of Tanglish words (src/silver_labels.py).")
CODE(source_of("src/silver_labels.py", "_lookup") + "\n\n" + source_of("src/silver_labels.py", "silver_label_word"))
CODE_CAP("Listing 6. Language-aware normalization of one word (src/normalizer.py).")
CODE(source_of("src/normalizer.py", "Normalizer._reduce_elongation") + "\n\n" + source_of("src/normalizer.py", "Normalizer._lookup") + "\n\n" + source_of("src/normalizer.py", "Normalizer.normalize_word"))
CODE_CAP("Listing 7. Interpolated Kneser-Ney trigram probability, vectorised (src/ngram_model.py).")
CODE(source_of("src/ngram_model.py", "KNTrigramLM._refresh_p1") + "\n\n" + source_of("src/ngram_model.py", "KNTrigramLM.prob_batch") + "\n\n" + source_of("src/ngram_model.py", "KNTrigramLM.tune_discount"))
CODE_CAP("Listing 8. LSTM language model definition and next-word prediction (src/lstm_model.py).")
CODE(source_of("src/lstm_model.py", "LSTMLM") + "\n\n" + source_of("src/lstm_model.py", "LSTMPredictor.predict_next"))
CODE_CAP("Listing 9. Core of the LSTM training step with checkpoint selection (src/lstm_model.py, excerpt of train_lstm).")
tl = source_of("src/lstm_model.py", "train_lstm").split("\n")
i0 = next(i for i, l in enumerate(tl) if l.strip().startswith("for bi, (_, batch)"))
i1 = next(i for i, l in enumerate(tl) if "loss.item() * n" in l)
j0 = next(i for i, l in enumerate(tl) if "val_nll, _ = evaluate_nll" in l)
j1 = next(i for i, l in enumerate(tl) if "bad += 1" in l)
CODE("\n".join(tl[i0:i1 + 1]) + "\n        ...\n" + "\n".join(tl[j0:j1 + 1]) + "\n    ...\n    model.load_state_dict(best_state)")
CODE_CAP("Listing 10. Word error rate and the top-k next-word protocol shared by both language models (src/evaluation.py).")
CODE(source_of("src/evaluation.py", "wer_cer") + "\n\n" + source_of("src/evaluation.py", "topk_accuracy_ngram"))
CODE_CAP("Listing 11. Single inference entry point used by the app, CLI and evaluation (src/pipeline.py, excerpt).")
an = source_of("src/pipeline.py", "CodeMixedPipeline.analyze").split("\n")
keep = [l for l in an if not l.strip().startswith("res.fluency") and "ctx_words" not in l]
a0 = next(i for i, l in enumerate(an) if "toks, labels, conf" in l)
a1 = next(i for i, l in enumerate(an) if "res.lstm = " in l)
CODE("def analyze(self, text, lid_model=\"char_ngram\", k=5):\n    ...\n" + "\n".join(an[a0:a1 + 1]) + "\n    ...  # fluency scoring and return")

# ---- 4.3 screenshots
H2("4.3 Output Screenshot", "toc_shots", page_before=True)
P("All figures below are genuine screenshots of the project's Streamlit dashboard, cropped only to remove empty margins and the navigation sidebar. Figures 7, 9, 10 and 12 are the files already stored in the project's `results/` folder; Figures 8, 11 and 13 were captured from the same running application (inference only, no retraining). No screenshot is generated or edited.", size=11, line=1.0, after=4)
from PIL import Image


def SHOT(img, cap, maxh=3.9, maxw=5.6):
    w, h = Image.open(A / img).size
    wi = min(maxw, maxh * w / h)
    FIG(img, wi, cap)


SHOT("shot_dashboard_ta_lid.png", "Figure 7. Main dashboard (Analyze page): example inputs, Tanglish sentence, auto-detected variant, code-mixing index and token-level language identification (naan[TA] inniku[TA] college[EN] ku[TA] late[EN] ah[TA] varen[TA]).", maxh=7.3, maxw=6.0)
SHOT("shot_hi_lid.png", "Figure 8. Hinglish input and token-level language identification (Hindi tags HI, English tags EN, punctuation by rule).", maxh=3.8, maxw=5.9)
SHOT("shot_ta_norm.png", "Figure 9. Text normalization of the Tanglish input *plsss send pannunga brooo*, giving *please send pannunga bro* (spelling only, no translation).", maxh=4.0)
SHOT("shot_ta_next.png", "Figure 10. Next-word prediction for the Tanglish sentence: trigram (Kneser-Ney) and LSTM top-5 suggestions side by side.", maxh=4.25)
SHOT("shot_hi_next.png", "Figure 11. Next-word prediction for the Hinglish sentence: trigram (Kneser-Ney) and LSTM top-5 suggestions side by side.", maxh=4.25)
SHOT("shot_eval.png", "Figure 12. Evaluation dashboard: headline results (Hinglish LID 96.3%, Tanglish LID 95.3%, normalization WER 10.1%, LSTM perplexity 142), per-class scores and confusion matrix, read from results/*.json (upper part of the page).", maxh=7.5, maxw=5.9)
SHOT("shot_about.png", "Figure 13. About page with the dataset split report (sentence counts per split), the LSTM and trigram evaluation and the data sources.", maxh=7.4, maxw=5.9)

# ----------------------------------------------------------------------------- 5. CONCLUSION
H1("CONCLUSION", "toc_conc")
H2("5.1 Summary of achievements")
P(f"The project implemented three core NLP tasks for Romanized Tanglish and Hinglish and combined them in one tested pipeline and dashboard. Token-level language identification reached {pc(LID['hi_a']['accuracy'])} (character n-gram) and {pc(LID['hi_b']['accuracy'])} (BiLSTM) word accuracy on the gold Hinglish test set, and {pc(LID['ta_a']['accuracy'])} and {pc(LID['ta_b']['accuracy'])} on the hand-reviewed Tanglish set. Rule- and dictionary-based normalization lowered Hinglish WER from {pc(N0['best_of_refs']['wer'])} to {pc(N2['best_of_refs']['wer'])} against the best of three references, at a level where the human annotators themselves differ by {pc(nm['hinglish']['inter_annotator']['mean_wer'])}. Two language models were built on identical data and compared under one protocol.")
H2("5.2 Statistical versus neural models")
P(f"The neural LSTM outperformed the from-scratch Kneser-Ney trigram model on every language-modelling metric: perplexity fell from {mh['ngram']['test_perplexity']:.0f} to {mh['lstm']['test_perplexity']:.0f} (Hinglish) and from {mt['ngram']['test_perplexity']:.0f} to {mt['lstm']['test_perplexity']:.0f} (Tanglish), top-1 accuracy rose by 1.2 to 1.3 points and top-5 accuracy by 2.1 to 2.2 points. The LSTM generalizes better to unseen word combinations because it shares information through embeddings, while the trigram model needs exact two-word contexts; the trigram model, however, trains in seconds to minutes, is exactly interpretable and still gives strong results. Character n-gram features alone were competitive for language identification: the BiLSTM was slightly better on Hinglish and not better on Tanglish, so sentence context added little there. These conclusions hold for these datasets and splits. Practically, code-mixed NLP of this kind supports spell-aware search, moderation and keyboard prediction for users who write in Romanized Indian languages.")
H2("5.3 Limitations")
P("The following limitations affect how the results should be interpreted:", after=2)
BUL([
    "**Tanglish identification labels are weak.** Training labels are generated by rules and curated dictionaries (silver labels), so the models can at best reproduce those rules; the 99% silver-test score is not generalization accuracy.",
    "**The Tanglish gold set is small and not independently validated.** It contains 120 sentences (1,327 word tokens) and was reviewed by the project team with an AI assistant, not by independent native-speaker annotators; about 4% of silver labels were corrected during review.",
    "**Names and acronyms are weak.** Recall of the OTHER class (names, acronyms) on the Tanglish gold set is about 46%.",
    "**Tanglish normalization is not independently validated.** Its 97% exact match uses 68 self-authored reference pairs derived with the same dictionary conventions as the system, so it is not evidence of generalization.",
    "**Vocabulary cap.** The Tanglish LSTM vocabulary is capped at 20,000 words (test out-of-vocabulary rate 10.9%) because a 30,000-word vocabulary caused training problems on the MPS backend of the development machine; the cap is applied to both models.",
    "**No cross-variant transfer.** The Hinglish and Tanglish models do not transfer reliably to each other's language mix and must be evaluated separately.",
    "**Rebuild not verified.** The one-command clean rebuild `python -m src.run_all` has not been verified end to end in a single run; individual stages were run separately and the training pipeline was modified afterwards.",
    "**Other caveats.** Near-duplicate Hinglish tweets can exist across the official train/test split, which slightly flatters all Hinglish models; the Tanglish LSTM trained for only 10 epochs; the discount grid optimum lies at its upper edge; perplexity does not guarantee meaningful generated text; and the corpora are noisy social-media text.",
], size=12)
H2("5.4 Future work")
BUL([
    "Native-speaker validation and expansion of the Tanglish gold data, with measured inter-annotator agreement.",
    "An independent Tanglish normalization benchmark with human references.",
    "Better handling of names, acronyms and other non-linguistic tokens.",
    "A larger vocabulary and more robust GPU training (for example sub-word or sampled-softmax output layers).",
    "Cross-language generalization studies, and additional Indic languages (for example Telugu, Kannada, Malayalam, Bengali).",
    "Language-aware normalization that uses context, and verification of the complete one-command rebuild.",
])

# ----------------------------------------------------------------------------- REFERENCES
H1("REFERENCES", "toc_refs")
refs = [
    "R. Sheth, H. Beniwal and M. Singh, \"COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing,\" in *Findings of the Association for Computational Linguistics: EMNLP 2025*, 2025 (ACL Anthology 2025.findings-emnlp.422; arXiv:2503.21670). Dataset: https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA (CC-BY-4.0).",
    "U. Barman, A. Das, J. Wagner and J. Foster, \"Code Mixing: A Challenge for Language Identification in the Language of Social Media,\" in *Proc. First Workshop on Computational Approaches to Code Switching*, Doha, Qatar, 2014, pp. 13-23. doi:10.3115/v1/W14-3902.",
    "B. Han and T. Baldwin, \"Lexical Normalisation of Short Text Messages: Makn Sens a #twitter,\" in *Proc. 49th Annual Meeting of the ACL*, Portland, OR, 2011, pp. 368-378. https://aclanthology.org/P11-1038/",
    "A. Pratapa, G. Bhat, M. Choudhury, S. Sitaram, S. Dandapat and K. Bali, \"Language Modeling for Code-Mixing: The Role of Linguistic Theory based Synthetic Data,\" in *Proc. 56th Annual Meeting of the ACL*, 2018, pp. 1543-1553. doi:10.18653/v1/P18-1143.",
    "B. R. Chakravarthi, V. Muralidaran, R. Priyadharshini and J. P. McCrae, \"Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text,\" in *Proc. 1st Joint Workshop on Spoken Language Technologies for Under-resourced Languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)*, LREC 2020, pp. 202-210. https://aclanthology.org/2020.sltu-1.28",
    "S. F. Chen and J. Goodman, \"An empirical study of smoothing techniques for language modeling,\" *Computer Speech & Language*, vol. 13, no. 4, pp. 359-394, 1999.",
    "R. Kneser and H. Ney, \"Improved backing-off for m-gram language modeling,\" in *Proc. IEEE ICASSP*, vol. 1, 1995.",
    "S. Hochreiter and J. Schmidhuber, \"Long short-term memory,\" *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.",
    "Y. Kim, Y. Jernite, D. Sontag and A. M. Rush, \"Character-aware neural language models,\" in *Proc. AAAI*, 2016. arXiv:1508.06615.",
    "O. Press and L. Wolf, \"Using the output embedding to improve language models,\" in *Proc. 15th Conf. of the European Chapter of the ACL (Short Papers)*, Valencia, 2017, pp. 157-163.",
    "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in *Proc. ICLR*, 2015. arXiv:1412.6980.",
    "Vishnu N., \"Tanglish-Corpus-185k,\" Hugging Face dataset (CC-BY-4.0). https://huggingface.co/datasets/vishnu-n/Tanglish-Corpus-185k",
    "A. Paszke et al., \"PyTorch: An imperative style, high-performance deep learning library,\" in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, 2019.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.",
]
for i, r_ in enumerate(refs, 1):
    p = doc.add_paragraph(); fmt(p, L, 0, 4, 1.15, left=0.4, first=-0.4)
    add_runs(p, f"[{i}]\t" + r_, TNR, 11)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))

# core properties
cp = doc.core_properties
cp.title = TITLE
cp.subject = "25AI2008 Multilingual AI - Skill Based Evaluation Report"
cp.author = ", ".join(n for n, _ in TEAM)
cp.keywords = "code-mixing, Tanglish, Hinglish, language identification, normalization, language model"

doc.save(OUT)
print("saved", OUT)
