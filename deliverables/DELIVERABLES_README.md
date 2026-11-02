# Deliverables: 25AI2008 Multilingual AI, Skill Based Evaluation

Project: Code-Mixed (Tanglish/Hinglish) Text: Language Identification, Normalization and N-gram/LSTM Language Modelling
Team: Angel Sweety P (5051), Shally S (5078), K Kokila Dharshini (5037), Selvarani S (5088)

## Files

| File | What it is |
|---|---|
| `25AI2008_REGNO.docx` | Editable report, 37 pages (6 front matter + 31 numbered body pages). Coding = 4 pages (body pp. 20-23), output screenshots = 5 pages (pp. 24-28). |
| `25AI2008_REGNO.pdf` | Preview of the report (rendered with Apple Pages). |
| `25AI2008_REGNO_Review.pptx` | Editable review deck, exactly 11 slides, native shapes and native (editable) charts. |
| `25AI2008_REGNO_Review.pdf` | Preview of the deck (rendered with Keynote; see "Rendering notes"). |
| `screenshots/` | Four screenshots captured from the running dashboard for this report (Hinglish LID, Hinglish next-word, Hinglish normalization, About page) plus the Tanglish "plsss send pannunga brooo" Language-tags view used on slide 9. |
| `build/` | Scripts and assets used to generate everything (`make_figures.py`, `build_report.py`, `build_deck.py`, helpers). Re-run in that order to regenerate. |

**Rename** `REGNO` in both file names to your registration number (the brief says not to invent one, and a team of four has four numbers).

The original template PDFs on the Desktop were not touched. Nothing in `src/`, `models/`, `data/` or `results/` was modified, and no model was retrained or rebuilt.

## Placeholders you still need to fill (highlighted yellow in Word; shown as `<<...>>`)

- `<<Class Teacher>>` and `<<Designation>>`: bonafide certificate page and evaluation page signature boxes
- `<<Course Teacher>>`: declaration page
- `<<AI/AIML>>`: degree line of the bonafide certificate
- `<<Faculty Name, Designation>>`: slide 1
- Individual contributions table: report section 3.8 (`<<to be filled>>` for each member)
- Module owners: slides 4 and 7 (`<<to be filled by the team>>`)
- "Submitted for the assessment held on ____": bonafide page
- Signatures and the whole Evaluation Report marks table are blank by design.
- Registration numbers are the four numbers you supplied (5051, 5078, 5037, 5088). They look like partial numbers; replace them with full register numbers if needed (cover, title page, bonafide, declaration, evaluation signature box, slide 1).

## Things to check or decide

1. **Hinglish baseline WER is 13.6%, not 13.7%.** `results/normalization.json` stores 0.13647 (no normalization, best of 3 references), which rounds to 13.6%. The README and the brief say 13.7%. Both documents use 13.6% so they match the saved file. If your faculty has seen 13.7%, either say so or change the README.
2. **Who reviewed the Tanglish gold set.** The brief says "the developer"; the project README says "an AI assistant, not independent native-speaker annotators". The report states: reviewed by the project team with an AI assistant, not validated by independent native speakers. Edit this (report Sections 3.2 and 5.3, deck slides 5 and 10) if it is not accurate.
3. **Hardware** (Apple M4, 16 GB, macOS) was read from the machine this project sits on; "MPS backend" comes from the saved result files.
4. **Software versions** (Python 3.14.3, PyTorch 2.14.0, scikit-learn 1.9.1, NumPy 2.4.0, pandas 3.0.3, Streamlit 1.62.0) were read from the project's virtual environment.
5. **Saved vs current training settings.** `results/lm_hinglish.json` records `epochs: 40` (early-stopped at 25) while `train_lm.py` now defaults to 15; the report quotes the saved results. This is consistent with limitation 7 (the pipeline changed after stages were run).
6. **Literature.** COMI-LINGUA (Sheth et al.), Barman et al., Han and Baldwin, Pratapa et al., Chakravarthi et al., and Press and Wolf were checked online against ACL Anthology / arXiv records this session. Chen and Goodman, Kneser and Ney, Hochreiter and Schmidhuber, Kim et al., Kingma and Ba, PyTorch and scikit-learn are standard references cited from knowledge without a live lookup (no DOI is given for Kneser-Ney 1995). Please spot-check them before submission.

## Deliberate decisions about the template

- Cover, title page, bonafide, declaration, evaluation report and table of contents follow the template pages (logo banner, Arial headings, signature boxes). Bonafide and declaration text is Arial 12 as in the template; all report body text is Times New Roman 12 pt, 1.5 line spacing; the title is 18 pt and the team line 16 pt.
- In the template the "SCHOOL OF COMPUTER SCIENCE AND TECHNOLOGY" line and the "TABLE OF CONTENTS" heading were split from their pages (page-break artifacts). They are kept together with their pages here.
- The TOC keeps the template numbering (1 Abstract, 2 Problem Statement, 3 Methodology/Architecture, 4.1 Implementation, 4.2 Coding, 4.3 Output Screenshot, 5 Conclusion). One row was added, **6 References**, because the declaration requires cited sources. Dataset description is inside chapter 3; results are inside 4.1; limitations are inside the Conclusion (5.3).
- TOC page numbers are Word fields (PAGEREF) with cached values from the Pages render. If Word paginates differently, right-click the table, then Update Field.
- Code listings use Courier New 7.5 pt, single spacing, to stay within the 10-page limit; they are unmodified excerpts taken from the source files.

## Rendering notes

- Report PDF: exported from Apple Pages. Word may paginate slightly differently; no Microsoft Word or PowerPoint was available here, so **the files were not opened in Office**. Open both once in Word/PowerPoint and check page count and the TOC.
- Deck PDF: exported from Keynote using a temporary copy of the deck with Carlito/Caladea (metric-compatible substitutes for Calibri/Cambria), because Calibri and Cambria are not installed on this Mac. The delivered `.pptx` itself uses Calibri/Cambria as in the template. Yellow highlight on placeholders is not drawn by Keynote, so the PDF shows them without highlight.
- Carlito and Caladea were installed temporarily in `~/Library/Fonts` for the preview and have been removed.
- The Streamlit dashboard was started twice (inference only, local port 8601) to capture screenshots and was stopped afterwards.

## Not verified

- `python -m src.run_all` was not run, and the clean end-to-end rebuild is **not** verified (stated as limitation 7 in the report and on slide 10). The 25 unit/integration tests were run and passed.
- Screenshots of the older dashboard layout (`results/app_demo_tanglish.png`, `results/app_eval.png`) and the Fluency/Generate pages were deliberately not used.
