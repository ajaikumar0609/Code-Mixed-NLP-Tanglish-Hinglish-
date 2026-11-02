"""Small python-docx helper layer used by build_report.py."""
import ast
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor, Emu

ROOT = Path("/Users/ajaikumarn/Desktop/shally")
A = ROOT / "deliverables/build/assets"
TNR, ARIAL, MONO = "Times New Roman", "Arial", "Courier New"
J, C, L, R = (WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT)


def set_font(run, name=None, size=None, bold=None, italic=None, color=None, hl=False):
    if name:
        run.font.name = name
        rpr = run._r.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), name)
    if size: run.font.size = Pt(size)
    if bold is not None: run.font.bold = bold
    if italic is not None: run.font.italic = italic
    if color: run.font.color.rgb = RGBColor.from_string(color)
    if hl: run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def fmt(p, align=None, before=None, after=None, line=None, exact=None, keep_next=False, keep_lines=False, page_before=False, left=None, first=None):
    pf = p.paragraph_format
    if align is not None: pf.alignment = align
    if before is not None: pf.space_before = Pt(before)
    if after is not None: pf.space_after = Pt(after)
    if exact is not None:
        pf.line_spacing_rule = WD_LINE_SPACING.EXACTLY; pf.line_spacing = Pt(exact)
    elif line is not None:
        pf.line_spacing = line
    if keep_next: pf.keep_with_next = True
    if keep_lines: pf.keep_together = True
    if page_before: pf.page_break_before = True
    if left is not None: pf.left_indent = Inches(left)
    if first is not None: pf.first_line_indent = Inches(first)
    return p


_MARK = re.compile(r"(\*\*.+?\*\*|`.+?`|\*[^*\s][^*]*?\*|\[\[.+?\]\])")


def add_runs(p, text, name=TNR, size=12, bold=False, italic=False, color=None):
    """Mini markup: **bold**, *italic*, `code`, [[placeholder]] (yellow highlight)."""
    for part in _MARK.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); set_font(r, name, size, True, italic, color)
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1]); set_font(r, MONO, size - 1.5, bold, False, color)
        elif part.startswith("[[") and part.endswith("]]"):
            r = p.add_run("<<" + part[2:-2] + ">>"); set_font(r, name, size, bold, italic, color, hl=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); set_font(r, name, size, bold, True, color)
        else:
            r = p.add_run(part); set_font(r, name, size, bold, italic, color)
    return p


def add_field(p, instr, cached="1", name=TNR, size=11, bold=False):
    def mk(t):
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); return e
    r1 = p.add_run(); set_font(r1, name, size, bold); r1._r.append(mk("begin"))
    r2 = p.add_run(); set_font(r2, name, size, bold)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "; r2._r.append(it)
    r3 = p.add_run(); set_font(r3, name, size, bold); r3._r.append(mk("separate"))
    r4 = p.add_run(cached); set_font(r4, name, size, bold)
    r5 = p.add_run(); r5._r.append(mk("end"))


_bm_id = [100]


def bookmark(p, name):
    _bm_id[0] += 1
    s = OxmlElement("w:bookmarkStart"); s.set(qn("w:id"), str(_bm_id[0])); s.set(qn("w:name"), name)
    e = OxmlElement("w:bookmarkEnd"); e.set(qn("w:id"), str(_bm_id[0]))
    p._p.insert(1 if p._p.pPr is not None else 0, s)
    p._p.append(e)


def cell_borders(cell, sz=6, color="000000", sides=("top", "left", "bottom", "right")):
    tcPr = cell._tc.get_or_add_tcPr()
    b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.insert_element_before(b, "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")
    for s in sides:
        e = OxmlElement(f"w:{s}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
        b.append(e)


def cell_shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill)
    tcPr.insert_element_before(sh, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")


def cell_margins(cell, top=40, bottom=40, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    m = OxmlElement("w:tcMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tcPr.insert_element_before(m, "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")


def set_col_widths(table, widths_in):
    table.autofit = False
    tblPr = table._tbl.tblPr
    # table.autofit = False above already writes <w:tblLayout w:type="fixed"/>
    tw = tblPr.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW"); tblPr.insert_element_before(tw, "w:jc", "w:tblCellSpacing", "w:tblInd", "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook")
    tw.set(qn("w:w"), str(int(sum(widths_in) * 1440))); tw.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths_in):
        gc.set(qn("w:w"), str(int(w * 1440)))
    for row in table.rows:
        for i, w in enumerate(widths_in):
            if i < len(row.cells):
                row.cells[i].width = Inches(w)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader"); e.set(qn("w:val"), "true"); trPr.append(e)


def no_split(row):
    trPr = row._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit"); e.set(qn("w:val"), "true"); trPr.append(e)


def para_shade(p, fill="F4F5F7"):
    pPr = p._p.get_or_add_pPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), fill)
    pPr.insert_element_before(sh, 'w:tabs','w:suppressAutoHyphens','w:kinsoku','w:wordWrap','w:overflowPunct','w:topLinePunct','w:autoSpaceDE','w:autoSpaceDN','w:bidi','w:adjustRightInd','w:snapToGrid','w:spacing','w:ind','w:contextualSpacing','w:mirrorIndents','w:suppressOverlap','w:jc','w:textDirection','w:textAlignment','w:textboxTightWrap','w:outlineLvl','w:divId','w:cnfStyle','w:rPr','w:sectPr','w:pPrChange')


def para_border(p, sides=("bottom",), sz=8, color="000000"):
    pPr = p._p.get_or_add_pPr()
    b = OxmlElement("w:pBdr")
    for s in sides:
        e = OxmlElement(f"w:{s}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:space"), "1"); e.set(qn("w:color"), color)
        b.append(e)
    pPr.insert_element_before(b, 'w:shd','w:tabs','w:suppressAutoHyphens','w:kinsoku','w:wordWrap','w:overflowPunct','w:topLinePunct','w:autoSpaceDE','w:autoSpaceDN','w:bidi','w:adjustRightInd','w:snapToGrid','w:spacing','w:ind','w:contextualSpacing','w:mirrorIndents','w:suppressOverlap','w:jc','w:textDirection','w:textAlignment','w:textboxTightWrap','w:outlineLvl','w:divId','w:cnfStyle','w:rPr','w:sectPr','w:pPrChange')


def source_of(path, qual, drop_doc=False):
    """Exact source text of a function/class (``Class.method`` allowed) from a repo file."""
    import textwrap
    text = (ROOT / path).read_text()
    tree = ast.parse(text)
    parts = qual.split(".")
    scope = tree.body
    node = None
    for part in parts:
        node = next(n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        scope = node.body
    start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
    out = textwrap.dedent("\n".join(text.splitlines()[start:node.end_lineno]))
    if drop_doc:
        out = re.sub(r'^(\s*)""".*?"""\n', "", out, count=1, flags=re.S | re.M) if False else out
    return out
