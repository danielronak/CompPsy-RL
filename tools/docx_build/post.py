"""Post-process pandoc's docx into IEEE conference layout.
Usage: python3 post.py in.docx out.docx
"""
import copy, sys
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Emu, Twips

TITLE = ("Computational Analogues of Post-Decision and Self-Evaluative "
         "Biases in Reinforcement Learning Agents")
AUTHORS = [
    ("Ronak Daniel", ["Dept. of Artificial Intelligence", "MPSTME, NMIMS University"],
     "Mumbai, India", "ronak.daniel27@nmims.in"),
    ("Avinash Tandle", ["Dept. of Electronics and Telecommunication", "MPSTME, NMIMS University"],
     "Mumbai, India", "avinash.tandle@nmims.edu"),
]
COL_W = Inches(3.45)          # one IEEE column (3.5" minus a hair)
C = WD_ALIGN_PARAGRAPH.CENTER

d = Document(sys.argv[1])
body = d.element.body
first = d.paragraphs[0]


def run(p, text, size, bold=False, italic=False):
    r = p.add_run(text)
    r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
    r.font.name = "Times New Roman"
    return r


def fmt(p, align=C, before=0, after=0, indent=Inches(0)):
    pf = p.paragraph_format
    pf.alignment, pf.space_before, pf.space_after = align, Pt(before), Pt(after)
    pf.first_line_indent = indent


# ---- title ----
t = first.insert_paragraph_before(style="Normal")
fmt(t, before=0, after=12)
run(t, TITLE, 24)

# ---- author table (2 side-by-side blocks, no borders) ----
tbl = d.add_table(rows=1, cols=2)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
tblPr = tbl._tbl.tblPr
borders = OxmlElement("w:tblBorders")
for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
    e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "nil"); borders.append(e)
tblPr.append(borders)
for cell, (name, affil, city, email) in zip(tbl.rows[0].cells, AUTHORS):
    cell.width = Inches(3.6)
    p = cell.paragraphs[0]; fmt(p); run(p, name, 11)
    for line in affil:
        q = cell.add_paragraph(); fmt(q); run(q, line, 10, italic=True)
    for line in (city, email):
        q = cell.add_paragraph(); fmt(q); run(q, line, 10)
t._p.addnext(tbl._tbl)

# ---- section break: single-column header section, then two-column body ----
brk = OxmlElement("w:p")
tbl._tbl.addnext(brk)
final = body.sectPr
head_sect = copy.deepcopy(final)
brk_ppr = OxmlElement("w:pPr")
sp = OxmlElement("w:spacing"); sp.set(qn("w:before"), "0"); sp.set(qn("w:after"), "240")
brk_ppr.append(sp); brk_ppr.append(head_sect); brk.append(brk_ppr)   # 12pt gap before abstract


def set_cols(sectPr, num, space=360):
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        anchor = sectPr.find(qn("w:docGrid"))
        anchor.addprevious(cols) if anchor is not None else sectPr.append(cols)
    cols.set(qn("w:num"), str(num)); cols.set(qn("w:space"), str(space))


set_cols(head_sect, 1)
set_cols(final, 2, 360)                       # 0.25" gutter
d.sections[-1].start_type = WD_SECTION.CONTINUOUS

# ---- abstract + index terms: bold 9pt, lead-ins italic ----
in_abs = False
for p in d.paragraphs:
    txt = p.text
    if txt.startswith("Abstract—"):
        in_abs = True
    if in_abs:
        for r in p.runs:
            r.font.bold, r.font.size = True, Pt(9)
        if txt.startswith("Index Terms—"):
            p.paragraph_format.space_after = Pt(6)
            break

# ---- references: 8pt hanging indent ----
in_refs = False
for p in d.paragraphs:
    if p.style.name.startswith("Heading") and p.text.strip() == "References":
        in_refs = True
        continue
    if in_refs and p.text.strip():
        pf = p.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.3), Inches(-0.3)
        pf.space_after = Pt(2)
        for r in p.runs:
            r.font.size = Pt(8)

# ---- display equations: no first-line indent so they centre properly ----
for p in d.paragraphs:
    if p._p.find(".//" + qn("m:oMathPara")) is not None:
        p.paragraph_format.first_line_indent = Inches(0)

# ---- figures: scale every image to the column width, keep aspect ----
for inline in body.iter(qn("wp:inline")):
    ext = inline.find(qn("wp:extent"))
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    aspect = cx / cy
    w = Inches(2.4) if aspect < 1.3 else COL_W   # the near-square DQN bar chart stays compact
    h = int(w / aspect)
    ext.set("cx", str(int(w))); ext.set("cy", str(h))
    for a_ext in inline.iter(qn("a:ext")):
        if a_ext.get("cx") is not None:
            a_ext.set("cx", str(int(w))); a_ext.set("cy", str(h))

# ---- data tables: fixed column widths summing to one IEEE column, 8pt ----
WIDTHS = [[980, 560, 1110, 1110, 1208],   # Table I   (sum 4968 tw = 3.45")
          [1300, 1834, 1834],              # Table II
          [1242, 1242, 1242, 1242],        # Table III
          [1268, 925, 925, 925, 925]]      # Table IV
for table, widths in zip(d.tables[1:], WIDTHS):
    assert len(table.columns) == len(widths), (len(table.columns), widths)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tp = table._tbl.tblPr
    tw = tp.find(qn("w:tblW"))
    if tw is None:
        tw = OxmlElement("w:tblW"); tp.append(tw)
    tw.set(qn("w:type"), "dxa"); tw.set(qn("w:w"), str(sum(widths)))
    mar = tp.find(qn("w:tblCellMar"))
    if mar is None:
        mar = OxmlElement("w:tblCellMar"); tp.append(mar)
    for side in ("left", "right"):
        e = mar.find(qn(f"w:{side}"))
        if e is None:
            e = OxmlElement(f"w:{side}"); mar.append(e)
        e.set(qn("w:w"), "50"); e.set(qn("w:type"), "dxa")
    for gc, w in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(w))
    for row in table.rows:
        for i, (cell, w) in enumerate(zip(row.cells, widths)):
            cell.width = Twips(w)
            for p in cell.paragraphs:
                p.paragraph_format.first_line_indent = Inches(0)
                p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else C
                for r in p.runs:
                    r.font.size = Pt(8)

# ---- inline/display math: size it like the surrounding text; tidy glyphs ----
def para_half_points(p):
    for r in p.runs:
        if r.font.size:
            return int(r.font.size.pt * 2)
    st = p.style
    while st is not None:
        if st.font.size:
            return int(st.font.size.pt * 2)
        st = st.base_style
    return 20

for p in d.paragraphs:
    mruns = list(p._p.iter(qn("m:r")))
    if not mruns:
        continue
    hp = str(para_half_points(p))
    for mr in mruns:
        for mt in mr.findall(qn("m:t")):
            txt = (mt.text or "").replace("​", "")
            mt.text = "∗" if txt == "*" else txt          # LaTeX ^* -> asterisk operator
        mts = mr.findall(qn("m:t"))
        if mts and all(not (mt.text or "") for mt in mts):
            mr.getparent().remove(mr)                    # empty run left by a {} group
            continue
        rpr = mr.find(qn("w:rPr"))
        if rpr is None:
            rpr = OxmlElement("w:rPr")
            mrpr = mr.find(qn("m:rPr"))
            (mrpr.addnext(rpr) if mrpr is not None else mr.insert(0, rpr))
        for tag in ("w:sz", "w:szCs"):
            e = rpr.find(qn(tag))
            if e is None:
                e = OxmlElement(tag); rpr.append(e)
            e.set(qn("w:val"), hp)

# ---- schema-order normalisation (pandoc/texmath + our table borders) ----
def reorder(parent, order):
    rank = {qn(t): i for i, t in enumerate(order)}
    kids = list(parent)
    kids.sort(key=lambda e: rank.get(e.tag, len(order)))
    for k in kids:
        parent.remove(k)
    for k in kids:
        parent.append(k)

TBLPR = ["w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual", "w:tblStyleRowBandSize",
         "w:tblStyleColBandSize", "w:tblW", "w:jc", "w:tblCellSpacing", "w:tblInd",
         "w:tblBorders", "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
         "w:tblCaption", "w:tblDescription", "w:tblPrChange"]
for el in body.iter(qn("w:tblPr")):
    reorder(el, TBLPR)
for el in body.iter(qn("m:dPr")):
    reorder(el, ["m:begChr", "m:sepChr", "m:endChr", "m:grow", "m:shp", "m:ctrlPr"])
for el in body.iter(qn("m:rPr")):
    if el.find(qn("m:nor")) is not None:          # nor and sty are mutually exclusive
        for sty in el.findall(qn("m:sty")):
            el.remove(sty)
    reorder(el, ["m:lit", "m:nor", "m:scr", "m:sty", "m:brk", "m:aln"])
for el in body.iter(qn("m:mcPr")):
    reorder(el, ["m:count", "m:mcJc"])
for pm in body.iter(qn("w:pgMar")):
    for attr, val in (("w:header", "720"), ("w:footer", "720"), ("w:gutter", "0")):
        if pm.get(qn(attr)) is None:
            pm.set(qn(attr), val)

d.core_properties.title = TITLE
d.core_properties.author = "Ronak Daniel; Avinash Tandle"
d.save(sys.argv[2])
print("saved", sys.argv[2])
