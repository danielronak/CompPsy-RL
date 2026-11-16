"""Style pandoc's default reference.docx to IEEE conference conventions.
Usage: python3 refdoc.py ref_default.docx ref_ieee.docx
"""
import sys
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

TNR = "Times New Roman"
d = Document(sys.argv[1])

# US Letter, IEEE conference margins
s = d.sections[0]
s.page_width, s.page_height = Inches(8.5), Inches(11)
s.top_margin, s.bottom_margin = Inches(0.75), Inches(1.0)
s.left_margin = s.right_margin = Inches(0.625)


def set_fonts(rpr):
    """Hard-set Times New Roman, removing theme-font attributes that would override it."""
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for k in list(rf.attrib):
        del rf.attrib[k]
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), TNR)


# document defaults
rpr_default = d.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
set_fonts(rpr_default)


def style(name, size, align=None, indent=None, before=0, after=0,
          bold=False, italic=False, small_caps=False):
    st = next((x for x in d.styles if x.name == name), None)
    if st is None:
        raise KeyError(name)
    set_fonts(st.element.get_or_add_rPr())
    f = st.font
    f.size, f.bold, f.italic, f.small_caps = Pt(size), bold, italic, small_caps
    f.color.rgb = RGBColor(0, 0, 0)
    pf = st.paragraph_format
    if align is not None:
        pf.alignment = align
    pf.first_line_indent = indent
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0


J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
style("Normal", 10)
style("Body Text", 10, J, Inches(0.2))
style("First Paragraph", 10, J, Inches(0.2))
style("Compact", 8, C, None)
style("Heading 1", 10, C, None, before=8, after=4, small_caps=True)
style("Heading 2", 10, L, None, before=6, after=3, italic=True)
style("Image Caption", 8, J, None, before=2, after=4)
style("Table Caption", 8, C, None, before=8, after=3)
for nm in ("Captioned Figure", "Figure"):
    try:
        style(nm, 10, C, None, before=6, after=0)
    except KeyError:
        pass
for nm in ("Heading 1", "Heading 2"):
    next(x for x in d.styles if x.name == nm).paragraph_format.keep_with_next = True

d.save(sys.argv[2])
print("reference doc styled ->", sys.argv[2])
