"""Turn the IEEEtran paper.tex into pandoc-friendly LaTeX.

Citation numbers and Table/Fig numbers are taken from the compiled paper.aux,
and the reference list from paper.bbl, so the Word file matches the PDF exactly.
Usage: python3 prep.py <latex_dir> <out.tex>
"""
import re, sys
from pathlib import Path

src = Path(sys.argv[1])
tex = (src / "paper.tex").read_text()
aux = (src / "paper.aux").read_text()
bbl = (src / "paper.bbl").read_text()

# ---------- maps from aux ----------
cite = {}   # key -> (num, short author)
for m in re.finditer(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}\{[^}]*\}\{\{([^}]*)\}\}", aux):
    cite[m.group(1)] = (int(m.group(2)), m.group(3).replace("~", " "))
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
    labels[m.group(1)] = m.group(2)

def fmt_nums(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}--{nums[j]}")
        else:
            out.extend(str(n) for n in nums[i:j + 1])
        i = j + 1
    return "[" + ", ".join(out) + "]"

def citep(m):
    return fmt_nums([cite[k.strip()][0] for k in m.group(1).split(",")])

def citet(m):
    parts = []
    for k in m.group(1).split(","):
        n, a = cite[k.strip()]
        parts.append(f"{a} [{n}]")
    return ", ".join(parts)

# ---------- body ----------
body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
body = re.sub(r"(?m)^\s*%.*$", "", body)                 # drop comment lines
body = body.replace(r"\maketitle", "")

body = re.sub(r"\\citep\{([^}]+)\}", citep, body)
body = re.sub(r"\\citet\{([^}]+)\}", citet, body)
body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], body)

# abstract + keywords -> IEEE lead-in paragraphs (styled later)
def abstract(m):
    paras = [p.strip() for p in m.group(1).strip().split("\n\n") if p.strip()]
    paras[0] = r"\textit{Abstract}---" + paras[0]
    return "\n\n".join(paras) + "\n\n"
body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", abstract, body, flags=re.S)
body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
              lambda m: r"\textit{Index Terms}---" + m.group(1).strip() + "\n\n", body, flags=re.S)

# IEEE section numbering (I., II. ...; A., B. ...)
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
n_sec = n_sub = 0
def number_heading(m):
    global n_sec, n_sub
    kind, star, title = m.group(1), m.group(2), m.group(3)
    if star:
        return "\\" + kind + "*{" + title + "}"
    if kind == "section":
        n_sec += 1; n_sub = 0
        return "\\section*{" + ROMAN[n_sec - 1] + ". " + title + "}"
    n_sub += 1
    return "\\subsection*{" + "ABCDEFGHIJ"[n_sub - 1] + ". " + title + "}"
body = re.sub(r"\\(section|subsection)(\*?)\{([^}]*)\}", number_heading, body)

# tables: tabularx -> tabular, strip layout-only commands
body = re.sub(r"\\begin\{tabularx\}\{[^}]*\}\{@\{\}([^@]*)@\{\}\}",
              lambda m: r"\begin{tabular}{" + m.group(1).replace("L", "l") + "}", body)
body = body.replace(r"\end{tabularx}", r"\end{tabular}")
for cmd in [r"\raggedright", r"\small", r"\footnotesize", r"\centering"]:
    body = body.replace(cmd + "\n", "\n")
body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)

# math inside table cells -> formatted text (Word equation objects don't wrap in narrow cells)
SPECIAL = {
    r"C_{\text{global}}": r"\textit{C}\textsubscript{global}",
    r"\hat{H}(\pi)": r"\textit{Ĥ}(\textit{π})",
    r"\mathcal{G}": r"\textit{G}",
}
GREEK = {r"\kappa": r"\textit{κ}", r"\lambda": r"\textit{λ}", r"\pi": r"\textit{π}", r"\Delta": "Δ"}

def cell_math(m):
    s = m.group(1)
    if s in SPECIAL:
        return SPECIAL[s]
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = s.replace(r"\pm", "±").replace(r"\times", "×")
    s = re.sub(r"^([A-Za-z])(?=$|=)", r"\\textit{\1}", s)     # single-letter variable
    s = s.replace("=", " = ").replace(",", ", ").replace("-", "−")
    assert not re.search(r"\\(?!%|textit|textsubscript)[A-Za-z]", s), f"unconverted table math: {s}"
    return s

body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}",
              lambda t: re.sub(r"\$([^$]+)\$", cell_math, t.group(0)).replace("\\ ", " "),
              body, flags=re.S)   # also: "\ " (non-breaking in Word) -> normal space in cells

# captions: prefix TABLE I. / Fig. 1. from the aux label numbers
def caption_with_label(env_body, kind):
    lab = re.search(r"\\label\{([^}]+)\}", env_body)
    if not lab:
        return env_body
    num = labels[lab.group(1)]
    prefix = f"TABLE {num}. " if kind == "table" else f"Fig. {num}. "
    return env_body.replace(r"\caption{", r"\caption{" + prefix, 1)
body = re.sub(r"(\\begin\{table\}.*?\\end\{table\})",
              lambda m: caption_with_label(m.group(1), "table"), body, flags=re.S)

# DQN joint figure (subfigures) -> one figure, both images, main caption
def dqn(m):
    blk = m.group(0)
    cap = re.search(r"\\end\{subfigure\}\s*\\caption\{(.*)\}\s*\\label\{fig:dqn_joint\}", blk, re.S).group(1)
    # two stacked image paragraphs (side by side overflows one column); caption under the second
    return ("\\begin{figure}\n\\includegraphics{figures/dqn_buffer_ablation.png}\n\\end{figure}\n\n"
            "\\begin{figure}\n\\includegraphics{figures/dqn_cognitive_dissonance.png}\n"
            "\\caption{" + cap + "}\n\\label{fig:dqn_joint}\n\\end{figure}")
body = re.sub(r"\\begin\{figure\*\}(?:\[t\])?\s*(?:\\centering)?\s*\\begin\{subfigure\}.*?\\end\{figure\*\}",
              dqn, body, flags=re.S)
body = body.replace(r"\begin{figure*}", r"\begin{figure}").replace(r"\end{figure*}", r"\end{figure}")
# \makebox wrapper around the ChoiceRoom figure
body = re.sub(r"\\makebox\[\\linewidth\]\[c\]\{%\s*(\\includegraphics[^\n]*?)%\s*\}", r"\1", body)
# image options (trim/clip/width) are re-done in post-processing
body = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", body)
body = re.sub(r"\[(?:htbp|t|b|h|!t|!h)\]", "", body)
body = re.sub(r"(\\begin\{figure\}.*?\\end\{figure\})",
              lambda m: caption_with_label(m.group(1), "figure"), body, flags=re.S)

# aligned split -> two centred lines (no empty alignment cells; reads well in one column)
body = re.sub(r"\\begin\{split\}(.*?)\\end\{split\}",
              lambda m: r"\begin{gathered}" + m.group(1).replace("{}&", "").replace("&", "") + r"\end{gathered}",
              body, flags=re.S)

# number display equations (1), (2) ... like the PDF
eq = {"n": 0}
def eqnum(m):
    eq["n"] += 1
    return "\\qquad(" + str(eq["n"]) + ")\n\\end{equation}"
body = re.sub(r"\\end\{equation\}", eqnum, body)

# bibliography -> literal reference list from the .bbl
items = re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}(.*?)(?=\\bibitem|\\end\{thebibliography\})",
                   bbl, flags=re.S)
refs = []
for key, txt in items:
    t = txt.replace(r"\newblock", " ").replace(r"\penalty0", "")
    t = re.sub(r"\\doi\{([^}]*)\}", r"doi: \1", t)
    t = re.sub(r"\s+", " ", t).strip()
    refs.append(f"[{cite[key][0]}] {t}")
reflist = "\\section*{References}\n\n" + "\n\n".join(refs) + "\n"
body = re.sub(r"\{[^{}]*?\\setlength\{\\bibsep\}\{[^}]*\}\s*\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}\s*\}",
              lambda m: reflist, body, flags=re.S)
assert "References" in body, "bibliography block not replaced"

doc = (r"\documentclass{article}" "\n" r"\usepackage{amsmath,amssymb,graphicx,hyperref}" "\n"
       r"\begin{document}" "\n" + body + "\n" r"\end{document}" "\n")
Path(sys.argv[2]).write_text(doc)
print("refs:", len(refs), "| labels:", {k: v for k, v in labels.items() if k.startswith(("tab:", "fig:"))})
