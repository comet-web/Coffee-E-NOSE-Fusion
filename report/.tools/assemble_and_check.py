"""Assemble report/MiniProject2_Report.tex (template preamble verbatim + body) and run static
LaTeX checks, since no TeX engine is available locally.
Run: python report/.tools/assemble_and_check.py
"""
import re
from collections import Counter
from pathlib import Path

REP = Path(__file__).resolve().parents[1]
TEMPLATE = REP / "template_original" / "MiniProjectTemplate" / "MiniProjectTemplate.tex"
OUT = REP / "MiniProject2_Report.tex"

t = TEMPLATE.read_text(encoding="utf-8")
pre = t[: t.index("\\begin{document}")]
body = (REP / "_body.tex").read_text(encoding="utf-8")
# The only addition to the template preamble: hyperref (loaded last, as its manual requires) so that
# citations, cross-references and contents entries are clickable, plus DOI links in the references.
HYPERREF = r"""
% ---- Added after the template preamble: clickable citations, cross-references and contents
% ---- entries. hyperref must be the last package loaded (hyperref manual).
\definecolor{linkblue}{rgb}{0.05,0.2,0.55}
\usepackage[colorlinks=true, linkcolor=linkblue, citecolor=linkblue, urlcolor=linkblue,
            linktoc=all, bookmarksnumbered=true,
            pdftitle={Data Fusion and Ablation Study for Coffee Quality Assessment Using Electronic Nose Data},
            pdfauthor={Sai Aravind Chabala, Chevireddy Pothan Sai Reddy, Lingala Sahith Akash Manikanta}]{hyperref}
% DOIs in the reference list link to the paper (plainnat's .bbl only \providecommand's \doi).
\newcommand{\doi}[1]{doi: \href{https://doi.org/#1}{\urlstyle{rm}\nolinkurl{#1}}}

"""
OUT.write_text(pre + HYPERREF + body, encoding="utf-8")
print("assembled", OUT.name, f"({len((pre + HYPERREF + body).splitlines())} lines)")

problems = []
doc = pre + HYPERREF + body
inputs = {m: (REP / "tables" / (m.split("/")[-1] + ".tex")).read_text(encoding="ascii")
          for m in re.findall(r"\\input\{(tables/[^}]+)\}", body)}
full_body = body + "\n".join(inputs.values())


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in s.splitlines())


b = strip_comments(full_body)

# 1. ASCII only (pdfLaTeX without extra Unicode setup)
bad = sorted({(i + 1, ch) for i, line in enumerate(full_body.splitlines()) for ch in line if ord(ch) > 127})
if bad:
    problems.append(f"non-ASCII characters: {bad[:10]}")

# 2. environment balance
begins = Counter(re.findall(r"\\begin\{([^}]+)\}", b))
ends = Counter(re.findall(r"\\end\{([^}]+)\}", b))
for env in set(begins) | set(ends):
    if begins[env] != ends[env]:
        problems.append(f"environment {env}: {begins[env]} begin vs {ends[env]} end")

# 3. brace balance (escaped braces removed)
bb = re.sub(r"\\[{}]", "", b)
if bb.count("{") != bb.count("}"):
    problems.append(f"brace imbalance: {bb.count('{')} {{ vs {bb.count('}')} }}")

# 4. citations vs bibliography
bib = (REP / "Myreferences.bib").read_text(encoding="ascii")
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set(k.strip() for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", b) for k in grp.split(","))
if cites - bibkeys:
    problems.append(f"citations without bib entry: {sorted(cites - bibkeys)}")
print(f"citations: {len(cites)} keys used, {len(bibkeys)} in bib; uncited (dropped by BibTeX): {sorted(bibkeys - cites)}")

# 5. labels vs refs
labels = re.findall(r"\\label\{([^}]+)\}", b)
dup = [l for l, c in Counter(labels).items() if c > 1]
if dup:
    problems.append(f"duplicate labels: {dup}")
refs = set(re.findall(r"\\ref\{([^}]+)\}", b))
if refs - set(labels):
    problems.append(f"refs without label: {sorted(refs - set(labels))}")
unused = set(labels) - refs
print(f"labels: {len(labels)}; never referenced: {sorted(unused)}")

# 6. files
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", b):
    if not (REP / g).exists():
        problems.append(f"missing graphic: {g}")
for m in re.findall(r"\\input\{([^}]+)\}", b):
    if not (REP / (m + ".tex")).exists():
        problems.append(f"missing input: {m}")

# 7. every float has caption + label (label after caption)
for env in ["table", "figure"]:
    for blk in re.findall(r"\\begin\{" + env + r"\}(.*?)\\end\{" + env + r"\}", b, re.S):
        if "\\includegraphics" in blk and "iiIT-Dharwad" in blk:
            continue  # title-page logo
        if env == "table" and "Roll No" in blk:
            continue  # certificate table (as in the template)
        if "\\caption" not in blk or "\\label" not in blk:
            problems.append(f"{env} without caption/label: {blk.strip()[:60]!r}")
        elif blk.index("\\label") < blk.index("\\caption"):
            problems.append(f"{env} label before caption: {blk.strip()[:60]!r}")

# 8. bare special characters outside math / tabular / align
def outside_math(s):
    s = re.sub(r"\\begin\{(equation|align|algorithmic)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    return s


txt = outside_math(b)
# file-name arguments may legitimately contain underscores
txt = re.sub(r"\\(includegraphics(\[[^\]]*\])?|input|label|ref|cite[pt]?|bibliography)\{[^}]*\}", " ", txt)
nontab = re.sub(r"\\begin\{(tabular|tabularx)\}.*?\\end\{\1\}", " ", txt, flags=re.S)
for ch, name in [("_", "underscore"), ("^", "caret")]:
    hits = [m.start() for m in re.finditer(re.escape(ch), txt) if txt[m.start() - 1] != "\\"]
    ctx = [txt[max(0, h - 30): h + 10].replace("\n", " ") for h in hits]
    ctx = [c for c in ctx if "\\label{" not in c and "\\ref{" not in c and "\\cite" not in c and ".png" not in c
           and "\\input{" not in c and "\\includegraphics" not in c]
    if ctx:
        problems.append(f"bare {name} outside math: {ctx[:5]}")
amp = [m.start() for m in re.finditer("&", nontab) if nontab[m.start() - 1] != "\\"]
if amp:
    problems.append(f"bare & outside tabular: {[nontab[max(0, a - 30):a + 10] for a in amp[:5]]}")
pct = [m.start() for m in re.finditer(r"(?<=\d)%", strip_comments(full_body))]
if pct:
    problems.append(f"unescaped % after a number: {len(pct)}")

# 9. template structure preserved
for sec in ["\\section{Introduction}", "\\section{Related Work}", "\\section{Data and Methods}",
            "\\section{Results and Discussions}", "\\section{Conclusion}", "\\bibliography{Myreferences}",
            "\\tableofcontents", "\\listoffigures", "\\listoftables", "Certificate"]:
    if sec not in body:
        problems.append(f"template element missing: {sec}")
if pre != t[: t.index("\\begin{document}")]:
    problems.append("preamble differs from template")

print("PROBLEMS:" if problems else "all checks passed")
for p in problems:
    print("  -", p)
print("words in body (approx.):", len(re.sub(r"\\[a-zA-Z]+|[{}$]", " ", strip_comments(body)).split()))
