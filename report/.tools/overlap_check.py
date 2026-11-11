r"""Text-overlap (plagiarism) self-check of the report against the source texts available locally:
the abstracts of the cited works (docs/references_abstracts.json), the dataset README files under
data/raw, and any extra text files given on the command line (e.g. the project brief).
Reports every run of >= N consecutive identical words (default 8). This is a self-check only; it
does not replace the institute's plagiarism software.
Run: python report/.tools/overlap_check.py [--n 8] [extra.txt|.md ...]
"""
import json
import re
import sys
from pathlib import Path

REP = Path(__file__).resolve().parents[1]
ROOT = REP.parent
args = sys.argv[1:]
N = int(args[args.index("--n") + 1]) if "--n" in args else 8
extra = [Path(a) for a in args if a.endswith((".md", ".txt", ".tex"))]


def latex_words(s):
    s = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in s.splitlines())
    s = s[s.index("\\section{Introduction}"):s.index("\\settocbibname")]
    s = re.sub(r"\\begin\{(equation|algorithmic)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\(cite|ref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", s)
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", s)
    return words(s)


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("-", " "))


def ngrams(ws, n):
    return {tuple(ws[i:i + n]): i for i in range(len(ws) - n + 1)}


report = latex_words((REP / "_body.tex").read_text(encoding="utf-8"))
grams = ngrams(report, N)
sources = {}
abstracts = json.loads((ROOT / "docs" / "references_abstracts.json").read_text(encoding="utf-8"))
for key, rec in abstracts.items():
    sources[f"abstract {key}"] = (rec.get("title") or "") + " " + (rec.get("abstract") or "")
for f in sorted((ROOT / "data" / "raw").rglob("*")):
    if f.suffix.lower() in (".md", ".txt") and f.stat().st_size < 2_000_000:
        sources[str(f.relative_to(ROOT))] = f.read_text(encoding="utf-8", errors="ignore")
for f in extra:
    sources[f.name] = f.read_text(encoding="utf-8", errors="ignore")

hits = 0
for name, text in sources.items():
    sw = words(text)
    common = sorted({grams[g] for g in ngrams(sw, N) if g in grams})
    if not common:
        continue
    # merge consecutive n-gram starts into spans
    spans, start, prev = [], common[0], common[0]
    for i in common[1:]:
        if i != prev + 1:
            spans.append((start, prev + N))
            start = i
        prev = i
    spans.append((start, prev + N))
    for a, b in spans:
        hits += 1
        print(f"[{name}] {b - a} words: {' '.join(report[a:b])}")
print(f"{len(sources)} source texts checked, report body {len(report)} words; "
      f"{hits} overlapping runs of >= {N} words")
