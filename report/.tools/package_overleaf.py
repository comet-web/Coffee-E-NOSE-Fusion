r"""Build report/MiniProject2_Report_Overleaf.zip with exactly the files the report uses:
the assembled .tex, the .bib, the logo, and every \includegraphics / \input target.
Run after assemble_and_check.py: python report/.tools/package_overleaf.py
"""
import re
import zipfile
from pathlib import Path

REP = Path(__file__).resolve().parents[1]
TEX = REP / "MiniProject2_Report.tex"
OUT = REP / "MiniProject2_Report_Overleaf.zip"

src = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in TEX.read_text(encoding="utf-8").splitlines())
files = [TEX.name, "Myreferences.bib"]
files += re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)
files += [m + ".tex" for m in re.findall(r"\\input\{([^}]+)\}", src)]
files = list(dict.fromkeys(files))
missing = [f for f in files if not (REP / f).exists()]
assert not missing, missing
tmp = OUT.with_suffix(".tmp")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(REP / f, f)
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
tmp.replace(OUT)
print(f"{OUT.name}: {len(files)} files, {OUT.stat().st_size / 1024:.0f} KB")
for f in files:
    print("  ", f)
