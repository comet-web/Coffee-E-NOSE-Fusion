"""Extract Computer Modern glyph widths (1/1000 em) from the template's compiled PDF, so that the
page simulator can measure text exactly as pdfLaTeX sets it. Writes report/.tools/cm_metrics.json.
Run: python report/.tools/extract_font_metrics.py
"""
import json
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF = HERE.parent / "template_original" / "MiniProjectTemplate" / "MiniProjectTemplate.pdf"

fonts = {}
for page in PdfReader(PDF).pages:
    res = page.get("/Resources")
    if not res or "/Font" not in res:
        continue
    for f in res["/Font"].values():
        f = f.get_object()
        name = str(f["/BaseFont"]).split("+")[-1]
        first, widths = int(f["/FirstChar"]), [float(w) for w in f["/Widths"]]
        table = fonts.setdefault(name, {})
        for i, w in enumerate(widths):
            if w > 0:
                table[str(first + i)] = w
out = HERE / "cm_metrics.json"
out.write_text(json.dumps(fonts, indent=0, sort_keys=True), encoding="utf-8")
print({k: len(v) for k, v in fonts.items()}, "->", out.name)
