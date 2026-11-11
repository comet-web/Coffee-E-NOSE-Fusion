r"""Page-layout simulator for the report (no TeX engine is available locally).

It reproduces the geometry measured on the template's own compiled PDF
(template_original/MiniProjectTemplate/MiniProjectTemplate.pdf): 12 pt Computer Modern, a
17 cm x 20 cm text block, \onehalfspacing (17.93 bp between baselines, 31 lines on a full page)
and single-spaced floats. Text is broken into lines with the real glyph widths of the template's
fonts (cm_metrics.json, written by extract_font_metrics.py), figures use the real PNG aspect
ratios, tables are measured from their LaTeX source, and floats are placed with simplified LaTeX
rules: here/top on the current page if they fit, otherwise deferred to the next page, a float-only
page when deferred floats fill at least \floatpagefraction, and everything flushed at \clearpage.
Expect an accuracy of about +/- 1 page.

Run: python report/.tools/simulate_pages.py [--verbose] [body.tex]
"""
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REP = HERE.parent
ARGS = sys.argv[1:]
VERBOSE = "--verbose" in ARGS
BODY = REP / next((a for a in ARGS if a.endswith(".tex")), "_body.tex")

# ----------------------------------------------------------------------------- geometry (bp)
PT = 72 / 72.27
TEXTWIDTH = 17 / 2.54 * 72          # 481.89
TEXTHEIGHT = 20 / 2.54 * 72         # 566.93
TOPSKIP = 12 * PT                   # first baseline below the top of the text block
BASE = 17.93                        # measured baselineskip of body text
PARINDENT = 1 / 2.54 * 72
EM = 11.71                          # quad of cmr12 at 12 pt
FLOATSEP, TEXTFLOATSEP, INTEXTSEP = 8 * PT, 10 * PT, 8 * PT     # values set in _body.tex
TOPFRACTION, FLOATPAGEFRACTION, TOPNUMBER = 0.9, 0.75, 3
CAPSKIP = 8 * PT                    # \abovecaptionskip + \belowcaptionskip (4 pt each)
ROWBASE = {12: 14.5 * PT, 10.95: 13.6 * PT, 10: 12 * PT}       # single-spaced rows in floats
RULES = 13.3                        # \toprule + \midrule + \bottomrule incl. booktabs spacing
TABCOLSEP = 6 * PT
PROTRUDE, EXPAND = 1.5, 0.01        # microtype: margin kerning, usable glyph expansion

# ----------------------------------------------------------------------------- fonts
M = json.loads((HERE / "cm_metrics.json").read_text(encoding="utf-8"))


def _t(name):
    return {int(k): v for k, v in M.get(name, {}).items()}


RM12 = _t("CMR12")
RM12.update({k: v for k, v in {122: 435.2, 123: 489.6, 124: 979.2}.items() if k not in RM12})
BF12 = _t("CMBX12")
IT12 = _t("CMTI12")
for c, w in RM12.items():
    BF12.setdefault(c, w * 1.12)
    IT12.setdefault(c, w)
RM10 = _t("CMR10")
_CMR10 = {33: 277.8, 34: 500.0, 35: 833.3, 36: 500.0, 37: 833.3, 38: 777.8, 39: 277.8, 40: 388.9,
          41: 388.9, 42: 500.0, 43: 777.8, 44: 277.8, 45: 333.3, 46: 277.8, 47: 500.0, 58: 277.8,
          59: 277.8, 60: 277.8, 61: 777.8, 62: 472.2, 63: 472.2, 64: 777.8, 65: 750.0, 66: 708.3,
          67: 722.2, 68: 763.9, 69: 680.6, 70: 652.8, 71: 784.7, 72: 750.0, 73: 361.1, 74: 513.9,
          75: 777.8, 76: 625.0, 77: 916.7, 78: 750.0, 79: 777.8, 80: 680.6, 81: 777.8, 82: 736.1,
          122: 444.4, 123: 500.0, 124: 1000.0}
for d in range(48, 58):
    _CMR10[d] = 500.0
for k, v in _CMR10.items():
    RM10.setdefault(k, v)
BF10 = {c: w * 1.12 for c, w in RM10.items()}
BF10.update(_t("CMBX10"))
IT10 = {c: w * 0.98 for c, w in RM10.items()}
FONTS = {("rm", 12): RM12, ("bf", 12): BF12, ("it", 12): IT12,
         ("rm", 10): RM10, ("bf", 10): BF10, ("it", 10): IT10}
VOWELS = set("aeiouyAEIOUY")


class Ctx:
    """Font size context: 12 pt body text, 10.95 pt (\\small) or 10 pt (\\footnotesize)."""

    def __init__(self, size_pt):
        self.size = size_pt
        self.em = size_pt * PT
        self.key = 12 if size_pt >= 11.5 else 10
        self.space = (0.3264 if self.key == 12 else 0.3333) * self.em
        self.stretch, self.shrink = self.space / 2, self.space / 3


C12, C10, C11 = Ctx(12), Ctx(10), Ctx(10.95)


def glyphs(text, font, ctx):
    if font == "tt":
        return len(text) * 0.5145 * ctx.em
    t = FONTS[(font, ctx.key)]
    return sum(t.get(ord(ch), RM12.get(ord(ch), 500.0)) for ch in text) / 1000 * ctx.em


# ----------------------------------------------------------------------------- LaTeX helpers
CMD = re.compile(r"\\([A-Za-z]+)\*?|\\(.)", re.S)


def strip_comments(s):
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in s.splitlines())


def read_arg(s, j):
    """Return (content, end) of the braced argument (or single token) starting at s[j]."""
    while j < len(s) and s[j] in " \t\n":
        j += 1
    if j >= len(s):
        return "", j
    if s[j] != "{":
        if s[j] == "\\":
            m = CMD.match(s, j)
            return s[j:m.end()], m.end()
        return s[j], j + 1
    depth, k = 0, j
    while k < len(s):
        if s[k] == "\\":
            k += 2
            continue
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k], k + 1
        k += 1
    return s[j + 1:], len(s)


def skip_opt(s, j):
    m = re.match(r"\s*\[[^\]]*\]", s[j:])
    return j + m.end() if m else j


# ----------------------------------------------------------------------------- math widths (em)
REL_CMDS = {"leq", "geq", "le", "ge", "in", "approx", "sim", "propto", "rightarrow", "to", "neq",
            "leftrightarrow", "equiv", "mapsto", "subset", "notin", "ll", "gg", "Rightarrow"}
BIN_CMDS = {"pm", "times", "cdot", "mp", "circ", "otimes", "oplus", "setminus"}
FUNCS = {"max", "min", "arg", "log", "exp", "cos", "sin", "lim", "sup", "inf", "det", "Pr", "ln",
         "tanh", "argmax", "argmin", "vec"}
SYM = {"alpha": .64, "beta": .57, "gamma": .52, "delta": .45, "epsilon": .41, "varepsilon": .47,
       "zeta": .45, "eta": .5, "theta": .48, "kappa": .58, "lambda": .58, "mu": .6, "nu": .5,
       "xi": .47, "pi": .57, "rho": .52, "sigma": .57, "tau": .44, "phi": .6, "varphi": .65,
       "chi": .63, "psi": .65, "omega": .62, "Gamma": .61, "Delta": .82, "Theta": .76,
       "Lambda": .69, "Xi": .68, "Pi": .73, "Sigma": .71, "Phi": .71, "Psi": .76, "Omega": .71,
       "infty": 1.0, "partial": .53, "nabla": .83, "top": .78, "prime": .3, "ell": .42,
       "dots": 1.0, "ldots": 1.0, "cdots": 1.0, "sum": 1.06, "prod": 1.06, "int": .56,
       "langle": .39, "rangle": .39, "lfloor": .45, "rfloor": .45, "emptyset": .5, "mid": .28,
       "forall": .56, "exists": .56, "neg": .67, "star": .5, "ast": .5}
SKIP_CMDS = {"left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr", "displaystyle",
             "textstyle", "limits", "nolimits", "mathstrut"}


def roman_em(text, bold=False):
    t = BF12 if bold else RM12
    text = re.sub(r"\\[A-Za-z]+", "x", text).replace("{", "").replace("}", "").replace("~", " ")
    return sum(t.get(ord(ch), 500.0) for ch in text) / 1000 * 12 * PT / EM


def mw(s, sc=1.0, display=False):
    """Approximate width (em) of math material s at relative scale sc."""
    w, i, n, prev = 0.0, 0, len(s), None
    while i < n:
        c = s[i]
        if c.isspace() or c == "&":
            i += 1
            continue
        if c == "\\":
            m = CMD.match(s, i)
            name = m.group(1) or m.group(2)
            i = m.end()
            if name in SKIP_CMDS:
                continue
            if name in ("mathrm", "text", "textrm", "operatorname", "mathit", "textit", "mbox"):
                arg, i = read_arg(s, i)
                w += roman_em(arg) * sc
                prev = "ord"
            elif name in ("mathbf", "bm", "boldsymbol", "textbf"):
                arg, i = read_arg(s, i)
                w += roman_em(arg, bold=True) * sc
                prev = "ord"
            elif name in ("mathbb", "mathcal", "mathscr"):
                arg, i = read_arg(s, i)
                w += 0.72 * len(arg.strip()) * sc
                prev = "ord"
            elif name in ("frac", "dfrac", "tfrac"):
                a, i = read_arg(s, i)
                b, i = read_arg(s, i)
                inner = sc if display else sc * 0.72
                w += max(mw(a, inner, display), mw(b, inner, display)) + 0.24 * sc
                prev = "ord"
            elif name in ("hat", "bar", "tilde", "vec", "widehat", "overline", "underline", "dot"):
                arg, i = read_arg(s, i)
                w += mw(arg, sc, display)
                prev = "ord"
            elif name == "sqrt":
                arg, i = read_arg(s, i)
                w += mw(arg, sc, display) + 0.8 * sc
            elif name in FUNCS:
                w += (roman_em(name) + 0.17) * sc
                prev = "op"
            elif name in REL_CMDS:
                w += (0.78 + (0.556 if sc >= 1 else 0)) * sc
                prev = "rel"
            elif name in BIN_CMDS:
                unary = prev in (None, "rel", "open", "bin", "punct")
                w += (0.78 + (0 if unary or sc < 1 else 0.444)) * sc
                prev = "bin"
            elif name in (",", ";", ":", "!", "quad", "qquad"):
                w += {",": .167, ";": .278, ":": .222, "!": -.167, "quad": 1.0, "qquad": 2.0}[name] * sc
            elif name in ("{", "}", "|"):
                w += 0.5 * sc
                prev = "open" if name == "{" else "ord"
            elif name in ("%", "_", "#", "$"):
                w += 0.6 * sc
            else:
                w += (1.44 if display and name in ("sum", "prod") else SYM.get(name, 0.6)) * sc
                prev = "ord"
            continue
        if c in "_^":
            arg, i = read_arg(s, i + 1)
            sw = mw(arg, sc * 0.72, False)
            if i < n and s[i] in "_^" and s[i] != c:          # both sub- and superscript
                arg2, i = read_arg(s, i + 1)
                sw = max(sw, mw(arg2, sc * 0.72, False))
            w += sw
            continue
        if c == "{":
            arg, i = read_arg(s, i)
            w += mw(arg, sc, display)
            continue
        i += 1
        if c in "=<>":
            w += (0.78 + (0.556 if sc >= 1 else 0)) * sc
            prev = "rel"
        elif c in "+-":
            unary = prev in (None, "rel", "open", "bin", "punct")
            w += (0.78 + (0 if unary or sc < 1 else 0.444)) * sc
            prev = "bin"
        elif c in "([":
            w += (0.39 if c == "(" else 0.27) * sc
            prev = "open"
        elif c in ")]":
            w += (0.39 if c == ")" else 0.27) * sc
            prev = "ord"
        elif c in ",;":
            w += 0.45 * sc
            prev = "punct"
        elif c in "|.!'":
            w += 0.28 * sc
        elif c == "/":
            w += 0.5 * sc
        elif c.isdigit():
            w += 0.49 * sc
            prev = "ord"
        elif c.isalpha():
            w += IT12.get(ord(c), 500.0) / 1000 * 1.06 * sc * 12 * PT / EM
            prev = "ord"
        else:
            w += 0.5 * sc
    return w


def math_chunks(s):
    """Split inline math at top-level relations (where TeX may break a line). Widths in em."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = CMD.match(s, i)
            name = m.group(1) or m.group(2)
            i = m.end()
            if depth == 0 and name in REL_CMDS:
                parts.append(s[start:i])
                start = i
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and c in "=<>":
            parts.append(s[start:i + 1])
            start = i + 1
        i += 1
    parts.append(s[start:])
    parts = [p for p in parts if p.strip()]
    return [(mw(p), k < len(parts) - 1) for k, p in enumerate(parts)]


# ----------------------------------------------------------------------------- inline text
STYLE = {"textbf": "bf", "emph": "it", "textit": "it", "texttt": "tt", "textrm": "rm",
         "mbox": None, "text": None, "textsc": None, "makebox": None, "textnormal": "rm"}
IGNORE_WITH_ARG = {"label", "index", "vspace", "hspace", "phantom", "addcontentsline", "input",
                   "setlength", "pagenumbering", "thispagestyle", "settocbibname"}


def inline(s, ctx, font="rm"):
    """Tokenise inline LaTeX: ('box', w, text, font) ('kern', w) ('space',) ('tie',) ('brk',)."""
    seq, buf = [], []

    def flush():
        if buf:
            txt = "".join(buf)
            seq.append(("box", glyphs(txt, font, ctx), txt, font))
            buf.clear()

    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            m = CMD.match(s, i)
            j = m.end()
            if m.group(2) is not None:
                ch = m.group(2)
                if ch in "%&_#$":
                    buf.append(ch)
                elif ch == ",":
                    flush()
                    seq.append(("kern", 0.1667 * ctx.em))
                elif ch in " \n\\":
                    flush()
                    seq.append(("space",))
                    j = skip_opt(s, j) if ch == "\\" else j
                elif ch in "'`\"^~=.":
                    arg, j = read_arg(s, j)
                    base = arg.replace("\\i", "i").replace("{", "").replace("}", "").strip()
                    buf.append(base[:1] if base else "e")
                i = j
                continue
            name = m.group(1)
            if name in STYLE:
                arg, j = read_arg(s, j)
                flush()
                f2 = STYLE[name] or font
                if f2 == "it" and font == "it":
                    f2 = "rm"
                seq.extend(inline(arg, ctx, f2))
            elif name in ("cite", "citep", "citet"):
                j = skip_opt(s, j)
                arg, j = read_arg(s, j)
                flush()
                txt = "[" + ", ".join("00" for k in arg.split(",") if k.strip()) + "]"
                seq.append(("box", glyphs(txt, "rm", ctx), txt, "rm"))
            elif name in ("ref", "eqref", "pageref"):
                arg, j = read_arg(s, j)
                buf.append("4.7" if arg.startswith("sec") else "12")
            elif name in IGNORE_WITH_ARG:
                arg, j = read_arg(s, j)
            elif name in ("dots", "ldots"):
                buf.append("...")
            elif name == "textsuperscript":
                arg, j = read_arg(s, j)
                flush()
                seq.append(("kern", glyphs(arg, font, ctx) * 0.7))
            elif name in ("quad", "qquad"):
                flush()
                seq.append(("kern", ctx.em * (1 if name == "quad" else 2)))
            elif j < n and s[j] == "{":
                arg, j = read_arg(s, j)
                flush()
                seq.extend(inline(arg, ctx, font))
            i = j
        elif c == "$":
            j = s.index("$", i + 1)
            flush()
            for w, brk in math_chunks(s[i + 1:j]):
                seq.append(("box", w * ctx.em, "", "math"))
                if brk:
                    seq.append(("brk",))
            i = j + 1
        elif c == "~":
            flush()
            seq.append(("tie",))
            i += 1
        elif c.isspace():
            flush()
            if seq and seq[-1][0] != "space":
                seq.append(("space",))
            i += 1
        elif c in "{}":
            i += 1
        elif c == "-":
            if s.startswith("---", i):
                buf.append(chr(124))
                i += 3
            elif s.startswith("--", i):
                buf.append(chr(123))
                i += 2
            else:
                buf.append("-")
                flush()
                seq.append(("brk",))
                i += 1
        elif s.startswith("``", i):
            buf.append(chr(92))
            i += 2
        elif s.startswith("''", i):
            buf.append(chr(34))
            i += 2
        else:
            buf.append(c)
            i += 1
    flush()
    while seq and seq[-1][0] == "space":
        seq.pop()
    while seq and seq[0][0] == "space":
        seq.pop(0)
    return seq


def chunks_of(seq, ctx):
    """Group tokens into unbreakable chunks; each carries the glue that follows it."""
    out, cur = [], None

    def new():
        return {"w": 0.0, "g": 0.0, "st": 0.0, "sh": 0.0, "words": [], "glue": None}

    for tok in seq:
        if cur is None:
            cur = new()
        kind = tok[0]
        if kind == "box":
            cur["w"] += tok[1]
            cur["g"] += tok[1]
            cur["words"].append((tok[2], tok[3]))
        elif kind == "kern":
            cur["w"] += tok[1]
        elif kind == "tie":
            cur["w"] += ctx.space
            cur["st"] += ctx.stretch
            cur["sh"] += ctx.shrink
            cur["words"].append((" ", "rm"))
        elif kind in ("space", "brk"):
            if kind == "space":
                last = "".join(t for t, _ in cur["words"]).rstrip()
                sentence = bool(last) and last[-1] in ".?!" and not (len(last) > 1 and last[-2].isupper())
                extra = ctx.shrink if sentence else 0.0
                cur["glue"] = (ctx.space + extra, ctx.stretch * (3 if sentence else 1),
                               ctx.shrink / (3 if sentence else 1))
            else:
                cur["glue"] = (0.0, 0.0, 0.0)
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def hyph_points(word):
    return [p for p in range(2, len(word) - 2)
            if word[p] not in VOWELS and word[p + 1] in VOWELS and word[p].isalpha()]


def break_lines(seq, ctx, first_width, width, ragged=False):
    """Greedy line filling (close to TeX on wide lines). Returns (lines, natural width of last line)."""
    chunks = chunks_of(seq, ctx)
    if not chunks:
        return 0, 0.0
    lines, L = 1, first_width
    nat = st = sh = g = 0.0
    empty, glue = True, (0.0, 0.0, 0.0)
    for ch in chunks:
        gl = (0.0, 0.0, 0.0) if empty else glue
        cap = L + PROTRUDE + sh + gl[2] + ch["sh"] + (0 if ragged else EXPAND * (g + ch["g"]))
        if empty or nat + gl[0] + ch["w"] <= cap:
            nat += gl[0] + ch["w"]
            st += gl[1] + ch["st"]
            sh += gl[2] + ch["sh"]
            g += ch["g"]
            empty = False
        else:
            placed = False
            single = len(ch["words"]) == 1 and ch["words"][0][1] in ("rm", "it", "bf")
            word = ch["words"][0][0] if single else ""
            if not ragged and single and word.isalpha() and len(word) >= 6 and L - nat > 1.26 * (st + 0.02 * g):
                font = ch["words"][0][1]
                for p in sorted(hyph_points(word), reverse=True):
                    pre = glyphs(word[:p] + "-", font, ctx)
                    if nat + gl[0] + pre <= cap - ch["sh"]:
                        rest = glyphs(word[p:], font, ctx)
                        lines, L = lines + 1, width
                        nat, st, sh, g = rest, 0.0, 0.0, rest
                        placed = True
                        break
            if not placed:
                lines, L = lines + 1, width
                nat, st, sh, g = ch["w"], ch["st"], ch["sh"], ch["g"]
        glue = ch["glue"] or (0.0, 0.0, 0.0)
    return lines, nat


def text_width(s, ctx, font="rm"):
    total = 0.0
    for tok in inline(s, ctx, font):
        if tok[0] in ("box", "kern"):
            total += tok[1]
        elif tok[0] in ("space", "tie"):
            total += ctx.space
    return total


# ----------------------------------------------------------------------------- floats
def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def caption_parts(block):
    i = block.index("\\caption")
    j = i + len("\\caption")
    short = None
    if block[j] == "[":
        k, depth = j, 0
        while True:
            if block[k] == "[":
                depth += 1
            elif block[k] == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        short = block[j + 1:k]
        j = k + 1
    long, _ = read_arg(block, j)
    return short, long


def caption_height(text, label, newline):
    seq = inline(text, C10)
    if newline:   # table captions: label on its own line
        lines, _ = break_lines(seq, C10, TEXTWIDTH, TEXTWIDTH, ragged=True)
        lines += 1
    else:
        lines, _ = break_lines(inline(label + ". ", C10) + [("space",)] + seq, C10, TEXTWIDTH, TEXTWIDTH,
                               ragged=True)
    return lines * ROWBASE[10] + CAPSKIP


def split_cells(row):
    cells, depth, cur, i = [], 0, [], 0
    while i < len(row):
        c = row[i]
        if c == "\\" and i + 1 < len(row):
            cur.append(row[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "&" and depth == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur))
    return cells


def parse_colspec(spec):
    cols, i = [], 0
    while i < len(spec):
        c = spec[i]
        if c in "lcr":
            cols.append(("n", None))
            i += 1
        elif c in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            arg, i = read_arg(spec, i + 1)
            cm = float(re.match(r"([\d.]+)cm", arg).group(1))
            cols.append(("p", cm / 2.54 * 72))
        elif c == "X":
            cols.append(("X", None))
            i += 1
        elif c in ">@<" and i + 1 < len(spec) and spec[i + 1] == "{":
            _, i = read_arg(spec, i + 1)
        elif c == "*":
            num, i = read_arg(spec, i + 1)
            sub, i = read_arg(spec, i)
            cols.extend(parse_colspec(sub) * int(num))
        else:
            i += 1
    return cols


def measure_table(block):
    size = 10 if "\\footnotesize" in block else 10.95 if "\\small" in block else 12
    ctx = {10: C10, 10.95: C11, 12: C12}[size]
    inp = re.search(r"\\input\{([^}]+)\}", block)
    src = (REP / (inp.group(1) + ".tex")).read_text(encoding="utf-8") if inp else block
    m = re.search(r"\\begin\{(tabularx|tabular)\}", src)
    env = m.group(1)
    j = m.end()
    if env == "tabularx":
        _, j = read_arg(src, j)
    spec, j = read_arg(src, j)
    body = src[j:src.index("\\end{" + env + "}")]
    cols = parse_colspec(spec)
    rules = len(re.findall(r"\\(toprule|midrule|bottomrule)", body))
    cmid = len(re.findall(r"\\cmidrule", body))
    body = re.sub(r"\\(toprule|midrule|bottomrule)|\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
    rows = [r for r in re.split(r"\\\\(?:\[[^\]]*\])?", body) if r.strip()]
    fixed = sum(w for k, w in cols if k == "p") + 2 * TABCOLSEP * len(cols)
    n_x = sum(1 for k, _ in cols if k == "X")
    xw = (TEXTWIDTH - fixed) / n_x if n_x else 0
    natural = [0.0] * len(cols)
    height = 0.0
    for r in rows:
        cells = split_cells(r)
        lines_row = 1
        for ci, cell in enumerate(cells[:len(cols)]):
            cell = cell.strip()
            mc = re.match(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}$", cell, re.S)
            span = 1
            if mc:
                span, cell = int(mc.group(1)), mc.group(2)
            kind, w = cols[ci]
            if kind == "n":
                natural[ci] = max(natural[ci], text_width(cell, ctx) / span)
            else:
                width = w if kind == "p" else xw
                n, _ = break_lines(inline(cell, ctx), ctx, width, width, ragged=True)
                lines_row = max(lines_row, n)
        height += lines_row * ROWBASE[size]
    width = sum(natural[i] for i, (k, _) in enumerate(cols) if k == "n") + fixed + n_x * xw
    height += RULES * rules / 3 + 6 * cmid
    scale = 1.0
    if "adjustbox" in block and width > TEXTWIDTH:
        scale = TEXTWIDTH / width
    return height * scale, width, scale


def measure_algorithm(block):
    body = block[block.index("\\begin{algorithmic}"):block.index("\\end{algorithmic}")]
    h, depth = 0.0, 0
    for m in re.finditer(r"\\(REQUIRE|STATE|FOR|ENDFOR|RETURN|IF|ENDIF|ELSE)(\{[^\n]*\})?([^\n]*)", body):
        cmd, arg, rest = m.group(1), m.group(2) or "", m.group(3)
        if cmd in ("ENDFOR", "ENDIF"):
            depth -= 1
        text = {"REQUIRE": "Require: ", "FOR": "for  do", "RETURN": "return ", "IF": "if  then",
                "ENDFOR": "end for", "ENDIF": "end if", "ELSE": "else"}.get(cmd, "") + arg.strip("{}") + rest
        width = TEXTWIDTH - 2.0 * EM - depth * 1.0 * EM
        n, _ = break_lines(inline(text, C12), C12, width, width)
        h += max(n, 1) * ROWBASE[12]
        if cmd in ("FOR", "IF"):
            depth += 1
    return h + ROWBASE[10] + 12   # caption line between rules + rule spacing


def measure_float(env, spec, block, number):
    short, long = caption_parts(block)
    label = re.search(r"\\label\{([^}]+)\}", block).group(1)
    info = {"env": env, "spec": spec, "label": label, "short": short or long, "wide": None}
    if env == "figure":
        g = re.search(r"\\includegraphics\[width=([0-9.]*)\\textwidth\]\{([^}]+)\}", block)
        frac = float(g.group(1) or 1.0)
        pw, ph = png_size(REP / g.group(2))
        h = TEXTWIDTH * frac * ph / pw + caption_height(long, f"Figure {number}", newline=False)
    elif env == "table":
        th, width, scale = measure_table(block)
        info["wide"] = (width, scale)
        h = th + caption_height(long, f"Table {number}", newline=True)
    else:
        h = measure_algorithm(block) + caption_height(long, f"Algorithm {number}", newline=False) - CAPSKIP
    info["h"] = h
    return info


# ----------------------------------------------------------------------------- document stream
def display_increments(eq, short):
    h, d = 8.5, 3.5
    if "_" in eq:
        d = 5.0
    if re.search(r"\\(max|min|arg|sum|prod|lim)\w*_", eq):
        d = 9.0
    if "\\sum" in eq:
        h, d = 15.0, 12.0
    if "\\frac" in eq:
        h, d = max(h, 17.0), max(d, 9.0)
        if "\\sum" in eq or "e^{" in eq:
            h, d = 21.0, 15.0
    if "\\left" in eq:
        h, d = h + 2, d + 2
    above = (0.0 if short else 12 * PT) + max(BASE, 3.0 + h + 1)
    below = (6.5 * PT if short else 12 * PT) + max(BASE, d + 9.5)
    return above, below


def build_stream(src):
    s = strip_comments(src)
    s = s[s.index("\\section{Introduction}"):s.index("\\settocbibname")]
    blocks = []

    def keep(kind, payload, inline_mark=False):
        blocks.append((kind, payload))
        return f"\x01{len(blocks) - 1}\x01" if inline_mark else f"\n\n\x00{len(blocks) - 1}\x00\n\n"

    counters = {"figure": 0, "table": 0, "algorithm": 0}

    def float_sub(m):
        env = m.group(1)
        counters[env] += 1
        return keep("float", measure_float(env, m.group(2) or "", m.group(3), counters[env]))

    s = re.sub(r"\\begin\{(figure|table|algorithm)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", float_sub, s, flags=re.S)
    s = re.sub(r"\\begin\{(equation|align)\*?\}(.*?)\\end\{\1\*?\}",
               lambda m: keep("display", m.group(2), True), s, flags=re.S)
    s = re.sub(r"\\begin\{(itemize|enumerate)\}(\[[^\]]*\])?(.*?)\\end\{\1\}",
               lambda m: keep("list", m.group(3)), s, flags=re.S)
    s = re.sub(r"\\(section|subsection)\*?\{([^}]*)\}(\s*\\label\{[^}]*\})?",
               lambda m: keep(m.group(1), m.group(2)), s)
    s = re.sub(r"\\(clearpage|newpage)", lambda m: keep(m.group(1), None), s)

    stream, after_heading = [], False
    for para in re.split(r"\n\s*\n", s):
        para = para.strip()
        if not para:
            continue
        mb = re.fullmatch(r"\x00(\d+)\x00", para)
        if mb:
            kind, payload = blocks[int(mb.group(1))]
            if kind in ("section", "subsection"):
                stream.append(("heading", kind, payload))
                after_heading = True
            elif kind == "float":
                stream.append(("float", payload))
            elif kind == "list":
                items = [it for it in re.split(r"\\item(?:\[[^\]]*\])?", payload) if it.strip()]
                stream.append(("list", [paragraph(it, TEXTWIDTH - 2.5 * EM, 0, blocks) for it in items]))
                after_heading = False
            else:
                stream.append((kind,))
            continue
        noindent = after_heading or para.startswith("\\noindent")
        stream.append(("par", paragraph(para, TEXTWIDTH, 0 if noindent else PARINDENT, blocks)))
        after_heading = False
    return stream, counters


def paragraph(text, width, indent, blocks):
    """Return a list of vertical pieces: ('lines', n) and ('display', above, below)."""
    pieces, first, last_w = [], True, width
    for k, seg in enumerate(re.split(r"\x01(\d+)\x01", text)):
        if k % 2 == 1:
            eq = blocks[int(seg)][1]
            disp_w = mw(eq, 1.0, display=True) * EM
            short = last_w + 2 * EM < (width - disp_w) / 2
            pieces.append(("display",) + display_increments(eq, short))
            first = False
            continue
        if not seg.strip():
            continue
        n, last_w = break_lines(inline(seg, C12), C12, width - (indent if first else 0), width)
        if n:
            pieces.append(("lines", n))
        first = False
    return pieces


def to_boxes(stream):
    """Flatten into vertical boxes: dict(inc, top, brk) or floats / clearpage markers."""
    out = []
    pending_head = None      # (inc_to_first_line, lines that must stay together)
    list_end = False
    for item in stream:
        kind = item[0]
        if kind == "heading":
            if item[1] == "section":
                out.append({"inc": 50.0, "top": TOPSKIP, "brk": True, "what": "section " + item[2]})
                pending_head = 11.84 + BASE
            else:
                out.append({"inc": 16.73 + 22.26, "top": TOPSKIP, "brk": True, "what": "subsection " + item[2]})
                pending_head = 7.72 + BASE
            continue
        if kind in ("float", "clearpage", "newpage"):
            out.append({"marker": kind, "info": item[1] if kind == "float" else None})
            continue
        pieces_list = item[1] if kind == "list" else [item[1]]
        for pi, pieces in enumerate(pieces_list):
            first_line = True
            for piece in pieces:
                if piece[0] == "display":
                    out.append({"inc": piece[1], "top": piece[1], "brk": False, "what": "display"})
                    out.append({"inc": piece[2] - BASE, "top": 0.0, "brk": True, "what": "after-display",
                                "glue": True})
                    continue
                for li in range(piece[1]):
                    inc = BASE
                    brk = True
                    if first_line and kind == "list":
                        inc += 3 * PT if pi == 0 else 1 * PT
                    if list_end and first_line:
                        inc += 3 * PT
                    if pending_head == "club":
                        brk = False
                        pending_head = None
                    elif pending_head is not None:
                        inc, brk = pending_head, False
                        pending_head = "club"
                    out.append({"inc": inc, "top": TOPSKIP, "brk": brk, "what": "line"})
                    first_line = False
                    list_end = False
            if pending_head == "club":
                pending_head = None
        list_end = kind == "list"
    return out


# ----------------------------------------------------------------------------- page builder
def paginate(boxes):
    pages, deferred = [], []
    page = {"used": 0.0, "lines": 0, "floats": [], "text": False, "items": []}

    def close(pg):
        if pg["text"] or pg["floats"]:
            pages.append(pg)

    def float_pages(force):
        nonlocal deferred
        while deferred:
            take, h = [], 0.0
            for f in deferred:
                add = f["h"] + (FLOATSEP if take else 0)
                if h + add <= TEXTHEIGHT:
                    take.append(f)
                    h += add
                else:
                    break
            if not take:
                take, h = [deferred[0]], deferred[0]["h"]
            if force or h >= FLOATPAGEFRACTION * TEXTHEIGHT:
                pages.append({"used": h, "lines": 0, "floats": [f["label"] for f in take], "text": False,
                              "items": [], "floatpage": True})
                deferred = deferred[len(take):]
            else:
                break

    def new_page():
        nonlocal page, deferred
        close(page)
        float_pages(False)
        page = {"used": 0.0, "lines": 0, "floats": [], "text": False, "items": []}
        h, n = 0.0, 0
        while deferred and n < TOPNUMBER and h + deferred[0]["h"] <= TOPFRACTION * TEXTHEIGHT:
            f = deferred.pop(0)
            h += f["h"] + (FLOATSEP if n else 0)
            page["floats"].append(f["label"])
            n += 1
        page["used"] = h + (TEXTFLOATSEP if n else 0)

    held = []   # boxes on the current page since the last legal breakpoint
    for b in boxes:
        if "marker" in b:
            if b["marker"] == "float":
                f = b["info"]
                same_type_waiting = any(d["env"] == f["env"] for d in deferred)
                cost = f["h"] + (2 * INTEXTSEP if page["text"] else TEXTFLOATSEP)
                if not same_type_waiting and page["used"] + cost <= TEXTHEIGHT:
                    page["used"] += cost
                    page["floats"].append(f["label"])
                else:
                    deferred.append(f)
            elif b["marker"] == "clearpage":
                close(page)
                float_pages(True)
                page = {"used": 0.0, "lines": 0, "floats": [], "text": False, "items": []}
                held = []
            continue
        if b["brk"]:
            held = []          # a page break is allowed before this box
        if b.get("glue") and not page["text"]:
            continue
        inc = b["inc"] if page["text"] else b["top"]
        if page["used"] + inc > TEXTHEIGHT and page["text"]:
            carry = [] if b["brk"] else held
            if carry and sum(1 for hb, _ in carry if hb["what"] == "line") >= page["lines"]:
                carry = []     # the whole page is one unbreakable chain: break here anyway
            if carry:
                page["used"] -= sum(a for _, a in carry)
                page["lines"] -= sum(1 for hb, _ in carry if hb["what"] == "line")
                page["items"] = page["items"][:len(page["items"]) - len(carry)]
            new_page()
            held = []
            for cb in [hb for hb, _ in carry] + [b]:
                a = cb["top"] if not page["text"] else cb["inc"]
                page["used"] += a
                page["text"] = True
                page["lines"] += cb["what"] == "line"
                page["items"].append(cb["what"])
                held.append((cb, a))
            continue
        page["used"] += inc
        page["text"] = True
        page["lines"] += b["what"] == "line"
        page["items"].append(b["what"])
        held.append((b, inc))
    close(page)
    float_pages(True)
    return pages


# ----------------------------------------------------------------------------- front and back matter
def list_pages(entries, first, steps):
    """entries: list of (lines, step_before) ; returns pages needed."""
    pages, y = 1, first
    for k, (lines, step) in enumerate(entries):
        if k:
            y += step
        y += (lines - 1) * BASE
        if y > TEXTHEIGHT:
            pages += 1
            y = TOPSKIP + (lines - 1) * BASE
    return pages


def toc_pages(src):
    s = strip_comments(src)
    s = s[s.index("\\section{Introduction}"):]
    heads = re.findall(r"\\(section|subsection)\{([^}]*)\}", s)
    entries = [(1, 29.63), (1, 29.63)]           # List of Figures, List of Tables (tocbibind)
    for kind, title in heads:
        if kind == "section":
            w = TEXTWIDTH - 3.05 * EM
            n, _ = break_lines(inline(title, C12, "bf"), C12, w, w)
            entries.append((n, 29.63))
        else:
            w = TEXTWIDTH - 6.35 * EM
            n, _ = break_lines(inline(title, C12), C12, w, w)
            entries.append((n, BASE))
    entries.append((1, 29.63))                   # References
    return list_pages(entries, TOPSKIP + 29.76, None), sum(1 for k, _ in heads if k == "subsection")


def lof_pages(floats):
    entries = []
    for f in floats:
        w = TEXTWIDTH - 6.35 * EM
        n, _ = break_lines(inline(f["short"], C12), C12, w, w)
        entries.append((n, BASE))
    return list_pages(entries, TOPSKIP + 11.84 + BASE, None) if entries else 1


def parse_bib(text):
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        typ, key = m.group(1).lower(), m.group(2)
        j, depth = m.end(), 1
        k = j
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        body = text[j:k - 1]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            v, _ = read_arg(body, fm.end())
            fields.setdefault(fm.group(1).lower(), v)
        entries[key] = (typ, fields)
    return entries


def names(a):
    out = []
    for n in re.split(r"\s+and\s+", a):
        n = n.strip()
        if n.startswith("{") and n.endswith("}") and "," not in n.strip("{}"):
            out.append(n.strip("{}"))
            continue
        if "," in n:
            last, first = n.split(",", 1)
            out.append(first.strip() + " " + last.strip())
        else:
            out.append(n)
    if len(out) == 1:
        return out[0]
    if len(out) == 2:
        return f"{out[0]} and {out[1]}"
    return ", ".join(out[:-1]) + ", and " + out[-1]


def bib_text(typ, f):
    parts = [names(f.get("author", "Anon")) + ".", f.get("title", "") + "."]
    vol = f.get("volume", "") + (f"({f['number']})" if f.get("number") else "") + (
        f":{f['pages']}" if f.get("pages") else "")
    if typ == "article":
        parts.append("\\emph{" + f.get("journal", "") + "}, " + (vol + ", " if vol else "") + f.get("year", "") + ".")
    elif typ in ("inproceedings", "incollection"):
        pg = f", pages {f['pages']}" if f.get("pages") else ""
        parts.append("In \\emph{" + f.get("booktitle", "") + "}" + pg + ", " + f.get("year", "") + ".")
    elif typ == "book":
        parts[1] = "\\emph{" + f.get("title", "") + "}."
        parts.append(", ".join(x for x in [f.get("series", ""), f.get("publisher", ""), f.get("address", ""),
                                           f.get("year", "")] if x) + ".")
    else:
        parts.append(", ".join(x for x in [f.get("howpublished", ""), f.get("year", "")] if x) + ".")
    if f.get("note"):
        parts.append(f["note"] + ".")
    if f.get("doi"):
        parts.append("doi: " + f["doi"] + ".")
    elif f.get("url"):
        parts.append("URL \\texttt{" + f["url"] + "}.")
    return " ".join(parts)


def bib_pages(src):
    s = strip_comments(src)
    keys = []
    for grp in re.findall(r"\\cite[pt]?(?:\[[^\]]*\])?\{([^}]+)\}", s):
        for k in grp.split(","):
            if k.strip() not in keys:
                keys.append(k.strip())
    bib = parse_bib((REP / "Myreferences.bib").read_text(encoding="utf-8"))
    label = glyphs(f"[{len(keys)}]", "rm", C12) + 0.5 * EM
    w = TEXTWIDTH - label
    entries = []
    for k in keys:
        typ, f = bib[k]
        n, _ = break_lines(inline(bib_text(typ, f), C12), C12, w, w)
        entries.append((n, BASE + 2 * PT))
    return list_pages(entries, TOPSKIP + 11.84 + BASE + 0.2, None), len(keys)


# ----------------------------------------------------------------------------- main
def main():
    src = BODY.read_text(encoding="utf-8")
    stream, counters = build_stream(src)
    floats = [it[1] for it in stream if it[0] == "float"]
    boxes = to_boxes(stream)
    pages = paginate(boxes)
    toc, n_sub = toc_pages(src)
    lof = lof_pages([f for f in floats if f["env"] == "figure"])
    lot = lof_pages([f for f in floats if f["env"] == "table"])
    refs, n_refs = bib_pages(src)
    front = 1 + 1 + 1 + toc + lof + lot + 1
    print(f"front matter: title, blank, certificate, contents {toc} (subsections: {n_sub}), "
          f"figures {lof}, tables {lot}, blank -> {front} pages")
    sparse = []
    for k, p in enumerate(pages, 1):
        fill = p["used"] / TEXTHEIGHT
        heads = [it.split(" ", 1)[1] for it in p["items"] if it.startswith(("section", "subsection"))]
        kind = "FLOAT PAGE" if p.get("floatpage") else f"{p['lines']:2d} lines"
        if VERBOSE:
            print(f"  p{k:02d} {fill:5.0%} {kind:10s} floats={','.join(p['floats']) or '-':38s} "
                  f"{'; '.join(heads)[:60]}")
        nxt = pages[k] if k < len(pages) else None
        section_end = nxt is None or any(it.startswith("section") for it in nxt["items"][:1])
        if fill < 0.6 and not section_end:
            sparse.append(f"p{k} ({fill:.0%}, {kind.strip()})")
        if p.get("floatpage") and fill < 0.6:
            sparse.append(f"p{k} float page only {fill:.0%} full: {', '.join(p['floats'])}")
    ends = [f"p{k} {p['used'] / TEXTHEIGHT:.0%}" for k, p in enumerate(pages, 1)
            if k == len(pages) or any(it.startswith("section") for it in pages[k]["items"][:1])]
    print(f"main matter: {len(pages)} pages (section-end pages: {', '.join(ends)})")
    print(f"references: {refs} pages ({n_refs} cited entries)")
    for f in floats:
        if f["wide"]:
            width, scale = f["wide"]
            if scale < 1.0:
                print(f"  note: {f['label']} natural width {width:.0f} bp > text width, scaled to {scale:.2f}")
            elif width > TEXTWIDTH + 1:
                print(f"  WARNING: {f['label']} is {width - TEXTWIDTH:.0f} bp wider than the text block (overfull)")
    if sparse:
        print("  sparse pages:", "; ".join(sparse))
    print(f"ESTIMATED TOTAL: {front + len(pages) + refs} pages "
          f"({counters['figure']} figures, {counters['table']} tables, {counters['algorithm']} algorithms)")


if __name__ == "__main__":
    main()
