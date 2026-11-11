"""Verify literature references against Crossref (DOI lookup). Writes docs/references_verified.csv.
Only DOIs are sent to the API; no user data. Usage: python -I src/utils/verify_refs.py"""
import csv, json, sys, time, urllib.parse, urllib.request
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

DOIS = """10.3390/s100100036
10.1016/0925-4005(92)80033-T
10.1109/TIM.2002.808038
10.1016/s0925-4005(00)00499-8
10.1016/j.snb.2005.01.019
10.1016/j.aca.2015.04.042
10.1126/science.ade4401
10.1186/1471-2105-7-91
10.1145/2382577.2382579
10.1038/299352a0
10.1016/j.snb.2012.01.074
10.3390/s90705099
10.1109/5.554205
10.1109/JSEN.2012.2192920
10.1016/j.aca.2011.03.025
10.1016/S0893-6080(05)80023-1
10.1162/089976698300017197
10.1016/j.foodchem.2017.02.149
10.1016/0925-4005(96)80047-5
10.1109/icsens.2003.1278889
10.1109/isie.1997.651750
10.1016/j.snb.2009.04.030
10.1016/0925-4005(91)80185-m
10.1016/j.aca.2007.02.013
10.1063/1.3156580
10.1111/j.1750-3841.2012.02851.x
10.1007/s12161-017-0909-z
10.1109/tim.2012.2184011
10.3390/s21062016
10.3390/s17051007
10.1016/b978-0-12-800243-8.00004-4
10.3390/s150101354
10.1111/joss.12104
10.1111/joss.12226
10.1039/9781782622437-00335
10.1007/s10068-017-0168-1""".split()

out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/references_verified.csv")
rows = []
for d in DOIS:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(d)
    req = urllib.request.Request(url, headers={"User-Agent": "reference-check/1.0"})
    try:
        m = json.load(urllib.request.urlopen(req, timeout=30))["message"]
        auth = "; ".join(f"{a.get('family','')}, {a.get('given','')}" for a in m.get("author", []))
        yr = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
        rows.append({"doi": d, "status": "found", "title": m.get("title", [""])[0], "authors": auth, "year": yr,
                     "venue": (m.get("container-title") or [""])[0], "volume": m.get("volume", ""),
                     "issue": m.get("issue", ""), "pages": m.get("page", "")})
    except Exception as e:
        rows.append({"doi": d, "status": f"NOT FOUND ({e})"})
    time.sleep(0.3)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["doi", "status", "title", "authors", "year", "venue", "volume", "issue", "pages"])
    w.writeheader(); w.writerows(rows)
for r in rows:
    print(r["doi"], "|", r["status"], "|", r.get("year"), "|", r.get("title", "")[:90], "|", r.get("authors", "")[:70], "|", r.get("venue", "")[:40])
