#!/usr/bin/env python
"""One-shot: shrink fonts/boxes in day-2 presentation for a compact look."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "docs" / "Презентация_День2_EDA.tex"
t = p.read_text(encoding="utf-8")

pairs = [
    # body text in cards / columns
    ("{\\footnotesize\\color{mvgray}", "{\\smalltext\\color{mvgray}"),
    ("{\\InterSemi\\small ", "{\\InterSemi\\fontsize{8.5}{10.5}\\selectfont "),
    # numbered lists blocks
    ("{\\footnotesize\n\\numdot", "{\\smalltext\n\\numdot"),
    ("{\\small\n\\numdot", "{\\smalltext\n\\numdot"),
    # big stat numbers smaller
    ("\\InterBold\\fontsize{17}{19}\\selectfont", "\\InterBold\\fontsize{13.5}{15.5}\\selectfont"),
    ("{\\fontsize{8}{10}\\selectfont\\color{mvgray}", "{\\smalltext\\color{mvgray}"),
    # tables smaller
    ("{\\footnotesize\n\\renewcommand{\\arraystretch}{1.35}", "{\\smalltext\n\\renewcommand{\\arraystretch}{1.25}"),
    ("{\\footnotesize\n\\renewcommand{\\arraystretch}{1.3}", "{\\smalltext\n\\renewcommand{\\arraystretch}{1.2}"),
    # card paddings smaller
    ("inner sep=10pt, text width=\\dimexpr\\textwidth-20pt\\relax", "inner sep=7pt, text width=\\dimexpr\\textwidth-14pt\\relax"),
    ("inner sep=11pt, text width=\\dimexpr\\textwidth-22pt\\relax", "inner sep=7pt, text width=\\dimexpr\\textwidth-14pt\\relax"),
    ("inner sep=8pt, text width=\\dimexpr\\textwidth-16pt\\relax", "inner sep=6pt, text width=\\dimexpr\\textwidth-12pt\\relax"),
    # layer boxes on architecture slide
    ("inner sep=10pt, text width=3.55cm, align=left, minimum height=2.9cm", "inner sep=7pt, text width=3.3cm, align=left, minimum height=2.4cm"),
    # markov token boxes
    ("inner sep=7pt, font=\\InterSemi\\small", "inner sep=5pt, font=\\InterSemi\\fontsize{8.5}{10}\\selectfont"),
    # match-flow boxes
    ("inner sep=7pt, text width=\\dimexpr\\textwidth-60pt\\relax, font=\\fontsize{8}{10.5}\\selectfont", "inner sep=5.5pt, text width=\\dimexpr\\textwidth-55pt\\relax, font=\\fontsize{7.5}{9.5}\\selectfont"),
    # sample click card fonts
    ("{\\InterSemi\\footnotesize Пример строки клика}", "{\\InterSemi\\fontsize{8}{10}\\selectfont Пример строки клика}"),
    ("{\\fontsize{7.5}{10}\\selectfont\\color{mvgray}\nquery", "{\\fontsize{7}{9}\\selectfont\\color{mvgray}\nquery"),
    # table header fonts inside InterSemi already sized by smalltext
]

for a, b in pairs:
    if a in t:
        t = t.replace(a, b)
    else:
        print("MISS:", a[:60])

p.write_text(t, encoding="utf-8")
print("done")
