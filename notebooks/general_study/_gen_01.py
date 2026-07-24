"""Generate notebooks/general_study/01_general_silver.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "01_general_silver.ipynb"
cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
    )


md(
    """# 01. General study — joint silver + broken queries

Общий источник данных для [`02_general_train_eval`](./02_general_train_eval.ipynb):
**один** silver-датасет, на котором учатся **и CRF, и GLiNER** (в отличие от
`notebooks/crf_ner_classifier` и `notebooks/gliner`, у которых каждый свой прогон).

Плюс — синтетический **broken_queries_eval**: держим его отдельно, НЕ в трейне,
специально под тему "появились broken queries" (см. `artifacts/history/broken_queries.md`).

Пайплайн: `query → SpellFixer v2 (typo+units+homoglyphs+алиасы) → WeakLabeler → silver BIO`.

CLI: `python notebooks/general_study/_run_01.py`
"""
)

md("## 0. Setup")
code(
    """%matplotlib inline
import sys, json, random
from pathlib import Path
from collections import Counter

ROOT = Path.cwd().resolve()
if ROOT.name in {"general_study", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "general_study" else ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    ARTIFACTS_DIR, FIGURES_DIR, ensure_dirs, apply_plot_style,
    load_query_clicks, brands_path, categories_path, model_phrases_path,
    MVIDEO_RED, DARK_SLATE,
)
from src.ner.labeling import WeakLabeler, bio_to_entities, tokenize
from src.preprocessing.pipeline import basic_clean, _norm_key
from src.preprocessing.spellfix import SpellFixer

ensure_dirs()
apply_plot_style()
OUT = ARTIFACTS_DIR / "silver" / "general_study"; OUT.mkdir(parents=True, exist_ok=True)
FIG = FIGURES_DIR / "general_study"; FIG.mkdir(parents=True, exist_ok=True)
SEED, MAX_QUERIES, BROKEN_N = 42, 6000, 400
print("OUT", OUT)
"""
)

md("## 1. SpellFixer v2 (typo + units + гомоглифы + алиасы транслитерации)")
code(
    """spell = SpellFixer.from_artifacts(ARTIFACTS_DIR)
print("vocab:", len(spell._vocab_set), "aliases:", len(spell._alias_to_canon))

for q in ["телфон 16 гь", "аsus tuf", "cамсунг galaxy", "сони плейстейшен 5", "laptop ксяоми"]:
    fixed, changes = spell.fix_query(q)
    print(f"{q!r:30s} -> {fixed!r:30s} {changes}")
"""
)

md("## 2. Сэмпл кликов -> spellfix -> WeakLabeler -> silver BIO")
code(
    """labeler = WeakLabeler.from_files(brands_path(), categories_path(), models_path=model_phrases_path())

clicks = load_query_clicks()
if len(clicks) > 60_000:
    clicks = clicks.sample(n=60_000, random_state=SEED)
qcol = "query_text" if "query_text" in clicks.columns else "query"
queries = (
    clicks[qcol].astype(str).map(lambda x: basic_clean(x, lowercase=False)).map(_norm_key).drop_duplicates()
)
queries = [q for q in queries.tolist() if len(q) >= 2][:MAX_QUERIES]
print("queries:", len(queries))

n_fixed = 0
fixed_queries = []
for q in queries:
    q2, ch = spell.fix_query(q)
    if ch:
        n_fixed += 1
    fixed_queries.append(q2)
print(f"spellfix touched {n_fixed}/{len(fixed_queries)}")
"""
)

code(
    """def entity_counts(tags):
    c = Counter()
    for t in tags:
        if t.startswith("B-"):
            c[t[2:]] += 1
    return c

rows = []
for q in fixed_queries:
    sent = labeler.label_query(q)
    if not sent:
        continue
    tokens, tags = [t for t, _ in sent], [g for _, g in sent]
    rows.append({
        "query": q, "n_tokens": len(tokens), "tokens": tokens, "tags": tags,
        "n_entities": sum(1 for t in tags if t.startswith("B-")),
        "has_entity": any(t != "O" for t in tags),
        "bio_str": " ".join(f"{a}/{b}" for a, b in sent),
    })
silver = pd.DataFrame(rows)
silver_ent = silver[silver["has_entity"]].copy()
print(f"silver={len(silver)} with_entity={len(silver_ent)}")
display(silver_ent[["query", "n_tokens", "bio_str"]].head(10))
"""
)

md(
    """## 3. Синтетические broken_queries: порча токенов (раскладка/гомоглиф/дубль/пропуск)

Количество токенов НЕ меняется — BIO-теги остаются валидны без пересчёта."""
)
code(
    """_KEYBOARD_NEIGHBORS = {
    "а": "оыв", "о": "аыр", "е": "ирп", "и": "еыу", "у": "ицк", "с": "ачм",
    "н": "гоь", "р": "оеп", "л": "од", "в": "аыц", "м": "сит",
    "a": "sq", "s": "adw", "e": "wrd", "o": "ipl",
}
_HOMOGLYPH_INJECT = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у"}

def corrupt_token(tok, rng):
    if len(tok) < 3 or not tok.isalpha():
        return tok, False
    op = rng.choice(["keyboard", "homoglyph", "duplicate", "drop"])
    pos = rng.randrange(1, len(tok) - 1) if len(tok) > 2 else 0
    ch = tok[pos]
    if op == "keyboard":
        neigh = _KEYBOARD_NEIGHBORS.get(ch.lower())
        if not neigh: return tok, False
        rep = rng.choice(neigh); rep = rep.upper() if ch.isupper() else rep
        return tok[:pos] + rep + tok[pos+1:], True
    if op == "homoglyph":
        rep = _HOMOGLYPH_INJECT.get(ch.lower())
        if not rep: return tok, False
        rep = rep.upper() if ch.isupper() else rep
        return tok[:pos] + rep + tok[pos+1:], True
    if op == "duplicate":
        return tok[:pos+1] + tok[pos] + tok[pos+1:], True
    return tok[:pos] + tok[pos+1:], True

rng = random.Random(SEED)
pool = silver_ent[silver_ent["n_tokens"] >= 2].sample(n=min(BROKEN_N, len(silver_ent)), random_state=SEED)
broken_rows = []
for _, r in pool.iterrows():
    tokens, tags = list(r["tokens"]), list(r["tags"])
    new_tokens, n_changed = list(tokens), 0
    for i, t in enumerate(tokens):
        if rng.random() < 0.5:
            nt, ch = corrupt_token(t, rng)
            if ch:
                new_tokens[i] = nt; n_changed += 1
    if n_changed == 0:
        continue
    q_broken = " ".join(new_tokens)
    retok = [t for t, _, _ in tokenize(q_broken)]
    if len(retok) != len(tags):
        continue
    broken_rows.append({"query": q_broken, "query_orig": r["query"], "tokens": retok, "tags": tags, "n_corrupted_tokens": n_changed})
broken = pd.DataFrame(broken_rows)
print("broken_queries_eval:", len(broken))
display(broken[["query_orig", "query", "n_corrupted_tokens"]].head(10))
"""
)

md("## 4. Save")
code(
    """silver.to_parquet(OUT / "general_silver_bio.parquet", index=False)
silver_ent.to_parquet(OUT / "general_silver_bio_ent.parquet", index=False)
broken.to_parquet(OUT / "broken_queries_eval.parquet", index=False)
print("saved ->", OUT)
"""
)

md(
    """## Дальше

[`02_general_train_eval.ipynb`](./02_general_train_eval.ipynb) — обучение CRF + GLiNER на
этом silver и метрика **всего каскада** на gold + broken_queries_eval.

Headless: `python notebooks/general_study/_run_01.py`
"""
)

NB.write_text(
    json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": cells,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print("wrote", NB, "cells", len(cells))
