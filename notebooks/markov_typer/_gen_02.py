"""Generate notebooks/markov_typer/02_attr_type_silver.ipynb — silver + EDA (no clf)."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_attr_type_silver.ipynb"
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
    """# 02. ATTR type silver + EDA

Задача: собрать **качественный silver** для типизации ATTR-span  
(`memory_storage`, `size`, `color`, …) — **без** локальных словарей и самодельных brand/model regex.

Учитель = тот же пайплайн, что NER:

```text
query → WeakLabeler.label_query → BIO
      → bio_to_entities → ATTR spans
      → _guess_attr_type (ATTR_PATTERNS + COLORS)
```

## Несколько ATTR в одном запросе

`ноутбук asus 16 гб 15.6 дюйм` → **две** строки silver:

| span_text | y |
|---|---|
| `16 гб` | memory_storage |
| `15.6 дюйм` | size |

Для будущего clf: n-grams **только по `span_text`**; чужие ATTR в контексте **маскируем**, не склеиваем.

Обучение классификаторов — **отдельный** ноутбук (после того как датасет устраивает).
"""
)

md("## 0. Setup")

code(
    """%matplotlib inline
import sys
import json
import re
import warnings
from pathlib import Path
from collections import Counter

ROOT = Path.cwd().resolve()
if ROOT.name in {"markov_typer", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "markov_typer" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    apply_plot_style, ensure_dirs, load_query_clicks,
    ARTIFACTS_DIR, FIGURES_DIR, MVIDEO_RED, DARK_SLATE, MUTED, save_stats,
)
from src.ner.labeling import (
    WeakLabeler,
    bio_to_entities,
    entities_to_structured,
    _guess_attr_type,
    ATTR_PATTERNS,
)
from src.preprocessing.pipeline import basic_clean, _norm_key

warnings.filterwarnings("ignore", category=FutureWarning)
ensure_dirs()
apply_plot_style()

OUT = ROOT / "artifacts" / "attr_type"
OUT.mkdir(parents=True, exist_ok=True)
FIG = FIGURES_DIR / "attr_type"
FIG.mkdir(parents=True, exist_ok=True)
print("OUT", OUT)
print("ATTR types in labeling:", [n for _, n in ATTR_PATTERNS])
"""
)

md(
    """## 1. Конфиг

| параметр | смысл |
|---|---|
| `SAMPLE_N` / `MAX_QUERIES` | сэмпл кликов → unique queries |
| `MIN_SPAN_LEN` | дроп пустых/крошечных span |
| unit-aug | доп. строки `гб↔gb↔…` с `is_aug=True` |
"""
)

code(
    """SAMPLE_N = 120_000
MAX_QUERIES = 40_000
SEED = 42
MIN_SPAN_LEN = 1
RARE_SUPPORT = 25  # только для отчёта EDA, в датасет не схлопываем

UNIT_AUG = {
    "гб": ["gb", "кб", "kb"],
    "gb": ["гб", "кб", "kb"],
    "тб": ["tb"],
    "tb": ["тб"],
    "мб": ["mb"],
    "mb": ["мб"],
    "вт": ["w"],
    "w": ["вт"],
    "кг": ["kg"],
    "kg": ["кг"],
}
"""
)

md(
    """## 2. WeakLabeler — единственный источник сущностей

Словари: `artifacts/brands.txt`, `categories.txt`, `model_phrases.txt`.  
`BRAND` / `MODEL` / `CATEGORY` / `ATTR` уже разведены внутри `label_query` — **не** копируем find_brand / looks_like_model сюда.
"""
)

code(
    """labeler = WeakLabeler.from_files(
    ARTIFACTS_DIR / "brands.txt",
    ARTIFACTS_DIR / "categories.txt",
    models_path=ARTIFACTS_DIR / "model_phrases.txt",
)
print(
    f"brands={len(labeler.brands)} cats={len(labeler.categories)} "
    f"models={len(labeler.models)} colors={len(labeler.colors)}"
)

# sanity: типы из labeling.py
for demo in ["16 гб", "15.6 дюйм", "белый", "1920x1080", "xyz"]:
    print(f"  {_guess_attr_type(demo)!r:20} <- {demo!r}")
"""
)

md(
    """## 3. Схема silver-строки и маскирование

Одна строка = **один** ATTR-span.

| колонка | смысл |
|---|---|
| `query`, `query_norm` | исходник / текст для labeler |
| `span_text`, `span_start`, `span_end` | текст и char-span ATTR |
| `y` | `_guess_attr_type` → `memory_storage` / `color` / `other` / … |
| `brand`, `category`, `model` | из `entities_to_structured` |
| `n_attrs_in_query` | сколько ATTR в запросе |
| `bio_tags` | BIO-строка для аудита |
| `query_masked_all_attr` | **все** ATTR → `<ATTR>` |
| `query_keep_span_mask_others` | текущий span текстом, остальные ATTR → `<ATTR>` |
| `query_masked_entities` | BRAND/CAT/MODEL/ATTR → плейсхолдеры |
| `is_aug` | unit-aug строка или нет |

**Политика:** n-grams типа — только `span_text`; чужие ATTR из контекста **исключаем** маской, не склеиваем.
"""
)

code(
    """_MULTI_SPACE = re.compile(r"\\s+")
_ENT_TOKEN = {
    "BRAND": "<BRAND>",
    "CATEGORY": "<CAT>",
    "MODEL": "<MODEL>",
    "ATTR": "<ATTR>",
    "GENRE": "<GENRE>",
    "PERSON": "<PERSON>",
}


def _mask_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    \"\"\"spans: (start, end, replacement), applied right-to-left.\"\"\"
    chars = list(text)
    for a, b, repl in sorted(spans, key=lambda x: -x[0]):
        chars[a:b] = list(repl)
    return _MULTI_SPACE.sub(" ", "".join(chars)).strip()


def mask_all_attr(text: str, attr_ents: list[dict]) -> str:
    spans = [(e["span"][0], e["span"][1], "<ATTR>") for e in attr_ents if "span" in e]
    return _mask_spans(text, spans)


def mask_keep_span(text: str, attr_ents: list[dict], keep: dict) -> str:
    spans = []
    for e in attr_ents:
        if "span" not in e:
            continue
        a, b = e["span"]
        if e is keep or (a == keep.get("span", [None])[0] and b == keep.get("span", [None, None])[1]):
            continue
        spans.append((a, b, "<ATTR>"))
    return _mask_spans(text, spans)


def mask_all_entities(text: str, ents: list[dict]) -> str:
    spans = []
    for e in ents:
        if "span" not in e:
            continue
        repl = _ENT_TOKEN.get(e["label"], f"<{e['label']}>")
        spans.append((e["span"][0], e["span"][1], repl))
    return _mask_spans(text, spans)


def aug_span_text(span_text: str, y: str) -> list[tuple[str, bool]]:
    \"\"\"[(text, is_aug), ...] — оригинал + unit synonyms.\"\"\"
    out = [(span_text, False)]
    if y in {"other", "color"}:
        return out
    parts = span_text.split()
    if len(parts) < 2:
        return out
    last = parts[-1].lower()
    for alt in UNIT_AUG.get(last, []):
        out.append((" ".join(parts[:-1] + [alt]), True))
    return out


def build_rows_for_query(query: str, labeler: WeakLabeler) -> list[dict]:
    q_clean = basic_clean(query, lowercase=False)
    q_norm = _norm_key(q_clean)
    if len(q_norm) < 2:
        return []
    tags = labeler.label_query(q_norm)
    ents = bio_to_entities(tags, query=q_norm)
    struct = entities_to_structured(ents, labeler)
    attr_ents = [e for e in ents if e["label"] == "ATTR" and (e.get("text") or "").strip()]
    if not attr_ents:
        return []
    bio_str = " ".join(f"{t}/{g}" for t, g in tags)
    masked_all = mask_all_attr(q_norm, attr_ents)
    masked_ent = mask_all_entities(q_norm, ents)
    brand = struct.get("brand") or ""
    category = struct.get("category") or ""
    model = struct.get("model") or ""
    rows = []
    for e in attr_ents:
        st0 = (e.get("text") or "").strip()
        if len(st0) < MIN_SPAN_LEN:
            continue
        span = e.get("span") or [None, None]
        y = _guess_attr_type(st0)
        keep_masked = mask_keep_span(q_norm, attr_ents, e)
        for st, is_aug in aug_span_text(st0, y):
            rows.append({
                "query": query,
                "query_norm": q_norm,
                "span_text": st,
                "span_start": span[0],
                "span_end": span[1],
                "y": y,
                "brand": brand,
                "category": category,
                "model": model,
                "n_attrs_in_query": len(attr_ents),
                "bio_tags": bio_str,
                "query_masked_all_attr": masked_all,
                "query_keep_span_mask_others": keep_masked,
                "query_masked_entities": masked_ent,
                "is_aug": is_aug,
            })
    return rows


# demo multi-ATTR + MODEL (+ glued unit caveats)
for q in [
    "ноутбук asus 16 гб 15.6 дюйм",
    "asus tuf gaming a15 16 гб",
    "asus tuf gaming a15 16gb",  # glued: model_phrases may swallow units
]:
    demo = pd.DataFrame(build_rows_for_query(q, labeler))
    print("\\nQ:", q)
    tags = labeler.label_query(_norm_key(basic_clean(q, lowercase=False)))
    print("  BIO:", tags)
    if demo.empty:
        print("  (no ATTR rows)")
    else:
        display(demo.loc[~demo["is_aug"], [
            "span_text", "y", "brand", "category", "model",
            "query_keep_span_mask_others", "query_masked_all_attr",
        ]])
"""
)

md("## 4. Сбор silver по кликам")

code(
    """clicks = load_query_clicks(n=SAMPLE_N, seed=SEED, random=True, columns=["query_text"])
queries = (
    clicks["query_text"].fillna("").astype(str).str.strip()
    .loc[lambda s: s.str.len().between(2, 120)]
    .drop_duplicates()
    .head(MAX_QUERIES)
    .tolist()
)
print(f"unique queries: {len(queries):,}")

rows: list[dict] = []
n_with_attr = 0
n_multi = 0
for q in queries:
    r = build_rows_for_query(q, labeler)
    if not r:
        continue
    n_with_attr += 1
    if r[0]["n_attrs_in_query"] >= 2:
        n_multi += 1
    rows.extend(r)

silver = pd.DataFrame(rows)
print(f"queries with ATTR: {n_with_attr:,}  multi-ATTR queries: {n_multi:,} ({n_multi/max(n_with_attr,1):.1%})")
print(f"silver rows (with aug): {len(silver):,}  raw: {(~silver['is_aug']).sum():,}  aug: {silver['is_aug'].sum():,}")
print("\\ny value_counts (raw only):")
print(silver.loc[~silver["is_aug"], "y"].value_counts())
display(silver.head(6))
"""
)

md(
    """## 5. EDA

Смотрим покрытие типов, multi-ATTR, контекст brand/cat/model, дыры (`other`), путаницу `dimensions` ↔ `resolution_exact`.
"""
)

code(
    """raw = silver.loc[~silver["is_aug"]].copy()

# --- overview ---
overview = pd.DataFrame([
    {"metric": "queries_sampled", "value": len(queries)},
    {"metric": "queries_with_ATTR", "value": n_with_attr},
    {"metric": "multi_ATTR_queries", "value": n_multi},
    {"metric": "multi_ATTR_share", "value": round(n_multi / max(n_with_attr, 1), 4)},
    {"metric": "silver_rows_raw", "value": int((~silver["is_aug"]).sum())},
    {"metric": "silver_rows_aug", "value": int(silver["is_aug"].sum())},
    {"metric": "n_types", "value": int(raw["y"].nunique())},
    {"metric": "share_with_brand", "value": round((raw["brand"].astype(str).str.len() > 0).mean(), 4)},
    {"metric": "share_with_category", "value": round((raw["category"].astype(str).str.len() > 0).mean(), 4)},
    {"metric": "share_with_model", "value": round((raw["model"].astype(str).str.len() > 0).mean(), 4)},
    {"metric": "share_y_other", "value": round((raw["y"] == "other").mean(), 4)},
    {"metric": "share_y_color", "value": round((raw["y"] == "color").mean(), 4)},
])
display(overview)

vc = raw["y"].value_counts()
rare = vc[vc < RARE_SUPPORT]
print(f"rare types (support < {RARE_SUPPORT}) — leave in dataset, flag for train later:")
display(rare.to_frame("support") if len(rare) else pd.DataFrame({"support": []}))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
vc.head(18).iloc[::-1].plot.barh(ax=axes[0], color=MVIDEO_RED)
axes[0].set_title("y distribution (raw spans)")
axes[0].set_xlabel("count")

n_attr_hist = raw.drop_duplicates("query_norm")["n_attrs_in_query"].clip(upper=6).value_counts().sort_index()
axes[1].bar(n_attr_hist.index.astype(str), n_attr_hist.values, color=DARK_SLATE)
axes[1].set_title("ATTR count per query (clipped at 6)")
axes[1].set_xlabel("n_attrs")
fig.tight_layout()
fig.savefig(FIG / "01_y_and_multi.png", dpi=120, bbox_inches="tight")
plt.show()
"""
)

code(
    """# multi-ATTR examples
multi = raw[raw["n_attrs_in_query"] >= 2]
print(f"multi-ATTR rows: {len(multi):,}")
# one query → list of (span, y)
ex_q = (
    multi.groupby("query_norm")
    .agg(spans=("span_text", list), types=("y", list), brand=("brand", "first"), n=("n_attrs_in_query", "first"))
    .reset_index()
    .head(12)
)
display(ex_q)

# top span texts per frequent type
for typ in vc.head(6).index:
    top = raw.loc[raw["y"] == typ, "span_text"].value_counts().head(8)
    print(f"\\n=== {typ} ===")
    print(top.to_string())
"""
)

code(
    """# dimensions vs resolution_exact — confusion-prone
confusable = raw[raw["y"].isin(["dimensions", "resolution_exact"])]
print("dimensions / resolution_exact sample:")
display(confusable[["query_norm", "span_text", "y", "brand", "category"]].head(20))

# y=other — regex holes / leftover ATTR
other = raw[raw["y"] == "other"]
print(f"\\nother count: {len(other):,}")
display(other["span_text"].value_counts().head(25).to_frame("count"))
display(other[["query_norm", "span_text", "brand", "category", "model"]].head(15))

# color sample
colors = raw[raw["y"] == "color"]
print(f"\\ncolor count: {len(colors):,}")
display(colors["span_text"].value_counts().head(15).to_frame("count"))
"""
)

code(
    """# masking sanity on multi-ATTR
sample_masks = multi.drop_duplicates("query_norm").head(5)
for _, r in sample_masks.iterrows():
    print("Q:", r["query_norm"])
    sub = raw[raw["query_norm"] == r["query_norm"]][["span_text", "y", "query_keep_span_mask_others"]]
    display(sub)
    print("  all_attr:", r["query_masked_all_attr"])
    print("  entities:", r["query_masked_entities"])
    print()

# unit-aug effect
aug_vs = pd.DataFrame({
    "raw_rows": [(~silver["is_aug"]).sum()],
    "aug_rows": [silver["is_aug"].sum()],
    "types_raw": [raw["y"].nunique()],
    "types_all": [silver["y"].nunique()],
})
display(aug_vs)
print("aug examples:")
display(silver.loc[silver["is_aug"], ["span_text", "y", "query_norm"]].head(12))
"""
)

code(
    """# MODEL vs ATTR: model_phrases иногда «съедают» число+единицу (особенно glued 16gb)
model_eat = []
for q in queries[:3000]:
    qn = _norm_key(basic_clean(q, lowercase=False))
    tags = labeler.label_query(qn)
    ents = bio_to_entities(tags, query=qn)
    models = [e for e in ents if e["label"] == "MODEL"]
    for m in models:
        t = (m.get("text") or "").lower()
        if re.search(r"\\d", t) and re.search(r"(gb|гб|tb|тб|mb|мб|вт|w|кг|kg)", t):
            model_eat.append({"query_norm": qn, "model_span": m.get("text")})
            if len(model_eat) >= 15:
                break
    if len(model_eat) >= 15:
        break
print(f"sample MODEL spans that look like ATTR units (up to 15): {len(model_eat)}")
display(pd.DataFrame(model_eat) if model_eat else pd.DataFrame({"note": ["none in first 3k queries"]}))
"""
)

md(
    """## 6. Сохранение

- `artifacts/attr_type/attr_type_silver.parquet` — полный silver (raw + aug)
- `artifacts/attr_type/attr_type_silver_raw.parquet` — только `is_aug=False`
- `artifacts/attr_type/attr_type_silver_meta.json` — N, классы, описание масок
"""
)

code(
    """meta = {
    "seed": SEED,
    "sample_n": SAMPLE_N,
    "max_queries": MAX_QUERIES,
    "n_queries_sampled": len(queries),
    "n_queries_with_attr": n_with_attr,
    "n_multi_attr_queries": n_multi,
    "n_rows_raw": int((~silver["is_aug"]).sum()),
    "n_rows_aug": int(silver["is_aug"].sum()),
    "n_rows_total": len(silver),
    "classes": sorted(raw["y"].unique().tolist()),
    "class_counts_raw": raw["y"].value_counts().to_dict(),
    "rare_types_support_lt": RARE_SUPPORT,
    "rare_types": rare.index.tolist() if len(rare) else [],
    "teacher": "WeakLabeler + _guess_attr_type(ATTR_PATTERNS, COLORS)",
    "mask_columns": {
        "query_masked_all_attr": "all ATTR spans -> <ATTR>",
        "query_keep_span_mask_others": "current span kept; other ATTR -> <ATTR>",
        "query_masked_entities": "BRAND/CAT/MODEL/ATTR -> placeholders",
    },
    "unit_aug_keys": sorted(UNIT_AUG.keys()),
    "design_note": (
        "one row per ATTR span; char n-grams for typing should use span_text only; "
        "other ATTR excluded via mask columns, not concatenated"
    ),
}

silver.to_parquet(OUT / "attr_type_silver.parquet", index=False)
raw.to_parquet(OUT / "attr_type_silver_raw.parquet", index=False)
(OUT / "attr_type_silver_meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)
overview.to_csv(OUT / "attr_type_silver_overview.csv", index=False)
save_stats({"attr_type_silver": meta}, name="attr_type_silver.json")
print("saved:", OUT / "attr_type_silver.parquet")
print(json.dumps({k: meta[k] for k in ["n_rows_raw", "n_rows_aug", "n_queries_with_attr", "n_multi_attr_queries"]}, indent=2))
"""
)

md(
    """## 7. Вердикт

| вопрос | ответ |
|---|---|
| Откуда spans? | `WeakLabeler` (те же словари, что NER) |
| Откуда `y`? | `_guess_attr_type` = COLORS + `ATTR_PATTERNS` (имена групп совпадают) |
| Multi-ATTR? | одна строка на span; маски исключают чужие ATTR |
| `other`? | оставляем — дыры regex / кандидаты в gold |
| MODEL съел единицу? | бывает на glued (`16gb`) — смотри EDA; править `model_phrases`, не локальный regex |
| UNKNOWN / rare collapse? | **не** в этом ноутбуке — политика train-clf |
| Готов к clf? | да, если coverage типов и маски ок по EDA выше |

Дальше: train-ноутбук на `attr_type_silver.parquet` (span TF-IDF + context columns).
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", NB, "cells", len(cells))
