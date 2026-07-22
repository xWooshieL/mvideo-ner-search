"""Generate notebooks/preprocessing/01_data_preprocessing.ipynb"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_data_preprocessing.ipynb"


def nid() -> str:
    return uuid.uuid4().hex[:8]


def md(s: str) -> dict:
    return {"cell_type": "markdown", "id": nid(), "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "id": nid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(md("""# 01. Data preprocessing — переиспользуемый слой

Собираем все договорённости проекта по предобработке запросов и реализуем их в `src.preprocessing`, чтобы другие ноутбуки импортировали **один** пайплайн.

Полный список гипотез: [`README.md`](./README.md).

> Фокус ноутбука — только preprocess (+ построение словарей MODEL / protected brands).  
> Обучение CRF/классификаторов здесь не делаем.
"""))

cells.append(md("""## 0. Setup
"""))

cells.append(code(r"""%matplotlib inline
import sys
from pathlib import Path
from collections import Counter

ROOT = Path.cwd().resolve()
if ROOT.name in {"preprocessing", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "preprocessing" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    apply_plot_style, ensure_dirs, load_query_clicks, ARTIFACTS_DIR, FIGURES_DIR,
    MVIDEO_RED, DARK_SLATE, save_stats,
)
from src.ner.labeling import WeakLabeler, bio_to_entities
from src.preprocessing import (
    QueryPreprocessor,
    build_model_lexicon_from_titles,
    save_phrase_list,
    load_phrase_list,
    split_glued_alnum,
)
from src.preprocessing.pipeline import MODEL_SEEDS, PROTECTED_BRAND_SEEDS, basic_clean

ensure_dirs()
apply_plot_style()
FIG = FIGURES_DIR / "preprocessing"
FIG.mkdir(parents=True, exist_ok=True)
ART = ARTIFACTS_DIR
ART.mkdir(exist_ok=True)

def save_local(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print("saved", p)
    return p

SAMPLE_N = 150_000
print("ROOT:", ROOT)
"""))

cells.append(md("""## 1. Гипотезы (кратко)

| ID | Гипотеза | Что делаем в preprocess |
|---|---|---|
| H1 | Грязный ввод (`128гб`, `g-pro`) | split units, нормализация сепараторов |
| H2 | Регистр важен («Красный Октябрь») | не слепой lower; protected brands; Title Case hints |
| H3 | Хвосты `g pro x se` = MODEL | lexicon сиды + майнинг из `sku_name` |
| H4 | ATTR-regex не ловит линейки | не пытаемся «допилить regex»; отдельный MODEL |
| H5 | Бренд часто вне текста | **не** чинится preprocess → clf/клики |
| H6 | ё/×/NBSP | унификация |

Слабые места — в конце ноутбука.
"""))

cells.append(md("""## 2. Демо шагов на жёстких примерах
"""))

cells.append(code(r"""examples = [
    "наушники logitech g-pro x se",
    "Ноутбук ASUS 16гб",
    "Красный Октябрь конфеты",
    "пылесос dyson v15",
    "iphone 15 pro max 256gb",
    "телевизор 55дюйм samsung",
]

rows = []
for q in examples:
    cleaned = basic_clean(q)
    rows.append({
        "original": q,
        "basic_clean": cleaned,
        "split_only": split_glued_alnum(q),
        "text_norm": QueryPreprocessor()(q).text_norm,
    })
pd.DataFrame(rows)
"""))

cells.append(md("""## 3. Майнинг MODEL-фраз из кликнутых SKU + сиды

Строим `artifacts/model_phrases.txt`: сиды (`g pro`, `v15`, …) ∪ частые хвосты после бренда в `sku_name`.
"""))

cells.append(code(r"""clicks = load_query_clicks(n=SAMPLE_N, seed=42, random=True)
brands = (
    clicks["sku_brand_name"].astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(400).index.tolist()
)
titles = clicks["sku_name"].astype(str).dropna().tolist()

mined = build_model_lexicon_from_titles(titles, brands, min_count=4, max_phrase_tokens=4)
print(f"mined phrases (≥4): {len(mined)}")

# объединяем с сидами
all_models = set(MODEL_SEEDS) | set(mined)
# немного ручного приоритета: оставляем фразы, где есть digit или известный префикс
def keep(p: str) -> bool:
    toks = p.split()
    if not toks:
        return False
    # не MODEL: чистые числа (ATTR)
    if all(t.replace(".", "").isdigit() for t in toks):
        return False
    if p in MODEL_SEEDS:
        return True
    if any(ch.isdigit() for ch in p) and any(ch.isalpha() for ch in p):
        return len(toks) >= 1
    if toks and toks[0] in {"g", "v", "ps", "galaxy", "redmi", "poco", "iphone", "macbook"}:
        return len(toks) >= 2
    return len(toks) >= 2 and mined.get(p, 0) >= 10

filtered = {p for p in all_models if keep(p)}
path_models = save_phrase_list(filtered, ART / "model_phrases.txt")
path_prot = save_phrase_list(PROTECTED_BRAND_SEEDS | set(load_phrase_list(ART / "protected_brands.txt")), ART / "protected_brands.txt")
print("wrote", path_models, "n=", len(filtered))
print("wrote", path_prot)

top_mined = pd.DataFrame(sorted(mined.items(), key=lambda x: -x[1])[:25], columns=["phrase", "count"])
display(top_mined)

fig, ax = plt.subplots(figsize=(9, 4))
show = top_mined.head(15)
ax.barh(show["phrase"][::-1], show["count"][::-1], color=MVIDEO_RED)
ax.set_title("Топ смайненных MODEL-фраз из sku_name")
fig.tight_layout()
save_local(fig, "01_mined_model_phrases.png")
plt.show()
"""))

cells.append(md("""## 4. QueryPreprocessor: MODEL + protected brand на запросах
"""))

cells.append(code(r"""pp = QueryPreprocessor.from_artifacts(ART)
labeler = WeakLabeler.from_files(ART / "brands.txt", ART / "categories.txt")

demo_queries = [
    "наушники logitech g pro x se",
    "наушники logitech g-pro x se",
    "Красный Октябрь",
    "красный телефон",
    "dyson v15 detect",
    "samsung galaxy s24 ultra",
]

demo_rows = []
for q in demo_queries:
    r = pp(q)
    weak = labeler.label_query(r.text_norm)
    merged = pp.merge_bio_hints(weak, r)
    demo_rows.append({
        "query": q,
        "text_norm": r.text_norm,
        "model_spans": r.model_spans,
        "protected": r.protected_spans,
        "titlecase": r.titlecase_hints,
        "bio_weak": weak,
        "bio_merged": merged,
    })

pd.DataFrame([{k: demo_rows[i][k] for k in ["query", "text_norm", "model_spans", "protected", "bio_merged"]} for i in range(len(demo_rows))])
"""))

cells.append(md("""## 5. Метрика эффекта на семпле уникальных запросов

Считаем, как часто появляется `MODEL` после preprocess+merge, и как часто хвост после бренда перестаёт быть полностью `O`.
"""))

cells.append(code(r"""uq = clicks["query_text"].astype(str).str.strip()
uq = uq[uq.str.len() >= 2].drop_duplicates()
sample_q = uq.sample(n=min(12_000, len(uq)), random_state=42).tolist()

n_model = 0
n_protected = 0
n_glued_fixed = 0
n_tail_o_before = 0
n_tail_improved = 0

def has_all_o_tail_after_brand(tags):
    idxs = [i for i, (_, t) in enumerate(tags) if "BRAND" in t]
    if not idxs:
        return False
    last = max(idxs)
    tail = tags[last + 1:]
    return bool(tail) and all(t == "O" for _, t in tail)

for q in sample_q:
    if split_glued_alnum(q) != q:
        n_glued_fixed += 1
    r = pp(q)
    if r.model_spans:
        n_model += 1
    if r.protected_spans or r.titlecase_hints:
        n_protected += 1
    weak = labeler.label_query(r.text_norm)
    merged = pp.merge_bio_hints(weak, r)
    before_tail = has_all_o_tail_after_brand(weak)
    after_has_model = any(t.endswith("MODEL") for _, t in merged)
    if before_tail:
        n_tail_o_before += 1
        if after_has_model:
            n_tail_improved += 1

stats = {
    "n_queries": len(sample_q),
    "share_model_span": n_model / len(sample_q),
    "share_protected_or_titlecase": n_protected / len(sample_q),
    "share_glued_unit_fixable": n_glued_fixed / len(sample_q),
    "n_brand_tail_all_O": n_tail_o_before,
    "n_tail_improved_with_MODEL": n_tail_improved,
    "tail_improve_rate": n_tail_improved / max(n_tail_o_before, 1),
    "n_model_phrases_dict": len(load_phrase_list(ART / "model_phrases.txt")),
}
display(pd.DataFrame(stats.items(), columns=["metric", "value"]))

fig, ax = plt.subplots(figsize=(7, 3.6))
keys = ["share_model_span", "share_glued_unit_fixable", "share_protected_or_titlecase", "tail_improve_rate"]
ax.bar(keys, [stats[k] for k in keys], color=MVIDEO_RED)
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
ax.set_title("Эффект предобработки на семпле запросов")
fig.tight_layout()
save_local(fig, "02_preprocess_effect.png")
plt.show()
"""))

cells.append(md("""## 6. Слабые места: что preprocess + словари НЕ закроют

### 6.1. `g pro` вне лексикона
Если завтра появится `g pro x 3e`, а фразы нет в `model_phrases.txt` и она редка в `sku_name` — останется `O`.

**Фикс не правилами:** разметить gold с тегом `MODEL` (300–1000 запросов) и учить CRF/transformer; lexicon — только bootstrap.

### 6.2. Омонимы (`pro`, `air`, `mini`)
`pro` может быть линейкой или словом. Lexicon longest-match ошибается без бренда/категории рядом.

**Фикс:** контекстный NER; не раздувать сиды однотокенным `pro` без необходимости (у нас `pro` есть в сидах — осознанный риск).

### 6.3. Бренд не написан (~74% кликов)
Preprocess не вытащит Apple из «айфон» без alias (alias — да) или из «телефон 128» без clf.

### 6.4. «красный» без «октябрь»
Title Case / protected list не сработают на lowercase «красный октябрь» — для этого фраза должна быть в `protected_brands.txt` (мы нормализуем в lower для матчинга — этот кейс **ловим**).  
А вот одиночный «красный» как бренд vs цвет — только модель/контекст.

### 6.5. Качественный датасет — когда обязателен

| Задача | Хватает preprocess+dict? | Нужен gold |
|---|---|---|
| `128гб` → токены | да | нет |
| `g-pro` → `g pro` | да | нет |
| `g pro x se` известный | да (lexicon) | желательно для обобщения |
| новая линейка / омоним | нет | **да** |
| бренд вне строки | нет | клик-лейблы / clf |
| color vs brand | частично | **да** на спорных |

Рекомендуемый формат gold (JSONL), чтобы добить хвосты:

```json
{"query": "наушники logitech g pro x se", "entities": [
  {"text": "наушники", "label": "CATEGORY"},
  {"text": "logitech", "label": "BRAND"},
  {"text": "g pro x se", "label": "MODEL"}
]}
```

Путь: `artifacts/gold/query_entities.jsonl` (создадите при разметке).
"""))

cells.append(code(r"""# Примеры, где lexicon/preprocess всё ещё слаб (для ручного просмотра)
hard = []
for q in sample_q:
    r = pp(q)
    weak = labeler.label_query(r.text_norm)
    merged = pp.merge_bio_hints(weak, r)
    # бренд есть, после него хвост из ≥2 O-токенов латиницы без MODEL
    brand_idx = [i for i, (_, t) in enumerate(merged) if t.endswith("BRAND")]
    if not brand_idx:
        continue
    last = max(brand_idx)
    tail = merged[last + 1:]
    if len(tail) >= 2 and all(t == "O" for _, t in tail):
        if all(tok.isascii() and tok.isalpha() and len(tok) <= 6 for tok, _ in tail[:4]):
            hard.append({"query": q, "tail": " ".join(t for t, _ in tail[:5]), "bio": merged})
    if len(hard) >= 15:
        break

print("Примеры хвостов, которые всё ещё O (кандидаты в gold MODEL):")
display(pd.DataFrame(hard)[["query", "tail"]] if hard else pd.DataFrame({"msg": ["мало примеров в семпле"]}))
"""))

cells.append(md("""## 7. Как использовать в других ноутбуках

```python
from src.preprocessing import QueryPreprocessor

pp = QueryPreprocessor.from_artifacts(ROOT / "artifacts")
r = pp(query)
# дальше: labeler / CRF на r.text_norm
tags = labeler.label_query(r.text_norm)
tags = pp.merge_bio_hints(tags, r)
```

Не копируйте локальные `lower()` / regex split — иначе разъедется словари `model_phrases.txt`.
"""))

cells.append(code(r"""save_stats(stats, "preprocessing_stats.json")
print("README:", ROOT / "notebooks" / "preprocessing" / "README.md")
print("Figures:", FIG)
print("Artifacts:", ART / "model_phrases.txt", ART / "protected_brands.txt")
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "cells", len(cells))
