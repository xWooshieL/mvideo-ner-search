"""Generate notebooks/preprocessing/02_preprocessed_data_overview.ipynb"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "02_preprocessed_data_overview.ipynb"


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

cells.append(md("""# 02. Preprocessed data overview — готовность silver → gold / CRF

Контекст пайплайна:

```text
1) regex + словари + (опц.) CRF   ← вы здесь: обучающие BIO-данные
2) ATTR typing / MM / classifiers
```

Цель ноутбука: прогнать **полный** preprocess + `WeakLabeler` (с `MODEL`), оценить качество silver-разметки и словаря `model_phrases.txt`, дать вердикт: **можно ли начинать gold**.

Связанные артефакты: `artifacts/model_phrases.txt`, `artifacts/brands.txt`, `artifacts/categories.txt`.
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
    MVIDEO_RED, DARK_SLATE, MUTED, save_stats,
)
from src.ner.labeling import (
    WeakLabeler, tokenize, lemmatize_text, bio_to_entities, entities_to_structured,
)
from src.preprocessing import QueryPreprocessor, load_phrase_list

ensure_dirs()
apply_plot_style()
FIG = FIGURES_DIR / "preprocessing"
FIG.mkdir(parents=True, exist_ok=True)
ART = ARTIFACTS_DIR

def save_local(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print("saved", p)
    return p

SAMPLE_N = 120_000
N_EVAL = 8_000
print("ROOT:", ROOT)
"""))

cells.append(md("""## 1. Полный пайплайн разметки одной строки

**Важно:** `models_path` обязателен. Без него получите ровно ваш кейс:

`asus tuf gaming a15 → BRAND + O O O`, хотя `tuf gaming a15` уже есть в словаре.
"""))

cells.append(code(r"""pp = QueryPreprocessor.from_artifacts(ART)
lab_no_model = WeakLabeler.from_files(ART / "brands.txt", ART / "categories.txt")
lab = WeakLabeler.from_files(
    ART / "brands.txt",
    ART / "categories.txt",
    models_path=ART / "model_phrases.txt",
)

def full_label(query: str, labeler=lab):
    r = pp(query)
    tags = labeler.label_query(r.text_norm)
    ents = bio_to_entities(tags, query=r.text_norm)
    return {
        "original": query,
        "text_norm": r.text_norm,
        "preprocess_model_spans": r.model_spans,
        "bio": tags,
        "entities": ents,
        "structured": entities_to_structured(ents, labeler),
        "lemma_len": len(lemmatize_text(r.text_norm)),
        "tok_len": len(tokenize(r.text_norm)),
    }

demos = [
    "asus tuf gaming a15",
    "наушники logitech g-pro x se",
    "красные холодильники samsung",
    "ноутбук 16гб",
    "пылесос dyson v15",
]
rows = []
for q in demos:
    a = full_label(q, lab_no_model)
    b = full_label(q, lab)
    rows.append({
        "query": q,
        "bio_WITHOUT_models_path": a["bio"],
        "bio_WITH_models_path": b["bio"],
        "structured": b["structured"],
        "lemma_vs_tok": f"{b['lemma_len']} vs {b['tok_len']}",
    })
pd.DataFrame(rows)
"""))

cells.append(md("""### Диагноз кейса `asus tuf gaming a15`

| Наблюдение | Вывод |
|---|---|
| В `model_phrases.txt` есть `tuf gaming a15` | словарь знает линейку |
| Без `models_path` → `tuf/gaming/a15 = O` | это не «дыра словаря», а **не подключенный** словарь |
| Natasha иногда режет `a15` → `a`+`15` | при mismatch длин включается fallback на `_normalize(token)` — матч MODEL всё равно возможен |
| Preprocess может пометить весь `asus tuf...` как MODEL | расхождение с labeler (asus = BRAND) — нормально: brand приоритетнее |

**Правильный вызов для CRF-данных:**

```python
lab = WeakLabeler.from_files(brands, categories, models_path=ART / "model_phrases.txt")
tags = lab.label_query(pp(query).text_norm)
```
"""))

cells.append(md("""## 2. Аудит `model_phrases.txt`
"""))

cells.append(code(r"""phrases = load_phrase_list(ART / "model_phrases.txt")
print("n_phrases:", len(phrases))

def phrase_flags(p: str) -> dict:
    toks = p.split()
    long_sku = any(sum(ch.isdigit() for ch in t) >= 5 for t in toks)
    has_alpha = any(any(c.isalpha() for c in t) for t in toks)
    pure_num = all(t.replace(".", "").isdigit() for t in toks)
    very_long = len(toks) >= 5
    looks_accessory = any(w in p for w in ["earpods", "cable", "чехол", "power bank", "silicone"])
    return {
        "n_tok": len(toks),
        "long_sku_digit": long_sku,
        "pure_num": pure_num,
        "very_long": very_long,
        "looks_accessory": looks_accessory,
        "ok_candidate": has_alpha and not long_sku and not pure_num and not very_long,
    }

flag_df = pd.DataFrame([{**{"phrase": p}, **phrase_flags(p)} for p in phrases])
summary = pd.DataFrame({
    "metric": [
        "total",
        "ok_candidate",
        "long_sku_digit (артикулы)",
        "very_long (>=5 tok)",
        "looks_accessory",
        "pure_num",
    ],
    "count": [
        len(flag_df),
        int(flag_df["ok_candidate"].sum()),
        int(flag_df["long_sku_digit"].sum()),
        int(flag_df["very_long"].sum()),
        int(flag_df["looks_accessory"].sum()),
        int(flag_df["pure_num"].sum()),
    ],
})
summary["share"] = summary["count"] / len(flag_df)
display(summary)

print("Примеры шума (артикулы/длинные):")
display(flag_df[flag_df["long_sku_digit"] | flag_df["very_long"]].head(12)[["phrase", "n_tok", "long_sku_digit", "very_long"]])
print("Примеры ок (tuf/g pro/...):")
display(flag_df[flag_df["phrase"].str.contains(r"^(tuf|g pro|v1[0-9]|galaxy)", regex=True)].head(12))
"""))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(
    ["ok_candidate", "sku/noise-ish"],
    [flag_df["ok_candidate"].mean(), 1 - flag_df["ok_candidate"].mean()],
    color=[DARK_SLATE, MVIDEO_RED],
)
axes[0].set_ylim(0, 1)
axes[0].set_title("model_phrases: доля «чистых» vs шумных кандидатов")
axes[0].set_ylabel("доля")

axes[1].hist(flag_df["n_tok"], bins=range(1, int(flag_df["n_tok"].max()) + 2),
             color=MVIDEO_RED, edgecolor="white", align="left")
axes[1].set_title("Длина фраз MODEL (токены)")
axes[1].set_xlabel("n tokens")
fig.tight_layout()
save_local(fig, "10_model_phrases_quality.png")
plt.show()
"""))

cells.append(md("""## 3. Покрытие silver BIO на семпле запросов

Сравниваем labeler **без** / **с** `models_path` после preprocess.
"""))

cells.append(code(r"""clicks = load_query_clicks(n=SAMPLE_N, seed=42, random=True)
uq = clicks["query_text"].astype(str).str.strip()
uq = uq[uq.str.len() >= 2].drop_duplicates()
sample_q = uq.sample(n=min(N_EVAL, len(uq)), random_state=42).tolist()

def eval_labeler(labeler):
    type_counts = Counter()
    n_empty = 0
    n_lemma_mismatch = 0
    n_has_model = 0
    n_brand_o_tail = 0  # BRAND + латинский хвост из O
    hard = []
    for q in sample_q:
        r = pp(q)
        qn = r.text_norm
        toks = tokenize(qn)
        lems = lemmatize_text(qn)
        if len(toks) != len(lems):
            n_lemma_mismatch += 1
        tags = labeler.label_query(qn)
        bs = [t[2:] for _, t in tags if t.startswith("B-")]
        type_counts.update(bs)
        if not bs:
            n_empty += 1
        if "MODEL" in bs:
            n_has_model += 1
        # brand + O latin tail
        bidx = [i for i, (_, t) in enumerate(tags) if t.endswith("BRAND")]
        if bidx:
            last = max(bidx)
            tail = tags[last + 1:]
            if tail and all(t == "O" for _, t in tail):
                tt = [tok for tok, _ in tail]
                if np.mean([x.isascii() and x.isalnum() for x in tt]) >= 0.6:
                    n_brand_o_tail += 1
                    if len(hard) < 15 and any(c.isalpha() for c in "".join(tt)):
                        hard.append({"query": qn, "o_tail": " ".join(tt[:6]), "bio": tags})
    n = len(sample_q)
    return {
        "n": n,
        "empty_share": n_empty / n,
        "lemma_mismatch_share": n_lemma_mismatch / n,
        "has_MODEL_share": n_has_model / n,
        "brand_latin_O_tail_share": n_brand_o_tail / n,
        "type_counts": dict(type_counts),
        "hard_tails": hard,
    }

stats_off = eval_labeler(lab_no_model)
stats_on = eval_labeler(lab)

cmp = pd.DataFrame([
    {"setting": "without models_path", **{k: stats_off[k] for k in ["empty_share", "has_MODEL_share", "brand_latin_O_tail_share", "lemma_mismatch_share"]}},
    {"setting": "with models_path", **{k: stats_on[k] for k in ["empty_share", "has_MODEL_share", "brand_latin_O_tail_share", "lemma_mismatch_share"]}},
])
display(cmp)
print("entity B-* counts WITH models:", stats_on["type_counts"])
print("Оставшиеся hard O-хвосты (кандидаты в gold MODEL):")
display(pd.DataFrame(stats_on["hard_tails"])[["query", "o_tail"]].head(12) if stats_on["hard_tails"] else pd.DataFrame({"msg": ["мало"]}))
"""))

cells.append(code(r"""fig, ax = plt.subplots(figsize=(8.5, 4))
x = np.arange(2)
w = 0.25
metrics = ["empty_share", "has_MODEL_share", "brand_latin_O_tail_share"]
colors = [MUTED, DARK_SLATE, MVIDEO_RED]
for i, m in enumerate(metrics):
    ax.bar(x + (i - 1) * w, [stats_off[m], stats_on[m]], w, label=m, color=colors[i])
ax.set_xticks(x)
ax.set_xticklabels(["no models_path", "with models_path"])
ax.set_ylim(0, 1)
ax.set_title("Silver BIO: эффект подключения model_phrases")
ax.legend(fontsize=8)
fig.tight_layout()
save_local(fig, "11_silver_coverage_with_without_model.png")
plt.show()
"""))

cells.append(md("""## 4. Вердикт для CRF / gold

### Можно ли начинать ручную разметку (gold)?

**Да — можно и нужно начинать**, но не как «замену» словаря, а как:

1. **eval / test** (честные метрики CRF, не silver-vs-silver);
2. **дообучение /纠偏** на hard tails, которых нет в `model_phrases` или которые словарь шумит.

Рекомендуемый объём первой волны: **300–500** запросов, стратификация:
- с BRAND + латинским хвостом;
- с ATTR (память/диагональ);
- «пустые» / только CATEGORY;
- спорные цвета / Title Case.

Формат (JSONL):

```json
{"query": "asus tuf gaming a15", "entities": [
  {"text": "asus", "label": "BRAND"},
  {"text": "tuf gaming a15", "label": "MODEL"}
]}
```

Путь: `artifacts/gold/query_entities.jsonl`.

### Подводные камни обучающих данных (stage-1 NER)

| Риск | Почему важно | Что сделать |
|---|---|---|
| Silver без `models_path` | CRF не видит класс MODEL | всегда передавать `model_phrases.txt` |
| Шум в `model_phrases` (~артикулы, 5+ токенов, аксессуары) | ложные B-MODEL → CRF учит мусор | пересобрать с `min_count>=6`, выкинуть long digit SKU |
| Natasha режет `a15`→`a`+`15` | mismatch токенов; иногда ломает матч | fallback уже есть; в gold размечать по **вашей** токенизации |
| Brand aliases (`galaxy` = BRAND) | часть «модели» уезжает в BRAND | в gold явно решить политику: `samsung` BRAND, `galaxy s24` MODEL |
| Клики ≠ золото intent | majority brand шумный | gold только по тексту запроса, не по клику |
| Train/test leak | одни и те же запросы | split по `query`, не по кликам |

### Трешхолды (практика)

| Параметр | Рекомендация сейчас |
|---|---|
| `min_count` для майнинга MODEL | **6–8** (словарь 6k слишком жирный — много шума в head) |
| однотокен MODEL | только `v15`/`g305`/`m2`-like (буква+цифра) |
| max tokens | **4** |
| отсев | digit-run ≥5 в токене (артикул), accessory keywords |
| доля gold от silver | не гнаться за 100% coverage словаря — gold на хвост |

### Итог одной фразой

Предобработка и схема BIO **готовы как bootstrap** для CRF; словарь MODEL работает, если его **подключать**.  
К gold можно приступать **сейчас**, параллельно почистив `model_phrases` от артикулов — иначе CRF получит шумные silver-лейблы на MODEL.
"""))

cells.append(code(r"""# Рекомендуемый «чистый» срез словаря (не перезаписываем файл без явного флага)
CLEAN = flag_df[
    flag_df["ok_candidate"]
    & ~flag_df["looks_accessory"]
].copy()
print(f"clean subset: {len(CLEAN)} / {len(flag_df)} ({len(CLEAN)/len(flag_df):.1%})")

# Показать, что tuf gaming a15 сохранится
must = ["tuf gaming a15", "g pro x se", "v15"]
for m in must:
    print(m, "in clean:", m in set(CLEAN["phrase"]) or m in load_phrase_list(ART/"model_phrases.txt"))

REWRITE = False  # поставьте True, чтобы перезаписать artifacts/model_phrases.txt очищенным срезом + MODEL_SEEDS
if REWRITE:
    from src.preprocessing.pipeline import MODEL_SEEDS
    from src.preprocessing import save_phrase_list
    keep = set(CLEAN["phrase"]) | set(MODEL_SEEDS)
    save_phrase_list(keep, ART / "model_phrases.txt")
    print("rewrote model_phrases.txt", len(keep))
else:
    print("REWRITE=False — словарь на диске не трогаем; для очистки переключите флаг.")

out = {
    "n_model_phrases": int(len(phrases)),
    "n_ok_candidate": int(flag_df["ok_candidate"].sum()),
    "ok_share": float(flag_df["ok_candidate"].mean()),
    "silver_with_model": {k: stats_on[k] for k in ["empty_share", "has_MODEL_share", "brand_latin_O_tail_share", "lemma_mismatch_share"]},
    "silver_without_model": {k: stats_off[k] for k in ["empty_share", "has_MODEL_share", "brand_latin_O_tail_share", "lemma_mismatch_share"]},
    "verdict": "start_gold_now_parallel_to_dict_cleanup",
    "asus_tuf_note": "needs models_path; phrase exists in model_phrases.txt",
}
save_stats(out, "preprocessed_data_overview.json")
pd.DataFrame([(k, str(v)) for k, v in out.items()], columns=["key", "value"])
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
