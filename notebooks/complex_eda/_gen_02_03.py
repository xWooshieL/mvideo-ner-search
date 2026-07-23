"""Generate 02_markov_eda.ipynb and 03_click_eda.ipynb"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent


def nid() -> str:
    return uuid.uuid4().hex[:8]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "id": nid(), "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": nid(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def write_nb(path: Path, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path, "cells=", len(cells))


# ---------------------------------------------------------------------------
# 02_markov_eda
# ---------------------------------------------------------------------------
c02: list[dict] = []

c02.append(md("""# 02. Markov EDA: типизация атрибутов после BIO

Связка с выводами из `01_methods_eda`:
- хвосты вроде `g pro x se` часто остаются `O` — regex атрибутов их **не** ловит (это модель/линейка, не `16 гб`);
- идеал пайплайна: **сначала** `B/I-ATTR` (или более общие spans), **потом** тип атрибута;
- миллиард голов `memory/diagonal/...` не нужен → отдельный **типизатор** поверх ATTR.

Этот ноутбук: принцип Маркова, бейзлайн на биграммах, сравнение с расширенными правилами из `temp/labeling.py`, вердикт «имеет ли смысл» vs LogReg/RNN.
"""))

c02.append(md("""## 0. Setup

Импортируем **расширенную** карту `ATTR_PATTERNS` из `temp/labeling.py` (не урезанный `src/ner/labeling.py`).
"""))

c02.append(code(r"""%matplotlib inline
import sys
import importlib.util
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

ROOT = Path.cwd().resolve()
if ROOT.name in {"complex_eda", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "complex_eda" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    apply_plot_style, ensure_dirs, load_query_clicks, ARTIFACTS_DIR, FIGURES_DIR,
    MVIDEO_RED, DARK_SLATE, MUTED, save_stats,
)
from src.ner.labeling import WeakLabeler, tokenize, bio_to_entities

# --- load temp/labeling.py ATTR map without breaking src package ---
_spec = importlib.util.spec_from_file_location("temp_labeling", ROOT / "temp" / "labeling.py")
temp_lab = importlib.util.module_from_spec(_spec)
sys.modules["temp_labeling"] = temp_lab
_spec.loader.exec_module(temp_lab)
TEMP_ATTR_PATTERNS = temp_lab.ATTR_PATTERNS
temp_tokenize = temp_lab.tokenize

ensure_dirs()
apply_plot_style()
FIG = FIGURES_DIR / "complex_eda" / "markov"
FIG.mkdir(parents=True, exist_ok=True)

def save_local(fig, name):
    p = FIG / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print("saved", p)
    return p

SAMPLE_N = 150_000
print("ROOT:", ROOT)
print("temp ATTR patterns:", len(TEMP_ATTR_PATTERNS))
print("types:", sorted({t for _, t in TEMP_ATTR_PATTERNS}))
"""))

c02.append(code(r"""clicks = load_query_clicks(n=SAMPLE_N, seed=42, random=True)
ql = clicks["query_text"].astype(str).str.strip().str.lower()
uq = ql.drop_duplicates()
print(f"clicks={len(clicks):,}  unique_queries={uq.nunique():,}")
labeler = WeakLabeler.from_files(ARTIFACTS_DIR / "brands.txt", ARTIFACTS_DIR / "categories.txt")
"""))

c02.append(md("""## 1. Принцип: что именно предсказывает «марковская» типизация

### Пайплайн (двухступенчатый)

```text
query tokens
   → (A) span detection:  BIO ∈ {O, B-ATTR, I-ATTR, ...}
   → (B) attribute typing: span tokens → {memory_storage, size, power, ...}
```

Шаг (B) — это **классификатор типа**, а не второй NER.

### Модель 1-го порядка (идея «16 → гб ⇒ memory»)

Оцениваем по weak/regex-примерам условные частоты:

\\[
P(t_{i+1} \\mid t_i),\\qquad
P(\\text{type} \\mid t_i, t_{i+1})
\\]

Инференс для span `[16, гб]`:
1. смотрим биграмму `(16, гб)`;
2. берём argmax по эмпирическому \\(P(\\text{type}\\mid 16, гб)\\)  
   или эвристику: единица измерения (`гб`) → словарь типов.

Это **табличная / частотная** модель (сглаженный lookup), формально — марковская цепь / bigram Naive Bayes, **не** нейросеть.

### Markov ≈ правила?

| | Regex | Markov bigram | LogReg/RNN на span |
|---|---|---|---|
| Что хранит | явные шаблоны | частоты из данных | веса признаков / скрытое состояние |
| Новые формулировки | ломается | ловит, если встречались | обобщает лучше |
| Интерпретируемость | высокая | высокая | средняя/низкая |
| Нужны данные | нет | silver labels | silver/gold |
| Эквивалентность | — | **близко к выученным правилам** | нет |

**Вывод заранее:** чистый Markov на дискретных токенах **действительно близок к выученным правилам**. Смысл есть как:
1. **автоматический** способ собрать unit→type из данных (вместо ручного словаря);
2. бейзлайн перед LogReg/CRF-типизатором;
3. не как замена span-детекции для хвостов `g pro`.

RNN/Transformer на span имеет смысл, если тип зависит от **длинного/шумного** контекста (`ноут 16` без `гб`, омонимы `32` = Ом vs ГБ). На коротких запросах часто хватает bigram + словарь единиц.
"""))

c02.append(md("""## 2. Гипотеза про хвост `g pro`

Regex из `temp/labeling.py` ловит **число+единица** / стандарты (`4k`, `wifi`).  
Токены `g`, `pro`, `x`, `se` — линейка продукта → их должен ловить **словарь моделей / BRAND-алиасы / NER**, не ATTR-Markov.

Ниже измерим: какая доля «хвостовых» токенов после последней сущности похожа на ATTR-паттерн vs на буквенный хвост модели.
"""))

c02.append(code(r"""def last_entity_tail(query: str):
    tags = labeler.label_query(query)
    if not tags:
        return []
    last_ent = -1
    for i, (_, t) in enumerate(tags):
        if t != "O":
            last_ent = i
    if last_ent < 0 or last_ent >= len(tags) - 1:
        return []
    return [tok for tok, t in tags[last_ent + 1:] if t == "O"]

rng = np.random.default_rng(42)
sample_q = uq.sample(n=min(8000, len(uq)), random_state=42).tolist()

tail_toks = Counter()
n_with_tail = 0
n_tail_looks_numeric = 0
examples_model_tail = []
examples_attrish = []

for q in sample_q:
    tail = last_entity_tail(q)
    if not tail:
        continue
    n_with_tail += 1
    for t in tail:
        tail_toks[t] += 1
    joined = " ".join(tail)
    if any(ch.isdigit() for ch in joined):
        n_tail_looks_numeric += 1
        if len(examples_attrish) < 8:
            examples_attrish.append((q, tail))
    else:
        if len(examples_model_tail) < 8:
            examples_model_tail.append((q, tail))

print(f"queries with O-tail after entity: {n_with_tail}/{len(sample_q)} = {n_with_tail/len(sample_q):.1%}")
print(f"among them, tail contains digit: {n_tail_looks_numeric}/{n_with_tail} = {n_tail_looks_numeric/max(n_with_tail,1):.1%}")

tail_df = pd.DataFrame(tail_toks.most_common(25), columns=["token", "count"])
display(tail_df.head(15))

fig, ax = plt.subplots(figsize=(9, 3.8))
top = tail_df.head(20)
ax.barh(top["token"][::-1], top["count"][::-1], color=MVIDEO_RED)
ax.set_title("Самые частые O-токены после последней сущности (хвост)")
fig.tight_layout()
save_local(fig, "01_entity_tails.png")
plt.show()

print("Примеры буквенного хвоста (модель/линейка) — Markov ATTR не поможет:")
for q, t in examples_model_tail[:5]:
    print(" ", q, "→", t)
print("Примеры хвоста с цифрой — кандидат в ATTR / typing:")
for q, t in examples_attrish[:5]:
    print(" ", q, "→", t)
"""))

c02.append(md("""## 3. Coverage: старые vs temp ATTR regex

Считаем, какую долю запросов закрывает расширенная карта из `temp/labeling.py`.
"""))

c02.append(code(r"""from src.ner.labeling import ATTR_PATTERNS as SRC_ATTR_PATTERNS

def coverage(queries, patterns):
    cov = Counter()
    any_hit = 0
    for q in queries:
        hit = False
        for pat, name in patterns:
            if pat.search(q):
                cov[name] += 1
                hit = True
        if hit:
            any_hit += 1
    return any_hit / len(queries), cov

# subsample for speed
cov_q = uq.sample(n=min(30_000, len(uq)), random_state=0).tolist()
any_src, cov_src = coverage(cov_q, SRC_ATTR_PATTERNS)
any_tmp, cov_tmp = coverage(cov_q, TEMP_ATTR_PATTERNS)

cmp = pd.DataFrame({
    "source": ["src/ner/labeling.py", "temp/labeling.py"],
    "n_patterns": [len(SRC_ATTR_PATTERNS), len(TEMP_ATTR_PATTERNS)],
    "share_queries_with_any_attr": [any_src, any_tmp],
})
display(cmp)

tmp_share = pd.DataFrame({
    "attr_type": list(cov_tmp.keys()),
    "share": [cov_tmp[k] / len(cov_q) for k in cov_tmp],
}).sort_values("share", ascending=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(tmp_share["attr_type"], tmp_share["share"], color=MVIDEO_RED)
ax.set_title("temp/labeling.py — доля запросов по типам ATTR")
ax.tick_params(axis="x", rotation=55)
fig.tight_layout()
save_local(fig, "02_temp_attr_coverage.png")
plt.show()
display(tmp_share.head(12))
"""))

c02.append(md("""## 4. Бейзлайн Markov: собираем биграммы число→единица→тип

Из regex-матчей строим:
- `unit_to_types[unit] → Counter(types)`
- `bigram_to_types[(tok_i, tok_{i+1})] → Counter(types)`
- transition counts `P(next|prev)` среди токенов ATTR-спанов
"""))

c02.append(code(r"""import re as _re

def span_tokens_from_match(query: str, start: int, end: int):
    # tokenize normalized like temp (split 128гб)
    toks = temp_tokenize(query)
    return [t for t, s, e in toks if not (e <= start or s >= end)]

records = []  # dicts: type, tokens, bigrams, units
UNIT_RE = _re.compile(r"^[а-яa-z°\"]{1,8}$", _re.I)

for q in cov_q:
    for pat, typ in TEMP_ATTR_PATTERNS:
        for m in pat.finditer(q):
            toks = [t.lower().replace("ё", "е") for t in span_tokens_from_match(q, m.start(), m.end())]
            if not toks:
                # fallback: split match text
                toks = [t for t in _re.split(r"\s+", m.group(0).lower()) if t]
            bigrams = list(zip(toks, toks[1:])) if len(toks) >= 2 else []
            units = [t for t in toks if UNIT_RE.match(t) and not t.isdigit()]
            records.append({"type": typ, "tokens": toks, "bigrams": bigrams, "units": units, "text": m.group(0)})

rec_df = pd.DataFrame(records)
print("extracted ATTR spans:", len(rec_df))
display(rec_df["type"].value_counts().head(15).to_frame("count"))

unit_to_types = defaultdict(Counter)
bigram_to_types = defaultdict(Counter)
trans = defaultdict(Counter)  # prev -> next counts inside spans

for r in records:
    for u in r["units"]:
        unit_to_types[u][r["type"]] += 1
    for bg in r["bigrams"]:
        bigram_to_types[bg][r["type"]] += 1
        trans[bg[0]][bg[1]] += 1

# top transitions after digits
digit_prev = {k: v for k, v in trans.items() if any(ch.isdigit() for ch in k)}
rows = []
for prev, nxts in digit_prev.items():
    for nxt, c in nxts.most_common(3):
        rows.append({"prev": prev, "next": nxt, "count": c, "top_type": unit_to_types[nxt].most_common(1)[0][0] if unit_to_types[nxt] else "?"})
trans_df = pd.DataFrame(rows).sort_values("count", ascending=False)
display(trans_df.head(20))

fig, ax = plt.subplots(figsize=(9, 4))
show = trans_df.head(15).copy()
show["edge"] = show["prev"] + " → " + show["next"]
ax.barh(show["edge"][::-1], show["count"][::-1], color=MVIDEO_RED)
ax.set_title("Частые переходы после числовых токенов (Markov bigrams)")
fig.tight_layout()
save_local(fig, "03_markov_transitions.png")
plt.show()
"""))

c02.append(md("""## 5. Инференс бейзлайна и качество на silver labels

Алгоритм `predict_type(tokens)`:
1. если биграмма известна → majority type;
2. иначе по единицам измерения;
3. иначе `unknown`.

Метрика: accuracy vs regex-тип на held-out spans (это **верхняя** оценка — учитель = те же правила; проверяем, воспроизводит ли Markov regex).
"""))

c02.append(code(r"""def predict_type(tokens: list[str]) -> str:
    toks = [t.lower().replace("ё", "е") for t in tokens]
    # bigrams
    scores = Counter()
    for bg in zip(toks, toks[1:]):
        scores.update(bigram_to_types.get(bg, {}))
    if scores:
        return scores.most_common(1)[0][0]
    for t in toks:
        if unit_to_types[t]:
            scores.update(unit_to_types[t])
    if scores:
        return scores.most_common(1)[0][0]
    return "unknown"

# train/test split of records
rng = np.random.default_rng(0)
idx = np.arange(len(records))
rng.shuffle(idx)
cut = int(0.75 * len(idx))
train_idx, test_idx = idx[:cut], idx[cut:]

# rebuild dictionaries on TRAIN only
unit_to_types = defaultdict(Counter)
bigram_to_types = defaultdict(Counter)
for i in train_idx:
    r = records[i]
    for u in r["units"]:
        unit_to_types[u][r["type"]] += 1
    for bg in r["bigrams"]:
        bigram_to_types[bg][r["type"]] += 1

y_true, y_pred = [], []
for i in test_idx:
    r = records[i]
    y_true.append(r["type"])
    y_pred.append(predict_type(r["tokens"]))

acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
unk = float(np.mean([p == "unknown" for p in y_pred]))
print(f"test spans={len(test_idx)}  accuracy vs regex-teacher={acc:.3f}  unknown_rate={unk:.3f}")

cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
# show top types only
top_types = rec_df["type"].value_counts().head(8).index.tolist() + ["unknown"]
cm2 = cm.reindex(index=[t for t in top_types if t in cm.index], columns=[t for t in top_types if t in cm.columns], fill_value=0)
display(cm2)

# confusion heatmap (small)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm2.values, cmap="Reds")
ax.set_xticks(range(cm2.shape[1]))
ax.set_yticks(range(cm2.shape[0]))
ax.set_xticklabels(cm2.columns, rotation=60, ha="right")
ax.set_yticklabels(cm2.index)
ax.set_title("Markov typer vs regex teacher (top types)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
save_local(fig, "04_markov_confusion.png")
plt.show()
"""))

c02.append(md("""## 6. Когда Markov НЕ поможет: омонимы и отсутствие единицы

Примеры, где нужна более сильная модель (LogReg на контексте / RNN / весь query):
- `32` без единицы — память? сопротивление? размер?
- `55` — дюймы TV vs что-то ещё
- тип зависит от категории (`наушники 32` vs `ssd 32`)
"""))

c02.append(code(r"""# ambiguous numbers: same number token appears in multiple ATTR types
num_types = defaultdict(Counter)
for r in records:
    for t in r["tokens"]:
        if any(ch.isdigit() for ch in t):
            num_types[t][r["type"]] += 1

amb = []
for num, ctr in num_types.items():
    if len(ctr) >= 2 and sum(ctr.values()) >= 20:
        amb.append({
            "number_token": num,
            "n_types": len(ctr),
            "total": sum(ctr.values()),
            "distribution": dict(ctr.most_common(4)),
        })
amb_df = pd.DataFrame(amb).sort_values(["n_types", "total"], ascending=False)
display(amb_df.head(15))

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(amb_df["n_types"], bins=range(2, int(amb_df["n_types"].max()) + 2), color=DARK_SLATE, edgecolor="white")
ax.set_title("Сколько разных ATTR-типов бывает у одного числового токена")
ax.set_xlabel("# types")
fig.tight_layout()
save_local(fig, "05_number_ambiguity.png")
plt.show()

print("Вердикт: при наличии единицы Markov ≈ выученное правило (ок как бейзлайн).")
print("Без единицы / при омонимии нужен контекст категории → LogReg(span+query) или лёгкий RNN/Transformer.")
"""))

c02.append(md(r"""## 7. Практический вердикт и план

| Вопрос | Ответ |
|---|---|
| Имеет ли смысл Markov? | **Да как бейзлайн типизации** и автосбор unit→type из `temp` regex |
| Эквивалентен ли правилам? | **Почти**, если признаки = те же токены; плюс — частоты из данных |
| Нужна ли сразу RNN? | **Не обязательно.** Сначала: BIO ATTR → Markov/LogReg typer. RNN — если gold покажет ошибку на омонимах |
| Починит ли `g pro`? | **Нет.** Это не ATTR; нужны словарь линеек / NER на MODEL / brand aliases |
| Масштабирование | O(|V|^2) биграмм редко плотно; на практике — hashmap по встреченным биграммам, RAM крошечный, latency << 1 мс |

### Рекомендуемый следующий шаг
1. Перенести `temp/labeling.py` ATTR map (+ улучшенный `tokenize`) в `src/ner/labeling.py`.
2. Детект span: словари + CRF (`B-ATTR`).
3. Typing: Markov/unit lexicon → затем LogReg(`tokens + category_hint`).
4. Отдельно: словарь product-line (`g pro`, `v15`, `air`) как `MODEL` или часть BRAND.
"""))

c02.append(code(r"""out = {
    "sample_clicks": int(len(clicks)),
    "temp_n_patterns": len(TEMP_ATTR_PATTERNS),
    "src_n_patterns": len(SRC_ATTR_PATTERNS),
    "attr_coverage_src": float(any_src),
    "attr_coverage_temp": float(any_tmp),
    "n_attr_spans": int(len(records)),
    "markov_test_accuracy_vs_regex": float(acc),
    "markov_unknown_rate": float(unk),
    "n_ambiguous_numbers": int(len(amb_df)),
    "queries_with_entity_tail_rate": float(n_with_tail / len(sample_q)),
    "tail_with_digit_rate": float(n_tail_looks_numeric / max(n_with_tail, 1)),
}
save_stats(out, "markov_attr_eda_stats.json")
pd.DataFrame(list(out.items()), columns=["metric", "value"])
"""))


write_nb(OUT / "03_attr_types_eda.ipynb", c02)


# ---------------------------------------------------------------------------
# 03_click_eda
# ---------------------------------------------------------------------------
c03: list[dict] = []

c03.append(md("""# 03. Click relevance: структура разметки и пайплайн очистки кликов

Идея: клик query→SKU — **шумный** weak label. Хотим бинарный классификатор

\\[
f(query, sku) \\rightarrow \\{0,1\\}
\\]

«пара релевантна / нет», обученный на **ручной** разметке 100+ пар; дальше модель доразмечает остальное и фильтрует шелуху для brand/category clf и retrieval.

> Пока разметки нет — этот ноутбук задаёт **схему, семпл кандидатов, инструкцию разметчику и скелет обучения**.
"""))

c03.append(md("""## 0. Setup
"""))

c03.append(code(r"""%matplotlib inline
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name in {"complex_eda", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "complex_eda" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    apply_plot_style, ensure_dirs, load_query_clicks, ARTIFACTS_DIR, FIGURES_DIR,
    MVIDEO_RED, DARK_SLATE,
)

ensure_dirs()
apply_plot_style()

LABEL_DIR = ARTIFACTS_DIR / "click_relevance"
LABEL_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_PATH = LABEL_DIR / "candidates_to_label.csv"
LABELS_PATH = LABEL_DIR / "labels.csv"
LABELS_JSONL = LABEL_DIR / "labels.jsonl"
README_LABEL = LABEL_DIR / "LABELING_GUIDE.md"

SAMPLE_N = 200_000
print("ROOT:", ROOT)
print("label dir:", LABEL_DIR)
"""))

c03.append(md("""## 1. Зачем это нужно (связь с EDA)

Из `01_methods_eda`:
- ~73–74% кликов: бренд не написан в запросе → учим brand clf на кликах;
- ~17% запросов кликают ≥2 бренда → часть кликов случайна / exploratory;
- позиция и цена есть, `user_id` нет → классический user-RecSys недоступен, но **pairwise relevance** — да.

Ручная разметка 100–300 пар → supervised denoiser → чище train для brand/category и retrieval.
"""))

c03.append(md("""## 2. Как размечать

### Файл для вашей разметки
Основной: `artifacts/click_relevance/labels.csv`  
(дублировать можно в `labels.jsonl` — удобнее для потоковой доразметки).

### Схема колонок

| column | тип | описание |
|---|---|---|
| `pair_id` | str | стабильный id (`hash query+sku_id`) |
| `query` | str | текст запроса |
| `sku_id` | str/int | id товара |
| `sku_name` | str | название из клика |
| `sku_brand_name` | str | бренд клика |
| `sku_position` | int | позиция в выдаче |
| `sku_price` | float | цена |
| `label` | int | **1** = релевантно, **0** = нет, пусто = ещё не размечено |
| `confidence` | int | 1–3 (опц.) насколько уверены |
| `notes` | str | свободный комментарий |
| `annotator` | str | кто разметил |
| `labeled_at` | str | ISO дата |

### Правила решения `label`

Ставьте **1**, если по запросу разумно ожидать этот SKU (тот же intent: категория/бренд/модель/ключевой ATTR).  
Ставьте **0**, если клик явный шум (другая категория, случайный бренд, аксессуар вместо устройства без сигнала в запросе и т.п.).

Примеры:
| query | sku | label |
|---|---|---:|
| `ноутбук asus 16` | ASUS VivoBook 16/512 | 1 |
| `ноутбук asus 16` | Чехол для MacBook | 0 |
| `наушники logitech g pro` | Logitech G PRO X | 1 |
| `холодильник` | Стиральная машина LG | 0 |

### Процесс
1. Запустите ячейку «кандидаты» → появится `candidates_to_label.csv`.
2. Скопируйте строки в `labels.csv` **или** заполняйте `label` прямо в candidates и сохраните как `labels.csv`.
3. Цель первой волны: **≥ 100** пар, лучше **200–300**, баланс примерно 50/50 если возможно (добавляем hard negatives).
4. Не править `pair_id`.
"""))

c03.append(code(r"""guide = '''# Click relevance labeling guide

## Files
- `candidates_to_label.csv` — пул пар для разметки (генерируется ноутбуком)
- `labels.csv` — ваши ответы (тот же schema + заполненный `label`)
- `labels.jsonl` — опционально, по одной JSON-записи на строку

## label
- 1 = query и SKU соответствуют одному поисковому намерению
- 0 = клик шумовой / другой intent

## Tips
- Смотрите бренд, категорию в названии, ключевые ATTR в query
- Если сомневаетесь — `confidence=1` и короткий `notes`
- Старайтесь размечать и top-position клики, и случайные negatives
'''
README_LABEL.write_text(guide, encoding="utf-8")
print("wrote", README_LABEL)
"""))

c03.append(md("""## 3. Генерация кандидатов на разметку

Стратификация:
1. **позитивы-кандидаты** — частые запросы, клик в топ-3 позиции;
2. **сомнительные** — тот же query, но другой бренд / позиция ≥ 10;
3. **hard negatives** — случайный SKU к тому же query (для баланса 0).
"""))

c03.append(code(r"""clicks = load_query_clicks(n=SAMPLE_N, seed=42, random=True)
df = clicks.copy()
df["query"] = df["query_text"].astype(str).str.strip()
df["brand"] = df["sku_brand_name"].astype(str).str.strip()
df = df[df["query"].str.len() >= 2]

# frequent queries
q_freq = df["query"].str.lower().value_counts()
frequent = set(q_freq[q_freq >= 20].index)

sub = df[df["query"].str.lower().isin(frequent)].copy()

rng = np.random.default_rng(42)

# 1) top-position candidates
top = sub[sub["sku_position"] <= 3].drop_duplicates(["query", "sku_id"]).sample(n=min(80, len(sub)), random_state=42)

# 2) suspicious: deep position
deep = sub[sub["sku_position"] >= 10].drop_duplicates(["query", "sku_id"])
deep = deep.sample(n=min(60, len(deep)), random_state=1) if len(deep) else deep

# 3) hard negatives: random sku for a query
neg_rows = []
queries_sample = sub["query"].drop_duplicates().sample(n=min(60, sub["query"].nunique()), random_state=2)
sku_pool = sub[["sku_id", "sku_name", "brand", "sku_price", "sku_position"]].drop_duplicates("sku_id")
for q in queries_sample:
    true_skus = set(sub.loc[sub["query"] == q, "sku_id"])
    cand = sku_pool[~sku_pool["sku_id"].isin(true_skus)]
    if cand.empty:
        continue
    row = cand.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    neg_rows.append({
        "query": q,
        "sku_id": row["sku_id"],
        "sku_name": row["sku_name"],
        "sku_brand_name": row["brand"],
        "sku_position": row["sku_position"],
        "sku_price": row["sku_price"],
        "candidate_source": "random_negative",
    })

def pack(frame, source):
    out = frame[["query", "sku_id", "sku_name", "brand", "sku_position", "sku_price"]].copy()
    out = out.rename(columns={"brand": "sku_brand_name"})
    out["candidate_source"] = source
    return out

cand = pd.concat([
    pack(top, "top_position"),
    pack(deep, "deep_position"),
    pd.DataFrame(neg_rows),
], ignore_index=True)

cand["pair_id"] = [
    str(abs(hash((str(q).lower(), str(s))))) for q, s in zip(cand["query"], cand["sku_id"])
]
cand["label"] = pd.NA
cand["confidence"] = pd.NA
cand["notes"] = ""
cand["annotator"] = ""
cand["labeled_at"] = ""

# de-dup
cand = cand.drop_duplicates("pair_id").reset_index(drop=True)
cand.to_csv(CANDIDATES_PATH, index=False, encoding="utf-8-sig")
print("candidates:", len(cand), "→", CANDIDATES_PATH)
display(cand["candidate_source"].value_counts().to_frame("n"))
display(cand.head(8))

# seed empty labels file if missing
if not LABELS_PATH.exists():
    seed = cand.head(0).copy()
    seed.to_csv(LABELS_PATH, index=False, encoding="utf-8-sig")
    print("created empty", LABELS_PATH)
else:
    print("labels file already exists:", LABELS_PATH)
"""))

c03.append(md("""## 4. Загрузка вашей разметки (когда появится)

Ячейка безопасно читает `labels.csv` и показывает баланс классов. Пока файл пустой — просто заглушка.
"""))

c03.append(code(r"""labels = pd.read_csv(LABELS_PATH) if LABELS_PATH.exists() else pd.DataFrame()
print("rows in labels.csv:", len(labels))
if len(labels) and "label" in labels.columns:
    labeled = labels.dropna(subset=["label"])
    print("labeled:", len(labeled))
    if len(labeled):
        display(labeled["label"].value_counts(dropna=False).to_frame("n"))
        display(labeled.head())
    else:
        print("Файл есть, но label ещё не заполнены — разметьте candidates и сохраните в labels.csv")
else:
    print("Разметки пока нет. Откройте candidates_to_label.csv и заполните labels.csv по гайду.")
"""))

c03.append(md("""## 5. Скелет модели (запустится после ≥50–100 лейблов)

Фичи (черновик):
- char TF-IDF по `query` и `sku_name` + cosine;
- совпадение бренда / токенов;
- `sku_position`, log(price);
- (позже) эмбеддинги.

Таргет: `label ∈ {0,1}`.
"""))

c03.append(code(r"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

MIN_LABELS = 50

if len(labels) and labels["label"].notna().sum() >= MIN_LABELS:
    lab = labels.dropna(subset=["label"]).copy()
    lab["label"] = lab["label"].astype(int)
    lab["q"] = lab["query"].astype(str)
    lab["t"] = lab["sku_name"].astype(str)

    # simple overlap features
    def tokset(s):
        return set(str(s).lower().split())
    lab["jaccard"] = [len(tokset(a) & tokset(b)) / max(1, len(tokset(a) | tokset(b))) for a, b in zip(lab["q"], lab["t"])]
    lab["brand_in_query"] = [
        str(b).lower() in str(q).lower() if len(str(b)) >= 2 else False
        for q, b in zip(lab["q"], lab["sku_brand_name"])
    ]

    Xtr, Xte, ytr, yte = train_test_split(lab, lab["label"], test_size=0.25, random_state=42, stratify=lab["label"])
    vec_q = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20_000)
    vec_t = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20_000)
    Qtr, Qte = vec_q.fit_transform(Xtr["q"]), vec_q.transform(Xte["q"])
    Ttr, Tte = vec_t.fit_transform(Xtr["t"]), vec_t.transform(Xte["t"])

    # cosine between query/title in a joint space (fit on concat)
    vec_j = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20_000)
    Jtr = vec_j.fit_transform(Xtr["q"] + " || " + Xtr["t"])
    Jte = vec_j.transform(Xte["q"] + " || " + Xte["t"])

    num_tr = np.c_[Xtr["jaccard"].values, Xtr["brand_in_query"].astype(float).values]
    num_te = np.c_[Xte["jaccard"].values, Xte["brand_in_query"].astype(float).values]

    Xtr_m = hstack([Qtr, Ttr, Jtr, num_tr])
    Xte_m = hstack([Qte, Tte, Jte, num_te])

    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(Xtr_m, ytr)
    pred = clf.predict(Xte_m)
    proba = clf.predict_proba(Xte_m)[:, 1]
    print(classification_report(yte, pred, digits=3))
    try:
        print("ROC-AUC:", round(roc_auc_score(yte, proba), 3))
    except Exception as e:
        print("AUC n/a:", e)
else:
    print(f"Нужно ≥{MIN_LABELS} размеченных строк в {LABELS_PATH}")
    print("Сейчас можно размечать candidates_to_label.csv → labels.csv")
"""))

c03.append(md("""## 6. Как использовать модель дальше (план)

```text
все клики
  → score = P(relevant|query,sku)
  → порог τ (например 0.6)
  → clean_clicks
       → brand/category classifier
       → retrieval positives
       → eval без шелухи
```

Active learning: периодически брать пары с `score ≈ 0.5`, доразмечать руками, дообучать.

### Чеклист первой волны разметки
- [ ] 100+ пар в `labels.csv`
- [ ] есть и 0, и 1
- [ ] есть top_position и random_negative
- [ ] перезапустить секцию 5
- [ ] сохранить `models/click_relevance_logreg.joblib` (добавим, когда появятся лейблы)
"""))

c03.append(code(r"""# Быстрая сводка путей
pd.DataFrame([
    {"path": str(CANDIDATES_PATH.relative_to(ROOT)), "role": "пул на разметку"},
    {"path": str(LABELS_PATH.relative_to(ROOT)), "role": "ваши 0/1 лейблы"},
    {"path": str(LABELS_JSONL.relative_to(ROOT)), "role": "опциональный jsonl"},
    {"path": str(README_LABEL.relative_to(ROOT)), "role": "краткий гайд"},
])
"""))

write_nb(OUT.parent / "_legacy" / "03_click_eda.ipynb", c03)
print("done")
