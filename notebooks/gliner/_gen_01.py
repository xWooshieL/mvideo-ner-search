"""Generate notebooks/gliner/01_gliner_zero_shot.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "01_gliner_zero_shot.ipynb"
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
    """# 01. GLiNER — zero-shot baseline

Хвост каскада: `spellfix → rules/dicts → CRF → GLiNER`. Здесь — точка отсчёта
**без обучения**: что даёт готовый чекпоинт [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1)
на наших запросах из коробки.

## Библиотека vs веса

| | Что это |
|---|---|
| `pip install gliner` | движок: `GLiNER.from_pretrained`, `predict_entities`, `train_model` |
| `urchade/gliner_multi-v2.1` | конкретный чекпоинт (мультиязычный, apache-2.0, ~209M) |

Обучение (fine-tune) — в [`02_gliner_finetune.ipynb`](./02_gliner_finetune.ipynb).

CLI: `python notebooks/gliner/_run_01.py`
"""
)

md("## 0. Setup")
code(
    """%matplotlib inline
import sys, json, time, warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name in {"gliner", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "gliner" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import (
    ensure_dirs, apply_plot_style, FIGURES_DIR, METRICS_DIR, MVIDEO_RED, DARK_SLATE,
)
from src.ner.gliner_utils import (
    EDGE_CASES, OUR_LABELS, LABEL_PROMPTS, evaluate_rows_char, gliner_labels,
    load_gold_bio_rows, predict_edge_cases,
)

warnings.filterwarnings("ignore")
ensure_dirs()
apply_plot_style()

MODEL_ID = "urchade/gliner_multi-v2.1"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "gliner"
FIG.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.3  # см. 02: 0.5 режет верные ATTR-спаны (score ~0.33-0.42)
SCHEMES = ["en", "ru"]
print("GOLD", GOLD)
"""
)

md(
    """## 1. Gold BIO -> строки для eval

Ретокенизируем тем же `tokenize()`, что и CRF — теги/токены согласованы."""
)
code(
    """rows = load_gold_bio_rows(GOLD)
print("gold rows usable (tokenize-aligned):", len(rows))
rows[0]
"""
)

md("## 2. Загрузка GLiNER (уже скачан в HF-кэш при первом вызове)")
code(
    """from gliner import GLiNER

t0 = time.perf_counter()
model = GLiNER.from_pretrained(MODEL_ID)
print(f"loaded in {time.perf_counter() - t0:.1f}s, type={type(model).__name__}")
"""
)

md(
    """## 3. Промпт-схемы: en vs ru

GLiNER — zero-shot по произвольным текстовым лейблам. Сравниваем два набора
промптов (`brand` / `бренд` и т.д.) — какой лучше заходит модели на наших запросах."""
)
code(
    """scheme_metrics = {}
for scheme in SCHEMES:
    labels = gliner_labels(scheme)
    print(f"--- scheme={scheme} labels={labels} ---")
    m = evaluate_rows_char(model, rows, scheme=scheme, threshold=THRESHOLD)
    scheme_metrics[scheme] = m
    print(f"microF1={m['micro']['f1']:.3f} P={m['micro']['precision']:.3f} "
          f"R={m['micro']['recall']:.3f} avg_latency={m['avg_latency_ms']:.1f}ms/query")

best_scheme = max(scheme_metrics, key=lambda s: scheme_metrics[s]["micro"]["f1"])
print("best_scheme:", best_scheme)
"""
)

md("## 4. Per-label F1")
code(
    """labs = OUR_LABELS
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(labs)); w = 0.35
en_f1 = [scheme_metrics["en"]["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
ru_f1 = [scheme_metrics["ru"]["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
ax.bar(x - w/2, en_f1, w, color=MVIDEO_RED, label="prompts=en")
ax.bar(x + w/2, ru_f1, w, color=DARK_SLATE, label="prompts=ru")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 1.05)
ax.set_title(f"GLiNER {MODEL_ID} — zero-shot"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "01_zero_shot_f1.png", dpi=120, bbox_inches="tight")
plt.show()
"""
)

md(
    """## 5. Edge cases (хвостовые кейсы, не из gold)

Опечатки, кириллица вместо латиницы, короткие/длинные запросы, чужой алфавит —
то, на чём rules+CRF обычно проседают."""
)
code(
    """edge_rows = predict_edge_cases(model, EDGE_CASES, scheme=best_scheme, threshold=THRESHOLD)
for e in edge_rows:
    print(f"{e['query']!r} -> {[(x['label'], x['text']) for x in e['entities']]}")
"""
)

md("## 6. Save artifacts")
code(
    """result = {
    "model_id": MODEL_ID, "threshold": THRESHOLD, "gold_n": len(rows),
    "schemes": scheme_metrics, "best_scheme": best_scheme, "edge_cases": edge_rows,
    "note": "zero-shot baseline (без fine-tune) — точка отсчёта для notebook 02",
}
(METRICS_DIR / "gliner_zero_shot.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("saved", METRICS_DIR / "gliner_zero_shot.json")
"""
)

md(
    """## Выводы

1. Zero-shot: **высокая precision, низкий recall** (модель осторожна без обучения) —
   много сущностей (`CATEGORY`/`MODEL`/`ATTR`) просто не находит.
2. Схема промптов `ru` обычно чуть лучше `en` на русских запросах.
3. GLiNER медленнее CRF (десятки-сотни ms/запрос на CPU) — роль **хвост** каскада,
   не замена CRF на всём трафике.
4. Дальше: [`02_gliner_finetune.ipynb`](./02_gliner_finetune.ipynb) — обучение на gold + silver.

Headless: `python notebooks/gliner/_run_01.py`
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
