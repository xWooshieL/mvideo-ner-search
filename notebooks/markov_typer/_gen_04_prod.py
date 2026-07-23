"""Generate 04_attr_type_prod.ipynb — thin wrapper around prod artifacts + interactive sanity."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "04_attr_type_prod.ipynb"
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
    """# 04. ATTR type — production pack

Прод-версия типизатора ATTR после отладки учителя и sanity.

Полный прогон обучения: `python notebooks/markov_typer/_run_04_prod.py`  
Отчёт: [`attr_type_prod_report.md`](./attr_type_prod_report.md)

## Фичи на примере `ноутбук asus 16 г`

| колонка | значение | роль |
|---|---|---|
| `span_text` | `16 г` | TF-IDF **только** на span |
| `context_text` | `asus ноутбук` | бренд+категория отдельно |
| `query_masked` | `ноутбук asus <ATTR>` | контекст без чужих единиц |
"""
)

md("## Setup + load prod model")

code(
    """import sys, json
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name in {"markov_typer", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "markov_typer" else ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import display, Markdown
from src.data_utils import MODELS, ARTIFACTS_DIR, FIGURES_DIR
from src.ner.attr_type_clf import predict_attr_type, load_policy
from src.ner.labeling import _guess_attr_type

OUT = ARTIFACTS_DIR / "attr_type"
policy = load_policy(OUT / "inference_policy.json")
print(json.dumps(policy, ensure_ascii=False, indent=2)[:800])
display(pd.read_csv(OUT / "prod_models_summary.csv"))
display(pd.read_csv(OUT / "prod_sanity_10.csv"))
"""
)

md("## Feature walkthrough")

code(
    """feat = json.loads((OUT / "prod_feature_example.json").read_text(encoding="utf-8"))
print("example query:", feat["example_query"])
print("entities:", feat["entities"])
for name, block in feat["feature_columns"].items():
    print(f"\\n[{name}] = {block['value']!r}")
    print("  how:", block["how"])
    print("  why:", block["why"])
print("\\nprediction:", feat["prediction"])
"""
)

md("## Interactive sanity (правь SPAN)")

code(
    """SPAN = "16 г"
BRAND = "asus"
CATEGORY = "ноутбук"
QUERY_MASKED = "ноутбук asus <ATTR>"

det = predict_attr_type(
    SPAN, brand=BRAND, category=CATEGORY, query_masked=QUERY_MASKED, return_details=True
)
print("teacher:", _guess_attr_type(SPAN))
print(json.dumps(det, ensure_ascii=False, indent=2))

BATCH = [
    ("16 г", "asus", "ноутбук"),
    ("16 гб", "asus", "ноутбук"),
    ("256 g", "samsung", "смартфон"),
    ("5 g", "samsung", "смартфон"),
    ("2 кг", "bosch", "пылесос"),
    ("150 грамм", "", "весы"),
    ("1920x1080", "xiaomi", "монитор"),
    ("4k", "lg", "телевизор"),
    ("15.6 дюйм", "asus", "ноутбук"),
    ("g pro", "logitech", "наушники"),
]
rows = []
for span, b, c in BATCH:
    d = predict_attr_type(span, brand=b, category=c, query_masked=f"{c} {b} <ATTR>".strip(), return_details=True)
    rows.append({
        "span": span, "brand": b, "category": c,
        "teacher": _guess_attr_type(span),
        "pred": d["label"], "raw": d["raw_pred"],
        "conf": round(d["confidence"], 3), "reason": d["reason"],
    })
display(pd.DataFrame(rows))
"""
)

md(
    """## Картинки

![](../../figures/attr_type/prod_01_models_threshold.png)

![](../../figures/attr_type/prod_02_class_support.png)
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
print("wrote", NB)
