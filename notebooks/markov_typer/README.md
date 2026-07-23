# Markov ATTR typer

Типизация span `B/I-ATTR` → `memory_storage` / `size` / …  
Учитель: [`src/ner/labeling.py`](../../src/ner/labeling.py).

| Файл | |
|---|---|
| [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) | silver + EDA |
| [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb) | **обучение prod** (4 clf + Markov baseline, policy, sanity) |
| [`attr_type_prod_report.md`](./attr_type_prod_report.md) | отчёт (метрики, sanity, gold agree) |

> Разведочный `01_markov_eda` перенесён в `notebooks/complex_eda/03_attr_types_eda.ipynb`
> (весь EDA собран в `complex_eda`). Марковская цепь здесь — только baseline против прод-clf.

Прод-модель: `models/attr_type_clf.joblib` (лучший — `sgd_span_ctx_masked`, silver-val macro-F1 ≈ 0.95).  
Policy: `artifacts/silver/attr_type/inference_policy.json`.  
Silver (канон): `artifacts/silver/attr_type/`. Словари: `artifacts/dicts/`.  
Метрики на gold: `artifacts/metrics/gold_metrics_attr_type.csv`.  
CLI: `python notebooks/markov_typer/_run_02.py` (silver) → `python notebooks/markov_typer/_run_04_prod.py` (обучение).
