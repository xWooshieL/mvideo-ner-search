# Markov ATTR typer

Типизация span `B/I-ATTR` → `memory_storage` / `size` / …  
Учитель: [`src/ner/labeling.py`](../../src/ner/labeling.py).

| Файл | |
|---|---|
| [`01_markov_eda.ipynb`](./01_markov_eda.ipynb) | смысл Markov |
| [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) | silver + EDA |
| [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb) | **обучение prod** (4 clf + Markov, policy, sanity) |
| [`04_attr_type_prod.ipynb`](./04_attr_type_prod.ipynb) | интерактив после обучения / разбор фич |
| [`attr_type_prod_report.md`](./attr_type_prod_report.md) | отчёт |

Прод-артефакты: `models/attr_type_clf.joblib`, `artifacts/attr_type/inference_policy.json`.  
Silver (канон): `artifacts/silver/attr_type/` (зеркало в `artifacts/attr_type/`).  
Словари: `artifacts/dicts/` (см. [`data/README.md`](../../data/README.md)).  
Эквивалент CLI: `python notebooks/markov_typer/_run_04_prod.py`
