# Markov ATTR typer

Типизация span `B/I-ATTR` → `memory_storage` / `size` / `color` / …  
Baseline Markov: [`src/ner/markov_typer.py`](../../src/ner/markov_typer.py).  
Учитель типов: [`src/ner/labeling.py`](../../src/ner/labeling.py) (`ATTR_PATTERNS` + `WeakLabeler`).

| Файл | |
|---|---|
| [`01_markov_eda.ipynb`](./01_markov_eda.ipynb) | смысл Markov, agreement с regex |
| [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) | silver + EDA на WeakLabeler |
| [`attr_type_silver.md`](./attr_type_silver.md) | схема датасета / маски |
| [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb) | **4 TF-IDF clf + Markov**, метрики |
| [`attr_type_classifier.md`](./attr_type_classifier.md) | отчёт по моделям |

Дизайн clf: n-grams только на `span_text`; brand/category/masked query — отдельно.
