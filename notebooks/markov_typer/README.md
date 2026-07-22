# Markov ATTR typer

Типизация span `B/I-ATTR` → `memory_storage` / `size` / `color` / …  
Baseline Markov: [`src/ner/markov_typer.py`](../../src/ner/markov_typer.py).  
Учитель типов: [`src/ner/labeling.py`](../../src/ner/labeling.py) (`ATTR_PATTERNS` + `WeakLabeler`).

| Файл | |
|---|---|
| [`01_markov_eda.ipynb`](./01_markov_eda.ipynb) | смысл Markov, agreement с regex |
| [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) | **silver + EDA** на WeakLabeler (схема, маски) |
| [`attr_type_silver.md`](./attr_type_silver.md) | договорённости по датасету |
| [`attr_type_classifier.md`](./attr_type_classifier.md) | отчёт clf (старый прогон; переобучать после нового silver) |

Дизайн: одна строка = один ATTR-span; n-grams типа — только `span_text`; чужие ATTR маскируем.
