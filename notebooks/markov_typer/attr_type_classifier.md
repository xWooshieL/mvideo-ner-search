# ATTR type classifier — отчёт

Ноутбук: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb)  
Silver: [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) / [`attr_type_silver.md`](./attr_type_silver.md)  
Лучшая sklearn: **`sgd_span_ctx_masked`** → `models/attr_type_clf.joblib`

## Дизайн

- Char/word n-grams **только по `span_text`**.
- Контекст: `brand`+`category` и/или `query_masked_all_attr`.
- `UNKNOWN` = редкие типы + `other` (+ modelish rule на инференсе).
- Unit-aug из silver (`is_aug`).

## Сводка моделей (val)

| model | acc | f1_macro | f1_micro | multi_f1_macro | f1_UNKNOWN |
|---|---:|---:|---:|---:|---:|
| `sgd_span_ctx_masked` | 0.977 | **0.943** | 0.977 | 0.931 | 0.889 |
| `logreg_span_char` | 0.957 | 0.909 | 0.957 | 0.915 | 0.605 |
| `logreg_span_ctx` | 0.963 | 0.907 | 0.963 | 0.882 | 0.745 |
| `logreg_span_wordchar` | 0.956 | 0.896 | 0.956 | 0.897 | 0.617 |
| `markov_lookup` | 0.937 | 0.895 | 0.937 | 0.899 | 0.404 |

`multi_f1_macro` — строки с ≥2 ATTR в запросе.

## Per-class F1 (best)

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| color | 1.000 | 1.000 | 1.000 | 201 |
| resolution_standard | 1.000 | 1.000 | 1.000 | 10 |
| memory_storage | 0.987 | 1.000 | 0.994 | 311 |
| power | 0.984 | 1.000 | 0.992 | 62 |
| weight | 1.000 | 0.968 | 0.984 | 95 |
| volume | 0.961 | 1.000 | 0.980 | 49 |
| size | 0.957 | 1.000 | 0.978 | 132 |
| frequency | 0.933 | 1.000 | 0.966 | 14 |
| connectivity | 1.000 | 0.909 | 0.952 | 55 |
| current | 1.000 | 0.875 | 0.933 | 8 |
| time | 1.000 | 0.833 | 0.909 | 6 |
| UNKNOWN | 0.917 | 0.863 | 0.889 | 51 |
| dimensions | 1.000 | 0.708 | 0.829 | 24 |
| voltage | 0.667 | 1.000 | 0.800 | 12 |

## Вывод

- Лучшая модель: `sgd_span_ctx_masked` (macro F1 **0.943**).
- Markov baseline: macro **0.895**.
- Слабые классы — в per-class таблице (часто `dimensions` / хвост).
