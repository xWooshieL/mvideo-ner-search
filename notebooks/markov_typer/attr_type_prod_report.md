# ATTR type classifier — prod report

Ноутбук обучения: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb)  
Модель: **`sgd_span_ctx_masked`** → `models/attr_type_clf.joblib`  
Policy: `artifacts/attr_type/inference_policy.json` (τ=`0.55`)  
Sanity: **8/10**

## Классы train

| class | train_n | val_n |
|---|---:|---:|
| `color` | 475 | 119 |
| `UNKNOWN` | 362 | 90 |
| `size` | 347 | 87 |
| `memory_storage` | 222 | 56 |
| `connectivity` | 141 | 35 |
| `volume` | 122 | 30 |
| `weight` | 89 | 22 |
| `power` | 77 | 19 |
| `resolution_exact` | 38 | 10 |
| `frequency` | 33 | 8 |
| `voltage` | 31 | 8 |
| `current` | 22 | 6 |
| `dimensions` | 18 | 5 |
| `resolution_standard` | 18 | 5 |
| `time` | 15 | 3 |

## Фичи (пример `ноутбук asus 16 г`)

| фича | значение |
|---|---|
| `span_text` | `16 г` — char TF-IDF |
| `context_text` | `asus ноутбук` |
| `query_masked` | `ноутбук asus <ATTR>` — **word** TF-IDF |

## Сводка моделей

| model | acc | f1_macro | multi_f1_macro | f1_UNKNOWN |
|---|---:|---:|---:|---:|
| `sgd_span_ctx_masked` | 0.984 | **0.956** | 0.985 | 0.978 |
| `markov_lookup` | 0.966 | 0.929 | 0.975 | 0.917 |
| `logreg_span_char` | 0.974 | 0.922 | 1.000 | 0.941 |
| `logreg_span_wordchar` | 0.978 | 0.921 | 0.988 | 0.946 |
| `logreg_span_ctx` | 0.966 | 0.907 | 0.988 | 0.925 |

Reject τ=0.55: см. `inference_policy.json`.

## Sanity 10

| span | expect | pred | conf | ok | teacher |
|---|---|---|---:|:---:|---|
| `16 г` | memory_storage | memory_storage | 0.94 | OK | memory_storage |
| `16 гб` | memory_storage | memory_storage | 0.98 | OK | memory_storage |
| `256 g` | memory_storage | memory_storage | 0.97 | OK | memory_storage |
| `5 g` | UNKNOWN | UNKNOWN | 0.21 | OK | other |
| `2 кг` | weight | weight | 0.99 | OK | weight |
| `150 грамм` | weight | size | 0.60 | FAIL | weight |
| `1920x1080` | resolution_exact | resolution_exact | 0.90 | OK | resolution_exact |
| `4k` | resolution_standard | color | 0.71 | FAIL | resolution_standard |
| `15.6 дюйм` | size | size | 0.98 | OK | size |
| `g pro` | UNKNOWN | UNKNOWN | 1.00 | OK | other |

## Per-class F1

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| connectivity | 1.000 | 1.000 | 1.000 | 35 |
| current | 1.000 | 1.000 | 1.000 | 6 |
| frequency | 1.000 | 1.000 | 1.000 | 8 |
| power | 1.000 | 1.000 | 1.000 | 19 |
| volume | 1.000 | 1.000 | 1.000 | 30 |
| time | 1.000 | 1.000 | 1.000 | 3 |
| resolution_standard | 1.000 | 1.000 | 1.000 | 5 |
| weight | 1.000 | 1.000 | 1.000 | 22 |
| voltage | 1.000 | 1.000 | 1.000 | 8 |
| color | 1.000 | 0.992 | 0.996 | 119 |
| size | 0.989 | 0.989 | 0.989 | 87 |
| memory_storage | 0.966 | 1.000 | 0.982 | 56 |
| UNKNOWN | 0.978 | 0.978 | 0.978 | 90 |
| resolution_exact | 0.750 | 0.900 | 0.818 | 10 |
| dimensions | 1.000 | 0.400 | 0.571 | 5 |
