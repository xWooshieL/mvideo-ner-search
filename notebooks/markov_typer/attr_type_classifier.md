# ATTR type classifier — отчёт

> **Устарело относительно нового silver.** Датасет пересобран в [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb)  
> на `WeakLabeler` ([`attr_type_silver.md`](./attr_type_silver.md)). Метрики ниже — прошлый прогон; clf нужно переобучить.

Ноутбук (старый пайплайн): `02_attr_type_silver.ipynb` раньше включал train.  
Лучшая sklearn того прогона: **`sgd_span_ctx_masked`** → `models/attr_type_clf.joblib`
## Дизайн (несколько ATTR в запросе)

Пример: `ноутбук asus 16 гб 15.6 дюйм` → **две** silver-строки:

| span_text | y |
|---|---|
| `16 гб` | memory_storage |
| `15.6 дюйм` | size |

- Char/word n-grams считаются **только по `span_text`** (не по склейке всех ATTR и не по всему query).
- Brand / category / `query_masked` (все ATTR заменены на `<ATTR>`) — **отдельные** фичи контекста.
- `UNKNOWN` — modelish-линейки (`g pro`) и редкие типы; на инференсе modelish режется правилом до clf.
- Аугментация единиц: `гб↔gb↔кб↔kb`, … чтобы не падать на опечатке/синониме.

## Сводка моделей (val)

| model | acc | f1_macro | f1_micro | multi_f1_macro | f1_UNKNOWN |
|---|---:|---:|---:|---:|---:|
| `sgd_span_ctx_masked` | 0.935 | **0.889** | 0.935 | 0.881 | 0.837 |
| `logreg_span_char` | 0.921 | 0.886 | 0.921 | 0.888 | 0.716 |
| `logreg_span_ctx` | 0.921 | 0.877 | 0.921 | 0.879 | 0.744 |
| `logreg_span_wordchar` | 0.915 | 0.877 | 0.915 | 0.873 | 0.721 |
| `markov_lookup` | 0.691 | 0.619 | 0.691 | 0.694 | 0.012 |

`multi_f1_macro` — только строки, где в запросе ≥2 ATTR (~27% val).

## Demo (после unit-aug + modelish-rule)

```
16 гб  -> memory_storage
16 кб  -> memory_storage
16 kb  -> memory_storage
256 gb -> memory_storage   # не MODEL, несмотря на model_phrases
g pro  -> UNKNOWN          # rule до clf (не weight)
2 кг   -> weight
15.6 дюйм -> size
```

## Per-class F1

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| voltage | 1.000 | 1.000 | 1.000 | 13 |
| time | 1.000 | 1.000 | 1.000 | 6 |
| current | 1.000 | 1.000 | 1.000 | 9 |
| frequency | 1.000 | 1.000 | 1.000 | 14 |
| resolution_standard | 1.000 | 1.000 | 1.000 | 10 |
| connectivity | 1.000 | 0.986 | 0.993 | 74 |
| size | 0.986 | 1.000 | 0.993 | 140 |
| weight | 1.000 | 0.983 | 0.992 | 121 |
| volume | 0.965 | 1.000 | 0.982 | 55 |
| power | 0.944 | 0.986 | 0.965 | 69 |
| memory_storage | 0.983 | 0.937 | 0.960 | 430 |
| UNKNOWN | 0.788 | 0.893 | 0.837 | 121 |
| dimensions | 0.568 | 0.636 | 0.600 | 33 |
| resolution_exact | 0.143 | 0.111 | 0.125 | 18 |

## Вывод

- Markov как бейзлайн заметно слабее (macro **0.62** vs **0.89**).
- Изоляция span-фичей + masked context работает на multi-ATTR.
- Дальше: fuzzy единиц шире, gold на ambiguous `x`, не тащить MODEL в ATTR-train.
