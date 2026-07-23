# CRF NER classifier

Sequence tagging (BIO) для поисковых запросов: `BRAND` / `CATEGORY` / `MODEL` / `ATTR`.

## Статус стека (MVP)

| Компонент | Статус | Где |
|---|---|---|
| Brand clf (query → brand) | MVP есть | `notebooks/brand_classifier_train/` |
| ATTR type clf (ATTR span → type/purpose/…) | MVP есть | `notebooks/markov_typer/03_…` |
| **CRF NER (токены → BIO spans)** | ✅ обучен, gold-eval есть | этот каталог |

Все три модели каскада воспроизводимы (см. блок «Воспроизведение» в корневом README).
Честные метрики CRF на gold — `artifacts/metrics/gold_metrics_ner.csv`.

## Как классифицирует CRF (это не TF-IDF)

CRF (`sklearn_crfsuite`) учит **последовательность тегов**, фичи — **на токен** ([`src/ner/features.py`](../../src/ner/features.py)):

| фича | пример | зачем |
|---|---|---|
| `word.lower` | `asus` | лексика |
| prefix/suffix | `asu`, `sus` | лёгкая устойчивость к обрезаниям |
| `word.shape` | `Xxx` / `dd` | паттерн регистра/цифр |
| digit / latin / cyrillic | флаги | единицы vs бренды |
| ±1 / ±2 соседи | контекст | `B-BRAND` → `I-MODEL` |
| BOS/EOS | края | |

**Перепутанные буквы:** нет fuzzy/edit-distance. Помогают shape + аффиксы + контекст, но `асус`≠`asus` без словаря/лемм в фичах. Опечатки — слабость CRF; brand-clf / dict — отдельные слои.

**TF-IDF** — у brand/attr-type clf, не у CRF.

## Пайплайн

```text
query → WeakLabeler (silver BIO) → CRF → spans
                                      └→ ATTR span → attr_type_clf
query → brand_clf (fallback / отдельный трек)
```

## Файлы

| Файл | |
|---|---|
| [`01_crf_eda.ipynb`](./01_crf_eda.ipynb) | срез silver, quality, gold parity |
| [`02_crf_classifier.ipynb`](./02_crf_classifier.ipynb) | **обучение CRF** + gold eval |
| [`02_crf_report.md`](./02_crf_report.md) | метрики train |
| [`01_crf_eda_report.md`](./01_crf_eda_report.md) | отчёт EDA |

CLI:
- EDA: `python notebooks/crf_ner_classifier/_run_01.py`
- Train: `python notebooks/crf_ner_classifier/_run_02.py`
- Silver check: `python notebooks/crf_ner_classifier/_check_silver.py`

Модель: `models/ner_crf.pkl`  
Silver BIO (канон): `artifacts/silver/ner_bio/` — см. [`data/README.md`](../../data/README.md).
