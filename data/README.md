# Данные и silver: откуда что берётся

Одна схема «сырьё → словари → silver → модели».

---

## 1. Сырьё в `./data`

| Файл | Роль |
|---|---|
| `query_clicks.parquet` | основной корпус запросов (~31M) |
| `sku_desc.parquet` | EDA / каталог; не основной train NER |
| `skus.pkl` | runtime / демо |
| `gold/bio_*.jsonl` | ручной eval |

Обучение clf/CRF-silver = в основном **query_text**. Карточки косвенно → майнинг словарей (`sku_name` → `model_phrases`).

---

## 2. Каноническая раскладка `artifacts/` (единый источник правды)

```text
artifacts/
  dicts/                         # словари
    brands.txt  categories.txt  model_phrases.txt  protected_brands.txt
  silver/                        # silver-датасеты + метрики/policy по задаче
    ner_bio/                     #   CRF BIO: silver_bio_slice.parquet(*), overview, crf_train_metrics
    attr_type/                   #   ATTR-span typing: silver(*), prod_* метрики, inference_policy
    brand_clf/                   #   query→brand: silver(*), train_runs/, label_map, inference_policy
  gold/                          # эталон: bio_liza.jsonl + gold_stats.json
  metrics/                       # сводные таблицы для презентаций (gold_metrics_*, model_comparison, silver_row_counts)
```

(*) parquet/joblib лежат локально и в `.gitignore`; в git — только превью-CSV, JSON-статистики и policy.

**Читатели:** `resolve_dict()` / `resolve_silver()` в [`src/data_utils.py`](../src/data_utils.py). Раньше была
двойная (legacy) раскладка `artifacts/{attr_type,brand_clf,ner}/` — теперь константы `ATTR_TYPE_DIR` /
`BRAND_CLF_DIR` / `NER_DIR_LEGACY` указывают прямо на `artifacts/silver/<kind>/`, дублирования нет.

| Silver | Путь |
|---|---|
| NER BIO | `artifacts/silver/ner_bio/` |
| ATTR-type | `artifacts/silver/attr_type/` |
| Brand clf | `artifacts/silver/brand_clf/` |

---

## 3. Поток

```text
query_clicks.query_text
  → WeakLabeler(dicts + labeling.py)
       → silver/ner_bio          → CRF
       → ATTR spans + _guess_attr_type → silver/attr_type → ATTR-type clf
  → brand labels                 → silver/brand_clf → brand clf

sku_name (клики/каталог)
  → майнинг → dicts/model_phrases.txt → тег MODEL
```

---

## 4. Карточки дальше

Можно добавить `source=sku_title|description` в silver без ломки query-пайплайна — отдельным полем, не вперемешку без метки.
