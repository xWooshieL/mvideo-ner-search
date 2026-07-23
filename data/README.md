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

## 2. Каноническая раскладка `artifacts/` (с legacy-fallback)

```text
artifacts/
  dicts/                         # канон словарей
    brands.txt
    categories.txt
    model_phrases.txt
    protected_brands.txt
  silver/
    ner_bio/                     # CRF BIO
    attr_type/                   # ATTR-span typing tables
    brand_clf/                   # query→brand tables
  attr_type/                     # runtime: joblib, metrics, policy (не silver)
  brand_clf/                     # runtime: train_runs, policy (+ legacy silver mirror)
  ner/                           # legacy mirror NER silver
  brands.txt …                   # legacy копии словарей (не удаляем)
```

**Читатели:** `resolve_dict()` / `resolve_silver()` в [`src/data_utils.py`](../src/data_utils.py) — сначала canonical, иначе legacy.  
**Запись silver:** `save_silver_parquet(..., mirror=True)` пишет в оба места.  
**Синк:** `python scripts/sync_artifact_layout.py` (копирует legacy→canonical, ничего не удаляет).

| Silver | Канон | Legacy (зеркало) |
|---|---|---|
| NER BIO | `artifacts/silver/ner_bio/` | `artifacts/ner/` |
| ATTR-type | `artifacts/silver/attr_type/` | `artifacts/attr_type/*silver*` |
| Brand clf | `artifacts/silver/brand_clf/` | `artifacts/brand_clf/silver_*` |

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
