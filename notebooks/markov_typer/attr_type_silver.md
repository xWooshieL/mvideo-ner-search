# ATTR type silver — схема и договорённости

Документ к ноутбуку [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb)  
и артефактам `artifacts/silver/attr_type/`.

---

## 1. Зачем этот датасет

NER помечает span как общий **`ATTR`**. Тип (`memory_storage`, `size`, `color`, …) нужен
второй стадией (clf / postprocess).

Silver учит типизатор на weak labels из того же пайплайна, что и BIO-разметка:

```text
query → WeakLabeler.label_query → BIO
      → bio_to_entities → ATTR spans
      → _guess_attr_type(ATTR_PATTERNS + COLORS) → y
```

**Не** используем локальные `find_brand` / самодельные model-regex в ноутбуке — словари
уже в `WeakLabeler` (`brands.txt`, `categories.txt`, `model_phrases.txt`).

---

## 2. Одна строка = один ATTR-span

Пример: `ноутбук asus 16 гб 15.6 дюйм` → **две** строки:

| span_text | y | brand | category |
|---|---|---|---|
| `16 гб` | memory_storage | asus / ASUS | ноутбук… |
| `15.6 дюйм` | size | … | … |

Так каждый найденный атрибут типизируется **отдельно**.

---

## 3. Колонки

| колонка | смысл |
|---|---|
| `query`, `query_norm` | исходник / текст для labeler |
| `span_text`, `span_start`, `span_end` | текст и char-span ATTR |
| `y` | тип: имена групп `ATTR_PATTERNS` + `color` + `other` |
| `brand`, `category`, `model` | из `entities_to_structured` |
| `n_attrs_in_query` | сколько ATTR в этом запросе |
| `bio_tags` | BIO для аудита |
| `query_masked_all_attr` | все ATTR → `<ATTR>` |
| `query_keep_span_mask_others` | текущий span текстом, остальные ATTR → `<ATTR>` |
| `query_masked_entities` | BRAND/CAT/MODEL/ATTR → плейсхолдеры |
| `is_aug` | unit-aug (`гб↔gb↔кб↔kb`, …) |

Файлы:

- `attr_type_silver.parquet` — raw + aug  
- `attr_type_silver_raw.parquet` — только `is_aug=False`  
- `attr_type_silver_meta.json` — N, классы, описание масок  

---

## 4. Что маскируем (и зачем)

Для будущего clf:

- **Char/word n-grams только по `span_text`** (не склейка всех ATTR, не весь query).
- Чужие ATTR в контексте **исключаем**:
  - `query_keep_span_mask_others` — виден текущий span + окружение без чужих единиц;
  - `query_masked_all_attr` — полный контекст без единиц измерения;
  - `query_masked_entities` — ещё и brand/cat/model как плейсхолдеры.

Нельзя кормить TF-IDF строкой вроде `"16 гб 15.6 дюйм"`: лексика типов смешается.

---

## 5. Откуда `y`

`_guess_attr_type` в [`src/ner/labeling.py`](../../src/ner/labeling.py):

1. `COLORS` (+ лемма) → `color`
2. первый матч по `ATTR_PATTERNS` → имя группы (`memory_storage`, …)
3. иначе → `other`

`other` **оставляем** в датасете — дыры regex / кандидаты в gold.  
Редкие классы **не** схлопываем в `UNKNOWN` здесь — это политика train-clf.

`MODEL` не должен попадать в ATTR, если сработал `model_phrases.txt`.
Иногда наоборот: glued `16gb` попадает в MODEL — это качество словаря MODEL, чинить там
(см. EDA в ноутбуке), а не локальным `looks_like_model` в silver.

---

## 6. Unit-aug

Доп. строки с тем же `y` и тем же masked-контекстом (`is_aug=True`), чтобы синонимы
единиц (`16 кб`, `256 gb`) не были OOV. В EDA смотрим raw vs aug отдельно.

---

## 7. Связь с clf

Обучение: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb) → отчёт [`attr_type_classifier.md`](./attr_type_classifier.md).  
Модель: `models/attr_type_clf.joblib`.
