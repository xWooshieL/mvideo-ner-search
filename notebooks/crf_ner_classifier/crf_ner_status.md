# CRF NER — статус 06/08 vs текущий код / gold

## MVP: что уже собрано

| Слой | Задача | Метод | Артефакт |
|---|---|---|---|
| Brand clf | query → бренд | TF-IDF + logreg/SGD | `models/brand_clf.joblib` |
| ATTR type clf | ATTR-span → type/purpose/size/… | TF-IDF char/word | `models/attr_type_clf.joblib` |
| **CRF NER** | токены → BIO spans | handcrafted feats + CRF | `models/ner_crf.pkl` (устарел: без MODEL) |

Далее для span-NER: привести silver к 4 тегам (+`MODEL`), сохранить артефакт, учить CRF с eval на gold.

## Разбор `06_weak_supervision_ner.ipynb`

| | |
|---|---|
| Вход | `brands.txt`, `categories.txt`, `query_clicks` |
| Учитель | `WeakLabeler.from_files` **без** `models_path` |
| Теги | `BRAND`, `CATEGORY`, `ATTR` — **нет MODEL** |
| Выход | только `figures/20_entity_distribution.png` |
| Gap | silver **не сохраняется** → нет воспроизводимого датасета |

## Разбор `08_train_ner_model.ipynb`

| | |
|---|---|
| Данные | weak on-the-fly (~23k sents), path `файлы/` vs `data/` в 06 |
| Фичи | `sent2features` — **не TF-IDF** |
| Метрики | silver↔silver F1~0.95 — круговые |
| Модель | `models/ner_crf.pkl` |
| Gap | нет MODEL; нет gold-eval; нет saved silver |

## Согласованность

| Источник | Теги |
|---|---|
| `labeling.py` (full) | BRAND, MODEL, CATEGORY, ATTR (+ GENRE/PERSON опц.) |
| 06/08 / текущий `ner_crf.pkl` | BRAND, CATEGORY, ATTR |
| Gold `bio_liza.jsonl` | BRAND, CATEGORY, MODEL, ATTR + `subtypes` для ATTR |

- Gold `subtypes` — **не** теги CRF (это attr-type).
- `artifacts/model_phrases.txt` есть, но 06/08/`train_all` его не подключают.
- Отдельного `artifacts/ner/*silver*` нет.

## Ошибки / риски

1. Без `models_path` хвосты вроде `tuf gaming a15` остаются `O` / путаются.
2. Val F1 завышен (тот же teacher).
3. Токенизация gold: `query.split()` vs `tokenize()` (`_split_glued`) — нужна проверка паритета.
4. Пути данных разъехались (`data/` vs `файлы/`).

## Вывод для EDA (`01`)

Пересобрать срез silver **с** `models_path`, сохранить parquet, сравнить с gold, зафиксировать gaps до обучения в `02`.
