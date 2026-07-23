# models/

Здесь лежат обученные модели MVP. В git хранятся **только 3 продакшн-модели** каскада
(остальные `*.joblib` / `*.json` — локальные эксперименты и в репозиторий не попадают,
см. `.gitignore`).

| Файл | Модуль каскада | Что делает | Чем обучалась |
|---|---|---|---|
| `ner_crf.pkl` | CRF NER | размечает токены запроса в BIO (`BRAND/CATEGORY/MODEL/ATTR`) | silver BIO из `artifacts/silver/ner_bio/` |
| `brand_clf.joblib` | Классификатор бренда | достаёт скрытый бренд, когда его нет в тексте (`айфон 15` → Apple) | silver из `artifacts/silver/brand_clf/` |
| `attr_type_clf.joblib` | Типизатор атрибутов | тип ATTR-спана (`16 гб` → memory, `для смузи` → purpose) | silver из `artifacts/silver/attr_type/` |

## Как воспроизвести

Из корня репозитория с активированным `.venv`:

```bash
python scripts/build_dictionaries.py                 # словари brands/categories/model_phrases
python notebooks/brand_classifier_train/_run_01.py   # -> models/brand_clf.joblib
python notebooks/markov_typer/_run_02.py             # silver типизатора
python notebooks/markov_typer/_run_04_prod.py        # -> models/attr_type_clf.joblib
python notebooks/crf_ner_classifier/_run_01.py       # silver BIO
python notebooks/crf_ner_classifier/_run_02.py       # -> models/ner_crf.pkl
```

## Локальные (нетрекаемые) артефакты

`markov_typer.json`, `*__<variant>.joblib`, `category_clf.joblib`, `tfidf_queries.joblib`,
`w2v_tfidf_svd.joblib` — baseline / сравнительные прогоны. Полезны локально, но в git не нужны:
их размер (десятки МБ) раздувал историю.
