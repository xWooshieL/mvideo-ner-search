# notebooks/_legacy/

Архив ранних разведочных ноутбуков. **Не поддерживаются** и не входят в MVP-пайплайн —
оставлены только как история исследования (могут ссылаться на старую раскладку `artifacts/`
и уже отсутствующие пути).

Актуальная работа собрана в:

| Тема | Где сейчас |
|---|---|
| Весь EDA (данные, тег MODEL, типы ATTR, методы) | `notebooks/complex_eda/` (3 ноутбука + README) |
| Предобработка + brand silver | `notebooks/preprocessing/` |
| CRF NER | `notebooks/crf_ner_classifier/` |
| Классификатор бренда | `notebooks/brand_classifier_train/` |
| Типизатор атрибутов | `notebooks/markov_typer/` |

Что архивировано:
- `01_data_overview` … `12_end_to_end_demo` — первичные EDA/бейзлайны (Дни 1–2).
- `06_weak_supervision_ner`, `08_train_ner_model` — ранний NER **без тега MODEL** и без gold-сверки
  (заменены на `crf_ner_classifier/`); `07_train_baseline_classifier` — старый бренд-бейзлайн.
- `markov_typer_01_markov_eda` — разведка марковской типизации (вошла в `complex_eda/03_attr_types_eda`).
- `03_click_eda` — эксперимент с релевантностью кликов (roadmap-денойзер, вне MVP-каскада).
