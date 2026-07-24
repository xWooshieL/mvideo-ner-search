# GLiNER (хвост каскада)

Zero-shot / fine-tuned NER поверх правил + CRF.

```text
spellfix → rules/dicts → CRF → GLiNER (дыры rules+CRF, новые типы)
```

Не заменяет CRF на всём трафике: GLiNER (~0.3B, CPU ~150-200 ms/запрос) — **хвост**,
подключается там, где rules+CRF не нашли сущность или встретился незнакомый паттерн.

## Библиотека vs веса (частый вопрос)

| | Что это | Пример |
|---|---|---|
| **библиотека `gliner`** | код: `GLiNER.from_pretrained`, `predict_entities`, `train_model` | `pip install gliner` |
| **чекпоинт на HF** | обученные веса конкретной модели | [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) |

```python
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
```

Без библиотеки веса не запустить; без чекпоинта библиотека пустая. `multi-v2.1` — мультиязычная
(apache-2.0), поэтому она понимает и русские, и латинские запросы без доп. перевода.

## Файлы

| Файл | |
|---|---|
| [`01_gliner_zero_shot.ipynb`](./01_gliner_zero_shot.ipynb) / [`_run_01.py`](./_run_01.py) | baseline без обучения, выбор промпт-схемы (en/ru) |
| [`02_gliner_finetune.ipynb`](./02_gliner_finetune.ipynb) / [`_run_02.py`](./_run_02.py) | fine-tune на gold + silver, авто-раунд 2, калибровка порога |
| [`01_gliner_zero_shot_report.md`](./01_gliner_zero_shot_report.md) | отчёт baseline |
| [`02_gliner_finetune_report.md`](./02_gliner_finetune_report.md) | отчёт fine-tune (до/после, edge cases) |
| [`src/ner/gliner_utils.py`](../../src/ner/gliner_utils.py) | конвертация BIO↔GLiNER, span P/R/F1, edge cases, eval-хелперы |

CLI:
```bash
python notebooks/gliner/_run_01.py   # zero-shot baseline -> artifacts/metrics/gliner_zero_shot.json
python notebooks/gliner/_run_02.py   # fine-tune -> models/gliner_ner/ (~10-15 мин на CPU)
```

## Результат (`bio_liza.jsonl`, span-level exact match, val-сплит)

| | micro P | micro R | micro F1 |
|---|---:|---:|---:|
| zero-shot (без обучения) | 0.70 | 0.28 | 0.39 |
| **fine-tuned** (gold 154 + silver 200, 3 эпохи) | 0.76 | 0.73 | **0.74** |

По лейблам сильнее всего подрос recall на `CATEGORY` и `ATTR` — zero-shot их почти не находил
(F1≈0), после обучения `CATEGORY` F1≈0.88, `ATTR` F1≈0.36 (самый скудный/шумный класс в gold).

Полные таблицы, edge cases (опечатки, кириллица вместо латиницы, чужой алфавит, 1-словные
запросы) — в [отчёте `02`](./02_gliner_finetune_report.md).

## Как обучение "само себя чинит"

`_run_02.py` не просто обучает один раз — двухступенчатая самопроверка:

1. **Мало прироста над zero-shot** (`< F1_MIN_GAIN`) → автоматически запускает второй раунд
   (больше эпох, ниже lr) и берёт лучший по val F1.
2. **Калибровка порога** после обучения: default `threshold=0.5` часто резал верные, но
   неуверенные спаны (особенно `ATTR`, score ~0.33-0.45) — это гиперпараметр **инференса**,
   не требует ретрейна. Скрипт перебирает сетку порогов на val и сам выбирает лучший
   (при близких F1 — более низкий порог, т.к. GLiNER здесь хвост и recall важнее чистой precision).

## Данные

- **Gold**: `data/gold/bio_liza.jsonl` (~180 размеченных строк после ретокенизации) — train/val 85/15.
- **Silver-добавка**: `artifacts/silver/ner_bio/silver_bio_slice.parquet` — тот же silver, что и у
  CRF (`notebooks/crf_ner_classifier`), уже пропущенный через `SpellFixer`. 200 примеров сверху
  скудного gold, чтобы модель не переобучилась на 150 строк.
- Gold для **eval** не трогаем — теги уже на исходных токенах, sanity-сравнение zero-shot vs
  fine-tuned должно быть честным "до/после" на одном и том же val.

## Модель

Fine-tuned веса — `models/gliner_ner/` (не в git, `.gitignore` — большая, safetensors/bin).
Загрузка:
```python
from gliner import GLiNER
model = GLiNER.from_pretrained("models/gliner_ner")
```

Пересобрать: `python notebooks/gliner/_run_02.py` (перезатрёт `models/gliner_ner/`).

## Дальше

1. Больше gold (`data/gold/bio_liza.jsonl` растёт) → перезапустить `_run_02.py`, F1 должен подрасти.
2. Подключить `models/gliner_ner` в `src/service/extractor.py` как явный **хвост** каскада
   (вызывать только когда rules+CRF не нашли сущность в спане) — сейчас GLiNER самостоятельный
   трек, в проде ещё не подключён.
3. `ATTR` — самый шумный gold-класс (смесь regex-атрибутов из silver и произвольных
   прилагательных/фраз у аннотаторов) — качественная разметка ATTR даст больше, чем ещё эпохи.
