# General study — совместное обучение CRF + GLiNER, метрика всего пайплайна

В отличие от [`crf_ner_classifier`](../crf_ner_classifier/) и [`gliner`](../gliner/)
(каждый тренируется на своих данных и меряет себя по отдельности), здесь:

1. **Один silver-датасет** — CRF и GLiNER учатся на одном и том же (со SpellFix v2).
2. **gold не в трейне вообще** — чистый held-out для обеих моделей.
3. Метрика — для **каскада целиком** (`rules → +CRF → +CRF+GLiNER`), не для одной модели.
4. Отдельный **`broken_queries_eval`** — синтетические опечатки (held out, не в трейне),
   стресс-тест устойчивости пайплайна к шуму (см. `artifacts/history/broken_queries.md`).

## Файлы

| Файл | |
|---|---|
| [`01_general_silver.ipynb`](./01_general_silver.ipynb) / [`_run_01.py`](./_run_01.py) | joint silver + синтетические broken queries |
| [`02_general_train_eval.ipynb`](./02_general_train_eval.ipynb) / [`_run_02.py`](./_run_02.py) | обучение CRF+GLiNER, метрика всего каскада |
| [`01_general_silver_report.md`](./01_general_silver_report.md) | отчёт по silver + примеры spellfix/порчи |
| [`02_general_train_eval_report.md`](./02_general_train_eval_report.md) | отчёт: F1 по стадиям x eval-сетам |

CLI:
```bash
python notebooks/general_study/_run_01.py   # -> artifacts/silver/general_study/
python notebooks/general_study/_run_02.py   # -> models/general_study/, ~15-20 мин на CPU
```

## Результат последнего прогона (span-level micro F1)

| eval set | n | rules | +CRF | +CRF+GLiNER |
|---|---:|---:|---:|---:|
| gold (чистый held-out) | 181 | 0.604 | 0.609 | **0.640** |
| broken_queries (синтетические опечатки) | 270 | 0.578 | **0.846** | 0.795 |
| silver_val (само-согласованность, teacher) | 799 | 1.000 | 0.987 | 0.931 |

**Главный вывод: GLiNER не всегда помогает поверх CRF.** На gold — небольшой плюс
(+0.031), но на `broken_queries` и `silver_val` — **просадка** (0.846→0.795, 0.987→0.931):
recall у GLiNER растёт, а precision падает сильнее — на зашумлённом/собственном
teacher-тексте он добавляет ложные срабатывания быстрее, чем закрывает реальные дыры CRF.

CRF, наоборот, даёт самый большой и устойчивый прирост — особенно на `broken_queries`
(+0.268 над rules) — контекст+shape-фичи гораздо надёжнее словарного точного совпадения
при опечатках.

Полные P/R/F1 по стадиям — в [`02_general_train_eval_report.md`](./02_general_train_eval_report.md).

## SpellFix v2 (новое: гомоглифы + алиасы транслитерации)

`01` использует обновлённый `src/preprocessing/spellfix.py`:

- **Гомоглифы**: `с`/`c`, `а`/`a`, `о`/`o`, `р`/`p`, `е`/`e`, `х`/`x`, `у`/`y` (и их пары в
  верхнем регистре) — детерминированная нормализация смешанного алфавита внутри слова
  (`аsus` → `asus`, `cамсунг` → `самсунг`), без словаря.
- **Алиасы** (`artifacts/dicts/spell_aliases.txt`): транслитерация "на слух", где буквы
  кириллицы/латиницы не похожи (`сони` → `sony`, `плейстейшен` → `playstation`,
  `ксяоми` → `xiaomi`) — обычный Левенштейн это не поймает, whitelist — надёжно.

Известное ограничение: сильно искажённые/усечённые слова (`моильник` вместо `холодильник`,
пропущено 3 буквы подряд) — вне охвата character-level fuzzy-подхода; нужна другая техника
(фонетическая/subsequence), не форсируем через ослабление порогов (риск ложных срабатываний
на коротких словах).

## Не трогает

- `notebooks/crf_ner_classifier/*`, `notebooks/gliner/*` — отдельные треки, свои данные/модели.
- `models/ner_crf.pkl`, `models/gliner_ner/` — веса других треков; свои — в `models/general_study/`.
- `src/service/extractor.py` — каскад здесь самостоятельная копия для эксперимента,
  в прод не подключена (см. `_run_02.py::cascade_entities`).
