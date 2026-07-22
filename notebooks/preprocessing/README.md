# Предобработка запросов (M.Video NER)

Цель папки: **единый, переиспользуемый** слой предобработки до NER / классификаторов / weak labels.

Код для импорта в другие ноутбуки:

```python
from src.preprocessing import QueryPreprocessor, build_model_lexicon_from_titles, save_phrase_list

pp = QueryPreprocessor.from_artifacts("../artifacts")  # или ROOT/"artifacts"
r = pp("Наушники Logitech G-Pro X SE 128гб")
print(r.text_norm, r.model_spans, r.protected_spans)
```

Ноутбук: [`01_data_preprocessing.ipynb`](./01_data_preprocessing.ipynb).

Артефакты:
- `artifacts/model_phrases.txt` — линейки/модели (`g pro x se`, …)
- `artifacts/protected_brands.txt` — бренды с «ложным» цветом (`красный октябрь`)

---

## Гипотезы и предположения (собрано по проекту)

### H1. Короткие запросы, грязный ввод
Пользователи пишут `128гб`, `g-pro`, лишние пробелы/NBSP.  
**Предобработка:** отклеить число от единицы, унифицировать `-`/`_` → пробел, схлопнуть whitespace.

### H2. Регистр несёт смысл (контекст)
«красный» как цвет ≠ «Красный Октябрь» как бренд.  
Слепое `lower()` до всех правил ломает сигнал.  
**Предобработка:** чистить текст с сохранением регистра в `text`; для матчинга держать `text_norm`; ловить Title Case фразы; список `protected_brands`.

### H3. Brand aliases уже работают, MODEL — нет
`iphone→Apple` есть в `labeling.py`. Хвосты `g pro x se` остаются `O`: это не ATTR-regex.  
**Предобработка:** лексикон MODEL (сиды + майнинг из `sku_name`) → spans `MODEL` для пайплайна.

### H4. Расширенные ATTR regex (`temp` → `src`) помогают числу+единице, не линейкам
`16 гб`, `55 дюйм` — да. `g`, `pro`, `x`, `se` — нет.  
**Предобработка не заменяет** словарь моделей / обучаемый NER.

### H5. Клики шумные; бренд часто вне текста (~74%)
Это **не** чинится предобработкой строки. Нужны clf по кликам и/или denoiser разметки (`complex_eda/03`).

### H6. ё/Е, латиница/кириллица «х/x»
Унификация `ё→е`, `×→x` повышает recall словарей.  
Осторожно: не ломать осмысленный токен `x` внутри `g pro x` (не удалять, только нормализовать крестик размера).

---

## Что делает пайплайн (`QueryPreprocessor`)

| Шаг | Действие | Зачем |
|---|---|---|
| 1 | `basic_clean` | NBSP, trim, ё→е |
| 2 | `split_glued_alnum` | `128гб` → `128 гб` |
| 3 | сепараторы `-`/`_` между латиницей → пробел | `g-pro` → `g pro` |
| 4 | tokenize + `text_norm` | стабильный ключ для словарей |
| 5 | Title Case hints | кандидат «Красный Октябрь» |
| 6 | match `protected_brands` | не красить как ATTR-color; пометить BRAND |
| 7 | match `model_phrases` | поймать `g pro x se` как MODEL |
| 8 | `merge_bio_hints` | наложить MODEL/BRAND поверх weak BIO |

---

## Слабые места (не фиксится одной предобработкой)

| Проблема | Почему preprocess мало | Что делать |
|---|---|---|
| Неизвестная линейка вне словаря (`foo bar baz`) | Нет сигнала в правилах | Майнинг + **gold-разметка MODEL** + CRF/NER |
| Омонимы (`pro` = линейка vs «профессиональный») | Нужен контекст бренда/категории | NER с контекстом; clf; не только lexicon |
| Бренд не написан в запросе | В строке нечего нормализовать | Brand classifier по кликам |
| Случайные клики | Не про текст запроса | Разметка relevance 0/1 |
| «Красный» без «Октябрь» | Не отличить цвет от бренда | Контекст / каталог / модель |

### Идеальный путь для `g pro` и хвостов

```text
preprocess (нормализация g-pro → g pro)
  → lexicon MODEL (сиды + mine из sku_name)
  → weak BIO + merge_bio_hints
  → обучение CRF/transformer уже с тегом MODEL
  → закрытие дыр: gold 300–1000 запросов с MODEL spans
```

Качественный датасет нужен там, где словарь конечен, а хвост бесконечен: новые линейки каждый сезон, омонимы, смешанные языки.

---

## Контракт для других ноутбуков

```python
result = QueryPreprocessor.from_artifacts(ARTIFACTS)(query)
# result.text_norm  — строка для словарей / TF-IDF
# result.tokens     — токены
# result.model_spans / protected_spans — подсказки
tags = labeler.label_query(result.text_norm)
tags = pp.merge_bio_hints(tags, result)
```

Не дублируйте `lower()`/`split` локально — зовите `QueryPreprocessor`.
