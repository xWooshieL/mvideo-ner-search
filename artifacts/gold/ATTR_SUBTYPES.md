# ATTR subtypes: app (gold) ↔ teacher canon

Контракт для eval и silver/clf. **Gold jsonl / UI приложения не меняем** —
редкие сабтипы app сохраняются; на метриках сводим в канон.

## Canon teacher (`_guess_attr_type` / clf classes)

- Unit-типы из `ATTR_PATTERNS` в `src/ner/labeling.py`
- `color` (COLORS)
- **`type`**, **`purpose`** (новые lexical ATTR)
- `other` (нет матча; на train часто → `UNKNOWN`)

Приоритет угадывания: `color` → units → `purpose` → `type` → `other`.

## gold → canon (eval map)

| gold (app) | canon |
|---|---|
| `type`, `gas`, `floor`, `style`, `function`, `feature` | `type` |
| `purpose`, `food` | `purpose` |
| `depth`, `width`, `heigh`, `length` | `size` |
| `counts` | `quantity` |
| `material`, `chip`, `sim`, `game`, `used`, `new`, `country`, `release date`, `delivery`, `speed` | `other` |
| unit-типы + `color` | as-is |

Код маппинга: `GOLD_TO_CANON` / `gold_subtype_to_canon()` в `src/ner/labeling.py`.

## App guidance (будущие разметки, опционально)

Не требует переразметки существующих файлов:

- `memory_storage` — только память/накопитель
- диагональ ТВ → `size`; вес → `weight`
- «беспроводные / узкая / смарт» → `type`
- «для спорта / смузи» → `purpose`
