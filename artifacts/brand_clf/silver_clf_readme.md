# Silver brand-clf: договорённости и подводные камни

Документ к ноутбуку [`03_brand_data_preprocessing.ipynb`](./03_brand_data_preprocessing.ipynb)  
и артефактам `artifacts/brand_clf/`.

---

## 1. Зачем этот датасет

Brand-clf — **fallback**, когда NER/словари не вытащили бренд из текста  
(типичный кейс: бренд не написан, но подразумевается — `айфон` → Apple).

Он **не** должен отвечать на каждый запрос.  
Иначе на `холодильник` модель ткнет в частый клик-бренд (Indesit/Haier) — false positive.

Поэтому в silver есть не только бренды top-K, но и спец-классы:

| Лейбл | Смысл на обучении | На инференсе |
|---|---|---|
| `Apple`, `Samsung`, … | конкретный бренд из closed-set (top-K) | можно принять при высокой уверенности |
| `NO_BRAND` | бренда нет и не подразумевается | → `brand = null` |
| `UNKNOWN` | бренд вне top-K / спорный / OOD | → `brand = null` (+ опц. retrieval) |

---

## 2. Что такое OOD (не «OOB»)

В колонке `label_reason` пишется **`ood_brand_surface`**.

**OOD = Out-Of-Distribution** — «вне распределения классов модели».

Это **не** опечатка и не «out of bag».

### Пример из silver

| query_norm | click_brand | brand (y) | reason |
|---|---|---|---|
| `стинол холодильник` | Стинол | `UNKNOWN` | `ood_brand_surface` |
| `darina холодильник` | Darina | `UNKNOWN` | `ood_brand_surface` |
| `hotpoint ariston холодильник` | Hotpoint | `UNKNOWN` | `ood_brand_surface` |

Бренд **есть в запросе** и совпадает с кликом — разметка «правильная» с точки зрения мира.  
Но **Стинол / Darina / Hotpoint не входят в top-K** обучающих классов clf → в silver их кладут как `UNKNOWN`.

Иначе пришлось бы либо:
- раздувать число классов до тысяч (хвост брендов), либо  
- на инференсе всё равно схлопывать неизвестных в «ближайший» Samsung/Indesit.

`UNKNOWN` учит модель: «вижу бренд, но это не мой closed-set → откажись», а не «угадай похожий».

### Другие `label_reason`

| reason | Когда | Пример |
|---|---|---|
| `category_only` | запрос ≈ категория, без brand/alias | `холодильник` → `NO_BRAND` |
| `brand_in_query` | имя бренда (top-K) в тексте + conf ок | `samsung galaxy` → Samsung |
| `alias_hint` | алиас (`айфон`, `haier`, …) | `айфон 15` → Apple |
| `ood_brand_surface` | бренд в тексте, но **не** top-K | `стинол холодильник` → UNKNOWN |
| `ambiguous_clicks` | нет brand/alias, много разных кликов | `холодильники двухкамерные 55 см` → UNKNOWN |
| `low_conf_with_evidence` | есть surface, но слабый majority | → UNKNOWN |
| `no_evidence_drop` | нет brand/alias и не category-only | **не попадает** в silver |

---

## 3. Confidence — откуда берётся

Да, **confidence из формулы majority по кликам**, не из модели clf.

На один `query_norm` собираем все клики с брендами. Вес клика:

\[
w_i = \frac{1}{1 + \mathrm{sku\_position}_i}
\]

(клик выше в выдаче весит больше).

Пусть \(W_b = \sum_i w_i\) по кликам бренда \(b\), \(W = \sum_b W_b\).  
Majority-бренд: \(b^\star = \arg\max_b W_b\).

\[
\mathrm{confidence} = \frac{W_{b^\star}}{W}
\]

Интерпретация: доля «веса» у победившего бренда.

| confidence | Типичная ситуация |
|---|---|
| ≈ 1.0 | все клики в один бренд (часто 1 клик) |
| 0.6–0.9 | есть runner-up, но majority явный |
| &lt; 0.55 (`MIN_CONFIDENCE`) | спорно → чаще `UNKNOWN` / дроп |

`sample_weight` в parquet ≈ этот confidence (для `NO_BRAND`/`UNKNOWN` фиксированные веса).

**Важно:** высокий confidence при *одном* клике — слабый сигнал. Поэтому отдельно смотрим `n_clicks`, `n_brands`, `brand_in_query`.

---

## 4. Как собирается `y` (кратко)

```text
если category-only          → NO_BRAND
иначе если alias ∈ top-K    → канон алиаса (Apple, …)
иначе если бренд в тексте ∈ top-K и conf≥τ → click_brand
иначе если бренд в тексте ∉ top-K и conf≥τ → UNKNOWN   ← твой Стинол
иначе если нет evidence     → DROP (не в датасет)
иначе спорное               → UNKNOWN / DROP
```

Категории берутся из **`artifacts/categories.txt` целиком** (не ручной короткий EXTRA-список).

Алиасы — из `BRAND_ALIASES` в `src/ner/labeling.py`.

---

## 5. Когда звать clf (inference policy)

Файл: `artifacts/brand_clf/inference_policy.json`.

### Cascade

1. NER/dicts/alias уже дали BRAND → **clf не звать**  
2. category-only → `brand = null`, **clf не звать**  
3. есть evidence (alias / model-ish) → **звать clf**  
4. иначе → `brand = null`

### Пороги принятия ответа clf

| Порог | Дефолт | Смысл |
|---|---|---|
| `TAU_ACCEPT` | 0.42 | min Softmax top-1 для конкретного бренда |
| `TAU_MARGIN` | 0.08 | min (top1 − top2) |
| `TAU_NO_BRAND` | 0.35 | принять `NO_BRAND` → null |
| `TAU_UNKNOWN` | 0.30 | принять `UNKNOWN` → null + OOD |

Если пороги не пройдены → **reject**, `brand = null` (лучше пусто, чем ложный Indesit).

---

## 6. Проблемные места датасета (осознанные)

### 6.1. Click ≠ intent
Клики шумят: сравнение товаров, случайный клик, «посмотрел не то».  
Majority снижает шум, но не убирает.

### 6.2. Popular-brand prior
Без `NO_BRAND` и дропа `no_evidence` модель учит `P(бренд | категория)`.  
Именно это ломает `холодильник → Indesit`.

### 6.3. Closed-set top-K → много `UNKNOWN`
Хвост брендов (Стинол, Darina, Kraft, Weissgauff, …) при **явном** написании в запросе всё равно `UNKNOWN`.  
Это **не баг разметки**, а следствие closed-set clf.  
Если нужно предсказывать и их — поднимать `TOP_K_BRANDS` / отдельный retrieval по surface-матчу (словарь брендов), не softmax на 80 классов.

### 6.4. `ambiguous_clicks` на «почти категориях»
`холодильники двухкамерные 55 см` — не category-only (есть число/`см`), клики разъехались → `UNKNOWN`.  
Ок для reject; на gold лучше разметить как `NO_BRAND`.

### 6.5. Alias vs click конфликт
`айфон` + клик не-Apple: сейчас чаще доверяем алиасу (`alias_hint`).  
Клики могут врать; алиас в тексте обычно сильнее.

### 6.6. Category lexicon шум
В `categories.txt` есть мусорные/широкие фразы (`smart`, `блок`, `игровой`).  
Эвристика `category_only` требует покрытие токенов ≥ 0.6 и отсутствие model-ish — но ложные category-only / пропуски возможны. Чистить словарь категорий отдельно.

### 6.7. Один клик → confidence = 1
Формально «уверенно», по сути мало данных. Смотреть `n_clicks` и tier.

### 6.8. Preprocess без MODEL-lexicon
Для скорости: `basic_clean` + `_norm_key`, без матчинга `model_phrases`.  
На brand-clf этого достаточно; для NER MODEL — другой пайплайн.

---

## 7. Файлы

| Путь | Содержимое |
|---|---|
| `artifacts/brand_clf/silver_brand_{train,val,all}.parquet` | silver |
| `artifacts/brand_clf/label_map.json` | id↔label, `special` |
| `artifacts/brand_clf/inference_policy.json` | cascade + τ |
| `artifacts/brand_clf/silver_brand_stats.json` | сводка прогона |
| `figures/preprocessing/brand_clf/` | EDA-картинки |

Колонки parquet (главные):  
`query_norm`, `brand` (y), `click_brand`, `label_reason`, `confidence`, `sample_weight`,  
`brand_in_query`, `has_alias`, `is_category_only`, `tier`, …

---

## 8. Метрики, которые имеют смысл

Не только accuracy на val:

1. **F1 на brand-present** (реальные бренды top-K)  
2. **False brand rate** на category-only / `NO_BRAND` (цель → 0)  
3. **Reject rate** и калибровка τ  
4. Отдельно качество на alias-кейсах (`айфон`, `редми`, …)

Gold позже: 300–500 запросов stratified по `label_reason` × tier.
