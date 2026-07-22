# Brand classifier — отчёт обучения

Ноутбук: [`01_classifier_train.ipynb`](./01_classifier_train.ipynb)  
Silver: `artifacts/brand_clf/silver_brand_*.parquet` (прогон `03`)  
Лучшая модель: **`logreg_wordchar`** → `models/brand_clf.joblib`

## 1. Постановка

Мультикласс query→brand на silver с классами top-K + `NO_BRAND` + `UNKNOWN` (OOD).
Clf — fallback после NER/alias; на category-only не должен навязывать Indesit.

См. также [`../preprocessing/silver_clf_readme.md`](../preprocessing/silver_clf_readme.md).

## 2. Данные

- train: **22846**, val: **5710**, classes: **67**
- NO_BRAND share (silver stats): 0.20513536142897013
- UNKNOWN share: 0.10713787329053866

### Inference thresholds

```json
{
  "TAU_ACCEPT": 0.42,
  "TAU_MARGIN": 0.08,
  "TAU_NO_BRAND": 0.35,
  "TAU_UNKNOWN": 0.3,
  "REQUIRE_BRAND_EVIDENCE": true
}
```

## 3. Сравнение моделей

| model | acc | f1_macro | f1_micro | f1_weighted | f1_NO_BRAND | f1_UNKNOWN | false_brand@cat |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sgd_char` | 0.944 | 0.951 | 0.944 | 0.943 | 0.875 | 0.856 | 0.094 |
| `logreg_wordchar` | 0.944 | 0.950 | 0.944 | 0.943 | 0.877 | 0.853 | 0.069 |
| `logreg_char` | 0.928 | 0.910 | 0.928 | 0.927 | 0.849 | 0.829 | 0.079 |
| `logreg_char_bal` | 0.909 | 0.874 | 0.909 | 0.907 | 0.782 | 0.788 | 0.197 |

### Лучшая (`logreg_wordchar`) raw

- accuracy **0.944**
- F1-macro **0.950**, micro **0.944**, weighted **0.943**
- F1(`NO_BRAND`) **0.877**, F1(`UNKNOWN`) **0.853**
- false brand rate на category-only val: **0.06887755102040816** (n=1176)

## 4. Per-class F1 (best model)

### Топ по F1

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| BQ | 1.000 | 1.000 | 1.000 | 6 |
| Deppa | 1.000 | 1.000 | 1.000 | 12 |
| Dyson | 1.000 | 1.000 | 1.000 | 30 |
| Grundig | 1.000 | 1.000 | 1.000 | 7 |
| HUAWEI | 1.000 | 1.000 | 1.000 | 255 |
| Infinix | 1.000 | 1.000 | 1.000 | 23 |
| MSI | 1.000 | 1.000 | 1.000 | 32 |
| Pantum | 1.000 | 1.000 | 1.000 | 10 |
| Oasis | 1.000 | 1.000 | 1.000 | 4 |
| Scarlett | 1.000 | 1.000 | 1.000 | 4 |
| Polaris | 1.000 | 1.000 | 1.000 | 27 |
| Rowenta | 1.000 | 1.000 | 1.000 | 15 |
| Tecno | 1.000 | 1.000 | 1.000 | 35 |
| Ariston | 1.000 | 1.000 | 1.000 | 7 |
| ATLANT | 1.000 | 1.000 | 1.000 | 10 |

### Хвост (худшие F1)

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| PlayStation | 1.000 | 0.250 | 0.400 | 4 |
| Electrolux | 0.850 | 0.850 | 0.850 | 20 |
| UNKNOWN | 0.932 | 0.787 | 0.853 | 614 |
| NO_BRAND | 0.848 | 0.908 | 0.877 | 1176 |
| Krona | 0.917 | 0.846 | 0.880 | 13 |
| Rapid | 1.000 | 0.800 | 0.889 | 5 |
| Candy | 0.857 | 0.938 | 0.896 | 32 |
| Gefest | 0.880 | 0.917 | 0.898 | 24 |
| Beko | 0.872 | 0.944 | 0.907 | 36 |
| TalleR | 1.000 | 0.833 | 0.909 | 6 |
| Moulinex | 1.000 | 0.833 | 0.909 | 12 |
| Centek | 1.000 | 0.833 | 0.909 | 12 |
| Midea | 0.902 | 0.920 | 0.911 | 50 |
| Hi | 1.000 | 0.857 | 0.923 | 21 |
| Atvel | 1.000 | 0.857 | 0.923 | 7 |

Полные CSV: `artifacts/brand_clf/train_runs/per_class_f1__*.csv`.

## 5. Reject-policy

После Softmax применяем τ из `inference_policy.json` (reject → трактуем как `NO_BRAND`/null).

| model | raw f1_macro | reject f1_macro | raw acc | reject acc |
|---|---:|---:|---:|---:|
| `sgd_char` | 0.951 | 0.949 | 0.944 | 0.944 |
| `logreg_wordchar` | 0.950 | 0.929 | 0.944 | 0.935 |
| `logreg_char` | 0.910 | 0.865 | 0.928 | 0.914 |
| `logreg_char_bal` | 0.874 | 0.880 | 0.909 | 0.856 |

## 6. Интерпретация

- **macro-F1** важнее accuracy: классы несбалансированы (`NO_BRAND` / Samsung / хвост).
- Высокий **F1(NO_BRAND)** + низкий **false_brand@cat** — модель не тащит Indesit на `холодильник`.
- **UNKNOWN** обычно сложнее: OOD-бренды орфографически разнообразны; F1 ниже — ожидаемо.
- Классы с F1≈0 и малым support — кандидаты в `UNKNOWN` или на ↑ семпла в `03`.

## 7. Нужно ли что-то перезапускать?

| Что | Когда |
|---|---|
| `03_brand_data_preprocessing.ipynb` | сменили пороги silver / top-K / NO_BRAND logic |
| этот ноутбук / `_run_01.py` | после обновления silver parquet |
| gold-разметка | для честного test и калибровки τ (пока нет) |

Перезапускать preprocess NER (`01`/`02`) **не нужно** — brand-clf живёт на `query_norm` + кликах.

## 8. Артефакты

| Путь |
|---|
| `models/brand_clf.joblib` |
| `artifacts/brand_clf/train_runs/metrics.json` |
| `artifacts/brand_clf/train_runs/models_summary.csv` |
| `figures/brand_clf/01_confusion_best.png` |
| `figures/brand_clf/02_models_compare.png` |

---
*Сгенерировано `_run_01.py`, best=`logreg_wordchar`.*

## 9. Рекомендация к деплою

По **macro-F1** формально лидирует `sgd_char` (~0.951), но разница с `logreg_wordchar` ~0.001, а **false brand rate на category-only** у word+char ниже (**0.069 vs 0.094**).

В `models/brand_clf.joblib` сохранён **`logreg_wordchar`** (word 1–2 + char_wb 2–4 + LogReg + `sample_weight`).

Калибровку τ и честный test — на mini-gold (пока нет). Перезапускать `03` не нужно, пока не меняете silver-правила. Перезапустить этот ноутбук / `_run_01.py` — если обновили parquet после `03`.
