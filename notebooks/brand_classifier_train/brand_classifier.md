# Brand classifier — отчёт обучения

Ноутбук: [`01_classifier_train.ipynb`](./01_classifier_train.ipynb)  
Silver: `artifacts/silver/brand_clf/silver_brand_*.parquet` (прогон `03`)  
Лучшая модель: **`sgd_char`** → `models/brand_clf.joblib`

## 1. Постановка

Мультикласс query→brand на silver с классами top-K + `NO_BRAND` + `UNKNOWN` (OOD).
Clf — fallback после NER/alias; на category-only не должен навязывать Indesit.

См. также [`../preprocessing/silver_clf_readme.md`](../preprocessing/silver_clf_readme.md).

## 2. Данные

- train: **22879**, val: **5717**, classes: **67**
- NO_BRAND share (silver stats): 0.2048494983277592
- UNKNOWN share: 0.10712792642140469

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
| `sgd_char` | 0.945 | 0.940 | 0.945 | 0.945 | 0.879 | 0.880 | 0.096 |
| `logreg_char_bal` | 0.927 | 0.936 | 0.927 | 0.926 | 0.823 | 0.827 | 0.163 |
| `logreg_wordchar` | 0.944 | 0.929 | 0.944 | 0.943 | 0.879 | 0.873 | 0.078 |
| `logreg_char` | 0.933 | 0.912 | 0.933 | 0.933 | 0.863 | 0.852 | 0.082 |

### Лучшая (`sgd_char`) raw

- accuracy **0.945**
- F1-macro **0.940**, micro **0.945**, weighted **0.945**
- F1(`NO_BRAND`) **0.879**, F1(`UNKNOWN`) **0.880**
- false brand rate на category-only val: **0.09608843537414966** (n=1176)

## 4. Per-class F1 (best model)

### Топ по F1

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| Infinix | 1.000 | 1.000 | 1.000 | 23 |
| Thunderobot | 1.000 | 1.000 | 1.000 | 14 |
| Canon | 1.000 | 1.000 | 1.000 | 26 |
| Tecno | 1.000 | 1.000 | 1.000 | 35 |
| Samsung | 1.000 | 1.000 | 1.000 | 647 |
| BQ | 1.000 | 1.000 | 1.000 | 6 |
| Atvel | 1.000 | 1.000 | 1.000 | 7 |
| HUAWEI | 1.000 | 1.000 | 1.000 | 255 |
| HONOR | 1.000 | 1.000 | 1.000 | 141 |
| Ariston | 1.000 | 1.000 | 1.000 | 7 |
| DeLonghi | 1.000 | 1.000 | 1.000 | 31 |
| Scarlett | 1.000 | 1.000 | 1.000 | 4 |
| Moulinex | 1.000 | 1.000 | 1.000 | 12 |
| Brayer | 1.000 | 1.000 | 1.000 | 11 |
| Haier | 1.000 | 0.995 | 0.998 | 204 |

### Хвост (худшие F1)

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| PlayStation | 1.000 | 0.500 | 0.667 | 4 |
| Grundig | 0.833 | 0.714 | 0.769 | 7 |
| Electrolux | 0.750 | 0.900 | 0.818 | 20 |
| Rapid | 0.714 | 1.000 | 0.833 | 5 |
| TalleR | 0.833 | 0.833 | 0.833 | 6 |
| Oasis | 1.000 | 0.750 | 0.857 | 4 |
| Candy | 0.871 | 0.844 | 0.857 | 32 |
| Centek | 1.000 | 0.750 | 0.857 | 12 |
| NO_BRAND | 0.869 | 0.889 | 0.879 | 1176 |
| UNKNOWN | 0.941 | 0.826 | 0.880 | 615 |
| Ballu | 0.917 | 0.846 | 0.880 | 13 |
| Sber | 0.929 | 0.867 | 0.897 | 15 |
| Kuppersberg | 0.814 | 1.000 | 0.897 | 35 |
| Beko | 0.892 | 0.917 | 0.904 | 36 |
| Logitech | 0.861 | 0.969 | 0.912 | 32 |

Полные CSV: `artifacts/silver/brand_clf/train_runs/per_class_f1__*.csv`.

## 5. Reject-policy

После Softmax применяем τ из `inference_policy.json` (reject → трактуем как `NO_BRAND`/null).

| model | raw f1_macro | reject f1_macro | raw acc | reject acc |
|---|---:|---:|---:|---:|
| `sgd_char` | 0.940 | 0.941 | 0.945 | 0.947 |
| `logreg_char_bal` | 0.936 | 0.928 | 0.927 | 0.865 |
| `logreg_wordchar` | 0.929 | 0.911 | 0.944 | 0.938 |
| `logreg_char` | 0.912 | 0.852 | 0.933 | 0.920 |

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
| `artifacts/silver/brand_clf/train_runs/metrics.json` |
| `artifacts/silver/brand_clf/train_runs/models_summary.csv` |
| `figures/brand_clf/01_confusion_best.png` |
| `figures/brand_clf/02_models_compare.png` |

---
*Сгенерировано `_run_01.py`, best=`sgd_char`.*
