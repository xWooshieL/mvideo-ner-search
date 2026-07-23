# Brand classifier train

| Файл | Назначение |
|---|---|
| [`01_classifier_train.ipynb`](./01_classifier_train.ipynb) | обучение + метрики + per-class F1 |
| [`brand_classifier.md`](./brand_classifier.md) | отчёт по прогону |
| `_run_01.py` | smoke/CI без Jupyter |
| `_gen_01.py` | пересборка ноутбука |

Данные (канон): `artifacts/silver/brand_clf/silver_brand_{train,val}.parquet` (из `../preprocessing/03_…` / `_run_03.py`).  
Модель: `models/brand_clf.joblib` (прод — `sgd_char`, silver-val macro-F1 ≈ 0.94).  
Метрики: `artifacts/silver/brand_clf/train_runs/models_summary.csv`, сравнение — `artifacts/metrics/model_comparison.csv`.

> Brand-level gold пока нет — приводятся только silver-val числа (бренд из majority кликов).
