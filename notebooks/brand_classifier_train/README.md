# Brand classifier train

| Файл | Назначение |
|---|---|
| [`01_classifier_train.ipynb`](./01_classifier_train.ipynb) | обучение + метрики + per-class F1 |
| [`brand_classifier.md`](./brand_classifier.md) | отчёт по прогону |
| `_run_01.py` | smoke/CI без Jupyter |
| `_gen_01.py` | пересборка ноутбука |

Данные: `artifacts/brand_clf/silver_brand_{train,val}.parquet` (из `../preprocessing/03_…`).  
Модель: `models/brand_clf.joblib`.
