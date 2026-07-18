#!/usr/bin/env python
"""Generate analysis Jupyter notebooks (executed cells + markdown)."""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def write(name: str, cells: list[dict]):
    path = NB_DIR / name
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", path)


COMMON = r'''
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.data_utils import load_query_clicks, load_sku_desc, FIGURES, ARTIFACTS, ensure_dirs
ensure_dirs()
sns.set_theme(style="whitegrid", context="notebook")
RED, ACCENT, TEAL = "#E31E24", "#0984E3", "#00B894"
'''


def main():
    write(
        "01_data_overview.ipynb",
        [
            md("# 01. Обзор данных М.Видео\n\nСхема датасетов, размеры, пропуски, примеры строк."),
            code(COMMON),
            code(
                """
# Полные размеры (из спецификации / артефактов)
import json
stats_path = ARTIFACTS / "dataset_stats.json"
if stats_path.exists():
    print(json.dumps(json.loads(stats_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2))
else:
    print("Запустите scripts/run_pipeline.py для полного stats")
"""
            ),
            code(
                """
df = load_query_clicks(nrows=50_000)
df.head()
"""
            ),
            code(
                """
print("shape", df.shape)
print(df.dtypes)
print("nulls:\\n", df.isna().sum())
print("unique queries", df["query_text"].nunique())
print("unique brands", df["sku_brand_name"].replace("", np.nan).nunique())
df.describe(include="all").T
"""
            ),
            code(
                """
desc = load_sku_desc(nrows=20_000)
print(desc.shape)
desc.head(3)
"""
            ),
            md("## Файлы кейса\n\n| Файл | Содержание |\n|------|------------|\n| `query_clicks.parquet` | ~31M кликов: запрос → SKU |\n| `sku_desc.parquet` | ~1.18M карточек: title/description |\n| `skus.pkl` | YML-каталог (`yml_catalog.shop`) |"),
        ],
    )

    write(
        "02_eda_queries.ipynb",
        [
            md("# 02. EDA поисковых запросов\n\nДлины, частоты, токены, позиции кликов."),
            code(COMMON),
            code(
                """
df = load_query_clicks(nrows=200_000)
df["qlen"] = df["query_text"].astype(str).str.len()
fig, ax = plt.subplots(figsize=(9,4))
ax.hist(df["qlen"].clip(0,80), bins=40, color=RED, edgecolor="white")
ax.set_title("Длина запросов"); ax.set_xlabel("символы")
plt.savefig(FIGURES/"01_query_length_dist.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
            code(
                """
top = df["query_text"].str.lower().value_counts().head(20)
fig, ax = plt.subplots(figsize=(9,6))
top.iloc[::-1].plot(kind="barh", ax=ax, color=RED)
ax.set_title("Топ запросов")
plt.savefig(FIGURES/"02_top_queries.png", dpi=200, bbox_inches="tight"); plt.show()
top
"""
            ),
            code(
                """
from src.ner.labeling import tokenize
from collections import Counter
cnt = Counter()
for q in df["query_text"].str.lower().sample(40_000, random_state=0):
    cnt.update([t for t,_,_ in tokenize(q) if len(t)>2])
pd.Series(dict(cnt.most_common(20)))
"""
            ),
        ],
    )

    write(
        "03_eda_products.ipynb",
        [
            md("# 03. EDA товаров и брендов\n\nЦены, бренды, subject_id, описания."),
            code(COMMON),
            code(
                """
df = load_query_clicks(nrows=200_000)
desc = load_sku_desc(nrows=80_000)
brands = df.loc[df.sku_brand_name.str.len()>0, "sku_brand_name"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9,6))
brands.iloc[::-1].plot(kind="barh", color=ACCENT, ax=ax)
ax.set_title("Топ брендов"); plt.savefig(FIGURES/"03_top_brands.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
            code(
                """
prices = df["sku_price"].clip(upper=df["sku_price"].quantile(0.99))
fig, ax = plt.subplots(figsize=(9,4))
ax.hist(prices, bins=50, color=TEAL, edgecolor="white")
ax.set_title("Цены кликнутых SKU"); plt.savefig(FIGURES/"04_price_distribution.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
            code(
                """
fig, axes = plt.subplots(1,2, figsize=(11,4))
axes[0].hist(desc["title"].astype(str).str.len().clip(0,200), bins=40, color=ACCENT)
axes[0].set_title("len(title)")
axes[1].hist(desc["description"].astype(str).str.len().clip(0,5000), bins=40, color=TEAL)
axes[1].set_title("len(description)")
plt.savefig(FIGURES/"09_title_length_dist.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
        ],
    )

    write(
        "04_similarity_matrices.ipynb",
        [
            md("# 04. Матрицы схожести и co-occurrence\n\nTF-IDF cosine, heatmap запрос×бренд."),
            code(COMMON),
            code(
                """
from sklearn.feature_extraction.text import TfidfVectorizer
df = load_query_clicks(nrows=150_000)
uniq = df["query_text"].str.lower().value_counts().head(25).index.tolist()
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)).fit_transform(uniq)
sim = (X @ X.T).toarray()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(sim, xticklabels=uniq, yticklabels=uniq, cmap="YlOrRd", ax=ax)
plt.xticks(rotation=90, fontsize=7); plt.yticks(fontsize=7)
ax.set_title("TF-IDF similarity топ-запросов")
plt.savefig(FIGURES/"07_tfidf_similarity_heatmap.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
            code(
                """
top_q = df["query_text"].str.lower().value_counts().head(10).index
top_b = df.loc[df.sku_brand_name.str.len()>0, "sku_brand_name"].value_counts().head(10).index
sub = df[df["query_text"].str.lower().isin(top_q) & df["sku_brand_name"].isin(top_b)]
mat = pd.crosstab(sub["query_text"].str.lower(), sub["sku_brand_name"])
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(np.log1p(mat), cmap="Reds", ax=ax)
ax.set_title("log co-occurrence query×brand")
plt.savefig(FIGURES/"06_query_brand_heatmap.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
        ],
    )

    write(
        "05_click_patterns.ipynb",
        [
            md("# 05. Паттерны кликов\n\nПозиция, цена, связь с выдачей."),
            code(COMMON),
            code(
                """
df = load_query_clicks(nrows=200_000)
fig, ax = plt.subplots(figsize=(9,4))
ax.hist(df["sku_position"].clip(0,50), bins=51, color="#E17055", edgecolor="white")
ax.set_title("Позиция при клике"); plt.savefig(FIGURES/"05_position_distribution.png", dpi=200, bbox_inches="tight"); plt.show()
print("доля кликов в топ-5:", (df.sku_position<=5).mean())
"""
            ),
            code(
                """
s = df.sample(min(25000,len(df)), random_state=1)
s = s[(s.sku_position<=40) & (s.sku_price<=s.sku_price.quantile(0.95))]
fig, ax = plt.subplots(figsize=(9,5))
hb = ax.hexbin(s.sku_position, s.sku_price, gridsize=30, cmap="Reds", mincnt=1)
fig.colorbar(hb); ax.set_title("Цена vs позиция")
plt.savefig(FIGURES/"10_price_vs_position.png", dpi=200, bbox_inches="tight"); plt.show()
"""
            ),
        ],
    )

    write(
        "06_weak_supervision_ner.ipynb",
        [
            md("# 06. Weak supervision для NER\n\nСловари брендов/категорий → BIO-разметка запросов."),
            code(COMMON),
            code(
                """
from src.ner.labeling import WeakLabeler, bio_to_entities
if (ARTIFACTS/"brands.txt").exists():
    lab = WeakLabeler.from_files(ARTIFACTS/"brands.txt", ARTIFACTS/"categories.txt")
else:
    lab = WeakLabeler.from_iterables(["Apple","Samsung","Xiaomi","ASUS"], ["смартфон","ноутбук","холодильник"])
examples = ["смартфон apple 256", "холодильник samsung", "ноутбук asus 16 гб", "пылесос dyson", "аэрогриль"]
for q in examples:
    bio = lab.label_query(q)
    print(q, "=>", bio, "ents=", bio_to_entities(bio, query=q))
"""
            ),
            code(
                """
df = load_query_clicks(nrows=30_000)
from collections import Counter
c = Counter()
for q in df["query_text"].drop_duplicates().sample(min(5000, df["query_text"].nunique()), random_state=0):
    for _,t in lab.label_query(q):
        if t.startswith("B-"): c[t[2:]] += 1
pd.Series(c).plot(kind="bar", color=RED, title="Сущности в weak labels"); plt.show()
"""
            ),
        ],
    )

    write(
        "07_train_baseline_classifier.ipynb",
        [
            md("# 07. Baseline: TF-IDF + LogisticRegression (бренд)\n\nМетрики Accuracy / F1."),
            code(COMMON),
            code(
                """
import json, joblib
from pathlib import Path
m = ARTIFACTS/"metrics.json"
if m.exists():
    metrics = json.loads(m.read_text(encoding="utf-8"))
    print("Baseline:", json.dumps(metrics.get("baseline_brand_classifier",{}), ensure_ascii=False, indent=2))
else:
    print("Сначала: python scripts/run_pipeline.py")
from IPython.display import Image, display
p = FIGURES/"13_confusion_matrix.png"
if p.exists(): display(Image(filename=str(p)))
"""
            ),
            code(
                """
# Быстрый мини-baseline на выборке
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = load_query_clicks(nrows=80_000)
g = df[df.sku_brand_name.str.len()>0].groupby(["query_text","sku_brand_name"]).size().reset_index(name="n")
idx = g.groupby("query_text")["n"].idxmax(); qb = g.loc[idx]
keep = qb["sku_brand_name"].value_counts()
qb = qb[qb.sku_brand_name.isin(keep[keep>=20].index)]
Xtr,Xte,ytr,yte = train_test_split(qb.query_text, qb.sku_brand_name, test_size=0.25, random_state=42)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4), min_df=2, max_features=30000)
Xt = vec.fit_transform(Xtr); Xe = vec.transform(Xte)
clf = LogisticRegression(max_iter=150, n_jobs=-1).fit(Xt,ytr)
pred = clf.predict(Xe)
print(classification_report(yte, pred, zero_division=0)[:1500])
"""
            ),
        ],
    )

    write(
        "08_train_ner_model.ipynb",
        [
            md("# 08. Обучение CRF NER\n\nChar/word features → sklearn-crfsuite. Entity F1 / token Accuracy."),
            code(COMMON),
            code(
                """
import json
from IPython.display import Image, display
m = json.loads((ARTIFACTS/"metrics.json").read_text(encoding="utf-8")) if (ARTIFACTS/"metrics.json").exists() else {}
print(json.dumps(m.get("ner_crf",{}), ensure_ascii=False, indent=2))
for name in ["14_ner_f1_by_entity.png","15_learning_curve.png","20_entity_distribution.png","18_metrics_summary.png"]:
    p = FIGURES/name
    if p.exists():
        display(Image(filename=str(p)))
"""
            ),
            code(
                """
from src.ner.labeling import WeakLabeler
from src.ner.model_crf import CRFNerModel
from src.ner.metrics import summarize_metrics
from src.ner.features import sent2labels
lab = WeakLabeler.from_files(ARTIFACTS/"brands.txt", ARTIFACTS/"categories.txt")
df = load_query_clicks(nrows=40_000)
qs = df["query_text"].drop_duplicates().sample(min(3000, df["query_text"].nunique()), random_state=1)
data = [s for s in (lab.label_query(q) for q in qs) if s]
split = int(0.8*len(data))
model = CRFNerModel(max_iterations=30).fit(data[:split])
yt = [sent2labels(s) for s in data[split:]]
yp = model.predict(data[split:])
print(summarize_metrics(yt, yp))
print(model.predict_query("смартфон apple 256"))
"""
            ),
        ],
    )

    write(
        "09_embeddings_similarity.ipynb",
        [
            md("# 09. Эмбеддинги и близость запросов\n\nTF-IDF char n-grams, t-SNE, similarity matrix."),
            code(COMMON),
            code(
                """
from IPython.display import Image, display
for name in ["16_embedding_tsne.png","19_similarity_queries.png","07_tfidf_similarity_heatmap.png"]:
    p = FIGURES/name
    if p.exists(): display(Image(filename=str(p)))
"""
            ),
            code(
                """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
df = load_query_clicks(nrows=100_000)
q = df["query_text"].str.lower().value_counts().head(300).index.tolist()
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5)).fit_transform(q)
Xr = TruncatedSVD(20, random_state=0).fit_transform(X)
emb = TSNE(2, random_state=0, perplexity=25, init="pca").fit_transform(Xr)
plt.figure(figsize=(8,6)); plt.scatter(emb[:,0], emb[:,1], s=16, c=RED, alpha=0.7)
plt.title("t-SNE запросов"); plt.show()
"""
            ),
        ],
    )

    write(
        "10_service_benchmark.ipynb",
        [
            md("# 10. Сервис и бенчмарк latency\n\nЦель кейса: JSON-ответ **< 100 мс**."),
            code(COMMON),
            code(
                """
import json
from IPython.display import Image, display
m = json.loads((ARTIFACTS/"metrics.json").read_text(encoding="utf-8")) if (ARTIFACTS/"metrics.json").exists() else {}
print(json.dumps(m.get("latency",{}), ensure_ascii=False, indent=2)[:2000])
for name in ["17_latency_histogram.png","18_metrics_summary.png"]:
    p = FIGURES/name
    if p.exists(): display(Image(filename=str(p)))
"""
            ),
            code(
                """
from src.service.extractor import QueryEntityExtractor
ext = QueryEntityExtractor.from_artifacts(ARTIFACTS, ROOT/"models")
for q in ["смартфон apple 256", "холодильник samsung", "ноутбук asus 16 гб", "аэрогриль"]:
    print(ext.extract(q))
"""
            ),
            md("## Запуск API\n\n```bash\nuvicorn src.service.app:app --host 0.0.0.0 --port 8000\ncurl -X POST localhost:8000/extract -H 'Content-Type: application/json' -d '{\"query\":\"пылесос dyson\"}'\n```"),
        ],
    )

    write(
        "11_skus_catalog_explore.ipynb",
        [
            md("# 11. Разбор `skus.pkl` (YML-каталог)\n\nСтруктура Yandex Market Language."),
            code(COMMON),
            code(
                """
import pickle
from pathlib import Path
p = ROOT/"файлы"/"skus.pkl"
print("size_gb", p.stat().st_size/1e9)
# Осторожно: файл ~1.5GB, загрузка может занять несколько минут
with open(p, "rb") as f:
    obj = pickle.load(f)
print(type(obj), obj.keys() if isinstance(obj, dict) else None)
shop = obj["yml_catalog"]["shop"]
print("shop keys:", list(shop.keys()))
for k,v in shop.items():
    print(k, type(v), (len(v) if hasattr(v,"__len__") and not isinstance(v,str) else ""))
"""
            ),
        ],
    )

    write(
        "12_end_to_end_demo.ipynb",
        [
            md("# 12. End-to-end демо\n\nОт сырого запроса к структурированному JSON."),
            code(COMMON),
            code(
                """
from src.service.extractor import QueryEntityExtractor
import json
ext = QueryEntityExtractor.from_artifacts(ARTIFACTS, ROOT/"models")
demo = [
    "айфон 15 pro 256",
    "samsung galaxy buds",
    "стиральная машинка bosch",
    "телевизор 55 дюймов lg oled",
    "яндекс станция мини",
]
rows = [ext.extract(q) for q in demo]
print(json.dumps(rows, ensure_ascii=False, indent=2))
pd.DataFrame([{**{k:r[k] for k in ["query","brand","category","latency_ms"]}, "n_entities":len(r["entities"])} for r in rows])
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
