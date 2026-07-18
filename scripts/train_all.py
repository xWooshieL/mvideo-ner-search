#!/usr/bin/env python
"""End-to-end training: dictionaries → weak BIO labels → CRF NER + TF-IDF classifiers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import learning_curve, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ner.features import sent2labels  # noqa: E402
from src.ner.labeling import WeakLabeler, bio_to_entities, entities_to_structured  # noqa: E402
from src.ner.metrics import summarize_metrics  # noqa: E402
from src.ner.model_crf import CRFNerModel  # noqa: E402

COL_BRAND = "toValidUTF8(sku_brand_name)"
COL_NAME = "toValidUTF8(sku_name)"
COL_QUERY = "toValidUTF8(query_text)"


def load_click_sample(path: Path, max_rows: int) -> "object":
    import pandas as pd

    pf = pq.ParquetFile(path)
    parts = []
    seen = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=[COL_BRAND, COL_NAME, COL_QUERY])
        df = t.to_pandas()
        df.columns = ["brand", "name", "query"]
        parts.append(df)
        seen += len(df)
        if seen >= max_rows:
            break
    out = pd.concat(parts, ignore_index=True).head(max_rows)
    out["query"] = out["query"].fillna("").astype(str).str.strip()
    out["brand"] = out["brand"].fillna("").astype(str).str.strip()
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out = out[out["query"].str.len() >= 2].drop_duplicates(subset=["query"])
    return out


def ensure_dictionaries(artifacts: Path, data: Path, max_rows: int) -> None:
    brands = artifacts / "brands.txt"
    cats = artifacts / "categories.txt"
    if brands.exists() and cats.exists():
        print("Dictionaries already exist, skipping build.")
        return
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_dictionaries.py"),
            "--data",
            str(data),
            "--out",
            str(artifacts),
            "--max-rows",
            str(max_rows),
        ]
    )


def plot_entity_distribution(sents, figures: Path) -> None:
    labels = []
    for s in sents:
        for _, t in s:
            if t.startswith("B-"):
                labels.append(t[2:])
    c = Counter(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    labs, vals = zip(*c.most_common()) if c else ([], [])
    ax.bar(labs, vals, color="#E31E24")
    ax.set_title("Распределение сущностей (weak labels)")
    ax.set_ylabel("Количество")
    fig.tight_layout()
    fig.savefig(figures / "20_entity_distribution.png", dpi=140)
    plt.close(fig)


def train_classifiers(df, models: Path, figures: Path, artifacts: Path) -> dict:
    # Brand classifier on rows with brand
    brand_df = df[df["brand"].str.len() >= 2].copy()
    # Keep top brands for tractable multi-class
    top_brands = {b for b, _ in Counter(brand_df["brand"]).most_common(80)}
    brand_df = brand_df[brand_df["brand"].isin(top_brands)]

    # Category proxy: first matching category keyword in query/name via labeler
    labeler = WeakLabeler.from_files(artifacts / "brands.txt", artifacts / "categories.txt")
    cats = []
    for q in df["query"]:
        ents = bio_to_entities(labeler.label_query(q), query=q)
        st = entities_to_structured(ents, labeler)
        cats.append(st["category"] or "__UNK__")
    df = df.copy()
    df["category_label"] = cats
    cat_df = df[df["category_label"] != "__UNK__"].copy()
    top_cats = {c for c, _ in Counter(cat_df["category_label"]).most_common(40)}
    cat_df = cat_df[cat_df["category_label"].isin(top_cats)]

    metrics = {}

    # --- Brand ---
    if len(brand_df) >= 200:
        Xb_train, Xb_test, yb_train, yb_test = train_test_split(
            brand_df["query"],
            brand_df["brand"],
            test_size=0.2,
            random_state=42,
            stratify=brand_df["brand"] if brand_df["brand"].value_counts().min() >= 2 else None,
        )
        brand_pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50000)),
                (
                    "clf",
                    LogisticRegression(max_iter=200, n_jobs=-1, solver="saga"),
                ),
            ]
        )
        brand_pipe.fit(Xb_train, yb_train)
        pred = brand_pipe.predict(Xb_test)
        metrics["brand_clf"] = {
            "f1_macro": float(f1_score(yb_test, pred, average="macro")),
            "f1_micro": float(f1_score(yb_test, pred, average="micro")),
            "accuracy": float((pred == yb_test).mean()),
            "n_train": int(len(Xb_train)),
            "n_test": int(len(Xb_test)),
            "n_classes": int(brand_df["brand"].nunique()),
        }
        joblib.dump(brand_pipe, models / "brand_clf.joblib")

        # Confusion matrix for top-15 brands
        top15 = [b for b, _ in Counter(yb_test).most_common(15)]
        mask = yb_test.isin(top15)
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(
            yb_test[mask],
            pred[mask],
            labels=top15,
            ax=ax,
            xticks_rotation=45,
            colorbar=False,
        )
        ax.set_title("Confusion matrix — бренды (top-15)")
        fig.tight_layout()
        fig.savefig(figures / "13_confusion_matrix.png", dpi=140)
        plt.close(fig)
        print("Brand clf:", metrics["brand_clf"])
    else:
        print("Not enough brand rows for classifier")

    # --- Category ---
    if len(cat_df) >= 200:
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            cat_df["query"],
            cat_df["category_label"],
            test_size=0.2,
            random_state=42,
            stratify=cat_df["category_label"] if cat_df["category_label"].value_counts().min() >= 2 else None,
        )
        cat_pipe = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40000)),
                (
                    "clf",
                    LogisticRegression(max_iter=200, n_jobs=-1, solver="saga"),
                ),
            ]
        )
        cat_pipe.fit(Xc_train, yc_train)
        pred = cat_pipe.predict(Xc_test)
        metrics["category_clf"] = {
            "f1_macro": float(f1_score(yc_test, pred, average="macro")),
            "f1_micro": float(f1_score(yc_test, pred, average="micro")),
            "accuracy": float((pred == yc_test).mean()),
            "n_train": int(len(Xc_train)),
            "n_test": int(len(Xc_test)),
            "n_classes": int(cat_df["category_label"].nunique()),
            "report": classification_report(yc_test, pred, output_dict=True, zero_division=0),
        }
        joblib.dump(cat_pipe, models / "category_clf.joblib")
        print("Category clf:", {k: metrics["category_clf"][k] for k in ("f1_macro", "f1_micro", "accuracy")})
    return metrics


def train_ner(df, labeler: WeakLabeler, models: Path, figures: Path, max_sents: int) -> dict:
    queries = df["query"].tolist()
    # Prefer queries that get at least one entity
    sents = labeler.label_dataset(queries, min_entities=1)
    if len(sents) > max_sents:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sents), size=max_sents, replace=False)
        sents = [sents[i] for i in idx]
    print(f"NER labeled sentences with ≥1 entity: {len(sents)}")
    plot_entity_distribution(sents, figures)

    train_sents, test_sents = train_test_split(sents, test_size=0.2, random_state=42)

    # Learning curve: train on growing subsets
    sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    curve_f1 = []
    curve_acc = []
    curve_n = []
    for frac in sizes:
        n = max(50, int(len(train_sents) * frac))
        subset = train_sents[:n]
        model = CRFNerModel(max_iterations=60)
        t0 = time.time()
        model.fit(subset)
        print(f"  trained on {n} sents in {time.time() - t0:.1f}s")
        y_true = [sent2labels(s) for s in test_sents]
        y_pred = model.predict(test_sents)
        m = summarize_metrics(y_true, y_pred)
        curve_f1.append(m["micro"]["f1"])
        curve_acc.append(m["token_accuracy"])
        curve_n.append(n)

    # Final model on full train
    final = CRFNerModel(max_iterations=80)
    final.fit(train_sents)
    final.save(models / "ner_crf.pkl")
    y_true = [sent2labels(s) for s in test_sents]
    y_pred = final.predict(test_sents)
    report = summarize_metrics(y_true, y_pred)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve_n, curve_f1, "o-", color="#E31E24", label="Entity micro-F1")
    ax.plot(curve_n, curve_acc, "s--", color="#333333", label="Token Accuracy")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("Метрика")
    ax.set_title("Learning curve — CRF NER")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures / "15_learning_curve.png", dpi=140)
    plt.close(fig)

    per = report["per_label"]
    labs = list(per.keys())
    f1s = [per[l]["f1"] for l in labs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labs, f1s, color=["#E31E24", "#555555", "#888888"][: len(labs)])
    ax.set_ylim(0, 1.05)
    ax.set_title("NER F1 по типам сущностей")
    ax.set_ylabel("F1")
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(figures / "14_ner_f1_by_entity.png", dpi=140)
    plt.close(fig)

    return {
        "n_train": len(train_sents),
        "n_test": len(test_sents),
        "token_accuracy": report["token_accuracy"],
        "entity_micro_f1": report["micro"]["f1"],
        "entity_micro_precision": report["micro"]["precision"],
        "entity_micro_recall": report["micro"]["recall"],
        "entity_macro_f1": report["macro_f1"],
        "per_label": report["per_label"],
        "learning_curve": {"n": curve_n, "f1": curve_f1, "accuracy": curve_acc},
    }


def plot_metrics_summary(metrics: dict, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names, vals = [], []
    ner = metrics.get("ner", {})
    if ner:
        names += ["NER Acc", "NER micro-F1", "NER macro-F1"]
        vals += [
            ner.get("token_accuracy", 0),
            ner.get("entity_micro_f1", 0),
            ner.get("entity_macro_f1", 0),
        ]
    if "brand_clf" in metrics.get("classifiers", {}):
        names.append("Brand F1-macro")
        vals.append(metrics["classifiers"]["brand_clf"]["f1_macro"])
    if "category_clf" in metrics.get("classifiers", {}):
        names.append("Cat F1-macro")
        vals.append(metrics["classifiers"]["category_clf"]["f1_macro"])
    if "latency" in metrics:
        names.append("p95 lat/100ms")
        vals.append(min(metrics["latency"].get("p95_ms", 0) / 100.0, 1.5))
    colors = sns.color_palette("Reds_r", n_colors=max(len(names), 1))
    ax.barh(names, vals, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_title("Сводка метрик пайплайна")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 1.0), i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(figures / "18_metrics_summary.png", dpi=140)
    plt.close(fig)


def quick_latency(artifacts: Path, models: Path, queries, figures: Path) -> dict:
    from src.service.extractor import QueryEntityExtractor

    ext = QueryEntityExtractor.from_artifacts(artifacts, models)
    # warmup
    for q in queries[:20]:
        ext.extract(q)
    lat = []
    for q in queries[:500]:
        r = ext.extract(q)
        lat.append(r["latency_ms"])
    arr = np.array(lat)
    stats = {
        "n": int(len(arr)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=40, color="#E31E24", edgecolor="white")
    ax.axvline(100, color="black", linestyle="--", label="100 ms SLA")
    ax.set_title("Latency histogram — /extract")
    ax.set_xlabel("ms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "17_latency_histogram.png", dpi=140)
    plt.close(fig)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "файлы" / "query_clicks.parquet")
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--max-ner-sents", type=int, default=40_000)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    args = ap.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    print("=== 1. Dictionaries ===")
    ensure_dictionaries(args.artifacts, args.data, min(args.max_rows, 400_000))

    print("=== 2. Load sample ===")
    df = load_click_sample(args.data, args.max_rows)
    print(f"Unique queries: {len(df)}")

    labeler = WeakLabeler.from_files(args.artifacts / "brands.txt", args.artifacts / "categories.txt")

    print("=== 3. Baseline classifiers ===")
    clf_metrics = train_classifiers(df, args.models, args.figures, args.artifacts)

    print("=== 4. CRF NER ===")
    ner_metrics = train_ner(df, labeler, args.models, args.figures, args.max_ner_sents)

    print("=== 5. Latency probe ===")
    lat_metrics = quick_latency(args.artifacts, args.models, df["query"].tolist(), args.figures)

    metrics = {
        "sample_rows": int(args.max_rows),
        "unique_queries": int(len(df)),
        "classifiers": clf_metrics,
        "ner": ner_metrics,
        "latency": lat_metrics,
    }
    plot_metrics_summary(metrics, args.figures)

    out = args.artifacts / "metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics → {out}")
    print(json.dumps({k: metrics[k] for k in ("ner", "latency")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
