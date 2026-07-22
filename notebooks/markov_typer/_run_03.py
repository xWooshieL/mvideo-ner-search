"""Headless train for ATTR-type TF-IDF classifiers (same as 03 notebook)."""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data_utils import MODELS, ensure_dirs, save_stats
from src.ner.attr_type_clf import Col, LABEL_UNKNOWN, looks_like_model
from src.ner.markov_typer import MarkovAttrTyper, _norm_token

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
VAL_SIZE = 0.2
MIN_CLASS = 25
UNKNOWN_SHARE_CAP = 0.18
INCLUDE_AUG = True
OUT = ROOT / "artifacts" / "attr_type"
SILVER = OUT / "attr_type_silver.parquet"
REPORT = ROOT / "notebooks" / "markov_typer" / "attr_type_classifier.md"


def log(msg: str) -> None:
    print(msg, flush=True)


def stratified_split(df, label_col, test_size, seed):
    y = df[label_col].astype(str)
    vc = y.value_counts()
    rare1 = set(vc[vc < 2].index)
    ok = df[~y.isin(rare1)]
    hold = df[y.isin(rare1)]
    if ok[label_col].nunique() < 2 or len(ok) < 10:
        return df.copy(), df.iloc[0:0].copy()
    tr, va = train_test_split(
        ok, test_size=test_size, random_state=seed, stratify=ok[label_col]
    )
    if len(hold):
        tr = pd.concat([tr, hold], ignore_index=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def make_pipelines(seed=SEED):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40_000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=15_000)
    lr = dict(max_iter=300, solver="lbfgs", random_state=seed)
    return {
        "logreg_span_char": Pipeline(
            [
                ("span", Col("span_text")),
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40_000
                    ),
                ),
                ("clf", LogisticRegression(**lr)),
            ]
        ),
        "logreg_span_wordchar": Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            ("char", Pipeline([("c", Col("span_text")), ("t", char)])),
                            ("word", Pipeline([("c", Col("span_text")), ("t", word)])),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(**lr)),
            ]
        ),
        "logreg_span_ctx": Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            (
                                "span",
                                Pipeline(
                                    [
                                        ("c", Col("span_text")),
                                        (
                                            "t",
                                            TfidfVectorizer(
                                                analyzer="char_wb",
                                                ngram_range=(2, 4),
                                                min_df=2,
                                                max_features=40_000,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                            (
                                "ctx",
                                Pipeline(
                                    [
                                        ("c", Col("context_text")),
                                        (
                                            "t",
                                            TfidfVectorizer(
                                                analyzer="char_wb",
                                                ngram_range=(2, 4),
                                                min_df=1,
                                                max_features=15_000,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(**lr)),
            ]
        ),
        "sgd_span_ctx_masked": Pipeline(
            [
                (
                    "feats",
                    FeatureUnion(
                        [
                            (
                                "span",
                                Pipeline(
                                    [
                                        ("c", Col("span_text")),
                                        (
                                            "t",
                                            TfidfVectorizer(
                                                analyzer="char_wb",
                                                ngram_range=(2, 5),
                                                min_df=2,
                                                max_features=50_000,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                            (
                                "masked",
                                Pipeline(
                                    [
                                        ("c", Col("query_masked")),
                                        (
                                            "t",
                                            TfidfVectorizer(
                                                analyzer="word",
                                                ngram_range=(1, 2),
                                                min_df=2,
                                                max_features=20_000,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss", alpha=1e-5, max_iter=40, random_state=seed, tol=1e-3
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    assert SILVER.exists(), SILVER

    silver = pd.read_parquet(SILVER)
    if not INCLUDE_AUG:
        silver = silver.loc[~silver["is_aug"]].copy()
    log(f"silver={len(silver)} aug={int(silver['is_aug'].sum())}")

    raw_vc = silver.loc[~silver["is_aug"], "y"].value_counts()
    rare = set(raw_vc[raw_vc < MIN_CLASS].index) | {"other"}
    log(f"rare→UNKNOWN {sorted(rare)}")

    df = silver.copy()
    df["y_raw"] = df["y"].astype(str)
    df["y"] = df["y_raw"].where(~df["y_raw"].isin(rare), LABEL_UNKNOWN)
    df.loc[df["span_text"].map(lambda s: looks_like_model(str(s))), "y"] = LABEL_UNKNOWN

    unk = df[df["y"] == LABEL_UNKNOWN]
    pos = df[df["y"] != LABEL_UNKNOWN]
    max_unk = max(80, int(UNKNOWN_SHARE_CAP * len(pos) / max(1e-6, 1 - UNKNOWN_SHARE_CAP)))
    if len(unk) > max_unk:
        unk = unk.sample(n=max_unk, random_state=SEED)
    df = pd.concat([pos, unk], ignore_index=True)
    vc = df["y"].value_counts()
    df = df[df["y"].isin(vc[vc >= MIN_CLASS].index)].copy()
    df["context_text"] = (
        df["brand"].fillna("").astype(str) + " " + df["category"].fillna("").astype(str)
    ).str.strip()
    df["query_masked"] = df["query_masked_all_attr"].fillna("").astype(str)

    train_df, val_df = stratified_split(df, "y", VAL_SIZE, SEED)
    train_df.to_parquet(OUT / "attr_type_train.parquet", index=False)
    val_df.to_parquet(OUT / "attr_type_val.parquet", index=False)
    log(f"train={len(train_df)} val={len(val_df)} classes={train_df['y'].nunique()}")

    classes = sorted(train_df["y"].unique(), key=lambda x: (x == LABEL_UNKNOWN, x))
    ytr = train_df["y"].astype(str).values
    yva = val_df["y"].astype(str).values
    multi_m = (val_df["n_attrs_in_query"] >= 2).values

    results = {}
    fitted = {}
    for name, pipe in make_pipelines().items():
        log(f"train {name}")
        pipe.fit(train_df, ytr)
        pred = pipe.predict(val_df)
        pack = {
            "accuracy": float(accuracy_score(yva, pred)),
            "f1_macro": float(f1_score(yva, pred, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(yva, pred, average="micro", zero_division=0)),
            "f1_weighted": float(f1_score(yva, pred, average="weighted", zero_division=0)),
            "report": classification_report(
                yva, pred, labels=classes, output_dict=True, zero_division=0
            ),
            "pred": pred,
        }
        if multi_m.any():
            pack["multi_f1_macro"] = float(
                f1_score(yva[multi_m], pred[multi_m], average="macro", zero_division=0)
            )
            pack["multi_accuracy"] = float(accuracy_score(yva[multi_m], pred[multi_m]))
        else:
            pack["multi_f1_macro"] = None
            pack["multi_accuracy"] = None
        pack["f1_UNKNOWN"] = float(pack["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0))
        results[name] = pack
        fitted[name] = pipe
        joblib.dump(pipe, OUT / f"{name}.joblib")
        log(
            f"  acc={pack['accuracy']:.3f} macro={pack['f1_macro']:.3f} "
            f"multi={pack['multi_f1_macro']} unk={pack['f1_UNKNOWN']:.3f}"
        )

    markov = MarkovAttrTyper()
    _big: dict = defaultdict(Counter)
    _unit: dict = defaultdict(Counter)
    _trans: dict = defaultdict(Counter)
    for _, r in train_df.iterrows():
        toks = [_norm_token(t) for t in str(r["span_text"]).split()]
        y = str(r["y"])
        markov.n_spans += 1
        for a, b in zip(toks, toks[1:]):
            _big[f"{a}|{b}"][y] += 1
            _trans[a][b] += 1
        if toks:
            _unit[toks[-1]][y] += 1
    markov.bigram_to_type = {k: dict(v) for k, v in _big.items()}
    markov.unit_to_type = {k: dict(v) for k, v in _unit.items()}
    markov.transitions = {k: dict(v) for k, v in _trans.items()}

    m_pred = []
    for s in val_df["span_text"].astype(str):
        lab, _ = markov.predict(s.split())
        m_pred.append(LABEL_UNKNOWN if lab == "unknown" else lab)
    m_pred = np.array(m_pred)
    results["markov_lookup"] = {
        "accuracy": float(accuracy_score(yva, m_pred)),
        "f1_macro": float(f1_score(yva, m_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(yva, m_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(yva, m_pred, average="weighted", zero_division=0)),
        "multi_f1_macro": float(
            f1_score(yva[multi_m], m_pred[multi_m], average="macro", zero_division=0)
        )
        if multi_m.any()
        else None,
        "multi_accuracy": float(accuracy_score(yva[multi_m], m_pred[multi_m]))
        if multi_m.any()
        else None,
        "report": classification_report(
            yva, m_pred, labels=classes, output_dict=True, zero_division=0
        ),
        "pred": m_pred,
    }
    results["markov_lookup"]["f1_UNKNOWN"] = float(
        results["markov_lookup"]["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0)
    )
    markov.save(MODELS / "markov_typer_attr_clf_baseline.json")
    log(
        f"markov acc={results['markov_lookup']['accuracy']:.3f} "
        f"macro={results['markov_lookup']['f1_macro']:.3f}"
    )

    summary = (
        pd.DataFrame(
            [
                {
                    "model": name,
                    "accuracy": v["accuracy"],
                    "f1_macro": v["f1_macro"],
                    "f1_micro": v["f1_micro"],
                    "f1_weighted": v["f1_weighted"],
                    "multi_f1_macro": v.get("multi_f1_macro"),
                    "multi_accuracy": v.get("multi_accuracy"),
                    "f1_UNKNOWN": v.get("f1_UNKNOWN"),
                }
                for name, v in results.items()
            ]
        )
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(OUT / "models_summary.csv", index=False)
    log(summary.to_string(index=False))

    sklearn_names = [n for n in summary["model"] if n != "markov_lookup"]
    best_name = summary.loc[summary["model"].isin(sklearn_names)].iloc[0]["model"]
    pipe_best = fitted[best_name]
    pred_best = results[best_name]["pred"]
    log(f"BEST={best_name}")
    log(classification_report(yva, pred_best, labels=classes, digits=3, zero_division=0))

    rep = results[best_name]["report"]
    per_class = pd.DataFrame(
        [
            {
                "class": c,
                "precision": rep[c]["precision"],
                "recall": rep[c]["recall"],
                "f1": rep[c]["f1-score"],
                "support": int(rep[c]["support"]),
            }
            for c in classes
            if c in rep
        ]
    ).sort_values("f1", ascending=False)
    per_class.to_csv(OUT / "per_class_f1__BEST.csv", index=False)
    per_class.to_csv(OUT / f"per_class_f1__{best_name}.csv", index=False)
    for name in make_pipelines():
        r = results[name]["report"]
        pd.DataFrame(
            [
                {
                    "class": c,
                    "precision": r[c]["precision"],
                    "recall": r[c]["recall"],
                    "f1": r[c]["f1-score"],
                    "support": int(r[c]["support"]),
                }
                for c in classes
                if c in r
            ]
        ).to_csv(OUT / f"per_class_f1__{name}.csv", index=False)

    joblib.dump(pipe_best, MODELS / "attr_type_clf.joblib")
    joblib.dump(pipe_best, MODELS / f"attr_type_clf__{best_name}.joblib")

    demos = ["16 гб", "16 кб", "16 kb", "256 gb", "15.6 дюйм", "g pro", "2 кг", "белый"]
    for s in demos:
        if looks_like_model(s):
            log(f"demo {s!r} → UNKNOWN (rule)")
            continue
        row = pd.DataFrame(
            [
                {
                    "span_text": s,
                    "context_text": "ноутбук asus",
                    "query_masked": "ноутбук asus <ATTR>",
                    "brand": "asus",
                    "category": "ноутбук",
                }
            ]
        )
        log(f"demo {s!r} → {pipe_best.predict(row)[0]}")

    metrics = {
        "best_model": best_name,
        "summary": summary.to_dict(orient="records"),
        "best_raw": {
            k: results[best_name][k]
            for k in ["accuracy", "f1_macro", "f1_micro", "multi_f1_macro", "f1_UNKNOWN"]
        },
        "markov": {
            k: results["markov_lookup"][k]
            for k in ["accuracy", "f1_macro", "multi_f1_macro", "f1_UNKNOWN"]
        },
        "n_train": len(train_df),
        "n_val": len(val_df),
        "classes": classes,
        "min_class": MIN_CLASS,
        "unknown_share_cap": UNKNOWN_SHARE_CAP,
        "include_aug": INCLUDE_AUG,
        "design_note": (
            "char n-grams on span only; brand/category/masked query separate; "
            "one row per ATTR span"
        ),
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_stats({"attr_type_clf": metrics}, name="attr_type_clf_metrics.json")

    lines = [
        "# ATTR type classifier — отчёт",
        "",
        "Ноутбук: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb)  ",
        "Silver: [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) / "
        "[`attr_type_silver.md`](./attr_type_silver.md)  ",
        f"Лучшая sklearn: **`{best_name}`** → `models/attr_type_clf.joblib`",
        "",
        "## Дизайн",
        "",
        "- Char/word n-grams **только по `span_text`**.",
        "- Контекст: `brand`+`category` и/или `query_masked_all_attr`.",
        "- `UNKNOWN` = редкие типы + `other` (+ modelish rule на инференсе).",
        "- Unit-aug из silver (`is_aug`).",
        "",
        "## Сводка моделей (val)",
        "",
        "| model | acc | f1_macro | f1_micro | multi_f1_macro | f1_UNKNOWN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        mm = "—" if pd.isna(r["multi_f1_macro"]) else f"{r['multi_f1_macro']:.3f}"
        star = "**" if r["model"] == best_name else ""
        lines.append(
            f"| `{r['model']}` | {r['accuracy']:.3f} | {star}{r['f1_macro']:.3f}{star} | "
            f"{r['f1_micro']:.3f} | {mm} | {r['f1_UNKNOWN']:.3f} |"
        )
    lines += [
        "",
        "`multi_f1_macro` — строки с ≥2 ATTR в запросе.",
        "",
        "## Per-class F1 (best)",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in per_class.iterrows():
        lines.append(
            f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "## Вывод",
        "",
        f"- Лучшая модель: `{best_name}` (macro F1 **{results[best_name]['f1_macro']:.3f}**).",
        f"- Markov baseline: macro **{results['markov_lookup']['f1_macro']:.3f}**.",
        "- Слабые классы — в per-class таблице (часто `dimensions` / хвост).",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE best={best_name} macro={results[best_name]['f1_macro']:.4f} → {REPORT}")


if __name__ == "__main__":
    main()
