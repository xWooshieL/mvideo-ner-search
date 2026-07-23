"""Production ATTR-type clf: relabel silver, retrain, calibrate, sanity, report."""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data_utils import (
    DARK_SLATE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    save_stats,
    resolve_silver,
    save_silver_parquet,
    ATTR_TYPE_DIR,
)
from src.ner.attr_type_clf import Col, LABEL_UNKNOWN, looks_like_model, predict_attr_type
from src.ner.labeling import _guess_attr_type
from src.ner.markov_typer import MarkovAttrTyper, _norm_token

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
VAL_SIZE = 0.2
MIN_CLASS = 15
UNKNOWN_SHARE_CAP = 0.18
OUT = ATTR_TYPE_DIR
FIG = FIGURES_DIR / "attr_type"
REPORT = ROOT / "notebooks" / "markov_typer" / "attr_type_prod_report.md"
SILVER_RAW = resolve_silver("attr_type", "attr_type_silver_raw.parquet")
SILVER_ALL = resolve_silver("attr_type", "attr_type_silver.parquet")


def log(msg: str) -> None:
    print(msg, flush=True)


def stratified_split(df, label_col, test_size, seed):
    y = df[label_col].astype(str)
    vc = y.value_counts()
    rare1 = set(vc[vc < 2].index)
    ok = df[~y.isin(rare1)]
    hold = df[y.isin(rare1)]
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
                        loss="log_loss", alpha=1e-5, max_iter=50, random_state=seed, tol=1e-3
                    ),
                ),
            ]
        ),
    }


def calibrate_threshold(pipe, val_df, yva, classes) -> tuple[float, pd.DataFrame]:
    """Pick min_confidence on val; floor at 0.55 for prod safety."""
    proba = pipe.predict_proba(val_df)
    classes_ = list(pipe.classes_)
    raw = np.array([classes_[i] for i in proba.argmax(axis=1)])
    conf = proba.max(axis=1)

    rows = []
    for tau in np.round(np.linspace(0.35, 0.90, 12), 2):
        pred = np.where(conf >= tau, raw, LABEL_UNKNOWN)
        macro = float(f1_score(yva, pred, average="macro", zero_division=0))
        acc = float(accuracy_score(yva, pred))
        cov = float((conf >= tau).mean())
        rows.append({"tau": float(tau), "f1_macro": macro, "accuracy": acc, "coverage": cov})
    cal = pd.DataFrame(rows)
    # prod floor: don't go below 0.55 even if val likes lower
    cal_safe = cal[cal["tau"] >= 0.55]
    if len(cal_safe):
        cal_safe = cal_safe.sort_values(
            ["f1_macro", "coverage"], ascending=[False, False]
        ).reset_index(drop=True)
        best_tau = float(cal_safe.iloc[0]["tau"])
    else:
        best_tau = 0.55
    return best_tau, cal


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    assert SILVER_RAW.exists(), SILVER_RAW
    raw = pd.read_parquet(SILVER_RAW)
    log(f"silver_raw={len(raw)}")

    # --- relabel with fixed teacher ---
    before = raw["y"].value_counts()
    raw = raw.copy()
    raw["y_old"] = raw["y"].astype(str)
    raw["y"] = raw["span_text"].map(_guess_attr_type)
    changed = raw[raw["y"] != raw["y_old"]]
    log(f"relabeled spans: {len(changed)} ({len(changed)/len(raw):.1%})")
    if len(changed):
        log(
            changed.groupby(["y_old", "y"])
            .size()
            .sort_values(ascending=False)
            .head(15)
            .to_string()
        )

    # teacher checks (labeling.py only)
    for s in ["16 г", "16 гб", "5 g", "256 g", "1920x1080", "4k", "150 грамм", "2 кг"]:
        log(f"  teacher {s!r:12} -> {_guess_attr_type(s)}")

    # no unit-aug / synthetic rows — types only from labeling.py
    df = raw
    log(f"teacher type counts:\n{df['y'].value_counts().to_string()}")

    # rare / other → UNKNOWN (train hygiene)
    raw_vc = df["y"].value_counts()
    rare = set(raw_vc[raw_vc < MIN_CLASS].index) | {"other"}
    log(f"rare->UNKNOWN: {sorted(rare)}")
    df = df.copy()
    df["y"] = df["y"].where(~df["y"].isin(rare), LABEL_UNKNOWN)
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

    save_silver_parquet(df, "attr_type", "attr_type_silver_prod.parquet")
    train_df, val_df = stratified_split(df, "y", VAL_SIZE, SEED)
    save_silver_parquet(train_df, "attr_type", "attr_type_train_prod.parquet")
    save_silver_parquet(val_df, "attr_type", "attr_type_val_prod.parquet")
    save_silver_parquet(train_df, "attr_type", "attr_type_train.parquet")
    save_silver_parquet(val_df, "attr_type", "attr_type_val.parquet")
    log(f"train={len(train_df)} val={len(val_df)} classes={train_df.y.nunique()}")
    log(train_df.y.value_counts().to_string())

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
        else:
            pack["multi_f1_macro"] = None
        pack["f1_UNKNOWN"] = float(pack["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0))
        results[name] = pack
        fitted[name] = pipe
        joblib.dump(pipe, OUT / f"prod__{name}.joblib")
        log(
            f"  acc={pack['accuracy']:.3f} macro={pack['f1_macro']:.3f} "
            f"multi={pack['multi_f1_macro']} unk={pack['f1_UNKNOWN']:.3f}"
        )

    # markov baseline
    markov = MarkovAttrTyper()
    _big, _unit, _trans = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
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
    m_pred = np.array(
        [
            LABEL_UNKNOWN if (lab := markov.predict(s.split())[0]) == "unknown" else lab
            for s in val_df["span_text"].astype(str)
        ]
    )
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
        "report": classification_report(
            yva, m_pred, labels=classes, output_dict=True, zero_division=0
        ),
        "pred": m_pred,
    }
    results["markov_lookup"]["f1_UNKNOWN"] = float(
        results["markov_lookup"]["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0)
    )

    summary = (
        pd.DataFrame(
            [
                {
                    "model": n,
                    "accuracy": v["accuracy"],
                    "f1_macro": v["f1_macro"],
                    "f1_micro": v["f1_micro"],
                    "multi_f1_macro": v.get("multi_f1_macro"),
                    "f1_UNKNOWN": v.get("f1_UNKNOWN"),
                }
                for n, v in results.items()
            ]
        )
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(OUT / "prod_models_summary.csv", index=False)
    log(summary.to_string(index=False))

    best_name = summary.loc[summary["model"] != "markov_lookup"].iloc[0]["model"]
    pipe_best = fitted[best_name]
    tau, cal = calibrate_threshold(pipe_best, val_df, yva, classes)
    cal.to_csv(OUT / "prod_threshold_curve.csv", index=False)
    log(f"BEST={best_name} calibrated tau={tau}")

    # apply reject
    proba = pipe_best.predict_proba(val_df)
    raw_pred = pipe_best.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    pred_rej = np.where(conf >= tau, raw_pred, LABEL_UNKNOWN)
    rej_macro = float(f1_score(yva, pred_rej, average="macro", zero_division=0))
    rej_acc = float(accuracy_score(yva, pred_rej))
    log(f"with reject@{tau}: acc={rej_acc:.3f} macro={rej_macro:.3f} coverage={(conf>=tau).mean():.1%}")

    joblib.dump(pipe_best, MODELS / "attr_type_clf.joblib")
    joblib.dump(pipe_best, MODELS / f"attr_type_clf__{best_name}.joblib")
    policy = {
        "model": best_name,
        "model_path": "models/attr_type_clf.joblib",
        "min_confidence": tau,
        "reject_label": LABEL_UNKNOWN,
        "features": {
            "span_text": "TF-IDF char/word n-grams ONLY on the ATTR span",
            "context_text": "brand + category string (separate TF-IDF)",
            "query_masked": "query with all ATTR spans replaced by <ATTR>",
        },
        "teacher_fixes": [
            "bare г/g with RAM-like numbers → memory_storage",
            "weight no longer matches bare г/g",
            "resolution_exact before dimensions",
            "unit aug includes гб→г and 4k aliases",
        ],
        "val_metrics_raw": results[best_name]["best_raw"]
        if "best_raw" in results[best_name]
        else {
            "accuracy": results[best_name]["accuracy"],
            "f1_macro": results[best_name]["f1_macro"],
        },
        "val_metrics_with_reject": {
            "accuracy": rej_acc,
            "f1_macro": rej_macro,
            "coverage": float((conf >= tau).mean()),
            "tau": tau,
        },
    }
    # fix policy metrics
    policy["val_metrics_raw"] = {
        "accuracy": results[best_name]["accuracy"],
        "f1_macro": results[best_name]["f1_macro"],
        "multi_f1_macro": results[best_name]["multi_f1_macro"],
        "f1_UNKNOWN": results[best_name]["f1_UNKNOWN"],
    }
    (OUT / "inference_policy.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].barh(summary["model"][::-1], summary["f1_macro"][::-1], color=MVIDEO_RED)
    axes[0].set_xlabel("f1_macro")
    axes[0].set_title("Prod models (val)")
    axes[1].plot(cal["tau"], cal["f1_macro"], "o-", color=MVIDEO_RED, label="f1_macro")
    axes[1].plot(cal["tau"], cal["coverage"], "s--", color=DARK_SLATE, label="coverage")
    axes[1].axvline(tau, color="gray", ls=":", label=f"tau={tau}")
    axes[1].set_xlabel("min_confidence")
    axes[1].set_title("Reject threshold curve")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "prod_01_models_threshold.png", dpi=130, bbox_inches="tight")
    plt.close()

    # class support
    fig, ax = plt.subplots(figsize=(8, 4))
    vc_plot = train_df["y"].value_counts()
    ax.barh(vc_plot.index.astype(str)[::-1], vc_plot.values[::-1], color=DARK_SLATE)
    ax.set_title("Train class support (prod relabel+aug)")
    ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(FIG / "prod_02_class_support.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --- 10 sanity cases ---
    sanity = [
        {"span": "16 г", "brand": "asus", "category": "ноутбук", "expect": "memory_storage"},
        {"span": "16 гб", "brand": "asus", "category": "ноутбук", "expect": "memory_storage"},
        {"span": "256 g", "brand": "samsung", "category": "смартфон", "expect": "memory_storage"},
        {"span": "5 g", "brand": "samsung", "category": "смартфон", "expect": "UNKNOWN"},  # ambiguous / not RAM size list as 5
        {"span": "2 кг", "brand": "bosch", "category": "пылесос", "expect": "weight"},
        {"span": "150 грамм", "brand": "", "category": "весы", "expect": "weight"},
        {"span": "1920x1080", "brand": "xiaomi", "category": "монитор", "expect": "resolution_exact"},
        {"span": "4k", "brand": "lg", "category": "телевизор", "expect": "resolution_standard"},
        {"span": "15.6 дюйм", "brand": "asus", "category": "ноутбук", "expect": "size"},
        {"span": "g pro", "brand": "logitech", "category": "наушники", "expect": "UNKNOWN"},
    ]
    # note: 5 alone + g is NOT in RAM list {8,16,...} — teacher → other → UNKNOWN. Good for prod reject of 5G noise.

    sanity_rows = []
    for c in sanity:
        masked = f"{c['category']} {c['brand']} <ATTR>".strip()
        det = predict_attr_type(
            c["span"],
            brand=c["brand"],
            category=c["category"],
            query_masked=masked,
            model_path=MODELS / "attr_type_clf.joblib",
            min_confidence=tau,
            return_details=True,
        )
        assert isinstance(det, dict)
        ok = det["label"] == c["expect"]
        sanity_rows.append(
            {
                "span": c["span"],
                "brand": c["brand"],
                "category": c["category"],
                "expect": c["expect"],
                "pred": det["label"],
                "raw_pred": det["raw_pred"],
                "confidence": round(det["confidence"], 3),
                "reason": det["reason"],
                "ok": ok,
                "teacher": _guess_attr_type(c["span"]),
            }
        )
        log(
            f"SANITY {c['span']!r:12} expect={c['expect']:20} "
            f"pred={det['label']:20} conf={det['confidence']:.2f} "
            f"{'OK' if ok else 'FAIL'} teacher={_guess_attr_type(c['span'])}"
        )

    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv(OUT / "prod_sanity_10.csv", index=False)
    log(f"sanity pass {sanity_df.ok.sum()}/{len(sanity_df)}")

    # --- feature explainer example ---
    example_span = "16 г"
    example_brand = "asus"
    example_cat = "ноутбук"
    example_masked = "ноутбук asus <ATTR>"
    feat_doc = {
        "example_query": "ноутбук asus 16 г",
        "entities": {
            "CATEGORY": "ноутбук",
            "BRAND": "asus",
            "ATTR_span": "16 г",
        },
        "feature_columns": {
            "span_text": {
                "value": example_span,
                "how": "TF-IDF char_wb(2–5) and/or word n-grams ONLY on this string",
                "why": "тип единицы живёт в самом span; не смешиваем 16 г с другими ATTR",
                "example_char_ngrams": [" 1", "16", "6 ", " г", "г "],
            },
            "context_text": {
                "value": f"{example_brand} {example_cat}",
                "how": "отдельный TF-IDF char по brand+category",
                "why": "подсказка домена (ноутбук → память вероятнее веса)",
            },
            "query_masked": {
                "value": example_masked,
                "how": "word TF-IDF; все ATTR заменены на <ATTR>",
                "why": "левый/правый контекст без чужих единиц (гб/дюйм не протекают)",
            },
        },
        "prediction": predict_attr_type(
            example_span,
            brand=example_brand,
            category=example_cat,
            query_masked=example_masked,
            min_confidence=tau,
            return_details=True,
        ),
    }
    (OUT / "prod_feature_example.json").write_text(
        json.dumps(feat_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # per-class for best
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
    per_class.to_csv(OUT / "prod_per_class_f1.csv", index=False)

    # --- gold mapped agree (read-only jsonl) ---
    gold_path = ROOT / "data" / "gold" / "bio_liza.jsonl"
    gold_agree_line = "gold mapped agree: (no file)"
    if gold_path.exists():
        from src.ner.labeling import gold_subtype_to_canon

        g_ok = g_n = 0
        for line in gold_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            toks = r["query"].split()
            tags = r["tags"]
            if len(toks) != len(tags):
                continue
            subtypes = {int(k): v for k, v in (r.get("subtypes") or {}).items()}
            i = 0
            while i < len(tags):
                if tags[i].startswith("B-") and tags[i][2:] == "ATTR":
                    j = i + 1
                    while j < len(tags) and tags[j] == "I-ATTR":
                        j += 1
                    span = " ".join(toks[i:j])
                    st = subtypes.get(i)
                    if st is None:
                        for k in range(i, j):
                            if k in subtypes:
                                st = subtypes[k]
                                break
                    if st:
                        g_n += 1
                        if gold_subtype_to_canon(st) == _guess_attr_type(span):
                            g_ok += 1
                    i = j
                else:
                    i += 1
        gold_agree_line = (
            f"gold mapped agree (teacher vs app→canon): **{g_ok}/{g_n}** "
            f"({g_ok / max(1, g_n):.1%}) — jsonl не меняли"
        )

    # --- markdown report ---
    lines = [
        "# ATTR type classifier — prod report",
        "",
        f"Модель: **`{best_name}`** → `models/attr_type_clf.joblib`  ",
        f"Policy: `artifacts/silver/attr_type/inference_policy.json` (τ=`{tau}`)  ",
        f"Sanity: **{int(sanity_df.ok.sum())}/{len(sanity_df)}** кейсов  ",
        gold_agree_line,
        "",
        "Канон: unit-типы + `color` + **`type`/`purpose`** (см. `data/gold/ATTR_SUBTYPES.md`).",
        "",
        "## Зачем классификатор, а не только regex",
        "",
        "NER отдаёт общий `ATTR`. Тип нужен для фактов/поиска. Regex-учитель даёт silver,",
        "clf обобщает опечатки/усечения (`16 г`, `4к`) и умеет **reject** при низкой уверенности.",
        "",
        "## Что такое фичи (пример: `ноутбук asus 16 г`)",
        "",
        "| фича | значение | что кодируем |",
        "|---|---|---|",
        f"| `span_text` | `{example_span}` | n-grams **только** span → сигнал единицы |",
        f"| `context_text` | `{example_brand} {example_cat}` | бренд+категория отдельно |",
        f"| `query_masked` | `{example_masked}` | запрос без чужих ATTR (`<ATTR>`) |",
        "",
        "Важно: **не** кормим TF-IDF строкой `16 г 15.6 дюйм` — типы смешаются.",
        "",
        "```text",
        "query:  ноутбук asus 16 г",
        "         CATEGORY BRAND ATTR",
        "",
        "X1 = TFIDF(\"16 г\")              # char: \"16\", \"6 \", \" г\", ...",
        "X2 = TFIDF(\"asus ноутбук\")     # контекст",
        "X3 = TFIDF(\"ноутбук asus <ATTR>\")",
        "y  = memory_storage",
        "```",
        "",
        "## Исправления учителя (гипотезы)",
        "",
        "| было | стало | зачем |",
        "|---|---|---|",
        "| `16 г` → weight | → **memory_storage** (RAM-like + `г`/`g`) | усечение «гб» в поиске |",
        "| `5 g` / `256 g` → weight | 5g → other/UNKNOWN; 256 g → memory | шум 5G / truncated gb |",
        "| `1920x1080` → dimensions | → **resolution_exact** | порядок regex |",
        "| weight = `г\\|g\\|грамм` | weight = кг/грамм | убрали омонимию |",
        "| нет type/purpose | **type** / **purpose** lexical | sync с app gold |",
        "",
        "## Метрики val (после relabel)",
        "",
        "| model | acc | f1_macro | multi_f1_macro | f1_UNKNOWN |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        mm = "—" if pd.isna(r["multi_f1_macro"]) else f"{r['multi_f1_macro']:.3f}"
        star = "**" if r["model"] == best_name else ""
        lines.append(
            f"| `{r['model']}` | {r['accuracy']:.3f} | {star}{r['f1_macro']:.3f}{star} | "
            f"{mm} | {r['f1_UNKNOWN']:.3f} |"
        )
    lines += [
        "",
        f"С reject τ={tau}: acc **{rej_acc:.3f}**, macro **{rej_macro:.3f}**, "
        f"coverage **{(conf>=tau).mean():.1%}**.",
        "",
        "![models](../../figures/attr_type/prod_01_models_threshold.png)",
        "",
        "## Sanity 10 (ручной демо-набор)",
        "",
        "| span | expect | pred | conf | ok | teacher |",
        "|---|---|---|---:|:---:|---|",
    ]
    for _, r in sanity_df.iterrows():
        lines.append(
            f"| `{r['span']}` | {r['expect']} | {r['pred']} | {r['confidence']:.2f} | "
            f"{'OK' if r['ok'] else 'FAIL'} | {r['teacher']} |"
        )
    lines += [
        "",
        "## Per-class F1 (best, без reject)",
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
        "## Применимость в прод / демку",
        "",
        "| Можно | Нельзя / осторожно |",
        "|---|---|",
        "| Типичные единицы с полной формой (`16 гб`, `2 кг`, `55 дюйм`) | Голые токены `v`, `l`, `tb` без числа |",
        "| Усечённая память RAM-like (`16 г`, `256 g`) | Редкие типы (ом, мпикс) → UNKNOWN |",
        "| Reject при conf < τ | Верить val 0.97 как gold — нет, ждём gold-set |",
        "| Fallback после NER ATTR-span | Заменять regex-учитель целиком без мониторинга |",
        "",
        "## Обоснованность",
        "",
        "1. **Задача узкая**: тип span, не полный NER — TF-IDF n-gram достаточен.",
        "2. **Фичи изолированы** — нет протекания единиц между ATTR.",
        "3. **Учитель починен** под реальные запросы М.Видео (усечения, 5G-шум).",
        "4. **Reject** снижает риск уверенно-неправильных ответов в демке.",
        "5. **Ограничение**: silver всё ещё regex-circular; final sign-off — на gold.",
        "",
        "## Как звать в демо",
        "",
        "```python",
        "from src.ner.attr_type_clf import predict_attr_type",
        "predict_attr_type('16 г', brand='asus', category='ноутбук',",
        "                  query_masked='ноутбук asus <ATTR>', return_details=True)",
        "```",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "best_model": best_name,
        "tau": tau,
        "summary": summary.to_dict(orient="records"),
        "reject": policy["val_metrics_with_reject"],
        "sanity_pass": int(sanity_df.ok.sum()),
        "sanity_total": len(sanity_df),
        "relabeled": int(len(changed)),
    }
    (OUT / "prod_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_stats({"attr_type_prod": metrics}, METRICS_DIR / "attr_type_prod_metrics.json")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
