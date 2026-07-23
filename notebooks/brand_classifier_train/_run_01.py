"""Train brand classifiers (smoke / CI) and write brand_classifier.md."""
from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    MODELS,
    MUTED,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    save_stats,
)

LOG = Path(__file__).resolve().parent / "_run_01_log.txt"
LABEL_NO_BRAND = "NO_BRAND"
LABEL_UNKNOWN = "UNKNOWN"
SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def make_pipelines(seed: int = SEED) -> dict[str, Pipeline]:
    # компактнее фичи / меньше итераций — иначе Windows-smoke рвёт долгий saga
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50_000)
    char5 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=60_000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25_000)
    union = FeatureUnion([("word", word), ("char", char)])
    lr_kw = dict(max_iter=200, solver="lbfgs", random_state=seed)
    return {
        "logreg_char": Pipeline(
            [
                ("tfidf", char),
                ("clf", LogisticRegression(**lr_kw)),
            ]
        ),
        "logreg_char_bal": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50_000
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(**lr_kw, class_weight="balanced"),
                ),
            ]
        ),
        "logreg_wordchar": Pipeline(
            [
                ("tfidf", union),
                ("clf", LogisticRegression(**lr_kw)),
            ]
        ),
        "sgd_char": Pipeline(
            [
                ("tfidf", char5),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=1e-5,
                        max_iter=25,
                        random_state=seed,
                        tol=1e-3,
                    ),
                ),
            ]
        ),
    }


def proba_matrix(pipe: Pipeline, X):
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        return pipe.predict_proba(X)
    return None


def apply_reject(pred, proba, class_names, tau_accept, tau_margin, tau_no_brand, tau_unknown):
    out = pred.astype(object).copy()
    if proba is None:
        return out
    for i, p in enumerate(pred):
        row = proba[i]
        order = np.argsort(row)[::-1]
        p1 = float(row[order[0]])
        p2 = float(row[order[1]]) if len(order) > 1 else 0.0
        margin = p1 - p2
        if p == LABEL_NO_BRAND:
            out[i] = LABEL_NO_BRAND
            continue
        if p == LABEL_UNKNOWN:
            out[i] = LABEL_UNKNOWN if p1 >= tau_unknown else LABEL_NO_BRAND
            continue
        if p1 >= tau_accept and margin >= tau_margin:
            out[i] = p
        else:
            out[i] = LABEL_NO_BRAND
    return out


def eval_pack(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def subset_metrics(val_df, y_pred):
    yt = val_df["brand"].astype(str).values
    mask_co = val_df["is_category_only"].astype(bool).values
    if mask_co.any():
        pred_co = y_pred[mask_co]
        false_brand = float(np.mean([(p not in {LABEL_NO_BRAND, LABEL_UNKNOWN}) for p in pred_co]))
    else:
        false_brand = None

    def _f1(label):
        return float(f1_score(yt == label, y_pred == label, zero_division=0))

    return {
        "n_category_only_val": int(mask_co.sum()),
        "false_brand_rate_category_only": false_brand,
        "f1_NO_BRAND": _f1(LABEL_NO_BRAND),
        "f1_UNKNOWN": _f1(LABEL_UNKNOWN),
        "support_NO_BRAND": int((yt == LABEL_NO_BRAND).sum()),
        "support_UNKNOWN": int((yt == LABEL_UNKNOWN).sum()),
    }


def write_report(md_path: Path, metrics: dict, summary: pd.DataFrame, best_tbl: pd.DataFrame) -> None:
    best = metrics["best_model"]
    br = metrics["best_raw"]
    bs = metrics["best_subset_raw"]
    lines = [
        "# Brand classifier — отчёт обучения",
        "",
        f"Ноутбук: [`01_classifier_train.ipynb`](./01_classifier_train.ipynb)  ",
        f"Silver: `artifacts/brand_clf/silver_brand_*.parquet` (прогон `03`)  ",
        f"Лучшая модель: **`{best}`** → `models/brand_clf.joblib`",
        "",
        "## 1. Постановка",
        "",
        "Мультикласс query→brand на silver с классами top-K + `NO_BRAND` + `UNKNOWN` (OOD).",
        "Clf — fallback после NER/alias; на category-only не должен навязывать Indesit.",
        "",
        "См. также [`../preprocessing/silver_clf_readme.md`](../preprocessing/silver_clf_readme.md).",
        "",
        "## 2. Данные",
        "",
        f"- train: **{metrics['n_train']}**, val: **{metrics['n_val']}**, classes: **{metrics['n_classes']}**",
        f"- NO_BRAND share (silver stats): {metrics.get('silver_stats_ref', {}).get('no_brand_share')}",
        f"- UNKNOWN share: {metrics.get('silver_stats_ref', {}).get('unknown_share')}",
        "",
        "### Inference thresholds",
        "",
        "```json",
        json.dumps(metrics["thresholds"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. Сравнение моделей",
        "",
        "| model | acc | f1_macro | f1_micro | f1_weighted | f1_NO_BRAND | f1_UNKNOWN | false_brand@cat |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        fb = r["false_brand_cat"]
        fb_s = "—" if pd.isna(fb) else f"{fb:.3f}"
        lines.append(
            f"| `{r['model']}` | {r['raw_accuracy']:.3f} | {r['raw_f1_macro']:.3f} | "
            f"{r['raw_f1_micro']:.3f} | {r['raw_f1_weighted']:.3f} | {r['f1_NO_BRAND']:.3f} | "
            f"{r['f1_UNKNOWN']:.3f} | {fb_s} |"
        )
    lines += [
        "",
        f"### Лучшая (`{best}`) raw",
        "",
        f"- accuracy **{br['accuracy']:.3f}**",
        f"- F1-macro **{br['f1_macro']:.3f}**, micro **{br['f1_micro']:.3f}**, weighted **{br['f1_weighted']:.3f}**",
        f"- F1(`NO_BRAND`) **{bs['f1_NO_BRAND']:.3f}**, F1(`UNKNOWN`) **{bs['f1_UNKNOWN']:.3f}**",
        f"- false brand rate на category-only val: **{bs['false_brand_rate_category_only']}** "
        f"(n={bs['n_category_only_val']})",
        "",
        "## 4. Per-class F1 (best model)",
        "",
        "### Топ по F1",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    top = best_tbl.sort_values("f1", ascending=False).head(15)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {int(row['support'])} |"
        )
    lines += [
        "",
        "### Хвост (худшие F1)",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    worst = best_tbl.sort_values("f1").head(15)
    for _, row in worst.iterrows():
        lines.append(
            f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {int(row['support'])} |"
        )
    lines += [
        "",
        "Полные CSV: `artifacts/brand_clf/train_runs/per_class_f1__*.csv`.",
        "",
        "## 5. Reject-policy",
        "",
        "После Softmax применяем τ из `inference_policy.json` (reject → трактуем как `NO_BRAND`/null).",
        "",
        "| model | raw f1_macro | reject f1_macro | raw acc | reject acc |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['model']}` | {r['raw_f1_macro']:.3f} | {r['rej_f1_macro']:.3f} | "
            f"{r['raw_accuracy']:.3f} | {r['rej_accuracy']:.3f} |"
        )
    lines += [
        "",
        "## 6. Интерпретация",
        "",
        "- **macro-F1** важнее accuracy: классы несбалансированы (`NO_BRAND` / Samsung / хвост).",
        "- Высокий **F1(NO_BRAND)** + низкий **false_brand@cat** — модель не тащит Indesit на `холодильник`.",
        "- **UNKNOWN** обычно сложнее: OOD-бренды орфографически разнообразны; F1 ниже — ожидаемо.",
        "- Классы с F1≈0 и малым support — кандидаты в `UNKNOWN` или на ↑ семпла в `03`.",
        "",
        "## 7. Нужно ли что-то перезапускать?",
        "",
        "| Что | Когда |",
        "|---|---|",
        "| `03_brand_data_preprocessing.ipynb` | сменили пороги silver / top-K / NO_BRAND logic |",
        "| этот ноутбук / `_run_01.py` | после обновления silver parquet |",
        "| gold-разметка | для честного test и калибровки τ (пока нет) |",
        "",
        "Перезапускать preprocess NER (`01`/`02`) **не нужно** — brand-clf живёт на `query_norm` + кликах.",
        "",
        "## 8. Артефакты",
        "",
        "| Путь |",
        "|---|",
        "| `models/brand_clf.joblib` |",
        "| `artifacts/brand_clf/train_runs/metrics.json` |",
        "| `artifacts/brand_clf/train_runs/models_summary.csv` |",
        "| `figures/brand_clf/01_confusion_best.png` |",
        "| `figures/brand_clf/02_models_compare.png` |",
        "",
        "---",
        f"*Сгенерировано `_run_01.py`, best=`{best}`.*",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    warnings.filterwarnings("ignore", category=FutureWarning)
    ensure_dirs()
    apply_plot_style()

    DATA = ARTIFACTS_DIR / "brand_clf"
    FIG = FIGURES_DIR / "brand_clf"
    OUT = DATA / "train_runs"
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    from src.data_utils import resolve_silver

    train = pd.read_parquet(resolve_silver("brand_clf", "silver_brand_train.parquet"))
    val = pd.read_parquet(resolve_silver("brand_clf", "silver_brand_val.parquet"))
    policy = json.loads((DATA / "inference_policy.json").read_text(encoding="utf-8"))
    stats_silver = json.loads(
        resolve_silver("brand_clf", "silver_brand_stats.json").read_text(encoding="utf-8")
    )
    TAU_ACCEPT = float(policy["thresholds"]["TAU_ACCEPT"])
    TAU_MARGIN = float(policy["thresholds"]["TAU_MARGIN"])
    TAU_NO_BRAND = float(policy["thresholds"]["TAU_NO_BRAND"])
    TAU_UNKNOWN = float(policy["thresholds"]["TAU_UNKNOWN"])

    Xtr = train["query_norm"].astype(str).values
    ytr = train["brand"].astype(str).values
    wtr = train["sample_weight"].astype(float).values
    Xva = val["query_norm"].astype(str).values
    yva = val["brand"].astype(str).values
    classes = sorted(set(ytr) | set(yva), key=lambda x: (x in {LABEL_NO_BRAND, LABEL_UNKNOWN}, x))
    log(f"train={len(train)} val={len(val)} classes={len(classes)}")

    results = {}
    per_class_tables = {}
    for name, pipe in make_pipelines().items():
        log(f"train {name}")
        if name != "logreg_char_bal":
            pipe.fit(Xtr, ytr, clf__sample_weight=wtr)
        else:
            pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        proba = proba_matrix(pipe, Xva)
        class_names = list(pipe.named_steps["clf"].classes_)
        pred_rej = apply_reject(
            pred, proba, class_names, TAU_ACCEPT, TAU_MARGIN, TAU_NO_BRAND, TAU_UNKNOWN
        )
        pack = eval_pack(yva, pred, classes)
        pack_rej = eval_pack(yva, pred_rej, classes)
        sub = subset_metrics(val, pred)
        sub_rej = subset_metrics(val, pred_rej)
        results[name] = {
            "raw": {k: pack[k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
            "reject": {k: pack_rej[k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
            "subset_raw": sub,
            "subset_reject": sub_rej,
        }
        rows = []
        for lab in classes:
            if lab not in pack["report"]:
                continue
            r = pack["report"][lab]
            rows.append(
                {
                    "class": lab,
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1-score"],
                    "support": int(r["support"]),
                }
            )
        tbl = pd.DataFrame(rows).sort_values("f1")
        per_class_tables[name] = tbl
        results[name]["per_class"] = tbl.to_dict(orient="records")
        joblib.dump(pipe, OUT / f"{name}.joblib")
        tbl.to_csv(OUT / f"per_class_f1__{name}.csv", index=False)
        log(
            f"  acc={pack['accuracy']:.3f} macro={pack['f1_macro']:.3f} "
            f"NO_BRAND={sub['f1_NO_BRAND']:.3f} UNKNOWN={sub['f1_UNKNOWN']:.3f} "
            f"fb={sub['false_brand_rate_category_only']}"
        )

    summary = pd.DataFrame(
        [
            {
                "model": n,
                **{f"raw_{k}": v["raw"][k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
                **{f"rej_{k}": v["reject"][k] for k in ["accuracy", "f1_macro", "f1_micro"]},
                "f1_NO_BRAND": v["subset_raw"]["f1_NO_BRAND"],
                "f1_UNKNOWN": v["subset_raw"]["f1_UNKNOWN"],
                "false_brand_cat": v["subset_raw"]["false_brand_rate_category_only"],
            }
            for n, v in results.items()
        ]
    ).sort_values("raw_f1_macro", ascending=False)
    summary.to_csv(OUT / "models_summary.csv", index=False)
    best_name = summary.iloc[0]["model"]
    best_tbl = per_class_tables[best_name].copy()
    best_tbl.to_csv(OUT / "per_class_f1__BEST.csv", index=False)

    best_pipe = joblib.load(OUT / f"{best_name}.joblib")
    joblib.dump(best_pipe, MODELS / "brand_clf.joblib")
    joblib.dump(best_pipe, MODELS / f"brand_clf__{best_name}.joblib")

    pred_best = best_pipe.predict(Xva)
    top_show = list(pd.Series(yva).value_counts().head(12).index)
    for s in (LABEL_NO_BRAND, LABEL_UNKNOWN):
        if s not in top_show:
            top_show.append(s)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(yva, pred_best, labels=top_show)
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks(range(len(top_show)))
    ax.set_yticks(range(len(top_show)))
    ax.set_xticklabels(top_show, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_show, fontsize=8)
    ax.set_title(f"Confusion — {best_name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG / "01_confusion_best.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(summary))
    w = 0.35
    axes[0].bar(x - w / 2, summary["raw_f1_macro"], w, label="raw", color=MVIDEO_RED)
    axes[0].bar(x + w / 2, summary["rej_f1_macro"], w, label="reject", color=DARK_SLATE)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["model"], rotation=20, ha="right")
    axes[0].legend()
    axes[0].set_title("F1-macro")
    axes[1].bar(summary["model"], summary["false_brand_cat"].fillna(0), color=MUTED)
    axes[1].set_title("False brand @ category-only")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG / "02_models_compare.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    err = val.copy()
    err["pred"] = pred_best
    err_bad = err[err["brand"] != err["pred"]]
    err_bad.head(500).to_csv(OUT / "errors_sample.csv", index=False)

    metrics_out = {
        "best_model": best_name,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_classes": int(len(classes)),
        "thresholds": policy["thresholds"],
        "summary": summary.to_dict(orient="records"),
        "best_raw": results[best_name]["raw"],
        "best_reject": results[best_name]["reject"],
        "best_subset_raw": results[best_name]["subset_raw"],
        "best_subset_reject": results[best_name]["subset_reject"],
        "worst_classes": best_tbl.sort_values("f1").head(15).to_dict(orient="records"),
        "best_classes": best_tbl.sort_values("f1", ascending=False).head(15).to_dict(orient="records"),
        "silver_stats_ref": {
            "no_brand_share": stats_silver.get("no_brand_share"),
            "unknown_share": stats_silver.get("unknown_share"),
        },
        "n_errors_val": int(len(err_bad)),
    }
    save_stats(metrics_out, OUT / "metrics.json")
    save_stats(metrics_out, DATA / "train_metrics.json")

    md_path = Path(__file__).resolve().parent / "brand_classifier.md"
    write_report(md_path, metrics_out, summary, best_tbl)
    log(f"BEST={best_name} macro={results[best_name]['raw']['f1_macro']:.4f}")
    log("DONE wrote " + str(md_path))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
