"""Generate notebooks/brand_classifier_train/01_classifier_train.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "01_classifier_train.ipynb"
cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
    )


md(
    """# 01. Brand classifier train

Обучение query→brand на silver из `artifacts/brand_clf/`.

Спец-классы: **`NO_BRAND`**, **`UNKNOWN`** (OOD).  
Договорённости: [`../preprocessing/silver_clf_readme.md`](../preprocessing/silver_clf_readme.md)  
Отчёт: [`brand_classifier.md`](./brand_classifier.md)

Сравниваем несколько sklearn-пайплайнов (TF-IDF + линейные модели), считаем:
- accuracy / F1 macro·micro·weighted  
- **F1 по каждому классу** (таблица)  
- метрики на `NO_BRAND` / `UNKNOWN`  
- false-brand rate на category-only  
- эффект reject-порогов из `inference_policy.json`
"""
)

md("## 0. Setup")

code(
    """%matplotlib inline
import sys
import json
import warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name in {"brand_classifier_train", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "brand_classifier_train" else ROOT.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from src.data_utils import (
    ARTIFACTS_DIR,
    FIGURES_DIR,
    MODELS,
    ensure_dirs,
    apply_plot_style,
    MVIDEO_RED,
    DARK_SLATE,
    MUTED,
    save_stats,
)

warnings.filterwarnings("ignore", category=FutureWarning)
ensure_dirs()
apply_plot_style()

DATA = ARTIFACTS_DIR / "brand_clf"
FIG = FIGURES_DIR / "brand_clf"
OUT = ARTIFACTS_DIR / "brand_clf" / "train_runs"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

LABEL_NO_BRAND = "NO_BRAND"
LABEL_UNKNOWN = "UNKNOWN"
SEED = 42
print("DATA", DATA)
print("FIG", FIG)
"""
)

md("## 1. Загрузка silver + policy")

code(
    """train = pd.read_parquet(DATA / "silver_brand_train.parquet")
val = pd.read_parquet(DATA / "silver_brand_val.parquet")
policy = json.loads((DATA / "inference_policy.json").read_text(encoding="utf-8"))
label_map = json.loads((DATA / "label_map.json").read_text(encoding="utf-8"))
stats_silver = json.loads((DATA / "silver_brand_stats.json").read_text(encoding="utf-8"))

TAU_ACCEPT = float(policy["thresholds"]["TAU_ACCEPT"])
TAU_MARGIN = float(policy["thresholds"]["TAU_MARGIN"])
TAU_NO_BRAND = float(policy["thresholds"]["TAU_NO_BRAND"])
TAU_UNKNOWN = float(policy["thresholds"]["TAU_UNKNOWN"])

print(f"train={len(train):,}  val={len(val):,}  classes={train['brand'].nunique()}")
print("specials in train:", {
    LABEL_NO_BRAND: int((train.brand == LABEL_NO_BRAND).sum()),
    LABEL_UNKNOWN: int((train.brand == LABEL_UNKNOWN).sum()),
})
print("thresholds:", policy["thresholds"])
display(train["brand"].value_counts().head(12).to_frame("train_n"))
display(train.sample(5, random_state=0)[["query_norm", "brand", "label_reason", "sample_weight", "is_category_only"]])
"""
)

md(
    """## 2. X / y / weights

- `X` = `query_norm`  
- `y` = `brand` (включая NO_BRAND / UNKNOWN)  
- `sample_weight` из silver (click-confidence); для specials уже выставлены в `03`
"""
)

code(
    """Xtr = train["query_norm"].astype(str).values
ytr = train["brand"].astype(str).values
wtr = train["sample_weight"].astype(float).values

Xva = val["query_norm"].astype(str).values
yva = val["brand"].astype(str).values

classes = sorted(set(ytr) | set(yva), key=lambda x: (x in {LABEL_NO_BRAND, LABEL_UNKNOWN}, x))
print("n_classes", len(classes))
assert set(yva).issubset(set(ytr)), "val has unseen labels — пересоберите silver split"
"""
)

md(
    """## 3. Модели

| id | Векторизация | Классификатор | Зачем |
|---|---|---|---|
| `logreg_char` | char_wb (2–4) | LogReg + sample_weight | близко к старому `07` |
| `logreg_char_bal` | char_wb (2–4) | LogReg class_weight=balanced | редкие бренды |
| `logreg_wordchar` | word(1–2) + char_wb(2–4) | LogReg + weights | алиасы + опечатки |
| `sgd_char` | char_wb (2–5) | SGD log_loss | быстрый линейный baseline |
"""
)

code(
    """def make_pipelines(seed: int = SEED) -> dict[str, Pipeline]:
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50_000)
    char5 = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=60_000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=25_000)
    union = FeatureUnion([("word", word), ("char", char)])
    lr_kw = dict(max_iter=200, solver="lbfgs", random_state=seed)

    return {
        "logreg_char": Pipeline([
            ("tfidf", char),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "logreg_char_bal": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=50_000)),
            ("clf", LogisticRegression(**lr_kw, class_weight="balanced")),
        ]),
        "logreg_wordchar": Pipeline([
            ("tfidf", union),
            ("clf", LogisticRegression(**lr_kw)),
        ]),
        "sgd_char": Pipeline([
            ("tfidf", char5),
            ("clf", SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-5,
                max_iter=25, random_state=seed, tol=1e-3,
            )),
        ]),
    }

PIPELINES = make_pipelines()
print(list(PIPELINES))
"""
)

md("## 4. Обучение + метрики")

code(
    """def proba_matrix(pipe: Pipeline, X) -> np.ndarray | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)
    return None


def apply_reject(pred: np.ndarray, proba: np.ndarray | None, class_names: list[str]) -> np.ndarray:
    \"\"\"Политика inference: низкая уверенность / NO_BRAND / UNKNOWN → служебный REJECT.
    Для сравнения метрик REJECT считаем как NO_BRAND (brand=null).\"\"\"
    out = pred.astype(object).copy()
    if proba is None:
        # без вероятностей: UNKNOWN/NO_BRAND оставляем, иначе без reject
        return out
    name_to_i = {c: i for i, c in enumerate(class_names)}
    for i, p in enumerate(pred):
        row = proba[i]
        order = np.argsort(row)[::-1]
        p1 = float(row[order[0]])
        p2 = float(row[order[1]]) if len(order) > 1 else 0.0
        margin = p1 - p2
        if p == LABEL_NO_BRAND:
            out[i] = LABEL_NO_BRAND if p1 >= TAU_NO_BRAND else LABEL_NO_BRAND
            continue
        if p == LABEL_UNKNOWN:
            out[i] = LABEL_UNKNOWN if p1 >= TAU_UNKNOWN else LABEL_NO_BRAND
            continue
        if p1 >= TAU_ACCEPT and margin >= TAU_MARGIN:
            out[i] = p
        else:
            out[i] = LABEL_NO_BRAND  # reject → null
    return out


def eval_pack(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def subset_metrics(val_df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    yt = val_df["brand"].astype(str).values
    # false brand: category-only, предсказали конкретный бренд
    mask_co = val_df["is_category_only"].astype(bool).values
    if mask_co.any():
        pred_co = y_pred[mask_co]
        false_brand = np.mean([
            (p not in {LABEL_NO_BRAND, LABEL_UNKNOWN}) for p in pred_co
        ])
    else:
        false_brand = None
    # NO_BRAND / UNKNOWN F1
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


results = {}
per_class_tables = {}

for name, pipe in PIPELINES.items():
    print(f"\\n=== train {name} ===")
    use_w = name != "logreg_char_bal"  # balanced уже компенсирует частоты
    if use_w:
        pipe.fit(Xtr, ytr, clf__sample_weight=wtr)
    else:
        pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xva)
    proba = proba_matrix(pipe, Xva)
    class_names = list(pipe.named_steps["clf"].classes_)
    pred_rej = apply_reject(pred, proba, class_names)

    pack = eval_pack(yva, pred, classes)
    pack_rej = eval_pack(yva, pred_rej, classes)
    sub = subset_metrics(val, pred)
    sub_rej = subset_metrics(val, pred_rej)

    results[name] = {
        "raw": {k: pack[k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
        "reject": {k: pack_rej[k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
        "subset_raw": sub,
        "subset_reject": sub_rej,
        "n_params_hint": name,
    }
    # per-class table
    rep = pack["report"]
    rows = []
    for lab in classes:
        if lab not in rep:
            continue
        r = rep[lab]
        rows.append({
            "class": lab,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })
    per_class_tables[name] = pd.DataFrame(rows).sort_values("f1")
    results[name]["per_class"] = per_class_tables[name].to_dict(orient="records")

    joblib.dump(pipe, OUT / f"{name}.joblib")
    print(
        f"  acc={pack['accuracy']:.3f}  f1_macro={pack['f1_macro']:.3f}  "
        f"f1_NO_BRAND={sub['f1_NO_BRAND']:.3f}  f1_UNKNOWN={sub['f1_UNKNOWN']:.3f}  "
        f"false_brand@cat={sub['false_brand_rate_category_only']}"
    )

summary = pd.DataFrame([
    {
        "model": n,
        **{f"raw_{k}": v["raw"][k] for k in ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]},
        **{f"rej_{k}": v["reject"][k] for k in ["accuracy", "f1_macro", "f1_micro"]},
        "f1_NO_BRAND": v["subset_raw"]["f1_NO_BRAND"],
        "f1_UNKNOWN": v["subset_raw"]["f1_UNKNOWN"],
        "false_brand_cat": v["subset_raw"]["false_brand_rate_category_only"],
    }
    for n, v in results.items()
]).sort_values("raw_f1_macro", ascending=False)

display(summary.round(4))
summary.to_csv(OUT / "models_summary.csv", index=False)
"""
)

md("## 5. Per-class F1 (лучший по macro-F1)")

code(
    """best_name = summary.iloc[0]["model"]
print("BEST:", best_name)
best_tbl = per_class_tables[best_name].copy()
best_tbl["is_special"] = best_tbl["class"].isin([LABEL_NO_BRAND, LABEL_UNKNOWN])
display(best_tbl.sort_values("f1", ascending=False).head(20).round(3))
print("\\nХудшие классы по F1:")
display(best_tbl.sort_values("f1").head(20).round(3))

# сохранить все per-class
for n, tbl in per_class_tables.items():
    tbl.to_csv(OUT / f"per_class_f1__{n}.csv", index=False)

best_tbl.to_csv(OUT / "per_class_f1__BEST.csv", index=False)
"""
)

md("## 6. Confusion / ошибки на спец-классах")

code(
    """best_pipe = joblib.load(OUT / f"{best_name}.joblib")
pred_best = best_pipe.predict(Xva)
proba_best = proba_matrix(best_pipe, Xva)
pred_best_rej = apply_reject(pred_best, proba_best, list(best_pipe.named_steps["clf"].classes_))

# top brands + specials for CM
top_show = (
    list(pd.Series(yva).value_counts().head(12).index)
)
# ensure specials
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
ax.set_xlabel("pred")
ax.set_ylabel("true")
ax.set_title(f"Confusion (top+specials) — {best_name}")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIG / "01_confusion_best.png", dpi=140, bbox_inches="tight")
plt.show()

# error gallery
err = val.copy()
err["pred"] = pred_best
err["pred_reject"] = pred_best_rej
err_bad = err[err["brand"] != err["pred"]]
print(f"errors: {len(err_bad):,} / {len(err):,} ({len(err_bad)/len(err):.1%})")

print("\\nNO_BRAND → wrongly predicted as brand:")
display(
    err_bad[err_bad["brand"] == LABEL_NO_BRAND]
    .groupby("pred").size().sort_values(ascending=False).head(10).to_frame("n")
)
print("\\nTrue brand → predicted NO_BRAND / UNKNOWN:")
miss = err_bad[err_bad["pred"].isin([LABEL_NO_BRAND, LABEL_UNKNOWN])]
display(miss.groupby(["brand", "pred"]).size().sort_values(ascending=False).head(15).to_frame("n"))

print("\\nsample errors:")
display(err_bad.sample(min(12, len(err_bad)), random_state=1)[
    ["query_norm", "brand", "pred", "pred_reject", "label_reason", "is_category_only"]
])
err_bad.head(500).to_csv(OUT / "errors_sample.csv", index=False)
"""
)

md("## 7. Сравнение raw vs reject-policy")

code(
    """cmp = summary[["model", "raw_accuracy", "raw_f1_macro", "rej_accuracy", "rej_f1_macro",
               "f1_NO_BRAND", "f1_UNKNOWN", "false_brand_cat"]].copy()
display(cmp.round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(len(summary))
w = 0.35
axes[0].bar(x - w/2, summary["raw_f1_macro"], w, label="raw", color=MVIDEO_RED)
axes[0].bar(x + w/2, summary["rej_f1_macro"], w, label="reject", color=DARK_SLATE)
axes[0].set_xticks(x)
axes[0].set_xticklabels(summary["model"], rotation=20, ha="right")
axes[0].set_title("F1-macro: raw vs reject")
axes[0].legend()

axes[1].bar(summary["model"], summary["false_brand_cat"].fillna(0), color=MUTED)
axes[1].set_title("False brand rate @ category-only (raw)")
axes[1].tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(FIG / "02_models_compare.png", dpi=140, bbox_inches="tight")
plt.show()
"""
)

md("## 8. Сохранение лучшей модели + metrics.json")

code(
    """# финальный артефакт для сервиса / 07-замены
best_pipe = joblib.load(OUT / f"{best_name}.joblib")
joblib.dump(best_pipe, MODELS / "brand_clf.joblib")
joblib.dump(best_pipe, MODELS / f"brand_clf__{best_name}.joblib")

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
}
save_stats(metrics_out, OUT / "metrics.json")
save_stats(metrics_out, DATA / "train_metrics.json")
print("saved model →", MODELS / "brand_clf.joblib")
print("saved metrics →", OUT / "metrics.json")
print(json.dumps({k: metrics_out[k] for k in ["best_model", "best_raw", "best_subset_raw"]}, indent=2, ensure_ascii=False))
"""
)

md(
    """## 9. Выводы (заполнить после Run All)

Смотри авто-отчёт [`brand_classifier.md`](./brand_classifier.md) — он обновляется скриптом `_run_01.py` / по цифрам из `train_runs/metrics.json`.

Краткий чеклист:
- [ ] macro-F1 лучше старого ~0.67?  
- [ ] `false_brand_rate` на category-only низкий?  
- [ ] F1(`NO_BRAND`) / F1(`UNKNOWN`) приемлемы?  
- [ ] хвост классов с F1≈0 — поднять support / объединить / оставить UNKNOWN  
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", NB, "cells", len(cells))
