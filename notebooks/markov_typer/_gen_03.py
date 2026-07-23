"""Generate notebooks/markov_typer/03_attr_type_classifier.ipynb — full prod train in ipynb."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "03_attr_type_classifier.ipynb"
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
    """# 03. ATTR type classifier (prod train)

Обучение типизатора ATTR-span **в этом ноутбуке** (тот же пайплайн, что `_run_04_prod.py`).

**Вход:** `artifacts/attr_type/attr_type_silver_raw.parquet` из [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb).

**Выход:**
- `models/attr_type_clf.joblib`
- `artifacts/attr_type/inference_policy.json` (τ reject)
- отчёт [`attr_type_prod_report.md`](./attr_type_prod_report.md)

## Фичи

| колонка | роль |
|---|---|
| `span_text` | TF-IDF char/word **только** на span |
| `context_text` | `brand` + `category` |
| `query_masked` | word TF-IDF; ATTR → `<ATTR>` |

Пример: `ноутбук asus 16 г` → span=`16 г`, masked=`ноутбук asus <ATTR>`.
"""
)

md("## 0. Setup")

code(
    """%matplotlib inline
import sys, json, warnings
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path.cwd().resolve()
if ROOT.name in {"markov_typer", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "markov_typer" else ROOT.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_utils import (
    apply_plot_style, ensure_dirs, ARTIFACTS_DIR, FIGURES_DIR, MODELS,
    MVIDEO_RED, DARK_SLATE, save_stats,
)
from src.ner.attr_type_clf import Col, LABEL_UNKNOWN, looks_like_model, predict_attr_type
from src.ner.labeling import ATTR_PATTERNS, _guess_attr_type
from src.ner.markov_typer import MarkovAttrTyper, _norm_token

warnings.filterwarnings("ignore", category=FutureWarning)
ensure_dirs()
apply_plot_style()

OUT = ROOT / "artifacts" / "attr_type"
OUT.mkdir(parents=True, exist_ok=True)
FIG = FIGURES_DIR / "attr_type"
FIG.mkdir(parents=True, exist_ok=True)
SILVER_RAW = OUT / "attr_type_silver_raw.parquet"
REPORT = ROOT / "notebooks" / "markov_typer" / "attr_type_prod_report.md"
print("SILVER_RAW", SILVER_RAW.exists(), SILVER_RAW)
"""
)

md("## 1. Конфиг")

code(
    """SEED = 42
VAL_SIZE = 0.2
MIN_CLASS = 15
UNKNOWN_SHARE_CAP = 0.18
TAU_FLOOR = 0.55   # prod reject: не ниже

assert SILVER_RAW.exists(), "Сначала прогони 02_attr_type_silver.ipynb"
"""
)

md(
    """## 2. Все типы атрибутов (учитель)

Имена из `ATTR_PATTERNS` + `color` / `other` / `UNKNOWN` (на train).
"""
)

code(
    """pattern_types = sorted({name for _, name in ATTR_PATTERNS})
print(f"ATTR_PATTERNS types ({len(pattern_types)}):")
for t in pattern_types:
    print(f"  - {t}")
print("\\n+ color (COLORS), other (no match), UNKNOWN (rare/other на train)")

# sanity teacher fixes
for s in ["16 г", "16 гб", "5 g", "256 g", "1920x1080", "4k", "150 грамм", "2 кг"]:
    print(f"  teacher {s!r:12} -> {_guess_attr_type(s)}")
"""
)

md(
    """## 3. Подготовка обучающей выборки

Типы и лейблы — **только** из `src/ner/labeling.py` (`ATTR_PATTERNS` / `_guess_attr_type` / `COLORS`).  
Никаких unit-aug и синтетических строк: учимся на span’ах из silver_raw.

1. Relabel `y` через `_guess_attr_type`  
2. rare / `other` / model-like → `UNKNOWN` (train hygiene)  
3. stratified train/val
"""
)

code(
    """def stratified_split(df, label_col, test_size, seed):
    y = df[label_col].astype(str)
    vc = y.value_counts()
    rare1 = set(vc[vc < 2].index)
    ok, hold = df[~y.isin(rare1)], df[y.isin(rare1)]
    tr, va = train_test_split(ok, test_size=test_size, random_state=seed, stratify=ok[label_col])
    if len(hold):
        tr = pd.concat([tr, hold], ignore_index=True)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


raw = pd.read_parquet(SILVER_RAW)
print(f"silver_raw rows: {len(raw):,}")

# учитель = labeling.py (никаких новых типов)
df = raw.copy()
df["y_old"] = df["y"].astype(str)
df["y"] = df["span_text"].map(_guess_attr_type)
changed = df[df["y"] != df["y_old"]]
print(f"relabeled: {len(changed):,} ({len(changed)/len(df):.1%})")
if len(changed):
    display(changed.groupby(["y_old", "y"]).size().sort_values(ascending=False).head(12).to_frame("n"))

print("teacher type counts:")
display(df["y"].value_counts().to_frame("n"))

# train hygiene only: редкие + other + model-like → UNKNOWN
raw_vc = df["y"].value_counts()
rare = set(raw_vc[raw_vc < MIN_CLASS].index) | {"other"}
print("rare/other -> UNKNOWN:", sorted(rare))

df["y"] = df["y"].where(~df["y"].isin(rare), LABEL_UNKNOWN)
df.loc[df["span_text"].map(lambda s: looks_like_model(str(s))), "y"] = LABEL_UNKNOWN

unk, pos = df[df["y"] == LABEL_UNKNOWN], df[df["y"] != LABEL_UNKNOWN]
max_unk = max(80, int(UNKNOWN_SHARE_CAP * len(pos) / max(1e-6, 1 - UNKNOWN_SHARE_CAP)))
if len(unk) > max_unk:
    unk = unk.sample(n=max_unk, random_state=SEED)
df = pd.concat([pos, unk], ignore_index=True)
vc = df["y"].value_counts()
df = df[df["y"].isin(vc[vc >= MIN_CLASS].index)].copy()

df["context_text"] = (df["brand"].fillna("").astype(str) + " " + df["category"].fillna("").astype(str)).str.strip()
df["query_masked"] = df["query_masked_all_attr"].fillna("").astype(str)

train_df, val_df = stratified_split(df, "y", VAL_SIZE, SEED)
df.to_parquet(OUT / "attr_type_silver_prod.parquet", index=False)
train_df.to_parquet(OUT / "attr_type_train_prod.parquet", index=False)
val_df.to_parquet(OUT / "attr_type_val_prod.parquet", index=False)
train_df.to_parquet(OUT / "attr_type_train.parquet", index=False)
val_df.to_parquet(OUT / "attr_type_val.parquet", index=False)
print(f"train={len(train_df):,} val={len(val_df):,} classes={train_df['y'].nunique()}")
"""
)

md(
    """## 4. Классы и head обучающего датасета

Ниже — **на чём реально учимся**: список классов, support, примеры строк.
"""
)

code(
    """classes = sorted(train_df["y"].unique(), key=lambda x: (x == LABEL_UNKNOWN, x))
print("=== TRAIN CLASSES ===")
print(classes)

cls_tbl = (
    train_df["y"].value_counts()
    .rename("train_n")
    .to_frame()
    .join(val_df["y"].value_counts().rename("val_n"))
    .fillna(0).astype(int)
)
cls_tbl["share_train"] = (cls_tbl["train_n"] / cls_tbl["train_n"].sum()).round(4)
display(cls_tbl)

print("\\n=== train_df.head(15) ===")
show_cols = [c for c in [
    "span_text", "y", "brand", "category", "context_text", "query_masked",
    "query_norm", "n_attrs_in_query",
] if c in train_df.columns]
display(train_df[show_cols].head(15))

print("\\n=== examples per class (up to 3) ===")
for c in classes:
    ex = train_df.loc[train_df["y"] == c, "span_text"].drop_duplicates().head(3).tolist()
    print(f"  {c:22} {ex}")

fig, ax = plt.subplots(figsize=(8, 4.2))
vc_plot = train_df["y"].value_counts()
ax.barh(vc_plot.index.astype(str)[::-1], vc_plot.values[::-1], color=DARK_SLATE)
ax.set_title("Train class support")
ax.set_xlabel("count")
fig.tight_layout()
fig.savefig(FIG / "prod_02_class_support.png", dpi=120, bbox_inches="tight")
plt.show()
"""
)

md(
    """## 5. Четыре модели + Markov

| id | фичи |
|---|---|
| `logreg_span_char` | char на span |
| `logreg_span_wordchar` | word+char на span |
| `logreg_span_ctx` | span char + context char |
| `sgd_span_ctx_masked` | span char + **word** на `query_masked` |
"""
)

code(
    """def make_pipelines(seed=SEED):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40_000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=15_000)
    lr = dict(max_iter=300, solver="lbfgs", random_state=seed)
    return {
        "logreg_span_char": Pipeline([
            ("span", Col("span_text")),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40_000)),
            ("clf", LogisticRegression(**lr)),
        ]),
        "logreg_span_wordchar": Pipeline([
            ("feats", FeatureUnion([
                ("char", Pipeline([("c", Col("span_text")), ("t", char)])),
                ("word", Pipeline([("c", Col("span_text")), ("t", word)])),
            ])),
            ("clf", LogisticRegression(**lr)),
        ]),
        "logreg_span_ctx": Pipeline([
            ("feats", FeatureUnion([
                ("span", Pipeline([
                    ("c", Col("span_text")),
                    ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=40_000)),
                ])),
                ("ctx", Pipeline([
                    ("c", Col("context_text")),
                    ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=15_000)),
                ])),
            ])),
            ("clf", LogisticRegression(**lr)),
        ]),
        "sgd_span_ctx_masked": Pipeline([
            ("feats", FeatureUnion([
                ("span", Pipeline([
                    ("c", Col("span_text")),
                    ("t", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50_000)),
                ])),
                ("masked", Pipeline([
                    ("c", Col("query_masked")),
                    ("t", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=20_000)),
                ])),
            ])),
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, random_state=seed, tol=1e-3)),
        ]),
    }

PIPELINES = make_pipelines()
print(list(PIPELINES))
"""
)

md("## 6. Обучение + сводная таблица")

code(
    """ytr = train_df["y"].astype(str).values
yva = val_df["y"].astype(str).values
multi_m = (val_df["n_attrs_in_query"] >= 2).values

results, fitted = {}, {}
for name, pipe in PIPELINES.items():
    print(f"=== train {name} ===")
    pipe.fit(train_df, ytr)
    pred = pipe.predict(val_df)
    pack = {
        "accuracy": float(accuracy_score(yva, pred)),
        "f1_macro": float(f1_score(yva, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(yva, pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(yva, pred, average="weighted", zero_division=0)),
        "report": classification_report(yva, pred, labels=classes, output_dict=True, zero_division=0),
        "pred": pred,
    }
    pack["multi_f1_macro"] = (
        float(f1_score(yva[multi_m], pred[multi_m], average="macro", zero_division=0))
        if multi_m.any() else None
    )
    pack["f1_UNKNOWN"] = float(pack["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0))
    results[name] = pack
    fitted[name] = pipe
    joblib.dump(pipe, OUT / f"prod__{name}.joblib")
    print(f"  acc={pack['accuracy']:.3f} macro={pack['f1_macro']:.3f} multi={pack['multi_f1_macro']} unk={pack['f1_UNKNOWN']:.3f}")

# Markov baseline
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
m_pred = np.array([
    LABEL_UNKNOWN if (lab := markov.predict(s.split())[0]) == "unknown" else lab
    for s in val_df["span_text"].astype(str)
])
results["markov_lookup"] = {
    "accuracy": float(accuracy_score(yva, m_pred)),
    "f1_macro": float(f1_score(yva, m_pred, average="macro", zero_division=0)),
    "f1_micro": float(f1_score(yva, m_pred, average="micro", zero_division=0)),
    "f1_weighted": float(f1_score(yva, m_pred, average="weighted", zero_division=0)),
    "multi_f1_macro": float(f1_score(yva[multi_m], m_pred[multi_m], average="macro", zero_division=0)) if multi_m.any() else None,
    "report": classification_report(yva, m_pred, labels=classes, output_dict=True, zero_division=0),
    "pred": m_pred,
}
results["markov_lookup"]["f1_UNKNOWN"] = float(
    results["markov_lookup"]["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0)
)

summary = pd.DataFrame([
    {
        "model": n,
        "accuracy": v["accuracy"],
        "f1_macro": v["f1_macro"],
        "f1_micro": v["f1_micro"],
        "multi_f1_macro": v.get("multi_f1_macro"),
        "f1_UNKNOWN": v.get("f1_UNKNOWN"),
    }
    for n, v in results.items()
]).sort_values("f1_macro", ascending=False).reset_index(drop=True)
summary.to_csv(OUT / "prod_models_summary.csv", index=False)
summary.to_csv(OUT / "models_summary.csv", index=False)
display(summary)
"""
)

md("## 7. Best model: report, threshold, save")

code(
    """best_name = summary.loc[summary["model"] != "markov_lookup"].iloc[0]["model"]
pipe_best = fitted[best_name]
pred_best = results[best_name]["pred"]
print("BEST =", best_name)
print(classification_report(yva, pred_best, labels=classes, digits=3, zero_division=0))

rep = results[best_name]["report"]
per_class = pd.DataFrame([
    {"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"],
     "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
    for c in classes if c in rep
]).sort_values("f1", ascending=False)
per_class.to_csv(OUT / "prod_per_class_f1.csv", index=False)
per_class.to_csv(OUT / "per_class_f1__BEST.csv", index=False)
display(per_class)

# calibrate tau >= TAU_FLOOR
proba = pipe_best.predict_proba(val_df)
raw_pred = pipe_best.classes_[proba.argmax(axis=1)]
conf = proba.max(axis=1)
cal_rows = []
for tau in np.round(np.linspace(0.35, 0.90, 12), 2):
    pred = np.where(conf >= tau, raw_pred, LABEL_UNKNOWN)
    cal_rows.append({
        "tau": float(tau),
        "f1_macro": float(f1_score(yva, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(yva, pred)),
        "coverage": float((conf >= tau).mean()),
    })
cal = pd.DataFrame(cal_rows)
cal_safe = cal[cal["tau"] >= TAU_FLOOR].sort_values(["f1_macro", "coverage"], ascending=[False, False])
tau = float(cal_safe.iloc[0]["tau"]) if len(cal_safe) else TAU_FLOOR
cal.to_csv(OUT / "prod_threshold_curve.csv", index=False)
pred_rej = np.where(conf >= tau, raw_pred, LABEL_UNKNOWN)
print(f"tau={tau}  acc={accuracy_score(yva, pred_rej):.3f}  "
      f"macro={f1_score(yva, pred_rej, average='macro', zero_division=0):.3f}  "
      f"coverage={(conf>=tau).mean():.1%}")

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
        "query_masked": "query with all ATTR spans replaced by <ATTR> (word TF-IDF)",
    },
    "n_train": len(train_df),
    "n_val": len(val_df),
    "classes": classes,
    "val_metrics_raw": {
        "accuracy": results[best_name]["accuracy"],
        "f1_macro": results[best_name]["f1_macro"],
        "multi_f1_macro": results[best_name]["multi_f1_macro"],
        "f1_UNKNOWN": results[best_name]["f1_UNKNOWN"],
    },
    "val_metrics_with_reject": {
        "tau": tau,
        "accuracy": float(accuracy_score(yva, pred_rej)),
        "f1_macro": float(f1_score(yva, pred_rej, average="macro", zero_division=0)),
        "coverage": float((conf >= tau).mean()),
    },
}
(OUT / "inference_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "metrics.json").write_text(json.dumps({"best_model": best_name, **policy}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved", MODELS / "attr_type_clf.joblib")
print("policy tau=", tau)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].barh(summary["model"][::-1], summary["f1_macro"][::-1], color=MVIDEO_RED)
axes[0].set_xlabel("f1_macro"); axes[0].set_title("Models (val)")
axes[1].plot(cal["tau"], cal["f1_macro"], "o-", color=MVIDEO_RED, label="f1_macro")
axes[1].plot(cal["tau"], cal["coverage"], "s--", color=DARK_SLATE, label="coverage")
axes[1].axvline(tau, color="gray", ls=":", label=f"tau={tau}")
axes[1].set_xlabel("min_confidence"); axes[1].legend(fontsize=8)
axes[1].set_title("Reject threshold")
fig.tight_layout()
fig.savefig(FIG / "prod_01_models_threshold.png", dpi=120, bbox_inches="tight")
plt.show()
"""
)

md("## 8. Sanity 10 + ручной тест")

code(
    """sanity = [
    {"span": "16 г", "brand": "asus", "category": "ноутбук", "expect": "memory_storage"},
    {"span": "16 гб", "brand": "asus", "category": "ноутбук", "expect": "memory_storage"},
    {"span": "256 g", "brand": "samsung", "category": "смартфон", "expect": "memory_storage"},
    {"span": "5 g", "brand": "samsung", "category": "смартфон", "expect": "UNKNOWN"},
    {"span": "2 кг", "brand": "bosch", "category": "пылесос", "expect": "weight"},
    {"span": "150 грамм", "brand": "", "category": "весы", "expect": "weight"},
    {"span": "1920x1080", "brand": "xiaomi", "category": "монитор", "expect": "resolution_exact"},
    {"span": "4k", "brand": "lg", "category": "телевизор", "expect": "resolution_standard"},
    {"span": "15.6 дюйм", "brand": "asus", "category": "ноутбук", "expect": "size"},
    {"span": "g pro", "brand": "logitech", "category": "наушники", "expect": "UNKNOWN"},
]
sanity_rows = []
for c in sanity:
    det = predict_attr_type(
        c["span"], brand=c["brand"], category=c["category"],
        query_masked=f"{c['category']} {c['brand']} <ATTR>".strip(),
        model_path=MODELS / "attr_type_clf.joblib",
        min_confidence=tau, return_details=True,
    )
    ok = det["label"] == c["expect"]
    sanity_rows.append({
        "span": c["span"], "expect": c["expect"], "pred": det["label"],
        "conf": round(det["confidence"], 3), "ok": ok, "teacher": _guess_attr_type(c["span"]),
    })
    print(f"{'OK' if ok else 'FAIL'} {c['span']!r:12} expect={c['expect']:20} pred={det['label']:20} conf={det['confidence']:.2f}")
sanity_df = pd.DataFrame(sanity_rows)
sanity_df.to_csv(OUT / "prod_sanity_10.csv", index=False)
display(sanity_df)
print(f"sanity {sanity_df.ok.sum()}/{len(sanity_df)}")
"""
)

code(
    """# --- правь здесь ---
SPAN = "16 г"
BRAND = "asus"
CATEGORY = "ноутбук"
QUERY_MASKED = "ноутбук asus <ATTR>"
# -------------------
det = predict_attr_type(
    SPAN, brand=BRAND, category=CATEGORY, query_masked=QUERY_MASKED,
    model_path=MODELS / "attr_type_clf.joblib", min_confidence=tau, return_details=True,
)
print("teacher:", _guess_attr_type(SPAN))
print(json.dumps({k: ([(a, float(b)) for a, b in v] if k == "top" else v) for k, v in det.items()},
                 ensure_ascii=False, indent=2))
"""
)

md("## 9. Отчёт")

code(
    """lines = [
    "# ATTR type classifier — prod report",
    "",
    f"Ноутбук обучения: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb)  ",
    f"Модель: **`{best_name}`** → `models/attr_type_clf.joblib`  ",
    f"Policy: `artifacts/attr_type/inference_policy.json` (τ=`{tau}`)  ",
    f"Sanity: **{int(sanity_df.ok.sum())}/{len(sanity_df)}**",
    "",
    "## Классы train",
    "",
    "| class | train_n | val_n |",
    "|---|---:|---:|",
]
for c, r in cls_tbl.iterrows():
    lines.append(f"| `{c}` | {int(r['train_n'])} | {int(r['val_n'])} |")
lines += [
    "",
    "## Фичи (пример `ноутбук asus 16 г`)",
    "",
    "| фича | значение |",
    "|---|---|",
    "| `span_text` | `16 г` — char TF-IDF |",
    "| `context_text` | `asus ноутбук` |",
    "| `query_masked` | `ноутбук asus <ATTR>` — **word** TF-IDF |",
    "",
    "## Сводка моделей",
    "",
    "| model | acc | f1_macro | multi_f1_macro | f1_UNKNOWN |",
    "|---|---:|---:|---:|---:|",
]
for _, r in summary.iterrows():
    mm = "—" if pd.isna(r["multi_f1_macro"]) else f"{r['multi_f1_macro']:.3f}"
    star = "**" if r["model"] == best_name else ""
    lines.append(
        f"| `{r['model']}` | {r['accuracy']:.3f} | {star}{r['f1_macro']:.3f}{star} | {mm} | {r['f1_UNKNOWN']:.3f} |"
    )
lines += [
    "",
    f"Reject τ={tau}: см. `inference_policy.json`.",
    "",
    "## Sanity 10",
    "",
    "| span | expect | pred | conf | ok | teacher |",
    "|---|---|---|---:|:---:|---|",
]
for _, r in sanity_df.iterrows():
    lines.append(
        f"| `{r['span']}` | {r['expect']} | {r['pred']} | {r['conf']:.2f} | "
        f"{'OK' if r['ok'] else 'FAIL'} | {r['teacher']} |"
    )
lines += ["", "## Per-class F1", "", "| class | precision | recall | f1 | support |", "|---|---:|---:|---:|---:|"]
for _, r in per_class.iterrows():
    lines.append(
        f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
    )
REPORT.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
save_stats({"attr_type_clf": policy}, name="attr_type_prod_metrics.json")
print("report ->", REPORT)
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
