"""Generate notebooks/markov_typer/03_attr_type_classifier.ipynb"""
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
    """# 03. ATTR type classifier (TF-IDF n-grams)

Обучение типизатора ATTR-span на silver из [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb).

## Дизайн фичей

| | |
|---|---|
| **X_span** | TF-IDF char/word **только** на `span_text` |
| **X_ctx** | `brand` + `category` (отдельно) |
| **X_masked** | `query_masked_all_attr` (все ATTR → `<ATTR>`) |
| **y** | тип + `UNKNOWN` (редкие / `other`) |

Несколько ATTR в запросе → уже разложены по строкам silver; чужие единицы в контексте маскированы.

4 sklearn-модели + `markov_lookup` как бейзлайн.
"""
)

md("## 0. Setup")

code(
    """%matplotlib inline
import sys
import json
import warnings
from pathlib import Path
from collections import Counter

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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.data_utils import (
    apply_plot_style, ensure_dirs, ARTIFACTS_DIR, FIGURES_DIR, MODELS,
    MVIDEO_RED, DARK_SLATE, MUTED, save_stats,
)
from src.ner.attr_type_clf import Col, LABEL_UNKNOWN, looks_like_model, predict_attr_type
from src.ner.markov_typer import MarkovAttrTyper, _norm_token

warnings.filterwarnings("ignore", category=FutureWarning)
ensure_dirs()
apply_plot_style()

OUT = ROOT / "artifacts" / "attr_type"
OUT.mkdir(parents=True, exist_ok=True)
FIG = FIGURES_DIR / "attr_type"
FIG.mkdir(parents=True, exist_ok=True)
SILVER = OUT / "attr_type_silver.parquet"
print("SILVER", SILVER.exists(), SILVER)
print("OUT", OUT)
"""
)

md("## 1. Конфиг")

code(
    """SEED = 42
VAL_SIZE = 0.2
MIN_CLASS = 25          # rare → UNKNOWN
UNKNOWN_SHARE_CAP = 0.18
INCLUDE_AUG = True      # unit-aug строки из silver

assert SILVER.exists(), f"Нет {SILVER} — сначала 02_attr_type_silver.ipynb"
"""
)

md(
    """## 2. Загрузка silver + подготовка y

- Редкие типы (`support < MIN_CLASS` на raw) и `other` → **`UNKNOWN`**
- Cap доли UNKNOWN
- `context_text`, `query_masked` для пайплайнов
"""
)

code(
    """def stratified_split(df, label_col, test_size, seed):
    y = df[label_col].astype(str)
    vc = y.value_counts()
    # классы с 1 примером — целиком в train
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


silver = pd.read_parquet(SILVER)
if not INCLUDE_AUG:
    silver = silver.loc[~silver["is_aug"]].copy()
print(f"silver rows: {len(silver):,}  aug={int(silver['is_aug'].sum()):,}")

# rare по raw-support
raw_vc = (
    silver.loc[~silver["is_aug"], "y"].value_counts()
    if "is_aug" in silver.columns
    else silver["y"].value_counts()
)
rare = set(raw_vc[raw_vc < MIN_CLASS].index) | {"other"}
print("→ UNKNOWN:", sorted(rare))

df = silver.copy()
df["y_raw"] = df["y"].astype(str)
df["y"] = df["y_raw"].where(~df["y_raw"].isin(rare), LABEL_UNKNOWN)

# modelish spans (если вдруг попали как ATTR) → UNKNOWN
df.loc[df["span_text"].map(lambda s: looks_like_model(str(s))), "y"] = LABEL_UNKNOWN

# cap UNKNOWN
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

print("class counts:")
print(df["y"].value_counts())
train_df, val_df = stratified_split(df, "y", VAL_SIZE, SEED)
train_df.to_parquet(OUT / "attr_type_train.parquet", index=False)
val_df.to_parquet(OUT / "attr_type_val.parquet", index=False)
print(f"train={len(train_df):,} val={len(val_df):,} classes={train_df['y'].nunique()}")
print(f"multi-ATTR share val: {(val_df['n_attrs_in_query'] >= 2).mean():.1%}")
"""
)

md(
    """## 3. Четыре модели

| id | фичи |
|---|---|
| `logreg_span_char` | char_wb(2–4) на span |
| `logreg_span_wordchar` | word(1–2) + char на span |
| `logreg_span_ctx` | span char + `context_text` char |
| `sgd_span_ctx_masked` | span char + `query_masked` word |
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
            ("clf", SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=40, random_state=seed, tol=1e-3)),
        ]),
    }

PIPELINES = make_pipelines()
print(list(PIPELINES))
"""
)

md("## 4. Обучение + сводная таблица метрик")

code(
    """classes = sorted(train_df["y"].unique(), key=lambda x: (x == LABEL_UNKNOWN, x))
ytr = train_df["y"].astype(str).values
yva = val_df["y"].astype(str).values
multi_m = (val_df["n_attrs_in_query"] >= 2).values

results = {}
fitted = {}

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
    if multi_m.any():
        pack["multi_f1_macro"] = float(f1_score(yva[multi_m], pred[multi_m], average="macro", zero_division=0))
        pack["multi_accuracy"] = float(accuracy_score(yva[multi_m], pred[multi_m]))
    else:
        pack["multi_f1_macro"] = None
        pack["multi_accuracy"] = None
    pack["f1_UNKNOWN"] = float(pack["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0))
    results[name] = pack
    fitted[name] = pipe
    joblib.dump(pipe, OUT / f"{name}.joblib")
    print(
        f"  acc={pack['accuracy']:.3f} macro={pack['f1_macro']:.3f} "
        f"multi_macro={pack['multi_f1_macro']} unk={pack['f1_UNKNOWN']:.3f}"
    )

# Markov baseline: учим на train span→y (не regex-цикл), lookup на val
markov = MarkovAttrTyper()
big = Counter()
# reuse structure via temporary train
from collections import defaultdict
_big = defaultdict(Counter)
_unit = defaultdict(Counter)
_trans = defaultdict(Counter)
for _, r in train_df.iterrows():
    toks = [ _norm_token(t) for t in str(r["span_text"]).split() ]
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
    if lab == "unknown":
        lab = LABEL_UNKNOWN
    m_pred.append(lab)
m_pred = np.array(m_pred)
results["markov_lookup"] = {
    "accuracy": float(accuracy_score(yva, m_pred)),
    "f1_macro": float(f1_score(yva, m_pred, average="macro", zero_division=0)),
    "f1_micro": float(f1_score(yva, m_pred, average="micro", zero_division=0)),
    "f1_weighted": float(f1_score(yva, m_pred, average="weighted", zero_division=0)),
    "multi_f1_macro": float(f1_score(yva[multi_m], m_pred[multi_m], average="macro", zero_division=0)) if multi_m.any() else None,
    "multi_accuracy": float(accuracy_score(yva[multi_m], m_pred[multi_m])) if multi_m.any() else None,
    "report": classification_report(yva, m_pred, labels=classes, output_dict=True, zero_division=0),
    "pred": m_pred,
}
results["markov_lookup"]["f1_UNKNOWN"] = float(
    results["markov_lookup"]["report"].get(LABEL_UNKNOWN, {}).get("f1-score", 0.0)
)
markov.save(MODELS / "markov_typer_attr_clf_baseline.json")
print("markov_lookup", {k: results["markov_lookup"][k] for k in ["accuracy", "f1_macro", "multi_f1_macro"]})

summary = pd.DataFrame([
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
]).sort_values("f1_macro", ascending=False).reset_index(drop=True)
summary.to_csv(OUT / "models_summary.csv", index=False)
display(summary)
"""
)

md("## 5. Classification report лучшей sklearn-модели")

code(
    """sklearn_names = [n for n in summary["model"] if n != "markov_lookup"]
best_name = summary.loc[summary["model"].isin(sklearn_names)].iloc[0]["model"]
pipe_best = fitted[best_name]
pred_best = results[best_name]["pred"]

print(f"BEST = {best_name}")
print(classification_report(yva, pred_best, labels=classes, digits=3, zero_division=0))

# per-class table
rep = results[best_name]["report"]
per_class = pd.DataFrame([
    {
        "class": c,
        "precision": rep[c]["precision"],
        "recall": rep[c]["recall"],
        "f1": rep[c]["f1-score"],
        "support": int(rep[c]["support"]),
    }
    for c in classes if c in rep
]).sort_values("f1", ascending=False)
per_class.to_csv(OUT / "per_class_f1__BEST.csv", index=False)
per_class.to_csv(OUT / f"per_class_f1__{best_name}.csv", index=False)
display(per_class)

for name in PIPELINES:
    r = results[name]["report"]
    pc = pd.DataFrame([
        {"class": c, "precision": r[c]["precision"], "recall": r[c]["recall"],
         "f1": r[c]["f1-score"], "support": int(r[c]["support"])}
        for c in classes if c in r
    ])
    pc.to_csv(OUT / f"per_class_f1__{name}.csv", index=False)
"""
)

md("## 6. Confusion + сравнение + ошибки")

code(
    """top_lab = list(val_df["y"].value_counts().head(10).index)
if LABEL_UNKNOWN in classes and LABEL_UNKNOWN not in top_lab:
    top_lab = top_lab[:-1] + [LABEL_UNKNOWN]

cm = confusion_matrix(yva, pred_best, labels=top_lab)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
im = axes[0].imshow(cm, cmap="Reds")
axes[0].set_xticks(range(len(top_lab)))
axes[0].set_yticks(range(len(top_lab)))
axes[0].set_xticklabels(top_lab, rotation=45, ha="right", fontsize=8)
axes[0].set_yticklabels(top_lab, fontsize=8)
axes[0].set_title(f"Confusion (top) — {best_name}")
fig.colorbar(im, ax=axes[0], fraction=0.046)

axes[1].barh(summary["model"][::-1], summary["f1_macro"][::-1], color=MVIDEO_RED, label="f1_macro")
axes[1].barh(
    summary["model"][::-1],
    summary["multi_f1_macro"].fillna(0)[::-1],
    color=DARK_SLATE, alpha=0.55, label="multi_f1_macro",
)
axes[1].set_xlabel("F1")
axes[1].set_title("Models compare")
axes[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "02_clf_compare.png", dpi=120, bbox_inches="tight")
plt.show()

err = val_df.copy()
err["pred"] = pred_best
err = err[err["y"] != err["pred"]]
print(f"errors: {len(err):,} / {len(val_df):,}")
display(err[["span_text", "y", "pred", "brand", "category", "query_norm", "n_attrs_in_query"]].head(20))
"""
)

md("## 7. Demo + сохранение артефактов / отчёт")

code(
    """joblib.dump(pipe_best, MODELS / "attr_type_clf.joblib")
joblib.dump(pipe_best, MODELS / f"attr_type_clf__{best_name}.joblib")

demos = ["16 гб", "16 кб", "16 kb", "256 gb", "15.6 дюйм", "g pro", "2 кг", "белый", "55 дюймов"]
print("demo predict_attr_type / pipe:")
for s in demos:
    rule = looks_like_model(s)
    if rule:
        print(f"  {s!r:16} → UNKNOWN (modelish rule)")
        continue
    row = pd.DataFrame([{
        "span_text": s,
        "context_text": "ноутбук asus",
        "query_masked": "ноутбук asus <ATTR>",
        "brand": "asus",
        "category": "ноутбук",
    }])
    print(f"  {s!r:16} → {pipe_best.predict(row)[0]}")

metrics = {
    "best_model": best_name,
    "summary": summary.to_dict(orient="records"),
    "best_raw": {k: results[best_name][k] for k in ["accuracy", "f1_macro", "f1_micro", "multi_f1_macro", "f1_UNKNOWN"]},
    "markov": {k: results["markov_lookup"][k] for k in ["accuracy", "f1_macro", "multi_f1_macro", "f1_UNKNOWN"]},
    "n_train": len(train_df),
    "n_val": len(val_df),
    "classes": classes,
    "min_class": MIN_CLASS,
    "unknown_share_cap": UNKNOWN_SHARE_CAP,
    "include_aug": INCLUDE_AUG,
    "design_note": "char n-grams on span only; brand/category/masked query separate; one row per ATTR span",
}
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
save_stats({"attr_type_clf": metrics}, name="attr_type_clf_metrics.json")

# markdown report
lines = [
    "# ATTR type classifier — отчёт",
    "",
    f"Ноутбук: [`03_attr_type_classifier.ipynb`](./03_attr_type_classifier.ipynb)  ",
    f"Silver: [`02_attr_type_silver.ipynb`](./02_attr_type_silver.ipynb) / [`attr_type_silver.md`](./attr_type_silver.md)  ",
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
        f"| {r['class']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
    )
lines += [
    "",
    "## Вывод",
    "",
    f"- Лучшая модель: `{best_name}` (macro F1 **{results[best_name]['f1_macro']:.3f}**).",
    f"- Markov baseline: macro **{results['markov_lookup']['f1_macro']:.3f}**.",
    "- Слабые классы смотри в per-class таблице (часто `dimensions` / хвост).",
    "",
]
report_path = ROOT / "notebooks" / "markov_typer" / "attr_type_classifier.md"
report_path.write_text("\\n".join(lines), encoding="utf-8")
print("report →", report_path)
print("model  →", MODELS / "attr_type_clf.joblib")
"""
)

md(
    """## 8. Ручной тест

Правишь `SPAN` / контекст ниже и перезапускаешь ячейку.  
Если обучение уже было — берётся `pipe_best`; иначе грузится `models/attr_type_clf.joblib`.
"""
)

code(
    """# --- правь здесь ---
SPAN = "16 гб"
BRAND = "asus"
CATEGORY = "ноутбук"
QUERY_MASKED = "ноутбук asus <ATTR>"   # как в проде: другие ATTR уже <ATTR>
TOP_K = 5
# -------------------

clf_path = MODELS / "attr_type_clf.joblib"
pipe = globals().get("pipe_best")
if pipe is None:
    assert clf_path.exists(), f"Нет модели: {clf_path} — сначала секции 4–7"
    pipe = joblib.load(clf_path)
    print("loaded", clf_path)
else:
    print("using pipe_best from training")

modelish = looks_like_model(SPAN)
row = pd.DataFrame([{
    "span_text": SPAN,
    "context_text": f"{BRAND} {CATEGORY}".strip(),
    "query_masked": QUERY_MASKED or SPAN,
    "brand": BRAND,
    "category": CATEGORY,
}])

print(f"\\nspan={SPAN!r}")
print(f"context={row.loc[0, 'context_text']!r}")
print(f"masked={row.loc[0, 'query_masked']!r}")
print(f"modelish_rule={modelish}")

if modelish:
    print(f"\\n→ pred = {LABEL_UNKNOWN}  (looks_like_model до clf)")
else:
    pred = pipe.predict(row)[0]
    print(f"\\n→ pred = {pred}")
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(row)[0]
        classes_ = list(pipe.classes_)
        ranking = sorted(zip(classes_, proba), key=lambda x: -x[1])[:TOP_K]
        print(f"\\ntop-{TOP_K} probabilities:")
        for lab, p in ranking:
            mark = " ←" if lab == pred else ""
            print(f"  {lab:22} {p:.4f}{mark}")

BATCH = [
    "16 гб", "16 кб", "256 gb", "15.6 дюйм", "2 кг", "белый",
    "55 дюймов", "500 вт", "g pro", "1920x1080", "4k",
]
rows = []
for s in BATCH:
    if looks_like_model(s):
        rows.append({"span_text": s, "pred": LABEL_UNKNOWN, "via": "modelish_rule", "p_max": 1.0})
        continue
    r = pd.DataFrame([{
        "span_text": s,
        "context_text": f"{BRAND} {CATEGORY}".strip(),
        "query_masked": QUERY_MASKED,
        "brand": BRAND,
        "category": CATEGORY,
    }])
    yhat = pipe.predict(r)[0]
    pmax = float(pipe.predict_proba(r).max()) if hasattr(pipe, "predict_proba") else None
    rows.append({"span_text": s, "pred": yhat, "via": "clf", "p_max": pmax})

display(pd.DataFrame(rows))
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
