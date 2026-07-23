"""Generate notebooks/crf_ner_classifier/02_crf_classifier.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_crf_classifier.ipynb"
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
    """# 02. CRF NER classifier

Обучение BIO-спанов (`BRAND` / `CATEGORY` / `MODEL` / `ATTR`) на silver из [`01_crf_eda`](./01_crf_eda.ipynb).

| | |
|---|---|
| Silver | `artifacts/silver/ner_bio/silver_bio_slice.parquet` |
| Модель | `models/ner_crf.pkl` |
| Eval | silver-val (optimistic) + **gold** `bio_liza.jsonl` |

Фичи — **не TF-IDF**: shape / аффиксы / соседи (`src/ner/features.py`).

CLI: `python notebooks/crf_ner_classifier/_run_02.py`
"""
)

md("## 0. Setup")
code(
    """%matplotlib inline
import sys, json, warnings
from pathlib import Path
from collections import Counter

ROOT = Path.cwd().resolve()
if ROOT.name in {"crf_ner_classifier", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "crf_ner_classifier" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_utils import (
    ensure_dirs, apply_plot_style, resolve_silver, MODELS, FIGURES_DIR,
    MVIDEO_RED, DARK_SLATE,
)
from src.ner.labeling import tokenize
from src.ner.metrics import summarize_metrics
from src.ner.model_crf import CRFNerModel

warnings.filterwarnings("ignore", category=FutureWarning)
ensure_dirs()
apply_plot_style()

SEED, VAL_SIZE, MAX_ITERS = 42, 0.2, 80
LABELS = ["BRAND", "CATEGORY", "MODEL", "ATTR"]
FIG = FIGURES_DIR / "ner"
FIG.mkdir(parents=True, exist_ok=True)
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
print("MODELS", MODELS)
"""
)

md("## 1. Load silver NER BIO")
code(
    """silver_path = resolve_silver("ner_bio", "silver_bio_slice.parquet")
df = pd.read_parquet(silver_path)
print(silver_path, "rows", len(df))
if "has_entity" in df.columns:
    df = df[df["has_entity"]].copy()
print("with_entity", len(df))
display(df[["query", "n_BRAND", "n_CATEGORY", "n_MODEL", "n_ATTR", "bio_str"]].head(8))

bc = Counter()
for tags in df["tags"]:
    for t in tags:
        if str(t).startswith("B-"):
            bc[str(t)[2:]] += 1
print("B-* counts", dict(bc))
"""
)

md("## 2. Train / val split + fit CRF")
code(
    """def rows_to_sents(frame):
    out = []
    for _, r in frame.iterrows():
        toks, tags = list(r["tokens"]), list(r["tags"])
        if toks and len(toks) == len(tags):
            out.append(list(zip(toks, tags)))
    return out

train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=SEED)
train_sents, val_sents = rows_to_sents(train_df), rows_to_sents(val_df)
print(f"train={len(train_sents)} val={len(val_sents)}")

model = CRFNerModel(max_iterations=MAX_ITERS)
model.fit(train_sents)
print("fitted")
"""
)

md("## 3. Silver-val metrics (weak↔weak)")
code(
    """y_val = [[t for _, t in s] for s in val_sents]
pred_val = model.predict(val_sents)
silver_m = summarize_metrics(y_val, pred_val)
print(f"tokAcc={silver_m['token_accuracy']:.3f} microF1={silver_m['micro']['f1']:.3f} macro={silver_m['macro_f1']:.3f}")
per = silver_m["per_label"]
display(pd.DataFrame([
    {"label": l, **{k: per[l][k] for k in ("precision","recall","f1","support")}}
    for l in LABELS if l in per
]))
"""
)

md("## 4. Gold eval (primary)")
code(
    """gold_sents, meta = [], {"n": 0, "tokenize_align": 0, "used": 0, "skipped": 0}
for line in GOLD.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    meta["n"] += 1
    q, tags = r["query"], r["tags"]
    toks = [t for t, _, _ in tokenize(q)]
    if len(toks) == len(tags):
        meta["tokenize_align"] += 1
        gold_sents.append(list(zip(toks, tags)))
        meta["used"] += 1
    elif len(q.split()) == len(tags):
        gold_sents.append(list(zip(q.split(), tags)))
        meta["used"] += 1
    else:
        meta["skipped"] += 1

y_gold = [[t for _, t in s] for s in gold_sents]
pred_gold = model.predict_tokens([[t for t, _ in s] for s in gold_sents])
gold_m = summarize_metrics(y_gold, pred_gold)
print(meta)
print(f"GOLD tokAcc={gold_m['token_accuracy']:.3f} microF1={gold_m['micro']['f1']:.3f} macro={gold_m['macro_f1']:.3f}")
gp = gold_m["per_label"]
display(pd.DataFrame([
    {"label": l, **{k: gp[l][k] for k in ("precision","recall","f1","support")}}
    for l in LABELS if l in gp
]))
"""
)

md("## 5. Save + demos + plot")
code(
    """model.save(MODELS / "ner_crf.pkl")
model.save(MODELS / "ner_crf__silver_v1.pkl")
print("saved", MODELS / "ner_crf.pkl")

labs = [l for l in LABELS if l in per]
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(labs)); w = 0.35
ax.bar(x - w/2, [per[l]["f1"] for l in labs], w, color=MVIDEO_RED, label="silver-val")
ax.bar(x + w/2, [gp.get(l, {}).get("f1", 0) for l in labs], w, color=DARK_SLATE, label="gold")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 1.05)
ax.set_title("CRF entity F1"); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "02_crf_entity_f1.png", dpi=120, bbox_inches="tight"); plt.show()

for q in ["asus tuf gaming a15 16 гб", "ноутбук asus 16 гб", "iphone 15 pro max"]:
    print(q, "->", " ".join(f"{a}/{b}" for a, b in model.predict_query(q)))
"""
)

md(
    """## Verdict

- Silver-val завышен (тот же teacher).
- Смотри **gold micro-F1** в ячейке выше / [`02_crf_report.md`](./02_crf_report.md).
- Дальше: больше silver (`_run_01` ↑ MAX_QUERIES) и правки teacher на MODEL/ATTR misses.
"""
)

NB.write_text(
    json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "cells": cells,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print("wrote", NB, "cells", len(cells))
