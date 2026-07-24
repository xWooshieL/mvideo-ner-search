"""Generate notebooks/general_study/02_general_train_eval.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_general_train_eval.ipynb"
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
    """# 02. General study — совместное обучение CRF + GLiNER, метрика всего каскада

CRF и GLiNER учатся на **одном и том же** silver из [`01`](./01_general_silver.ipynb).
**Gold не участвует в обучении** — чистый held-out. Метрика считается для **каскада
целиком** (rules → +CRF → +CRF+GLiNER), на трёх eval-сетах: `gold`, `broken_queries`,
`silver_val`.

CLI: `python notebooks/general_study/_run_02.py` (CRF — секунды, GLiNER — ~15-20 мин на CPU
с учётом финальной оценки трёх стадий на трёх сетах).
"""
)

md("## 0. Setup")
code(
    """%matplotlib inline
import sys, json, time
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name in {"general_study", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "general_study" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.data_utils import (
    ARTIFACTS_DIR, FIGURES_DIR, METRICS_DIR, ensure_dirs, apply_plot_style,
    brands_path, categories_path, model_phrases_path, MVIDEO_RED, DARK_SLATE,
)
from src.ner.labeling import WeakLabeler, bio_to_entities, tokenize
from src.ner.model_crf import CRFNerModel
from src.ner.gliner_utils import (
    OUR_LABELS, bio_row_to_gliner_example, gliner_labels,
    gliner_preds_to_our_labels, gold_row_char_entities, load_gold_bio_rows, span_prf1,
)

ensure_dirs(); apply_plot_style()
SILVER_DIR = ARTIFACTS_DIR / "silver" / "general_study"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "general_study"; FIG.mkdir(parents=True, exist_ok=True)
MODELS_DIR = ROOT / "models" / "general_study"; MODELS_DIR.mkdir(parents=True, exist_ok=True)
SEED, VAL_SIZE = 42, 0.15
STAGES = ["rules", "rules_crf", "rules_crf_gliner"]
print("SILVER_DIR", SILVER_DIR)
"""
)

md("## 1. Загрузка: general silver / broken_queries / gold")
code(
    """silver = pd.read_parquet(SILVER_DIR / "general_silver_bio_ent.parquet")
broken_df = pd.read_parquet(SILVER_DIR / "broken_queries_eval.parquet")
gold_rows = load_gold_bio_rows(GOLD)
print(f"silver={len(silver)} broken={len(broken_df)} gold(held out)={len(gold_rows)}")

train_df, val_df = train_test_split(silver, test_size=VAL_SIZE, random_state=SEED)
print(f"train={len(train_df)} val={len(val_df)}")

def rows_from_df(df):
    out = []
    for _, r in df.iterrows():
        q = r["query"]
        toks_spans = tokenize(q)
        toks = [t for t, _, _ in toks_spans]
        tags = list(r["tags"])
        if len(toks) != len(tags):
            continue
        out.append({"query": q, "tokens": toks, "tags": tags, "char_spans": [(s, e) for _, s, e in toks_spans]})
    return out

silver_val_rows = rows_from_df(val_df)
broken_rows = rows_from_df(broken_df)
labeler = WeakLabeler.from_files(brands_path(), categories_path(), models_path=model_phrases_path())
"""
)

md("## 2. Обучение CRF (на всём train)")
code(
    """def rows_to_sents(df):
    out = []
    for _, r in df.iterrows():
        toks, tags = list(r["tokens"]), list(r["tags"])
        if toks and len(toks) == len(tags):
            out.append(list(zip(toks, tags)))
    return out

train_sents = rows_to_sents(train_df)
t0 = time.perf_counter()
crf = CRFNerModel(max_iterations=80)
crf.fit(train_sents)
print(f"CRF fit in {time.perf_counter()-t0:.1f}s on {len(train_sents)} sents")
crf.save(MODELS_DIR / "ner_crf.pkl")
"""
)

md("## 3. Обучение GLiNER (на сэмпле того же train)")
code(
    """from gliner import GLiNER

GLINER_SCHEME, GLINER_TRAIN_N, GLINER_BATCH, GLINER_EPOCHS = "ru", 500, 8, 3
gliner_train_df = train_df.sample(n=min(GLINER_TRAIN_N, len(train_df)), random_state=SEED)
gliner_examples = [
    bio_row_to_gliner_example(list(r["tokens"]), list(r["tags"]), scheme=GLINER_SCHEME)
    for _, r in gliner_train_df.iterrows()
]
gliner_examples = [e for e in gliner_examples if e["ner"]]
print("GLiNER train examples:", len(gliner_examples))

gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
steps_per_epoch = max(1, -(-len(gliner_examples) // GLINER_BATCH))
max_steps = steps_per_epoch * GLINER_EPOCHS
t0 = time.perf_counter()
gliner_model.train_model(
    train_dataset=gliner_examples, eval_dataset=None,
    output_dir=str(MODELS_DIR / "_gliner_ckpt"),
    learning_rate=1e-5, others_lr=5e-5, weight_decay=0.01,
    per_device_train_batch_size=GLINER_BATCH, max_steps=max_steps,
    warmup_ratio=0.1, logging_steps=max(1, steps_per_epoch),
    save_steps=max(1, max_steps), save_total_limit=1, use_cpu=True, report_to="none",
)
print(f"GLiNER fit in {time.perf_counter()-t0:.1f}s")
"""
)

md(
    """## 4. Каскад: rules → +CRF → +CRF+GLiNER

Своя копия merge-логики (как в `src/service/extractor.py`, но не трогаем прод-код)."""
)
code(
    """def merge_entities(primary, secondary):
    result = list(primary)
    occupied = [tuple(e["span"]) for e in primary if e.get("span")]
    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])
    for ent in secondary:
        span = ent.get("span")
        if span is None:
            result.append(ent); continue
        if any(overlaps(span, o) for o in occupied):
            continue
        result.append(ent); occupied.append(tuple(span))
    result.sort(key=lambda e: (e.get("span") or [0])[0])
    return result

def to_char_ents(entities):
    return [{"label": e["label"], "start": e["span"][0], "end": e["span"][1], "text": e.get("text", "")}
            for e in entities if e.get("span")]

def cascade_entities(query, stage, threshold):
    dict_tags = labeler.label_query(query)
    entities = bio_to_entities(dict_tags, query=query)
    if stage == "rules":
        return entities
    crf_ents = bio_to_entities(crf.predict_query(query), query=query)
    entities = merge_entities(entities, crf_ents)
    if stage == "rules_crf":
        return entities
    raw = gliner_model.predict_entities(query, gliner_labels(GLINER_SCHEME), threshold=threshold)
    gl_ents = gliner_preds_to_our_labels(raw, scheme=GLINER_SCHEME)
    gl_conv = [{"text": e["text"], "label": e["label"], "span": [e["start"], e["end"]]} for e in gl_ents]
    return merge_entities(entities, gl_conv)

def eval_cascade(rows, stage, threshold):
    gold_ents, pred_ents = [], []
    t0 = time.perf_counter()
    for r in rows:
        pred_ents.append(to_char_ents(cascade_entities(r["query"], stage, threshold)))
        gold_ents.append(gold_row_char_entities(r))
    m = span_prf1(gold_ents, pred_ents)
    m["avg_latency_ms"] = round((time.perf_counter() - t0) * 1000.0 / max(1, len(rows)), 2)
    return m
"""
)

md("## 5. Калибровка threshold GLiNER (на silver_val, как в notebooks/gliner/_run_02)")
code(
    """calib_rows = silver_val_rows[: min(150, len(silver_val_rows))]
best_thr, best_f1 = 0.5, -1.0
for thr in [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]:
    m = eval_cascade(calib_rows, "rules_crf_gliner", thr)
    print(f"thr={thr:.2f} microF1={m['micro']['f1']:.3f}")
    if m["micro"]["f1"] > best_f1:
        best_f1, best_thr = m["micro"]["f1"], thr
print("calibrated threshold:", best_thr)

gliner_model.save_pretrained(str(MODELS_DIR / "gliner_ner"))
"""
)

md("## 6. Метрика всего пайплайна: 3 стадии x 3 eval-сета")
code(
    """eval_sets = {"gold": gold_rows, "broken_queries": broken_rows, "silver_val": silver_val_rows}
results = {}
for name, rows in eval_sets.items():
    results[name] = {}
    for stage in STAGES:
        m = eval_cascade(rows, stage, best_thr)
        results[name][stage] = m
        print(f"[{name}/{stage}] n={len(rows)} F1={m['micro']['f1']:.3f} P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f}")

rows_summary = [
    {"eval_set": name, **{s: results[name][s]["micro"]["f1"] for s in STAGES}}
    for name in eval_sets
]
display(pd.DataFrame(rows_summary))
"""
)

md("## 7. Plot + save")
code(
    """fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(STAGES)); w = 0.25
colors = [MVIDEO_RED, DARK_SLATE, "#5B8C5A"]
for i, (name, color) in enumerate(zip(eval_sets, colors)):
    f1s = [results[name][s]["micro"]["f1"] for s in STAGES]
    ax.bar(x + (i - 1) * w, f1s, w, color=color, label=name)
ax.set_xticks(x); ax.set_xticklabels(STAGES, rotation=10); ax.set_ylim(0, 1.05)
ax.set_title("Метрика всего пайплайна: rules -> +CRF -> +GLiNER"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "02_pipeline_f1.png", dpi=120, bbox_inches="tight")
plt.show()

(METRICS_DIR / "general_study_pipeline.json").write_text(
    json.dumps({"gliner_threshold": best_thr, "results": results}, ensure_ascii=False, indent=2, default=str),
    encoding="utf-8",
)
print("saved metrics + models/general_study/*")
"""
)

md(
    """## Выводы (пример прогона — числа обновятся при перезапуске)

1. **CRF** обычно даёт наибольший прирост над `rules` — особенно на `broken_queries`
   (устойчивость к шуму за счёт контекстных/shape-фич, а не точного словарного совпадения).
2. **GLiNER не всегда помогает поверх rules+CRF**: смотри P/R по стадиям — если recall растёт,
   а precision падает сильнее, GLiNER на этом сете скорее вредит, чем помогает (типично для
   зашумлённого текста/собственного teacher-сигнала, где CRF уже близко к максимуму).
3. `silver_val` — самосогласованная (не внешняя) метрика: `rules` там даёт F1≈1.0 по
   построению (это же teacher). Ориентир — `gold` и `broken_queries`.
4. Обе модели обучены **без** gold — честный внешний тест.

Headless: `python notebooks/general_study/_run_02.py`
"""
)

NB.write_text(
    json.dumps(
        {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": cells,
        },
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)
print("wrote", NB, "cells", len(cells))
