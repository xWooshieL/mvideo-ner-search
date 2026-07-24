"""Generate notebooks/gliner/02_gliner_finetune.ipynb"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "02_gliner_finetune.ipynb"
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
    """# 02. GLiNER — fine-tune

Дообучаем [`urchade/gliner_multi-v2.1`](https://huggingface.co/urchade/gliner_multi-v2.1) на нашем
gold (`data/gold/bio_liza.jsonl`) + добавка weak silver из CRF-пайплайна (уже со spellfix).

| | |
|---|---|
| База | zero-shot из [`01_gliner_zero_shot`](./01_gliner_zero_shot.ipynb) |
| Train | gold (85%) + silver добавка |
| Eval | gold val (15%), span-level P/R/F1, **тот же split** для "до/после" |
| Модель | `models/gliner_ner/` (не в git — большая, см. `.gitignore`) |

Если после раунда 1 прирост над zero-shot маленький — скрипт **сам** запускает
раунд 2 (больше эпох, ниже lr) и берёт лучший по val F1.

CLI: `python notebooks/gliner/_run_02.py` (CPU: раунд ~10-20 минут, без GPU — это ок, GLiNER тут хвост, не прод на каждый запрос).
"""
)

md("## 0. Setup")
code(
    """%matplotlib inline
import sys, json, time, warnings
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name in {"gliner", "notebooks"}:
    ROOT = ROOT.parents[1] if ROOT.name == "gliner" else ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_utils import (
    ensure_dirs, apply_plot_style, resolve_silver, FIGURES_DIR, METRICS_DIR,
    MVIDEO_RED, DARK_SLATE,
)
from src.ner.gliner_utils import (
    EDGE_CASES, OUR_LABELS, bio_row_to_gliner_example, evaluate_rows_char,
    load_gold_bio_rows, predict_edge_cases,
)

warnings.filterwarnings("ignore")
ensure_dirs()
apply_plot_style()

MODEL_ID = "urchade/gliner_multi-v2.1"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "gliner"; FIG.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "models" / "gliner_ner"; OUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_JSON = METRICS_DIR / "gliner_zero_shot.json"

SEED, VAL_FRAC, SILVER_N, THRESHOLD = 42, 0.15, 200, 0.3  # 0.3: калибровка по val (см. раздел 5.1)
BATCH_SIZE, EPOCHS_ROUND1, EPOCHS_ROUND2 = 8, 3, 5
LR_ROUND1, LR_ROUND2, OTHERS_LR = 1e-5, 5e-6, 5e-5
F1_MIN_GAIN = 0.03  # прирост меньше -> "что-то не так", авто-раунд 2

scheme = "ru"
if BASELINE_JSON.exists():
    scheme = json.loads(BASELINE_JSON.read_text(encoding="utf-8")).get("best_scheme", "ru")
print("label scheme:", scheme)
"""
)

md("## 1. Gold split + silver-добавка")
code(
    """import random
gold_rows = load_gold_bio_rows(GOLD)
rnd = random.Random(SEED)
idx = list(range(len(gold_rows))); rnd.shuffle(idx)
n_val = max(1, int(len(gold_rows) * VAL_FRAC))
val_idx = set(idx[:n_val])
train_rows = [r for i, r in enumerate(gold_rows) if i not in val_idx]
val_rows = [r for i, r in enumerate(gold_rows) if i in val_idx]
print(f"gold: train={len(train_rows)} val={len(val_rows)}")

train_examples = [bio_row_to_gliner_example(r["tokens"], r["tags"], scheme=scheme) for r in train_rows]
val_examples = [bio_row_to_gliner_example(r["tokens"], r["tags"], scheme=scheme) for r in val_rows]

silver_path = resolve_silver("ner_bio", "silver_bio_slice.parquet")
silver_examples = []
if silver_path.exists():
    sdf = pd.read_parquet(silver_path)
    if "has_entity" in sdf.columns:
        sdf = sdf[sdf["has_entity"]]
    sdf = sdf[sdf["tokens"].map(len) == sdf["tags"].map(len)]
    if len(sdf) > SILVER_N:
        sdf = sdf.sample(n=SILVER_N, random_state=SEED)
    silver_examples = [
        bio_row_to_gliner_example(list(r["tokens"]), list(r["tags"]), scheme=scheme)
        for _, r in sdf.iterrows()
    ]
    silver_examples = [e for e in silver_examples if e["ner"]]
train_examples = train_examples + silver_examples
print(f"train examples: {len(train_examples)} (gold={len(train_rows)}, silver={len(silver_examples)})")
"""
)

md("## 2. Baseline (zero-shot) на этом val")
code(
    """from gliner import GLiNER
model = GLiNER.from_pretrained(MODEL_ID)

baseline_m = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
print(f"baseline microF1={baseline_m['micro']['f1']:.3f} "
      f"P={baseline_m['micro']['precision']:.3f} R={baseline_m['micro']['recall']:.3f}")
baseline_edge = predict_edge_cases(model, EDGE_CASES, scheme=scheme, threshold=THRESHOLD)
"""
)

md(
    """## 3. Раунд 1 обучения

**Важно**: `model.train_model(...)` уже сам запускает `Trainer.train()` внутри —
повторно вызывать `trainer.train()` не нужно (задвоит обучение)."""
)
code(
    """def train_round(epochs, lr, tag):
    steps_per_epoch = max(1, -(-len(train_examples) // BATCH_SIZE))
    max_steps = steps_per_epoch * epochs
    print(f"[{tag}] train={len(train_examples)} epochs={epochs} steps={max_steps} lr={lr}")
    t0 = time.perf_counter()
    model.train_model(
        train_dataset=train_examples,
        eval_dataset=val_examples,
        output_dir=str(OUT_DIR / f"_ckpt_{tag}"),
        learning_rate=lr, others_lr=OTHERS_LR, weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        max_steps=max_steps, warmup_ratio=0.1, logging_steps=max(1, steps_per_epoch),
        save_steps=max(1, max_steps), save_total_limit=1, use_cpu=True, report_to="none",
    )
    print(f"[{tag}] done in {time.perf_counter() - t0:.1f}s")

train_round(EPOCHS_ROUND1, LR_ROUND1, "r1")
m1 = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
print(f"[r1] microF1={m1['micro']['f1']:.3f} P={m1['micro']['precision']:.3f} R={m1['micro']['recall']:.3f}")
gain = m1["micro"]["f1"] - baseline_m["micro"]["f1"]
print(f"gain over zero-shot: {gain:+.3f}")
"""
)

md(
    """## 4. Авто-раунд 2 (если "что-то не так")

Если прирост меньше порога — сам продолжаю обучение дольше и с меньшим lr,
беру лучший результат по val F1."""
)
code(
    """best_metrics, best_round = m1, 1
if gain < F1_MIN_GAIN:
    print(f"gain < {F1_MIN_GAIN} -> авто-раунд 2 (epochs={EPOCHS_ROUND2}, lr={LR_ROUND2})")
    train_round(EPOCHS_ROUND2, LR_ROUND2, "r2")
    m2 = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
    print(f"[r2] microF1={m2['micro']['f1']:.3f} P={m2['micro']['precision']:.3f} R={m2['micro']['recall']:.3f}")
    if m2["micro"]["f1"] >= best_metrics["micro"]["f1"]:
        best_metrics, best_round = m2, 2
print(f"best_round={best_round} trainF1={best_metrics['micro']['f1']:.3f}")
"""
)

md(
    """## 5. Калибровка порога ("что-то не так" — но не ретрейн)

Обученная модель может отдавать верный спан+лейбл с score чуть ниже дефолтного
`threshold=0.5` (типичная история для `ATTR`). Это гиперпараметр **инференса**,
не требует повторного обучения — подбираем его отдельно по val."""
)
code(
    """THRESHOLD_GRID = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
calib = []
for thr in THRESHOLD_GRID:
    m = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=thr)
    calib.append({"threshold": thr, "metrics": m})
    print(f"thr={thr:.2f} microF1={m['micro']['f1']:.3f} "
          f"P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f} "
          f"ATTR_f1={m['per_label'].get('ATTR', {}).get('f1', 0):.3f}")

best_f1 = max(c["metrics"]["micro"]["f1"] for c in calib)
good = [c for c in calib if c["metrics"]["micro"]["f1"] >= best_f1 - 0.01]
best_calib = min(good, key=lambda c: c["threshold"])  # при близком F1 берём ниже — recall важнее (хвост каскада)
final_threshold = best_calib["threshold"]
final_m = best_calib["metrics"]
print(f"calibrated threshold={final_threshold} finalF1={final_m['micro']['f1']:.3f}")
"""
)

md("## 6. Edge cases: до vs после (на калиброванном пороге)")
code(
    """edge_rows = predict_edge_cases(model, EDGE_CASES, scheme=scheme, threshold=final_threshold)
for eb, ea in zip(baseline_edge, edge_rows):
    print(f"{ea['query']!r}")
    print(f"  zero-shot : {[(x['label'], x['text']) for x in eb['entities']]}")
    print(f"  fine-tuned: {[(x['label'], x['text']) for x in ea['entities']]}")
"""
)

md("## 7. Plot + save")
code(
    """labs = OUR_LABELS
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(labs)); w = 0.35
base_f1 = [baseline_m["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
fine_f1 = [final_m["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
ax.bar(x - w/2, base_f1, w, color=DARK_SLATE, label="zero-shot")
ax.bar(x + w/2, fine_f1, w, color=MVIDEO_RED, label="fine-tuned")
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 1.05)
ax.set_title("GLiNER: zero-shot vs fine-tuned"); ax.legend()
fig.tight_layout()
fig.savefig(FIG / "02_finetune_f1.png", dpi=120, bbox_inches="tight")
plt.show()

import shutil
for p in OUT_DIR.glob("_ckpt_*"):
    shutil.rmtree(p, ignore_errors=True)
model.save_pretrained(str(OUT_DIR))
print("saved fine-tuned model to", OUT_DIR)

result = {
    "model_id": MODEL_ID, "scheme": scheme, "gold_train_n": len(train_rows),
    "gold_val_n": len(val_rows), "silver_n": len(silver_examples),
    "baseline_val": baseline_m, "best_round": best_round, "final_val": final_m,
    "edge_cases_before": baseline_edge, "edge_cases_after": edge_rows,
    "model_dir": str(OUT_DIR.relative_to(ROOT)),
}
(METRICS_DIR / "gliner_finetune.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("saved", METRICS_DIR / "gliner_finetune.json")
"""
)

md(
    """## Выводы

1. Смотрим **прирост microF1 над zero-shot** на одном и том же val-сплите — это честное "до/после".
2. Gold всего ~200 строк — когда разметки станет больше, перезапустить этот ноутбук/скрипт,
   F1 должен подрасти дальше.
3. GLiNER остаётся **хвостом** каскада (rules → CRF → GLiNER), не заменой CRF на всём трафике
   (CPU-latency заметно выше, чем у CRF).
4. Модель — в `models/gliner_ner/` (гитигнор, большая — safetensors).

Headless: `python notebooks/gliner/_run_02.py`
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
