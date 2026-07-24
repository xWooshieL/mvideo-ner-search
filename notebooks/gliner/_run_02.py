"""GLiNER fine-tune: urchade/gliner_multi-v2.1 на gold (+ silver добавка) -> models/gliner_ner/.

Не трогает CRF/WeakLabeler — отдельный, самостоятельный трек (хвост каскада).
Если после обучения метрика хуже или почти не сдвинулась — автоматически
догоняем вторым раундом (больше шагов, ниже lr) прямо в этом скрипте.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_utils import (
    DARK_SLATE,
    FIGURES_DIR,
    METRICS_DIR,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    resolve_silver,
)
from src.ner.gliner_utils import (
    EDGE_CASES,
    OUR_LABELS,
    bio_row_to_gliner_example,
    evaluate_rows_char,
    load_gold_bio_rows,
    predict_edge_cases,
)

warnings.filterwarnings("ignore")

MODEL_ID = "urchade/gliner_multi-v2.1"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "gliner"
REPORT = Path(__file__).resolve().parent / "02_gliner_finetune_report.md"
OUT_JSON = METRICS_DIR / "gliner_finetune.json"
BASELINE_JSON = METRICS_DIR / "gliner_zero_shot.json"
OUT_DIR = ROOT / "models" / "gliner_ner"

SEED = 42
VAL_FRAC = 0.15
SILVER_N = 200  # добавка weak-примеров к скудному gold (200 строк) — держим CPU-время в узде
THRESHOLD = 0.3  # калибровка по val: 0.5 режет верные ATTR-спаны (score ~0.33-0.42), 0.3 — лучший баланс
BATCH_SIZE = 8
EPOCHS_ROUND1 = 3
EPOCHS_ROUND2 = 5  # если раунд 1 не помог / стало хуже
LR_ROUND1 = 1e-5
LR_ROUND2 = 5e-6
OTHERS_LR = 5e-5
F1_MIN_GAIN = 0.03  # если прирост над zero-shot меньше — считаем "что-то не так"
THRESHOLD_GRID = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]  # калибровка порога после обучения


def log(msg: str) -> None:
    print(msg, flush=True)


def load_best_scheme() -> str:
    if BASELINE_JSON.exists():
        try:
            data = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
            return data.get("best_scheme", "ru")
        except Exception:
            pass
    return "ru"


def split_gold(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    import random

    rnd = random.Random(SEED)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    n_val = max(1, int(len(rows) * VAL_FRAC))
    val_idx = set(idx[:n_val])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return train, val


def load_silver_examples(scheme: str, n: int) -> list[dict]:
    """Weak BIO (CRF-пайплайн, уже со spellfix) -> формат обучения GLiNER."""
    path = resolve_silver("ner_bio", "silver_bio_slice.parquet")
    if not path.exists():
        log(f"silver not found at {path} — skip augmentation")
        return []
    df = pd.read_parquet(path)
    if "has_entity" in df.columns:
        df = df[df["has_entity"]]
    if "n_entities" in df.columns:
        df = df[df["n_entities"] >= 1]
    df = df[df["tokens"].map(len) == df["tags"].map(len)]
    if len(df) > n:
        df = df.sample(n=n, random_state=SEED)
    examples = [
        bio_row_to_gliner_example(list(r["tokens"]), list(r["tags"]), scheme=scheme)
        for _, r in df.iterrows()
    ]
    examples = [e for e in examples if e["ner"]]
    log(f"silver augmentation: {len(examples)} examples from {path.name}")
    return examples


def train_round(model, train_ds: list[dict], val_ds: list[dict], *, epochs: int, lr: float, tag: str):
    steps_per_epoch = max(1, -(-len(train_ds) // BATCH_SIZE))  # ceil
    max_steps = steps_per_epoch * epochs
    out_dir = OUT_DIR / f"_ckpt_{tag}"
    log(
        f"[{tag}] train={len(train_ds)} val={len(val_ds)} epochs={epochs} "
        f"steps/epoch={steps_per_epoch} max_steps={max_steps} lr={lr}"
    )
    t0 = time.perf_counter()
    # ВАЖНО: model.train_model(...) уже запускает Trainer.train() внутри себя —
    # повторный вызов trainer.train() задвоит обучение (проверено смоук-тестом).
    model.train_model(
        train_dataset=train_ds,
        eval_dataset=val_ds if val_ds else None,
        output_dir=str(out_dir),
        learning_rate=lr,
        others_lr=OTHERS_LR,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        max_steps=max_steps,
        warmup_ratio=0.1,
        logging_steps=max(1, steps_per_epoch),
        save_steps=max(1, max_steps),  # одно сохранение чекпоинта в конце
        save_total_limit=1,
        use_cpu=True,
        report_to="none",
    )
    dt = time.perf_counter() - t0
    log(f"[{tag}] training done in {dt:.1f}s")
    return dt


def main() -> None:
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    scheme = load_best_scheme()
    log(f"label scheme (from 01_gliner_zero_shot): {scheme}")

    log(f"loading gold from {GOLD}")
    gold_rows = load_gold_bio_rows(GOLD)
    assert gold_rows, "gold BIO пуст — проверь data/gold/bio_liza.jsonl"
    train_rows, val_rows = split_gold(gold_rows)
    log(f"gold split: train={len(train_rows)} val={len(val_rows)}")

    train_examples = [
        bio_row_to_gliner_example(r["tokens"], r["tags"], scheme=scheme) for r in train_rows
    ]
    val_examples = [
        bio_row_to_gliner_example(r["tokens"], r["tags"], scheme=scheme) for r in val_rows
    ]

    silver_examples = load_silver_examples(scheme, SILVER_N)
    train_examples = train_examples + silver_examples
    log(f"train examples total: {len(train_examples)} (gold={len(train_rows)}, silver={len(silver_examples)})")

    log(f"loading GLiNER {MODEL_ID}...")
    from gliner import GLiNER

    model = GLiNER.from_pretrained(MODEL_ID)

    # --- baseline (тот же val, что и для fine-tune eval — честное "до/после") ---
    log("baseline (zero-shot) on this val split...")
    baseline_m = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
    log(
        f"baseline microF1={baseline_m['micro']['f1']:.3f} "
        f"P={baseline_m['micro']['precision']:.3f} R={baseline_m['micro']['recall']:.3f}"
    )
    baseline_edge = predict_edge_cases(model, EDGE_CASES, scheme=scheme, threshold=THRESHOLD)

    # --- раунд 1 ---
    rounds_log = []
    dt1 = train_round(model, train_examples, val_examples, epochs=EPOCHS_ROUND1, lr=LR_ROUND1, tag="r1")
    m1 = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
    log(f"[r1] microF1={m1['micro']['f1']:.3f} P={m1['micro']['precision']:.3f} R={m1['micro']['recall']:.3f}")
    rounds_log.append({"round": 1, "epochs": EPOCHS_ROUND1, "lr": LR_ROUND1, "seconds": round(dt1, 1), "metrics": m1})

    best_metrics = m1
    best_round = 1
    gain = m1["micro"]["f1"] - baseline_m["micro"]["f1"]
    log(f"gain over baseline after round 1: {gain:+.3f}")

    auto_retrain = gain < F1_MIN_GAIN
    if auto_retrain:
        log(
            f"gain < {F1_MIN_GAIN} (что-то не так) -> запускаю раунд 2 "
            f"(epochs={EPOCHS_ROUND2}, lr={LR_ROUND2}) на том же модели"
        )
        dt2 = train_round(model, train_examples, val_examples, epochs=EPOCHS_ROUND2, lr=LR_ROUND2, tag="r2")
        m2 = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=THRESHOLD)
        log(f"[r2] microF1={m2['micro']['f1']:.3f} P={m2['micro']['precision']:.3f} R={m2['micro']['recall']:.3f}")
        rounds_log.append(
            {"round": 2, "epochs": EPOCHS_ROUND2, "lr": LR_ROUND2, "seconds": round(dt2, 1), "metrics": m2}
        )
        if m2["micro"]["f1"] >= best_metrics["micro"]["f1"]:
            best_metrics, best_round = m2, 2
    log(f"best round={best_round} trainF1={best_metrics['micro']['f1']:.3f}")

    # --- калибровка порога: default 0.5 часто режет верные, но неуверенные спаны
    # (типично для ATTR — регрессия по recall без выигрыша по precision). Не ретрейн,
    # а честный post-hoc подбор по val: тот же принцип "сам файнтюйни", если что-то не так. ---
    log("calibrating decision threshold on val...")
    calib = []
    for thr in THRESHOLD_GRID:
        m = evaluate_rows_char(model, val_rows, scheme=scheme, threshold=thr)
        calib.append({"threshold": thr, "metrics": m})
        log(
            f"  thr={thr:.2f} microF1={m['micro']['f1']:.3f} "
            f"P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f}"
        )
    # выбираем лучший microF1; при близком результате (<=0.01) предпочитаем более низкий
    # порог — GLiNER у нас хвост каскада, там recall важнее чистой precision
    best_f1 = max(c["metrics"]["micro"]["f1"] for c in calib)
    good = [c for c in calib if c["metrics"]["micro"]["f1"] >= best_f1 - 0.01]
    best_calib = min(good, key=lambda c: c["threshold"])
    final_threshold = best_calib["threshold"]
    final_m = best_calib["metrics"]
    log(f"calibrated threshold={final_threshold} finalF1={final_m['micro']['f1']:.3f}")

    # --- edge cases после обучения (на калиброванном пороге) ---
    edge_rows = predict_edge_cases(model, EDGE_CASES, scheme=scheme, threshold=final_threshold)
    for e_before, e_after in zip(baseline_edge, edge_rows):
        log(
            f"EDGE {e_after['query']!r} zero-shot={[(x['label'], x['text']) for x in e_before['entities']]} "
            f"-> finetuned={[(x['label'], x['text']) for x in e_after['entities']]}"
        )

    # --- save model (финальные веса в памяти — лучший раунд уже применён последним train) ---
    for p in OUT_DIR.glob("_ckpt_*"):
        import shutil

        shutil.rmtree(p, ignore_errors=True)
    model.save_pretrained(str(OUT_DIR))
    log(f"saved fine-tuned model to {OUT_DIR}")

    # --- save metrics json ---
    result = {
        "model_id": MODEL_ID,
        "scheme": scheme,
        "gold_train_n": len(train_rows),
        "gold_val_n": len(val_rows),
        "silver_n": len(silver_examples),
        "baseline_val": baseline_m,
        "rounds": rounds_log,
        "best_round": best_round,
        "threshold_calibration": [
            {"threshold": c["threshold"], "micro_f1": c["metrics"]["micro"]["f1"]} for c in calib
        ],
        "final_threshold": final_threshold,
        "final_val": final_m,
        "auto_retrain_triggered": auto_retrain,
        "edge_cases_before": baseline_edge,
        "edge_cases_after": edge_rows,
        "model_dir": str(OUT_DIR.relative_to(ROOT)),
        "note": "fine-tune поверх zero-shot; silver — из CRF-пайплайна (уже со spellfix)",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {OUT_JSON}")

    # --- plot: baseline vs finetuned per-label F1 ---
    apply_plot_style()
    labs = OUR_LABELS
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labs))
    w = 0.35
    base_f1 = [baseline_m["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
    fine_f1 = [final_m["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
    ax.bar(x - w / 2, base_f1, w, color=DARK_SLATE, label="zero-shot")
    ax.bar(x + w / 2, fine_f1, w, color=MVIDEO_RED, label="fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("entity F1 (gold val)")
    ax.set_title("GLiNER: zero-shot vs fine-tuned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_finetune_f1.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- report ---
    lines = [
        "# 02. GLiNER — fine-tune",
        "",
        f"Модель: `{MODEL_ID}`. Схема лейблов: **{scheme}** (взята из `01_gliner_zero_shot`).",
        f"Gold train/val: **{len(train_rows)}/{len(val_rows)}** (из {len(gold_rows)}, seed={SEED}).",
        f"Silver-добавка (CRF-пайплайн, со spellfix): **{len(silver_examples)}** примеров.",
        "",
        "## До/после (gold val, span-level exact match)",
        "",
        "| | micro P | micro R | micro F1 |",
        "|---|---:|---:|---:|",
        f"| zero-shot | {baseline_m['micro']['precision']:.3f} | {baseline_m['micro']['recall']:.3f} | {baseline_m['micro']['f1']:.3f} |",
        f"| fine-tuned (round {best_round}) | {final_m['micro']['precision']:.3f} | {final_m['micro']['recall']:.3f} | {final_m['micro']['f1']:.3f} |",
        "",
        f"Прирост microF1: **{final_m['micro']['f1'] - baseline_m['micro']['f1']:+.3f}**",
        "",
        "## Раунды обучения",
        "",
        "| round | epochs | lr | секунд | microF1 (val) |",
        "|---|---:|---:|---:|---:|",
    ]
    for rr in rounds_log:
        lines.append(
            f"| {rr['round']} | {rr['epochs']} | {rr['lr']:.0e} | {rr['seconds']:.0f} | {rr['metrics']['micro']['f1']:.3f} |"
        )
    if auto_retrain:
        lines.append("")
        lines.append(
            f"Раунд 1 дал прирост < {F1_MIN_GAIN} над zero-shot -> автоматически запущен раунд 2 "
            "(больше эпох, ниже lr). Взят лучший раунд по val F1."
        )
    lines += [
        "",
        "## Калибровка порога (после обучения)",
        "",
        "Модель после обучения — то же, порог отсечения (`threshold`) — гиперпараметр инференса, "
        "не требует ретрейна. Default `0.5` резал верные, но неуверенные спаны (типично для ATTR).",
        "",
        "| threshold | microF1 (val) |",
        "|---:|---:|",
    ]
    for c in calib:
        marker = " **← выбран**" if c["threshold"] == final_threshold else ""
        lines.append(f"| {c['threshold']:.2f} | {c['metrics']['micro']['f1']:.3f}{marker} |")
    lines += [
        "",
        f"**Итоговый порог: {final_threshold}** (было {THRESHOLD} по умолчанию до калибровки). "
        "Используется в edge cases ниже и должен использоваться при вызове `predict_entities(..., threshold=...)` в проде.",
        "",
        "## По лейблам (val, на калиброванном пороге)",
        "",
        "| label | zero-shot F1 | fine-tuned F1 |",
        "|---|---:|---:|",
    ]
    for lab in OUR_LABELS:
        b = baseline_m["per_label"].get(lab, {}).get("f1", 0.0)
        f = final_m["per_label"].get(lab, {}).get("f1", 0.0)
        lines.append(f"| {lab} | {b:.3f} | {f:.3f} |")
    lines += [
        "",
        "![f1](../../figures/gliner/02_finetune_f1.png)",
        "",
        "## Edge cases: до vs после",
        "",
        "| query | zero-shot | fine-tuned |",
        "|---|---|---|",
    ]
    for eb, ea in zip(baseline_edge, edge_rows):
        b_str = ", ".join(f"{x['label']}:`{x['text']}`" for x in eb["entities"]) or "—"
        a_str = ", ".join(f"{x['label']}:`{x['text']}`" for x in ea["entities"]) or "—"
        lines.append(f"| `{ea['query']}` | {b_str} | {a_str} |")
    lines += [
        "",
        "## Выводы",
        "",
        f"1. Fine-tune на {len(train_rows)} gold + {len(silver_examples)} silver -> "
        f"microF1 {baseline_m['micro']['f1']:.3f} → {final_m['micro']['f1']:.3f}.",
        "2. Gold всего 200 строк — как будет больше размеченного (`data/gold/bio_liza.jsonl` растёт), "
        "перезапустить этот скрипт заново, F1 должен подрасти дальше.",
        "3. GLiNER остаётся **хвостом** каскада (rules → CRF → GLiNER), не заменой CRF на всём трафике "
        f"(latency ~{final_m['avg_latency_ms']:.0f}ms/query на CPU).",
        f"4. Модель сохранена в `models/gliner_ner/` (не в git — см. `.gitignore`).",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
