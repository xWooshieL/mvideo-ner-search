"""GLiNER zero-shot baseline: urchade/gliner_multi-v2.1 на нашем gold BIO + edge cases.

Не трогает CRF/WeakLabeler — отдельный, самостоятельный трек (хвост каскада).
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

from src.data_utils import (
    DARK_SLATE,
    FIGURES_DIR,
    METRICS_DIR,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
)
from src.ner.gliner_utils import (
    EDGE_CASES,
    OUR_LABELS,
    evaluate_rows_char,
    gliner_labels,
    load_gold_bio_rows,
    predict_edge_cases,
)

warnings.filterwarnings("ignore")

MODEL_ID = "urchade/gliner_multi-v2.1"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "gliner"
REPORT = Path(__file__).resolve().parent / "01_gliner_zero_shot_report.md"
OUT_JSON = METRICS_DIR / "gliner_zero_shot.json"
THRESHOLD = 0.3  # см. 02_gliner_finetune: 0.5 режет верные ATTR-спаны (score ~0.33-0.42)
SCHEMES = ["en", "ru"]


def log(msg: str) -> None:
    print(msg, flush=True)


def run_scheme(model, rows: list[dict], scheme: str) -> dict:
    return evaluate_rows_char(model, rows, scheme=scheme, threshold=THRESHOLD)


def main() -> None:
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading gold from {GOLD}")
    rows = load_gold_bio_rows(GOLD)
    log(f"gold rows usable (tokenize-aligned): {len(rows)}")
    assert rows, "gold BIO пуст или не читается — проверь data/gold/bio_liza.jsonl"

    log(f"loading GLiNER {MODEL_ID} (zero-shot, без обучения)...")
    t0 = time.perf_counter()
    from gliner import GLiNER

    model = GLiNER.from_pretrained(MODEL_ID)
    log(f"loaded in {time.perf_counter() - t0:.1f}s")

    scheme_metrics = {}
    for scheme in SCHEMES:
        log(f"--- scheme={scheme} labels={gliner_labels(scheme)} ---")
        m = run_scheme(model, rows, scheme)
        scheme_metrics[scheme] = m
        log(
            f"scheme={scheme} microF1={m['micro']['f1']:.3f} "
            f"P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f} "
            f"avg_latency={m['avg_latency_ms']:.1f}ms/query"
        )

    best_scheme = max(scheme_metrics, key=lambda s: scheme_metrics[s]["micro"]["f1"])
    log(f"best_scheme={best_scheme}")

    # --- edge cases (качественный разбор, best_scheme) ---
    edge_rows = predict_edge_cases(model, EDGE_CASES, scheme=best_scheme, threshold=THRESHOLD)
    for e in edge_rows:
        log(f"EDGE {e['query']!r} -> {[(x['label'], x['text']) for x in e['entities']]}")

    # --- save ---
    result = {
        "model_id": MODEL_ID,
        "threshold": THRESHOLD,
        "gold_n": len(rows),
        "schemes": scheme_metrics,
        "best_scheme": best_scheme,
        "edge_cases": edge_rows,
        "note": "zero-shot baseline (без fine-tune) — точка отсчёта для notebook 02",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {OUT_JSON}")

    # --- plot ---
    apply_plot_style()
    labs = OUR_LABELS
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(labs))
    w = 0.35
    en_f1 = [scheme_metrics["en"]["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
    ru_f1 = [scheme_metrics["ru"]["per_label"].get(l, {}).get("f1", 0.0) for l in labs]
    ax.bar(x - w / 2, en_f1, w, color=MVIDEO_RED, label="prompts=en")
    ax.bar(x + w / 2, ru_f1, w, color=DARK_SLATE, label="prompts=ru")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("entity F1 (gold, zero-shot)")
    ax.set_title(f"GLiNER {MODEL_ID} — zero-shot по лейбл-схемам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_zero_shot_f1.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- report ---
    lines = [
        "# 01. GLiNER — zero-shot baseline",
        "",
        f"Модель: `{MODEL_ID}` (multilingual, apache-2.0). Без обучения — то, что даёт HF-чекпоинт из коробки.",
        f"Gold: `data/gold/bio_liza.jsonl`, использовано **{len(rows)}** запросов.",
        "",
        "## Промпт-схемы (лейблы, которые видит GLiNER)",
        "",
        "| our label | en prompt | ru prompt |",
        "|---|---|---|",
    ]
    from src.ner.gliner_utils import LABEL_PROMPTS

    for l in OUR_LABELS:
        lines.append(f"| {l} | `{LABEL_PROMPTS['en'][l]}` | `{LABEL_PROMPTS['ru'][l]}` |")
    lines += [
        "",
        "## Метрики по схемам (span-level, exact match)",
        "",
        "| scheme | micro P | micro R | micro F1 | avg latency/query |",
        "|---|---:|---:|---:|---:|",
    ]
    for s, m in scheme_metrics.items():
        lines.append(
            f"| {s} | {m['micro']['precision']:.3f} | {m['micro']['recall']:.3f} | "
            f"{m['micro']['f1']:.3f} | {m['avg_latency_ms']:.1f} ms |"
        )
    lines += [
        "",
        f"**Лучшая схема: `{best_scheme}`** (используем в `02_gliner_finetune`).",
        "",
        f"| label | P | R | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    per = scheme_metrics[best_scheme]["per_label"]
    for lab in OUR_LABELS:
        if lab not in per:
            continue
        r = per[lab]
        lines.append(f"| {lab} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp'] + r['fn']} |")
    lines += [
        "",
        "![f1](../../figures/gliner/01_zero_shot_f1.png)",
        "",
        "## Edge cases (качественно, не в gold)",
        "",
        "| query | предсказанные сущности |",
        "|---|---|",
    ]
    for e in edge_rows:
        ents_str = ", ".join(f"{x['label']}:`{x['text']}`" for x in e["entities"]) or "—"
        lines.append(f"| `{e['query']}` | {ents_str} |")
    lines += [
        "",
        "## Выводы",
        "",
        "1. Это **точка отсчёта без обучения** — сравниваем с ней fine-tune в `02`.",
        "2. GLiNER — не замена CRF: тяжелее (~0.3B) и медленнее; роль — **хвост** каскада "
        "(дыры rules+CRF, новые типы), не каждый запрос.",
        "3. Дальше: `02_gliner_finetune.ipynb` — обучение на gold (+ по необходимости silver).",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
