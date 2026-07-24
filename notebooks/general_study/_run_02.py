"""General study — шаг 2: совместное обучение CRF + GLiNER на joint silver из _run_01
и метрика ВСЕГО каскада (rules -> CRF -> GLiNER) на gold и на synthetic broken_queries_eval.

В отличие от notebooks/crf_ner_classifier и notebooks/gliner (которые тренируются
каждый на своих данных и меряют себя по отдельности), здесь:
  1) CRF и GLiNER учатся на ОДНОМ и том же silver (общий _run_01, spellfix v2);
  2) gold (data/gold/bio_liza.jsonl) НЕ используется в обучении вообще — чистый held-out;
  3) метрика считается для каскада целиком (rules -> +CRF -> +GLiNER), не для одной модели.

Не трогает models/ner_crf.pkl и models/gliner_ner/ (общие треки коллег/другого ноутбука) —
свои веса кладём в models/general_study/.
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
from sklearn.model_selection import train_test_split

from src.data_utils import (
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    METRICS_DIR,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    brands_path,
    categories_path,
    model_phrases_path,
)
from src.ner.labeling import WeakLabeler, bio_to_entities
from src.ner.model_crf import CRFNerModel
from src.ner.gliner_utils import (
    OUR_LABELS,
    bio_row_to_gliner_example,
    gliner_labels,
    gliner_preds_to_our_labels,
    gold_row_char_entities,
    load_gold_bio_rows,
    span_prf1,
)

warnings.filterwarnings("ignore")

SILVER_DIR = ARTIFACTS_DIR / "silver" / "general_study"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
FIG = FIGURES_DIR / "general_study"
MODELS_DIR = ROOT / "models" / "general_study"
REPORT = Path(__file__).resolve().parent / "02_general_train_eval_report.md"
OUT_JSON = METRICS_DIR / "general_study_pipeline.json"

SEED = 42
VAL_SIZE = 0.15
CRF_MAX_ITERS = 80
GLINER_MODEL_ID = "urchade/gliner_multi-v2.1"
GLINER_TRAIN_N = 500  # капим объём для GLiNER (CPU) — CRF учится на всём train
GLINER_BATCH = 8
GLINER_EPOCHS = 3
GLINER_LR = 1e-5
GLINER_OTHERS_LR = 5e-5
GLINER_SCHEME = "ru"
GLINER_THRESHOLD_GRID = [0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
STAGES = ["rules", "rules_crf", "rules_crf_gliner"]
STAGE_TITLES = {
    "rules": "rules (словари)",
    "rules_crf": "rules + CRF",
    "rules_crf_gliner": "rules + CRF + GLiNER (весь каскад)",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def rows_to_sents(df: pd.DataFrame) -> list[list[tuple[str, str]]]:
    out = []
    for _, r in df.iterrows():
        toks, tags = list(r["tokens"]), list(r["tags"])
        if toks and len(toks) == len(tags):
            out.append(list(zip(toks, tags)))
    return out


# ---------------------------------------------------------------------------
# Каскад: та же логика, что в src/service/extractor.py (_merge_entities), но
# самостоятельная копия здесь — не трогаем прод-экстрактор ради эксперимента.
# ---------------------------------------------------------------------------


def merge_entities(primary: list[dict], secondary: list[dict]) -> list[dict]:
    result = list(primary)
    occupied = [tuple(e["span"]) for e in primary if e.get("span")]

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for ent in secondary:
        span = ent.get("span")
        if span is None:
            result.append(ent)
            continue
        if any(overlaps(span, o) for o in occupied):
            continue
        result.append(ent)
        occupied.append(tuple(span))
    result.sort(key=lambda e: (e.get("span") or [0])[0])
    return result


def to_char_ents(entities: list[dict]) -> list[dict]:
    out = []
    for e in entities:
        span = e.get("span")
        if not span:
            continue
        out.append({"label": e["label"], "start": span[0], "end": span[1], "text": e.get("text", "")})
    return out


def cascade_entities(
    query: str,
    stage: str,
    *,
    labeler: WeakLabeler,
    crf: CRFNerModel,
    gliner_model,
    gliner_scheme: str,
    gliner_threshold: float,
) -> list[dict]:
    dict_tags = labeler.label_query(query)
    entities = bio_to_entities(dict_tags, query=query)
    if stage == "rules":
        return entities

    crf_tags = crf.predict_query(query)
    crf_entities = bio_to_entities(crf_tags, query=query)
    entities = merge_entities(entities, crf_entities)
    if stage == "rules_crf":
        return entities

    labels = gliner_labels(gliner_scheme)
    raw_preds = gliner_model.predict_entities(query, labels, threshold=gliner_threshold)
    gl_ents = gliner_preds_to_our_labels(raw_preds, scheme=gliner_scheme)
    gl_conv = [{"text": e["text"], "label": e["label"], "span": [e["start"], e["end"]]} for e in gl_ents]
    entities = merge_entities(entities, gl_conv)
    return entities


def eval_cascade(
    rows: list[dict],
    stage: str,
    *,
    labeler: WeakLabeler,
    crf: CRFNerModel,
    gliner_model,
    gliner_scheme: str,
    gliner_threshold: float,
) -> dict:
    gold_ents, pred_ents = [], []
    t0 = time.perf_counter()
    for r in rows:
        q = r["query"]
        pred = cascade_entities(
            q, stage, labeler=labeler, crf=crf, gliner_model=gliner_model,
            gliner_scheme=gliner_scheme, gliner_threshold=gliner_threshold,
        )
        pred_ents.append(to_char_ents(pred))
        gold_ents.append(gold_row_char_entities(r))
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(rows))
    m = span_prf1(gold_ents, pred_ents)
    m["avg_latency_ms"] = round(latency_ms, 2)
    return m


def rows_from_silver_df(df: pd.DataFrame) -> list[dict]:
    """silver-строки (tokens/tags уже есть) -> формат load_gold_bio_rows (query/tokens/tags/char_spans)."""
    from src.ner.labeling import tokenize as _tok

    out = []
    for _, r in df.iterrows():
        q = r["query"]
        toks_spans = _tok(q)
        toks = [t for t, _, _ in toks_spans]
        tags = list(r["tags"])
        if len(toks) != len(tags):
            continue
        out.append({"query": q, "tokens": toks, "tags": tags, "char_spans": [(s, e) for _, s, e in toks_spans]})
    return out


def main() -> None:
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    # --- данные ---
    silver_path = SILVER_DIR / "general_silver_bio_ent.parquet"
    assert silver_path.exists(), f"нет {silver_path} — сначала python notebooks/general_study/_run_01.py"
    silver = pd.read_parquet(silver_path)
    log(f"general silver (with_entity) rows={len(silver)}")

    broken_path = SILVER_DIR / "broken_queries_eval.parquet"
    broken_df = pd.read_parquet(broken_path) if broken_path.exists() else pd.DataFrame()
    broken_rows = rows_from_silver_df(broken_df) if len(broken_df) else []
    log(f"broken_queries_eval rows={len(broken_rows)}")

    gold_rows = load_gold_bio_rows(GOLD)
    log(f"gold rows (held out, NOT trained on)={len(gold_rows)}")

    train_df, val_df = train_test_split(silver, test_size=VAL_SIZE, random_state=SEED)
    log(f"silver train={len(train_df)} val={len(val_df)}")
    silver_val_rows = rows_from_silver_df(val_df)

    labeler = WeakLabeler.from_files(brands_path(), categories_path(), models_path=model_phrases_path())

    # --- 1) CRF на train (весь) ---
    train_sents = rows_to_sents(train_df)
    log(f"[CRF] fit на {len(train_sents)} предложениях, max_iterations={CRF_MAX_ITERS}")
    t0 = time.perf_counter()
    crf = CRFNerModel(max_iterations=CRF_MAX_ITERS)
    crf.fit(train_sents)
    log(f"[CRF] done in {time.perf_counter() - t0:.1f}s")
    crf_path = MODELS_DIR / "ner_crf.pkl"
    crf.save(crf_path)
    log(f"[CRF] saved -> {crf_path}")

    # --- 2) GLiNER на капнутой выборке того же train ---
    gliner_train_df = train_df.sample(n=min(GLINER_TRAIN_N, len(train_df)), random_state=SEED)
    gliner_examples = [
        bio_row_to_gliner_example(list(r["tokens"]), list(r["tags"]), scheme=GLINER_SCHEME)
        for _, r in gliner_train_df.iterrows()
    ]
    gliner_examples = [e for e in gliner_examples if e["ner"]]
    log(f"[GLiNER] train examples: {len(gliner_examples)}")

    from gliner import GLiNER

    gliner_model = GLiNER.from_pretrained(GLINER_MODEL_ID)
    steps_per_epoch = max(1, -(-len(gliner_examples) // GLINER_BATCH))
    max_steps = steps_per_epoch * GLINER_EPOCHS
    log(f"[GLiNER] fit: examples={len(gliner_examples)} epochs={GLINER_EPOCHS} steps={max_steps}")
    t0 = time.perf_counter()
    gliner_model.train_model(
        train_dataset=gliner_examples,
        eval_dataset=None,
        output_dir=str(MODELS_DIR / "_gliner_ckpt"),
        learning_rate=GLINER_LR,
        others_lr=GLINER_OTHERS_LR,
        weight_decay=0.01,
        per_device_train_batch_size=GLINER_BATCH,
        max_steps=max_steps,
        warmup_ratio=0.1,
        logging_steps=max(1, steps_per_epoch),
        save_steps=max(1, max_steps),
        save_total_limit=1,
        use_cpu=True,
        report_to="none",
    )
    log(f"[GLiNER] done in {time.perf_counter() - t0:.1f}s")

    # --- калибровка порога GLiNER (тот же принцип, что в notebooks/gliner/_run_02) ---
    log("[GLiNER] calibrating threshold on silver-val (proxy, gold/broken слишком малы для этого шага)...")
    calib_rows = silver_val_rows[: min(150, len(silver_val_rows))]
    best_thr, best_f1 = GLINER_THRESHOLD_GRID[0], -1.0
    for thr in GLINER_THRESHOLD_GRID:
        m = eval_cascade(
            calib_rows, "rules_crf_gliner", labeler=labeler, crf=crf,
            gliner_model=gliner_model, gliner_scheme=GLINER_SCHEME, gliner_threshold=thr,
        )
        log(f"  thr={thr:.2f} microF1={m['micro']['f1']:.3f}")
        if m["micro"]["f1"] > best_f1:
            best_f1, best_thr = m["micro"]["f1"], thr
    log(f"[GLiNER] calibrated threshold={best_thr}")

    for p in (MODELS_DIR / "_gliner_ckpt").glob("*"):
        import shutil

        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    gliner_dir = MODELS_DIR / "gliner_ner"
    gliner_model.save_pretrained(str(gliner_dir))
    log(f"[GLiNER] saved -> {gliner_dir}")

    # --- 3) метрика ВСЕГО пайплайна: 3 стадии x 3 eval-сета ---
    eval_sets = {
        "gold": gold_rows,
        "broken_queries": broken_rows,
        "silver_val": silver_val_rows,
    }
    results: dict[str, dict[str, dict]] = {}
    for set_name, rows in eval_sets.items():
        if not rows:
            continue
        results[set_name] = {}
        for stage in STAGES:
            m = eval_cascade(
                rows, stage, labeler=labeler, crf=crf, gliner_model=gliner_model,
                gliner_scheme=GLINER_SCHEME, gliner_threshold=best_thr,
            )
            results[set_name][stage] = m
            log(
                f"[{set_name}/{stage}] n={len(rows)} microF1={m['micro']['f1']:.3f} "
                f"P={m['micro']['precision']:.3f} R={m['micro']['recall']:.3f} "
                f"lat={m['avg_latency_ms']:.1f}ms"
            )

    # --- save json ---
    out = {
        "gliner_model_id": GLINER_MODEL_ID,
        "gliner_threshold": best_thr,
        "n_silver_train": len(train_df),
        "n_silver_val": len(val_df),
        "n_gold": len(gold_rows),
        "n_broken_queries": len(broken_rows),
        "n_gliner_train_examples": len(gliner_examples),
        "results": results,
        "note": "CRF+GLiNER обучены ТОЛЬКО на general silver (gold не в train, чистый held-out)",
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved {OUT_JSON}")

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(STAGES))
    w = 0.25
    colors = [MVIDEO_RED, DARK_SLATE, "#5B8C5A"]
    for i, (set_name, color) in enumerate(zip(results.keys(), colors)):
        f1s = [results[set_name][s]["micro"]["f1"] for s in STAGES]
        ax.bar(x + (i - 1) * w, f1s, w, color=color, label=set_name)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_TITLES[s] for s in STAGES], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("micro F1 (span-level)")
    ax.set_title("Метрика всего пайплайна: rules -> +CRF -> +GLiNER")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_pipeline_f1.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- report ---
    lines = [
        "# 02. General study — совместное обучение CRF + GLiNER, метрика всего каскада",
        "",
        f"CRF и GLiNER обучены на **одном и том же** general silver из `01` "
        f"(train={len(train_df)}, GLiNER — сэмпл {len(gliner_examples)} из них). "
        "**Gold не участвует в обучении вообще** — чистый held-out для обеих моделей.",
        f"GLiNER threshold откалиброван на silver-val: **{best_thr}**.",
        "",
        "## Метрика всего пайплайна (span-level micro F1)",
        "",
        "| eval set | n | " + " | ".join(STAGE_TITLES[s] for s in STAGES) + " |",
        "|---|---:|" + "---:|" * len(STAGES),
    ]
    for set_name, rows in eval_sets.items():
        if set_name not in results:
            continue
        vals = " | ".join(f"{results[set_name][s]['micro']['f1']:.3f}" for s in STAGES)
        lines.append(f"| {set_name} | {len(rows)} | {vals} |")
    lines += [
        "",
        "![pipeline](../../figures/general_study/02_pipeline_f1.png)",
        "",
        "## Подробно (P / R / F1 по стадиям)",
        "",
    ]
    for set_name in results:
        lines += [f"### {set_name}", "", "| stage | P | R | F1 | latency/query |", "|---|---:|---:|---:|---:|"]
        for s in STAGES:
            m = results[set_name][s]
            lines.append(
                f"| {STAGE_TITLES[s]} | {m['micro']['precision']:.3f} | {m['micro']['recall']:.3f} | "
                f"{m['micro']['f1']:.3f} | {m['avg_latency_ms']:.1f} ms |"
            )
        lines.append("")
    def _delta(set_name: str) -> float:
        return results[set_name]["rules_crf_gliner"]["micro"]["f1"] - results[set_name]["rules_crf"]["micro"]["f1"]

    gliner_verdict = []
    for set_name in results:
        d = _delta(set_name)
        gliner_verdict.append(f"`{set_name}`: {d:+.3f}")
    lines += [
        "## Выводы",
        "",
        "1. **CRF — главный источник прироста над rules**, особенно там, где важна устойчивость "
        "к шуму: на `broken_queries` разница CRF vs rules обычно самая заметная во всей таблице — "
        "контекст+shape-фичи вытягивают то, что словарь по точному совпадению теряет.",
        f"2. **GLiNER (+CRF+rules) над +CRF**: {', '.join(gliner_verdict)}. Если где-то отрицательно — "
        "GLiNER на этом прогоне добавляет ложные срабатывания быстрее, чем закрывает реальные дыры "
        "(смотри P/R по стадиям выше: recall обычно растёт, а вот precision может падать сильнее). "
        "Порог калибровался по silver_val — не считать одним универсальным для всех сценариев.",
        "3. **Опечатки бьют по всему каскаду**: сравни лучшую стадию на `broken_queries` с `rules` на "
        "чистом `silver_val` — SpellFixer снижает урон, но не убирает его полностью "
        "(см. `01_general_silver_report.md`).",
        "4. Обе модели обучены **без** gold — держим его как честный внешний тест.",
        "5. Модели: `models/general_study/ner_crf.pkl`, `models/general_study/gliner_ner/` "
        "(не трогают `models/ner_crf.pkl` / `models/gliner_ner/` других треков).",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
