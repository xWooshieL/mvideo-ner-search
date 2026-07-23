"""Train CRF NER on silver/ner_bio; eval on silver-val + gold; save models/ner_crf.pkl."""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_utils import (
    DARK_SLATE,
    FIGURES_DIR,
    MODELS,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    resolve_silver,
    save_stats,
)
from src.ner.labeling import tokenize
from src.ner.metrics import summarize_metrics
from src.ner.model_crf import CRFNerModel

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
VAL_SIZE = 0.2
MAX_ITERS = 80
MIN_ENTITIES = 1  # keep sentences with ≥1 B-*
REPORT = Path(__file__).resolve().parent / "02_crf_report.md"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
OUT = ROOT / "artifacts" / "ner"
FIG = FIGURES_DIR / "ner"
LABELS = ["BRAND", "CATEGORY", "MODEL", "ATTR"]


def log(msg: str) -> None:
    print(msg, flush=True)


def rows_to_sents(df: pd.DataFrame) -> list[list[tuple[str, str]]]:
    sents = []
    for _, r in df.iterrows():
        toks = list(r["tokens"])
        tags = list(r["tags"])
        if len(toks) != len(tags) or not toks:
            continue
        sents.append(list(zip(toks, tags)))
    return sents


def load_gold_sents() -> tuple[list[list[tuple[str, str]]], dict]:
    """Gold BIO aligned to prod tokenize() when possible."""
    meta = {"n": 0, "tokenize_align": 0, "used": 0, "skipped": 0}
    sents = []
    if not GOLD.exists():
        return sents, meta
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        meta["n"] += 1
        q = r["query"]
        tags = r["tags"]
        toks_tok = [t for t, _, _ in tokenize(q)]
        if len(toks_tok) == len(tags):
            meta["tokenize_align"] += 1
            sents.append(list(zip(toks_tok, tags)))
            meta["used"] += 1
        else:
            # fallback: whitespace split (app tokenization)
            toks_split = q.split()
            if len(toks_split) == len(tags):
                sents.append(list(zip(toks_split, tags)))
                meta["used"] += 1
            else:
                meta["skipped"] += 1
    return sents, meta


def entity_counts_from_sents(sents: list[list[tuple[str, str]]]) -> Counter:
    c: Counter = Counter()
    for sent in sents:
        for _, t in sent:
            if t.startswith("B-"):
                c[t[2:]] += 1
    return c


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    silver_path = resolve_silver("ner_bio", "silver_bio_slice.parquet")
    assert silver_path.exists(), silver_path
    df = pd.read_parquet(silver_path)
    log(f"silver={silver_path} rows={len(df)}")

    if "has_entity" in df.columns:
        df = df[df["has_entity"]].copy()
    # filter min entities
    df = df[df["n_entities"] >= MIN_ENTITIES].copy() if "n_entities" in df.columns else df
    log(f"train pool with entities: {len(df)}")

    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=SEED)
    train_sents = rows_to_sents(train_df)
    val_sents = rows_to_sents(val_df)
    log(f"sents train={len(train_sents)} val={len(val_sents)}")
    log(f"train B-* {dict(entity_counts_from_sents(train_sents))}")

    # --- train ---
    log(f"fit CRF max_iterations={MAX_ITERS}")
    model = CRFNerModel(max_iterations=MAX_ITERS)
    model.fit(train_sents)

    y_val = [[t for _, t in s] for s in val_sents]
    pred_val = model.predict(val_sents)
    silver_m = summarize_metrics(y_val, pred_val)
    log(
        f"silver-val microF1={silver_m['micro']['f1']:.3f} "
        f"tokAcc={silver_m['token_accuracy']:.3f} macroF1={silver_m['macro_f1']:.3f}"
    )

    # --- gold ---
    gold_sents, gold_meta = load_gold_sents()
    gold_m = None
    if gold_sents:
        y_gold = [[t for _, t in s] for s in gold_sents]
        # predict from tokens only (same tokenization as gold sents)
        pred_gold = model.predict_tokens([[t for t, _ in s] for s in gold_sents])
        gold_m = summarize_metrics(y_gold, pred_gold)
        log(
            f"gold microF1={gold_m['micro']['f1']:.3f} "
            f"tokAcc={gold_m['token_accuracy']:.3f} "
            f"used={gold_meta['used']}/{gold_meta['n']} tokenize_align={gold_meta['tokenize_align']}"
        )

    # --- save ---
    model_path = MODELS / "ner_crf.pkl"
    model.save(model_path)
    # also versioned
    model.save(MODELS / "ner_crf__silver_v1.pkl")
    log(f"saved {model_path}")

    metrics = {
        "silver_path": str(silver_path),
        "n_train": len(train_sents),
        "n_val": len(val_sents),
        "max_iterations": MAX_ITERS,
        "seed": SEED,
        "labels": LABELS,
        "silver_val": silver_m,
        "gold": gold_m,
        "gold_meta": gold_meta,
        "train_entity_counts": dict(entity_counts_from_sents(train_sents)),
        "model_path": str(model_path),
        "note": "silver-val is weak↔weak (optimistic); gold is primary for MVP",
    }
    (OUT / "crf_train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # mirror under silver ner_bio
    from src.data_utils import SILVER_NER_BIO

    SILVER_NER_BIO.mkdir(parents=True, exist_ok=True)
    (SILVER_NER_BIO / "crf_train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_stats(
        {
            "ner_crf_silver_micro_f1": round(silver_m["micro"]["f1"], 4),
            "ner_crf_gold_micro_f1": round(gold_m["micro"]["f1"], 4) if gold_m else None,
            "n_train": len(train_sents),
        },
        name="ner_crf_metrics.json",
    )

    # --- plots ---
    per = silver_m["per_label"]
    labs = [l for l in LABELS if l in per]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(labs, [per[l]["f1"] for l in labs], color=MVIDEO_RED, label="silver-val")
    if gold_m:
        gp = gold_m["per_label"]
        x = np.arange(len(labs))
        w = 0.35
        ax.clear()
        ax.bar(x - w / 2, [per[l]["f1"] for l in labs], w, color=MVIDEO_RED, label="silver-val")
        ax.bar(
            x + w / 2,
            [gp.get(l, {}).get("f1", 0.0) for l in labs],
            w,
            color=DARK_SLATE,
            label="gold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("entity F1")
    ax.set_title("CRF NER entity F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_crf_entity_f1.png", dpi=120, bbox_inches="tight")
    plt.close()

    # sanity demos
    demos = [
        "asus tuf gaming a15 16 гб",
        "ноутбук asus 16 гб",
        "iphone 15 pro max",
        "беспроводные наушники sony",
    ]
    demo_rows = []
    for q in demos:
        pred = model.predict_query(q)
        demo_rows.append({"query": q, "bio": " ".join(f"{a}/{b}" for a, b in pred)})
        log(f"DEMO {q} -> {demo_rows[-1]['bio']}")

    # --- report ---
    lines = [
        "# 02 CRF NER — train report",
        "",
        f"Model: `{model_path.relative_to(ROOT)}`  ",
        f"Silver: `{silver_path}`  ",
        f"Train/val: **{len(train_sents)}** / **{len(val_sents)}** (seed={SEED})",
        "",
        "## Features (not TF-IDF)",
        "",
        "Per-token: `word.lower`, prefix/suffix, shape, digit/latin/cyrillic, ±1/±2 neighbors, BOS/EOS.",
        "Typos: weak (no edit-distance).",
        "",
        "## Silver-val (weak↔weak, optimistic)",
        "",
        f"- token accuracy: **{silver_m['token_accuracy']:.3f}**",
        f"- entity micro-F1: **{silver_m['micro']['f1']:.3f}** "
        f"(P={silver_m['micro']['precision']:.3f} R={silver_m['micro']['recall']:.3f})",
        f"- macro-F1: **{silver_m['macro_f1']:.3f}**",
        "",
        "| label | P | R | F1 | support |",
        "|---|---:|---:|---:|---:|",
    ]
    for lab in LABELS:
        if lab not in per:
            continue
        r = per[lab]
        lines.append(
            f"| {lab} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |"
        )
    if gold_m:
        lines += [
            "",
            "## Gold (`bio_liza.jsonl`) — primary MVP metric",
            "",
            f"- used **{gold_meta['used']}/{gold_meta['n']}** "
            f"(tokenize_align={gold_meta['tokenize_align']}, skipped={gold_meta['skipped']})",
            f"- token accuracy: **{gold_m['token_accuracy']:.3f}**",
            f"- entity micro-F1: **{gold_m['micro']['f1']:.3f}** "
            f"(P={gold_m['micro']['precision']:.3f} R={gold_m['micro']['recall']:.3f})",
            f"- macro-F1: **{gold_m['macro_f1']:.3f}**",
            "",
            "| label | P | R | F1 | support |",
            "|---|---:|---:|---:|---:|",
        ]
        gp = gold_m["per_label"]
        for lab in LABELS:
            if lab not in gp:
                continue
            r = gp[lab]
            lines.append(
                f"| {lab} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['support']} |"
            )
    lines += [
        "",
        "![f1](../../figures/ner/02_crf_entity_f1.png)",
        "",
        "## Demos",
        "",
        "| query | BIO |",
        "|---|---|",
    ]
    for d in demo_rows:
        lines.append(f"| `{d['query']}` | `{d['bio']}` |")
    lines += [
        "",
        "## Notes",
        "",
        "1. Silver includes **MODEL** (`models_path`); old 06/08 did not.",
        "2. Trust **gold** more than silver-val.",
        "3. Expand `silver_bio_slice` (more queries) before claiming prod-ready F1.",
        "",
        "Artifacts: `models/ner_crf.pkl`, `artifacts/ner/crf_train_metrics.json`.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")


if __name__ == "__main__":
    main()
