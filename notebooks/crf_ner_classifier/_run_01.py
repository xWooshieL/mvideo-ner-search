"""Headless CRF NER EDA: weak silver slice + gold parity + report."""
from __future__ import annotations

import json
import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import (
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    METRICS_DIR,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    load_query_clicks,
    save_stats,
    brands_path,
    categories_path,
    model_phrases_path,
    save_silver_parquet,
    SILVER_NER_BIO,
)
from src.ner.labeling import WeakLabeler, bio_to_entities, tokenize
from src.preprocessing.pipeline import basic_clean, _norm_key

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
SAMPLE_N = 60_000
MAX_QUERIES = 5_000
OUT = SILVER_NER_BIO
FIG = FIGURES_DIR / "ner"
REPORT = Path(__file__).resolve().parent / "01_crf_eda_report.md"
GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"


def log(msg: str) -> None:
    print(msg, flush=True)


def entity_counts(tags: list[str]) -> Counter:
    c: Counter = Counter()
    for t in tags:
        if t.startswith("B-"):
            c[t[2:]] += 1
    return c


def sent_to_row(query: str, tags_pairs: list[tuple[str, str]], source: str) -> dict:
    tokens = [t for t, _ in tags_pairs]
    tags = [g for _, g in tags_pairs]
    ents = bio_to_entities(tags_pairs, query=query)
    ec = entity_counts(tags)
    return {
        "query": query,
        "n_tokens": len(tokens),
        "tokens": tokens,
        "tags": tags,
        "n_entities": sum(1 for t in tags if t.startswith("B-")),
        "n_BRAND": ec["BRAND"],
        "n_CATEGORY": ec["CATEGORY"],
        "n_MODEL": ec["MODEL"],
        "n_ATTR": ec["ATTR"],
        "has_entity": any(t != "O" for t in tags),
        "bio_str": " ".join(f"{a}/{b}" for a, b in tags_pairs),
        "entities_json": json.dumps(
            [{"text": e["text"], "label": e["label"]} for e in ents],
            ensure_ascii=False,
        ),
        "source": source,
    }


def load_gold_rows() -> list[dict]:
    rows = []
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        q = r["query"]
        tags_split = r["tags"]
        toks_split = q.split()
        tok_spans = tokenize(q)
        toks_tok = [t for t, _, _ in tok_spans]
        split_ok = len(toks_split) == len(tags_split)
        tokenize_ok = len(toks_tok) == len(tags_split)
        rows.append(
            {
                "index": r.get("index"),
                "query": q,
                "tags": tags_split,
                "subtypes": r.get("subtypes") or {},
                "n_tokens_split": len(toks_split),
                "n_tags": len(tags_split),
                "n_tokens_tokenize": len(toks_tok),
                "split_align": split_ok,
                "tokenize_align": tokenize_ok,
                "tokens_tokenize": toks_tok,
            }
        )
    return rows


def bio_span_set(tokens: list[str], tags: list[str]) -> set[tuple[str, str]]:
    """{(label, span_text_lower)} from BIO."""
    out = set()
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            lab = tags[i][2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{lab}":
                j += 1
            out.add((lab, " ".join(tokens[i:j]).lower()))
            i = j
        else:
            i += 1
    return out


def main() -> None:
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    assert brands_path().exists()
    assert categories_path().exists()
    models_path = model_phrases_path()
    assert models_path.exists(), models_path
    assert GOLD.exists(), GOLD

    # --- labelers ---
    labeler_full = WeakLabeler.from_files(
        brands_path(),
        categories_path(),
        models_path=models_path,
    )
    labeler_legacy = WeakLabeler.from_files(
        brands_path(),
        categories_path(),
    )
    log(
        f"dicts brands={len(labeler_full.brands)} cats={len(labeler_full.categories)} "
        f"models={len(labeler_full.models)}"
    )

    # --- queries ---
    clicks = load_query_clicks()
    if len(clicks) > SAMPLE_N:
        clicks = clicks.sample(n=SAMPLE_N, random_state=SEED)
    qcol = "query_text" if "query_text" in clicks.columns else "query"
    queries = (
        clicks[qcol]
        .astype(str)
        .map(lambda x: basic_clean(x, lowercase=False))
        .map(_norm_key)
        .drop_duplicates()
    )
    queries = [q for q in queries.tolist() if len(q) >= 2][:MAX_QUERIES]
    log(f"unique queries for silver slice: {len(queries)}")

    # --- build silver with MODEL ---
    rows_full = []
    for i, q in enumerate(queries):
        if i and i % 500 == 0:
            log(f"  labeled {i}/{len(queries)}")
        sent = labeler_full.label_query(q)
        if not sent:
            continue
        rows_full.append(sent_to_row(q, sent, source="weak_with_model"))
    silver = pd.DataFrame(rows_full)
    silver_ent = silver[silver["has_entity"]].copy()
    log(f"silver rows={len(silver)} with_entity={len(silver_ent)}")

    # legacy without MODEL (A/B) — small subsample
    model_gain = 0
    legacy_model = 0
    for q in queries[: min(1000, len(queries))]:
        a = entity_counts([t for _, t in labeler_full.label_query(q)])
        b = entity_counts([t for _, t in labeler_legacy.label_query(q)])
        if a["MODEL"] > b["MODEL"]:
            model_gain += 1
        if b["MODEL"] > 0:
            legacy_model += 1
    log(f"A/B on 1k: full has more MODEL in {model_gain} queries; legacy MODEL hits={legacy_model}")

    # entity distribution
    type_counts = Counter()
    for _, r in silver_ent.iterrows():
        type_counts["BRAND"] += int(r["n_BRAND"])
        type_counts["CATEGORY"] += int(r["n_CATEGORY"])
        type_counts["MODEL"] += int(r["n_MODEL"])
        type_counts["ATTR"] += int(r["n_ATTR"])

    overview = pd.DataFrame(
        [
            {"metric": "queries_sampled", "value": len(queries)},
            {"metric": "silver_rows", "value": len(silver)},
            {"metric": "silver_with_entity", "value": len(silver_ent)},
            {"metric": "share_with_entity", "value": float(len(silver_ent) / max(1, len(silver)))},
            {"metric": "mean_tokens", "value": float(silver["n_tokens"].mean())},
            {"metric": "p95_tokens", "value": float(silver["n_tokens"].quantile(0.95))},
            {"metric": "n_BRAND", "value": type_counts["BRAND"]},
            {"metric": "n_CATEGORY", "value": type_counts["CATEGORY"]},
            {"metric": "n_MODEL", "value": type_counts["MODEL"]},
            {"metric": "n_ATTR", "value": type_counts["ATTR"]},
            {"metric": "model_gain_queries_4k", "value": model_gain},
        ]
    )
    overview.to_csv(OUT / "silver_overview.csv", index=False)
    save_silver_parquet(silver_ent.head(200), "ner_bio", "silver_bio_preview.parquet")
    # full slice for later train
    save_silver_parquet(silver, "ner_bio", "silver_bio_slice.parquet")
    log(f"saved {OUT / 'silver_bio_slice.parquet'} (+ legacy mirror)")

    # plots
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labs = ["BRAND", "CATEGORY", "MODEL", "ATTR"]
    vals = [type_counts[l] for l in labs]
    ax.bar(labs, vals, color=[MVIDEO_RED, DARK_SLATE, "#5B8C5A", "#C4A35A"])
    ax.set_title("Weak silver entity counts (with models_path)")
    ax.set_ylabel("B-* count")
    fig.tight_layout()
    fig.savefig(FIG / "01_entity_counts.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(silver["n_tokens"], bins=30, color=DARK_SLATE, edgecolor="white")
    ax.set_title("Tokens per query (silver slice)")
    ax.set_xlabel("n_tokens")
    fig.tight_layout()
    fig.savefig(FIG / "01_token_len.png", dpi=120, bbox_inches="tight")
    plt.close()

    # --- gold ---
    gold = load_gold_rows()
    gdf = pd.DataFrame(gold)
    split_ok = int(gdf["split_align"].sum())
    tok_ok = int(gdf["tokenize_align"].sum())
    gold_types = Counter()
    for tags in gdf["tags"]:
        gold_types.update(entity_counts(tags))

    # weak vs gold on gold queries (span-level)
    agree_tp = agree_fp = agree_fn = 0
    conf = Counter()
    for _, gr in gdf.iterrows():
        if not gr["split_align"]:
            continue
        q = gr["query"]
        gold_set = bio_span_set(q.split(), gr["tags"])
        pred_pairs = labeler_full.label_query(q)
        pred_toks = [t for t, _ in pred_pairs]
        pred_tags = [t for _, t in pred_pairs]
        # align if tokenize differs: compare on split tokens via re-label... use pred as-is
        # For fair compare when tokenize_align: use tokenize tokens + need gold tags on same toks
        if gr["tokenize_align"]:
            gold_set = bio_span_set(gr["tokens_tokenize"], gr["tags"])
            pred_set = bio_span_set(pred_toks, pred_tags)
        else:
            # fallback: weak on normalized query vs split gold (noisy)
            pred_set = bio_span_set(pred_toks, pred_tags)
            gold_set = bio_span_set(q.split(), gr["tags"])
        agree_tp += len(gold_set & pred_set)
        agree_fp += len(pred_set - gold_set)
        agree_fn += len(gold_set - pred_set)
        for lab, span in gold_set:
            # find teacher label for same span text if any
            hit = [p for p in pred_set if p[1] == span]
            if not hit:
                conf[(lab, "MISS")] += 1
            else:
                conf[(lab, hit[0][0])] += 1

    prec = agree_tp / max(1, agree_tp + agree_fp)
    rec = agree_tp / max(1, agree_tp + agree_fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    # samples: multi-entity, MODEL-heavy
    samples = silver_ent.sort_values("n_MODEL", ascending=False).head(8)
    hard = []
    for q in [
        "asus tuf gaming a15 16 гб",
        "ноутбук asus 16гб",
        "iphone 15 pro max",
        "беспроводные наушники sony",
        "телевизор tcl 65",
    ]:
        hard.append(
            {
                "query": q,
                "full": " ".join(f"{a}/{b}" for a, b in labeler_full.label_query(q)),
                "legacy": " ".join(f"{a}/{b}" for a, b in labeler_legacy.label_query(q)),
            }
        )

    meta = {
        "seed": SEED,
        "max_queries": MAX_QUERIES,
        "n_silver": len(silver),
        "n_with_entity": len(silver_ent),
        "entity_counts": dict(type_counts),
        "gold_n": len(gdf),
        "gold_split_align": split_ok,
        "gold_tokenize_align": tok_ok,
        "gold_entity_counts": dict(gold_types),
        "weak_vs_gold_span": {
            "tp": agree_tp,
            "fp": agree_fp,
            "fn": agree_fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        },
        "models_path": str(models_path),
        "note": "CRF features are handcrafted (not TF-IDF); silver is weak teacher",
    }
    (OUT / "eda_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_stats(
        {
            "silver_rows": len(silver),
            "with_entity": len(silver_ent),
            "gold_tokenize_align": tok_ok,
            "weak_gold_f1": round(f1, 4),
        },
        METRICS_DIR / "ner_eda_stats.json",
    )

    # report
    lines = [
        "# 01 CRF NER EDA report",
        "",
        "## MVP context",
        "",
        "- Brand clf + ATTR type clf — MVP готовы; CRF spans — следующий блок.",
        "- ATTR clf воспроизводим: `labeling.py` + markov `02/03` + `models/attr_type_clf.joblib`.",
        "- Этот EDA: качество **weak silver** для будущего `02_crf_classifier`.",
        "",
        "## Как классифицирует CRF (spoiler: не TF-IDF)",
        "",
        "| | Brand / ATTR-type clf | CRF NER |",
        "|---|---|---|",
        "| Объект | весь query / один ATTR-span | каждый токен в последовательности |",
        "| Фичи | TF-IDF char/word | lower, shape, prefix/suffix, ±соседи |",
        "| Модель | LogReg / SGD | sklearn_crfsuite CRF |",
        "| Опечатки | char n-grams частично ловят | слабо (нет edit-distance) |",
        "",
        "## Silver slice",
        "",
        f"- Queries: **{len(queries):,}** (cap `{MAX_QUERIES}`)",
        f"- Rows: **{len(silver):,}**, with ≥1 entity: **{len(silver_ent):,}** "
        f"({len(silver_ent)/max(1,len(silver)):.1%})",
        f"- Dicts: brands={len(labeler_full.brands)}, cats={len(labeler_full.categories)}, "
        f"models={len(labeler_full.models)}",
        f"- `models_path` ON → MODEL B-counts: **{type_counts['MODEL']}**; "
        f"legacy without models rarely emits MODEL (hits on 4k subsample: {legacy_model})",
        "",
        "| entity | B-* count |",
        "|---|---:|",
    ]
    for lab in labs:
        lines.append(f"| {lab} | {type_counts[lab]} |")
    lines += [
        "",
        f"![entities](../../figures/ner/01_entity_counts.png)",
        "",
        f"Mean tokens={silver['n_tokens'].mean():.1f}, p95={silver['n_tokens'].quantile(0.95):.0f}",
        "",
        "## Gold parity (`bio_liza.jsonl`)",
        "",
        f"- Queries: **{len(gdf)}**",
        f"- `split` align (tags vs `query.split()`): **{split_ok}/{len(gdf)}**",
        f"- `tokenize()` align (prod tokenizer): **{tok_ok}/{len(gdf)}**",
        "",
        "| gold entity | B-* |",
        "|---|---:|",
    ]
    for lab, n in gold_types.most_common():
        lines.append(f"| {lab} | {n} |")
    lines += [
        "",
        "CRF игнорирует `subtypes` (это attr-type слой).",
        "",
        "## Weak teacher vs gold (span micro)",
        "",
        f"- precision={prec:.3f} recall={rec:.3f} **F1={f1:.3f}** "
        f"(tp={agree_tp} fp={agree_fp} fn={agree_fn})",
        "- Это потолок silver-обучения: CRF учится копировать teacher, не gold.",
        "",
        "### Confusion gold_label → teacher (or MISS)",
        "",
        "| gold | teacher | n |",
        "|---|---|---:|",
    ]
    for (g, t), n in conf.most_common(20):
        lines.append(f"| {g} | {t} | {n} |")
    lines += [
        "",
        "## Hard examples (full vs legacy without MODEL)",
        "",
        "| query | with models_path | legacy (06/08 style) |",
        "|---|---|---|",
    ]
    for h in hard:
        lines.append(f"| `{h['query']}` | `{h['full']}` | `{h['legacy']}` |")
    lines += [
        "",
        "## Quality verdict",
        "",
        "1. Silver **можно** использовать для MVP CRF, но только с `models_path`.",
        "2. Старые 06/08 / `ner_crf.pkl` **не согласованы** с gold (нет MODEL).",
        "3. Gold tokenize mismatch нужно чинить или переразмечать под `tokenize()` перед hard eval.",
        "4. Val на silver↔silver будет завышен — в `02` обязателен gold F1.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT / 'silver_bio_slice.parquet'}`",
        f"- `{OUT / 'silver_bio_preview.parquet'}`",
        f"- `{OUT / 'eda_meta.json'}`",
        f"- figures: `figures/ner/01_*.png`",
        "",
        "Next: `02_crf_classifier.ipynb` — train CRF on this silver, eval on gold.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    log(f"DONE report={REPORT}")
    log(overview.to_string(index=False))


if __name__ == "__main__":
    main()
