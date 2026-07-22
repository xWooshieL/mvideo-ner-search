"""Smoke-run brand silver with NO_BRAND / UNKNOWN + inference policy."""
from __future__ import annotations

import ast
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    MUTED,
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    load_query_clicks,
    save_stats,
)
from src.preprocessing.pipeline import basic_clean, _norm_key  # noqa: E402

LOG = Path(__file__).resolve().parent / "_run_03_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_brand_aliases() -> dict:
    tree = ast.parse((ROOT / "src" / "ner" / "labeling.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "BRAND_ALIASES":
                    return ast.literal_eval(node.value)
    return {}


def stratified_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for _, g in df.groupby(label_col, sort=False):
        idx = np.array(g.index, dtype=np.int64, copy=True)
        rng.shuffle(idx)
        if len(idx) == 1:
            train_parts.append(g)
            continue
        n_val = max(1, int(round(len(idx) * test_size)))
        val_parts.append(g.loc[idx[:n_val]])
        train_parts.append(g.loc[idx[n_val:]])
    return (
        pd.concat(train_parts).sample(frac=1, random_state=seed),
        pd.concat(val_parts).sample(frac=1, random_state=seed),
    )


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    ensure_dirs()
    apply_plot_style()
    FIG = FIGURES_DIR / "preprocessing" / "brand_clf"
    OUT = ARTIFACTS_DIR / "brand_clf"
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    SAMPLE_N = 400_000
    SEED = 42
    TOP_K_BRANDS = 80
    MIN_CLICKS = 1
    MIN_CONFIDENCE = 0.55
    REQUIRE_SINGLE_BRAND = False
    LABEL_NO_BRAND = "NO_BRAND"
    LABEL_UNKNOWN = "UNKNOWN"
    HIGH_CONF = 0.75
    MED_CONF = 0.55
    NO_BRAND_MAX_TOKENS = 4
    NO_BRAND_TARGET_SHARE = 0.18
    UNKNOWN_FROM_TAIL = True
    KEEP_ALIAS_WITHOUT_SURFACE = True
    VAL_SIZE = 0.2
    MIN_PER_CLASS_TRAIN = 15
    TAU_ACCEPT = 0.42
    TAU_MARGIN = 0.08
    TAU_NO_BRAND = 0.35
    TAU_UNKNOWN = 0.30
    REQUIRE_BRAND_EVIDENCE = True

    BRAND_ALIASES = load_brand_aliases()
    log(f"load clicks n={SAMPLE_N}")
    clicks = load_query_clicks(
        n=SAMPLE_N,
        seed=SEED,
        random=True,
        columns=["query_text", "sku_brand_name", "sku_position", "sku_name"],
    )
    clicks["query_text"] = clicks["query_text"].fillna("").astype(str).str.strip()
    clicks["sku_brand_name"] = clicks["sku_brand_name"].fillna("").astype(str).str.strip()
    clicks["sku_position"] = (
        pd.to_numeric(clicks["sku_position"], errors="coerce").fillna(99).astype(int)
    )
    raw = clicks[
        (clicks["query_text"].str.len() >= 2) & (clicks["sku_brand_name"].str.len() >= 2)
    ].copy()
    log(f"raw={len(raw)} uq={raw['query_text'].nunique()}")

    uniq = raw["query_text"].drop_duplicates().tolist()
    norm_map = {q: _norm_key(basic_clean(q, lowercase=False)) for q in uniq}
    raw["query_norm"] = raw["query_text"].map(norm_map)

    rows = []
    for qn, g in raw.groupby("query_norm", sort=False):
        brands = g["sku_brand_name"].tolist()
        positions = g["sku_position"].tolist()
        raw_queries = g["query_text"].tolist()
        weights: dict[str, float] = defaultdict(float)
        counts: Counter = Counter()
        for b, pos in zip(brands, positions):
            w = 1.0 / (1.0 + max(int(pos), 0))
            weights[b] += w
            counts[b] += 1
        total_w = sum(weights.values())
        maj, maj_w = max(weights.items(), key=lambda kv: (kv[1], counts[kv[0]]))
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        runner = ranked[1][0] if len(ranked) > 1 else None
        runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
        rows.append(
            {
                "query_norm": qn,
                "query_raw": Counter(raw_queries).most_common(1)[0][0],
                "click_brand": maj,
                "confidence": maj_w / total_w if total_w else 0.0,
                "n_clicks": len(g),
                "n_brands": len(weights),
                "runner_up": runner,
                "weight_margin": (maj_w - runner_w) / total_w if total_w else 0.0,
            }
        )
    agg = pd.DataFrame(rows)
    log(f"agg={len(agg)} mean_conf={agg['confidence'].mean():.3f}")

    _alias_to_canon = {a.lower(): c for a, c in BRAND_ALIASES.items()}
    _canon_to_aliases: dict[str, set[str]] = defaultdict(set)
    for a, c in BRAND_ALIASES.items():
        _canon_to_aliases[c.lower()].add(a.lower())
        _canon_to_aliases[c.lower()].add(c.lower())

    cat_lines = [
        _norm_key(x)
        for x in (ARTIFACTS_DIR / "categories.txt").read_text(encoding="utf-8").splitlines()
        if len(_norm_key(x)) >= 3
    ]
    _cat_stop = {
        "для",
        "и",
        "с",
        "на",
        "по",
        "из",
        "или",
        "the",
        "a",
        "of",
        "to",
        "в",
        "без",
        "под",
        "над",
        "при",
        "от",
        "до",
        "как",
        "все",
        "всё",
    }
    CATEGORY_TOKENS: set[str] = set()
    for p in set(cat_lines):
        for t in p.split():
            if len(t) >= 3 and t not in _cat_stop:
                CATEGORY_TOKENS.add(t)
    log(f"categories phrases={len(set(cat_lines))} tokens={len(CATEGORY_TOKENS)}")
    ALIAS_KEYS = sorted(set(BRAND_ALIASES.keys()), key=len, reverse=True)

    def _word_in(q: str, key: str) -> bool:
        if len(key) < 2:
            return False
        return re.search(rf"(?<!\w){re.escape(key)}(?!\w)", q) is not None

    def brand_mentioned_in_text(query_norm: str, brand: str) -> bool:
        if not query_norm or not brand:
            return False
        q = query_norm.lower()
        b = brand.strip().lower()
        keys = set(_canon_to_aliases.get(b, set())) | {b}
        if b in _alias_to_canon:
            canon = _alias_to_canon[b].lower()
            keys |= _canon_to_aliases.get(canon, set()) | {canon}
        return any(_word_in(q, k) for k in keys)

    def any_alias_in_query(query_norm: str):
        q = (query_norm or "").lower()
        for a in ALIAS_KEYS:
            if _word_in(q, a.lower()):
                return True, _alias_to_canon.get(a.lower(), BRAND_ALIASES.get(a))
        return False, None

    def category_only_score(query_norm: str) -> dict:
        q = (query_norm or "").lower().strip()
        toks = [t for t in re.findall(r"[a-zа-яё0-9]+", q) if t]
        if not toks:
            return {"is_category_only": False, "has_modelish": False}
        has_alias, _ = any_alias_in_query(q)
        has_modelish = any(re.search(r"\d", t) for t in toks) or any(
            re.fullmatch(r"[a-z]{1,3}\d+[a-z0-9]*", t) for t in toks
        )
        cat_hits = sum(1 for t in toks if t in CATEGORY_TOKENS)
        cover = cat_hits / len(toks)
        is_co = (
            (not has_alias)
            and (not has_modelish)
            and len(toks) <= NO_BRAND_MAX_TOKENS
            and cover >= 0.6
        )
        return {"is_category_only": bool(is_co), "has_modelish": bool(has_modelish), "cat_cover": cover}

    brand_mass = agg.groupby("click_brand")["n_clicks"].sum().sort_values(ascending=False)
    top_brands = set(brand_mass.head(TOP_K_BRANDS).index)

    feat_rows = []
    for r in agg.itertuples(index=False):
        qn = r.query_norm
        has_alias, alias_canon = any_alias_in_query(qn)
        co = category_only_score(qn)
        feat_rows.append(
            {
                **r._asdict(),
                "brand_in_query": brand_mentioned_in_text(qn, r.click_brand),
                "has_alias": has_alias,
                "alias_canon": alias_canon,
                "is_category_only": co["is_category_only"],
                "has_modelish": co["has_modelish"],
                "click_in_topk": r.click_brand in top_brands,
            }
        )
    feat = pd.DataFrame(feat_rows)
    log(
        "evidence "
        + json.dumps(
            {
                "brand_in_query": float(feat["brand_in_query"].mean()),
                "has_alias": float(feat["has_alias"].mean()),
                "category_only": float(feat["is_category_only"].mean()),
            }
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    share = pd.Series(
        {
            "brand_in_query": feat["brand_in_query"].mean(),
            "has_alias": feat["has_alias"].mean(),
            "category_only": feat["is_category_only"].mean(),
        }
    )
    axes[0].bar(share.index, share.values, color=[DARK_SLATE, MVIDEO_RED, MUTED])
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)
    danger = feat[feat["is_category_only"]]
    if len(danger):
        axes[1].barh(
            danger["click_brand"].value_counts().head(12).index[::-1],
            danger["click_brand"].value_counts().head(12).values[::-1],
            color=MVIDEO_RED,
        )
    axes[1].set_title("click_brand on category-only")
    fig.tight_layout()
    fig.savefig(FIG / "03_brand_in_query.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    def assign_label(r):
        if r["is_category_only"]:
            return LABEL_NO_BRAND, "category_only"
        conf_ok = r["confidence"] >= MIN_CONFIDENCE and r["n_clicks"] >= MIN_CLICKS
        if REQUIRE_SINGLE_BRAND and r["n_brands"] > 1:
            conf_ok = conf_ok and r["confidence"] >= HIGH_CONF
        if KEEP_ALIAS_WITHOUT_SURFACE and r["has_alias"] and r["alias_canon"]:
            canon = r["alias_canon"]
            if canon in top_brands:
                return canon, "alias_hint"
        if r["brand_in_query"] and r["click_in_topk"] and conf_ok:
            return r["click_brand"], "brand_in_query"
        if UNKNOWN_FROM_TAIL and r["brand_in_query"] and (not r["click_in_topk"]) and conf_ok:
            return LABEL_UNKNOWN, "ood_brand_surface"
        if (not r["brand_in_query"]) and (not r["has_alias"]):
            if r["n_brands"] >= 3 and r["confidence"] < 0.5:
                return LABEL_UNKNOWN, "ambiguous_clicks"
            return None, "no_evidence_drop"
        if (r["brand_in_query"] or r["has_alias"]) and not conf_ok:
            return LABEL_UNKNOWN, "low_conf_with_evidence"
        return None, "other_drop"

    assigned = feat.apply(lambda r: assign_label(r), axis=1, result_type="expand")
    feat["brand"] = assigned[0]
    feat["label_reason"] = assigned[1]
    log("reasons " + json.dumps(feat["label_reason"].value_counts().to_dict(), ensure_ascii=False))

    silver = feat[feat["brand"].notna()].copy()
    silver = silver[silver["query_norm"].str.len().between(2, 120)].copy()
    nb = silver[silver["brand"] == LABEL_NO_BRAND]
    pos = silver[silver["brand"] != LABEL_NO_BRAND]
    target_nb = int(
        max(200, NO_BRAND_TARGET_SHARE * len(pos) / max(1e-6, 1 - NO_BRAND_TARGET_SHARE))
    )
    if len(nb) > target_nb:
        nb = nb.sample(n=target_nb, random_state=SEED)
    silver = pd.concat([pos, nb], ignore_index=True)
    unk = silver[silver["brand"] == LABEL_UNKNOWN]
    oth = silver[silver["brand"] != LABEL_UNKNOWN]
    max_unk = max(150, int(0.12 * len(oth)))
    if len(unk) > max_unk:
        unk = unk.sample(n=max_unk, random_state=SEED)
    silver = pd.concat([oth, unk], ignore_index=True)

    def tier_row(r) -> str:
        if r["brand"] in {LABEL_NO_BRAND, LABEL_UNKNOWN}:
            return "special"
        if r["confidence"] >= HIGH_CONF and (
            r["n_clicks"] >= 2 or r["brand_in_query"] or r["has_alias"]
        ):
            return "high"
        if r["confidence"] >= MED_CONF:
            return "medium"
        return "low"

    silver["tier"] = silver.apply(tier_row, axis=1)
    silver["sample_weight"] = np.where(
        silver["brand"] == LABEL_NO_BRAND,
        1.0,
        np.where(silver["brand"] == LABEL_UNKNOWN, 0.9, silver["confidence"].clip(0.35, 1.0)),
    )
    log(
        f"silver={len(silver)} no_brand={(silver['brand']==LABEL_NO_BRAND).mean():.3f} "
        f"unk={(silver['brand']==LABEL_UNKNOWN).mean():.3f}"
    )

    # fridge sanity
    fridge = silver[silver["query_norm"].str.contains("холодильник", na=False)]
    log("fridge_labels " + json.dumps(fridge["brand"].value_counts().head(5).to_dict(), ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    svc = silver["brand"].value_counts()
    show = list(svc.head(20).index)
    axes[0].barh([str(x) for x in show[::-1]], svc.loc[show].values[::-1], color=MVIDEO_RED)
    axes[0].set_title("Silver class counts")
    tier_vc = silver["tier"].value_counts()
    axes[1].bar(tier_vc.index.astype(str), tier_vc.values, color=DARK_SLATE)
    axes[1].set_title("Tiers")
    fig.tight_layout()
    fig.savefig(FIG / "04_silver_class_balance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # also keep 01/02 figs lightly
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vc = raw["sku_brand_name"].value_counts()
    axes[0].barh(vc.head(20).index[::-1], vc.head(20).values[::-1], color=MVIDEO_RED)
    axes[0].set_title("Top-20 brands by clicks")
    n_brands_per_q = raw.groupby("query_text")["sku_brand_name"].nunique()
    axes[1].hist(n_brands_per_q.clip(upper=6), bins=np.arange(0.5, 7.5, 1), color=DARK_SLATE)
    fig.tight_layout()
    fig.savefig(FIG / "01_raw_brand_dist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(agg["confidence"], bins=30, color=MVIDEO_RED, edgecolor="white")
    ax.axvline(MIN_CONFIDENCE, color=DARK_SLATE, ls="--")
    ax.set_title("Majority confidence")
    fig.tight_layout()
    fig.savefig(FIG / "02_majority_confidence.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    vc = silver["brand"].value_counts()
    keep = vc[vc >= max(MIN_PER_CLASS_TRAIN + 5, 20)].index
    for sp in (LABEL_NO_BRAND, LABEL_UNKNOWN):
        if sp in vc.index and vc[sp] >= MIN_PER_CLASS_TRAIN:
            keep = keep.union(pd.Index([sp]))
    data = silver[silver["brand"].isin(keep)].copy()
    train_df, val_df = stratified_split(data, "brand", VAL_SIZE, SEED)
    tr_vc = train_df["brand"].value_counts()
    ok = tr_vc[tr_vc >= MIN_PER_CLASS_TRAIN].index
    train_df = train_df[train_df["brand"].isin(ok)].copy()
    val_df = val_df[val_df["brand"].isin(ok)].copy()
    label_list = sorted(
        train_df["brand"].unique(), key=lambda x: (x in {LABEL_NO_BRAND, LABEL_UNKNOWN}, x)
    )
    label2id = {b: i for i, b in enumerate(label_list)}
    train_df["label_id"] = train_df["brand"].map(label2id)
    val_df["label_id"] = val_df["brand"].map(label2id)
    data = data[data["brand"].isin(ok)].copy()
    data["label_id"] = data["brand"].map(label2id)
    log(f"train={len(train_df)} val={len(val_df)} classes={len(label_list)}")

    cols = [
        "query_norm",
        "query_raw",
        "brand",
        "click_brand",
        "label_id",
        "label_reason",
        "confidence",
        "sample_weight",
        "n_clicks",
        "n_brands",
        "brand_in_query",
        "has_alias",
        "is_category_only",
        "tier",
        "runner_up",
        "weight_margin",
    ]
    inference_policy = {
        "cascade": [
            "1. If NER/dicts/alias yields BRAND → use it; do NOT call clf",
            "2. If category-only → brand=null (NO_BRAND); do NOT call clf",
            "3. Else if brand-evidence (alias or modelish) → call clf",
            "4. Else → brand=null; do NOT call clf",
        ],
        "thresholds": {
            "TAU_ACCEPT": TAU_ACCEPT,
            "TAU_MARGIN": TAU_MARGIN,
            "TAU_NO_BRAND": TAU_NO_BRAND,
            "TAU_UNKNOWN": TAU_UNKNOWN,
            "REQUIRE_BRAND_EVIDENCE": REQUIRE_BRAND_EVIDENCE,
        },
        "special_labels": {
            "NO_BRAND": "no brand implied; output brand=null",
            "UNKNOWN": "OOD / ambiguous; output brand=null + optional retrieval",
        },
        "accept_rules": [
            "pred in brands and P>=TAU_ACCEPT and margin>=TAU_MARGIN → accept",
            "pred==NO_BRAND and P>=TAU_NO_BRAND → brand=null",
            "pred==UNKNOWN and P>=TAU_UNKNOWN → brand=null (ood)",
            "else → reject brand=null",
        ],
    }
    stats = {
        "sample_n_clicks": int(SAMPLE_N),
        "seed": SEED,
        "top_k": TOP_K_BRANDS,
        "min_confidence": MIN_CONFIDENCE,
        "n_agg_queries": int(len(feat)),
        "n_silver": int(len(silver)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_classes": int(len(label_list)),
        "label_reason_counts": {str(k): int(v) for k, v in feat["label_reason"].value_counts().items()},
        "no_brand_share": float((silver["brand"] == LABEL_NO_BRAND).mean()),
        "unknown_share": float((silver["brand"] == LABEL_UNKNOWN).mean()),
        "category_only_in_agg": float(feat["is_category_only"].mean()),
        "thresholds": inference_policy["thresholds"],
        "has_no_brand": LABEL_NO_BRAND in label2id,
        "has_unknown": LABEL_UNKNOWN in label2id,
    }

    train_df[cols].to_parquet(OUT / "silver_brand_train.parquet", index=False)
    val_df[cols].to_parquet(OUT / "silver_brand_val.parquet", index=False)
    data[cols].to_parquet(OUT / "silver_brand_all.parquet", index=False)
    with open(OUT / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(i): b for b, i in label2id.items()},
                "special": [LABEL_NO_BRAND, LABEL_UNKNOWN],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(OUT / "inference_policy.json", "w", encoding="utf-8") as f:
        json.dump(inference_policy, f, ensure_ascii=False, indent=2)
    save_stats(stats, OUT / "silver_brand_stats.json")
    train_df[cols].head(500).to_csv(OUT / "silver_brand_train_preview.csv", index=False)
    log("DONE " + json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
