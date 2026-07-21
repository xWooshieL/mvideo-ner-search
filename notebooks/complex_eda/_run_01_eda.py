"""Execute complex_eda analyses (mirror of 01_eda.ipynb) and save figures/stats."""
from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    ARTIFACTS_DIR,
    DARK_SLATE,
    FIGURES_DIR,
    MVIDEO_RED,
    QUERY_CLICKS_PATH,
    SKU_DESC_PATH,
    SKUS_PKL_PATH,
    apply_plot_style,
    ensure_dirs,
    load_query_clicks,
    load_sku_desc,
    parquet_num_rows,
    save_stats,
)
from src.ner.features import sent2labels  # noqa: E402
from src.ner.labeling import (  # noqa: E402
    ATTR_PATTERNS,
    WeakLabeler,
    bio_to_entities,
    entities_to_structured,
)
from src.ner.metrics import summarize_metrics  # noqa: E402
from src.ner.model_crf import CRFNerModel  # noqa: E402

LOG = Path(__file__).resolve().parent / "_run_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save_local(fig, fig_dir: Path, name: str) -> None:
    path = fig_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"saved {path}")


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    ensure_dirs()
    apply_plot_style()
    fig_dir = FIGURES_DIR / "complex_eda"
    fig_dir.mkdir(parents=True, exist_ok=True)

    SAMPLE_N = 150_000
    N_WEAK = 8_000
    log(f"ROOT={ROOT}")

    clicks = load_query_clicks(n=SAMPLE_N, seed=42, random=True)
    desc = load_sku_desc(n=40_000, seed=42, random=True)
    log(f"loaded clicks={len(clicks)} desc={len(desc)}")

    overview = {
        "query_clicks_full_rows": parquet_num_rows(QUERY_CLICKS_PATH),
        "sku_desc_full_rows": parquet_num_rows(SKU_DESC_PATH),
        "skus_pkl_bytes": SKUS_PKL_PATH.stat().st_size,
        "sample_rows": len(clicks),
    }
    log(json.dumps(overview, ensure_ascii=False))

    q = clicks["query_text"].astype(str).str.strip()
    ql = q.str.lower()
    brand = clicks["sku_brand_name"].astype(str).str.strip()
    brand_l = brand.str.lower()

    tok_n = ql.str.split().str.len()
    summary = {
        "unique_queries": int(ql.nunique()),
        "unique_skus": int(clicks["sku_id"].nunique()),
        "unique_brands": int(brand.replace("", np.nan).nunique()),
        "unique_subjects": int(clicks["sku_subject_id"].nunique()),
        "query_tokens_p50": float(tok_n.quantile(0.50)),
        "query_tokens_p90": float(tok_n.quantile(0.90)),
        "query_tokens_p99": float(tok_n.quantile(0.99)),
        "query_chars_p50": float(ql.str.len().quantile(0.50)),
        "query_chars_p90": float(ql.str.len().quantile(0.90)),
        "price_median": float(clicks["sku_price"].median()),
    }
    log("summary " + json.dumps(summary, ensure_ascii=False))

    mask = brand_l.str.len() >= 2
    hit = np.fromiter(
        (b in qq for qq, b in zip(ql[mask], brand_l[mask])),
        dtype=bool,
        count=int(mask.sum()),
    )
    brand_in = float(hit.mean())
    brand_abs = float((~hit).mean())
    log(f"brand_in_query={brand_in:.4f} absent={brand_abs:.4f}")

    # 1 lengths
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(tok_n.clip(upper=15), bins=range(1, 16), color=MVIDEO_RED, edgecolor="white")
    axes[0].axvline(tok_n.median(), color=DARK_SLATE, ls="--")
    axes[0].set_title("Длина запроса (токены)")
    axes[1].hist(ql.str.len().clip(upper=60), bins=30, color=DARK_SLATE, edgecolor="white", alpha=0.85)
    axes[1].set_title("Длина запроса (символы)")
    fig.tight_layout()
    save_local(fig, fig_dir, "01_query_length.png")

    # 2 attr regex
    labeler = WeakLabeler.from_files(ARTIFACTS_DIR / "brands.txt", ARTIFACTS_DIR / "categories.txt")
    cov: Counter[str] = Counter()
    any_attr = 0
    for qq in ql:
        hit_any = False
        for pat, name in ATTR_PATTERNS:
            if pat.search(qq):
                cov[name] += 1
                hit_any = True
        if hit_any:
            any_attr += 1
    attr_cov_df = pd.DataFrame(
        {"attr_type": list(cov.keys()), "share": [cov[k] / len(ql) for k in cov]}
    ).sort_values("share", ascending=False)
    log("attr_cov " + attr_cov_df.to_string(index=False))
    log(f"any_attr={any_attr/len(ql):.4f}")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(attr_cov_df["attr_type"], attr_cov_df["share"], color=MVIDEO_RED)
    ax.set_title("Покрытие regex-атрибутов")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    save_local(fig, fig_dir, "02_attr_regex_coverage.png")

    demo_q = "ноутбук asus 16 гб"
    tags = labeler.label_query(demo_q)
    ents = bio_to_entities(tags, query=demo_q)
    struct = entities_to_structured(ents, labeler=labeler)
    log(f"DEMO {tags} -> {struct}")

    # 3 weak BIO
    uq_sample = ql.drop_duplicates().head(N_WEAK).tolist()
    tag_hist: Counter[str] = Counter()
    n_with = n_brand = n_cat = n_attr = multi = n_i = 0
    for query in uq_sample:
        tags = labeler.label_query(query)
        labs = [t for _, t in tags if t != "O"]
        if labs:
            n_with += 1
        types = {t[2:] for t in labs if t.startswith(("B-", "I-"))}
        if "BRAND" in types:
            n_brand += 1
        if "CATEGORY" in types:
            n_cat += 1
        if "ATTR" in types:
            n_attr += 1
        if len(types) >= 2:
            multi += 1
        if any(t.startswith("I-") for _, t in tags):
            n_i += 1
        for _, t in tags:
            tag_hist[t] += 1
    m = len(uq_sample)
    weak = {
        "n": m,
        "any_entity": n_with / m,
        "brand": n_brand / m,
        "category": n_cat / m,
        "attr": n_attr / m,
        "multi_type": multi / m,
        "has_I": n_i / m,
    }
    log("weak " + json.dumps(weak, ensure_ascii=False))

    bio_non_o = pd.Series({k: v for k, v in tag_hist.items() if k != "O"}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(bio_non_o.index.astype(str), bio_non_o.values, color=MVIDEO_RED)
    ax.set_title("BIO classes (без O)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_local(fig, fig_dir, "03_bio_class_distribution.png")

    # 4 CRF smoke
    sents = labeler.label_dataset(uq_sample, min_entities=1)
    log(f"sents={len(sents)}")
    crf_report = {}
    if len(sents) >= 400:
        train, test = train_test_split(sents, test_size=0.25, random_state=42)
        train_small = train[: min(2000, len(train))]
        model = CRFNerModel(max_iterations=35)
        model.fit(train_small)
        yt = [sent2labels(s) for s in test]
        yp = model.predict(test)
        crf_report = summarize_metrics(yt, yp)
        log(
            "crf "
            + json.dumps(
                {
                    "token_accuracy": crf_report.get("token_accuracy"),
                    "micro_f1": (crf_report.get("micro") or {}).get("f1"),
                    "per_label": {
                        k: v.get("f1") for k, v in (crf_report.get("per_label") or {}).items()
                    },
                },
                ensure_ascii=False,
            )
        )
        log(f"crf demo {model.predict_query('пылесос dyson v15')}")

    # 5 brand classifier evidence
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    share = pd.Series({"brand_in_query": brand_in, "brand_absent": brand_abs})
    axes[0].bar(share.index, share.values, color=[MVIDEO_RED, DARK_SLATE])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Бренд в тексте vs только в клике")
    vc = brand.replace("", np.nan).dropna().value_counts()
    top = vc.head(15)
    axes[1].barh(top.index[::-1], top.values[::-1], color=MVIDEO_RED)
    axes[1].set_title("Топ брендов")
    fig.tight_layout()
    save_local(fig, fig_dir, "04_brand_in_query_vs_click.png")

    g = clicks.groupby(ql)["sku_brand_name"].nunique()
    amb_share = float((g >= 2).mean())
    log(f"ambiguous_queries_share={amb_share:.4f}")

    agg = (
        clicks.assign(query=ql, brand=brand)
        .loc[lambda d: d["brand"].str.len() >= 2]
        .groupby("query")["brand"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    top_brands = agg["brand"].value_counts().head(40).index
    agg40 = agg[agg["brand"].isin(top_brands)].copy()
    le = LabelEncoder()
    y = le.fit_transform(agg40["brand"])
    Xtr, Xte, ytr, yte = train_test_split(
        agg40["query"], y, test_size=0.2, random_state=42, stratify=y
    )
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=40_000)
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(vec.fit_transform(Xtr), ytr)
    pred = clf.predict(vec.transform(Xte))
    clf_acc = float(accuracy_score(yte, pred))
    clf_f1 = float(f1_score(yte, pred, average="macro"))
    log(f"clf_top40 accuracy={clf_acc:.3f} macroF1={clf_f1:.3f}")

    # 6 multi-head agreement sample
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(clicks), size=min(5000, len(clicks)), replace=False)
    agree = has_span = 0
    for i in sub_idx:
        query = str(clicks.iloc[i]["query_text"])
        b = str(clicks.iloc[i]["sku_brand_name"]).strip()
        if len(b) < 2:
            continue
        tags = labeler.label_query(query)
        ents = bio_to_entities(tags, query=query)
        brands = [e["text"] for e in ents if e["label"] == "BRAND"]
        if brands:
            has_span += 1
            agree += int(any(b.lower() in br.lower() or br.lower() in b.lower() for br in brands))
    log(f"span_click_agree={agree}/{has_span}")

    # 7 seq2seq pairs
    pairs = []
    for query in uq_sample[:2000]:
        tags = labeler.label_query(query)
        ents = bio_to_entities(tags, query=query)
        if not ents:
            continue
        pairs.append({"input": query, "output": entities_to_structured(ents, labeler=labeler)})
    log(f"seq2seq_pairs={len(pairs)}")

    # 8 retrieval
    sku_titles = (
        clicks[["sku_id", "sku_name", "sku_brand_name"]]
        .drop_duplicates("sku_id")
        .dropna(subset=["sku_name"])
        .head(10_000)
        .reset_index(drop=True)
    )
    eval_q = ql.value_counts().head(40).index.tolist()
    rvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    sku_m = rvec.fit_transform(sku_titles["sku_name"].astype(str))
    sims = cosine_similarity(rvec.transform(eval_q), sku_m)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(sims.max(axis=1), bins=15, color=MVIDEO_RED, edgecolor="white")
    ax.set_title("TF-IDF retrieval top-1 scores")
    fig.tight_layout()
    save_local(fig, fig_dir, "05_retrieval_scores.png")

    # 9 position/price
    pos = clicks["sku_position"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(pos.clip(upper=40), bins=40, color=DARK_SLATE, edgecolor="white")
    axes[0].set_title("Позиция клика")
    sub = clicks[(clicks["sku_price"] > 0) & (clicks["sku_position"] <= 30)]
    sub = sub[sub["sku_price"] < sub["sku_price"].quantile(0.95)]
    axes[1].hexbin(sub["sku_position"], sub["sku_price"], gridsize=30, cmap="Reds", mincnt=3)
    axes[1].set_title("Цена × позиция")
    fig.tight_layout()
    save_local(fig, fig_dir, "06_position_price.png")

    out = {
        "overview": overview,
        "summary": summary,
        "brand_in_query_rate": brand_in,
        "brand_absent_rate": brand_abs,
        "weak_coverage": weak,
        "attr_regex_any": any_attr / len(ql),
        "attr_regex_by_type": {k: cov[k] / len(ql) for k in cov},
        "ambiguous_query_share": amb_share,
        "clf_top40": {"accuracy": clf_acc, "macro_f1": clf_f1},
        "span_click_agree": {"agree": agree, "has_span": has_span},
        "seq2seq_pairs": len(pairs),
        "crf": {
            "token_accuracy": crf_report.get("token_accuracy"),
            "micro_f1": (crf_report.get("micro") or {}).get("f1"),
        },
    }
    save_stats(out, "complex_eda_method_stats.json")
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
