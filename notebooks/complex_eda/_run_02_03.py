"""Smoke-run markov + click candidate generation."""
from __future__ import annotations

import importlib.util
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
    MVIDEO_RED,
    apply_plot_style,
    ensure_dirs,
    load_query_clicks,
    save_stats,
)
from src.ner.labeling import ATTR_PATTERNS as SRC_ATTR_PATTERNS  # noqa: E402
from src.ner.labeling import WeakLabeler  # noqa: E402

LOG = Path(__file__).resolve().parent / "_run_02_03_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    ensure_dirs()
    apply_plot_style()
    fig_dir = FIGURES_DIR / "complex_eda" / "markov"
    fig_dir.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location("temp_labeling", ROOT / "temp" / "labeling.py")
    temp_lab = importlib.util.module_from_spec(spec)
    sys.modules["temp_labeling"] = temp_lab
    spec.loader.exec_module(temp_lab)
    TEMP = temp_lab.ATTR_PATTERNS
    temp_tokenize = temp_lab.tokenize
    log(f"temp patterns={len(TEMP)} src={len(SRC_ATTR_PATTERNS)}")

    clicks = load_query_clicks(n=120_000, seed=42, random=True)
    ql = clicks["query_text"].astype(str).str.strip().str.lower()
    uq = ql.drop_duplicates()
    labeler = WeakLabeler.from_files(ARTIFACTS_DIR / "brands.txt", ARTIFACTS_DIR / "categories.txt")
    log(f"clicks={len(clicks)} uq={uq.nunique()}")

    def last_entity_tail(query: str):
        tags = labeler.label_query(query)
        last_ent = -1
        for i, (_, t) in enumerate(tags):
            if t != "O":
                last_ent = i
        if last_ent < 0 or last_ent >= len(tags) - 1:
            return []
        return [tok for tok, t in tags[last_ent + 1 :] if t == "O"]

    sample_q = uq.sample(n=min(6000, len(uq)), random_state=42).tolist()
    tail_toks = Counter()
    n_with_tail = n_digit = 0
    for q in sample_q:
        tail = last_entity_tail(q)
        if not tail:
            continue
        n_with_tail += 1
        for t in tail:
            tail_toks[t] += 1
        if any(ch.isdigit() for ch in " ".join(tail)):
            n_digit += 1
    log(f"tail_rate={n_with_tail/len(sample_q):.3f} digit_tail={n_digit/max(n_with_tail,1):.3f}")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    top = pd.DataFrame(tail_toks.most_common(20), columns=["token", "count"])
    ax.barh(top["token"][::-1], top["count"][::-1], color=MVIDEO_RED)
    ax.set_title("O-tokens after last entity")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_entity_tails.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    cov_q = uq.sample(n=min(20_000, len(uq)), random_state=0).tolist()

    def coverage(queries, patterns):
        cov = Counter()
        any_hit = 0
        for q in queries:
            hit = False
            for pat, name in patterns:
                if pat.search(q):
                    cov[name] += 1
                    hit = True
            if hit:
                any_hit += 1
        return any_hit / len(queries), cov

    any_src, _ = coverage(cov_q, SRC_ATTR_PATTERNS)
    any_tmp, cov_tmp = coverage(cov_q, TEMP)
    log(f"coverage src={any_src:.4f} temp={any_tmp:.4f}")

    tmp_share = pd.DataFrame(
        {"attr_type": list(cov_tmp.keys()), "share": [cov_tmp[k] / len(cov_q) for k in cov_tmp]}
    ).sort_values("share", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(tmp_share["attr_type"], tmp_share["share"], color=MVIDEO_RED)
    ax.tick_params(axis="x", rotation=55)
    ax.set_title("temp ATTR coverage")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_temp_attr_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    UNIT_RE = re.compile(r"^[а-яa-z°\"]{1,8}$", re.I)
    records = []
    for q in cov_q:
        for pat, typ in TEMP:
            for m in pat.finditer(q):
                toks = [t.lower().replace("ё", "е") for t, s, e in temp_tokenize(q) if not (e <= m.start() or s >= m.end())]
                if not toks:
                    toks = [t for t in re.split(r"\s+", m.group(0).lower()) if t]
                bigrams = list(zip(toks, toks[1:])) if len(toks) >= 2 else []
                units = [t for t in toks if UNIT_RE.match(t) and not t.isdigit()]
                records.append({"type": typ, "tokens": toks, "bigrams": bigrams, "units": units})
    log(f"spans={len(records)}")

    unit_to_types = defaultdict(Counter)
    bigram_to_types = defaultdict(Counter)
    trans = defaultdict(Counter)
    for r in records:
        for u in r["units"]:
            unit_to_types[u][r["type"]] += 1
        for bg in r["bigrams"]:
            bigram_to_types[bg][r["type"]] += 1
            trans[bg[0]][bg[1]] += 1

    rows = []
    for prev, nxts in trans.items():
        if not any(ch.isdigit() for ch in prev):
            continue
        for nxt, c in nxts.most_common(2):
            top_t = unit_to_types[nxt].most_common(1)[0][0] if unit_to_types[nxt] else "?"
            rows.append({"prev": prev, "next": nxt, "count": c, "top_type": top_t})
    trans_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    show = trans_df.head(15).copy()
    show["edge"] = show["prev"] + " → " + show["next"]
    ax.barh(show["edge"][::-1], show["count"][::-1], color=MVIDEO_RED)
    ax.set_title("Markov transitions after digits")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_markov_transitions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(0)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    cut = int(0.75 * len(idx))
    train_idx, test_idx = idx[:cut], idx[cut:]
    unit_to_types = defaultdict(Counter)
    bigram_to_types = defaultdict(Counter)
    for i in train_idx:
        r = records[i]
        for u in r["units"]:
            unit_to_types[u][r["type"]] += 1
        for bg in r["bigrams"]:
            bigram_to_types[bg][r["type"]] += 1

    def predict_type(tokens):
        toks = [t.lower().replace("ё", "е") for t in tokens]
        scores = Counter()
        for bg in zip(toks, toks[1:]):
            scores.update(bigram_to_types.get(bg, {}))
        if scores:
            return scores.most_common(1)[0][0]
        for t in toks:
            if unit_to_types[t]:
                scores.update(unit_to_types[t])
        return scores.most_common(1)[0][0] if scores else "unknown"

    y_true = [records[i]["type"] for i in test_idx]
    y_pred = [predict_type(records[i]["tokens"]) for i in test_idx]
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    unk = float(np.mean([p == "unknown" for p in y_pred]))
    log(f"markov_acc={acc:.3f} unk={unk:.3f}")

    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    top_types = pd.Series(y_true).value_counts().head(8).index.tolist() + ["unknown"]
    cm2 = cm.reindex(
        index=[t for t in top_types if t in cm.index],
        columns=[t for t in top_types if t in cm.columns],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm2.values, cmap="Reds")
    ax.set_xticks(range(cm2.shape[1]))
    ax.set_yticks(range(cm2.shape[0]))
    ax.set_xticklabels(cm2.columns, rotation=60, ha="right")
    ax.set_yticklabels(cm2.index)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_markov_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    num_types = defaultdict(Counter)
    for r in records:
        for t in r["tokens"]:
            if any(ch.isdigit() for ch in t):
                num_types[t][r["type"]] += 1
    amb = sum(1 for ctr in num_types.values() if len(ctr) >= 2 and sum(ctr.values()) >= 20)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ntypes = [len(ctr) for ctr in num_types.values() if len(ctr) >= 2 and sum(ctr.values()) >= 20]
    if ntypes:
        ax.hist(ntypes, bins=range(2, max(ntypes) + 2), color=DARK_SLATE, edgecolor="white")
    ax.set_title("Ambiguous number tokens")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_number_ambiguity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # click candidates
    label_dir = ARTIFACTS_DIR / "click_relevance"
    label_dir.mkdir(parents=True, exist_ok=True)
    df = clicks.copy()
    df["query"] = df["query_text"].astype(str).str.strip()
    df["brand"] = df["sku_brand_name"].astype(str).str.strip()
    q_freq = df["query"].str.lower().value_counts()
    frequent = set(q_freq[q_freq >= 15].index)
    sub = df[df["query"].str.lower().isin(frequent)].copy()
    top = sub[sub["sku_position"] <= 3].drop_duplicates(["query", "sku_id"]).sample(
        n=min(80, len(sub)), random_state=42
    )
    deep = sub[sub["sku_position"] >= 10].drop_duplicates(["query", "sku_id"])
    deep = deep.sample(n=min(60, len(deep)), random_state=1) if len(deep) else deep
    rng = np.random.default_rng(42)
    neg_rows = []
    queries_sample = sub["query"].drop_duplicates().sample(n=min(60, sub["query"].nunique()), random_state=2)
    sku_pool = sub[["sku_id", "sku_name", "brand", "sku_price", "sku_position"]].drop_duplicates("sku_id")
    for q in queries_sample:
        true_skus = set(sub.loc[sub["query"] == q, "sku_id"])
        cand = sku_pool[~sku_pool["sku_id"].isin(true_skus)]
        if cand.empty:
            continue
        row = cand.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        neg_rows.append(
            {
                "query": q,
                "sku_id": row["sku_id"],
                "sku_name": row["sku_name"],
                "sku_brand_name": row["brand"],
                "sku_position": row["sku_position"],
                "sku_price": row["sku_price"],
                "candidate_source": "random_negative",
            }
        )

    def pack(frame, source):
        out = frame[["query", "sku_id", "sku_name", "brand", "sku_position", "sku_price"]].copy()
        out = out.rename(columns={"brand": "sku_brand_name"})
        out["candidate_source"] = source
        return out

    cand = pd.concat([pack(top, "top_position"), pack(deep, "deep_position"), pd.DataFrame(neg_rows)], ignore_index=True)
    cand["pair_id"] = [str(abs(hash((str(q).lower(), str(s))))) for q, s in zip(cand["query"], cand["sku_id"])]
    for col, val in [("label", pd.NA), ("confidence", pd.NA), ("notes", ""), ("annotator", ""), ("labeled_at", "")]:
        cand[col] = val
    cand = cand.drop_duplicates("pair_id")
    cand_path = label_dir / "candidates_to_label.csv"
    cand.to_csv(cand_path, index=False, encoding="utf-8-sig")
    labels_path = label_dir / "labels.csv"
    if not labels_path.exists():
        cand.head(0).to_csv(labels_path, index=False, encoding="utf-8-sig")
    (label_dir / "LABELING_GUIDE.md").write_text(
        "# Click relevance\n\nFill `labels.csv` with label=0/1. See notebooks/complex_eda/03_click_eda.ipynb.\n",
        encoding="utf-8",
    )
    log(f"candidates={len(cand)} -> {cand_path}")

    save_stats(
        {
            "attr_coverage_src": any_src,
            "attr_coverage_temp": any_tmp,
            "n_attr_spans": len(records),
            "markov_test_accuracy_vs_regex": acc,
            "markov_unknown_rate": unk,
            "n_ambiguous_numbers": amb,
            "queries_with_entity_tail_rate": n_with_tail / len(sample_q),
            "tail_with_digit_rate": n_digit / max(n_with_tail, 1),
            "n_click_candidates": len(cand),
        },
        "markov_attr_eda_stats.json",
    )
    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        raise
