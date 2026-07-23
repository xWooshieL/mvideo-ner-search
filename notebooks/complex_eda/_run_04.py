"""Smoke-run for 04_model_tag_eda figures + model_phrases thresholds."""
from __future__ import annotations

import sys
from collections import Counter
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
from src.ner.labeling import WeakLabeler  # noqa: E402
from src.preprocessing import (  # noqa: E402
    QueryPreprocessor,
    build_model_lexicon_from_titles,
    save_phrase_list,
)
from src.preprocessing.pipeline import MODEL_SEEDS  # noqa: E402

LOG = Path(__file__).resolve().parent / "_run_04_log.txt"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def brand_latin_o_tail(tags):
    brand_idx = [i for i, (_, t) in enumerate(tags) if t.endswith("BRAND")]
    if not brand_idx:
        return None
    last = max(brand_idx)
    tail = tags[last + 1 :]
    if not tail or not all(t == "O" for _, t in tail):
        return None
    toks = [tok for tok, _ in tail]
    ascii_share = np.mean([tok.isascii() and tok.isalnum() for tok in toks])
    if ascii_share < 0.6:
        return None
    return toks


def accept_phrase(phrase: str, count: int, min_count: int, min_tokens: int = 2) -> bool:
    toks = phrase.split()
    if count < min_count and phrase not in MODEL_SEEDS:
        return False
    if phrase in MODEL_SEEDS:
        return True
    if all(t.replace(".", "").isdigit() for t in toks):
        return False
    if len(toks) == 1:
        t = toks[0]
        return any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and 2 <= len(t) <= 10
    return len(toks) >= min_tokens


def coverage_for_phrases(phrases_set, o_tails_list):
    plist = sorted(phrases_set, key=lambda s: (-len(s.split()), -len(s)))
    hit = 0
    for tail_toks in o_tails_list:
        ok = False
        for p in plist:
            pt = p.split()
            if tail_toks[: len(pt)] == pt:
                ok = True
                break
        hit += int(ok)
    return hit / max(len(o_tails_list), 1)


def main() -> None:
    if LOG.exists():
        LOG.unlink()
    ensure_dirs()
    apply_plot_style()
    fig_dir = FIGURES_DIR / "complex_eda" / "model_tag"
    fig_dir.mkdir(parents=True, exist_ok=True)
    art = ARTIFACTS_DIR

    clicks = load_query_clicks(n=150_000, seed=42, random=True)
    log(f"clicks={len(clicks)}")
    labeler_base = WeakLabeler.from_files(art / "brands.txt", art / "categories.txt")

    uq = clicks["query_text"].astype(str).str.strip()
    sample_q = uq[uq.str.len() >= 2].drop_duplicates().sample(n=min(12_000, uq.nunique()), random_state=42).tolist()

    n_with_brand = n_o_tail = 0
    tail_lens = []
    tail_phrases = Counter()
    o_tails_list = []
    pp0 = QueryPreprocessor()

    for q in sample_q:
        qn = pp0(q).text_norm
        tags = labeler_base.label_query(qn)
        if any(t.endswith("BRAND") for _, t in tags):
            n_with_brand += 1
        tail = brand_latin_o_tail(tags)
        if tail is None:
            continue
        n_o_tail += 1
        tail_lens.append(len(tail))
        tail_phrases[" ".join(tail[:5])] += 1
        o_tails_list.append(tail)

    log(f"brand={n_with_brand} o_tail={n_o_tail} share={n_o_tail/max(n_with_brand,1):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    clean = n_with_brand - n_o_tail
    axes[0].bar([0], [clean], color=DARK_SLATE, label="BRAND without latin O-tail")
    axes[0].bar([0], [n_o_tail], bottom=[clean], color=MVIDEO_RED, label="BRAND + latin O-tail")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["queries with BRAND"])
    axes[0].set_title("Missing MODEL tag — problem")
    axes[0].legend(fontsize=8)
    axes[1].hist(tail_lens, bins=range(1, max(tail_lens + [1]) + 2), color=MVIDEO_RED, edgecolor="white", align="left")
    axes[1].set_title("Length of unlabeled tail after BRAND")
    axes[1].set_xlabel("tokens")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_missing_model_problem.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    top_lost = pd.DataFrame(tail_phrases.most_common(20), columns=["o_tail_phrase", "count"])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(top_lost["o_tail_phrase"][::-1], top_lost["count"][::-1], color=MVIDEO_RED)
    ax.set_title("Top O-tails after BRAND — MODEL candidates")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_top_lost_tails.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    log("top tails: " + str(top_lost.head(8).to_dict("records")))

    brands_top = (
        clicks["sku_brand_name"].astype(str).str.strip().replace("", np.nan).dropna().value_counts().head(400).index.tolist()
    )
    titles = clicks["sku_name"].astype(str).dropna().drop_duplicates().tolist()
    raw = build_model_lexicon_from_titles(titles, brands_top, min_count=2, max_phrase_tokens=4)
    log(f"raw={len(raw)} titles={len(titles)}")

    rows = []
    for mc in range(2, 21):
        kept = {p for p, c in raw.items() if accept_phrase(p, c, min_count=mc)} | set(MODEL_SEEDS)
        rows.append(
            {
                "min_count": mc,
                "dict_size": len(kept),
                "o_tail_coverage": coverage_for_phrases(kept, o_tails_list),
            }
        )
    thr_df = pd.DataFrame(rows)
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax2 = ax1.twinx()
    ax1.plot(thr_df["min_count"], thr_df["dict_size"], "o-", color=DARK_SLATE, lw=2, label="dict size")
    ax2.plot(thr_df["min_count"], thr_df["o_tail_coverage"], "s--", color=MVIDEO_RED, lw=2, label="O-tail coverage")
    ax1.axvline(6, color=MUTED, ls=":")
    ax1.set_xlabel("min_count")
    ax1.set_title("Threshold trade-off: dict size vs O-tail coverage")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_threshold_tradeoff.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    MIN_COUNT = 6
    final = {p for p, c in raw.items() if accept_phrase(p, c, min_count=MIN_COUNT)} | set(MODEL_SEEDS)
    save_phrase_list(final, art / "model_phrases.txt")
    dicts = art / "dicts"
    dicts.mkdir(parents=True, exist_ok=True)
    save_phrase_list(final, dicts / "model_phrases.txt")
    log(f"saved model_phrases n={len(final)} (+ dicts/)")

    from src.data_utils import brands_path, categories_path, model_phrases_path

    bp = brands_path() if brands_path().exists() else art / "brands.txt"
    cp = categories_path() if categories_path().exists() else art / "categories.txt"
    mp = model_phrases_path() if model_phrases_path().exists() else art / "model_phrases.txt"
    labeler_model = WeakLabeler.from_files(bp, cp, models_path=mp)
    for q in ["наушники logitech g-pro x se", "dyson v15", "samsung galaxy s24"]:
        qn = pp0(q).text_norm
        log(f"AFTER {qn} -> {labeler_model.label_query(qn)}")

    n_model_after = n_o_tail_after = 0
    for q in sample_q:
        qn = pp0(q).text_norm
        tags = labeler_model.label_query(qn)
        if any(t.endswith("MODEL") for _, t in tags):
            n_model_after += 1
        if brand_latin_o_tail(tags) is not None:
            n_o_tail_after += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [n_o_tail, n_o_tail_after], w, color=MVIDEO_RED, label="latin O-tails after BRAND")
    ax.bar(x + w / 2, [0, n_model_after], w, color=DARK_SLATE, label="queries with MODEL")
    ax.set_xticks(x)
    ax.set_xticklabels(["before (no MODEL)", "after (model_phrases)"])
    ax.set_title("Effect of MODEL dictionary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_before_after_model_tag.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    reduction = 1 - n_o_tail_after / max(n_o_tail, 1)
    log(f"reduction={reduction:.3f} model_queries={n_model_after}")
    save_stats(
        {
            "sample_queries": len(sample_q),
            "with_brand": n_with_brand,
            "latin_o_tails_before": n_o_tail,
            "latin_o_tails_after": n_o_tail_after,
            "o_tail_reduction": float(reduction),
            "queries_with_model_after": n_model_after,
            "min_count": MIN_COUNT,
            "n_model_phrases": len(final),
        },
        "model_tag_eda_stats.json",
    )
    log("DONE")


if __name__ == "__main__":
    main()
