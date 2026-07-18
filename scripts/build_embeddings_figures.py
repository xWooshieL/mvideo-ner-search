#!/usr/bin/env python
"""TF-IDF / Word2Vec embeddings: t-SNE, similarity matrix figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COL_QUERY = "toValidUTF8(query_text)"
COL_BRAND = "toValidUTF8(sku_brand_name)"


def load_queries(path: Path, n: int):
    pf = pq.ParquetFile(path)
    rows = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=[COL_QUERY, COL_BRAND])
        df = t.to_pandas()
        df.columns = ["query", "brand"]
        rows.append(df)
        if sum(len(r) for r in rows) >= n * 5:
            break
    import pandas as pd

    df = pd.concat(rows, ignore_index=True)
    df["query"] = df["query"].fillna("").astype(str).str.strip()
    df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    df = df[df["query"].str.len() >= 2].drop_duplicates("query")
    return df.head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "файлы" / "query_clicks.parquet")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    args = ap.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    df = load_queries(args.data, args.n)
    queries = df["query"].tolist()
    print(f"Embedding {len(queries)} queries")

    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000)
    X = tfidf.fit_transform(queries)

    # Similarity among a small sample of popular queries
    sample_idx = np.linspace(0, len(queries) - 1, num=min(25, len(queries)), dtype=int)
    Xs = X[sample_idx]
    sim = cosine_similarity(Xs)
    labels = [queries[i][:28] for i in sample_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Косинусная близость запросов (TF-IDF)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.figures / "19_similarity_queries.png", dpi=140)
    plt.close(fig)

    # t-SNE on subset
    n_tsne = min(1500, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n_tsne, replace=False)
    X_dense = X[idx].astype(np.float32).toarray()
    # Reduce dims first if needed
    if X_dense.shape[1] > 100:
        from sklearn.decomposition import TruncatedSVD

        X_dense = TruncatedSVD(n_components=50, random_state=42).fit_transform(X[idx])

    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    emb = tsne.fit_transform(X_dense)

    brands = df["brand"].iloc[idx].tolist()
    # color top brands
    from collections import Counter

    top = {b for b, _ in Counter([b for b in brands if b]).most_common(12)}
    brand_plot = [b if b in top else "other" for b in brands]

    fig, ax = plt.subplots(figsize=(9, 7))
    cats = sorted(set(brand_plot))
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(cats), 1))
    for i, c in enumerate(cats):
        m = [j for j, v in enumerate(brand_plot) if v == c]
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, color=cmap(i), label=c)
    ax.legend(markerscale=2, fontsize=8, loc="best", frameon=False)
    ax.set_title("t-SNE запросов (TF-IDF), цвет = бренд клика")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.figures / "16_embedding_tsne.png", dpi=140)
    plt.close(fig)

    # Word2Vec-like co-occurrence via TruncatedSVD on word TF-IDF (gensim optional)
    try:
        from gensim.models import Word2Vec

        tokenized = [q.lower().split() for q in queries]
        w2v = Word2Vec(sentences=tokenized, vector_size=64, window=3, min_count=3, workers=2, epochs=10)
        w2v.save(str(args.models / "w2v_queries.model"))
        print("Saved Word2Vec → models/w2v_queries.model")
    except Exception as e:
        print("Word2Vec skipped (use TF-IDF SVD fallback):", e)
        word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=15000)
        Xw = word_tfidf.fit_transform(queries)
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=64, random_state=42)
        svd.fit(Xw)
        import joblib as _joblib

        _joblib.dump({"tfidf": word_tfidf, "svd": svd}, args.models / "w2v_tfidf_svd.joblib")
        print("Saved TF-IDF+SVD embedding → models/w2v_tfidf_svd.joblib")

    import joblib

    joblib.dump(tfidf, args.models / "tfidf_queries.joblib")
    print("Saved figures 16 & 19")


if __name__ == "__main__":
    main()
