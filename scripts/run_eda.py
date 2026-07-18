#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Генератор EDA-графиков для кейса M.Video Search/NER.

Запуск из корня репозитория:
    python scripts/run_eda.py

Создаёт 12+ PNG в figures/ и artifacts/dataset_stats.json.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    DARK_SLATE,
    DPI,
    MVIDEO_RED,
    MUTED,
    PALETTE,
    apply_plot_style,
    dataset_overview_stats,
    ensure_dirs,
    load_query_clicks,
    load_sku_desc,
    parquet_num_rows,
    save_fig,
    save_stats,
    text_len,
    QUERY_CLICKS_PATH,
    SKU_DESC_PATH,
    SKUS_PKL_PATH,
)

warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLE_CLICKS = 400_000
SAMPLE_DESC = 250_000
SEED = 42


def _barh_top(ax, series: pd.Series, title: str, xlabel: str, color=MVIDEO_RED, top=20):
    top_s = series.value_counts().head(top).iloc[::-1]
    ax.barh(top_s.index.astype(str), top_s.values, color=color, edgecolor="none")
    ax.set_title(title, color=DARK_SLATE, pad=10)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_query_length(df: pd.DataFrame) -> None:
    lengths = text_len(df["query_text"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(lengths.clip(upper=lengths.quantile(0.99)), bins=50, color=MVIDEO_RED, edgecolor="white", ax=ax)
    ax.axvline(lengths.median(), color=DARK_SLATE, ls="--", lw=1.5, label=f"Медиана = {lengths.median():.0f}")
    ax.set_title("Распределение длины поисковых запросов (символы)")
    ax.set_xlabel("Длина запроса")
    ax.set_ylabel("Число наблюдений")
    ax.legend()
    save_fig(fig, "01_query_length_dist.png")
    plt.close(fig)


def fig_top_queries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    _barh_top(ax, df["query_text"], "Топ-20 поисковых запросов (по семплу)", "Частота кликов")
    save_fig(fig, "02_top_queries.png")
    plt.close(fig)


def fig_top_brands(df: pd.DataFrame) -> None:
    brands = df["sku_brand_name"].replace("", np.nan).dropna()
    fig, ax = plt.subplots(figsize=(10, 7))
    _barh_top(ax, brands, "Топ-20 брендов по кликам", "Частота кликов", color=DARK_SLATE)
    save_fig(fig, "03_top_brands.png")
    plt.close(fig)


def fig_price_distribution(df: pd.DataFrame) -> None:
    prices = df["sku_price"].dropna()
    prices = prices[(prices > 0) & (prices < prices.quantile(0.98))]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(prices, bins=60, color=MVIDEO_RED, edgecolor="white", ax=ax, log_scale=(False, False))
    ax.set_title("Распределение цен SKU (обрезка 98-го перцентиля)")
    ax.set_xlabel("Цена, ₽")
    ax.set_ylabel("Число наблюдений")
    save_fig(fig, "04_price_distribution.png")
    plt.close(fig)


def fig_position_distribution(df: pd.DataFrame) -> None:
    pos = df["sku_position"].dropna()
    pos = pos[pos <= 50]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(pos, bins=51, discrete=True, color=MVIDEO_RED, edgecolor="white", ax=ax)
    ax.set_title("Распределение позиций SKU в выдаче (0–50)")
    ax.set_xlabel("Позиция (sku_position)")
    ax.set_ylabel("Число кликов")
    save_fig(fig, "05_position_distribution.png")
    plt.close(fig)


def fig_query_brand_heatmap(df: pd.DataFrame) -> None:
    top_q = df["query_text"].value_counts().head(15).index
    top_b = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(15).index
    sub = df[df["query_text"].isin(top_q) & df["sku_brand_name"].isin(top_b)]
    mat = pd.crosstab(sub["query_text"], sub["sku_brand_name"])
    mat = mat.reindex(index=top_q, columns=top_b, fill_value=0)
    # log1p для читаемости
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        np.log1p(mat),
        cmap=sns.light_palette(MVIDEO_RED, as_cmap=True),
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "log(1 + частота)"},
    )
    ax.set_title("Совместная встречаемость: запрос × бренд")
    ax.set_xlabel("Бренд")
    ax.set_ylabel("Запрос")
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, "06_query_brand_heatmap.png")
    plt.close(fig)


def fig_tfidf_similarity(df: pd.DataFrame) -> None:
    top_q = df["query_text"].value_counts().head(25).index.tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(top_q)
    sim = cosine_similarity(X)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        sim,
        xticklabels=top_q,
        yticklabels=top_q,
        cmap=sns.light_palette(MVIDEO_RED, as_cmap=True),
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Косинусное сходство (TF-IDF)"},
    )
    ax.set_title("Матрица сходства топ-запросов (TF-IDF char n-grams)")
    plt.xticks(rotation=55, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    save_fig(fig, "07_tfidf_similarity_heatmap.png")
    plt.close(fig)


def fig_subject_distribution(df: pd.DataFrame) -> None:
    subj = df["sku_subject_id"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    _barh_top(ax, subj, "Топ-20 категорий (sku_subject_id)", "Частота кликов", color=MUTED)
    save_fig(fig, "08_subject_distribution.png")
    plt.close(fig)


def fig_title_length(desc: pd.DataFrame) -> None:
    tlen = text_len(desc["title"])
    dlen = text_len(desc["description"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(tlen.clip(upper=tlen.quantile(0.99)), bins=40, color=MVIDEO_RED, ax=axes[0], edgecolor="white")
    axes[0].set_title("Длина title")
    axes[0].set_xlabel("Символы")
    sns.histplot(dlen.clip(upper=dlen.quantile(0.95)), bins=40, color=DARK_SLATE, ax=axes[1], edgecolor="white")
    axes[1].set_title("Длина description")
    axes[1].set_xlabel("Символы")
    fig.suptitle("Длины текстовых полей каталога sku_desc", fontweight="bold", color=DARK_SLATE)
    fig.tight_layout()
    save_fig(fig, "09_title_length_dist.png")
    plt.close(fig)


def fig_price_vs_position(df: pd.DataFrame) -> None:
    sub = df[["sku_price", "sku_position"]].dropna()
    sub = sub[(sub["sku_price"] > 0) & (sub["sku_position"] <= 30)]
    sub = sub[sub["sku_price"] < sub["sku_price"].quantile(0.95)]
    # агрегат по позиции
    agg = sub.groupby("sku_position")["sku_price"].median().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(agg["sku_position"], agg["sku_price"], color=MVIDEO_RED, lw=2.2, marker="o", ms=4)
    ax.fill_between(agg["sku_position"], agg["sku_price"], alpha=0.12, color=MVIDEO_RED)
    ax.set_title("Медианная цена vs позиция в выдаче")
    ax.set_xlabel("Позиция")
    ax.set_ylabel("Медианная цена, ₽")
    save_fig(fig, "10_price_vs_position.png")
    plt.close(fig)


def fig_query_word_frequency(df: pd.DataFrame) -> None:
    # Простая токенизация по пробелам + нижний регистр
    stop = {
        "и", "в", "на", "с", "для", "по", "из", "к", "от", "the", "a", "of",
        "или", "не", "до", "за", "без", "под",
    }
    counts: Counter = Counter()
    for q in df["query_text"].dropna().astype(str).head(200_000):
        for tok in q.lower().replace("-", " ").split():
            tok = tok.strip(".,!?«»\"'()[]")
            if len(tok) < 3 or tok in stop or tok.isdigit():
                continue
            counts[tok] += 1
    top = counts.most_common(30)
    words, freqs = zip(*top[::-1])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(words, freqs, color=MVIDEO_RED, edgecolor="none")
    ax.set_title("Частота слов в поисковых запросах (топ-30)")
    ax.set_xlabel("Частота")
    save_fig(fig, "11_query_word_frequency.png")
    plt.close(fig)

    # Word cloud (optional)
    try:
        from wordcloud import WordCloud

        wc = WordCloud(
            width=1400,
            height=800,
            background_color="white",
            colormap="Reds",
            max_words=120,
            prefer_horizontal=0.9,
            relative_scaling=0.4,
        ).generate_from_frequencies(dict(counts.most_common(200)))
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Облако слов поисковых запросов", color=DARK_SLATE, pad=12)
        save_fig(fig, "13_query_wordcloud.png")
        plt.close(fig)
    except Exception as e:
        print(f"WordCloud пропущен: {e}")


def fig_brand_price_boxplot(df: pd.DataFrame) -> None:
    brands = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(12).index
    sub = df[df["sku_brand_name"].isin(brands)].copy()
    sub = sub[(sub["sku_price"] > 0) & (sub["sku_price"] < sub["sku_price"].quantile(0.95))]
    order = sub.groupby("sku_brand_name")["sku_price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=sub,
        x="sku_brand_name",
        y="sku_price",
        order=order,
        color="#FECACA",
        width=0.6,
        fliersize=1.5,
        ax=ax,
    )
    ax.set_title("Распределение цен по топ-брендам")
    ax.set_xlabel("Бренд")
    ax.set_ylabel("Цена, ₽")
    plt.xticks(rotation=35, ha="right")
    save_fig(fig, "12_brand_price_boxplot.png")
    plt.close(fig)


def fig_brand_similarity(df: pd.DataFrame) -> None:
    brands = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(20).index.tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vec.fit_transform(brands)
    sim = cosine_similarity(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        sim,
        xticklabels=brands,
        yticklabels=brands,
        cmap=sns.light_palette(DARK_SLATE, as_cmap=True),
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Сходство названий брендов"},
    )
    ax.set_title("TF-IDF сходство названий топ-брендов")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    save_fig(fig, "14_brand_name_similarity.png")
    plt.close(fig)


def fig_clicks_by_position_cumsum(df: pd.DataFrame) -> None:
    pos = df["sku_position"].dropna()
    pos = pos[pos <= 40]
    vc = pos.value_counts().sort_index()
    share = vc / vc.sum()
    cum = share.cumsum()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(share.index, share.values, color=MVIDEO_RED, alpha=0.85, label="Доля кликов")
    ax.plot(cum.index, cum.values, color=DARK_SLATE, lw=2, marker="o", ms=3, label="Накопленная доля")
    ax.set_title("Доля кликов по позициям и накопленный охват")
    ax.set_xlabel("Позиция")
    ax.set_ylabel("Доля")
    ax.legend()
    save_fig(fig, "15_position_click_share.png")
    plt.close(fig)


def fig_nulls_overview(df: pd.DataFrame, desc: pd.DataFrame) -> None:
    null_c = (df.isna() | (df.astype(str).eq(""))).mean().sort_values(ascending=True)
    null_d = (desc.isna() | (desc.astype(str).eq(""))).mean().sort_values(ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(null_c.index, null_c.values, color=MVIDEO_RED)
    axes[0].set_title("Доля пропусков / пустых строк\nquery_clicks (семпл)")
    axes[0].set_xlabel("Доля")
    axes[1].barh(null_d.index, null_d.values, color=DARK_SLATE)
    axes[1].set_title("Доля пропусков / пустых строк\nsku_desc (семпл)")
    axes[1].set_xlabel("Доля")
    fig.tight_layout()
    save_fig(fig, "16_nulls_overview.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    apply_plot_style()

    full_clicks = parquet_num_rows(QUERY_CLICKS_PATH)
    full_desc = parquet_num_rows(SKU_DESC_PATH)
    print(f"Полный query_clicks: {full_clicks:,} строк")
    print(f"Полный sku_desc:     {full_desc:,} строк")
    print(f"Загрузка семпла кликов n={SAMPLE_CLICKS:,} ...")
    df = load_query_clicks(n=SAMPLE_CLICKS, seed=SEED, random=True)
    print(f"  загружено: {len(df):,}")
    print(f"Загрузка семпла описаний n={SAMPLE_DESC:,} ...")
    desc = load_sku_desc(n=SAMPLE_DESC, seed=SEED, random=True)
    print(f"  загружено: {len(desc):,}")

    # Полные уникальности — оценка по семплу + известные ориентиры
    stats = dataset_overview_stats(df)
    stats["known_full_unique_queries_approx"] = 1_790_000
    stats["known_full_unique_skus_approx"] = 332_000
    stats["known_full_unique_brands_approx"] = 7_150
    stats["known_full_unique_subjects_approx"] = 4_103
    stats["sample_top_brands"] = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(15).to_dict()
    stats["sample_top_queries"] = df["query_text"].value_counts().head(15).to_dict()
    stats["skus_pkl_exists"] = SKUS_PKL_PATH.exists()
    stats["sku_desc_sample_rows"] = int(len(desc))
    stats["figures_generated_by"] = "scripts/run_eda.py"

    print("Рисуем графики...")
    fig_query_length(df)
    fig_top_queries(df)
    fig_top_brands(df)
    fig_price_distribution(df)
    fig_position_distribution(df)
    fig_query_brand_heatmap(df)
    fig_tfidf_similarity(df)
    fig_subject_distribution(df)
    fig_title_length(desc)
    fig_price_vs_position(df)
    fig_query_word_frequency(df)
    fig_brand_price_boxplot(df)
    fig_brand_similarity(df)
    fig_clicks_by_position_cumsum(df)
    fig_nulls_overview(df, desc)

    save_stats(stats)
    figs = sorted((ROOT / "figures").glob("*.png"))
    print(f"\nГотово: {len(figs)} PNG в figures/")
    for p in figs:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
