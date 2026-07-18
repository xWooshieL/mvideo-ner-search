# -*- coding: utf-8 -*-
"""Generate rich EDA notebooks 01-05."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(s: str):
    lines = s.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else []),
    }


def code(s: str):
    lines = s.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else []),
    }


SETUP = r'''%matplotlib inline
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import (
    apply_plot_style, save_fig, ensure_dirs, load_query_clicks, load_sku_desc,
    text_len, parquet_num_rows, parquet_schema_names, dataset_overview_stats,
    save_stats, QUERY_CLICKS_PATH, SKU_DESC_PATH, SKUS_PKL_PATH,
    MVIDEO_RED, DARK_SLATE, MUTED,
)
ensure_dirs()
apply_plot_style()
pd.set_option("display.max_colwidth", 80)
print("ROOT:", ROOT)'''


def write(name: str, cells):
    path = NB_DIR / name
    path.write_text(json.dumps(nb(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


write(
    "01_data_overview.ipynb",
    [
        md(
            "# 01. Обзор данных — M.Video Search / NER\n"
            "\n"
            "Ноутбук описывает схему и базовые характеристики трёх источников:\n"
            "\n"
            "1. `query_clicks.parquet` — клики по поисковой выдаче (~31M строк)\n"
            "2. `sku_desc.parquet` — названия и описания SKU (~1.18M строк)\n"
            "3. `skus.pkl` — YML-каталог (Yandex Market Language)\n"
            "\n"
            "> Для скорости используем **семплы**; полные размеры датасетов фиксируем в статистике."
        ),
        code(SETUP),
        md("## Полные размеры файлов"),
        code(
            "n_clicks = parquet_num_rows(QUERY_CLICKS_PATH)\n"
            "n_desc = parquet_num_rows(SKU_DESC_PATH)\n"
            'print(f"query_clicks.parquet: {n_clicks:,} строк")\n'
            'print(f"sku_desc.parquet:     {n_desc:,} строк")\n'
            'print(f"skus.pkl size:        {SKUS_PKL_PATH.stat().st_size / 1e9:.2f} GB")\n'
            'print("\\nСырые колонки query_clicks:")\n'
            "for c in parquet_schema_names(QUERY_CLICKS_PATH):\n"
            '    print(" -", c)\n'
            'print("\\nКолонки sku_desc:")\n'
            "for c in parquet_schema_names(SKU_DESC_PATH):\n"
            '    print(" -", c)'
        ),
        md(
            "## Семпл query_clicks\n\n"
            "Колонки `toValidUTF8(...)` переименовываются в `sku_name`, `sku_brand_name`, `query_text`."
        ),
        code(
            "SAMPLE_N = 300_000\n"
            "df = load_query_clicks(n=SAMPLE_N, seed=42, random=True)\n"
            'print("Форма семпла:", df.shape)\n'
            'print("\\ndtypes:")\n'
            "print(df.dtypes)\n"
            "display(df.head(8))\n"
            'display(df.describe(include="all").T)'
        ),
        md("## Пропуски и пустые строки"),
        code(
            "nulls = df.isna().sum()\n"
            'empty = (df.astype(str).eq("") | df.astype(str).eq("nan")).sum()\n'
            'overview = pd.DataFrame({"nulls": nulls, "empty_str": empty, '
            '"null_or_empty_share": (nulls + empty) / len(df)})\n'
            "display(overview)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            'share = overview["null_or_empty_share"].sort_values()\n'
            "ax.barh(share.index, share.values, color=MVIDEO_RED)\n"
            'ax.set_title("Доля пропусков / пустых значений (query_clicks, семпл)")\n'
            'ax.set_xlabel("Доля")\n'
            'save_fig(fig, "16_nulls_overview.png")\n'
            "plt.show()"
        ),
        md("## Семпл sku_desc"),
        code(
            "desc = load_sku_desc(n=200_000, seed=42, random=True)\n"
            'print("Форма:", desc.shape)\n'
            "print(desc.dtypes)\n"
            "display(desc.head(5))\n"
            'print("\\nПропуски:")\n'
            "print(desc.isna().sum())\n"
            'print("\\nПустые title:", (desc["title"].fillna("").eq("")).sum())\n'
            'print("Пустые description:", (desc["description"].fillna("").eq("")).sum())'
        ),
        md(
            "## Структура skus.pkl (YML)\n\n"
            "Файл большой (~1.5 GB). Загрузка может занять несколько минут."
        ),
        code(
            "import pickle\n"
            "with open(SKUS_PKL_PATH, \"rb\") as f:\n"
            "    skus = pickle.load(f)\n"
            "print(type(skus))\n"
            "if isinstance(skus, dict):\n"
            '    print("Ключи:", list(skus.keys())[:30])\n'
            '    cat = skus.get("yml_catalog", skus)\n'
            "    if isinstance(cat, dict):\n"
            '        print("yml_catalog keys:", list(cat.keys())[:30])\n'
            '        shop = cat.get("shop", {})\n'
            "        if isinstance(shop, dict):\n"
            '            print("shop keys:", list(shop.keys())[:40])'
        ),
        md("## Ключевые статистики → artifacts"),
        code(
            "stats = dataset_overview_stats(df)\n"
            'stats["known_full_unique_queries_approx"] = 1_790_000\n'
            'stats["known_full_unique_skus_approx"] = 332_000\n'
            'stats["known_full_unique_brands_approx"] = 7150\n'
            'stats["known_full_unique_subjects_approx"] = 4103\n'
            'stats["sample_top_queries"] = df["query_text"].value_counts().head(10).to_dict()\n'
            'stats["sample_top_brands"] = (\n'
            '    df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(10).to_dict()\n'
            ")\n"
            "save_stats(stats)\n"
            "stats"
        ),
    ],
)

write(
    "02_eda_queries.ipynb",
    [
        md(
            "# 02. EDA поисковых запросов\n\n"
            "Анализируем длину запросов, топ-частоты, слова/n-граммы и позиции кликов.\n\n"
            "**Полный датасет:** ~30.99M кликов, ~1.79M уникальных запросов.  \n"
            "**Семпл:** 400k строк."
        ),
        code(
            SETUP
            + "\nfrom collections import Counter\n"
            + "df = load_query_clicks(n=400_000, seed=42)\n"
            + "print(df.shape)\n"
            + 'print("Уникальных запросов в семпле:", df["query_text"].nunique())'
        ),
        md("## Длина запроса"),
        code(
            'lengths = text_len(df["query_text"])\n'
            "print(lengths.describe())\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            "sns.histplot(lengths.clip(upper=lengths.quantile(0.99)), bins=50, "
            'color=MVIDEO_RED, edgecolor="white", ax=ax)\n'
            'ax.axvline(lengths.median(), color=DARK_SLATE, ls="--", lw=1.5, '
            'label=f"Медиана = {lengths.median():.0f}")\n'
            'ax.set_title("Распределение длины поисковых запросов (символы)")\n'
            'ax.set_xlabel("Длина запроса"); ax.set_ylabel("Число наблюдений"); ax.legend()\n'
            'save_fig(fig, "01_query_length_dist.png"); plt.show()'
        ),
        md("## Топ запросов"),
        code(
            'top_q = df["query_text"].value_counts().head(20)\n'
            'display(top_q.to_frame("clicks"))\n'
            "fig, ax = plt.subplots(figsize=(10, 7))\n"
            "ax.barh(top_q.index.astype(str)[::-1], top_q.values[::-1], color=MVIDEO_RED)\n"
            'ax.set_title("Топ-20 поисковых запросов (по семплу)"); ax.set_xlabel("Частота кликов")\n'
            'save_fig(fig, "02_top_queries.png"); plt.show()'
        ),
        md("## Частота слов и облако"),
        code(
            'stop = {"и", "в", "на", "с", "для", "по", "из", "к", "от", "или", "не", "до", "за", "без", "под"}\n'
            "counts = Counter()\n"
            'for q in df["query_text"].dropna().astype(str):\n'
            '    for tok in q.lower().replace("-", " ").split():\n'
            "        tok = tok.strip(\".!?«»\\\"'()[]\")\n"
            "        if len(tok) >= 3 and tok not in stop and not tok.isdigit():\n"
            "            counts[tok] += 1\n"
            "top_words = counts.most_common(30)\n"
            "words, freqs = zip(*top_words[::-1])\n"
            "fig, ax = plt.subplots(figsize=(10, 8))\n"
            "ax.barh(words, freqs, color=MVIDEO_RED)\n"
            'ax.set_title("Частота слов в поисковых запросах (топ-30)"); ax.set_xlabel("Частота")\n'
            'save_fig(fig, "11_query_word_frequency.png"); plt.show()\n'
            "try:\n"
            "    from wordcloud import WordCloud\n"
            "    wc = WordCloud(width=1400, height=800, background_color=\"white\", "
            "colormap=\"Reds\", max_words=120).generate_from_frequencies(dict(counts.most_common(200)))\n"
            "    fig, ax = plt.subplots(figsize=(12, 7))\n"
            '    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")\n'
            '    ax.set_title("Облако слов поисковых запросов")\n'
            '    save_fig(fig, "13_query_wordcloud.png"); plt.show()\n'
            "except Exception as e:\n"
            '    print("WordCloud недоступен:", e)'
        ),
        md("## Распределение позиций клика"),
        code(
            'pos = df["sku_position"].dropna()\n'
            "pos = pos[pos <= 50]\n"
            "print(pos.describe())\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            'sns.histplot(pos, bins=51, discrete=True, color=MVIDEO_RED, edgecolor="white", ax=ax)\n'
            'ax.set_title("Распределение позиций SKU в выдаче (0–50)")\n'
            'ax.set_xlabel("Позиция (sku_position)"); ax.set_ylabel("Число кликов")\n'
            'save_fig(fig, "05_position_distribution.png"); plt.show()'
        ),
        md("## Биграммы (топ)"),
        code(
            "bigrams = Counter()\n"
            'for q in df["query_text"].dropna().astype(str).head(150_000):\n'
            "    toks = [t.strip(\".!?«»\\\"'()[]\") for t in q.lower().split() if len(t.strip(\".!?\")) >= 2]\n"
            "    for a, b in zip(toks, toks[1:]):\n"
            '        bigrams[f"{a} {b}"] += 1\n'
            "bg = bigrams.most_common(20)\n"
            "labels, vals = zip(*bg[::-1])\n"
            "fig, ax = plt.subplots(figsize=(10, 7))\n"
            "ax.barh(labels, vals, color=DARK_SLATE)\n"
            'ax.set_title("Топ-20 биграмм в запросах"); ax.set_xlabel("Частота")\n'
            'save_fig(fig, "17_query_bigrams.png"); plt.show()'
        ),
    ],
)

write(
    "03_eda_products.ipynb",
    [
        md(
            "# 03. EDA товаров: бренды, цены, категории\n\n"
            "Смотрим бренды, ценовые распределения, `sku_subject_id` и длины title/description.\n\n"
            "**Ориентиры полного датасета:** ~332k SKU, ~7150 брендов, ~4103 subject."
        ),
        code(
            SETUP
            + "\ndf = load_query_clicks(n=400_000, seed=42)\n"
            + "desc = load_sku_desc(n=250_000, seed=42)\n"
            + 'print("clicks", df.shape, "desc", desc.shape)'
        ),
        md("## Топ брендов"),
        code(
            'brands = df["sku_brand_name"].replace("", np.nan).dropna()\n'
            'print("Уникальных брендов в семпле:", brands.nunique())\n'
            "top_b = brands.value_counts().head(20)\n"
            'display(top_b.to_frame("clicks"))\n'
            "fig, ax = plt.subplots(figsize=(10, 7))\n"
            "ax.barh(top_b.index.astype(str)[::-1], top_b.values[::-1], color=DARK_SLATE)\n"
            'ax.set_title("Топ-20 брендов по кликам"); ax.set_xlabel("Частота кликов")\n'
            'save_fig(fig, "03_top_brands.png"); plt.show()'
        ),
        md("## Распределение цен"),
        code(
            'prices = df["sku_price"].dropna()\n'
            "prices = prices[(prices > 0) & (prices < prices.quantile(0.98))]\n"
            "print(prices.describe())\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            'sns.histplot(prices, bins=60, color=MVIDEO_RED, edgecolor="white", ax=ax)\n'
            'ax.set_title("Распределение цен SKU (обрезка 98-го перцентиля)")\n'
            'ax.set_xlabel("Цена, ₽"); ax.set_ylabel("Число наблюдений")\n'
            'save_fig(fig, "04_price_distribution.png"); plt.show()'
        ),
        md("## Boxplot цен по брендам"),
        code(
            "top12 = brands.value_counts().head(12).index\n"
            'sub = df[df["sku_brand_name"].isin(top12)].copy()\n'
            'sub = sub[(sub["sku_price"] > 0) & (sub["sku_price"] < sub["sku_price"].quantile(0.95))]\n'
            'order = sub.groupby("sku_brand_name")["sku_price"].median().sort_values(ascending=False).index\n'
            "fig, ax = plt.subplots(figsize=(12, 6))\n"
            'sns.boxplot(data=sub, x="sku_brand_name", y="sku_price", order=order, '
            'color="#FECACA", width=0.6, fliersize=1.5, ax=ax)\n'
            'ax.set_title("Распределение цен по топ-брендам")\n'
            'ax.set_xlabel("Бренд"); ax.set_ylabel("Цена, ₽")\n'
            'plt.xticks(rotation=35, ha="right")\n'
            'save_fig(fig, "12_brand_price_boxplot.png"); plt.show()'
        ),
        md("## Категории (sku_subject_id)"),
        code(
            'print("Уникальных subject в семпле:", df["sku_subject_id"].nunique())\n'
            'subj = df["sku_subject_id"].astype(str).value_counts().head(20)\n'
            "fig, ax = plt.subplots(figsize=(10, 7))\n"
            "ax.barh(subj.index.astype(str)[::-1], subj.values[::-1], color=MUTED)\n"
            'ax.set_title("Топ-20 категорий (sku_subject_id)"); ax.set_xlabel("Частота кликов")\n'
            'save_fig(fig, "08_subject_distribution.png"); plt.show()'
        ),
        md("## Длины title / description"),
        code(
            'tlen = text_len(desc["title"]); dlen = text_len(desc["description"])\n'
            'print("title:", tlen.describe()); print("description:", dlen.describe())\n'
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "sns.histplot(tlen.clip(upper=tlen.quantile(0.99)), bins=40, color=MVIDEO_RED, ax=axes[0], edgecolor=\"white\")\n"
            'axes[0].set_title("Длина title"); axes[0].set_xlabel("Символы")\n'
            "sns.histplot(dlen.clip(upper=dlen.quantile(0.95)), bins=40, color=DARK_SLATE, ax=axes[1], edgecolor=\"white\")\n"
            'axes[1].set_title("Длина description"); axes[1].set_xlabel("Символы")\n'
            'fig.suptitle("Длины текстовых полей каталога sku_desc", fontweight="bold"); fig.tight_layout()\n'
            'save_fig(fig, "09_title_length_dist.png"); plt.show()'
        ),
    ],
)

write(
    "04_similarity_matrices.ipynb",
    [
        md(
            "# 04. Матрицы сходства и совместной встречаемости\n\n"
            "Строим TF-IDF cosine similarity между топ-запросами, heatmap query×brand "
            "и сходство названий брендов."
        ),
        code(
            SETUP
            + "\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n"
            + "from sklearn.metrics.pairwise import cosine_similarity\n"
            + "df = load_query_clicks(n=400_000, seed=42)\n"
            + "print(df.shape)"
        ),
        md("## Heatmap: запрос × бренд"),
        code(
            'top_q = df["query_text"].value_counts().head(15).index\n'
            'top_b = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(15).index\n'
            'sub = df[df["query_text"].isin(top_q) & df["sku_brand_name"].isin(top_b)]\n'
            'mat = pd.crosstab(sub["query_text"], sub["sku_brand_name"]).reindex('
            "index=top_q, columns=top_b, fill_value=0)\n"
            "fig, ax = plt.subplots(figsize=(12, 8))\n"
            "sns.heatmap(np.log1p(mat), cmap=sns.light_palette(MVIDEO_RED, as_cmap=True), ax=ax, "
            'linewidths=0.3, linecolor="white", cbar_kws={"label": "log(1 + частота)"})\n'
            'ax.set_title("Совместная встречаемость: запрос × бренд")\n'
            'ax.set_xlabel("Бренд"); ax.set_ylabel("Запрос")\n'
            'plt.xticks(rotation=45, ha="right")\n'
            'save_fig(fig, "06_query_brand_heatmap.png"); plt.show()'
        ),
        md("## TF-IDF сходство топ-запросов"),
        code(
            'queries = df["query_text"].value_counts().head(25).index.tolist()\n'
            'vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))\n'
            "sim = cosine_similarity(vec.fit_transform(queries))\n"
            "fig, ax = plt.subplots(figsize=(11, 9))\n"
            "sns.heatmap(sim, xticklabels=queries, yticklabels=queries, "
            "cmap=sns.light_palette(MVIDEO_RED, as_cmap=True), vmin=0, vmax=1, square=True, ax=ax, "
            'cbar_kws={"label": "Косинусное сходство (TF-IDF)"})\n'
            'ax.set_title("Матрица сходства топ-запросов (TF-IDF char n-grams)")\n'
            'plt.xticks(rotation=55, ha="right", fontsize=8); plt.yticks(fontsize=8)\n'
            'save_fig(fig, "07_tfidf_similarity_heatmap.png"); plt.show()'
        ),
        md("## Сходство названий брендов"),
        code(
            'brands = df["sku_brand_name"].replace("", np.nan).dropna().value_counts().head(20).index.tolist()\n'
            'sim_b = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(brands))\n'
            "fig, ax = plt.subplots(figsize=(10, 8))\n"
            "sns.heatmap(sim_b, xticklabels=brands, yticklabels=brands, "
            "cmap=sns.light_palette(DARK_SLATE, as_cmap=True), vmin=0, vmax=1, square=True, ax=ax, "
            'cbar_kws={"label": "Сходство названий брендов"})\n'
            'ax.set_title("TF-IDF сходство названий топ-брендов")\n'
            'plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)\n'
            'save_fig(fig, "14_brand_name_similarity.png"); plt.show()'
        ),
        md("## Co-occurrence: запрос × subject"),
        code(
            'top_subj = df["sku_subject_id"].value_counts().head(12).index\n'
            'sub2 = df[df["query_text"].isin(top_q) & df["sku_subject_id"].isin(top_subj)]\n'
            'mat2 = pd.crosstab(sub2["query_text"], sub2["sku_subject_id"].astype(str))\n'
            "fig, ax = plt.subplots(figsize=(12, 8))\n"
            "sns.heatmap(np.log1p(mat2), cmap=sns.light_palette(DARK_SLATE, as_cmap=True), ax=ax, "
            'linewidths=0.3, linecolor="white", cbar_kws={"label": "log(1 + частота)"})\n'
            'ax.set_title("Совместная встречаемость: запрос × subject_id")\n'
            'save_fig(fig, "18_query_subject_heatmap.png"); plt.show()'
        ),
    ],
)

write(
    "05_click_patterns.ipynb",
    [
        md(
            "# 05. Паттерны кликов\n\n"
            "Исследуем связь позиции в выдаче с кликами, ценой и брендом.\n\n"
            "Прокси CTR: доля кликов бренда на топ-позициях."
        ),
        code(
            SETUP
            + "\ndf = load_query_clicks(n=400_000, seed=42)\n"
            + "print(df.shape)"
        ),
        md("## Доля кликов по позициям"),
        code(
            'pos = df["sku_position"].dropna()\n'
            "pos = pos[pos <= 40]\n"
            "vc = pos.value_counts().sort_index(); share = vc / vc.sum(); cum = share.cumsum()\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            'ax.bar(share.index, share.values, color=MVIDEO_RED, alpha=0.85, label="Доля кликов")\n'
            'ax.plot(cum.index, cum.values, color=DARK_SLATE, lw=2, marker="o", ms=3, label="Накопленная доля")\n'
            'ax.set_title("Доля кликов по позициям и накопленный охват")\n'
            'ax.set_xlabel("Позиция"); ax.set_ylabel("Доля"); ax.legend()\n'
            'save_fig(fig, "15_position_click_share.png"); plt.show()\n'
            'print(f"Доля кликов на позициях 0–2: {share.loc[share.index <= 2].sum():.1%}")'
        ),
        md("## Цена vs позиция"),
        code(
            'sub = df[["sku_price", "sku_position"]].dropna()\n'
            'sub = sub[(sub["sku_price"] > 0) & (sub["sku_position"] <= 30)]\n'
            'sub = sub[sub["sku_price"] < sub["sku_price"].quantile(0.95)]\n'
            'agg = sub.groupby("sku_position")["sku_price"].median().reset_index()\n'
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            'ax.plot(agg["sku_position"], agg["sku_price"], color=MVIDEO_RED, lw=2.2, marker="o", ms=4)\n'
            'ax.fill_between(agg["sku_position"], agg["sku_price"], alpha=0.12, color=MVIDEO_RED)\n'
            'ax.set_title("Медианная цена vs позиция в выдаче")\n'
            'ax.set_xlabel("Позиция"); ax.set_ylabel("Медианная цена, ₽")\n'
            'save_fig(fig, "10_price_vs_position.png"); plt.show()'
        ),
        md("## Прокси CTR брендов: доля кликов на топ-3 позициях"),
        code(
            'bdf = df[df["sku_brand_name"].replace("", np.nan).notna()].copy()\n'
            'top_brands = bdf["sku_brand_name"].value_counts().head(15).index\n'
            'bdf = bdf[bdf["sku_brand_name"].isin(top_brands)]\n'
            'proxy = (bdf.assign(top3=bdf["sku_position"] <= 2)'
            '.groupby("sku_brand_name")["top3"].mean().sort_values())\n'
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "ax.barh(proxy.index.astype(str), proxy.values, color=MVIDEO_RED)\n"
            'ax.set_title("Прокси CTR: доля кликов бренда на позициях 0–2")\n'
            'ax.set_xlabel("Доля кликов на топ-3")\n'
            'save_fig(fig, "19_brand_top3_share.png"); plt.show()\n'
            'display(proxy.sort_values(ascending=False).to_frame("top3_share"))'
        ),
        md("## Средняя позиция по брендам"),
        code(
            'avg_pos = bdf.groupby("sku_brand_name")["sku_position"].median().sort_values()\n'
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "ax.barh(avg_pos.index.astype(str), avg_pos.values, color=DARK_SLATE)\n"
            'ax.set_title("Медианная позиция клика по топ-брендам"); ax.set_xlabel("Медианная позиция")\n'
            'save_fig(fig, "20_brand_median_position.png"); plt.show()'
        ),
        md("## Цена × позиция: hexbin"),
        code(
            'plot_df = df[(df["sku_price"] > 0) & (df["sku_position"] <= 25)].copy()\n'
            'plot_df = plot_df[plot_df["sku_price"] < plot_df["sku_price"].quantile(0.9)]\n'
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            'hb = ax.hexbin(plot_df["sku_position"], plot_df["sku_price"], gridsize=35, cmap="Reds", mincnt=5)\n'
            "cb = fig.colorbar(hb, ax=ax); cb.set_label(\"Число кликов\")\n"
            'ax.set_title("Плотность кликов: позиция × цена")\n'
            'ax.set_xlabel("Позиция"); ax.set_ylabel("Цена, ₽")\n'
            'save_fig(fig, "21_price_position_hexbin.png"); plt.show()'
        ),
    ],
)

print("Done.")
