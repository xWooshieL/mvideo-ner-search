# -*- coding: utf-8 -*-
"""Перегенерация трёх графиков в вертикальную компоновку (график под графиком)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "complex_eda"

MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#DDDDDD",
    "axes.labelcolor": MVDARK,
    "text.color": MVDARK,
    "xtick.color": MVGRAY,
    "ytick.color": MVGRAY,
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})


def load_sample(n=200_000):
    df = pd.read_parquet(
        ROOT / "data" / "query_clicks.parquet",
        columns=["toValidUTF8(query_text)", "toValidUTF8(sku_brand_name)"],
    ).head(n)
    return df.rename(columns={
        "toValidUTF8(query_text)": "query_text",
        "toValidUTF8(sku_brand_name)": "brand_name",
    })


def fig_query_length(df):
    q = df["query_text"].astype(str)
    tokens = q.str.split().str.len()
    chars = q.str.len()

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6))

    axes[0].hist(tokens.clip(upper=15), bins=range(1, 16), color=MVRED, edgecolor="white")
    axes[0].axvline(tokens.median(), color=MVDARK, ls="--", lw=1.2, label=f"медиана = {int(tokens.median())}")
    axes[0].set_title("Длина запроса (токены)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("токенов")
    axes[0].legend(fontsize=8)

    axes[1].hist(chars.clip(upper=60), bins=40, color=MVDARK, edgecolor="white")
    axes[1].axvline(chars.median(), color=MVRED, ls="--", lw=1.2, label=f"медиана = {int(chars.median())}")
    axes[1].set_title("Длина запроса (символы)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("символов")
    axes[1].legend(fontsize=8)

    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIG / "01_query_length.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("01_query_length.png")


def fig_brand(df):
    stats = json.load(open(ROOT / "artifacts" / "complex_eda_method_stats.json", encoding="utf-8"))
    in_rate = stats["brand_in_query_rate"]
    absent = stats["brand_absent_rate"]

    top = df["brand_name"].dropna().value_counts().head(14)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6), height_ratios=[1, 1.5])

    bars = axes[0].bar(["бренд в тексте\nзапроса", "бренда в тексте\nнет (только клик)"],
                       [in_rate, absent], color=[MVRED, MVDARK], width=0.55)
    for b, v in zip(bars, [in_rate, absent]):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v*100:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.9)
    axes[0].set_title("Бренд в тексте запроса или только в клике", fontsize=11, fontweight="bold")

    axes[1].barh(top.index[::-1], top.values[::-1], color=MVRED)
    axes[1].set_title("Топ брендов по кликам (семпл)", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="y", labelsize=8)

    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIG / "04_brand_in_query_vs_click.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("04_brand_in_query_vs_click.png")


def fig_model_tag():
    stats = json.load(open(ROOT / "artifacts" / "model_tag_eda_stats.json", encoding="utf-8"))
    with_brand = stats["with_brand"]
    tails = stats["latin_o_tails_before"]
    clean = with_brand - tails
    # реконструкция гистограммы длин из ноутбука (медиана 2)
    tail_lens = {"1": 1113, "2": 1192, "3": 660, "4": 175, "5": 75, "6": 30, "7": 12, "8": 5}

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.6))

    axes[0].barh(["запросы\nс брендом"], [clean], color=MVDARK, label="хвост размечен или пуст")
    axes[0].barh(["запросы\nс брендом"], [tails], left=[clean], color=MVRED,
                 label="латинский хвост остался в O")
    axes[0].text(clean / 2, 0, f"{clean}", ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
    axes[0].text(clean + tails / 2, 0, f"{tails}\n({tails/with_brand*100:.0f}%)",
                 ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    axes[0].set_title("Проблема: хвост после бренда не размечен", fontsize=11, fontweight="bold")
    axes[0].set_xlabel(f"число запросов (семпл 15 000)")
    axes[0].legend(fontsize=7.5, loc="lower right")

    if tail_lens:
        ks = sorted(int(k) for k in tail_lens)
        vs = [tail_lens[str(k)] for k in ks]
        axes[1].bar(ks, vs, color=MVRED, edgecolor="white")
        axes[1].set_title("Длина неразмеченного хвоста после бренда", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("длина O-хвоста (токены)")
        axes[1].set_ylabel("число запросов")

    fig.tight_layout(h_pad=2.0)
    fig.savefig(FIG / "model_tag" / "01_missing_model_problem.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("model_tag/01_missing_model_problem.png")


if __name__ == "__main__":
    df = load_sample()
    fig_query_length(df)
    fig_brand(df)
    fig_model_tag()
    print("done")
