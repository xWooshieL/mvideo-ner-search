# -*- coding: utf-8 -*-
"""Графики brand-классификатора в стиле презентации М.Видео (красно-белый)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "brand_clf"
OUT = ROOT / "figures" / "brand_clf"
OUT.mkdir(parents=True, exist_ok=True)

MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"
MVSOFT = "#FDECEC"

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
    "font.size": 11,
})

metrics = json.loads((ART / "train_metrics.json").read_text(encoding="utf-8"))
stats = json.loads((ART / "silver_brand_stats.json").read_text(encoding="utf-8"))
summary = pd.DataFrame(metrics["summary"])

NICE = {
    "sgd_char": "SGD · символьные n-граммы",
    "logreg_wordchar": "LogReg · слова + символы",
    "logreg_char_bal": "LogReg · симв., баланс классов",
    "logreg_char": "LogReg · символьные n-граммы",
}


def fig_models_compare():
    """Сравнение 4 моделей: macro-F1 и false brand rate — вертикально."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4))
    order = ["sgd_char", "logreg_wordchar", "logreg_char", "logreg_char_bal"]
    df = summary.set_index("model").loc[order]
    labels = [NICE[m] for m in order]
    y = np.arange(len(order))

    ax1.barh(y - 0.2, df["raw_f1_macro"], height=0.36, color=MVRED, label="без порогов (raw)")
    ax1.barh(y + 0.2, df["rej_f1_macro"], height=0.36, color=MVDARK, label="с reject-порогами")
    ax1.set_yticks(y, labels)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.invert_yaxis()
    ax1.set_xlim(0.80, 1.02)
    ax1.set_title("Macro-F1: сырой ответ и после порогов", fontweight="bold", fontsize=12)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=9)
    for i, (raw, rej) in enumerate(zip(df["raw_f1_macro"], df["rej_f1_macro"])):
        ax1.text(raw + 0.003, i - 0.2, f"{raw:.3f}", va="center", fontsize=8.5, color=MVRED, fontweight="bold")
        ax1.text(rej + 0.003, i + 0.2, f"{rej:.3f}", va="center", fontsize=8.5, color=MVDARK)

    short = ["SGD симв.", "LogReg сл.+симв.", "LogReg симв.", "LogReg симв. бал."]
    bars = ax2.bar(short, df["false_brand_cat"] * 100,
                   color=[MVDARK if m != "logreg_wordchar" else MVRED for m in order], width=0.55)
    ax2.set_title("Ложный бренд на запросах-категориях, % (цель — 0)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("«холодильник → Indesit», %")
    ax2.tick_params(axis="x", labelsize=9)
    ax2.set_ylim(0, 19)
    for b, v in zip(bars, df["false_brand_cat"] * 100):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout(h_pad=2.4)
    fig.savefig(OUT / "mv_01_models_compare.png", dpi=150)
    plt.close(fig)
    print("mv_01_models_compare.png")


def fig_silver_composition():
    """Из чего собран silver: причины меток + спец-классы."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4))

    reasons = stats["label_reason_counts"]
    nice_r = {
        "no_evidence_drop": "нет сигнала — выброшено",
        "alias_hint": "алиас в тексте («айфон»)",
        "brand_in_query": "бренд написан в запросе",
        "category_only": "чистая категория → NO_BRAND",
        "ood_brand_surface": "бренд вне top-K → UNKNOWN",
        "ambiguous_clicks": "клики разъехались → UNKNOWN",
        "low_conf_with_evidence": "слабый majority → UNKNOWN",
        "other_drop": "прочее — выброшено",
    }
    items = sorted(reasons.items(), key=lambda kv: kv[1])
    names = [nice_r.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [MVGRAY if "выброшено" in n else (MVRED if "→" in n else MVDARK) for n in names]
    ax1.barh(names, vals, color=colors, height=0.62)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.set_title("Причина метки → количество запросов", fontweight="bold", fontsize=12)
    for i, v in enumerate(vals):
        ax1.text(v + 250, i, f"{v:,}".replace(",", " "), va="center", fontsize=9)
    ax1.set_xlim(0, max(vals) * 1.16)

    sizes = [1 - stats["no_brand_share"] - stats["unknown_share"],
             stats["no_brand_share"], stats["unknown_share"]]
    labels = [f"бренды top-K\n{sizes[0]*100:.0f}%",
              f"NO_BRAND\n{sizes[1]*100:.0f}%",
              f"UNKNOWN\n{sizes[2]*100:.0f}%"]
    wedges, texts = ax2.pie(sizes, labels=labels, colors=[MVRED, MVDARK, MVGRAY],
                            startangle=90, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
                            labeldistance=1.12, textprops=dict(fontsize=10, fontweight="bold"))
    ax2.text(0, 0, f"{stats['n_silver']:,}".replace(",", " ") + "\nзапросов",
             ha="center", va="center", fontsize=12, fontweight="bold")
    ax2.set_title("Состав silver: 65 брендов + 2 спец-класса", fontweight="bold", fontsize=12)

    fig.tight_layout(h_pad=2.2)
    fig.savefig(OUT / "mv_02_silver_composition.png", dpi=150)
    plt.close(fig)
    print("mv_02_silver_composition.png")


def fig_per_class():
    """Per-class F1 лучшей модели: топ и хвост."""
    df = pd.read_csv(ART / "train_runs" / "per_class_f1__BEST.csv")
    df = df.sort_values("f1")

    worst = df.head(10)
    best = df.tail(10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4))

    def draw(ax, sub, title):
        colors = [MVRED if sp else MVDARK for sp in sub["is_special"]]
        ax.barh(sub["class"], sub["f1"], color=colors, height=0.62)
        ax.set_xlim(0, 1.18)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(title, fontweight="bold", fontsize=12)
        for i, (f1, sup) in enumerate(zip(sub["f1"], sub["support"])):
            ax.text(f1 + 0.015, i, f"{f1:.2f} · n={sup}", va="center", fontsize=8.5)

    draw(ax1, best.iloc[::-1], "Лучшие классы по F1 (валидация, 5 710 запросов)")
    draw(ax2, worst, "Хвост: где модель ошибается (красным — спец-классы)")

    fig.tight_layout(h_pad=2.4)
    fig.savefig(OUT / "mv_03_per_class_f1.png", dpi=150)
    plt.close(fig)
    print("mv_03_per_class_f1.png")


def fig_thresholds():
    """Как работают пороги: схема принятия ответа."""
    th = metrics["thresholds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ["TAU_ACCEPT\nмин. уверенность top-1", "TAU_MARGIN\nотрыв top-1 от top-2",
             "TAU_NO_BRAND\nпринять «бренда нет»", "TAU_UNKNOWN\nпринять «вне списка»"]
    vals = [th["TAU_ACCEPT"], th["TAU_MARGIN"], th["TAU_NO_BRAND"], th["TAU_UNKNOWN"]]
    bars = ax.bar(names, vals, color=[MVRED, MVDARK, MVDARK, MVGRAY], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.55)
    ax.set_title("Пороги принятия ответа: не прошёл — brand = null", fontweight="bold", fontsize=12)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "mv_04_thresholds.png", dpi=150)
    plt.close(fig)
    print("mv_04_thresholds.png")


if __name__ == "__main__":
    fig_models_compare()
    fig_silver_composition()
    fig_per_class()
    fig_thresholds()
    print("done")
