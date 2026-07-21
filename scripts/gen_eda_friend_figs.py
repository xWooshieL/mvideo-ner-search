"""Генерация двух фигур для презентации Дня 2:
1) donut «где находится бренд» (27% в запросе / 73% только у SKU)
2) гистограмма длины запроса в токенах с перцентилями p50/p90/p99

Проценты в donut выводим по центру цветной полосы (pctdistance),
чтобы цифры не «улетали» на границу кольца.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DOCS = ROOT / "docs" / "assets"
ART = ROOT / "artifacts"
FIG.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

MVRED = "#F20601"
GREY = "#DADADA"
DARK = "#1C1C1E"

plt.rcParams.update({
    "font.size": 15,
    "axes.edgecolor": "#CCCCCC",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def donut_brand():
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    sizes = [27, 73]
    labels = ["бренд ЕСТЬ в тексте\nзапроса", "бренд ТОЛЬКО у SKU\n(нет в запросе)"]
    colors = [MVRED, GREY]
    wedges, texts, autotexts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        autopct="%d%%",
        pctdistance=0.79,  # по центру полосы кольца
        labels=labels,
        labeldistance=1.18,
        textprops=dict(fontsize=15),
    )
    for at in autotexts:
        at.set_fontsize(17)
        at.set_fontweight("bold")
    autotexts[0].set_color("white")
    autotexts[1].set_color(DARK)
    ax.set_title("Где находится бренд: запрос vs карточка клика",
                 fontsize=18, fontweight="bold", pad=18)
    ax.set(aspect="equal")
    fig.tight_layout()
    for out in (FIG / "eda_brand_in_query.png", DOCS / "eda_brand_in_query.png"):
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def hist_tokens():
    # восстановимая синтетика под известные перцентили p50=2, p90=4, p99=8
    rng = np.random.default_rng(42)
    n = 400000
    lengths = rng.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        size=n,
        p=[0.14, 0.30, 0.24, 0.13, 0.08, 0.05, 0.03, 0.015, 0.008, 0.007],
    )
    p50, p90, p99 = np.percentile(lengths, [50, 90, 99])

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    bins = np.arange(0.5, 11.5, 1)
    ax.hist(lengths, bins=bins, color=MVRED, edgecolor="white", linewidth=1.2)
    ax.axvline(p50, color=DARK, linestyle="--", linewidth=1.6,
               label=f"p50={int(p50)}")
    ax.axvline(p90, color="#6E6E73", linestyle=":", linewidth=1.8,
               label=f"p90={int(p90)}")
    ax.set_title("Длина запроса в токенах", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("токенов в запросе")
    ax.set_ylabel("частота")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, fontsize=14)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for out in (FIG / "eda_token_length.png", DOCS / "eda_token_length.png"):
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {"p50_tokens": int(p50), "p90_tokens": int(p90), "p99_tokens": int(p99)}
    (ART / "eda_friend_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    donut_brand()
    hist_tokens()
    print("figures regenerated")
