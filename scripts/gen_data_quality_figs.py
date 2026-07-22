"""Фигуры для блока «Качество данных» в презентации Дня 2.

1) eda_dq_completeness.png — донат: доля строк с полными / частично пустыми
   атрибутами карточки (бренд/название/цена).
2) eda_dq_issues.png — горизонтальный бар: масштаб найденных проблем
   (в % от семпла или в штуках).

Цифры считаются из того же семпла query_clicks, что и аудит
(scripts/audit_data_quality.py), чтобы всё было согласовано.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_utils import load_query_clicks

FIG = ROOT / "figures"
DOCS = ROOT / "docs" / "assets"
FIG.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

MVRED = "#F20601"
GREY = "#DADADA"
DARK = "#1C1C1E"
ORANGE = "#E8833A"

plt.rcParams.update({
    "font.size": 14,
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
})


def main() -> None:
    print("Загружаю семпл 600k (seed=7)...")
    df = load_query_clicks(n=600_000, seed=7)
    n = len(df)
    brand = df["sku_brand_name"].fillna("").astype(str).str.strip()
    name = df["sku_name"].fillna("").astype(str).str.strip()
    price = df["sku_price"]

    empty_brand = (brand == "")
    empty_name = (name == "")
    bad_price = (price <= 0)

    # полнота карточки: все три поля заполнены vs есть хотя бы одна дыра
    full = (~empty_brand) & (~empty_name) & (price > 0)
    partial = (~full) & ((~empty_brand) | (~empty_name))
    lost = empty_brand & empty_name  # карточка не подтянулась

    share_full = full.mean()
    share_lost = lost.mean()
    share_partial = 1 - share_full - share_lost

    print(f"full={share_full:.3f} partial={share_partial:.3f} lost={share_lost:.3f}")

    # --- 1) донат полноты ---
    fig, ax = plt.subplots(figsize=(6.6, 5.0), dpi=150)
    sizes = [share_full, share_partial, share_lost]
    labels = ["карточка полная\n(бренд+название+цена)",
              "частично пусто",
              "карточка потеряна\n(нет бренда и названия)"]
    colors = [GREY, ORANGE, MVRED]
    wedges, _t, autot = ax.pie(
        sizes, colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        autopct=lambda p: f"{p:.0f}%", pctdistance=0.79,
        labels=labels, labeldistance=1.16,
        textprops=dict(fontsize=13),
    )
    for a in autot:
        a.set_fontsize(15)
        a.set_fontweight("bold")
    autot[-1].set_color("white")
    ax.set_title("Полнота карточки в кликах", fontsize=17, fontweight="bold", pad=16)
    ax.set(aspect="equal")
    fig.tight_layout()
    for out in (FIG / "eda_dq_completeness.png", DOCS / "eda_dq_completeness.png"):
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # --- 2) бар масштаба проблем (% от семпла) ---
    uq = df["query_text"].dropna().astype(str)
    uq = uq[uq.str.strip() != ""].drop_duplicates()
    mixed_q = uq[uq.apply(lambda s: bool(re.search("[а-яё]", s, re.I)) and bool(re.search("[a-z]", s, re.I)))]
    share_mixed = len(mixed_q) / max(len(uq), 1)

    dup_rows = df.duplicated().mean()
    zero_pos = (df["sku_position"] == 0).mean() if "sku_position" in df else 0.0

    metrics = [
        ("Смешанная раскладка\nв запросах", share_mixed),
        ("Пустой бренд", empty_brand.mean()),
        ("Пустое название", empty_name.mean()),
        ("Позиция = 0\n(нет данных)", zero_pos),
        ("Полные дубли строк", dup_rows),
        ("Цена ≤ 0", bad_price.mean()),
    ]
    metrics.sort(key=lambda x: x[1])
    lbls = [m[0] for m in metrics]
    vals = [m[1] * 100 for m in metrics]

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150)
    bars = ax.barh(lbls, vals, color=MVRED, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=12, fontweight="bold", color=DARK)
    ax.set_title("Масштаб проблем данных (% от семпла)", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("% строк / уникальных запросов")
    ax.set_xlim(0, max(vals) * 1.18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for out in (FIG / "eda_dq_issues.png", DOCS / "eda_dq_issues.png"):
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("data-quality figures saved")


if __name__ == "__main__":
    main()
