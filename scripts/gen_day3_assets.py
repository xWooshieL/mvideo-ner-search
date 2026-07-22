# -*- coding: utf-8 -*-
"""День 3: перегенерация графика model_tag/01 + выгрузка данных для разметки.

1. figures/complex_eda/model_tag/01_missing_model_problem.png — без наложений.
2. apps/labeling/data/queries_<name>.json — 3×1500 непересекающихся запросов.
3. apps/labeling/data/pairs_<name>.json — пары (запрос, карточка) для 1/0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "complex_eda" / "model_tag"
OUT = ROOT / "apps" / "labeling" / "data"
OUT.mkdir(parents=True, exist_ok=True)

MVRED = "#F20601"
MVDARK = "#1C1C1E"
MVGRAY = "#6E6E73"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#DDDDDD",
    "axes.grid": True,
    "grid.color": "#EEEEEE",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def regen_model_tag_fig() -> None:
    stats = json.load(open(ROOT / "artifacts" / "model_tag_eda_stats.json", encoding="utf-8"))
    with_brand = stats["with_brand"]
    tails = stats["latin_o_tails_before"]
    clean = with_brand - tails

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # левая панель: горизонтальный стек вместо наложенного вертикального
    ax = axes[0]
    ax.barh(["запросы\nс брендом"], [clean], color=MVDARK, label="хвост размечен или пуст")
    ax.barh(["запросы\nс брендом"], [tails], left=[clean], color=MVRED,
            label="латинский хвост остался в O")
    ax.set_xlim(0, with_brand * 1.05)
    ax.set_title("Проблема: хвост после бренда не размечен", fontsize=11, color=MVDARK, pad=12)
    ax.set_xlabel("число запросов (семпл 15 000)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.text(clean / 2, 0, f"{clean}", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(clean + tails / 2, 0, f"{tails}\n(46%)", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")

    # правая панель: длина хвоста
    ax2 = axes[1]
    # реконструкция гистограммы длин из ноутбука (медиана 2)
    lens = [1] * 1113 + [2] * 1192 + [3] * 660 + [4] * 175 + [5] * 75 + [6] * 30 + [7] * 12 + [8] * 5
    ax2.hist(lens, bins=range(1, 10), color=MVRED, edgecolor="white", linewidth=1.2)
    ax2.set_title("Длина неразмеченного хвоста после бренда", fontsize=11, color=MVDARK, pad=12)
    ax2.set_xlabel("длина O-хвоста (токены)")
    ax2.set_ylabel("число запросов")

    fig.tight_layout()
    fig.savefig(FIG / "01_missing_model_problem.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figure regenerated:", FIG / "01_missing_model_problem.png")


def export_labeling_data() -> None:
    rng = random.Random(42)
    print("reading parquet sample...")
    df = pd.read_parquet(
        ROOT / "data" / "query_clicks.parquet",
        columns=["toValidUTF8(query_text)", "toValidUTF8(sku_name)",
                 "toValidUTF8(sku_brand_name)", "sku_price", "sku_position", "sku_id"],
    ).head(3_000_000)
    df.columns = ["query", "sku_name", "brand", "price", "position", "sku_id"]
    df["query"] = df["query"].astype(str).str.strip()
    df = df[(df["query"].str.len() >= 3) & (df["query"].str.len() <= 80)]

    # частотные запросы, чтобы разметка была осмысленной
    vc = df["query"].value_counts()
    pool = [q for q in vc.index if vc[q] >= 2][:40_000]
    rng.shuffle(pool)
    need = 1500 * 3
    uniq = pool[:need]
    if len(uniq) < need:
        rest = [q for q in vc.index if q not in set(uniq)]
        uniq += rest[: need - len(uniq)]
    print("unique queries picked:", len(uniq))

    annotators = ["nikita", "nekit", "liza"]
    display = {"nikita": "Никита", "nekit": "Некит", "liza": "Лиза"}
    for i, name in enumerate(annotators):
        chunk = uniq[i * 1500: (i + 1) * 1500]
        (OUT / f"queries_{name}.json").write_text(
            json.dumps({"annotator": display[name], "queries": chunk},
                       ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
        print(f"queries_{name}.json: {len(chunk)}")

    # пары для 1/0: к каждому запросу — реально кликнутая карточка (или случайная как hard negative)
    first_click = df.drop_duplicates("query").set_index("query")
    all_rows = df.sample(min(len(df), 200_000), random_state=42)
    for i, name in enumerate(annotators):
        chunk = uniq[i * 1500: (i + 1) * 1500]
        pairs = []
        for j, q in enumerate(chunk):
            if q in first_click.index and rng.random() > 0.35:
                row = first_click.loc[q]
                pairs.append({
                    "query": q,
                    "sku_name": str(row["sku_name"]),
                    "brand": str(row["brand"]),
                    "price": float(row["price"]) if pd.notna(row["price"]) else None,
                    "kind": "clicked",
                })
            else:
                row = all_rows.iloc[rng.randrange(len(all_rows))]
                pairs.append({
                    "query": q,
                    "sku_name": str(row["sku_name"]),
                    "brand": str(row["brand"]),
                    "price": float(row["price"]) if pd.notna(row["price"]) else None,
                    "kind": "random",
                })
        (OUT / f"pairs_{name}.json").write_text(
            json.dumps({"annotator": display[name], "pairs": pairs},
                       ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
        print(f"pairs_{name}.json: {len(pairs)}")


if __name__ == "__main__":
    regen_model_tag_fig()
    export_labeling_data()
    print("done")
