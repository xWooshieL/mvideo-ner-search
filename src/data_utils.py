"""
Общие утилиты загрузки и семплирования данных кейса M.Video Search/NER.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------



ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
ARTIFACTS_DIR = ROOT / "artifacts"
NOTEBOOKS_DIR = ROOT / "notebooks"
ARTIFACTS = ARTIFACTS_DIR
FIGURES = FIGURES_DIR
MODELS = ROOT / "models"

# --- каноническая раскладка артефактов (legacy пути тоже поддерживаются) ---
DICTS_DIR = ARTIFACTS_DIR / "dicts"
SILVER_DIR = ARTIFACTS_DIR / "silver"
SILVER_NER_BIO = SILVER_DIR / "ner_bio"
SILVER_ATTR_TYPE = SILVER_DIR / "attr_type"
SILVER_BRAND_CLF = SILVER_DIR / "brand_clf"

# метрики/joblib типизатора и brand clf остаются рядом с «рантаймом»
ATTR_TYPE_DIR = ARTIFACTS_DIR / "attr_type"
BRAND_CLF_DIR = ARTIFACTS_DIR / "brand_clf"
NER_DIR_LEGACY = ARTIFACTS_DIR / "ner"

DICT_FILENAMES = (
    "brands.txt",
    "categories.txt",
    "model_phrases.txt",
    "protected_brands.txt",
)

QUERY_CLICKS_PATH = DATA_DIR / "query_clicks.parquet"
SKU_DESC_PATH = DATA_DIR / "sku_desc.parquet"
SKUS_PKL_PATH = DATA_DIR / "skus.pkl"

# Переименование «сырых» ClickHouse-колонок
QUERY_CLICKS_RENAME = {
    "toValidUTF8(sku_name)": "sku_name",
    "toValidUTF8(sku_brand_name)": "sku_brand_name",
    "toValidUTF8(query_text)": "query_text",
}
QUERY_RENAME = QUERY_CLICKS_RENAME

# Палитра в духе M.Video
MVIDEO_RED = "#E31E24"
DARK_SLATE = "#1F2937"
SLATE = "#334155"
MUTED = "#64748B"
LIGHT_BG = "#F8FAFC"
ACCENT_GRAY = "#94A3B8"
PALETTE = [MVIDEO_RED, DARK_SLATE, "#DC2626", "#475569", "#B91C1C", "#0F172A", MUTED, "#991B1B"]

DPI = 200


def ensure_dirs() -> None:
    """Создаёт figures/, artifacts/, models/ и каноническую раскладку dicts/silver."""
    for p in (
        FIGURES_DIR,
        ARTIFACTS_DIR,
        MODELS,
        ROOT / "docs" / "figures",
        DICTS_DIR,
        SILVER_NER_BIO,
        SILVER_ATTR_TYPE,
        SILVER_BRAND_CLF,
        ATTR_TYPE_DIR,
        BRAND_CLF_DIR,
        NER_DIR_LEGACY,
    ):
        p.mkdir(parents=True, exist_ok=True)


def resolve_dict(name: str) -> Path:
    """Словарь: `artifacts/dicts/<name>` → fallback `artifacts/<name>`."""
    new = DICTS_DIR / name
    if new.exists():
        return new
    return ARTIFACTS_DIR / name


def brands_path() -> Path:
    return resolve_dict("brands.txt")


def categories_path() -> Path:
    return resolve_dict("categories.txt")


def model_phrases_path() -> Path:
    return resolve_dict("model_phrases.txt")


def protected_brands_path() -> Path:
    return resolve_dict("protected_brands.txt")


def _silver_dirs(kind: str) -> tuple[Path, Path]:
    """(canonical, legacy) для kind: ner_bio | attr_type | brand_clf."""
    if kind == "ner_bio":
        return SILVER_NER_BIO, NER_DIR_LEGACY
    if kind == "attr_type":
        return SILVER_ATTR_TYPE, ATTR_TYPE_DIR
    if kind == "brand_clf":
        return SILVER_BRAND_CLF, BRAND_CLF_DIR
    raise ValueError(f"unknown silver kind: {kind}")


def resolve_silver(kind: str, name: str) -> Path:
    """Файл silver: canonical → legacy. Если нет нигде — путь canonical (для записи)."""
    canon, legacy = _silver_dirs(kind)
    p_new, p_old = canon / name, legacy / name
    if p_new.exists():
        return p_new
    if p_old.exists():
        return p_old
    return p_new


def save_silver_parquet(df: pd.DataFrame, kind: str, name: str, *, mirror: bool = True) -> Path:
    """Пишет silver parquet в canonical; при mirror=True дублирует в legacy (ничего не ломаем)."""
    ensure_dirs()
    canon, legacy = _silver_dirs(kind)
    primary = canon / name
    df.to_parquet(primary, index=False)
    if mirror:
        secondary = legacy / name
        if secondary.resolve() != primary.resolve():
            secondary.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(secondary, index=False)
    return primary


def sync_artifact_layout(*, dry_run: bool = False) -> dict[str, list[str]]:
    """Идемпотентно копирует legacy → canonical. Исходники не удаляет."""
    ensure_dirs()
    report: dict[str, list[str]] = {"dicts": [], "ner_bio": [], "attr_type": [], "brand_clf": []}

    def _copy(src: Path, dst: Path, bucket: str) -> None:
        if not src.exists() or not src.is_file():
            return
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return
        if dry_run:
            report[bucket].append(f"would_copy {src.name} -> {dst}")
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        report[bucket].append(str(dst.relative_to(ROOT)))

    for name in DICT_FILENAMES:
        _copy(ARTIFACTS_DIR / name, DICTS_DIR / name, "dicts")

    ner_names = [
        "silver_bio_slice.parquet",
        "silver_bio_preview.parquet",
        "silver_overview.csv",
        "eda_meta.json",
        "eda_stats.json",
    ]
    for name in ner_names:
        _copy(NER_DIR_LEGACY / name, SILVER_NER_BIO / name, "ner_bio")

    attr_silver = [
        "attr_type_silver.parquet",
        "attr_type_silver_raw.parquet",
        "attr_type_silver_prod.parquet",
        "attr_type_all.parquet",
        "attr_type_train.parquet",
        "attr_type_val.parquet",
        "attr_type_train_prod.parquet",
        "attr_type_val_prod.parquet",
        "attr_type_silver_meta.json",
        "attr_type_silver_overview.csv",
    ]
    for name in attr_silver:
        _copy(ATTR_TYPE_DIR / name, SILVER_ATTR_TYPE / name, "attr_type")

    brand_silver = [
        "silver_brand_all.parquet",
        "silver_brand_train.parquet",
        "silver_brand_val.parquet",
        "silver_brand_stats.json",
        "silver_brand_train_preview.csv",
        "silver_clf_readme.md",
    ]
    for name in brand_silver:
        _copy(BRAND_CLF_DIR / name, SILVER_BRAND_CLF / name, "brand_clf")

    return report


def parquet_num_rows(path: Path | str) -> int:
    """Число строк в parquet без полной загрузки."""
    return int(pq.ParquetFile(str(path)).metadata.num_rows)


def parquet_schema_names(path: Path | str) -> list[str]:
    """Имена колонок parquet."""
    return list(pq.ParquetFile(str(path)).schema_arrow.names)


def _rename_query_clicks(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=QUERY_CLICKS_RENAME)


def load_query_clicks(
    n: Optional[int] = 300_000,
    nrows: Optional[int] = None,
    columns: Optional[list[str]] = None,
    seed: int = 42,
    random_state: Optional[int] = None,
    random: bool = True,
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Загружает query_clicks.parquet.

    Parameters
    ----------
    n / nrows : int | None
        Размер выборки. None — весь файл (осторожно: ~31M строк).
    columns : list[str] | None
        Колонки после переименования (sku_name, query_text, ...).
    seed / random_state : int
        Зерно для воспроизводимого семпла.
    random : bool
        True — случайный семпл по row groups; False — первые n строк.
    """
    if nrows is not None:
        n = nrows
    if random_state is not None:
        seed = random_state

    path = QUERY_CLICKS_PATH
    pf = pq.ParquetFile(str(path))
    total = pf.metadata.num_rows
    raw_cols = None
    if columns is not None:
        inv = {v: k for k, v in QUERY_CLICKS_RENAME.items()}
        raw_cols = [inv.get(c, c) for c in columns]

    if sample_frac is not None:
        n = max(1, int(total * sample_frac))

    if n is None or n >= total:
        table = pf.read(columns=raw_cols)
        return _rename_query_clicks(table.to_pandas())

    if not random:
        chunks: list[pd.DataFrame] = []
        taken = 0
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i, columns=raw_cols)
            part = _rename_query_clicks(t.to_pandas())
            need = n - taken
            if len(part) > need:
                part = part.iloc[:need]
            chunks.append(part)
            taken += len(part)
            if taken >= n:
                break
        return pd.concat(chunks, ignore_index=True)

    rng = np.random.default_rng(seed)
    n_rg = pf.num_row_groups
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    weights = np.array(rg_sizes, dtype=float)
    weights /= weights.sum()
    per_rg = rng.multinomial(n, weights)

    chunks = []
    for i, k in enumerate(per_rg):
        if k == 0:
            continue
        t = pf.read_row_group(i, columns=raw_cols)
        part = _rename_query_clicks(t.to_pandas())
        if k < len(part):
            idx = rng.choice(len(part), size=k, replace=False)
            part = part.iloc[idx]
        chunks.append(part)
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def load_sku_desc(
    n: Optional[int] = 200_000,
    nrows: Optional[int] = None,
    columns: Optional[list[str]] = None,
    seed: int = 42,
    random: bool = True,
) -> pd.DataFrame:
    """Загружает sku_desc.parquet (title, description)."""
    if nrows is not None:
        n = nrows
    path = SKU_DESC_PATH
    pf = pq.ParquetFile(str(path))
    total = pf.metadata.num_rows

    if n is None or n >= total:
        return pf.read(columns=columns).to_pandas()

    if not random:
        chunks: list[pd.DataFrame] = []
        taken = 0
        for i in range(pf.num_row_groups):
            part = pf.read_row_group(i, columns=columns).to_pandas()
            need = n - taken
            if len(part) > need:
                part = part.iloc[:need]
            chunks.append(part)
            taken += len(part)
            if taken >= n:
                break
        return pd.concat(chunks, ignore_index=True)

    rng = np.random.default_rng(seed)
    n_rg = pf.num_row_groups
    rg_sizes = [pf.metadata.row_group(i).num_rows for i in range(n_rg)]
    weights = np.asarray(rg_sizes, dtype=float)
    weights /= weights.sum()
    per_rg = rng.multinomial(n, weights)

    chunks = []
    for i, k in enumerate(per_rg):
        if k == 0:
            continue
        part = pf.read_row_group(i, columns=columns).to_pandas()
        if k < len(part):
            idx = rng.choice(len(part), size=k, replace=False)
            part = part.iloc[idx]
        chunks.append(part)
    df = pd.concat(chunks, ignore_index=True)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def load_skus_catalog(max_offers: Optional[int] = 5_000) -> dict[str, Any]:
    """
    Загружает skus.pkl (YML-каталог).

    Возвращает dict с ключами raw / meta / offers_sample.
    Для обратной совместимости: если вызвать без использования ключей —
    используйте load_skus_raw().
    """
    with open(SKUS_PKL_PATH, "rb") as f:
        raw = pickle.load(f)

    meta: dict[str, Any] = {"type": type(raw).__name__}
    offers_sample = pd.DataFrame()

    if isinstance(raw, dict):
        meta["keys"] = list(raw.keys())
        catalog = raw.get("yml_catalog", raw)
        if isinstance(catalog, dict):
            meta["catalog_keys"] = list(catalog.keys())
            shop = catalog.get("shop", catalog)
            if isinstance(shop, dict):
                meta["shop_keys"] = list(shop.keys())
                offers = shop.get("offers") or shop.get("offer")
                items: list = []
                if isinstance(offers, dict):
                    items = list(offers.values())
                    if items and not isinstance(items[0], dict):
                        items = offers.get("offer", items) if isinstance(offers.get("offer"), list) else []
                elif isinstance(offers, list):
                    items = offers
                meta["n_offers_raw"] = len(items) if hasattr(items, "__len__") else None
                if items:
                    rows = []
                    for o in items[: max_offers or len(items)]:
                        if not isinstance(o, dict):
                            continue
                        rows.append(
                            {
                                "id": o.get("@id") or o.get("id"),
                                "name": o.get("name") or o.get("model"),
                                "vendor": o.get("vendor") or o.get("brand"),
                                "price": o.get("price"),
                                "categoryId": o.get("categoryId"),
                                "url": o.get("url"),
                            }
                        )
                    offers_sample = pd.DataFrame(rows)
                    meta["n_offers_sample"] = len(offers_sample)

    return {"raw": raw, "meta": meta, "offers_sample": offers_sample}


def apply_plot_style() -> None:
    """Единый стиль matplotlib/seaborn для всех ноутбуков."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.05,
        rc={
            "axes.facecolor": LIGHT_BG,
            "figure.facecolor": "white",
            "axes.edgecolor": ACCENT_GRAY,
            "axes.labelcolor": DARK_SLATE,
            "text.color": DARK_SLATE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "grid.color": "#E2E8F0",
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
        },
    )
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, name: str, dpi: int = DPI) -> Path:
    """Сохраняет фигуру в figures/ с высоким DPI."""
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Сохранено: {path}")
    return path


def dataset_overview_stats(sample_clicks: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Ключевые статистики датасета (полные размеры + оценки по семплу)."""
    stats: dict[str, Any] = {
        "query_clicks_full_rows": parquet_num_rows(QUERY_CLICKS_PATH),
        "sku_desc_full_rows": parquet_num_rows(SKU_DESC_PATH),
        "query_clicks_columns_raw": parquet_schema_names(QUERY_CLICKS_PATH),
        "sku_desc_columns": parquet_schema_names(SKU_DESC_PATH),
        "skus_pkl_size_bytes": SKUS_PKL_PATH.stat().st_size if SKUS_PKL_PATH.exists() else None,
    }
    if sample_clicks is not None and len(sample_clicks) > 0:
        df = sample_clicks
        stats["sample_rows"] = int(len(df))
        stats["sample_unique_queries"] = int(df["query_text"].nunique())
        stats["sample_unique_skus"] = int(df["sku_id"].nunique())
        stats["sample_unique_brands"] = int(df["sku_brand_name"].nunique())
        stats["sample_unique_subjects"] = int(df["sku_subject_id"].nunique())
        stats["sample_price_median"] = float(df["sku_price"].median())
        stats["sample_price_mean"] = float(df["sku_price"].mean())
        stats["sample_null_counts"] = {c: int(df[c].isna().sum()) for c in df.columns}
    return stats


def save_stats(stats: dict[str, Any], name: str = "dataset_stats.json") -> Path:
    ensure_dirs()
    path = ARTIFACTS_DIR / name

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=_default)
    print(f"Статистика сохранена: {path}")
    return path


def text_len(series: pd.Series) -> pd.Series:
    """Длина строки; NaN → 0."""
    return series.fillna("").astype(str).str.len()
