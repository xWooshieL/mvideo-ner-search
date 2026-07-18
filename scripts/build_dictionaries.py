#!/usr/bin/env python
"""Build brand and category dictionaries from click parquet (weak supervision)."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ner.labeling import CATEGORY_SEEDS, _normalize  # noqa: E402

COL_BRAND = "toValidUTF8(sku_brand_name)"
COL_NAME = "toValidUTF8(sku_name)"
COL_QUERY = "toValidUTF8(query_text)"


def iter_row_groups(path: Path, max_rows: int | None = None):
    pf = pq.ParquetFile(path)
    seen = 0
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=[COL_BRAND, COL_NAME, COL_QUERY])
        df = t.to_pandas()
        df.columns = ["brand", "name", "query"]
        yield df
        seen += len(df)
        if max_rows and seen >= max_rows:
            break


SHORT_BRAND_WHITELIST = {"lg", "hp", "jbl", "bq", "tcl", "msi", "bbk", "aoc", "nec", "akg"}


def extract_brands(df, counter: Counter, min_len: int = 2) -> None:
    for b in df["brand"].dropna().astype(str):
        b = b.strip()
        if len(b) < min_len:
            continue
        # skip numeric junk / overly short noisy tokens (keep known short brands)
        if b.isdigit():
            continue
        if len(b) <= 2 and b.lower() not in SHORT_BRAND_WHITELIST:
            continue
        counter[b] += 1


def extract_categories_from_queries(df, counter: Counter) -> None:
    """Short single/double-token queries often are category intents."""
    for q in df["query"].dropna().astype(str):
        qn = _normalize(q)
        if not qn or len(qn) > 40:
            continue
        tokens = qn.split()
        if 1 <= len(tokens) <= 3 and all(re.search(r"[а-яa-z]", t) for t in tokens):
            # prefer cyrillic-heavy category-like queries
            if any(t in CATEGORY_SEEDS for t in tokens) or len(tokens) <= 2:
                counter[qn] += 1


def extract_categories_from_names(df, counter: Counter) -> None:
    for name in df["name"].dropna().astype(str):
        name_n = _normalize(name)
        if not name_n:
            continue
        for seed in CATEGORY_SEEDS:
            if seed in name_n:
                counter[seed] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "файлы" / "query_clicks.parquet",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--max-rows", type=int, default=500_000)
    ap.add_argument("--min-brand-count", type=int, default=5)
    ap.add_argument("--min-category-count", type=int, default=20)
    ap.add_argument("--top-brands", type=int, default=5000)
    ap.add_argument("--top-categories", type=int, default=2000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    brand_c: Counter = Counter()
    cat_c: Counter = Counter()

    print(f"Reading up to {args.max_rows} rows from {args.data} ...")
    for df in iter_row_groups(args.data, args.max_rows):
        extract_brands(df, brand_c)
        extract_categories_from_queries(df, cat_c)
        extract_categories_from_names(df, cat_c)

    # Always include seeds
    for s in CATEGORY_SEEDS:
        cat_c[s] += args.min_category_count

    brands = [
        b
        for b, c in brand_c.most_common(args.top_brands * 2)
        if c >= args.min_brand_count
        and (
            len(b) >= 3
            or b.lower() in SHORT_BRAND_WHITELIST
        )
    ][: args.top_brands]
    brand_norm = {_normalize(b) for b in brands}

    # Categories: drop entries that are actually brands (e.g. iphone/samsung as query)
    categories = []
    for c, n in cat_c.most_common(args.top_categories * 3):
        if n < args.min_category_count or not (2 <= len(c) <= 40):
            continue
        cn = _normalize(c)
        if cn in brand_norm:
            continue
        # drop pure latin single tokens that look like product lines/brands
        toks = cn.split()
        if len(toks) == 1 and re.fullmatch(r"[a-z0-9\-\.]+", toks[0]) and toks[0] not in CATEGORY_SEEDS:
            continue
        categories.append(c)
        if len(categories) >= args.top_categories:
            break

    brands_path = args.out / "brands.txt"
    cats_path = args.out / "categories.txt"
    brands_path.write_text("\n".join(brands) + "\n", encoding="utf-8")
    cats_path.write_text("\n".join(categories) + "\n", encoding="utf-8")

    print(f"Saved {len(brands)} brands → {brands_path}")
    print(f"Saved {len(categories)} categories → {cats_path}")
    print("Top brands:", brands[:15])
    print("Top categories:", categories[:15])


if __name__ == "__main__":
    main()
