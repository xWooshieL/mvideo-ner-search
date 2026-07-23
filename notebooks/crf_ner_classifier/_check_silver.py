"""Quick health check for silver datasets (ner / attr_type / brand_clf)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_utils import (
    brands_path,
    categories_path,
    model_phrases_path,
    resolve_silver,
)


def main() -> None:
    failed = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("OK  " if cond else "FAIL"), name, detail)
        if not cond:
            failed.append(name)

    for p, lab in [
        (brands_path(), "brands"),
        (categories_path(), "categories"),
        (model_phrases_path(), "model_phrases"),
    ]:
        check(f"dict:{lab}", p.exists() and p.stat().st_size > 0, str(p))

    ner = resolve_silver("ner_bio", "silver_bio_slice.parquet")
    check("ner_slice", ner.exists(), str(ner))
    if ner.exists():
        df = pd.read_parquet(ner)
        need = ["query", "tokens", "tags", "n_BRAND", "n_CATEGORY", "n_MODEL", "n_ATTR", "has_entity"]
        miss = [c for c in need if c not in df.columns]
        n_ent = int(df["has_entity"].sum()) if "has_entity" in df.columns else -1
        check("ner_cols", not miss, f"rows={len(df)} with_ent={n_ent} miss={miss}")
        sample = df.head(min(2000, len(df)))
        bad = sum(1 for _, r in sample.iterrows() if len(r["tokens"]) != len(r["tags"]))
        check("ner_tok_tag_align", bad == 0, f"bad_in_{len(sample)}={bad}")
        bc: Counter[str] = Counter()
        for tags in df.loc[df["has_entity"], "tags"].head(5000):
            for t in tags:
                if str(t).startswith("B-"):
                    bc[str(t)[2:]] += 1
        check("ner_has_MODEL", bc.get("MODEL", 0) > 0, dict(bc))

    raw = resolve_silver("attr_type", "attr_type_silver_raw.parquet")
    check("attr_raw", raw.exists(), str(raw))
    if raw.exists():
        a = pd.read_parquet(raw)
        need = ["span_text", "y", "query_masked_all_attr"]
        miss = [c for c in need if c not in a.columns]
        ys = set(a["y"].astype(str)) if "y" in a.columns else set()
        check(
            "attr_cols",
            not miss,
            f"rows={len(a)} nunique_y={a['y'].nunique() if 'y' in a.columns else None} miss={miss}",
        )
        check("attr_has_type_or_purpose", ("type" in ys) or ("purpose" in ys), sorted(ys)[:12])

    bt = resolve_silver("brand_clf", "silver_brand_train.parquet")
    check("brand_train", bt.exists(), str(bt))
    if bt.exists():
        b = pd.read_parquet(bt)
        need = ["query_norm", "brand"]
        miss = [c for c in need if c not in b.columns]
        check(
            "brand_cols",
            not miss,
            f"rows={len(b)} brands={b['brand'].nunique() if 'brand' in b.columns else None} miss={miss}",
        )

    print("---")
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)
    print("ALL SILVER OK")


if __name__ == "__main__":
    main()
