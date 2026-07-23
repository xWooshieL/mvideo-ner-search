"""Smoke / headless build of ATTR-type silver (WeakLabeler + EDA artifacts)."""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data_utils import (
    ARTIFACTS_DIR,
    METRICS_DIR,
    ensure_dirs,
    load_query_clicks,
    save_stats,
    brands_path,
    categories_path,
    model_phrases_path,
    save_silver_parquet,
    ATTR_TYPE_DIR,
)
from src.ner.labeling import (
    ATTR_PATTERNS,
    WeakLabeler,
    _guess_attr_type,
    bio_to_entities,
    entities_to_structured,
)
from src.preprocessing.pipeline import basic_clean, _norm_key

warnings.filterwarnings("ignore", category=FutureWarning)

SAMPLE_N = 120_000
MAX_QUERIES = 25_000
SEED = 42
MIN_SPAN_LEN = 1
RARE_SUPPORT = 15

UNIT_AUG = {
    "гб": ["gb", "кб", "kb"],
    "gb": ["гб", "кб", "kb"],
    "тб": ["tb"],
    "tb": ["тб"],
    "мб": ["mb"],
    "mb": ["мб"],
    "вт": ["w"],
    "w": ["вт"],
    "кг": ["kg"],
    "kg": ["кг"],
}

_MULTI_SPACE = re.compile(r"\s+")
_ENT_TOKEN = {
    "BRAND": "<BRAND>",
    "CATEGORY": "<CAT>",
    "MODEL": "<MODEL>",
    "ATTR": "<ATTR>",
    "GENRE": "<GENRE>",
    "PERSON": "<PERSON>",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _mask_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    chars = list(text)
    for a, b, repl in sorted(spans, key=lambda x: -x[0]):
        chars[a:b] = list(repl)
    return _MULTI_SPACE.sub(" ", "".join(chars)).strip()


def mask_all_attr(text: str, attr_ents: list[dict]) -> str:
    spans = [(e["span"][0], e["span"][1], "<ATTR>") for e in attr_ents if "span" in e]
    return _mask_spans(text, spans)


def mask_keep_span(text: str, attr_ents: list[dict], keep: dict) -> str:
    ka, kb = (keep.get("span") or [None, None])[:2]
    spans = []
    for e in attr_ents:
        if "span" not in e:
            continue
        a, b = e["span"]
        if a == ka and b == kb:
            continue
        spans.append((a, b, "<ATTR>"))
    return _mask_spans(text, spans)


def mask_all_entities(text: str, ents: list[dict]) -> str:
    spans = []
    for e in ents:
        if "span" not in e:
            continue
        repl = _ENT_TOKEN.get(e["label"], f"<{e['label']}>")
        spans.append((e["span"][0], e["span"][1], repl))
    return _mask_spans(text, spans)


def aug_span_text(span_text: str, y: str) -> list[tuple[str, bool]]:
    out = [(span_text, False)]
    if y in {"other", "color", "type", "purpose"}:
        return out
    parts = span_text.split()
    if len(parts) < 2:
        return out
    last = parts[-1].lower()
    for alt in UNIT_AUG.get(last, []):
        out.append((" ".join(parts[:-1] + [alt]), True))
    return out


def build_rows_for_query(query: str, labeler: WeakLabeler) -> list[dict]:
    q_clean = basic_clean(query, lowercase=False)
    q_norm = _norm_key(q_clean)
    if len(q_norm) < 2:
        return []
    tags = labeler.label_query(q_norm)
    ents = bio_to_entities(tags, query=q_norm)
    struct = entities_to_structured(ents, labeler)
    attr_ents = [e for e in ents if e["label"] == "ATTR" and (e.get("text") or "").strip()]
    if not attr_ents:
        return []
    bio_str = " ".join(f"{t}/{g}" for t, g in tags)
    masked_all = mask_all_attr(q_norm, attr_ents)
    masked_ent = mask_all_entities(q_norm, ents)
    brand = struct.get("brand") or ""
    category = struct.get("category") or ""
    model = struct.get("model") or ""
    rows = []
    for e in attr_ents:
        st0 = (e.get("text") or "").strip()
        if len(st0) < MIN_SPAN_LEN:
            continue
        span = e.get("span") or [None, None]
        y = _guess_attr_type(st0)
        keep_masked = mask_keep_span(q_norm, attr_ents, e)
        for st, is_aug in aug_span_text(st0, y):
            rows.append(
                {
                    "query": query,
                    "query_norm": q_norm,
                    "span_text": st,
                    "span_start": span[0],
                    "span_end": span[1],
                    "y": y,
                    "brand": brand,
                    "category": category,
                    "model": model,
                    "n_attrs_in_query": len(attr_ents),
                    "bio_tags": bio_str,
                    "query_masked_all_attr": masked_all,
                    "query_keep_span_mask_others": keep_masked,
                    "query_masked_entities": masked_ent,
                    "is_aug": is_aug,
                }
            )
    return rows


def main() -> None:
    ensure_dirs()
    out = ATTR_TYPE_DIR  # metrics/meta рядом с рантаймом
    out.mkdir(parents=True, exist_ok=True)

    log("load WeakLabeler")
    labeler = WeakLabeler.from_files(
        brands_path(),
        categories_path(),
        models_path=model_phrases_path() if model_phrases_path().exists() else None,
    )
    log(
        f"dicts brands={len(labeler.brands)} cats={len(labeler.categories)} "
        f"models={len(labeler.models)} patterns={len(ATTR_PATTERNS)}"
    )

    # demos
    for q in [
        "ноутбук asus 16 гб 15.6 дюйм",
        "asus tuf gaming a15 16 гб",
        "asus tuf gaming a15 16gb",
    ]:
        demo = pd.DataFrame(build_rows_for_query(q, labeler))
        tags = labeler.label_query(_norm_key(basic_clean(q, lowercase=False)))
        log(f"DEMO {q!r}")
        log(f"  BIO={tags}")
        if demo.empty:
            log("  no ATTR rows")
        else:
            for _, r in demo.loc[~demo["is_aug"]].iterrows():
                log(
                    f"  span={r['span_text']!r} y={r['y']} brand={r['brand']!r} "
                    f"model={r['model']!r} keep={r['query_keep_span_mask_others']!r}"
                )

    log("load clicks")
    clicks = load_query_clicks(n=SAMPLE_N, seed=SEED, random=True, columns=["query_text"])
    queries = (
        clicks["query_text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.len().between(2, 120)]
        .drop_duplicates()
        .head(MAX_QUERIES)
        .tolist()
    )
    log(f"queries={len(queries)}")

    rows: list[dict] = []
    n_with_attr = 0
    n_multi = 0
    for q in queries:
        r = build_rows_for_query(q, labeler)
        if not r:
            continue
        n_with_attr += 1
        if r[0]["n_attrs_in_query"] >= 2:
            n_multi += 1
        rows.extend(r)

    silver = pd.DataFrame(rows)
    raw = silver.loc[~silver["is_aug"]].copy()
    vc = raw["y"].value_counts()
    rare = vc[vc < RARE_SUPPORT]

    overview = pd.DataFrame(
        [
            {"metric": "queries_sampled", "value": len(queries)},
            {"metric": "queries_with_ATTR", "value": n_with_attr},
            {"metric": "multi_ATTR_queries", "value": n_multi},
            {"metric": "silver_rows_raw", "value": int(len(raw))},
            {"metric": "silver_rows_aug", "value": int(silver["is_aug"].sum())},
            {"metric": "n_types", "value": int(raw["y"].nunique())},
            {"metric": "share_y_other", "value": round(float((raw["y"] == "other").mean()), 4)},
        ]
    )
    log(overview.to_string(index=False))
    log("top y:\n" + vc.head(12).to_string())

    meta = {
        "seed": SEED,
        "sample_n": SAMPLE_N,
        "max_queries": MAX_QUERIES,
        "n_queries_sampled": len(queries),
        "n_queries_with_attr": n_with_attr,
        "n_multi_attr_queries": n_multi,
        "n_rows_raw": int(len(raw)),
        "n_rows_aug": int(silver["is_aug"].sum()),
        "n_rows_total": len(silver),
        "classes": sorted(raw["y"].unique().tolist()),
        "class_counts_raw": vc.to_dict(),
        "rare_types_support_lt": RARE_SUPPORT,
        "rare_types": rare.index.tolist() if len(rare) else [],
        "teacher": "WeakLabeler + _guess_attr_type(ATTR_PATTERNS, COLORS)",
        "mask_columns": {
            "query_masked_all_attr": "all ATTR spans -> <ATTR>",
            "query_keep_span_mask_others": "current span kept; other ATTR -> <ATTR>",
            "query_masked_entities": "BRAND/CAT/MODEL/ATTR -> placeholders",
        },
        "unit_aug_keys": sorted(UNIT_AUG.keys()),
        "design_note": (
            "one row per ATTR span; char n-grams for typing should use span_text only; "
            "other ATTR excluded via mask columns, not concatenated"
        ),
        "smoke": True,
    }

    save_silver_parquet(silver, "attr_type", "attr_type_silver.parquet")
    save_silver_parquet(raw, "attr_type", "attr_type_silver_raw.parquet")
    # meta/overview — в runtime dir + silver (mirror via write both)
    (out / "attr_type_silver_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from src.data_utils import SILVER_ATTR_TYPE

    SILVER_ATTR_TYPE.mkdir(parents=True, exist_ok=True)
    (SILVER_ATTR_TYPE / "attr_type_silver_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    overview.to_csv(out / "attr_type_silver_overview.csv", index=False)
    overview.to_csv(SILVER_ATTR_TYPE / "attr_type_silver_overview.csv", index=False)
    save_stats({"attr_type_silver": meta}, METRICS_DIR / "attr_type_silver.json")

    # required columns check
    need = [
        "query",
        "query_norm",
        "span_text",
        "span_start",
        "span_end",
        "y",
        "brand",
        "category",
        "model",
        "n_attrs_in_query",
        "bio_tags",
        "query_masked_all_attr",
        "query_keep_span_mask_others",
        "query_masked_entities",
        "is_aug",
    ]
    missing = [c for c in need if c not in silver.columns]
    assert not missing, missing
    assert "memory_storage" in set(raw["y"]), "expected memory_storage in y"
    log(f"DONE rows_raw={len(raw)} rows_aug={silver['is_aug'].sum()} types={raw['y'].nunique()}")


if __name__ == "__main__":
    main()
