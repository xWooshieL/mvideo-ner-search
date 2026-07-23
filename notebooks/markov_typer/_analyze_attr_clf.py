"""ATTR-type clf quality / error analysis."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.ner.attr_type_clf import LABEL_UNKNOWN, looks_like_model
from src.ner.labeling import ATTR_PATTERNS, _guess_attr_type

OUT = Path("artifacts/attr_type")
pipe = joblib.load(Path("models") / "attr_type_clf.joblib")
train = pd.read_parquet(OUT / "attr_type_train.parquet")
val = pd.read_parquet(OUT / "attr_type_val.parquet")
silver = pd.read_parquet(OUT / "attr_type_silver_raw.parquet")

print("=== SIZE ===")
print(f"silver_raw={len(silver)} train={len(train)} val={len(val)} queries={silver.query_norm.nunique()}")
print("train class counts:")
print(train.y.value_counts().to_string())

val = val.copy()
if "context_text" not in val.columns:
    val["context_text"] = (
        val.brand.fillna("").astype(str) + " " + val.category.fillna("").astype(str)
    ).str.strip()
if "query_masked" not in val.columns:
    val["query_masked"] = val["query_masked_all_attr"].fillna("").astype(str)

pred = pipe.predict(val)
val["pred"] = pred
err = val[val.y.astype(str) != val.pred.astype(str)]
print(f"\n=== VAL ERRORS {len(err)}/{len(val)} ({len(err)/len(val):.1%}) ===")
print("error pairs true -> pred counts:")
print(err.groupby(["y", "pred"]).size().sort_values(ascending=False).to_string() if len(err) else "none")
print("\nall error rows:")
cols = [c for c in ["span_text", "y", "pred", "query_norm", "brand", "category"] if c in err.columns]
print(err[cols].to_string())

proba = pipe.predict_proba(val)
val["p_max"] = proba.max(axis=1)
print("\n=== confidence ===")
print("mean p_max correct", float(val.loc[val.y == val.pred, "p_max"].mean()))
print("mean p_max wrong", float(val.loc[val.y != val.pred, "p_max"].mean()) if len(err) else None)
print("wrong with p_max>0.8", int(((val.y != val.pred) & (val.p_max > 0.8)).sum()))
print("correct with p_max<0.5", int(((val.y == val.pred) & (val.p_max < 0.5)).sum()))

silver2 = silver.copy()
silver2["teacher"] = silver2.span_text.map(_guess_attr_type)
print("\n=== silver y vs teacher ===")
print("agree", float((silver2.y == silver2.teacher).mean()))
mismatch = silver2[silver2.y != silver2.teacher]
print("mismatches", len(mismatch))
if len(mismatch):
    print(mismatch[["span_text", "y", "teacher"]].head(20).to_string())

print("\n=== teacher first-match for ambiguous ===")
for s in ["1920x1080", "100x50", "4k", "16 грамм", "16 г", "16 g", "full hd"]:
    print(f"teacher {s!r} -> {_guess_attr_type(s)}")
    for pat, name in ATTR_PATTERNS:
        if pat.search(s.lower().replace("ё", "е")):
            print(f"  first ATTR_PATTERNS hit: {name}")
            break

print("\n=== weight units in silver ===")
w = silver[silver.y == "weight"]
print(w.span_text.value_counts().head(15).to_string())
print("грамм count", int(w.span_text.str.contains("грамм", case=False).sum()))

print("\n=== resolution_standard in silver ===")
rs = silver[silver.y == "resolution_standard"]
print("n", len(rs))
print(rs.span_text.value_counts().head(12).to_string() if len(rs) else "none")

print("\n=== dimensions in silver (top) ===")
print(silver.loc[silver.y == "dimensions", "span_text"].value_counts().head(12).to_string())

print("\n=== MANUAL PROBES ===")
manual = [
    ("16 грамм", "weight"),
    ("16 г", "weight"),
    ("16 g", "weight"),
    ("4k", "resolution_standard"),
    ("1920x1080", "resolution? but teacher=dimensions"),
    ("full hd", "resolution_standard"),
    ("wi-fi", "connectivity"),
    ('55"', "size"),
    ("16gb", "memory_storage"),
    ("от 16 гб", "memory_storage"),
    ("2литра", "volume"),
    ("500вт", "power"),
    ("белая", "color"),
    ("32 ом", "UNKNOWN(rare)"),
]
for s, expect in manual:
    row = pd.DataFrame(
        [
            {
                "span_text": s,
                "context_text": "ноутбук asus",
                "query_masked": "ноутбук asus <ATTR>",
                "brand": "asus",
                "category": "ноутбук",
            }
        ]
    )
    if looks_like_model(s):
        print(f"{s!r:18} RULE UNKNOWN | expect {expect} | teacher={_guess_attr_type(s)}")
        continue
    p = pipe.predict(row)[0]
    pr = pipe.predict_proba(row)[0]
    top = sorted(zip(pipe.classes_, pr), key=lambda x: -x[1])[:3]
    tops = " | ".join(f"{a}:{b:.2f}" for a, b in top)
    t = _guess_attr_type(s)
    print(
        f"{s!r:18} pred={p:18} p={pr.max():.2f} teacher={t:18} expect≈{expect}\n"
        f"    {tops}"
    )

print("\n=== lexical diversity (train) ===")
for c in [
    "memory_storage",
    "size",
    "weight",
    "color",
    "dimensions",
    "resolution_standard",
    "UNKNOWN",
]:
    sub = train[train.y == c]
    units = sub.span_text.map(lambda x: x.split()[-1].lower() if str(x).split() else "")
    print(
        f"{c:22} n={len(sub):4d} unique_spans={sub.span_text.nunique():4d} "
        f"unique_last_tok={units.nunique():3d}"
    )

# OOV units: last token not seen in train for that class
print("\n=== OOV-ish: last token never in train ===")
train_units = set(
    train.span_text.map(lambda x: x.split()[-1].lower() if str(x).split() else "")
)
for s, _ in manual:
    last = s.split()[-1].lower() if s.split() else s.lower()
    print(f"  last={last!r:12} in_train_units={last in train_units}")
