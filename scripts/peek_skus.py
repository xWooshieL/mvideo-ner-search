#!/usr/bin/env python
"""Lightweight peek into skus.pkl YML catalog (may take minutes to unpickle)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "skus_catalog_summary.json"


def main():
    path = ROOT / "файлы" / "skus.pkl"
    print(f"Loading {path} ({path.stat().st_size / 1e9:.2f} GB)...")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    shop = obj["yml_catalog"]["shop"]
    summary = {
        "yml_date": obj["yml_catalog"].get("@date"),
        "shop_keys": list(shop.keys()),
    }
    for k, v in shop.items():
        if isinstance(v, list):
            summary[f"{k}_len"] = len(v)
            if v and isinstance(v[0], dict):
                summary[f"{k}_item0_keys"] = list(v[0].keys())[:30]
        elif isinstance(v, dict):
            summary[f"{k}_keys"] = list(v.keys())[:40]
            if "offer" in v and isinstance(v["offer"], list):
                summary["n_offers"] = len(v["offer"])
                if v["offer"]:
                    summary["offer0_keys"] = list(v["offer"][0].keys()) if isinstance(v["offer"][0], dict) else str(type(v["offer"][0]))
            if "category" in v:
                cats = v["category"]
                summary["n_categories"] = len(cats) if hasattr(cats, "__len__") else None
        else:
            summary[k] = str(v)[:200]
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
