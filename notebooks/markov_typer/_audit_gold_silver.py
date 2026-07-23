"""Audit gold ATTR subtypes vs labeling.py teacher + silver (read-only gold)."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ner.labeling import (
    ATTR_PATTERNS,
    _guess_attr_type,
    gold_subtype_to_canon,
)

GOLD = ROOT / "data" / "gold" / "bio_liza.jsonl"
SILVER_RAW = ROOT / "artifacts" / "attr_type" / "attr_type_silver_raw.parquet"
TRAIN = ROOT / "artifacts" / "attr_type" / "attr_type_train_prod.parquet"


def spans_from_bio(query: str, tags: list[str]) -> list[tuple[str, int, int, str]]:
    toks = query.split()
    if len(toks) != len(tags):
        return []
    spans = []
    i = 0
    while i < len(tags):
        t = tags[i]
        if t.startswith("B-"):
            lab = t[2:]
            j = i + 1
            while j < len(tags) and tags[j] == f"I-{lab}":
                j += 1
            spans.append((lab, i, j, " ".join(toks[i:j])))
            i = j
        else:
            i += 1
    return spans


def main() -> None:
    rows = [
        json.loads(l)
        for l in GOLD.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    teacher_types = sorted({n for _, n in ATTR_PATTERNS})
    teacher_inv = set(teacher_types) | {"color", "other", "type", "purpose"}

    gold_subtypes: Counter[str] = Counter()
    gold_ex: dict[str, list[str]] = defaultdict(list)
    pairs = []

    for r in rows:
        q = r["query"]
        tags = r["tags"]
        toks = q.split()
        if len(toks) != len(tags):
            continue
        subtypes = {int(k): v for k, v in (r.get("subtypes") or {}).items()}
        for lab, i, j, text in spans_from_bio(q, tags):
            if lab != "ATTR":
                continue
            st = subtypes.get(i)
            if st is None:
                for k in range(i, j):
                    if k in subtypes:
                        st = subtypes[k]
                        break
            if st is None:
                st = "<MISSING>"
            gold_subtypes[st] += 1
            if len(gold_ex[st]) < 4:
                gold_ex[st].append(text)
            teacher = _guess_attr_type(text)
            canon = gold_subtype_to_canon(st) if st != "<MISSING>" else "<MISSING>"
            pairs.append(
                {
                    "query": q,
                    "span": text,
                    "gold": st,
                    "canon": canon,
                    "teacher": teacher,
                    "agree_raw": st == teacher,
                    "agree_mapped": canon == teacher,
                }
            )

    print("=== GOLD (read-only) ===")
    print(f"queries={len(rows)} ATTR={len(pairs)}")
    print("subtypes:", dict(gold_subtypes.most_common()))

    labeled = [x for x in pairs if x["gold"] != "<MISSING>"]
    raw_ok = sum(1 for x in labeled if x["agree_raw"])
    map_ok = sum(1 for x in labeled if x["agree_mapped"])
    print("\n=== AGREEMENT ===")
    print(f"exact gold==teacher: {raw_ok}/{len(labeled)} ({raw_ok / max(1, len(labeled)):.1%})")
    print(f"mapped canon==teacher: {map_ok}/{len(labeled)} ({map_ok / max(1, len(labeled)):.1%})")

    print("\n=== mapped confusion (canon -> teacher) ===")
    ct = Counter((x["canon"], x["teacher"]) for x in labeled)
    for (g, t), n in ct.most_common(25):
        mark = "OK" if g == t else "XX"
        print(f"  {mark} {g:22} -> {t:22} {n}")

    print("\n=== mapped disagree (sample) ===")
    for x in [x for x in labeled if not x["agree_mapped"]][:25]:
        print(
            f"  canon={x['canon']:12} teacher={x['teacher']:18} "
            f"gold={x['gold']:12} span={x['span']!r:28} | {x['query']}"
        )

    ooc = sorted({x["gold"] for x in labeled} - teacher_inv - set(gold_subtype_to_canon(g) for g in {x["gold"] for x in labeled}))
    # OOC = gold labels that map to something outside teacher after map? 
    # better: gold subtypes whose canon is not in teacher_inv
    ooc_after = sorted(
        {
            x["gold"]
            for x in labeled
            if gold_subtype_to_canon(x["gold"]) not in teacher_inv
        }
    )
    print("\n=== gold subtypes whose canon not in teacher inv ===")
    print(ooc_after or "(none)")

    print("\n=== SMOKE teacher ===")
    for s in [
        "16 г",
        "16 гб",
        "150 грамм",
        "4k",
        "беспроводные",
        "узкая",
        "смарт",
        "для спорта",
        "спорта",
        "смузи",
        "6 кг",
        "512gb",
    ]:
        print(f"  {_guess_attr_type(s):18} <- {s!r}")

    if SILVER_RAW.exists():
        raw = pd.read_parquet(SILVER_RAW)
        y_new = raw["span_text"].map(_guess_attr_type)
        print(f"\n=== SILVER_RAW re-teacher ===")
        print(y_new.value_counts().head(25).to_string())
        print("type/purpose counts:", int((y_new == "type").sum()), int((y_new == "purpose").sum()))

    if TRAIN.exists():
        tr = pd.read_parquet(TRAIN)
        print(f"\ntrain_prod classes: {sorted(tr['y'].unique())}")


if __name__ == "__main__":
    main()
