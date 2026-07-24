"""Honest evaluation of every MVP module against the gold set.

Writes presentation-ready tables to artifacts/metrics/:
  - gold_metrics_ner.csv        entity P/R/F1 per tag for RULES (labeling.py) and CRF
  - gold_metrics_attr_type.csv  attr-type agreement on gold spans (teacher + clf)
  - silver_row_counts.csv       rows per silver dataset (per task)
  - model_comparison.csv        candidate models per task (brand / attr / crf)
  - gold_eval_summary.json      machine-readable roll-up

Gold = data/gold/bio_liza.jsonl (canonical; mirrored under artifacts/gold/).
Single annotator; size grows over time. Metrics are indicative, not final.
Brand classifier has no brand-level gold yet -> silver-val only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import (  # noqa: E402
    METRICS_DIR,
    MODELS,
    SILVER_ATTR_TYPE,
    SILVER_BRAND_CLF,
    SILVER_NER_BIO,
    parquet_num_rows,
)
from src.ner.labeling import (  # noqa: E402
    WeakLabeler,
    _guess_attr_type,
    bio_to_entities,
    gold_subtype_to_canon,
    tokenize,
)
from src.data_utils import brands_path, categories_path, model_phrases_path  # noqa: E402
from src.ner.metrics import _get_spans  # noqa: E402
from src.ner.model_crf import CRFNerModel  # noqa: E402

# Canonical source of truth; artifacts/gold is a mirror for packaging.
GOLD_CANDIDATES = (
    ROOT / "data" / "gold" / "bio_liza.jsonl",
    ROOT / "artifacts" / "gold" / "bio_liza.jsonl",
)
GOLD = next((p for p in GOLD_CANDIDATES if p.exists()), GOLD_CANDIDATES[0])
NER_LABELS = ["BRAND", "CATEGORY", "MODEL", "ATTR"]
_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return _WS.sub(" ", str(s).strip().lower())


def load_gold(path: Path | None = None) -> tuple[list[dict], dict]:
    path = path or GOLD
    rows = []
    n_raw = 0
    n_skip = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n_raw += 1
        r = json.loads(line)
        q = r["query"]
        tags = r["tags"]
        toks = q.split()
        if len(toks) != len(tags):
            tk = [t for t, _, _ in tokenize(q)]
            if len(tk) == len(tags):
                toks = tk
            else:
                n_skip += 1
                continue
        rows.append({"query": q, "tokens": toks, "tags": tags, "subtypes": r.get("subtypes") or {}})
    meta = {"path": str(path.as_posix()), "n_raw": n_raw, "n_usable": len(rows), "n_skipped": n_skip}
    return rows, meta


def span_set(tokens: list[str], tags: list[str]) -> set[tuple[str, str]]:
    return {(lab, norm(" ".join(tokens[s:e]))) for lab, s, e in _get_spans(tags)}


def prf_table(gold_sets: list[set], pred_sets: list[set], module: str) -> tuple[pd.DataFrame, dict]:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for g, p in zip(gold_sets, pred_sets):
        for lab, _ in g & p:
            tp[lab] += 1
        for lab, _ in p - g:
            fp[lab] += 1
        for lab, _ in g - p:
            fn[lab] += 1
    rows = []
    stp = sfp = sfn = 0
    for lab in NER_LABELS:
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append(
            {"module": module, "label": lab, "precision": round(p, 4), "recall": round(r, 4),
             "f1": round(f1, 4), "support": tp[lab] + fn[lab]}
        )
        stp += tp[lab]; sfp += fp[lab]; sfn += fn[lab]
    mp = stp / (stp + sfp) if (stp + sfp) else 0.0
    mr = stp / (stp + sfn) if (stp + sfn) else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0
    rows.append({"module": module, "label": "MICRO", "precision": round(mp, 4),
                 "recall": round(mr, 4), "f1": round(mf, 4), "support": stp + sfn})
    return pd.DataFrame(rows), {"micro_precision": mp, "micro_recall": mr, "micro_f1": mf}


def eval_ner(gold: list[dict]) -> tuple[pd.DataFrame, dict]:
    labeler = WeakLabeler.from_files(
        brands_path(), categories_path(),
        models_path=model_phrases_path() if model_phrases_path().exists() else None,
    )
    crf = None
    crf_path = MODELS / "ner_crf.pkl"
    if crf_path.exists():
        crf = CRFNerModel.load(crf_path)

    gold_sets, rule_sets, crf_sets = [], [], []
    for g in gold:
        gold_sets.append(span_set(g["tokens"], g["tags"]))
        rp = labeler.label_query(g["query"])
        rule_sets.append(span_set([t for t, _ in rp], [t for _, t in rp]))
        if crf is not None:
            cp = crf.predict_query(g["query"])
            crf_sets.append(span_set([t for t, _ in cp], [t for _, t in cp]))

    t_rules, m_rules = prf_table(gold_sets, rule_sets, "rules_regex")
    tables = [t_rules]
    summary = {"rules_regex": m_rules}
    if crf is not None:
        t_crf, m_crf = prf_table(gold_sets, crf_sets, "crf")
        tables.append(t_crf)
        summary["crf"] = m_crf
    return pd.concat(tables, ignore_index=True), summary


def eval_attr_type(gold: list[dict]) -> tuple[pd.DataFrame, dict]:
    from src.ner.attr_type_clf import predict_attr_type

    model_phrases = set()
    mp = model_phrases_path()
    if mp.exists():
        model_phrases = {norm(x) for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()}

    n = teacher_ok = clf_ok = 0
    per_canon = defaultdict(lambda: {"n": 0, "teacher_ok": 0, "clf_ok": 0})
    for g in gold:
        toks, tags = g["tokens"], g["tags"]
        subtypes = {int(k): v for k, v in g["subtypes"].items()}
        ents = {lab: norm(" ".join(toks[s:e])) for lab, s, e in _get_spans(tags) if lab in {"BRAND", "CATEGORY"}}
        for lab, s, e in _get_spans(tags):
            if lab != "ATTR":
                continue
            st = subtypes.get(s)
            if st is None:
                for k in range(s, e):
                    if k in subtypes:
                        st = subtypes[k]
                        break
            if not st:
                continue
            gold_canon = gold_subtype_to_canon(st)
            span = " ".join(toks[s:e])
            teacher = _guess_attr_type(span)
            try:
                clf = predict_attr_type(
                    span, brand=ents.get("BRAND", ""), category=ents.get("CATEGORY", ""),
                    query_masked=f"{ents.get('CATEGORY','')} {ents.get('BRAND','')} <ATTR>".strip(),
                    model_phrases=model_phrases,
                )
            except Exception:
                clf = "ERROR"
            n += 1
            teacher_ok += int(teacher == gold_canon)
            clf_ok += int(clf == gold_canon)
            pc = per_canon[gold_canon]
            pc["n"] += 1
            pc["teacher_ok"] += int(teacher == gold_canon)
            pc["clf_ok"] += int(clf == gold_canon)

    rows = [{"canon": "ALL", "n": n,
             "teacher_agree": round(teacher_ok / max(1, n), 4),
             "clf_agree": round(clf_ok / max(1, n), 4)}]
    for canon, d in sorted(per_canon.items(), key=lambda kv: -kv[1]["n"]):
        rows.append({"canon": canon, "n": d["n"],
                     "teacher_agree": round(d["teacher_ok"] / max(1, d["n"]), 4),
                     "clf_agree": round(d["clf_ok"] / max(1, d["n"]), 4)})
    summary = {"n_gold_attr_spans": n,
               "teacher_agree": round(teacher_ok / max(1, n), 4),
               "clf_agree": round(clf_ok / max(1, n), 4)}
    return pd.DataFrame(rows), summary


def silver_row_counts() -> pd.DataFrame:
    specs = [
        ("ner_bio", "silver_bio_slice.parquet", SILVER_NER_BIO),
        ("attr_type", "attr_type_silver_raw.parquet", SILVER_ATTR_TYPE),
        ("attr_type", "attr_type_train_prod.parquet", SILVER_ATTR_TYPE),
        ("attr_type", "attr_type_val_prod.parquet", SILVER_ATTR_TYPE),
        ("brand_clf", "silver_brand_train.parquet", SILVER_BRAND_CLF),
        ("brand_clf", "silver_brand_val.parquet", SILVER_BRAND_CLF),
        ("brand_clf", "silver_brand_all.parquet", SILVER_BRAND_CLF),
    ]
    rows = []
    for task, name, d in specs:
        p = d / name
        rows.append({"task": task, "dataset": name, "rows": parquet_num_rows(p) if p.exists() else None})
    return pd.DataFrame(rows)


def model_comparison() -> pd.DataFrame:
    rows = []
    bs = SILVER_BRAND_CLF / "train_runs" / "models_summary.csv"
    if bs.exists():
        for _, r in pd.read_csv(bs).iterrows():
            rows.append({"task": "brand_clf", "model": r["model"], "f1_macro": round(float(r["raw_f1_macro"]), 4),
                         "note": "silver-val (no brand gold)"})
    at = SILVER_ATTR_TYPE / "prod_models_summary.csv"
    if at.exists():
        for _, r in pd.read_csv(at).iterrows():
            rows.append({"task": "attr_type", "model": r["model"], "f1_macro": round(float(r["f1_macro"]), 4),
                         "note": "silver-val (relabelled teacher)"})
    crf_m = SILVER_NER_BIO / "crf_train_metrics.json"
    if crf_m.exists():
        m = json.loads(crf_m.read_text(encoding="utf-8"))
        sv = m.get("silver_val", {}).get("micro", {}).get("f1")
        gm = (m.get("gold") or {}).get("micro", {}).get("f1")
        rows.append({"task": "ner_crf", "model": "linear_chain_crf",
                     "f1_macro": round(sv, 4) if sv else None,
                     "note": f"silver-val microF1; gold microF1={round(gm,4) if gm else 'NA'}"})
    return pd.DataFrame(rows)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    assert GOLD.exists(), GOLD
    gold, gold_meta = load_gold()
    print(f"gold path: {gold_meta['path']}")
    print(
        f"gold queries usable: {gold_meta['n_usable']} / raw {gold_meta['n_raw']} "
        f"(skipped {gold_meta['n_skipped']})"
    )

    summary = {"gold_queries": len(gold), "gold_meta": gold_meta}

    ner_tbl, ner_sum = eval_ner(gold)
    ner_tbl.to_csv(METRICS_DIR / "gold_metrics_ner.csv", index=False)
    summary["ner"] = ner_sum
    print(ner_tbl.to_string(index=False))

    attr_tbl, attr_sum = eval_attr_type(gold)
    attr_tbl.to_csv(METRICS_DIR / "gold_metrics_attr_type.csv", index=False)
    summary["attr_type"] = attr_sum
    print(attr_tbl.to_string(index=False))

    rc = silver_row_counts()
    rc.to_csv(METRICS_DIR / "silver_row_counts.csv", index=False)
    print(rc.to_string(index=False))

    mc = model_comparison()
    crf_f1 = (ner_sum.get("crf") or {}).get("micro_f1")
    if crf_f1 is not None and not mc.empty:
        mask = mc["task"] == "ner_crf"
        if mask.any():
            mc.loc[mask, "note"] = (
                f"silver-val microF1; gold microF1={round(float(crf_f1), 4)} (n={len(gold)})"
            )
    mc.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(mc.to_string(index=False))

    (METRICS_DIR / "gold_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE ->", METRICS_DIR)


if __name__ == "__main__":
    main()
