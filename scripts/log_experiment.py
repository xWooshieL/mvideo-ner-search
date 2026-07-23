#!/usr/bin/env python
"""Append an experiment snapshot to artifacts/history/.

Examples:
  python scripts/log_experiment.py --module ner_crf --tag baseline-pre-spellfix \\
      --note "Before SpellFixer rebuild" --from-metrics

  python scripts/log_experiment.py --module ner_crf --tag spellfix-v1 \\
      --note "SpellFixer in silver+_run_01 + extractor" --from-metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORY = ROOT / "artifacts" / "history"
METRICS = ROOT / "artifacts" / "metrics"
SILVER_NER = ROOT / "artifacts" / "silver" / "ner_bio"
SILVER_BRAND = ROOT / "artifacts" / "silver" / "brand_clf"
SILVER_ATTR = ROOT / "artifacts" / "silver" / "attr_type"

MODULE_MD = {
    "ner_crf": "ner_crf.md",
    "brand_clf": "brand_clf.md",
    "attr_type_clf": "attr_type_clf.md",
    "cascade": "broken_queries.md",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_from_metrics(module: str) -> dict[str, Any]:
    out: dict[str, Any] = {"module": module}
    if module == "ner_crf":
        crf = _load_json(SILVER_NER / "crf_train_metrics.json") or {}
        gold = _load_json(METRICS / "gold_eval_summary.json") or {}
        sv = (crf.get("silver_val") or {}).get("micro") or {}
        gm = ((gold.get("ner") or {}).get("crf") or {})
        rules = ((gold.get("ner") or {}).get("rules_regex") or {})
        out["metrics"] = {
            "silver_val_micro_f1": sv.get("f1"),
            "silver_val_tok_acc": (crf.get("silver_val") or {}).get("token_accuracy"),
            "gold_micro_f1": gm.get("micro_f1"),
            "gold_micro_p": gm.get("micro_precision"),
            "gold_micro_r": gm.get("micro_recall"),
            "rules_gold_micro_f1": rules.get("micro_f1"),
            "n_train": crf.get("n_train"),
            "n_val": crf.get("n_val"),
            "per_label_gold": None,
        }
        # per-label from gold_metrics_ner.csv if present
        csv = METRICS / "gold_metrics_ner.csv"
        if csv.exists():
            import pandas as pd

            df = pd.read_csv(csv)
            crf_rows = df[df["module"] == "crf"]
            out["metrics"]["per_label_gold"] = {
                str(r["label"]): float(r["f1"])
                for _, r in crf_rows.iterrows()
                if r["label"] != "MICRO"
            }
        meta = _load_json(SILVER_NER / "eda_meta.json") or {}
        out["metrics"]["n_spell_fixed"] = meta.get("n_spell_fixed")
        out["metrics"]["n_silver"] = meta.get("n_silver")
    elif module == "brand_clf":
        m = _load_json(SILVER_BRAND / "train_metrics.json") or {}
        out["metrics"] = {
            "best_model": m.get("best_model"),
            "f1_macro": (m.get("best_raw") or {}).get("f1_macro"),
            "accuracy": (m.get("best_raw") or {}).get("accuracy"),
            "f1_NO_BRAND": (m.get("best_subset_raw") or {}).get("f1_NO_BRAND"),
            "f1_UNKNOWN": (m.get("best_subset_raw") or {}).get("f1_UNKNOWN"),
            "false_brand_cat": (m.get("best_subset_raw") or {}).get(
                "false_brand_rate_category_only"
            ),
            "n_train": m.get("n_train"),
            "n_val": m.get("n_val"),
            "n_classes": m.get("n_classes"),
        }
    elif module == "attr_type_clf":
        m = _load_json(SILVER_ATTR / "prod_metrics.json") or {}
        gold = _load_json(METRICS / "gold_eval_summary.json") or {}
        at = gold.get("attr_type") or {}
        best = None
        for row in m.get("summary") or []:
            if row.get("model") == m.get("best_model"):
                best = row
                break
        out["metrics"] = {
            "best_model": m.get("best_model"),
            "tau": m.get("tau"),
            "f1_macro": (best or {}).get("f1_macro"),
            "accuracy": (best or {}).get("accuracy"),
            "sanity_pass": m.get("sanity_pass"),
            "sanity_total": m.get("sanity_total"),
            "gold_teacher_agree": at.get("teacher_agree"),
            "gold_clf_agree": at.get("clf_agree"),
        }
    else:
        out["metrics"] = {}
    return out


def append_jsonl(entry: dict[str, Any]) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    path = HISTORY / "experiments.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_md(module: str, entry: dict[str, Any]) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    name = MODULE_MD.get(module, f"{module}.md")
    path = HISTORY / name
    if not path.exists():
        path.write_text(f"# {module} — история экспериментов\n\n", encoding="utf-8")

    m = entry.get("metrics") or {}
    lines = [
        f"## {entry['ts']} — `{entry['tag']}`",
        "",
        f"**Note:** {entry.get('note') or '—'}",
        "",
    ]
    if module == "ner_crf":
        lines += [
            "| метрика | значение |",
            "|---|---:|",
            f"| silver-val micro-F1 | {_fmt(m.get('silver_val_micro_f1'))} |",
            f"| silver-val tokAcc | {_fmt(m.get('silver_val_tok_acc'))} |",
            f"| **gold micro-F1** | {_fmt(m.get('gold_micro_f1'))} |",
            f"| gold P / R | {_fmt(m.get('gold_micro_p'))} / {_fmt(m.get('gold_micro_r'))} |",
            f"| rules gold micro-F1 | {_fmt(m.get('rules_gold_micro_f1'))} |",
            f"| n_train / n_val | {m.get('n_train')} / {m.get('n_val')} |",
            f"| spellfix touched | {m.get('n_spell_fixed')} / {m.get('n_silver')} |",
            "",
        ]
        pl = m.get("per_label_gold") or {}
        if pl:
            lines += [
                "Gold entity F1 per tag:",
                "",
                "| tag | F1 |",
                "|---|---:|",
            ]
            for k, v in pl.items():
                lines.append(f"| {k} | {_fmt(v)} |")
            lines.append("")
    elif module == "brand_clf":
        lines += [
            "| метрика | значение |",
            "|---|---:|",
            f"| best | `{m.get('best_model')}` |",
            f"| f1_macro | {_fmt(m.get('f1_macro'))} |",
            f"| accuracy | {_fmt(m.get('accuracy'))} |",
            f"| F1 NO_BRAND / UNKNOWN | {_fmt(m.get('f1_NO_BRAND'))} / {_fmt(m.get('f1_UNKNOWN'))} |",
            f"| false_brand@cat | {_fmt(m.get('false_brand_cat'))} |",
            f"| train/val/classes | {m.get('n_train')} / {m.get('n_val')} / {m.get('n_classes')} |",
            "",
        ]
    elif module == "attr_type_clf":
        lines += [
            "| метрика | значение |",
            "|---|---:|",
            f"| best | `{m.get('best_model')}` τ={m.get('tau')} |",
            f"| f1_macro (silver-val) | {_fmt(m.get('f1_macro'))} |",
            f"| sanity | {m.get('sanity_pass')}/{m.get('sanity_total')} |",
            f"| gold teacher / clf agree | {_fmt(m.get('gold_teacher_agree'))} / {_fmt(m.get('gold_clf_agree'))} |",
            "",
        ]
    else:
        lines += ["```json", json.dumps(m, ensure_ascii=False, indent=2), "```", ""]

    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, choices=list(MODULE_MD) + ["ner_crf", "brand_clf", "attr_type_clf"])
    ap.add_argument("--tag", required=True, help="short id, e.g. spellfix-v1")
    ap.add_argument("--note", default="", help="why this run")
    ap.add_argument("--from-metrics", action="store_true", help="read artifacts/metrics + silver")
    ap.add_argument("--metrics-json", type=Path, default=None, help="optional override JSON")
    args = ap.parse_args()

    if args.metrics_json:
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
        entry = {"module": args.module, "metrics": metrics}
    elif args.from_metrics:
        entry = collect_from_metrics(args.module)
    else:
        print("Need --from-metrics or --metrics-json", file=sys.stderr)
        return 2

    entry.update(
        {
            "ts": _now(),
            "tag": args.tag,
            "note": args.note,
        }
    )
    append_jsonl(entry)
    append_md(args.module, entry)
    print(json.dumps(entry, ensure_ascii=False, indent=2))
    print(f"logged → {HISTORY / MODULE_MD[args.module]} + experiments.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
