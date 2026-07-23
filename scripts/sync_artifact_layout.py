"""Синхронизация artifacts → канон dicts/ + silver/* без удаления legacy.

Usage:
  python scripts/sync_artifact_layout.py
  python scripts/sync_artifact_layout.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import sync_artifact_layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    report = sync_artifact_layout(dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    n = sum(len(v) for v in report.values())
    print(f"{'would copy' if args.dry_run else 'copied/updated'}: {n} files")
    print("Legacy paths kept. Readers: resolve_dict / resolve_silver.")


if __name__ == "__main__":
    main()
