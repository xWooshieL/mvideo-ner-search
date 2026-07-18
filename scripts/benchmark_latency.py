#!/usr/bin/env python
"""Benchmark QueryEntityExtractor latency (p50/p95/p99) and dump JSON examples."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.service.extractor import QueryEntityExtractor  # noqa: E402

COL_QUERY = "toValidUTF8(query_text)"


def load_queries(path: Path, n: int) -> list[str]:
    pf = pq.ParquetFile(path)
    qs: list[str] = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=[COL_QUERY])
        for q in t.column(0).to_pylist():
            if q and str(q).strip():
                qs.append(str(q).strip())
                if len(qs) >= n * 3:
                    break
        if len(qs) >= n * 3:
            break
    # unique preserve order
    seen = set()
    out = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "файлы" / "query_clicks.parquet")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "benchmark.json")
    args = ap.parse_args()

    args.figures.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.data, args.n)
    print(f"Loaded {len(queries)} unique queries")

    ext = QueryEntityExtractor.from_artifacts(args.artifacts, args.models)
    for q in queries[:30]:
        ext.extract(q)

    latencies = []
    examples = []
    t_all = time.perf_counter()
    for q in queries:
        r = ext.extract(q)
        latencies.append(r["latency_ms"])
        if len(examples) < 15 and r.get("entities"):
            examples.append(r)
    wall = time.perf_counter() - t_all

    arr = np.asarray(latencies, dtype=float)
    stats = {
        "n": int(len(arr)),
        "wall_sec": float(wall),
        "qps": float(len(arr) / wall) if wall else 0.0,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "under_100ms_share": float((arr < 100).mean()),
        "examples": examples,
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(arr, bins=50, color="#E31E24", edgecolor="white")
    ax.axvline(100, color="black", ls="--", label="100 ms")
    ax.axvline(stats["p95_ms"], color="#555", ls=":", label=f"p95={stats['p95_ms']:.1f}")
    ax.set_title("Latency histogram")
    ax.set_xlabel("ms")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures / "17_latency_histogram.png", dpi=140)
    plt.close(fig)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # merge into metrics.json if present
    metrics_path = args.artifacts / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        metrics["latency"] = {k: stats[k] for k in stats if k != "examples"}
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: stats[k] for k in stats if k != "examples"}, ensure_ascii=False, indent=2))
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
