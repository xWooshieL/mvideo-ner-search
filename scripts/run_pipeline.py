#!/usr/bin/env python
"""Orchestrator: EDA figures → dictionaries → train → embeddings → latency."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run("run_eda.py")
    run("build_dictionaries.py", "--max-rows", "500000")
    run("train_all.py", "--max-rows", "200000")
    run("build_embeddings_figures.py")
    run("benchmark_latency.py")
    # copy figures to docs
    src = ROOT / "figures"
    dst = ROOT / "docs" / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.png"):
        target = dst / p.name
        target.write_bytes(p.read_bytes())
    print("\nPipeline complete. See artifacts/metrics.json and figures/")


if __name__ == "__main__":
    main()
