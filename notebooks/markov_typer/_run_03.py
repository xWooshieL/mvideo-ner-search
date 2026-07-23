"""CLI: same as 03 notebook / prod train. Delegates to _run_04_prod."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("_run_04_prod.py")), run_name="__main__")
