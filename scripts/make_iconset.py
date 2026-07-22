# -*- coding: utf-8 -*-
"""Готовит .iconset (набор PNG нужных размеров) для сборки .icns на macOS.

Запускать где угодно (Windows/macOS) — просто раскладывает PNG по размерам.
Сам .icns собирается на маке командой `iconutil -c icns` (см. build-macos.sh).
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SIZES = [16, 32, 64, 128, 256, 512, 1024]


def make_iconset(src_png: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Image.open(src_png).convert("RGBA")
    for size in SIZES:
        img = base.resize((size, size), Image.LANCZOS)
        img.save(out_dir / f"icon_{size}x{size}.png")
        if size <= 512:
            img2x = base.resize((size * 2, size * 2), Image.LANCZOS)
            img2x.save(out_dir / f"icon_{size}x{size}@2x.png")
    print(f"iconset готов: {out_dir}")


if __name__ == "__main__":
    make_iconset(ROOT / "cpp" / "mvsearch" / "assets" / "icon.png",
                 ROOT / "cpp" / "mvsearch" / "assets" / "icon.iconset")
    make_iconset(ROOT / "cpp" / "mvlabel" / "assets" / "icon.png",
                 ROOT / "cpp" / "mvlabel" / "assets" / "icon.iconset")
