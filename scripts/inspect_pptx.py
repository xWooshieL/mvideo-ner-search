#!/usr/bin/env python
"""Inspect pptx templates: theme fonts, colors, media; extract logos."""
import re
import zipfile
from pathlib import Path

DOWNLOADS = Path(r"C:\Users\kamau\Downloads")
OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

for name in ["Курс ЦУ (2).pptx", "Курс ЦУ (1).pptx", "М.ТЕХ (корр) (2) (1).pptx"]:
    p = DOWNLOADS / name
    if not p.exists():
        print("MISSING:", name)
        continue
    print("=" * 30, name)
    z = zipfile.ZipFile(p)
    themes = [n for n in z.namelist() if "theme" in n and n.endswith(".xml")]
    for t in themes[:2]:
        xml = z.read(t).decode("utf-8", errors="ignore")
        fonts = re.findall(r'<a:latin typeface="([^"]+)"', xml)
        colors = re.findall(r'<a:srgbClr val="([0-9A-Fa-f]{6})"', xml)
        print(t)
        print("  fonts:", sorted(set(fonts)))
        print("  colors:", colors[:16])
    # fonts used in slides
    slide_fonts = set()
    for n in z.namelist():
        if n.startswith("ppt/slides/slide") and n.endswith(".xml"):
            xml = z.read(n).decode("utf-8", errors="ignore")
            slide_fonts.update(re.findall(r'typeface="([^"]+)"', xml))
    print("  slide fonts:", sorted(slide_fonts))
    media = [n for n in z.namelist() if n.startswith("ppt/media/")]
    print("  media:", len(media))
    for m in media:
        info = z.getinfo(m)
        print(f"    {m}  {info.file_size}")
    # extract images
    prefix = re.sub(r"[^\w]+", "_", name)[:12]
    for m in media:
        data = z.read(m)
        ext = Path(m).suffix
        target = OUT / f"{prefix}_{Path(m).name}"
        target.write_bytes(data)
    print("  extracted to", OUT)
