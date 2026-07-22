# -*- coding: utf-8 -*-
"""Чистит и пересобирает логотип/иконки приложений и установщиков.

Баг: logo_mark_white.png оказался сплошным белым прямоугольником (альфа = 255
везде, без формы буквы «М»), из-за чего:
  - на splash-экранах обоих приложений вместо белой «М» рисовался белый блок;
  - в сайдбаре инсталлятора (wizard-sidebar.bmp) получался белый прямоугольник
    на красном фоне;
  - icon.png/icon.ico содержали "призрачный" полупрозрачный прямоугольник
    вокруг буквы (из старой неудачной векторизации), из-за чего иконка
    выглядела грязной на цветных фонах.

Всё это чинится генерацией заново из единственного чистого источника —
logo_mark.png (красная «М» на честно прозрачном фоне).
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MVIDEO_RED = (242, 6, 1)  # #f20601, Theme.accent

APPS = [
    ROOT / "cpp" / "mvsearch" / "assets",
    ROOT / "cpp" / "mvlabel" / "assets",
]

ICON_SIZES = [16, 24, 32, 48, 64, 128, 256]
ICONSET_SIZES = [16, 32, 64, 128, 256, 512, 1024]


def load_clean_mark(assets_dir: Path) -> Image.Image:
    """logo_mark.png — единственный проверенно-чистый источник (только буква, альфа честная)."""
    src = Image.open(assets_dir / "logo_mark.png").convert("RGBA")
    bbox = src.getbbox()
    if bbox:
        src = src.crop(bbox)
    return src


def make_white_mark(mark: Image.Image) -> Image.Image:
    """Белый силуэт буквы «М» на прозрачном фоне (для тёмных/красных подложек)."""
    alpha = mark.split()[3]
    white = Image.new("RGBA", mark.size, (255, 255, 255, 0))
    white.putalpha(alpha)
    return white


def make_icon_square(mark: Image.Image, size: int = 512, pad_ratio: float = 0.16) -> Image.Image:
    """Квадратная иконка: буква «М» по центру на ПОЛНОСТЬЮ прозрачном фоне, без подложек."""
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pad = int(size * pad_ratio)
    avail = size - 2 * pad
    w, h = mark.size
    scale = min(avail / w, avail / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = mark.resize((new_w, new_h), Image.LANCZOS)
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def save_ico(icon_png: Image.Image, dest: Path) -> None:
    icon_png.save(dest, format="ICO", sizes=[(s, s) for s in ICON_SIZES])


def make_iconset(icon_png: Image.Image, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for size in ICONSET_SIZES:
        icon_png.resize((size, size), Image.LANCZOS).save(out_dir / f"icon_{size}x{size}.png")
        if size <= 512:
            icon_png.resize((size * 2, size * 2), Image.LANCZOS).save(out_dir / f"icon_{size}x{size}@2x.png")


def make_wizard_header(mark: Image.Image, dest: Path, size=(55, 55)) -> None:
    """Маленький логотип в заголовке инсталлятора: красная «М» на белом."""
    canvas = Image.new("RGB", size, (255, 255, 255))
    pad = int(size[1] * 0.14)
    avail = size[1] - 2 * pad
    w, h = mark.size
    scale = avail / h
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = mark.resize((new_w, new_h), Image.LANCZOS)
    x = (size[0] - new_w) // 2
    y = (size[1] - new_h) // 2
    canvas.paste(resized, (x, y), resized)
    canvas.save(dest, format="BMP")


def make_wizard_sidebar(white_mark: Image.Image, dest: Path, size=(164, 314)) -> None:
    """Боковая панель инсталлятора: белая «М» по центру на брендовом красном."""
    canvas = Image.new("RGB", size, MVIDEO_RED)
    pad_w = int(size[0] * 0.30)
    avail_w = size[0] - 2 * pad_w
    w, h = white_mark.size
    scale = avail_w / w
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = white_mark.resize((new_w, new_h), Image.LANCZOS)
    x = (size[0] - new_w) // 2
    y = (size[1] - new_h) // 2 - int(size[1] * 0.06)
    canvas.paste(resized, (x, y), resized)

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    text = "M.Video"
    tw = draw.textlength(text, font=font)
    draw.text(((size[0] - tw) / 2, y + new_h + 18), text, fill=(255, 255, 255), font=font)

    canvas.save(dest, format="BMP")


def main() -> None:
    for assets_dir in APPS:
        print(f"== {assets_dir}")
        mark = load_clean_mark(assets_dir)
        white_mark = make_white_mark(mark)

        white_mark.save(assets_dir / "logo_mark_white.png")
        print("  logo_mark_white.png восстановлен (силуэт, не сплошной блок)")

        icon = make_icon_square(mark, size=512, pad_ratio=0.16)
        icon.resize((256, 256), Image.LANCZOS).save(assets_dir / "icon.png")
        save_ico(icon, assets_dir / "icon.ico")
        print("  icon.png / icon.ico пересобраны (чистая прозрачность, без 'призрака')")

        make_iconset(icon, assets_dir / "icon.iconset")
        icns = assets_dir / "icon.icns"
        if icns.exists():
            icns.unlink()  # пересобирается на маке через iconutil (build-macos.sh)
        print("  icon.iconset обновлён")

    installer_assets = ROOT / "installer" / "assets"
    mark = load_clean_mark(APPS[0])
    white_mark = make_white_mark(mark)
    make_wizard_header(mark, installer_assets / "wizard-header.bmp")
    make_wizard_sidebar(white_mark, installer_assets / "wizard-sidebar.bmp")
    print(f"== {installer_assets}: wizard-header.bmp и wizard-sidebar.bmp пересобраны")


if __name__ == "__main__":
    main()
