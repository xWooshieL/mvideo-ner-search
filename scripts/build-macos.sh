#!/usr/bin/env bash
# Сборка обоих приложений (Умный поиск + Разметка) под macOS.
# Запускать НА МАКЕ (Qt для macOS собирается только там, кросс-компиляция с Windows не работает).
#
# Требования:
#   - Xcode Command Line Tools:  xcode-select --install
#   - CMake 3.21+:                brew install cmake
#   - Qt 6.5+ для macOS:          через Qt Online Installer (qt.io) или `brew install qt`
#
# Использование:
#   chmod +x scripts/build-macos.sh
#   ./scripts/build-macos.sh [путь_к_Qt/lib/cmake]
#
# Если Qt поставлен через Qt Online Installer, путь обычно:
#   ~/Qt/6.8.2/macos/lib/cmake
# Если через brew:
#   $(brew --prefix qt)/lib/cmake

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QT_PREFIX="${1:-$HOME/Qt/6.8.2/macos}"
DIST_DIR="$ROOT_DIR/dist-macos"

echo "==> Repo root: $ROOT_DIR"
echo "==> Qt prefix: $QT_PREFIX"

if [ ! -d "$QT_PREFIX" ]; then
    echo "!! Не найден Qt по пути $QT_PREFIX"
    echo "   Укажи путь явно: ./scripts/build-macos.sh /путь/к/Qt/6.8.2/macos"
    exit 1
fi

MACDEPLOYQT="$QT_PREFIX/bin/macdeployqt"
if [ ! -x "$MACDEPLOYQT" ]; then
    echo "!! macdeployqt не найден в $QT_PREFIX/bin"
    exit 1
fi

# .icns из подготовленных .iconset (генерируются scripts/make_iconset.py на любой ОС)
make_icns() {
    local app_dir="$1"
    local iconset="$app_dir/assets/icon.iconset"
    local icns="$app_dir/assets/icon.icns"
    if [ -d "$iconset" ] && [ ! -f "$icns" ]; then
        echo "==> iconutil: $iconset -> $icns"
        iconutil -c icns "$iconset" -o "$icns"
    fi
}

build_app() {
    local name="$1"          # mvsearch | mvlabel
    local src_dir="$ROOT_DIR/cpp/$name"
    local build_dir="$src_dir/build-macos"

    make_icns "$src_dir"

    echo "==> Конфигурирую $name"
    cmake -S "$src_dir" -B "$build_dir" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$QT_PREFIX/lib/cmake" \
        >/dev/null || cmake -S "$src_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$QT_PREFIX/lib/cmake"

    echo "==> Собираю $name (Release)"
    cmake --build "$build_dir" --config Release -j"$(sysctl -n hw.ncpu)"

    local bundle
    bundle=$(find "$build_dir" -maxdepth 2 -name "*.app" | head -n1)
    if [ -z "$bundle" ]; then
        echo "!! Не найден .app бандл после сборки $name"
        exit 1
    fi

    echo "==> macdeployqt: $bundle"
    "$MACDEPLOYQT" "$bundle" -qmldir="$src_dir/qml"

    mkdir -p "$DIST_DIR"
    rm -rf "$DIST_DIR/$(basename "$bundle")"
    cp -R "$bundle" "$DIST_DIR/"
    echo "==> Готово: $DIST_DIR/$(basename "$bundle")"
}

build_app mvsearch
build_app mvlabel

echo ""
echo "==> Упаковка в .dmg"
for app in "$DIST_DIR"/*.app; do
    base=$(basename "$app" .app)
    dmg="$DIST_DIR/${base}.dmg"
    rm -f "$dmg"
    hdiutil create -volname "$base" -srcfolder "$app" -ov -format UDZO "$dmg" >/dev/null
    echo "    $dmg"
done

echo ""
echo "Всё готово. Результаты в $DIST_DIR:"
ls -la "$DIST_DIR"
