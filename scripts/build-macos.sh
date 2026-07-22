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
#   ~/Qt/6.8.2/macos
# Если через brew (рекомендуемый вызов):
#   ./scripts/build-macos.sh "$(brew --prefix qt)"
#
# Замечание: Homebrew Qt разбит на qtbase/qtdeclarative/qtsvg/... —
# macdeployqt иногда ругается на VirtualKeyboard/Svg. Для локального запуска
# это обычно терпимо; для раздачи .dmg лучше Qt Online Installer.

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
    # Homebrew иногда кладёт macdeployqt в qtbase, а не в meta-пакет qt
    for cand in \
        "$QT_PREFIX/bin/macdeployqt" \
        "$(brew --prefix qtbase 2>/dev/null)/bin/macdeployqt" \
        "$(brew --prefix qt 2>/dev/null)/bin/macdeployqt"
    do
        if [ -n "$cand" ] && [ -x "$cand" ]; then
            MACDEPLOYQT="$cand"
            break
        fi
    done
fi
if [ ! -x "$MACDEPLOYQT" ]; then
    echo "!! macdeployqt не найден в $QT_PREFIX/bin"
    exit 1
fi

# Homebrew Qt разбит на keg'и (qtbase/qtdeclarative/qtsvg/...).
# macdeployqt сам их не видит — собираем -libpath для всех модулей.
collect_brew_libpaths() {
    local -a paths=()
    local brew_prefix=""
    if command -v brew >/dev/null 2>&1; then
        brew_prefix="$(brew --prefix 2>/dev/null || true)"
    fi
    if [ -n "$brew_prefix" ]; then
        paths+=("$brew_prefix/lib")
        local d
        for d in "$brew_prefix"/opt/qt*/lib; do
            [ -d "$d" ] && paths+=("$d")
        done
    fi
    # префикс, который передали вручную / Cellar
    [ -d "$QT_PREFIX/lib" ] && paths+=("$QT_PREFIX/lib")
    # уникализируем
    printf '%s\n' "${paths[@]}" | awk 'NF && !seen[$0]++'
}

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
    # Homebrew: CMAKE_PREFIX_PATH должен включать lib/cmake (или сам prefix)
    local cmake_prefix="$QT_PREFIX"
    if [ -d "$QT_PREFIX/lib/cmake" ]; then
        cmake_prefix="$QT_PREFIX"
    fi
    cmake -S "$src_dir" -B "$build_dir" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$cmake_prefix" \
        >/dev/null || cmake -S "$src_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$cmake_prefix"

    echo "==> Собираю $name (Release)"
    cmake --build "$build_dir" --config Release -j"$(sysctl -n hw.ncpu)"

    local bundle
    bundle=$(find "$build_dir" -maxdepth 2 -name "*.app" | head -n1)
    if [ -z "$bundle" ]; then
        echo "!! Не найден .app бандл после сборки $name"
        exit 1
    fi

    echo "==> macdeployqt: $bundle"
    local -a deploy_args=("$bundle" "-qmldir=$src_dir/qml")
    local lib
    while IFS= read -r lib; do
        [ -n "$lib" ] && deploy_args+=("-libpath=$lib")
    done < <(collect_brew_libpaths)
    echo "    libpaths: $(collect_brew_libpaths | tr '\n' ' ')"
    # Homebrew Qt тянет лишние QML-плагины (VirtualKeyboard и т.п.) — не валимся на их rpath,
    # если сам .app уже собран. Для переноса на другой Mac лучше Qt Online Installer.
    set +e
    "$MACDEPLOYQT" "${deploy_args[@]}"
    local deploy_rc=$?
    set -e
    if [ "$deploy_rc" -ne 0 ]; then
        echo "!! macdeployqt вернул код $deploy_rc (часто из-за Homebrew split-пакетов)."
        echo "   Если .app запускается локально — ок. Для раздачи лучше Qt из Online Installer."
        echo "   Дополнительно можно поставить: brew install qtsvg qtvirtualkeyboard"
    fi

    # macdeployqt ломает подпись → на macOS 15 краш: Code Signature Invalid / Invalid Page.
    # Ad-hoc codesign (-) + снятие quarantine чинят запуск на своей машине.
    echo "==> codesign (ad-hoc) + xattr: $bundle"
    xattr -cr "$bundle" 2>/dev/null || true
    codesign --force --deep --sign - "$bundle"

    mkdir -p "$DIST_DIR"
    rm -rf "$DIST_DIR/$(basename "$bundle")"
    cp -R "$bundle" "$DIST_DIR/"
    local dist_app="$DIST_DIR/$(basename "$bundle")"
    xattr -cr "$dist_app" 2>/dev/null || true
    codesign --force --deep --sign - "$dist_app"
    echo "==> Готово: $dist_app"
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
