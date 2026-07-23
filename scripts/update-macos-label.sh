#!/usr/bin/env bash
# Обновление MvLabel на macOS БЕЗ потери разметки.
#
# Разметка лежит НЕ внутри .app, а в:
#   ~/Library/Application Support/MVideo/М.Видео Разметка/labels/
# (см. LabelStore::labelsDir). Поэтому достаточно заменить бандл.
#
# Использование (на Маке участницы, после git pull / скачивания релиза):
#   chmod +x scripts/update-macos-label.sh
#   ./scripts/update-macos-label.sh                  # берёт dist-macos/MvLabel.app
#   ./scripts/update-macos-label.sh /path/to/MvLabel.app
#   ./scripts/update-macos-label.sh /path/to/MvLabel.dmg
#
# Скрипт:
#   1) делает страховочный бэкап labels в ~/Desktop/mvlabel_labels_backup_<дата>
#   2) закрывает старое приложение
#   3) ставит новый .app в /Applications (или рядом со старым)
#   4) снимает quarantine (Gatekeeper) и ad-hoc codesign
#   5) проверяет, что бэкап и живые labels на месте

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT_DIR/dist-macos/MvLabel.app}"
APP_SUPPORT="$HOME/Library/Application Support/MVideo"
# Qt использует organizationName=MVideo + applicationName из main.cpp
LABELS_CANDIDATES=(
  "$APP_SUPPORT/М.Видео Разметка/labels"
  "$APP_SUPPORT/MVideo Labeling/labels"
  "$HOME/Library/Application Support/ru.mvideo.labeling/labels"
)

stamp="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$HOME/Desktop/mvlabel_labels_backup_$stamp"

echo "==> Источник обновления: $SRC"

# если передали .dmg — смонтировать и найти .app внутри
TMP_MOUNT=""
cleanup() {
  if [ -n "$TMP_MOUNT" ] && [ -d "$TMP_MOUNT" ]; then
    hdiutil detach "$TMP_MOUNT" -quiet || true
  fi
}
trap cleanup EXIT

if [[ "$SRC" == *.dmg ]]; then
  echo "==> Монтирую DMG…"
  TMP_MOUNT="$(mktemp -d /tmp/mvlabel_dmg.XXXX)"
  hdiutil attach "$SRC" -nobrowse -quiet -mountpoint "$TMP_MOUNT"
  FOUND="$(find "$TMP_MOUNT" -maxdepth 2 -name 'MvLabel.app' -print -quit || true)"
  if [ -z "$FOUND" ]; then
    echo "!! В DMG нет MvLabel.app"
    exit 1
  fi
  SRC="$FOUND"
fi

if [ ! -d "$SRC" ]; then
  echo "!! Не найден $SRC"
  echo "   Сначала собери: ./scripts/build-macos.sh \"\$(brew --prefix qt)\""
  echo "   или скачай .app/.dmg из GitHub Releases и передай путь аргументом."
  exit 1
fi

# --- 1. бэкап разметки ---
mkdir -p "$BACKUP_DIR"
backed=0
for d in "${LABELS_CANDIDATES[@]}"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null || true)" ]; then
    echo "==> Бэкап разметки: $d"
    cp -R "$d" "$BACKUP_DIR/$(basename "$(dirname "$d")")_labels"
    backed=1
  fi
done
# на всякий случай — всё дерево Application Support/MVideo
if [ -d "$APP_SUPPORT" ]; then
  cp -R "$APP_SUPPORT" "$BACKUP_DIR/ApplicationSupport_MVideo" 2>/dev/null || true
  backed=1
fi
if [ "$backed" -eq 0 ]; then
  echo "==> Разметки пока нет (или новый аккаунт) — бэкап пустой, это ок"
else
  echo "==> Страховочный бэкап: $BACKUP_DIR"
fi

# --- 2. закрыть старое приложение ---
echo "==> Закрываю MvLabel (если запущено)…"
osascript -e 'tell application "MvLabel" to quit' 2>/dev/null || true
killall MvLabel 2>/dev/null || true
sleep 1

# --- 3. куда ставить ---
DEST="/Applications/MvLabel.app"
# если уже стоит в другом месте — обновим там
for cand in \
  "/Applications/MvLabel.app" \
  "$HOME/Applications/MvLabel.app" \
  "$(mdfind 'kMDItemCFBundleIdentifier == ru.mvideo.labeling' 2>/dev/null | head -n1)"
do
  if [ -n "$cand" ] && [ -d "$cand" ]; then
    DEST="$cand"
    break
  fi
done

echo "==> Устанавливаю в: $DEST"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -R "$SRC" "$DEST"

# --- 4. Gatekeeper / quarantine ---
echo "==> Снимаю quarantine и делаю ad-hoc codesign…"
xattr -cr "$DEST" || true
codesign --force --deep --sign - "$DEST" 2>/dev/null || true

# --- 5. проверка ---
echo ""
echo "==> Готово. Разметка НЕ внутри .app — она здесь:"
for d in "${LABELS_CANDIDATES[@]}"; do
  if [ -d "$d" ]; then
    n=$(find "$d" -type f | wc -l | tr -d ' ')
    echo "    $d  ($n файлов)"
  fi
done
echo "    бэкап: $BACKUP_DIR"
echo ""
echo "Запуск: open \"$DEST\""
echo "Если Gatekeeper ругается: ПКМ → Открыть, или:"
echo "  xattr -cr \"$DEST\" && open \"$DEST\""
