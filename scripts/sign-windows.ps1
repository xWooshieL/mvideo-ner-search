# Подписывает .exe кодовым сертификатом M.Video (self-signed).
#
# ВАЖНО про Windows Defender / SmartScreen:
#   - Настоящий "доверенный" сертификат (DigiCert/Sectigo и т.п.) стоит денег
#     и требует юр. лицо + проверку личности — сделать его мгновенно и бесплатно
#     нельзя физически.
#   - Self-signed сертификат (этот скрипт) убирает "Unknown Publisher" и
#     показывает имя издателя, но SmartScreen у ПОСТОРОННИХ пользователей
#     всё равно может предупреждать, пока сертификат не в их Trusted Root.
#   - Чтобы у команды (Никита/Некит/Лиза и т.д.) предупреждений не было вовсе —
#     один раз импортируйте installer/certs/mvideo_codesign.cer в
#     "Доверенные корневые центры сертификации" (Trusted Root CA) на их машинах:
#       certutil -addstore -f "Root" installer\certs\mvideo_codesign.cer
#
# Использование:
#   .\scripts\sign-windows.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$PFX = Join-Path $ROOT "installer\certs\mvideo_codesign.pfx"
$PFX_PASSWORD = "MVideoNer2026!"

$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter "signtool.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*\x64\*" } | Select-Object -First 1 -ExpandProperty FullName
if (-not $signtool) {
    Write-Error "signtool.exe не найден. Установи Windows SDK (Signing Tools) через Visual Studio Installer."
    exit 1
}

$targets = @(
    "$ROOT\installer\output\MVideo_SmartSearch_Setup_0.1.0.exe",
    "$ROOT\installer\output\MVideo_Labeling_Setup_0.1.3.exe"
)

foreach ($f in $targets) {
    if (Test-Path $f) {
        Write-Output "Подписываю: $f"
        & $signtool sign /f $PFX /p $PFX_PASSWORD /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 "$f"
    } else {
        Write-Warning "Не найден: $f (сначала собери установщик)"
    }
}

Write-Output ""
Write-Output "Готово. Сертификат для импорта командой (убирает предупреждение на их машинах):"
Write-Output "  certutil -addstore -f `"Root`" `"$ROOT\installer\certs\mvideo_codesign.cer`""
