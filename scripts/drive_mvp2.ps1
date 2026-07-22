# второй проход: убрать выделение кликом в пустое место, скриншоты с JSON и без
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WDrv2 {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, int e);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

function Shot($rect, $file) {
    $w = $rect.Right - $rect.Left
    $h = $rect.Bottom - $rect.Top
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bmp.Size)
    $bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Write-Output "saved $file"
}

function Click($x, $y) {
    [WDrv2]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 200
    [WDrv2]::mouse_event(2, 0, 0, 0, 0)
    [WDrv2]::mouse_event(4, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 350
}

$proc = Get-Process MvSearch -ErrorAction Stop | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
[WDrv2]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 600

$r = New-Object WDrv2+RECT
[WDrv2]::GetWindowRect($proc.MainWindowHandle, [ref]$r) | Out-Null

# 1) кадр с открытым JSON: кликаем на заголовок «Извлечённые факты», чтобы снять выделение
Click ($r.Left + 320) ($r.Top + 186)
Start-Sleep -Milliseconds 400
Shot $r "figures\apps\cpp_mvp_json.png"

# 2) скрываем JSON кнопкой (справа сверху карточки фактов) и снимаем кадр с ранжированием
Click ($r.Left + 942) ($r.Top + 185)
Start-Sleep -Milliseconds 700
Shot $r "figures\apps\cpp_mvp_search.png"
