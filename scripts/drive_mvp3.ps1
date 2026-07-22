# третий проход: скрыть JSON -> кадр с ранжированием; открыть статистику -> кадр
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WDrv3 {
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
    [WDrv3]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 250
    [WDrv3]::mouse_event(2, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 60
    [WDrv3]::mouse_event(4, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 500
}

$proc = Get-Process MvSearch -ErrorAction Stop | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
[WDrv3]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 700

$r = New-Object WDrv3+RECT
[WDrv3]::GetWindowRect($proc.MainWindowHandle, [ref]$r) | Out-Null

# кнопка «Скрыть JSON» — правый край карточки фактов
Click ($r.Left + 941) ($r.Top + 186)
Start-Sleep -Milliseconds 600
Shot $r "figures\apps\cpp_mvp_search.png"

# страница статистики
Click ($r.Left + 94) ($r.Top + 151)
Start-Sleep -Milliseconds 800
Shot $r "figures\apps\cpp_mvp_stats.png"

# вернёмся на поиск
Click ($r.Left + 94) ($r.Top + 114)
