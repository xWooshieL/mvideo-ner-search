# полный набор скриншотов C++ приложений (запрос через MV_QUERY, utf-8)
$OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WShot {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

function Shot-Window($procName, $file, $waitMs) {
    Start-Sleep -Milliseconds $waitMs
    $proc = Get-Process $procName -ErrorAction Stop | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    [WShot]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
    Start-Sleep -Milliseconds 700
    $r = New-Object WShot+RECT
    [WShot]::GetWindowRect($proc.MainWindowHandle, [ref]$r) | Out-Null
    $w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size)
    $bmp.Save($file, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
    Write-Output "saved $file"
    Stop-Process -Name $procName -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 700
}

$dist = "C:\mv-app\dist\mvsearch\MvSearch.exe"
$out = "figures\apps"

# 1) поиск с результатами (JSON скрыт)
$env:MV_QUERY = "наушники logitech g pro x se 128гб"
Start-Process $dist
Shot-Window "MvSearch" "$out\cpp_mvp_search.png" 7500

# 2) поиск с открытым JSON
$env:MV_QUERY = "айфон 15 про 256гб"
Start-Process $dist -ArgumentList "--json"
Shot-Window "MvSearch" "$out\cpp_mvp_json.png" 7500

# 3) статистика
$env:MV_QUERY = ""
Start-Process $dist -ArgumentList "--stats"
Shot-Window "MvSearch" "$out\cpp_mvp_stats.png" 7500
