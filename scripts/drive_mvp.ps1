# самодостаточный сценарий: очистить поле, ввести запрос, скриншот
param(
    [string]$Query = "наушники logitech g pro x se 128гб",
    [string]$OutFile = "figures\apps\cpp_mvp_search.png",
    [int]$FieldY = 128,
    [switch]$SkipType
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class WDrv {
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, int e);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@

$proc = Get-Process MvSearch -ErrorAction Stop | Where-Object { $_.MainWindowHandle -ne 0 } | Select-Object -First 1
[WDrv]::SetForegroundWindow($proc.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 600

$r = New-Object WDrv+RECT
[WDrv]::GetWindowRect($proc.MainWindowHandle, [ref]$r) | Out-Null

if (-not $SkipType) {
    [WDrv]::SetCursorPos($r.Left + 550, $r.Top + $FieldY) | Out-Null
    Start-Sleep -Milliseconds 200
    [WDrv]::mouse_event(2, 0, 0, 0, 0)
    [WDrv]::mouse_event(4, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 300
    [System.Windows.Forms.SendKeys]::SendWait("^a")
    Start-Sleep -Milliseconds 150
    [System.Windows.Forms.SendKeys]::SendWait("{DEL}")
    Start-Sleep -Milliseconds 200
    [System.Windows.Forms.SendKeys]::SendWait($Query)
    Start-Sleep -Milliseconds 300
    [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
    Start-Sleep -Milliseconds 1800
}

$w = $r.Right - $r.Left
$h = $r.Bottom - $r.Top
$bmp = New-Object System.Drawing.Bitmap($w, $h)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size)
$bmp.Save($OutFile, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output "saved $OutFile"
