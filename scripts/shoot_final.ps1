# все скриншоты через самосъёмку (MV_SHOOT): точные кадры без обрезки экрана
$root = "C:\Users\kamau\Documents\мвидео"
$mvp = "C:\mv-app\dist\mvsearch\MvSearch.exe"
$lbl = "C:\mv-app\dist\mvlabel\MvLabel.exe"
$out = "$root\figures\apps"

function RunShot($exe, $shot, $query, $demo, $extraArgs) {
    $env:MV_SHOOT = $shot
    $env:MV_QUERY = $query
    $env:MV_DEMO = $demo
    if ($extraArgs) { Start-Process $exe -ArgumentList $extraArgs -Wait } else { Start-Process $exe -Wait }
    if (Test-Path $shot) { Write-Output "saved $shot" } else { Write-Output "FAILED $shot" }
}

RunShot $mvp "$out\cpp_mvp_search.png" "наушники logitech g pro x se 128гб" "" $null
RunShot $mvp "$out\cpp_mvp_json.png" "айфон 15 про 256гб" "" "--json"
RunShot $mvp "$out\cpp_mvp_stats.png" "" "" "--stats"

RunShot $lbl "$out\cpp_label_stage1.png" "" "stage1" $null
RunShot $lbl "$out\cpp_label_stage2.png" "" "stage2" $null
RunShot $lbl "$out\cpp_label_stage3.png" "" "stage3" $null
RunShot $lbl "$out\cpp_label_match.png" "" "match" $null

$env:MV_SHOOT = ""; $env:MV_QUERY = ""; $env:MV_DEMO = ""
