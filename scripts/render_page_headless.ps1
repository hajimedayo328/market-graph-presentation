param([string]$Url, [string]$OutDir, [int]$Width = 1200, [int]$Height = 6400, [string]$Name = "page")
# 画面に出さずに Edge でページを描画し、描画後の DOM と全体スクリーンショットを保存する（自作ページの動作確認用）
$ErrorActionPreference = "Stop"
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
$prof = Join-Path $OutDir "edge_profile"
$common = @("--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--virtual-time-budget=8000", "--user-data-dir=`"$prof`"")
$dom = Join-Path $OutDir "$Name.dom.html"
$err = Join-Path $OutDir "$Name.err.txt"
Start-Process -FilePath $edge -ArgumentList ($common + @("--dump-dom", $Url)) -RedirectStandardOutput $dom -RedirectStandardError $err -Wait -NoNewWindow
$png = Join-Path $OutDir "$Name.png"
Start-Process -FilePath $edge -ArgumentList ($common + @("--window-size=$Width,$Height", "--screenshot=`"$png`"", $Url)) -RedirectStandardError (Join-Path $OutDir "$Name.err2.txt") -Wait -NoNewWindow
Write-Output ("dom {0} bytes, png {1} bytes" -f (Get-Item $dom).Length, (Get-Item $png).Length)
