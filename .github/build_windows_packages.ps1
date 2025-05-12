$ErrorActionPreference = "Stop"

$today = Get-Date -Format "MMdd"
$cuda = $env:TORCH_CUDA
if (-not $cuda) {
    Write-Error "Missing TORCH_CUDA env (e.g., cu124 or cu128)"
    exit 1
}
$pkgName = "GPT-SoVITS-$today-$cuda"
$tmpDir = "tmp"
$srcDir = $PWD

Write-Host "[INFO] Cleaning .git..."
Remove-Item "$srcDir\.git" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[INFO] Creating tmp dir..."
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

Write-Host "[INFO] Downloading Python..."
$zst = "$tmpDir\python.tar.zst"
Invoke-WebRequest "https://github.com/astral-sh/python-build-standalone/releases/download/20250409/cpython-3.11.12+20250409-x86_64-pc-windows-msvc-pgo-full.tar.zst" -OutFile $zst

Write-Host "1111"
Get-ChildItem $tmpDir
& "C:\Program Files\7-Zip\7z.exe" e $zst -o"$tmpDir" -aoa
Write-Host "2222"
Get-ChildItem $tmpDir
$tar = Get-ChildItem "$tmpDir" -Filter "*.tar" | Select-Object -First 1
& "C:\Program Files\7-Zip\7z.exe" x $tar.FullName -o"$tmpDir\extracted" -aoa
Write-Host "3333"
Get-ChildItem $tmpDir
Move-Item "$tmpDir\extracted\python\install" "$srcDir\runtime"

$baseHF = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
$PRETRAINED_URL = "$baseHF/pretrained_models.zip"
$G2PW_URL = "$baseHF/G2PWModel.zip"
$UVR5_URL = "$baseHF/uvr5_weights.zip"

function DownloadAndUnzip($url, $targetRelPath) {
    $filename = Split-Path $url -Leaf
    $tmpZip = "$tmpDir\$filename"
    Invoke-WebRequest $url -OutFile $tmpZip
    Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force
    Move-Item "$tmpDir\$($filename -replace '\.zip$', '')" "$srcDir\$targetRelPath" -Force
    Remove-Item $tmpZip
}

Write-Host "[INFO] Download pretrained_models..."
DownloadAndUnzip $PRETRAINED_URL "GPT_SoVITS\pretrained_models"

Write-Host "[INFO] Download G2PWModel..."
DownloadAndUnzip $G2PW_URL "GPT_SoVITS\text\G2PWModel"

Write-Host "[INFO] Download UVR5 model..."
DownloadAndUnzip $UVR5_URL "tools\uvr5\uvr5_weights"

Write-Host "[INFO] Downloading funasr..."
$funasrUrl = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/funasr.zip"
$funasrZip = "$tmpDir\funasr.zip"
Invoke-WebRequest -Uri $funasrUrl -OutFile $funasrZip
Expand-Archive -Path $funasrZip -DestinationPath "$srcDir\tools\asr\models" -Force
Remove-Item $funasrZip

Write-Host "[INFO] Download ffmpeg..."
$ffUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$ffZip = "$tmpDir\ffmpeg.zip"
Invoke-WebRequest -Uri $ffUrl -OutFile $ffZip
Expand-Archive $ffZip -DestinationPath $tmpDir -Force
$ffDir = Get-ChildItem -Directory "$tmpDir" | Where-Object { $_.Name -like "ffmpeg*" } | Select-Object -First 1
Move-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$srcDir"
Move-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$srcDir"
Remove-Item $ffZip
Remove-Item $ffDir.FullName -Recurse -Force

Write-Host "[INFO] Installing PyTorch..."
$torchCmd = switch ($cuda) {
    "cu124" { "pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124" }
    "cu128" { "pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128" }
    default { Write-Error "Unsupported CUDA version: $cuda"; exit 1 }
}
& ".\runtime\python.exe" -m ensurepip
& ".\runtime\python.exe" -c "$torchCmd"

Write-Host "[INFO] Installing dependencies..."
& ".\runtime\python.exe" -m pip install -r extra-req.txt --no-deps
& ".\runtime\python.exe" -m pip install -r requirements.txt
& ".\runtime\python.exe" -c "import nltk; nltk.download(['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict'])"

Write-Host "[INFO] Preparing final directory..."
$finalDir = "..\$pkgName"
Move-Item $srcDir $finalDir -Force
Compress-Archive -Path "$finalDir\*" -DestinationPath "$pkgName.zip" -Force

$msUser = $env:MODELSCOPE_USERNAME
$msToken = $env:MODELSCOPE_TOKEN
if (-not $msUser -or -not $msToken) {
    Write-Error "Missing MODELSCOPE_USERNAME or MODELSCOPE_TOKEN"
    exit 1
}
modelscope login --token $msToken
modelscope upload "$msUser/GPT-SoVITS-Packages" "$pkgName.zip" "data/$pkgName.zip" --repo-type model

Write-Host "[SUCCESS] Uploaded: $pkgName.zip"