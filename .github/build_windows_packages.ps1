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

$baseHF = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
$PRETRAINED_URL = "$baseHF/pretrained_models.zip"
$G2PW_URL = "$baseHF/G2PWModel.zip"
$UVR5_URL = "$baseHF/uvr5_weights.zip"
$NLTK_URL = "$baseHF/nltk_data.zip"
$JTALK_URL = "$baseHF/open_jtalk_dic_utf_8-1.11.tar.gz"

$PYTHON_VERSION = "3.11.12"
$RELEASE_VERSION = "20250409"

Write-Host "[INFO] Cleaning .git..."
Remove-Item "$srcDir\.git" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[INFO] Creating tmp dir..."
New-Item -ItemType Directory -Force -Path $tmpDir

Write-Host "[INFO] System Python version:"
python --version
python -m site

Write-Host "[INFO] Downloading Python $PYTHON_VERSION..."
$zst = "$tmpDir\python.tar.zst"
Invoke-WebRequest "https://github.com/astral-sh/python-build-standalone/releases/download/$RELEASE_VERSION/cpython-$PYTHON_VERSION+$RELEASE_VERSION-x86_64-pc-windows-msvc-pgo-full.tar.zst" -OutFile $zst
& "C:\Program Files\7-Zip\7z.exe" e $zst -o"$tmpDir" -aoa
$tar = Get-ChildItem "$tmpDir" -Filter "*.tar" | Select-Object -First 1
& "C:\Program Files\7-Zip\7z.exe" x $tar.FullName -o"$tmpDir\extracted" -aoa
Move-Item "$tmpDir\extracted\python\install" "$srcDir\runtime"

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
& ".\runtime\python.exe" -m ensurepip
switch ($cuda) {
    "cu124" {
        & ".\runtime\python.exe" -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    }
    "cu128" {
        & ".\runtime\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    }
    default {
        Write-Error "Unsupported CUDA version: $cuda"
        exit 1
    }
}

Write-Host "[INFO] Installing dependencies..."
& ".\runtime\python.exe" -m pip install -r extra-req.txt --no-deps --no-warn-script-location
& ".\runtime\python.exe" -m pip install -r requirements.txt --no-warn-script-location

Write-Host "[INFO] Downloading NLTK and pyopenjtalk dictionary..."
$PYTHON = ".\runtime\python.exe"
$prefix = & $PYTHON -c "import sys; print(sys.prefix)"
$jtalkPath = & $PYTHON -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))"
$nltkZip = "$tmpDir\nltk_data.zip"
$jtalkTar = "$tmpDir\open_jtalk_dic_utf_8-1.11.tar.gz"

Invoke-WebRequest -Uri $NLTK_URL -OutFile $nltkZip
Expand-Archive -Path $nltkZip -DestinationPath $prefix -Force
Remove-Item $nltkZip

Invoke-WebRequest -Uri $JTALK_URL -OutFile $jtalkTar
& "C:\Program Files\7-Zip\7z.exe" e $jtalkTar -o"$tmpDir" -aoa
$innerTar = Get-ChildItem "$tmpDir" -Filter "*.tar" | Select-Object -First 1
& "C:\Program Files\7-Zip\7z.exe" x $innerTar.FullName -o"$jtalkPath" -aoa
Remove-Item $jtalkTar
Remove-Item $innerTar.FullName

Write-Host "[INFO] Preparing final directory... $pkgName"
Get-ChildItem ../
Get-ChildItem .
Remove-Item "$tmpDir" -Force -Recurse -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path ../$pkgName
Copy-Item "$srcDir\*" -Destination ../$pkgName -Recurse -Force
Set-Location ../
Get-ChildItem .
Compress-Archive -Path "$pkgName" -DestinationPath "$pkgName.zip" -Force

Write-Host "[INFO] Uploading to ModelScope..."
$msUser = $env:MODELSCOPE_USERNAME
$msToken = $env:MODELSCOPE_TOKEN
if (-not $msUser -or -not $msToken) {
    Write-Error "Missing MODELSCOPE_USERNAME or MODELSCOPE_TOKEN"
    exit 1
}
python -m pip install --upgrade pip
python -m pip install modelscope --no-warn-script-location.
modelscope login --token $msToken
modelscope upload "$msUser/GPT-SoVITS-Packages" "$pkgName.zip" "data/$pkgName.zip" --repo-type model

Write-Host "[SUCCESS] Uploaded: $pkgName.zip"