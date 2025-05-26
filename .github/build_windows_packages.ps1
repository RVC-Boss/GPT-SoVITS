$ErrorActionPreference = "Stop"

Write-Host "Current location: $(Get-Location)"

$cuda = $env:TORCH_CUDA
if (-not $cuda) {
    Write-Error "Missing TORCH_CUDA env (cu124 or cu128)"
    exit 1
}

$date = $env:DATE_SUFFIX
if ([string]::IsNullOrWhiteSpace($date)) {
    $date = Get-Date -Format "MMdd"
}

$pkgName = "GPT-SoVITS-$date"
$tmpDir = "tmp"
$srcDir = $PWD

$suffix = $env:PKG_SUFFIX
if (-not [string]::IsNullOrWhiteSpace($suffix)) {
    $pkgName = "$pkgName$suffix"
}

$pkgName = "$pkgName-$cuda"

$baseHF = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
$PRETRAINED_URL = "$baseHF/pretrained_models.zip"
$G2PW_URL = "$baseHF/G2PWModel.zip"
$UVR5_URL = "$baseHF/uvr5_weights.zip"
$NLTK_URL = "$baseHF/nltk_data.zip"
$JTALK_URL = "$baseHF/open_jtalk_dic_utf_8-1.11.tar.gz"

$PYTHON_VERSION = "3.11.12"
$PY_RELEASE_VERSION = "20250409"

Write-Host "[INFO] Cleaning .git..."
Remove-Item "$srcDir\.git" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[INFO] Creating tmp dir..."
New-Item -ItemType Directory -Force -Path $tmpDir

Write-Host "[INFO] System Python version:"
python --version
python -m site

Write-Host "[INFO] Downloading Python $PYTHON_VERSION..."
$zst = "$tmpDir\python.tar.zst"
Invoke-WebRequest "https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE_VERSION/cpython-$PYTHON_VERSION+$PY_RELEASE_VERSION-x86_64-pc-windows-msvc-pgo-full.tar.zst" -OutFile $zst
& "C:\Program Files\7-Zip\7z.exe" e $zst -o"$tmpDir" -aoa
$tar = Get-ChildItem "$tmpDir" -Filter "*.tar" | Select-Object -First 1
& "C:\Program Files\7-Zip\7z.exe" x $tar.FullName -o"$tmpDir\extracted" -aoa
Move-Item "$tmpDir\extracted\python\install" "$srcDir\runtime"

Write-Host "[INFO] Copying Redistributing Visual C++ Runtime..."
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$redistRoot = Join-Path $vsPath "VC\Redist\MSVC"
$targetVer = Get-ChildItem -Path $redistRoot -Directory |
    Where-Object { $_.Name -match "^14\." } |
    Sort-Object Name -Descending |
    Select-Object -First 1
$x64Path = Join-Path $targetVer.FullName "x64"
Get-ChildItem -Path $x64Path -Directory | Where-Object {
    $_.Name -match '^Microsoft\..*\.(CRT|OpenMP)$'
} | ForEach-Object {
    Get-ChildItem -Path $_.FullName -Filter "*.dll" | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination "$srcDir\runtime" -Force
    }
}

function DownloadAndUnzip($url, $targetRelPath) {
    $filename = Split-Path $url -Leaf
    $tmpZip = "$tmpDir\$filename"
    Invoke-WebRequest $url -OutFile $tmpZip
    Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force
    $subdirName = $filename -replace '\.zip$', ''
    $sourcePath = Join-Path $tmpDir $subdirName
    $destRoot = Join-Path $srcDir $targetRelPath
    $destPath = Join-Path $destRoot $subdirName
    if (Test-Path $destPath) {
        Remove-Item $destPath -Recurse -Force
    }
    Move-Item $sourcePath $destRoot
    Remove-Item $tmpZip
}

Write-Host "[INFO] Download pretrained_models..."
DownloadAndUnzip $PRETRAINED_URL "GPT_SoVITS"

Write-Host "[INFO] Download G2PWModel..."
DownloadAndUnzip $G2PW_URL "GPT_SoVITS\text"

Write-Host "[INFO] Download UVR5 model..."
DownloadAndUnzip $UVR5_URL "tools\uvr5"

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
Move-Item "$($ffDir.FullName)\bin\ffmpeg.exe" "$srcDir\runtime"
Move-Item "$($ffDir.FullName)\bin\ffprobe.exe" "$srcDir\runtime"
Remove-Item $ffZip
Remove-Item $ffDir.FullName -Recurse -Force

Write-Host "[INFO] Installing PyTorch..."
& ".\runtime\python.exe" -m ensurepip
& ".\runtime\python.exe" -m pip install --upgrade pip --no-warn-script-location
switch ($cuda) {
    "cu124" {
        & ".\runtime\python.exe" -m pip install torch==2.6 torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-warn-script-location
    }
    "cu128" {
        & ".\runtime\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128 --no-warn-script-location
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

Write-Host "[INFO] Preparing final directory $pkgName ..."
$items = @(Get-ChildItem -Filter "*.sh") +
         @(Get-ChildItem -Filter "*.ipynb") +
         @("$tmpDir", ".github", "Docker", "docs", ".gitignore", ".dockerignore", "README.md")
Remove-Item $items -Force -Recurse -ErrorAction SilentlyContinue
$curr = Get-Location
Set-Location ../
Get-ChildItem .
Copy-Item -Path $curr -Destination $pkgName -Recurse
$7zPath = "$pkgName.7z"
$start = Get-Date
Write-Host "Compress Starting at $start"
& "C:\Program Files\7-Zip\7z.exe" a -t7z "$7zPath" "$pkgName" -m0=lzma2 -mx=9 -md=1g -ms=1g -mmc=500 -mfb=273 -mlc=0 -mlp=4 -mpb=4 -mc=8g -mmt=on -bsp1
$end = Get-Date
Write-Host "Elapsed time: $($end - $start)"
Get-ChildItem .

python -m pip install --upgrade pip
python -m pip install "modelscope" "huggingface_hub[hf_transfer]" --no-warn-script-location

Write-Host "[INFO] Uploading to ModelScope..."
$msUser = $env:MODELSCOPE_USERNAME
$msToken = $env:MODELSCOPE_TOKEN
if (-not $msUser -or -not $msToken) {
    Write-Error "Missing MODELSCOPE_USERNAME or MODELSCOPE_TOKEN"
    exit 1
}
modelscope upload "$msUser/GPT-SoVITS-Packages" "$7zPath" "$7zPath" --repo-type model --token $msToken

Write-Host "[SUCCESS] Uploaded: $7zPath to ModelScope"

Write-Host "[INFO] Uploading to HuggingFace..."
$hfUser = $env:HUGGINGFACE_USERNAME
$hfToken = $env:HUGGINGFACE_TOKEN
if (-not $hfUser -or -not $hfToken) {
    Write-Error "Missing HUGGINGFACE_USERNAME or HUGGINGFACE_TOKEN"
    exit 1
}
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
huggingface-cli upload "$hfUser/GPT-SoVITS-Packages" "$7zPath" "$7zPath" --repo-type model --token $hfToken

Write-Host "[SUCCESS] Uploaded: $7zPath to HuggingFace"
