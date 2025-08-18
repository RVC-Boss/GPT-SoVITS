Param (
    [Parameter(Mandatory=$true)][ValidateSet("CU126", "CU128", "CPU")][string]$Device,
    [Parameter(Mandatory=$true)][ValidateSet("HF", "HF-Mirror", "ModelScope")][string]$Source,
    [switch]$DownloadUVR5
)

$global:ErrorActionPreference = 'Stop'

trap {
    Write-ErrorLog $_
}

function Write-ErrorLog {
    param (
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    Write-Host "`n[ERROR] Command failed:" -ForegroundColor Red
    if (-not $ErrorRecord.Exception.Message){
    } else {
        Write-Host "Message:" -ForegroundColor Red 
        $ErrorRecord.Exception.Message -split "`n" | ForEach-Object {
            Write-Host "    $_"
        }
    }

    Write-Host "Command:" -ForegroundColor Red  -NoNewline
    Write-Host " $($ErrorRecord.InvocationInfo.Line)".Replace("`r", "").Replace("`n", "")
    Write-Host "Location:" -ForegroundColor Red -NoNewline
    Write-Host " $($ErrorRecord.InvocationInfo.ScriptName):$($ErrorRecord.InvocationInfo.ScriptLineNumber)"
    Write-Host "Call Stack:" -ForegroundColor DarkRed
    $ErrorRecord.ScriptStackTrace -split "`n" | ForEach-Object {
        Write-Host "    $_" -ForegroundColor DarkRed
    }

    exit 1
}

function Write-Info($msg) {
    Write-Host "[INFO]:" -ForegroundColor Green -NoNewline
    Write-Host " $msg"
}
function Write-Success($msg) {
    Write-Host "[SUCCESS]:" -ForegroundColor Blue -NoNewline
    Write-Host " $msg"
}


function Invoke-Conda {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    $output = & conda install -y -q -c conda-forge @Args 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "Conda Install $Args Failed" -ForegroundColor Red
        $errorMessages = @()
        foreach ($item in $output) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $msg = $item.Exception.Message
                Write-Host "$msg" -ForegroundColor Red
                $errorMessages += $msg
            }
            else {
                Write-Host $item
                $errorMessages += $item
            }
        }
        throw [System.Exception]::new(($errorMessages -join "`n"))
    }
}

function Invoke-Pip {
    param (
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )
    
    $output = & pip install @Args 2>&1
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -ne 0) {
        $errorMessages = @()
        Write-Host "Pip Install $Args Failed" -ForegroundColor Red
        foreach ($item in $output) {
            if ($item -is [System.Management.Automation.ErrorRecord]) {
                $msg = $item.Exception.Message
                Write-Host "$msg" -ForegroundColor Red
                $errorMessages += $msg
            }
            else {
                Write-Host $item
                $errorMessages += $item
            }
        }
        throw [System.Exception]::new(($errorMessages -join "`n"))
    }
}

function Invoke-Download {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [Parameter()]
        [string]$OutFile
    )

    try {
        $params = @{
            Uri = $Uri
        }

        if ($OutFile) {
            $params["OutFile"] = $OutFile
        }

        $null = Invoke-WebRequest @params -ErrorAction Stop

    } catch {
        Write-Host "Failed to download:" -ForegroundColor Red
        Write-Host "  $Uri"
        throw
    }
}

function Invoke-Unzip {
    param($ZipPath, $DestPath)
    Expand-Archive -Path $ZipPath -DestinationPath $DestPath -Force
    Remove-Item $ZipPath -Force
}

chcp 65001
Set-Location $PSScriptRoot

Write-Info "Installing FFmpeg & CMake..."
Invoke-Conda  ffmpeg cmake
Write-Success "FFmpeg & CMake Installed"

$PretrainedURL  = ""
$G2PWURL        = ""
$UVR5URL        = ""
$NLTKURL        = ""
$OpenJTalkURL   = ""

switch ($Source) {
    "HF" {
        Write-Info "Download Model From HuggingFace"
        $PretrainedURL = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
        $G2PWURL       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        $UVR5URL       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
        $NLTKURL       = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL  = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "HF-Mirror" {
        Write-Info "Download Model From HuggingFace-Mirror"
        $PretrainedURL = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
        $G2PWURL       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        $UVR5URL       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/uvr5_weights.zip"
        $NLTKURL       = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        $OpenJTalkURL  = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
    "ModelScope" {
        Write-Info "Download Model From ModelScope"
        $PretrainedURL = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
        $G2PWURL       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
        $UVR5URL       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/uvr5_weights.zip"
        $NLTKURL       = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
        $OpenJTalkURL  = "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
    }
}

if (-not (Test-Path "GPT_SoVITS/pretrained_models/sv")) {
    Write-Info "Downloading Pretrained Models..."
    Invoke-Download -Uri $PretrainedURL -OutFile "pretrained_models.zip"
    Invoke-Unzip "pretrained_models.zip" "GPT_SoVITS"
    Write-Success "Pretrained Models Downloaded"
} else {
    Write-Info "Pretrained Model Exists"
    Write-Info "Skip Downloading Pretrained Models"
}


if (-not (Test-Path "GPT_SoVITS/text/G2PWModel")) {
    Write-Info "Downloading G2PWModel..."
    Invoke-Download -Uri $G2PWURL -OutFile "G2PWModel.zip"
    Invoke-Unzip "G2PWModel.zip" "GPT_SoVITS/text"
    Write-Success "G2PWModel Downloaded"
} else {
    Write-Info "G2PWModel Exists"
    Write-Info "Skip Downloading G2PWModel"
}

if ($DownloadUVR5) {
    if (-not (Test-Path "tools/uvr5/uvr5_weights")) {
        Write-Info "Downloading UVR5 Models..."
        Invoke-Download -Uri $UVR5URL -OutFile "uvr5_weights.zip"
        Invoke-Unzip "uvr5_weights.zip" "tools/uvr5"
        Write-Success "UVR5 Models Downloaded"
    } else {
        Write-Info "UVR5 Models Exists"
        Write-Info "Skip Downloading UVR5 Models"
    }
}

switch ($Device) {
    "CU128" {
        Write-Info "Installing PyTorch For CUDA 12.8..."
        Invoke-Pip torch torchaudio --index-url "https://download.pytorch.org/whl/cu128"
    }
    "CU126" {
        Write-Info "Installing PyTorch For CUDA 12.6..."
        Invoke-Pip torch torchaudio --index-url "https://download.pytorch.org/whl/cu126"
    }
    "CPU" {
        Write-Info "Installing PyTorch For CPU..."
        Invoke-Pip torch torchaudio --index-url "https://download.pytorch.org/whl/cpu"
    }
}
Write-Success "PyTorch Installed"

Write-Info "Installing Python Dependencies From requirements.txt..."
Invoke-Pip -r extra-req.txt --no-deps
Invoke-Pip -r requirements.txt
Write-Success "Python Dependencies Installed"

Write-Info "Downloading NLTK Data..."
Invoke-Download -Uri $NLTKURL -OutFile "nltk_data.zip"
Invoke-Unzip "nltk_data.zip" (python -c "import sys; print(sys.prefix)").Trim()

Write-Info "Downloading Open JTalk Dict..."
Invoke-Download -Uri $OpenJTalkURL -OutFile "open_jtalk_dic_utf_8-1.11.tar.gz"
$target = (python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))").Trim()
tar -xzf open_jtalk_dic_utf_8-1.11.tar.gz -C $target
Remove-Item "open_jtalk_dic_utf_8-1.11.tar.gz" -Force
Write-Success "Open JTalk Dic Downloaded"

Write-Success "Installation Completed"
