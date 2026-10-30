[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("CU126", "CU128", "CPU")]
    [string]$Device
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "python is required" }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is required on PATH" }

function Invoke-Python {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

Invoke-Python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), 'Python 3.10-3.12 is required'"
Invoke-Python -m pip install --upgrade pip
switch ($Device) {
    "CU126" { Invoke-Python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126 }
    "CU128" { Invoke-Python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 }
    "CPU" { Invoke-Python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu }
}
Invoke-Python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
Invoke-Python -c "import nltk; [nltk.download(x, quiet=True, raise_on_error=True) for x in ('cmudict','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]"
Invoke-Python -c "import pyopenjtalk; pyopenjtalk.g2p('test')"
Write-Host "Inference runtime installed. Start with: python api_v2.py -a 0.0.0.0 -p 9880"
