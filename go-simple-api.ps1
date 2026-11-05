$ErrorActionPreference = "Stop"
chcp 65001
Set-Location $PSScriptRoot

$runtimePython = Join-Path $PSScriptRoot "runtime\python.exe"
$apiScript = Join-Path $PSScriptRoot "simple_api.py"
$apiConfig = Join-Path $PSScriptRoot "simple_api.yaml"

if (Test-Path $runtimePython) {
    & $runtimePython $apiScript -c $apiConfig @args
} else {
    python $apiScript -c $apiConfig @args
}
