$ErrorActionPreference = "SilentlyContinue"
chcp 65001
Set-Location $PSScriptRoot
$runtimePath = Join-Path $PSScriptRoot "runtime"
$env:PATH = "$runtimePath"
& "$runtimePath\python.exe" -s "$PSScriptRoot\webui.py" zh_CN
pause
