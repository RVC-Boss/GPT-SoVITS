$ErrorActionPreference = "SilentlyContinue"
chcp 65001
Set-Location $PSScriptRoot
$runtimePath = Join-Path $PSScriptRoot "runtime"
$env:PATH = "$runtimePath;$env:PATH"
& "$runtimePath\python.exe" -I "$PSScriptRoot\webui.py" zh_CN
pause
