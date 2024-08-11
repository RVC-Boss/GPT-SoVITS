$ErrorActionPreference = "SilentlyContinue"
chcp 65001
& "$PSScriptRoot\runtime\python.exe" "$PSScriptRoot\webui.py zh_CN"
pause
