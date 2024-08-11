$ErrorActionPreference = "SilentlyContinue"
chcp 65001
& "$PSScriptRoot\runtime\python.exe" "$PSScriptRoot\webui.py v1 zh_CN"
pause
