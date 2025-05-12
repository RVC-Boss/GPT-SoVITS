$ErrorActionPreference = "SilentlyContinue"
chcp 65001
& "$PSScriptRoot\runtime\python.exe" -I "$PSScriptRoot\webui.py" zh_CN
pause
