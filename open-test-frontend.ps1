$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
Start-Process (Join-Path $PSScriptRoot "test_frontend\index.html")
