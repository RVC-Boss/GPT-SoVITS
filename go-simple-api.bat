@echo off
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

if exist "%SCRIPT_DIR%\runtime\python.exe" (
  "%SCRIPT_DIR%\runtime\python.exe" "%SCRIPT_DIR%\simple_api.py" -c "%SCRIPT_DIR%\simple_api.yaml" %*
) else (
  python "%SCRIPT_DIR%\simple_api.py" -c "%SCRIPT_DIR%\simple_api.yaml" %*
)

pause
