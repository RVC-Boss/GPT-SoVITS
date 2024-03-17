CHCP 65001
@echo off
setlocal


echo Set the local repo path
set REPO_PATH=../

echo cd to the local repo path
cd /d %REPO_PATH%

echo setting the PortableGit path
set GIT_PATH=PortableGit/bin


echo Update submodule
"%GIT_PATH%\git.exe" submodule update --init --recursive
"%GIT_PATH%\git.exe" submodule foreach --recursive "git fetch origin plug_in && git reset --hard origin/plug_in"

echo git reset --hard
"%GIT_PATH%\git.exe" fetch https://gitee.com/xxoy/GPT-SoVITS-Inference.git main
"%GIT_PATH%\git.exe" reset --hard FETCH_HEAD

echo.
pause