CHCP 65001
@echo off
setlocal


:: 设置需要同步的本地仓库路径
set REPO_PATH=../

:: 切换到仓库目录
cd /d %REPO_PATH%

:: 设置 PortableGit 的路径
set GIT_PATH=PortableGit/bin



echo 强制覆盖所有子模块
"%GIT_PATH%\git.exe" submodule update --init --recursive
"%GIT_PATH%\git.exe" submodule foreach --recursive "git fetch origin plug_in && git reset --hard origin/plug_in"

echo 执行 git pull 更新本地仓库
"%GIT_PATH%\git.exe" fetch https://github.com/X-T-E-R/GPT-SoVITS-Inference.git main
"%GIT_PATH%\git.exe" reset --hard FETCH_HEAD

echo.
echo 更新完成！
pause