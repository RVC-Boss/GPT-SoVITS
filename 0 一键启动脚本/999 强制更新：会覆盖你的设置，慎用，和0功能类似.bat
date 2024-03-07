CHCP 65001
@echo off
setlocal

:: 设置 PortableGit 的路径
set GIT_PATH=../PortableGit/bin

:: 设置需要同步的本地仓库路径
set REPO_PATH=./

:: 添加 PortableGit 到 PATH，以便可以执行 git 命令
set PATH=%GIT_PATH%;%PATH%

:: 切换到仓库目录
cd /d %REPO_PATH%

:: 执行 git pull 更新本地仓库
git fetch https://github.com/X-T-E-R/GPT-SoVITS-Inference.git main
git reset --hard FETCH_HEAD
git submodule update --init --recursive

echo.
echo 更新完成！
pause