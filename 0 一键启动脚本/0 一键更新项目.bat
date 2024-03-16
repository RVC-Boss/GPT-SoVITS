CHCP 65001
@echo off
setlocal


echo 设置需要同步的本地仓库路径
set REPO_PATH=../

echo 切换到仓库目录
cd /d %REPO_PATH%

echo 设置 PortableGit 的路径
set GIT_PATH=PortableGit/bin

echo 更新所有子模块
"%GIT_PATH%\git.exe" submodule update --init --recursive

echo 执行 git pull 更新本地仓库
"%GIT_PATH%\git.exe" stash
"%GIT_PATH%\git.exe" pull https://gitee.com/xxoy/GPT-SoVITS-Inference.git main

echo.
echo 更新完成！
pause
