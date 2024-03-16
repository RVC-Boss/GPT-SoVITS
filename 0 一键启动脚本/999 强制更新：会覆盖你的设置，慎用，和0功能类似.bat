CHCP 65001
@echo off
setlocal


echo 设置需要同步的本地仓库路径
set REPO_PATH=../

echo 切换到仓库目录
cd /d %REPO_PATH%

echo 设置 PortableGit 的路径
set GIT_PATH=PortableGit/bin

echo 撤销之前的强制换源
"%GIT_PATH%\git.exe" config --global --unset url."https://gitclone.com/".insteadOf

echo 强制覆盖所有子模块
"%GIT_PATH%\git.exe" submodule update --init --recursive
"%GIT_PATH%\git.exe" submodule foreach --recursive "git fetch origin plug_in && git reset --hard origin/plug_in"

echo 强制覆盖本地仓库
"%GIT_PATH%\git.exe" fetch https://gitee.com/xxoy/GPT-SoVITS-Inference.git main
"%GIT_PATH%\git.exe" reset --hard FETCH_HEAD

echo.
echo 更新完成！
pause