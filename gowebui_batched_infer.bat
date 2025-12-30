@echo off
:: 1. 切换命令行编码为UTF-8，解决中文显示乱码（必须放在最前面）
chcp 65001 > nul

:: 2. 获取当前bat文件所在目录并格式化
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: 3. 切换到脚本根目录
cd /d "%SCRIPT_DIR%"

:: 4. 创建专属TEMP目录（补充主页面的核心步骤）
if not exist "TEMP" md "TEMP"
set "TEMP=%SCRIPT_DIR%\TEMP"

:: 5. 设置核心环境变量（补充推理脚本依赖的配置）
set "version=v2Pro"
:: 语言配置
set "language=zh_CN"
:: 启用半精度推理（GPU用户推荐，CPU用户改为False）
set "is_half=True"
:: 指定GPU卡号（多卡可修改，无GPU则删除此行）
set "_CUDA_VISIBLE_DEVICES=0"

:: 6. 将runtime目录加入环境变量，确保能调用内置python
set "PATH=%SCRIPT_DIR%\runtime;%PATH%"

:: 7. 直接启动并行推理脚本，传入中文语言参数
echo 正在启动GPT-SoVITS并行推理页面...
runtime\python.exe -I GPT_SoVITS/inference_webui_fast.py zh_CN

:: 8. 执行完成后暂停，便于查看报错信息
pause