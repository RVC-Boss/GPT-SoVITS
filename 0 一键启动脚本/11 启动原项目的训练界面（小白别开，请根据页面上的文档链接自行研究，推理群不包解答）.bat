CHCP 65001
@echo off 
cd ../
echo 尝试启动原版的训练推理界面
start  http://127.0.0.1:9874
runtime\python.exe ./webui.py

pause