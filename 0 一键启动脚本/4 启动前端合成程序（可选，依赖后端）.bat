CHCP 65001
@echo off 
cd ../
echo 尝试启动程序
start  http://127.0.0.1:9867
runtime\python.exe ./Inference/src/TTS_Webui.py

pause