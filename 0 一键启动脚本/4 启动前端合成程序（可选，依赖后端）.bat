CHCP 65001
@echo off 
cd ../
echo 尝试启动程序，请耐心等待gradio启动，等待十几秒
start  http://127.0.0.1:9867
runtime\python.exe ./Inference/src/TTS_Webui.py

pause