CHCP 65001
@echo off 
cd ../
echo 尝试启动后端程序
echo 等待一分钟以上没有出现新的内容说明不正常
runtime\python.exe ./Inference/src/tts_backend.py

pause