CHCP 65001
@echo off
 
cd ../
echo Try to start the program, please wait patiently for the backend to start, wait for more than ten seconds
echo if there is no new content, it means that the backend is not normal
runtime\python.exe ./Inference/src/tts_backend.py

pause