CHCP 65001
@echo off 
cd ../
echo Try to start the program, please wait patiently for gradio to start
echo if the browser does not pop up automatically, please manually open the browser and enter http://127.0.0.1:9868
runtime\python.exe ./Inference/src/Character_Manager.py

pause