CHCP 65001
@echo off 
cd ../

echo Try to start the program, please wait patiently for the frontend to start, wait for more than ten seconds
echo if there is no new content, it means that the frontend is not normal
echo if the browser does not pop up automatically, please manually open the browser and enter http://127.0.0.1:9867
runtime\python.exe ./Inference/src/TTS_Webui.py

pause