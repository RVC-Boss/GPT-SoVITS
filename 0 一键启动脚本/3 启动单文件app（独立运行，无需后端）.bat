CHCP 65001
@echo off 
cd ../
echo 尝试启动程序，请耐心等待gradio启动，等待十几秒，若未自动弹出浏览器，请手动打开浏览器输入http://127.0.0.1:7860
runtime\python.exe app.py

pause