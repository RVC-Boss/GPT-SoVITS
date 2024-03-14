CHCP 65001
@echo off 
cd ../
echo 请确保您的主项目运行正常
runtime\python.exe -m pip  config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
runtime\python.exe -m pip  config set install.trusted-host pypi.tuna.tsinghua.edu.cn
runtime\python.exe -m pip install -r ./requirements.txt

pause