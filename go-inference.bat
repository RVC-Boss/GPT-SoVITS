call %CONDA_PREFIX%\Scripts\activate.bat %CONDA_PREFIX%
call conda activate GPTSoVits
cd /d %~dp0
python .\GPT_SoVITS\inference_webui.py
pause
