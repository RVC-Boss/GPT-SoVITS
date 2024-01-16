I am organizing and uploading the codes. It will be public in one day.

# requirment (How to install)

## python and pytorch version
py39+pytorch2.0.1+cu11 passed the test.

## pip packages
pip install torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm==4.59.0 cn2an pypinyin pyopenjtalk g2p_en

## additionally
If you need the Chinese ASR feature supported by funasr, you should

pip install modelscope sentencepiece funasr

## You need ffmpeg.

### Ubuntu/Debian users
```bash
sudo apt install ffmpeg
```
### MacOS users
```bash
brew install ffmpeg
```
### Windows users
download and put it to the GPT-SoVITS root.
- download [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- download [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

