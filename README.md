I am organizing and uploading the codes. It will be public in one day.

# demo video and features

https://www.bilibili.com/video/BV12g4y1m7Uw/

todo

# todolist

todo

# Requirments (How to install)

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
download and put them in the GPT-SoVITS root.
- download [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

- download [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe)

## You need download some pretrained models

### pretrained GPT-SoVITS models/SSL feature model/Chinese BERT model

put these files

https://huggingface.co/lj1995/GPT-SoVITS

to 

GPT_SoVITS\pretrained_models

### Chinese ASR (Additionally)

put these files

https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files

https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files

https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files

 to 

tools/damo_asr/models

 ![image](https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/aa376752-9f9d-4101-9a09-867bf4df6f6a)

### UVR5 (Vocals/Accompaniment Separation & Reverberation Removal. Additionally) 

put the models you need from 

https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights

to

tools/uvr5/uvr5_weights

# Credits

todo


