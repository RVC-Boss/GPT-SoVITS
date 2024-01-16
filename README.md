
# demo video and features

demo video in Chinese: https://www.bilibili.com/video/BV12g4y1m7Uw/

few shot fine tuning demo:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

features:

1、input 5s vocal, zero shot TTS

2、1min training dataset, fine tune (few shot TTS. The TTS model trained using few-shot techniques exhibits significantly better similarity and realism in the speaker's voice compared to zero-shot.)

3、Cross lingual (inference another language that is different from the training dataset language), now support English, Japanese and Chinese

4、This WebUI integrates tools such as voice accompaniment separation, automatic segmentation of training sets, Chinese ASR, text labeling, etc., to help beginners quickly create their own training datasets and GPT/SoVITS models.

# todolist

1、zero shot voice conversion(5s) /few shot voice converion(1min)

2、TTS speaking speed control

3、more TTS emotion control

4、experiment about change sovits token inputs to probability distribution of vocabs

5、better English and Japanese text frontend

6、tiny version and larger-sized TTS models

7、colab scripts

8、more training dataset(2k->10k)

# Requirments (How to install)

## python and pytorch version
py39+pytorch2.0.1+cu11 passed the test.

## pip packages
pip install torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm==4.59.0 cn2an pypinyin pyopenjtalk g2p_en

## additionally
If you need the Chinese ASR feature supported by funasr, you should

pip install modelscope torchaudio sentencepiece funasr

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

# dataset format

The format of the TTS annotation .list file:

vocal path|speaker_name|language|text

e.g. D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.

language dictionary:

    'zh': Chinese
    
    "ja": Japanese
    
    'en': English
    


# Credits

https://github.com/innnky/ar-vits

https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR

https://github.com/jaywalnut310/vits

https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556

https://github.com/TencentGameMate/chinese_speech_pretrain

https://github.com/auspicious3000/contentvec/

https://github.com/jik876/hifi-gan

https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41

https://github.com/Anjok07/ultimatevocalremovergui

https://github.com/openvpi/audio-slicer

https://github.com/cronrpc/SubFix

https://modelscope.cn/models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch

https://github.com/FFmpeg/FFmpeg

https://github.com/gradio-app/gradio


