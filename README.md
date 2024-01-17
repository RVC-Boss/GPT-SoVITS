# GPT-SoVITS - Voice Conversion and Text-to-Speech WebUI

## Demo Video and Features

Check out our demo video in Chinese: [Bilibili Demo](https://www.bilibili.com/video/BV12g4y1m7Uw/)

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

### Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, and Chinese.

4. **WebUI Tools:** Integrated tools include voice accompaniment separation, automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

## Todo List

0. **High Priority:**
   - Localization in Japanese and English.
   - User guide.

1. **Features:**
   - Zero-shot voice conversion (5s) / few-shot voice conversion (1min).
   - TTS speaking speed control.
   - Enhanced TTS emotion control.
   - Experiment with changing SoVITS token inputs to probability distribution of vocabs.
   - Improve English and Japanese text frontend.
   - Develop tiny and larger-sized TTS models.
   - Colab scripts.
   - Expand training dataset (2k -> 10k).

## Requirements (How to Install)

### Python and PyTorch Version

Tested with Python 3.9, PyTorch 2.0.1, and CUDA 11.

### Quick Install with Conda

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
### Install Manually
#### Pip Packages

```bash
pip install torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm cn2an pypinyin pyopenjtalk g2p_en chardet
```

#### Additional Requirements

If you need Chinese ASR (supported by FunASR), install:

```bash
pip install modelscope torchaudio sentencepiece funasr
```

#### FFmpeg

##### Conda Users
```bash
conda install ffmpeg
```

##### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### MacOS Users

```bash
brew install ffmpeg
```

##### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root.

### Pretrained Models

Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS\pretrained_models`.

For Chinese ASR, download models from [Damo ASR Models](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files) and place them in `tools/damo_asr/models`.

For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

## Dataset Format

The TTS annotation .list file format:

```
vocal_path|speaker_name|language|text
```

Example:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

Language dictionary:

- 'zh': Chinese
- 'ja': Japanese
- 'en': English

## Credits

Special thanks to the following projects and contributors:

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
