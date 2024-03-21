# GSVI : GPT-SoVITS Inference Plugin

Welcome to GSVI, an inference-specialized plugin built on top of GPT-SoVITS to enhance your text-to-speech (TTS) experience with a user-friendly API interface. This plugin enriches the [original GPT-SoVITS project](https://github.com/RVC-Boss/GPT-SoVITS), making voice synthesis more accessible and versatile.

Please note that we do not recommend using GSVI for training. Its existence is to make the process of using GPT-soVITS simpler and more comfortable for others, and to make model sharing easier.

This fork is mainly based on the `fast_inference_` branch, using a lot of PR code contributed by [ChasonJiang](https://github.com/ChasonJiang). Thanks to this great developer. ”Dalao NB！“

At the same time, the Inference folder used by this branch is the main submodule, coming from https://github.com/X-T-E-R/TTS-for-GPT-soVITS. If you find that the Inference is empty after pulling, please manually execute:

```
git submodule add -b plug_in https://github.com/X-T-E-R/TTS-for-GPT-soVITS.git Inference
git submodule update --init --recursive
```

## Features

- High-level abstract interface for easy character and emotion selection
- Comprehensive TTS engine support (speaker selection, speed adjustment, volume control)
- User-friendly design for everyone
- Simply place the shared character model folder, and you can quickly use it.
- High compatibility and extensibility for various platforms and applications (for example: SillyTavern)

## Getting Started

1. Install manually or use prezip for Windows
2. Put your character model folders
3. Run bat file or run python file manually
4. If you encounter issues, join our community or consult the FAQ. QQ Group: 863760614 , Discord (AI Hub): 

We look forward to seeing how you use GSVI to bring your creative projects to life!

Prezip : https://huggingface.co/XTer123/GSVI_prezip/tree/main
## Usage

### Use With Bat Files

You could see a bunch of bat files in `0 Bat Files/`

- If you want to update, then run bat 0 and 1 (or 999 0 1)
- If you want to start with a single gradio file, then run bat 3
- If you want to start with backend and frontend , run bat 5 and 6
- If you want to manage your models, run 10.bat

### Python Files

#### Start with a single gradio file

- Gradio Application: `app.py`  (In the root of GSVI)

#### Start with backend and frontend mod

- Flask Backend Program: `Inference/src/tts_backend.py`
- Gradio Frontend Application: `Inference/src/TTS_Webui.py`
- Other Frontend Applications or Services Using Our API 

### Model Management

- Gradio Model Management Interface: `Inference/src/Character_Manager.py`

##  API Documentation

For API documentation, visit our [Yuque documentation page](https://www.yuque.com/xter/zibxlp/knu8p82lb5ipufqy). or [API Doc.md](./api_doc.md)

## Model Folder Format

In a character model folder, like `trained/Character1/`

Put the pth / ckpt / wav files in it, the wav should be named as the prompt text

Like :

```
trained
--hutao
----hutao-e75.ckpt
----hutao_e60_s3360.pth
----hutao said something.wav
```

### Add a emotion for your model

To make that, open the Model Manage Tool (10.bat / Inference/src/Character_Manager.py)

It can assign a reference audio to each emotion, aiming to achieve the implementation of emotion options.

## Installation

You could install this with the guide bellow, then download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`, and put your character model folder in `trained`

Or just download the pre-packaged distribution for Windows. ( then put your character model folder in `trained` )

About the character model folder, see below

### Tested Environments

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.3.0.dev20240122, macOS 14.3 (Apple silicon)

_Note: numba==0.56.4 requires py<3.11_

### Windows

If you are a Windows user (tested with win>=10), you can directly download the [pre-packaged distribution]() and double-click on _go-webui.bat_ to start GPT-SoVITS-WebUI.

Or  ```pip install -r requirements.txt``` , and then double click the `install.bat`

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**

First make sure you have installed FFmpeg by running `brew install ffmpeg` or `conda install ffmpeg`, then install by using the following commands:

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
git submodule init
git submodule update --init --recursive
```

### Install FFmpeg ( No need if use prezip )

#### Conda Users

```bash
conda install ffmpeg
```

#### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

#### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root.

### Pretrained Models ( No need if use prezip )

Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.


## Docker

Writing Now, Please Wait

Remove the `pyaudio` in the `requirements.txt` !!!!


## Credits

This fork is mainly based on the `fast_inference_` branch of [GPT-soVITS](https://github.com/RVC-Boss/GPT-SoVITS) project, using a lot of PR code contributed by [ChasonJiang](https://github.com/ChasonJiang).

Special thanks to the following projects and contributors:

### Theoretical
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
### Pretrained Models
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
### Text Frontend for Inference
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [LangSegment](https://github.com/juntaosun/LangSegment)
### WebUI Tools
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
  
## Thanks to all contributors for their efforts

<a href="https://github.com/X-T-E-R/GPT-SoVITS-Inference/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=X-T-E-R/GPT-SoVITS-Inference" />
</a>

