<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
A Powerful Few-shot Voice Conversion and Text-to-Speech WebUI.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)

[**English**](./README.md) | [**中文简体**](./docs/cn/README.md) | [**日本語**](./docs/ja/README.md) | [**한국어**](./docs/ko/README.md)

</div>

---

> Check out our [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw) here!

Unseen speakers few-shot fine-tuning demo:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

For users in China region, you can use AutoDL Cloud Docker to experience the full functionality online: https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official

## Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, and Chinese.

4. **WebUI Tools:** Integrated tools include voice accompaniment separation, automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

## Environment Preparation

If you are a Windows user (tested with win>=10) you can install directly via the prezip. Just download the [prezip](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true), unzip it and double-click go-webui.bat to start GPT-SoVITS-WebUI.

### Tested Environments

- Python 3.9, PyTorch 2.0.1, CUDA 11
- Python 3.10.13, PyTorch 2.1.2, CUDA 12.3
- Python 3.9, PyTorch 2.3.0.dev20240122, macOS 14.3 (Apple silicon, GPU)

_Note: numba==0.56.4 require py<3.11_

### Quick Install with Conda

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### Install Manually

#### Pip Packages

```bash
pip install -r requirements.txt
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

Download pretrained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS/pretrained_models`.

For UVR5 (Vocals/Accompaniment Separation & Reverberation Removal, additionally), download models from [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) and place them in `tools/uvr5/uvr5_weights`.

Users in China region can download these two models by entering the links below and clicking "Download a copy"

- [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)

- [UVR5 Weights](https://www.icloud.com.cn/iclouddrive/0bekRKDiJXboFhbfm3lM2fVbA#UVR5_Weights)

For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/damo_asr/models`.

### For Mac Users

If you are a Mac user, make sure you meet the following conditions for training and inferencing with GPU:

- Mac computers with Apple silicon
- macOS 12.3 or later
- Xcode command-line tools installed by running `xcode-select --install`

_Other Macs can do inference with CPU only._

Then install by using the following commands:

#### Create Environment

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
```

#### Install Requirements

```bash
pip install -r requirements.txt
pip uninstall torch torchaudio
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### Using Docker

#### docker-compose.yaml configuration

0. Regarding image tags: Due to rapid updates in the codebase and the slow process of packaging and testing images, please check [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) for the currently packaged latest images and select as per your situation, or alternatively, build locally using a Dockerfile according to your own needs.
1. Environment Variables：

- is_half: Controls half-precision/double-precision. This is typically the cause if the content under the directories 4-cnhubert/5-wav32k is not generated correctly during the "SSL extracting" step. Adjust to True or False based on your actual situation.

2. Volumes Configuration，The application's root directory inside the container is set to /workspace. The default docker-compose.yaml lists some practical examples for uploading/downloading content.
3. shm_size： The default available memory for Docker Desktop on Windows is too small, which can cause abnormal operations. Adjust according to your own situation.
4. Under the deploy section, GPU-related settings should be adjusted cautiously according to your system and actual circumstances.

#### Running with docker compose

```
docker compose -f "docker-compose.yaml" up -d
```

#### Running with docker command

As above, modify the corresponding parameters based on your actual situation, then run the following command:

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## Dataset Format

The TTS annotation .list file format:

```
vocal_path|speaker_name|language|text
```

Language dictionary:

- 'zh': Chinese
- 'ja': Japanese
- 'en': English

Example:

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```

## Todo List

- [ ] **High Priority:**

  - [x] Localization in Japanese and English.
  - [ ] User guide.
  - [x] Japanese and English dataset fine tune training.

- [ ] **Features:**
  - [ ] Zero-shot voice conversion (5s) / few-shot voice conversion (1min).
  - [ ] TTS speaking speed control.
  - [ ] Enhanced TTS emotion control.
  - [ ] Experiment with changing SoVITS token inputs to probability distribution of vocabs.
  - [ ] Improve English and Japanese text frontend.
  - [ ] Develop tiny and larger-sized TTS models.
  - [x] Colab scripts.
  - [ ] Try expand training dataset (2k hours -> 10k hours).
  - [ ] better sovits base model (enhanced audio quality)
  - [ ] model mix

## (Optional) If you need, here will provide the command line operation mode
Use the command line to open the WebUI for UVR5
```
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
```
If you can't open a browser, follow the format below for UVR processing,This is using mdxnet for audio processing
```
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
```
This is how the audio segmentation of the dataset is done using the command line
```
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips> 
    --hop_size <step_size_for_computing_volume_curve>
```
This is how dataset ASR processing is done using the command line(Only Chinese)
```
python tools/damo_asr/cmd-asr.py "<Path to the directory containing input audio files>"
```
ASR processing is performed through Faster_Whisper(ASR marking except Chinese)

(No progress bars, GPU performance may cause time delays)
```
python ./tools/damo_asr/WhisperASR.py -i <input> -o <output> -f <file_name.list> -l <language>
```
A custom list save path is enabled
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

## Thanks to all contributors for their efforts

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
