<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
强大的少样本语音转换与语音合成Web用户界面。<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)

[**English**](./README.md) | [**中文简体**](./README_ZH.md)

</div>

------



> 查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw) 

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

## 功能：
1. **零样本文本到语音（TTS）：** 输入5秒的声音样本，即刻体验文本到语音转换。

2. **少样本TTS：** 仅需1分钟的训练数据即可微调模型，提升声音相似度和真实感。

3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语和中文。

4. **WebUI工具：** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和GPT/SoVITS模型。

## 环境准备

如果你是Windows用户（已在win>=10上测试），可以直接通过预打包文件安装。只需下载[预打包文件](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true)，解压后双击go-webui.bat即可启动GPT-SoVITS-WebUI。

### Python和PyTorch版本

已在Python 3.9、PyTorch 2.0.1和CUDA 11上测试。

### 使用Conda快速安装

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```
### 手动安装包
#### Pip包

```bash
pip install torch numpy scipy tensorboard librosa==0.9.2 numba==0.56.4 pytorch-lightning gradio==3.14.0 ffmpeg-python onnxruntime tqdm cn2an pypinyin pyopenjtalk g2p_en chardet transformers
```

#### 额外要求

如果你需要中文自动语音识别（由FunASR支持），请安装：

```bash
pip install modelscope torchaudio sentencepiece funasr
```

#### FFmpeg

##### Conda 使用者
```bash
conda install ffmpeg
```

##### Ubuntu/Debian 使用者

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### MacOS 使用者

```bash
brew install ffmpeg
```

##### Windows 使用者

下载并将 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) 放置在 GPT-SoVITS 根目录下。

### 预训练模型


从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中。

对于中文自动语音识别（另外），从 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), 和 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 下载模型，并将它们放置在 `tools/damo_asr/models` 中。

对于UVR5（人声/伴奏分离和混响移除，另外），从 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 下载模型，并将它们放置在 `tools/uvr5/uvr5_weights` 中。


## 数据集格式

文本到语音（TTS）注释 .list 文件格式：

```
vocal_path|speaker_name|language|text
```

语言字典：

- 'zh': Chinese
- 'ja': Japanese
- 'en': English

示例：

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|en|I like playing Genshin.
```
## 待办事项清单

- [ ] **高优先级：**
   - [ ] 日语和英语的本地化。
   - [ ] 用户指南。
   - [ ] 日语和英语数据集微调训练。

- [ ] **Features:**
   - [ ] 零样本声音转换（5秒）/ 少样本声音转换（1分钟）。
   - [ ] TTS语速控制。
   - [ ] 增强的TTS情感控制。
   - [ ] 尝试将SoVITS令牌输入更改为词汇的概率分布。
   - [ ] 改进英语和日语文本前端。
   - [ ] 开发体积小和更大的TTS模型。
   - [ ] Colab脚本。
   - [ ] 扩展训练数据集（从2k小时到10k小时）。
   - [ ] 更好的sovits基础模型（增强的音频质量）。
   - [ ] 模型混合。

## 致谢

特别感谢以下项目和贡献者：

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

## 感谢所有贡献者的努力
<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
