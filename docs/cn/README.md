<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
强大的少样本语音转换与语音合成Web用户界面。<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br> -->

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/lj1995/GPT-SoVITS-v2)
[![Discord](https://img.shields.io/discord/1198701940511617164?color=%23738ADB&label=Discord&style=for-the-badge)](https://discord.gg/dnrgs5GHfG)

[**English**](../../README.md) | **简体中文** | [**日本語**](../ja/README.md) | [**한국어**](../ko/README.md) | [**Türkçe**](../tr/README.md)

</div>

---

## 功能：

1. **零样本文本到语音（TTS）：** 输入 5 秒的声音样本，即刻体验文本到语音转换。

2. **少样本 TTS：** 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感。

3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语、韩语、粤语和中文。

4. **WebUI 工具：** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和 GPT/SoVITS 模型。

**查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**

未见过的说话者 few-shot 微调演示：

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**用户手册: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## 安装

中国地区的用户可[点击此处](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official)使用 AutoDL 云端镜像进行体验。

### 测试通过的环境

- Python 3.9，PyTorch 2.0.1，CUDA 11
- Python 3.10.13，PyTorch 2.1.2，CUDA 12.3
- Python 3.9，Pytorch 2.2.2，macOS 14.4.1（Apple 芯片）
- Python 3.9，PyTorch 2.2.2，CPU 设备

_注: numba==0.56.4 需要 python<3.11_

### Windows

如果你是 Windows 用户（已在 win>=10 上测试），可以下载[整合包](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true)，解压后双击 go-webui.bat 即可启动 GPT-SoVITS-WebUI。

**中国地区的用户可以[在此处下载整合包](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO)。**

### Linux

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
bash install.sh
```

### macOS

**注：在 Mac 上使用 GPU 训练的模型效果显著低于其他设备训练的模型，所以我们暂时使用 CPU 进行训练。**

1. 运行 `xcode-select --install` 安装 Xcode command-line tools。
2. 运行 `brew install ffmpeg` 安装 FFmpeg。
3. 完成上述步骤后，运行以下的命令来安装本项目：

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
pip install -r requirements.txt
```

### 手动安装

#### 安装 FFmpeg

##### Conda 用户

```bash
conda install ffmpeg
```

##### Ubuntu/Debian 用户

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
conda install -c conda-forge 'ffmpeg<7'
```

##### Windows 用户

下载并将 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) 放置在 GPT-SoVITS 根目录下。

安装 [Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) 环境(仅限韩语TTS)

##### MacOS 用户
```bash
brew install ffmpeg
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

### 在 Docker 中使用

#### docker-compose.yaml 设置

0. image 的标签：由于代码库更新很快，镜像的打包和测试又很慢，所以请自行在 [Docker Hub](https://hub.docker.com/r/breakstring/gpt-sovits) 查看当前打包好的最新的镜像并根据自己的情况选用，或者在本地根据您自己的需求通过 Dockerfile 进行构建。
1. 环境变量：

- is_half: 半精度/双精度控制。在进行 "SSL extracting" 步骤时如果无法正确生成 4-cnhubert/5-wav32k 目录下的内容时，一般都是它引起的，可以根据实际情况来调整为 True 或者 False。

2. Volume 设置，容器内的应用根目录设置为 /workspace。 默认的 docker-compose.yaml 中列出了一些实际的例子，便于上传/下载内容。
3. shm_size：Windows 下的 Docker Desktop 默认可用内存过小，会导致运行异常，根据自己情况酌情设置。
4. deploy 小节下的 gpu 相关内容，请根据您的系统和实际情况酌情设置。

#### 通过 docker compose 运行

```
docker compose -f "docker-compose.yaml" up -d
```

#### 通过 docker 命令运行

同上，根据您自己的实际情况修改对应的参数，然后运行如下命令：

```
docker run --rm -it --gpus=all --env=is_half=False --volume=G:\GPT-SoVITS-DockerTest\output:/workspace/output --volume=G:\GPT-SoVITS-DockerTest\logs:/workspace/logs --volume=G:\GPT-SoVITS-DockerTest\SoVITS_weights:/workspace/SoVITS_weights --workdir=/workspace -p 9880:9880 -p 9871:9871 -p 9872:9872 -p 9873:9873 -p 9874:9874 --shm-size="16G" -d breakstring/gpt-sovits:xxxxx
```

## 预训练模型

**中国地区的用户可以[在此处下载这些模型](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#nVNhX)。**

1. 从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将其放置在 `GPT_SoVITS/pretrained_models` 目录中。

2. 从 [G2PWModel_1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip) 下载模型，解压并重命名为 `G2PWModel`，然后将其放置在 `GPT_SoVITS/text` 目录中。（仅限中文TTS）

3. 对于 UVR5（人声/伴奏分离和混响移除，额外功能），从 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 下载模型，并将其放置在 `tools/uvr5/uvr5_weights` 目录中。

     - 如果你在 UVR5 中使用 `bs_roformer` 或 `mel_band_roformer`模型，你可以手动下载模型和相应的配置文件，并将它们放在 `tools/UVR5/UVR5_weights` 中。**重命名模型文件和配置文件，确保除后缀外**，模型和配置文件具有相同且对应的名称。此外，模型和配置文件名**必须包含“roformer”**，才能被识别为 roformer 类的模型。

     - 建议在模型名称和配置文件名中**直接指定模型类型**，例如`mel_mand_roformer`、`bs_roformer`。如果未指定，将从配置文中比对特征，以确定它是哪种类型的模型。例如，模型`bs_roformer_ep_368_sdr_12.9628.ckpt` 和对应的配置文件`bs_roformer_ep_368_sdr_12.9628.yaml` 是一对。`kim_mel_band_roformer.ckpt` 和 `kim_mel_band_roformer.yaml` 也是一对。


4. 对于中文 ASR（额外功能），从 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files)、[Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files) 和 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 下载模型，并将它们放置在 `tools/asr/models` 目录中。

5. 对于英语或日语 ASR（额外功能），从 [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3) 下载模型，并将其放置在 `tools/asr/models` 目录中。此外，[其他模型](https://huggingface.co/Systran) 可能具有类似效果且占用更少的磁盘空间。

## 数据集格式

文本到语音（TTS）注释 .list 文件格式：

```
vocal_path|speaker_name|language|text
```

语言字典：

- 'zh': 中文
- 'ja': 日语
- 'en': 英语
- 'ko': 韩语
- 'yue': 粤语

示例：

```
D:\GPT-SoVITS\xxx/xxx.wav|xxx|zh|我爱玩原神。
```

## 微调与推理

### 打开WebUI

#### 整合包用户

双击`go-webui.bat`或者使用`go-webui.ps1`
若想使用V1,则双击`go-webui-v1.bat`或者使用`go-webui-v1.ps1`

#### 其他

```bash
python webui.py <language(optional)>
```

若想使用V1,则

```bash
python webui.py v1 <language(optional)>
```
或者在webUI内动态切换

### 微调

#### 现已支持自动填充路径

    1. 填入训练音频路径
    2. 切割音频
    3. 进行降噪(可选)
    4. 进行ASR
    5. 校对标注
    6. 前往下一个窗口,点击训练

### 打开推理WebUI

#### 整合包用户

双击 `go-webui.bat` 或者使用 `go-webui.ps1` ,然后在 `1-GPT-SoVITS-TTS/1C-推理` 中打开推理webUI

#### 其他

```bash
python GPT_SoVITS/inference_webui.py <language(optional)>
```
或者

```bash
python webui.py
```
然后在 `1-GPT-SoVITS-TTS/1C-推理` 中打开推理webUI

## V2发布说明

新特性:

1. 支持韩语及粤语

2. 更好的文本前端

3. 底模由2k小时扩展至5k小时

4. 对低音质参考音频（尤其是来源于网络的高频严重缺失、听着很闷的音频）合成出来音质更好

    详见[wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

从v1环境迁移至v2

1. 需要pip安装requirements.txt更新环境

2. 需要克隆github上的最新代码

3. 需要从[huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main/gsv-v2final-pretrained) 下载预训练模型文件放到GPT_SoVITS\pretrained_models\gsv-v2final-pretrained下

    中文额外需要下载[G2PWModel_1.1.zip](https://paddlespeech.cdn.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip)（下载G2PW模型,解压并重命名为`G2PWModel`,将其放到`GPT_SoVITS/text`目录下）

## V3更新说明

新模型特点:

1. 音色相似度更像，需要更少训练集来逼近本人（不训练直接使用底模模式下音色相似性提升更大）

2. GPT合成更稳定，重复漏字更少，也更容易跑出丰富情感

    详见[wiki](https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7))

从v2环境迁移至v3

1. 需要pip安装requirements.txt更新环境

2. 需要克隆github上的最新代码

3. 从[huggingface](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)下载这些v3新增预训练模型 (s1v3.ckpt, s2Gv3.pth and models--nvidia--bigvgan_v2_24khz_100band_256x folder)将他们放到`GPT_SoVITS\pretrained_models`目录下

    如果想用音频超分功能缓解v3模型生成24k音频觉得闷的问题，需要下载额外的模型参数，参考[how to download](../../tools/AP_BWE_main/24kto48k/readme.txt)


## 待办事项清单

- [x] **高优先级：**

  - [x] 日语和英语的本地化。
  - [x] 用户指南。
  - [x] 日语和英语数据集微调训练。

- [ ] **功能:**
  - [x] 零样本声音转换（5 秒）/ 少样本声音转换（1 分钟）。
  - [x] TTS 语速控制。
  - [ ] ~~增强的 TTS 情感控制。~~
  - [ ] 尝试将 SoVITS 令牌输入更改为词汇的概率分布。
  - [x] 改进英语和日语文本前端。
  - [ ] 开发体积小和更大的 TTS 模型。
  - [x] Colab 脚本。
  - [x] 扩展训练数据集（从 2k 小时到 10k 小时）。
  - [x] 更好的 sovits 基础模型（增强的音频质量）。
  - [ ] 模型混合。

## （附加）命令行运行方式
使用命令行打开UVR5的WebUI
````
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
````
<!-- 如果打不开浏览器，请按照下面的格式进行UVR处理，这是使用mdxnet进行音频处理的方式
````
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision
```` -->
这是使用命令行完成数据集的音频切分的方式
````
python audio_slicer.py \
    --input_path "<path_to_original_audio_file_or_directory>" \
    --output_root "<directory_where_subdivided_audio_clips_will_be_saved>" \
    --threshold <volume_threshold> \
    --min_length <minimum_duration_of_each_subclip> \
    --min_interval <shortest_time_gap_between_adjacent_subclips>
    --hop_size <step_size_for_computing_volume_curve>
````
这是使用命令行完成数据集ASR处理的方式（仅限中文）
````
python tools/asr/funasr_asr.py -i <input> -o <output>
````
通过Faster_Whisper进行ASR处理（除中文之外的ASR标记）

（没有进度条，GPU性能可能会导致时间延迟）
```
python ./tools/asr/fasterwhisper_asr.py -i <input> -o <output> -l <language> -p <precision>
```
启用自定义列表保存路径

## 致谢

特别感谢以下项目和贡献者：

### 理论研究
- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)
### 预训练模型
- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
### 推理用文本前端
- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)
### WebUI 工具
- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

感谢 @Naozumi520 提供粤语训练集，并在粤语相关知识方面给予指导。

## 感谢所有贡献者的努力

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
