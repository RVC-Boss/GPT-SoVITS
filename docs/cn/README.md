<div align="center">

<h1>GPT-SoVITS-WebUI</h1>
强大的少样本语音转换与语音合成Web用户界面。<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/GPT-SoVITS/tree/main)

[**English**](../../README.md) | [**中文简体**](./README.md) | [**日本語**](../ja/README.md) | [**한국어**](../ko/README.md)

</div>

---

## 功能：

1. **零样本文本到语音（TTS）：** 输入 5 秒的声音样本，即刻体验文本到语音转换。

2. **少样本 TTS：** 仅需 1 分钟的训练数据即可微调模型，提升声音相似度和真实感。

3. **跨语言支持：** 支持与训练数据集不同语言的推理，目前支持英语、日语和中文。

4. **WebUI 工具：** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注，协助初学者创建训练数据集和 GPT/SoVITS 模型。

**查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**

未见过的说话者 few-shot 微调演示：

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**用户手册: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## 安装

中国地区用户可[点击此处](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official)使用 AutoDL 云端镜像进行体验。

### 测试通过的环境

- Python 3.9、PyTorch 2.0.1 和 CUDA 11
- Python 3.10.13, PyTorch 2.1.2 和 CUDA 12.3
- Python 3.9、Pytorch 2.3.0.dev20240122 和 macOS 14.3（Apple 芯片）

_注意: numba==0.56.4 需要 python<3.11_

### Windows

如果你是 Windows 用户（已在 win>=10 上测试），可以直接下载[预打包文件](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-beta.7z?download=true)，解压后双击 go-webui.bat 即可启动 GPT-SoVITS-WebUI。

### Linux

#### Step 1:下载 GPT-SoVITS 源代码

请通过本项目首页通过HTTP或SSH或下载ZIP压缩包的方式完整下载本项目

#### Step 2:安装 conda

可以根据 conda 的[清华镜像源](https://link.zhihu.com/?target=https%3A//mirror.tuna.tsinghua.edu.cn/help/anaconda/)去进行下载

```text
wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh        #原网址

wget -c https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh        #清华的镜像源latest的版本的话就是说以后一直会更新最新的版本
```

上述命令得到的是.sh 文件，使用如下命令安装：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

具体过程不再赘述，可自行查阅

#### Step 3:安装其他

在上述 cunda 安装完成后请重启命令行界面。
这里要求先开启 cunda 环境，以免造成 GPT-SoVITS 的配置影响其他软件运行

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits
```

此时你的命令行前应该会出现`(GPTSoVits)`的标志

然后请进入你之前下载好的 GPT-SoVITS 文件夹内，如果此时使用`ls`命令，你可以在里面找到两个文件：`install.sh`和`requirements.txt`
此时运行指令，等待安装完成即可：

```bash
bash install.sh
```

（另：好像用`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt`也可以安装，这里的回忆缺失了.jpg😭）
参考教程：[MAC 教程 (yuque.com)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/znoph9dtetg437xb)
（对，我是看着 MAC 的教程安的）

下次再启动，只需要打开终端，定位到项目目录，进入 conda 环境，运行启动conda 环境即可

```bash
cd /XXX/GPT-SoVITS

conda activate GPTSoVits
```

#### Step 4:推理

> 在/GPT-SoVITS-main/路径下，运行以下命令即可启动 webui 界面：
>
> ```bash
> python webui.py
> ```

当然，你没有下载预训练模型肯定会报错，我是直接把 windows 的整合包里面的东西丢到报错缺失的文件夹内的，你可以这样做：

> 从  [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS)  下载预训练模型，并将它们放置在  `GPT_SoVITS\pretrained_models`  中。
>
> 对于 UVR5（人声/伴奏分离和混响移除，附加），从  [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights)  下载模型，并将它们放置在  `tools/uvr5/uvr5_weights`  中。
> 中国地区用户可以进入以下链接并点击“下载副本”下载以上两个模型：
>
> - [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)
>
> - [UVR5 Weights](https://www.icloud.com.cn/iclouddrive/0bekRKDiJXboFhbfm3lM2fVbA#UVR5_Weights)
>
> 对于中文自动语音识别（附加），从  [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), 和  [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files)  下载模型，并将它们放置在  `tools/damo_asr/models`  中。

~~我的目的是推理，又不是训练，而且都用命令行了还要什么 ui 界面 🤪~~

如果你只希望推理，那么找到推理界面直接运行推理即可

**（以下部分为错误示范）**

- 在 GPT-SoVITS-main/GPT_SoVITS 内存有二级界面的启动.py 文件
- 推理界面的.py 文件为`inference_webui.py`
- 该文件需要依赖 GPT-SoVITS-main 文件夹下的其他内容，并且作者将其写成了相对路径

理所当然的的就把他从 GPT-SoVITS-main/GPT_SoVITS 复制到了 GPT-SoVITS-main 下面。并且使用命令成功启动：

```bash
python inference_webui.py
```

但是此时问题来了，我如果仿照 Part1 中用 curl 的方法推送并获取结果，服务器会报错：

```bash
{'detail': 'Method Not Allowed'}
```

很好，只能另寻他法。
然后我在 GPT-SoVITS-main 下翻到了 api.py

**（以下为正确做法）**

这个时候就简单了，直接启动远程端口：

```bash
python api.py
```

这里的端口号是 9880，使用<http://localhost:9880/>即可访问。

这里使用 curl 方法推送参数并解析返回值，我已经写成了 python 文件如下：

`getvoice.py`

```python
import requests
import json

# 读取文本内容
with open("/XXX/你需要转化的文本.txt", "r") as f:
    text = f.read()

# 定义请求参数
url = "http://localhost:9880/"
headers = {"Content-Type": "application/json"}
data = {
    "refer_wav_path": "/xxx/示例语音，和网页端的要求相同，建议5-10.wav,
    "prompt_text": "这是你上面示例语音的文本",
    "prompt_language": "zh",
    "text": text,
    "text_language": "zh",
}

# 发送请求并获取响应
response = requests.post(url, headers=headers, data=json.dumps(data))

# 处理结果
if response.status_code == 200:
    # 成功
    # 这里可以将音频数据保存到文件
    with open("~/output.wav", "wb") as f:
        f.write(response.content)
else:
    # 失败
    error_info = json.loads(response.content)
    print(error_info)
```
安装完成后启动总结如下：
- 在~/GPT-SoVITS-main 中先使用 conda activate GPTSoVits 启动虚拟环境

- 再使用 python api.py 启动远程端口

- 使用 python getvoice.py 读取/XXX/你需要转化的文本.txt 的内容并在~/下生成 wav 文件

### macOS

**注：在 Mac 上使用 GPU 训练的模型效果显著低于其他设备训练的模型，所以我们暂时使用CPU进行训练。**

首先确保你已通过运行 `brew install ffmpeg` 或 `conda install ffmpeg` 安装 FFmpeg，然后运行以下命令安装：

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt
```

### 手动安装

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 安装 FFmpeg

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

##### Windows 使用者

下载并将 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) 放置在 GPT-SoVITS 根目录下。

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

从 [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) 下载预训练模型，并将它们放置在 `GPT_SoVITS\pretrained_models` 中。

对于 UVR5（人声/伴奏分离和混响移除，附加），从 [UVR5 Weights](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/uvr5_weights) 下载模型，并将它们放置在 `tools/uvr5/uvr5_weights` 中。

中国地区用户可以进入以下链接并点击“下载副本”下载以上两个模型：

- [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)

- [UVR5 Weights](https://www.icloud.com.cn/iclouddrive/0bekRKDiJXboFhbfm3lM2fVbA#UVR5_Weights)

对于中文自动语音识别（附加），从 [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), 和 [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) 下载模型，并将它们放置在 `tools/damo_asr/models` 中。

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

  - [x] 日语和英语的本地化。
  - [ ] 用户指南。
  - [x] 日语和英语数据集微调训练。

- [ ] **Features:**
  - [ ] 零样本声音转换（5 秒）/ 少样本声音转换（1 分钟）。
  - [ ] TTS 语速控制。
  - [ ] 增强的 TTS 情感控制。
  - [ ] 尝试将 SoVITS 令牌输入更改为词汇的概率分布。
  - [ ] 改进英语和日语文本前端。
  - [ ] 开发体积小和更大的 TTS 模型。
  - [x] Colab 脚本。
  - [ ] 扩展训练数据集（从 2k 小时到 10k 小时）。
  - [ ] 更好的 sovits 基础模型（增强的音频质量）。
  - [ ] 模型混合。

## （可选）命令行的操作方式
使用命令行打开UVR5的WebUI
````
python tools/uvr5/webui.py "<infer_device>" <is_half> <webui_port_uvr5>
````
如果打不开浏览器，请按照下面的格式进行UVR处理，这是使用mdxnet进行音频处理的方式
````
python mdxnet.py --model --input_root --output_vocal --output_ins --agg_level --format --device --is_half_precision 
````
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
python tools/damo_asr/cmd-asr.py "<Path to the directory containing input audio files>"
````
通过Faster_Whisper进行ASR处理（除中文之外的ASR标记）

（没有进度条，GPU性能可能会导致时间延迟）
````
python ./tools/damo_asr/WhisperASR.py -i <input> -o <output> -f <file_name.list> -l <language>
````
启用自定义列表保存路径
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
