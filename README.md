# GPT-SoVITS 推理服务

这是从 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 裁剪得到的语音推理服务。给定一组模型权重、参考音频和待合成文字，服务生成对应音色的语音，供其他程序通过 HTTP 调用。

唯一服务入口是 `api_v2.py`，默认地址为 `http://127.0.0.1:9880`。仓库提供推理代码、配置样例、安装脚本和 Docker 部署文件；模型权重需要另行准备。训练、微调、数据集处理、ASR、音频分离和 WebUI 已移除。

- 模型版本：v1、v2、v2Pro、v2ProPlus、v3、v4、v5dev、v5turbo。
- 语言：中文、英文、日文、韩文、粤语及混合语言模式；具体取决于模型版本。
- 输出：WAV、RAW PCM、OGG、AAC；支持分段返回及指定版本的 GPT 语义流式。
- 部署：Windows / Linux，CPU 或匹配 PyTorch 的 GPU 环境。

## 对外提供哪些 API

| 方法与路径 | 用途 |
| --- | --- |
| `POST /tts` | 用 JSON 参数合成语音，响应体为音频二进制 |
| `GET /tts` | 使用查询参数执行同样的推理 |
| `GET /set_sovits_weights` | 加载指定 SoVITS `.pth` 权重 |
| `GET /set_gpt_weights` | 加载指定 GPT `.ckpt` 权重 |
| `GET /set_refer_audio` | 预加载指定参考音频 |

服务启动后可打开 `/docs` 查看交互式接口说明，或读取 `/openapi.json`。完整参数、流式约束、返回格式和调用示例见 [API 文档](docs/api.md)。

## 快速开始：用现有 v4 模型生成 WAV

以下命令均在仓库根目录执行。这个例子使用 CPU，适用于已有角色 GPT/SoVITS 权重的情况；其他版本的资源与配置见 [模型指南](docs/models.md)。

### 1. 安装到项目隔离环境

先安装 Python 3.12 和 FFmpeg，确认 `python --version` 与 `ffmpeg -version` 可运行。Windows 需使用 64 位 Python；安装脚本支持 Python 3.10–3.12，推荐 3.12。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:NLTK_DATA = Join-Path $PWD '.codex/runtime-cache/nltk_data'
New-Item -ItemType Directory -Force $env:NLTK_DATA | Out-Null
& .\install.ps1 -Device CPU
```

Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export NLTK_DATA="$PWD/.codex/runtime-cache/nltk_data"
mkdir -p "$NLTK_DATA"
bash install.sh --device CPU
```

后续命令继续使用这个已激活的环境。CUDA、ROCm、系统依赖和 Docker 的完整安装步骤见 [部署指南](docs/deployment.md)。

### 2. 准备模型和参考音频

将角色模型和参考音频放入以下位置，或在配置、请求中填写实际路径：

```text
models/voice/model.ckpt       # 角色 GPT 权重
models/voice/model.pth        # 配套 SoVITS 权重
references/reference.wav     # 3～10 秒参考音频
```

另外准备中文 RoBERTa、CNHuBERT、中文 G2PW 和 v4 声码器。**v4 LoRA 权重还需要原版 `s2Gv4.pth` 基座，不能只放角色 `.pth` 文件。**

默认公共资源布局如下；具体文件清单和下载来源见 [模型指南](docs/models.md)：

```text
GPT_SoVITS/pretrained_models/
├── chinese-roberta-wwm-ext-large/
├── chinese-hubert-base/
└── gsv-v4-pretrained/
    ├── s2Gv4.pth
    └── vocoder.pth
GPT_SoVITS/text/G2PWModel/
```

参考音频用于提供音色和说话方式；`prompt_text` 必须填写该音频实际说出的内容，`text` 才是希望模型生成的新内容。

### 3. 配置要加载的模型

复制 [v4 CPU 配置样例](examples/tts_v4_cpu.yaml)：

```powershell
# Windows
Copy-Item examples/tts_v4_cpu.yaml GPT_SoVITS/configs/tts_local.yaml
```

```bash
# Linux
cp examples/tts_v4_cpu.yaml GPT_SoVITS/configs/tts_local.yaml
```

编辑复制后的文件：

- `custom.t2s_weights_path` 指向角色 `.ckpt`。
- `custom.vits_weights_path` 指向角色 `.pth`。
- `custom.device: cpu`、`custom.is_half: false` 用于 CPU。
- `v4.vits_weights_path` 指向原版 `s2Gv4.pth`，供 LoRA 加载使用。

`custom` 决定启动时加载什么；`v1` 至 `v5turbo` 配置块提供各版本的默认资源路径。修改 `version` 字段不会转换模型格式。

### 4. 启动 HTTP 服务

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_local.yaml
```

模型在启动阶段加载。查看终端中的实际权重路径，确认加载的是目标角色；不存在的配置路径可能触发默认路径回退。看到服务启动完成后，在浏览器打开 `http://127.0.0.1:9880/docs`。

需要从其他机器访问时，将 `-a` 改为 `0.0.0.0`，客户端使用服务所在机器的 IP。端口由 `-p` 指定，配置文件由 `-c` / `--tts_config` 指定。

### 5. 发送推理请求

保持服务终端运行，另开一个终端并切到仓库根目录发送请求。

打开 [examples/tts_request.json](examples/tts_request.json)，把 `ref_audio_path` 改为参考音频路径，把 `prompt_text` 改为准确转写。示例使用中文默认模式 `all_zh`，其他语言代码见 API 文档。

Windows：

```powershell
curl.exe --fail-with-body -X POST http://127.0.0.1:9880/tts -H "Content-Type: application/json" --data-binary "@examples/tts_request.json" --output output.wav
```

Linux：

```bash
curl --fail-with-body -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' --data-binary @examples/tts_request.json --output output.wav
```

成功响应保存为 `output.wav`。所有模型和参考音频路径都由**服务端**读取：远程客户端的本地路径不能直接供服务器使用；容器部署时填写容器内路径。

### 6. 换角色或使用流式输出

切换角色时，暂停其他推理请求，依次调用 `/set_sovits_weights`、`/set_gpt_weights`，再更新参考音频。切换只影响当前进程，重启仍读取启动配置。这几个请求分别加锁，整套切换并非原子事务。

需要流式输出时，先按上述步骤确认普通 WAV 推理正常，再根据 [API 文档的流式说明](docs/api.md) 选择模式。普通流式使用 `batch_size=1`；GPT 语义流式仅适用于 v2ProPlus、v5dev、v5turbo，需要可用的 CUDA 加速后端，并有额外参数限制。

## 文档与验证

| 文档 | 内容 |
| --- | --- |
| [目录与架构](docs/architecture.md) | 目录职责、请求链路、模型加载、流式生命周期和开发入口 |
| [模型指南](docs/models.md) | 权重来源、八个版本的资源清单、LoRA、参考音频和配置 |
| [API 文档](docs/api.md) | 完整接口、参数默认值、Python/curl 调用、热切换、流式和错误处理 |
| [部署指南](docs/deployment.md) | Windows/Linux 隔离环境、GPU、Docker、离线资源和排错 |

安装好语言资源后，可执行回归测试：

```bash
python -m unittest discover -s tests -t . -p "test_*.py"
```

仅检查入口导入时使用 `python -c "import api_v2"`；这个命令不会加载 TTS 权重，不能代替真实服务启动与语音生成检查。

## Limitations / Known Issues

真实音频验证已覆盖 Windows CPU 下的两套 v4 LoRA；其他版本有检查点兼容与可控输入数值回归。Linux、Docker、GPU/CUDA/Flash Attention 尚未在本机完成实跑验证。

当前部分内部合成错误可能以静音音频返回，因此 HTTP 200 本身不能证明合成正确。普通流式的批量限制、声码器版本的分段回退，以及流开始后的错误表现详见 [API 文档](docs/api.md)。

项目许可见 [LICENSE](LICENSE)。第三方模块的许可文件保留在对应目录。
