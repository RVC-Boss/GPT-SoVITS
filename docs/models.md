# 使用现有 GPT-SoVITS 模型

准备一组版本匹配的 GPT `.ckpt`、SoVITS `.pth`、共享模型，以及一段 3–10 秒的参考音频，即可通过 `api_v2.py` 推理。运行以下命令时，当前目录应为包含 `api_v2.py` 的仓库根目录。

## 角色模型与参考音频

| 资源 | 用途 | 示例路径 |
| --- | --- | --- |
| GPT `.ckpt` | 根据文本和参考语音生成语义 token | `models/voice/model.ckpt` |
| SoVITS `.pth` | 将语义 token 合成为语音；部分版本还需要外部声码器 | `models/voice/model.pth` |
| 参考 WAV | 提供音色、语气及语义提示；服务会读取并重采样 | `references/reference.wav` |
| 参考音频的准确转写 | 请求中的 `prompt_text`，内容应与参考音频一致 | 例如参考音频实际说出的“你好，欢迎回来。” |
| 要生成的文字 | 请求中的 `text` | 例如“我们开始今天的介绍。” |

优先使用模型提供者说明中配套的 `.ckpt` 和 `.pth`。参考音频建议为单人、清晰、无配乐的 WAV；服务按重采样到 16 kHz 后的长度检查 3–10 秒范围。`prompt_lang` 对应参考转写的语言，`text_lang` 对应生成文本的语言。v3、v4 和 v5 的请求必须提供非空 `prompt_text`。

保留下载文件的原始 checkpoint 格式。服务可直接加载受支持的已有模型及其兼容配置；修改文件名或 YAML 中的 `version` 不会转换模型版本。实际 SoVITS 类型由加载器根据权重标识和内容确定。

## 共享资源与模型来源

GPT、SoVITS、BERT、CNHuBERT、Pro 声纹模型和各版本声码器可从 [GPT-SoVITS 模型库](https://huggingface.co/lj1995/GPT-SoVITS/tree/main) 下载。只需下载所使用版本的资源，并保留下面的目录与文件名。

| 资源 | 默认位置 | 准备方式 |
| --- | --- | --- |
| 中文 BERT | `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/` | 下载模型库中的整个同名目录，包括权重、配置和 tokenizer 文件 |
| CNHuBERT | `GPT_SoVITS/pretrained_models/chinese-hubert-base/` | 下载整个同名目录，包括权重、配置和音频特征提取配置 |
| G2PW 中文多音字模型 | `GPT_SoVITS/text/G2PWModel/` | 下载 [G2PWModel.zip](https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip)，解压为该目录 |

BERT 与 CNHuBERT 是启动时加载的共享模型。G2PW 供 v2 及后续版本的中文前端使用，在处理中文时加载。G2PW 目录应直接包含 `config.py`、`g2pW.onnx`（也支持 `g2pw.onnx`）、`POLYPHONIC_CHARS.txt`、`MONOPHONIC_CHARS.txt` 及两个拼音映射 JSON 文件，避免解压成多套嵌套的 `G2PWModel/G2PWModel/`。

G2PW 目录不存在时，代码会尝试下载 [G2PW 1.1 资源包](https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip) 并写入该位置。离线或只读部署应预先放好完整资源；已有目录缺少文件时，应补齐该目录。

### 八个版本的默认权重

下表中的路径均相对于 `GPT_SoVITS/pretrained_models/`。它们与 `GPT_SoVITS/configs/tts_infer.yaml` 的版本块及 `TTS_Config.default_configs` 一致。

| 版本 | 默认 GPT `.ckpt` | 默认 SoVITS `.pth` | 额外资源 |
| --- | --- | --- | --- |
| v1 | `s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt` | `s2G488k.pth` | 共享模型 |
| v2 | `gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt` | `gsv-v2final-pretrained/s2G2333k.pth` | 共享模型 |
| v2Pro | `s1v3.ckpt` | `v2Pro/s2Gv2Pro.pth` | Pro 声纹模型 |
| v2ProPlus | `s1v3.ckpt` | `v2Pro/s2Gv2ProPlus.pth` | Pro 声纹模型 |
| v3 | `s1v3.ckpt` | `s2Gv3.pth` | BigVGAN 声码器 |
| v4 | `s1v3.ckpt` | `gsv-v4-pretrained/s2Gv4.pth` | v4 声码器 |
| v5dev | `s1v3.ckpt` | `gsv-v5-pretrained/s2Gv5dev.pth` | v5 声码器 |
| v5turbo | `s1v3.ckpt` | `gsv-v5-pretrained/s2Gv5turbo.pth` | v5 声码器 |

所有版本都使用前述 BERT 和 CNHuBERT；使用中文时再按对应前端准备 G2PW。v1 的语言集合为中、英、日及现有混合模式；其他版本还支持粤语、韩语及相应混合模式。

### 声码器、Pro 声纹与超采样

```text
GPT_SoVITS/pretrained_models/
├── models--nvidia--bigvgan_v2_24khz_100band_256x/   # v3
│   ├── config.json
│   └── bigvgan_generator.pt
├── gsv-v4-pretrained/
│   ├── s2Gv4.pth
│   └── vocoder.pth                              # v4
├── gsv-v5-pretrained/
│   ├── s2Gv5dev.pth
│   ├── s2Gv5turbo.pth
│   └── vocoder.pth                              # v5dev/v5turbo 共用
└── sv/
    └── pretrained_eres2netv2w24s4ep4.ckpt         # v2Pro/v2ProPlus

tools/AP_BWE_main/24kto48k/
├── config.json
└── g_24kto48k.zip                               # 可选的 v3 超采样
```

v3 的 BigVGAN 资源可取模型库中的同名目录，或取 [NVIDIA BigVGAN 模型](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main) 中的 `config.json` 和 `bigvgan_generator.pt`，直接放入上述本地目录。运行时读取这两个本地文件，目录名不表示需要额外的 Hugging Face 缓存层级。v3 默认输出 24 kHz；v4 与两个 v5 版本的声码器输出 48 kHz。

v3 的可选超采样使用 [AP-BWE](https://github.com/yxlu-0102/AP-BWE) 提供的 [预训练资源](https://drive.google.com/drive/folders/1IIYTf2zbJWzelu4IftKD6ooHloJ8mnZF?usp=share_link)。将 `g_24kto48k.zip` 与对应 `config.json` 放在同一目录；这里的 ZIP 文件由加载器直接读取，不需解压为其他权重文件。请求设置 `super_sampling: true` 后才会加载，当前该选项用于 v3 的 24→48 kHz 输出。缺少超采样权重时，服务记录提示并保留原采样率输出。

## 可复制的 v4 CPU 配置

[examples/tts_v4_cpu.yaml](../examples/tts_v4_cpu.yaml) 已包含 CPU、完整精度、共享资源默认路径，以及独立的 v4 基座配置。将已有模型放到 `models/voice/model.ckpt` 和 `models/voice/model.pth`，或修改示例中 `custom` 的这两个路径。

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force models/voice | Out-Null
Copy-Item examples/tts_v4_cpu.yaml GPT_SoVITS/configs/tts_local.yaml
python api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 127.0.0.1 -p 9880
```

Linux：

```bash
mkdir -p models/voice
cp examples/tts_v4_cpu.yaml GPT_SoVITS/configs/tts_local.yaml
python api_v2.py -c GPT_SoVITS/configs/tts_local.yaml -a 127.0.0.1 -p 9880
```

启动前还需准备共享模型、G2PW 与 `gsv-v4-pretrained/vocoder.pth`；若使用 v4 LoRA，还需下一节说明的完整 v4 基座。CPU 使用 `device: cpu` 和 `is_half: false`，运行速度取决于模型与机器；CUDA 环境可将这两项改为 `cuda` 和 `true`。

请求中的模型路径和音频路径属于服务所在机器。参考音频可放在 `references/reference.wav`，并在 `/tts` 中提供：

```json
{
  "text": "我们开始今天的介绍。",
  "text_lang": "zh",
  "ref_audio_path": "references/reference.wav",
  "prompt_text": "你好，欢迎回来。",
  "prompt_lang": "zh",
  "media_type": "wav",
  "streaming_mode": false
}
```

将 `prompt_text` 改为该 WAV 的准确转写。完整请求方法与流式参数见 [API 文档](api.md)。

### `custom` 与 `v4` 块的区别

`custom` 决定启动时实际使用的设备、GPT、SoVITS 和共享模型。其他版本块提供相应版本的默认路径；它们不会自动成为当前配置。仅编辑 `v4` 块而保留原来的 `custom`，不会将服务切到 v4。

v4 LoRA `.pth` 需要与其配套的完整 v4 基座。加载器先读取 `v4.vits_weights_path` 指向的基座，再合并 `custom.vits_weights_path` 指向的 LoRA 权重。示例中：

- `custom.vits_weights_path: models/voice/model.pth` 是准备使用的角色模型。
- `v4.vits_weights_path: GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth` 是完整基座。

两者应分别指向角色 LoRA 和其基座。使用完整 v4 权重时，加载器直接加载该完整权重。声码器仍需单独准备。

如果基座放在其他目录，修改示例的 `v4.vits_weights_path`，并保留 `v4` 块的其他字段。顶层同名版本块会整体覆盖内置版本块，不按字段递归合并；完整块也用于缺失路径时的默认值解析。

### 直接使用仓库的默认模型

如果使用上表的默认模型，可从仓库 YAML 复制对应版本块到 `custom`。以下命令将默认 v4 配置写到独立文件，不加载权重：

```bash
python -c "from pathlib import Path; import yaml; d=yaml.safe_load(Path('GPT_SoVITS/configs/tts_infer.yaml').read_text(encoding='utf-8')); d['custom']=dict(d['v4']); d['custom'].update(device='cpu', is_half=False); p=Path('models/voice/tts-default-v4.yaml'); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding='utf-8')"
python api_v2.py -c models/voice/tts-default-v4.yaml
```

选择其他版本时，改为复制该版本块，并下载匹配权重及额外资源；修改 `version` 字段本身不会转换 `.ckpt` 或 `.pth`。

## 路径规则与 G2PW tokenizer

| 位置 | 路径规则 |
| --- | --- |
| `-c/--tts_config` | 可使用相对路径或绝对路径 |
| `custom` 中的 `t2s_weights_path`、`vits_weights_path`、`bert_base_path`、`cnhuhbert_base_path` | 可使用相对路径或绝对路径；BERT/CNHuBERT 指向完整模型目录 |
| `v4.vits_weights_path` 等版本默认路径 | 可使用相对路径或绝对路径，LoRA 用它查找基座 |
| `/tts` 的参考音频、辅助参考音频与 `/set_*` 路径 | 指向服务进程可访问的本地文件，可使用相对路径或绝对路径 |
| G2PW 模型目录、BigVGAN、v4/v5 声码器、Pro 声纹模型 | 当前按上面的项目固定布局查找，YAML 没有单独的重定位字段 |
| AP-BWE 超采样 | 固定在 `tools/AP_BWE_main/24kto48k/`，相对于该工具源码位置 |

相对路径以服务的工作目录为基准，不以 YAML 文件所在目录为基准。Windows 的绝对路径建议在 YAML 中使用正斜杠或单引号，避免反斜杠被双引号当作转义字符。

G2PW tokenizer 与 TTS 的 BERT 模型分别加载。`custom.bert_base_path` 只设置 TTS 的 BERT 路径；G2PW 读取小写环境变量 `bert_path`，未设置时仍使用 `GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`。如果将 BERT 迁移到 `models/shared/chinese-roberta-wwm-ext-large/`，应同时修改 YAML 的 `bert_base_path` 并在启动前设置：

```powershell
$env:bert_path = (Resolve-Path models/shared/chinese-roberta-wwm-ext-large).Path
python api_v2.py -c GPT_SoVITS/configs/tts_local.yaml
```

```bash
export bert_path="$PWD/models/shared/chinese-roberta-wwm-ext-large"
python api_v2.py -c GPT_SoVITS/configs/tts_local.yaml
```

这个环境变量只重定位 G2PW tokenizer 的来源；G2PW 的 ONNX 模型和字典仍放在 `GPT_SoVITS/text/G2PWModel/`。

## Limitations / Known Issues

`TTS_Config` 发现配置中的 GPT、SoVITS、BERT 或 CNHuBERT 路径不存在时，会尝试回退到该版本块的默认路径。因此，角色路径拼错时可能启动了默认模型，而不是立即报错。启动后应检查 `fall back to default ...` 提示及 `Loading Text2Semantic weights`、`Loading VITS weights`、`Loading BERT weights`、`Loading CNHuBERT weights` 中的实际路径。LoRA 基座的真实路径应检查对应版本块的 vits_weights_path；当前预训练加载日志显示的是传入的角色权重路径，不能据此确认基座路径。缺失配置文件本身也会回退默认配置，应确认传入的 YAML 文件真实存在。

本仓库不附带模型权重。完整离线部署还需提前准备相应语言资源；安装和容器挂载方法见 [部署文档](deployment.md)。
