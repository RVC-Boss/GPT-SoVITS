# GPT-SoVITS 本地环境修复总结（uv）

本文档记录在本机（Arch / CUDA 13.3 / RTX 4060 Laptop / Python via uv）搭建
`GPT-SoVITS` 时踩到的问题、已合入代码/依赖的修复，以及一键重建环境的方法。

相关脚本：

| 文件 | 作用 |
|------|------|
| `setup_uv.sh` | 用 uv 创建 venv、装依赖、布局模型 |
| `scripts/setup_sitecustomize.py` | 写入 sitecustomize，预加载 CUDA12 NPP |
| `scripts/link_npp_for_torchcodec.py` | 把 NPP `.so.12` 链到 torchcodec 目录 |
| `requirements.txt` | 已固化兼容版本与额外依赖 |

一键重建：

```bash
cd /path/to/GPT-SoVITS   # 本仓库 root
bash setup_uv.sh --device CU128
source .venv/bin/activate
python webui.py zh_CN
```

可选参数见 `bash setup_uv.sh --help`。

---

## 1. 环境与依赖（可复用，已写入 requirements / setup 脚本）

### 1.1 工具链

- 用 **uv** 建 Python **3.10** venv（项目官方测过 3.10–3.12）
- 系统需：`ffmpeg`、`unzip`、`cmake`/`gcc`（编译 opencc 等时用到）
- GPU 包：`torch` / `torchcodec` / `torchaudio` 均从官方 cu128（或 cu126）index 安装，**三者 CUDA 标签必须一致**

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install torch torchcodec --index-url https://download.pytorch.org/whl/cu128
uv pip install torchaudio --index-url https://download.pytorch.org/whl/cu128 --reinstall
uv pip install -r extra-req.txt --no-deps
uv pip install -r requirements.txt
```

### 1.2 `transformers` 上限过旧 → Fun-ASR-Nano 无法加载 Qwen3

**现象**

```
ValueError: The checkpoint you are trying to load has model type `qwen3`
but Transformers does not recognize this architecture.
```

**原因**

- Fun-ASR-Nano（`FunAudioLLM/Fun-ASR-Nano-2512`）内部 LLM 是 Qwen3-0.6B
- 原 `requirements.txt` 为 `transformers>=4.43,<=4.50`，4.50 尚无 `qwen3`

**修复（已写入 requirements.txt）**

```
transformers>=4.51,<4.53
```

本机验证：`transformers==4.52.4`，`CONFIG_MAPPING` 含 `qwen3`，Fun-ASR-Nano 可加载。

### 1.3 Gradio WebUI 空白 / `TypeError: unhashable type: 'dict'`

**现象**

访问 `http://0.0.0.0:9874/` 时 ASGI 报错：

```
File ".../jinja2/utils.py", line 515, in __getitem__
    rv = self._mapping[key]
TypeError: unhashable type: 'dict'
```

**原因**

- `requirements.txt` 只有 `fastapi[standard]>=0.115.2`，无上界
- uv 解析到了 `fastapi 0.139` + `starlette 1.3`
- Starlette 新 API：`TemplateResponse(request, name, context=...)`
- Gradio 4.44 仍调用旧 API：`TemplateResponse(name, {"request": request, ...})`
- 模板名位置被塞进了 dict，Jinja2 缓存 key 不可哈希

**修复（已写入 requirements.txt）**

```
fastapi[standard]>=0.115.2,<0.116
starlette>=0.37.2,<0.39
```

本机验证：`fastapi==0.115.2` + `starlette==0.38.6`，首页 HTTP 200。

### 1.4 `torchaudio.load` 失败：缺 `libnppicc.so.12`

**现象**（`2-get-sv.py` 等）

```
OSError: libnppicc.so.12: cannot open shared object file
RuntimeError: Could not load libtorchcodec
```

**原因**

- torchaudio 2.11 默认走 **torchcodec** 解码
- torchcodec 依赖 **CUDA 12** 的 NPP 库（soname `.so.12`）
- 系统是 CUDA 13，只有 `libnppicc.so.13`

**修复（可复用，已写入 requirements + setup 脚本）**

1. 安装 `nvidia-npp-cu12`（已加入 `requirements.txt`）
2. 把 `nvidia/npp/lib/libnpp*.so.12` **符号链接**到 `torchcodec` 包目录  
   （`scripts/link_npp_for_torchcodec.py`）
3. 写入 `sitecustomize.py` 预加载 NPP，并在 `.venv/bin/activate` 追加 `LD_LIBRARY_PATH`  
   （`scripts/setup_sitecustomize.py`）

验证：

```python
import torchaudio
w, sr = torchaudio.load("some.wav")  # should succeed
```

---

## 2. 代码修复（上游逻辑漏洞，已改仓库文件）

### 2.1 ASR backend 未从 WebUI 传到 CLI

**现象**

在 WebUI 选「达摩 ASR (中文经典)」仍加载 Fun-ASR-Nano，并触发 Qwen3/transformers 错误。

**原因**

上游 PR 给 `create_model(..., backend=...)` 加了分支，但：

1. `tools/asr/config.py` 的 `asr_dict` **没有** `backend` 字段  
2. `tools/asr/funasr_asr.py` CLI **没有** `-b/--backend`  
3. `webui.py` `open_asr` **没有**把 backend 拼进命令  
4. 默认 dropdown value 还是旧名字 `"达摩 ASR (中文)"`，与新 key `"达摩 ASR (中文经典)"` 对不上  

于是 CLI 永远 default=`fun-asr-nano`。

**修复文件**

- `tools/asr/config.py`：为 Fun-ASR-Nano / SenseVoice / 达摩 增加 `backend`
- `tools/asr/funasr_asr.py`：增加 `-b/--backend` 并传入 `execute_asr`
- `webui.py`：
  - `open_asr` 拼接 `-b ...`
  - 默认模型改为 `"达摩 ASR (中文经典)"`

对应 backend：

| WebUI 选项 | backend |
|------------|---------|
| Fun-ASR-Nano (31语种+方言, 推荐) | `fun-asr-nano` |
| SenseVoice (极速, 5语种) | `sensevoice` |
| 达摩 ASR (中文经典) | `paraformer`（本地 Paraformer+VAD+Punc） |

中文数据建议默认用 **达摩 ASR**（离线、已预装权重）。

---

## 3. 模型文件布局（安装/摆放，非代码）

README 约定的路径：

| 内容 | 路径 |
|------|------|
| 预训练底模 | `GPT_SoVITS/pretrained_models/` |
| G2PW（中文多音字） | `GPT_SoVITS/text/G2PWModel/` |
| UVR5 人声分离 | `tools/uvr5/uvr5_weights/` |
| FunASR / Faster-Whisper | `tools/asr/models/` |

本机实际做法：

1. **G2PW**：解压根目录 `G2PWModel.zip` → `GPT_SoVITS/text/`
2. **UVR5**：解压根目录 `uvr5_weights.zip` → `tools/uvr5/`
3. **ASR**：把 `aaa/` 下 FunASR + faster-whisper 拷到 `tools/asr/models/`  
   （注意 docker 下载出来可能是 root 权限，需 `chown`）
4. **pretrained_models**：
   - 旁边 `../GPT-SoVITS/pretrained_models.zip` 当时 **损坏**（缺 EOCD）
   - 从正在跑的 Docker 镜像内路径  
     `/workspace/models/pretrained_models/`  
     `docker cp` 到本地 `GPT_SoVITS/pretrained_models/`
   - `setup_uv.sh` 会按：容器 → 旁目录 → zip 的顺序尝试

校验清单：

```text
GPT_SoVITS/pretrained_models/s2Gv3.pth
GPT_SoVITS/pretrained_models/s1v3.ckpt
GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/
GPT_SoVITS/pretrained_models/v2Pro/
GPT_SoVITS/pretrained_models/chinese-hubert-base/
GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
GPT_SoVITS/pretrained_models/sv/
GPT_SoVITS/text/G2PWModel/
tools/uvr5/uvr5_weights/*.pth
tools/asr/models/speech_paraformer-.../model.pt
tools/asr/models/speech_fsmn_vad_.../model.pt
tools/asr/models/punc_ct-transformer_.../model.pt
tools/asr/models/faster-whisper-large-v3/model.bin
```

可选运行时数据（日/英 g2p）：

- `$VIRTUAL_ENV/nltk_data`
- `site-packages/pyopenjtalk/open_jtalk_dic_utf_8-1.11`  
  （`setup_uv.sh` 会尝试从 hf-mirror 拉取）

---

## 4. 问题速查

| 症状 | 处理 |
|------|------|
| WebUI 打开报 `unhashable type: 'dict'` | 固定 fastapi&lt;0.116、starlette&lt;0.39，重启 webui |
| Fun-ASR-Nano 报 `model type qwen3` | `transformers>=4.51`；或直接用「达摩 ASR」 |
| 选达摩却仍加载 Nano | 更新后的 `config.py`/`funasr_asr.py`/`webui.py`，重启 webui |
| `2-get-sv` / `torchaudio.load` 缺 `libnppicc.so.12` | 装 `nvidia-npp-cu12` + 跑两个 scripts hook |
| `torchaudio` 与 `torch` CUDA 不一致 | 用同一 index 重装 `torchaudio`（`--reinstall`） |
| pretrained zip 解压失败 | 检查 zip 完整性；或从官方 Docker 镜像/HF 重新下 |
| ASR 模型目录几乎空、权限 denied | `aaa/` 是 root 文件时 `sudo chown -R $USER aaa` 再拷 |

---

## 5. 建议启动流程

```bash
cd /home/ogios/work/voice/local
source .venv/bin/activate   # 会带上 NPP LD_LIBRARY_PATH
python webui.py zh_CN
```

或：

```bash
uv run webui.py zh_CN
```

数据处理推荐顺序：

1. UVR5 人声分离（可选）
2. 切片 slicer
3. ASR：**达摩 ASR (中文经典)** + `zh`
4. 1A 文本 / 1B Hubert+SV / 1C semantic（或一键三连）
5. 微调 / 推理

---

## 6. 与官方 install.sh 的差异

官方 `install.sh` 依赖 **conda**。本机方案等价物：

| install.sh | 本方案 |
|------------|--------|
| conda env + pip | uv venv + uv pip |
| 自动 wget 模型 zip | `setup_uv.sh` 优先用本地 zip/`aaa`/docker |
| conda ffmpeg | 系统 ffmpeg |
| 无 fastapi 上界 | 显式 pin 兼容 Gradio 4 |
| 无 npp-cu12 | 显式安装 + hook |

代码侧 ASR backend 接线是对上游遗漏的补丁，升级上游时注意是否已合并，避免冲突。

---

## 7. 本机已验证组合（2026-07-24）

```
Python          3.10.20 (uv)
torch           2.11.0+cu128
torchaudio      2.11.0+cu128
torchcodec      0.11.1+cu128
transformers    4.52.4
fastapi         0.115.2
starlette       0.38.6
gradio          4.44.1
funasr          1.3.26
nvidia-npp-cu12 12.4.1.87
GPU             NVIDIA GeForce RTX 4060 Laptop GPU
```

验证过：

- WebUI 首页 200
- Paraformer ASR 转写成功
- Fun-ASR-Nano 模型加载成功
- `2-get-sv` 写出 `logs/*/7-sv_cn/*.pt`
