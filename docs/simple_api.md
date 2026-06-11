# GPT-SoVITS 简化接口文档

本项目新增 `simple_api.py` 作为中间层，封装 GPT-SoVITS 推理引擎，提供更简洁的调用方式。

## 快速开始

```bash
# 安装依赖
python -m pip install -r requirements.txt

# 启动
python simple_api.py -c simple_api.yaml

# 访问
Swagger UI:  http://127.0.0.1:9881/docs
ReDoc:       http://127.0.0.1:9881/redoc
测试前端:     http://127.0.0.1:9881/test/
```

## 接口总览

| 方法 | 路径 | 说明 | 标签 |
|------|------|------|------|
| GET | `/health` | 健康检查（含 GPU 信息） | System |
| GET | `/voices` | 列出 voice profiles | System |
| **POST** | **`/api/tts`** | **核心 TTS 接口（MVP）** | **MVP** |
| GET | `/speak` | voice profile TTS (GET) | Profile |
| POST | `/speak` | voice profile TTS (POST) | Profile |
| POST | `/v1/tts` | OpenAI 兼容格式 TTS | Profile |
| POST | `/speak/base64` | 返回 Base64 音频 | Profile |
| POST | `/admin/reload-config` | 热加载配置 | Admin |
| POST | `/admin/weights` | 切换模型权重 | Admin |

---

## 1. POST /api/tts — 核心 TTS 接口

**推荐使用此接口**。上传参考音频和文字，直接返回生成的音频。

### 请求格式

```
Content-Type: multipart/form-data
```

### 字段说明

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | **是** | — | 需要生成的文字 |
| `ref_audio` | file | **是** | — | 主参考音频，3-10 秒（支持 wav/flac/ogg/mp3/m4a/aac） |
| `aux_ref_audio` | file[] | 否 | — | 辅助参考音频，可上传多个 |
| `prompt_text` | string | 否 | `""` | 主参考音频对应文字（v2 可留空；v3/v4 必填） |
| `text_lang` | string | 否 | `zh` | 生成文字语言：zh/en/ja/ko/yue/auto |
| `prompt_lang` | string | 否 | `zh` | 参考音频语言：zh/en/ja/ko/yue/auto |
| `format` | string | 否 | `wav` | 返回格式：wav/ogg/aac/raw |
| `emotion` | string | 否 | `neutral` | 情绪预设：neutral/happy/calm/sad/angry |
| `speed` | float | 否 | — | 语速（0.5-2.0），覆盖情绪预设中的语速 |
| `seed` | int | 否 | `-1` | 随机种子，-1 为随机 |

### 情绪预设参数映射

| 情绪 | temperature | top_p | top_k | speed_factor | repetition_penalty |
|------|-------------|-------|-------|--------------|-------------------|
| neutral | — | — | — | — | — |
| happy | 1.1 | 0.95 | — | — | — |
| calm | 0.8 | 0.85 | — | 0.92 | — |
| sad | 0.75 | 0.85 | — | 0.9 | — |
| angry | 1.2 | — | 20 | — | 1.25 |

> 显式传入 `speed` 会覆盖情绪预设中的 `speed_factor`。

### curl 示例

**基础调用：**

```powershell
curl.exe -X POST http://127.0.0.1:9881/api/tts `
  -F "text=你好，欢迎使用这个声音。" `
  -F "ref_audio=@D:\audio\ref.wav" `
  --output output.wav
```

**带辅助参考音频和情绪：**

```powershell
curl.exe -X POST http://127.0.0.1:9881/api/tts `
  -F "text=你好，欢迎使用这个声音。" `
  -F "ref_audio=@D:\audio\ref.wav" `
  -F "aux_ref_audio=@D:\audio\aux1.wav" `
  -F "emotion=happy" `
  -F "speed=1.1" `
  --output output.wav
```

**Linux/macOS：**

```bash
curl -X POST http://127.0.0.1:9881/api/tts \
  -F "text=你好，欢迎使用这个声音。" \
  -F "ref_audio=@/path/to/ref.wav" \
  -F "emotion=calm" \
  --output output.wav
```

### 返回

- 成功：音频二进制流（Content-Type: `audio/wav` 等）
- 失败：JSON 错误信息

```json
{"message": "tts failed", "exception": "..."}
```

### 常见错误

| HTTP 状态码 | 原因 |
|------------|------|
| 400 | text 为空 / ref_audio 缺失 / 音频时长不在 3-10 秒 / 不支持的 format / v3/v4 时 prompt_text 为空 |
| 404 | voice profile 不存在（仅 /speak 接口） |
| 503 | TTS pipeline 未就绪（模型未加载） |

---

## 2. GET /health — 健康检查

```bash
curl http://127.0.0.1:9881/health
```

返回示例：

```json
{
  "status": "ok",
  "tts_config": "GPT_SoVITS/configs/tts_infer.yaml",
  "version": "v2",
  "languages": ["auto", "en", "zh"],
  "pid": 12345,
  "memory_mb": 2048.5,
  "gpu": {
    "name": "NVIDIA GeForce RTX 3080",
    "memory_used_mb": 4096.2,
    "memory_total_mb": 10240.0
  }
}
```

---

## 3. GET /voices — 列出 voice profiles

```bash
curl http://127.0.0.1:9881/voices
```

返回示例：

```json
{
  "default_voice": "default",
  "voices": [
    {
      "name": "default",
      "description": "Replace this profile with your reference voice.",
      "text_lang": "zh",
      "prompt_lang": "zh",
      "ref_audio_path": "reference.wav",
      "ready": true
    }
  ]
}
```

---

## 4. POST /speak — voice profile TTS

基于 `simple_api.yaml` 中配置的 voice profile 调用 TTS。

### 请求体（JSON）

```json
{
  "text": "hello world",
  "voice": "default",
  "text_lang": "zh",
  "format": "wav",
  "speed": 1.0
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | **是** | 需要生成的文字 |
| `voice` | string | 否 | voice profile 名称，不传则使用 default |
| `text_lang` | string | 否 | 生成文字语言 |
| `format` | string | 否 | 返回格式 |
| `stream` | bool | 否 | 是否流式返回 |
| `speed` | float | 否 | 语速 |

### curl 示例

```bash
curl -X POST http://127.0.0.1:9881/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"你好世界","voice":"default"}' \
  --output output.wav
```

---

## 5. GET /speak — voice profile TTS (GET)

与 POST /speak 相同，但通过 URL 参数传递。

```
GET /speak?text=hello&voice=default&format=wav
```

---

## 6. POST /speak/base64 — 返回 Base64 音频

返回 Base64 编码的音频，适合 Web 前端直接使用。

```bash
curl -X POST http://127.0.0.1:9881/speak/base64 \
  -H "Content-Type: application/json" \
  -d '{"text":"hello","voice":"default"}'
```

返回：

```json
{
  "media_type": "audio/wav",
  "audio_base64": "UklGRi..."
}
```

---

## 7. POST /v1/tts — OpenAI 兼容格式

请求格式与 POST /speak 相同，路径兼容 OpenAI TTS API 风格。

---

## 8. POST /admin/reload-config — 热加载配置

重新加载 `simple_api.yaml`，无需重启服务。

```bash
curl -X POST http://127.0.0.1:9881/admin/reload-config
```

返回：`{"message": "success", "default_voice": "default"}`

---

## 9. POST /admin/weights — 切换模型权重

运行时切换 GPT-SoVITS 模型权重文件。

```bash
curl -X POST http://127.0.0.1:9881/admin/weights \
  -H "Content-Type: application/json" \
  -d '{"gpt_weights_path":"path/to/gpt.pt","sovits_weights_path":"path/to/sovits.pt"}'
```

---

## 配置文件

`simple_api.yaml`：

```yaml
server:
  host: 127.0.0.1
  port: 9881
  tts_config: GPT_SoVITS/configs/tts_infer.yaml

cors_allow_origins:
  - "*"

upload:
  dir: runtime/uploads
  min_ref_seconds: 3
  max_ref_seconds: 10
  max_upload_mb: 80

defaults:
  text_lang: zh
  prompt_lang: zh
  media_type: wav
  text_split_method: cut5
  batch_size: 1
  speed_factor: 1.0
  seed: -1

emotion_presets:
  neutral: {}
  happy:
    temperature: 1.1
    top_p: 0.95
  calm:
    temperature: 0.8
    top_p: 0.85
    speed_factor: 0.92
  sad:
    temperature: 0.75
    top_p: 0.85
    speed_factor: 0.9
  angry:
    temperature: 1.2
    top_k: 20
    repetition_penalty: 1.25

voices:
  default:
    description: Replace this profile with your reference voice.
    ref_audio_path: reference.wav
    prompt_text: Replace this with the exact text spoken in reference.wav.
    prompt_lang: zh
    text_lang: zh
```

### 配置说明

| 配置项 | 说明 |
|--------|------|
| `server.host` | 监听地址 |
| `server.port` | 监听端口 |
| `server.tts_config` | GPT-SoVITS 推理配置文件路径 |
| `upload.dir` | 临时上传目录 |
| `upload.min_ref_seconds` | 主参考音频最短秒数 |
| `upload.max_ref_seconds` | 主参考音频最长秒数 |
| `upload.max_upload_mb` | 单个上传文件最大体积 (MB) |
| `defaults.*` | 所有接口的默认参数 |
| `emotion_presets.*` | 情绪预设参数映射 |
| `voices.*` | 固定音色 profile |

---

## 添加自定义音色

编辑 `simple_api.yaml`，在 `voices` 下添加：

```yaml
voices:
  narrator:
    description: "男声旁白"
    ref_audio_path: voices/narrator.wav
    prompt_text: "旁白参考音频的逐字稿"
    prompt_lang: zh
    text_lang: zh
```

然后热加载：

```bash
curl -X POST http://127.0.0.1:9881/admin/reload-config
```

---

## 测试

### 契约测试（无需 GPU）

```bash
python -m unittest tests.test_simple_api_contract -v
```

覆盖：

- `/api/tts` 路由注册
- 上传接口参数构造
- 主参考音频 3-10 秒校验
- v2 空 prompt_text 允许 / v3/v4 空 prompt_text 拒绝
- 临时上传目录清理
- 情绪预设应用与 speed 覆盖

### 前端测试

1. 启动后端
2. 访问 `http://127.0.0.1:9881/test/`
3. 上传音频或视频（视频会自动提取音频）
4. 使用波形裁剪工具选择 3-10 秒片段
5. 填写文字，选择情绪和语速
6. 点击生成

---

## 启动脚本

| 脚本 | 平台 | 说明 |
|------|------|------|
| `go-simple-api.ps1` | Windows PowerShell | 自动检测 runtime\python.exe |
| `go-simple-api.bat` | Windows CMD | 同上 |
| `open-test-frontend.ps1` | Windows PowerShell | 直接打开测试前端 HTML |
