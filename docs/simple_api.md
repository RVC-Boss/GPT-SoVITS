# GPT-SoVITS 简化接口教程

本项目原本已经有 `api_v2.py`，但调用时需要传很多 GPT-SoVITS 参数。新增的 `simple_api.py` 是一个中间层，目标是让前端或业务方用更简单的方式调用。

当前 MVP 推荐使用：

```http
POST /api/tts
Content-Type: multipart/form-data
```

适合你的流程：

1. 前端从视频中提取音频（或直接上传已裁剪的 3-10 秒音频）。
2. 用户人工裁剪主参考音频到 3-10 秒。
3. 前端把主参考音频、要生成的文字、可选辅助音频提交给后端。
4. 后端调用 GPT-SoVITS 生成音频并返回。

## 1. 安装依赖

在 GPT-SoVITS 使用的 Python 环境里运行：

```bash
python -m pip install -r requirements.txt
```

本中间层额外需要：

- `python-multipart`：用于接收前端上传的音频文件。
- `soundfile`：用于读取音频信息和校验参考音频时长。

这两个依赖已经写入 `requirements.txt`。

## 2. 检查配置

配置文件：

```text
simple_api.yaml
```

常用配置：

```yaml
server:
  host: 127.0.0.1
  port: 9881
  tts_config: GPT_SoVITS/configs/tts_infer.yaml

upload:
  dir: runtime/uploads
  min_ref_seconds: 3
  max_ref_seconds: 10
  max_upload_mb: 80
```

说明：

- `server.host`：后端监听地址。
- `server.port`：后端端口。
- `server.tts_config`：GPT-SoVITS 推理配置文件。
- `upload.dir`：临时上传目录。
- `upload.min_ref_seconds`：主参考音频最短秒数。
- `upload.max_ref_seconds`：主参考音频最长秒数。
- `upload.max_upload_mb`：单个上传音频最大体积。

当前默认后端地址：

```text
http://127.0.0.1:9881
```

## 3. 启动后端

进入项目目录：

```powershell
cd D:\tts\GPT-SoVITS
```

方式一：

```powershell
python simple_api.py -c simple_api.yaml
```

方式二，Windows PowerShell：

```powershell
.\go-simple-api.ps1
```

方式三，Windows 批处理：

```bat
go-simple-api.bat
```

启动成功后，后端会监听：

```text
http://127.0.0.1:9881
```

## 4. 健康检查

后端启动后，可以访问：

```text
http://127.0.0.1:9881/health
```

或者命令行测试：

```bash
curl http://127.0.0.1:9881/health
```

返回示例：

```json
{
  "status": "ok",
  "tts_config": "GPT_SoVITS/configs/tts_infer.yaml",
  "version": "v2",
  "languages": ["auto", "auto_yue", "en", "zh"]
}
```

## 5. 打开测试前端

后端启动后，推荐直接打开：

```text
http://127.0.0.1:9881/test/
```

这个页面已经挂载在后端服务里，不需要额外启动前端服务。

也可以直接打开本地文件：

```text
test_frontend/index.html
```

Windows 下也可以运行：

```powershell
.\open-test-frontend.ps1
```

测试前端里有一个“后端接口地址”输入框。

默认值：

```text
http://127.0.0.1:9881/api/tts
```

如果你修改了 `server.host` 或 `server.port`，记得同步修改页面里的接口地址。

## 6. MVP 接口

接口地址：

```http
POST /api/tts
```

请求类型：

```http
multipart/form-data
```

字段说明：

```text
text           必填。需要生成的文字。
ref_audio      必填。主参考音频（支持上传视频，前端会自动提取音频），要求 3-10 秒。
aux_ref_audio  可选。辅助参考音频，可以上传多个。
prompt_text    可选。主参考音频对应文字，可以留空。
text_lang      可选。生成文字语言，默认 zh。
prompt_lang    可选。参考音频语言，默认 zh。即使 prompt_text 为空，也需要传给 GPT-SoVITS 内部。
format         可选。返回格式，默认 wav。
emotion        可选。情绪 preset：neutral、happy、calm、sad、angry。
speed          可选。语速，对应 GPT-SoVITS 的 speed_factor。
seed           可选。随机种子，默认 -1。
```

## 7. curl 调用示例

PowerShell 示例：

```powershell
curl.exe -X POST http://127.0.0.1:9881/api/tts `
  -F "text=你好，欢迎使用这个声音。" `
  -F "ref_audio=@D:\audio\ref.wav" `
  -F "prompt_text=" `
  -F "text_lang=zh" `
  -F "prompt_lang=zh" `
  -F "emotion=neutral" `
  --output output.wav
```

如果有辅助参考音频：

```powershell
curl.exe -X POST http://127.0.0.1:9881/api/tts `
  -F "text=你好，欢迎使用这个声音。" `
  -F "ref_audio=@D:\audio\ref.wav" `
  -F "aux_ref_audio=@D:\audio\aux1.wav" `
  -F "aux_ref_audio=@D:\audio\aux2.wav" `
  -F "prompt_text=" `
  -F "text_lang=zh" `
  -F "prompt_lang=zh" `
  --output output.wav
```

## 8. 重要规则

- 主参考音频必须是 3-10 秒。
- `aux_ref_audio` 是可选项。
- `prompt_text` 可以为空，但当前主要针对 GPT-SoVITS v2。
- 如果切到 GPT-SoVITS v3/v4，空 `prompt_text` 会被中间层直接返回 400。
- 生成文字固定使用 `cut5`，也就是按照标点符号切句。
- `emotion` 目前是轻量 preset，本质是映射到采样和语速参数；更稳定的情绪控制仍然依赖带情绪的参考音频。

## 9. 测试前端使用步骤

1. 启动后端。
2. 打开 `http://127.0.0.1:9881/test/`。
3. 检查“后端接口地址”是否为：

```text
http://127.0.0.1:9881/api/tts
```

4. 填写“需要生成的文字”。
5. 上传主参考音频。
6. 可选上传辅助参考音频。
7. 可选填写参考音频文字。
8. 选择语言、情绪、语速。
9. 点击“生成音频”。
10. 页面右侧会显示返回结果，可以在线播放和下载。

## 10. 其他接口

除了 MVP 上传接口，还保留了 profile 调用接口：

```http
GET /voices
GET /speak?text=hello&voice=default
POST /speak
POST /speak/base64
POST /v1/tts
POST /admin/reload-config
POST /admin/weights
```

这些接口适合后续做固定音色 profile，不是当前 MVP 的主流程。

## 11. Base64 返回示例

```bash
curl -X POST http://127.0.0.1:9881/speak/base64 ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"hello\",\"voice\":\"default\"}"
```

返回格式：

```json
{
  "media_type": "audio/wav",
  "audio_base64": "..."
}
```

## 12. 添加固定音色 profile

如果后续要做固定音色，可以编辑 `simple_api.yaml`：

```yaml
voices:
  default:
    ref_audio_path: reference.wav
    prompt_text: exact transcript
    prompt_lang: zh
    text_lang: zh

  narrator:
    ref_audio_path: voices/narrator.wav
    prompt_text: exact transcript of narrator.wav
    prompt_lang: zh
    text_lang: zh
```

编辑后调用：

```bash
curl -X POST http://127.0.0.1:9881/admin/reload-config
```

## 13. 契约测试

这个测试使用 mock，不会加载 GPT-SoVITS 模型：

```bash
python -m unittest tests.test_simple_api_contract
```

用于确认：

- `/api/tts` 路由存在。
- 上传接口能构造正确参数。
- 主参考音频 3-10 秒校验正常。
- 空 `prompt_text` 在 v2 可用。
- v3/v4 空 `prompt_text` 会返回 400。
- 临时上传目录会被清理。
