# 简化 TTS 接口快速启动

完整教程见：`docs/simple_api.md`

## 一句话流程

启动后端，打开测试页，上传 3-10 秒参考音频或视频（视频自动提取音频），填写生成文字，点击生成。

## 1. 启动后端

```powershell
cd D:\tts\GPT-SoVITS
python -m pip install -r requirements.txt
.\go-simple-api.ps1
```

也可以运行：

```bat
go-simple-api.bat
```

默认后端地址：

```text
http://127.0.0.1:9881
```

## 2. 打开页面

| 页面 | 地址 | 说明 |
|------|------|------|
| 测试前端 | http://127.0.0.1:9881/test/ | 带波形裁剪的测试 UI |
| Swagger UI | http://127.0.0.1:9881/docs | 交互式 API 文档 |
| ReDoc | http://127.0.0.1:9881/redoc | 可读式 API 文档 |

## 3. 调用接口

接口：

```http
POST /api/tts
Content-Type: multipart/form-data
```

最小字段：

```text
text       要生成的文字
ref_audio  3-10 秒主参考音频（或视频，前端自动提取）
```

常用可选字段：

```text
aux_ref_audio  辅助参考音频，可多个
prompt_text    参考音频文本，可留空
text_lang      默认 zh
prompt_lang    默认 zh
emotion        neutral / happy / calm / sad / angry
speed          语速，默认 1
seed           默认 -1
```

## 4. PowerShell 示例

```powershell
curl.exe -X POST http://127.0.0.1:9881/api/tts `
  -F "text=你好，欢迎使用这个声音。" `
  -F "ref_audio=@D:\audio\ref.wav" `
  -F "prompt_text=" `
  -F "text_lang=zh" `
  -F "prompt_lang=zh" `
  --output output.wav
```

## 5. 注意事项

- 主参考音频必须是 3-10 秒。
- 前端支持上传视频文件，会自动提取音频。
- 前端提供波形裁剪工具，可直接选择 3-10 秒片段。
- `prompt_text` 在当前 v2 配置下可以为空。
- 如果切换到 v3/v4，空 `prompt_text` 会返回 400。
- 生成文字固定按标点符号切句。
- 更详细的配置、profile、base64 接口见 `docs/simple_api.md`。
