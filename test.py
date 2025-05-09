import requests

# API地址（本地运行时）
url = "http://127.0.0.1:9880"

# 请求体
payload = {
    "refer_wav_path": "Arona_Academy_In_2.ogg.wav",         # 替换为你的参考音频路径
    "prompt_text": "様々な授業やイベントが準備されているので、ご希望のスケジュールを選んでください！",            # 参考音频中的文字
    "prompt_language": "ja",             # 语言
    "text": "你好。你好。你好。你好。你好。你好。你好。你好。你好。你好。你好。",
    "text_language": "zh"
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 检查返回
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("生成成功，保存为 output.wav")
else:
    print(f"生成失败: {response.status_code}, 返回信息: {response.text}")
