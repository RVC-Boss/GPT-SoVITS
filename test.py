import requests

# API地址（本地运行）
url = "http://127.0.0.1:9880/tts"

# 请求体（对齐 api_v2.py 的 POST 定义）
payload = {
    "ref_audio_path": "Arona_Academy_In_2.ogg.wav",
    "prompt_text": "様々な授業やイベントが準備されているので、ご希望のスケジュールを選んでください！",
    "prompt_lang": "ja",
    "text": "中国大陆一共有31个省份啦！",
    "text_lang": "zh",
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 1.0,
    "text_split_method": "cut0",
    "batch_size": 1,
    "batch_threshold": 0.75,
    "split_bucket": True,
    "speed_factor": 1.0,
    "fragment_interval": 0.3,
    "seed": -1,
    "media_type": "wav",
    "streaming_mode": False,
    "parallel_infer": True,
    "repetition_penalty": 1.35,
    "sample_steps": 32,
    "super_sampling": False
}

# 发送 POST 请求
response = requests.post(url, json=payload)

# 检查返回并保存音频
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("✅ 生成成功，保存为 output.wav")
else:
    print(f"❌ 生成失败: {response.status_code}, 返回信息: {response.text}")
