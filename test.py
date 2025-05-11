import requests

# API地址（本地运行）
url = "http://127.0.0.1:9880/tts"

# 请求体（对齐 api_v2.py 的 POST 定义）
payload = {
    "ref_audio_path": r"C:\Users\bdxly\Desktop\GPT-SoVITS\Arona_Academy_In_2.ogg.wav",
    "prompt_text": "様々な授業やイベントが準備されているので、ご希望のスケジュールを選んでください！",
    "prompt_lang": "ja",
    "text": "这是我的失误。我的选择，和因它发生的这一切。 直到最后，迎来了这样的结局，我才明白您是对的。 …我知道，事到如今再来说这些，挺厚脸皮的。但还是拜托您了。老师。 我想，您一定会忘记我说的这些话，不过…没关系。因为就算您什么都不记得了，在相同的情况下，应该还是会做那样的选择吧…… 所以重要的不是经历，是选择。 很多很多，只有您才能做出的选择。 我们以前聊过……关于负责人之人的话题吧。 我当时不懂……但是现在，我能理解了。 身为大人的责任与义务。以及在其延长线上的，您所做出的选择。 甚至还有，您做出选择时的那份心情。…… 所以，老师。 您是我唯一可以信任的大人，我相信您一定能找到，通往与这条扭曲的终点截然不同的……另一个结局的正确选项。所以，老师，请您一定要",
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
    print(" 生成成功，保存为 output.wav")
else:
    print(f" 生成失败: {response.status_code}, 返回信息: {response.text}")

