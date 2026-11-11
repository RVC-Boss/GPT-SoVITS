def get_models():
    model_size_list = [
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    ]
    return model_size_list


asr_dict = {
    "达摩 ASR (中文经典)": {
        "lang": ["zh"],
        "size": ["large"],
        "path": "funasr_asr.py",
        "backend": "damo",
        "precision": ["float32"],
    },
    "Fun-ASR-Nano（速度更慢，更准）": {
        "lang": ["zh", "en", "ja", "ko", "yue", "auto"],
        "size": ["large"],
        "path": "funasr_asr.py",
        "backend": "fun-asr-nano",
        "precision": ["float32"],
    },
    "Faster Whisper (多语种)": {
        "lang": ["auto", "en", "ja", "ko"],
        "size": get_models(),
        "path": "fasterwhisper_asr.py",
        "precision": ["float32", "float16", "int8"],
    },
}
