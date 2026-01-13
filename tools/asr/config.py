def get_models():
    model_size_list = [
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        #"distil-large-v2",
        #"distil-large-v3",
        #"distil-large-v3.5",
    ]
    return model_size_list


asr_dict = {
    "达摩 ASR (中文)": {"lang": ["zh", "yue"], "size": ["large"], "path": "funasr_asr.py", "precision": ["float32"]},
    "Faster Whisper (多语种)": {
        "lang": ["auto", "en", "ja", "ko"],
        "size": get_models(),
        "path": "fasterwhisper_asr.py",
        "precision": ["float32", "float16", "int8"],
    },
}
