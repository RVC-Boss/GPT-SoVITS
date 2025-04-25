import os


def check_fw_local_models():
    """
    启动时检查本地是否有 Faster Whisper 模型.
    """
    model_size_list = [
        "tiny",
        "tiny.en",
        "base",
        "base.en",
        "small",
        "small.en",
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
    ]
    for i, size in enumerate(model_size_list):
        if os.path.exists(f"tools/asr/models/faster-whisper-{size}"):
            model_size_list[i] = size + "-local"
    return model_size_list


asr_dict = {
    "达摩 ASR (中文)": {"lang": ["zh", "yue"], "size": ["large"], "path": "funasr_asr.py", "precision": ["float32"]},
    "Faster Whisper (多语种)": {
        "lang": ["auto", "zh", "en", "ja", "ko", "yue"],
        "size": check_fw_local_models(),
        "path": "fasterwhisper_asr.py",
        "precision": ["float32", "float16", "int8"],
    },
}
