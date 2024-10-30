import os
import re

pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)

speaker_verification_models = {
    'speech_campplus_sv_zh-cn_16k-common': {
        'task': 'speaker-verification',
        'model': 'Ref_Audio_Selector/tool/speaker_verification/models/speech_campplus_sv_zh-cn_16k-common',
        'model_revision': 'v1.0.0'
    },
    'speech_eres2net_sv_zh-cn_16k-common': {
        'task': 'speaker-verification',
        'model': 'Ref_Audio_Selector/tool/speaker_verification/models/speech_eres2net_sv_zh-cn_16k-common',
        'model_revision': 'v1.0.5'
    }
}

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def get_gpt_model_names():
    gpt_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): gpt_names.append("%s/%s" % (GPT_weight_root, name))
    sorted(gpt_names, key=custom_sort_key)
    return gpt_names


def get_sovits_model_names():
    sovits_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): sovits_names.append("%s/%s" % (SoVITS_weight_root, name))
    sorted(sovits_names, key=custom_sort_key)
    return sovits_names

