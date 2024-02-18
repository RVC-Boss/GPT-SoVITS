# Download moda ASR related models
from modelscope import snapshot_download
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',revision="v2.0.4")
