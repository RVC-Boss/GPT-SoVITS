# Download moda ASR related models
from modelscope import snapshot_download
import shutil
import os
import time
from modelscope.hub.snapshot_download import snapshot_download

def safe_download(model_id, revision='v2.0.4', max_retries=3):
    cache_base = os.path.expanduser(f'~/.cache/modelscope/hub/{model_id.replace("/", "/")}')
    if os.path.exists(cache_base):
        shutil.rmtree(cache_base)  # Remove possibly corrupted model cache

    for attempt in range(max_retries):
        try:
            print(f"Attempting download for: {model_id} (Attempt {attempt + 1})")
            model_dir = snapshot_download(model_id, revision=revision)
            print(f"✅ Downloaded {model_id} to {model_dir}")
            return model_dir
        except Exception as e:
            print(f"❌ Failed on attempt {attempt + 1} for {model_id}: {e}")
            time.sleep(5)
    raise RuntimeError(f"❌ All attempts failed for {model_id}")

# Model downloads
safe_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
safe_download('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch')
safe_download('damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch')
