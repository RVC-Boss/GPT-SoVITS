# Download moda ASR related models
from modelscope import snapshot_download
model_dir = snapshot_download('damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',revision="v2.0.4")
model_dir = snapshot_download('damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',revision="v2.0.4")

import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Download https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip unzip and rename to G2PWModel, and then place them in GPT_SoVITS/text.

import os
import zipfile
import shutil
import requests

# 获取当前文件的路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 定义下载链接和目标路径
url = 'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/g2p/G2PWModel_1.1.zip'
download_path = os.path.join(current_dir, 'G2PWModel_1.1.zip')
target_dir = os.path.join(current_dir, '../GPT_SoVITS/text/')

# 下载文件
response = requests.get(url)
with open(download_path, 'wb') as file:
    file.write(response.content)

# 解压文件
with zipfile.ZipFile(download_path, 'r') as zip_ref:
    zip_ref.extractall(current_dir)

# 重命名解压后的文件夹
os.rename(os.path.join(current_dir, 'G2PWModel_1.1'), os.path.join(current_dir, 'G2PWModel'))

# 移动文件夹到目标目录
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
shutil.move(os.path.join(current_dir, 'G2PWModel'), target_dir)

# 清理临时文件
os.remove(download_path)

print("Download G2PWModel successfully")