import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import math

bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)


tokenizer = AutoTokenizer.from_pretrained(bert_path)
model = AutoModel.from_pretrained(bert_path)


def calculate_similarity(text1, text2, max_length=512):
    # 预处理文本，设置最大长度
    inputs1 = tokenizer(text1, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    inputs2 = tokenizer(text2, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    # 获取句子向量（这里是取CLS token的向量并展平为一维）
    with torch.no_grad():
        encoded_text1 = model(**inputs1)[0][:, 0, :].flatten()
        encoded_text2 = model(**inputs2)[0][:, 0, :].flatten()

    # 确保转换为numpy数组并且是一维的
    similarity = 1 - cosine(encoded_text1.cpu().numpy().flatten(), encoded_text2.cpu().numpy().flatten())

    return similarity

# 对0.8-1区间的值进行放大
def adjusted_similarity(similarity_score2, boundary=0.8):

    if similarity_score2 < boundary:
        return 0

    # 倍数
    multiple = 1/(1 - boundary)

    adjusted_score = (similarity_score2 - boundary)*multiple

    return adjusted_score


def calculate_result(t1, t2):
    # 计算并打印相似度
    similarity_score2 = calculate_similarity(t1, t2)

    # 调整相似度
    adjusted_similarity_score2 = adjusted_similarity(similarity_score2)

    return similarity_score2, adjusted_similarity_score2


