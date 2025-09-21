import os
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from Ref_Audio_Selector.config_param.log_config import logger

bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f'使用计算设备: {device}')

tokenizer = AutoTokenizer.from_pretrained(bert_path)
model = AutoModel.from_pretrained(bert_path).to(device)


def calculate_similarity(text1, text2, max_length=512):
    # 预处理文本，设置最大长度
    inputs1 = tokenizer(text1, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    inputs2 = tokenizer(text2, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)

    # 获取句子向量（这里是取CLS token的向量并展平为一维）
    with torch.no_grad():
        encoded_text1 = model(**inputs1)[0][:, 0, :].flatten()
        encoded_text2 = model(**inputs2)[0][:, 0, :].flatten()

    # 确保转换为numpy数组并且是一维的
    similarity = 1 - cosine(encoded_text1.cpu().numpy().flatten(), encoded_text2.cpu().numpy().flatten())

    return similarity


# 对boundary到1区间的值进行放大
def adjusted_similarity(similarity_score2, boundary=0.8):
    if similarity_score2 < boundary:
        return 0

    # 倍数
    multiple = 1 / (1 - boundary)

    adjusted_score = (similarity_score2 - boundary) * multiple

    return adjusted_score


def calculate_result(t1, t2, boundary):
    # 计算并打印相似度
    similarity_score2 = calculate_similarity(t1, t2)

    # 调整相似度
    adjusted_similarity_score2 = adjusted_similarity(similarity_score2, boundary)

    return similarity_score2, adjusted_similarity_score2


def print_result(t1, t2, boundary):
    print(f't2: {t2}')
    # 计算并打印相似度
    similarity_score2 = calculate_similarity(t1, t2)
    print(f"两句话的相似度为: {similarity_score2:.4f}")

    # 调整相似度
    adjusted_similarity_score2 = adjusted_similarity(similarity_score2, boundary)
    print(f"调整后的相似度为: {adjusted_similarity_score2:.4f}")


def test(boundary):
    # 原始文本
    text1 = "这是第一个句子"
    list = """
    这是第一个句子
    这是第二个句子。
    那么，这是第三个表达。
    当前呈现的是第四个句子。
    接下来，我们有第五句话。
    在此，展示第六条陈述。
    继续下去，这是第七个短句。
    不容忽视的是第八个表述。
    顺延着序列，这是第九句。
    此处列举的是第十个说法。
    进入新的篇章，这是第十一个句子。
    下一段内容即为第十二个句子。
    显而易见，这是第十三个叙述。
    渐进地，我们来到第十四句话。
    向下滚动，您会看到第十五个表达。
    此刻，呈现在眼前的是第十六个句子。
    它们中的一个——第十七个句子在此。
    如同链条般连接，这是第十八个断言。
    按照顺序排列，接下来是第十九个话语。
    逐一列举，这是第二十个陈述句。
    结构相似，本例给出第二十一个实例句。
    这是最初的陈述句。
    首先表达的是这一个句子。
    第一句内容即为此处所示。
    这是起始的叙述段落。
    开篇所展示的第一句话就是这个。
    明媚的阳光洒满大地
    窗外飘落粉色樱花瓣
    笔尖轻触纸面思绪万千
    深夜的月光如水般静谧
    穿越丛林的小径蜿蜒曲折
    浅酌清茶品味人生百态
    破晓时分雄鸡一唱天下白
    草原上奔驰的骏马无拘无束
    秋叶纷飞描绘季节更替画卷
    寒冬雪夜炉火旁围坐共话家常
    kszdRjYXw
    pfsMgTlVHnB
    uQaGxIbWz
    ZtqNhPmKcOe
    jfyrXsStVUo
    wDiEgLkZbn
    yhNvAfUmqC
    TpKjxMrWgs
    eBzHUaFJtYd
    oQnXcVSiPkL
    00000
    """
    list2 = list.strip().split('\n')
    for item in list2:
        print_result(text1, item, boundary)


if __name__ == '__main__':
    test(0.9)
