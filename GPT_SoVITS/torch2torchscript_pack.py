from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from export_torch_script_v3 import MyBertModel, build_phone_level_feature

bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
model = AutoModelForMaskedLM.from_pretrained(bert_path, output_hidden_states=True)

# 构建包装模型
wrapped_model = MyBertModel(model)

# 准备示例输入
text = "这是一条用于导出TorchScript的示例文本"
encoded = tokenizer(text, return_tensors="pt")
word2ph = torch.tensor([2 if c not in "，。？！,.?" else 1 for c in text], dtype=torch.int)

# 包装成输入
example_inputs = {
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"],
    "token_type_ids": encoded["token_type_ids"],
    "word2ph": word2ph
}

# Trace 模型并保存
traced = torch.jit.trace(wrapped_model, example_kwarg_inputs=example_inputs)
traced.save("pretrained_models/bert_script.pt")
print("✅ BERT TorchScript 模型导出完成！")
