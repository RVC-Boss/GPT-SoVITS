from functools import lru_cache
import torch
import torch
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Tuple

@lru_cache(maxsize=1000)
def get_cached_bert(norm_text: str, word2ph_tuple: tuple, device_str: str = "cuda"):
    """
    缓存 BERT 提取函数，用于相同 norm_text 时复用特征

    Args:
        norm_text (str): 清洗后的文本（可复用）
        word2ph_tuple (tuple): word2ph 列表转换成 tuple（因为 lru_cache 不支持 list）
        device_str (str): 设备信息，用于转移到正确设备上

    Returns:
        Tensor: 形状 [hidden_dim, total_phonemes]
    """
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    # 如果你在类里，可以改成 self.tokenizer 和 self.model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese", output_hidden_states=True).eval().to(device_str)

    inputs = tokenizer(norm_text, return_tensors="pt").to(device_str)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = torch.cat(outputs.hidden_states[-3:-2], dim=-1)[0][1:-1]  # 去掉 CLS/SEP
        word2ph = torch.tensor(list(word2ph_tuple), device=hidden.device)
        indices = torch.repeat_interleave(torch.arange(len(word2ph), device=hidden.device), word2ph)
        phone_level_feature = hidden[indices]
    return phone_level_feature.T.cpu()





class CachedBertExtractor:
    def __init__(self, model_name_or_path: str = "bert-base-chinese", device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.device = device
        self.bert_model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, output_hidden_states=True
        ).eval().to(device)

    def get_bert_feature(self, norm_text: str, word2ph: List[int]) -> torch.Tensor:
        """
        Public method: gets cached BERT feature tensor
        """
        word2ph_tuple = tuple(word2ph)
        return self._cached_bert(norm_text, word2ph_tuple).to(self.device)

    @lru_cache(maxsize=1024)
    def _cached_bert(self, norm_text: str, word2ph_tuple: Tuple[int, ...]) -> torch.Tensor:
        """
        Cached private method: returns CPU tensor (for lru_cache compatibility)
        """
        inputs = self.tokenizer(norm_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            hidden = torch.cat(outputs.hidden_states[-3:-2], dim=-1)[0][1:-1]  # shape: [seq_len-2, hidden_dim]

        word2ph_tensor = torch.tensor(list(word2ph_tuple), device=self.device)
        indices = torch.repeat_interleave(torch.arange(len(word2ph_tuple), device=self.device), word2ph_tensor)
        phone_level_feature = hidden[indices]  # [sum(word2ph), hidden_size]

        return phone_level_feature.T.cpu()  # cache-safe

    def clear_cache(self):
        """
        Clear the internal BERT feature cache
        """
        self._cached_bert.cache_clear()
