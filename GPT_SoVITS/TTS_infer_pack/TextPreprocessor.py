
import os, sys

from tqdm import tqdm
now_dir = os.getcwd()
sys.path.append(now_dir)

import re
import torch
import LangSegment
from typing import Dict, List, Tuple
from text.cleaner import clean_text
from text import cleaned_text_to_sequence
from transformers import AutoModelForMaskedLM, AutoTokenizer
from TTS_infer_pack.text_segmentation_method import split_big_text, splits, get_method as get_seg_method

from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

def get_first(text:str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text

def merge_short_text_in_array(texts:str, threshold:int) -> list:
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result






class TextPreprocessor:
    def __init__(self, bert_model:AutoModelForMaskedLM, 
                 tokenizer:AutoTokenizer, device:torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        
    def preprocess(self, text:str, lang:str, text_split_method:str)->List[Dict]:
        print(i18n("############ 切分文本 ############"))
        texts = self.pre_seg_text(text, lang, text_split_method)
        result = []
        print(i18n("############ 提取文本Bert特征 ############"))
        for text in tqdm(texts):
            phones, bert_features, norm_text = self.segment_and_extract_feature_for_text(text, lang)
            if phones is None:
                continue
            res={
                "phones": phones,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            result.append(res)
        return result

    def pre_seg_text(self, text:str, lang:str, text_split_method:str):
        text = text.strip("\n")
        if (text[0] not in splits and len(get_first(text)) < 4): 
            text = "。" + text if lang != "en" else "." + text
        print(i18n("实际输入的目标文本:"))
        print(text)
        
        seg_method = get_seg_method(text_split_method)
        text = seg_method(text)
        
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        _texts = text.split("\n")
        _texts = merge_short_text_in_array(_texts, 5)
        texts = []

        
        for text in _texts:
            # 解决输入目标文本的空行导致报错的问题
            if (len(text.strip()) == 0):
                continue
            if (text[-1] not in splits): text += "。" if lang != "en" else "."
            
            # 解决句子过长导致Bert报错的问题
            if (len(text) > 510):
                texts.extend(split_big_text(text))
            else:
                texts.append(text)
            
        print(i18n("实际输入的目标文本(切句后):"))
        print(texts)
        return texts
    
    def segment_and_extract_feature_for_text(self, texts:list, language:str)->Tuple[list, torch.Tensor, str]:
        textlist, langlist = self.seg_text(texts, language)
        if len(textlist) == 0:
            return None, None, None
        
        phones, bert_features, norm_text = self.extract_bert_feature(textlist, langlist)
        return phones, bert_features, norm_text


    def seg_text(self, text:str, language:str)->Tuple[list, list]:

        textlist=[]
        langlist=[]
        if language in ["auto", "zh", "ja"]:
            LangSegment.setfilters(["zh","ja","en","ko"])
            for tmp in LangSegment.getTexts(text):
                if tmp["text"] == "":
                    continue
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                elif tmp["lang"] == "en":
                    langlist.append("en")
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language if language!="auto" else tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if formattext != "":
                textlist.append(formattext)
                langlist.append("en")
            
        elif language in ["all_zh","all_ja"]:

            formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            language = language.replace("all_","")
            if text == "":
                return [],[]
            textlist.append(formattext)
            langlist.append(language) 
        
        else:
            raise ValueError(f"language {language} not supported")
        
        return textlist, langlist
    

    def extract_bert_feature(self, textlist:list, langlist:list):
        phones_list = []
        bert_feature_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = self.clean_text_inf(textlist[i], lang)
            _bert_feature = self.get_bert_inf(phones, word2ph, norm_text, lang)
            # phones_list.append(phones)
            phones_list.extend(phones)
            norm_text_list.append(norm_text)
            bert_feature_list.append(_bert_feature)
        bert_feature = torch.cat(bert_feature_list, dim=1)
        # phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)
        return phones_list, bert_feature, norm_text


    def get_bert_feature(self, text:str, word2ph:list)->torch.Tensor:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T
    
    def clean_text_inf(self, text:str, language:str):
        phones, word2ph, norm_text = clean_text(text, language)
        phones = cleaned_text_to_sequence(phones)
        return phones, word2ph, norm_text

    def get_bert_inf(self, phones:list, word2ph:list, norm_text:str, language:str):
        language=language.replace("all_","")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(self.device)

        return feature
    



