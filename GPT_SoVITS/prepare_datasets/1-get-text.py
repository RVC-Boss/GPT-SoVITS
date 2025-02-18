# -*- coding: utf-8 -*-

import os
import re
import LangSegment
from text import chinese

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get('version', None)
import sys, numpy as np, traceback, pdb
import os.path
from glob import glob
from tqdm import tqdm
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from tools.my_utils import clean_path

# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"

from time import time as ttime
import shutil


def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    if os.path.exists(bert_pretrained_dir):...
    else:raise FileNotFoundError(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def get_bert_inf(phones:list, word2ph:list, norm_text:str, language:str):
        language=language.replace("all_","")
        if language == "zh":
            feature = get_bert_feature(norm_text, word2ph).to(device)
        else:
            feature = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float32,
            ).to(device)

        return feature

    def get_phones_and_bert(text:str, language:str, version:str, final:bool=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_","")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return get_phones_and_bert(formattext,"zh",version)
                else:
                    phones, word2ph, norm_text = clean_text(formattext, language, version)
                    bert = get_bert_feature(norm_text, word2ph).to(device)
            elif language == "yue" and re.search(r'[A-Za-z]', formattext):
                    formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                    formattext = chinese.mix_text_normalize(formattext)
                    return get_phones_and_bert(formattext,"yue",version)
            else:
                phones, word2ph, norm_text = clean_text(formattext, language, version)
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=torch.float32,
                ).to(device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist=[]
            langlist=[]
            LangSegment.setfilters(["zh","ja","en","ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            # print(textlist)
            # print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text(textlist[i], lang, version)
                bert = get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = ''.join(norm_text_list)

        return phones, bert, norm_text

    def process(data, res):
        for name, text, lan in data:
            try:
                name=clean_path(name)
                name = os.path.basename(name)
                print(name)
                phones, bert_feature, norm_text = get_phones_and_bert(
                    text.replace("%", "-").replace("￥", ","), lan, 'v2'
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)
                if os.path.exists(path_bert) == False and lan == "zh":
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, None, norm_text])
            except:
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append(
                    [wav_name, text, language_v1_to_language_v2.get(language, language)]
                )
            else:
                print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
        except:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
