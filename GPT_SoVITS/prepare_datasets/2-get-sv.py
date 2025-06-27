# -*- coding: utf-8 -*-

import sys
import os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

opt_dir = os.environ.get("opt_dir")
sv_path = os.environ.get("sv_path")
import torch

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

import traceback
import torchaudio

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(f"{now_dir}/GPT_SoVITS/eres2net")
from tools.my_utils import clean_path
from time import time as ttime
import shutil
from ERes2NetV2 import ERes2NetV2
import kaldi as Kaldi


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


sv_cn_dir = "%s/7-sv_cn" % (opt_dir)
wav32dir = "%s/5-wav32k" % (opt_dir)
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(sv_cn_dir, exist_ok=True)
os.makedirs(wav32dir, exist_ok=True)

maxx = 0.95
alpha = 0.5
if torch.cuda.is_available():
    device = "cuda:0"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"


class SV:
    def __init__(self, device, is_half):
        pretrained_state = torch.load(sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.res = torchaudio.transforms.Resample(32000, 16000).to(device)
        if is_half == False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)
        self.is_half = is_half

    def compute_embedding3(self, wav):  # (1,x)#-1~1
        with torch.no_grad():
            wav = self.res(wav)
            if self.is_half == True:
                wav = wav.half()
            feat = torch.stack(
                [Kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0) for wav0 in wav]
            )
            sv_emb = self.embedding_model.forward3(feat)
        return sv_emb


sv = SV(device, is_half)


def name2go(wav_name, wav_path):
    sv_cn_path = "%s/%s.pt" % (sv_cn_dir, wav_name)
    if os.path.exists(sv_cn_path):
        return
    wav_path = "%s/%s" % (wav32dir, wav_name)
    wav32k, sr0 = torchaudio.load(wav_path)
    assert sr0 == 32000
    wav32k = wav32k.to(device)
    emb = sv.compute_embedding3(wav32k).cpu()  # torch.Size([1, 20480])
    my_save(emb, sv_cn_path)


with open(inp_text, "r", encoding="utf8") as f:
    lines = f.read().strip("\n").split("\n")

for line in lines[int(i_part) :: int(all_parts)]:
    try:
        wav_name, spk_name, language, text = line.split("|")
        wav_name = clean_path(wav_name)
        if inp_wav_dir != "" and inp_wav_dir != None:
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s" % (inp_wav_dir, wav_name)

        else:
            wav_path = wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name, wav_path)
    except:
        print(line, traceback.format_exc())
