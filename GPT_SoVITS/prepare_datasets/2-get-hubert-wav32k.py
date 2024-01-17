# -*- coding: utf-8 -*-

import sys,os
inp_text=                           os.environ.get("inp_text")
inp_wav_dir=                        os.environ.get("inp_wav_dir")
exp_name=                           os.environ.get("exp_name")
i_part=                             os.environ.get("i_part")
all_parts=                          os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"]= os.environ.get("_CUDA_VISIBLE_DEVICES")
from feature_extractor import cnhubert
opt_dir=                            os.environ.get("opt_dir")
cnhubert.cnhubert_base_path=                os.environ.get("cnhubert_base_dir")
is_half=eval(os.environ.get("is_half","True"))

import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
from my_utils import load_audio

# from config import cnhubert_base_path
# cnhubert.cnhubert_base_path=cnhubert_base_path
# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]
# cnhubert.cnhubert_base_path=sys.argv[7]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name

from time import time as ttime
import shutil
def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

maxx=0.95
alpha=0.5
device="cuda:0"
model=cnhubert.get_model()
if(is_half==True):
    model=model.half().to(device)
else:
    model = model.to(device)
def name2go(wav_name):
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):return
    if(inp_wav_dir!=""):
        wav_path="%s/%s"%(inp_wav_dir,wav_name)
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32, orig_sr=32000, target_sr=16000
    )
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:return
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    # torch.save(ssl,hubert_path )
    my_save(ssl,hubert_path )

with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")

for line in lines[int(i_part)::int(all_parts)]:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        wav_name=os.path.basename(wav_name)
        name2go(wav_name)
    except:
        print(line,traceback.format_exc())
