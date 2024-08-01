import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")
import torch
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import math, traceback
import multiprocessing
import sys, pdb

now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

# inp_text=sys.argv[1]
# exp_name=sys.argv[2]
# i_part=sys.argv[3]
# all_parts=sys.argv[4]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[5]
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name


hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False
        )
    )

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        if is_half == True:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines[int(i_part) :: int(all_parts)]:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
