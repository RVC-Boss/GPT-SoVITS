import traceback
from collections import OrderedDict
from time import time as ttime
import shutil
import os
import torch
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (ttime())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


from io import BytesIO

model_version2byte = {
    "v4": b"04",
    "v2Pro": b"05",
    "v2ProPlus": b"06",
}


def my_save2(fea, path, model_version):
    bio = BytesIO()
    torch.save(fea, bio)
    bio.seek(0)
    data = bio.getvalue()
    byte = model_version2byte[model_version]
    data = byte + data[2:]
    with open(path, "wb") as f:
        f.write(data)


def savee(ckpt, name, epoch, steps, hps, model_version=None, lora_rank=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = hps
        opt["info"] = "%sepoch_%siteration" % (epoch, steps)
        if lora_rank:
            opt["lora_rank"] = lora_rank
            my_save2(opt, "%s/%s.pth" % (hps.save_weight_dir, name), model_version)
        elif model_version != None and "Pro" in model_version:
            my_save2(opt, "%s/%s.pth" % (hps.save_weight_dir, name), model_version)
        else:
            my_save(opt, "%s/%s.pth" % (hps.save_weight_dir, name))
        return "Success."
    except:
        return traceback.format_exc()


"""
Only V4 and V2Pro series are supported:
04:v4lora
05:v2Pro
06:v2ProPlus
"""
head2version = {
    b"04": ["v2", "v4", True],
    b"05": ["v2", "v2Pro", False],
    b"06": ["v2", "v2ProPlus", False],
}
hash_pretrained_dict = {
    "4f26b9476d0c5033e04162c486074374": ["v2", "v4", False],  # s2Gv4.pth#sovits_v4_pretrained
    "c7e9fce2223f3db685cdfa1e6368728a": ["v2", "v2Pro", False],  # s2Gv2Pro.pth#sovits_v2Pro_pretrained
    "66b313e39455b57ab1b0bc0b239c9d0a": ["v2", "v2ProPlus", False],  # s2Gv2ProPlus.pth#sovits_v2ProPlus_pretrained
}
import hashlib


def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


def get_sovits_version_from_path_fast(sovits_path):
    ###1-if it is pretrained sovits models, by hash
    hash = get_hash_from_file(sovits_path)
    if hash in hash_pretrained_dict:
        return hash_pretrained_dict[hash]
    ###2-new weights, by head
    with open(sovits_path, "rb") as f:
        version = f.read(2)
    if version != b"PK":
        return head2version[version]
    ###3-legacy weights, not supported
    if_lora_v3 = False
    size = os.path.getsize(sovits_path)
    """
    Legacy file size detection - no longer supported:
        v1weights:about 82942KB
        v2weights:about 83014KB
        v3weights:about 750MB
    Only V4 and V2Pro series are supported.
    """
    if size < 700 * 1024 * 1024:
        raise ValueError(f"Legacy model detected (file size: {size/1024/1024:.1f}MB). Only V4 and V2Pro series are supported.")
    else:
        raise ValueError(f"Unsupported model format. Please use V4 or V2Pro series models.")


def load_sovits_new(sovits_path):
    f = open(sovits_path, "rb")
    meta = f.read(2)
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    return torch.load(sovits_path, map_location="cpu", weights_only=False)
