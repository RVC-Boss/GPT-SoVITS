import os
import shutil
import traceback
from collections import OrderedDict
from time import time as ttime
from typing import Any

import torch

from GPT_SoVITS.module.models import set_serialization
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
set_serialization()


def save(fea, path):  # fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = f"{ttime()}.pth"
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, f"{dir}/{name}")


def save_ckpt(ckpt, name, epoch, steps, hps, lora_rank=None):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = hps.to_dict()
        opt["info"] = f"{epoch}epoch_{steps}iteration"
        if lora_rank:
            opt["lora_rank"] = lora_rank
        save(opt, f"{hps.save_weight_dir}/{name}.pth")
        return "Success."
    except Exception:
        return traceback.format_exc()


def inspect_version(
    f: str,
) -> tuple[str, str, bool, Any, dict]:
    """

    Returns:
        tuple[model_version, lang_version, is_lora, hps, state_dict]
    """
    dict_s2 = torch.load(f, map_location="cpu", mmap=True)
    hps = dict_s2["config"]
    version: str | None = None
    if "version" in hps.keys():
        version = hps["version"]
    is_lora = "lora_rank" in dict_s2.keys()

    if version is not None:
        # V3 V4 Lora & Finetuned V2 Pro
        lang_version = "v2"
        model_version = version
    else:
        # V2 Pro Pretrain
        if hps["model"]["gin_channels"] == 1024:
            if hps["model"]["upsample_initial_channel"] == 768:
                lang_version = "v2"
                model_version = "v2ProPlus"
            else:
                lang_version = "v2"
                model_version = "v2Pro"

            return model_version, lang_version, is_lora, hps, dict_s2

        # Old V1/V2
        if "dec.conv_pre.weight" in dict_s2["weight"].keys():
            if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
                lang_version = model_version = "v1"
            else:
                lang_version = model_version = "v2"
        else:  # Old Finetuned V3 & V3/V4 Pretrain
            lang_version = "v2"
            model_version = "v3"
            if dict_s2["info"] == "pretrained_s2G_v4":
                model_version = "v4"

    return model_version, lang_version, is_lora, hps, dict_s2
