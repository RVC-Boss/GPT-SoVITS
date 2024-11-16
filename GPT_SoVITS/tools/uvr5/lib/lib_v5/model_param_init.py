import json
import os
import pathlib

default_param = {}
default_param["bins"] = 768
default_param["unstable_bins"] = 9  # training only
default_param["reduction_bins"] = 762  # training only
default_param["sr"] = 44100
default_param["pre_filter_start"] = 757
default_param["pre_filter_stop"] = 768
default_param["band"] = {}


default_param["band"][1] = {
    "sr": 11025,
    "hl": 128,
    "n_fft": 960,
    "crop_start": 0,
    "crop_stop": 245,
    "lpf_start": 61,  # inference only
    "res_type": "polyphase",
}

default_param["band"][2] = {
    "sr": 44100,
    "hl": 512,
    "n_fft": 1536,
    "crop_start": 24,
    "crop_stop": 547,
    "hpf_start": 81,  # inference only
    "res_type": "sinc_best",
}


def int_keys(d):
    r = {}
    for k, v in d:
        if k.isdigit():
            k = int(k)
        r[k] = v
    return r


class ModelParameters(object):
    def __init__(self, config_path=""):
        if ".pth" == pathlib.Path(config_path).suffix:
            import zipfile

            with zipfile.ZipFile(config_path, "r") as zip:
                self.param = json.loads(
                    zip.read("param.json"), object_pairs_hook=int_keys
                )
        elif ".json" == pathlib.Path(config_path).suffix:
            with open(config_path, "r") as f:
                self.param = json.loads(f.read(), object_pairs_hook=int_keys)
        else:
            self.param = default_param

        for k in [
            "mid_side",
            "mid_side_b",
            "mid_side_b2",
            "stereo_w",
            "stereo_n",
            "reverse",
        ]:
            if not k in self.param:
                self.param[k] = False
