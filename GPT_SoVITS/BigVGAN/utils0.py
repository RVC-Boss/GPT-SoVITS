# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import glob
import os

import torch
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix, renamed_file=None):
    # Fallback to original scanning logic first
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)

    if len(cp_list) > 0:
        last_checkpoint_path = sorted(cp_list)[-1]
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return last_checkpoint_path

    # If no pattern-based checkpoints are found, check for renamed file
    if renamed_file:
        renamed_path = os.path.join(cp_dir, renamed_file)
        if os.path.isfile(renamed_path):
            print(f"[INFO] Resuming from renamed checkpoint: '{renamed_file}'")
            return renamed_path

    return None
