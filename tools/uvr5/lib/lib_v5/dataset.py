import os
import random

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from . import spec_utils


class VocalRemoverValidationSet(torch.utils.data.Dataset):
    def __init__(self, patch_list):
        self.patch_list = patch_list

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        path = self.patch_list[idx]
        data = np.load(path)

        X, y = data["X"], data["y"]

        X_mag = np.abs(X)
        y_mag = np.abs(y)

        return X_mag, y_mag


def make_pair(mix_dir, inst_dir):
    input_exts = [".wav", ".m4a", ".mp3", ".mp4", ".flac"]

    X_list = sorted(
        [
            os.path.join(mix_dir, fname)
            for fname in os.listdir(mix_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )
    y_list = sorted(
        [
            os.path.join(inst_dir, fname)
            for fname in os.listdir(inst_dir)
            if os.path.splitext(fname)[1] in input_exts
        ]
    )

    filelist = list(zip(X_list, y_list))

    return filelist


def train_val_split(dataset_dir, split_mode, val_rate, val_filelist):
    if split_mode == "random":
        filelist = make_pair(
            os.path.join(dataset_dir, "mixtures"),
            os.path.join(dataset_dir, "instruments"),
        )

        random.shuffle(filelist)

        if len(val_filelist) == 0:
            val_size = int(len(filelist) * val_rate)
            train_filelist = filelist[:-val_size]
            val_filelist = filelist[-val_size:]
        else:
            train_filelist = [
                pair for pair in filelist if list(pair) not in val_filelist
            ]
    elif split_mode == "subdirs":
        if len(val_filelist) != 0:
            raise ValueError(
                "The `val_filelist` option is not available in `subdirs` mode"
            )

        train_filelist = make_pair(
            os.path.join(dataset_dir, "training/mixtures"),
            os.path.join(dataset_dir, "training/instruments"),
        )

        val_filelist = make_pair(
            os.path.join(dataset_dir, "validation/mixtures"),
            os.path.join(dataset_dir, "validation/instruments"),
        )

    return train_filelist, val_filelist


def augment(X, y, reduction_rate, reduction_mask, mixup_rate, mixup_alpha):
    perm = np.random.permutation(len(X))
    for i, idx in enumerate(tqdm(perm)):
        if np.random.uniform() < reduction_rate:
            y[idx] = spec_utils.reduce_vocal_aggressively(
                X[idx], y[idx], reduction_mask
            )

        if np.random.uniform() < 0.5:
            # swap channel
            X[idx] = X[idx, ::-1]
            y[idx] = y[idx, ::-1]
        if np.random.uniform() < 0.02:
            # mono
            X[idx] = X[idx].mean(axis=0, keepdims=True)
            y[idx] = y[idx].mean(axis=0, keepdims=True)
        if np.random.uniform() < 0.02:
            # inst
            X[idx] = y[idx]

        if np.random.uniform() < mixup_rate and i < len(perm) - 1:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            X[idx] = lam * X[idx] + (1 - lam) * X[perm[i + 1]]
            y[idx] = lam * y[idx] + (1 - lam) * y[perm[i + 1]]

    return X, y


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - left * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def make_training_set(filelist, cropsize, patches, sr, hop_length, n_fft, offset):
    len_dataset = patches * len(filelist)

    X_dataset = np.zeros((len_dataset, 2, n_fft // 2 + 1, cropsize), dtype=np.complex64)
    y_dataset = np.zeros((len_dataset, 2, n_fft // 2 + 1, cropsize), dtype=np.complex64)

    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft)
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        X, y = X / coef, y / coef

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode="constant")
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode="constant")

        starts = np.random.randint(0, X_pad.shape[2] - cropsize, patches)
        ends = starts + cropsize
        for j in range(patches):
            idx = i * patches + j
            X_dataset[idx] = X_pad[:, :, starts[j] : ends[j]]
            y_dataset[idx] = y_pad[:, :, starts[j] : ends[j]]

    return X_dataset, y_dataset


def make_validation_set(filelist, cropsize, sr, hop_length, n_fft, offset):
    patch_list = []
    patch_dir = "cs{}_sr{}_hl{}_nf{}_of{}".format(
        cropsize, sr, hop_length, n_fft, offset
    )
    os.makedirs(patch_dir, exist_ok=True)

    for i, (X_path, y_path) in enumerate(tqdm(filelist)):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        X, y = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft)
        coef = np.max([np.abs(X).max(), np.abs(y).max()])
        X, y = X / coef, y / coef

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode="constant")
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode="constant")

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, "{}_p{}.npz".format(basename, j))
            start = j * roi_size
            if not os.path.exists(outpath):
                np.savez(
                    outpath,
                    X=X_pad[:, :, start : start + cropsize],
                    y=y_pad[:, :, start : start + cropsize],
                )
            patch_list.append(outpath)

    return VocalRemoverValidationSet(patch_list)
