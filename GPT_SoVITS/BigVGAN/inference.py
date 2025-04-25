# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
import librosa
from utils import load_checkpoint
from meldataset import get_mel_spectrogram
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from bigvgan import BigVGAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False


def inference(a, h):
    generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # Load the ground truth audio and resample if necessary
            wav, sr = librosa.load(os.path.join(a.input_wavs_dir, filname), sr=h.sampling_rate, mono=True)
            wav = torch.FloatTensor(wav).to(device)
            # Compute mel spectrogram from the ground truth audio
            x = get_mel_spectrogram(wav.unsqueeze(0), generator.h)

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + "_generated.wav")
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="test_files")
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a, h)


if __name__ == "__main__":
    main()
