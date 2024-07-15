# This code is modified from https://github.com/ZFTurbo/

import time
import librosa
from tqdm import tqdm
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class BsRoformer_Loader:
    def get_model_from_config(self):
        config = {
            "attn_dropout": 0.1,
            "depth": 12,
            "dim": 512,
            "dim_freqs_in": 1025,
            "dim_head": 64,
            "ff_dropout": 0.1,
            "flash_attn": True,
            "freq_transformer_depth": 1,
            "freqs_per_bands":(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 128, 129),
            "heads": 8,
            "linear_transformer_depth": 0,
            "mask_estimator_depth": 2,
            "multi_stft_hop_size": 147,
            "multi_stft_normalized": False,
            "multi_stft_resolution_loss_weight": 1.0,
            "multi_stft_resolutions_window_sizes":(4096, 2048, 1024, 512, 256),
            "num_stems": 1,
            "stereo": True,
            "stft_hop_length": 441,
            "stft_n_fft": 2048,
            "stft_normalized": False,
            "stft_win_length": 2048,
            "time_transformer_depth": 1,

        }

        from bs_roformer.bs_roformer import BSRoformer
        model = BSRoformer(
            **dict(config)
        )

        return model
    

    def demix_track(self, model, mix, device):
        C = 352800
        N = 2
        fade_size = C // 10
        step = int(C // N)
        border = C - step
        batch_size = 4

        length_init = mix.shape[-1]

        progress_bar = tqdm(total=(length_init//step)+3)
        progress_bar.set_description("Processing")

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            mix = nn.functional.pad(mix, (border, border), mode='reflect')

        # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
        window_size = C
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_start = torch.ones(window_size)
        window_middle = torch.ones(window_size)
        window_finish = torch.ones(window_size)
        window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
        window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        with torch.cuda.amp.autocast():
            with torch.inference_mode():
                req_shape = (1, ) + tuple(mix.shape)

                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i:i + C].to(device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step
                    progress_bar.update(1)

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        x = model(arr)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                            counter[..., start:start+l] += window[..., :l]

                        batch_data = []
                        batch_locations = []

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    # Remove pad
                    estimated_sources = estimated_sources[..., border:-border]

        progress_bar.close()

        return {k: v for k, v in zip(['vocals', 'other'], estimated_sources)}


    def run_folder(self,input, vocal_root, others_root, format):
        # start_time = time.time()
        self.model.eval()
        path = input

        if not os.path.isdir(vocal_root):
            os.mkdir(vocal_root)

        if not os.path.isdir(others_root):
            os.mkdir(others_root)

        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            return

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)

        mix_orig = mix.copy()

        mixture = torch.tensor(mix, dtype=torch.float32)
        res = self.demix_track(self.model, mixture, self.device)

        estimates = res['vocals'].T
        print("{}/{}_{}.{}".format(vocal_root, os.path.basename(path)[:-4], 'vocals', format))
        
        if format in ["wav", "flac"]:
            sf.write("{}/{}_{}.{}".format(vocal_root, os.path.basename(path)[:-4], 'vocals', format), estimates, sr)
            sf.write("{}/{}_{}.{}".format(others_root, os.path.basename(path)[:-4], 'instrumental', format), mix_orig.T - estimates, sr)
        else:
            path_vocal = "%s/%s_vocals.wav" % (vocal_root, os.path.basename(path)[:-4])
            path_other = "%s/%s_instrumental.wav" % (others_root, os.path.basename(path)[:-4])
            sf.write(path_vocal, estimates, sr)
            sf.write(path_other, mix_orig.T - estimates, sr)
            opt_path_vocal = path_vocal[:-4] + ".%s" % format
            opt_path_other = path_other[:-4] + ".%s" % format
            if os.path.exists(path_vocal):
                os.system(
                    "ffmpeg -i '%s' -vn '%s' -q:a 2 -y" % (path_vocal, opt_path_vocal)
                )
                if os.path.exists(opt_path_vocal):
                    try:
                        os.remove(path_vocal)
                    except:
                        pass
            if os.path.exists(path_other):
                os.system(
                    "ffmpeg -i '%s' -vn '%s' -q:a 2 -y" % (path_other, opt_path_other)
                )
                if os.path.exists(opt_path_other):
                    try:
                        os.remove(path_other)
                    except:
                        pass

        # print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


    def __init__(self, model_path, device):
        self.device = device
        self.extract_instrumental=True

        model = self.get_model_from_config()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        self.model = model.to(device)


    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        self.run_folder(input, vocal_root, others_root, format)

