# This code is modified from https://github.com/ZFTurbo/
import os
import warnings

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Roformer_Loader:
    def get_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            # use fullloader to load tag !!python/tuple, code can be improved
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def get_default_config(self):
        default_config = None
        if self.model_type == "bs_roformer":
            # Use model_bs_roformer_ep_368_sdr_12.9628.yaml and model_bs_roformer_ep_317_sdr_12.9755.yaml as default configuration files
            # Other BS_Roformer models may not be compatible
            # fmt: off
            default_config = {
                "audio": {"chunk_size": 352800, "sample_rate": 44100},
                "model": {
                    "dim": 512,
                    "depth": 12,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "linear_transformer_depth": 0,
                    "freqs_per_bands": (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 12, 12, 12, 12, 12, 12, 12, 24, 24, 24, 24, 24, 24, 24, 24, 48, 48, 48, 48, 48, 48, 48, 48, 128, 129),
                    "dim_head": 64,
                    "heads": 8,
                    "attn_dropout": 0.1,
                    "ff_dropout": 0.1,
                    "flash_attn": True,
                    "dim_freqs_in": 1025,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 441,
                    "stft_win_length": 2048,
                    "stft_normalized": False,
                    "mask_estimator_depth": 2,
                    "multi_stft_resolution_loss_weight": 1.0,
                    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                    "multi_stft_hop_size": 147,
                    "multi_stft_normalized": False,
                },
                "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
                "inference": {"batch_size": 2, "num_overlap": 2},
            }
            # fmt: on
        elif self.model_type == "mel_band_roformer":
            # Use model_mel_band_roformer_ep_3005_sdr_11.4360.yaml as default configuration files
            # Other Mel_Band_Roformer models may not be compatible
            default_config = {
                "audio": {"chunk_size": 352800, "sample_rate": 44100},
                "model": {
                    "dim": 384,
                    "depth": 12,
                    "stereo": True,
                    "num_stems": 1,
                    "time_transformer_depth": 1,
                    "freq_transformer_depth": 1,
                    "linear_transformer_depth": 0,
                    "num_bands": 60,
                    "dim_head": 64,
                    "heads": 8,
                    "attn_dropout": 0.1,
                    "ff_dropout": 0.1,
                    "flash_attn": True,
                    "dim_freqs_in": 1025,
                    "sample_rate": 44100,
                    "stft_n_fft": 2048,
                    "stft_hop_length": 441,
                    "stft_win_length": 2048,
                    "stft_normalized": False,
                    "mask_estimator_depth": 2,
                    "multi_stft_resolution_loss_weight": 1.0,
                    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
                    "multi_stft_hop_size": 147,
                    "multi_stft_normalized": False,
                },
                "training": {"instruments": ["vocals", "other"], "target_instrument": "vocals"},
                "inference": {"batch_size": 2, "num_overlap": 2},
            }

        return default_config

    def get_model_from_config(self):
        if self.model_type == "bs_roformer":
            from bs_roformer.bs_roformer import BSRoformer

            model = BSRoformer(**dict(self.config["model"]))
        elif self.model_type == "mel_band_roformer":
            from bs_roformer.mel_band_roformer import MelBandRoformer

            model = MelBandRoformer(**dict(self.config["model"]))
        else:
            print("Error: Unknown model: {}".format(self.model_type))
            model = None
        return model

    def demix_track(self, model, mix, device):
        C = self.config["audio"]["chunk_size"]  # chunk_size
        N = self.config["inference"]["num_overlap"]
        fade_size = C // 10
        step = int(C // N)
        border = C - step
        batch_size = self.config["inference"]["batch_size"]

        length_init = mix.shape[-1]
        progress_bar = tqdm(total=length_init // step + 1, desc="Processing", leave=False)

        # Do pad from the beginning and end to account floating window results better
        if length_init > 2 * border and (border > 0):
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

        # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
        window_size = C
        fadein = torch.linspace(0, 1, fade_size)
        fadeout = torch.linspace(1, 0, fade_size)
        window_start = torch.ones(window_size)
        window_middle = torch.ones(window_size)
        window_finish = torch.ones(window_size)
        window_start[-fade_size:] *= fadeout  # First audio chunk, no fadein
        window_finish[:fade_size] *= fadein  # Last audio chunk, no fadeout
        window_middle[-fade_size:] *= fadeout
        window_middle[:fade_size] *= fadein

        with torch.amp.autocast("cuda"):
            with torch.inference_mode():
                if self.config["training"]["target_instrument"] is None:
                    req_shape = (len(self.config["training"]["instruments"]),) + tuple(mix.shape)
                else:
                    req_shape = (1,) + tuple(mix.shape)

                result = torch.zeros(req_shape, dtype=torch.float32)
                counter = torch.zeros(req_shape, dtype=torch.float32)
                i = 0
                batch_data = []
                batch_locations = []
                while i < mix.shape[1]:
                    part = mix[:, i : i + C].to(device)
                    length = part.shape[-1]
                    if length < C:
                        if length > C // 2 + 1:
                            part = nn.functional.pad(input=part, pad=(0, C - length), mode="reflect")
                        else:
                            part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode="constant", value=0)
                    if self.is_half:
                        part = part.half()
                    batch_data.append(part)
                    batch_locations.append((i, length))
                    i += step
                    progress_bar.update(1)

                    if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                        arr = torch.stack(batch_data, dim=0)
                        # print(23333333,arr.dtype)
                        x = model(arr)

                        window = window_middle
                        if i - step == 0:  # First audio chunk, no fadein
                            window = window_start
                        elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                            window = window_finish

                        for j in range(len(batch_locations)):
                            start, l = batch_locations[j]
                            result[..., start : start + l] += x[j][..., :l].cpu() * window[..., :l]
                            counter[..., start : start + l] += window[..., :l]

                        batch_data = []
                        batch_locations = []

                estimated_sources = result / counter
                estimated_sources = estimated_sources.cpu().numpy()
                np.nan_to_num(estimated_sources, copy=False, nan=0.0)

                if length_init > 2 * border and (border > 0):
                    # Remove pad
                    estimated_sources = estimated_sources[..., border:-border]

        progress_bar.close()

        if self.config["training"]["target_instrument"] is None:
            return {k: v for k, v in zip(self.config["training"]["instruments"], estimated_sources)}
        else:
            return {k: v for k, v in zip([self.config["training"]["target_instrument"]], estimated_sources)}

    def run_folder(self, input, vocal_root, others_root, format):
        self.model.eval()
        path = input
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        file_base_name = os.path.splitext(os.path.basename(path))[0]

        sample_rate = 44100
        if "sample_rate" in self.config["audio"]:
            sample_rate = self.config["audio"]["sample_rate"]

        try:
            mix, sr = librosa.load(path, sr=sample_rate, mono=False)
        except Exception as e:
            print("Can read track: {}".format(path))
            print("Error message: {}".format(str(e)))
            return

        # in case if model only supports mono tracks
        isstereo = self.config["model"].get("stereo", True)
        if not isstereo and len(mix.shape) != 1:
            mix = np.mean(mix, axis=0)  # if more than 2 channels, take mean
            print("Warning: Track has more than 1 channels, but model is mono, taking mean of all channels.")

        mix_orig = mix.copy()

        mixture = torch.tensor(mix, dtype=torch.float32)
        res = self.demix_track(self.model, mixture, self.device)

        if self.config["training"]["target_instrument"] is not None:
            # if target instrument is specified, save target instrument as vocal and other instruments as others
            # other instruments are caculated by subtracting target instrument from mixture
            target_instrument = self.config["training"]["target_instrument"]
            other_instruments = [i for i in self.config["training"]["instruments"] if i != target_instrument]
            other = mix_orig - res[target_instrument]  # caculate other instruments

            path_vocal = "{}/{}_{}.wav".format(vocal_root, file_base_name, target_instrument)
            path_other = "{}/{}_{}.wav".format(others_root, file_base_name, other_instruments[0])
            self.save_audio(path_vocal, res[target_instrument].T, sr, format)
            self.save_audio(path_other, other.T, sr, format)
        else:
            # if target instrument is not specified, save the first instrument as vocal and the rest as others
            vocal_inst = self.config["training"]["instruments"][0]
            path_vocal = "{}/{}_{}.wav".format(vocal_root, file_base_name, vocal_inst)
            self.save_audio(path_vocal, res[vocal_inst].T, sr, format)
            for other in self.config["training"]["instruments"][1:]:  # save other instruments
                path_other = "{}/{}_{}.wav".format(others_root, file_base_name, other)
                self.save_audio(path_other, res[other].T, sr, format)

    def save_audio(self, path, data, sr, format):
        # input path should be endwith '.wav'
        if format in ["wav", "flac"]:
            if format == "flac":
                path = path[:-3] + "flac"
            sf.write(path, data, sr)
        else:
            sf.write(path, data, sr)
            os.system('ffmpeg -i "{}" -vn "{}" -q:a 2 -y'.format(path, path[:-3] + format))
            try:
                os.remove(path)
            except:
                pass

    def __init__(self, model_path, config_path, device, is_half):
        self.device = device
        self.is_half = is_half
        self.model_type = None
        self.config = None

        # get model_type, first try:
        if "bs_roformer" in model_path.lower() or "bsroformer" in model_path.lower():
            self.model_type = "bs_roformer"
        elif "mel_band_roformer" in model_path.lower() or "melbandroformer" in model_path.lower():
            self.model_type = "mel_band_roformer"

        if not os.path.exists(config_path):
            if self.model_type is None:
                # if model_type is still None, raise an error
                raise ValueError(
                    "Error: Unknown model type. If you are using a model without a configuration file, Ensure that your model name includes 'bs_roformer', 'bsroformer', 'mel_band_roformer', or 'melbandroformer'. Otherwise, you can manually place the model configuration file into 'tools/uvr5/uvr5w_weights' and ensure that the configuration file is named as '<model_name>.yaml' then try it again."
                )
            self.config = self.get_default_config()
        else:
            # if there is a configuration file
            self.config = self.get_config(config_path)
            if self.model_type is None:
                # if model_type is still None, second try, get model_type from the configuration file
                if "freqs_per_bands" in self.config["model"]:
                    # if freqs_per_bands in config, it's a bs_roformer model
                    self.model_type = "bs_roformer"
                else:
                    # else it's a mel_band_roformer model
                    self.model_type = "mel_band_roformer"

        print("Detected model type: {}".format(self.model_type))
        model = self.get_model_from_config()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)

        if is_half == False:
            self.model = model.to(device)
        else:
            self.model = model.half().to(device)

    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        self.run_folder(input, vocal_root, others_root, format)
