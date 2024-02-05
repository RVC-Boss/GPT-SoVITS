import hashlib
import json
import math
import os

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def crop_center(h1, h2):
    h1_shape = h1.size()
    h2_shape = h2.size()

    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")

    # s_freq = (h2_shape[2] - h1_shape[2]) // 2
    # e_freq = s_freq + h1_shape[2]
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]

    return h1


def wave_to_spectrogram(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):
    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    spec_left  = librosa.stft(wave_left,  n_fft=n_fft, hop_length=hop_length)
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)

    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def wave_to_spectrogram_mt(
    wave, hop_length, n_fft, mid_side=False, mid_side_b2=False, reverse=False
):
    import threading

    if reverse:
        wave_left = np.flip(np.asfortranarray(wave[0]))
        wave_right = np.flip(np.asfortranarray(wave[1]))
    elif mid_side:
        wave_left = np.asfortranarray(np.add(wave[0], wave[1]) / 2)
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1]))
    elif mid_side_b2:
        wave_left = np.asfortranarray(np.add(wave[1], wave[0] * 0.5))
        wave_right = np.asfortranarray(np.subtract(wave[0], wave[1] * 0.5))
    else:
        wave_left = np.asfortranarray(wave[0])
        wave_right = np.asfortranarray(wave[1])

    def run_thread(**kwargs):
        global spec_left
        spec_left = librosa.stft(**kwargs)

    thread = threading.Thread(
        target=run_thread,
        kwargs={"y": wave_left, "n_fft": n_fft, "hop_length": hop_length},
    )
    thread.start()
    spec_right = librosa.stft(wave_right, n_fft=n_fft, hop_length=hop_length)
    thread.join()

    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def combine_spectrograms(specs, mp):
    l = min([specs[i].shape[2] for i in specs])
    spec_c = np.zeros(shape=(2, mp.param["bins"] + 1, l), dtype=np.complex64)
    offset = 0
    bands_n = len(mp.param["band"])

    for d in range(1, bands_n + 1):
        h = mp.param["band"][d]["crop_stop"] - mp.param["band"][d]["crop_start"]
        spec_c[:, offset : offset + h, :l] = specs[d][
            :, mp.param["band"][d]["crop_start"] : mp.param["band"][d]["crop_stop"], :l
        ]
        offset += h

    if offset > mp.param["bins"]:
        raise ValueError("Too much bins")

    # lowpass fiter
    if (
        mp.param["pre_filter_start"] > 0
    ):  # and mp.param['band'][bands_n]['res_type'] in ['scipy', 'polyphase']:
        if bands_n == 1:
            spec_c = fft_lp_filter(
                spec_c, mp.param["pre_filter_start"], mp.param["pre_filter_stop"]
            )
        else:
            gp = 1
            for b in range(
                mp.param["pre_filter_start"] + 1, mp.param["pre_filter_stop"]
            ):
                g = math.pow(
                    10, -(b - mp.param["pre_filter_start"]) * (3.5 - gp) / 20.0
                )
                gp = g
                spec_c[:, b, :] *= g

    return np.asfortranarray(spec_c)


def spectrogram_to_image(spec, mode="magnitude"):
    if mode == "magnitude":
        if np.iscomplexobj(spec):
            y = np.abs(spec)
        else:
            y = spec
        y = np.log10(y**2 + 1e-8)
    elif mode == "phase":
        if np.iscomplexobj(spec):
            y = np.angle(spec)
        else:
            y = spec

    y -= y.min()
    y *= 255 / y.max()
    img = np.uint8(y)

    if y.ndim == 3:
        img = img.transpose(1, 2, 0)
        img = np.concatenate([np.max(img, axis=2, keepdims=True), img], axis=2)

    return img


def reduce_vocal_aggressively(X, y, softmask):
    v = X - y
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(v)

    v_mask = v_mag_tmp > y_mag_tmp
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)

    return y_mag * np.exp(1.0j * np.angle(y))


def mask_silence(mag, ref, thres=0.2, min_range=64, fade_size=32):
    if min_range < fade_size * 2:
        raise ValueError("min_range must be >= fade_area * 2")

    mag = mag.copy()

    idx = np.where(ref.mean(axis=(0, 1)) < thres)[0]
    starts = np.insert(idx[np.where(np.diff(idx) != 1)[0] + 1], 0, idx[0])
    ends = np.append(idx[np.where(np.diff(idx) != 1)[0]], idx[-1])
    uninformative = np.where(ends - starts > min_range)[0]
    if len(uninformative) > 0:
        starts = starts[uninformative]
        ends = ends[uninformative]
        old_e = None
        for s, e in zip(starts, ends):
            if old_e is not None and s - old_e < fade_size:
                s = old_e - fade_size * 2

            if s != 0:
                weight = np.linspace(0, 1, fade_size)
                mag[:, :, s : s + fade_size] += weight * ref[:, :, s : s + fade_size]
            else:
                s -= fade_size

            if e != mag.shape[2]:
                weight = np.linspace(1, 0, fade_size)
                mag[:, :, e - fade_size : e] += weight * ref[:, :, e - fade_size : e]
            else:
                e += fade_size

            mag[:, :, s + fade_size : e - fade_size] += ref[
                :, :, s + fade_size : e - fade_size
            ]
            old_e = e

    return mag


def align_wave_head_and_tail(a, b):
    l = min([a[0].size, b[0].size])

    return a[:l, :l], b[:l, :l]


def cache_or_load(mix_path, inst_path, mp):
    mix_basename = os.path.splitext(os.path.basename(mix_path))[0]
    inst_basename = os.path.splitext(os.path.basename(inst_path))[0]

    cache_dir = "mph{}".format(
        hashlib.sha1(json.dumps(mp.param, sort_keys=True).encode("utf-8")).hexdigest()
    )
    mix_cache_dir = os.path.join("cache", cache_dir)
    inst_cache_dir = os.path.join("cache", cache_dir)

    os.makedirs(mix_cache_dir, exist_ok=True)
    os.makedirs(inst_cache_dir, exist_ok=True)

    mix_cache_path = os.path.join(mix_cache_dir, mix_basename + ".npy")
    inst_cache_path = os.path.join(inst_cache_dir, inst_basename + ".npy")

    if os.path.exists(mix_cache_path) and os.path.exists(inst_cache_path):
        X_spec_m = np.load(mix_cache_path)
        y_spec_m = np.load(inst_cache_path)
    else:
        X_wave, y_wave, X_spec_s, y_spec_s = {}, {}, {}, {}

        for d in range(len(mp.param["band"]), 0, -1):
            bp = mp.param["band"][d]

            if d == len(mp.param["band"]):  # high-end band
                X_wave[d], _ = librosa.load(
                    mix_path, 
                    sr       = bp["sr"],
                    mono     = False,
                    dtype    = np.float32,
                    res_type = bp["res_type"]
                )
                y_wave[d], _ = librosa.load(
                    inst_path,
                    sr       = bp["sr"],
                    mono     = False,
                    dtype    = np.float32,
                    res_type = bp["res_type"],
                )
            else:  # lower bands
                X_wave[d] = librosa.resample(
                    X_wave[d + 1],
                    orig_sr   = mp.param["band"][d + 1]["sr"],
                    target_sr = bp["sr"],
                    res_type  = bp["res_type"],
                )
                y_wave[d] = librosa.resample(
                    y_wave[d + 1],
                    orig_sr   = mp.param["band"][d + 1]["sr"],
                    target_sr = bp["sr"],
                    res_type  = bp["res_type"],
                )

            X_wave[d], y_wave[d] = align_wave_head_and_tail(X_wave[d], y_wave[d])

            X_spec_s[d] = wave_to_spectrogram(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )
            y_spec_s[d] = wave_to_spectrogram(
                y_wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        del X_wave, y_wave

        X_spec_m = combine_spectrograms(X_spec_s, mp)
        y_spec_m = combine_spectrograms(y_spec_s, mp)

        if X_spec_m.shape != y_spec_m.shape:
            raise ValueError("The combined spectrograms are different: " + mix_path)

        _, ext = os.path.splitext(mix_path)

        np.save(mix_cache_path, X_spec_m)
        np.save(inst_cache_path, y_spec_m)

    return X_spec_m, y_spec_m


def spectrogram_to_wave(spec, hop_length, mid_side, mid_side_b2, reverse):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hop_length)
    wave_right = librosa.istft(spec_right, hop_length=hop_length)

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray(
            [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
        )
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def spectrogram_to_wave_mt(spec, hop_length, mid_side, reverse, mid_side_b2):
    import threading

    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    def run_thread(**kwargs):
        global wave_left
        wave_left = librosa.istft(**kwargs)

    thread = threading.Thread(
        target=run_thread, kwargs={"stft_matrix": spec_left, "hop_length": hop_length}
    )
    thread.start()
    wave_right = librosa.istft(spec_right, hop_length=hop_length)
    thread.join()

    if reverse:
        return np.asfortranarray([np.flip(wave_left), np.flip(wave_right)])
    elif mid_side:
        return np.asfortranarray(
            [np.add(wave_left, wave_right / 2), np.subtract(wave_left, wave_right / 2)]
        )
    elif mid_side_b2:
        return np.asfortranarray(
            [
                np.add(wave_right / 1.25, 0.4 * wave_left),
                np.subtract(wave_left / 1.25, 0.4 * wave_right),
            ]
        )
    else:
        return np.asfortranarray([wave_left, wave_right])


def cmb_spectrogram_to_wave(spec_m, mp, extra_bins_h=None, extra_bins=None):
    wave_band = {}
    bands_n = len(mp.param["band"])
    offset = 0

    for d in range(1, bands_n + 1):
        bp = mp.param["band"][d]
        spec_s = np.ndarray(
            shape=(2, bp["n_fft"] // 2 + 1, spec_m.shape[2]), dtype=complex
        )
        h = bp["crop_stop"] - bp["crop_start"]
        spec_s[:, bp["crop_start"] : bp["crop_stop"], :] = spec_m[
            :, offset : offset + h, :
        ]

        offset += h
        if d == bands_n:  # higher
            if extra_bins_h:  # if --high_end_process bypass
                max_bin = bp["n_fft"] // 2
                spec_s[:, max_bin - extra_bins_h : max_bin, :] = extra_bins[
                    :, :extra_bins_h, :
                ]
            if bp["hpf_start"] > 0:
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
            if bands_n == 1:
                wave = spectrogram_to_wave(
                    spec_s,
                    bp["hl"],
                    mp.param["mid_side"],
                    mp.param["mid_side_b2"],
                    mp.param["reverse"],
                )
            else:
                wave = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
        else:
            sr = mp.param["band"][d + 1]["sr"]
            if d == 1:  # lower
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave = librosa.resample(
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                    orig_sr   = bp["sr"],
                    target_sr = sr,
                    res_type  = "sinc_fastest",
                )
            else:  # mid
                spec_s = fft_hp_filter(spec_s, bp["hpf_start"], bp["hpf_stop"] - 1)
                spec_s = fft_lp_filter(spec_s, bp["lpf_start"], bp["lpf_stop"])
                wave2 = np.add(
                    wave,
                    spectrogram_to_wave(
                        spec_s,
                        bp["hl"],
                        mp.param["mid_side"],
                        mp.param["mid_side_b2"],
                        mp.param["reverse"],
                    ),
                )
                # wave = librosa.core.resample(wave2, orig_sr=bp['sr'], target_sr=sr, res_type="sinc_fastest")
                wave = librosa.core.resample(wave2, orig_sr=bp["sr"], target_sr=sr, res_type="scipy")

    return wave.T


def fft_lp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop):
        g -= 1 / (bin_stop - bin_start)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, bin_stop:, :] *= 0

    return spec


def fft_hp_filter(spec, bin_start, bin_stop):
    g = 1.0
    for b in range(bin_start, bin_stop, -1):
        g -= 1 / (bin_start - bin_stop)
        spec[:, b, :] = g * spec[:, b, :]

    spec[:, 0 : bin_stop + 1, :] *= 0

    return spec


def mirroring(a, spec_m, input_high_end, mp):
    if "mirroring" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mirror = mirror * np.exp(1.0j * np.angle(input_high_end))

        return np.where(
            np.abs(input_high_end) <= np.abs(mirror), input_high_end, mirror
        )

    if "mirroring2" == a:
        mirror = np.flip(
            np.abs(
                spec_m[
                    :,
                    mp.param["pre_filter_start"]
                    - 10
                    - input_high_end.shape[1] : mp.param["pre_filter_start"]
                    - 10,
                    :,
                ]
            ),
            1,
        )
        mi = np.multiply(mirror, input_high_end * 1.7)

        return np.where(np.abs(input_high_end) <= np.abs(mi), input_high_end, mi)


def ensembling(a, specs):
    for i in range(1, len(specs)):
        if i == 1:
            spec = specs[0]

        ln = min([spec.shape[2], specs[i].shape[2]])
        spec = spec[:, :, :ln]
        specs[i] = specs[i][:, :, :ln]

        if "min_mag" == a:
            spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
        if "max_mag" == a:
            spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

    return spec


def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])

    return spec


def istft(spec, hl):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])

    wave_left = librosa.istft(spec_left, hop_length=hl)
    wave_right = librosa.istft(spec_right, hop_length=hl)
    wave = np.asfortranarray([wave_left, wave_right])


if __name__ == "__main__":
    import argparse
    import sys
    import time

    import cv2
    from model_param_init import ModelParameters

    p = argparse.ArgumentParser()
    p.add_argument(
        "--algorithm",
        "-a",
        type=str,
        choices=["invert", "invert_p", "min_mag", "max_mag", "deep", "align"],
        default="min_mag",
    )
    p.add_argument(
        "--model_params",
        "-m",
        type=str,
        default=os.path.join("modelparams", "1band_sr44100_hl512.json"),
    )
    p.add_argument("--output_name", "-o", type=str, default="output")
    p.add_argument("--vocals_only", "-v", action="store_true")
    p.add_argument("input", nargs="+")
    args = p.parse_args()

    start_time = time.time()

    if args.algorithm.startswith("invert") and len(args.input) != 2:
        raise ValueError("There should be two input files.")

    if not args.algorithm.startswith("invert") and len(args.input) < 2:
        raise ValueError("There must be at least two input files.")

    wave, specs = {}, {}
    mp = ModelParameters(args.model_params)

    for i in range(len(args.input)):
        spec = {}

        for d in range(len(mp.param["band"]), 0, -1):
            bp = mp.param["band"][d]

            if d == len(mp.param["band"]):  # high-end band
                wave[d], _ = librosa.load(
                    args.input[i],
                    sr       = bp["sr"],
                    mono     = False,
                    dtype    = np.float32,
                    res_type = bp["res_type"],
                )

                if len(wave[d].shape) == 1:  # mono to stereo
                    wave[d] = np.array([wave[d], wave[d]])
            else:  # lower bands
                wave[d] = librosa.resample(
                    wave[d + 1],
                    orig_sr   = mp.param["band"][d + 1]["sr"],
                    target_sr = bp["sr"],
                    res_type  = bp["res_type"],
                )

            spec[d] = wave_to_spectrogram(
                wave[d],
                bp["hl"],
                bp["n_fft"],
                mp.param["mid_side"],
                mp.param["mid_side_b2"],
                mp.param["reverse"],
            )

        specs[i] = combine_spectrograms(spec, mp)

    del wave

    if args.algorithm == "deep":
        d_spec = np.where(np.abs(specs[0]) <= np.abs(spec[1]), specs[0], spec[1])
        v_spec = d_spec - specs[1]
        sf.write(
            os.path.join("{}.wav".format(args.output_name)),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )

    if args.algorithm.startswith("invert"):
        ln = min([specs[0].shape[2], specs[1].shape[2]])
        specs[0] = specs[0][:, :, :ln]
        specs[1] = specs[1][:, :, :ln]

        if "invert_p" == args.algorithm:
            X_mag = np.abs(specs[0])
            y_mag = np.abs(specs[1])
            max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)
            v_spec = specs[1] - max_mag * np.exp(1.0j * np.angle(specs[0]))
        else:
            specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
            v_spec = specs[0] - specs[1]

            if not args.vocals_only:
                X_mag = np.abs(specs[0])
                y_mag = np.abs(specs[1])
                v_mag = np.abs(v_spec)

                X_image = spectrogram_to_image(X_mag)
                y_image = spectrogram_to_image(y_mag)
                v_image = spectrogram_to_image(v_mag)

                cv2.imwrite("{}_X.png".format(args.output_name), X_image)
                cv2.imwrite("{}_y.png".format(args.output_name), y_image)
                cv2.imwrite("{}_v.png".format(args.output_name), v_image)

                sf.write(
                    "{}_X.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[0], mp),
                    mp.param["sr"],
                )
                sf.write(
                    "{}_y.wav".format(args.output_name),
                    cmb_spectrogram_to_wave(specs[1], mp),
                    mp.param["sr"],
                )

        sf.write(
            "{}_v.wav".format(args.output_name),
            cmb_spectrogram_to_wave(v_spec, mp),
            mp.param["sr"],
        )
    else:
        if not args.algorithm == "deep":
            sf.write(
                os.path.join("ensembled", "{}.wav".format(args.output_name)),
                cmb_spectrogram_to_wave(ensembling(args.algorithm, specs), mp),
                mp.param["sr"],
            )

    if args.algorithm == "align":
        trackalignment = [
            {
                "file1": '"{}"'.format(args.input[0]),
                "file2": '"{}"'.format(args.input[1]),
            }
        ]

        for i, e in tqdm(enumerate(trackalignment), desc="Performing Alignment..."):
            os.system(f"python lib/align_tracks.py {e['file1']} {e['file2']}")

    # print('Total time: {0:.{1}f}s'.format(time.time() - start_time, 1))
