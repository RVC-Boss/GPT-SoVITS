# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import sys

# to import modules from parent_dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
import json
from env import AttrDict
from bigvgan import BigVGAN
from time import time
from tqdm import tqdm
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from scipy.io.wavfile import write
import numpy as np

import argparse

torch.backends.cudnn.benchmark = True

# For easier debugging
torch.set_printoptions(linewidth=200, threshold=10_000)


def generate_soundwave(duration=5.0, sr=24000):
    t = np.linspace(0, duration, int(sr * duration), False, dtype=np.float32)

    modulation = np.sin(2 * np.pi * t / duration)

    min_freq = 220
    max_freq = 1760
    frequencies = min_freq + (max_freq - min_freq) * (modulation + 1) / 2
    soundwave = np.sin(2 * np.pi * frequencies * t)

    soundwave = soundwave / np.max(np.abs(soundwave)) * 0.95

    return soundwave, sr


def get_mel(x, h):
    return mel_spectrogram(
        x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax
    )


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test script to check CUDA kernel correctness."
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        required=True,
        help="Path to the checkpoint file. Assumes config.json exists in the directory.",
    )

    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        config = f.read()
    json_config = json.loads(config)
    h = AttrDict({**json_config})

    print("loading plain Pytorch BigVGAN")
    generator_original = BigVGAN(h).to("cuda")
    print("loading CUDA kernel BigVGAN with auto-build")
    generator_cuda_kernel = BigVGAN(h, use_cuda_kernel=True).to("cuda")

    state_dict_g = load_checkpoint(args.checkpoint_file, "cuda")
    generator_original.load_state_dict(state_dict_g["generator"])
    generator_cuda_kernel.load_state_dict(state_dict_g["generator"])

    generator_original.remove_weight_norm()
    generator_original.eval()
    generator_cuda_kernel.remove_weight_norm()
    generator_cuda_kernel.eval()

    # define number of samples and length of mel frame to benchmark
    num_sample = 10
    num_mel_frame = 16384
    
    # CUDA kernel correctness check
    diff = 0.0
    for i in tqdm(range(num_sample)):
        # Random mel
        data = torch.rand((1, h.num_mels, num_mel_frame), device="cuda")
                
        with torch.inference_mode():
            audio_original = generator_original(data)
            
        with torch.inference_mode():
            audio_cuda_kernel = generator_cuda_kernel(data)

        # Both outputs should be (almost) the same
        test_result = (audio_original - audio_cuda_kernel).abs()
        diff += test_result.mean(dim=-1).item()
    
    diff /= num_sample
    if (
        diff <= 2e-3
    ):  # We can expect a small difference (~1e-3) which does not affect perceptual quality
        print(
            f"\n[Success] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}"
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test CUDA fused vs. plain torch BigVGAN inference"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={audio_cuda_kernel[-1][-1][-30:].tolist()}, "
            f"\n > torch_values={audio_original[-1][-1][-30:].tolist()}"
        )
    
    del data, audio_original, audio_cuda_kernel
    
    # Variables for tracking total time and VRAM usage
    toc_total_original = 0
    toc_total_cuda_kernel = 0
    vram_used_original_total = 0
    vram_used_cuda_kernel_total = 0
    audio_length_total = 0

    # Measure Original inference in isolation
    for i in tqdm(range(num_sample)):
        torch.cuda.reset_peak_memory_stats(device="cuda")
        data = torch.rand((1, h.num_mels, num_mel_frame), device="cuda")
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_original = generator_original(data)
        torch.cuda.synchronize()
        toc = time() - tic
        toc_total_original += toc   

        vram_used_original_total += torch.cuda.max_memory_allocated(device="cuda")
        
        del data, audio_original
        torch.cuda.empty_cache()

    # Measure CUDA kernel inference in isolation
    for i in tqdm(range(num_sample)):
        torch.cuda.reset_peak_memory_stats(device="cuda")
        data = torch.rand((1, h.num_mels, num_mel_frame), device="cuda")
        torch.cuda.synchronize()
        tic = time()
        with torch.inference_mode():
            audio_cuda_kernel = generator_cuda_kernel(data)
        torch.cuda.synchronize()
        toc = time() - tic
        toc_total_cuda_kernel += toc
        
        audio_length_total += audio_cuda_kernel.shape[-1]
        
        vram_used_cuda_kernel_total += torch.cuda.max_memory_allocated(device="cuda")
        
        del data, audio_cuda_kernel
        torch.cuda.empty_cache()

    # Calculate metrics
    audio_second = audio_length_total / h.sampling_rate
    khz_original = audio_length_total / toc_total_original / 1000
    khz_cuda_kernel = audio_length_total / toc_total_cuda_kernel / 1000
    vram_used_original_gb = vram_used_original_total / num_sample / (1024 ** 3)
    vram_used_cuda_kernel_gb = vram_used_cuda_kernel_total / num_sample / (1024 ** 3)

    # Print results
    print(
        f"Original BigVGAN: took {toc_total_original:.2f} seconds to generate {audio_second:.2f} seconds of audio, {khz_original:.1f}kHz, {audio_second / toc_total_original:.1f} faster than realtime, VRAM used {vram_used_original_gb:.1f} GB"
    )
    print(
        f"CUDA kernel BigVGAN: took {toc_total_cuda_kernel:.2f} seconds to generate {audio_second:.2f} seconds of audio, {khz_cuda_kernel:.1f}kHz, {audio_second / toc_total_cuda_kernel:.1f} faster than realtime, VRAM used {vram_used_cuda_kernel_gb:.1f} GB"
    )
    print(f"speedup of CUDA kernel: {khz_cuda_kernel / khz_original}")
    print(f"VRAM saving of CUDA kernel: {vram_used_original_gb / vram_used_cuda_kernel_gb}")

    # Use artificial sine waves for inference test
    audio_real, sr = generate_soundwave(duration=5.0, sr=h.sampling_rate)
    audio_real = torch.tensor(audio_real).to("cuda")
    # Compute mel spectrogram from the ground truth audio
    x = get_mel(audio_real.unsqueeze(0), h)

    with torch.inference_mode():
        y_g_hat_original = generator_original(x)
        y_g_hat_cuda_kernel = generator_cuda_kernel(x)

    audio_real = audio_real.squeeze()
    audio_real = audio_real * MAX_WAV_VALUE
    audio_real = audio_real.cpu().numpy().astype("int16")

    audio_original = y_g_hat_original.squeeze()
    audio_original = audio_original * MAX_WAV_VALUE
    audio_original = audio_original.cpu().numpy().astype("int16")

    audio_cuda_kernel = y_g_hat_cuda_kernel.squeeze()
    audio_cuda_kernel = audio_cuda_kernel * MAX_WAV_VALUE
    audio_cuda_kernel = audio_cuda_kernel.cpu().numpy().astype("int16")

    os.makedirs("tmp", exist_ok=True)
    output_file_real = os.path.join("tmp", "audio_real.wav")
    output_file_original = os.path.join("tmp", "audio_generated_original.wav")
    output_file_cuda_kernel = os.path.join("tmp", "audio_generated_cuda_kernel.wav")
    write(output_file_real, h.sampling_rate, audio_real)
    write(output_file_original, h.sampling_rate, audio_original)
    write(output_file_cuda_kernel, h.sampling_rate, audio_cuda_kernel)
    print("Example generated audios of original vs. fused CUDA kernel written to tmp!")
    print("Done")
