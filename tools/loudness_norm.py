# modifiled from https://github.com/fishaudio/audio-preprocess/blob/main/fish_audio_preprocess/cli/loudness_norm.py

from pathlib import Path
from typing import Union
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os


def loudness_norm_(
    input_dir: str,
    peak: float,
    loudness: float,
    block_size: float,
    num_workers: int,
):
    """Perform loudness normalization (ITU-R BS.1770-4) on audio files."""
    
    if isinstance(input_dir, str):
        path = Path(input_dir)
    input_dir, output_dir = Path(input_dir), Path(input_dir)
    
    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")
    files = (
       [f for f in path.glob("*") if f.is_file() and f.suffix == ".wav"]
    )
    

    print(f"Found {len(files)} files, normalizing loudness")


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_dir)
            new_file = output_dir / relative_path

            if new_file.parent.exists() is False:
                new_file.parent.mkdir(parents=True)
            
            tasks.append(
                executor.submit(
                    loudness_norm_file, file, new_file, peak, loudness, block_size
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    print("Done!")



def loudness_norm_file(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    peak=-1.0,
    loudness=-23.0,
    block_size=0.400,
) -> None:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        input_file: input audio file
        output_file: output audio file
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)
    """

    # Thanks to .against's feedback
    # https://github.com/librosa/librosa/issues/1236

    input_file, output_file = str(input_file), str(output_file)

    audio, rate = sf.read(input_file)
    audio = loudness_norm(audio, rate, peak, loudness, block_size)
    sf.write(output_file, audio, rate)




def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)

    Returns:
        loudness normalized audio
    """

    # peak normalize audio to [peak] dB
    audio = pyln.normalize.peak(audio, peak)

    # measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    return pyln.normalize.loudness(audio, _loudness, loudness)




parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_dir",help="匹配响度输入文件夹")
parser.add_argument("-l","--loudness",help="响度")
parser.add_argument("-p","--peak",help="响度峰值")
parser.add_argument("-n","--num_worker")
args = parser.parse_args()
input_dir = args.input_dir
loudness = float(args.loudness)
peak = float(args.peak)
num_worker = int(args.num_worker) 

if __name__ == "__main__":
    loudness_norm_(input_dir=input_dir,peak=peak,loudness=loudness,block_size=0.4,num_workers=num_worker)