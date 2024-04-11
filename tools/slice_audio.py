# modified from https://github.com/fishaudio/audio-preprocess/blob/main/fish_audio_preprocess/cli/slice_audio.py

from pathlib import Path
from typing import Union, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import librosa
import numpy as np
import soundfile as sf
import math
from my_utils import load_audio
import argparse
import os


AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}


def list_files(
    path: Union[Path, str],
    extensions: set[str] = None,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory.

    Args:
        path (Path): Path to the directory.
        extensions (set, optional): Extensions to filter. Defaults to None.
        sort (bool, optional): Whether to sort the files. Defaults to True.

    Returns:
        list: List of files.
    """

    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Directory {path} does not exist.")

    files = [f for f in path.glob("*") if f.is_file()]

    if extensions is not None:
        files = [f for f in files if f.suffix in extensions]

    if sort:
        files = sorted(files)
    
    return files


def make_dirs(path: Union[Path, str]):
    """Make directories.

    Args:
        path (Union[Path, str]): Path to the directory.
    """
    if isinstance(path, str):
        path = Path(path)

    if path.exists():
       print(f"Output directory already exists: {path}")

    path.mkdir(parents=True, exist_ok=True)


def slice_audio_v2_(
    input_path: str,
    output_dir: str,
    num_workers: int,
    min_duration: float,
    max_duration: float,
    min_silence_duration: float,
    top_db: int,
    hop_length: int,
    max_silence_kept: float,
    merge_short:bool
):
    """(OpenVPI version) Slice audio files into smaller chunks by silence."""

    input_path_, output_dir_ = Path(input_path), Path(output_dir)
    if not input_path_.exists():
        raise RuntimeError("You input a wrong audio path that does not exists, please fix it!")
    make_dirs(output_dir_)
    if input_path_.is_dir():
        files = list_files(input_path_, extensions=AUDIO_EXTENSIONS)
    elif input_path_.is_file() and input_path_.suffix in AUDIO_EXTENSIONS:
        files = [input_path_]
        input_path_ = input_path_.parent
    else:
        raise RuntimeError("The input path is not file or dir, please fixes it")     
    print(f"Found {len(files)} files, processing...")


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        tasks = []

        for file in tqdm(files, desc="Preparing tasks"):
            # Get relative path to input_dir
            relative_path = file.relative_to(input_path_)
            save_path = output_dir_ / relative_path.parent / relative_path.stem

            tasks.append(
                executor.submit(
                    slice_audio_file_v2,
                    input_file=str(file),
                    output_dir=save_path,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    min_silence_duration=min_silence_duration,
                    top_db=top_db,
                    hop_length=hop_length,
                    max_silence_kept=max_silence_kept,
                    merge_short=merge_short
                )
            )

        for i in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            assert i.exception() is None, i.exception()

    print("Done!")
    print(f"Total: {len(files)}")
    print(f"Output directory: {output_dir}")


def slice_audio_file_v2(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    min_duration: float = 4.0,
    max_duration: float = 12.0,
    min_silence_duration: float = 0.3,
    top_db: int = -40,
    hop_length: int = 10,
    max_silence_kept: float = 0.4,
    merge_short: bool = False
) -> None:
    """
    Slice audio by silence and save to output folder

    Args:
        input_file: input audio file
        output_dir: output folder
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        top_db: threshold to detect silence
        hop_length: hop length to detect silence
        max_silence_kept: maximum duration of silence to be kept
    """

    output_dir = Path(output_dir)

    audio = load_audio(str(input_file),32000)
    rate = 32000
    for idx, sliced in enumerate(
        slice_audio_v2(
            audio,
            rate,
            min_duration=min_duration,
            max_duration=max_duration,
            min_silence_duration=min_silence_duration,
            top_db=top_db,
            hop_length=hop_length,
            max_silence_kept=max_silence_kept,
            merge_short=merge_short
        )
    ):
        if len(sliced) <= 3*rate: continue
        max_audio=np.abs(sliced).max()#防止爆音,懒得搞混合了,后面有响度匹配
        if max_audio>1: 
            sliced/=max_audio
        sf.write(str(output_dir) + f"_{idx:04d}.wav", sliced, rate)


def slice_audio_v2(
    audio: np.ndarray,
    rate: int,
    min_duration: float = 4.0,
    max_duration: float = 12.0,
    min_silence_duration: float = 0.3,
    top_db: int = -40,
    hop_length: int = 10,
    max_silence_kept: float = 0.5,
    merge_short: bool = False
) -> Iterable[np.ndarray]:
    """Slice audio by silence

    Args:
        audio: audio data, in shape (samples, channels)
        rate: sample rate
        min_duration: minimum duration of each slice
        max_duration: maximum duration of each slice
        min_silence_duration: minimum duration of silence
        top_db: threshold to detect silence
        hop_length: hop length to detect silence
        max_silence_kept: maximum duration of silence to be kept
        merge_short: merge short slices automatically

    Returns:
        Iterable of sliced audio
    """

    if len(audio) / rate < min_duration:
        sliced_by_max_duration_chunk = slice_by_max_duration(audio, max_duration, rate)
        yield from merge_short_chunks(
            sliced_by_max_duration_chunk, max_duration, rate
        ) if merge_short else sliced_by_max_duration_chunk
        return

    slicer = Slicer(
        sr=rate,
        threshold=top_db,
        min_length=min_duration * 1000,
        min_interval=min_silence_duration * 1000,
        hop_size=hop_length,
        max_sil_kept=max_silence_kept * 1000,
    )

    sliced_audio = slicer.slice(audio)
    if merge_short:
        sliced_audio = merge_short_chunks(sliced_audio, max_duration, rate)

    for chunk in sliced_audio:
        sliced_by_max_duration_chunk = slice_by_max_duration(chunk, max_duration, rate)
        yield from sliced_by_max_duration_chunk


def slice_by_max_duration(
    gen: np.ndarray, slice_max_duration: float, rate: int
) -> Iterable[np.ndarray]:
    """Slice audio by max duration

    Args:
        gen: audio data, in shape (samples, channels)
        slice_max_duration: maximum duration of each slice
        rate: sample rate

    Returns:
        generator of sliced audio data
    """

    if len(gen) > slice_max_duration * rate:
        # Evenly split _gen into multiple slices
        n_chunks = math.ceil(len(gen) / (slice_max_duration * rate))
        chunk_size = math.ceil(len(gen) / n_chunks)

        for i in range(0, len(gen), chunk_size):
            yield gen[i : i + chunk_size]
    else:
        yield gen


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 4000,
        min_interval: int = 300,
        hop_size: int = 10,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )

        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )

        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        if samples.shape[0] <= self.min_length:
            return [waveform]

        rms_list = librosa.feature.rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0

        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue

            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue

            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start

                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))

                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))

                clip_start = pos_r
            silence_start = None

        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []

            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )

            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )

            return chunks
        

# def merge_short_chunks(chunks, max_duration, rate):
#     merged_chunks = []
#     buffer, length = [], 0
#     lengths = [len(chunk)/rate for chunk in chunks]
#     print(lengths)
#     for chunk in chunks:
#         if length + len(chunk) > max_duration * rate and len(buffer) > 0:
#             print(len(buffer))
#             merged_chunks.append(np.concatenate(buffer))
#             buffer, length = [], 0
#         else:
#             buffer.append(chunk)
#             length += len(chunk)
            

#     if len(buffer) > 0:
#         print(len(buffer))
#         merged_chunks.append(np.concatenate(buffer))
    
#     print([len(chunk)/rate for chunk in merged_chunks])

#     return merged_chunks


def merge_short_chunks(chunks, max_duration, rate):
    if not chunks:
        return []
    
    max_length = int(max_duration * rate)  
    merged = []
    current = chunks[0]  
    for chunk in chunks[1:]:  
        if len(current) + len(chunk) <= max_length:
            current = np.concatenate((current, np.zeros(int(0.1*rate)), chunk))  # 在合并前后加入一个0.1s作为间隔
        else:
            merged.append(current)
            current = chunk  

    merged.append(current)  # 添加最后一个块
    return merged










        
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_dir",help="切割输入文件夹")
parser.add_argument("-o","--output_dir",help="切割输入文件夹")
parser.add_argument("--threshold",default=-40,help="音量小于这个值视作静音的备选切割点")
parser.add_argument("--min_duration",default=4,help="每段最短多长，如果第一段太短一直和后面段连起来直到超过这个值")
parser.add_argument("--max_duration",default=12,help="每段最长多长")
parser.add_argument("--min_interval",default=0.3,help="最短切割间隔")
parser.add_argument("--hop_size",default=10,help="怎么算音量曲线，越小精度越大计算量越高（不是精度越大效果越好）")
parser.add_argument("--max_sil_kept",default=0.4,help="切完后静音最多留多长")
parser.add_argument("--num_worker",default=os.cpu_count(),help="切使用的进程数")
parser.add_argument("--merge_short",default="False",help="响割使用的进程数")


args = parser.parse_args()
input_path = args.input_dir
output_dir =  args.output_dir
threshold = float(args.threshold)
min_duration = float(args.min_duration)
max_duration = float(args.max_duration)
min_interval = float(args.min_interval)
hop_size = float(args.hop_size)
max_sil_kept = float(args.max_sil_kept)
num_worker = int(args.num_worker)
merge_short = eval(args.merge_short)

if __name__ == "__main__":
    slice_audio_v2_(input_path, output_dir, num_worker, min_duration, max_duration, min_interval, threshold, hop_size, max_sil_kept,merge_short)
