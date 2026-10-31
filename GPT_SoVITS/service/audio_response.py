"""Audio container encoding for HTTP responses."""

import subprocess
import sys
import threading
import wave
from io import BytesIO

import numpy as np
import soundfile as sf

if "tools.portable_runtime" in sys.modules:
    from tools.portable_runtime import FFMPEG_EXE
else:
    try:
        from GPT_SoVITS.runtime.portable import FFMPEG_EXE
    except ImportError:
        from tools.portable_runtime import FFMPEG_EXE


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    """Write OGG in a large-stack thread to avoid libsndfile stack overflow."""

    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as error:
        print("RuntimeError: {}".format(error))
        print("Changing the thread stack size is unsupported.")
    except ValueError as error:
        print("ValueError: {}".format(error))
        print("The specified stack size is invalid.")
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            str(FFMPEG_EXE),
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    """Create the single RIFF header prepended to a streaming WAV response."""

    wav_buffer = BytesIO()
    with wave.open(wav_buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(frame_input)
    wav_buffer.seek(0)
    return wav_buffer.read()
