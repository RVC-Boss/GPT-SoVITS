import enum
import os
import os.path as osp
import queue
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.multiprocessing as tmp
import torchaudio
import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from scipy.io import wavfile
from torch.multiprocessing.spawn import spawn

from GPT_SoVITS.Accelerate.logger import SpeedColumnIteration, console, logger
from GPT_SoVITS.eres2net.ERes2NetV2 import ERes2NetV2
from GPT_SoVITS.feature_extractor import cnhubert as cnhubert_mod
from tools.my_utils import clean_path, load_audio

warnings.filterwarnings("ignore", message=".*ComplexHalf support is experimental.*")

torch.set_grad_enabled(False)

tmp.set_start_method("spawn", force=True)

MAXX = 0.95
ALPHA = 0.5


class Device(str, enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


class SV:
    def __init__(self, device: torch.device, fp16: bool, sv_path: str):
        pretrained_state = torch.load(sv_path, map_location="cpu")
        embedding_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
        embedding_model.load_state_dict(pretrained_state)
        embedding_model.eval()
        self.embedding_model = embedding_model
        self.dtype = torch.float16 if fp16 else torch.float32
        if fp16 is False:
            self.embedding_model = self.embedding_model.to(device)
        else:
            self.embedding_model = self.embedding_model.half().to(device)

    def compute_embedding(self, wav: torch.Tensor):
        if not torch.cuda.is_available():
            wav = wav.float()
        feat = torch.stack(
            [
                torchaudio.compliance.kaldi.fbank(wav0.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0)
                for wav0 in wav
            ]
        ).to(self.dtype)
        sv_emb: torch.Tensor = self.embedding_model.forward3(feat)
        return sv_emb


def parse_inp_text_line(line: str) -> str:
    wav_name, _, __, ___ = line.split("|", 3)
    return wav_name


def build_device_strings(device_type: str, device_ids: List[int], procs_per_device: int) -> List[str]:
    devices: List[str] = []
    for device_id in device_ids:
        dstr = f"{device_type}:{device_id}"
        devices.extend([dstr] * procs_per_device)
    return devices


def worker_entry(
    rank: int,
    device_strs: List[str],
    tasks_q: "tmp.Queue[tuple[int, str] | None]",
    results_q: "tmp.Queue[int]",
    cnhubert_base_dir: str,
    sv: Optional[str],
    opt_dir: str,
    fp16: bool,
):
    device_str = device_strs[rank]
    device = torch.device(device_str)

    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_device(device.index)
    elif device.type == "mps":
        assert torch.mps.is_available()
    elif device.type == "xpu":
        assert torch.xpu.is_available()

    hubert_dir = osp.join(opt_dir, "4-cnhubert")
    wav32dir = osp.join(opt_dir, "5-wav32k")
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    if not osp.exists(cnhubert_base_dir):
        raise FileNotFoundError(f"CNHuBERT Base Dir not found: {cnhubert_base_dir}")
    cnhubert_mod.cnhubert_base_path = cnhubert_base_dir

    model = cnhubert_mod.get_model()
    resample = torchaudio.transforms.Resample(32000, 16000)

    if fp16:
        model = model.half().to(device)
        resample = resample.half().to(device)
    else:
        model = model.to(device)
        resample = resample.to(device)

    sv_model: SV | None = None

    sv_cn_dir = osp.join(opt_dir, "7-sv_cn")
    if sv:
        os.makedirs(sv_cn_dir, exist_ok=True)
        extract_sv = True
        sv_model = SV(device, fp16, sv)
    else:
        extract_sv = False

    def process_one_item(
        wav_name: str,
        wav_path: str,
        model_: cnhubert_mod.CNHubert,
        resample_: torchaudio.transforms.Resample,
        use_fp16: bool = False,
        extract_sv: bool = False,
    ) -> bool:
        hubert_path = osp.join(hubert_dir, f"{wav_name}.pt")
        if osp.exists(hubert_path):
            return False

        tmp_audio = load_audio(wav_path, 32000)
        tmp_max = float(np.abs(tmp_audio).max()) if tmp_audio.size > 0 else 0.0
        if tmp_max <= 0:
            logger.warning(f"[W{rank}] Filtered: Empty or silent audio: {wav_path}")
            return False
        if tmp_max > 2.2:
            logger.warning(f"[W{rank}] Filtered: peak={tmp_max:.3f}")
            return False

        tmp_audio32 = (tmp_audio / tmp_max * (MAXX * ALPHA * 32768.0)) + ((1.0 - ALPHA) * 32768.0) * tmp_audio

        if extract_sv:
            assert sv_cn_dir
            assert sv_model
            sv_path = osp.join(sv_cn_dir, f"{wav_name}.pt")
            if not osp.exists(sv_path):
                tmp_audio32_sv = (tmp_audio / tmp_max * (MAXX * ALPHA)) + (1.0 - ALPHA) * tmp_audio
                tensor_wav32_sv = torch.from_numpy(tmp_audio32_sv).to(device)
                if use_fp16:
                    tensor_wav32_sv = tensor_wav32_sv.half()
                tensor_wav16_sv: torch.Tensor = resample_(tensor_wav32_sv)
                out_sv = sv_model.compute_embedding(tensor_wav16_sv.unsqueeze(0)).cpu()
                torch.save(out_sv, sv_path)

        tensor_wav32 = torch.from_numpy(tmp_audio32).to(device)

        if use_fp16:
            tensor_wav32 = tensor_wav32.half()

        tensor_wav16 = resample_(tensor_wav32)

        out: torch.Tensor = model_.model(tensor_wav16.unsqueeze(0))["last_hidden_state"]  # [1, T, 768]
        ssl = out.transpose(1, 2).contiguous().cpu()  # [1, 768, T]

        if torch.isnan(ssl).any():
            return True

        wavfile.write(
            osp.join(wav32dir, f"{osp.splitext(wav_name)[0]}.wav"),
            32000,
            tmp_audio32.astype(np.int16),
        )

        torch.save(ssl, hubert_path)
        return False

    i = 0
    while True:
        item = tasks_q.get()
        if item is None:
            break

        idx, wav_path = item

        i += 1
        if i % 10 == 0:
            match device.index:
                case "cuda":
                    torch.cuda.empty_cache()
                case "mps":
                    torch.mps.empty_cache()
                case "xpu":
                    torch.xpu.empty_cache()

        try:
            name = clean_path(osp.basename(wav_path))

            is_nan = process_one_item(
                wav_name=name,
                wav_path=wav_path,
                model_=model,
                resample_=resample,
                use_fp16=fp16,
                extract_sv=extract_sv,
            )

            if is_nan and fp16:
                model = model.float()
                resample = resample.float()
                is_nan = process_one_item(
                    wav_name=name,
                    wav_path=wav_path,
                    model_=model,
                    resample_=resample,
                    use_fp16=False,
                    extract_sv=False,
                )
                if is_nan:
                    logger.error(f"[W{rank}] Failed: NaN Audio {name}")

                model = model.half()
                resample = resample.half()

        except Exception as e:
            del (
                device_str,
                hubert_dir,
                wav32dir,
                cnhubert_base_dir,
                tasks_q,
                results_q,
                opt_dir,
                model,
                resample,
                sv_cn_dir,
                sv_model,
                device_strs,
                idx,
                sv,
                i,
            )
            logger.exception(f"[W{rank}] Failed: {wav_path}")
            raise e

        results_q.put(idx)

    sys.exit(0)


@app.command()
def main(
    inp_list: Path = typer.Option(
        ...,
        "--inp-list",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        show_default=False,
        help="list File: wav|spk|lang|text",
    ),
    wav_dir: Optional[Path] = typer.Option(
        None, "--wav-dir", file_okay=False, dir_okay=True, readable=True, show_default=False, help="Wav Audio Dir"
    ),
    opt: Path = typer.Option(
        ..., "--opt", file_okay=False, dir_okay=True, writable=True, show_default=False, help="Output Directory"
    ),
    cnhubert_dir: Path = typer.Option(
        ...,
        "--cnhubert",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        show_default=False,
        help="Path to CNHuBERT Pretrained Models",
    ),
    sv: Optional[Path] = typer.Option(
        None,
        "--sv",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        show_default=False,
        help="(optional) SV Model Path, If Set, Extract SV Embeddings",
    ),
    device: Device = typer.Option(Device.cpu, "--device", help="Compute device"),
    device_id: str = typer.Option("0", "--device-id", help="CUDA_VISIBLE_DEVICE, Such as '0,1,2'"),
    nproc: int = typer.Option(1, "--nproc", min=1, help="Number of processes per GPU"),
    fp16: bool = typer.Option(False, is_flag=True, flag_value=True, help="Use FP16"),
):
    device_ids = [int(x) for x in device_id.split(",") if x.strip() != ""]
    if device in {"cpu", "mps"} and device_ids != [0]:
        raise ValueError(f"Invalid Device IDs for {device=}: {device_ids}")
    if nproc < 1:
        raise ValueError(f"Invalid nproc: {nproc}")

    os.makedirs(opt, exist_ok=True)

    with open(inp_list, "r", encoding="utf8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    tasks_all: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        try:
            wav_name = parse_inp_text_line(line)
            if wav_dir:
                wav_name = clean_path(osp.basename(wav_name))
                wav_path = osp.join(str(wav_dir), wav_name)
            else:
                wav_path = wav_name
            tasks_all.append((idx, wav_path))
        except Exception:
            logger.exception(f"Skip line {idx}: {line}")

    n_tasks = len(tasks_all)
    if n_tasks == 0:
        logger.warning("Empty list. Nothing to do.")
        return

    device_strs = build_device_strings(device, device_ids, nproc)
    world_size = len(device_strs)

    tasks_q: "tmp.Queue[tuple[int, str] | None]" = tmp.Queue()
    results_q: "tmp.Queue[int]" = tmp.Queue()

    for task in tasks_all:
        tasks_q.put(task)
    for _ in range(world_size):
        tasks_q.put(None)

    completed = 0

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        SpeedColumnIteration(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True),
        console=console,
        redirect_stderr=False,
        redirect_stdout=False,
    ) as progress:
        if sv:
            progress_task = progress.add_task("Extract CNHuBERT/SV & Save Wav 32k", total=n_tasks)
        else:
            progress_task = progress.add_task("Extract CNHuBERT & Save Wav 32k", total=n_tasks)

        ctx = spawn(
            worker_entry,
            args=(device_strs, tasks_q, results_q, cnhubert_dir, sv, opt, fp16),
            nprocs=world_size,
            join=False,
            daemon=False,
        )
        assert ctx is not None

        while completed < n_tasks:
            try:
                _ = results_q.get(timeout=0.01)
                completed += 1
                progress.update(progress_task, advance=1)
            except queue.Empty:
                pass

            for p in ctx.processes:
                if p is None:
                    continue
                if (p.exitcode is not None and p.exitcode != 0) or (not p.is_alive()):
                    progress.live.stop()
                    try:
                        ctx.join()
                    except Exception as e:
                        console.print(e)
                    finally:
                        logger.critical(f"Worker PID {p.pid} crashed with exit code {p.exitcode}.")
                        sys.exit(1)

        ctx.join()

    logger.info(f"Done. Output dir: {opt}")


def is_powershell_env(env: dict) -> bool:
    return any(k in env for k in ("PSHOME", "POWERSHELL_DISTRIBUTION_CHANNEL", "PSModulePath"))


def get_prog_name() -> str:
    script_rel = ".".join(["GPT_SoVITS", "prepare_datasets", osp.basename(__file__)]).strip(".py")
    return f"python -s -m {script_rel}"


if __name__ == "__main__":
    t = time.perf_counter()
    app(prog_name=get_prog_name())
    logger.info(f"Exec Time: {time.perf_counter() - t:.2f} secs")
