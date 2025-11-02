import enum
import gc
import os
import os.path as osp
import queue
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.multiprocessing as tmp
import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.multiprocessing.spawn import spawn

from GPT_SoVITS.Accel.logger import SpeedColumnIteration, console, logger
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3
from GPT_SoVITS.process_ckpt import inspect_version
from tools.my_utils import DictToAttrRecursive, clean_path

torch.set_grad_enabled(False)

tmp.set_start_method("spawn", force=True)


class Device(str, enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


def parse_inp_text_line(line: str) -> str:
    wav_name, _, __, ___ = line.split("|", 3)
    return wav_name


def build_device_strings(device_type: str, device_ids: List[int], procs_per_device: int) -> List[str]:
    devices: List[str] = []
    for device_id in device_ids:
        dstr = f"{device_type}:{device_id}" if device_type in {"cuda", "mps"} else "cpu"
        devices.extend([dstr] * procs_per_device)
    return devices


def worker_entry(
    rank: int,
    device_strs: List[str],
    tasks_q: "tmp.Queue[Tuple[int, str] | None]",
    results_q: "tmp.Queue[Tuple[int, str]]",
    pretrained_s2g: str,
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
    os.makedirs(opt_dir, exist_ok=True)

    if not osp.exists(hubert_dir):
        raise FileNotFoundError(hubert_dir)

    version, _, _, hps_dict, dict_s2 = inspect_version(pretrained_s2g)
    hps = DictToAttrRecursive(hps_dict)

    if version in {"v3", "v4"}:
        vq_model: SynthesizerTrn | SynthesizerTrnV3 = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version=version,
            **hps.model,
        )
    else:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version=version,
            **hps.model,
        )

    load_result = vq_model.load_state_dict(dict_s2["weight"])
    if rank == 0:
        console.print("")
        console.print(load_result)

    for name in list(vq_model._modules.keys()):
        if name not in ["quantizer", "ssl_proj"]:
            del vq_model._modules[name]
    del dict_s2

    if fp16:
        vq_model = vq_model.to(device).half()
    else:
        vq_model.to(device)

    match device.index:
        case "cuda":
            torch.cuda.empty_cache()
        case "mps":
            torch.mps.empty_cache()
        case "xpu":
            torch.xpu.empty_cache()
    gc.collect()

    def extract_semantic_from_hubert_pt(wav_basename: str) -> str | None:
        hubert_path = osp.join(hubert_dir, f"{wav_basename}.pt")
        if not osp.exists(hubert_path):
            return None

        ssl_content: torch.Tensor = torch.load(hubert_path, map_location="cpu")
        if fp16:
            ssl_content = ssl_content.half().to(device)
        else:
            ssl_content = ssl_content.to(device)

        codes = vq_model.extract_latent(ssl_content)
        vec = codes[0, 0, :].tolist()

        return " ".join(str(i) for i in vec)

    i = 0
    while True:
        item = tasks_q.get()
        if item is None:
            break

        idx, wav_name = item

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
            name = clean_path(osp.basename(wav_name))
            semantic = extract_semantic_from_hubert_pt(name)
            if semantic is None:
                results_q.put((idx, ""))
            else:
                results_q.put((idx, f"{name}\t{semantic}"))
        except Exception as e:
            del device_str, vq_model, hubert_dir, _, version, hps_dict, hps, item, idx, i, load_result
            logger.exception(f"[W{rank}] Failed on: {wav_name}")
            raise e

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
        help="list file: wav|spk|lang|text",
    ),
    opt: Path = typer.Option(
        ..., "--opt", file_okay=False, dir_okay=True, writable=True, show_default=False, help="Output Directory"
    ),
    pretrained_s2g: Path = typer.Option(
        ...,
        "--pretrained-s2g",
        file_okay=True,
        dir_okay=False,
        exists=True,
        readable=True,
        show_default=False,
        help="Path to pretrained s2G checkpoint",
    ),
    device: Device = typer.Option(Device.cpu, "--device", help="Compute device"),
    device_id: str = typer.Option("0", "--device-id", help="CUDA_VISIBLE_DEVICES style, e.g. '0,1'"),
    nproc: int = typer.Option(1, "--nproc", min=1, help="Processes per device"),
    fp16: bool = typer.Option(True, is_flag=True, flag_value=True, help="Use FP16 on CUDA"),
):
    device_ids = [int(x) for x in device_id.split(",") if x.strip() != ""]
    if device in {"cpu", "mps"} and device_ids != [0]:
        raise ValueError(f"Invalid Device ID {device_ids}")
    if nproc < 1:
        raise ValueError(f"Invalid Num Process {nproc}")

    os.makedirs(opt, exist_ok=True)
    merged_path = osp.join(opt, "6-name2semantic.tsv")

    with open(inp_list, "r", encoding="utf8") as f:
        raw_lines = [ln for ln in f.read().splitlines() if ln.strip()]

    tasks_all: List[Tuple[int, str]] = []
    for idx, line in enumerate(raw_lines):
        try:
            wav_name = parse_inp_text_line(line)
            tasks_all.append((idx, wav_name))
        except Exception:
            logger.exception(f"Skip line {idx}: {line}")

    n_tasks = len(tasks_all)
    if n_tasks == 0:
        logger.warning("Empty list")
        with open(merged_path, "w", encoding="utf8") as fout:
            pass
        return

    device_strs = build_device_strings(device, device_ids, nproc)
    world_size = len(device_strs)

    tasks_q: "tmp.Queue[Tuple[int, str] | None]" = tmp.Queue()
    results_q: "tmp.Queue[Tuple[int, str]]" = tmp.Queue()

    for task in tasks_all:
        tasks_q.put(task)
    for _ in range(world_size):
        tasks_q.put(None)

    ordered: List[str] = [""] * n_tasks
    completed = 0

    with Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        SpeedColumnIteration(show_speed=True),
        TimeRemainingColumn(elapsed_when_finished=True),
        console=console,
    ) as progress:
        progress_task = progress.add_task("Extract Semantic Codes", total=n_tasks)

        ctx = spawn(
            worker_entry,
            args=(device_strs, tasks_q, results_q, pretrained_s2g, opt, fp16),
            nprocs=world_size,
            join=False,
            daemon=False,
        )
        assert ctx

        while completed < n_tasks:
            try:
                idx, line = results_q.get(timeout=0.05)
                if line:
                    ordered[idx] = line
                completed += 1
                progress.update(progress_task, advance=1)
            except queue.Empty:
                pass

            for p in ctx.processes:
                assert p
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

    with open(merged_path, "w", encoding="utf8") as fout:
        for line in ordered:
            if line:
                fout.write(line + "\n")

    logger.info(f"Done: {merged_path}")


def is_powershell_env(env: dict) -> bool:
    return any(k in env for k in ("PSHOME", "POWERSHELL_DISTRIBUTION_CHANNEL", "PSModulePath"))


def get_prog_name() -> str:
    script_rel = ".".join(["GPT_SoVITS", "prepare_datasets", osp.basename(__file__)]).strip(".py")
    return f"python -s -m {script_rel}"


if __name__ == "__main__":
    t = time.perf_counter()
    app(prog_name=get_prog_name())
    logger.info(f"Exec Time: {time.perf_counter() - t:.2f} secs")
