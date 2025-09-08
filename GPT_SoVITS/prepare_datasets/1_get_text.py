import enum
import os
import os.path as osp
import queue
import sys
import time
import warnings
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as tmp
import typer
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.multiprocessing.spawn import spawn
from transformers import BertForMaskedLM, BertTokenizerFast

from GPT_SoVITS.Accelerate.logger import SpeedColumnIteration, console, logger
from GPT_SoVITS.text.cleaner import clean_text
from tools.my_utils import clean_path

torch.set_grad_enabled(False)

tmp.set_start_method("spawn", force=True)

warnings.filterwarnings("ignore", category=UserWarning, module="jieba_fast._compat")


class Device(str, enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
)


def lang_map(lang: str) -> str:
    m = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    return m.get(lang, "")


def parse_inp_text_line(line: str) -> tuple[str, str, str]:
    wav_name, _, language, text = line.split("|", 3)
    return wav_name, language, text


def build_device_strings(device_type: str, device_ids: list[int], procs_per_device: int) -> list[str]:
    devices: list[str] = []
    for device_id in device_ids:
        dstr = f"{device_type}:{device_id}"
        devices.extend([dstr] * procs_per_device)
    return devices


def worker_entry(
    rank: int,
    device_strs: List[str],
    tasks_q: "tmp.Queue[tuple[int, str, str, str] | None]",
    results_q: "tmp.Queue[tuple[int, tuple[str, str, list[int] | None, str]]]",
    bert_pretrained_dir: str,
    opt_dir: str,
    fp16: bool,
    version: str | None,
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

    bert_dir = osp.join(opt_dir, "3-bert")
    os.makedirs(bert_dir, exist_ok=True)

    if not osp.exists(bert_pretrained_dir):
        raise FileNotFoundError(bert_pretrained_dir)

    tokenizer = BertTokenizerFast.from_pretrained(bert_pretrained_dir)
    bert_model = BertForMaskedLM.from_pretrained(bert_pretrained_dir, device_map=device)

    if fp16:
        bert_model = bert_model.half()

    def get_bert_feature(text: str, word2ph: list[int]) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        out: torch.Tensor = bert_model(**inputs, output_hidden_states=True).hidden_states  # type: ignore
        layer = out[-3][0].cpu()[1:-1]  # [seq-2, hid]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            phone_level_feature.append(layer[i].repeat(word2ph[i], 1))
        feats = torch.cat(phone_level_feature, dim=0)  # [phones, hid]
        return feats.T  # [hid, phones]

    i = 0
    while True:
        item = tasks_q.get()
        if item is None:
            break

        idx, wav_name, language, text = item

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
            mapped_lang = lang_map(language)
            if not mapped_lang:
                logger.warning(f"[W{rank}] Unsupported language: {language} of {wav_name}")
                results_q.put((idx, ("", "", [], "")))
                continue

            phones, word2ph, norm_text = clean_text(
                text.replace("%", "-").replace("￥", ","),
                mapped_lang,
                version,
            )

            if mapped_lang == "zh":
                path_bert = osp.join(bert_dir, f"{name}.pt")
                if not osp.exists(path_bert):
                    assert word2ph
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    torch.save(bert_feature, path_bert)

            phones_str = " ".join(phones)
            results_q.put((idx, (name, phones_str, word2ph, norm_text)))
        except Exception as e:
            del (
                device_str,
                tokenizer,
                bert_model,
                bert_dir,
                bert_pretrained_dir,
                tasks_q,
                results_q,
                opt_dir,
                item,
                idx,
                i,
            )
            logger.exception(f"[W{rank}] Failed: {wav_name} | {text}")
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
        help="list File: wav|spk|lang|text",
    ),
    opt: Path = typer.Option(
        ..., "--opt", file_okay=False, dir_okay=True, writable=True, show_default=False, help="Output Directory"
    ),
    bert: Path = typer.Option(
        ..., "--bert", exists=True, readable=True, show_default=False, help="Path to Bert Pretrained Models"
    ),
    version: str = typer.Option("v2", "--version", help="SoVITS Language Version"),
    device: Device = typer.Option(Device.cpu, "--device", help="Compute device"),
    device_id: str = typer.Option("0", "--device-id", help="CUDA_VISIBLE_DEVICE, Such as '0,1,2'"),
    nproc: int = typer.Option(1, "--nproc", min=1, help="Number of processes per GPU"),
    fp16: bool = typer.Option(False, is_flag=True, flag_value=True, help="Use FP16"),
):
    device_ids = [int(x) for x in device_id.split(",") if x.strip() != ""]
    if device in {"cpu", "mps"} and device_ids != [0]:
        raise ValueError(f"Invalid Device ID {device_ids}")
    if nproc < 1:
        raise ValueError(f"Invalid Num Process {nproc}")

    os.makedirs(opt, exist_ok=True)
    merged_path = osp.join(opt, "2-name2text.txt")

    with open(inp_list, "r", encoding="utf8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]

    tasks_all: list[tuple[int, str, str, str]] = []
    for idx, line in enumerate(lines):
        try:
            wav_name, language, text = parse_inp_text_line(line)
            tasks_all.append((idx, wav_name, language, text))
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

    tasks_q: "tmp.Queue[tuple[int, str, str, str] | None]" = tmp.Queue()
    results_q: "tmp.Queue[tuple[int, tuple[str, str, list[int] | None, str]]]" = tmp.Queue()

    for task in tasks_all:
        tasks_q.put(task)
    for _ in range(world_size):
        tasks_q.put(None)

    ordered: list[tuple[str, str, list[int] | None, str]] = [("", "", [], "")] * n_tasks
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
        progress_task = progress.add_task("G2P & Extract Bert", total=n_tasks)

        ctx = spawn(
            worker_entry,
            args=(device_strs, tasks_q, results_q, bert, opt, fp16, version),
            nprocs=world_size,
            join=False,
            daemon=False,
        )
        assert ctx

        while completed < n_tasks:
            try:
                idx, tup = results_q.get(timeout=0.01)
                ordered[idx] = tup
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
        for name, phones_str, word2ph, norm_text in ordered:
            if name:
                fout.write(f"{name}\t{phones_str}\t{word2ph}\t{norm_text}\n")

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
