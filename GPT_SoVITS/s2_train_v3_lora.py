import logging
import os
import platform
import sys
import warnings
from collections import OrderedDict as od
from contextlib import nullcontext
from random import randint
from typing import Any

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from rich import print
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import GPT_SoVITS.utils as utils
from GPT_SoVITS.Accelerate import console, logger
from GPT_SoVITS.Accelerate.logger import SpeedColumnIteration
from GPT_SoVITS.module import commons
from GPT_SoVITS.module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollateV3,
    TextAudioSpeakerCollateV4,
    TextAudioSpeakerLoaderV3,
    TextAudioSpeakerLoaderV4,
)
from GPT_SoVITS.module.models import SynthesizerTrnV3
from GPT_SoVITS.process_ckpt import save_ckpt

hps = utils.get_hparams(stage=2)

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False  # 反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
torch.set_grad_enabled(True)

global_step = 0
save_root = ""
no_grad_names: set[Any] = set()
lora_rank = 0

if torch.cuda.is_available():
    device_str = "cuda"
elif torch.mps.is_available():
    device_str = "mps"
else:
    device_str = "cpu"


multigpu = torch.cuda.device_count() > 1 if torch.cuda.is_available() else False


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"

    spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step, save_root, no_grad_names, lora_rank
    device = torch.device(f"{device_str}:{rank}")

    if rank == 0:
        logger.add(
            os.path.join(hps.data.exp_dir, "train.log"),
            level="INFO",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YY-MM-DD HH:mm:ss}\t{name}\t{level}\t{message}",
        )
        console.print(hps.to_dict())
        writer: SummaryWriter | None = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval: SummaryWriter | None = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))
    else:
        writer = writer_eval = None

    if multigpu:
        dist.init_process_group(
            backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
            init_method="env://",
            world_size=n_gpus,
            rank=rank,
        )

    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    TextAudioSpeakerLoader = TextAudioSpeakerLoaderV3 if hps.model.version == "v3" else TextAudioSpeakerLoaderV4
    TextAudioSpeakerCollate = TextAudioSpeakerCollateV3 if hps.model.version == "v3" else TextAudioSpeakerCollateV4
    train_dataset = TextAudioSpeakerLoader(hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [
            32,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=6,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )
    save_root = f"{hps.data.exp_dir}/logs_s2_{hps.model.version}_lora_{hps.train.lora_rank}"
    os.makedirs(save_root, exist_ok=True)
    lora_rank = int(hps.train.lora_rank)
    lora_config = LoraConfig(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights=True,
    )

    def get_model(hps):
        return SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

    def get_optim(net_g):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net_g.parameters()),  ###默认所有层lr一致
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

    def model2DDP(net_g, rank):
        if multigpu:
            net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        else:
            pass
        return net_g

    try:  # 如果能加载自动resume
        net_g = get_model(hps)
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2DDP(net_g, rank)
        optim_g = get_optim(net_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(save_root, "G_*.pth"),
            net_g,
            optim_g,
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:  # 如果首次不能加载，加载pretrain
        epoch_str = 1
        global_step = 0
        net_g = get_model(hps)
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G is not None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info(f"loaded pretrained {hps.train.pretrained_s2G}")
            console.print(
                f"loaded pretrained {hps.train.pretrained_s2G}",
                net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)["weight"],
                    strict=False,
                ),
            )
        net_g.cfm = get_peft_model(net_g.cfm, lora_config)
        net_g = model2DDP(net_g, rank)
        optim_g = get_optim(net_g)

    no_grad_names = set()
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            no_grad_names.add(name.replace("module.", ""))

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
    for _ in range(epoch_str):
        scheduler_g.step()

    scaler = GradScaler(device=device.type, enabled=hps.train.fp16_run)

    net_d = optim_d = scheduler_d = None
    print(f"start training from epoch {epoch_str}")

    with (
        Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            redirect_stderr=True,
            redirect_stdout=True,
        )
        if rank == 0
        else nullcontext() as progress
    ):
        if isinstance(progress, Progress):
            epoch_task: TaskID | None = progress.add_task(
                "Epoch",
                total=int(hps.train.epochs),
                completed=int(epoch_str) - 1,
            )
        else:
            epoch_task = step_task = None

        for epoch in range(epoch_str, hps.train.epochs + 1):
            if rank == 0:
                assert epoch_task is not None
                assert progress is not None
                progress.advance(epoch_task, 1)
                train_and_evaluate(
                    device,
                    epoch,
                    hps,
                    (net_g, net_d),
                    (optim_g, optim_d),
                    (scheduler_g, scheduler_d),
                    scaler,
                    (train_loader, None),
                    logger,
                    (writer, writer_eval),
                )
            else:
                train_and_evaluate(
                    device,
                    epoch,
                    hps,
                    (net_g, net_d),
                    (optim_g, optim_d),
                    (scheduler_g, scheduler_d),
                    scaler,
                    (train_loader, None),
                    None,
                    (None, None),
                )
            scheduler_g.step()
    if rank == 0:
        assert progress
        progress.stop()
        logger.info("Training Done")
    sys.exit(0)


def train_and_evaluate(
    device: torch.device,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    logger,
    writers,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()

    with (
        Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            SpeedColumnIteration(show_speed=True),
            TimeRemainingColumn(elapsed_when_finished=True),
            console=console,
            redirect_stderr=True,
            redirect_stdout=True,
            transient=not (int(epoch) == int(hps.train.epochs)),
        )
        if device.index == 0
        else nullcontext() as progress
    ):
        if isinstance(progress, Progress):
            step_task: TaskID | None = progress.add_task("Steps", total=len(train_loader))
        else:
            step_task = None

        for batch_idx, (ssl, spec, mel, ssl_lengths, spec_lengths, text, text_lengths, mel_lengths) in enumerate(
            train_loader
        ):
            spec, spec_lengths = spec.to(device, non_blocking=True), spec_lengths.to(device, non_blocking=True)
            mel, mel_lengths = mel.to(device, non_blocking=True), mel_lengths.to(device, non_blocking=True)
            ssl = ssl.to(device, non_blocking=True)
            ssl.requires_grad = False
            text, text_lengths = text.to(device, non_blocking=True), text_lengths.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=hps.train.fp16_run):
                cfm_loss = net_g(
                    ssl,
                    spec,
                    mel,
                    ssl_lengths,
                    spec_lengths,
                    text,
                    text_lengths,
                    mel_lengths,
                    use_grad_ckpt=hps.train.grad_ckpt,
                )
                loss_gen_all = cfm_loss
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if device.index == 0 and progress is not None and step_task is not None:
                progress.advance(step_task, 1)

            if device.index == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [cfm_loss]
                    logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_loader)))
                    logger.info([x.item() for x in losses] + [global_step, lr])

                    scalar_dict = {"loss/g/total": loss_gen_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars=scalar_dict,
                    )

            global_step += 1

    if hps.train.if_save_latest == 0:
        utils.save_checkpoint(
            net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(save_root, f"G_{global_step}.pth"), logger
        )
    else:
        utils.save_checkpoint(
            net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(save_root, "G_233333333333.pth"), logger
        )

    if epoch % hps.train.save_every_epoch == 0 and device.index == 0:
        if hps.train.if_save_every_weights is True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            sim_ckpt = od()
            for key in ckpt:
                if key not in no_grad_names:
                    sim_ckpt[key] = ckpt[key].half().cpu()
            save_info = save_ckpt(
                sim_ckpt,
                hps.name + f"_e{epoch}_s{global_step}_l{lora_rank}",
                epoch,
                global_step,
                hps,
                lora_rank=lora_rank,
            )
            logger.info(f"saving ckpt {hps.name}_e{epoch}:{save_info}")


if __name__ == "__main__":
    main()
