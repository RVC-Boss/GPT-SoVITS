import logging
import os
import platform
import sys
import warnings
from contextlib import nullcontext
from random import randint

import torch
import torch.distributed as dist
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.multiprocessing.spawn import spawn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import GPT_SoVITS.utils as utils
from GPT_SoVITS.Accelerate import console, logger
from GPT_SoVITS.Accelerate.logger import SpeedColumnIteration
from GPT_SoVITS.module import commons
from GPT_SoVITS.module.data_utils import (
    DistributedBucketSampler,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from GPT_SoVITS.module.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from GPT_SoVITS.module.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from GPT_SoVITS.process_ckpt import save_ckpt

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)

hps = utils.get_hparams(stage=2)

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True  # 反正A100fp32更快，那试试tf32吧
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
torch.set_grad_enabled(True)

global_step = 0

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
    global global_step
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

    train_dataset = TextAudioSpeakerLoader(hps.data, version=hps.model.version)
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
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioSpeakerCollate(version=hps.model.version)
    train_loader = DataLoader(
        train_dataset,
        num_workers=5,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )

    net_g: SynthesizerTrn = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(
        hps.model.use_spectral_norm,
        version=hps.model.version,
    ).to(device)

    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            console.print(name, "not requires_grad")

    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )

    optim_g = torch.optim.AdamW(
        # filter(lambda p: p.requires_grad, net_g.parameters()),###默认所有层lr一致
        [
            {"params": base_params, "lr": hps.train.learning_rate},
            {
                "params": net_g.enc_p.text_embedding.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.encoder_text.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
            {
                "params": net_g.enc_p.mrte.parameters(),
                "lr": hps.train.learning_rate * hps.train.text_low_lr_rate,
            },
        ],
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if multigpu:
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)  # type: ignore
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)  # type: ignore
    else:
        pass

    try:  # 如果能加载自动resume
        epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(f"{hps.data.exp_dir}/logs_s2_{hps.model.version}", "D_*.pth"),
            net_d,
            optim_d,
        )[-1]  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(f"{hps.data.exp_dir}/logs_s2_{hps.model.version}", "G_*.pth"),
            net_g,
            optim_g,
        )[-1]
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0
        if (
            hps.train.pretrained_s2G != ""
            and hps.train.pretrained_s2G is not None
            and os.path.exists(hps.train.pretrained_s2G)
        ):
            if rank == 0:
                logger.info(f"loaded pretrained {hps.train.pretrained_s2G}")
            console.print(
                f"loaded pretrained {hps.train.pretrained_s2G}",
                net_g.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                )
                if multigpu
                else net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                ),
            )  ##测试不加载优化器
        if (
            hps.train.pretrained_s2D != ""
            and hps.train.pretrained_s2D is not None
            and os.path.exists(hps.train.pretrained_s2D)
        ):
            if rank == 0:
                logger.info(f"loaded pretrained {hps.train.pretrained_s2D}")
            console.print(
                f"loaded pretrained {hps.train.pretrained_s2D}",
                net_d.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"], strict=False
                )
                if multigpu
                else net_d.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"],
                ),
            )

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hps.train.lr_decay,
        last_epoch=-1,
    )
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()

    scaler = GradScaler(device=device.type, enabled=hps.train.fp16_run)

    if rank == 0:
        logger.info(f"start training from epoch {epoch_str}")

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
                    # [train_loader, eval_loader], logger, [writer, writer_eval])
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
            scheduler_d.step()
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
    net_d.train()

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

        for batch_idx, data in enumerate(train_loader):
            if hps.model.version in {"v2Pro", "v2ProPlus"}:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = data
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb = map(
                    lambda x: x.to(device, non_blocking=True),
                    (ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths, sv_emb),
                )
            else:
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = data
                ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths = map(
                    lambda x: x.to(device, non_blocking=True),
                    (ssl, ssl_lengths, spec, spec_lengths, y, y_lengths, text, text_lengths),
                )
                sv_emb = None
            ssl.requires_grad = False

            with autocast(device_type=device.type, dtype=torch.float16, enabled=hps.train.fp16_run):
                if hps.model.version in {"v2Pro", "v2ProPlus"}:
                    (y_hat, kl_ssl, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), stats_ssl) = net_g(
                        ssl, spec, spec_lengths, text, text_lengths, sv_emb
                    )
                else:
                    (
                        y_hat,
                        kl_ssl,
                        ids_slice,
                        x_mask,
                        z_mask,
                        (z, z_p, m_p, logs_p, m_q, logs_q),
                        stats_ssl,
                    ) = net_g(ssl, spec, spec_lengths, text, text_lengths)

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                with autocast(device_type=device.type, enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r,
                        y_d_hat_g,
                    )
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(device_type=device.type, enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl

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
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                    logger.info(
                        "Train Epoch: {} [{:.0f}%]".format(
                            epoch,
                            100.0 * batch_idx / len(train_loader),
                        )
                    )
                    logger.info([x.item() for x in losses] + [global_step, lr])

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl_ssl": kl_ssl,
                            "loss/g/kl": loss_kl,
                        }
                    )

                    # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    image_dict = None
                    try:  # Some people installed the wrong version of matplotlib.
                        image_dict = {
                            "slice/mel_org": utils.plot_spectrogram_to_numpy(
                                y_mel[0].data.cpu().numpy(),
                            ),
                            "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                                y_hat_mel[0].data.cpu().numpy(),
                            ),
                            "all/mel": utils.plot_spectrogram_to_numpy(
                                mel[0].data.cpu().numpy(),
                            ),
                            "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                                stats_ssl[0].data.cpu().numpy(),
                            ),
                        }
                    except Exception as _:
                        pass
                    if image_dict:
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            scalars=scalar_dict,
                        )
                    else:
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            scalars=scalar_dict,
                        )
            global_step += 1

    if hps.train.if_save_latest == 0:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                f"{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                f"G_{global_step}.pth",
            ),
            logger,
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                "{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                "D_{global_step}.pth",
            ),
            logger,
        )
    else:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                f"{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                "G_233333333333.pth",
            ),
            logger,
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(
                f"{hps.data.exp_dir}/logs_s2_{hps.model.version}",
                "D_233333333333.pth",
            ),
            logger,
        )

    if epoch % hps.train.save_every_epoch == 0 and device.index == 0:
        if hps.train.if_save_every_weights is True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            save_info = save_ckpt(
                ckpt,
                hps.name + f"_e{epoch}_s{global_step}",
                epoch,
                global_step,
                hps,
            )
            logger.info(f"saving ckpt {hps.name}_e{epoch}:{save_info}")


def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    logger.info("Evaluating ...")
    with torch.no_grad():
        for batch_idx, (
            ssl,
            ssl_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            text,
            text_lengths,
        ) in enumerate(eval_loader):
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl = ssl.to(device)
            text, text_lengths = text.to(device), text_lengths.to(device)
            for test in [0, 1]:
                y_hat, mask, *_ = (
                    generator.module.infer(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                        test=test,
                    )
                    if torch.cuda.is_available()
                    else generator.infer(
                        ssl,
                        spec,
                        spec_lengths,
                        text,
                        text_lengths,
                        test=test,
                    )
                )
                y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1).float(),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                image_dict.update(
                    {
                        f"gen/mel_{batch_idx}_{test}": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].cpu().numpy(),
                        ),
                    }
                )
                audio_dict.update(
                    {
                        f"gen/audio_{batch_idx}_{test}": y_hat[0, :, : y_hat_lengths[0]],
                    },
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
                    },
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


if __name__ == "__main__":
    main()
