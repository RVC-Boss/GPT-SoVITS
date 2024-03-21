import utils, os

hps = utils.get_hparams(stage=2)
os.environ["CUDA_VISIBLE_DEVICES"] = hps.train.gpu_numbers.replace("-", ",")
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist, traceback
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging, traceback

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
from random import randint
from module import commons

from module.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler,
)
from module.models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from module.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from module.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from process_ckpt import savee

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
###反正A100fp32更快，那试试tf32吧
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")  # 最低精度但最快（也就快一丁点），对于结果造成不了影响
# from config import pretrained_s2G,pretrained_s2D
global_step = 0

device = "cpu"  # cuda以外的设备，等mps优化后加入


def main():

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.data.exp_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.s2_ckpt_dir)
        writer = SummaryWriter(log_dir=hps.s2_ckpt_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.s2_ckpt_dir, "eval"))

    dist.init_process_group(
        backend = "gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data)  ########
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
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=6,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=16,
    )
    # if rank == 0:
    #     eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, val=True)
    #     eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
    #                              batch_size=1, pin_memory=True,
    #                              drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).cuda(rank) if torch.cuda.is_available() else SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)

    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank) if torch.cuda.is_available() else MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)
    for name, param in net_g.named_parameters():
        if not param.requires_grad:
            print(name, "not requires_grad")

    te_p = list(map(id, net_g.enc_p.text_embedding.parameters()))
    et_p = list(map(id, net_g.enc_p.encoder_text.parameters()))
    mrte_p = list(map(id, net_g.enc_p.mrte.parameters()))
    base_params = filter(
        lambda p: id(p) not in te_p + et_p + mrte_p and p.requires_grad,
        net_g.parameters(),
    )

    # te_p=net_g.enc_p.text_embedding.parameters()
    # et_p=net_g.enc_p.encoder_text.parameters()
    # mrte_p=net_g.enc_p.mrte.parameters()

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
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
        net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2" % hps.data.exp_dir, "D_*.pth"),
            net_d,
            optim_d,
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path("%s/logs_s2" % hps.data.exp_dir, "G_*.pth"),
            net_g,
            optim_g,
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.train.pretrained_s2G != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2G)
            print(
                net_g.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                ) if torch.cuda.is_available() else net_g.load_state_dict(
                    torch.load(hps.train.pretrained_s2G, map_location="cpu")["weight"],
                    strict=False,
                )
            )  ##测试不加载优化器
        if hps.train.pretrained_s2D != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % hps.train.pretrained_s2D)
            print(
                net_d.module.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"]
                ) if torch.cuda.is_available() else net_d.load_state_dict(
                    torch.load(hps.train.pretrained_s2D, map_location="cpu")["weight"]
                )
            )

    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=-1
    )
    for _ in range(epoch_str):
        scheduler_g.step()
        scheduler_d.step()

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                # [train_loader, eval_loader], logger, [writer, writer_eval])
                [train_loader, None],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    # scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (
        ssl,
        ssl_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        text,
        text_lengths,
    ) in tqdm(enumerate(train_loader)):
        if torch.cuda.is_available():
            spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
                rank, non_blocking=True
            )
            y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
                rank, non_blocking=True
            )
            ssl = ssl.cuda(rank, non_blocking=True)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
            text, text_lengths = text.cuda(rank, non_blocking=True), text_lengths.cuda(
                rank, non_blocking=True
            )
        else:
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            ssl = ssl.to(device)
            ssl.requires_grad = False
            # ssl_lengths = ssl_lengths.cuda(rank, non_blocking=True)
            text, text_lengths = text.to(device), text_lengths.to(device)

        with autocast(enabled=hps.train.fp16_run):
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
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
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

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
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

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
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
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                    "all/stats_ssl": utils.plot_spectrogram_to_numpy(
                        stats_ssl[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        if hps.train.if_save_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2" % hps.data.exp_dir, "G_{}.pth".format(global_step)
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2" % hps.data.exp_dir, "D_{}.pth".format(global_step)
                ),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2" % hps.data.exp_dir, "G_{}.pth".format(233333333333)
                ),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(
                    "%s/logs_s2" % hps.data.exp_dir, "D_{}.pth".format(233333333333)
                ),
            )
        if rank == 0 and hps.train.if_save_every_weights == True:
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
            logger.info(
                "saving ckpt %s_e%s:%s"
                % (
                    hps.name,
                    epoch,
                    savee(
                        ckpt,
                        hps.name + "_e%s_s%s" % (epoch, global_step),
                        epoch,
                        global_step,
                        hps,
                    ),
                )
            )

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    print("Evaluating ...")
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
            print(111)
            if torch.cuda.is_available():
                spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
                y, y_lengths = y.cuda(), y_lengths.cuda()
                ssl = ssl.cuda()
                text, text_lengths = text.cuda(), text_lengths.cuda()
            else:
                spec, spec_lengths = spec.to(device), spec_lengths.to(device)
                y, y_lengths = y.to(device), y_lengths.to(device)
                ssl = ssl.to(device)
                text, text_lengths = text.to(device), text_lengths.to(device)
            for test in [0, 1]:
                y_hat, mask, *_ = generator.module.infer(
                    ssl, spec, spec_lengths, text, text_lengths, test=test
                ) if torch.cuda.is_available() else generator.infer(
                    ssl, spec, spec_lengths, text, text_lengths, test=test
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
                            y_hat_mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update(
                    {f"gen/audio_{batch_idx}_{test}": y_hat[0, :, : y_hat_lengths[0]]}
                )
                image_dict.update(
                    {
                        f"gt/mel_{batch_idx}": utils.plot_spectrogram_to_numpy(
                            mel[0].cpu().numpy()
                        )
                    }
                )
                audio_dict.update({f"gt/audio_{batch_idx}": y[0, :, : y_lengths[0]]})

        # y_hat, mask, *_ = generator.module.infer(ssl, spec_lengths, speakers, y=None)
        # audio_dict.update({
        #     f"gen/audio_{batch_idx}_style_pred": y_hat[0, :, :]
        # })

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
