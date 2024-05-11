import argparse
import random
from functools import partial
from pathlib import Path

import soundfile
import torch
from deepspeed import DeepSpeedConfig
from torch import Tensor
from tqdm import tqdm

from ..data import create_dataloaders, mix_fg_bg
from ..utils import Engine, TrainLoop, save_mels, setup_logging, tree_map
from ..utils.distributed import is_local_leader
from .enhancer import Enhancer
from .hparams import HParams
from .univnet.discriminator import Discriminator


def load_G(run_dir: Path, hp: HParams | None = None, training=True):
    if hp is None:
        hp = HParams.load(run_dir)
        assert isinstance(hp, HParams)
    model = Enhancer(hp)
    engine = Engine(model=model, config_class=DeepSpeedConfig(hp.deepspeed_config), ckpt_dir=run_dir / "ds" / "G")
    if training:
        engine.load_checkpoint()
    else:
        engine.load_checkpoint(load_optimizer_states=False, load_lr_scheduler_states=False)
    return engine


def load_D(run_dir: Path, hp: HParams):
    if hp is None:
        hp = HParams.load(run_dir)
        assert isinstance(hp, HParams)
    model = Discriminator(hp)
    engine = Engine(model=model, config_class=DeepSpeedConfig(hp.deepspeed_config), ckpt_dir=run_dir / "ds" / "D")
    engine.load_checkpoint()
    return engine


def save_wav(path: Path, wav: Tensor, rate: int):
    wav = wav.detach().cpu().numpy()
    soundfile.write(path, wav, samplerate=rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--yaml", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging(args.run_dir)
    hp = HParams.load(args.run_dir, yaml=args.yaml)

    if is_local_leader():
        hp.save_if_not_exists(args.run_dir)
        hp.print()

    train_dl, val_dl = create_dataloaders(hp, mode="enhancer")

    def feed_G(engine: Engine, batch: dict[str, Tensor]):
        if hp.lcfm_training_mode == "ae":
            pred = engine(batch["fg_wavs"], batch["fg_wavs"])
        elif hp.lcfm_training_mode == "cfm":
            alpha_fn = lambda: random.uniform(*hp.mix_alpha_range)
            mx_dwavs = mix_fg_bg(batch["fg_dwavs"], batch["bg_dwavs"], alpha=alpha_fn)
            pred = engine(mx_dwavs, batch["fg_wavs"], batch["fg_dwavs"])
        else:
            raise ValueError(f"Unknown training mode: {hp.lcfm_training_mode}")
        losses = engine.gather_attribute("losses")
        return pred, losses

    def feed_D(engine: Engine, batch: dict | None, fake: Tensor):
        if batch is None:
            losses = engine(fake=fake)
        else:
            losses = engine(fake=fake, real=batch["fg_wavs"])
        return losses

    @torch.no_grad()
    def eval_fn(engine: Engine, eval_dir, n_saved=10):
        assert isinstance(hp, HParams)

        model = engine.module
        model.eval()

        step = engine.global_step

        for i, batch in enumerate(tqdm(val_dl), 1):
            batch = tree_map(lambda x: x.to(args.device) if isinstance(x, Tensor) else x, batch)

            fg_wavs = batch["fg_wavs"]  # 1 t

            if hp.lcfm_training_mode == "ae":
                in_dwavs = fg_wavs
            elif hp.lcfm_training_mode == "cfm":
                in_dwavs = mix_fg_bg(fg_wavs, batch["bg_dwavs"])
            else:
                raise ValueError(f"Unknown training mode: {hp.lcfm_training_mode}")

            pred_fg_wavs = model(in_dwavs)  # 1 t

            in_mels = model.to_mel(in_dwavs)  # 1 c t
            fg_mels = model.to_mel(fg_wavs)  # 1 c t
            pred_fg_mels = model.to_mel(pred_fg_wavs)  # 1 c t

            rate = model.hp.wav_rate
            get_path = lambda suffix: eval_dir / f"step_{step:08}_{i:03}{suffix}"

            save_wav(get_path("_input.wav"), in_dwavs[0], rate=rate)
            save_wav(get_path("_predict.wav"), pred_fg_wavs[0], rate=rate)
            save_wav(get_path("_target.wav"), fg_wavs[0], rate=rate)

            save_mels(
                get_path(".png"),
                cond_mel=in_mels[0].cpu().numpy(),
                pred_mel=pred_fg_mels[0].cpu().numpy(),
                targ_mel=fg_mels[0].cpu().numpy(),
            )

            if i >= n_saved:
                break

    train_loop = TrainLoop(
        run_dir=args.run_dir,
        train_dl=train_dl,
        load_G=partial(load_G, hp=hp),
        load_D=partial(load_D, hp=hp),
        device=args.device,
        feed_G=feed_G,
        feed_D=feed_D,
        eval_fn=eval_fn,
        gan_training_start_step=hp.gan_training_start_step,
    )

    train_loop.run(max_steps=hp.max_steps)


if __name__ == "__main__":
    main()
