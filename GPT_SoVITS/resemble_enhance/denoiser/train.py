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
from .denoiser import Denoiser
from .hparams import HParams


def load_G(run_dir: Path, hp: HParams | None = None, training=True):
    if hp is None:
        hp = HParams.load(run_dir)
    assert isinstance(hp, HParams)
    model = Denoiser(hp)
    engine = Engine(model=model, config_class=DeepSpeedConfig(hp.deepspeed_config), ckpt_dir=run_dir / "ds" / "G")
    if training:
        engine.load_checkpoint()
    else:
        engine.load_checkpoint(load_optimizer_states=False, load_lr_scheduler_states=False)
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

    train_dl, val_dl = create_dataloaders(hp, mode="denoiser")

    def feed_G(engine: Engine, batch: dict[str, Tensor]):
        alpha_fn = lambda: random.uniform(*hp.mix_alpha_range)
        if random.random() < hp.distort_prob:
            fg_wavs = batch["fg_dwavs"]
        else:
            fg_wavs = batch["fg_wavs"]
        mx_dwavs = mix_fg_bg(fg_wavs, batch["bg_dwavs"], alpha=alpha_fn)
        pred = engine(mx_dwavs, fg_wavs)
        losses = engine.gather_attribute("losses", prefix="losses")
        return pred, losses

    @torch.no_grad()
    def eval_fn(engine: Engine, eval_dir, n_saved=10):
        model = engine.module
        model.eval()

        step = engine.global_step

        for i, batch in enumerate(tqdm(val_dl), 1):
            batch = tree_map(lambda x: x.to(args.device) if isinstance(x, Tensor) else x, batch)

            fg_dwavs = batch["fg_dwavs"]  # 1 t
            mx_dwavs = mix_fg_bg(fg_dwavs, batch["bg_dwavs"])
            pred_fg_dwavs = model(mx_dwavs)  # 1 t

            mx_mels = model.to_mel(mx_dwavs)  # 1 c t
            fg_mels = model.to_mel(fg_dwavs)  # 1 c t
            pred_fg_mels = model.to_mel(pred_fg_dwavs)  # 1 c t

            rate = model.hp.wav_rate
            get_path = lambda suffix: eval_dir / f"step_{step:08}_{i:03}{suffix}"

            save_wav(get_path("_input.wav"), mx_dwavs[0], rate=rate)
            save_wav(get_path("_predict.wav"), pred_fg_dwavs[0], rate=rate)
            save_wav(get_path("_target.wav"), fg_dwavs[0], rate=rate)

            save_mels(
                get_path(".png"),
                cond_mel=mx_mels[0].cpu().numpy(),
                pred_mel=pred_fg_mels[0].cpu().numpy(),
                targ_mel=fg_mels[0].cpu().numpy(),
            )

            if i >= n_saved:
                break

    train_loop = TrainLoop(
        run_dir=args.run_dir,
        train_dl=train_dl,
        load_G=partial(load_G, hp=hp),
        device=args.device,
        feed_G=feed_G,
        eval_fn=eval_fn,
    )

    train_loop.run(max_steps=hp.max_steps)


if __name__ == "__main__":
    main()
