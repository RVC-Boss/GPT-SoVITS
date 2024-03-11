import logging

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import Tensor, nn
from torch.distributions import Beta

from ..common import Normalizer
from ..denoiser.inference import load_denoiser
from ..melspec import MelSpectrogram
from ..utils.distributed import global_leader_only
from ..utils.train_loop import TrainLoop
from .hparams import HParams
from .lcfm import CFM, IRMAE, LCFM
from .univnet import UnivNet

logger = logging.getLogger(__name__)


def _maybe(fn):
    def _fn(*args):
        if args[0] is None:
            return None
        return fn(*args)

    return _fn


def _normalize_wav(x: Tensor):
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class Enhancer(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        n_mels = self.hp.num_mels
        vocoder_input_dim = n_mels + self.hp.vocoder_extra_dim
        latent_dim = self.hp.lcfm_latent_dim

        self.lcfm = LCFM(
            IRMAE(
                input_dim=n_mels,
                output_dim=vocoder_input_dim,
                latent_dim=latent_dim,
            ),
            CFM(
                cond_dim=n_mels,
                output_dim=self.hp.lcfm_latent_dim,
                solver_nfe=self.hp.cfm_solver_nfe,
                solver_method=self.hp.cfm_solver_method,
                time_mapping_divisor=self.hp.cfm_time_mapping_divisor,
            ),
            z_scale=self.hp.lcfm_z_scale,
        )

        self.lcfm.set_mode_(self.hp.lcfm_training_mode)

        self.mel_fn = MelSpectrogram(hp)
        self.vocoder = UnivNet(self.hp, vocoder_input_dim)
        self.denoiser = load_denoiser(self.hp.denoiser_run_dir, "cpu")
        self.normalizer = Normalizer()

        self._eval_lambd = 0.0

        self.dummy: Tensor
        self.register_buffer("dummy", torch.zeros(1))

        if self.hp.enhancer_stage1_run_dir is not None:
            pretrained_path = self.hp.enhancer_stage1_run_dir / "ds/G/default/mp_rank_00_model_states.pt"
            self._load_pretrained(pretrained_path)

        logger.info(f"{self.__class__.__name__} summary")
        logger.info(f"{self.summarize()}")

    def _load_pretrained(self, path):
        # Clone is necessary as otherwise it holds a reference to the original model
        cfm_state_dict = {k: v.clone() for k, v in self.lcfm.cfm.state_dict().items()}
        denoiser_state_dict = {k: v.clone() for k, v in self.denoiser.state_dict().items()}
        state_dict = torch.load(path, map_location="cpu")["module"]
        self.load_state_dict(state_dict, strict=False)
        self.lcfm.cfm.load_state_dict(cfm_state_dict)  # Reset cfm
        self.denoiser.load_state_dict(denoiser_state_dict)  # Reset denoiser
        logger.info(f"Loaded pretrained model from {path}")

    def summarize(self):
        npa_train = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        npa = lambda m: sum(p.numel() for p in m.parameters())
        rows = []
        for name, module in self.named_children():
            rows.append(dict(name=name, trainable=npa_train(module), total=npa(module)))
        rows.append(dict(name="total", trainable=npa_train(self), total=npa(self)))
        df = pd.DataFrame(rows)
        return df.to_markdown(index=False)

    def to_mel(self, x: Tensor, drop_last=True):
        """
        Args:
            x: (b t), wavs
        Returns:
            o: (b c t), mels
        """
        if drop_last:
            return self.mel_fn(x)[..., :-1]  # (b d t)
        return self.mel_fn(x)

    @global_leader_only
    @torch.no_grad()
    def _visualize(self, original_mel, denoised_mel):
        loop = TrainLoop.get_running_loop()
        if loop is None or loop.global_step % 100 != 0:
            return

        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        plt.title("Original")
        plt.imshow(original_mel[0].cpu().numpy(), origin="lower", interpolation="none")
        plt.subplot(212)
        plt.title("Denoised")
        plt.imshow(denoised_mel[0].cpu().numpy(), origin="lower", interpolation="none")
        plt.tight_layout()

        path = loop.get_running_loop_viz_path("input", ".png")
        plt.savefig(path, dpi=300)

    def _may_denoise(self, x: Tensor, y: Tensor | None = None):
        if self.hp.lcfm_training_mode == "cfm":
            return self.denoiser(x, y)
        return x

    def configurate_(self, nfe, solver, lambd, tau):
        """
        Args:
            nfe: number of function evaluations
            solver: solver method
            lambd: denoiser strength [0, 1]
            tau: prior temperature [0, 1]
        """
        self.lcfm.cfm.solver.configurate_(nfe, solver)
        self.lcfm.eval_tau_(tau)
        self._eval_lambd = lambd

    def forward(self, x: Tensor, y: Tensor | None = None, z: Tensor | None = None):
        """
        Args:
            x: (b t), mix wavs (fg + bg)
            y: (b t), fg clean wavs
            z: (b t), fg distorted wavs
        Returns:
            o: (b t), reconstructed wavs
        """
        assert x.dim() == 2, f"Expected (b t), got {x.size()}"
        assert y is None or y.dim() == 2, f"Expected (b t), got {y.size()}"

        if self.hp.lcfm_training_mode == "cfm":
            self.normalizer.eval()

        x = _normalize_wav(x)
        y = _maybe(_normalize_wav)(y)
        z = _maybe(_normalize_wav)(z)

        x_mel_original = self.normalizer(self.to_mel(x), update=False)  # (b d t)

        if self.hp.lcfm_training_mode == "cfm":
            if self.training:
                lambd = Beta(0.2, 0.2).sample(x.shape[:1]).to(x.device)
                lambd = lambd[:, None, None]
                x_mel_denoised = self.normalizer(self.to_mel(self._may_denoise(x, z)), update=False)
                x_mel_denoised = x_mel_denoised.detach()
                x_mel_denoised = lambd * x_mel_denoised + (1 - lambd) * x_mel_original
                self._visualize(x_mel_original, x_mel_denoised)
            else:
                lambd = self._eval_lambd
                if lambd == 0:
                    x_mel_denoised = x_mel_original
                else:
                    x_mel_denoised = self.normalizer(self.to_mel(self._may_denoise(x, z)), update=False)
                    x_mel_denoised = x_mel_denoised.detach()
                    x_mel_denoised = lambd * x_mel_denoised + (1 - lambd) * x_mel_original
        else:
            x_mel_denoised = x_mel_original

        y_mel = _maybe(self.to_mel)(y)  # (b d t)
        y_mel = _maybe(self.normalizer)(y_mel)

        lcfm_decoded = self.lcfm(x_mel_denoised, y_mel, ψ0=x_mel_original)  # (b d t)

        if lcfm_decoded is None:
            o = None
        else:
            o = self.vocoder(lcfm_decoded, y)

        return o
