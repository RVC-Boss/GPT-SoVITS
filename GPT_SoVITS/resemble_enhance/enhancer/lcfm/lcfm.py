import logging
from enum import Enum

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor, nn

from .cfm import CFM
from .irmae import IRMAE, IRMAEOutput

logger = logging.getLogger(__name__)


def freeze_(module):
    for p in module.parameters():
        p.requires_grad_(False)


class LCFM(nn.Module):
    class Mode(Enum):
        AE = "ae"
        CFM = "cfm"

    def __init__(self, ae: IRMAE, cfm: CFM, z_scale: float = 1.0):
        super().__init__()
        self.ae = ae
        self.cfm = cfm
        self.z_scale = z_scale
        self._mode = None
        self._eval_tau = 0.5

    @property
    def mode(self):
        return self._mode

    def set_mode_(self, mode):
        mode = self.Mode(mode)
        self._mode = mode

        if mode == mode.AE:
            freeze_(self.cfm)
            logger.info("Freeze cfm")
        elif mode == mode.CFM:
            freeze_(self.ae)
            logger.info("Freeze ae (encoder and decoder)")
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def get_running_train_loop(self):
        try:
            # Lazy import
            from ...utils.train_loop import TrainLoop

            return TrainLoop.get_running_loop()
        except ImportError:
            return None

    @property
    def global_step(self):
        loop = self.get_running_train_loop()
        if loop is None:
            return None
        return loop.global_step

    @torch.no_grad()
    def _visualize(self, x, y, y_):
        loop = self.get_running_train_loop()
        if loop is None:
            return

        plt.subplot(221)
        plt.imshow(y[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.title("GT")

        plt.subplot(222)
        y_ = y_[:, : y.shape[1]]
        plt.imshow(y_[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.title("Posterior")

        plt.subplot(223)
        z_ = self.cfm(x)
        y__ = self.ae.decode(z_)
        y__ = y__[:, : y.shape[1]]
        plt.imshow(y__[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.title("C-Prior")
        del y__

        plt.subplot(224)
        z_ = torch.randn_like(z_)
        y__ = self.ae.decode(z_)
        y__ = y__[:, : y.shape[1]]
        plt.imshow(y__[0].detach().cpu().numpy(), aspect="auto", origin="lower", interpolation="none")
        plt.title("Prior")
        del z_, y__

        path = loop.make_current_step_viz_path("recon", ".png")
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.tight_layout()
        plt.savefig(path, dpi=500)
        plt.close()

    def _scale(self, z: Tensor):
        return z * self.z_scale

    def _unscale(self, z: Tensor):
        return z / self.z_scale

    def eval_tau_(self, tau):
        self._eval_tau = tau

    def forward(self, x, y: Tensor | None = None, ψ0: Tensor | None = None):
        """
        Args:
            x: (b d t), condition mel
            y: (b d t), target mel
            ψ0: (b d t), starting mel
        """
        if self.mode == self.Mode.CFM:
            self.ae.eval()  # Always set to eval when training cfm

        if ψ0 is not None:
            ψ0 = self._scale(self.ae.encode(ψ0))
            if self.training:
                tau = torch.rand_like(ψ0[:, :1, :1])
            else:
                tau = self._eval_tau
            ψ0 = tau * torch.randn_like(ψ0) + (1 - tau) * ψ0

        if y is None:
            if self.mode == self.Mode.AE:
                with torch.no_grad():
                    training = self.ae.training
                    self.ae.eval()
                    z = self.ae.encode(x)
                    self.ae.train(training)
            else:
                z = self._unscale(self.cfm(x, ψ0=ψ0))

            h = self.ae.decode(z)
        else:
            ae_output: IRMAEOutput = self.ae(y, skip_decoding=self.mode == self.Mode.CFM)

            if self.mode == self.Mode.CFM:
                _ = self.cfm(x, self._scale(ae_output.latent.detach()), ψ0=ψ0)

            h = ae_output.decoded

            if h is not None and self.global_step is not None and self.global_step % 100 == 0:
                self._visualize(x[:1], y[:1], h[:1])

        return h
