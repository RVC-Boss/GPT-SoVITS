import logging
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import trange

from .wn import WN

logger = logging.getLogger(__name__)


class VelocityField(Protocol):
    def __call__(self, *, t: Tensor, ψt: Tensor, dt: Tensor) -> Tensor:
        ...


class Solver:
    def __init__(
        self,
        method="midpoint",
        nfe=32,
        viz_name="solver",
        viz_every=100,
        mel_fn=None,
        time_mapping_divisor=4,
        verbose=False,
    ):
        self.configurate_(nfe=nfe, method=method)

        self.verbose = verbose
        self.viz_every = viz_every
        self.viz_name = viz_name

        self._camera = None
        self._mel_fn = mel_fn
        self._time_mapping = partial(self.exponential_decay_mapping, n=time_mapping_divisor)

    def configurate_(self, nfe=None, method=None):
        if nfe is None:
            nfe = self.nfe

        if method is None:
            method = self.method

        if nfe == 1 and method in ("midpoint", "rk4"):
            logger.warning(f"1 NFE is not supported for {method}, using euler method instead.")
            method = "euler"

        self.nfe = nfe
        self.method = method

    @property
    def time_mapping(self):
        return self._time_mapping

    @staticmethod
    def exponential_decay_mapping(t, n=4):
        """
        Args:
            n: target step
        """

        def h(t, a):
            return (a**t - 1) / (a - 1)

        # Solve h(1/n) = 0.5
        a = float(scipy.optimize.fsolve(lambda a: h(1 / n, a) - 0.5, x0=0))

        t = h(t, a=a)

        return t

    @torch.no_grad()
    def _maybe_camera_snap(self, *, ψt, t):
        camera = self._camera
        if camera is not None:
            if ψt.shape[1] == 1:
                # Waveform, b 1 t, plot every 100 samples
                plt.subplot(211)
                plt.plot(ψt.detach().cpu().numpy()[0, 0, ::100], color="blue")
                if self._mel_fn is not None:
                    plt.subplot(212)
                    mel = self._mel_fn(ψt.detach().cpu().numpy()[0, 0])
                    plt.imshow(mel, origin="lower", interpolation="none")
            elif ψt.shape[1] == 2:
                # Complex
                plt.subplot(121)
                plt.imshow(
                    ψt.detach().cpu().numpy()[0, 0],
                    origin="lower",
                    interpolation="none",
                )
                plt.subplot(122)
                plt.imshow(
                    ψt.detach().cpu().numpy()[0, 1],
                    origin="lower",
                    interpolation="none",
                )
            else:
                # Spectrogram, b c t
                plt.imshow(ψt.detach().cpu().numpy()[0], origin="lower", interpolation="none")
            ax = plt.gca()
            ax.text(0.5, 1.01, f"t={t:.2f}", transform=ax.transAxes, ha="center")
            camera.snap()

    @staticmethod
    def _euler_step(t, ψt, dt, f: VelocityField):
        return ψt + dt * f(t=t, ψt=ψt, dt=dt)

    @staticmethod
    def _midpoint_step(t, ψt, dt, f: VelocityField):
        return ψt + dt * f(t=t + dt / 2, ψt=ψt + dt * f(t=t, ψt=ψt, dt=dt) / 2, dt=dt)

    @staticmethod
    def _rk4_step(t, ψt, dt, f: VelocityField):
        k1 = f(t=t, ψt=ψt, dt=dt)
        k2 = f(t=t + dt / 2, ψt=ψt + dt * k1 / 2, dt=dt)
        k3 = f(t=t + dt / 2, ψt=ψt + dt * k2 / 2, dt=dt)
        k4 = f(t=t + dt, ψt=ψt + dt * k3, dt=dt)
        return ψt + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @property
    def _step(self):
        if self.method == "euler":
            return self._euler_step
        elif self.method == "midpoint":
            return self._midpoint_step
        elif self.method == "rk4":
            return self._rk4_step
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_running_train_loop(self):
        try:
            # Lazy import
            from ...utils.train_loop import TrainLoop

            return TrainLoop.get_running_loop()
        except ImportError:
            return None

    @property
    def visualizing(self):
        loop = self.get_running_train_loop()
        if loop is None:
            return
        out_path = loop.make_current_step_viz_path(self.viz_name, ".gif")
        return loop.global_step % self.viz_every == 0 and not out_path.exists()

    def _reset_camera(self):
        try:
            from celluloid import Camera

            self._camera = Camera(plt.figure())
        except:
            pass

    def _maybe_dump_camera(self):
        camera = self._camera
        loop = self.get_running_train_loop()
        if camera is not None and loop is not None:
            animation = camera.animate()
            out_path = loop.make_current_step_viz_path(self.viz_name, ".gif")
            out_path.parent.mkdir(exist_ok=True, parents=True)
            animation.save(out_path, writer="pillow", fps=4)
            plt.close()
            self._camera = None

    @property
    def n_steps(self):
        n = self.nfe
        if self.method == "euler":
            pass
        elif self.method == "midpoint":
            n //= 2
        elif self.method == "rk4":
            n //= 4
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return n

    def solve(self, f: VelocityField, ψ0: Tensor, t0=0.0, t1=1.0):
        ts = self._time_mapping(np.linspace(t0, t1, self.n_steps + 1))

        if self.visualizing:
            self._reset_camera()

        if self.verbose:
            steps = trange(self.n_steps, desc="CFM inference")
        else:
            steps = range(self.n_steps)

        ψt = ψ0

        for i in steps:
            dt = ts[i + 1] - ts[i]
            t = ts[i]
            self._maybe_camera_snap(ψt=ψt, t=t)
            ψt = self._step(t=t, ψt=ψt, dt=dt, f=f)

        self._maybe_camera_snap(ψt=ψt, t=ts[-1])

        ψ1 = ψt
        del ψt

        self._maybe_dump_camera()

        return ψ1

    def __call__(self, f: VelocityField, ψ0: Tensor, t0=0.0, t1=1.0):
        return self.solve(f=f, ψ0=ψ0, t0=t0, t1=t1)


class SinusodialTimeEmbedding(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed
        assert d_embed % 2 == 0

    def forward(self, t):
        t = t.unsqueeze(-1)  # ... 1
        p = torch.linspace(0, 4, self.d_embed // 2).to(t)
        while p.dim() < t.dim():
            p = p.unsqueeze(0)  # ... d/2
        sin = torch.sin(t * 10**p)
        cos = torch.cos(t * 10**p)
        return torch.cat([sin, cos], dim=-1)


@dataclass(eq=False)
class CFM(nn.Module):
    """
    This mixin is for general diffusion models.

    ψ0 stands for the gaussian noise, and ψ1 is the data point.

    Here we follow the CFM style:
        The generation process (reverse process) is from t=0 to t=1.
        The forward process is from t=1 to t=0.
    """

    cond_dim: int
    output_dim: int
    time_emb_dim: int = 128
    viz_name: str = "cfm"
    solver_nfe: int = 32
    solver_method: str = "midpoint"
    time_mapping_divisor: int = 4

    def __post_init__(self):
        super().__init__()
        self.solver = Solver(
            viz_name=self.viz_name,
            viz_every=1,
            nfe=self.solver_nfe,
            method=self.solver_method,
            time_mapping_divisor=self.time_mapping_divisor,
        )
        self.emb = SinusodialTimeEmbedding(self.time_emb_dim)
        self.net = WN(
            input_dim=self.output_dim,
            output_dim=self.output_dim,
            local_dim=self.cond_dim,
            global_dim=self.time_emb_dim,
        )

    def _perturb(self, ψ1: Tensor, t: Tensor | None = None):
        """
        Perturb ψ1 to ψt.
        """
        raise NotImplementedError

    def _sample_ψ0(self, x: Tensor):
        """
        Args:
            x: (b c t), which implies the shape of ψ0
        """
        shape = list(x.shape)
        shape[1] = self.output_dim
        if self.training:
            g = None
        else:
            g = torch.Generator(device=x.device)
            g.manual_seed(0)  # deterministic sampling during eval
        ψ0 = torch.randn(shape, device=x.device, dtype=x.dtype, generator=g)
        return ψ0

    @property
    def sigma(self):
        return 1e-4

    def _to_ψt(self, *, ψ1: Tensor, ψ0: Tensor, t: Tensor):
        """
        Eq (22)
        """
        while t.dim() < ψ1.dim():
            t = t.unsqueeze(-1)
        μ = t * ψ1 + (1 - t) * ψ0
        return μ + torch.randn_like(μ) * self.sigma

    def _to_u(self, *, ψ1, ψ0: Tensor):
        """
        Eq (21)
        """
        return ψ1 - ψ0

    def _to_v(self, *, ψt, x, t: float | Tensor):
        """
        Args:
            ψt: (b c t)
            x: (b c t)
            t: (b)
        Returns:
            v: (b c t)
        """
        if isinstance(t, (float, int)):
            t = torch.full(ψt.shape[:1], t).to(ψt)
        t = t.clamp(0, 1)  # [0, 1)
        g = self.emb(t)  # (b d)
        v = self.net(ψt, l=x, g=g)
        return v

    def compute_losses(self, x, y, ψ0) -> dict:
        """
        Args:
            x: (b c t)
            y: (b c t)
        Returns:
            losses: dict
        """
        t = torch.rand(len(x), device=x.device, dtype=x.dtype)
        t = self.solver.time_mapping(t)

        if ψ0 is None:
            ψ0 = self._sample_ψ0(x)

        ψt = self._to_ψt(ψ1=y, t=t, ψ0=ψ0)

        v = self._to_v(ψt=ψt, t=t, x=x)
        u = self._to_u(ψ1=y, ψ0=ψ0)

        losses = dict(l1=F.l1_loss(v, u))

        return losses

    @torch.inference_mode()
    def sample(self, x, ψ0=None, t0=0.0):
        """
        Args:
            x: (b c t)
        Returns:
            y: (b ... t)
        """
        if ψ0 is None:
            ψ0 = self._sample_ψ0(x)
        f = lambda t, ψt, dt: self._to_v(ψt=ψt, t=t, x=x)
        ψ1 = self.solver(f=f, ψ0=ψ0, t0=t0)
        return ψ1

    def forward(self, x: Tensor, y: Tensor | None = None, ψ0: Tensor | None = None, t0=0.0):
        if y is None:
            y = self.sample(x, ψ0=ψ0, t0=t0)
        else:
            self.losses = self.compute_losses(x, y, ψ0=ψ0)
        return y
