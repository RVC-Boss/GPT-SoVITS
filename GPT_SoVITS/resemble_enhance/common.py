import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class Normalizer(nn.Module):
    def __init__(self, momentum=0.01, eps=1e-9):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.running_mean_unsafe: Tensor
        self.running_var_unsafe: Tensor
        self.register_buffer("running_mean_unsafe", torch.full([], torch.nan))
        self.register_buffer("running_var_unsafe", torch.full([], torch.nan))

    @property
    def started(self):
        return not torch.isnan(self.running_mean_unsafe)

    @property
    def running_mean(self):
        if not self.started:
            return torch.zeros_like(self.running_mean_unsafe)
        return self.running_mean_unsafe

    @property
    def running_std(self):
        if not self.started:
            return torch.ones_like(self.running_var_unsafe)
        return (self.running_var_unsafe + self.eps).sqrt()

    @torch.no_grad()
    def _ema(self, a: Tensor, x: Tensor):
        return (1 - self.momentum) * a + self.momentum * x

    def update_(self, x):
        if not self.started:
            self.running_mean_unsafe = x.mean()
            self.running_var_unsafe = x.var()
        else:
            self.running_mean_unsafe = self._ema(self.running_mean_unsafe, x.mean())
            self.running_var_unsafe = self._ema(self.running_var_unsafe, (x - self.running_mean).pow(2).mean())

    def forward(self, x: Tensor, update=True):
        if self.training and update:
            self.update_(x)
        self.stats = dict(mean=self.running_mean.item(), std=self.running_std.item())
        x = (x - self.running_mean) / self.running_std
        return x

    def inverse(self, x: Tensor):
        return x * self.running_std + self.running_mean
