# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/model/lr_schedulers.py
import math

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam


class WarmupCosineLRSchedule(torch.optim.lr_scheduler._LRScheduler):
    """
    Implements Warmup learning rate schedule until 'warmup_steps', going from 'init_lr' to 'peak_lr' for multiple optimizers.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        peak_lr,
        end_lr,
        warmup_steps=10000,
        total_steps=400000,
        current_step=0,
    ):
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.optimizer = optimizer
        self._warmup_rate = (peak_lr - init_lr) / warmup_steps
        self._decay_rate = (end_lr - peak_lr) / (total_steps - warmup_steps)
        self._current_step = current_step
        self.lr = init_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._last_lr = [self.lr]

    def set_lr(self, lr):
        self._last_lr = [g["lr"] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            # g['lr'] = lr
            g["lr"] = self.end_lr  ###锁定用线性

    def step(self):
        if self._current_step < self.warmup_steps:
            lr = self.init_lr + self._warmup_rate * self._current_step

        elif self._current_step > self.total_steps:
            lr = self.end_lr

        else:
            decay_ratio = (self._current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            if decay_ratio < 0.0 or decay_ratio > 1.0:
                raise RuntimeError(
                    "Decay ratio must be in [0.0, 1.0]. Fix LR scheduler settings."
                )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = self.end_lr + coeff * (self.peak_lr - self.end_lr)

        self.lr = lr = self.end_lr = 0.002  ###锁定用线性###不听话，直接锁定！
        self.set_lr(lr)
        self.lr = lr
        self._current_step += 1
        return self.lr


if __name__ == "__main__":
    m = nn.Linear(10, 10)
    opt = Adam(m.parameters(), lr=1e-4)
    s = WarmupCosineLRSchedule(
        opt, 1e-6, 2e-4, 1e-6, warmup_steps=2000, total_steps=20000, current_step=0
    )
    lrs = []
    for i in range(25000):
        s.step()
        lrs.append(s.lr)
        print(s.lr)

    plt.plot(lrs)
    plt.plot(range(0, 25000), lrs)
    plt.show()
