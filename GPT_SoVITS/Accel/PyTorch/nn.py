"""
Enhanced Type Hint nn.Module
Modified From https://github.com/labmlai/labml/blob/master/helpers/labml_helpers/module.py
"""

from typing import Any

import torch.nn
from torch.nn import (
    functional as functional,
)
from torch.nn import (
    utils as utils,
)
from torch.nn.modules import *  # type: ignore # noqa: F403
from torch.nn.parameter import (
    Parameter as Parameter,
)

Tensor = torch.Tensor


class Module(torch.nn.Module):
    r"""
    Wraps ``torch.nn.Module`` to overload ``__call__`` instead of
    ``forward`` for better type checking.

    `PyTorch Github issue for clarification <https://github.com/pytorch/pytorch/issues/44605>`_
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        # To stop PyTorch from giving abstract methods warning
        pass

    def __init_subclass__(cls, **kwargs):
        if cls.__dict__.get("__call__", None) is None:
            return

        setattr(cls, "forward", cls.__dict__["__call__"])
        delattr(cls, "__call__")

    @property
    def device(self) -> torch.device:
        params = self.parameters()
        try:
            sample_param = next(params)
            return sample_param.device
        except StopIteration:
            raise RuntimeError(f"Unable to determine device of {self.__class__.__name__}") from None


class Linear(torch.nn.Linear):
    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class Dropout(torch.nn.Dropout):
    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class Embedding(torch.nn.Embedding):
    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)


class LayerNorm(torch.nn.LayerNorm):
    def __call__(self, input: Tensor) -> Tensor:
        return super().__call__(input)
