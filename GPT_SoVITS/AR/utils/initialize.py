#!/usr/bin/env python3
"""Initialize modules for espnet2 neural networks."""
import torch
from typeguard import check_argument_types


def initialize(model: torch.nn.Module, init: str):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules
    as function `espnet_initialization_fn` within the custom module.

    Args:
        model: Target.
        init: Method of initialization.
    """
    assert check_argument_types()
    print("init with", init)

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for name, p in model.named_parameters():
        if ".bias" in name and p.dim() == 1:
            p.data.zero_()
