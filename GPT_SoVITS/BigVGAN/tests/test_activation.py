# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import sys
# to import modules from parent_dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch
from alias_free_activation.cuda import activation1d
from activations import Snake


def test_load_fused_kernels():
    try:
        print("[Success] load_fused_kernels")
    except ImportError as e:
        print("[Fail] load_fused_kernels")
        raise e


def test_anti_alias_activation():
    data = torch.rand((10, 10, 200), device="cuda")

    # Check activations.Snake cuda vs. torch
    fused_anti_alias_activation = activation1d.Activation1d(
        activation=Snake(10), fused=True
    ).cuda()
    fused_activation_output = fused_anti_alias_activation(data)

    torch_anti_alias_activation = activation1d.Activation1d(
        activation=Snake(10), fused=False
    ).cuda()
    torch_activation_output = torch_anti_alias_activation(data)

    test_result = (fused_activation_output - torch_activation_output).abs()

    while test_result.dim() != 1:
        test_result = test_result.mean(dim=-1)

    diff = test_result.mean(dim=-1)

    if diff <= 1e-3:
        print(
            f"\n[Success] test_fused_anti_alias_activation"
            f"\n > mean_difference={diff}"
            f"\n > fused_values={fused_activation_output[-1][-1][:].tolist()}"
            f"\n > torch_values={torch_activation_output[-1][-1][:].tolist()}"
        )
    else:
        print(
            f"\n[Fail] test_fused_anti_alias_activation"
            f"\n > mean_difference={diff}, "
            f"\n > fused_values={fused_activation_output[-1][-1][:].tolist()}, "
            f"\n > torch_values={torch_activation_output[-1][-1][:].tolist()}"
        )


if __name__ == "__main__":
    from alias_free_activation.cuda import load

    load.load()
    test_load_fused_kernels()
    test_anti_alias_activation()
