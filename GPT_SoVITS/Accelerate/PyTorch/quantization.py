from typing import cast

import torch

from . import nn

Tensor = torch.Tensor


# based on ComfyUI's and MinusZoneAI's fp8_linear optimization
def fp8_linear_forward(cls: nn.Linear, input: Tensor):
    weight_dtype = cls.weight.dtype
    base_dtype = input.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            input_shape = input.shape

            scale_weight: Tensor | None = getattr(cls, "scale_weight", None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device).squeeze()

            scale_input = torch.ones((), device=input.device, dtype=torch.float32)

            input = torch.clamp(input, min=-448, max=448, out=input)
            inn = (
                input.reshape(-1, input_shape[2]).to(torch.float8_e4m3fn).contiguous()
            )  # always e4m3fn because e5m2 * e5m2 is not supported

            bias = cls.bias if cls.bias is not None else None

            o = torch._scaled_mm(
                inn, cls.weight.t(), out_dtype=base_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight
            )

            return o.reshape((-1, input_shape[1], cls.weight.shape[0]))
        else:
            raise
    else:
        raise


def convert_fp8_linear(
    module: nn.Module,
):
    apply_fn = fp8_linear_forward

    for _, sub in list(module.named_modules()):
        if isinstance(sub, nn.Linear):
            if getattr(sub, "_fp8", False):
                continue
            setattr(sub, "forward", apply_fn)
            setattr(sub, "_fp8", True)
    return module


def per_tensor_quantize(tensor: torch.Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a tensor using per-tensor static scaling factor.
    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    if tensor.numel() == 0:
        # Deal with empty tensors (triggered by empty MoE experts)
        min_val, max_val = (
            torch.tensor(0.0, dtype=tensor.dtype),
            torch.tensor(1.0, dtype=tensor.dtype),
        )
    else:
        min_val, max_val = tensor.aminmax()
    amax = min_val.abs().max(max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


def fp8_gemm(A, A_scale, B, B_scale, bias, out_dtype):
    cuda_compute_capability = torch.cuda.get_device_capability()
    if cuda_compute_capability >= (9, 0):
        output, _ = torch._scaled_mm(
            A,
            B.t(),
            out_dtype=out_dtype,
            scale_a=A_scale,
            scale_b=B_scale,
            bias=bias,
        )
    else:
        output = torch.nn.functional.linear(
            A.to(out_dtype) * A_scale,
            B.to(out_dtype) * B_scale.to(out_dtype),
            bias=bias,
        )
    return output


class FP8DynamicLinear(nn.Module):
    def __init__(self, qweight: Tensor, scale: Tensor, bias: Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
        self.bias = bias

    def __call__(self, x):
        qinput, x_scale = per_tensor_quantize(x)
        output = fp8_gemm(
            A=qinput,
            A_scale=x_scale,
            B=self.weight,
            B_scale=self.weight_scale,
            bias=self.bias,
            out_dtype=x.dtype,
        )
        return output


def replace_all_linear_with_fp8(model: nn.Module):
    """
    Recursively replace every nn.Linear with FP8DynamicLinear in-place.
    """

    def _recursively_replace(parent: nn.Module):
        for child_name, child in list(parent.named_children()):
            child = cast(nn.Module, child)
            if isinstance(child, FP8DynamicLinear):
                continue

            if isinstance(child, nn.Linear):
                device = child.weight.device

                w = child.weight

                b = child.bias.clone()

                qw, qs = per_tensor_quantize(w)

                quant_linear = FP8DynamicLinear(qw, qs, b)

                quant_linear.to(device)

                setattr(parent, child_name, quant_linear)

                del child
            else:
                _recursively_replace(child)

    _recursively_replace(model)
