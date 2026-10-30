import os

import torch

torch.set_grad_enabled(False)

if torch.__version__ >= "2.9.0":
    torch.backends.fp32_precision = "tf32"  # type: ignore
else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


cpu_count = os.cpu_count() or 1
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)
