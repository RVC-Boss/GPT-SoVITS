import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@torch.jit.script
def _fused_tanh_sigmoid(h):
    a, b = h.chunk(2, dim=1)
    h = a.tanh() * b.sigmoid()
    return h


class WNLayer(nn.Module):
    """
    A DiffWave-like WN
    """

    def __init__(self, hidden_dim, local_dim, global_dim, kernel_size, dilation):
        super().__init__()

        local_output_dim = hidden_dim * 2

        if global_dim is not None:
            self.gconv = nn.Conv1d(global_dim, hidden_dim, 1)

        if local_dim is not None:
            self.lconv = nn.Conv1d(local_dim, local_output_dim, 1)

        self.dconv = nn.Conv1d(hidden_dim, local_output_dim, kernel_size, dilation=dilation, padding="same")

        self.out = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1)

    def forward(self, z, l, g):
        identity = z

        if g is not None:
            if g.dim() == 2:
                g = g.unsqueeze(-1)
            z = z + self.gconv(g)

        z = self.dconv(z)

        if l is not None:
            z = z + self.lconv(l)

        z = _fused_tanh_sigmoid(z)

        h = self.out(z)

        z, s = h.chunk(2, dim=1)

        o = (z + identity) / math.sqrt(2)

        return o, s


class WN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        local_dim=None,
        global_dim=None,
        n_layers=30,
        kernel_size=3,
        dilation_cycle=5,
        hidden_dim=512,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        assert hidden_dim % 2 == 0

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.local_dim = local_dim
        self.global_dim = global_dim

        self.start = nn.Conv1d(input_dim, hidden_dim, 1)
        if local_dim is not None:
            self.local_norm = nn.InstanceNorm1d(local_dim)

        self.layers = nn.ModuleList(
            [
                WNLayer(
                    hidden_dim=hidden_dim,
                    local_dim=local_dim,
                    global_dim=global_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** (i % dilation_cycle),
                )
                for i in range(n_layers)
            ]
        )

        self.end = nn.Conv1d(hidden_dim, output_dim, 1)

    def forward(self, z, l=None, g=None):
        """
        Args:
            z: input (b c t)
            l: local condition (b c t)
            g: global condition (b d)
        """
        z = self.start(z)

        if l is not None:
            l = self.local_norm(l)

        # Skips
        s_list = []

        for layer in self.layers:
            z, s = layer(z, l, g)
            s_list.append(s)

        s_list = torch.stack(s_list, dim=0).sum(dim=0)
        s_list = s_list / math.sqrt(len(self.layers))

        o = self.end(s_list)

        return o

    def summarize(self, length=100):
        from ptflops import get_model_complexity_info

        x = torch.randn(1, self.input_dim, length)

        macs, params = get_model_complexity_info(
            self,
            (self.input_dim, length),
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True,
        )

        print(f"Input shape: {x.shape}")
        print(f"Computational complexity: {macs}")
        print(f"Number of parameters: {params}")


if __name__ == "__main__":
    model = WN(input_dim=64, output_dim=64)
    model.summarize()
