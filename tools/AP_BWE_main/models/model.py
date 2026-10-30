import torch
import torch.nn as nn


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        layer_scale_init_value (float, optional): Initial value for the layer scale. Zero disables scaling.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value=0,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * 3)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim * 3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class APNet_BWE_Model(torch.nn.Module):
    def __init__(self, h):
        super(APNet_BWE_Model, self).__init__()
        self.h = h
        layer_scale_init_value = 1 / h.ConvNeXt_layers

        self.conv_pre_mag = nn.Conv1d(h.n_fft // 2 + 1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(h.n_fft // 2 + 1, h.ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)

        self.convnext_mag = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=h.ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for _ in range(h.ConvNeXt_layers)
            ]
        )

        self.norm_post_mag = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(h.ConvNeXt_channels, eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(h.ConvNeXt_channels, h.n_fft // 2 + 1)
        self.linear_post_pha_r = nn.Linear(h.ConvNeXt_channels, h.n_fft // 2 + 1)
        self.linear_post_pha_i = nn.Linear(h.ConvNeXt_channels, h.n_fft // 2 + 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, mag_nb, pha_nb):
        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag)
            x_pha = conv_block_pha(x_pha)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)

        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)

        com_wb = torch.stack((torch.exp(mag_wb) * torch.cos(pha_wb), torch.exp(mag_wb) * torch.sin(pha_wb)), dim=-1)

        return mag_wb, pha_wb, com_wb
