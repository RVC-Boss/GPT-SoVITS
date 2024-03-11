import torch.nn.functional as F
from torch import nn


class PreactResBlock(nn.Sequential):
    def __init__(self, dim):
        super().__init__(
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(dim // 16, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + super().forward(x)


class UNetBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, scale_factor=1.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.pre_conv = nn.Conv2d(input_dim, output_dim, 3, padding=1)
        self.res_block1 = PreactResBlock(output_dim)
        self.res_block2 = PreactResBlock(output_dim)
        self.downsample = self.upsample = nn.Identity()
        if scale_factor > 1:
            self.upsample = nn.Upsample(scale_factor=scale_factor)
        elif scale_factor < 1:
            self.downsample = nn.Upsample(scale_factor=scale_factor)

    def forward(self, x, h=None):
        """
        Args:
            x: (b c h w), last output
            h: (b c h w), skip output
        Returns:
            o: (b c h w), output
            s: (b c h w), skip output
        """
        x = self.upsample(x)
        if h is not None:
            assert x.shape == h.shape, f"{x.shape} != {h.shape}"
            x = x + h
        x = self.pre_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.downsample(x), x


class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, num_blocks=4, num_middle_blocks=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [
                UNetBlock(input_dim=hidden_dim * 2**i, output_dim=hidden_dim * 2 ** (i + 1), scale_factor=0.5)
                for i in range(num_blocks)
            ]
        )
        self.middle_blocks = nn.ModuleList(
            [UNetBlock(input_dim=hidden_dim * 2**num_blocks) for _ in range(num_middle_blocks)]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                UNetBlock(input_dim=hidden_dim * 2 ** (i + 1), output_dim=hidden_dim * 2**i, scale_factor=2)
                for i in reversed(range(num_blocks))
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )

    @property
    def scale_factor(self):
        return 2 ** len(self.encoder_blocks)

    def pad_to_fit(self, x):
        """
        Args:
            x: (b c h w), input
        Returns:
            x: (b c h' w'), padded input
        """
        hpad = (self.scale_factor - x.shape[2] % self.scale_factor) % self.scale_factor
        wpad = (self.scale_factor - x.shape[3] % self.scale_factor) % self.scale_factor
        return F.pad(x, (0, wpad, 0, hpad))

    def forward(self, x):
        """
        Args:
            x: (b c h w), input
        Returns:
            o: (b c h w), output
        """
        shape = x.shape

        x = self.pad_to_fit(x)
        x = self.input_proj(x)

        s_list = []
        for block in self.encoder_blocks:
            x, s = block(x)
            s_list.append(s)

        for block in self.middle_blocks:
            x, _ = block(x)

        for block, s in zip(self.decoder_blocks, reversed(s_list)):
            x, _ = block(x, s)

        x = self.head(x)
        x = x[..., : shape[2], : shape[3]]

        return x

    def test(self, shape=(3, 512, 256)):
        import ptflops

        macs, params = ptflops.get_model_complexity_info(
            self,
            shape,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True,
        )

        print(f"macs: {macs}")
        print(f"params: {params}")


def main():
    model = UNet(3, 3)
    model.test()


if __name__ == "__main__":
    main()
