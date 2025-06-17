# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
To further improve the short-duration feature extraction capability of ERes2Net, we expand the channel dimension
within each stage. However, this modification also increases the number of model parameters and computational complexity.
To alleviate this problem, we propose an improved ERes2NetV2 by pruning redundant structures, ultimately reducing
both the model parameters and its computational cost.
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pooling_layers as pooling_layers
from fusion import AFF


class ReLU(nn.Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = "inplace" if self.inplace else ""
        return self.__class__.__name__ + " (" + inplace_str + ")"


class BasicBlockERes2NetV2(nn.Module):
    def __init__(self, in_planes, planes, stride=1, baseWidth=26, scale=2, expansion=2):
        super(BasicBlockERes2NetV2, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class BasicBlockERes2NetV2AFF(nn.Module):
    def __init__(self, in_planes, planes, stride=1, baseWidth=26, scale=2, expansion=2):
        super(BasicBlockERes2NetV2AFF, self).__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(in_planes, width * scale, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = scale
        self.expansion = expansion

        convs = []
        fuse_models = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        for j in range(self.nums - 1):
            fuse_models.append(AFF(channels=width, r=4))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.fuse_models = nn.ModuleList(fuse_models)
        self.relu = ReLU(inplace=True)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        self.stride = stride
        self.width = width
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = self.fuse_models[i - 1](sp, spx[i])

            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.shortcut(x)
        out += residual
        out = self.relu(out)

        return out


class ERes2NetV2(nn.Module):
    def __init__(
        self,
        block=BasicBlockERes2NetV2,
        block_fuse=BasicBlockERes2NetV2AFF,
        num_blocks=[3, 4, 6, 3],
        m_channels=64,
        feat_dim=80,
        embedding_size=192,
        baseWidth=26,
        scale=2,
        expansion=2,
        pooling_func="TSTP",
        two_emb_layer=False,
    ):
        super(ERes2NetV2, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embedding_size = embedding_size
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer
        self.baseWidth = baseWidth
        self.scale = scale
        self.expansion = expansion

        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_fuse, m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_fuse, m_channels * 8, num_blocks[3], stride=2)

        # Downsampling module
        self.layer3_ds = nn.Conv2d(
            m_channels * 4 * self.expansion,
            m_channels * 8 * self.expansion,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
        )

        # Bottom-up fusion module
        self.fuse34 = AFF(channels=m_channels * 8 * self.expansion, r=4)

        self.n_stats = 1 if pooling_func == "TAP" or pooling_func == "TSDP" else 2
        self.pool = getattr(pooling_layers, pooling_func)(in_dim=self.stats_dim * self.expansion)
        self.seg_1 = nn.Linear(self.stats_dim * self.expansion * self.n_stats, embedding_size)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embedding_size, affine=False)
            self.seg_2 = nn.Linear(embedding_size, embedding_size)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes, planes, stride, baseWidth=self.baseWidth, scale=self.scale, expansion=self.expansion
                )
            )
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        stats = self.pool(fuse_out34)

        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            return embed_b
        else:
            return embed_a

    def forward3(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out3_ds = self.layer3_ds(out3)
        fuse_out34 = self.fuse34(out4, out3_ds)
        # print(111111111,fuse_out34.shape)#111111111 torch.Size([16, 2048, 10, 72])
        return fuse_out34.flatten(start_dim=1, end_dim=2).mean(-1)
        # stats = self.pool(fuse_out34)
        #
        # embed_a = self.seg_1(stats)
        # if self.two_emb_layer:
        #     out = F.relu(embed_a)
        #     out = self.seg_bn_1(out)
        #     embed_b = self.seg_2(out)
        #     return embed_b
        # else:
        #     return embed_a


if __name__ == "__main__":
    x = torch.randn(1, 300, 80)
    model = ERes2NetV2(feat_dim=80, embedding_size=192, m_channels=64, baseWidth=26, scale=2, expansion=2)
    model.eval()
    y = model(x)
    print(y.size())
    macs, num_params = profile(model, inputs=(x,))
    print("Params: {} M".format(num_params / 1e6))  # 17.86 M
    print("MACs: {} G".format(macs / 1e9))  # 12.69 G
