import torch
import torch.nn as nn

from net import AKDEConv, ChannelAttention, PixelAttention, SpatialAttention


class SingleInputCGAEnhance(nn.Module):
    def __init__(self, dim=64, reduction=8):
        super(SingleInputCGAEnhance, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self._init_output_identity(dim)

    def _init_output_identity(self, dim):
        with torch.no_grad():
            self.conv.weight.zero_()
            for idx in range(dim):
                self.conv.weight[idx, idx, 0, 0] = 1.0
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x):
        initial = x
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.pa(initial, pattn1)
        fused_local = pattn2 * x
        result = initial + fused_local
        return self.conv(result)


class AKDECGALocalFeatureExtraction(nn.Module):
    def __init__(self, dim=64, reduction=8):
        super(AKDECGALocalFeatureExtraction, self).__init__()
        self.akdeconv = AKDEConv(dim)
        self.cga = SingleInputCGAEnhance(dim=dim, reduction=reduction)

    def forward(self, x):
        x = self.akdeconv(x)
        x = self.cga(x)
        return x
