import torch
import torch.nn as nn

from .CommonZMamba import GlobalMamba4Path
from .SpatialMamba import LayerNorm
from .net import AKCBlock


class CommenMambaFusionBlock(nn.Module):
    def __init__(self, dim=64, use_private=True):
        super(CommenMambaFusionBlock, self).__init__()
        self.use_private = use_private
        self.ir_norm = LayerNorm(dim, 'WithBias')
        self.vi_norm = LayerNorm(dim, 'WithBias')
        self.cross_mixer = GlobalMamba4Path(dim=dim)
        if use_private:
            self.ir_private = AKCBlock(dim)
            self.vi_private = AKCBlock(dim)
        self.fusion_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self._init_fusion_sum(dim)

    def _init_fusion_sum(self, dim):
        with torch.no_grad():
            self.fusion_proj.weight.zero_()
            for channel in range(dim):
                self.fusion_proj.weight[channel, channel, 0, 0] = 1.0
                self.fusion_proj.weight[channel, channel + dim, 0, 0] = 1.0
            if self.fusion_proj.bias is not None:
                self.fusion_proj.bias.zero_()

    def forward(self, ir_feature, vi_feature):
        if ir_feature.shape != vi_feature.shape:
            raise ValueError(
                f"ir_feature and vi_feature must have the same shape, got "
                f"{ir_feature.shape} and {vi_feature.shape}."
            )

        ir_cross, vi_cross = self.cross_mixer(
            self.ir_norm(ir_feature),
            self.vi_norm(vi_feature),
        )
        ir_enhanced, vi_enhanced = ir_cross, vi_cross
        if self.use_private:
            ir_enhanced = ir_enhanced + self.ir_private(ir_feature)
            vi_enhanced = vi_enhanced + self.vi_private(vi_feature)
        return self.fusion_proj(torch.cat((ir_enhanced, vi_enhanced), dim=1))
