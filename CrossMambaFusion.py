import torch
import torch.nn as nn

from GMEM import GlobalMamba4Path
from SpatialMamba import LayerNorm
from net import AKCBlock


CROSS_MAMBA_FUSION_STRUCTURE = 'low_high_cross_mamba_private_akc'
LEGACY_CROSS_MAMBA_FUSION_STRUCTURES = ('low_high_cross_mamba',)


def is_cross_mamba_fusion_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        return False
    fusion_structure = checkpoint.get('fusion_structure')
    return fusion_structure in (
        CROSS_MAMBA_FUSION_STRUCTURE,
        *LEGACY_CROSS_MAMBA_FUSION_STRUCTURES,
    ) or (
        'ModalEnhanceLayer' in checkpoint and 'CrossMambaFusionLayer' in checkpoint
    )


def infer_cross_mamba_share_mode(checkpoint):
    if isinstance(checkpoint, dict) and 'cross_mamba_share_mode' in checkpoint:
        return str(checkpoint['cross_mamba_share_mode']).lower()
    return 'independent'


def get_decoder_residual_input(mode, data_ir, data_vis):
    mode = str(mode or 'none').lower()
    if mode == 'none':
        return None
    if mode == 'ir':
        return data_ir
    if mode == 'vis':
        return data_vis
    if mode == 'ir+vis':
        return data_ir + data_vis
    raise ValueError(f"Unsupported decoder residual mode: {mode}")


class IntraModalEnhanceBlock(nn.Module):
    def __init__(self, dim=64, out_dim=None):
        super(IntraModalEnhanceBlock, self).__init__()
        out_dim = dim if out_dim is None else int(out_dim)
        self.proj = nn.Conv2d(dim * 2, out_dim, kernel_size=1, bias=True)

    def forward(self, low_feature, high_feature):
        if low_feature.shape != high_feature.shape:
            raise ValueError(
                f"low_feature and high_feature must have the same shape, got "
                f"{low_feature.shape} and {high_feature.shape}."
            )
        return self.proj(torch.cat((low_feature, high_feature), dim=1))


class CrossMambaFusionBlock(nn.Module):
    def __init__(self, dim=64, share_mode='independent', use_checkpoint=True):
        super(CrossMambaFusionBlock, self).__init__()
        self.ir_norm = LayerNorm(dim, 'WithBias')
        self.vi_norm = LayerNorm(dim, 'WithBias')
        self.cross_mixer = GlobalMamba4Path(
            dim=dim,
            share_mode=share_mode,
            use_checkpoint=use_checkpoint,
        )
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
        p_ir = self.ir_private(ir_feature)
        p_vi = self.vi_private(vi_feature)
        ir_enhanced = ir_cross + p_ir
        vi_enhanced = vi_cross + p_vi
        return self.fusion_proj(torch.cat((ir_enhanced, vi_enhanced), dim=1))
