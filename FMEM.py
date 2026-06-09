import torch.nn as nn

from SpatialMamba import LayerNorm, SpatialMamba4Path


def infer_fmem_share_mamba(checkpoint):
    if isinstance(checkpoint, dict) and 'fmem_share_mamba' in checkpoint:
        return bool(checkpoint['fmem_share_mamba'])

    state_dict = checkpoint.get('FMEMLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = [
        key[7:] if isinstance(key, str) and key.startswith('module.') else key
        for key in state_dict.keys()
    ]
    if any(str(key).startswith('global_mixer.mambas.') for key in keys):
        return False
    if any(str(key).startswith('global_mixer.mamba.') for key in keys):
        return True
    return False


class EfficientChannelAttention(nn.Module):
    def __init__(self, dim=64, kernel_size=3):
        super(EfficientChannelAttention, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class FusionMambaEnhanceModule(nn.Module):
    def __init__(self, dim=64, share_mamba=True, eca_kernel_size=3):
        super(FusionMambaEnhanceModule, self).__init__()
        self.detail_norm = LayerNorm(dim, 'WithBias')
        self.base_norm = LayerNorm(dim, 'WithBias')
        self.detail_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.base_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.detail_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)
        self.base_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

        self.global_mixer = SpatialMamba4Path(dim=dim, share_mamba=share_mamba)
        self.global_norm = LayerNorm(dim, 'WithBias')
        self.detail_linear = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.base_linear = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.merge_linear = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.eca = EfficientChannelAttention(dim=dim, kernel_size=eca_kernel_size)

    def forward(self, detail_feature, base_feature):
        detail_context = self.detail_dwconv(self.detail_proj(self.detail_norm(detail_feature)))
        base_context = self.base_dwconv(self.base_proj(self.base_norm(base_feature)))

        hybrid = detail_context * base_context + detail_context + base_context
        global_gate = self.global_norm(self.global_mixer(hybrid))

        detail_enhanced = global_gate * self.detail_linear(detail_feature)
        base_enhanced = global_gate * self.base_linear(base_feature)
        enhanced_sum = detail_enhanced + base_enhanced
        enhanced = self.eca(self.merge_linear(enhanced_sum)) + enhanced_sum

        return enhanced + detail_feature + base_feature
