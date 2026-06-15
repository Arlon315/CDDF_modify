import torch
import torch.nn as nn
import torch.nn.functional as F

from net import HTBFeedForward, LayerNorm as HTBLayerNorm
from SpatialMamba import LayerNorm as SpatialLayerNorm, SpatialMamba4Path


FUSION_ENHANCE_MAMBA = 'mamba'
FUSION_ENHANCE_CROSS_HISTOGRAM = 'cross_histogram'


def _strip_module_prefixes(state_dict):
    return [
        key[7:] if isinstance(key, str) and key.startswith('module.') else key
        for key in state_dict.keys()
    ]


def infer_fusion_enhance_type(checkpoint):
    if not isinstance(checkpoint, dict) or 'FMEMLayer' not in checkpoint:
        return None

    enhance_type = checkpoint.get('fusion_enhance_type')
    if enhance_type:
        enhance_type = str(enhance_type).lower()
        if enhance_type in (FUSION_ENHANCE_MAMBA, FUSION_ENHANCE_CROSS_HISTOGRAM):
            return enhance_type
        raise ValueError(f"Unsupported fusion_enhance_type: {enhance_type}")

    keys = _strip_module_prefixes(checkpoint.get('FMEMLayer', {}))
    if any(str(key).startswith('global_mixer.') for key in keys):
        return FUSION_ENHANCE_MAMBA
    if any(
        str(key).startswith((
            'cross_attention.detail_branch.',
            'cross_attention.base_branch.',
        ))
        for key in keys
    ):
        return FUSION_ENHANCE_CROSS_HISTOGRAM
    return FUSION_ENHANCE_MAMBA


def infer_fmem_share_mamba(checkpoint):
    if isinstance(checkpoint, dict) and 'fmem_share_mamba' in checkpoint:
        return bool(checkpoint['fmem_share_mamba'])

    state_dict = checkpoint.get('FMEMLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = _strip_module_prefixes(state_dict)
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
        self.detail_norm = SpatialLayerNorm(dim, 'WithBias')
        self.base_norm = SpatialLayerNorm(dim, 'WithBias')
        self.detail_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.base_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.detail_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)
        self.base_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

        self.global_mixer = SpatialMamba4Path(dim=dim, share_mamba=share_mamba)
        self.global_norm = SpatialLayerNorm(dim, 'WithBias')
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


class HistogramBranch(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(HistogramBranch, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.norm = HTBLayerNorm(dim, 'WithBias')
        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 5,
            dim * 5,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 5,
            bias=bias,
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)

        x_half, idx_h = x[:, :c // 2].sort(dim=-2)
        x_half, idx_w = x_half.sort(dim=-1)
        x = torch.cat((x_half, x[:, c // 2:]), dim=1)

        q1, k1, q2, k2, v = self.qkv_dwconv(self.qkv(x)).chunk(5, dim=1)
        v, idx_v = v.reshape(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.reshape(b, c, -1), dim=2, index=idx_v)
        k1 = torch.gather(k1.reshape(b, c, -1), dim=2, index=idx_v)
        q2 = torch.gather(q2.reshape(b, c, -1), dim=2, index=idx_v)
        k2 = torch.gather(k2.reshape(b, c, -1), dim=2, index=idx_v)

        return {
            'q1': q1,
            'k1': k1,
            'q2': q2,
            'k2': k2,
            'v': v,
            'idx_v': idx_v,
            'idx_h': idx_h,
            'idx_w': idx_w,
            'shape': (b, c, h, w),
            'temperature': self.temperature,
        }


class CrossHistogramAttention(nn.Module):
    def __init__(self, dim=64, num_heads=4, bias=False):
        super(CrossHistogramAttention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

        self.dim = dim
        self.num_heads = num_heads
        self.factor = num_heads
        self.detail_branch = HistogramBranch(dim=dim, num_heads=num_heads, bias=bias)
        self.base_branch = HistogramBranch(dim=dim, num_heads=num_heads, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def _pad(self, x):
        length = x.shape[-1]
        pad_right = (-length) % self.factor
        if pad_right:
            x = F.pad(x, (0, pad_right), mode='constant', value=0)
        return x

    def _reshape_histogram(self, x, if_box):
        b, c, _ = x.shape
        x = self._pad(x)
        grouped_length = x.shape[-1] // self.factor
        channels_per_head = c // self.num_heads

        if if_box:
            x = x.reshape(
                b, self.num_heads, channels_per_head, self.factor, grouped_length)
        else:
            x = x.reshape(
                b, self.num_heads, channels_per_head, grouped_length, self.factor)
            x = x.permute(0, 1, 2, 4, 3)

        return x.reshape(
            b, self.num_heads, channels_per_head * self.factor, grouped_length)

    def _restore_histogram(self, x, if_box, output_length):
        b, _, _, grouped_length = x.shape
        channels_per_head = self.dim // self.num_heads
        x = x.reshape(
            b, self.num_heads, channels_per_head, self.factor, grouped_length)
        if if_box:
            x = x.reshape(b, self.dim, self.factor * grouped_length)
        else:
            x = x.permute(0, 1, 2, 4, 3)
            x = x.reshape(b, self.dim, grouped_length * self.factor)
        return x[:, :, :output_length]

    @staticmethod
    def _softmax_1(x):
        logit = x.exp()
        return logit / (logit.sum(dim=-1, keepdim=True) + 1)

    def _build_relation(self, q, k, temperature, if_box):
        q = F.normalize(self._reshape_histogram(q, if_box), dim=-1)
        k = F.normalize(self._reshape_histogram(k, if_box), dim=-1)
        return self._softmax_1((q @ k.transpose(-2, -1)) * temperature)

    def _apply_relation(self, relation, target_v, if_box):
        output_length = target_v.shape[-1]
        target_v = self._reshape_histogram(target_v, if_box)
        return self._restore_histogram(
            relation @ target_v,
            if_box,
            output_length,
        )

    def _restore_target(self, bhr, fhr, target):
        b, c, h, w = target['shape']
        idx_v = target['idx_v']
        bhr = torch.zeros_like(bhr).scatter(dim=2, index=idx_v, src=bhr)
        fhr = torch.zeros_like(fhr).scatter(dim=2, index=idx_v, src=fhr)
        out = self.project_out(
            bhr.reshape(b, c, h, w) * fhr.reshape(b, c, h, w))

        out_half = out[:, :c // 2]
        out_half = torch.zeros_like(out_half).scatter(
            dim=-1, index=target['idx_w'], src=out_half)
        out_half = torch.zeros_like(out_half).scatter(
            dim=-2, index=target['idx_h'], src=out_half)
        return torch.cat((out_half, out[:, c // 2:]), dim=1)

    def forward(self, detail_feature, base_feature):
        if detail_feature.shape != base_feature.shape:
            raise ValueError(
                "detail_feature and base_feature must have the same shape, "
                f"got {tuple(detail_feature.shape)} and {tuple(base_feature.shape)}."
            )
        if detail_feature.shape[1] != self.dim:
            raise ValueError(
                f"Expected {self.dim} channels, got {detail_feature.shape[1]}."
            )

        detail = self.detail_branch(detail_feature)
        base = self.base_branch(base_feature)

        base_bhr = self._build_relation(
            base['q1'], base['k1'], base['temperature'], True)
        base_fhr = self._build_relation(
            base['q2'], base['k2'], base['temperature'], False)
        detail_bhr = self._build_relation(
            detail['q1'], detail['k1'], detail['temperature'], True)
        detail_fhr = self._build_relation(
            detail['q2'], detail['k2'], detail['temperature'], False)

        detail_from_base = self._restore_target(
            self._apply_relation(base_bhr, detail['v'], True),
            self._apply_relation(base_fhr, detail['v'], False),
            detail,
        )
        base_from_detail = self._restore_target(
            self._apply_relation(detail_bhr, base['v'], True),
            self._apply_relation(detail_fhr, base['v'], False),
            base,
        )
        return 0.5 * (detail_from_base + base_from_detail)


class FusionHistogramEnhanceModule(nn.Module):
    def __init__(
        self,
        dim=64,
        num_heads=4,
        ffn_expansion_factor=2.5,
        bias=False,
    ):
        super(FusionHistogramEnhanceModule, self).__init__()
        self.cross_attention = CrossHistogramAttention(
            dim=dim,
            num_heads=num_heads,
            bias=bias,
        )
        self.norm_ffn = HTBLayerNorm(dim, 'WithBias')
        self.dgff = HTBFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, detail_feature, base_feature):
        cross = self.cross_attention(detail_feature, base_feature)
        x = detail_feature + base_feature + cross
        return x + self.dgff(self.norm_ffn(x))


def build_fusion_enhance_module(
    checkpoint=None,
    fusion_enhance_type=None,
    dim=64,
    num_heads=4,
    ffn_expansion_factor=2.5,
):
    enhance_type = fusion_enhance_type or infer_fusion_enhance_type(checkpoint)
    if enhance_type is None:
        return None

    enhance_type = str(enhance_type).lower()
    if enhance_type == FUSION_ENHANCE_CROSS_HISTOGRAM:
        return FusionHistogramEnhanceModule(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
        )
    if enhance_type == FUSION_ENHANCE_MAMBA:
        return FusionMambaEnhanceModule(
            dim=dim,
            share_mamba=infer_fmem_share_mamba(checkpoint),
        )
    raise ValueError(f"Unsupported fusion_enhance_type: {enhance_type}")
