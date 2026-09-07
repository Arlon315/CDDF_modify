import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatialMamba import SpatialMambaGlobalFeature


ENCODER_GLOBAL_LOCAL_SEMANTICS = 'global_local'


def _require_global_local_checkpoint(checkpoint):
    semantics = checkpoint.get('encoder_feature_semantics') if isinstance(checkpoint, dict) else None
    if semantics != ENCODER_GLOBAL_LOCAL_SEMANTICS:
        raise ValueError('Checkpoint does not use the required global-local encoder semantics.')


def rearrange(x, pattern, **kwargs):
    if pattern == 'b (head c) h w -> b head c (h w)':
        head = kwargs['head']
        b, head_channels, h, w = x.shape
        return x.reshape(b, head, head_channels // head, h * w)
    if pattern == 'b head c (h w) -> b (head c) h w':
        head, h, w = kwargs['head'], kwargs['h'], kwargs['w']
        b, _, c, _ = x.shape
        return x.reshape(b, head * c, h, w)
    if pattern == 'b c h w -> b (h w) c':
        b, c, h, w = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    if pattern == 'b (h w) c -> b c h w':
        h, w = kwargs['h'], kwargs['w']
        b, _, c = x.shape
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    raise NotImplementedError(f'Unsupported rearrange pattern: {pattern}')


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = conv_weight.reshape(conv_shape[0], conv_shape[1], -1)
        conv_weight_cd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - self.theta * conv_weight[:, :, :].sum(2)
        conv_weight_cd = conv_weight_cd.reshape(conv_shape[0], conv_shape[1], 3, 3)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_ad, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = conv_weight.reshape(conv_shape[0], conv_shape[1], -1)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = conv_weight_ad.reshape(conv_shape[0], conv_shape[1], 3, 3)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_hd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = conv_weight_hd.reshape(conv_shape[0], conv_shape[1], 3, 3)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Conv2d_vd, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = conv_weight_vd.reshape(conv_shape[0], conv_shape[1], 3, 3)
        return conv_weight_vd, self.conv.bias


class AKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_points=9, stride=1, bias=True):
        super(AKConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_points = num_points
        self.stride = stride

        side = int(math.sqrt(num_points))
        if side * side != num_points:
            raise ValueError("AKConv2d currently expects a square number of sampling points.")
        radius = side // 2
        offsets = []
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                offsets.append((y, x))
        self.register_buffer('base_offsets', torch.tensor(offsets, dtype=torch.float32))

        self.offset = nn.Conv2d(
            in_channels,
            2 * num_points,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=True,
        )
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, num_points))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_channels * self.num_points)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        batch, channels, height, width = x.shape
        offset = self.offset(x)
        out_height, out_width = offset.shape[-2:]
        offset = offset.view(batch, self.num_points, 2, out_height, out_width)

        y_base = torch.arange(out_height, device=x.device, dtype=x.dtype) * self.stride
        x_base = torch.arange(out_width, device=x.device, dtype=x.dtype) * self.stride
        yy, xx = torch.meshgrid(y_base, x_base, indexing='ij')
        base = torch.stack((yy, xx), dim=0).view(1, 1, 2, out_height, out_width)
        kernel_offsets = self.base_offsets.to(dtype=x.dtype).view(1, self.num_points, 2, 1, 1)
        coords = base + kernel_offsets + offset

        if height > 1:
            grid_y = 2.0 * coords[:, :, 0] / (height - 1) - 1.0
        else:
            grid_y = coords[:, :, 0] * 0.0
        if width > 1:
            grid_x = 2.0 * coords[:, :, 1] / (width - 1) - 1.0
        else:
            grid_x = coords[:, :, 1] * 0.0

        samples = []
        for point in range(self.num_points):
            grid = torch.stack((grid_x[:, point], grid_y[:, point]), dim=-1)
            samples.append(
                F.grid_sample(
                    x,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True,
                )
            )
        sampled = torch.stack(samples, dim=2)
        out = torch.einsum('b c n h w, o c n -> b o h w', sampled, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class AKCBlock(nn.Module):
    def __init__(self, dim):
        super(AKCBlock, self).__init__()
        self.akconv = AKConv2d(dim, dim, num_points=9, bias=True)
        self.mix1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.mix2 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        return x + self.mix2(self.mix1(self.akconv(x)))


class AKDEConv(nn.Module):
    def __init__(self, dim):
        super(AKDEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv1_6 = AKCBlock(dim)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        weight = w1 + w2 + w3 + w4 + w5
        bias = b1 + b2 + b3 + b4 + b5
        deconv = F.conv2d(input=x, weight=weight, bias=bias, stride=1, padding=1, groups=1)
        return deconv + self.conv1_6(x)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        hidden_dim = max(dim // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(
            2 * dim,
            dim,
            kernel_size=7,
            padding=3,
            padding_mode='reflect',
            groups=dim,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        b, c, h, w = x.shape
        x = torch.stack((x, pattn1), dim=2).reshape(b, 2 * c, h, w)
        return self.sigmoid(self.pa2(x))


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type != 'WithBias':
            raise ValueError("GLoC-Mamba only supports WithBias LayerNorm.")
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)
class AttentionHistogram(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(AttentionHistogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, _ = self.pad(k, self.factor)
        v, _ = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        channels_per_head = c // self.num_heads

        if ifBox:
            q = q.view(b, self.num_heads, channels_per_head, self.factor, hw)
            k = k.view(b, self.num_heads, channels_per_head, self.factor, hw)
            v = v.view(b, self.num_heads, channels_per_head, self.factor, hw)
        else:
            q = q.view(b, self.num_heads, channels_per_head, hw, self.factor).permute(0, 1, 2, 4, 3)
            k = k.view(b, self.num_heads, channels_per_head, hw, self.factor).permute(0, 1, 2, 4, 3)
            v = v.view(b, self.num_heads, channels_per_head, hw, self.factor).permute(0, 1, 2, 4, 3)

        q = q.reshape(b, self.num_heads, channels_per_head * self.factor, hw)
        k = k.reshape(b, self.num_heads, channels_per_head * self.factor, hw)
        v = v.reshape(b, self.num_heads, channels_per_head * self.factor, hw)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v

        out = out.view(b, self.num_heads, channels_per_head, self.factor, hw)
        if ifBox:
            out = out.reshape(b, c, self.factor * hw)
        else:
            out = out.permute(0, 1, 2, 4, 3).reshape(b, c, hw * self.factor)
        return self.unpad(out, t_pad)

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = torch.cat((x_sort, x[:, c // 2:]), dim=1)
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.zeros_like(out1).scatter(2, idx, out1).view(b, c, h, w)
        out2 = torch.zeros_like(out2).scatter(2, idx, out2).view(b, c, h, w)
        out = self.project_out(out1 * out2)
        out_replace = out[:, :c // 2]
        out_replace = torch.zeros_like(out_replace).scatter(-1, idx_w, out_replace)
        out_replace = torch.zeros_like(out_replace).scatter(-2, idx_h, out_replace)
        return torch.cat((out_replace, out[:, c // 2:]), dim=1)


##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class HTBFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(HTBFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(
            hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1,
            padding=2, groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(
            hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1,
            padding=2, groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


##########################################################################
## Histogram Transformer Block (HTB)
class HTB(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.5, bias=False, LayerNorm_type='WithBias'):
        super(HTB, self).__init__()

        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.attn_g = AttentionHistogram(dim, num_heads, bias, True)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)
        self.ffn = HTBFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        return x + self.ffn(self.norm_ff1(x))


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


def make_feature_blocks(block_type, dim, num_blocks, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
    block_type = str(block_type).lower()
    if block_type == 'restormer':
        return [
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            )
            for _ in range(num_blocks)
        ]
    if block_type == 'htb':
        return [
            HTB(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
            )
            for _ in range(num_blocks)
        ]
    raise ValueError(f"Unsupported block_type: {block_type}")


class Restormer_Encoder(nn.Module):
    def __init__(
        self,
        inp_channels=1,
        dim=64,
        num_blocks=(4, 4),
        heads=(8, 8, 8),
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        use_global_local=True,
    ):
        super(Restormer_Encoder, self).__init__()
        self.use_global_local = use_global_local
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*make_feature_blocks(
            'restormer',
            dim,
            num_blocks[0],
            heads[0],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        ))
        if use_global_local:
            self.globalFeature = SpatialMambaGlobalFeature(
                dim=dim,
                num_layers=1,
            )
            from .LocalFeatureCGA import AKDECGALocalFeatureExtraction
            self.localFeature = AKDECGALocalFeatureExtraction(dim=dim)

    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(self.patch_embed(inp_img))
        if not self.use_global_local:
            return out_enc_level1
        global_feature = self.globalFeature(out_enc_level1)
        local_feature = self.localFeature(out_enc_level1)
        return global_feature, local_feature, out_enc_level1


class Restormer_Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        dim=64,
        num_blocks=(4, 2),
        heads=(8, 8, 8),
        ffn_expansion_factor=2.5,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super(Restormer_Decoder, self).__init__()
        # This layer is unused by the fixed forward, but is retained because
        # its parameters are present in the supported checkpoint.
        self.reduce_channel = nn.Conv2d(int(dim * 2), dim, kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*make_feature_blocks(
            'htb',
            dim,
            num_blocks[0],
            heads[1],
            ffn_expansion_factor,
            bias,
            LayerNorm_type,
        ))
        self.output = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(dim // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, fused_feature):
        output = self.output(self.encoder_level2(fused_feature))
        return self.sigmoid(output), fused_feature


CURRENT_GLCM_CONFIG = {
    'backbone': 'restormer',
    'encoder_global_feature': 'spatial_mamba',
    'encoder_local_feature': 'AKDEConv+CGA',
    'decoder_block': 'htb',
    'fusion_structure': 'glcm_mamba_common_mamba_private_akc',
    'modal_enhance_structure': 'cross_modal_global_local',
    'cross_mamba_share_mode': 'independent',
    'decoder_residual': 'none',
}


def require_current_glcm_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a dictionary.')
    _require_global_local_checkpoint(checkpoint)

    for key, expected in CURRENT_GLCM_CONFIG.items():
        actual = checkpoint.get(key)
        if actual != expected:
            raise ValueError(
                f'Checkpoint {key} is {actual!r}, but GLoC-Mamba requires {expected!r}.'
            )

    required_states = (
        'DIDF_Encoder',
        'DIDF_Decoder',
        'ModalEnhanceLayer',
        'CrossMambaFusionLayer',
    )
    missing_states = [key for key in required_states if key not in checkpoint]
    if missing_states:
        raise KeyError(f'Checkpoint is missing required model states: {missing_states}.')


def build_current_glcm_modules():
    return Restormer_Encoder(), Restormer_Decoder()
