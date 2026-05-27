import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from RandomMamba import BaseRandomMambaExtraction


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
    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out
    
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()
        hidden_features = int(in_features*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ChannelLayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(ChannelLayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class NAFBlock(nn.Module):
    """Lightweight NAFNet-style block for dense image restoration features."""

    def __init__(self, dim, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super(NAFBlock, self).__init__()
        dw_channel = dim * dw_expand
        ffn_channel = dim * ffn_expand

        self.norm1 = ChannelLayerNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm2 = ChannelLayerNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = self.dropout1(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg(y)
        y = self.conv5(y)
        y = self.dropout2(y)
        return x + y * self.gamma


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super(SAFM, self).__init__()
        if dim % n_levels != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_levels ({n_levels}).")

        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
            for _ in range(self.n_levels)
        ])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        return self.act(out) * x


class BaseFuseWithSAFM(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=1., qkv_bias=False, n_levels=4):
        super(BaseFuseWithSAFM, self).__init__()
        self.base = BaseFeatureExtraction(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            qkv_bias=qkv_bias,
        )
        self.safm = SAFM(dim, n_levels=n_levels)
        # self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.safm(x)
        # x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B):
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x


def calculate_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, 1, H, W), device=device)
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowMCAMBlock(nn.Module):
    def __init__(self, dim=64, inter_channels=16, window_size=8, shift_size=0):
        super(WindowMCAMBlock, self).__init__()
        self.dim = dim
        self.inter_channels = inter_channels
        self.window_size = window_size
        self.shift_size = shift_size

        self.g_ir = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.g_vi = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.theta_ir = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.theta_vi = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.phi_ir = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.phi_vi = nn.Conv2d(dim, inter_channels, 1, bias=False)
        self.proj = nn.Conv2d(inter_channels, dim, 1, bias=True)

        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def _pad_to_window(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        return x, H, W

    def forward(self, ir, vi):
        B, C, H0, W0 = ir.shape
        del H0, W0

        ir, H_ori, W_ori = self._pad_to_window(ir)
        vi, _, _ = self._pad_to_window(vi)

        B, C, H, W = ir.shape
        ws = self.window_size
        ss = self.shift_size

        if ss > 0:
            ir = torch.roll(ir, shifts=(-ss, -ss), dims=(2, 3))
            vi = torch.roll(vi, shifts=(-ss, -ss), dims=(2, 3))
            attn_mask = calculate_mask(H, W, ws, ss, ir.device)
        else:
            attn_mask = None

        g_ir = window_partition(self.g_ir(ir), ws)
        g_vi = window_partition(self.g_vi(vi), ws)
        theta_ir = window_partition(self.theta_ir(ir), ws)
        theta_vi = window_partition(self.theta_vi(vi), ws)
        phi_ir = window_partition(self.phi_ir(ir), ws)
        phi_vi = window_partition(self.phi_vi(vi), ws)

        attn_ir = torch.matmul(theta_ir, phi_ir.transpose(-2, -1))
        attn_vi = torch.matmul(theta_vi, phi_vi.transpose(-2, -1))

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn_ir = attn_ir.view(B, nW, ws * ws, ws * ws)
            attn_vi = attn_vi.view(B, nW, ws * ws, ws * ws)
            attn_ir = attn_ir + attn_mask.unsqueeze(0)
            attn_vi = attn_vi + attn_mask.unsqueeze(0)
            attn_ir = attn_ir.view(-1, ws * ws, ws * ws)
            attn_vi = attn_vi.view(-1, ws * ws, ws * ws)

        attn_ir = F.softmax(attn_ir, dim=-1)
        attn_vi = F.softmax(attn_vi, dim=-1)
        common_attn = attn_ir * attn_vi
        common_attn = common_attn / (common_attn.sum(dim=-1, keepdim=True) + 1e-6)

        y_ir = torch.matmul(common_attn, g_ir)
        y_vi = torch.matmul(common_attn, g_vi)
        y = y_ir * y_vi
        y = window_reverse(y, ws, H, W, B)

        if ss > 0:
            y = torch.roll(y, shifts=(ss, ss), dims=(2, 3))

        y = y[:, :, :H_ori, :W_ori].contiguous()
        y = self.proj(y)
        return y


class SwinWindowMCAMBaseFusion(nn.Module):
    dual_input = True

    def __init__(self, dim=64, inter_channels=16, window_size=8):
        super(SwinWindowMCAMBaseFusion, self).__init__()
        self.window_mcam = WindowMCAMBlock(
            dim=dim,
            inter_channels=inter_channels,
            window_size=window_size,
            shift_size=0,
        )
        self.shift_window_mcam = WindowMCAMBlock(
            dim=dim,
            inter_channels=inter_channels,
            window_size=window_size,
            shift_size=window_size // 2,
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, ir_base, vi_base):
        shortcut = ir_base + vi_base
        common_1 = self.window_mcam(ir_base, vi_base)
        common_2 = self.shift_window_mcam(ir_base, vi_base)
        common = common_1 + common_2
        common = self.refine(common)
        out = shortcut + torch.tanh(self.alpha) * common
        return out


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return self.bottleneckBlock(x)

class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)
    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        return z1, z2
    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)
    def forward(self, x):
        z1, z2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


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


class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        weight = w1 + w2 + w3 + w4 + w5
        bias = b1 + b2 + b3 + b4 + b5
        return F.conv2d(input=x, weight=weight, bias=bias, stride=1, padding=1, groups=1)


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


class CGAFusion(nn.Module):
    dual_input = True

    def __init__(self, dim=64, reduction=8, preserve_sum_range=False):
        super(CGAFusion, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.preserve_sum_range = preserve_sum_range
        self.detail_residual_scale = nn.Parameter(torch.zeros(1))
        self._init_output_identity(dim)

    def _init_output_identity(self, dim):
        with torch.no_grad():
            self.conv.weight.zero_()
            for idx in range(dim):
                self.conv.weight[idx, idx, 0, 0] = 1.0
            if self.conv.bias is not None:
                self.conv.bias.zero_()

    def forward(self, x, y=None):
        if y is None:
            return self.conv(x)

        initial = x + y
        pattn1 = self.sa(initial) + self.ca(initial)
        pattn2 = self.pa(initial, pattn1)
        fused_detail = pattn2 * x + (1.0 - pattn2) * y

        if self.preserve_sum_range:
            correction = fused_detail - 0.5 * initial
            result = initial + torch.tanh(self.detail_residual_scale) * correction
        else:
            result = initial + fused_detail
        return self.conv(result)


def _unwrap_module(module):
    return module.module if isinstance(module, nn.DataParallel) else module


def fuse_base_features(base_fuse_layer, feature_i_b, feature_v_b):
    module = _unwrap_module(base_fuse_layer)
    if getattr(module, 'dual_input', False):
        return base_fuse_layer(feature_i_b, feature_v_b)
    return base_fuse_layer(feature_i_b + feature_v_b)


def fuse_detail_features(detail_fuse_layer, feature_i_d, feature_v_d):
    module = _unwrap_module(detail_fuse_layer)
    if getattr(module, 'dual_input', False):
        return detail_fuse_layer(feature_i_d, feature_v_d)
    return detail_fuse_layer(feature_i_d + feature_v_d)

# =============================================================================

# =============================================================================
import numbers
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


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
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
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
    if block_type == 'naf':
        return [NAFBlock(dim=dim, ffn_expand=ffn_expansion_factor) for _ in range(num_blocks)]
    raise ValueError(f"Unsupported block_type: {block_type}")


def _normalize_encoder_base_feature(encoder_base_feature):
    encoder_base_feature = str(encoder_base_feature or 'random_mamba').lower()
    if encoder_base_feature in ('random_mamba', 'randommamba', 'shuffle_mamba', 'shufflemamba', 'mamba'):
        return 'random_mamba'
    if encoder_base_feature in ('base', 'base_feature', 'basefeature', 'restormer_base'):
        return 'base'
    raise ValueError(f"Unsupported encoder_base_feature: {encoder_base_feature}")


class Restormer_Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 block_type='restormer',
                 detail_enhance_layers=2,
                 encoder_detail_num_layers=1,
                 encoder_base_feature='random_mamba',
                 random_mamba_layers=4,
                 ):

        super(Restormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*make_feature_blocks(
            block_type, dim, num_blocks[0], heads[0], ffn_expansion_factor, bias, LayerNorm_type))
        encoder_detail_num_layers = int(encoder_detail_num_layers or 1)
        if str(block_type).lower() == 'naf':
            self.baseFeature = NAFBlock(dim=dim)
            self.detailFeature = DetailFeatureExtraction(num_layers=encoder_detail_num_layers)
        else:
            encoder_base_feature = _normalize_encoder_base_feature(encoder_base_feature)
            if encoder_base_feature == 'random_mamba':
                self.baseFeature = BaseRandomMambaExtraction(
                    dim=dim,
                    num_layers=int(random_mamba_layers or 4),
                    repeat=1,
                )
            else:
                self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads = heads[2])
            self.detailFeature = DetailFeatureExtraction(num_layers=encoder_detail_num_layers)
        detail_enhance_layers = int(detail_enhance_layers or 0)
        if detail_enhance_layers > 0:
            self.detailEnhance = nn.Sequential(*[DEConv(dim) for _ in range(detail_enhance_layers)])
        else:
            self.detailEnhance = nn.Identity()
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        detail_feature = self.detailEnhance(detail_feature)
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 block_type='restormer',
                 ):

        super(Restormer_Decoder, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*make_feature_blocks(
            block_type, dim, num_blocks[1], heads[1], ffn_expansion_factor, bias, LayerNorm_type))
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, inp_img, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0


class FastRestormer_Encoder(Restormer_Encoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('block_type', 'naf')
        super(FastRestormer_Encoder, self).__init__(*args, **kwargs)


class FastRestormer_Decoder(Restormer_Decoder):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('block_type', 'naf')
        super(FastRestormer_Decoder, self).__init__(*args, **kwargs)


def _build_detail_fusion_module(detail_fusion, detail_fusion_num_layers=1):
    detail_fusion = str(detail_fusion or 'cga').lower()
    if detail_fusion == 'cga':
        return CGAFusion(dim=64)
    if detail_fusion in ('inn', 'detail', 'detail_feature'):
        return DetailFeatureExtraction(num_layers=int(detail_fusion_num_layers or 1))
    raise ValueError(f"Unsupported detail_fusion: {detail_fusion}")


def _normalize_base_fusion(base_fusion):
    base_fusion = str(base_fusion or 'base')
    if base_fusion.lower() == 'base':
        return 'base'
    if base_fusion.lower() in ('basesafm', 'base_safm', 'safm'):
        return 'baseSAFM'
    if base_fusion.lower() in ('windowmcam', 'window_mcam', 'swinwindowmcam', 'swin_window_mcam', 'mcam'):
        return 'windowMCAM'
    raise ValueError(f"Unsupported base_fusion: {base_fusion}")


def _build_base_fusion_module(base_fusion, backbone):
    base_fusion = _normalize_base_fusion(base_fusion)
    if base_fusion == 'windowMCAM':
        return SwinWindowMCAMBaseFusion(dim=64, inter_channels=16, window_size=8)
    if base_fusion == 'baseSAFM':
        return BaseFuseWithSAFM(dim=64, num_heads=8)
    if str(backbone).lower() == 'fast':
        return NAFBlock(dim=64)
    return BaseFeatureExtraction(dim=64, num_heads=8)


def build_cddfuse_modules(
    backbone='restormer',
    detail_fusion='cga',
    detail_fusion_num_layers=1,
    encoder_detail_enhance_layers=2,
    encoder_detail_num_layers=1,
    base_fusion='base',
    encoder_base_feature='random_mamba',
    encoder_random_mamba_layers=4,
):
    backbone = str(backbone or 'restormer').lower()
    if backbone == 'fast':
        return (
            FastRestormer_Encoder(
                detail_enhance_layers=encoder_detail_enhance_layers,
                encoder_detail_num_layers=encoder_detail_num_layers,
                encoder_base_feature=encoder_base_feature,
                random_mamba_layers=encoder_random_mamba_layers,
            ),
            FastRestormer_Decoder(),
            _build_base_fusion_module(base_fusion, backbone),
            _build_detail_fusion_module(detail_fusion, detail_fusion_num_layers),
        )
    if backbone == 'restormer':
        return (
            Restormer_Encoder(
                detail_enhance_layers=encoder_detail_enhance_layers,
                encoder_detail_num_layers=encoder_detail_num_layers,
                encoder_base_feature=encoder_base_feature,
                random_mamba_layers=encoder_random_mamba_layers,
            ),
            Restormer_Decoder(),
            _build_base_fusion_module(base_fusion, backbone),
            _build_detail_fusion_module(detail_fusion, detail_fusion_num_layers),
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def infer_cddfuse_backbone(checkpoint):
    backbone = checkpoint.get('backbone') if isinstance(checkpoint, dict) else None
    if backbone:
        return str(backbone).lower()

    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    keys = [
        key[7:] if isinstance(key, str) and key.startswith('module.') else key
        for key in encoder_state.keys()
    ]

    if any(str(key).startswith('encoder_level1.') and '.conv1.' in str(key) for key in keys):
        return 'fast'
    if any(str(key).startswith('baseFeature.') and str(key).endswith(('beta', 'gamma')) for key in keys):
        return 'fast'
    if any(str(key).startswith('encoder_level1.') and '.attn.' in str(key) for key in keys):
        return 'restormer'
    if any(str(key).startswith('baseFeature.attn.') for key in keys):
        return 'restormer'

    return 'restormer'


def infer_cddfuse_detail_fusion(checkpoint):
    detail_fusion = checkpoint.get('detail_fusion') if isinstance(checkpoint, dict) else None
    if detail_fusion:
        return str(detail_fusion).lower()

    detail_state = checkpoint.get('DetailFuseLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = [
        key[7:] if isinstance(key, str) and key.startswith('module.') else key
        for key in detail_state.keys()
    ]

    if any(str(key).startswith(('sa.', 'ca.', 'pa.', 'conv.')) for key in keys):
        return 'cga'
    if any(str(key).startswith('net.') or 'theta_' in str(key) for key in keys):
        return 'inn'

    return 'inn'


def _strip_module_prefixes(state_dict):
    return [
        key[7:] if isinstance(key, str) and key.startswith('module.') else key
        for key in state_dict.keys()
    ]


def infer_cddfuse_base_fusion(checkpoint):
    base_fusion = checkpoint.get('base_fusion') if isinstance(checkpoint, dict) else None
    if base_fusion:
        return _normalize_base_fusion(base_fusion)

    base_state = checkpoint.get('BaseFuseLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = _strip_module_prefixes(base_state)
    if any(str(key).startswith(('window_mcam.', 'shift_window_mcam.', 'refine.')) or str(key) == 'alpha' for key in keys):
        return 'windowMCAM'
    if any(str(key).startswith(('base.', 'safm.', 'proj.')) for key in keys):
        return 'baseSAFM'

    return 'base'


def infer_cddfuse_encoder_base_feature(checkpoint):
    encoder_base_feature = checkpoint.get('encoder_base_feature') if isinstance(checkpoint, dict) else None
    if encoder_base_feature:
        return _normalize_encoder_base_feature(encoder_base_feature)

    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    keys = _strip_module_prefixes(encoder_state)
    if any(str(key).startswith(('baseFeature.layers.', 'baseFeature.norm_f.')) for key in keys):
        return 'random_mamba'
    if any(str(key).startswith(('baseFeature.attn.', 'baseFeature.mlp.', 'baseFeature.norm1.', 'baseFeature.norm2.')) for key in keys):
        return 'base'
    return 'base'


def infer_cddfuse_encoder_random_mamba_layers(checkpoint):
    num_layers = checkpoint.get('encoder_random_mamba_layers') if isinstance(checkpoint, dict) else None
    if num_layers is not None:
        return int(num_layers)

    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    layer_indices = []
    for key in _strip_module_prefixes(encoder_state):
        parts = str(key).split('.')
        if len(parts) > 2 and parts[0] == 'baseFeature' and parts[1] == 'layers' and parts[2].isdigit():
            layer_indices.append(int(parts[2]))
    return max(layer_indices) + 1 if layer_indices else 4


def infer_cddfuse_encoder_detail_num_layers(checkpoint):
    num_layers = checkpoint.get('encoder_detail_num_layers') if isinstance(checkpoint, dict) else None
    if num_layers is not None:
        return int(num_layers)

    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    layer_indices = []
    for key in _strip_module_prefixes(encoder_state):
        parts = str(key).split('.')
        if len(parts) > 2 and parts[0] == 'detailFeature' and parts[1] == 'net' and parts[2].isdigit():
            layer_indices.append(int(parts[2]))
    return max(layer_indices) + 1 if layer_indices else 1


def infer_cddfuse_detail_num_layers(checkpoint):
    num_layers = checkpoint.get('detail_fusion_num_layers') if isinstance(checkpoint, dict) else None
    if num_layers is not None:
        return int(num_layers)

    detail_state = checkpoint.get('DetailFuseLayer', {}) if isinstance(checkpoint, dict) else {}
    layer_indices = []
    for key in _strip_module_prefixes(detail_state):
        parts = str(key).split('.')
        if len(parts) > 1 and parts[0] == 'net' and parts[1].isdigit():
            layer_indices.append(int(parts[1]))
    return max(layer_indices) + 1 if layer_indices else 1


def infer_cddfuse_encoder_detail_enhance_layers(checkpoint):
    num_layers = checkpoint.get('encoder_detail_enhance_layers') if isinstance(checkpoint, dict) else None
    if num_layers is not None:
        return int(num_layers)

    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    layer_indices = []
    for key in _strip_module_prefixes(encoder_state):
        parts = str(key).split('.')
        if len(parts) > 2 and parts[0] == 'detailEnhance' and parts[1].isdigit():
            layer_indices.append(int(parts[1]))
    return max(layer_indices) + 1 if layer_indices else 0
    
if __name__ == '__main__':
    height = 128
    width = 128
    window_size = 8
    modelE = Restormer_Encoder().cuda()
    modelD = Restormer_Decoder().cuda()
