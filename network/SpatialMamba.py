import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    Mamba = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if layer_norm_type != 'WithBias':
            raise ValueError("SpatialMamba LayerNorm only supports 'WithBias'.")
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        return self.body(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 ffn_expansion_factor=2,
                 bias=False):
        super(Mlp, self).__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2,
                                kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class SpatialMamba4Path(nn.Module):
    def __init__(self, dim=64):
        super(SpatialMamba4Path, self).__init__()
        if Mamba is None:
            raise ImportError(
                "SpatialMamba4Path requires mamba_ssm. Install mamba-ssm "
                "in the training environment before using this block. "
                f"Original import error: {MAMBA_IMPORT_ERROR!r}"
            )

        self.mambas = nn.ModuleList([
            Mamba(d_model=dim),
            Mamba(d_model=dim),
            Mamba(d_model=dim),
            Mamba(d_model=dim),
        ])

        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def _h_seq_to_img(self, seq, B, C, H, W):
        return seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _v_seq_to_img(self, seq, B, C, H, W):
        return seq.reshape(B, W, H, C).permute(0, 3, 2, 1).contiguous()

    def forward(self, x):
        B, C, H, W = x.shape

        h_fwd = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h_rev = torch.flip(h_fwd, dims=[1])
        v_fwd = x.permute(0, 3, 2, 1).reshape(B, W * H, C)
        v_rev = torch.flip(v_fwd, dims=[1])

        h_fwd = self.mambas[0](h_fwd)
        h_rev = self.mambas[1](h_rev)
        v_fwd = self.mambas[2](v_fwd)
        v_rev = self.mambas[3](v_rev)

        h_rev = torch.flip(h_rev, dims=[1])
        v_rev = torch.flip(v_rev, dims=[1])

        h_fwd = self._h_seq_to_img(h_fwd, B, C, H, W)
        h_rev = self._h_seq_to_img(h_rev, B, C, H, W)
        v_fwd = self._v_seq_to_img(v_fwd, B, C, H, W)
        v_rev = self._v_seq_to_img(v_rev, B, C, H, W)

        out = h_fwd + h_rev + v_fwd + v_rev
        out = self.proj(out)
        return out


class SpatialMambaBaseLayer(nn.Module):
    def __init__(self, dim=64, ffn_expansion_factor=1.0):
        super(SpatialMambaBaseLayer, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.spatial_mamba = SpatialMamba4Path(dim=dim)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(
            in_features=dim,
            ffn_expansion_factor=ffn_expansion_factor
        )

    def forward(self, x):
        x = x + self.spatial_mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialMambaGlobalFeature(nn.Module):
    def __init__(self,
                 dim=64,
                 num_layers=1,
                 ffn_expansion_factor=1.0):
        super(SpatialMambaGlobalFeature, self).__init__()
        self.layers = nn.ModuleList([
            SpatialMambaBaseLayer(
                dim=dim,
                ffn_expansion_factor=ffn_expansion_factor,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
