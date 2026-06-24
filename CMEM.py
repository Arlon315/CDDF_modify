import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from SpatialMamba import LayerNorm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


def _strip_module_prefix(key):
    return key[7:] if isinstance(key, str) and key.startswith('module.') else key


def is_cmem_checkpoint(checkpoint):
    state_dict = checkpoint.get('FMEMLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = [_strip_module_prefix(key) for key in state_dict.keys()]
    return any(str(key).startswith('cross_mixer.') for key in keys)


def infer_cmem_share_mamba(checkpoint):
    if isinstance(checkpoint, dict) and 'cmem_share_mamba' in checkpoint:
        return bool(checkpoint['cmem_share_mamba'])

    state_dict = checkpoint.get('FMEMLayer', {}) if isinstance(checkpoint, dict) else {}
    keys = [_strip_module_prefix(key) for key in state_dict.keys()]
    if any(str(key).startswith('cross_mixer.blocks.') for key in keys):
        return False
    if any(str(key).startswith('cross_mixer.block.') for key in keys):
        return True
    return False


class CrossMambaSeqBlock(nn.Module):
    def __init__(
        self,
        dim=64,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank='auto',
        dt_min=0.001,
        dt_max=0.1,
        dt_init='random',
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
    ):
        super(CrossMambaSeqBlock, self).__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "CrossMambaSeqBlock requires mamba_ssm. Install mamba-ssm "
                "in the training environment before using CMEM."
            )

        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == 'auto' else int(dt_rank)

        self.in_proj_d = nn.Linear(dim, self.d_inner * 2, bias=bias)
        self.in_proj_b = nn.Linear(dim, self.d_inner * 2, bias=bias)
        self.conv1d_d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.conv1d_b = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj_d = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_d = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj(self.dt_proj_d, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)
        self._init_dt_proj(self.dt_proj_b, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        self.A_log_d = nn.Parameter(self._init_a_log())
        self.A_log_b = nn.Parameter(self._init_a_log())
        self.A_log_d._no_weight_decay = True
        self.A_log_b._no_weight_decay = True

        self.D_skip_d = nn.Parameter(torch.ones(self.d_inner))
        self.D_skip_b = nn.Parameter(torch.ones(self.d_inner))
        self.D_skip_d._no_weight_decay = True
        self.D_skip_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

    def _init_dt_proj(self, module, dt_min, dt_max, dt_init, dt_scale, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(module.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(module.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            module.bias.copy_(inv_dt)
        module.bias._no_reinit = True

    def _init_a_log(self):
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a = a.repeat(self.d_inner, 1).contiguous()
        return torch.log(a)

    def _conv_act(self, x, conv1d, seqlen):
        if causal_conv1d_fn is not None and x.is_cuda:
            return causal_conv1d_fn(
                x=x,
                weight=conv1d.weight.squeeze(1),
                bias=conv1d.bias,
                activation='silu',
            )
        return self.act(conv1d(x)[..., :seqlen])

    def _make_ssm_params(self, x, x_proj, dt_proj):
        batch, _, seqlen = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(batch * seqlen, self.d_inner)
        x_dbl = x_proj(x_flat)
        dt, b_param, c_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.linear(dt, dt_proj.weight)
        dt = dt.view(batch, seqlen, self.d_inner).permute(0, 2, 1).contiguous()
        b_param = b_param.view(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        c_param = c_param.view(batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        return dt, b_param, c_param

    def forward(self, detail_seq, base_seq):
        if detail_seq.shape != base_seq.shape:
            raise ValueError(
                f"detail_seq and base_seq must have the same shape, got "
                f"{detail_seq.shape} and {base_seq.shape}."
            )

        seqlen = detail_seq.shape[1]
        detail_xz = self.in_proj_d(detail_seq)
        base_xz = self.in_proj_b(base_seq)
        detail_x, detail_z = detail_xz.chunk(2, dim=-1)
        base_x, base_z = base_xz.chunk(2, dim=-1)

        detail_x = detail_x.transpose(1, 2).contiguous()
        base_x = base_x.transpose(1, 2).contiguous()
        detail_z = detail_z.transpose(1, 2).contiguous()
        base_z = base_z.transpose(1, 2).contiguous()

        detail_x = self._conv_act(detail_x, self.conv1d_d, seqlen)
        base_x = self._conv_act(base_x, self.conv1d_b, seqlen)

        detail_dt, detail_b, detail_c = self._make_ssm_params(
            detail_x, self.x_proj_d, self.dt_proj_d)
        base_dt, base_b, base_c = self._make_ssm_params(
            base_x, self.x_proj_b, self.dt_proj_b)

        detail_a = -torch.exp(self.A_log_d.float())
        base_a = -torch.exp(self.A_log_b.float())
        detail_y = selective_scan_fn(
            detail_x,
            detail_dt,
            detail_a,
            detail_b,
            detail_c,
            self.D_skip_d.float(),
            z=base_z,
            delta_bias=self.dt_proj_d.bias.float(),
            delta_softplus=True,
        )
        base_y = selective_scan_fn(
            base_x,
            base_dt,
            base_a,
            base_b,
            base_c,
            self.D_skip_b.float(),
            z=detail_z,
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
        )

        out = (detail_y + base_y).transpose(1, 2).contiguous()
        return self.out_proj(out)


class CrossSpatialMamba4Path(nn.Module):
    def __init__(self, dim=64, share_mamba=False):
        super(CrossSpatialMamba4Path, self).__init__()
        self.share_mamba = share_mamba
        if share_mamba:
            self.block = CrossMambaSeqBlock(dim=dim)
        else:
            self.blocks = nn.ModuleList([
                CrossMambaSeqBlock(dim=dim),
                CrossMambaSeqBlock(dim=dim),
                CrossMambaSeqBlock(dim=dim),
                CrossMambaSeqBlock(dim=dim),
            ])
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def _h_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _v_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(batch, width, height, channels).permute(0, 3, 2, 1).contiguous()

    def forward(self, detail_feature, base_feature):
        if detail_feature.shape != base_feature.shape:
            raise ValueError(
                f"detail_feature and base_feature must have the same shape, got "
                f"{detail_feature.shape} and {base_feature.shape}."
            )

        batch, channels, height, width = detail_feature.shape
        detail_h_fwd = detail_feature.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        base_h_fwd = base_feature.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        detail_h_rev = torch.flip(detail_h_fwd, dims=[1])
        base_h_rev = torch.flip(base_h_fwd, dims=[1])

        detail_v_fwd = detail_feature.permute(0, 3, 2, 1).reshape(
            batch, width * height, channels)
        base_v_fwd = base_feature.permute(0, 3, 2, 1).reshape(
            batch, width * height, channels)
        detail_v_rev = torch.flip(detail_v_fwd, dims=[1])
        base_v_rev = torch.flip(base_v_fwd, dims=[1])

        if self.share_mamba:
            detail_seq = torch.cat(
                [detail_h_fwd, detail_h_rev, detail_v_fwd, detail_v_rev], dim=0)
            base_seq = torch.cat(
                [base_h_fwd, base_h_rev, base_v_fwd, base_v_rev], dim=0)
            out_seq = self.block(detail_seq, base_seq)
            h_fwd, h_rev, v_fwd, v_rev = torch.chunk(out_seq, 4, dim=0)
        else:
            h_fwd = self.blocks[0](detail_h_fwd, base_h_fwd)
            h_rev = self.blocks[1](detail_h_rev, base_h_rev)
            v_fwd = self.blocks[2](detail_v_fwd, base_v_fwd)
            v_rev = self.blocks[3](detail_v_rev, base_v_rev)

        h_rev = torch.flip(h_rev, dims=[1])
        v_rev = torch.flip(v_rev, dims=[1])

        h_fwd = self._h_seq_to_img(h_fwd, batch, channels, height, width)
        h_rev = self._h_seq_to_img(h_rev, batch, channels, height, width)
        v_fwd = self._v_seq_to_img(v_fwd, batch, channels, height, width)
        v_rev = self._v_seq_to_img(v_rev, batch, channels, height, width)

        out = (h_fwd + h_rev + v_fwd + v_rev) / 4.0
        return self.proj(out)


class CrossMambaEnhanceModule(nn.Module):
    def __init__(self, dim=64, share_mamba=False):
        super(CrossMambaEnhanceModule, self).__init__()
        self.detail_norm = LayerNorm(dim, 'WithBias')
        self.base_norm = LayerNorm(dim, 'WithBias')
        self.cross_mixer = CrossSpatialMamba4Path(dim=dim, share_mamba=share_mamba)
        self.merge = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, detail_feature, base_feature):
        cross = self.cross_mixer(
            self.detail_norm(detail_feature),
            self.base_norm(base_feature),
        )
        cross = self.merge(cross)
        return detail_feature + base_feature + torch.tanh(self.alpha) * cross
