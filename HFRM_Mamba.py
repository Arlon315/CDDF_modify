import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from SpatialMamba import LayerNorm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class SSMOnlySeqBlock(nn.Module):
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
        super(SSMOnlySeqBlock, self).__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "SSMOnlySeqBlock requires mamba_ssm. Install mamba-ssm "
                "in the training environment before using HFRM-Mamba."
            )

        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == 'auto' else int(dt_rank)

        self.in_proj = nn.Linear(dim, self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj(
            dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        self.A_log = nn.Parameter(self._init_a_log())
        self.A_log._no_weight_decay = True

        self.D_skip = nn.Parameter(torch.ones(self.d_inner))
        self.D_skip._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, dim, bias=bias)

    def _init_dt_proj(
        self, dt_min, dt_max, dt_init, dt_scale, dt_init_floor
    ):
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(
                self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def _init_a_log(self):
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a = a.repeat(self.d_inner, 1).contiguous()
        return torch.log(a)

    def _conv_act(self, x, seqlen):
        if causal_conv1d_fn is not None and x.is_cuda:
            return causal_conv1d_fn(
                x=x,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation='silu',
            )
        return self.act(self.conv1d(x)[..., :seqlen])

    def _make_ssm_params(self, x):
        batch, _, seqlen = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(
            batch * seqlen, self.d_inner)
        x_dbl = self.x_proj(x_flat)
        dt, b_param, c_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.linear(dt, self.dt_proj.weight)
        dt = dt.view(
            batch, seqlen, self.d_inner).permute(0, 2, 1).contiguous()
        b_param = b_param.view(
            batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        c_param = c_param.view(
            batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        return dt, b_param, c_param

    def forward(self, sequence):
        seqlen = sequence.shape[1]
        x = self.in_proj(sequence).transpose(1, 2).contiguous()
        x = self._conv_act(x, seqlen)

        dt, b_param, c_param = self._make_ssm_params(x)
        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(self.A_log.float()),
            b_param,
            c_param,
            self.D_skip.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = y.transpose(1, 2).contiguous()
        return self.out_proj(y)


class SSMOnly4Path(nn.Module):
    def __init__(self, dim=64):
        super(SSMOnly4Path, self).__init__()
        self.mambas = nn.ModuleList([
            SSMOnlySeqBlock(dim=dim),
            SSMOnlySeqBlock(dim=dim),
            SSMOnlySeqBlock(dim=dim),
            SSMOnlySeqBlock(dim=dim),
        ])
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def _run_mamba(self, mamba, sequence):
        if self.training and sequence.requires_grad:
            return checkpoint_fn(
                mamba, sequence, use_reentrant=False)
        return mamba(sequence)

    def _h_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(
            batch, height, width, channels
        ).permute(0, 3, 1, 2).contiguous()

    def _v_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(
            batch, width, height, channels
        ).permute(0, 3, 2, 1).contiguous()

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        h_fwd = feature.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        h_rev = torch.flip(h_fwd, dims=[1])
        v_fwd = feature.permute(0, 3, 2, 1).reshape(
            batch, width * height, channels)
        v_rev = torch.flip(v_fwd, dims=[1])

        h_fwd = self._run_mamba(self.mambas[0], h_fwd)
        h_rev = self._run_mamba(self.mambas[1], h_rev)
        v_fwd = self._run_mamba(self.mambas[2], v_fwd)
        v_rev = self._run_mamba(self.mambas[3], v_rev)

        h_rev = torch.flip(h_rev, dims=[1])
        v_rev = torch.flip(v_rev, dims=[1])

        h_fwd = self._h_seq_to_img(
            h_fwd, batch, channels, height, width)
        h_rev = self._h_seq_to_img(
            h_rev, batch, channels, height, width)
        v_fwd = self._v_seq_to_img(
            v_fwd, batch, channels, height, width)
        v_rev = self._v_seq_to_img(
            v_rev, batch, channels, height, width)

        return self.proj(h_fwd + h_rev + v_fwd + v_rev)


class HighLowFrequencyReciprocalMambaBlock(nn.Module):
    def __init__(self, dim=64, out_dim=None):
        super(HighLowFrequencyReciprocalMambaBlock, self).__init__()
        out_dim = dim if out_dim is None else int(out_dim)

        self.low_norm = LayerNorm(dim, 'WithBias')
        self.high_norm = LayerNorm(dim, 'WithBias')
        self.low_mamba = SSMOnly4Path(dim=dim)
        self.high_mamba = SSMOnly4Path(dim=dim)

        self.low_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.high_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_gate(self.low_gate)
        self._init_gate(self.high_gate)

        self.low_value = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.high_value = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.fusion_proj = nn.Conv2d(
            dim, out_dim, kernel_size=1, bias=True)

    def _init_gate(self, gate):
        pointwise = gate[2]
        with torch.no_grad():
            pointwise.weight.zero_()
            pointwise.bias.fill_(-4.0)

    def forward(self, low_feature, high_feature):
        if low_feature.shape != high_feature.shape:
            raise ValueError(
                f"low_feature and high_feature must have the same shape, got "
                f"{low_feature.shape} and {high_feature.shape}."
            )

        low_feature = self.low_norm(low_feature)
        high_feature = self.high_norm(high_feature)

        low_context = self.low_mamba(low_feature)
        high_context = self.high_mamba(high_feature)
        low_attention = self.low_gate(low_context)
        high_attention = self.high_gate(high_context)

        low_content = self.low_value(low_feature)
        high_content = self.high_value(high_feature)
        high_enhanced = high_content * (1.0 + low_attention)
        low_enhanced = low_content * (1.0 + high_attention)

        return self.fusion_proj(low_enhanced + high_enhanced)
