import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

from .SpatialMamba import LayerNorm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError as exc:
    selective_scan_fn = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None

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
                "in the training environment before using HFRM-Mamba. "
                f"Original import error: {MAMBA_IMPORT_ERROR!r}"
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


class GlobalLocalCrossModalMambaBlock(nn.Module):
    def __init__(self, dim=64):
        super(GlobalLocalCrossModalMambaBlock, self).__init__()
        self.ir_global_norm = LayerNorm(dim, 'WithBias')
        self.vi_global_norm = LayerNorm(dim, 'WithBias')
        self.ir_local_norm = LayerNorm(dim, 'WithBias')
        self.vi_local_norm = LayerNorm(dim, 'WithBias')

        self.ir_global_mamba = SSMOnly4Path(dim=dim)
        self.vi_global_mamba = SSMOnly4Path(dim=dim)
        self.ir_local_mamba = SSMOnly4Path(dim=dim)
        self.vi_local_mamba = SSMOnly4Path(dim=dim)

        self.ir_global_gate = self._make_gate(dim)
        self.vi_global_gate = self._make_gate(dim)
        self.ir_local_gate = self._make_gate(dim)
        self.vi_local_gate = self._make_gate(dim)

        self.ir_fusion_proj = nn.Conv2d(
            dim * 2, dim, kernel_size=1, bias=True)
        self.vi_fusion_proj = nn.Conv2d(
            dim * 2, dim, kernel_size=1, bias=True)

    def _make_gate(self, dim):
        gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self._init_gate(gate)
        return gate

    def _init_gate(self, gate):
        pointwise = gate[2]
        with torch.no_grad():
            pointwise.weight.zero_()
            pointwise.bias.fill_(-4.0)

    def forward(
        self,
        ir_global_feature,
        ir_local_feature,
        vi_global_feature,
        vi_local_feature,
    ):
        features = (
            ir_global_feature,
            ir_local_feature,
            vi_global_feature,
            vi_local_feature,
        )
        if any(feature.shape != ir_global_feature.shape for feature in features[1:]):
            raise ValueError(
                'IR/VI global/local features must have the same shape, got '
                f'{[feature.shape for feature in features]}.'
            )

        ir_global_attention = self.ir_global_gate(
            self.ir_global_mamba(self.ir_global_norm(ir_global_feature)))
        vi_global_attention = self.vi_global_gate(
            self.vi_global_mamba(self.vi_global_norm(vi_global_feature)))
        ir_local_attention = self.ir_local_gate(
            self.ir_local_mamba(self.ir_local_norm(ir_local_feature)))
        vi_local_attention = self.vi_local_gate(
            self.vi_local_mamba(self.vi_local_norm(vi_local_feature)))

        ir_global_enhanced = ir_global_feature * vi_global_attention + ir_global_feature
        vi_global_enhanced = vi_global_feature * ir_global_attention + vi_global_feature
        ir_local_enhanced = ir_local_feature * vi_local_attention + ir_local_feature
        vi_local_enhanced = vi_local_feature * ir_local_attention + vi_local_feature

        ir_enhanced = self.ir_fusion_proj(torch.cat(
            [ir_global_enhanced, ir_local_enhanced], dim=1))
        vi_enhanced = self.vi_fusion_proj(torch.cat(
            [vi_global_enhanced, vi_local_enhanced], dim=1))
        return ir_enhanced, vi_enhanced
