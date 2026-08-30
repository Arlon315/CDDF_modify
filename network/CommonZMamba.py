import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as checkpoint_fn

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


class CommonZMambaSeqBlock(nn.Module):
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
        super(CommonZMambaSeqBlock, self).__init__()
        if selective_scan_fn is None:
            raise ImportError(
                "CommonZMambaSeqBlock requires mamba_ssm. Install mamba-ssm "
                "in the training environment before using GMEM. "
                f"Original import error: {MAMBA_IMPORT_ERROR!r}"
            )

        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == 'auto' else int(dt_rank)

        self.ir_x_proj = nn.Linear(dim, self.d_inner, bias=bias)
        self.vi_x_proj = nn.Linear(dim, self.d_inner, bias=bias)
        self.common_norm = nn.LayerNorm(dim)
        self.common_z_proj = nn.Linear(dim, self.d_inner, bias=bias)

        self.ir_conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.vi_conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.ir_ssm_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.vi_ssm_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.ir_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.vi_dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj(
            self.ir_dt_proj, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)
        self._init_dt_proj(
            self.vi_dt_proj, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        self.ir_a_log = nn.Parameter(self._init_a_log())
        self.vi_a_log = nn.Parameter(self._init_a_log())
        self.ir_a_log._no_weight_decay = True
        self.vi_a_log._no_weight_decay = True

        self.ir_d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.vi_d_skip = nn.Parameter(torch.ones(self.d_inner))
        self.ir_d_skip._no_weight_decay = True
        self.vi_d_skip._no_weight_decay = True

        self.ir_out_norm = nn.LayerNorm(self.d_inner)
        self.vi_out_norm = nn.LayerNorm(self.d_inner)
        self.ir_out_proj = nn.Linear(self.d_inner, dim, bias=bias)
        self.vi_out_proj = nn.Linear(self.d_inner, dim, bias=bias)

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

    def _make_ssm_params(self, x, ssm_proj, dt_proj):
        batch, _, seqlen = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(batch * seqlen, self.d_inner)
        x_dbl = ssm_proj(x_flat)
        dt, b_param, c_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.linear(dt, dt_proj.weight)
        dt = dt.view(batch, seqlen, self.d_inner).permute(0, 2, 1).contiguous()
        b_param = b_param.view(
            batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        c_param = c_param.view(
            batch, seqlen, self.d_state).permute(0, 2, 1).contiguous()
        return dt, b_param, c_param

    def forward(self, ir_seq, vi_seq):
        if ir_seq.shape != vi_seq.shape:
            raise ValueError(
                f"ir_seq and vi_seq must have the same shape, got "
                f"{ir_seq.shape} and {vi_seq.shape}."
            )

        seqlen = ir_seq.shape[1]
        ir_x = self.ir_x_proj(ir_seq).transpose(1, 2).contiguous()
        vi_x = self.vi_x_proj(vi_seq).transpose(1, 2).contiguous()

        common = ir_seq + vi_seq + ir_seq * vi_seq
        common_z = self.common_z_proj(self.common_norm(common))
        common_z = common_z.transpose(1, 2).contiguous()

        ir_x = self._conv_act(ir_x, self.ir_conv1d, seqlen)
        vi_x = self._conv_act(vi_x, self.vi_conv1d, seqlen)
        ir_dt, ir_b, ir_c = self._make_ssm_params(
            ir_x, self.ir_ssm_proj, self.ir_dt_proj)
        vi_dt, vi_b, vi_c = self._make_ssm_params(
            vi_x, self.vi_ssm_proj, self.vi_dt_proj)

        ir_y = selective_scan_fn(
            ir_x,
            ir_dt,
            -torch.exp(self.ir_a_log.float()),
            ir_b,
            ir_c,
            self.ir_d_skip.float(),
            z=common_z,
            delta_bias=self.ir_dt_proj.bias.float(),
            delta_softplus=True,
        )
        vi_y = selective_scan_fn(
            vi_x,
            vi_dt,
            -torch.exp(self.vi_a_log.float()),
            vi_b,
            vi_c,
            self.vi_d_skip.float(),
            z=common_z,
            delta_bias=self.vi_dt_proj.bias.float(),
            delta_softplus=True,
        )

        ir_y = ir_y.transpose(1, 2).contiguous()
        vi_y = vi_y.transpose(1, 2).contiguous()
        ir_y = self.ir_out_proj(self.ir_out_norm(ir_y))
        vi_y = self.vi_out_proj(self.vi_out_norm(vi_y))
        return ir_y, vi_y


class GlobalMamba4Path(nn.Module):
    def __init__(self, dim=64):
        super(GlobalMamba4Path, self).__init__()
        self.use_checkpoint = True
        self.blocks = nn.ModuleList([
            CommonZMambaSeqBlock(dim=dim),
            CommonZMambaSeqBlock(dim=dim),
            CommonZMambaSeqBlock(dim=dim),
            CommonZMambaSeqBlock(dim=dim),
        ])

    def _run_block(self, block, ir_seq, vi_seq):
        if self.use_checkpoint and self.training and (
            ir_seq.requires_grad or vi_seq.requires_grad
        ):
            return checkpoint_fn(block, ir_seq, vi_seq)
        return block(ir_seq, vi_seq)

    def _h_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(
            batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _v_seq_to_img(self, seq, batch, channels, height, width):
        return seq.reshape(
            batch, width, height, channels).permute(0, 3, 2, 1).contiguous()

    def forward(self, ir_feature, vi_feature):
        if ir_feature.shape != vi_feature.shape:
            raise ValueError(
                f"ir_feature and vi_feature must have the same shape, got "
                f"{ir_feature.shape} and {vi_feature.shape}."
            )

        batch, channels, height, width = ir_feature.shape
        ir_h_fwd = ir_feature.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        vi_h_fwd = vi_feature.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels)
        ir_h_rev = torch.flip(ir_h_fwd, dims=[1])
        vi_h_rev = torch.flip(vi_h_fwd, dims=[1])

        ir_v_fwd = ir_feature.permute(0, 3, 2, 1).reshape(
            batch, width * height, channels)
        vi_v_fwd = vi_feature.permute(0, 3, 2, 1).reshape(
            batch, width * height, channels)
        ir_v_rev = torch.flip(ir_v_fwd, dims=[1])
        vi_v_rev = torch.flip(vi_v_fwd, dims=[1])

        h_fwd_block, h_rev_block, v_fwd_block, v_rev_block = self.blocks
        ir_h_fwd, vi_h_fwd = self._run_block(
            h_fwd_block, ir_h_fwd, vi_h_fwd)
        ir_h_rev, vi_h_rev = self._run_block(
            h_rev_block, ir_h_rev, vi_h_rev)
        ir_v_fwd, vi_v_fwd = self._run_block(
            v_fwd_block, ir_v_fwd, vi_v_fwd)
        ir_v_rev, vi_v_rev = self._run_block(
            v_rev_block, ir_v_rev, vi_v_rev)

        ir_h_rev = torch.flip(ir_h_rev, dims=[1])
        vi_h_rev = torch.flip(vi_h_rev, dims=[1])
        ir_v_rev = torch.flip(ir_v_rev, dims=[1])
        vi_v_rev = torch.flip(vi_v_rev, dims=[1])

        ir_h_fwd = self._h_seq_to_img(
            ir_h_fwd, batch, channels, height, width)
        vi_h_fwd = self._h_seq_to_img(
            vi_h_fwd, batch, channels, height, width)
        ir_h_rev = self._h_seq_to_img(
            ir_h_rev, batch, channels, height, width)
        vi_h_rev = self._h_seq_to_img(
            vi_h_rev, batch, channels, height, width)
        ir_v_fwd = self._v_seq_to_img(
            ir_v_fwd, batch, channels, height, width)
        vi_v_fwd = self._v_seq_to_img(
            vi_v_fwd, batch, channels, height, width)
        ir_v_rev = self._v_seq_to_img(
            ir_v_rev, batch, channels, height, width)
        vi_v_rev = self._v_seq_to_img(
            vi_v_rev, batch, channels, height, width)

        ir_long = (ir_h_fwd + ir_h_rev + ir_v_fwd + ir_v_rev)
        vi_long = (vi_h_fwd + vi_h_rev + vi_v_fwd + vi_v_rev)
        return ir_long, vi_long
