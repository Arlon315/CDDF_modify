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
                "in the training environment before using CMEM_detail."
            )

        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank == 'auto' else int(dt_rank)

        self.in_proj_ir = nn.Linear(dim, self.d_inner * 2, bias=bias)
        self.in_proj_vi = nn.Linear(dim, self.d_inner * 2, bias=bias)
        self.conv1d_ir = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.conv1d_vi = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        self.x_proj_ir = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.x_proj_vi = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj_ir = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.dt_proj_vi = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj(self.dt_proj_ir, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)
        self._init_dt_proj(self.dt_proj_vi, dt_min, dt_max, dt_init, dt_scale, dt_init_floor)

        self.A_log_ir = nn.Parameter(self._init_a_log())
        self.A_log_vi = nn.Parameter(self._init_a_log())
        self.A_log_ir._no_weight_decay = True
        self.A_log_vi._no_weight_decay = True

        self.D_skip_ir = nn.Parameter(torch.ones(self.d_inner))
        self.D_skip_vi = nn.Parameter(torch.ones(self.d_inner))
        self.D_skip_ir._no_weight_decay = True
        self.D_skip_vi._no_weight_decay = True

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

    def forward(self, ir_seq, vi_seq):
        if ir_seq.shape != vi_seq.shape:
            raise ValueError(
                f"ir_seq and vi_seq must have the same shape, got "
                f"{ir_seq.shape} and {vi_seq.shape}."
            )

        seqlen = ir_seq.shape[1]
        ir_xz = self.in_proj_ir(ir_seq)
        vi_xz = self.in_proj_vi(vi_seq)
        ir_x, ir_z = ir_xz.chunk(2, dim=-1)
        vi_x, vi_z = vi_xz.chunk(2, dim=-1)

        ir_x = ir_x.transpose(1, 2).contiguous()
        vi_x = vi_x.transpose(1, 2).contiguous()
        ir_z = ir_z.transpose(1, 2).contiguous()
        vi_z = vi_z.transpose(1, 2).contiguous()

        ir_x = self._conv_act(ir_x, self.conv1d_ir, seqlen)
        vi_x = self._conv_act(vi_x, self.conv1d_vi, seqlen)

        ir_dt, ir_b, ir_c = self._make_ssm_params(ir_x, self.x_proj_ir, self.dt_proj_ir)
        vi_dt, vi_b, vi_c = self._make_ssm_params(vi_x, self.x_proj_vi, self.dt_proj_vi)

        ir_a = -torch.exp(self.A_log_ir.float())
        vi_a = -torch.exp(self.A_log_vi.float())
        ir_y = selective_scan_fn(
            ir_x,
            ir_dt,
            ir_a,
            ir_b,
            ir_c,
            self.D_skip_ir.float(),
            z=vi_z,
            delta_bias=self.dt_proj_ir.bias.float(),
            delta_softplus=True,
        )
        vi_y = selective_scan_fn(
            vi_x,
            vi_dt,
            vi_a,
            vi_b,
            vi_c,
            self.D_skip_vi.float(),
            z=ir_z,
            delta_bias=self.dt_proj_vi.bias.float(),
            delta_softplus=True,
        )

        out = (ir_y + vi_y).transpose(1, 2).contiguous()
        return self.out_proj(out)


class CrossSpatialMamba4Path(nn.Module):
    def __init__(self, dim=64, share_mamba=False, use_checkpoint=True):
        super(CrossSpatialMamba4Path, self).__init__()
        self.share_mamba = share_mamba
        self.use_checkpoint = use_checkpoint
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

    def _run_block(self, block, ir_seq, vi_seq):
        if self.use_checkpoint and self.training and (ir_seq.requires_grad or vi_seq.requires_grad):
            return checkpoint_fn(block, ir_seq, vi_seq)
        return block(ir_seq, vi_seq)

    def forward(self, ir_feature, vi_feature):
        if ir_feature.shape != vi_feature.shape:
            raise ValueError(
                f"ir_feature and vi_feature must have the same shape, got "
                f"{ir_feature.shape} and {vi_feature.shape}."
            )

        batch, channels, height, width = ir_feature.shape
        ir_h_fwd = ir_feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        vi_h_fwd = vi_feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        ir_h_rev = torch.flip(ir_h_fwd, dims=[1])
        vi_h_rev = torch.flip(vi_h_fwd, dims=[1])

        ir_v_fwd = ir_feature.permute(0, 3, 2, 1).reshape(batch, width * height, channels)
        vi_v_fwd = vi_feature.permute(0, 3, 2, 1).reshape(batch, width * height, channels)
        ir_v_rev = torch.flip(ir_v_fwd, dims=[1])
        vi_v_rev = torch.flip(vi_v_fwd, dims=[1])

        if self.share_mamba:
            ir_seq = torch.cat([ir_h_fwd, ir_h_rev, ir_v_fwd, ir_v_rev], dim=0)
            vi_seq = torch.cat([vi_h_fwd, vi_h_rev, vi_v_fwd, vi_v_rev], dim=0)
            out_seq = self._run_block(self.block, ir_seq, vi_seq)
            h_fwd, h_rev, v_fwd, v_rev = torch.chunk(out_seq, 4, dim=0)
        else:
            h_fwd = self._run_block(self.blocks[0], ir_h_fwd, vi_h_fwd)
            h_rev = self._run_block(self.blocks[1], ir_h_rev, vi_h_rev)
            v_fwd = self._run_block(self.blocks[2], ir_v_fwd, vi_v_fwd)
            v_rev = self._run_block(self.blocks[3], ir_v_rev, vi_v_rev)

        h_rev = torch.flip(h_rev, dims=[1])
        v_rev = torch.flip(v_rev, dims=[1])

        h_fwd = self._h_seq_to_img(h_fwd, batch, channels, height, width)
        h_rev = self._h_seq_to_img(h_rev, batch, channels, height, width)
        v_fwd = self._v_seq_to_img(v_fwd, batch, channels, height, width)
        v_rev = self._v_seq_to_img(v_rev, batch, channels, height, width)

        out = (h_fwd + h_rev + v_fwd + v_rev) / 4.0
        return self.proj(out)


class CMEMDetailFusion(nn.Module):
    dual_input = True

    def __init__(self, dim=64, share_mamba=False, use_checkpoint=True):
        super(CMEMDetailFusion, self).__init__()
        self.ir_norm = LayerNorm(dim, 'WithBias')
        self.vi_norm = LayerNorm(dim, 'WithBias')
        self.cross_mixer = CrossSpatialMamba4Path(
            dim=dim,
            share_mamba=share_mamba,
            use_checkpoint=use_checkpoint,
        )
        self.merge = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, ir_detail_feature, vi_detail_feature):
        cross = self.cross_mixer(
            self.ir_norm(ir_detail_feature),
            self.vi_norm(vi_detail_feature),
        )
        cross = self.merge(cross)
        return ir_detail_feature + vi_detail_feature + cross
