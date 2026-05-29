import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_SELECTIVE_SCAN_CUDA = None


def _load_selective_scan_cuda():
    global _SELECTIVE_SCAN_CUDA
    if _SELECTIVE_SCAN_CUDA is not None:
        return _SELECTIVE_SCAN_CUDA

    try:
        import mamba_ssm  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "CMFMOnlyDetailFusion requires `mamba_ssm` and its selective-scan CUDA "
            "runtime. Install a compatible `mamba-ssm` build before using "
            "detail_fusion='cmfm'."
        ) from exc

    try:
        import selective_scan_cuda
    except ImportError as exc:
        raise ImportError(
            "CMFMOnlyDetailFusion could not import `selective_scan_cuda`. Install "
            "a `mamba-ssm` build that matches the active PyTorch/CUDA runtime before "
            "using detail_fusion='cmfm'."
        ) from exc

    _SELECTIVE_SCAN_CUDA = selective_scan_cuda
    return _SELECTIVE_SCAN_CUDA


class EfficientMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys, ori_h, ori_w, step_size=2):
        if step_size != 2:
            raise ValueError("EfficientMerge currently supports step_size=2.")

        b, _, c, _ = ys.shape
        h = math.ceil(ori_h / step_size)
        w = math.ceil(ori_w / step_size)
        ctx.shape = (h, w)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size

        new_h = h * step_size
        new_w = w * step_size
        y = ys.new_empty((b, c, new_h, new_w))
        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(b, c, h, w)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(b, c, w, h).transpose(2, 3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(b, c, h, w)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(b, c, w, h).transpose(2, 3)

        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        return y.view(b, c, -1)

    @staticmethod
    def backward(ctx, grad_x):
        b, c, _ = grad_x.shape
        step_size = ctx.step_size
        grad_x = grad_x.view(b, c, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))
        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))

        h = grad_x.shape[2] // step_size
        w = grad_x.shape[3] // step_size
        grad_xs = grad_x.new_empty((b, 4, c, h * w))
        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(b, c, -1)
        grad_xs[:, 1] = grad_x.transpose(2, 3)[:, :, ::step_size, 1::step_size].reshape(b, c, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(b, c, -1)
        grad_xs[:, 3] = grad_x.transpose(2, 3)[:, :, 1::step_size, 1::step_size].reshape(b, c, -1)
        return grad_xs, None, None, None


class EfficientScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size=2):
        if step_size != 2:
            raise ValueError("EfficientScan currently supports step_size=2.")

        b, c, org_h, org_w = x.shape
        ctx.shape = (b, c, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))

        h = x.shape[2] // step_size
        w = x.shape[3] // step_size
        xs = x.new_empty((b, 4, c, h * w))
        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(b, c, -1)
        xs[:, 1] = x.transpose(2, 3)[:, :, ::step_size, 1::step_size].contiguous().view(b, c, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(b, c, -1)
        xs[:, 3] = x.transpose(2, 3)[:, :, 1::step_size, 1::step_size].contiguous().view(b, c, -1)
        return xs.view(b, 4, c, -1)

    @staticmethod
    def backward(ctx, grad_xs):
        b, c, org_h, org_w = ctx.shape
        step_size = ctx.step_size
        new_h = math.ceil(org_h / step_size)
        new_w = math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((b, c, new_h * step_size, new_w * step_size))
        grad_xs = grad_xs.view(b, 4, c, new_h, new_w)

        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(b, c, new_h, new_w)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(b, c, new_w, new_h).transpose(2, 3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(b, c, new_h, new_w)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(b, c, new_w, new_h).transpose(2, 3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]
        return grad_x, None


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, a, b, c, d=None, delta_bias=None, delta_softplus=False, nrows=1):
        selective_scan_cuda = _load_selective_scan_cuda()
        if nrows not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported nrows for selective scan: {nrows}")
        if u.shape[1] % (b.shape[1] * nrows) != 0:
            raise ValueError(f"Incompatible selective scan shapes: nrows={nrows}, u={u.shape}, B={b.shape}")

        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.squeeze_b = False
        ctx.squeeze_c = False

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if d is not None:
            d = d.contiguous()
        if b.stride(-1) != 1:
            b = b.contiguous()
        if c.stride(-1) != 1:
            c = c.contiguous()
        if b.dim() == 3:
            b = b.unsqueeze(1)
            ctx.squeeze_b = True
        if c.dim() == 3:
            c = c.unsqueeze(1)
            ctx.squeeze_c = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, a, b, c, d, None, delta_bias, delta_softplus)
        del rest
        ctx.save_for_backward(u, delta, a, b, c, d, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        selective_scan_cuda = _load_selective_scan_cuda()
        u, delta, a, b, c, d, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, da, db, dc, dd, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u,
            delta,
            a,
            b,
            c,
            d,
            None,
            delta_bias,
            dout,
            x,
            None,
            None,
            ctx.delta_softplus,
            False,
        )
        del rest
        if ctx.squeeze_b:
            db = db.squeeze(1)
        if ctx.squeeze_c:
            dc = dc.squeeze(1)
        return du, ddelta, da, db, dc, dd, ddelta_bias, None, None


def cross_selective_scan_cross(
    x1,
    x2,
    x_proj_weight,
    x_proj_bias,
    dt_projs_weight,
    dt_projs_bias,
    a_logs,
    ds,
    out_norm,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    step_size=2,
):
    b, _, h, w = x1.shape
    d_total, d_state = a_logs.shape
    k, d_inner, dt_rank = dt_projs_weight.shape
    del d_total, dt_rank

    if nrows < 1:
        if d_inner % 4 == 0:
            nrows = 4
        elif d_inner % 3 == 0:
            nrows = 3
        elif d_inner % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    x = x1 * x2 + x1 + x2
    xs = EfficientScan.apply(x, step_size)
    scan_h = math.ceil(h / step_size)
    scan_w = math.ceil(w / step_size)
    scan_l = scan_h * scan_w

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, k, -1, 1)
    dts, bs, cs = torch.split(x_dbl, [dt_projs_weight.shape[2], d_state, d_state], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(b, -1, scan_l).to(torch.float)
    dts = dts.contiguous().view(b, -1, scan_l).to(torch.float)
    scan_as = -torch.exp(a_logs.to(torch.float))
    bs = bs.contiguous().to(torch.float)
    cs = cs.contiguous().to(torch.float)
    ds = ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    ys = SelectiveScan.apply(
        xs,
        dts,
        scan_as,
        bs,
        cs,
        ds,
        delta_bias,
        delta_softplus,
        nrows,
    ).view(b, k, -1, scan_l)

    y = EfficientMerge.apply(ys, int(h), int(w), step_size)
    y = y.transpose(1, 2).contiguous()
    y = out_norm(y).view(b, h, w, -1)
    return y.to(x1.dtype) if to_dtype else y


class SS2D_cross_new(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        forward_type="v2",
        step_size=2,
        **kwargs,
    ):
        super(SS2D_cross_new, self).__init__()
        del kwargs
        if step_size != 2:
            raise ValueError("SS2D_cross_new currently supports step_size=2.")

        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.step_size = step_size

        self.disable_z_act = forward_type.endswith("nozact")
        if self.disable_z_act:
            forward_type = forward_type[: -len("nozact")]

        if forward_type.endswith("softmax"):
            forward_type = forward_type[: -len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type.endswith("sigmoid"):
            forward_type = forward_type[: -len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        if forward_type not in ("v1", "v2"):
            raise NotImplementedError("SS2D_cross_new only enables forward_type='v2' in this project.")
        self.forward_core_type = forward_type
        self.k = 4 if forward_type not in ("share_ssm",) else 1
        self.k2 = self.k if forward_type not in ("share_a",) else 1

        self.in_proj1 = nn.Linear(d_model, d_expand * 2, bias=bias)
        self.in_proj2 = nn.Linear(d_model, d_expand * 2, bias=bias)
        self.act1 = act_layer()
        self.act2 = act_layer()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False)

        x_proj = [
            nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.k)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = [
            self.dt_init(
                self.dt_rank,
                d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
            )
            for _ in range(self.k)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.k2, merge=True)
        self.Ds = self.D_init(d_inner, copies=self.k2, merge=True)
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.k2 * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k2 * d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(torch.randn((self.k, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.k, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        a = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, d_state)
        a = a.repeat(d_inner, 1).contiguous()
        a_log = torch.log(a)
        if copies > 0:
            a_log = a_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge:
                a_log = a_log.flatten(0, 1)
        a_log = nn.Parameter(a_log)
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        d = torch.ones(d_inner, device=device)
        if copies > 0:
            d = d.unsqueeze(0).repeat(copies, 1)
            if merge:
                d = d.flatten(0, 1)
        d = nn.Parameter(d)
        d._no_weight_decay = True
        return d

    def forward_corev0(self, *args, **kwargs):
        raise NotImplementedError("SS2D_cross_new only enables forward_type='v2' in this project.")

    def forward_corev0_seq(self, *args, **kwargs):
        raise NotImplementedError("SS2D_cross_new only enables forward_type='v2' in this project.")

    def forward_corev0_share_ssm(self, *args, **kwargs):
        raise NotImplementedError("SS2D_cross_new only enables forward_type='v2' in this project.")

    def forward_corev0_share_a(self, *args, **kwargs):
        raise NotImplementedError("SS2D_cross_new only enables forward_type='v2' in this project.")

    def forward_corev2(self, x1, x2, nrows=-1, channel_first=False, step_size=2):
        nrows = 1
        if not channel_first:
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x1 = self.in_rank(x1)
            x2 = self.in_rank(x2)
        x = cross_selective_scan_cross(
            x1,
            x2,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            self.out_norm,
            nrows=nrows,
            delta_softplus=True,
            step_size=step_size,
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x1, x2, **kwargs):
        del kwargs
        xz1 = self.in_proj1(x1)
        xz2 = self.in_proj2(x2)
        if self.d_conv > 1:
            x1, z1 = xz1.chunk(2, dim=-1)
            x2, z2 = xz2.chunk(2, dim=-1)
            if not self.disable_z_act:
                z1 = self.act1(z1)
                z2 = self.act2(z2)
            x1 = x1.permute(0, 3, 1, 2).contiguous()
            x2 = x2.permute(0, 3, 1, 2).contiguous()
            x1 = self.act1(self.conv2d(x1))
            x2 = self.act2(self.conv2d(x2))
        elif self.disable_z_act:
            x1, z1 = xz1.chunk(2, dim=-1)
            x2, z2 = xz2.chunk(2, dim=-1)
            x1 = self.act1(x1)
            x2 = self.act2(x2)
        else:
            xz1 = self.act1(xz1)
            xz2 = self.act2(xz2)
            x1, z1 = xz1.chunk(2, dim=-1)
            x2, z2 = xz2.chunk(2, dim=-1)

        y = self.forward_corev2(x1, x2, channel_first=(self.d_conv > 1), step_size=self.step_size)
        y = y * z1 + y * z2
        return self.dropout(self.out_proj(y))


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        del channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CMFMOnlyDetailFusion(nn.Module):
    dual_input = True

    def __init__(self, dim=64, d_state=16, step_size=2):
        super(CMFMOnlyDetailFusion, self).__init__()
        _load_selective_scan_cuda()
        self.dim = dim
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.cmfm = SS2D_cross_new(
            d_model=dim,
            d_state=d_state,
            dropout=0.0,
            step_size=step_size,
        )
        self.eca = eca_layer(channel=dim)

    def forward(self, feature_i_d, feature_v_d):
        if feature_i_d.shape != feature_v_d.shape:
            raise ValueError(
                f"CMFMOnlyDetailFusion input shapes must match, got "
                f"{tuple(feature_i_d.shape)} and {tuple(feature_v_d.shape)}."
            )
        if feature_i_d.dim() != 4 or feature_i_d.shape[1] != self.dim:
            raise ValueError(
                f"CMFMOnlyDetailFusion expects NCHW tensors with {self.dim} channels, "
                f"got {tuple(feature_i_d.shape)}."
            )

        input1 = feature_i_d.permute(0, 2, 3, 1).contiguous()
        input2 = feature_v_d.permute(0, 2, 3, 1).contiguous()
        cross = self.cmfm(self.ln_1(input1), self.ln_2(input2))

        cross_nchw = cross.permute(0, 3, 1, 2).contiguous()
        cross_eca = self.eca(cross_nchw).permute(0, 2, 3, 1).contiguous()
        out = input1 + input2 + cross + cross_eca
        return out.permute(0, 3, 1, 2).contiguous()
