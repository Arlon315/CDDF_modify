import torch
import torch.nn as nn


def _load_mamba_cls():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except ImportError as exc:
        raise ImportError(
            "RandomMamba requires `mamba_ssm`. Install a compatible "
            "`mamba-ssm[causal-conv1d]` build in the ImageFusion conda "
            "environment before using encoder_base_feature='random_mamba'."
        ) from exc
    return Mamba


def _build_mamba(dim, d_state, d_conv, expand):
    Mamba = _load_mamba_cls()
    kwargs = dict(d_state=d_state, d_conv=d_conv, expand=expand)
    try:
        return Mamba(d_model=dim, **kwargs)
    except TypeError:
        return Mamba(dim, **kwargs)


class RandomMamba(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        repeat=2,
    ):
        super(RandomMamba, self).__init__()
        self.dim = dim
        self.repeat = max(int(repeat), 1)
        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.mixer = _build_mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def _pos_tokens(self, x, h, w):
        b, n, c = x.shape
        if n != h * w:
            raise ValueError(f"Token count {n} does not match H*W={h*w}.")
        x_4d = x.transpose(1, 2).contiguous().view(b, c, h, w)
        pos = self.pos_embed(x_4d)
        return pos.flatten(2).transpose(1, 2).contiguous()

    def _run_mamba_once(self, x):
        b, n, c = x.shape
        del b, c
        shuffle = torch.randperm(n, device=x.device)
        inverse = torch.empty_like(shuffle)
        inverse[shuffle] = torch.arange(n, device=x.device)
        x = x[:, shuffle, :]
        x = self.mixer(x)
        if isinstance(x, tuple):
            x = x[0]
        return x[:, inverse, :].contiguous()

    def forward(self, x, h, w, residual=None):
        x = x + self._pos_tokens(x, h, w)
        residual = x if residual is None else residual + x
        x = self.norm(residual)

        if self.repeat == 1:
            x = self._run_mamba_once(x)
        else:
            mixed = None
            for _ in range(self.repeat):
                y = self._run_mamba_once(x)
                mixed = y if mixed is None else mixed + y
            x = mixed / float(self.repeat)
        return x, residual


class BaseRandomMambaExtraction(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=4,
        repeat=1,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super(BaseRandomMambaExtraction, self).__init__()
        self.dim = dim
        self.num_layers = int(num_layers)
        self.repeat = int(repeat)
        self.layers = nn.ModuleList([
            RandomMamba(
                dim=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                repeat=repeat,
            )
            for _ in range(self.num_layers)
        ])
        self.norm_f = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {c}.")

        hidden_states = x.flatten(2).transpose(1, 2).contiguous()
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, h, w, residual)

        residual = hidden_states if residual is None else residual + hidden_states
        hidden_states = self.norm_f(residual)
        return hidden_states.transpose(1, 2).contiguous().view(b, c, h, w)
