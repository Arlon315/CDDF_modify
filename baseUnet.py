import torch
import torch.nn as nn
import torch.nn.functional as F

from net import TransformerBlock


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False, use_relu=False):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        ]
        if use_bn:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_relu:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False, use_relu=False):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        ]
        if use_bn:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_relu:
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat((x, self.layer(x)), dim=1)


class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        dense_Layer_out=32,
        num_layers=3,
        out_channels=None,
        kernel_size=3,
        stride=1,
        use_relu=False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        current_channels = in_channels
        for index in range(self.num_layers):
            layer = DenseConvLayer(
                in_channels=current_channels,
                out_channels=dense_Layer_out,
                kernel_size=kernel_size,
                stride=stride,
                use_relu=use_relu,
            )
            self.add_module(f"dense_conv{index + 1}", layer)
            current_channels += dense_Layer_out
        self.adjust_conv = ConvLayer(current_channels, out_channels or current_channels, kernel_size=kernel_size)

    def forward(self, x):
        out = x
        for index in range(self.num_layers):
            out = getattr(self, f"dense_conv{index + 1}")(out)
        return self.adjust_conv(out)


class UpCatTransformer(nn.Module):
    def __init__(self, up_ch, skip_ch, out_ch, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(up_ch, out_ch, kernel_size=3, padding=1, padding_mode="reflect", bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.reduce = nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=1, bias=bias)
        self.trans = TransformerBlock(
            dim=out_ch,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.trans(x)
        return x


class BaseUNetEncoder(nn.Module):
    def __init__(
        self,
        dim=64,
        mid_dim=96,
        deep_dim=128,
        num_heads=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.shallow_dense = DenseBlock(
            in_channels=dim,
            dense_Layer_out=dim // 2,
            num_layers=3,
            out_channels=dim,
            use_relu=True,
        )
        self.shallow_trans = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.down1 = nn.AvgPool2d(2)

        self.mid_dense = DenseBlock(
            in_channels=dim,
            dense_Layer_out=dim // 2,
            num_layers=3,
            out_channels=mid_dim,
            use_relu=True,
        )
        self.mid_trans = TransformerBlock(
            dim=mid_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.down2 = nn.AvgPool2d(2)

        self.deep_dense = DenseBlock(
            in_channels=mid_dim,
            dense_Layer_out=dim // 2,
            num_layers=3,
            out_channels=deep_dim,
            use_relu=True,
        )
        self.deep_trans = TransformerBlock(
            dim=deep_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, x):
        shallow = self.shallow_trans(self.shallow_dense(x))

        mid = self.down1(shallow)
        mid = self.mid_trans(self.mid_dense(mid))

        deep = self.down2(mid)
        deep = self.deep_trans(self.deep_dense(deep))

        return shallow, mid, deep


class BaseUNetDecoder(nn.Module):
    def __init__(
        self,
        dim=64,
        mid_dim=96,
        deep_dim=128,
        num_heads=8,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super().__init__()

        self.deep_trans = TransformerBlock(
            dim=deep_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.up_mid = UpCatTransformer(
            up_ch=deep_dim,
            skip_ch=mid_dim,
            out_ch=mid_dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.up_shallow = UpCatTransformer(
            up_ch=mid_dim,
            skip_ch=dim,
            out_ch=dim,
            num_heads=num_heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, shallow, mid, deep, residual=None):
        deep = self.deep_trans(deep) 
        x = self.up_mid(deep, mid)
        x = self.up_shallow(x, shallow)
        x = self.out_proj(x)

        if residual is not None:
            x = residual + self.beta * x

        return x
