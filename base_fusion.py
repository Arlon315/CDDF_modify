import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.conv(torch.cat([avg_out, max_out], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        return self.shared_mlp(self.avg_pool(x)) + self.shared_mlp(self.max_pool(x))


class PixelAttention(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2 * channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="reflect",
            groups=channels,
            bias=True,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attention):
        b, c, h, w = x.shape
        x = torch.cat([x.unsqueeze(2), attention.unsqueeze(2)], dim=2)
        x = x.reshape(b, 2 * c, h, w)
        return self.sigmoid(self.conv(x))


class MidCGAFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(channels)
        self.pixel_attention = PixelAttention(channels)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, infrared, visible):
        initial = infrared + visible
        attention = self.channel_attention(initial) + self.spatial_attention(initial)
        pixel_attention = self.pixel_attention(initial, attention)
        fused = initial + pixel_attention * infrared + (1.0 - pixel_attention) * visible
        return self.out_conv(fused)


class MCAM(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or max(in_channels // 2, 1)

        self.g_infrared = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g_visible = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta_infrared = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta_visible = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi_infrared = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi_visible = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        if bn_layer:
            self.out = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
            )
            nn.init.constant_(self.out[1].weight, 0)
            nn.init.constant_(self.out[1].bias, 0)
        else:
            self.out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.out.weight, 0)
            nn.init.constant_(self.out.bias, 0)

        if sub_sample:
            pool = nn.MaxPool2d(kernel_size=2)
            self.g_infrared = nn.Sequential(self.g_infrared, pool)
            self.g_visible = nn.Sequential(self.g_visible, pool)
            self.phi_infrared = nn.Sequential(self.phi_infrared, pool)
            self.phi_visible = nn.Sequential(self.phi_visible, pool)

    def forward(self, infrared, visible):
        batch_size = infrared.size(0)

        g_infrared = self.g_infrared(infrared).view(batch_size, self.inter_channels, -1)
        g_infrared = g_infrared.permute(0, 2, 1)

        g_visible = self.g_visible(visible).view(batch_size, self.inter_channels, -1)
        g_visible = g_visible.permute(0, 2, 1)

        theta_infrared = self.theta_infrared(infrared).view(batch_size, self.inter_channels, -1)
        theta_infrared = theta_infrared.permute(0, 2, 1)

        theta_visible = self.theta_visible(visible).view(batch_size, self.inter_channels, -1)
        theta_visible = theta_visible.permute(0, 2, 1)

        phi_infrared = self.phi_infrared(infrared).view(batch_size, self.inter_channels, -1)
        phi_visible = self.phi_visible(visible).view(batch_size, self.inter_channels, -1)

        attention_infrared = F.softmax(torch.matmul(theta_infrared, phi_infrared), dim=-1)
        attention_visible = F.softmax(torch.matmul(theta_visible, phi_visible), dim=-1)
        mutual_attention = attention_infrared * attention_visible

        response_infrared = torch.matmul(mutual_attention, g_infrared)
        response_visible = torch.matmul(mutual_attention, g_visible)
        response = response_infrared * response_visible
        response = response.permute(0, 2, 1).contiguous()
        response = response.view(batch_size, self.inter_channels, *infrared.size()[2:])
        return self.out(response)


class DeepMCAMFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mcam = MCAM(channels)

    def forward(self, infrared, visible):
        return infrared + visible + self.mcam(infrared, visible)


class MultiScaleBaseFusion(nn.Module):
    def __init__(self, dim=64, mid_dim=96, deep_dim=128):
        super().__init__()
        self.mid_fuse = MidCGAFusion(channels=mid_dim)
        self.deep_fuse = DeepMCAMFusion(channels=deep_dim)

    def forward(
        self,
        I_B_s,
        I_B_m,
        I_B_d,
        V_B_s,
        V_B_m,
        V_B_d,
    ):
        F_B_s = I_B_s + V_B_s
        F_B_m = self.mid_fuse(I_B_m, V_B_m)
        F_B_d = self.deep_fuse(I_B_d, V_B_d)
        return F_B_s, F_B_m, F_B_d
