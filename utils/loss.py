import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        loss_in = F.l1_loss(torch.max(image_y, image_ir), generate_img)
        source_gradient = torch.max(
            self.sobelconv(image_y),
            self.sobelconv(image_ir),
        )
        loss_grad = F.l1_loss(source_gradient, self.sobelconv(generate_img))
        return loss_in + 10 * loss_grad, loss_in, loss_grad, None


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        kernely = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kernelx)
        self.register_buffer('weighty', kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)
