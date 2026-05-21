import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxAmpFocalFrequencyLoss(nn.Module):
    """
    Fusion version of Focal Frequency Loss.

    Target:
        target_amp = max(abs(FFT(IR)), abs(FFT(VIS_Y)))

    It keeps the focal idea from FFL:
        larger frequency error -> larger weight
        smaller frequency error -> smaller weight
    """

    def __init__(
        self,
        loss_weight=0.03,
        alpha=1.0,
        patch_factor=1,
        high_freq_only=True,
        high_start=0.08,
        log_matrix=False,
        batch_matrix=False,
        eps=1e-8,
    ):
        super(MaxAmpFocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.high_freq_only = high_freq_only
        self.high_start = high_start
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix
        self.eps = eps

    def _crop_patches(self, x):
        """
        x: [B, C, H, W]
        return: [B, P, C, patch_h, patch_w]
        """
        patch_factor = self.patch_factor
        b, c, h, w = x.shape

        assert h % patch_factor == 0 and w % patch_factor == 0, \
            "H and W must be divisible by patch_factor."

        patch_h = h // patch_factor
        patch_w = w // patch_factor

        patches = []
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch = x[
                    :,
                    :,
                    i * patch_h:(i + 1) * patch_h,
                    j * patch_w:(j + 1) * patch_w,
                ]
                patches.append(patch)

        return torch.stack(patches, dim=1)

    def _high_freq_mask(self, h, w, device, dtype):
        """
        Full fft2 mask.
        shape: [1, 1, 1, H, W]
        """
        fy = torch.fft.fftfreq(h, device=device).view(h, 1)
        fx = torch.fft.fftfreq(w, device=device).view(1, w)
        radius = torch.sqrt(fy ** 2 + fx ** 2)

        mask = (radius >= self.high_start).to(dtype)
        return mask.view(1, 1, 1, h, w)

    def forward(self, fused, vis_y, ir):
        fused = fused.float()
        vis_y = vis_y.float()
        ir = ir.float()

        fused_p = self._crop_patches(fused)
        vis_p = self._crop_patches(vis_y)
        ir_p = self._crop_patches(ir)

        # [B, P, C, H, W], complex
        fft_f = torch.fft.fft2(fused_p, norm='ortho')
        fft_v = torch.fft.fft2(vis_p, norm='ortho')
        fft_i = torch.fft.fft2(ir_p, norm='ortho')

        amp_f = torch.abs(fft_f)
        amp_v = torch.abs(fft_v)
        amp_i = torch.abs(fft_i)

        target_amp = torch.max(amp_v, amp_i).detach()

        diff = torch.abs(amp_f - target_amp)

        if self.high_freq_only:
            h, w = diff.shape[-2:]
            mask = self._high_freq_mask(h, w, diff.device, diff.dtype)
            diff = diff * mask

        # focal weight matrix
        with torch.no_grad():
            weight_matrix = diff.pow(self.alpha)

            if self.log_matrix:
                weight_matrix = torch.log(weight_matrix + 1.0)

            if self.batch_matrix:
                weight_matrix = weight_matrix / (weight_matrix.max() + self.eps)
            else:
                weight_matrix = weight_matrix / (
                    weight_matrix.amax(dim=(-2, -1), keepdim=True) + self.eps
                )

            weight_matrix = torch.nan_to_num(weight_matrix, nan=0.0, posinf=1.0, neginf=0.0)
            weight_matrix = torch.clamp(weight_matrix, 0.0, 1.0)

        loss = weight_matrix * diff.pow(2)

        if self.high_freq_only:
            denom = mask.sum() * diff.shape[0] * diff.shape[1] * diff.shape[2] + self.eps
            loss = loss.sum() / denom
        else:
            loss = loss.mean()

        return self.loss_weight * loss




class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

        # 频率损失
        self.freq_loss = MaxAmpFocalFrequencyLoss(
            loss_weight=0.03,
            alpha=1.0,
            patch_factor=1,
            high_freq_only=True,
            high_start=0.08,
            log_matrix=False,
            batch_matrix=False,
        )


    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis[:,:1,:,:]
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        
        # loss_freq = self.freq_loss(generate_img, image_y, image_ir)
        
        loss_total=loss_in+10*loss_grad
        return loss_total,loss_in,loss_grad,None

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()