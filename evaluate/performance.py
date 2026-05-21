"""
Evaluator-ported image-fusion metrics used by batch evaluation.

The batch evaluator keeps all metric formulas in this file so projects can share
the same preprocessing and aggregation rules while matching utils/Evaluator.py
for metric behavior and metric coverage.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
from PIL import Image
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
try:
    import torch
except ModuleNotFoundError:
    torch = None
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as structural_similarity


METRIC_COLUMNS: Tuple[str, ...] = (
    "EN",
    "SD",
    "SF",
    "AG",
    "MI",
    "MSE",
    "CC",
    "PSNR",
    "SCD",
    "VIF",
    "Qabf",
    "SSIM",
)
METRIC_IMPLEMENTATION_ID = "evaluator_port_v2"


@dataclass(frozen=True)
class MetricProfile:
    name: str = "evaluator_port_v2"
    implementation_id: str = METRIC_IMPLEMENTATION_ID
    source_implementation: str = "utils/Evaluator.py"
    grayscale_conversion: str = "OpenCV BGR2GRAY (Evaluator-compatible, with PIL fallback)"
    value_range: str = "rounded grayscale float64 in [0,255]"
    alignment_mode: str = "batch evaluator normalizes shapes before Evaluator-aligned formulas"
    mi_log_base: str = "e"
    vif_scales: int = 4
    vif_noise_variance: float = 2.0
    vif_aggregation: str = "sum"
    ssim_backend: str = "skimage structural_similarity default on uint8"
    ssim_gaussian_weights: bool = False
    ssim_mode: str = "sum(SSIM(F,A), SSIM(F,B))"
    ssim_window_size: int = 7
    ssim_sigma: float = 0.0
    ssim_k1: float = 0.01
    ssim_k2: float = 0.03
    ssim_data_range: float = 255.0

    def metadata(self) -> Dict[str, object]:
        return asdict(self)


PAPER_PROFILE = MetricProfile()


def _to_255_range(array: np.ndarray) -> np.ndarray:
    gray = np.asarray(array).astype(np.float64, copy=False)
    if gray.size == 0:
        raise ValueError("Empty image is not supported")
    if gray.min() >= -1.0 and gray.max() <= 1.0:
        if gray.min() < 0.0:
            gray = (gray + 1.0) * 127.5
        else:
            gray = gray * 255.0
    elif gray.max() > 255.0 or gray.min() < 0.0:
        gray = np.clip(gray, 0.0, 255.0)
    return np.round(gray)


def _to_project_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] >= 3:
            color = np.clip(_to_255_range(array[:, :, :3]), 0.0, 255.0).astype(np.uint8)
            if cv2 is not None:
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            else:
                gray = np.round(
                    0.114 * color[:, :, 0].astype(np.float64)
                    + 0.587 * color[:, :, 1].astype(np.float64)
                    + 0.299 * color[:, :, 2].astype(np.float64)
                ).astype(np.uint8)
            return np.round(gray.astype(np.float64, copy=False))
        if array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.ndim != 2:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    return _to_255_range(array)


def _resize_to_shape(gray: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if gray.shape == shape:
        return gray
    if cv2 is not None:
        return cv2.resize(gray, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    gray_u8 = np.clip(np.round(gray), 0.0, 255.0).astype(np.uint8)
    resized = Image.fromarray(gray_u8).resize((int(shape[1]), int(shape[0])), Image.BILINEAR)
    return np.round(np.asarray(resized).astype(np.float64, copy=False))


def _prepare_triplet(
    src1: np.ndarray,
    src2: np.ndarray,
    fused: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fused_gray = _to_project_gray(fused)
    target_shape = fused_gray.shape
    src1_gray = _resize_to_shape(_to_project_gray(src1), target_shape)
    src2_gray = _resize_to_shape(_to_project_gray(src2), target_shape)
    return src1_gray, src2_gray, fused_gray


def _corrcoef2(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denominator = np.sqrt(np.sum((x - np.mean(x)) ** 2) * np.sum((y - np.mean(y)) ** 2))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def metric_en(fused: np.ndarray) -> float:
    gray = np.uint8(np.clip(_to_project_gray(fused), 0.0, 255.0))
    hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
    hist /= max(float(gray.size), 1.0)
    return float(-np.sum(hist * np.log2(hist + (hist == 0))))


def metric_sd(fused: np.ndarray) -> float:
    gray = _to_project_gray(fused)
    return float(np.std(gray))


def metric_sf(fused: np.ndarray) -> float:
    gray = _to_project_gray(fused)
    row_term = np.mean((gray[:, 1:] - gray[:, :-1]) ** 2) if gray.shape[1] > 1 else 0.0
    col_term = np.mean((gray[1:, :] - gray[:-1, :]) ** 2) if gray.shape[0] > 1 else 0.0
    return float(np.sqrt(row_term + col_term))


def metric_ag(fused: np.ndarray) -> float:
    gray = _to_project_gray(fused)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    if gray.shape[1] > 1:
        gx[:, 0] = gray[:, 1] - gray[:, 0]
        gx[:, -1] = gray[:, -1] - gray[:, -2]
    if gray.shape[1] > 2:
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0

    if gray.shape[0] > 1:
        gy[0, :] = gray[1, :] - gray[0, :]
        gy[-1, :] = gray[-1, :] - gray[-2, :]
    if gray.shape[0] > 2:
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0

    return float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0)))


def _mutual_information_discrete(image_x: np.ndarray, image_y: np.ndarray) -> float:
    x = np.uint8(np.clip(image_x, 0.0, 255.0)).ravel()
    y = np.uint8(np.clip(image_y, 0.0, 255.0)).ravel()
    if x.size != y.size:
        raise ValueError("Mutual information requires the same number of pixels.")

    joint = np.zeros((256, 256), dtype=np.float64)
    np.add.at(joint, (x, y), 1.0)
    joint_sum = joint.sum()
    if joint_sum <= 0.0:
        return 0.0

    joint /= joint_sum
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    px_py = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / px_py[mask])))


def metric_mi(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return float(_mutual_information_discrete(gf, g1) + _mutual_information_discrete(gf, g2))


def metric_mse(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return float((np.mean((g1 - gf) ** 2) + np.mean((g2 - gf) ** 2)) / 2.0)


def metric_cc(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return float((_corrcoef2(g1, gf) + _corrcoef2(g2, gf)) / 2.0)


def metric_psnr(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    gf = _to_project_gray(fused)
    mse = metric_mse(src1, src2, fused)
    return float(10.0 * np.log10(np.max(gf) ** 2 / mse))


def metric_scd(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    img_f_a = gf - g1
    img_f_b = gf - g2
    return float(_corrcoef2(g1, img_f_b) + _corrcoef2(g2, img_f_a))


def _compare_viff(reference: np.ndarray, distorted: np.ndarray, sigma_nsq: float, scales: int) -> float:
    ref = reference.astype(np.float64, copy=False)
    dist = distorted.astype(np.float64, copy=False)
    eps = 1e-10
    numerator = 0.0
    denominator = 0.0
    num_scales = max(int(scales), 1)

    for scale in range(1, num_scales + 1):
        size = 2 ** (num_scales - scale + 1) + 1
        sigma = size / 5.0

        m = (size - 1.0) / 2.0
        y, x = np.ogrid[-m : m + 1, -m : m + 1]
        kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
        kernel /= kernel.sum()

        if scale > 1:
            ref = convolve2d(ref, np.rot90(kernel, 2), mode="valid")[::2, ::2]
            dist = convolve2d(dist, np.rot90(kernel, 2), mode="valid")[::2, ::2]

        mu_ref = convolve2d(ref, np.rot90(kernel, 2), mode="valid")
        mu_dist = convolve2d(dist, np.rot90(kernel, 2), mode="valid")
        mu_ref_sq = mu_ref * mu_ref
        mu_dist_sq = mu_dist * mu_dist
        mu_ref_dist = mu_ref * mu_dist

        sigma_ref_sq = convolve2d(ref * ref, np.rot90(kernel, 2), mode="valid") - mu_ref_sq
        sigma_dist_sq = convolve2d(dist * dist, np.rot90(kernel, 2), mode="valid") - mu_dist_sq
        sigma_ref_dist = convolve2d(ref * dist, np.rot90(kernel, 2), mode="valid") - mu_ref_dist

        sigma_ref_sq[sigma_ref_sq < 0] = 0
        sigma_dist_sq[sigma_dist_sq < 0] = 0

        gain = sigma_ref_dist / (sigma_ref_sq + eps)
        sv_sq = sigma_dist_sq - gain * sigma_ref_dist

        gain[sigma_ref_sq < eps] = 0
        sv_sq[sigma_ref_sq < eps] = sigma_dist_sq[sigma_ref_sq < eps]
        sigma_ref_sq[sigma_ref_sq < eps] = 0

        gain[sigma_dist_sq < eps] = 0
        sv_sq[sigma_dist_sq < eps] = 0

        sv_sq[gain < 0] = sigma_dist_sq[gain < 0]
        gain[gain < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        numerator += np.sum(np.log10(1 + gain * gain * sigma_ref_sq / (sv_sq + sigma_nsq)))
        denominator += np.sum(np.log10(1 + sigma_ref_sq / sigma_nsq))

    vif = numerator / denominator if denominator > eps else 1.0
    return float(1.0 if np.isnan(vif) else vif)


def _metric_vif_prepared(
    g1: np.ndarray,
    g2: np.ndarray,
    gf: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
) -> float:
    vif_a = _compare_viff(g1, gf, sigma_nsq=profile.vif_noise_variance, scales=profile.vif_scales)
    vif_b = _compare_viff(g2, gf, sigma_nsq=profile.vif_noise_variance, scales=profile.vif_scales)
    if profile.vif_aggregation == "average":
        return float((vif_a + vif_b) / 2.0)
    if profile.vif_aggregation == "sum":
        return float(vif_a + vif_b)
    raise ValueError(f"Unsupported VIF aggregation mode: {profile.vif_aggregation}")


def _gaussian_kernel_torch(size: int, sigma: float, device) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch VIF path.")

    half = size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float64, device=device)
    y = coords.view(-1, 1)
    x = coords.view(1, -1)
    kernel = torch.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    kernel[kernel < torch.finfo(kernel.dtype).eps * kernel.max()] = 0
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def _conv2d_valid_torch(image: "torch.Tensor", kernel: "torch.Tensor") -> "torch.Tensor":
    return torch.nn.functional.conv2d(image.unsqueeze(0).unsqueeze(0), kernel).squeeze(0).squeeze(0)


def _compare_viff_torch(
    reference: np.ndarray,
    distorted: np.ndarray,
    sigma_nsq: float,
    scales: int,
    device=None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch VIF path.")
    if device is None:
        device = torch.device("cuda")

    ref = torch.as_tensor(reference, dtype=torch.float64, device=device)
    dist = torch.as_tensor(distorted, dtype=torch.float64, device=device)
    eps = 1e-10
    numerator = torch.tensor(0.0, dtype=torch.float64, device=device)
    denominator = torch.tensor(0.0, dtype=torch.float64, device=device)
    num_scales = max(int(scales), 1)

    for scale in range(1, num_scales + 1):
        size = 2 ** (num_scales - scale + 1) + 1
        sigma = size / 5.0
        kernel = _gaussian_kernel_torch(size, sigma, device=device)

        if scale > 1:
            ref = _conv2d_valid_torch(ref, kernel)[::2, ::2]
            dist = _conv2d_valid_torch(dist, kernel)[::2, ::2]

        mu_ref = _conv2d_valid_torch(ref, kernel)
        mu_dist = _conv2d_valid_torch(dist, kernel)
        mu_ref_sq = mu_ref * mu_ref
        mu_dist_sq = mu_dist * mu_dist
        mu_ref_dist = mu_ref * mu_dist

        sigma_ref_sq = _conv2d_valid_torch(ref * ref, kernel) - mu_ref_sq
        sigma_dist_sq = _conv2d_valid_torch(dist * dist, kernel) - mu_dist_sq
        sigma_ref_dist = _conv2d_valid_torch(ref * dist, kernel) - mu_ref_dist

        sigma_ref_sq = sigma_ref_sq.clone()
        sigma_dist_sq = sigma_dist_sq.clone()
        sigma_ref_sq[sigma_ref_sq < 0] = 0
        sigma_dist_sq[sigma_dist_sq < 0] = 0

        gain = sigma_ref_dist / (sigma_ref_sq + eps)
        sv_sq = sigma_dist_sq - gain * sigma_ref_dist

        mask = sigma_ref_sq < eps
        gain[mask] = 0
        sv_sq[mask] = sigma_dist_sq[mask]
        sigma_ref_sq[mask] = 0

        mask = sigma_dist_sq < eps
        gain[mask] = 0
        sv_sq[mask] = 0

        mask = gain < 0
        sv_sq[mask] = sigma_dist_sq[mask]
        gain[mask] = 0
        sv_sq[sv_sq <= eps] = eps

        numerator = numerator + torch.sum(torch.log10(1 + gain * gain * sigma_ref_sq / (sv_sq + sigma_nsq)))
        denominator = denominator + torch.sum(torch.log10(1 + sigma_ref_sq / sigma_nsq))

    denominator_value = float(denominator.item())
    vif = float((numerator / denominator).item()) if denominator_value > eps else 1.0
    return float(1.0 if math.isnan(vif) else vif)


def _metric_vif_prepared_torch(
    g1: np.ndarray,
    g2: np.ndarray,
    gf: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
    device=None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch VIF path.")

    with torch.no_grad():
        vif_a = _compare_viff_torch(
            g1,
            gf,
            sigma_nsq=profile.vif_noise_variance,
            scales=profile.vif_scales,
            device=device,
        )
        vif_b = _compare_viff_torch(
            g2,
            gf,
            sigma_nsq=profile.vif_noise_variance,
            scales=profile.vif_scales,
            device=device,
        )

    if profile.vif_aggregation == "average":
        return float((vif_a + vif_b) / 2.0)
    if profile.vif_aggregation == "sum":
        return float(vif_a + vif_b)
    raise ValueError(f"Unsupported VIF aggregation mode: {profile.vif_aggregation}")


def metric_vif(
    src1: np.ndarray,
    src2: np.ndarray,
    fused: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return _metric_vif_prepared(g1, g2, gf, profile=profile)


def _qabf_get_array(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sobel_x = convolve2d(image, h3, mode="same")
    sobel_y = convolve2d(image, h1, mode="same")
    grad = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    angle = np.zeros_like(image)
    angle[sobel_x == 0] = math.pi / 2
    angle[sobel_x != 0] = np.arctan(sobel_y[sobel_x != 0] / sobel_x[sobel_x != 0])
    return grad, angle


def _qabf_get_quality(a_src: np.ndarray, g_src: np.ndarray, a_fused: np.ndarray, g_fused: np.ndarray) -> np.ndarray:
    tg = 0.9994
    kg = -15
    dg = 0.5
    ta = 0.9879
    ka = -22
    da = 0.8

    relative_grad = np.zeros_like(a_src)
    relative_grad[g_src > g_fused] = g_fused[g_src > g_fused] / g_src[g_src > g_fused]
    relative_grad[g_src == g_fused] = g_fused[g_src == g_fused]
    relative_grad[g_src < g_fused] = g_src[g_src < g_fused] / g_fused[g_src < g_fused]

    relative_angle = 1 - np.abs(a_src - a_fused) / (math.pi / 2)
    q_grad = tg / (1 + np.exp(kg * (relative_grad - dg)))
    q_angle = ta / (1 + np.exp(ka * (relative_angle - da)))
    return q_grad * q_angle


def _metric_qabf_prepared(g1: np.ndarray, g2: np.ndarray, gf: np.ndarray) -> float:
    grad1, ang1 = _qabf_get_array(g1)
    grad2, ang2 = _qabf_get_array(g2)
    gradf, angf = _qabf_get_array(gf)
    q1 = _qabf_get_quality(ang1, grad1, angf, gradf)
    q2 = _qabf_get_quality(ang2, grad2, angf, gradf)
    denominator = np.sum(grad1 + grad2)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(q1 * grad1 + q2 * grad2)
    return float(numerator / denominator)


def _qabf_get_array_torch(image: np.ndarray, device) -> tuple["torch.Tensor", "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch Qabf path.")

    # Match scipy.signal.convolve2d(mode="same") exactly: float64, zero padding,
    # and flipped kernels because torch.conv2d performs correlation instead of convolution.
    kernel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=torch.float64,
        device=device,
    ).flip((0, 1)).view(1, 1, 3, 3)
    kernel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float64,
        device=device,
    ).flip((0, 1)).view(1, 1, 3, 3)

    image_t = torch.as_tensor(np.asarray(image), dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0)
    sobel_x = torch.nn.functional.conv2d(image_t, kernel_x, padding=1).squeeze(0).squeeze(0)
    sobel_y = torch.nn.functional.conv2d(image_t, kernel_y, padding=1).squeeze(0).squeeze(0)
    grad = torch.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    angle = torch.empty_like(sobel_x)
    zero_mask = sobel_x == 0
    angle[zero_mask] = math.pi / 2
    nonzero_mask = ~zero_mask
    angle[nonzero_mask] = torch.atan(sobel_y[nonzero_mask] / sobel_x[nonzero_mask])
    return grad, angle


def _qabf_get_quality_torch(
    a_src: "torch.Tensor",
    g_src: "torch.Tensor",
    a_fused: "torch.Tensor",
    g_fused: "torch.Tensor",
) -> "torch.Tensor":
    tg = 0.9994
    kg = -15.0
    dg = 0.5
    ta = 0.9879
    ka = -22.0
    da = 0.8

    relative_grad = torch.zeros_like(a_src)
    mask_gt = g_src > g_fused
    mask_eq = g_src == g_fused
    mask_lt = g_src < g_fused
    relative_grad[mask_gt] = g_fused[mask_gt] / g_src[mask_gt]
    relative_grad[mask_eq] = g_fused[mask_eq]
    relative_grad[mask_lt] = g_src[mask_lt] / g_fused[mask_lt]

    relative_angle = 1 - torch.abs(a_src - a_fused) / (math.pi / 2)
    q_grad = tg / (1 + torch.exp(kg * (relative_grad - dg)))
    q_angle = ta / (1 + torch.exp(ka * (relative_angle - da)))
    return q_grad * q_angle


def _metric_qabf_prepared_torch(g1: np.ndarray, g2: np.ndarray, gf: np.ndarray, device=None) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch Qabf path.")
    if device is None:
        device = torch.device("cuda")

    with torch.no_grad():
        grad1, ang1 = _qabf_get_array_torch(g1, device=device)
        grad2, ang2 = _qabf_get_array_torch(g2, device=device)
        gradf, angf = _qabf_get_array_torch(gf, device=device)
        q1 = _qabf_get_quality_torch(ang1, grad1, angf, gradf)
        q2 = _qabf_get_quality_torch(ang2, grad2, angf, gradf)
        denominator = torch.sum(grad1 + grad2)
        if denominator.item() <= 0:
            return 0.0
        numerator = torch.sum(q1 * grad1 + q2 * grad2)
        return float((numerator / denominator).item())


def metric_qabf(src1: np.ndarray, src2: np.ndarray, fused: np.ndarray) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return _metric_qabf_prepared(g1, g2, gf)


def _ssim_single(reference: np.ndarray, distorted: np.ndarray, profile: MetricProfile) -> float:
    del profile
    ref_u8 = np.uint8(np.clip(reference, 0.0, 255.0))
    dist_u8 = np.uint8(np.clip(distorted, 0.0, 255.0))
    return float(structural_similarity(ref_u8, dist_u8))


def _uniform_filter_torch(image: "torch.Tensor", size: int) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch SSIM path.")
    if size <= 0 or size % 2 != 1:
        raise ValueError("Window size must be a positive odd integer.")

    pad = size // 2
    kernel = torch.full((1, 1, size, size), 1.0 / (size * size), dtype=torch.float64, device=image.device)
    padded = torch.nn.functional.pad(image.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return torch.nn.functional.conv2d(padded, kernel).squeeze(0).squeeze(0)


def _ssim_single_torch(
    reference: np.ndarray,
    distorted: np.ndarray,
    profile: MetricProfile,
    device=None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch SSIM path.")
    if device is None:
        device = torch.device("cuda")

    ref_u8 = np.uint8(np.clip(reference, 0.0, 255.0))
    dist_u8 = np.uint8(np.clip(distorted, 0.0, 255.0))
    if ref_u8.shape != dist_u8.shape:
        raise ValueError("Input images must have the same dimensions.")

    win_size = int(profile.ssim_window_size)
    if min(ref_u8.shape) < win_size:
        raise ValueError(
            "win_size exceeds image extent. Either ensure that your images are at least 7x7; "
            "or pass win_size explicitly in the function call, with an odd value less than or "
            "equal to the smaller side of your images."
        )
    if win_size % 2 != 1:
        raise ValueError("Window size must be odd.")

    ref = torch.as_tensor(ref_u8, dtype=torch.float64, device=device)
    dist = torch.as_tensor(dist_u8, dtype=torch.float64, device=device)

    npix = win_size ** 2
    cov_norm = npix / (npix - 1)
    ux = _uniform_filter_torch(ref, win_size)
    uy = _uniform_filter_torch(dist, win_size)
    uxx = _uniform_filter_torch(ref * ref, win_size)
    uyy = _uniform_filter_torch(dist * dist, win_size)
    uxy = _uniform_filter_torch(ref * dist, win_size)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    data_range = float(profile.ssim_data_range)
    c1 = (float(profile.ssim_k1) * data_range) ** 2
    c2 = (float(profile.ssim_k2) * data_range) ** 2

    a1 = 2 * ux * uy + c1
    a2 = 2 * vxy + c2
    b1 = ux * ux + uy * uy + c1
    b2 = vx + vy + c2
    s = (a1 * a2) / (b1 * b2)

    pad = (win_size - 1) // 2
    if pad > 0:
        s = s[pad:-pad, pad:-pad]
    return float(s.mean(dtype=torch.float64).item())


def _metric_ssim_prepared(
    g1: np.ndarray,
    g2: np.ndarray,
    gf: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
) -> float:
    ssim_a = _ssim_single(gf, g1, profile=profile)
    ssim_b = _ssim_single(gf, g2, profile=profile)
    if profile.ssim_mode.startswith("average"):
        return float((ssim_a + ssim_b) / 2.0)
    if profile.ssim_mode.startswith("sum"):
        return float(ssim_a + ssim_b)
    raise ValueError(f"Unsupported SSIM aggregation mode: {profile.ssim_mode}")


def _metric_ssim_prepared_torch(
    g1: np.ndarray,
    g2: np.ndarray,
    gf: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
    device=None,
) -> float:
    if torch is None:
        raise RuntimeError("PyTorch is required for the torch SSIM path.")

    with torch.no_grad():
        ssim_a = _ssim_single_torch(gf, g1, profile=profile, device=device)
        ssim_b = _ssim_single_torch(gf, g2, profile=profile, device=device)

    if profile.ssim_mode.startswith("average"):
        return float((ssim_a + ssim_b) / 2.0)
    if profile.ssim_mode.startswith("sum"):
        return float(ssim_a + ssim_b)
    raise ValueError(f"Unsupported SSIM aggregation mode: {profile.ssim_mode}")


def metric_ssim(
    src1: np.ndarray,
    src2: np.ndarray,
    fused: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
) -> float:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)
    return _metric_ssim_prepared(g1, g2, gf, profile=profile)


def compute_all_metrics(
    src1: np.ndarray,
    src2: np.ndarray,
    fused: np.ndarray,
    profile: MetricProfile = PAPER_PROFILE,
) -> Dict[str, float]:
    g1, g2, gf = _prepare_triplet(src1, src2, fused)

    mse = float((np.mean((g1 - gf) ** 2) + np.mean((g2 - gf) ** 2)) / 2.0)
    cc = float((_corrcoef2(g1, gf) + _corrcoef2(g2, gf)) / 2.0)
    scd = float(_corrcoef2(g1, gf - g2) + _corrcoef2(g2, gf - g1))

    if torch is not None and torch.cuda.is_available():
        try:
            vif = _metric_vif_prepared_torch(g1, g2, gf, profile=profile, device=torch.device("cuda"))
        except RuntimeError:
            vif = _metric_vif_prepared(g1, g2, gf, profile=profile)
    else:
        vif = _metric_vif_prepared(g1, g2, gf, profile=profile)

    if torch is not None and torch.cuda.is_available():
        try:
            qabf = _metric_qabf_prepared_torch(g1, g2, gf, device=torch.device("cuda"))
        except RuntimeError:
            qabf = _metric_qabf_prepared(g1, g2, gf)
    else:
        qabf = _metric_qabf_prepared(g1, g2, gf)

    if torch is not None and torch.cuda.is_available():
        try:
            ssim = _metric_ssim_prepared_torch(g1, g2, gf, profile=profile, device=torch.device("cuda"))
        except RuntimeError:
            ssim = _metric_ssim_prepared(g1, g2, gf, profile=profile)
    else:
        ssim = _metric_ssim_prepared(g1, g2, gf, profile=profile)

    gray_u8 = np.uint8(np.clip(gf, 0.0, 255.0))
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    hist /= max(float(gray_u8.size), 1.0)

    gx = np.zeros_like(gf)
    gy = np.zeros_like(gf)
    if gf.shape[1] > 1:
        gx[:, 0] = gf[:, 1] - gf[:, 0]
        gx[:, -1] = gf[:, -1] - gf[:, -2]
    if gf.shape[1] > 2:
        gx[:, 1:-1] = (gf[:, 2:] - gf[:, :-2]) / 2.0
    if gf.shape[0] > 1:
        gy[0, :] = gf[1, :] - gf[0, :]
        gy[-1, :] = gf[-1, :] - gf[-2, :]
    if gf.shape[0] > 2:
        gy[1:-1, :] = (gf[2:, :] - gf[:-2, :]) / 2.0

    sf_row = np.mean((gf[:, 1:] - gf[:, :-1]) ** 2) if gf.shape[1] > 1 else 0.0
    sf_col = np.mean((gf[1:, :] - gf[:-1, :]) ** 2) if gf.shape[0] > 1 else 0.0

    psnr = float("inf") if mse <= 0.0 else float(10.0 * np.log10(np.max(gf) ** 2 / mse))

    return {
        "EN": float(-np.sum(hist * np.log2(hist + (hist == 0)))),
        "SD": float(np.std(gf)),
        "SF": float(np.sqrt(sf_row + sf_col)),
        "AG": float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0))),
        "MI": float(_mutual_information_discrete(gf, g1) + _mutual_information_discrete(gf, g2)),
        "MSE": mse,
        "CC": cc,
        "PSNR": psnr,
        "SCD": scd,
        "VIF": vif,
        "Qabf": qabf,
        "SSIM": ssim,
    }

def get_metric_metadata(profile: MetricProfile = PAPER_PROFILE) -> Mapping[str, object]:
    return profile.metadata()


__all__ = [
    "METRIC_COLUMNS",
    "METRIC_IMPLEMENTATION_ID",
    "MetricProfile",
    "PAPER_PROFILE",
    "compute_all_metrics",
    "get_metric_metadata",
    "metric_en",
    "metric_ag",
    "metric_cc",
    "metric_mi",
    "metric_mse",
    "metric_psnr",
    "metric_qabf",
    "metric_scd",
    "metric_sd",
    "metric_sf",
    "metric_ssim",
    "metric_vif",
]
