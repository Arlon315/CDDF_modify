#!/usr/bin/env python3
"""
Utilities for evaluating CDDFuse models in this repository.

The main adapter is `run_fusion_prediction`, which matches the runner contract
expected by `evaluate/batch_evaluate.py` while using the current project's
network definition, checkpoints, and test-data conventions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from net import (  # noqa: E402
    build_cddfuse_modules,
    fuse_base_features,
    fuse_detail_features,
    infer_cddfuse_base_fusion,
    infer_cddfuse_backbone,
    infer_cddfuse_detail_fusion,
    infer_cddfuse_detail_num_layers,
    infer_cddfuse_encoder_detail_enhance_layers,
)

try:
    from evaluate.performance import METRIC_COLUMNS, PAPER_PROFILE, compute_all_metrics  # type: ignore
except ImportError:
    from performance import METRIC_COLUMNS, PAPER_PROFILE, compute_all_metrics  # type: ignore


DEFAULT_FILE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_MODEL_CACHE: Dict[Tuple[str, str], Dict[str, torch.nn.Module]] = {}


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize_device(device: str) -> str:
    requested = str(device or "cuda").strip().lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    return cleaned


def _load_model_bundle(model_path: str, device: str) -> Dict[str, torch.nn.Module]:
    model_file = _resolve_project_path(model_path).resolve()
    cache_key = (str(model_file), device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    checkpoint = torch.load(str(model_file), map_location=device)

    encoder, decoder, base_fuse, detail_fuse = build_cddfuse_modules(
        infer_cddfuse_backbone(checkpoint),
        detail_fusion=infer_cddfuse_detail_fusion(checkpoint),
        detail_fusion_num_layers=infer_cddfuse_detail_num_layers(checkpoint),
        encoder_detail_enhance_layers=infer_cddfuse_encoder_detail_enhance_layers(checkpoint),
        base_fusion=infer_cddfuse_base_fusion(checkpoint),
    )
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    base_fuse = base_fuse.to(device)
    detail_fuse = detail_fuse.to(device)

    encoder.load_state_dict(_strip_module_prefix(checkpoint["DIDF_Encoder"]))
    decoder.load_state_dict(_strip_module_prefix(checkpoint["DIDF_Decoder"]))
    base_fuse.load_state_dict(_strip_module_prefix(checkpoint["BaseFuseLayer"]))
    detail_fuse.load_state_dict(_strip_module_prefix(checkpoint["DetailFuseLayer"]))

    encoder.eval()
    decoder.eval()
    base_fuse.eval()
    detail_fuse.eval()

    bundle = {
        "encoder": encoder,
        "decoder": decoder,
        "base_fuse": base_fuse,
        "detail_fuse": detail_fuse,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _parse_hw(size_value: Any) -> Optional[Tuple[int, int]]:
    if size_value is None:
        return None
    if isinstance(size_value, (tuple, list)) and len(size_value) == 2:
        return int(size_value[1]), int(size_value[0])
    if isinstance(size_value, int):
        return (size_value, size_value) if size_value > 0 else None

    text = str(size_value).strip().lower()
    if not text or text in {"0", "keep", "original", "none"}:
        return None
    for sep in ("x", ","):
        if sep in text:
            width_text, height_text = text.split(sep, 1)
            return int(height_text), int(width_text)
    side = int(text)
    return (side, side) if side > 0 else None


def _resize_image(image: np.ndarray, hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if hw is None or image.shape[:2] == hw:
        return image
    return cv2.resize(image, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)


def _read_gray_image(path: str) -> np.ndarray:
    image_path = _resolve_project_path(path)
    if not image_path.exists():
        raise FileNotFoundError(path)
    image_data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def _to_tensor(image_u8: np.ndarray, device: str) -> torch.Tensor:
    array = image_u8.astype(np.float32, copy=False) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)


def _normalize_fused_tensor(tensor: torch.Tensor) -> torch.Tensor:
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    denom = torch.clamp(max_value - min_value, min=1e-8)
    return (tensor - min_value) / denom


def _is_medical_task(ir_path: str, vis_path: str, fusion_params: Mapping[str, Any]) -> bool:
    task = str(fusion_params.get("task") or fusion_params.get("task_type") or "").strip().lower()
    if task:
        return task == "mif"

    parts = {part.upper() for part in Path(ir_path).parts + Path(vis_path).parts}
    medical_tokens = {"MRI_CT", "MRI_PET", "MRI_SPECT", "MRI", "CT", "PET", "SPECT"}
    return bool(parts.intersection(medical_tokens))


def _infer_decoder_input_mode(
    model_path: str,
    ir_path: str,
    vis_path: str,
    fusion_params: Mapping[str, Any],
) -> str:
    override = fusion_params.get("decoder_input_mode", fusion_params.get("decoder_mode", "auto"))
    mode = str(override).strip().lower()
    if mode and mode != "auto":
        if mode not in {"visible", "sum", "none"}:
            raise ValueError(f"Unsupported decoder_input_mode: {override}")
        return mode

    model_name = Path(model_path).stem.upper()
    if "MIF" in model_name:
        return "none"
    if _is_medical_task(ir_path, vis_path, fusion_params):
        return "sum"
    return "visible"


def _build_decoder_input(mode: str, vis_tensor: torch.Tensor, ir_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    if mode == "visible":
        return vis_tensor
    if mode == "sum":
        return vis_tensor + ir_tensor
    if mode == "none":
        return None
    raise ValueError(f"Unsupported decoder input mode: {mode}")


def _save_gray_image(path: str, image_float01: np.ndarray) -> None:
    output_path = _resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image_float01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    extension = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image_u8)
    if not ok:
        raise IOError(f"Failed to encode fused image: {output_path}")
    try:
        with output_path.open("wb") as handle:
            handle.write(encoded.tobytes())
    except OSError as exc:
        raise IOError(f"Failed to save fused image: {output_path}") from exc


def get_random_image_pairs_from_folder(
    ir_dir: str,
    vis_dir: str,
    num_pairs: int = 0,
    file_extensions: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    extensions = tuple(file_extensions or DEFAULT_FILE_EXTENSIONS)
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}

    def scan(folder: str) -> Dict[str, str]:
        folder_path = Path(folder)
        mapping: Dict[str, str] = {}
        if not folder_path.exists():
            return mapping
        for path in sorted(folder_path.iterdir()):
            if path.is_file() and path.suffix.lower() in allowed:
                mapping.setdefault(path.stem, str(path))
        return mapping

    ir_files = scan(ir_dir)
    vis_files = scan(vis_dir)
    names = sorted(set(ir_files).intersection(vis_files))
    if num_pairs > 0 and num_pairs < len(names):
        rng = np.random.default_rng(seed)
        names = list(rng.choice(names, size=num_pairs, replace=False))
        names.sort()
    return [(ir_files[name], vis_files[name]) for name in names]


def run_fusion_prediction(
    ir_path: str,
    vis_path: str,
    model_path: str = "models/CDDFuse_IVF.pth",
    output_path: Optional[str] = None,
    device: str = "cuda",
    debug: bool = False,
    input_size: Any = 0,
    use_test: bool = True,
    return_for_eval: bool = False,
    eval_size: Any = 0,
    save_output: bool = True,
    fusion_params: Optional[Mapping[str, Any]] = None,
) -> Any:
    del use_test, eval_size

    fusion_params = dict(fusion_params or {})
    resolved_device = _normalize_device(device)
    if debug and resolved_device != device:
        print(f"Requested device '{device}' is unavailable; falling back to '{resolved_device}'.")

    input_hw = _parse_hw(input_size)
    ir_image = _resize_image(_read_gray_image(ir_path), input_hw)
    vis_image = _resize_image(_read_gray_image(vis_path), input_hw)
    if ir_image.shape != vis_image.shape:
        raise ValueError(
            f"Input pair shape mismatch: {Path(ir_path).name}={ir_image.shape}, "
            f"{Path(vis_path).name}={vis_image.shape}"
        )

    bundle = _load_model_bundle(model_path=model_path, device=resolved_device)
    ir_tensor = _to_tensor(ir_image, resolved_device)
    vis_tensor = _to_tensor(vis_image, resolved_device)
    decoder_input_mode = _infer_decoder_input_mode(model_path, ir_path, vis_path, fusion_params)
    decoder_input = _build_decoder_input(decoder_input_mode, vis_tensor, ir_tensor)

    with torch.no_grad():
        feature_v_b, feature_v_d, _ = bundle["encoder"](vis_tensor)
        feature_i_b, feature_i_d, _ = bundle["encoder"](ir_tensor)
        feature_f_b = fuse_base_features(bundle["base_fuse"], feature_i_b, feature_v_b)
        feature_f_d = fuse_detail_features(bundle["detail_fuse"], feature_i_d, feature_v_d)
        fused_tensor, _ = bundle["decoder"](decoder_input, feature_f_b, feature_f_d)
        fused_tensor = _normalize_fused_tensor(fused_tensor)

    fused_float01 = np.squeeze(fused_tensor.detach().cpu().numpy()).astype(np.float32, copy=False)

    if save_output and output_path:
        _save_gray_image(output_path, fused_float01)

    if return_for_eval:
        return {
            "success": True,
            "fused_float01": fused_float01,
            "ir_eval_u8": ir_image,
            "vis_eval_u8": vis_image,
            "decoder_input_mode": decoder_input_mode,
        }
    return True


def evaluate_images(
    ir_paths: Sequence[str],
    vis_paths: Sequence[str],
    fused_paths: Sequence[str],
    eval_size: Any = 0,
) -> pd.DataFrame:
    eval_hw = _parse_hw(eval_size)
    rows: List[Dict[str, Any]] = []

    for ir_path, vis_path, fused_path in zip(ir_paths, vis_paths, fused_paths):
        ir_image = _resize_image(_read_gray_image(ir_path), eval_hw)
        vis_image = _resize_image(_read_gray_image(vis_path), eval_hw)
        fused_image = _resize_image(_read_gray_image(fused_path), eval_hw)

        metrics = compute_all_metrics(ir_image, vis_image, fused_image, profile=PAPER_PROFILE)
        row = {"filename": Path(fused_path).stem}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows, columns=("filename",) + METRIC_COLUMNS)


def save_results_to_csv(results_df: pd.DataFrame, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False, encoding="utf-8")


def save_results_to_excel(results_df: pd.DataFrame, output_path: str, sheet_name: str = "evaluation") -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        results_df.to_excel(writer, index=False, sheet_name=sheet_name)


def print_statistics(results_df: pd.DataFrame) -> None:
    if results_df.empty:
        print("No valid evaluation results.")
        return

    numeric = results_df[list(METRIC_COLUMNS)].astype(float)
    stats = pd.DataFrame(
        [
            {"stat": "mean", **numeric.mean().to_dict()},
            {"stat": "std", **numeric.std(ddof=1).fillna(0.0).to_dict()},
        ]
    )
    print(stats.to_string(index=False))


__all__ = [
    "DEFAULT_FILE_EXTENSIONS",
    "evaluate_images",
    "get_random_image_pairs_from_folder",
    "print_statistics",
    "run_fusion_prediction",
    "save_results_to_csv",
    "save_results_to_excel",
]
