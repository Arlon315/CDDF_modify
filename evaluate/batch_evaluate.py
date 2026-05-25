#!/usr/bin/env python3
"""
Portable batch evaluation for image-fusion models.

Recommended copy strategy across projects:
1. Copy this file together with `performance.py`.
2. Keep metric formulas only in `performance.py`.
3. Adapt cross-project behavior in just these hooks when needed:
   - `get_project_defaults()`
   - `build_runner_call_kwargs()`
   - `normalize_runner_output()`

If the target project's runner already supports a common callable shape, no code
changes are required. Prefer passing `--runner module:function` or
`--runner path\\to\\file.py:function` before editing this file.
项目默认值在 batch_evaluate.py (line 95)，
runner 参数适配在 batch_evaluate.py (line 314)，
runner 返回值适配在 batch_evaluate.py (line 346)，
批量评估主流程在 batch_evaluate.py (line 615)。
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_RUNNER_SPEC = "evaluate.evaluate_utils:run_fusion_prediction"
RUNNER_PARAM_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "ir_path": ("ir_image_path", "ir_path", "infrared_image_path", "infrared_path", "ir"),
    "vis_path": (
        "vi_image_path",
        "vis_image_path",
        "vis_path",
        "visible_image_path",
        "visible_path",
        "vi_path",
        "vi",
        "vis",
    ),
    "model_path": ("model_path", "checkpoint_path", "weights_path", "ckpt_path"),
    "output_path": ("output_path", "save_path", "result_path", "fused_path"),
    "device": ("device",),
    "debug": ("debug", "verbose"),
    "input_size": ("input_size", "resize_to", "img_size"),
    "use_test": ("use_test", "test_mode"),
    "return_for_eval": ("return_for_eval",),
    "eval_size": ("eval_size",),
    "save_output": ("save_output", "save_result", "save_image", "save_img"),
    "fusion_params": ("fusion_params",),
}


try:
    from evaluate.performance import (  # type: ignore
        METRIC_COLUMNS,
        METRIC_IMPLEMENTATION_ID,
        PAPER_PROFILE,
        MetricProfile,
        compute_all_metrics,
        get_metric_metadata,
    )
except ImportError:
    try:
        from performance import (  # type: ignore
            METRIC_COLUMNS,
            METRIC_IMPLEMENTATION_ID,
            PAPER_PROFILE,
            MetricProfile,
            compute_all_metrics,
            get_metric_metadata,
        )
    except ImportError as exc:
        raise ImportError(
            "batch_evaluate.py requires performance.py. Copy both files together "
            "or ensure either 'evaluate.performance' or 'performance' is importable."
        ) from exc


def get_project_defaults() -> Mapping[str, object]:
    return {
        "datasets": {
            "TNO": {
                "ir_dir": "test_img/TNO/ir",
                "vis_dir": "test_img/TNO/vi",
                "num_pairs": 25,
                "task": "ivf",
                "default_model_path": "models/CDDFuse_IVF.pth",
            },
            "RoadScene": {
                "ir_dir": "test_img/RoadScene/ir",
                "vis_dir": "test_img/RoadScene/vi",
                "num_pairs": 50,
                "task": "ivf",
                "default_model_path": "models/CDDFuse_IVF.pth",
            },
            "MRI_CT": {
                "ir_dir": "test_img/MRI_CT/CT",
                "vis_dir": "test_img/MRI_CT/MRI",
                "num_pairs": 21,
                "task": "mif",
                "default_model_path": "models/CDDFuse_MIF.pth",
            },
            "MRI_PET": {
                "ir_dir": "test_img/MRI_PET/PET",
                "vis_dir": "test_img/MRI_PET/MRI",
                "num_pairs": 42,
                "task": "mif",
                "default_model_path": "models/CDDFuse_MIF.pth",
            },
            "MRI_SPECT": {
                "ir_dir": "test_img/MRI_SPECT/SPECT",
                "vis_dir": "test_img/MRI_SPECT/MRI",
                "num_pairs": 73,
                "task": "mif",
                "default_model_path": "models/CDDFuse_MIF.pth",
            },
            # "M3FD_Fusion": {
            #     "ir_dir": "image/M3FD_Fusion/ir",
            #     "vis_dir": "image/M3FD_Fusion/vis",
            #     "num_pairs": 53,
            # },
            "21_pairs_tno": {
                "ir_dir": "image/21_pairs_tno/ir",
                "vis_dir": "image/21_pairs_tno/vis",
                "num_pairs": 21,
            },
            "40_vot_tno": {
                "ir_dir": "image/40_vot_tno/ir",
                "vis_dir": "image/40_vot_tno/vis",
                "num_pairs": 40,
            },
            "MSRS": {
                "ir_dir": "image/MSRS/ir",
                "vis_dir": "image/MSRS/vi",
                "num_pairs": 24,
            },
            "RoadScence": {
                "ir_dir": "E:/workspace/python_work/dataSet/MixDataSet/split_8_2/test/RoadScene/ir",
                "vis_dir": "E:/workspace/python_work/dataSet/MixDataSet/split_8_2/test/RoadScene/vi",
                "num_pairs": 44,
            },
            "M3FD_Fusion": {
                "ir_dir": "E:/workspace/python_work/dataSet/MixDataSet/split_8_2/test/M3FD/ir",
                "vis_dir": "E:/workspace/python_work/dataSet/MixDataSet/split_8_2/test/M3FD/vi",
                "num_pairs": 60,
            },
            "MSRS-main": {
                "ir_dir": "E:/workspace/python_work/dataSet/MSRS-main/test/ir",
                "vis_dir": "E:/workspace/python_work/dataSet/MSRS-main/test/vi",
                "num_pairs": 361,
            },
        },
        "default_dataset": "RoadScence",
        "default_model_path": "models/CDDFuse_05-13-16-32_epoch_120.pth",
        "default_runner": DEFAULT_RUNNER_SPEC,
        "default_device": "cuda",
        "default_debug": False,
        "default_use_test": True,
        "default_save_fused": True,
        "default_eval_size": "0",
        "default_input_size": "0",
    }


def resolve_project_path(path_value: str) -> str:
    if not str(path_value).strip():
        return ""
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def parse_size_arg(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if not text or text in {"0", "keep", "original", "none"}:
        return 0
    for sep in ("x", ","):
        if sep in text:
            width_text, height_text = text.split(sep, 1)
            return (int(width_text), int(height_text))
    return int(text)


def format_size_arg(value: Any) -> str:
    parsed = parse_size_arg(value)
    if isinstance(parsed, tuple):
        return f"{parsed[0]}x{parsed[1]}"
    if parsed <= 0:
        return "original"
    return str(parsed)


def resolve_hw(size_value: Any, fallback_shape: Tuple[int, int]) -> Tuple[int, int]:
    parsed = parse_size_arg(size_value)
    if isinstance(parsed, tuple):
        return int(parsed[1]), int(parsed[0])
    if isinstance(parsed, int) and parsed > 0:
        return int(parsed), int(parsed)
    return int(fallback_shape[0]), int(fallback_shape[1])


def parse_kv_overrides(items: Optional[Sequence[str]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE override, got: {item}")
        key, raw_value = item.split("=", 1)
        try:
            value = ast.literal_eval(raw_value.strip())
        except Exception:
            value = raw_value.strip()
        result[key.strip()] = value
    return result


def normalize_extensions(file_extensions: Optional[Sequence[str]]) -> Tuple[str, ...]:
    values = file_extensions or DEFAULT_FILE_EXTENSIONS
    return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in values)


def scan_image_dir(folder: str, file_extensions: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    folder_path = Path(folder)
    if not folder_path.exists():
        return mapping
    allowed = set(normalize_extensions(file_extensions))
    for path in sorted(folder_path.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            mapping.setdefault(path.stem, str(path))
    return mapping


def discover_image_pairs(
    ir_dir: str,
    vis_dir: str,
    num_pairs: int,
    file_extensions: Sequence[str],
    seed: int,
) -> List[Tuple[str, str]]:
    ir_images = scan_image_dir(ir_dir, file_extensions)
    vis_images = scan_image_dir(vis_dir, file_extensions)
    matching_names = sorted(set(ir_images).intersection(vis_images))
    if not matching_names:
        return []
    if num_pairs > 0 and num_pairs < len(matching_names):
        rng = random.Random(seed)
        rng.shuffle(matching_names)
        matching_names = matching_names[:num_pairs]
    return [(ir_images[name], vis_images[name]) for name in matching_names]


def to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def to_u8_image(image: Any) -> np.ndarray:
    array = to_numpy_array(image)
    if array.dtype == np.uint8:
        return array
    array = array.astype(np.float32, copy=False)
    if array.min() >= -1.0 and array.max() <= 1.0 and array.min() < 0.0:
        array = (array + 1.0) * 127.5
    elif array.max() <= 1.0:
        array = array * 255.0
    return (np.clip(array, 0.0, 255.0) + 0.5).astype(np.uint8)


def to_float01_image(image: Any) -> np.ndarray:
    array = np.squeeze(to_numpy_array(image)).astype(np.float32, copy=False)
    if array.min() >= -1.0 and array.max() <= 1.0 and array.min() < 0.0:
        array = (array + 1.0) / 2.0
    elif array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def read_image_for_eval(path: str) -> np.ndarray:
    image_path = Path(resolve_project_path(path))
    if not image_path.exists():
        raise FileNotFoundError(path)
    image_data = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def resize_image_like_eval(image: np.ndarray, eval_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if eval_hw is None or image.shape[:2] == eval_hw:
        return image
    height, width = eval_hw
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def build_eval_bundle_from_saved_output(
    ir_path: str,
    vis_path: str,
    fused_path: str,
    eval_size: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ir_img = read_image_for_eval(ir_path)
    vis_img = read_image_for_eval(vis_path)
    fused_img = read_image_for_eval(fused_path)
    eval_hw = resolve_hw(eval_size, fused_img.shape[:2]) if parse_size_arg(eval_size) else None
    ir_eval = resize_image_like_eval(to_u8_image(ir_img), eval_hw)
    vis_eval = resize_image_like_eval(to_u8_image(vis_img), eval_hw)
    fused_eval = resize_image_like_eval(to_u8_image(fused_img), eval_hw)
    return to_float01_image(fused_eval), ir_eval, vis_eval


def coerce_eval_bundle(bundle: Tuple[Any, Any, Any], eval_size: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fused_float01, ir_img, vis_img = bundle
    fused_float01 = to_float01_image(fused_float01)
    ir_u8 = to_u8_image(ir_img)
    vis_u8 = to_u8_image(vis_img)
    eval_hw = resolve_hw(eval_size, fused_float01.shape[:2]) if parse_size_arg(eval_size) else None
    if eval_hw is not None:
        fused_float01 = resize_image_like_eval(fused_float01, eval_hw)
        ir_u8 = resize_image_like_eval(ir_u8, eval_hw)
        vis_u8 = resize_image_like_eval(vis_u8, eval_hw)
    return fused_float01.astype(np.float32, copy=False), ir_u8, vis_u8


def load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("Runner spec must use 'module:function' or '/path/to/file.py:function'")
    module_ref, func_name = spec.rsplit(":", 1)
    module_ref = module_ref.strip()
    func_name = func_name.strip()
    if module_ref.endswith(".py") or os.path.sep in module_ref or (os.path.altsep and os.path.altsep in module_ref):
        module_path = Path(module_ref)
        if not module_path.is_absolute():
            module_path = (PROJECT_ROOT / module_path).resolve()
        module_name = f"_portable_batch_eval_{module_path.stem}"
        spec_obj = importlib.util.spec_from_file_location(module_name, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Unable to load module from file: {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        sys.modules[module_name] = module
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    runner = getattr(module, func_name)
    if not callable(runner):
        raise TypeError(f"Runner is not callable: {spec}")
    return runner


def build_runner_call_kwargs(
    runner: Callable[..., Any],
    request: Mapping[str, Any],
    extra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    signature = inspect.signature(runner)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: Dict[str, Any] = {}
    if accepts_kwargs:
        for semantic_name, aliases in RUNNER_PARAM_ALIASES.items():
            if semantic_name in request:
                kwargs[aliases[0]] = request[semantic_name]
        kwargs.update(extra_kwargs or {})
        return kwargs

    for semantic_name, aliases in RUNNER_PARAM_ALIASES.items():
        if semantic_name not in request:
            continue
        for alias in aliases:
            if alias in signature.parameters:
                kwargs[alias] = request[semantic_name]
                break

    for key, value in (extra_kwargs or {}).items():
        if key in signature.parameters:
            kwargs[key] = value
    return kwargs


def normalize_runner_output(
    result: Any,
    ir_path: str,
    vis_path: str,
    output_path: str,
    return_for_eval: bool,
    eval_size: Any,
) -> Tuple[bool, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    success = False
    bundle: Optional[Tuple[Any, Any, Any]] = None

    if isinstance(result, dict):
        success = bool(result.get("success", True))
        if {"fused_float01", "ir_eval_u8", "vis_eval_u8"} <= set(result):
            bundle = (
                result["fused_float01"],
                result["ir_eval_u8"],
                result["vis_eval_u8"],
            )
    elif isinstance(result, tuple):
        if len(result) == 4 and isinstance(result[0], bool):
            success = result[0]
            bundle = (result[1], result[2], result[3])
        elif len(result) == 3:
            success = True
            bundle = result
        elif len(result) == 2 and isinstance(result[0], bool):
            success = result[0]
    elif isinstance(result, bool):
        success = result
    else:
        success = result is not None

    if return_for_eval and bundle is not None:
        return True, coerce_eval_bundle(bundle, eval_size)
    if return_for_eval and os.path.exists(output_path):
        return True, build_eval_bundle_from_saved_output(ir_path, vis_path, output_path, eval_size)
    if not return_for_eval and os.path.exists(output_path):
        return True, None
    return success, None


def run_fusion_prediction(
    ir_path: str,
    vis_path: str,
    output_path: str,
    model_path: str,
    runner: Callable[..., Any],
    device: str,
    debug: bool,
    return_for_eval: bool,
    eval_size: Any,
    use_test: bool,
    input_size: Any,
    save_output: bool,
    fusion_params: Mapping[str, Any],
    runner_extra_kwargs: Mapping[str, Any],
) -> Tuple[bool, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    request = {
        "ir_path": ir_path,
        "vis_path": vis_path,
        "model_path": model_path,
        "output_path": output_path,
        "device": device,
        "debug": debug,
        "input_size": input_size,
        "use_test": use_test,
        "return_for_eval": return_for_eval,
        "eval_size": eval_size,
        "save_output": save_output,
        "fusion_params": dict(fusion_params),
    }
    kwargs = build_runner_call_kwargs(runner, request, extra_kwargs=runner_extra_kwargs)
    try:
        result = runner(**kwargs)
        return normalize_runner_output(
            result,
            ir_path=ir_path,
            vis_path=vis_path,
            output_path=output_path,
            return_for_eval=return_for_eval,
            eval_size=eval_size,
        )
    except Exception as exc:
        print(f"Fusion error for {os.path.basename(output_path)}: {exc}")
        return False, None


def evaluate_images_in_memory(
    eval_items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    eval_size: Any,
    profile: MetricProfile = PAPER_PROFILE,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for index, (filename, ir_img, vis_img, fused_float01) in enumerate(eval_items, start=1):
        try:
            eval_hw = resolve_hw(eval_size, fused_float01.shape[:2]) if parse_size_arg(eval_size) else None
            if eval_hw is not None:
                fused_float01 = resize_image_like_eval(fused_float01, eval_hw)
                ir_img = resize_image_like_eval(ir_img, eval_hw)
                vis_img = resize_image_like_eval(vis_img, eval_hw)
            fused_u8 = to_u8_image(fused_float01)
            metrics = compute_all_metrics(ir_img, vis_img, fused_u8, profile=profile)
            row = {"filename": filename}
            row.update({metric: metrics[metric] for metric in METRIC_COLUMNS})
            rows.append(row)
            print(f"Evaluated [{index}/{len(eval_items)}]: {filename}")
        except Exception as exc:
            print(f"Evaluation failed [{index}/{len(eval_items)}]: {filename}: {exc}")
    return pd.DataFrame(rows, columns=("filename",) + METRIC_COLUMNS)


def derive_model_tag(model_path: str) -> str:
    path = Path(model_path)
    stem = path.stem
    if "_epoch_" in stem:
        return stem.split("_epoch_", 1)[0]
    if path.parent.name and path.parent.name.lower() != "models":
        return path.parent.name
    return stem


def derive_model_epoch(model_path: str) -> Optional[str]:
    stem = Path(model_path).stem
    if "_epoch_" not in stem:
        return None
    return stem.split("_epoch_", 1)[1]


def derive_dataset_tag(dataset_key: Optional[str], ir_dir: str, vis_dir: str) -> str:
    if dataset_key:
        return dataset_key
    ir_parent = Path(ir_dir).parent.name
    vis_parent = Path(vis_dir).parent.name
    return ir_parent or vis_parent or Path(ir_dir).name


def build_default_output_csv(model_path: str, dataset_tag: str) -> str:
    base_dir = Path(build_default_fused_dir(model_path, dataset_tag))
    return str(base_dir / f"batch_evaluation_results_{dataset_tag}.csv")


def build_default_fused_dir(model_path: str, dataset_tag: str) -> str:
    base_dir = PROJECT_ROOT / "test_result" / dataset_tag / derive_model_tag(model_path)
    model_epoch = derive_model_epoch(model_path)
    if model_epoch:
        base_dir = base_dir / f"epoch{model_epoch}"
    return str(base_dir)


def build_metadata(
    args: argparse.Namespace,
    dataset_tag: str,
    num_pairs: int,
    matched_pairs: int,
    successful_pairs: int,
    failed_pairs: int,
) -> Dict[str, Any]:
    metadata = {
        "metric_profile": PAPER_PROFILE.name,
        "metric_implementation_id": METRIC_IMPLEMENTATION_ID,
        "metric_columns": list(METRIC_COLUMNS),
        "dataset_tag": dataset_tag,
        "ir_dir": str(args.ir_dir),
        "vis_dir": str(args.vis_dir),
        "model_path": str(args.model_path),
        "runner": args.runner,
        "device": args.device,
        "eval_size": format_size_arg(args.eval_size),
        "input_size": format_size_arg(args.input_size),
        "num_pairs_requested": int(num_pairs),
        "num_pairs_matched": int(matched_pairs),
        "num_pairs_successful": int(successful_pairs),
        "num_pairs_failed": int(failed_pairs),
        "save_fused": bool(args.save_fused),
        "seed": int(args.seed),
        "file_extensions": list(args.file_extensions),
        "fusion_params": dict(args.fusion_params),
        "runner_kwargs": dict(args.runner_kwargs),
    }
    metadata.update(dict(get_metric_metadata(PAPER_PROFILE)))
    return metadata


def save_results_table(results_df: pd.DataFrame, output_path: str, metadata: Mapping[str, Any]) -> None:
    if results_df.empty:
        print("No valid evaluation results to save.")
        return
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False, encoding="utf-8")

    numeric = results_df[list(METRIC_COLUMNS)].astype(float)
    stats_df = pd.DataFrame(
        [
            {"stat": "mean", **numeric.mean().to_dict()},
            {"stat": "std", **numeric.std(ddof=1).fillna(0.0).to_dict()},
        ]
    )
    stats_path = output_path.replace(".csv", "_stats.csv")
    stats_df.to_csv(stats_path, index=False, encoding="utf-8")

    meta_path = output_path.replace(".csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("Batch Evaluation Summary")
    print("=" * 72)
    mean_row = stats_df.loc[stats_df["stat"] == "mean"].iloc[0]
    std_row = stats_df.loc[stats_df["stat"] == "std"].iloc[0]
    for metric in METRIC_COLUMNS:
        print(f"{metric:<6} {mean_row[metric]:.4f} +- {std_row[metric]:.4f}")
    print("=" * 72)
    print(f"Saved detailed results: {output_path}")
    print(f"Saved statistics:      {stats_path}")
    print(f"Saved metadata:        {meta_path}")


def build_arg_parser(project_defaults: Mapping[str, object]) -> argparse.ArgumentParser:
    datasets = project_defaults["datasets"]
    parser = argparse.ArgumentParser(description="Batch evaluation for CDDFuse image-fusion models.")
    parser.add_argument("--dataset", type=str, default=project_defaults["default_dataset"], choices=sorted(datasets))
    parser.add_argument("--ir_dir", type=str, default=None, help="Infrared directory; overrides --dataset.")
    parser.add_argument("--vis_dir", type=str, default=None, help="Visible directory; overrides --dataset.")
    parser.add_argument("--model_path", type=str, default=None, help="Checkpoint path. Defaults to the dataset model.")
    parser.add_argument("--runner", type=str, default=project_defaults["default_runner"])
    parser.add_argument("--fused_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--num_pairs", type=int, default=None, help="0 or negative means all matched pairs.")
    parser.add_argument("--file_extensions", nargs="+", default=list(DEFAULT_FILE_EXTENSIONS))
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for deterministic pair selection.")
    parser.add_argument("--eval_size", type=str, default=project_defaults["default_eval_size"], help="0 | 256 | 640x480")
    parser.add_argument("--input_size", type=str, default=project_defaults["default_input_size"], help="0 | 256 | 640x480")
    parser.add_argument("--device", type=str, default=project_defaults["default_device"])
    parser.add_argument(
        "--fusion_param",
        action="append",
        default=[],
        help="Extra fusion override: KEY=VALUE, e.g. decoder_input_mode='sum' or task='mif'",
    )
    parser.add_argument("--runner_kwarg", action="append", default=[], help="Extra runner kwarg: KEY=VALUE")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=project_defaults["default_debug"])
    parser.add_argument("--use_test", action=argparse.BooleanOptionalAction, default=project_defaults["default_use_test"])
    parser.add_argument("--save_fused", action=argparse.BooleanOptionalAction, default=project_defaults["default_save_fused"])
    return parser


def resolve_dataset_args(args: argparse.Namespace, project_defaults: Mapping[str, object]) -> argparse.Namespace:
    datasets = project_defaults["datasets"]
    custom_dirs = args.ir_dir is not None or args.vis_dir is not None
    dataset_cfg = datasets.get(args.dataset, {})
    args.ir_dir = resolve_project_path(str(args.ir_dir or dataset_cfg.get("ir_dir", "")))
    args.vis_dir = resolve_project_path(str(args.vis_dir or dataset_cfg.get("vis_dir", "")))
    if not args.ir_dir or not args.vis_dir:
        raise ValueError("Both --ir_dir and --vis_dir must be provided.")
    default_pairs = int(dataset_cfg.get("num_pairs", 0))
    args.num_pairs = int(default_pairs if args.num_pairs is None else args.num_pairs)
    args.eval_size = parse_size_arg(args.eval_size)
    args.input_size = parse_size_arg(args.input_size)
    args.file_extensions = normalize_extensions(args.file_extensions)
    args.fusion_params = parse_kv_overrides(args.fusion_param)
    args.runner_kwargs = parse_kv_overrides(args.runner_kwarg)
    if not args.model_path:
        args.model_path = str(dataset_cfg.get("default_model_path", project_defaults["default_model_path"]))
    args.model_path = resolve_project_path(str(args.model_path))
    dataset_task = dataset_cfg.get("task")
    if dataset_task and "task" not in args.fusion_params and "task_type" not in args.fusion_params:
        args.fusion_params["task"] = str(dataset_task)
    args.dataset_tag = derive_dataset_tag(None if custom_dirs else args.dataset, args.ir_dir, args.vis_dir)
    if not args.fused_dir:
        args.fused_dir = build_default_fused_dir(args.model_path, args.dataset_tag)
    else:
        args.fused_dir = resolve_project_path(str(args.fused_dir))
    if not args.output_csv:
        args.output_csv = build_default_output_csv(args.model_path, args.dataset_tag)
    else:
        args.output_csv = resolve_project_path(str(args.output_csv))
    return args


def main() -> None:
    project_defaults = get_project_defaults()
    parser = build_arg_parser(project_defaults)
    args = resolve_dataset_args(parser.parse_args(), project_defaults)

    dataset_tag = args.dataset_tag
    runner = load_callable(args.runner)
    image_pairs = discover_image_pairs(
        ir_dir=args.ir_dir,
        vis_dir=args.vis_dir,
        num_pairs=args.num_pairs,
        file_extensions=args.file_extensions,
        seed=args.seed,
    )

    print(f"Metric profile: {PAPER_PROFILE.name} ({METRIC_IMPLEMENTATION_ID})")
    print(f"Dataset tag:    {dataset_tag}")
    print(f"IR dir:         {args.ir_dir}")
    print(f"VIS dir:        {args.vis_dir}")
    print(f"Model path:     {args.model_path}")
    print(f"Runner:         {args.runner}")
    print(f"Eval size:      {format_size_arg(args.eval_size)}")
    print(f"Input size:     {format_size_arg(args.input_size)}")
    print(f"Save fused:     {args.save_fused}")
    print(f"Matched pairs:  {len(image_pairs)}")
    print("-" * 72)

    if not image_pairs:
        print("No matched infrared-visible pairs were found.")
        return

    if args.save_fused:
        Path(args.fused_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    eval_items: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    failed_items: List[str] = []

    for index, (ir_path, vis_path) in enumerate(image_pairs, start=1):
        filename = Path(ir_path).stem
        output_path = str(Path(args.fused_dir) / f"{filename}.png")
        print(f"[{index}/{len(image_pairs)}] Running fusion: {filename}")

        ok, bundle = run_fusion_prediction(
            ir_path=ir_path,
            vis_path=vis_path,
            output_path=output_path,
            model_path=args.model_path,
            runner=runner,
            device=args.device,
            debug=args.debug,
            return_for_eval=True,
            eval_size=args.eval_size,
            use_test=args.use_test,
            input_size=args.input_size,
            save_output=args.save_fused,
            fusion_params=args.fusion_params,
            runner_extra_kwargs=args.runner_kwargs,
        )

        if ok and bundle is not None:
            fused_float01, ir_eval_u8, vis_eval_u8 = bundle
            eval_items.append((filename, ir_eval_u8, vis_eval_u8, fused_float01))
        else:
            failed_items.append(filename)

    print("-" * 72)
    print(f"Fusion success: {len(eval_items)}")
    print(f"Fusion failed:  {len(failed_items)}")

    if not eval_items:
        print("No fused outputs were available for evaluation.")
        print(
            "If you moved these files to another project, keep performance.py next "
            "to batch_evaluate.py and either make the runner return "
            "(fused_float01, ir_eval_u8, vis_eval_u8) or keep --save_fused "
            "enabled so fused images can be read back from disk."
        )
        return

    results_df = evaluate_images_in_memory(eval_items, eval_size=args.eval_size, profile=PAPER_PROFILE)
    metadata = build_metadata(
        args=args,
        dataset_tag=dataset_tag,
        num_pairs=args.num_pairs,
        matched_pairs=len(image_pairs),
        successful_pairs=len(eval_items),
        failed_pairs=len(failed_items),
    )
    save_results_table(results_df, args.output_csv, metadata)


if __name__ == "__main__":
    main()
