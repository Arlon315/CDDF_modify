#!/usr/bin/env python3
"""
闂傚倷绀佺紞濠傤焽瑜忕槐鐐寸節閸パ囨７濠电偛妯婃禍婵嬪磻閵娾晜鐓熼柡鍐ㄥ亞閻掗箖鏌ｈ箛鎾村窛缂佽鲸甯楃粚閬嶅础閻愬吀鎮ｆ俊?闂傚倷绀侀幉锟犳偡椤栫偛鍨傞柣銏ゆ交缂嶆牠鏌涢埄鍐槈缂佲偓閸屾凹鐔嗛悹杞拌閸庢劖绻涢崼婵堝煟闁哄瞼鍠撻幏鐘诲灳閾忣偆褰庢繝鐢靛仜椤р偓缂佺姵鎹囬獮鍡涘礃椤旇偐顦板銈嗘閺侇噣宕戦幘瓒佹椽顢旈崟顔ф洟姊洪崜鎻掍簽闁哥姵鎹囧畷褰掓嚃閳哄啰锛滃┑鐐村灦椤ㄥ懐鈧艾閰ｉ弻?

濠电姵顔栭崰妤冪紦閸ф纾归柡宥庡幖閸戠娀鏌涢幇銊︽珖妞も晝鍏橀弻鐔碱敍閻愯弓鍠婇梺鍝ュ枎闁帮綁骞冪憴鍕婵炴潙顑呮禍楣冩煠閻撳骸顣虫い锔垮嵆閺岋綁鎮╅崗鍛板焻闂佸憡姊归崹鎸庝繆閺夋埈鍚嬪璺侯儏閸擃喖顪冮妶鍡樷拻闁告鍛闁哄被鍎查悡娆撴煟閹惧啿顒㈤柟顖氱墦閺岋綁濮€閳轰椒鍠婇梺?
1. 婵犵數鍎戠徊钘壝洪敂鐐床闁告劦浜栭崑鎾诲垂椤愶綆妫冮悗娈垮櫘閸嬪﹪骞冭瀹曠厧鈹戦崼娑樹喊闂傚倸鍊搁崐绋课涘Δ鈧灋婵犲﹤鐗婇崕妤€螖閿濆懎鏆為柛瀣ㄥ姂閺屾稑螖閸愩劋娌柣搴㈢閻擄繝寮婚悢鍏煎仺闁割煈鍋勫▓灞筋渻閵堝繗鍚傞柡鍛█楠炲啴宕奸弴鐐茶€垮┑掳鍊撶粈渚€宕滈悜鑺モ拺缂備焦顭囬惌銈夋煕椤垵鐏ｇ紒顕呭幗瀵板嫰骞囬鍌滃幀婵犵妲呴崹宕囨兜閸洖纾?
2. 婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο璇茬窞闁归偊鍓涢ˇ褔姊洪崫鍕ⅱ闁轰焦鎮傞弻褔宕掑鍏煎瘜闂侀潧鐗嗛幊蹇曟嫻閳╁啰绠鹃柤纰卞墮椤ｅジ鏌熷畡鐗堝櫧闁圭懓瀚版俊鎼佹晜閼恒儺鍚傛繝鐢靛仦閸ㄥ爼骞愰崫銉х煋闁圭虎鍠栫粻鏍煙閹规劦鍤欓柛銊ュ€块幃妤呮晲鎼存繄鐩庢繝鐢靛仜濞差參寮婚妸銉㈡婵☆垰鐏濋顓㈡偡濠婂嫬惟闁搞儺鐓堝鍧楁⒑闁偛鑻晶瀵糕偓娈垮枛椤曨參鍩€椤掍胶鈯曢柨姘跺箹閺夋埊韬慨濠冩そ椤㈡洟濮€閻橆偄浜鹃柛褎顨呴崒銊╂煏閸繍妲搁梻鍌ゅ灦閺屾洘绻涜濡厼顭囬崼鏇熲拺?
3. 闂傚倸鍊烽悞锕併亹閸愵亞鐭撻柛顐ｆ礀閺嬩焦銇勯妷锝呰緟I闂傚倷绀侀幉锛勫垝瀹€鍕剹濞达絿鍎ら鑺ャ亜閹惧崬鐏╃紒鈧崟顖涚厱婵犻潧妫楅顐ｃ亜韫囨梹灏﹂柟顔斤耿閹瑩鍩℃担宄邦棜闂傚倸鍊风欢锟犲磻閸涱厙锝夊箳閺冣偓椤愯姤銇勯幇鍫曟闁哄拋鍓熼幃姗€鎮欑捄杞版睏闂佽崵鍠愮换鍫ュ蓟濞戙垹妫橀悹鍥ㄥ絻椤牓姊洪崨濠勬噮闁稿鎸剧划瀣箳濡も偓閸愨偓濡炪倖鎸鹃崑娑欐叏閵堝鈷戦柛娑橈工缁楁帗淇婇銏ゅ弰闁糕斁鍋?
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATA_ROOT = Path(r"E:\workspace\python_work\dataSet")
DEFAULT_CHECKPOINT = "models/GLoC-Mamba/GLoC-Mamba_latest.pth"
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    root_name: str
    ir_template: Tuple[str, ...]
    vis_template: Tuple[str, ...]
    output_template: Tuple[str, ...]
    default_checkpoint: str
    default_split: Optional[str] = None
    valid_splits: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedDataset:
    key: str
    display_name: str
    root_dir: Path
    split: Optional[str]
    ir_dir: Path
    vis_dir: Path
    output_dir: Path
    checkpoint: Path


@dataclass(frozen=True)
class PairItem:
    name: str
    ir_path: Path
    vis_path: Path
    output_path: Path


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "llvip": DatasetSpec(
        key="llvip",
        root_name="LLVIP",
        ir_template=("infrared",),
        vis_template=("visible",),
        output_template=("labels",),
        default_checkpoint=DEFAULT_CHECKPOINT,
    ),
    "m3fd": DatasetSpec(
        key="m3fd",
        root_name="M3FD_fusion",
        ir_template=("Ir",),
        vis_template=("Vis",),
        output_template=("labels",),
        default_checkpoint=DEFAULT_CHECKPOINT,
    ),
    "msrs": DatasetSpec(
        key="msrs",
        root_name="MSRS-main",
        ir_template=("{split}", "ir"),
        vis_template=("{split}", "vi"),
        output_template=("{split}", "labels"),
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_split="train",
        valid_splits=("train", "test"),
    ),
    "roadscene": DatasetSpec(
        key="roadscene",
        root_name="RoadScene",
        ir_template=("cropinfrared",),
        vis_template=("crop_LR_visible",),
        output_template=("{model_name}_labels",),
        default_checkpoint=DEFAULT_CHECKPOINT,
    ),
}


def normalize_extensions(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    entries = values or DEFAULT_EXTENSIONS
    return tuple(value.lower() if value.startswith(".") else f".{value.lower()}" for value in entries)


def parse_hw(value: object) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[1]), int(value[0])
    if isinstance(value, int):
        return (value, value) if value > 0 else None

    text = str(value).strip().lower()
    if not text or text in {"0", "keep", "original", "none"}:
        return None
    for separator in ("x", ","):
        if separator in text:
            width_text, height_text = text.split(separator, 1)
            return int(height_text), int(width_text)
    side = int(text)
    return (side, side) if side > 0 else None


def format_duration(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_path_parts(parts: Sequence[str], split: Optional[str], model_name: Optional[str] = None) -> Tuple[str, ...]:
    rendered: List[str] = []
    for part in parts:
        current = part
        if "{split}" in current:
            if not split:
                raise ValueError("This dataset layout requires --split.")
            current = current.replace("{split}", split)
        if "{model_name}" in current:
            if not model_name:
                raise ValueError("This dataset layout requires a resolved checkpoint name.")
            current = current.replace("{model_name}", model_name)
        rendered.append(current)
    return tuple(rendered)


def resolve_requested_device(requested: str, allow_cpu_fallback: bool) -> str:
    normalized = str(requested or "cuda").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            return normalized
        if allow_cpu_fallback:
            return "cpu"
        raise RuntimeError("CUDA was requested but is not available.")
    return normalized


def resolve_cli_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_dataset(args: argparse.Namespace) -> ResolvedDataset:
    dataset_key = str(args.dataset).strip().lower()
    if dataset_key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    spec = DATASET_SPECS[dataset_key]
    split = args.split or spec.default_split
    if spec.valid_splits and split not in spec.valid_splits:
        allowed = ", ".join(spec.valid_splits)
        raise ValueError(f"Unsupported split '{split}' for dataset '{dataset_key}'. Allowed: {allowed}")

    if args.ir_dir or args.vis_dir:
        if not args.ir_dir or not args.vis_dir:
            raise ValueError("Both --ir-dir and --vis-dir must be provided together.")
        ir_dir = resolve_cli_path(args.ir_dir)
        vis_dir = resolve_cli_path(args.vis_dir)
        common_parent = ir_dir.parent if ir_dir.parent == vis_dir.parent else ir_dir.parent
        output_dir = resolve_cli_path(args.output_dir) if args.output_dir else common_parent / "labels"
        checkpoint = resolve_cli_path(args.checkpoint) if args.checkpoint else (PROJECT_ROOT / spec.default_checkpoint).resolve()
        display_name = dataset_key.upper() if not split else f"{dataset_key.upper()}:{split}"
        return ResolvedDataset(
            key=dataset_key,
            display_name=display_name,
            root_dir=common_parent,
            split=split,
            ir_dir=ir_dir,
            vis_dir=vis_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
        )

    dataset_root = resolve_cli_path(args.dataset_root) if args.dataset_root else resolve_cli_path(str(Path(args.data_root) / spec.root_name))
    checkpoint = resolve_cli_path(args.checkpoint) if args.checkpoint else (PROJECT_ROOT / spec.default_checkpoint).resolve()
    model_name = checkpoint.stem
    ir_dir = dataset_root.joinpath(*render_path_parts(spec.ir_template, split, model_name=model_name))
    vis_dir = dataset_root.joinpath(*render_path_parts(spec.vis_template, split, model_name=model_name))
    output_dir = (
        resolve_cli_path(args.output_dir)
        if args.output_dir
        else dataset_root.joinpath(*render_path_parts(spec.output_template, split, model_name=model_name))
    )
    display_name = spec.key.upper() if not split else f"{spec.key.upper()}:{split}"
    return ResolvedDataset(
        key=spec.key,
        display_name=display_name,
        root_dir=dataset_root,
        split=split,
        ir_dir=ir_dir,
        vis_dir=vis_dir,
        output_dir=output_dir,
        checkpoint=checkpoint,
    )


def scan_image_dir(folder: Path, allowed_extensions: Sequence[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    allowed = set(normalize_extensions(allowed_extensions))
    if not folder.exists():
        return mapping
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            mapping.setdefault(path.stem, path)
    return mapping


def build_output_suffix(output_ext: str, source_path: Path) -> str:
    normalized = str(output_ext or "source").strip().lower()
    if normalized in {"source", "same", "keep"}:
        return source_path.suffix or ".png"
    return normalized if normalized.startswith(".") else f".{normalized}"


def discover_pairs(
    ir_dir: Path,
    vis_dir: Path,
    output_dir: Path,
    allowed_extensions: Sequence[str],
    output_ext: str,
    limit: int,
) -> List[PairItem]:
    ir_map = scan_image_dir(ir_dir, allowed_extensions)
    vis_map = scan_image_dir(vis_dir, allowed_extensions)
    matched_names = sorted(set(ir_map).intersection(vis_map))
    if limit > 0:
        matched_names = matched_names[:limit]

    items: List[PairItem] = []
    for name in matched_names:
        ir_path = ir_map[name]
        vis_path = vis_map[name]
        suffix = build_output_suffix(output_ext, ir_path)
        items.append(
            PairItem(
                name=name,
                ir_path=ir_path,
                vis_path=vis_path,
                output_path=output_dir / f"{name}{suffix}",
            )
        )
    return items


def save_gray_image(path: Path, image_float01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image_float01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), image_u8):
        raise IOError(f"Failed to save image: {path}")


class BaseFusionAdapter:
    """
    Project-specific logic belongs here.

    To migrate this file to another project, replace only this adapter with the
    target model's loading and inference code.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        input_size: Optional[Tuple[int, int]],
        amp: bool,
        decoder_input_mode: str,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.input_size = input_size
        self.amp = bool(amp and device.startswith("cuda"))
        self.decoder_input_mode = str(decoder_input_mode).strip().lower()

    def load(self) -> None:
        raise NotImplementedError

    def fuse_pair(self, ir_path: Path, vis_path: Path) -> np.ndarray:
        raise NotImplementedError


class GLoCMambaAdapter(BaseFusionAdapter):
    def __init__(
        self,
        checkpoint_path: Path,
        device: str,
        input_size: Optional[Tuple[int, int]],
        amp: bool,
        decoder_input_mode: str,
    ) -> None:
        super().__init__(checkpoint_path, device, input_size, amp, decoder_input_mode)
        self.encoder = None
        self.decoder = None
        self.modal_enhance = None
        self.cross_mamba_fusion = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from network.model_loader import build_current_glcm_model
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Missing dependency '{exc.name}' required by GLoC-Mamba."
            ) from exc

        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        model = build_current_glcm_model(checkpoint, self.device, data_parallel=False)
        self.encoder = model['encoder']
        self.decoder = model['decoder']
        self.modal_enhance = model['modal_enhance']
        self.cross_mamba_fusion = model['cross_mamba_fusion']
        self._loaded = True

    def _read_gray(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        if self.input_size is None or image.shape[:2] == self.input_size:
            return image
        return cv2.resize(
            image,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def _to_tensor(self, image_u8: np.ndarray) -> torch.Tensor:
        array = image_u8.astype(np.float32, copy=False) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device, non_blocking=self.device.startswith('cuda'))

    def _validate_decoder_input_mode(self) -> None:
        mode = self.decoder_input_mode or 'auto'
        if mode not in {'auto', 'none'}:
            raise ValueError(
                "GLoC-Mamba uses decoder_input_mode='none'; "
                f"received {self.decoder_input_mode!r}."
            )

    @staticmethod
    def _normalize_output(tensor: torch.Tensor) -> torch.Tensor:
        min_value = torch.min(tensor)
        max_value = torch.max(tensor)
        return (tensor - min_value) / torch.clamp(max_value - min_value, min=1e-8)

    def fuse_pair(self, ir_path: Path, vis_path: Path) -> np.ndarray:
        self.load()
        assert self.encoder is not None
        assert self.decoder is not None
        assert self.modal_enhance is not None
        assert self.cross_mamba_fusion is not None
        self._validate_decoder_input_mode()

        ir_image = self._read_gray(ir_path)
        vis_image = self._read_gray(vis_path)
        if ir_image.shape != vis_image.shape:
            raise ValueError(
                f"Input pair shape mismatch: {ir_path.name}={ir_image.shape}, "
                f"{vis_path.name}={vis_image.shape}"
            )

        ir_tensor = self._to_tensor(ir_image)
        vis_tensor = self._to_tensor(vis_image)
        amp_context = (
            torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
            if self.amp
            else contextlib.nullcontext()
        )

        with torch.inference_mode(), amp_context:
            feature_v_g, feature_v_l, _ = self.encoder(vis_tensor)
            feature_i_g, feature_i_l, _ = self.encoder(ir_tensor)
            feature_i_e, feature_v_e = self.modal_enhance(
                feature_i_g,
                feature_i_l,
                feature_v_g,
                feature_v_l,
            )
            fused_feature = self.cross_mamba_fusion(feature_i_e, feature_v_e)
            fused_tensor, _ = self.decoder(fused_feature)
            fused_tensor = self._normalize_output(fused_tensor)

        fused = np.squeeze(fused_tensor.detach().cpu().numpy()).astype(
            np.float32,
            copy=False,
        )
        return np.clip(fused, 0.0, 1.0)


def build_adapter(args: argparse.Namespace, checkpoint_path: Path, device: str) -> BaseFusionAdapter:
    adapter_name = str(args.adapter).strip().lower()
    if adapter_name != "gloc-mamba":
        raise ValueError(f"Unsupported adapter: {args.adapter}")
    return GLoCMambaAdapter(
        checkpoint_path=checkpoint_path,
        device=device,
        input_size=parse_hw(args.input_size),
        amp=args.amp,
        decoder_input_mode=args.decoder_input_mode,
    )


class ProgressPrinter:
    def __init__(self, total: int) -> None:
        self.total = max(int(total), 1)
        self.start_time = time.perf_counter()
        self.last_width = 0
        self.term_width = shutil.get_terminal_size((120, 20)).columns

    def update(self, index: int, saved: int, skipped: int, failed: int, current_name: str) -> None:
        elapsed = time.perf_counter() - self.start_time
        speed = index / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - index, 0)
        eta = remaining / speed if speed > 0 else 0.0
        bar_width = 24
        filled = int(round(bar_width * index / self.total))
        bar = "#" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {index}/{self.total} "
            f"saved={saved} skipped={skipped} failed={failed} "
            f"left={remaining} eta={format_duration(eta)} {current_name}"
        )
        trimmed = line[: max(self.term_width - 1, 1)]
        print(trimmed.ljust(self.last_width), end="", flush=True)
        self.last_width = len(trimmed)

    def finish(self) -> None:
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate fusion labels with GLoC-Mamba.")
    parser.add_argument("--dataset", type=str.lower, default="roadscene", choices=sorted(DATASET_SPECS))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-root", type=str, default=None, help="Override the selected dataset root.")
    parser.add_argument("--split", type=str.lower, default=None, help="Dataset split, such as train or test.")
    parser.add_argument("--ir-dir", type=str, default=None, help="Custom infrared folder; overrides dataset layout.")
    parser.add_argument("--vis-dir", type=str, default=None, help="Custom visible folder; overrides dataset layout.")
    parser.add_argument("--output-dir", type=str, default=None, help="Label output directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument("--adapter", type=str, default="gloc-mamba", help="Model adapter name.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | cuda:0 | cpu | auto")
    parser.add_argument(
        "--allow-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fall back to CPU if CUDA is unavailable.",
    )
    parser.add_argument("--input-size", type=str, default="0", help="0 keeps original size, or 256, or 640x480.")
    parser.add_argument(
        "--decoder-input-mode",
        type=str,
        default="auto",
        choices=("auto", "none"),
        help="GLoC-Mamba uses no decoder residual input.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA autocast for faster inference when supported.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing label images.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N matched pairs. 0 means all.")
    parser.add_argument("--output-ext", type=str, default="source", help="source or a specific extension such as .png.")
    parser.add_argument("--file-extensions", nargs="+", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    return parser


def configure_runtime(device: str) -> None:
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def run(args: argparse.Namespace) -> int:
    resolved = resolve_dataset(args)
    device = resolve_requested_device(args.device, allow_cpu_fallback=args.allow_cpu_fallback)
    configure_runtime(device)

    if not resolved.ir_dir.exists():
        raise FileNotFoundError(f"Infrared directory does not exist: {resolved.ir_dir}")
    if not resolved.vis_dir.exists():
        raise FileNotFoundError(f"Visible directory does not exist: {resolved.vis_dir}")
    if not resolved.checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved.checkpoint}")

    pairs = discover_pairs(
        ir_dir=resolved.ir_dir,
        vis_dir=resolved.vis_dir,
        output_dir=resolved.output_dir,
        allowed_extensions=args.file_extensions,
        output_ext=args.output_ext,
        limit=args.limit,
    )
    if not pairs:
        raise RuntimeError("No matched infrared-visible pairs were found.")

    print(f"Dataset: {resolved.display_name}")
    print(f"Output:  {resolved.output_dir}")

    if args.dry_run:
        print(f"Matched pairs: {len(pairs)}")
        return 0

    resolved.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = build_adapter(args, resolved.checkpoint, device)
    progress = ProgressPrinter(total=len(pairs))

    saved = 0
    skipped = 0
    failed = 0
    failed_items: List[Tuple[str, str]] = []

    for index, pair in enumerate(pairs, start=1):
        try:
            if pair.output_path.exists() and not args.overwrite:
                skipped += 1
            else:
                fused_float01 = adapter.fuse_pair(pair.ir_path, pair.vis_path)
                save_gray_image(pair.output_path, fused_float01)
                saved += 1
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            failed_items.append((pair.name, str(exc)))
            if args.debug:
                print()
                print(f"Failed: {pair.name} -> {exc}")
        progress.update(index=index, saved=saved, skipped=skipped, failed=failed, current_name=pair.name)

    progress.finish()
    elapsed = time.perf_counter() - progress.start_time
    print(f"Completed: total={len(pairs)} saved={saved} skipped={skipped} failed={failed} elapsed={format_duration(elapsed)}")
    if failed_items:
        preview = ", ".join(name for name, _ in failed_items[:10])
        print(f"Failed items: {preview}")
    return 0 if failed == 0 else 1


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:  # pylint: disable=broad-except
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
