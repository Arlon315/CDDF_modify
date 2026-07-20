"""Construction and checkpoint loading for the fixed GLoC-Mamba network."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from .CMFB import CommenMambaFusionBlock
from .GLCM_Mamba import GlobalLocalCrossModalMambaBlock
from .net import build_current_glcm_modules, require_current_glcm_checkpoint


def _strip_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def _load_state(module: nn.Module, checkpoint: Mapping[str, Any], key: str) -> None:
    module.load_state_dict(_strip_module_prefix(checkpoint[key]), strict=True)


def build_current_glcm_model(
    checkpoint: Mapping[str, Any],
    device: str | torch.device,
    *,
    data_parallel: bool = False,
) -> Dict[str, Any]:
    """Build the current network and strictly load a compatible checkpoint."""
    require_current_glcm_checkpoint(checkpoint)

    encoder, decoder = build_current_glcm_modules()
    modal_enhance = GlobalLocalCrossModalMambaBlock(dim=64)
    cross_mamba_fusion = CommenMambaFusionBlock(dim=64)

    _load_state(encoder, checkpoint, 'DIDF_Encoder')
    _load_state(decoder, checkpoint, 'DIDF_Decoder')
    _load_state(modal_enhance, checkpoint, 'ModalEnhanceLayer')
    _load_state(cross_mamba_fusion, checkpoint, 'CrossMambaFusionLayer')

    modules = {
        'encoder': encoder.to(device),
        'decoder': decoder.to(device),
        'modal_enhance': modal_enhance.to(device),
        'cross_mamba_fusion': cross_mamba_fusion.to(device),
    }
    if data_parallel:
        modules = {name: nn.DataParallel(module) for name, module in modules.items()}

    for module in modules.values():
        module.eval()

    return modules


__all__ = ['build_current_glcm_model']
