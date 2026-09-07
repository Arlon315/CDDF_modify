"""Construction and checkpoint loading for GLoC-Mamba and its ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn

from .ablation import build_ablation_modules, require_ablation_checkpoint


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
    """Select the saved ablation (legacy checkpoints use 0) and load strictly."""
    ablation = require_ablation_checkpoint(checkpoint)

    encoder, decoder, modal_enhance, cross_mamba_fusion = build_ablation_modules(ablation)

    _load_state(encoder, checkpoint, 'DIDF_Encoder')
    _load_state(decoder, checkpoint, 'DIDF_Decoder')
    if modal_enhance is not None:
        _load_state(modal_enhance, checkpoint, 'ModalEnhanceLayer')
    _load_state(cross_mamba_fusion, checkpoint, 'CrossMambaFusionLayer')

    modules = {
        'encoder': encoder.to(device),
        'decoder': decoder.to(device),
        'modal_enhance': modal_enhance.to(device) if modal_enhance is not None else None,
        'cross_mamba_fusion': cross_mamba_fusion.to(device),
    }
    if data_parallel:
        modules = {name: nn.DataParallel(module) if module is not None else None for name, module in modules.items()}

    for module in modules.values():
        if module is not None:
            module.eval()

    return modules


__all__ = ['build_current_glcm_model']
