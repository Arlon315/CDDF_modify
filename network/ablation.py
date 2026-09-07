"""The full model (0) and the six fixed ablation configurations (1-6)."""

import torch
import torch.nn as nn

from .CMFB import CommenMambaFusionBlock
from .GLCM_Mamba import GlobalLocalCrossModalMambaBlock
from .net import (
    CURRENT_GLCM_CONFIG,
    ENCODER_GLOBAL_LOCAL_SEMANTICS,
    Restormer_Encoder,
    Restormer_Decoder,
    TransformerBlock,
)


ABLATION_DESCRIPTIONS = {
    0: 'Full model (final baseline)',
    1: 'Full encoder + GLCM + CMFB without the entire AKC private branch',
    2: 'Full encoder + GLCM + TransformerBlock fusion',
    3: 'Full encoder + global/local cat projection + CMFB',
    4: 'Full encoder + global/local cat projection + TransformerBlock fusion',
    5: 'Restormer backbone only + CMFB (no GLCM)',
    6: 'Restormer backbone only + TransformerBlock fusion (no GLCM)',
}


class CatGlobalLocalBlock(nn.Module):
    def __init__(self, dim=64):
        super(CatGlobalLocalBlock, self).__init__()
        self.ir_fusion_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self.vi_fusion_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)

    def forward(self, ir_global, ir_local, vi_global, vi_local):
        ir_feature = self.ir_fusion_proj(torch.cat((ir_global, ir_local), dim=1))
        vi_feature = self.vi_fusion_proj(torch.cat((vi_global, vi_local), dim=1))
        return ir_feature, vi_feature


class RestormerFusionBlock(nn.Module):
    def __init__(self, dim=64):
        super(RestormerFusionBlock, self).__init__()
        self.fusion_proj = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=True)
        self.transformer = TransformerBlock(
            dim=dim,
            num_heads=8,
            ffn_expansion_factor=2,
            bias=False,
            LayerNorm_type='WithBias',
        )

    def forward(self, ir_feature, vi_feature):
        feature = self.fusion_proj(torch.cat((ir_feature, vi_feature), dim=1))
        return self.transformer(feature)


def get_ablation_config(ablation=0):
    if ablation not in ABLATION_DESCRIPTIONS:
        raise ValueError(f'Unknown ablation: {ablation!r}; expected an integer from 0 to 6.')
    config = {
        **CURRENT_GLCM_CONFIG,
        'ablation': ablation,
        'encoder_feature_semantics': ENCODER_GLOBAL_LOCAL_SEMANTICS,
    }
    if ablation == 1:
        config['fusion_structure'] = 'common_mamba_no_private'
    elif ablation in (3, 5):
        config['fusion_structure'] = 'common_mamba_private_akc'
    if ablation in (2, 4, 6):
        config.update(
            fusion_structure='cat_transformer_block',
            cross_mamba_share_mode='none',
            fusion_transformer_blocks=1,
            fusion_transformer_heads=8,
            fusion_transformer_ffn_expansion=2,
        )
    if ablation in (3, 4):
        config['modal_enhance_structure'] = 'global_local_cat'
    elif ablation in (5, 6):
        config.update(
            encoder_feature_semantics='backbone',
            encoder_global_feature='none',
            encoder_local_feature='none',
            modal_enhance_structure='none',
        )
    return config


def build_ablation_modules(ablation=0):
    get_ablation_config(ablation)
    encoder = Restormer_Encoder(use_global_local=ablation not in (5, 6))
    decoder = Restormer_Decoder()
    if ablation in (5, 6):
        modal_enhance = None
    elif ablation in (3, 4):
        modal_enhance = CatGlobalLocalBlock()
    else:
        modal_enhance = GlobalLocalCrossModalMambaBlock()
    if ablation in (2, 4, 6):
        fusion = RestormerFusionBlock()
    else:
        fusion = CommenMambaFusionBlock(use_private=ablation != 1)
    return encoder, decoder, modal_enhance, fusion


def require_ablation_checkpoint(checkpoint, ablation=None):
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a dictionary.')
    # Checkpoints from final predate this selector and represent experiment 0.
    saved_ablation = checkpoint.get('ablation', 0)
    if ablation is not None and saved_ablation != ablation:
        raise ValueError(
            f'Checkpoint ablation is {saved_ablation}, but --ablation {ablation} was selected.'
        )
    for key, expected in get_ablation_config(saved_ablation).items():
        actual = checkpoint.get(key, 0) if key == 'ablation' else checkpoint.get(key)
        if actual != expected:
            raise ValueError(f'Checkpoint {key} is {actual!r}, but requires {expected!r}.')
    for key in ('DIDF_Encoder', 'DIDF_Decoder', 'ModalEnhanceLayer', 'CrossMambaFusionLayer'):
        if key not in checkpoint:
            raise KeyError(f'Checkpoint is missing required model state: {key}.')
    if saved_ablation in (5, 6) and checkpoint['ModalEnhanceLayer'] != {}:
        raise ValueError('Experiments 5 and 6 must not contain GLCM parameters.')
    return saved_ablation
