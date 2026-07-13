# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import (
    build_cddfuse_modules,
    infer_cddfuse_backbone,
    infer_cddfuse_decoder_block,
    infer_cddfuse_encoder_base_feature,
    infer_cddfuse_encoder_detail_feature,
    resolve_cddfuse_encoder_base_feature,
    resolve_cddfuse_encoder_detail_feature,
    resolve_cddfuse_decoder_block,
)
from CrossMambaFusion import (
    CROSS_MAMBA_FUSION_STRUCTURE,
    CrossMambaFusionBlock,
    IntraModalEnhanceBlock,
    get_decoder_residual_input,
    infer_cross_mamba_share_mode,
)
from utils.dataset import H5Dataset
import argparse
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, PixelBSCLLoss, cc
import kornia


def set_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 42 3407 2026
seed = 42
set_seed(seed)


'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
criteria_pixel_bscl = PixelBSCLLoss()
model_str = 'CDDFuse'

parser = argparse.ArgumentParser(description="Train CDDFuse with checkpoint resume support.")
parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
parser.add_argument(
    "--resume_mode",
    choices=("auto", "full", "pretrain"),
    default="auto",
    help="full strictly resumes all modules; pretrain loads Phase I weights and starts Phase II; auto chooses by checkpoint structure.",
)
parser.add_argument("--checkpoint_dir", type=str, default="models/newStructure/", help="Directory for saved checkpoints.")
parser.add_argument("--save_interval", type=int, default=10, help="Save a checkpoint every N epochs.")
parser.add_argument(
    "--backbone",
    choices=("fast", "restormer"),
    default="restormer",
    help="Use the NAF-style fast backbone or the Restormer encoder backbone.",
)
parser.add_argument(
    "--encoder_base_feature",
    choices=("auto", "spatial_mamba", "base"),
    default="spatial_mamba",
    help="Encoder base branch. auto uses Spatial Mamba for restormer and NAF for fast.",
)
parser.add_argument(
    "--encoder_detail_feature",
    choices=("auto", "INN", "INN+DEConv", "INN+AKDEConv", "AKDEConv+CGA"),
    default="AKDEConv+CGA",
    help="Encoder detail branch. auto keeps INN+DEConv; INN uses three INN nodes; INN+AKDEConv adds an AKConv detail branch.",
)
parser.add_argument(
    "--decoder_block",
    choices=("auto", "htb", "restormer", "naf"),
    default="htb",
    help="Decoder reconstruction block. auto uses HTB for restormer and NAF for fast.",
)
parser.add_argument(
    "--detail_fusion",
    choices=("cga", "dff", "inn"),
    default="cga",
    help="Use CGAFusion, DFF-style fusion, or the original INN DetailFeatureExtraction fusion.",
)
parser.add_argument(
    "--base_fusion",
    choices=("base", "baseSAFM", "windowMCAM", "gmem"),
    default="gmem",
    help="Use base fusion, baseSAFM fusion, Swin-WindowMCAM fusion, or GMEM.",
)
parser.add_argument(
    "--gmem_share_mode",
    choices=("independent", "axis", "all"),
    default="independent",
    help="GMEM direction parameter sharing: independent, axis, or all.",
)
parser.add_argument(
    "--cross_mamba_share_mode",
    choices=("independent", "axis", "all"),
    default="independent",
    help="Cross-Mamba direction parameter sharing for the new low/high fusion path.",
)
parser.add_argument(
    "--skip_phase1",
    action="store_true",
    default=True,
    help="Skip the single-modality reconstruction phase and train fusion from epoch 0.",
)
parser.add_argument(
    "--use_decomp_loss",
    action="store_true",
    default=False,
    help="Enable the original low/high correlation decomposition loss.",
)
parser.add_argument(
    "--decoder_residual",
    choices=("none", "ir", "vis", "ir+vis"),
    default="none",
    help="Residual image passed to the decoder in the fusion phase.",
)

args = parser.parse_args()
encoder_base_feature = resolve_cddfuse_encoder_base_feature(args.encoder_base_feature, args.backbone)
encoder_detail_feature = resolve_cddfuse_encoder_detail_feature(args.encoder_detail_feature)
decoder_block = resolve_cddfuse_decoder_block(args.decoder_block, args.backbone)
decoder_block_suffix = "" if args.backbone == "fast" and decoder_block == "naf" else f"_{decoder_block}"
encoder_base_suffix = "" if encoder_base_feature in ("base", "naf") else f"_{encoder_base_feature}"
encoder_detail_suffix = f"_{encoder_detail_feature.replace('+', '_')}"
phase_suffix = "_skipP1" if args.skip_phase1 else ""
decomp_suffix = "_decomp" if args.use_decomp_loss else ""
residual_suffix = "" if args.decoder_residual == "none" else f"_res{args.decoder_residual.replace('+', '_')}"
model_str = (
    f"{encoder_base_suffix}{encoder_detail_suffix}{decoder_block_suffix}"
    f"_lowhigh_crossmamba_privateakc_{args.cross_mamba_share_mode}"
    f"{phase_suffix}{decomp_suffix}{residual_suffix}"
)

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 30  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.
coeff_pixel_bscl = 0.0

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_module, decoder_module, _, _ = build_cddfuse_modules(
    args.backbone,
    detail_fusion=args.detail_fusion,
    encoder_base_feature=encoder_base_feature,
    encoder_detail_feature=encoder_detail_feature,
    base_fusion=args.base_fusion,
    gmem_share_mode=args.gmem_share_mode,
    decoder_block=decoder_block,
)
DIDF_Encoder = nn.DataParallel(encoder_module).to(device)
DIDF_Decoder = nn.DataParallel(decoder_module).to(device)
ModalEnhanceLayer = nn.DataParallel(
    IntraModalEnhanceBlock(dim=64)
).to(device)
CrossMambaFusionLayer = nn.DataParallel(
    CrossMambaFusionBlock(dim=64, share_mode=args.cross_mamba_share_mode)
).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    ModalEnhanceLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    CrossMambaFusionLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')


# data loader
generator = torch.Generator()
generator.manual_seed(seed)

trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=16,
                         worker_init_fn=seed_worker,
                         generator=generator)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
start_epoch = 0

os.makedirs(args.checkpoint_dir, exist_ok=True)


def get_detail_fusion_num_layers():
    return None


def build_checkpoint(epoch):
    return {
        'epoch': epoch,
        'seed': seed,
        'timestamp': timestamp,
        'fusion_structure': CROSS_MAMBA_FUSION_STRUCTURE,
        'cross_mamba_share_mode': args.cross_mamba_share_mode,
        'skip_phase1': bool(args.skip_phase1),
        'use_decomp_loss': bool(args.use_decomp_loss),
        'decoder_residual': args.decoder_residual,
        'backbone': args.backbone,
        'encoder_base_feature': encoder_base_feature,
        'encoder_detail_feature': encoder_detail_feature,
        'decoder_block': decoder_block,
        'detail_fusion': args.detail_fusion,
        'base_fusion': args.base_fusion,
        'gmem_share_mode': args.gmem_share_mode,
        'encoder_detail_enhance': 'akdeconv_cga' if encoder_detail_feature == 'AKDEConv+CGA' else ('akdeconv' if encoder_detail_feature == 'INN+AKDEConv' else ('deconv' if encoder_detail_feature == 'INN+DEConv' else None)),
        'encoder_detail_enhance_layers': 0 if encoder_detail_feature == 'AKDEConv+CGA' else (2 if encoder_detail_feature in ('INN+DEConv', 'INN+AKDEConv') else 0),
        'encoder_detail_num_layers': 0 if encoder_detail_feature == 'AKDEConv+CGA' else (1 if encoder_detail_feature in ('INN+DEConv', 'INN+AKDEConv') else 3),
        'decoder_freq_enhance': 'dynamic_filter',
        'detail_fusion_num_layers': get_detail_fusion_num_layers(),
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'ModalEnhanceLayer': ModalEnhanceLayer.state_dict(),
        'CrossMambaFusionLayer': CrossMambaFusionLayer.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'optimizer3': optimizer3.state_dict(),
        'optimizer4': optimizer4.state_dict(),
        'scheduler1': scheduler1.state_dict(),
        'scheduler2': scheduler2.state_dict(),
        'scheduler3': scheduler3.state_dict(),
        'scheduler4': scheduler4.state_dict(),
    }

def save_checkpoint(epoch, save_tag=None):
    checkpoint = build_checkpoint(epoch)
    tag = save_tag or f"epoch_{epoch:03d}"
    ckpt_path = os.path.join(args.checkpoint_dir, f"{model_str}_{timestamp}_{tag}.pth")
    latest_path = os.path.join(args.checkpoint_dir, f"{model_str}_latest.pth")
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    print(f"\nSaved checkpoint: {ckpt_path}")


def load_state_if_present(module, checkpoint, key, required=True, strict=True):
    if key not in checkpoint:
        if required:
            raise KeyError(f"Checkpoint is missing required key: {key}")
        print(f"Skipped {key}: not found in checkpoint.")
        return False
    try:
        result = module.load_state_dict(checkpoint[key], strict=strict)
        if not strict and (result.missing_keys or result.unexpected_keys):
            print(
                f"Loaded {key} with strict=False. "
                f"Missing keys: {result.missing_keys}; unexpected keys: {result.unexpected_keys}"
            )
        return True
    except RuntimeError as exc:
        if required:
            raise
        print(f"Skipped {key}: incompatible state dict ({exc}).")
        return False


def load_compatible_state_if_present(module, checkpoint, key, required=True):
    if key not in checkpoint:
        if required:
            raise KeyError(f"Checkpoint is missing required key: {key}")
        print(f"Skipped {key}: not found in checkpoint.")
        return False

    current_state = module.state_dict()
    compatible_state = {}
    skipped_keys = []

    for source_key, value in checkpoint[key].items():
        target_key = source_key
        if target_key not in current_state:
            if source_key.startswith('module.') and source_key[7:] in current_state:
                target_key = source_key[7:]
            elif f"module.{source_key}" in current_state:
                target_key = f"module.{source_key}"
            else:
                skipped_keys.append(source_key)
                continue

        if current_state[target_key].shape != value.shape:
            skipped_keys.append(source_key)
            continue
        compatible_state[target_key] = value

    current_state.update(compatible_state)
    module.load_state_dict(current_state)
    print(
        f"Partially loaded {key}: {len(compatible_state)} compatible tensors, "
        f"skipped {len(skipped_keys)} incompatible tensors."
    )
    return True


def load_optimizer_if_present(optimizer, checkpoint, key):
    if key not in checkpoint:
        print(f"Skipped {key}: not found in checkpoint.")
        return False
    try:
        optimizer.load_state_dict(checkpoint[key])
        return True
    except ValueError as exc:
        print(f"Skipped {key}: incompatible optimizer state ({exc}).")
        return False


def load_scheduler_if_present(scheduler, checkpoint, key):
    if key not in checkpoint:
        print(f"Skipped {key}: not found in checkpoint.")
        return False
    try:
        scheduler.load_state_dict(checkpoint[key])
        return True
    except Exception as exc:
        print(f"Skipped {key}: incompatible scheduler state ({exc}).")
        return False


if args.resume:
    resume_path = os.path.expanduser(args.resume)
    checkpoint = torch.load(resume_path, map_location=device)
    checkpoint_backbone = infer_cddfuse_backbone(checkpoint)
    if checkpoint_backbone != args.backbone:
        raise ValueError(
            f"Checkpoint backbone is '{checkpoint_backbone}', but current --backbone is '{args.backbone}'."
        )

    checkpoint_decoder_block = infer_cddfuse_decoder_block(checkpoint, checkpoint_backbone)
    checkpoint_encoder_base_feature = infer_cddfuse_encoder_base_feature(checkpoint)
    checkpoint_encoder_detail_feature = infer_cddfuse_encoder_detail_feature(checkpoint)
    checkpoint_cross_mamba_share_mode = infer_cross_mamba_share_mode(checkpoint)
    checkpoint_fusion_structure = checkpoint.get('fusion_structure') if isinstance(checkpoint, dict) else None

    decoder_block_matches = checkpoint_decoder_block == decoder_block
    encoder_base_feature_matches = checkpoint_encoder_base_feature == encoder_base_feature
    encoder_detail_feature_matches = checkpoint_encoder_detail_feature == encoder_detail_feature
    fusion_structure_matches = checkpoint_fusion_structure == CROSS_MAMBA_FUSION_STRUCTURE
    cross_mamba_share_mode_matches = checkpoint_cross_mamba_share_mode == args.cross_mamba_share_mode

    resume_mode = args.resume_mode
    if resume_mode == "auto":
        resume_mode = (
            "full"
            if (
                decoder_block_matches
                and encoder_base_feature_matches
                and encoder_detail_feature_matches
                and fusion_structure_matches
                and cross_mamba_share_mode_matches
            )
            else "pretrain"
        )

    if resume_mode == "full":
        if not encoder_base_feature_matches:
            raise ValueError(
                f"Checkpoint encoder_base_feature is '{checkpoint_encoder_base_feature}', "
                f"but current --encoder_base_feature resolves to '{encoder_base_feature}'."
            )
        if not encoder_detail_feature_matches:
            raise ValueError(
                f"Checkpoint encoder_detail_feature is '{checkpoint_encoder_detail_feature}', "
                f"but current --encoder_detail_feature resolves to '{encoder_detail_feature}'."
            )
        if not decoder_block_matches:
            raise ValueError(
                f"Checkpoint decoder_block is '{checkpoint_decoder_block}', "
                f"but current --decoder_block resolves to '{decoder_block}'."
            )
        if not fusion_structure_matches:
            raise ValueError(
                f"Checkpoint fusion_structure is '{checkpoint_fusion_structure}', "
                f"but current training expects '{CROSS_MAMBA_FUSION_STRUCTURE}'."
            )
        if not cross_mamba_share_mode_matches:
            raise ValueError(
                f"Checkpoint cross_mamba_share_mode is '{checkpoint_cross_mamba_share_mode}', "
                f"but current --cross_mamba_share_mode is '{args.cross_mamba_share_mode}'."
            )

    load_state_if_present(DIDF_Encoder, checkpoint, 'DIDF_Encoder', strict=False)
    load_compatible_state_if_present(DIDF_Decoder, checkpoint, 'DIDF_Decoder')
    load_state_if_present(ModalEnhanceLayer, checkpoint, 'ModalEnhanceLayer', required=False)
    load_compatible_state_if_present(CrossMambaFusionLayer, checkpoint, 'CrossMambaFusionLayer', required=False)

    checkpoint_epoch = int(checkpoint.get('epoch', 0))
    if resume_mode == "full":
        load_optimizer_if_present(optimizer1, checkpoint, 'optimizer1')
        load_optimizer_if_present(optimizer2, checkpoint, 'optimizer2')
        load_optimizer_if_present(optimizer3, checkpoint, 'optimizer3')
        load_optimizer_if_present(optimizer4, checkpoint, 'optimizer4')
        load_scheduler_if_present(scheduler1, checkpoint, 'scheduler1')
        load_scheduler_if_present(scheduler2, checkpoint, 'scheduler2')
        load_scheduler_if_present(scheduler3, checkpoint, 'scheduler3')
        load_scheduler_if_present(scheduler4, checkpoint, 'scheduler4')
        start_epoch = checkpoint_epoch
        print(f"Resumed full checkpoint from {resume_path} at epoch {start_epoch}.")
    else:
        skipped_resume_parts = []
        if not encoder_base_feature_matches:
            skipped_resume_parts.append('DIDF_Encoder base branch optimizer/scheduler')
            print(
                f"Partially loaded DIDF_Encoder: checkpoint encoder_base_feature='{checkpoint_encoder_base_feature}', "
                f"current encoder_base_feature='{encoder_base_feature}'."
            )
        if not encoder_detail_feature_matches:
            skipped_resume_parts.append('DIDF_Encoder high branch optimizer/scheduler')
            print(
                f"Partially loaded DIDF_Encoder: checkpoint encoder_detail_feature="
                f"'{checkpoint_encoder_detail_feature}', current encoder_detail_feature="
                f"'{encoder_detail_feature}'."
            )
        if not decoder_block_matches:
            skipped_resume_parts.append('DIDF_Decoder optimizer/scheduler')
            print(
                f"Partially loaded DIDF_Decoder: checkpoint decoder_block='{checkpoint_decoder_block}', "
                f"current decoder_block='{decoder_block}'."
            )
        if not fusion_structure_matches:
            skipped_resume_parts.append('ModalEnhanceLayer/CrossMambaFusionLayer')
            print(
                f"Skipped new fusion layers: checkpoint fusion_structure='{checkpoint_fusion_structure}'."
            )
        elif not cross_mamba_share_mode_matches:
            skipped_resume_parts.append('CrossMambaFusionLayer optimizer/scheduler')
            print(
                f"Partially loaded fusion layers: checkpoint cross_mamba_share_mode="
                f"'{checkpoint_cross_mamba_share_mode}', current='{args.cross_mamba_share_mode}'."
            )

        if encoder_base_feature_matches and encoder_detail_feature_matches:
            load_optimizer_if_present(optimizer1, checkpoint, 'optimizer1')
            load_scheduler_if_present(scheduler1, checkpoint, 'scheduler1')
        if decoder_block_matches:
            load_optimizer_if_present(optimizer2, checkpoint, 'optimizer2')
            load_scheduler_if_present(scheduler2, checkpoint, 'scheduler2')
        if 'ModalEnhanceLayer' in checkpoint:
            load_optimizer_if_present(optimizer3, checkpoint, 'optimizer3')
            load_scheduler_if_present(scheduler3, checkpoint, 'scheduler3')
        if 'CrossMambaFusionLayer' in checkpoint and fusion_structure_matches and cross_mamba_share_mode_matches:
            load_optimizer_if_present(optimizer4, checkpoint, 'optimizer4')
            load_scheduler_if_present(scheduler4, checkpoint, 'scheduler4')

        start_epoch = checkpoint_epoch if args.skip_phase1 else max(checkpoint_epoch, epoch_gap)
        skipped_message = ', '.join(skipped_resume_parts) if skipped_resume_parts else 'no incompatible parts'
        print(
            f"Loaded pretrain from {resume_path}: "
            f"checkpoint encoder_base_feature='{checkpoint_encoder_base_feature}', "
            f"current encoder_base_feature='{encoder_base_feature}'; "
            f"checkpoint encoder_detail_feature='{checkpoint_encoder_detail_feature}', "
            f"current encoder_detail_feature='{encoder_detail_feature}'; "
            f"checkpoint decoder_block='{checkpoint_decoder_block}', current decoder_block='{decoder_block}'. "
            f"Skipped {skipped_message}; starting at epoch {start_epoch}."
        )
'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(start_epoch, num_epochs):
    ''' train '''
    epoch_loss = 0.0
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        ModalEnhanceLayer.train()
        CrossMambaFusionLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        ModalEnhanceLayer.zero_grad()
        CrossMambaFusionLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        run_phase1 = (not args.skip_phase1) and epoch < epoch_gap
        if run_phase1:
            feature_V_L, feature_V_H, _ = DIDF_Encoder(data_VIS)
            feature_I_L, feature_I_H, _ = DIDF_Encoder(data_IR)
            feature_V_E = ModalEnhanceLayer(feature_V_L, feature_V_H)
            feature_I_E = ModalEnhanceLayer(feature_I_L, feature_I_H)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, fused_feature=feature_V_E)
            data_IR_hat, _ = DIDF_Decoder(data_IR, fused_feature=feature_I_E)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)
            Gradient_loss = 0.5 * (
                L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                       kornia.filters.SpatialGradient()(data_VIS_hat))
                + L1Loss(kornia.filters.SpatialGradient()(data_IR),
                         kornia.filters.SpatialGradient()(data_IR_hat))
            )

            loss_decomp = torch.tensor(0.0, device=device)
            if args.use_decomp_loss:
                cc_loss_L = cc(feature_V_L, feature_I_L)
                cc_loss_H = cc(feature_V_H, feature_I_H)
                loss_decomp = (cc_loss_H) ** 2 / (1.01 + cc_loss_L)

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                ModalEnhanceLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
        else:
            feature_V_L, feature_V_H, _ = DIDF_Encoder(data_VIS)
            feature_I_L, feature_I_H, _ = DIDF_Encoder(data_IR)
            feature_V_E = ModalEnhanceLayer(feature_V_L, feature_V_H)
            feature_I_E = ModalEnhanceLayer(feature_I_L, feature_I_H)
            feature_F_E = CrossMambaFusionLayer(feature_I_E, feature_V_E)
            decoder_input = get_decoder_residual_input(args.decoder_residual, data_IR, data_VIS)
            data_Fuse, feature_F = DIDF_Decoder(decoder_input, fused_feature=feature_F_E)

            fusionloss, _, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
            pixel_bscl_loss = criteria_pixel_bscl(data_VIS, data_IR, data_Fuse)
            loss_decomp = torch.tensor(0.0, device=device)
            if args.use_decomp_loss:
                cc_loss_L = cc(feature_V_L, feature_I_L)
                cc_loss_H = cc(feature_V_H, feature_I_H)
                loss_decomp = (cc_loss_H) ** 2 / (1.01 + cc_loss_L)

            loss = fusionloss + coeff_decomp * loss_decomp + coeff_pixel_bscl * pixel_bscl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                ModalEnhanceLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                CrossMambaFusionLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
        loss_value = loss.item()
        epoch_loss += loss_value

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss_value,
                time_left,
            )
        )

    avg_loss = epoch_loss / max(1, len(loader['train']))
    print("\n[Epoch %d/%d] [avg_loss: %f]" % (epoch + 1, num_epochs, avg_loss))

    # adjust the learning rate

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    if args.skip_phase1 or epoch >= epoch_gap:
        scheduler4.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    finished_epoch = epoch + 1
    if args.save_interval > 0 and finished_epoch % args.save_interval == 0:
        save_checkpoint(finished_epoch)
    
save_checkpoint(num_epochs, save_tag="final")
