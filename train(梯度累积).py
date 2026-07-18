# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from net import (
    ENCODER_GLOBAL_LOCAL_SEMANTICS,
    require_cddfuse_global_local_checkpoint,
    build_cddfuse_modules,
    fuse_base_features,
    fuse_detail_features,
    infer_cddfuse_base_fusion,
    infer_cddfuse_gmem_share_mode,
    infer_cddfuse_backbone,
    infer_cddfuse_decoder_block,
    infer_cddfuse_encoder_global_feature,
    infer_cddfuse_encoder_local_feature,
    infer_cddfuse_detail_fusion,
    resolve_cddfuse_encoder_global_feature,
    resolve_cddfuse_encoder_local_feature,
    resolve_cddfuse_decoder_block,
)
from utils.dataset import H5Dataset
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc
import kornia



'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

parser = argparse.ArgumentParser(description="Train CDDFuse with checkpoint resume support.")
parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
parser.add_argument(
    "--resume_mode",
    choices=("auto", "full", "pretrain"),
    default="auto",
    help="full strictly resumes all modules; pretrain loads Phase I weights and starts Phase II; auto chooses by checkpoint structure.",
)
parser.add_argument("--checkpoint_dir", type=str, default="models/MCAM_HTB/", help="Directory for saved checkpoints.")
parser.add_argument("--save_interval", type=int, default=10, help="Save a checkpoint every N epochs.")
parser.add_argument(
    "--backbone",
    choices=("fast", "restormer"),
    default="restormer",
    help="Use the NAF-style fast backbone or the Restormer encoder backbone.",
)
parser.add_argument(
    "--encoder_global_feature",
    choices=("auto", "spatial_mamba", "global"),
    default="auto",
    help="Encoder global branch. auto uses Spatial Mamba for restormer and NAF for fast.",
)
parser.add_argument(
    "--encoder_local_feature",
    choices=("auto", "INN", "INN+DEConv", "INN+AKDEConv", "AKDEConv+CGA"),
    default="auto",
    help="Encoder local branch. auto keeps INN+DEConv; INN uses three INN nodes; INN+AKDEConv adds an AKConv detail branch.",
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

args = parser.parse_args()
encoder_global_feature = resolve_cddfuse_encoder_global_feature(args.encoder_global_feature, args.backbone)
encoder_local_feature = resolve_cddfuse_encoder_local_feature(args.encoder_local_feature)
decoder_block = resolve_cddfuse_decoder_block(args.decoder_block, args.backbone)
decoder_block_suffix = "" if args.backbone == "fast" and decoder_block == "naf" else f"_{decoder_block}"
encoder_global_suffix = "" if encoder_global_feature in ("global", "naf") else f"_{encoder_global_feature}"
encoder_local_suffix = f"_{encoder_local_feature.replace('+', '_')}"
base_fusion_suffix = (
    f"_gmem_{args.gmem_share_mode}"
    if args.base_fusion == "gmem"
    else ("" if args.base_fusion == "base" else f"_{args.base_fusion}")
)
model_str = f"{args.backbone}{encoder_global_suffix}{encoder_local_suffix}{decoder_block_suffix}_{args.detail_fusion}{base_fusion_suffix}"

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 30  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 4
accumulation_steps = 2
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
# Coefficients of the loss function
coeff_mse_loss_VF = 1. # alpha1
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      # alpha2 and alpha4
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5


# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_module, decoder_module, base_fuse_module, detail_fuse_module = build_cddfuse_modules(
    args.backbone,
    detail_fusion=args.detail_fusion,
    encoder_global_feature=encoder_global_feature,
    encoder_local_feature=encoder_local_feature,
    base_fusion=args.base_fusion,
    gmem_share_mode=args.gmem_share_mode,
    decoder_block=decoder_block,
)
DIDF_Encoder = nn.DataParallel(encoder_module).to(device)
DIDF_Decoder = nn.DataParallel(decoder_module).to(device)
BaseFuseLayer = nn.DataParallel(base_fuse_module).to(device)
DetailFuseLayer = nn.DataParallel(detail_fuse_module).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')


# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=16)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
start_epoch = 0

os.makedirs(args.checkpoint_dir, exist_ok=True)


def get_detail_fusion_num_layers():
    module = DetailFuseLayer.module if isinstance(DetailFuseLayer, nn.DataParallel) else DetailFuseLayer
    net = getattr(module, 'net', None)
    return len(net) if net is not None else None


def build_checkpoint(epoch):
    return {
        'epoch': epoch,
        'timestamp': timestamp,
        'encoder_feature_semantics': ENCODER_GLOBAL_LOCAL_SEMANTICS,
        'backbone': args.backbone,
        'encoder_global_feature': encoder_global_feature,
        'encoder_local_feature': encoder_local_feature,
        'decoder_block': decoder_block,
        'detail_fusion': args.detail_fusion,
        'base_fusion': args.base_fusion,
        'gmem_share_mode': args.gmem_share_mode,
        'accumulation_steps': accumulation_steps,
        'encoder_local_enhance': 'akdeconv_cga' if encoder_local_feature == 'AKDEConv+CGA' else ('akdeconv' if encoder_local_feature == 'INN+AKDEConv' else ('deconv' if encoder_local_feature == 'INN+DEConv' else None)),
        'encoder_local_enhance_layers': 0 if encoder_local_feature == 'AKDEConv+CGA' else (2 if encoder_local_feature in ('INN+DEConv', 'INN+AKDEConv') else 0),
        'encoder_local_num_layers': 0 if encoder_local_feature == 'AKDEConv+CGA' else (1 if encoder_local_feature in ('INN+DEConv', 'INN+AKDEConv') else 3),
        'decoder_freq_enhance': 'dynamic_filter',
        'detail_fusion_num_layers': get_detail_fusion_num_layers(),
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer': DetailFuseLayer.state_dict(),
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
    require_cddfuse_global_local_checkpoint(checkpoint)
    checkpoint_backbone = infer_cddfuse_backbone(checkpoint)
    if checkpoint_backbone != args.backbone:
        raise ValueError(
            f"Checkpoint backbone is '{checkpoint_backbone}', but current --backbone is '{args.backbone}'."
        )
    checkpoint_decoder_block = infer_cddfuse_decoder_block(checkpoint, checkpoint_backbone)
    checkpoint_encoder_global_feature = infer_cddfuse_encoder_global_feature(checkpoint)
    checkpoint_encoder_local_feature = infer_cddfuse_encoder_local_feature(checkpoint)
    checkpoint_detail_fusion = infer_cddfuse_detail_fusion(checkpoint)
    checkpoint_base_fusion = infer_cddfuse_base_fusion(checkpoint)
    checkpoint_gmem_share_mode = infer_cddfuse_gmem_share_mode(checkpoint)
    decoder_block_matches = checkpoint_decoder_block == decoder_block
    encoder_global_feature_matches = checkpoint_encoder_global_feature == encoder_global_feature
    encoder_local_feature_matches = checkpoint_encoder_local_feature == encoder_local_feature
    detail_fusion_matches = checkpoint_detail_fusion == args.detail_fusion
    gmem_share_mode_matches = (
        checkpoint_base_fusion != 'gmem'
        or args.base_fusion != 'gmem'
        or checkpoint_gmem_share_mode == args.gmem_share_mode
    )
    base_fusion_matches = checkpoint_base_fusion == args.base_fusion and gmem_share_mode_matches
    resume_mode = args.resume_mode
    if resume_mode == "auto":
        resume_mode = (
            "full"
            if decoder_block_matches and encoder_global_feature_matches and encoder_local_feature_matches and detail_fusion_matches and base_fusion_matches
            else "pretrain"
        )

    if resume_mode == "full":
        if not encoder_global_feature_matches:
            raise ValueError(
                f"Checkpoint encoder_global_feature is '{checkpoint_encoder_global_feature}', "
                f"but current --encoder_global_feature resolves to '{encoder_global_feature}'."
            )
        if not encoder_local_feature_matches:
            raise ValueError(
                f"Checkpoint encoder_local_feature is '{checkpoint_encoder_local_feature}', "
                f"but current --encoder_local_feature resolves to '{encoder_local_feature}'."
            )
        if not decoder_block_matches:
            raise ValueError(
                f"Checkpoint decoder_block is '{checkpoint_decoder_block}', "
                f"but current --decoder_block resolves to '{decoder_block}'."
            )
        if not detail_fusion_matches:
            raise ValueError(
                f"Checkpoint detail_fusion is '{checkpoint_detail_fusion}', "
                f"but current --detail_fusion is '{args.detail_fusion}'."
            )
        if not gmem_share_mode_matches:
            raise ValueError(
                f"Checkpoint gmem_share_mode is '{checkpoint_gmem_share_mode}', "
                f"but current --gmem_share_mode is '{args.gmem_share_mode}'."
            )
        if not base_fusion_matches:
            raise ValueError(
                f"Checkpoint base_fusion is '{checkpoint_base_fusion}', "
                f"but current --base_fusion is '{args.base_fusion}'."
            )

    load_state_if_present(DIDF_Encoder, checkpoint, 'DIDF_Encoder', strict=False)
    if decoder_block_matches:
        load_state_if_present(DIDF_Decoder, checkpoint, 'DIDF_Decoder')
    else:
        load_compatible_state_if_present(DIDF_Decoder, checkpoint, 'DIDF_Decoder')

    checkpoint_epoch = int(checkpoint.get('epoch', 0))
    if resume_mode == "full":
        load_state_if_present(BaseFuseLayer, checkpoint, 'BaseFuseLayer')
        load_state_if_present(DetailFuseLayer, checkpoint, 'DetailFuseLayer')
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
        if not encoder_global_feature_matches:
            skipped_resume_parts.append('DIDF_Encoder globalFeature weights/optimizer1/scheduler1')
            print(
                f"Partially loaded DIDF_Encoder: checkpoint encoder_global_feature='{checkpoint_encoder_global_feature}', "
                f"current encoder_global_feature='{encoder_global_feature}'."
            )
        if not encoder_local_feature_matches:
            skipped_resume_parts.append('DIDF_Encoder localFeature weights/optimizer1/scheduler1')
            print(
                f"Partially loaded DIDF_Encoder: checkpoint encoder_local_feature='{checkpoint_encoder_local_feature}', "
                f"current encoder_local_feature='{encoder_local_feature}'."
            )
        if not decoder_block_matches:
            skipped_resume_parts.append('DIDF_Decoder block weights/optimizer2/scheduler2')
            print(
                f"Partially loaded DIDF_Decoder: checkpoint decoder_block='{checkpoint_decoder_block}', "
                f"current decoder_block='{decoder_block}'."
            )
        if base_fusion_matches:
            load_state_if_present(BaseFuseLayer, checkpoint, 'BaseFuseLayer', required=False)
            skipped_resume_parts.append('optimizer3/scheduler3')
        else:
            skipped_resume_parts.append('BaseFuseLayer/optimizer3/scheduler3')
            print(
                f"Skipped BaseFuseLayer: checkpoint base_fusion='{checkpoint_base_fusion}', "
                f"current base_fusion='{args.base_fusion}'."
            )
        if detail_fusion_matches:
            load_state_if_present(DetailFuseLayer, checkpoint, 'DetailFuseLayer', required=False)
        else:
            skipped_resume_parts.append('DetailFuseLayer/optimizer4/scheduler4')
            print(
                f"Skipped DetailFuseLayer: checkpoint detail_fusion='{checkpoint_detail_fusion}', "
                f"current detail_fusion='{args.detail_fusion}'."
            )
        if encoder_global_feature_matches:
            load_optimizer_if_present(optimizer1, checkpoint, 'optimizer1')
            load_scheduler_if_present(scheduler1, checkpoint, 'scheduler1')
        if decoder_block_matches:
            load_optimizer_if_present(optimizer2, checkpoint, 'optimizer2')
            load_scheduler_if_present(scheduler2, checkpoint, 'scheduler2')
        if detail_fusion_matches:
            load_optimizer_if_present(optimizer4, checkpoint, 'optimizer4')
            load_scheduler_if_present(scheduler4, checkpoint, 'scheduler4')
        start_epoch = max(checkpoint_epoch, epoch_gap)
        skipped_message = ', '.join(skipped_resume_parts)
        print(
            f"Loaded Phase I pretrain from {resume_path}: "
            f"checkpoint encoder_global_feature='{checkpoint_encoder_global_feature}', "
            f"current encoder_global_feature='{encoder_global_feature}'; "
            f"checkpoint decoder_block='{checkpoint_decoder_block}', current decoder_block='{decoder_block}'; "
            f"checkpoint detail_fusion='{checkpoint_detail_fusion}', current detail_fusion='{args.detail_fusion}'; "
            f"checkpoint base_fusion='{checkpoint_base_fusion}', current base_fusion='{args.base_fusion}'. "
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
    num_batches = len(loader['train'])
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    optimizer4.zero_grad()
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        accum_group_start = (i // accumulation_steps) * accumulation_steps
        accum_group_size = min(accumulation_steps, num_batches - accum_group_start)
        should_step = ((i + 1) % accumulation_steps == 0) or (i + 1 == num_batches)

        if epoch < epoch_gap: #Phase I
            feature_V_G, feature_V_L, _ = DIDF_Encoder(data_VIS)
            feature_I_G, feature_I_L, _ = DIDF_Encoder(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_G, feature_V_L)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_G, feature_I_L)

            cc_loss_B = cc(feature_V_G, feature_I_G)
            cc_loss_D = cc(feature_V_L, feature_I_L)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            (loss / accum_group_size).backward()
            if should_step:
                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
        else:  #Phase II
            feature_V_G, feature_V_L, feature_V = DIDF_Encoder(data_VIS)
            feature_I_G, feature_I_L, feature_I = DIDF_Encoder(data_IR)
            feature_F_G = fuse_base_features(BaseFuseLayer, feature_I_G, feature_V_G)
            feature_F_L = fuse_detail_features(DetailFuseLayer, feature_I_L, feature_V_L)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_G, feature_F_L)

            
            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_G, feature_I_G)
            cc_loss_D = cc(feature_V_L, feature_I_L)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_ ,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss = fusionloss + coeff_decomp * loss_decomp
            (loss / accum_group_size).backward()
            if should_step:
                nn.utils.clip_grad_norm_(
                    DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                nn.utils.clip_grad_norm_(
                    DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                optimizer4.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer4.zero_grad()

        loss_value = loss.item()
        epoch_loss += loss_value

        # Determine approximate time left
        batches_done = epoch * num_batches + i
        batches_left = num_epochs * num_batches - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                num_batches,
                loss_value,
                time_left,
            )
        )

    avg_loss = epoch_loss / max(1, num_batches)
    print("\n[Epoch %d/%d] [avg_loss: %f]" % (epoch + 1, num_epochs, avg_loss))

    # adjust the learning rate

    scheduler1.step()  
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
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
