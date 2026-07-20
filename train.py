# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from network.net import (
    CURRENT_GLCM_CONFIG,
    ENCODER_GLOBAL_LOCAL_SEMANTICS,
    build_current_glcm_modules,
    require_current_glcm_checkpoint,
)
from network.GLCM_Mamba import GlobalLocalCrossModalMambaBlock
from network.CMFB import CommenMambaFusionBlock
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
from utils.loss import Fusionloss


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

parser = argparse.ArgumentParser(
    description="Train the fixed GLoC-Mamba architecture used by the supported checkpoint."
)
parser.add_argument("--resume", type=str, default="", help="Path to a compatible GLoC-Mamba checkpoint.")
parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", type=str, default="models/GLoC-Mamba/", help="Directory for saved checkpoints.")
parser.add_argument("--save-interval", "--save_interval", dest="save_interval", type=int, default=10, help="Save a checkpoint every N epochs.")
args = parser.parse_args()
model_str = 'GLoC-Mamba'

# . Set the hyper-parameters for training
num_epochs = 120
lr = 1e-4
weight_decay = 0
batch_size = 8
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_module, decoder_module = build_current_glcm_modules()
DIDF_Encoder = nn.DataParallel(encoder_module).to(device)
DIDF_Decoder = nn.DataParallel(decoder_module).to(device)
ModalEnhanceLayer = nn.DataParallel(GlobalLocalCrossModalMambaBlock(dim=64)).to(device)
CrossMambaFusionLayer = nn.DataParallel(
    CommenMambaFusionBlock(dim=64)
).to(device)

optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(ModalEnhanceLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(CrossMambaFusionLayer.parameters(), lr=lr, weight_decay=weight_decay)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
criteria_fusion = Fusionloss()

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


def build_checkpoint(epoch):
    return {
        'epoch': epoch,
        'seed': seed,
        'timestamp': timestamp,
        'encoder_feature_semantics': ENCODER_GLOBAL_LOCAL_SEMANTICS,
        **CURRENT_GLCM_CONFIG,
        'skip_phase1': True,
        'use_decomp_loss': False,
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
    require_current_glcm_checkpoint(checkpoint)
    load_state_if_present(DIDF_Encoder, checkpoint, 'DIDF_Encoder', strict=True)
    load_state_if_present(DIDF_Decoder, checkpoint, 'DIDF_Decoder', strict=True)
    load_state_if_present(ModalEnhanceLayer, checkpoint, 'ModalEnhanceLayer', strict=True)
    load_state_if_present(CrossMambaFusionLayer, checkpoint, 'CrossMambaFusionLayer', strict=True)
    load_optimizer_if_present(optimizer1, checkpoint, 'optimizer1')
    load_optimizer_if_present(optimizer2, checkpoint, 'optimizer2')
    load_optimizer_if_present(optimizer3, checkpoint, 'optimizer3')
    load_optimizer_if_present(optimizer4, checkpoint, 'optimizer4')
    load_scheduler_if_present(scheduler1, checkpoint, 'scheduler1')
    load_scheduler_if_present(scheduler2, checkpoint, 'scheduler2')
    load_scheduler_if_present(scheduler3, checkpoint, 'scheduler3')
    load_scheduler_if_present(scheduler4, checkpoint, 'scheduler4')
    start_epoch = int(checkpoint.get('epoch', 0))
    print(f'Resumed checkpoint from {resume_path} at epoch {start_epoch}.')

'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

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

        feature_V_G, feature_V_L, _ = DIDF_Encoder(data_VIS)
        feature_I_G, feature_I_L, _ = DIDF_Encoder(data_IR)
        feature_I_E, feature_V_E = ModalEnhanceLayer(
            feature_I_G, feature_I_L, feature_V_G, feature_V_L)
        feature_F_E = CrossMambaFusionLayer(feature_I_E, feature_V_E)
        data_Fuse, _ = DIDF_Decoder(feature_F_E)

        fusionloss, _, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
        loss = fusionloss
        loss.backward()
        nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(ModalEnhanceLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(CrossMambaFusionLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
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
    scheduler4.step()

    finished_epoch = epoch + 1
    if args.save_interval > 0 and finished_epoch % args.save_interval == 0:
        save_checkpoint(finished_epoch)
    
save_checkpoint(num_epochs, save_tag="final")
