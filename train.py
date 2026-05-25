# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''

from baseUnet import BaseUNetDecoder, BaseUNetEncoder
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
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


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

parser = argparse.ArgumentParser(description="Train CDDFuse with checkpoint resume support.")
parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory for saved checkpoints.")
parser.add_argument("--save_interval", type=int, default=10, help="Save a checkpoint every N epochs.")
args = parser.parse_args()

# . Set the hyper-parameters for training
num_epochs = 120 # total epoch
epoch_gap = 40  # epoches of Phase I 

lr = 1e-4
weight_decay = 0
batch_size = 16
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
DIDF_Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseUNet_Encoder = nn.DataParallel(BaseUNetEncoder()).to(device)
BaseUNet_Decoder = nn.DataParallel(BaseUNetDecoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)

# optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(
    list(DIDF_Encoder.parameters()) +
    list(BaseUNet_Encoder.parameters()) +
    list(BaseUNet_Decoder.parameters()),
    lr=lr,
    weight_decay=weight_decay,
)
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
Loss_ssim = kornia.losses.SSIM(11, reduction='mean')


# data loader
trainloader = DataLoader(H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
start_epoch = 0

os.makedirs(args.checkpoint_dir, exist_ok=True)


def build_checkpoint(epoch):
    return {
        'epoch': epoch,
        'timestamp': timestamp,
        'DIDF_Encoder': DIDF_Encoder.state_dict(),
        'BaseUNetEncoder': BaseUNet_Encoder.state_dict(),
        'BaseUNetDecoder': BaseUNet_Decoder.state_dict(),
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


if args.resume:
    resume_path = os.path.expanduser(args.resume)
    checkpoint = torch.load(resume_path, map_location=device)
    DIDF_Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    if 'BaseUNetEncoder' not in checkpoint or 'BaseUNetDecoder' not in checkpoint:
        raise KeyError("Checkpoint is missing BaseUNetEncoder/BaseUNetDecoder state dicts for the new Base U-Net architecture.")
    BaseUNet_Encoder.load_state_dict(checkpoint['BaseUNetEncoder'])
    BaseUNet_Decoder.load_state_dict(checkpoint['BaseUNetDecoder'])
    DIDF_Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
    if 'optimizer1' in checkpoint:
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        optimizer3.load_state_dict(checkpoint['optimizer3'])
        optimizer4.load_state_dict(checkpoint['optimizer4'])
    if 'scheduler1' in checkpoint:
        scheduler1.load_state_dict(checkpoint['scheduler1'])
        scheduler2.load_state_dict(checkpoint['scheduler2'])
        scheduler3.load_state_dict(checkpoint['scheduler3'])
        scheduler4.load_state_dict(checkpoint['scheduler4'])
    start_epoch = int(checkpoint.get('epoch', 0))
    print(f"Resumed from {resume_path} at epoch {start_epoch}.")


def extract_base_detail_features(input_tensor):
    detail_feature, shared_feature = DIDF_Encoder(input_tensor)
    base_shallow, base_mid, base_deep = BaseUNet_Encoder(shared_feature)
    base_feature = BaseUNet_Decoder(base_shallow, base_mid, base_deep)
    return base_feature, detail_feature, shared_feature

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
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseUNet_Encoder.train()
        BaseUNet_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseUNet_Encoder.zero_grad()
        BaseUNet_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap: #Phase I
            feature_V_B, feature_V_D, _ = extract_base_detail_features(data_VIS)
            feature_I_B, feature_I_D, _ = extract_base_detail_features(data_IR)
            data_VIS_hat, _ = DIDF_Decoder(data_VIS, feature_V_B, feature_V_D)
            data_IR_hat, _ = DIDF_Decoder(data_IR, feature_I_B, feature_I_D)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss_decomp =  (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * \
                   mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseUNet_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseUNet_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()  
            optimizer2.step()
        else:  #Phase II
            feature_V_B, feature_V_D, feature_V = extract_base_detail_features(data_VIS)
            feature_I_B, feature_I_D, feature_I = extract_base_detail_features(data_IR)
            feature_F_B = BaseFuseLayer(feature_I_B+feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D+feature_V_D)
            data_Fuse, feature_F = DIDF_Decoder(data_VIS, feature_F_B, feature_F_D)  

            
            mse_loss_V = 5*Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5*Loss_ssim(data_IR,  data_Fuse) + MSELoss(data_IR,  data_Fuse)

            cc_loss_B = cc(feature_V_B, feature_I_B)
            cc_loss_D = cc(feature_V_D, feature_I_D)
            loss_decomp =   (cc_loss_D) ** 2 / (1.01 + cc_loss_B)  
            fusionloss, _,_  = criteria_fusion(data_VIS, data_IR, data_Fuse)
            
            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseUNet_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseUNet_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
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
                loss.item(),
                time_left,
            )
        )

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
