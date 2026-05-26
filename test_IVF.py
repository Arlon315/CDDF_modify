from baseUnet import BaseUNetDecoder, BaseUNetEncoder
from base_fusion import MultiScaleBaseFusion
from net import DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


FEATURE_VIS_CHANNELS = 8
FEATURE_GRID_COLS = 4
DEFAULT_CKPT_PATH = r"models/Unet/CDDFuse_05-26-11-42_epoch_040.pth"
DEFAULT_DATASETS = ["TNO", "RoadScene"]


def normalize_feature_channel(channel):
    channel = np.asarray(channel, dtype=np.float32)
    channel_min = np.min(channel)
    channel_max = np.max(channel)
    if channel_max - channel_min < 1e-8:
        return np.zeros(channel.shape, dtype=np.uint8)
    channel = (channel - channel_min) / (channel_max - channel_min)
    return np.uint8(np.round(channel * 255.0))


def make_feature_grid(channel_images, cols=FEATURE_GRID_COLS, pad=2):
    if len(channel_images) == 1:
        return channel_images[0]

    h, w = channel_images[0].shape
    rows = (len(channel_images) + cols - 1) // cols
    grid_h = rows * h + max(rows - 1, 0) * pad
    grid_w = cols * w + max(cols - 1, 0) * pad
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    for idx, image in enumerate(channel_images):
        row = idx // cols
        col = idx % cols
        y = row * (h + pad)
        x = col * (w + pad)
        grid[y:y + h, x:x + w] = image
    return grid


def save_feature_visualizations(feature_dict, img_name, save_root, max_channels=FEATURE_VIS_CHANNELS):
    sample_name = os.path.splitext(img_name)[0]
    sample_save_dir = os.path.join(save_root, sample_name)

    for feature_name, feature_tensor in feature_dict.items():
        feature = feature_tensor.detach().float().cpu()
        if feature.dim() == 4:
            feature = feature[0]
        elif feature.dim() == 2:
            feature = feature.unsqueeze(0)
        elif feature.dim() != 3:
            raise ValueError(f"Unsupported feature shape for {feature_name}: {tuple(feature.shape)}")

        channel_count = min(max_channels, feature.shape[0])
        channel_images = []
        for channel_idx in range(channel_count):
            channel_image = normalize_feature_channel(feature[channel_idx].numpy())
            channel_images.append(channel_image)
            img_save(channel_image, f"{feature_name}_ch{channel_idx:02d}", sample_save_dir)

        grid_image = make_feature_grid(channel_images)
        img_save(grid_image, f"{feature_name}_first{channel_count:02d}_grid", sample_save_dir)


def infer_encoder_detail_num_layers(checkpoint):
    encoder_state = checkpoint.get('DIDF_Encoder', {}) if isinstance(checkpoint, dict) else {}
    layer_indices = []
    for key in encoder_state.keys():
        key = key[7:] if isinstance(key, str) and key.startswith('module.') else key
        parts = str(key).split('.')
        if len(parts) > 2 and parts[0] == 'detailFeature' and parts[1] == 'net' and parts[2].isdigit():
            layer_indices.append(int(parts[2]))
    return max(layer_indices) + 1 if layer_indices else 3


def extract_base_detail_features(encoder, base_unet_encoder, base_unet_decoder, input_tensor):
    detail_feature, shared_feature = encoder(input_tensor)
    base_shallow, base_mid, base_deep = base_unet_encoder(shared_feature)
    base_feature = base_unet_decoder(base_shallow, base_mid, base_deep)
    return {
        "base_feature": base_feature,
        "base_shallow": base_shallow,
        "base_mid": base_mid,
        "base_deep": base_deep,
        "detail_feature": detail_feature,
        "shared_feature": shared_feature,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=DEFAULT_CKPT_PATH)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--img-name', default=None, help='Only infer one named image, for example 01.png.')
    parser.add_argument('--limit', type=int, default=None, help='Infer the first N images in each dataset.')
    parser.add_argument('--feature-channels', type=int, default=FEATURE_VIS_CHANNELS)
    parser.add_argument('--no-eval', action='store_true', help='Skip metric evaluation.')
    return parser.parse_args()


def get_image_names(test_folder, img_name=None, limit=None):
    image_names = sorted(os.listdir(os.path.join(test_folder, "ir")))
    if img_name is not None:
        if img_name not in image_names:
            raise FileNotFoundError(f"{img_name} was not found in {os.path.join(test_folder, 'ir')}")
        image_names = [img_name]
    if limit is not None:
        image_names = image_names[:limit]
    return image_names


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for dataset_name in args.datasets:
        print("\n"*2+"="*80)
        model_name="CDDFuse    "
        print("The test result of "+dataset_name+' :')
        test_folder=os.path.join('test_img',dataset_name)
        test_out_folder=os.path.join('test_result',dataset_name)
        feature_vis_folder=os.path.join('test_result','feature_vis',dataset_name)
        image_names = get_image_names(test_folder, args.img_name, args.limit)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        encoder_module = Restormer_Encoder()
        encoder_module.detailFeature = DetailFeatureExtraction(num_layers=infer_encoder_detail_num_layers(checkpoint))
        decoder_module = Restormer_Decoder()
        base_fusion_module = MultiScaleBaseFusion(dim=64, mid_dim=96, deep_dim=128)
        detail_fuse_module = DetailFeatureExtraction(num_layers=1)
        Encoder = nn.DataParallel(encoder_module).to(device)
        BaseUNet_Encoder = nn.DataParallel(BaseUNetEncoder()).to(device)
        BaseUNet_Decoder = nn.DataParallel(BaseUNetDecoder()).to(device)
        Decoder = nn.DataParallel(decoder_module).to(device)
        BaseFusionLayer = nn.DataParallel(base_fusion_module).to(device)
        DetailFuseLayer = nn.DataParallel(detail_fuse_module).to(device)

        Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
        if 'BaseUNetEncoder' not in checkpoint or 'BaseUNetDecoder' not in checkpoint:
            raise KeyError("Checkpoint is missing BaseUNetEncoder/BaseUNetDecoder state dicts for the new Base U-Net architecture.")
        if 'BaseFusionLayer' not in checkpoint:
            raise KeyError("Checkpoint is missing BaseFusionLayer state dict. Old BaseFuseLayer checkpoints are not compatible with MultiScaleBaseFusion.")
        BaseUNet_Encoder.load_state_dict(checkpoint['BaseUNetEncoder'])
        BaseUNet_Decoder.load_state_dict(checkpoint['BaseUNetDecoder'])
        Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
        BaseFusionLayer.load_state_dict(checkpoint['BaseFusionLayer'])
        DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
        Encoder.eval()
        BaseUNet_Encoder.eval()
        BaseUNet_Decoder.eval()
        Decoder.eval()
        BaseFusionLayer.eval()
        DetailFuseLayer.eval()

        with torch.no_grad():
            for img_name in image_names:

                data_IR=image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

                feat_V = extract_base_detail_features(
                    Encoder, BaseUNet_Encoder, BaseUNet_Decoder, data_VIS)
                feat_I = extract_base_detail_features(
                    Encoder, BaseUNet_Encoder, BaseUNet_Decoder, data_IR)
                F_B_s, F_B_m, F_B_d = BaseFusionLayer(
                    feat_I["base_shallow"],
                    feat_I["base_mid"],
                    feat_I["base_deep"],
                    feat_V["base_shallow"],
                    feat_V["base_mid"],
                    feat_V["base_deep"],
                )
                feature_F_B = BaseUNet_Decoder(F_B_s, F_B_m, F_B_d)
                feature_F_D = DetailFuseLayer(feat_I["detail_feature"] + feat_V["detail_feature"])
                data_Fuse, out_enc_level0 = Decoder(data_VIS, feature_F_B, feature_F_D)
                # data_Fuse, _ = Decoder(None, feature_F_B, feature_F_D)
                save_feature_visualizations({
                    "feature_V_D": feat_V["detail_feature"],
                    "feature_I_D": feat_I["detail_feature"],
                    "feature_V_B": feat_V["base_feature"],
                    "feature_I_B": feat_I["base_feature"],
                    "out_enc_level0": out_enc_level0,
                }, img_name, feature_vis_folder, max_channels=args.feature_channels)
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                fi = np.uint8(np.round(np.squeeze((data_Fuse * 255).cpu().numpy())))
                img_save(fi, os.path.splitext(img_name)[0], test_out_folder)

        if args.no_eval:
            print(f"Saved fusion and feature visualizations for {len(image_names)} image(s).")
            print("="*80)
            continue

        from utils.Evaluator import Evaluator

        eval_folder=test_out_folder
        ori_img_folder=test_folder

        metric_result = np.zeros((8))
        for img_name in image_names:
                ir = image_read_cv2(os.path.join(ori_img_folder,"ir", img_name), 'GRAY')
                vi = image_read_cv2(os.path.join(ori_img_folder,"vi", img_name), 'GRAY')
                fi = image_read_cv2(os.path.join(eval_folder, os.path.splitext(img_name)[0]+".png"), 'GRAY')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                            , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                            , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                            , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

        metric_result /= len(image_names)
        print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
        print(model_name+'\t'+str(np.round(metric_result[0], 2))+'\t'
                +str(np.round(metric_result[1], 2))+'\t'
                +str(np.round(metric_result[2], 2))+'\t'
                +str(np.round(metric_result[3], 2))+'\t'
                +str(np.round(metric_result[4], 2))+'\t'
                +str(np.round(metric_result[5], 2))+'\t'
                +str(np.round(metric_result[6], 2))+'\t'
                +str(np.round(metric_result[7], 2))
                )
        print("="*80)


if __name__ == '__main__':
    main()
