from baseUnet import BaseUNetDecoder, BaseUNetEncoder
from net import BaseFeatureExtraction, DetailFeatureExtraction, Restormer_Decoder, Restormer_Encoder
import os
import numpy as np
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CDDFuse_path=r"models/CDDFuse_IVF.pth"
CDDFuse_MIF_path=r"models/CDDFuse_MIF.pth"


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
    return base_feature, detail_feature, shared_feature


for dataset_name in ["MRI_CT","MRI_PET","MRI_SPECT"]: 
    print("\n"*2+"="*80)
    print("The test result of "+dataset_name+" :")
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    for ckpt_path in [CDDFuse_path,CDDFuse_MIF_path]: 
        model_name=ckpt_path.split('/')[-1].split('.')[0]
        test_folder=os.path.join('test_img',dataset_name) 
        test_out_folder=os.path.join('test_result',dataset_name)

        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder_module = Restormer_Encoder()
        encoder_module.detailFeature = DetailFeatureExtraction(num_layers=infer_encoder_detail_num_layers(checkpoint))
        decoder_module = Restormer_Decoder()
        base_fuse_module = BaseFeatureExtraction(dim=64, num_heads=8)
        detail_fuse_module = DetailFeatureExtraction(num_layers=1)
        Encoder = nn.DataParallel(encoder_module).to(device)
        BaseUNet_Encoder = nn.DataParallel(BaseUNetEncoder()).to(device)
        BaseUNet_Decoder = nn.DataParallel(BaseUNetDecoder()).to(device)
        Decoder = nn.DataParallel(decoder_module).to(device)
        BaseFuseLayer = nn.DataParallel(base_fuse_module).to(device)
        DetailFuseLayer = nn.DataParallel(detail_fuse_module).to(device)

        Encoder.load_state_dict(checkpoint['DIDF_Encoder'])
        if 'BaseUNetEncoder' not in checkpoint or 'BaseUNetDecoder' not in checkpoint:
            raise KeyError("Checkpoint is missing BaseUNetEncoder/BaseUNetDecoder state dicts for the new Base U-Net architecture.")
        BaseUNet_Encoder.load_state_dict(checkpoint['BaseUNetEncoder'])
        BaseUNet_Decoder.load_state_dict(checkpoint['BaseUNetDecoder'])
        Decoder.load_state_dict(checkpoint['DIDF_Decoder'])
        BaseFuseLayer.load_state_dict(checkpoint['BaseFuseLayer'])
        DetailFuseLayer.load_state_dict(checkpoint['DetailFuseLayer'])
        Encoder.eval()
        BaseUNet_Encoder.eval()
        BaseUNet_Decoder.eval()
        Decoder.eval()
        BaseFuseLayer.eval()
        DetailFuseLayer.eval()

        with torch.no_grad():
            for img_name in os.listdir(os.path.join(test_folder,dataset_name.split('_')[0])):
                data_IR=image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[1],img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
                data_VIS = image_read_cv2(os.path.join(test_folder,dataset_name.split('_')[0],img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0

                data_IR,data_VIS = torch.FloatTensor(data_IR),torch.FloatTensor(data_VIS)
                data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

                feature_V_B, feature_V_D, feature_V = extract_base_detail_features(
                    Encoder, BaseUNet_Encoder, BaseUNet_Decoder, data_VIS)
                feature_I_B, feature_I_D, feature_I = extract_base_detail_features(
                    Encoder, BaseUNet_Encoder, BaseUNet_Decoder, data_IR)
                feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
                feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)
                if ckpt_path==CDDFuse_path:
                    data_Fuse, _ = Decoder(data_IR+data_VIS, feature_F_B, feature_F_D)
                else:
                    data_Fuse, _ = Decoder(None, feature_F_B, feature_F_D)
                data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
                fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                img_save(fi, img_name.split(sep='.')[0], test_out_folder)
        eval_folder=test_out_folder  
        ori_img_folder=test_folder

        metric_result = np.zeros((8))
        for img_name in os.listdir(os.path.join(ori_img_folder,dataset_name.split('_')[0])):
                ir = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[1], img_name), 'GRAY')
                vi = image_read_cv2(os.path.join(ori_img_folder,dataset_name.split('_')[0], img_name), 'GRAY')
                fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0]+".png"), 'GRAY')
                metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                            , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                            , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                            , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

        metric_result /= len(os.listdir(eval_folder))
        
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
