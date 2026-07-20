import argparse
import os
from pathlib import Path

import numpy as np
import torch

from network.model_loader import build_current_glcm_model
from utils.Evaluator import Evaluator
from utils.img_read_save import image_read_cv2, img_save


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the current GLoC-Mamba checkpoint on medical fusion datasets.')
    parser.add_argument('--ckpt-path', required=True, help='Path to a compatible GLoC-Mamba checkpoint.')
    parser.add_argument('--datasets', nargs='+', default=['MRI_CT', 'MRI_PET', 'MRI_SPECT'])
    parser.add_argument('--limit', type=int, default=None, help='Evaluate only the first N image pairs per dataset.')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model = build_current_glcm_model(checkpoint, device, data_parallel=True)
    encoder = model['encoder']
    decoder = model['decoder']
    modal_enhance = model['modal_enhance']
    cross_mamba_fusion = model['cross_mamba_fusion']

    for dataset_name in args.datasets:
        visible_modality, infrared_modality = dataset_name.split('_', maxsplit=1)
        test_folder = os.path.join('test_img', dataset_name)
        visible_folder = os.path.join(test_folder, visible_modality)
        infrared_folder = os.path.join(test_folder, infrared_modality)
        output_folder = os.path.join('test_result', dataset_name)
        image_names = sorted(os.listdir(visible_folder))
        if args.limit is not None:
            image_names = image_names[:args.limit]
        if not image_names:
            raise ValueError(f'No images found for {dataset_name}.')

        with torch.no_grad():
            for image_name in image_names:
                data_ir = image_read_cv2(os.path.join(infrared_folder, image_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
                data_vis = image_read_cv2(os.path.join(visible_folder, image_name), mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
                data_ir = torch.FloatTensor(data_ir).to(device)
                data_vis = torch.FloatTensor(data_vis).to(device)

                feature_v_g, feature_v_l, _ = encoder(data_vis)
                feature_i_g, feature_i_l, _ = encoder(data_ir)
                feature_i_e, feature_v_e = modal_enhance(
                    feature_i_g, feature_i_l, feature_v_g, feature_v_l)
                feature_f_e = cross_mamba_fusion(feature_i_e, feature_v_e)
                fused, _ = decoder(feature_f_e)
                fused = (fused - fused.min()) / (fused.max() - fused.min())
                img_save(np.uint8(np.round(np.squeeze((fused * 255).cpu().numpy()))), Path(image_name).stem, output_folder)

        metric_result = np.zeros(8)
        for image_name in image_names:
            ir = image_read_cv2(os.path.join(infrared_folder, image_name), 'GRAY')
            vis = image_read_cv2(os.path.join(visible_folder, image_name), 'GRAY')
            fused = image_read_cv2(os.path.join(output_folder, Path(image_name).stem + '.png'), 'GRAY')
            metric_result += np.array([
                Evaluator.EN(fused), Evaluator.SD(fused), Evaluator.SF(fused),
                Evaluator.MI(fused, ir, vis), Evaluator.SCD(fused, ir, vis),
                Evaluator.VIFF(fused, ir, vis), Evaluator.Qabf(fused, ir, vis),
                Evaluator.SSIM(fused, ir, vis),
            ])
        metric_result /= len(image_names)
        print(dataset_name + '\t' + '\t'.join(str(np.round(value, 2)) for value in metric_result))


if __name__ == '__main__':
    main()
