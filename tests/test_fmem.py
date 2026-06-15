import unittest

import torch
import torch.nn as nn

from FMEM import (
    FUSION_ENHANCE_CROSS_HISTOGRAM,
    FUSION_ENHANCE_MAMBA,
    CrossHistogramAttention,
    FusionHistogramEnhanceModule,
    build_fusion_enhance_module,
    infer_fusion_enhance_type,
)


class FusionHistogramEnhanceModuleTest(unittest.TestCase):
    def test_forward_backward_regular_and_odd_shapes(self):
        for shape in ((2, 64, 32, 32), (2, 64, 31, 29)):
            module = FusionHistogramEnhanceModule(
                dim=64,
                num_heads=4,
                ffn_expansion_factor=2.5,
            )
            detail = torch.randn(shape, requires_grad=True)
            base = torch.randn(shape, requires_grad=True)

            output = module(detail, base)
            self.assertEqual(output.shape, detail.shape)
            self.assertTrue(torch.isfinite(output).all())

            output.mean().backward()
            self.assertTrue(torch.isfinite(detail.grad).all())
            self.assertTrue(torch.isfinite(base.grad).all())
            for parameter in module.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_restore_uses_target_value_indices(self):
        attention = CrossHistogramAttention(dim=4, num_heads=2)
        attention.project_out = nn.Identity()

        bhr = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        fhr = torch.ones_like(bhr)
        idx_v = torch.tensor(
            [[
                [5, 4, 3, 2, 1, 0],
                [1, 3, 5, 0, 2, 4],
                [2, 0, 4, 1, 5, 3],
                [0, 2, 1, 5, 3, 4],
            ]]
        )
        idx_h = torch.tensor(
            [[
                [[0, 0, 0], [1, 1, 1]],
                [[0, 0, 0], [1, 1, 1]],
            ]]
        )
        idx_w = torch.tensor(
            [[
                [[0, 1, 2], [0, 1, 2]],
                [[0, 1, 2], [0, 1, 2]],
            ]]
        )
        target = {
            'idx_v': idx_v,
            'idx_h': idx_h,
            'idx_w': idx_w,
            'shape': (1, 4, 2, 3),
        }

        restored = attention._restore_target(bhr, fhr, target)
        expected = torch.zeros_like(bhr).scatter(2, idx_v, bhr).reshape(1, 4, 2, 3)
        self.assertTrue(torch.equal(restored, expected))

    def test_checkpoint_type_inference_and_cross_histogram_builder(self):
        module = build_fusion_enhance_module(
            fusion_enhance_type=FUSION_ENHANCE_CROSS_HISTOGRAM,
            dim=64,
        )
        checkpoint = {
            'fusion_enhance_type': FUSION_ENHANCE_CROSS_HISTOGRAM,
            'FMEMLayer': module.state_dict(),
        }
        self.assertEqual(
            infer_fusion_enhance_type(checkpoint),
            FUSION_ENHANCE_CROSS_HISTOGRAM,
        )
        self.assertIsInstance(
            build_fusion_enhance_module(checkpoint=checkpoint, dim=64),
            FusionHistogramEnhanceModule,
        )

        checkpoint_without_metadata = {'FMEMLayer': module.state_dict()}
        self.assertEqual(
            infer_fusion_enhance_type(checkpoint_without_metadata),
            FUSION_ENHANCE_CROSS_HISTOGRAM,
        )

        old_mamba_checkpoint = {
            'FMEMLayer': {
                'module.global_mixer.mambas.0.in_proj.weight': torch.empty(1),
            }
        }
        self.assertEqual(
            infer_fusion_enhance_type(old_mamba_checkpoint),
            FUSION_ENHANCE_MAMBA,
        )


if __name__ == '__main__':
    unittest.main()
