"""Compare the opt-in scheduling policy with SGLang's existing FP8 GEMM."""

import pathlib
import sys
import unittest

import torch

SOURCE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "flashinfer/experimental/mla_fp8_sm120"
)
sys.path.insert(0, str(SOURCE))

from projection_tuning import policy_mm
from sglang.srt.layers.quantization import fp8_utils


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestProjectionPolicy(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")

    def test_real_projection_shapes_match_existing_gemm(self):
        torch.manual_seed(326)
        for m, n, k in ((1024, 8960, 512), (5093, 8960, 512), (5093, 2048, 1536)):
            with self.subTest(m=m, n=n, k=k):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                a, sa = fp8_utils.sglang_per_token_group_quant_fp8_row_padded(x, 128)
                w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
                # Match the original checkpoint's scale-transpose view.
                sb = (torch.rand(n // 128, k // 128, device="cuda") + 0.1).T
                expected = fp8_utils.fp8_blockwise_scaled_mm(
                    a, w.T, sa, sb, out_dtype=torch.bfloat16
                )
                actual = policy_mm(a, w.T, sa, sb, out_dtype=torch.bfloat16)
                self.assertTrue(torch.isfinite(actual).all())
                relative_l2 = (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm()
                self.assertLess(float(relative_l2), 0.0001)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)


if __name__ == "__main__":
    unittest.main()
