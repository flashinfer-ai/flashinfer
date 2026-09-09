"""CUDA Graph regression coverage for the SM120 M537 LoRA-down route."""

import unittest

import torch

from flashinfer.gemm.svdquant_sm120_cutlass import get_nvfp4_svdquant_sm120_module


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class TestSm120M537CudaGraph(unittest.TestCase):
    def test_cublaslt_lora_down_does_not_configure_l2_during_capture(self) -> None:
        """The persisting-L2 hint must not issue a forbidden capture API."""
        if torch.cuda.get_device_properties(0).multi_processor_count < 105:
            self.skipTest("requires the >=105-SM M537 admission")

        m, k, rank = 537, 5120, 32
        torch.manual_seed(20260816)
        x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        pre_quant_scale = torch.ones((k,), dtype=torch.bfloat16, device="cuda")
        global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        l2t_smoothed = torch.randn(
            (k, rank), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
        sf = torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        )
        down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
        workspace = torch.empty((32 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
        module = get_nvfp4_svdquant_sm120_module()

        def run() -> None:
            module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
                x,
                pre_quant_scale,
                global_scale,
                l2t_smoothed,
                xq,
                sf,
                down,
                workspace,
            )

        run()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()

        self.assertTrue(bool(torch.isfinite(down).all()))


if __name__ == "__main__":
    unittest.main()
