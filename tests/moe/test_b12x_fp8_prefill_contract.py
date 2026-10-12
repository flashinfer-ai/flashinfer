"""Host configuration and masked-route contracts for opt-in FP8 activations."""

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
    W4A16GemmKernel,
    _w4a16_topk_sum_launch_flat,
)


def test_disabled_fp8_preserves_four_stage_pipeline(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kernel = W4A16GemmKernel(
        size_m=1024,
        size_n=2048,
        size_k=4096,
        num_experts=288,
        top_k=8,
        mul_topk_weights=False,
        tile_n=128,
        tile_k=64,
        moe_block_size=32,
        max_m_blocks=512,
        element_dtype="bf16",
        scale_format="e4m3_k32",
    )
    assert kernel.stages == 4
    assert kernel.tune["pair"] == 0
    assert all(value == 0 for key, value in kernel.tune.items() if key != "stages")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vector_sum_never_reads_unrouted_scratch():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    values = torch.full((3, 4, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor(
        [[0, -1, 1, -1], [-1, -1, -1, -1], [2, 0, -1, 1]],
        dtype=torch.int32,
        device="cuda",
    )
    values[ids >= 0] = 1
    expected = (ids >= 0).sum(-1)[:, None].expand(3, 128).bfloat16()
    for vec in (1, 8):
        output = torch.empty((3, 128), dtype=torch.bfloat16, device="cuda")
        _w4a16_topk_sum_launch_flat(
            values,
            ids,
            output,
            3,
            4,
            128,
            "bf16",
            int(torch.cuda.current_stream().cuda_stream),
            vec=vec,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
