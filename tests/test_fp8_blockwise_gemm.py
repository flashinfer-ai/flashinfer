"""
Based on https://github.com/sgl-project/sglang/blob/main/sgl-kernel/tests/test_fp8_blockwise_gemm.py
Copyright (c) 2025 SGLang Project (Apache 2.0 License)
"""

from typing import Optional, Type

import pytest
import torch

from flashinfer.gemm import gemm_fp8_blockwise_trtllm


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    return tuple(cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def baseline_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: Type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match the
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)
    output = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output


def _test_accuracy(M, N, K):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = "cuda"

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    assert N % 128 == 0
    assert K % 128 == 0

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    a_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    a_fp8 = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    b_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2 * fp8_max
    b_fp8 = b_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn).t()
    scale_a = (
        torch.randn(
            scale_shape(a_fp8.shape, (1, 128)), device=device, dtype=torch.float32
        )
        * 0.001
    )
    scale_b = (
        torch.randn(
            scale_shape(b_fp8.shape, (128, 128)), device=device, dtype=torch.float32
        )
        * 0.001
    )
    scale_a = scale_a.t().contiguous().t()
    scale_b = scale_b.t().contiguous().t()
    torch.testing.assert_close(
        baseline_scaled_mm(a_fp8, b_fp8, scale_a, scale_b, out_dtype=torch.bfloat16),
        gemm_fp8_blockwise_trtllm(a_fp8, b_fp8, scale_a, scale_b),
        rtol=0.02,
        atol=1,
    )


@pytest.mark.parametrize("M", [1, 5, 127, 128, 512, 4096])
@pytest.mark.parametrize("N", [128, 8192, 14080, 32768])
@pytest.mark.parametrize("K", [512, 14080, 16384])
def test_accuracy(M, N, K):
    _test_accuracy(M, N, K)


if __name__ == "__main__":
    pytest.main([__file__])
