from __future__ import annotations

import cutlass
import pytest
import torch

from b12x._lib.utils import mxfp6_packed_k_bytes
from b12x.quantization.mxfp6 import (
    allocate_bf16_to_fp6_tma_outputs,
    compile_bf16_to_fp6_tma,
)
# Underscore alias: pytest tries to collect any module-level class named
# Test* (PytestCollectionWarning — TestKernel has an __init__).
from b12x.quantization.mxfp6.bf16_to_fp6_tma import TestKernel as _TestKernel


def test_bf16_to_fp6_tma_kernel_import() -> None:
    assert _TestKernel is not None
    k = _TestKernel()
    assert k.tile_shape_mnk == (128, 128, 128)
    assert k.threads_per_cta == 160


def test_allocate_bf16_to_fp6_tma_outputs_shapes() -> None:
    m, k = 128, 128
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=torch.device("cpu"))
    assert out.packed_a_storage.shape == (1, m, mxfp6_packed_k_bytes(k))
    assert out.packed_a_storage.dtype == torch.uint8
    rows_pad = 128
    cols_pad_sf = 4
    assert out.scale_storage.shape == (rows_pad * cols_pad_sf,)
    assert out.scale_storage.dtype == torch.uint8


def test_allocate_bf16_to_fp6_tma_outputs_bytes_shapes() -> None:
    m, k = 128, 256
    out = allocate_bf16_to_fp6_tma_outputs(m, k, device=torch.device("cpu"), emit="bytes")
    assert out.packed_a_storage.shape == (1, m, k)
    assert out.packed_a_storage.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile")
def test_compile_bf16_to_fp6_tma_returns_callable() -> None:
    # ``compile_bf16_to_fp6_tma`` queries the device (active clusters / SM count),
    # which requires an established primary CUDA context. When this test runs before
    # any CUDA tensor is created the query raises CUDA_ERROR_INVALID_CONTEXT, so touch
    # the device first to establish the context deterministically.
    torch.cuda.init()
    _ = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()
    launch = compile_bf16_to_fp6_tma(128, 128)
    assert callable(launch)
    launch2 = compile_bf16_to_fp6_tma(128, 128)
    assert launch2 is launch


def test_compile_bf16_to_fp6_tma_rejects_bad_k() -> None:
    with pytest.raises(AssertionError):
        compile_bf16_to_fp6_tma(128, 129)
