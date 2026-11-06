import os

# Disable CUDA memory caching so out-of-bounds writes surface as immediate errors
# instead of silently corrupting adjacent cached allocations.
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import pytest
import torch
from tests.test_helpers.utils_fp4 import pack_fp4_values, ref_fp4_quant

from flashinfer import fp4_quantize
from flashinfer.utils import (
    is_sm100a_supported,
    is_sm110a_supported,
    is_sm12x_supported,
)

DTYPES = [torch.float16, torch.bfloat16]
UNALIGNED_M_SHAPES = [
    (17, 512),
    (33, 1024),
    (1025, 1024),
    (1025, 6144),
]
BACKENDS = ["cuda", "cute-dsl"]
SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

BLOCK_SIZE = 16


def _is_fp4_supported(device: torch.device) -> bool:
    return (
        is_sm100a_supported(device)
        or is_sm110a_supported(device)
        or is_sm12x_supported(device)
    )


def _is_cute_dsl_available() -> bool:
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            return False
        import torch as _torch

        if _torch.cuda.is_available():
            from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported

            return is_cute_dsl_arch_supported(*_torch.cuda.get_device_capability(0))
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", UNALIGNED_M_SHAPES)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_fp4_quantize_unaligned_m_non_swizzled(
    dtype: torch.dtype,
    shape: tuple[int, int],
    backend: str,
    seed: int,
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: fp4_quantize with M not a multiple of 16 for linear SF."""
    if not _is_fp4_supported(torch.device(device)):
        pytest.skip("Nvfp4 Requires compute capability >= 10 and CUDA >= 12.8")
    if backend == "cute-dsl" and not _is_cute_dsl_available():
        pytest.skip("CuTe-DSL not available")
    monkeypatch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "0")
    monkeypatch.setenv("TRTLLM_DISABLE_FP4_QUANT_FAST_MATH", "0")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
    torch.set_default_device(device)
    torch.manual_seed(seed)

    m, n = shape
    sf_vec_size = BLOCK_SIZE
    assert n % sf_vec_size == 0, f"cols needs to be {sf_vec_size} divisible"

    x = torch.randn((m, n), dtype=dtype)
    tensor_amax = torch.abs(x).max().to(torch.float32)
    global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / tensor_amax

    out_val, out_sf = fp4_quantize(
        x, global_scale, sf_vec_size, False, False, backend=backend
    )

    assert out_val.shape == (m, n // 2), (
        f"Expected val shape {(m, n // 2)}, got {out_val.shape}"
    )
    expected_sf_size = m * n // sf_vec_size
    assert out_sf.numel() == expected_sf_size, (
        f"Expected sf numel {expected_sf_size}, got {out_sf.numel()}"
    )

    out_ref, scale_ref = ref_fp4_quant(x, global_scale, sf_vec_size)
    torch.testing.assert_close(out_val, pack_fp4_values(out_ref), rtol=0, atol=0)
    torch.testing.assert_close(
        out_sf, scale_ref.to(torch.float8_e4m3fn).view(torch.uint8), rtol=0, atol=0
    )
