"""Forced SM103 CuTe DSL epilogue correctness and eligibility checks."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")

from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm103 import (
    Sm103BlockScaledPersistentDenseGemmKernel as Sm103Kernel,
)
from flashinfer.quantization.fp4_quantization import (
    _e2m1_and_ufp8sf_scale_to_float_cpu,
)
from flashinfer.utils import get_compute_capability


# Each configuration is (tile_MN, cluster_MN, use_tma_store, swap_ab).
_CONFIGURATIONS = [
    pytest.param(((128, 256), (1, 1), False, False), id="overlap-1cta"),
    pytest.param(((256, 256), (2, 1), False, False), id="overlap-2cta"),
    pytest.param(((128, 128), (1, 1), False, False), id="direct-256bit"),
    pytest.param(((128, 256), (1, 1), True, False), id="tma-store"),
    pytest.param(((128, 128), (1, 1), False, True), id="swap-ab"),
]
_OUTPUT_DTYPES = [
    pytest.param(torch.bfloat16, cutlass.BFloat16, id="bf16"),
    pytest.param(torch.float16, cutlass.Float16, id="fp16"),
]


def _require_supported_gpu():
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in ((10, 3), (10, 7)):
        pytest.skip("Requires SM103 or SM107 for the SM103 CuTe DSL kernel")


def _make_inputs(m, n, k):
    """Make packed FP4 operands and padded scale storage on the CPU."""
    generator = torch.Generator().manual_seed(103)
    packed, scales = [], []
    # E4M3 encodings of 0.5, 1 and 2: nonuniform, exactly representable scales.
    scale_values = torch.tensor([0x30, 0x38, 0x40], dtype=torch.uint8)
    for rows in (m, n):
        packed.append(
            torch.randint(256, (rows, k // 2), dtype=torch.uint8, generator=generator)
        )
        padded_rows = (rows + 127) // 128 * 128
        scales.append(
            scale_values[torch.randint(3, (padded_rows, k // 16), generator=generator)]
        )
    alpha = torch.tensor([0.125], dtype=torch.float32)
    return packed[0], packed[1], scales[0], scales[1], alpha


def _reference(a, b, sfa, sfb, alpha, dtype):
    def decode(packed, scales):
        rows, k = packed.shape[0], packed.shape[1] * 2
        # Undo the 128x4 scale layout, then drop the physical row padding.
        linear_scales = (
            scales.reshape(-1, k // 64, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(-1, k // 16)[:rows]
            .contiguous()
        )
        return _e2m1_and_ufp8sf_scale_to_float_cpu(
            packed, linear_scales, None, 16, 1, False
        )

    return ((decode(a, sfa) @ decode(b, sfb).T) * alpha).to(dtype)


def _can_implement(shape, c_dtype, config):
    tile, cluster, use_tma_store, swap_ab = config
    m, n, k = shape
    if swap_ab:
        m, n = n, m
    return Sm103Kernel.can_implement(
        cutlass.Float4E2M1FN,
        cutlass.Float8E4M3FN,
        16,
        c_dtype,
        tile,
        cluster,
        m,
        n,
        k,
        1,
        "k",
        "k",
        "m" if swap_ab else "n",
        use_tma_store,
    )


def _compile_and_run(inputs, out, c_dtype, config):
    tile, cluster, use_tma_store, swap_ab = config
    a, b, sfa, sfb, alpha = inputs
    if swap_ab:
        a, b, sfa, sfb = b, a, sfb, sfa
    sf_m, sf_n = (a.shape[0] + 127) // 128, (b.shape[0] + 127) // 128
    sf_k = (a.shape[1] * 2 // 16 + 3) // 4
    # A fresh cache forces each case through the production compile helper.
    compiled, _ = _compile_block_scaled_gemm(
        cache={},
        cache_key=(config, out.dtype),
        make_gemm_kernel=lambda: Sm103Kernel(
            16, tile, cluster, use_tma_store, enable_pdl=False
        ),
        ab_cutlass_dtype=cutlass.Uint8,
        sf_dtype=cutlass.Float8E4M3FN,
        c_cutlass_dtype=c_dtype,
        ab_assumed_align=32,
        cluster_shape_mn=cluster,
        swap_ab=swap_ab,
        sf_m=sf_m,
        sf_n=sf_n,
        sf_k=sf_k,
        batch_size=1,
    )
    # Match forward(): FFI expects a column-major view for swap_ab, while the
    # wrapper reconstructs the actual output layout from the storage pointer.
    launch_out = out.as_strided(out.shape, (1, out.shape[0])) if swap_ab else out
    compiled(a, b, launch_out, sf_m, sf_n, sf_k, sfa.data_ptr(), sfb.data_ptr(), alpha)


@pytest.mark.parametrize("config", _CONFIGURATIONS)
@pytest.mark.parametrize(
    "shape", [(2048, 4096, 1024), (200, 512, 768)], ids=["multi-tile", "partial-m"]
)
@pytest.mark.parametrize("dtype,c_dtype", _OUTPUT_DTYPES)
def test_sm103_epilogue(config, shape, dtype, c_dtype):
    _require_supported_gpu()
    assert _can_implement(shape, c_dtype, config), (shape, config)
    inputs = _make_inputs(*shape)
    expected = _reference(*inputs, dtype)
    device_inputs = tuple(tensor.cuda() for tensor in inputs)
    out = torch.full(shape[:2], float("nan"), dtype=dtype, device="cuda")
    _compile_and_run(device_inputs, out, c_dtype, config)
    assert torch.isfinite(out).all()
    # These FP4 values, power-of-two scales and alpha=1/8 have exact FP32 sums
    # at K <= 1024. Compare after rounding the reference to the output dtype.
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype,c_dtype", _OUTPUT_DTYPES)
@pytest.mark.parametrize(
    "shape,use_tma_store,swap_ab,expected",
    [
        pytest.param((128, 128, 256), False, False, True, id="direct-n128"),
        pytest.param((128, 96, 256), False, False, False, id="direct-n96-rejected"),
        pytest.param((128, 96, 256), True, False, True, id="tma-n96"),
        pytest.param((96, 128, 256), False, True, True, id="swapped-kernel-n96"),
    ],
)
def test_sm103_can_implement(shape, use_tma_store, swap_ab, expected, dtype, c_dtype):
    # N=96 meets the other alignment rules but not the new direct-store rule.
    config = ((128, 128), (1, 1), use_tma_store, swap_ab)
    assert _can_implement(shape, c_dtype, config) is expected
