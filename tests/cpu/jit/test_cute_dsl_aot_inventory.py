"""Check critical vLLM GEMM inventory coverage without compiling kernels."""

from itertools import product

import pytest
import torch

pytest.importorskip("cutlass")

from flashinfer.cute_dsl.utils import torch_to_cutlass_dtype  # noqa: E402
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import (  # noqa: E402
    _blockscaled_kernel_disk_name,
    _get_mm_fp4_cute_dsl_tactics,
    _mm_fp4_cache_key,
    mm_fp4_l2_policy,
)
from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import (  # noqa: E402
    _mm_mxfp8_cache_key,
)
from flashinfer.jit.gemm.cute_dsl_fp4 import mm_fp4_aot_configs  # noqa: E402
from flashinfer.jit.gemm.cute_dsl_mxfp8 import mm_mxfp8_aot_configs  # noqa: E402


def require_arch(arch):
    if arch == "sm107a":
        pytest.importorskip("flashinfer.gemm.kernels.dense_blockscaled_gemm_sm107")


def assert_packaged(configs, key):
    name = _blockscaled_kernel_disk_name(key, 1, None)
    assert configs.get(name) == key, f"Missing runtime specialization: {name}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ("sm100a", "sm107a"))
def test_fp4_inventory_covers_bf16_runtime_tactics(arch):
    require_arch(arch)
    configs = mm_fp4_aot_configs(arch)
    dtype = torch.bfloat16
    # Decode and prefill exercise split-K and both persistent L2 regions.
    # K >= 8192 avoids GPU queries. SM103 shares SM100's enumeration.
    for use_nvfp4, (m, n, k) in product(
        (True, False),
        ((8, 128, 16384), (8192, 8192, 8192)),
    ):
        tactics = _get_mm_fp4_cute_dsl_tactics(
            m, n, k, use_nvfp4, torch_to_cutlass_dtype(dtype), None, int(arch[2:-1])
        )
        assert tactics
        for tactic in tactics:
            assert_packaged(
                configs,
                _mm_fp4_cache_key(
                    16 if use_nvfp4 else 32,
                    tactic,
                    True,
                    dtype,
                    None,
                    mm_fp4_l2_policy(m, tactic[0], tactic[2], tactic[4]),
                ),
            )


@pytest.mark.cpu_only
def test_mxfp8_inventory_keeps_rubin_and_splitk():
    require_arch("sm107a")
    configs = mm_mxfp8_aot_configs("sm107a")
    # One public API call can use native Rubin without PDL or its low-M
    # SM100 split-K fallback with PDL. Both must survive inventory pruning.
    dtype = torch.bfloat16
    assert_packaged(
        configs,
        _mm_mxfp8_cache_key(
            ((128, 128), (1, 1), False, 128), False, dtype, native_sm107=True
        ),
    )
    assert_packaged(
        configs,
        _mm_mxfp8_cache_key(((128, 8), (1, 1), True, False, 2), True, dtype),
    )
