"""Smoke tests for MX-FP6 fused MoE kernel setup (no GPU compile)."""

# TODO(port): the FP6 extensions of the fused-MoE backends (the static backend
# base `_MoEStaticKernelBase`, the micro backend's `MXFP6_BLOCK_SIZE` /
# `is_supported_mxfp6`) were not ported upstream; the b12x FP6 MoE route
# is b12x.moe.fused_moe with quant_mode="w6a8_mx" instead. The affected
# tests are kept as reference behind pytest.skip gates.

import cutlass
import pytest

from b12x._lib.utils import mxfp6_tile_k, mxfp6_num_k_blocks
from b12x.moe._shared.kernels import mxfp6_moe
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend


def test_mxfp6_moe_helpers_import():
    assert mxfp6_moe.moe_emit_mma_k_block is not None


def test_mxfp6_tile_k_matches_moe_expectation():
    assert mxfp6_tile_k() == 128
    assert mxfp6_num_k_blocks(128) == 4


def test_moe_static_dynamic_init_tile_k_default():
    pytest.skip("pending: FP6 static-backend port (_MoEStaticKernelBase not upstream)")
    from b12x.moe._shared.kernels.static import _MoEStaticKernelBase

    backend = _MoEStaticKernelBase(
        sf_vec_size=32,
        mma_tiler_mn=(128, 128),
        output_tile_count_n=1,
    )
    assert backend.tile_shape_mnk[2] == 256
    dyn = MoEDynamicKernelBackend(sf_vec_size=32, mma_tiler_mn=(128, 128))
    assert dyn.tile_shape_mnk[2] == 256


def test_micro_mxfp6_not_supported():
    pytest.skip("pending: FP6 micro kernel port (is_supported_mxfp6 not upstream)")
    from b12x.moe._shared.kernels.micro import (
        MXFP6_BLOCK_SIZE,
        MoEMicroKernelBackend,
    )

    assert MXFP6_BLOCK_SIZE == 32
    assert MoEMicroKernelBackend.is_supported_mxfp6(1, 128, 128, 8, 8) is False


def test_fp6_element_types_exist():
    assert cutlass.Float6E3M2FN is not None
    assert cutlass.Float6E2M3FN is not None
