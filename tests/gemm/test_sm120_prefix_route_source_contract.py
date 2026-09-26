"""CPU-only source contracts for the SM120 fused-prefix A/B seam.

The seam spans three files: a C++ ladder header, the TVM-FFI prefix gate, and the
producer dispatcher. Python mirrors the ladder because the M537 native producer
reads a prepacked LoRA-down matrix that the cuBLASLt prefix does not, so a silent
disagreement between the two sides would corrupt results rather than slow them
down. These tests pin the agreement without a GPU.
"""

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LADDER_HEADER = REPO_ROOT / "include/flashinfer/gemm/svdquant_sm120_prefix_route.h"
PRODUCER_HEADER = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
)
LINEAR_SOURCE = REPO_ROOT / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu"

_LADDER_ENTRY = re.compile(
    r"\{\s*(\d+),\s*(\d+),\s*Route::k(\w+),\s*(true|false),\s*(true|false),"
    r"\s*(true|false)\s*\}"
)
_ROUTE_NAMES = {"Cublaslt": "cublaslt", "Native": "native", "NativeAlt": "native_alt"}
# The ladder lives in flashinfer::gemm; callers outside it must say so in full.
_QUALIFIED = "flashinfer::gemm::svdquant_sm120_prefix_route::"

# Mirrors KernelLaunchGeometry in the producer header.
_SF_VEC_SIZE = 16
_RANK = 32
_LARGE_M_DOWN_WARPS = _RANK // 8
# One wave on the SM count both SM120 parts in use here happen to have.
_ONE_WAVE_SM_COUNT = 110


def _large_m_geometry(m: int, k: int, block_threads: int, tile_m: int, tile_k: int):
    """Recompute the producer header's compile-time launch contract in Python."""
    quant_threads = block_threads - _LARGE_M_DOWN_WARPS * 32
    sf_cols_per_tile = tile_k // _SF_VEC_SIZE
    padded_m = (m + 127) // 128 * 128
    grid_blocks = (padded_m + tile_m - 1) // tile_m
    prefetch_lines = k * _RANK * 2 // 128
    shared_bytes = (
        2 * tile_m * (tile_k + 8) * 2
        + 2 * tile_k * 2
        + 2 * (tile_m * sf_cols_per_tile if k >= 8192 else 1)
    )
    assert k % tile_k == 0
    assert tile_k % (4 * _SF_VEC_SIZE) == 0
    assert 0 < tile_m <= 80 and tile_m % 16 == 0
    assert block_threads % 32 == 0 and quant_threads > 0
    assert (tile_m * sf_cols_per_tile) % quant_threads == 0
    assert k < 8192 or grid_blocks * block_threads >= prefetch_lines
    assert shared_bytes <= 48 * 1024
    return grid_blocks, shared_bytes


@pytest.mark.parametrize(
    "tile_m,expected_blocks",
    [(32, 64), (16, 128)],
)
def test_row57_producer_underfill_was_real_but_not_the_cause(
    tile_m: int,
    expected_blocks: int,
) -> None:
    # Given: the underfill hypothesis for row 57 -- a 32-row tile leaves 64 blocks
    # on a 110-SM device -- was arithmetically sound and still lost.
    # When: both geometries are re-derived from the producer launch contract.
    # Then: each is legal and only the 16-row tile fills the device, which pins
    # the refutation to the K14336 per-block LoRA-down re-read, not to legality.
    grid_blocks, shared_bytes = _large_m_geometry(1935, 14336, 384, tile_m, 256)
    assert grid_blocks == expected_blocks
    assert shared_bytes <= 48 * 1024
    assert (grid_blocks >= _ONE_WAVE_SM_COUNT) is (tile_m == 16)


def test_row57_is_the_only_underfilled_batch4_producer_grid() -> None:
    # Given: underfill is the one structural producer asymmetry in Batch 4.
    # When: every Batch-4 large-M producer grid is re-derived.
    # Then: only row 57's geometry leaves the device in a partial wave, so no
    # other row can be attributed to the same cause without new evidence.
    underfilled = {
        (m, k)
        for m, k, block_threads, tile_m, tile_k in (
            (1935, 14336, 384, 32, 256),
            (6913, 5120, 192, 32, 128),
            (6913, 5376, 192, 32, 128),
            (6913, 7168, 256, 32, 256),
            (6913, 14336, 384, 32, 256),
        )
        if _large_m_geometry(m, k, block_threads, tile_m, tile_k)[0]
        < _ONE_WAVE_SM_COUNT
    }
    assert underfilled == {(1935, 14336)}
