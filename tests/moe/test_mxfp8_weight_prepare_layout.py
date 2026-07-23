"""Layout regression for gh #4087: MxFp8 weight-scale preparation.

The MxFp8 unified-MoE weight path (``prepare.py``) must produce the SAME
scale-factor layout as the legacy ``shuffle_matrix_sf_a`` pipeline the
kernels were written against: row permutation THEN 128x4
``block_scale_interleave``. #4026's adapter did only the permutation; the
gap escaped its own unit test because that test hard-coded the incomplete
layout as expected, and escaped the functional test because atol=0.05
dwarfed the output scale (~0.0025).

Per review guidance the checks (a) compare directly against the legacy
reference pipeline and (b) use sentinel data where scale bytes are all
distinct — with random weights, misplaced reads often land on an equal
E8M0 exponent and cancel out.
"""

import pytest
import torch

from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.utils import get_compute_capability


def _skip_unless_sm100():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if get_compute_capability(torch.device("cuda:0"))[0] != 10:
        pytest.skip("MxFp8 trtllm path requires SM100-family")


def _sentinel_sf(rows, cols_sf, dev):
    # Cycle a prime-period pattern so any misplaced byte changes the result.
    return (
        (torch.arange(rows * cols_sf, dtype=torch.int64, device=dev) % 251)
        .to(torch.uint8)
        .reshape(rows, cols_sf)
    )


@pytest.mark.parametrize("rows,cols", [(1024, 768), (256, 512), (2048, 1024)])
def test_w2_sf_shuffle_matches_legacy(rows, cols):
    """Non-gated (w2) scale path: permute+interleave == shuffle_matrix_sf_a."""
    _skip_unless_sm100()
    from flashinfer.fused_moe.core import get_w2_permute_indices_with_cache
    from flashinfer.quantization.fp4_quantization import shuffle_matrix_sf_a

    dev = torch.device("cuda:0")
    sf = _sentinel_sf(rows, cols // 32, dev)

    cache = {}
    permute_sf = get_w2_permute_indices_with_cache(cache, sf, 128, num_elts_per_sf=32)
    got = block_scale_interleave(sf[permute_sf.to(dev)].contiguous())

    want = shuffle_matrix_sf_a(sf, 128, num_elts_per_sf=32)

    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    n_diff = int((got != want).sum())
    assert n_diff == 0, f"{n_diff}/{want.numel()} scale bytes misplaced"


@pytest.mark.parametrize("rows,cols", [(2 * 768, 1024), (2 * 256, 512)])
def test_w31_sf_is_interleaved(rows, cols):
    """Gated (w3_w1) scale path: the result must be 128x4-interleaved, i.e.
    NOT equal to the permutation-only layout #4026 produced (the exact gh
    #4087 regression), and elementwise-preserving (a permutation of bytes)."""
    _skip_unless_sm100()
    from flashinfer.fused_moe.core import _maybe_get_cached_w3_w1_permute_indices

    dev = torch.device("cuda:0")
    sf = _sentinel_sf(rows, cols // 32, dev)

    cache = {}
    permute_sf = _maybe_get_cached_w3_w1_permute_indices(
        cache, sf, 128, num_elts_per_sf=32, is_gated_act_gemm=True
    )
    permuted_only = sf[permute_sf.to(dev)].contiguous()
    got = block_scale_interleave(permuted_only)

    # Regression guard: the shipped layout must NOT be the permutation-only
    # layout #4026 produced (the exact gh #4087 regression).
    assert not torch.equal(
        got.flatten()[: permuted_only.numel()], permuted_only.flatten()
    ), "interleave was a no-op -- layout regression (gh #4087)"
    if got.numel() == permuted_only.numel():
        # Pure relayout (no padding): byte multiset must be preserved.
        assert torch.equal(
            got.flatten().sort().values, permuted_only.flatten().sort().values
        ), "interleave lost/duplicated scale bytes"
