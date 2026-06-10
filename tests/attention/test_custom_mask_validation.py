"""Unit tests for ``custom_mask`` shape validation in
``single_prefill_with_kv_cache``.

These tests cover the input validation added in PR #3467, which checks that a
provided ``custom_mask`` has shape ``(qo_len, kv_len)`` before it is packed and
forwarded to the attention kernel. An invalid shape must raise a ``ValueError``
with a descriptive message, while ``None`` or an already-packed mask must skip
the validation entirely.
"""

import pytest
import torch

import flashinfer

DEVICE = "cuda:0"

# The validation lives in the Python entry of the API, but the "valid shape"
# and "skip validation" paths fall through to the actual attention kernel, which
# requires a CUDA device. Skip the whole module when no GPU is available.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _make_qkv(qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout):
    """Create ``q``/``k``/``v`` tensors honoring the requested ``kv_layout``."""
    q = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=DEVICE
    )
    if kv_layout == "NHD":
        kv_shape = (kv_len, num_kv_heads, head_dim)
    else:  # HND
        kv_shape = (num_kv_heads, kv_len, head_dim)
    k = torch.randn(*kv_shape, dtype=torch.float16, device=DEVICE)
    v = torch.randn(*kv_shape, dtype=torch.float16, device=DEVICE)
    return q, k, v


# ---------------------------------------------------------------------------
# Valid scenarios: a correctly shaped (qo_len, kv_len) mask must pass.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_custom_mask_valid_shape_passes(kv_layout):
    qo_len, kv_len = 16, 32
    num_qo_heads = num_kv_heads = 4
    head_dim = 128
    q, k, v = _make_qkv(
        qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout
    )
    mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=DEVICE)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask, kv_layout=kv_layout
    )
    assert o.shape == (qo_len, num_qo_heads, head_dim)


# ---------------------------------------------------------------------------
# Invalid scenarios: a wrong shape must raise ValueError before the kernel runs.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_shape_fn",
    [
        lambda qo, kv: (qo - 1, kv),  # row count mismatch
        lambda qo, kv: (qo, kv - 5),  # column count mismatch
        lambda qo, kv: (1, qo, kv),  # extra dimension
        lambda qo, kv: (kv, qo),  # transposed rows/cols
    ],
    ids=["row-mismatch", "col-mismatch", "extra-dim", "transposed"],
)
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_custom_mask_invalid_shape_raises(kv_layout, bad_shape_fn):
    qo_len, kv_len = 16, 32
    num_qo_heads = num_kv_heads = 4
    head_dim = 128
    q, k, v = _make_qkv(
        qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, kv_layout
    )
    bad_shape = bad_shape_fn(qo_len, kv_len)
    mask = torch.ones(*bad_shape, dtype=torch.bool, device=DEVICE)

    with pytest.raises(ValueError, match="custom_mask tensor must have shape"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, custom_mask=mask, kv_layout=kv_layout
        )


# ---------------------------------------------------------------------------
# Boundary scenarios: validation must be skipped entirely.
# ---------------------------------------------------------------------------
def test_custom_mask_none_skips_validation():
    qo_len, kv_len = 16, 32
    num_qo_heads = num_kv_heads = 4
    head_dim = 128
    q, k, v = _make_qkv(
        qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, "NHD"
    )

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=None)
    assert o.shape == (qo_len, num_qo_heads, head_dim)


def test_packed_custom_mask_skips_validation():
    qo_len, kv_len = 16, 32
    num_qo_heads = num_kv_heads = 4
    head_dim = 128
    q, k, v = _make_qkv(
        qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, "NHD"
    )

    full_mask = torch.ones(qo_len, kv_len, dtype=torch.bool, device=DEVICE)
    packed = flashinfer.packbits(
        full_mask.contiguous().view(-1), bitorder="little"
    )
    # A deliberately wrong-shaped ``custom_mask`` must be ignored when a packed
    # mask is supplied, so no ValueError should be raised.
    wrong_mask = torch.ones(qo_len - 1, kv_len, dtype=torch.bool, device=DEVICE)

    o = flashinfer.single_prefill_with_kv_cache(
        q, k, v, custom_mask=wrong_mask, packed_custom_mask=packed
    )
    assert o.shape == (qo_len, num_qo_heads, head_dim)
