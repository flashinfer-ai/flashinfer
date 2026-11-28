import pytest
import torch

import flashinfer
from flashinfer.utils import _check_num_heads


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="num_heads tests require CUDA"
)


def _workspace():
    return torch.empty(4 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def test_check_num_heads_helper():
    with pytest.raises(ValueError, match="must be positive"):
        _check_num_heads(0, 8)
    with pytest.raises(ValueError, match="must be positive"):
        _check_num_heads(8, 0)
    with pytest.raises(ValueError, match="must be positive"):
        _check_num_heads(-1, 8)
    with pytest.raises(ValueError, match="must be positive"):
        _check_num_heads(8, -2)
    with pytest.raises(ValueError, match="must be divisible"):
        _check_num_heads(7, 4)

    # Valid configurations do not raise
    _check_num_heads(8, 8)
    _check_num_heads(32, 8)
    _check_num_heads(64, 1)


@pytest.mark.parametrize("kv_heads", [0, -1])
def test_single_decode_rejects_non_positive_kv_heads(kv_heads):
    q = torch.randn(8, 128, dtype=torch.float16, device="cuda")
    k = torch.zeros(1, max(kv_heads, 0), 128, dtype=torch.float16, device="cuda")
    if kv_heads < 0:
        # Pass via non-divisible or zero-headed proxy
        pass
    with pytest.raises(ValueError, match="must be positive"):
        flashinfer.single_decode_with_kv_cache(q, k, k)


def test_single_decode_rejects_zero_qo_heads():
    q = torch.randn(0, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 8, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="must be positive"):
        flashinfer.single_decode_with_kv_cache(q, k, k)


def test_single_decode_rejects_non_divisible_heads():
    q = torch.randn(7, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="must be divisible"):
        flashinfer.single_decode_with_kv_cache(q, k, k)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_single_prefill_rejects_zero_kv_heads(layout):
    q = torch.randn(1, 8, 128, dtype=torch.float16, device="cuda")
    if layout == "NHD":
        k = torch.zeros(1, 0, 128, dtype=torch.float16, device="cuda")
    else:
        k = torch.zeros(0, 1, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="must be positive"):
        flashinfer.single_prefill_with_kv_cache(q, k, k, kv_layout=layout)


def test_single_prefill_rejects_zero_qo_heads():
    q = torch.randn(1, 0, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 8, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="must be positive"):
        flashinfer.single_prefill_with_kv_cache(q, k, k)


def test_single_prefill_rejects_non_divisible_heads():
    q = torch.randn(1, 7, 128, dtype=torch.float16, device="cuda")
    k = torch.randn(1, 4, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="must be divisible"):
        flashinfer.single_prefill_with_kv_cache(q, k, k)


@pytest.mark.parametrize("method", ["plan", "workspace_size"])
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,expected_match",
    [
        (8, 0, "must be positive"),
        (0, 8, "must be positive"),
        (7, 4, "must be divisible"),
    ],
)
def test_paged_decode_wrapper_rejects_invalid_num_heads(
    method, num_qo_heads, num_kv_heads, expected_match
):
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(_workspace(), "NHD")
    indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    indices = torch.zeros(16, dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match=expected_match):
        getattr(w, method)(
            indptr, indices, last_page_len, num_qo_heads, num_kv_heads, 128, 16
        )


@pytest.mark.parametrize("method", ["plan", "workspace_size"])
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,expected_match",
    [
        (8, 0, "must be positive"),
        (0, 8, "must be positive"),
        (7, 4, "must be divisible"),
    ],
)
def test_paged_prefill_wrapper_rejects_invalid_num_heads(
    method, num_qo_heads, num_kv_heads, expected_match
):
    wp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_workspace(), "NHD")
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.zeros(16, dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match=expected_match):
        getattr(wp, method)(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=128,
            page_size=16,
        )


@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,expected_match",
    [
        (8, 0, "must be positive"),
        (0, 8, "must be positive"),
        (7, 4, "must be divisible"),
    ],
)
def test_ragged_prefill_wrapper_rejects_invalid_num_heads(
    num_qo_heads, num_kv_heads, expected_match
):
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(_workspace(), "NHD")
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match=expected_match):
        wr.plan(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=128,
            causal=True,
            q_data_type=torch.float16,
        )
