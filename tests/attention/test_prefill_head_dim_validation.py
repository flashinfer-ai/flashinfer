import pytest
import torch

import flashinfer
import flashinfer.prefill as prefill


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="prefill head-dimension tests require CUDA"
)


def _workspace():
    return torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _paged_inputs():
    qo_indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    paged_kv_last_page_len = torch.tensor([16], dtype=torch.int32, device="cuda")
    return qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len


@pytest.mark.parametrize("head_dim", [72, 96, 0])
def test_single_prefill_rejects_unsafe_head_dim(head_dim):
    q = torch.empty(16, 1, head_dim, dtype=torch.float16, device="cuda")
    k = torch.empty(16, 1, head_dim, dtype=torch.float16, device="cuda")
    v = torch.empty(16, 1, head_dim, dtype=torch.float16, device="cuda")

    with pytest.raises(ValueError, match="positive multiples of 64"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")


def test_single_prefill_accepts_safe_head_dim():
    q = torch.randn(16, 1, 64, dtype=torch.float16, device="cuda")
    k = torch.randn(16, 1, 64, dtype=torch.float16, device="cuda")
    v = torch.randn(16, 1, 64, dtype=torch.float16, device="cuda")

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")

    assert o.shape == q.shape
    assert torch.isfinite(o).all()


@pytest.mark.parametrize("method", ["workspace_size", "plan"])
def test_paged_prefill_rejects_unsafe_head_dim(method):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    args = (*_paged_inputs(), 1, 1, 96, 16)

    with pytest.raises(ValueError, match="positive multiples of 64"):
        getattr(wrapper, method)(*args)


def test_ragged_prefill_rejects_unsafe_head_dim():
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="positive multiples of 64"):
        wrapper.plan(indptr, indptr, 1, 1, 96)


def test_paged_workspace_size_selects_backend_with_head_dims(monkeypatch):
    class Module:
        @staticmethod
        def workspace_size(*args):
            return 0, 0

    selected_dims = None

    def select_backend(*args, head_dim_qk, head_dim_vo):
        nonlocal selected_dims
        selected_dims = head_dim_qk, head_dim_vo
        return "fa2"

    monkeypatch.setattr(prefill, "determine_attention_backend", select_backend)
    monkeypatch.setattr(prefill, "get_batch_prefill_module", lambda *args: Module())

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(_workspace())
    wrapper.workspace_size(*_paged_inputs(), 1, 1, 512, 16, head_dim_vo=512)

    assert selected_dims == (512, 512)
