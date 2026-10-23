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


@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo",
    [(72, 72), (96, 96), (64, 96), (96, 64)],
)
def test_single_prefill_rejects_unsafe_head_dim(head_dim_qk, head_dim_vo):
    q = torch.empty(16, 1, head_dim_qk, dtype=torch.float16, device="cuda")
    k = torch.empty(16, 1, head_dim_qk, dtype=torch.float16, device="cuda")
    v = torch.empty(16, 1, head_dim_vo, dtype=torch.float16, device="cuda")

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
@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo",
    [(96, 96), (64, 96), (96, 64)],
)
def test_paged_prefill_rejects_unsafe_head_dim(method, head_dim_qk, head_dim_vo):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    args = (*_paged_inputs(), 1, 1, head_dim_qk, 16)
    kwargs = {"head_dim_vo": head_dim_vo}

    with pytest.raises(ValueError, match="positive multiples of 64"):
        getattr(wrapper, method)(*args, **kwargs)


@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo",
    [(96, 96), (64, 96), (96, 64)],
)
def test_ragged_prefill_rejects_unsafe_head_dim(head_dim_qk, head_dim_vo):
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="positive multiples of 64"):
        wrapper.plan(indptr, indptr, 1, 1, head_dim_qk, head_dim_vo=head_dim_vo)


# FA2 loads one-byte K and V rows 128 elements at a time, or 64 when head_dim_vo
# is 64, and output tiles of more than 256 dims must divide head_dim_vo.
_ONE_BYTE_UNSAFE = [
    (192, 192),
    (64, 128),
    (128, 64),
    (192, 128),
    (448, 256),
    (576, 512),
]
_VO_TILE_UNSAFE = [(320, 320), (384, 384), (448, 448)]
_BATCH_UNSAFE = [
    (*dims, torch.float8_e4m3fn, "one-byte KV") for dims in _ONE_BYTE_UNSAFE
] + [(*dims, torch.float16, "multiple of 256") for dims in _VO_TILE_UNSAFE]


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("head_dim_qk,head_dim_vo", _ONE_BYTE_UNSAFE)
def test_single_prefill_rejects_one_byte_kv_head_dim(
    kv_dtype, head_dim_qk, head_dim_vo
):
    q = torch.empty(16, 1, head_dim_qk, dtype=torch.float16, device="cuda")
    k = torch.empty(16, 1, head_dim_qk, dtype=kv_dtype, device="cuda")
    v = torch.empty(16, 1, head_dim_vo, dtype=kv_dtype, device="cuda")

    with pytest.raises(ValueError, match="one-byte KV"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", _VO_TILE_UNSAFE)
def test_single_prefill_rejects_untileable_head_dim_vo(head_dim_qk, head_dim_vo):
    q = torch.empty(16, 1, head_dim_qk, dtype=torch.float16, device="cuda")
    k = torch.empty(16, 1, head_dim_qk, dtype=torch.float16, device="cuda")
    v = torch.empty(16, 1, head_dim_vo, dtype=torch.float16, device="cuda")

    with pytest.raises(ValueError, match="multiple of 256"):
        flashinfer.single_prefill_with_kv_cache(q, k, v, backend="fa2")


@pytest.mark.parametrize("method", ["workspace_size", "plan"])
@pytest.mark.parametrize("head_dim_qk,head_dim_vo,kv_dtype,match", _BATCH_UNSAFE)
def test_paged_prefill_rejects_kv_dtype_and_vo_tile_head_dims(
    method, head_dim_qk, head_dim_vo, kv_dtype, match
):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    args = (*_paged_inputs(), 1, 1, head_dim_qk, 16)
    kwargs = {
        "head_dim_vo": head_dim_vo,
        "q_data_type": torch.float16,
        "kv_data_type": kv_dtype,
    }

    with pytest.raises(ValueError, match=match):
        getattr(wrapper, method)(*args, **kwargs)


@pytest.mark.parametrize("head_dim_qk,head_dim_vo,kv_dtype,match", _BATCH_UNSAFE)
def test_ragged_prefill_rejects_kv_dtype_and_vo_tile_head_dims(
    head_dim_qk, head_dim_vo, kv_dtype, match
):
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _workspace(), backend="fa2"
    )
    indptr = torch.tensor([0, 16], dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match=match):
        wrapper.plan(
            indptr,
            indptr,
            1,
            1,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_data_type=torch.float16,
            kv_data_type=kv_dtype,
        )


@pytest.mark.parametrize(
    "head_dim_qk,head_dim_vo,kv_dtype",
    [
        (64, 64, torch.float8_e4m3fn),
        (128, 128, torch.float8_e4m3fn),
        (256, 256, torch.float8_e5m2),
        (512, 512, torch.float8_e4m3fn),
        (512, 256, torch.uint8),  # packed NVFP4, Gemma-4's asymmetric shape
        (192, 128, torch.float16),
        (512, 512, torch.bfloat16),
    ],
)
def test_head_dim_check_accepts_supported_head_dims(head_dim_qk, head_dim_vo, kv_dtype):
    prefill._check_head_dim(head_dim_qk, head_dim_vo, "fa2", kv_dtype)


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
