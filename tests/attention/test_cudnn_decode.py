import math

import pytest
import torch

import flashinfer
import flashinfer.cudnn.decode as cudnn_decode

# The fallback (cubin) decode path is bf16-only and has no lse output; tests
# exercising fp16 or return_lse need the cuDNN graph backend.
requires_cudnn_graph = pytest.mark.skipif(
    not cudnn_decode.CUDNN_AVAILABLE,
    reason="requires the cudnn-frontend python package (cuDNN graph backend)",
)


def _build_paged_kv(batch_size, s_kv, page_size, num_kv_heads, head_dim, dtype, device):
    """Interleaved HND paged KV cache plus strided K/V views and block tables.

    Mirrors the layout used by test_cudnn_decode: pages of sequence ``i`` are
    ``[i * num_pages_per_seq, (i + 1) * num_pages_per_seq)``.
    """
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=dtype).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    strides = (
        2 * page_size * num_kv_heads * head_dim,
        head_dim,
        num_kv_heads * head_dim,
        1,
    )
    k_cache = kv_cache[:, 0, :, :, :].as_strided(
        (total_num_pages, num_kv_heads, page_size, head_dim), strides
    )
    v_cache = kv_cache[:, 1, :, :, :].as_strided(
        (total_num_pages, num_kv_heads, page_size, head_dim), strides
    )

    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )
    return k_cache, v_cache, block_tables


def _decode_ref(q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale):
    """fp32 torch decode reference; returns ``(out, lse)``.

    ``lse`` is the natural-log log-sum-exp of the scaled scores over the valid
    KV positions (``scale`` folded in), shape ``(batch_size, num_heads_qo)``.
    """
    batch_size, num_heads_qo, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    d_vo = v_cache.shape[3]
    gqa_ratio = num_heads_qo // num_kv_heads

    out = torch.empty(
        batch_size, num_heads_qo, d_vo, dtype=torch.float32, device=q.device
    )
    lse = torch.empty(batch_size, num_heads_qo, dtype=torch.float32, device=q.device)
    for b in range(batch_size):
        kv_len = int(actual_seq_lens_kv.flatten()[b].item())
        pages = block_tables[b].to(torch.long)
        k_b = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim)[:, :kv_len]
            .float()
            .repeat_interleave(gqa_ratio, dim=0)
        )
        v_b = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, d_vo)[:, :kv_len]
            .float()
            .repeat_interleave(gqa_ratio, dim=0)
        )
        scores = torch.einsum("hd,hld->hl", q[b].float(), k_b) * scale
        lse[b] = torch.logsumexp(scores, dim=-1)
        out[b] = torch.einsum("hl,hld->hd", torch.softmax(scores, dim=-1), v_b)
    return out, lse


def _run_cudnn_decode(
    q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale, **kwargs
):
    device = q.device
    batch_size, num_qo_heads, head_dim = q.shape
    page_size = k_cache.shape[2]
    s_kv = block_tables.shape[1] * page_size
    ragged_q = torch.arange(0, batch_size + 1, device=device) * (
        num_qo_heads * head_dim
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_sequence_kv=s_kv,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        batch_offsets_q=ragged_q,
        batch_offsets_o=ragged_q,
        **kwargs,
    )


@pytest.mark.parametrize("batch_size", [8, 16, 32])
@pytest.mark.parametrize("s_kv", [512, 8192])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True, False])
def test_cudnn_decode(
    batch_size,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    is_cuda_graph_compatible,
):
    # test set up basics
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    s_qo = 1
    head_dim = 128

    # Initialize Q tensor
    # Since the number of tokens is 1, batch size is the token count
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]

    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )

    # Initialize scale
    scale = float(1.0 / (head_dim**0.5))

    # Actual sequence lengths (should be randomized across batches. )
    actual_seq_lens_kv = torch.randint(
        0, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    ragged_q = torch.arange(0, batch_size + 1, device=device) * (
        num_qo_heads * head_dim
    )

    workspace_buffer_size = math.ceil(
        (
            batch_size * s_qo * num_qo_heads * head_dim * 4
            + batch_size * s_qo * num_qo_heads * 4
        )
        / (1024 * 1024)
    ) * (1024 * 1024)

    workspace_buffer_size = max(workspace_buffer_size, 128 * 1024 * 1024)

    workspace_buffer = torch.empty(
        workspace_buffer_size, dtype=torch.int8, device=device
    )

    output = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale,
        workspace_buffer,
        max_sequence_kv=s_kv,
        actual_seq_lens_kv=actual_seq_lens_kv,
        block_tables=block_tables,
        is_cuda_graph_compatible=is_cuda_graph_compatible,
        batch_offsets_q=ragged_q,
        batch_offsets_o=ragged_q,
    )

    actual_seq_lens_kv_device = actual_seq_lens_kv.to(device)

    kv_indptr = (
        torch.cat(
            [
                torch.tensor([0], device=device),
                torch.cumsum(
                    (actual_seq_lens_kv_device.flatten() + page_size - 1) // page_size,
                    dim=0,
                ),
            ]
        )
        .int()
        .to(device)
    )

    # kv_indices
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )

    # kv_last_page_len
    kv_last_page_len = (
        torch.where(
            actual_seq_lens_kv_device.flatten() % page_size == 0,
            torch.full((batch_size,), page_size, device=device),
            actual_seq_lens_kv_device.flatten() % page_size,
        )
        .int()
        .to(device)
    )

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer_ref, "HND")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, rtol=1e-2, atol=1e-2)


@requires_cudnn_graph
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cudnn_decode_dtypes(dtype):
    """q.dtype must be honored: fp16 inputs vs an fp32 reference built from the
    same fp16 inputs (fails if fp16 buffers are silently reinterpreted as bf16)."""
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_kv, page_size = 8, 512, 16
    num_kv_heads, num_qo_heads, head_dim = 8, 32, 128

    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_cache, v_cache, block_tables = _build_paged_kv(
        batch_size, s_kv, page_size, num_kv_heads, head_dim, dtype, device
    )
    scale = float(1.0 / (head_dim**0.5))
    actual_seq_lens_kv = torch.randint(
        1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    output = _run_cudnn_decode(
        q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
    )
    assert output.dtype == dtype

    out_ref, _ = _decode_ref(
        q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
    )
    torch.testing.assert_close(output, out_ref.to(dtype), rtol=1e-2, atol=1e-2)


@requires_cudnn_graph
def test_cudnn_decode_return_lse():
    """return_lse=True returns (out, lse); out matches the return_lse=False
    output, and lse matches natural-log logsumexp of the scaled scores."""
    torch.manual_seed(1)
    device = "cuda:0"
    batch_size, s_kv, page_size = 8, 512, 16
    num_kv_heads, num_qo_heads, head_dim = 8, 32, 128
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_cache, v_cache, block_tables = _build_paged_kv(
        batch_size, s_kv, page_size, num_kv_heads, head_dim, dtype, device
    )
    scale = float(1.0 / (head_dim**0.5))
    actual_seq_lens_kv = torch.randint(
        1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    out_no_lse = _run_cudnn_decode(
        q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
    )

    # Pre-allocated lse buffer must be used as-is.
    lse_buf = torch.full(
        (batch_size, num_qo_heads), float("nan"), device=device, dtype=torch.float32
    )
    out, lse = _run_cudnn_decode(
        q,
        k_cache,
        v_cache,
        block_tables,
        actual_seq_lens_kv,
        scale,
        return_lse=True,
        lse=lse_buf,
    )
    assert lse is lse_buf

    torch.testing.assert_close(out, out_no_lse)

    out_ref, lse_ref = _decode_ref(
        q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
    )
    torch.testing.assert_close(out, out_ref.to(dtype), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=2e-2, atol=2e-2)

    # Internally-allocated lse path.
    out2, lse2 = _run_cudnn_decode(
        q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale, return_lse=True
    )
    torch.testing.assert_close(lse2, lse_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out2, out_no_lse)


@requires_cudnn_graph
def test_cudnn_decode_dtype_cache_no_collision():
    """Same-shape bf16 then fp16 calls must not share a cached graph."""
    torch.manual_seed(2)
    device = "cuda:0"
    batch_size, s_kv, page_size = 4, 256, 16
    num_kv_heads, num_qo_heads, head_dim = 4, 16, 128
    scale = float(1.0 / (head_dim**0.5))

    for dtype in (torch.bfloat16, torch.float16):
        q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=dtype)
        k_cache, v_cache, block_tables = _build_paged_kv(
            batch_size, s_kv, page_size, num_kv_heads, head_dim, dtype, device
        )
        actual_seq_lens_kv = torch.randint(
            1, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
        )
        output = _run_cudnn_decode(
            q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
        )
        out_ref, _ = _decode_ref(
            q, k_cache, v_cache, block_tables, actual_seq_lens_kv, scale
        )
        torch.testing.assert_close(output, out_ref.to(dtype), rtol=1e-2, atol=1e-2)


def test_cudnn_decode_unsupported_dtype_raises():
    torch.manual_seed(3)
    device = "cuda:0"
    batch_size, s_kv, page_size = 2, 64, 16
    num_kv_heads, num_qo_heads, head_dim = 2, 4, 128

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.float32
    )
    k_cache, v_cache, block_tables = _build_paged_kv(
        batch_size, s_kv, page_size, num_kv_heads, head_dim, torch.float32, device
    )
    actual_seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), s_kv, dtype=torch.int32, device=device
    )
    with pytest.raises(ValueError, match="only supports torch.float16"):
        _run_cudnn_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            actual_seq_lens_kv,
            float(1.0 / (head_dim**0.5)),
        )


def test_cudnn_decode_return_lse_requires_cudnn(monkeypatch):
    """The non-cuDNN cubin fallback has no lse output: raise NotImplementedError."""
    torch.manual_seed(4)
    device = "cuda:0"
    batch_size, s_kv, page_size = 2, 64, 16
    num_kv_heads, num_qo_heads, head_dim = 2, 4, 128

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_cache, v_cache, block_tables = _build_paged_kv(
        batch_size, s_kv, page_size, num_kv_heads, head_dim, torch.bfloat16, device
    )
    actual_seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), s_kv, dtype=torch.int32, device=device
    )
    monkeypatch.setattr(cudnn_decode, "CUDNN_AVAILABLE", False)
    with pytest.raises(NotImplementedError, match="return_lse"):
        _run_cudnn_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            actual_seq_lens_kv,
            float(1.0 / (head_dim**0.5)),
            return_lse=True,
        )


def test_sdpa_decode_key_fn_discriminates_baked_attributes():
    """Attributes _build_decode_graph bakes into a graph must key the cache.

    Uses meta tensors (the key fn is pure Python, no GPU): v_cache shape /
    strides, block table width and aux int dtypes are all baked via
    tensor_like, so same-shape calls differing only in them must not share a
    graph.
    """
    b, h, d = 4, 16, 128
    page_size = 16
    pages_per_seq = 4
    num_pages = b * pages_per_seq

    def meta(*shape, dtype=torch.bfloat16):
        return torch.empty(*shape, device="meta", dtype=dtype)

    def make_kwargs(**overrides):
        kwargs = dict(
            q=meta(b, h, d),
            k_cache=meta(num_pages, h, page_size, d),
            v_cache=meta(num_pages, h, page_size, d),
            scale=1.0 / (d**0.5),
            max_sequence_kv=page_size * pages_per_seq,
            actual_seq_lens_kv=meta(b, 1, 1, 1, dtype=torch.int32),
            block_tables=meta(b, pages_per_seq, dtype=torch.int32),
        )
        kwargs.update(overrides)
        return kwargs

    base = cudnn_decode._sdpa_decode_key_fn(**make_kwargs())
    assert base == cudnn_decode._sdpa_decode_key_fn(**make_kwargs())

    variants = {
        "v_cache d_vo": make_kwargs(v_cache=meta(num_pages, h, page_size, d // 2)),
        "block table width": make_kwargs(
            block_tables=meta(b, 2 * pages_per_seq, dtype=torch.int32)
        ),
        "seq-lens dtype": make_kwargs(
            actual_seq_lens_kv=meta(b, 1, 1, 1, dtype=torch.int64)
        ),
        "kv strides": make_kwargs(
            k_cache=meta(num_pages, page_size, h, d).permute(0, 2, 1, 3),
            v_cache=meta(num_pages, page_size, h, d).permute(0, 2, 1, 3),
        ),
    }
    for name, kwargs in variants.items():
        assert cudnn_decode._sdpa_decode_key_fn(**kwargs) != base, (
            f"cache key must change when {name} changes"
        )


def test_cudnn_decode_fp16_requires_cudnn(monkeypatch):
    """The non-cuDNN cubin fallback is bf16-only: fp16 must raise instead of
    being silently reinterpreted as bf16."""
    torch.manual_seed(5)
    device = "cuda:0"
    batch_size, s_kv, page_size = 2, 64, 16
    num_kv_heads, num_qo_heads, head_dim = 2, 4, 128

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device=device, dtype=torch.float16
    )
    k_cache, v_cache, block_tables = _build_paged_kv(
        batch_size, s_kv, page_size, num_kv_heads, head_dim, torch.float16, device
    )
    actual_seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), s_kv, dtype=torch.int32, device=device
    )
    monkeypatch.setattr(cudnn_decode, "CUDNN_AVAILABLE", False)
    with pytest.raises(NotImplementedError, match="bfloat16"):
        _run_cudnn_decode(
            q,
            k_cache,
            v_cache,
            block_tables,
            actual_seq_lens_kv,
            float(1.0 / (head_dim**0.5)),
        )
