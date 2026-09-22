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


def _decode_ref(
    q,
    k_cache,
    v_cache,
    block_tables,
    actual_seq_lens_kv,
    scale,
    *,
    q_len_per_req=1,
    window_left=-1,
    sinks=None,
):
    """fp32 torch decode reference; returns ``(out, lse)``.

    ``q`` has ``batch_size * q_len_per_req`` rows (one request's rows are
    consecutive). Row ``i`` of a request with ``kv_len`` keys sees keys
    ``0 .. kv_len - q_len_per_req + i`` (bottom-right causal; every key for
    ``q_len_per_req == 1``); ``window_left >= 0`` keeps the ``window_left``
    keys before that position plus the position itself; ``sinks[h]`` joins
    the softmax as one extra logit with a zero value row. ``lse`` is the
    base-2 log-sum-exp of the scaled scores (sink included), shape
    ``(rows, num_heads_qo)`` -- FlashInfer's LSE contract.
    """
    rows, num_heads_qo, head_dim = q.shape
    batch_size = rows // q_len_per_req
    num_kv_heads = k_cache.shape[1]
    d_vo = v_cache.shape[3]
    gqa_ratio = num_heads_qo // num_kv_heads

    out = torch.empty(rows, num_heads_qo, d_vo, dtype=torch.float32, device=q.device)
    lse = torch.empty(rows, num_heads_qo, dtype=torch.float32, device=q.device)
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
        for i in range(q_len_per_req):
            r = b * q_len_per_req + i
            hi = kv_len - q_len_per_req + i + 1  # keys 0 .. hi-1 are visible
            lo = max(0, hi - 1 - window_left) if window_left >= 0 else 0
            scores = torch.einsum("hd,hld->hl", q[r].float(), k_b[:, lo:hi]) * scale
            if sinks is not None:
                scores = torch.cat([scores, sinks.float().view(-1, 1)], dim=-1)
            lse[r] = torch.logsumexp(scores, dim=-1) * math.log2(math.e)
            attn = torch.softmax(scores, dim=-1)
            if sinks is not None:
                attn = attn[:, :-1]
            out[r] = torch.einsum("hl,hld->hd", attn, v_b[:, lo:hi])
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
    output, and lse matches the base-2 logsumexp of the scaled scores."""
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
    with pytest.raises(ValueError, match=r"only supports torch\.float16"):
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
        # The mask is baked into the graph: multi-token rows add the
        # bottom-right causal diagonal, the window its left bound; the sink
        # tensor is a graph input.
        "q_len_per_req": make_kwargs(
            q=meta(b, h, 2, d), q_len_per_req=2, actual_seq_lens_q=None
        ),
        "window_left": make_kwargs(window_left=128),
        "sinks": make_kwargs(sinks=meta(1, h, 1, 1, dtype=torch.float32)),
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


# --- BatchDecodeWithPagedKVCacheWrapper(backend="cudnn") -----------------------


def _wrapper_inputs(
    batch_size,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    dtype,
    kv_layout,
    device,
    q_len_per_req=1,
):
    """CSR paged-KV plan inputs + a 5-D paged cache in ``kv_layout`` + query
    (``batch_size * q_len_per_req`` rows, one request's rows consecutive)."""
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = batch_size * num_pages_per_seq
    kv_lens = torch.randint(
        max(1, s_kv // 2), s_kv + 1, (batch_size,), dtype=torch.int32
    )
    kv_lens[0] = s_kv
    pages_per_seq = (kv_lens + page_size - 1) // page_size
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages_per_seq, 0)
    indices = torch.randperm(total_num_pages, dtype=torch.int32)[: int(indptr[-1])]
    last_page_len = kv_lens - (pages_per_seq - 1) * page_size
    if kv_layout == "HND":
        shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    else:
        shape = (total_num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_cache = torch.randn(shape, dtype=dtype, device=device)
    q = torch.randn(
        batch_size * q_len_per_req, num_qo_heads, head_dim, dtype=dtype, device=device
    )
    return q, kv_cache, indptr.to(device), indices.to(device), last_page_len.to(device)


def _run_wrapper(
    backend,
    q,
    kv_cache,
    indptr,
    indices,
    last_page_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    dtype,
    kv_layout,
    run_kwargs=None,
    **plan_kwargs,
):
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, use_tensor_cores=(backend != "cudnn"), backend=backend
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        **plan_kwargs,
    )
    return wrapper.run(q, kv_cache, return_lse=True, **(run_kwargs or {}))


@requires_cudnn_graph
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (64, 4)])
def test_cudnn_wrapper_matches_fa2(
    dtype, kv_layout, page_size, num_qo_heads, num_kv_heads
):
    """backend='cudnn' on the paged-decode wrapper matches the fa2 tensor-core
    backend (output and base-2 LSE) for the same CSR plan inputs."""
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_kv, head_dim = 8, 2048, 128
    q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        batch_size,
        s_kv,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
        device,
    )
    args = (
        q,
        kv_cache,
        indptr,
        indices,
        last_page_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
    )
    out_ref, lse_ref = _run_wrapper("fa2", *args)
    out, lse = _run_wrapper("cudnn", *args)
    assert out.dtype == dtype and out.shape == out_ref.shape
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-2)


@requires_cudnn_graph
def test_cudnn_wrapper_replan_reuses_block_table_and_bucketed_max():
    """Consecutive plans with growing KV lengths inside one 1024-token bucket
    keep the same block-table buffer (and thus the same built cuDNN graph)."""
    torch.manual_seed(0)
    device = "cuda:0"
    dtype, kv_layout, page_size = torch.bfloat16, "HND", 16
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="cudnn"
    )
    tables = []
    for s_kv in (1500, 1600, 2048):
        q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
            4,
            s_kv,
            page_size,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            dtype,
            kv_layout,
            device,
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        assert wrapper._max_kv_len == 2048
        tables.append(wrapper._block_tables.data_ptr())
        out = wrapper.run(q, kv_cache)
        ref = _run_wrapper(
            "fa2",
            q,
            kv_cache,
            indptr,
            indices,
            last_page_len,
            page_size,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            dtype,
            kv_layout,
        )[0]
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    assert len(set(tables)) == 1


@requires_cudnn_graph
def test_cudnn_wrapper_cuda_graph_with_user_block_tables():
    """CUDA-graph mode: capture once with a caller-owned block table, re-plan
    with new page indices / lengths, replay, and match fa2."""
    torch.manual_seed(0)
    device = "cuda:0"
    dtype, kv_layout, page_size = torch.bfloat16, "HND", 16
    batch_size, s_kv = 4, 1024
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    max_pages = s_kv // page_size
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    indptr_buf = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indices_buf = torch.zeros(batch_size * max_pages, dtype=torch.int32, device=device)
    last_buf = torch.zeros(batch_size, dtype=torch.int32, device=device)
    block_tables = torch.zeros(batch_size, max_pages, dtype=torch.int32, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout,
        use_cuda_graph=True,
        backend="cudnn",
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_buf,
    )

    def fill_block_tables(indptr, indices):
        block_tables.zero_()
        for i in range(batch_size):
            b, e = int(indptr[i]), int(indptr[i + 1])
            block_tables[i, : e - b] = indices[b:e]

    q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        batch_size,
        s_kv,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
        device,
    )
    fill_block_tables(indptr, indices)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        block_tables=block_tables,
    )
    out = torch.empty_like(q)
    wrapper.run(q, kv_cache, out=out)  # warm-up builds the cuDNN graph outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        wrapper.run(q, kv_cache, out=out)

    for _ in range(2):
        q2, kv2, indptr, indices, last_page_len = _wrapper_inputs(
            batch_size,
            s_kv,
            page_size,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            dtype,
            kv_layout,
            device,
        )
        q.copy_(q2)
        kv_cache.copy_(kv2)
        fill_block_tables(indptr, indices)
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=dtype,
            kv_data_type=dtype,
            block_tables=block_tables,
        )
        g.replay()
        torch.cuda.synchronize()
        ref = _run_wrapper(
            "fa2",
            q,
            kv_cache,
            indptr,
            indices,
            last_page_len,
            page_size,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            dtype,
            kv_layout,
        )[0]
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_cudnn_graph
@pytest.mark.parametrize(
    "plan_kwargs",
    [
        dict(logits_soft_cap=30.0),
        dict(pos_encoding_mode="ROPE_LLAMA"),
        dict(kv_data_type=torch.float8_e4m3fn),
    ],
)
def test_cudnn_wrapper_rejects_unsupported(plan_kwargs):
    device = "cuda:0"
    dtype, page_size = torch.bfloat16, 16
    q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        4, 512, page_size, 4, 32, 128, dtype, "HND", device
    )
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", backend="cudnn"
    )
    kwargs = dict(q_data_type=dtype, kv_data_type=dtype)
    kwargs.update(plan_kwargs)
    with pytest.raises((NotImplementedError, ValueError)):
        wrapper.plan(indptr, indices, last_page_len, 32, 4, 128, page_size, **kwargs)


@requires_cudnn_graph
def test_cudnn_wrapper_rejects_malformed_sinks():
    """sinks must be a float32 (num_qo_heads,) tensor on q's device: a wrong
    shape or dtype is a ValueError before any graph is built."""
    device = "cuda:0"
    dtype, page_size = torch.bfloat16, 16
    q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        4, 512, page_size, 4, 32, 128, dtype, "HND", device
    )
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", backend="cudnn"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        32,
        4,
        128,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    with pytest.raises(ValueError):
        wrapper.run(q, kv_cache, sinks=torch.zeros(32, 1, device=device))
    with pytest.raises(ValueError):
        wrapper.run(q, kv_cache, sinks=torch.zeros(32, device=device, dtype=dtype))


@requires_cudnn_graph
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("packed_first", [True, False])
def test_cudnn_wrapper_packed_q_strides(dtype, kv_layout, packed_first):
    """A query sliced out of a packed QKV projection (batch stride > h*d) must
    give the same result as its contiguous copy, in either graph-cache
    population order (packed-Q graph first, or contiguous-Q graph first)."""
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_kv, page_size = 4, 128, 16
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    _, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        batch_size,
        s_kv,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
        device,
    )
    packed = torch.randn(
        batch_size,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        device=device,
        dtype=dtype,
    )
    q_packed = packed[:, : num_qo_heads * head_dim].view(
        batch_size, num_qo_heads, head_dim
    )
    assert not q_packed.is_contiguous()
    q_contig = q_packed.contiguous()

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="cudnn"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    order = (q_packed, q_contig) if packed_first else (q_contig, q_packed)
    results = [wrapper.run(q, kv_cache, return_lse=True) for q in order]
    (out_a, lse_a), (out_b, lse_b) = results
    torch.testing.assert_close(out_a, out_b, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse_a, lse_b, rtol=1e-4, atol=1e-4)

    out_ref, lse_ref = _run_wrapper(
        "fa2",
        q_contig,
        kv_cache,
        indptr,
        indices,
        last_page_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
    )
    torch.testing.assert_close(out_a, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse_a, lse_ref, rtol=1e-3, atol=1e-2)


def _mtp_args(
    dtype, kv_layout, q_len_per_req, num_qo_heads=32, num_kv_heads=8, head_dim=128
):
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_kv, page_size = 8, 2048, 16
    q, kv_cache, indptr, indices, last_page_len = _wrapper_inputs(
        batch_size,
        s_kv,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
        device,
        q_len_per_req=q_len_per_req,
    )
    return (
        q,
        kv_cache,
        indptr,
        indices,
        last_page_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        dtype,
        kv_layout,
    )


@requires_cudnn_graph
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("q_len_per_req", [2, 4])
def test_cudnn_wrapper_mtp_matches_fa2(dtype, kv_layout, q_len_per_req):
    """q_len_per_req > 1 (speculative / MTP verification rows): the cudnn
    backend applies the bottom-right causal diagonal per request and matches
    fa2's tensor-core decode on output and base-2 LSE, row for row."""
    args = _mtp_args(dtype, kv_layout, q_len_per_req)
    out_ref, lse_ref = _run_wrapper("fa2", *args, q_len_per_req=q_len_per_req)
    out, lse = _run_wrapper("cudnn", *args, q_len_per_req=q_len_per_req)
    assert out.shape == (8 * q_len_per_req, 32, 128) and lse.shape == (
        8 * q_len_per_req,
        32,
    )
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-2)


@requires_cudnn_graph
@pytest.mark.parametrize("window_left", [64, 128])
@pytest.mark.parametrize("q_len_per_req", [1, 2])
def test_cudnn_wrapper_sliding_window_matches_fa2(window_left, q_len_per_req):
    """FlashInfer's window_left counts the keys before a row's diagonal
    position (the position itself is always visible); cuDNN's band bound
    counts the diagonal too, so the wrapper passes window_left + 1. An
    off-by-one on either side shows up against fa2 at these small windows."""
    args = _mtp_args(torch.bfloat16, "HND", q_len_per_req)
    kw = dict(q_len_per_req=q_len_per_req, window_left=window_left)
    out_ref, lse_ref = _run_wrapper("fa2", *args, **kw)
    out, lse = _run_wrapper("cudnn", *args, **kw)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-2)


def _run_cudnn_or_skip_when_declined(feature, fn):
    """The cuDNN stacks differ in what their SDPA engines serve: the backend
    engine rejects an attention sink at s_q == 1, the FROST engines
    (cudnn-frontend 1.30+, enabled) serve it. A decline that names the
    feature is a skip; anything else is a failure."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- the frontend's error types vary
        if feature in str(exc).lower():
            pytest.skip(
                f"this cuDNN stack declines {feature} for this decode graph: "
                f"{type(exc).__name__}: {str(exc)[:200]}"
            )
        raise


@requires_cudnn_graph
@pytest.mark.parametrize(
    "num_qo_heads,num_kv_heads,head_dim", [(32, 8, 128), (64, 8, 64)]
)
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("q_len_per_req", [1, 2])
def test_cudnn_wrapper_sinks_match_reference(
    num_qo_heads, num_kv_heads, head_dim, window_left, q_len_per_req
):
    """Attention sinks (gpt-oss / Streaming-LLM): sinks[h] joins each row's
    softmax denominator as one extra logit with a zero value row, unscaled --
    FlashInfer's contract. Checked against the fp32 torch reference (output
    and base-2 LSE, the LSE including the sink), alone and with the sliding
    window (the gpt-oss decode graph: 64/8 heads at d=64) and with multi-token
    rows. The plain fa2 tensor-core module is not a reference here: it applies
    sinks only when built with the AttentionSink JIT variant. Skips where the
    cuDNN stack declines a sink at s_q == 1 (the backend engine); FROST-served
    stacks (cudnn-frontend 1.30+, engines enabled) run every case."""
    args = _mtp_args(
        torch.bfloat16, "HND", q_len_per_req, num_qo_heads, num_kv_heads, head_dim
    )
    q, kv_cache, indptr, indices, last_page_len, page_size = args[:6]
    device = q.device
    # Logits 6..10: large enough that the sink holds a visible share of the
    # softmax mass against 1k..2k unit-variance key scores (the no-sink
    # cross-check below needs the difference to show).
    sinks = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 4 + 6
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "HND", backend="cudnn"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len_per_req,
        window_left=window_left,
    )
    out, lse = _run_cudnn_or_skip_when_declined(
        "sink", lambda: wrapper.run(q, kv_cache, sinks=sinks, return_lse=True)
    )
    batch_size = indptr.numel() - 1
    out_ref, lse_ref = _decode_ref(
        q,
        kv_cache[:, 0],
        kv_cache[:, 1],
        wrapper._block_tables,
        wrapper._kv_lens_buffer[:batch_size],
        1.0 / math.sqrt(head_dim),
        q_len_per_req=q_len_per_req,
        window_left=window_left,
        sinks=sinks,
    )
    torch.testing.assert_close(out.float(), out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-2)
    # The sink must have taken part: without it the same graph gives a
    # different row wherever the sink logit competes with the keys.
    out_nosink, _ = _decode_ref(
        q,
        kv_cache[:, 0],
        kv_cache[:, 1],
        wrapper._block_tables,
        wrapper._kv_lens_buffer[:batch_size],
        1.0 / math.sqrt(head_dim),
        q_len_per_req=q_len_per_req,
        window_left=window_left,
    )
    assert (out_ref - out_nosink).abs().max().item() > 5e-2


@requires_cudnn_graph
def test_cudnn_decode_standalone_q_len_per_req_matches_wrapper():
    """cudnn_batch_decode_with_kv_cache(q_len_per_req=2) takes q as
    (batch * 2, H, D) consecutive rows, returns (batch * 2, H, D) and
    (batch * 2, H), and equals the wrapper's cudnn backend on the same plan
    (block table, KV lengths and declared maximum taken from the wrapper)."""
    dtype, kv_layout, q_len_per_req = torch.bfloat16, "HND", 2
    args = _mtp_args(dtype, kv_layout, q_len_per_req)
    q, kv_cache, indptr, indices, last_page_len, page_size = args[:6]
    num_kv_heads, num_qo_heads, head_dim = args[6:9]
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, backend="cudnn"
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
        q_len_per_req=q_len_per_req,
    )
    out_w, lse_w = wrapper.run(q, kv_cache, return_lse=True)
    batch_size = indptr.numel() - 1
    k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
    out, lse = cudnn_decode.cudnn_batch_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        1.0 / math.sqrt(head_dim),
        workspace,
        max_sequence_kv=wrapper._max_kv_len,
        actual_seq_lens_kv=wrapper._kv_lens_buffer[:batch_size].view(
            batch_size, 1, 1, 1
        ),
        block_tables=wrapper._block_tables,
        return_lse=True,
        q_len_per_req=q_len_per_req,
    )
    assert out.shape == (batch_size * q_len_per_req, num_qo_heads, head_dim)
    assert lse.shape == (batch_size * q_len_per_req, num_qo_heads)
    torch.testing.assert_close(out, out_w, rtol=0, atol=0)
    torch.testing.assert_close(lse, lse_w, rtol=0, atol=0)
    with pytest.raises(ValueError):
        cudnn_decode.cudnn_batch_decode_with_kv_cache(
            q[:-1],
            k_cache,
            v_cache,
            1.0 / math.sqrt(head_dim),
            workspace,
            max_sequence_kv=wrapper._max_kv_len,
            actual_seq_lens_kv=wrapper._kv_lens_buffer[:batch_size].view(
                batch_size, 1, 1, 1
            ),
            block_tables=wrapper._block_tables,
            q_len_per_req=q_len_per_req,
        )
