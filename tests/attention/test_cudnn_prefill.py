import math

import pytest
import torch

import flashinfer
import cudnn

from flashinfer.utils import get_compute_capability


def _paged_sink_reference(q, k, v, table, q_lens, kv_lens, sinks, causal):
    """FP64 reference with one zero-value sink column per query head."""
    result = torch.zeros_like(q)
    stats = torch.empty(q.shape[:2], dtype=torch.float64, device=q.device)
    head_map = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
    start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        if q_len == 0:
            continue
        pages = table[b, : (kv_len + k.shape[2] - 1) // k.shape[2]].long()
        keys = k[pages].permute(1, 0, 2, 3).reshape(k.shape[1], -1, k.shape[-1])
        values = v[pages].permute(1, 0, 2, 3).reshape(v.shape[1], -1, v.shape[-1])
        keys, values = (
            keys[head_map, :kv_len].double(),
            values[head_map, :kv_len].double(),
        )
        query = q[start : start + q_len].transpose(0, 1).double()
        logits = query @ keys.transpose(-1, -2) / q.shape[-1] ** 0.5
        if causal:
            allowed = torch.arange(kv_len, device=q.device)[None, :] <= (
                kv_len - q_len + torch.arange(q_len, device=q.device)[:, None]
            )
            logits.masked_fill_(~allowed, -float("inf"))
        logits = torch.cat(
            [logits, sinks.double()[:, None, None].expand(-1, q_len, 1)], -1
        )
        stats[start : start + q_len] = torch.logsumexp(logits, -1).transpose(0, 1)
        result[start : start + q_len] = (
            (torch.softmax(logits, -1)[..., :kv_len] @ values)
            .transpose(0, 1)
            .to(q.dtype)
        )
        start += q_len
    return result, stats


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize(
    "q_lens", [[1, 1, 1, 0], [4, 4, 4, 0], [8, 8, 8, 0], [1, 4, 0, 8], [33, 65, 0, 17]]
)
@pytest.mark.parametrize("return_lse", [False, True])
def test_cudnn_paged_sink_verification_and_replay(
    layout, num_kv_heads, q_lens, return_lse
):
    from flashinfer.cudnn import prefill as cp

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Rubin paged sink qualification")
    if tuple(map(int, cudnn.__version__.split(".")[:2])) < (1, 32):
        pytest.skip("needs FE containing Rubin paged sink admission")
    torch.manual_seed(42)
    device, page_size, hq, d = torch.device("cuda"), 16, 32, 128
    # Partial pages and an empty request. Physical page order is unrelated to request order.
    kv_lens = [129, 151, 0, 139]
    batch, pages_per_request = len(q_lens), 10
    table = (
        torch.randperm(batch * pages_per_request, device=device).int().view(batch, -1)
    )
    raw = torch.randn(
        batch * pages_per_request + 1,
        page_size,
        num_kv_heads,
        2 * d,
        dtype=torch.bfloat16,
        device=device,
    )
    raw[-1].fill_(torch.nan)
    k, v = raw[..., :d].transpose(1, 2), raw[..., d:].transpose(1, 2)
    cache = (k.transpose(1, 2), v.transpose(1, 2)) if layout == "NHD" else (k, v)
    q = torch.randn(sum(q_lens), 3, hq, d, dtype=torch.bfloat16, device=device)[:, 1]
    sinks = torch.linspace(4, 7, hq, device=device)
    q_indptr = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    indices = torch.empty(0, dtype=torch.int32, device=device)
    last_page = torch.zeros(batch, dtype=torch.int32)
    live_kv = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        layout,
        backend="cudnn",
        use_cuda_graph=True,
        qo_indptr_buf=torch.empty_like(q_indptr, device=device),
        paged_kv_indptr_buf=torch.empty_like(kv_indptr, device=device),
        paged_kv_indices_buf=torch.empty(
            batch * pages_per_request, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buf=torch.empty_like(last_page, device=device),
    )

    def plan(lengths):
        q_indptr[1:] = torch.tensor(lengths).cumsum(0)
        wrapper.plan(
            q_indptr,
            kv_indptr,
            indices,
            last_page,
            hq,
            num_kv_heads,
            d,
            page_size,
            q_data_type=torch.bfloat16,
            causal=True,
            block_tables=table,
            seq_lens=live_kv,
            seq_lens_q=torch.tensor(lengths, dtype=torch.int32, device=device),
            max_token_per_sequence=max(q_lens),
            max_sequence_kv=max(kv_lens),
        )

    plan(q_lens)
    out = torch.empty(q.shape, dtype=q.dtype, device=device)
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=device)
        if return_lse
        else None
    )

    def run(query=q, pools=cache, sink=sinks, output=out):
        return wrapper.run(
            query,
            pools,
            sinks=sink,
            out=output,
            lse=lse,
            return_lse=return_lse,
            lse_base="ln",
        )

    def check(lengths):
        total = sum(lengths)
        expected, expected_stats = _paged_sink_reference(
            q[:total], k, v, table, lengths, live_kv.tolist(), sinks, True
        )
        torch.testing.assert_close(
            out[:total].float(), expected.float(), atol=2e-2, rtol=2e-2
        )
        if return_lse:
            torch.testing.assert_close(
                lse[:total].double(), expected_stats, atol=5e-3, rtol=5e-3
            )

    run()
    check(q_lens)
    prepared = wrapper._cudnn_prepared
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    builds = cp._prefill_graph_builds
    # New execute pointers must not select a new FE graph or reuse old sink values.
    sink2, out2 = sinks + 1, torch.empty_like(out)
    q2 = torch.empty_strided(q.shape, q.stride(), dtype=q.dtype, device=device)
    q2.copy_(-q)
    raw2 = raw.clone().mul_(0.5)
    k2, v2 = raw2[..., :d].transpose(1, 2), raw2[..., d:].transpose(1, 2)
    cache2 = (k2.transpose(1, 2), v2.transpose(1, 2)) if layout == "NHD" else (k2, v2)
    run(q2, cache2, sink2, out2)
    expected, _ = _paged_sink_reference(q2, k2, v2, table, q_lens, kv_lens, sink2, True)
    torch.testing.assert_close(out2.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert cp._prefill_graph_builds == builds
    assert wrapper._cudnn_prepared is prepared
    # A retained capture observes changed page IDs, Q offsets, lengths and sink contents.
    table.copy_(table.roll(1, dims=1))
    live_kv.sub_(torch.tensor([2, 4, 0, 2], device=device))
    sinks.add_(0.5)
    changed_q = q_lens.copy()
    if changed_q[1] > 1:
        changed_q[1] -= 1
    plan(changed_q)
    out.fill_(torch.nan)
    if lse is not None:
        lse.fill_(torch.nan)
    graph.replay()
    check(changed_q)
    assert cp._prefill_graph_builds == builds
    if return_lse:
        # The public default is base-2 LSE, including native-log2 declines.
        total = sum(changed_q)
        wrapper.run(
            q[:total],
            cache,
            sinks=sinks,
            out=out[:total],
            lse=lse[:total],
            return_lse=True,
        )
        _, expected_stats = _paged_sink_reference(
            q[:total], k, v, table, changed_q, live_kv.tolist(), sinks, True
        )
        torch.testing.assert_close(
            lse[:total].double(),
            expected_stats / math.log(2.0),
            atol=5e-3,
            rtol=5e-3,
        )


@pytest.mark.parametrize("kind", ["shape", "dtype", "stride", "device", "query_dtype"])
def test_cudnn_prefill_sink_input_contract(kind):
    from flashinfer.cudnn.prefill import _validate_prefill_sinks

    q = torch.empty(4, 32, 128, dtype=torch.bfloat16)
    sinks = torch.empty(32, dtype=torch.float32)
    if kind == "shape":
        sinks = sinks.view(1, 32)
    elif kind == "dtype":
        sinks = sinks.bfloat16()
    elif kind == "stride":
        sinks = torch.empty(64)[::2]
    elif kind == "device":
        sinks = torch.empty(32, device="meta")
    else:
        q = q.to(torch.float8_e4m3fn)
    with pytest.raises((ValueError, NotImplementedError)):
        _validate_prefill_sinks(q, sinks)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

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

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

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
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

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

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    cache_nhd = (k_cache.transpose(1, 2), v_cache.transpose(1, 2))
    result = wrapper_cudnn.run(q, cache_nhd, return_lse=return_lse)
    output, stats = result if return_lse else (result, None)
    if is_cuda_graph_compatible:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper_cudnn.run(
                q, cache_nhd, out=output, lse=stats, return_lse=return_lse
            )
        output.fill_(torch.nan)
        if stats is not None:
            stats.fill_(torch.nan)
        graph.replay()

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    result_ref = wrapper.run(q, kv_cache, return_lse=return_lse)
    output_ref, stats_ref = result_ref if return_lse else (result_ref, None)
    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)
    if return_lse:
        torch.testing.assert_close(stats, stats_ref, atol=3e-3, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill_fp8(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    if cudnn.backend_version() < 91701:
        pytest.skip("cuDNN backend version is less than 9.17.1, skipping test")

    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # test set up basics
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"

    major, _ = get_compute_capability(torch.device(device))

    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, skipping test"
        )

    # TODO: Remove this xfail once cuDNN fixes FP8 prefill on Blackwell
    pytest.xfail(
        "cuDNN FP8 prefill has known issues on Blackwell; expected to be fixed in a subsequent cuDNN release"
    )

    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )

    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    q_scale = q.amax().item() / 256

    q_scale = torch.tensor(q_scale, device=device, dtype=torch.float32)
    q_fp8 = (q / q_scale).to(torch.float8_e4m3fn)

    q_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Initialize KV Cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device) * 0.05
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

    k_scale = k_cache.amax().item() / 256
    v_scale = v_cache.amax().item() / 256
    k_cache_fp8 = (k_cache / k_scale).to(torch.float8_e4m3fn)
    v_cache_fp8 = (v_cache / v_scale).to(torch.float8_e4m3fn)

    k_scale_tensor = torch.tensor(k_scale, device=device, dtype=torch.float32)
    v_scale_tensor = torch.tensor(v_scale, device=device, dtype=torch.float32)

    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()

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
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()

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

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_cudnn = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    wrapper_cudnn.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        seq_lens=actual_seq_lens_kv,
        seq_lens_q=actual_seq_lens_q,
        sm_scale=scale,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        block_tables=block_tables,
    )

    output = wrapper_cudnn.run(
        q_fp8,
        (k_cache_fp8.transpose(1, 2), v_cache_fp8.transpose(1, 2)),
        q_scale=q_scale,
        k_scale=k_scale_tensor,
        v_scale=v_scale_tensor,
    )

    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # Workspace buffer
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )

    output_ref = wrapper.run(q, kv_cache)

    torch.testing.assert_close(output, output_ref, atol=1e-2, rtol=1e-2)
