"""Split-KV prefill when the plan reserves more KV than the request holds."""

import math

import pytest
import torch

import flashinfer

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 2, 128
BATCH_SIZE, KV_LEN = 4, 700
DTYPE = torch.bfloat16
DEVICE = "cuda:0"


def _make_inputs(batch_size, kv_len, qo_len, page_size, surplus_pages, seed=0):
    torch.manual_seed(seed)
    pages = (kv_len + page_size - 1) // page_size + surplus_pages
    kv_cache = torch.randn(
        pages * batch_size,
        2,
        page_size,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    q = torch.randn(
        batch_size * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    qo_indptr = torch.arange(
        0, (batch_size + 1) * qo_len, qo_len, dtype=torch.int32, device=DEVICE
    )
    kv_indptr = torch.arange(
        0, (batch_size + 1) * pages, pages, dtype=torch.int32, device=DEVICE
    )
    kv_indices = torch.arange(pages * batch_size, dtype=torch.int32, device=DEVICE)
    # (pages - 1) * page_size + last_page_len == kv_len, whatever the page count
    last_page_len = torch.full(
        (batch_size,),
        kv_len - (pages - 1) * page_size,
        dtype=torch.int32,
        device=DEVICE,
    )
    return q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, pages


def _plan_kwargs(split_pages, window_left=-1):
    kwargs = dict(
        causal=True,
        window_left=window_left,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
    )
    kwargs.update(
        {"fixed_split_size": split_pages} if split_pages else {"disable_split_kv": True}
    )
    return kwargs


def _plan(wrapper, inputs, page_size, split_pages, window_left=-1):
    q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, _ = inputs
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        **_plan_kwargs(split_pages, window_left),
    )


def _run(workspace, inputs, page_size, split_pages, window_left=-1):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    _plan(wrapper, inputs, page_size, split_pages, window_left)
    return wrapper.run(inputs[0], inputs[1]).float()


def _skip_unless_counts_differ(pages, page_size, split_pages, kv_len=KV_LEN):
    """The surplus pages must actually add a chunk, or the case proves nothing."""
    planned = math.ceil(pages / split_pages)
    from_kv_len = math.ceil(kv_len / (split_pages * page_size))
    if planned == from_kv_len:
        pytest.skip(f"the surplus pages add no chunk (both {planned})")


def _assert_close(out, reference):
    torch.testing.assert_close(
        out, reference, rtol=2e-2, atol=2e-2 * reference.abs().max().item()
    )


@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("qo_len", [1, 2, 4])
@pytest.mark.parametrize("split_pages", [2, 3])
def test_batch_prefill_over_reserved_plan(page_size, qo_len, split_pages):
    """Pages that carry no tokens must not change the result.

    plan() lays the partial outputs out from the page counts it is given, so
    trailing pages with a non-positive last_page_len make it reserve one more
    chunk per row than the KV length implies.
    """
    inputs = _make_inputs(BATCH_SIZE, KV_LEN, qo_len, page_size, split_pages)
    _skip_unless_counts_differ(inputs[-1], page_size, split_pages)

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    reference = _run(workspace, inputs, page_size, split_pages=None)
    _assert_close(_run(workspace, inputs, page_size, split_pages), reference)


@pytest.mark.parametrize("surplus_pages", [0, 2])
@pytest.mark.parametrize("window_left", [64, 256])
def test_batch_prefill_over_reserved_plan_sliding_window(surplus_pages, window_left):
    """A sliding window clamps the planned KV span; both sides must clamp alike.

    The plan and the kernel agree here even with surplus pages, because both clamp
    to the window, so this pins that agreement rather than the divergence above.
    """
    page_size, qo_len, split_pages = 16, 4, 2
    inputs = _make_inputs(BATCH_SIZE, KV_LEN, qo_len, page_size, surplus_pages)

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    reference = _run(
        workspace, inputs, page_size, split_pages=None, window_left=window_left
    )
    out = _run(workspace, inputs, page_size, split_pages, window_left=window_left)
    _assert_close(out, reference)


def test_batch_prefill_over_reserved_plan_cuda_graph():
    """The same plan, captured and replayed."""
    page_size, qo_len, split_pages, batch_size = 16, 4, 2, 1
    inputs = _make_inputs(batch_size, KV_LEN, qo_len, page_size, split_pages)
    _skip_unless_counts_differ(inputs[-1], page_size, split_pages)
    q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, _ = inputs

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    reference = _run(workspace, inputs, page_size, split_pages=None)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE),
        "NHD",
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr_buf=qo_indptr.clone(),
        paged_kv_indptr_buf=kv_indptr.clone(),
        paged_kv_indices_buf=kv_indices.clone(),
        paged_kv_last_page_len_buf=last_page_len.clone(),
    )
    try:
        _plan(wrapper, inputs, page_size, split_pages)
    except RuntimeError as err:
        # A CUDA-graph plan pads to the split batch size the card's SM count
        # allows, which a fixed split size can exceed on a small GPU.
        if "padded batch size" not in str(err):
            raise
        pytest.skip(f"this GPU cannot hold the padded plan: {err}")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            wrapper.run(q, kv_cache)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = wrapper.run(q, kv_cache)
    _plan(wrapper, inputs, page_size, split_pages)
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(out.float(), reference)


def test_batch_prefill_over_reserved_plan_exact_workspace():
    """workspace_size() must cover the partial rows such a plan reserves."""
    page_size, qo_len, split_pages = 16, 4, 2
    inputs = _make_inputs(BATCH_SIZE, KV_LEN, qo_len, page_size, split_pages)
    _skip_unless_counts_differ(inputs[-1], page_size, split_pages)
    q, kv_cache, qo_indptr, kv_indptr, kv_indices, last_page_len, _ = inputs

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    reference = _run(workspace, inputs, page_size, split_pages=None)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="fa2"
    )
    float_size, int_size = wrapper.workspace_size(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        page_size,
        **_plan_kwargs(split_pages),
    )
    wrapper.reset_workspace_buffer(
        torch.zeros(float_size, dtype=torch.uint8, device=DEVICE),
        torch.zeros(int_size, dtype=torch.uint8, device=DEVICE),
    )
    _plan(wrapper, inputs, page_size, split_pages)
    _assert_close(wrapper.run(q, kv_cache).float(), reference)


@pytest.mark.parametrize("qo_len", [1, 4])
@pytest.mark.parametrize("split_tokens", [64, 128])
def test_batch_prefill_ragged_split_kv_matches_unsplit(qo_len, split_tokens):
    """The ragged planner gets KV lengths in tokens, so it cannot over-reserve.

    This pins the split-KV path there to the unsplit result.
    """
    kv_len = KV_LEN
    torch.manual_seed(0)
    q = torch.randn(
        BATCH_SIZE * qo_len, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    k = torch.randn(
        BATCH_SIZE * kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    v = torch.randn(
        BATCH_SIZE * kv_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE
    )
    qo_indptr = torch.arange(
        0, (BATCH_SIZE + 1) * qo_len, qo_len, dtype=torch.int32, device=DEVICE
    )
    kv_indptr = torch.arange(
        0, (BATCH_SIZE + 1) * kv_len, kv_len, dtype=torch.int32, device=DEVICE
    )
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    def run(split):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            **_plan_kwargs(split),
        )
        return wrapper.run(q, k, v).float()

    _assert_close(run(split_tokens), run(None))
