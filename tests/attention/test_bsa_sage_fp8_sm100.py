"""SM100 blk64 physical tails, sparse padding, scheduling and graph regression."""

import math

import pytest
import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
from flashinfer.utils import get_compute_capability, is_sm100a_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7))
    or not is_sm100a_supported(torch.device("cuda")),
    reason="requires SM100-family blk64 attention",
)


def inputs(sq, sk, fp8=True, variable=False, sizes_rank=0):
    torch.manual_seed(22)
    h, d = 4, 128
    q, k, v = [torch.randn(1, s, h, d, device="cuda") for s in (sq, sk, sk)]
    kwargs = {}
    if fp8:
        qs = q.abs().amax(-1).clamp_min(1e-6) / 448
        kp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, (-sk) % 16))
        ks = kp.reshape(1, -1, 16, h, d).abs().amax((2, 4)).clamp_min(1e-6) / 448
        vs = v.abs().amax((0, 1)).clamp_min(1e-6) / 448
        q = (q / qs[..., None]).to(torch.float8_e4m3fn)
        k = (k / ks.repeat_interleave(16, 1)[:, :sk, :, None]).to(torch.float8_e4m3fn)
        v = (v / vs).to(torch.float8_e4m3fn)
        kwargs = dict(
            q_scale=qs.transpose(1, 2).contiguous(),
            k_scale=ks.transpose(1, 2).contiguous(),
            v_scale=vs,
        )
    else:
        q, k, v = [x.to(torch.bfloat16) for x in (q, k, v)]
    nq, nk = (sq + 63) // 64, (sk + 63) // 64
    # Reverse order exercises physical tails independently of traversal order.
    index = torch.arange(nk - 1, -1, -1, device="cuda", dtype=torch.int32)
    index = index.expand(1, h, nq, nk).contiguous()
    if variable:
        counts = torch.arange(h * nq, device="cuda", dtype=torch.int32)
        counts = (counts % (nk + 1)).reshape(1, h, nq)
        kwargs["q2k_block_nums"] = counts
        # Unused index slots must never be dereferenced (including empty rows).
        index[torch.arange(nk, device="cuda") >= counts[..., None]] = 2**30
    if sizes_rank:
        shape = {1: (nk,), 2: (1, nk), 3: (1, h, nk)}[sizes_rank]
        sizes = torch.full(shape, 64, device="cuda", dtype=torch.int32)
        sizes[..., 0] = 37
        # Last size intentionally remains 64, even when physical tail is shorter.
        if sizes_rank == 3:
            sizes[0, :, 0] = torch.tensor([0, 17, 37, 64], device="cuda")
            sizes[0, 0, -1] = 0  # Include a nonempty sparse row with no valid tokens.
        kwargs["block_sizes"] = sizes
    return q, k, v, index, kwargs


def reference(q, k, v, index, kwargs):
    sq, sk, h = q.shape[1], k.shape[1], q.shape[2]
    q, k, v = [x.float() for x in (q, k, v)]
    if "q_scale" in kwargs:
        q = q * kwargs["q_scale"].transpose(1, 2)[..., None]
        k = (
            k
            * kwargs["k_scale"]
            .transpose(1, 2)
            .repeat_interleave(16, 1)[:, :sk, :, None]
        )
        v = v * kwargs["v_scale"]
    mask = torch.zeros(1, h, sq, sk, device="cuda", dtype=torch.bool)
    idx = index.cpu()
    counts = kwargs.get("q2k_block_nums")
    counts = counts.cpu() if counts is not None else None
    sizes = kwargs.get("block_sizes")
    sizes = sizes.cpu() if sizes is not None else None
    for head in range(h):
        for row in range(index.shape[2]):
            count = int(counts[0, head, row]) if counts is not None else index.shape[-1]
            for block in idx[0, head, row, :count].tolist():
                size = 64
                if sizes is not None:
                    size = int(
                        sizes[block]
                        if sizes.ndim == 1
                        else (
                            sizes[0, block]
                            if sizes.ndim == 2
                            else sizes[0, head, block]
                        )
                    )
                mask[
                    0,
                    head,
                    row * 64 : min((row + 1) * 64, sq),
                    block * 64 : min(block * 64 + size, sk),
                ] = True
    scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) / math.sqrt(128)
    scores.masked_fill_(~mask, -torch.inf)
    lse = torch.logsumexp(scores, -1)
    probs = torch.softmax(scores, -1).nan_to_num()
    return (probs @ v.transpose(1, 2)).transpose(1, 2), lse


def check(case, splits=1, clc=False):
    q, k, v, index, kwargs = case
    expected, expected_lse = reference(*case)
    out = torch.full(q.shape, torch.nan, device="cuda", dtype=torch.bfloat16)
    lse = torch.full(expected_lse.shape, torch.nan, device="cuda")
    actual, actual_lse = bsa_attn_sm100_blk64_fwd(
        q,
        k,
        v,
        index,
        index.shape[-1],
        **kwargs,
        kv_splits=splits,
        use_clc=clc,
        backend="cute",
        return_lse=True,
        out=out,
        lse=lse,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == lse.data_ptr()
    assert torch.isfinite(actual).all()
    tol = 0.04 if "q_scale" in kwargs else 0.015
    torch.testing.assert_close(actual.float(), expected, atol=tol, rtol=tol)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-3, rtol=2e-3)
    empty = torch.isneginf(expected_lse)
    assert (actual.transpose(1, 2)[empty] == 0).all()
    return actual.clone(), actual_lse.clone()


@pytest.mark.parametrize("sk", [1, 63, 64, 65, 127, 128, 193])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("splits,clc", [(1, False), (1, True), (4, False)])
def test_physical_tail(sk, fp8, splits, clc):
    check(inputs(100, sk, fp8), splits, clc)


@pytest.mark.parametrize("sq", [1, 63, 65])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("splits", [1, 4])
def test_short_query_tail(sq, fp8, splits):
    check(inputs(sq, 65, fp8), splits, False)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("splits", [1, 2, 4, 8, 16])
def test_variable_counts_and_padding(rank, splits):
    case = inputs(193, 193, variable=True, sizes_rank=rank)
    check(case, splits, False)
    if splits == 1:
        a, la = check(case, 1, True)
        b, lb = check(case, 1, False)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(la, lb, atol=0, rtol=0)


@pytest.mark.parametrize("fp8", [False, True])
def test_persistent_multiwave(fp8):
    # More Q tiles than SMs, with empty rows and a short physical last block.
    case = inputs(4161, 193, fp8=fp8, variable=True, sizes_rank=3)
    a, la = check(case, 1, True)
    b, lb = check(case, 1, False)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(la, lb, atol=0, rtol=0)


@pytest.mark.parametrize("fp8", [False, True])
def test_split_clc_alias_reuse(fp8):
    case = inputs(4161, 193, fp8=fp8, variable=True, sizes_rank=3)
    a, la = check(case, 4, True)
    b, lb = check(case, 4, False)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(la, lb, atol=0, rtol=0)


@pytest.mark.parametrize("splits", [1, 4, "auto"])
def test_graph_replay_changed_metadata(splits):
    case = inputs(193, 193, variable=True, sizes_rank=1)
    q, k, v, index, kwargs = case
    # Keep all indices valid as counts change on replay.
    index.copy_(torch.arange(3, -1, -1, device="cuda", dtype=torch.int32))
    # initcheck does not track TMA tensor stores. NaN sentinels independently
    # prove every logical output is written, including on each graph replay.
    out = torch.full(q.shape, torch.nan, device="cuda", dtype=torch.bfloat16)
    lse = torch.full((1, 4, 193), torch.nan, device="cuda")

    def fn():
        return bsa_attn_sm100_blk64_fwd(
            q,
            k,
            v,
            index,
            4,
            **kwargs,
            backend="cute",
            kv_splits=splits,
            out=out,
            lse=lse,
            return_lse=True,
        )

    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for count in (1, 4, 0, 2):
        q.copy_((q.float() * 0.75).to(q.dtype))
        kwargs["q_scale"].mul_(1.1)
        kwargs["k_scale"].mul_(0.9)
        kwargs["v_scale"].mul_(1.05)
        kwargs["q2k_block_nums"].fill_(count)
        kwargs["block_sizes"][0] = 17 + count
        index.copy_(index.flip(-1))
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        expected, expected_lse = reference(*case)
        torch.testing.assert_close(out.float(), expected, atol=0.04, rtol=0.04)
        torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("splits", [1, 2, 4])
@pytest.mark.parametrize("sk", [65, 127, 193])
def test_wrapper_partial_kv_tail(fp8, splits, sk):
    from flashinfer.sparse import BlockSparseAttentionWrapper

    q, k, v, index, kwargs = inputs(128, sk, fp8)
    expected, _ = reference(q, k, v, index, kwargs)
    workspace = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="vsa_sm100_blk64")
    nk = (sk + 63) // 64
    mask = torch.ones(4, 2, nk, device="cuda", dtype=torch.bool)
    wrapper.plan(
        None,
        None,
        128,
        sk,
        64,
        64,
        4,
        4,
        128,
        block_mask=mask,
        q_data_type=q.dtype,
        kv_splits=splits,
        use_clc=False,
        **kwargs,
    )
    actual = wrapper.run(q[0], k[0], v[0])
    tol = 0.04 if fp8 else 0.015
    torch.testing.assert_close(actual.float(), expected[0], atol=tol, rtol=tol)


def test_auto_split_uses_active_counts_and_capture_bound(monkeypatch):
    from flashinfer.cute_dsl.sparse import bsa_attn_sm100_blk64 as module

    q, k, v, index, kwargs = inputs(64, 16384, variable=True)
    index.copy_(torch.arange(255, -1, -1, device="cuda", dtype=torch.int32))
    kwargs["q2k_block_nums"].fill_(32)
    monkeypatch.setattr(module, "get_sm100_device_num_sms", lambda _: 160)
    original = module.resolve_sm100_blk64_split_workspace
    selected = []

    def resolve(q, dim, splits, **kw):
        selected.append(splits)
        return original(q, dim, splits, **kw)

    monkeypatch.setattr(module, "resolve_sm100_blk64_split_workspace", resolve)
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)

    def fn(splits):
        return bsa_attn_sm100_blk64_fwd(
            q,
            k,
            v,
            index,
            256,
            **kwargs,
            backend="cute",
            kv_splits=splits,
            out=out,
        )

    fn("auto")
    assert selected[-1] == 4  # 32 active blocks, rather than index capacity 256.
    fn(16)  # Warm the conservative capture specialization before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn("auto")
    assert selected[-1] == 16  # No GPU-to-host counts readback during capture.
    graph.replay()
    expected, _ = reference(q, k, v, index, kwargs)
    torch.testing.assert_close(out.float(), expected, atol=0.04, rtol=0.04)


@pytest.mark.parametrize("layout", ["offset", "strided"])
@pytest.mark.parametrize("splits", [1, 4])
def test_preallocated_output_layout(layout, splits):
    case = inputs(100, 65)
    q, k, v, index, kwargs = case
    if layout == "offset":
        backing = torch.empty(q.numel() + 1, device="cuda", dtype=torch.bfloat16)
        out = backing[1:].view(q.shape)
    else:
        out = torch.empty((*q.shape, 2), device="cuda", dtype=torch.bfloat16)[..., 0]
    actual, _ = bsa_attn_sm100_blk64_fwd(
        q,
        k,
        v,
        index,
        2,
        **kwargs,
        backend="cute",
        out=out,
        kv_splits=splits,
    )
    assert actual.data_ptr() == out.data_ptr()
    expected, _ = reference(*case)
    torch.testing.assert_close(out.float(), expected, atol=0.04, rtol=0.04)
