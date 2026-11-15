# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public dispatch and numerical contracts for SM107 ragged FMHA."""

import functools
import itertools
import math

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL unavailable", allow_module_level=True)

from flashinfer.attention.cute_dsl import fmha as wrapper
from flashinfer.cute_dsl.attention.fmha.compile_rubin import compile_rubin_fmha_kernel
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin FMHA requires SM107",
)

HEADS = 16
SCALE_Q, SCALE_K, SCALE_V, SCALE_O = 0.75, 1.25, 0.625, 1.5
SM_SCALE = 1 / math.sqrt(192)


@pytest.fixture(autouse=True)
def fp32_reference_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous


def compiler_args(**overrides):
    args = dict(
        qk_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim=192,
        head_dim_v=128,
        is_causal=False,
        with_lse=True,
        enable_sink=False,
        enable_skip_softmax=False,
        use_pdl=False,
        device=torch.device("cuda"),
    )
    args.update(overrides)
    return args


def indptr(lengths):
    return torch.tensor(
        [0] + list(itertools.accumulate(lengths)), dtype=torch.int32, device="cuda"
    )


def inputs(q_lens, kv_lens, offset=False):
    generator = torch.Generator(device="cuda").manual_seed(31)

    def tensor(tokens, dim):
        # A leading token is a nonzero, suitably aligned storage offset while
        # retaining the compact NHD strides required by the compiler ABI.
        storage = torch.randn(
            tokens + int(offset), HEADS, dim, device="cuda", generator=generator
        ).to(torch.float8_e4m3fn)
        return storage[int(offset) :]

    return (
        tensor(sum(q_lens), 192),
        tensor(sum(kv_lens), 192),
        tensor(sum(kv_lens), 128),
    )


def reference(
    q,
    k,
    v,
    q_lens,
    kv_lens,
    causal,
    *,
    sm_scale=SM_SCALE,
    q_scale=SCALE_Q,
    k_scale=SCALE_K,
    v_scale=SCALE_V,
    o_scale=SCALE_O,
):
    out = torch.zeros((*q.shape[:2], 128), device=q.device, dtype=torch.float32)
    lse = torch.full(q.shape[:2], -float("inf"), device=q.device, dtype=torch.float32)
    q_offset = kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        if q_len and kv_len:
            qs = slice(q_offset, q_offset + q_len)
            ks = slice(kv_offset, kv_offset + kv_len)
            logits = torch.einsum("qhd,khd->hqk", q[qs].float(), k[ks].float())
            logits *= q_scale * k_scale * sm_scale
            if causal:
                query_position = torch.arange(q_len, device=q.device)[:, None]
                key_position = torch.arange(kv_len, device=q.device)[None, :]
                logits.masked_fill_(
                    key_position > query_position + kv_len - q_len, -float("inf")
                )
            log_norm = torch.logsumexp(logits, dim=-1)
            # A fully masked query has O=0, LSE=-inf. Avoid softmax(-inf,...)
            # NaNs explicitly; this is distinct from an empty KV sequence.
            probabilities = torch.where(
                torch.isfinite(log_norm)[..., None], torch.softmax(logits, dim=-1), 0
            )
            out[qs] = torch.einsum("hqk,khd->qhd", probabilities, v[ks].float()) * (
                v_scale / o_scale
            )
            lse[qs] = log_norm.transpose(0, 1) / math.log(2)
        q_offset += q_len
        kv_offset += kv_len
    return out, lse


def assert_outputs(out, lse, expected, expected_lse):
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.05, rtol=0.05)
    relative_l2 = torch.linalg.vector_norm(
        out.float() - expected
    ) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    assert relative_l2.item() <= 0.04
    if lse is not None:
        assert not torch.isnan(lse).any()
        assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
        active = torch.isfinite(expected_lse)
        torch.testing.assert_close(
            lse[active], expected_lse[active], atol=0.05, rtol=0.005
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_lse", [False, True])
@pytest.mark.parametrize(
    "q_lens,kv_lens,offset",
    [([1], [1], False), ([128], [256], False), ([127, 129, 17], [129, 257, 193], True)],
)
def test_rubin_ragged_reference(causal, with_lse, q_lens, kv_lens, offset):
    q, k, v = inputs(q_lens, kv_lens, offset)
    out_storage = torch.empty(
        (q.shape[0] + int(offset), HEADS, 128), device=q.device, dtype=torch.bfloat16
    )
    out = out_storage[int(offset) :]
    out.fill_(float("nan"))
    lse = torch.full(q.shape[:2], float("nan"), device=q.device) if with_lse else None
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=causal)
    result = attn.run(
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=with_lse,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    assert (result[0] if with_lse else result) is out
    if with_lse:
        assert result[1] is lse
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, causal))


@pytest.mark.parametrize("q_lens,kv_lens", [([0, 3, 5], [8, 0, 9]), ([3, 5], [0, 0])])
@pytest.mark.parametrize("with_lse", [False, True])
def test_batch_empty_rows(q_lens, kv_lens, with_lse):
    q, k, v = inputs(q_lens, kv_lens)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    lse = torch.full(q.shape[:2], 19.0, device=q.device) if with_lse else None
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True)
    result = attn.run(
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=with_lse,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    assert (result[0] if with_lse else result) is out
    if with_lse:
        assert result[1] is lse
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))


@pytest.mark.parametrize(
    "q_len,kv_len,with_lse",
    [
        (17, 1, False),
        (128, 17, True),
        (129, 17, False),
        (257, 17, True),
        (271, 16, False),
        (271, 16, True),
        (272, 16, False),
        (272, 16, True),
        (273, 16, False),
        (273, 16, True),
        (528, 16, False),
        (528, 16, True),
    ],
)
def test_batch_fully_masked_queries(q_len, kv_len, with_lse):
    """Cover partial tiles and one or more wholly masked 256-row thread blocks."""
    q_lens, kv_lens = [q_len], [kv_len]
    q, k, v = inputs(q_lens, kv_lens)
    out = torch.full(
        (*q.shape[:2], 128), float("nan"), device=q.device, dtype=torch.bfloat16
    )
    lse = torch.full(q.shape[:2], float("nan"), device=q.device) if with_lse else None
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True)
    result = attn.run(
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=with_lse,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    assert (result[0] if with_lse else result) is out
    if with_lse:
        assert result[1] is lse
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))

    masked_rows = q_len - kv_len
    assert torch.equal(out[:masked_rows], torch.zeros_like(out[:masked_rows]))
    if with_lse:
        assert torch.isneginf(lse[:masked_rows]).all()


def batch_wrapper(q_lens, kv_lens, *, causal, graph=False):
    q_ptr, kv_ptr = indptr(q_lens), indptr(kv_lens)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
        use_cuda_graph=graph,
        qo_indptr_buf=q_ptr if graph else None,
        kv_indptr_buf=kv_ptr if graph else None,
    )
    attn.plan(
        q_ptr,
        kv_ptr,
        HEADS,
        HEADS,
        192,
        head_dim_vo=128,
        causal=causal,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    return attn, q_ptr, kv_ptr


def warmup_batch_graph(*, causal, with_lse):
    q, k, v = inputs([1], [128])
    attn, _, _ = batch_wrapper([1], [128], causal=causal)
    attn.run(q, k, v, return_lse=with_lse)


def review_plan(attn, q_lens, kv_lens, **overrides):
    params = dict(
        qo_indptr=indptr(q_lens),
        kv_indptr=indptr(kv_lens),
        num_qo_heads=HEADS,
        num_kv_heads=HEADS,
        head_dim_qk=192,
        head_dim_vo=128,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    params.update(overrides)
    attn.plan(**params)


def test_batch_pdl_hint():
    q_lens, kv_lens = [0, 3, 17, 129], [9, 0, 1, 257]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True)
    out = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device)
    result, result_lse = attn.run(
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
        enable_pdl=True,
    )
    assert result is out and result_lse is lse
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))


@pytest.mark.parametrize(
    "q_lens,initial_kv",
    [([3, 5], [64, 64]), ([3, 5], [128, 0]), ([0, 8], [128, 0])],
)
def test_batch_graph_retains_ragged_lengths(q_lens, initial_kv):
    q, k, v = inputs(q_lens, initial_kv)
    attn, _, kv_ptr = batch_wrapper(q_lens, initial_kv, causal=True, graph=True)
    out = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device)
    call = functools.partial(
        attn.run,
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # All-inactive metadata never launches eagerly; warm the same compiled
        # specialization with a positive Q/KV sequence before capture.
        warmup_batch_graph(causal=True, with_lse=True)
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for lengths in ([0, 128], [128, 0], [64, 64]):
        kv_ptr.copy_(indptr(lengths))
        out.fill_(19)
        lse.fill_(19)
        graph.replay()
        assert_outputs(out, lse, *reference(q, k, v, q_lens, lengths, True))


def test_batch_graph_all_empty_kv():
    q_lens, kv_lens = [3, 5], [0, 0]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True, graph=True)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    call = functools.partial(
        attn.run,
        q,
        k,
        v,
        out=out,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup_batch_graph(causal=True, with_lse=False)
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    out.fill_(19)
    graph.replay()
    assert_outputs(out, None, *reference(q, k, v, q_lens, kv_lens, True))


@pytest.mark.parametrize("causal", [False, True])
def test_batch_graph_repartitions_queries(causal):
    q_lens, kv_lens = [256, 256], [256, 256]
    q, k, v = inputs(q_lens, kv_lens)
    attn, q_ptr, kv_ptr = batch_wrapper(q_lens, kv_lens, causal=causal, graph=True)
    out = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device)
    call = functools.partial(
        attn.run,
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for q_lens, kv_lens in (
        ([512, 0], [256, 256]),
        ([0, 512], [512, 0]),
        ([0, 512], [0, 512]),
        ([512, 0], [0, 512]),
    ):
        # Storage and addresses stay fixed while either sequence grows beyond
        # its capture-time Q length, including transitions to/from empty KV.
        q_ptr.copy_(indptr(q_lens))
        kv_ptr.copy_(indptr(kv_lens))
        out.fill_(19)
        lse.fill_(19)
        graph.replay()
        assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, causal))


@pytest.mark.parametrize("new_kv_total", [64, 192], ids=["shrink", "grow"])
@pytest.mark.parametrize("capture_first", [False, True], ids=["before", "after"])
def test_batch_graph_kv_total_is_fixed(new_kv_total, capture_first):
    q_lens, kv_lens = [3, 5], [64, 64]
    q, k, v = inputs(q_lens, kv_lens)
    attn, q_ptr, kv_ptr = batch_wrapper(q_lens, kv_lens, causal=True, graph=True)
    original_q_ptr, original_kv_ptr = q_ptr.clone(), kv_ptr.clone()
    out = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device)
    call = functools.partial(
        attn.run,
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )

    def capture():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        return graph

    def replan(new_q_lens, new_kv_lens):
        attn.plan(
            indptr(new_q_lens),
            indptr(new_kv_lens),
            HEADS,
            HEADS,
            192,
            head_dim_vo=128,
            causal=True,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )

    graph = capture() if capture_first else None
    # Reject shrinking before capture too: otherwise shrink/capture/grow could
    # satisfy an initial-capacity bound but exceed the captured K/V descriptors.
    # If rejection is missing, stop here rather than replay unsafe metadata.
    with pytest.raises(ValueError, match="(?i)(fixed|total.*KV|KV.*total)"):
        replan([5, 3], [0, new_kv_total])
    torch.testing.assert_close(q_ptr, original_q_ptr)
    torch.testing.assert_close(kv_ptr, original_kv_ptr)
    out.fill_(19)
    lse.fill_(19)
    call()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))
    if graph is None:
        graph = capture()
    out.fill_(19)
    lse.fill_(19)
    graph.replay()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))

    # Fixed storage still permits redistributing both query and KV sequences.
    q_lens, kv_lens = [5, 3], [0, 128]
    replan(q_lens, kv_lens)
    out.fill_(19)
    lse.fill_(19)
    graph.replay()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))


@pytest.mark.parametrize("kv_len", [0, 17])
def test_batch_strided_output_rejected_without_writes(kv_len):
    q_lens, kv_lens = [17], [kv_len]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    backing = torch.full(
        (q.shape[0], HEADS, 256), 19.0, device=q.device, dtype=torch.bfloat16
    )
    out = backing[..., ::2]
    lse = torch.full(q.shape[:2], 19.0, device=q.device)
    original_inputs = [tensor.view(torch.uint8).clone() for tensor in (q, k, v)]
    # The backing also accommodates compact stores, keeping a missing guard
    # observable without an out-of-bounds write in the active-KV case.
    with pytest.raises(ValueError, match="contiguous.*out|out.*contiguous"):
        attn.run(q, k, v, out=out, lse=lse, return_lse=True)
    torch.testing.assert_close(backing, torch.full_like(backing, 19.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 19.0))
    for tensor, original in zip((q, k, v), original_inputs, strict=True):
        assert torch.equal(tensor.view(torch.uint8), original)


@pytest.mark.parametrize(
    "destination,source,lse_layout",
    [
        ("out", "q", "NH"),
        ("out", "k", "NH"),
        ("out", "v", "NH"),
        ("lse", "q", "NH"),
        ("lse", "k", "NH"),
        ("lse", "v", "NH"),
        ("lse", "out", "NH"),
        ("lse", "q", "HN"),
        ("lse", "out", "HN"),
    ],
)
def test_batch_output_alias_rejected_before_neutralization(
    destination, source, lse_layout
):
    # Both sequences are inactive, but Q, K and V all have nonempty storage.
    # Missing validation can only neutralize output; it cannot launch a kernel
    # that races aliased input and output accesses.
    q_lens, kv_lens = [3, 0], [0, 8]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    lse_shape = q.shape[:2] if lse_layout == "NH" else (q.shape[1], q.shape[0])
    tensors = dict(
        q=q,
        k=k,
        v=v,
        out=torch.full(
            (*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16
        ),
        lse=torch.full(lse_shape, 19.0, device=q.device),
    )
    backing = torch.full(
        (max(tensors[source].nbytes, tensors[destination].nbytes),),
        19,
        dtype=torch.uint8,
        device=q.device,
    )
    for name in (source, destination):
        tensor = tensors[name]
        tensors[name] = backing[: tensor.nbytes].view(tensor.dtype).view(tensor.shape)
    originals = {
        name: tensor.view(torch.uint8).clone() for name, tensor in tensors.items()
    }
    original_backing = backing.clone()
    with pytest.raises(ValueError, match="overlap"):
        attn.run(**tensors, return_lse=True, lse_layout=lse_layout)
    assert torch.equal(backing, original_backing)
    for name, tensor in tensors.items():
        assert torch.equal(tensor.view(torch.uint8), originals[name])


def test_batch_disjoint_views_of_shared_storage():
    q_lens, kv_lens = [3], [17]
    q, k, v = inputs(q_lens, kv_lens)
    tensors = [
        q,
        k,
        v,
        torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16),
        torch.empty(q.shape[:2], device=q.device),
    ]
    backing = torch.empty(
        sum(tensor.nbytes for tensor in tensors), dtype=torch.uint8, device=q.device
    )
    offset = 0
    shared = []
    for tensor in tensors:
        view = (
            backing[offset : offset + tensor.nbytes]
            .view(tensor.dtype)
            .view(tensor.shape)
        )
        shared.append(view)
        offset += tensor.nbytes
    for destination, source in zip(shared[:3], (q, k, v), strict=True):
        destination.copy_(source)
    q, k, v, out, lse = shared
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    result = attn.run(
        q,
        k,
        v,
        out=out,
        lse=lse,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    assert result[0] is out and result[1] is lse
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, False))


@pytest.mark.parametrize("kv_len", [0, 17])
def test_batch_strided_lse_rejected_without_writes(kv_len):
    q_lens, kv_lens = [17], [kv_len]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    backing = torch.full(
        (q.shape[0], 2 * HEADS), 19.0, device=q.device, dtype=torch.float32
    )
    lse = backing[:, ::2]
    with pytest.raises(ValueError, match="contiguous lse"):
        attn.run(q, k, v, out=out, lse=lse, return_lse=True)
    # Check both the requested LSE view and the interleaved sibling storage.
    torch.testing.assert_close(backing, torch.full_like(backing, 19.0))
    torch.testing.assert_close(out, torch.full_like(out, 19.0))


def test_batch_empty_kv_still_validates_plan():
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    # Validate even when every KV row is empty and eager run would skip launch.
    with pytest.raises(ValueError, match="Rubin"):
        attn.plan(
            indptr([3]),
            indptr([0]),
            HEADS,
            HEADS,
            192,
            head_dim_vo=128,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )


def test_batch_empty_kv_rejects_attention_sinks():
    q_lens, kv_lens = [3], [0]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True)
    attn._sinks = torch.zeros(HEADS, device=q.device)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    lse = torch.full(q.shape[:2], 19.0, device=q.device)
    with pytest.raises(NotImplementedError, match="Rubin.*sinks"):
        attn.run(q, k, v, out=out, lse=lse, return_lse=True)
    torch.testing.assert_close(out, torch.full_like(out, 19.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 19.0))


def test_capability_guard(monkeypatch):
    # A cached success must not bypass capability validation on a later call.
    compile_rubin_fmha_kernel(**compiler_args())
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 0)
    )
    with pytest.raises(ValueError, match="Rubin FMHA"):
        compile_rubin_fmha_kernel(**compiler_args())


def test_output_alignment_guard():
    q_lens, kv_lens = [1], [1]
    q, k, v = inputs(q_lens, kv_lens)
    storage = torch.empty(HEADS * 128 + 1, dtype=torch.bfloat16, device=q.device)
    out = storage[1:].view(1, HEADS, 128)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    with pytest.raises(AssertionError, match="32-byte aligned"):
        attn.run(q, k, v, out=out)


def test_kernel_construction_uses_requested_device(monkeypatch):
    """Check context ordering with a real compile, without requiring two GPUs."""
    from flashinfer.cute_dsl.attention.fmha.rubin.kernel import (
        RubinFusedMultiHeadAttentionForward,
    )

    requested = torch.device("cuda", torch.cuda.current_device())
    original_device = torch.cuda.device
    original_init = RubinFusedMultiHeadAttentionForward.__init__
    active_contexts = []
    constructed_on = []

    class DeviceSpy(original_device):
        # PyTorch itself uses isinstance(value, torch.cuda.device), so keep
        # the patched context manager a type rather than a generator function.
        def __enter__(self):
            result = super().__enter__()
            active_contexts.append(self.idx)
            return result

        def __exit__(self, *exc_info):
            try:
                return super().__exit__(*exc_info)
            finally:
                active_contexts.pop()

    def constructor_spy(self, *args, **kwargs):
        # Merely checking current_device is insufficient on a single GPU: it
        # also matches before entering the required explicit device context.
        assert requested.index in active_contexts
        assert torch.cuda.current_device() == requested.index
        constructed_on.append(requested)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "device", DeviceSpy)
    monkeypatch.setattr(
        RubinFusedMultiHeadAttentionForward, "__init__", constructor_spy
    )
    # Other successful compilations in this module use 16 heads, so this
    # configuration reaches the real constructor and compiler on a cache miss.
    handle = compile_rubin_fmha_kernel(
        **compiler_args(num_qo_heads=32, num_kv_heads=32, device=requested)
    )
    assert callable(handle)
    assert constructed_on == [requested]
    assert not active_contexts


def test_generic_cute_dsl_keeps_existing_artifact_route(monkeypatch):
    from flashinfer.cute_dsl.attention.fmha import compile_rubin

    class ArtifactLookupReached(Exception):
        pass

    def forbidden(*args, **kwargs):
        pytest.fail("Generic CuTe call selected Rubin")

    observed = []

    def artifact_lookup(*args, **kwargs):
        observed.append((args, kwargs))
        raise ArtifactLookupReached

    monkeypatch.setattr(
        wrapper, "_get_gpu_arch", lambda device: "existing-architecture"
    )
    monkeypatch.setattr(wrapper, "get_cute_dsl_fmha_kernel", artifact_lookup)
    monkeypatch.setattr(compile_rubin, "compile_rubin_fmha_kernel", forbidden)
    q, k, v = inputs([1], [1])
    out = torch.empty(1, HEADS, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ArtifactLookupReached):
        wrapper.cute_dsl_fmha_ragged_prefill(
            q, k, v, out, indptr([1]), indptr([1]), max_qo_len=1, max_kv_len=1
        )
    assert len(observed) == 1
    args, _ = observed[0]
    assert args[:5] == (
        "existing-architecture",
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        torch.bfloat16,
        192,
    )


@pytest.mark.parametrize(
    "change",
    ["q-shrink", "q-grow", "batch", "heads", "causal", "sm-scale", "head-dim"],
)
def test_review_graph_replan_preserves_execution(change):
    q_lens, kv_lens = [3, 5], [0, 128]
    q, k, v = inputs(q_lens, kv_lens)
    attn, q_ptr, kv_ptr = batch_wrapper(q_lens, kv_lens, causal=True, graph=True)
    original_q, original_kv = q_ptr.clone(), kv_ptr.clone()
    scales = dict(q_scale=SCALE_Q, k_scale=SCALE_K, v_scale=SCALE_V, o_scale=SCALE_O)
    out = torch.empty((*q.shape[:2], 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:2], device=q.device)
    call = functools.partial(
        attn.run, q, k, v, out=out, lse=lse, return_lse=True, **scales
    )
    new_q, new_kv, overrides = [5, 3], [128, 0], {}
    if change == "q-shrink":
        new_q = [1, 3]
    elif change == "q-grow":
        new_q = [5, 4]
    elif change == "batch":
        new_q, new_kv = [7], [128]
    elif change == "heads":
        overrides = dict(num_qo_heads=8, num_kv_heads=8)
    elif change == "causal":
        overrides = dict(causal=False)
    elif change == "sm-scale":
        overrides = dict(sm_scale=0.5 * SM_SCALE)
    elif change == "head-dim":
        overrides = dict(head_dim_vo=64)

    def reject_replan():
        # A missing refusal stops the test before unsafe metadata is replayed.
        with pytest.raises(ValueError):
            review_plan(attn, new_q, new_kv, **overrides)
        torch.testing.assert_close(q_ptr, original_q)
        torch.testing.assert_close(kv_ptr, original_kv)
        # Implicit allocation also exercises cached row count and head dimension.
        actual, actual_lse = attn.run(q, k, v, return_lse=True, **scales)
        assert_outputs(actual, actual_lse, *reference(q, k, v, q_lens, kv_lens, True))

    reject_replan()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    reject_replan()
    out.fill_(19)
    lse.fill_(19)
    graph.replay()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))

    # Explicit/default scale equivalence and fixed-total redistribution remain valid.
    q_lens, kv_lens = [5, 3], [128, 0]
    review_plan(attn, q_lens, kv_lens, sm_scale=SM_SCALE)
    call()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))
    out.fill_(19)
    lse.fill_(19)
    graph.replay()
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))


def test_review_equal_head_batch_grid_rejection_preserves_plan():
    q_lens, kv_lens = [3], [128]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=True)
    # Metadata alone exercises the invalid launch grid, without a large Q/K/V
    # allocation or launching the unsupported batch on the baseline.
    ptr = torch.arange(65537, dtype=torch.int32)
    with pytest.raises(ValueError, match="(?i)(batch|65535)"):
        review_plan(attn, [], [], qo_indptr=ptr, kv_indptr=ptr)
    out, lse = attn.run(
        q,
        k,
        v,
        return_lse=True,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, True))


@pytest.mark.parametrize(
    "plan_scale,scales",
    [
        (1e39, {}),
        (SM_SCALE, {"q_scale": -1e39}),
        (SM_SCALE, {"k_scale": math.inf}),
        (SM_SCALE, {"v_scale": math.nan}),
        (SM_SCALE, {"o_scale": 1e39}),
        (1e38, {"q_scale": 4.0}),
        (3e38, {}),
        (SM_SCALE, {"v_scale": 3e38, "o_scale": 0.5}),
        (SM_SCALE, {"o_scale": 0.0}),
    ],
    ids=[
        "sm-input",
        "q-input",
        "k-inf",
        "v-nan",
        "o-input",
        "score",
        "base2",
        "output",
        "zero-divisor",
    ],
)
def test_review_equal_head_invalid_scales_before_writes(plan_scale, scales):
    # Eager all-inactive rows avoid launching an invalid scale on the baseline.
    q, k, v = inputs([3], [0])
    attn, _, _ = batch_wrapper([3], [0], causal=True)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    lse = torch.full(q.shape[:2], 19.0, device=q.device)
    with pytest.raises(ValueError, match="(?i)scale"):
        review_plan(attn, [3], [0], sm_scale=plan_scale)
        attn.run(q, k, v, out=out, lse=lse, return_lse=True, **scales)
    torch.testing.assert_close(out, torch.full_like(out, 19))
    torch.testing.assert_close(lse, torch.full_like(lse, 19))


@pytest.mark.parametrize(
    "sm_scale,q_scale,k_scale",
    [
        (-SM_SCALE, SCALE_Q, SCALE_K),
        (0.0, SCALE_Q, SCALE_K),
        (SM_SCALE, 0.0, SCALE_K),
        (SM_SCALE, SCALE_Q, 0.0),
        (1e-50, SCALE_Q, SCALE_K),
        (SM_SCALE, 1e-30, 1e-30),
    ],
    ids=[
        "negative-score",
        "zero-sm",
        "zero-q",
        "zero-k",
        "underflow-sm",
        "underflow-product",
    ],
)
@pytest.mark.parametrize("kv_len", [0, 128], ids=["inactive", "active"])
def test_review_equal_head_nonpositive_score_rejected_before_writes(
    sm_scale, q_scale, k_scale, kv_len
):
    q_lens, kv_lens = [3], [kv_len]
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=False)
    # The effective score depends on runtime Q/K factors, so plan accepts the
    # individual finite sm_scale and run must reject before either output write.
    review_plan(attn, q_lens, kv_lens, causal=False, sm_scale=sm_scale)
    out = torch.full((*q.shape[:2], 128), 19.0, device=q.device, dtype=torch.bfloat16)
    lse = torch.full(q.shape[:2], 19.0, device=q.device)
    with pytest.raises(ValueError, match="(?i)scale"):
        attn.run(
            q,
            k,
            v,
            out=out,
            lse=lse,
            return_lse=True,
            q_scale=q_scale,
            k_scale=k_scale,
        )
    torch.testing.assert_close(out, torch.full_like(out, 19))
    torch.testing.assert_close(lse, torch.full_like(lse, 19))


@pytest.mark.parametrize("with_lse", [False, True])
def test_review_zero_q_graph_capture(with_lse):
    q, k, v = inputs([0], [128])
    attn, _, _ = batch_wrapper([0], [128], causal=True, graph=True)
    out = torch.empty((0, HEADS, 128), device=q.device, dtype=torch.bfloat16)
    lse = torch.empty((0, HEADS), device=q.device) if with_lse else None
    call = functools.partial(attn.run, q, k, v, out=out, lse=lse, return_lse=with_lse)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warmup_batch_graph(causal=True, with_lse=with_lse)
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert out.shape == (0, HEADS, 128)
    if with_lse:
        assert lse.shape == (0, HEADS)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(q_scale=-SCALE_Q, k_scale=-SCALE_K),
        dict(sm_scale=-SM_SCALE, q_scale=-SCALE_Q),
        dict(v_scale=-SCALE_V),
        dict(o_scale=-SCALE_O),
        dict(v_scale=0.0),
        dict(q_scale=1e-50, k_scale=1e25),
    ],
    ids=[
        "negative-qk",
        "negative-sm-q",
        "negative-v",
        "negative-o",
        "zero-v",
        "compensated-tiny-q",
    ],
)
@pytest.mark.parametrize(
    "causal,q_lens,kv_lens",
    [(False, [3], [128]), (True, [3, 5], [0, 129])],
    ids=["noncausal", "causal-partial-kv"],
)
def test_review_equal_head_supported_scale_controls(overrides, causal, q_lens, kv_lens):
    scales = dict(
        sm_scale=SM_SCALE,
        q_scale=SCALE_Q,
        k_scale=SCALE_K,
        v_scale=SCALE_V,
        o_scale=SCALE_O,
    )
    scales.update(overrides)
    q, k, v = inputs(q_lens, kv_lens)
    attn, _, _ = batch_wrapper(q_lens, kv_lens, causal=causal)
    review_plan(attn, q_lens, kv_lens, causal=causal, sm_scale=scales["sm_scale"])
    out, lse = attn.run(
        q,
        k,
        v,
        return_lse=True,
        **{key: value for key, value in scales.items() if key != "sm_scale"},
    )
    assert_outputs(out, lse, *reference(q, k, v, q_lens, kv_lens, causal, **scales))
