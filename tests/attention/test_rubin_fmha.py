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


def reference(q, k, v, q_lens, kv_lens, causal):
    out = torch.zeros((*q.shape[:2], 128), device=q.device, dtype=torch.float32)
    lse = torch.full(q.shape[:2], -float("inf"), device=q.device, dtype=torch.float32)
    q_offset = kv_offset = 0
    for q_len, kv_len in zip(q_lens, kv_lens, strict=True):
        if q_len and kv_len:
            qs = slice(q_offset, q_offset + q_len)
            ks = slice(kv_offset, kv_offset + kv_len)
            logits = torch.einsum("qhd,khd->hqk", q[qs].float(), k[ks].float())
            logits *= SCALE_Q * SCALE_K * SM_SCALE
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
                SCALE_V / SCALE_O
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
