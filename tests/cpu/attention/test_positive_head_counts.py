# SPDX-License-Identifier: Apache-2.0

"""CPU checks for attention head-count validation before backend dispatch."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import flashinfer
from flashinfer import decode, prefill
from flashinfer.cute_dsl.attention import (
    BatchDecodeCuteDSLWrapper,
    BatchDecodePagedCuteDSLWrapper,
    BatchPrefillCuteDSLWrapper,
)
from flashinfer.jit.attention.modules import get_batch_prefill_bidirectional_ranges_spec

pytestmark = pytest.mark.cpu_only

_HEAD_DIM = 128
_PAGE_SIZE = 16
_HEAD_RATIO_ERROR = "must be a multiple of num_kv_heads"


@pytest.fixture(
    params=["decode", "tensor-core-decode", "prefill", "jit-decode", "jit-prefill"]
)
def single_attention(request, monkeypatch):
    variant = request.param

    def bind(run):
        module = SimpleNamespace(run=run)
        if variant == "jit-decode":
            return partial(decode.single_decode_with_kv_cache_with_jit_module, module)
        if variant == "jit-prefill":
            return partial(prefill.single_prefill_with_kv_cache_with_jit_module, module)
        if variant == "decode":
            monkeypatch.setattr(
                decode, "get_single_decode_module", lambda *args: module
            )
            return flashinfer.single_decode_with_kv_cache
        if variant == "tensor-core-decode":
            monkeypatch.setattr(
                decode, "get_single_prefill_module", lambda *args: module
            )
            return partial(
                flashinfer.single_decode_with_kv_cache, use_tensor_cores=True
            )
        monkeypatch.setattr(prefill, "get_single_prefill_module", lambda *args: module)
        return partial(flashinfer.single_prefill_with_kv_cache, backend="fa2")

    return bind, "prefill" in variant


def _q_tensor(is_prefill, num_heads, dtype=torch.float16):
    shape = (2, num_heads, _HEAD_DIM) if is_prefill else (num_heads, _HEAD_DIM)
    return torch.empty(shape, dtype=dtype)


def _kv_tensor(layout, seq_len, num_heads, dtype=torch.float16):
    prefix = (seq_len, num_heads) if layout == "NHD" else (num_heads, seq_len)
    return torch.empty((*prefix, _HEAD_DIM), dtype=dtype)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "num_qo_heads, num_k_heads, num_v_heads, error",
    [
        pytest.param(0, 8, 8, "num_qo_heads must be positive", id="zero-q"),
        pytest.param(8, 0, 0, "num_kv_heads must be positive", id="zero-kv"),
        pytest.param(8, 8, 0, "num_kv_heads in v must be positive", id="zero-v"),
        pytest.param(7, 2, 2, _HEAD_RATIO_ERROR, id="non-divisible"),
        pytest.param(2, 8, 8, _HEAD_RATIO_ERROR, id="q-heads-less-than-kv"),
    ],
)
def test_single_attention_rejects_invalid_head_counts(
    single_attention, layout, num_qo_heads, num_k_heads, num_v_heads, error
):
    bind, is_prefill = single_attention

    def unexpected_backend(*args):
        pytest.fail("invalid head counts reached backend dispatch")

    attention = bind(unexpected_backend)
    q = _q_tensor(is_prefill, num_qo_heads)
    k = _kv_tensor(layout, 3, num_k_heads)
    v = _kv_tensor(layout, 3, num_v_heads)
    with pytest.raises(ValueError, match=error):
        attention(q, k, v, kv_layout=layout)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_single_attention_accepts_empty_kv_with_positive_heads(
    single_attention, layout
):
    bind, is_prefill = single_attention
    q = _q_tensor(is_prefill, 16)
    k = _kv_tensor(layout, 0, 8)
    v = torch.empty_like(k)
    calls = []

    def run(query, key, value, tmp, out, *args):
        calls.append(key)
        assert key is k and value is v
        assert query.shape[-2:] == q.shape[-2:]
        out.fill_(1)

    out = bind(run)(q, k, v, kv_layout=layout)
    assert calls == [k]
    torch.testing.assert_close(out, torch.ones_like(q))


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("kv_len", [0, 3])
def test_single_fp8_prefill_uses_head_axis_for_default_scales(
    monkeypatch, layout, kv_len
):
    q = _q_tensor(True, 16, torch.float8_e4m3fn)
    k = _kv_tensor(layout, kv_len, 8, q.dtype)
    v = torch.empty_like(k)
    scales = []

    def run(*args):
        # The native run signature places Q/K/V dequantization scales after sm_scale.
        scales.extend(args[13:16])

    module = SimpleNamespace(run=run)
    monkeypatch.setattr(prefill, "get_single_prefill_module", lambda *args: module)
    flashinfer.single_prefill_with_kv_cache(q, k, v, kv_layout=layout, backend="fa3")

    assert len(scales) == 3
    for scale, num_heads in zip(scales, (16, 8, 8), strict=True):
        torch.testing.assert_close(scale, torch.ones(num_heads, dtype=torch.float32))


_PLANNERS = [
    (
        flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        "decode",
        ("plan", "begin_forward", "workspace_size", "fast_decode_plan"),
    ),
    (
        flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        "prefill",
        ("plan", "begin_forward", "workspace_size"),
    ),
    (
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
        "ragged",
        ("plan", "begin_forward"),
    ),
    (
        flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
        "decode",
        ("begin_forward",),
    ),
    (
        flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
        "prefill",
        ("begin_forward",),
    ),
    (
        flashinfer.MultiLevelCascadeAttentionWrapper,
        "cascade",
        ("plan", "begin_forward"),
    ),
    (flashinfer.PODWithPagedKVCacheWrapper, "decode", ("plan", "begin_forward")),
    (
        flashinfer.BatchPODWithPagedKVCacheWrapper,
        "batch-pod",
        ("plan", "begin_forward"),
    ),
    (
        flashinfer.BatchDecodeWithAttentionSinkWrapper,
        "decode",
        ("begin_forward", "plan"),
    ),
    (
        flashinfer.BatchAttentionWithAttentionSinkWrapper,
        "prefill",
        ("begin_forward", "plan"),
    ),
    (
        flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper,
        "prefill",
        ("begin_forward", "plan", "workspace_size"),
    ),
    (flashinfer.BatchAttention, "holistic", ("plan",)),
    (flashinfer.BlockSparseAttentionWrapper, "block-sparse", ("plan", "begin_forward")),
    (
        flashinfer.VariableBlockSparseAttentionWrapper,
        "variable-block-sparse",
        ("plan",),
    ),
    (BatchDecodeCuteDSLWrapper, "cute-decode", ("plan",)),
    (BatchDecodePagedCuteDSLWrapper, "decode", ("plan",)),
    (BatchPrefillCuteDSLWrapper, "ragged", ("plan",)),
]


@pytest.fixture
def plan_args():
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    indices = torch.tensor([0], dtype=torch.int32)
    lengths = torch.tensor([_PAGE_SIZE], dtype=torch.int32)
    paged = (kv_indptr, indices, lengths)
    return {
        "cute-decode": ((1, 1), (_HEAD_DIM,)),
        "decode": (paged, (_HEAD_DIM, _PAGE_SIZE)),
        "prefill": ((qo_indptr, *paged), (_HEAD_DIM, _PAGE_SIZE)),
        "ragged": ((qo_indptr, kv_indptr), (_HEAD_DIM,)),
        "cascade": (
            ([qo_indptr], [kv_indptr], [indices], [lengths]),
            (_HEAD_DIM, _PAGE_SIZE),
        ),
        "batch-pod": ((qo_indptr, *paged, qo_indptr, *paged), (_HEAD_DIM, _PAGE_SIZE)),
        "holistic": ((qo_indptr, *paged), (_HEAD_DIM, _HEAD_DIM, _PAGE_SIZE)),
        "block-sparse": ((kv_indptr, indices, 16, 16, 16, 16), (_HEAD_DIM,)),
        "variable-block-sparse": (
            (torch.ones((8, 1, 1), dtype=torch.bool), lengths, lengths),
            (_HEAD_DIM,),
        ),
    }


@pytest.mark.parametrize(
    "wrapper_cls, metadata, method",
    [
        pytest.param(
            cls,
            metadata,
            method,
            id=method if method == "fast_decode_plan" else f"{cls.__name__}.{method}",
        )
        for cls, metadata, methods in _PLANNERS
        for method in methods
    ],
)
@pytest.mark.parametrize(
    "num_qo_heads, num_kv_heads, error",
    [
        pytest.param(0, 8, "num_qo_heads must be positive", id="zero-q"),
        pytest.param(8, 0, "num_kv_heads must be positive", id="zero-kv"),
        pytest.param(-8, 8, "num_qo_heads must be positive", id="negative-q"),
        pytest.param(8, -8, "num_kv_heads must be positive", id="negative-kv"),
        pytest.param(7, 2, _HEAD_RATIO_ERROR, id="non-divisible"),
        pytest.param(2, 8, _HEAD_RATIO_ERROR, id="q-heads-less-than-kv"),
    ],
)
def test_attention_plan_rejects_invalid_head_counts(
    wrapper_cls, metadata, method, plan_args, num_qo_heads, num_kv_heads, error
):
    # Skip workspace allocation and JIT in constructors, while still calling real planners.
    wrapper = wrapper_cls.__new__(wrapper_cls)
    if wrapper_cls is flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper:
        cls = flashinfer.BatchDecodeWithPagedKVCacheWrapper
        wrapper._batch_decode_wrapper = cls.__new__(cls)
    elif wrapper_cls is flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper:
        cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        wrapper._batch_prefill_wrapper = cls.__new__(cls)
    elif wrapper_cls is flashinfer.MultiLevelCascadeAttentionWrapper:
        cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        wrapper._batch_prefill_wrappers = [cls.__new__(cls)]
        wrapper._num_levels = 1
    elif wrapper_cls is flashinfer.BatchPrefillWithCausalBidirectionalRangesWrapper:
        wrapper._spec = get_batch_prefill_bidirectional_ranges_spec(
            torch.float16,
            torch.float16,
            torch.float16,
            torch.int32,
            _HEAD_DIM,
            _HEAD_DIM,
        )
    prefix, suffix = plan_args[metadata]
    planner = (
        partial(flashinfer.fast_decode_plan, wrapper)
        if method == "fast_decode_plan"
        else getattr(wrapper, method)
    )
    with pytest.raises(ValueError, match=error):
        planner(*prefix, num_qo_heads, num_kv_heads, *suffix)


@pytest.mark.parametrize(
    "wrapper_cls, variant",
    [
        pytest.param(
            flashinfer.BatchDecodeWithPagedKVCacheWrapper, "paged", id="paged-decode"
        ),
        pytest.param(
            flashinfer.BatchPrefillWithPagedKVCacheWrapper, "paged", id="paged-prefill"
        ),
        pytest.param(
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper,
            "ragged",
            id="ragged-prefill",
        ),
        pytest.param(flashinfer.BatchAttention, "paged", id="holistic"),
        pytest.param(
            flashinfer.PODWithPagedKVCacheWrapper, "pod-prefill", id="pod-prefill"
        ),
        pytest.param(
            flashinfer.PODWithPagedKVCacheWrapper, "pod-decode", id="pod-decode"
        ),
        pytest.param(
            flashinfer.BatchPODWithPagedKVCacheWrapper,
            "batch-pod-prefill",
            id="batch-pod-prefill",
        ),
        pytest.param(
            flashinfer.BatchPODWithPagedKVCacheWrapper,
            "batch-pod-decode",
            id="batch-pod-decode",
        ),
    ],
)
@pytest.mark.parametrize(
    "num_qo_heads, num_kv_heads, error",
    [
        pytest.param(0, 8, "num_qo_heads must be positive", id="zero-q"),
        pytest.param(8, 0, "num_kv_heads must be positive", id="zero-kv"),
    ],
)
def test_attention_run_rejects_invalid_head_counts(
    wrapper_cls, variant, num_qo_heads, num_kv_heads, error
):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    wrapper._kv_layout = "NHD"
    if wrapper_cls is flashinfer.BatchAttention:
        wrapper._use_profiler = False

    q = _q_tensor(True, num_qo_heads)
    k = _kv_tensor("NHD", _PAGE_SIZE, num_kv_heads)
    v = torch.empty_like(k)
    paged = (k.unsqueeze(0), v.unsqueeze(0))
    good_q = _q_tensor(True, 8)
    good_k = _kv_tensor("NHD", _PAGE_SIZE, 8)
    good_v = torch.empty_like(good_k)
    good_paged = (good_k.unsqueeze(0), good_v.unsqueeze(0))
    args = {
        "paged": (q, paged),
        "ragged": (q, k, v),
        "pod-prefill": (q, k, v, good_q, good_paged),
        "pod-decode": (good_q, good_k, good_v, q, paged),
        "batch-pod-prefill": (q, paged, good_q, good_paged),
        "batch-pod-decode": (good_q, good_paged, q, paged),
    }
    with pytest.raises(ValueError, match=error):
        wrapper.run(*args[variant])


@pytest.mark.parametrize(
    "num_qo_heads, num_kv_heads, error",
    [
        pytest.param(0, 8, "num_qo_heads must be positive", id="zero-q"),
        pytest.param(8, 0, "num_kv_heads must be positive", id="zero-kv"),
        pytest.param(8, -8, "num_kv_heads must be positive", id="negative-kv"),
        pytest.param(8, 3, _HEAD_RATIO_ERROR, id="non-divisible"),
    ],
)
def test_xqa_rejects_invalid_head_counts(num_qo_heads, num_kv_heads, error):
    q = _q_tensor(True, num_qo_heads).unsqueeze(1)
    kv = _kv_tensor("NHD", _PAGE_SIZE, 8).unsqueeze(0)
    block_tables = torch.zeros((2, 1), dtype=torch.int32)
    seq_lens = torch.full((2, 1), _PAGE_SIZE, dtype=torch.uint32)
    scratch = torch.empty(1, dtype=torch.uint8)
    semaphore = torch.zeros(1, dtype=torch.uint32)
    with pytest.raises(ValueError, match=error):
        flashinfer.xqa(
            q,
            kv,
            kv,
            block_tables,
            seq_lens,
            torch.empty_like(q),
            scratch,
            semaphore,
            num_kv_heads,
            _PAGE_SIZE,
            sm_count=1,
            enable_pdl=False,
        )


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_xqa_batch_decode_keeps_shared_cache_axes(monkeypatch, layout):
    run = Mock()
    monkeypatch.setattr(decode, "xqa", run)
    monkeypatch.setattr(decode, "get_device_sm_count", lambda device: 1)
    q = _q_tensor(True, 16)
    kv = _kv_tensor(layout, _PAGE_SIZE, 8).unsqueeze(0).unsqueeze(0)
    workspace = torch.empty(8 * 1024 * 1024 + 64, dtype=torch.uint8)
    block_tables = torch.zeros((2, 1), dtype=torch.int32)
    seq_lens = torch.full((2,), _PAGE_SIZE, dtype=torch.uint32)

    out = decode.xqa_batch_decode_with_kv_cache(
        q,
        kv,
        workspace,
        block_tables,
        seq_lens,
        _PAGE_SIZE,
        kv_layout=layout,
        enable_pdl=False,
    )

    assert out.shape == q.shape
    run.assert_called_once()
    assert run.call_args.args[1] is kv and run.call_args.args[2] is kv
    assert run.call_args.args[-2:] == (8, _PAGE_SIZE)
