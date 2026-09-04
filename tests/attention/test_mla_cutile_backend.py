# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the planned cuTile MLA backend."""

import math

import pytest
import torch


class _FakeCutileKernel:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        seq_lens,
        block_tables,
        k_scale,
        v_scale,
        max_seq_len=-1,
        outputs=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "q_nope": q_nope,
                "q_pe": q_pe,
                "ckv_cache": ckv_cache,
                "kpe_cache": kpe_cache,
                "seq_lens": seq_lens,
                "block_tables": block_tables,
                "k_scale": k_scale,
                "v_scale": v_scale,
                "max_seq_len": max_seq_len,
                "outputs": outputs,
                "kwargs": kwargs,
            }
        )
        if outputs is None:
            outputs = torch.empty_like(q_nope)
        outputs.copy_(q_nope)
        return outputs


def _patch_cutile_runtime(monkeypatch, kernel):
    from flashinfer.mla._batch_mla._backends import cutile_backend

    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", lambda: kernel)


def _metadata():
    from flashinfer.mla import MLAPlanMetadata

    return MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_tables=torch.tensor([[1, 0], [0, 1]], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
    )


def _plan_kwargs(metadata=None, **overrides):
    kwargs = {
        "metadata": _metadata() if metadata is None else metadata,
        "num_heads": 16,
        "head_dim_ckv": 512,
        "head_dim_kpe": 64,
        "page_size": 2,
        "causal": False,
        "sm_scale": 1.0 / math.sqrt(576),
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "query_layout": "split",
        "kv_cache_layout": "split",
    }
    kwargs.update(overrides)
    return kwargs


def _inputs(*, num_heads=16, page_size=2, dtype=torch.bfloat16):
    return (
        (
            torch.empty(2, num_heads, 512, dtype=dtype),
            torch.empty(2, num_heads, 64, dtype=dtype),
        ),
        (
            torch.empty(2, page_size, 512, dtype=dtype),
            torch.empty(2, page_size, 64, dtype=dtype),
        ),
    )


def _planned_wrapper(monkeypatch, *, use_cuda_graph=False, metadata=None):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8),
        use_cuda_graph=use_cuda_graph,
        backend="cutile",
    )
    wrapper.plan(**_plan_kwargs(metadata))
    return wrapper, kernel


def test_cutile_backend_is_explicitly_registered_with_split_lowering():
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _BatchMLAPagedAttentionCutileBackend,
    )
    from flashinfer.mla._batch_mla._wrapper import _BACKEND_TYPES

    assert _BACKEND_TYPES["cutile"] is _BatchMLAPagedAttentionCutileBackend
    capabilities = _BatchMLAPagedAttentionCutileBackend._plan_capabilities
    assert capabilities.backend_name == "cutile"
    assert capabilities.lse_modes == frozenset({"none"})
    assert capabilities.output_scales == frozenset({"none"})
    assert capabilities.scale_modes == frozenset({"default"})
    assert {"combined", "independent-split"} <= capabilities.kv_layouts
    assert capabilities.requires_packed_query is False
    assert capabilities.requires_packed_kv_cache is False


def test_cutile_lazy_kernel_lookup_and_retained_dense_metadata(monkeypatch):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    metadata = _metadata()
    kernel = _FakeCutileKernel()
    getter_calls = []
    from flashinfer.mla._batch_mla._backends import cutile_backend

    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: (10, 0)
    )

    def get_kernel():
        getter_calls.append(None)
        return kernel

    monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", get_kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    with pytest.raises(ValueError, match="causal"):
        wrapper.plan(**_plan_kwargs(metadata, causal=True))
    assert getter_calls == []

    wrapper.plan(**_plan_kwargs(metadata))
    assert getter_calls == [None]

    query, kv_cache = _inputs()
    out = torch.empty_like(query[0])
    actual = wrapper.run(query=query, kv_cache=kv_cache, out=out)

    assert actual is out
    assert getter_calls == [None]
    assert len(kernel.calls) == 1
    call = kernel.calls[0]
    assert call["q_nope"] is query[0]
    assert call["q_pe"] is query[1]
    assert call["ckv_cache"] is kv_cache[0]
    assert call["kpe_cache"] is kv_cache[1]
    assert call["seq_lens"] is metadata.seq_lens
    assert call["block_tables"] is metadata.block_tables
    assert call["outputs"] is out


def test_cutile_packed_contract_is_lowered_to_zero_copy_split_views(monkeypatch):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    wrapper.plan(**_plan_kwargs(query_layout="packed", kv_cache_layout="packed"))
    query = torch.empty(2, 16, 576, dtype=torch.bfloat16)
    kv_cache = torch.empty(2, 2, 576, dtype=torch.bfloat16)
    wrapper.run(query=query, kv_cache=kv_cache)

    call = kernel.calls[0]
    assert (
        call["q_nope"].untyped_storage().data_ptr()
        == query.untyped_storage().data_ptr()
    )
    assert (
        call["q_pe"].untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
    )
    assert (
        call["ckv_cache"].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )
    assert (
        call["kpe_cache"].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cutile_supported_float_dtypes_dispatch(monkeypatch, dtype):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )
    wrapper.plan(**_plan_kwargs(q_data_type=dtype, kv_data_type=dtype))
    query = (
        torch.empty(2, 16, 512, dtype=dtype),
        torch.empty(2, 16, 64, dtype=dtype),
    )
    kv_cache = (
        torch.empty(2, 2, 512, dtype=dtype),
        torch.empty(2, 2, 64, dtype=dtype),
    )

    actual = wrapper.run(query=query, kv_cache=kv_cache)

    assert actual.dtype == dtype
    assert len(kernel.calls) == 1


def test_cutile_runtime_metadata_override_is_paired_and_zero_copy(monkeypatch):
    wrapper, kernel = _planned_wrapper(monkeypatch)
    query, kv_cache = _inputs()
    seq_lens = torch.tensor([2, 3], dtype=torch.int32)
    block_tables = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="both be omitted or both be provided"):
        wrapper.run(query=query, kv_cache=kv_cache, kv_len=seq_lens)
    with pytest.raises(ValueError, match="both be omitted or both be provided"):
        wrapper.run(query=query, kv_cache=kv_cache, page_table=block_tables)

    wrapper.run(
        query=query,
        kv_cache=kv_cache,
        kv_len=seq_lens,
        page_table=block_tables,
    )
    call = kernel.calls[-1]
    assert call["seq_lens"] is seq_lens
    assert call["block_tables"] is block_tables


@pytest.mark.parametrize(
    ("kv_len", "page_table", "match"),
    [
        (
            torch.tensor([2, 3], dtype=torch.int64),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "torch.int32",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            "torch.int32",
        ),
        (
            torch.empty(4, dtype=torch.int32)[::2],
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "contiguous",
        ),
        (
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "shape",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.tensor([[0, 1]], dtype=torch.int32),
            "shape",
        ),
        (
            torch.empty(2, dtype=torch.int32, device="meta"),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
            "workspace device",
        ),
        (
            torch.tensor([2, 3], dtype=torch.int32),
            torch.empty((2, 2), dtype=torch.int32, device="meta"),
            "workspace device",
        ),
    ],
)
def test_cutile_runtime_metadata_override_rejects_unsafe_tensors(
    monkeypatch, kv_len, page_table, match
):
    wrapper, kernel = _planned_wrapper(monkeypatch)
    query, kv_cache = _inputs()

    with pytest.raises(ValueError, match=match):
        wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
        )
    assert kernel.calls == []


@pytest.mark.parametrize(
    "plan_overrides",
    [
        {"lse_mode": "base2"},
        {"output_dtype": torch.float8_e4m3fn, "output_scale": "per-tensor"},
        {"scale_mode": "kv-per-tensor"},
        {"skip_softmax": True},
        {"causal": True},
        {"head_dim_ckv": 256},
        {"q_data_type": torch.float32, "kv_data_type": torch.float32},
    ],
)
def test_cutile_plan_rejects_unsupported_contracts(monkeypatch, plan_overrides):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    with pytest.raises(ValueError):
        wrapper.plan(**_plan_kwargs(**plan_overrides))
    assert kernel.calls == []


@pytest.mark.parametrize("page_size", [2, 128])
def test_cutile_plan_accepts_supported_page_sizes(monkeypatch, page_size):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    wrapper.plan(**_plan_kwargs(page_size=page_size))
    assert kernel.calls == []


@pytest.mark.parametrize("page_size", [True, 1, 3, 129])
def test_cutile_rejects_unsupported_page_sizes(page_size):
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _validate_cutile_page_size,
    )

    with pytest.raises(ValueError, match=r"\[2, 128\]"):
        _validate_cutile_page_size(page_size)


@pytest.mark.parametrize("num_heads", [16, 128])
def test_cutile_plan_accepts_supported_head_counts(monkeypatch, num_heads):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    kernel = _FakeCutileKernel()
    _patch_cutile_runtime(monkeypatch, kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    wrapper.plan(**_plan_kwargs(num_heads=num_heads))
    query, kv_cache = _inputs(num_heads=num_heads)
    wrapper.run(query=query, kv_cache=kv_cache)
    assert len(kernel.calls) == 1


@pytest.mark.parametrize("num_heads", [True, 7, 9, 129])
def test_cutile_rejects_unsupported_head_counts(num_heads):
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _validate_cutile_num_heads,
    )

    with pytest.raises(ValueError, match=r"multiple of 8.*\[8, 128\]"):
        _validate_cutile_num_heads(num_heads)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0), (12, 1)])
def test_cutile_plan_accepts_supported_blackwell_architectures(monkeypatch, capability):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.mla._batch_mla._backends import cutile_backend

    kernel = _FakeCutileKernel()
    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: capability
    )
    monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", lambda: kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    wrapper.plan(**_plan_kwargs())
    query, kv_cache = _inputs()
    wrapper.run(query=query, kv_cache=kv_cache)
    assert len(kernel.calls) == 1


@pytest.mark.parametrize("capability", [(9, 0), (12, 2)])
def test_cutile_plan_rejects_undemonstrated_architectures(monkeypatch, capability):
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    from flashinfer.mla._batch_mla._backends import cutile_backend

    kernel = _FakeCutileKernel()
    monkeypatch.setattr(
        cutile_backend, "_get_compute_capability", lambda device: capability
    )
    monkeypatch.setattr(cutile_backend, "get_cutile_mla_decode", lambda: kernel)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(1024, dtype=torch.uint8), backend="cutile"
    )

    with pytest.raises(ValueError, match="SM100, SM103, SM120, and SM121"):
        wrapper.plan(**_plan_kwargs())
    assert kernel.calls == []


@pytest.mark.parametrize(
    ("run_overrides", "match"),
    [
        ({"return_lse": True}, "LSE mode"),
        ({"lse": torch.empty(2, 16)}, "LSE mode"),
        ({"o_scale": 0.5}, "o_scale"),
        ({"ckv_scale": 1.0, "kpe_scale": 1.0}, "only valid"),
        ({"profiler_buffer": torch.empty(1)}, "profiler"),
    ],
)
def test_cutile_run_rejects_unsupported_options_before_dispatch(
    monkeypatch, run_overrides, match
):
    wrapper, kernel = _planned_wrapper(monkeypatch)
    query, kv_cache = _inputs()

    with pytest.raises(ValueError, match=match):
        wrapper.run(query=query, kv_cache=kv_cache, **run_overrides)
    assert kernel.calls == []


@pytest.mark.parametrize("case", ["shape", "overlap"])
def test_cutile_rejects_unsafe_output_before_dispatch(monkeypatch, case):
    wrapper, kernel = _planned_wrapper(monkeypatch)
    query, kv_cache = _inputs()
    if case == "shape":
        out = torch.empty(2, 16, 511, dtype=torch.bfloat16)
    else:
        out = query[0]

    with pytest.raises(ValueError):
        wrapper.run(query=query, kv_cache=kv_cache, out=out)
    assert kernel.calls == []


@pytest.mark.parametrize("case", ["shape", "dtype"])
def test_cutile_rejects_unsafe_launch_input_before_dispatch(monkeypatch, case):
    wrapper, kernel = _planned_wrapper(monkeypatch)
    query, kv_cache = _inputs()
    q_nope, q_pe = query
    if case == "shape":
        q_nope = torch.empty(2, 16, 511, dtype=torch.bfloat16)
    elif case == "dtype":
        q_nope = torch.empty(2, 16, 512, dtype=torch.float16)
    with pytest.raises(ValueError):
        wrapper.run(query=(q_nope, q_pe), kv_cache=kv_cache)
    assert kernel.calls == []


def test_cutile_failed_replan_is_transactional(monkeypatch):
    wrapper, _ = _planned_wrapper(monkeypatch)
    prior_backend = wrapper._planned_backend
    prior_contract = wrapper._input_contract

    with pytest.raises(ValueError, match="causal"):
        wrapper.plan(**_plan_kwargs(causal=True))

    assert wrapper._planned_backend is prior_backend
    assert wrapper._input_contract is prior_contract


def test_cutile_cuda_graph_plan_retains_metadata_and_rejects_replan(monkeypatch):
    metadata = _metadata()
    wrapper, kernel = _planned_wrapper(
        monkeypatch, use_cuda_graph=True, metadata=metadata
    )
    query, kv_cache = _inputs()
    wrapper.run(query=query, kv_cache=kv_cache)

    assert kernel.calls[-1]["seq_lens"] is metadata.seq_lens
    assert kernel.calls[-1]["block_tables"] is metadata.block_tables
    with pytest.raises(RuntimeError, match=r"CUDA graph.*replan"):
        wrapper.plan(**_plan_kwargs(metadata))
