# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only public-contract checks for PrimTS paged context attention."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import torch

import flashinfer.attention.prims_ts as prims_ts
import flashinfer.attention.prims_ts.context as context_api


def _allow_cpu_plan_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep semantic checks while bypassing only CUDA storage requirements."""

    def validate_base(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        context_api._validate_qkv_dtype(q, k, v)
        if k.shape != v.shape:
            raise ValueError("v must have the same shape as k")

    def validate_indptr(
        tensor: torch.Tensor,
        name: str,
        *,
        device: torch.device,
    ) -> None:
        del device
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")
        if tensor.ndim != 1 or tensor.numel() < 2:
            raise ValueError(f"{name} must be a nonempty indptr")
        context_api._validate_compact(tensor, name, "[B+1]")

    def validate_metadata(
        tensor: torch.Tensor,
        name: str,
        *,
        device: torch.device,
    ) -> None:
        del device
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype torch.int32")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be rank 1")
        context_api._validate_compact(tensor, name, "one-dimensional")

    monkeypatch.setattr(context_api, "_validate_base_tensors", validate_base)
    monkeypatch.setattr(context_api, "_validate_device", lambda device: 0)
    monkeypatch.setattr(context_api, "_validate_indptr_tensor", validate_indptr)
    monkeypatch.setattr(
        context_api, "_validate_paged_metadata_tensor", validate_metadata
    )


@pytest.mark.parametrize("head_dim", (128, 256))
def test_paged_geometry_translates_nonidentity_csr_for_both_head_dims(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
):
    _allow_cpu_plan_metadata(monkeypatch)
    q = torch.empty((97, 32, head_dim), dtype=torch.float16)
    k_cache = torch.empty((6, 4, 32, head_dim), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    qo_indptr = torch.tensor([0, 33, 97], dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 2, 5], dtype=torch.int32)
    paged_kv_indices = torch.tensor([4, 1, 3, 0, 2], dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([17, 5], dtype=torch.int32)

    geometry, metadata = context_api._resolve_paged_geometry(
        q,
        k_cache,
        v_cache,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        page_size=32,
        mask_type="dense",
        window_left=-1,
        output_dtype=q.dtype,
    )

    assert geometry.head_dim == head_dim
    assert geometry.max_seq_len_q == 64
    assert geometry.max_seq_len_k == 69
    assert geometry.max_num_pages_per_seq_kv == 3
    assert metadata.seq_lens == (49, 69)
    assert metadata.kv_indptr == (0, 49, 118)
    assert metadata.dense_page_indices == (
        4,
        1,
        0,
        4,
        1,
        0,
        3,
        0,
        2,
        3,
        0,
        2,
    )


def test_paged_public_validation_is_strict_and_has_no_layout_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    assert prims_ts.BatchPrefillPagedTSWrapper is context_api.BatchPrefillPagedTSWrapper
    assert (
        prims_ts.batch_prefill_with_paged_kv_cache
        is context_api.batch_prefill_with_paged_kv_cache
    )
    assert context_api._dtype_key(torch.float8_e4m3fn) == "float8_e4m3fn"
    with pytest.raises(NotImplementedError, match="page_size=32"):
        context_api._validate_page_size(16)
    with pytest.raises(NotImplementedError, match="kv_layout='HND'"):
        context_api.BatchPrefillPagedTSWrapper(kv_layout="NHD")

    _allow_cpu_plan_metadata(monkeypatch)
    q = torch.empty((4, 4, 128), dtype=torch.float16)
    k_cache = torch.empty((2, 2, 32, 128), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    common = {
        "qo_indptr": torch.tensor([0, 4], dtype=torch.int32),
        "paged_kv_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "paged_kv_last_page_len": torch.tensor([4], dtype=torch.int32),
        "page_size": 32,
        "mask_type": "dense",
        "window_left": -1,
        "output_dtype": q.dtype,
    }
    with pytest.raises(ValueError, match="physical page pool"):
        context_api._resolve_paged_geometry(
            q,
            k_cache,
            v_cache,
            paged_kv_indices=torch.tensor([2], dtype=torch.int32),
            **common,
        )

    strided_k = torch.empty((2, 2, 32, 129), dtype=torch.float16)[..., :128]
    strided_v = torch.empty_like(strided_k)
    with pytest.raises(ValueError, match="compact.*Hkv"):
        context_api._resolve_paged_geometry(
            q,
            strided_k,
            strided_v,
            paged_kv_indices=torch.tensor([1], dtype=torch.int32),
            **common,
        )


def test_paged_public_signatures_and_defaults_are_stable():
    wrapper_init = inspect.signature(context_api.BatchPrefillPagedTSWrapper)
    assert tuple(wrapper_init.parameters) == ("kv_layout",)
    assert wrapper_init.parameters["kv_layout"].default == "HND"

    plan = inspect.signature(context_api.BatchPrefillPagedTSWrapper.plan).parameters
    assert tuple(plan) == (
        "self",
        "q",
        "k_cache",
        "v_cache",
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "page_size",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
        "out_dtype",
    )
    assert all(
        plan[name].default is inspect.Parameter.empty
        for name in (
            "self",
            "q",
            "k_cache",
            "v_cache",
            "qo_indptr",
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
        )
    )
    assert {
        name: plan[name].default
        for name in (
            "page_size",
            "mask_type",
            "window_left",
            "sm_scale",
            "output_scale",
            "out_dtype",
        )
    } == {
        "page_size": 32,
        "mask_type": "dense",
        "window_left": -1,
        "sm_scale": None,
        "output_scale": 1.0,
        "out_dtype": None,
    }
    assert all(
        plan[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in (
            "page_size",
            "mask_type",
            "window_left",
            "sm_scale",
            "output_scale",
            "out_dtype",
        )
    )

    run = inspect.signature(context_api.BatchPrefillPagedTSWrapper.run).parameters
    assert tuple(run) == ("self", "q", "k_cache", "v_cache", "out")
    assert run["out"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run["out"].default is None

    one_shot = inspect.signature(
        context_api.batch_prefill_with_paged_kv_cache
    ).parameters
    assert tuple(one_shot) == (
        "q",
        "k_cache",
        "v_cache",
        "qo_indptr",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "page_size",
        "kv_layout",
        "mask_type",
        "window_left",
        "sm_scale",
        "output_scale",
        "out_dtype",
        "out",
    )
    assert one_shot["page_size"].default == 32
    assert one_shot["kv_layout"].default == "HND"
    assert one_shot["mask_type"].default == "dense"
    assert one_shot["window_left"].default == -1
    assert one_shot["out"].default is None


def test_paged_compile_policy_is_static_persistent_on_cutlass_dsl_47(
    monkeypatch: pytest.MonkeyPatch,
):
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.base_dsl.dsl import BaseDSL

    from flashinfer.attention.prims_ts.kernels.fmha_context import fmha_kernel

    constructor_args: dict[str, object] = {}

    class FakeFmha:
        def __init__(self, **kwargs: object) -> None:
            constructor_args.update(kwargs)
            self.cfg = SimpleNamespace()

    monkeypatch.setattr(fmha_kernel, "FmhaTs", FakeFmha)
    monkeypatch.setattr(
        utils,
        "HardwareInfo",
        lambda: SimpleNamespace(get_max_active_clusters=lambda cluster_size: 7),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        cute.runtime, "make_fake_compact_tensor", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        cute.runtime, "make_fake_stream", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(cute, "sym_int", lambda: 1)
    monkeypatch.setattr(cute, "compile", lambda *args, **kwargs: "compiled")
    monkeypatch.setattr(BaseDSL, "enable_pyir", staticmethod(lambda: nullcontext()))
    context_api._get_compiled_paged_context.cache_clear()
    try:
        compiled, policy = context_api._get_compiled_paged_context(
            0,
            4,
            1024,
            1024,
            32,
            32,
            4,
            256,
            "float8_e4m3fn",
            "float8_e4m3fn",
            "dense",
            -1,
            False,
            False,
            0,
        )
    finally:
        context_api._get_compiled_paged_context.cache_clear()

    assert compiled == "compiled"
    assert dict(policy) == {
        "scheduler": "static_persistent",
        "pairing": "query",
        "kv_layout": "paged_hnd",
        "page_size": 32,
        "causal_single_kv_tile": False,
    }
    assert constructor_args["d"] == 256
    assert constructor_args["is_persistent"] is True
    assert constructor_args["is_clc_dynamic"] is False
    assert constructor_args["use_paged_kv"] is True
    assert constructor_args["num_tokens_per_page"] == 32
    assert constructor_args["max_num_pages_per_seq_kv"] == 32


def test_paged_run_with_supplied_out_has_no_torch_allocation_or_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    q = torch.empty((2, 4, 128), dtype=torch.float16)
    k_cache = torch.empty((3, 2, 32, 128), dtype=torch.float16)
    v_cache = torch.empty_like(k_cache)
    out = torch.empty_like(q)
    wrapper = context_api.BatchPrefillPagedTSWrapper()
    wrapper._geometry = context_api._PagedContextGeometry(
        device=q.device,
        device_index=0,
        batch_size=1,
        total_q=2,
        max_seq_len_q=2,
        max_seq_len_k=33,
        max_num_pages_per_seq_kv=2,
        num_physical_pages=3,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=q.dtype,
        output_dtype=out.dtype,
        mask_type="dense",
        window_left=-1,
        head_paired=False,
        has_q_offset=False,
        max_q_offset=0,
        q_shape=tuple(q.shape),
        kv_shape=tuple(k_cache.shape),
    )
    wrapper._qo_indptr = torch.tensor([0, 2], dtype=torch.int32)
    wrapper._paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
    wrapper._paged_kv_indices = torch.tensor([2, 0], dtype=torch.int32)
    wrapper._paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    wrapper._logical_kv_indptr = torch.tensor([0, 33], dtype=torch.int32)
    wrapper._seq_lens_kv = torch.tensor([33], dtype=torch.int32)
    wrapper._dense_page_idx_kv = torch.tensor([[[2, 0], [2, 0]]], dtype=torch.int32)
    wrapper._scale_softmax_log2 = torch.tensor([0.1], dtype=torch.float32)
    wrapper._output_scale = torch.tensor([1.0], dtype=torch.float32)
    compiled_calls: list[tuple[object, ...]] = []
    wrapper._compiled = lambda *args: compiled_calls.append(args)
    wrapper._planned = True

    monkeypatch.setattr(
        context_api,
        "_validate_tensor",
        lambda tensor, name: (
            None
            if isinstance(tensor, torch.Tensor)
            else (_ for _ in ()).throw(TypeError(f"{name} must be a torch.Tensor"))
        ),
    )
    monkeypatch.setattr(
        context_api.torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run(out=...) must not allocate a torch tensor")
        ),
    )

    result = wrapper.run(q, k_cache, v_cache, out=out)

    assert result is out
    assert len(compiled_calls) == 1
    call = compiled_calls[0]
    assert call[:4] == (q, k_cache, v_cache, out)
    assert call[6:] == (
        wrapper._qo_indptr,
        wrapper._logical_kv_indptr,
        wrapper._dense_page_idx_kv,
        wrapper._seq_lens_kv,
    )
    assert ".tolist(" not in inspect.getsource(
        context_api.BatchPrefillPagedTSWrapper.run
    )
