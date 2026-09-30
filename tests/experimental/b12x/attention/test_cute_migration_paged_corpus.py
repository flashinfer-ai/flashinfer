from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

import cutlass

from b12x.attention.paged._forward import (
    _build_extend_forward_kernel,
    _descriptor_row_ptrs,
    _encode_plane_tma_descriptors,
    _to_kernel_tensor,
    _torch_to_cutlass_dtype,
)
from b12x.attention.paged.forward_paged import (
    PagedBf16ExtendRawForwardKernel,
    PagedFp8DecodeRawForwardKernel,
    PagedFp8ExtendRawForwardKernel,
)
from b12x.attention.paged.merge import PagedPersistentMergeKernel
from b12x.attention.paged.reference import paged_attention_reference
from b12x.attention.paged.traits import select_paged_forward_traits_from_plan
from b12x._lib.compiler import (
    KernelCompileSpec,
    compile as b12x_compile,
    run_compiled,
)
from b12x._lib.utils import current_cuda_stream
from b12x.attention._shared.contiguous.api import clear_attention_caches
from b12x.attention.paged._forward import paged_attention_forward
from b12x.attention.paged._scratch import B12XPagedAttentionScratchCaps, plan_paged_attention_scratch
from b12x.attention.paged.planner import create_paged_plan

from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import make_paged_inputs, quantize_paged_kv_cache_e4m3


_Q_HEADS = 8
_KV_HEADS = 1
_HEAD_DIM = 256
_PAGE_SIZE = 64


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f32 = actual.float().reshape(-1)
    expected_f32 = expected.float().reshape(-1)
    return float(
        torch.nn.functional.cosine_similarity(
            actual_f32,
            expected_f32,
            dim=0,
        ).item()
    )


def _assert_paged_result(
    actual: torch.Tensor,
    actual_lse_base2: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
    *,
    fp8_kv: bool,
) -> None:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    assert bool(torch.isfinite(actual_f32).all().item())
    assert bool((actual_f32 != 0).any().item())
    assert _cosine_similarity(actual_f32, expected_f32) >= (
        0.999 if fp8_kv else 0.99999
    )
    assert float((actual_f32 - expected_f32).abs().max().item()) <= (
        0.06 if fp8_kv else 0.03
    )
    actual_lse = actual_lse_base2.float() * math.log(2.0)
    assert bool(torch.isfinite(actual_lse).all().item())
    assert float((actual_lse - expected_lse.float()).abs().max().item()) <= (
        0.09 if fp8_kv else 0.05
    )


def _make_fixed_graph_binding(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    output: torch.Tensor,
    mode: str,
    disable_split_kv: bool = False,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
):
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode=mode,
        disable_split_kv=disable_split_kv,
        enable_cuda_graph=True,
        graph_chunk_policy=True,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=plan.total_q,
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=max(plan.new_batch_size, plan.padded_batch_size, 1),
            max_partial_rows=max(plan.total_num_partial_rows, 0),
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=True,
        )
    )
    if mode == "decode":
        scratch_plan.prepare_decode_graph_replay_state(
            batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_cache_page_count=page_table.shape[1],
            force_split_kv=False,
        )
    else:
        scratch_plan.prepare_graph_replay_state(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            active_total_q=q.shape[0],
            disable_split_kv=disable_split_kv,
        )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=q.device)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=q.shape[0],
        disable_split_kv=disable_split_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return binding, scratch, scratch_plan


@pytest.mark.parametrize(
    "q_len",
    (8, 16, 64, 128, 256, 1024, 8),
    ids=("q8", "q16", "q64", "q128", "q256", "q1024", "kv-fp8"),
)
@torch.inference_mode()
def test_paged_fp8_prefill_size_graph_oracle(q_len: int) -> None:
    """Pin every requested generic-prefill size to an oracle and graph replay."""
    require_b12x()
    clear_attention_caches()
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[q_len],
            cache_seqlens=[q_len],
            page_size=_PAGE_SIZE,
            q_heads=_Q_HEADS,
            kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=9100 + q_len,
        )
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    # The serving API accepts [batch] descales directly. Keeping this view
    # avoids a per-call contiguous() conversion while CUDA capture is active.
    k_descale = k_descale.reshape(-1)
    v_descale = v_descale.reshape(-1)
    expected, expected_lse = paged_attention_reference(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    output = torch.empty_like(q)
    binding, scratch, scratch_plan = _make_fixed_graph_binding(
        q=q,
        k_cache=k_fp8,
        v_cache=v_fp8,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        output=output,
        mode="extend",
        k_descale=k_descale,
        v_descale=v_descale,
    )
    del scratch, scratch_plan

    # Compile and validate eager launch before capture, then prove that replay
    # overwrites poisoned outputs without any allocation or replanning.
    actual, actual_lse = paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    _assert_paged_result(
        actual,
        actual_lse,
        expected,
        expected_lse,
        fp8_kv=True,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_attention_forward(binding=binding)
    output.fill_(float("nan"))
    binding.scratch.lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_paged_result(
        output,
        binding.scratch.current_lse_view(),
        expected,
        expected_lse,
        fp8_kv=True,
    )


def _run_paged_forward_graph_oracle(
    kv_dtype: torch.dtype,
    *,
    disable_split_kv: bool,
) -> None:
    """Exercise an API-routed PagedForward graph with live inputs."""
    require_b12x()
    clear_attention_caches()
    mode = "decode" if disable_split_kv else "verify"
    q_len = 1 if mode == "decode" else 4
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[q_len],
            cache_seqlens=[256],
            page_size=_PAGE_SIZE,
            q_heads=_Q_HEADS,
            kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=9200 + (1 if kv_dtype == torch.float8_e4m3fn else 0),
        )
    )
    k_descale = None
    v_descale = None
    if kv_dtype == torch.float8_e4m3fn:
        k_cache, v_cache, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
        )
        k_descale = k_descale.reshape(-1)
        v_descale = v_descale.reshape(-1)
    expected, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    output = torch.empty_like(q)
    binding, scratch, _scratch_plan = _make_fixed_graph_binding(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        output=output,
        mode=mode,
        disable_split_kv=disable_split_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert binding.scratch.plan.split_kv is (not disable_split_kv)

    stable_tensors = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
        "output": output,
        "lse": binding.scratch.lse,
        **{f"scratch_{index}": tensor for index, tensor in enumerate(scratch)},
    }
    stable_ptrs = {name: tensor.data_ptr() for name, tensor in stable_tensors.items()}

    actual, actual_lse = paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    _assert_paged_result(
        actual,
        actual_lse,
        expected,
        expected_lse,
        fp8_kv=kv_dtype == torch.float8_e4m3fn,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert binding.scratch.page_table is not None
        assert binding.scratch.cache_seqlens is not None
        assert binding.scratch.cu_seqlens_q is not None
        binding.scratch.page_table[: page_table.shape[0], : page_table.shape[1]].copy_(
            page_table
        )
        binding.scratch.cache_seqlens[: cache_seqlens.shape[0]].copy_(cache_seqlens)
        binding.scratch.cu_seqlens_q[: cu_seqlens_q.shape[0]].copy_(cu_seqlens_q)
        paged_attention_forward(binding=binding)
    output.fill_(float("nan"))
    binding.scratch.lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_paged_result(
        output,
        binding.scratch.current_lse_view(),
        expected,
        expected_lse,
        fp8_kv=kv_dtype == torch.float8_e4m3fn,
    )

    first_output = output.clone()
    q_2, k_cache_2, v_cache_2, page_table_2, cache_seqlens_2, cu_seqlens_q_2 = (
        make_paged_inputs(
            q_seqlens=[q_len],
            cache_seqlens=[256],
            page_size=_PAGE_SIZE,
            q_heads=_Q_HEADS,
            kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=9300 + (1 if kv_dtype == torch.float8_e4m3fn else 0),
            page_table_width=page_table.shape[1],
            num_pages=k_cache.shape[0],
        )
    )
    if kv_dtype == torch.float8_e4m3fn:
        (
            k_cache_2,
            v_cache_2,
            k_descale_2,
            v_descale_2,
        ) = quantize_paged_kv_cache_e4m3(
            k_cache_2,
            v_cache_2,
            page_table_2,
            cache_seqlens_2,
        )
        assert k_descale is not None and v_descale is not None
        k_descale.copy_(k_descale_2.reshape(-1))
        v_descale.copy_(v_descale_2.reshape(-1))

    q.copy_(q_2)
    k_cache.copy_(k_cache_2)
    v_cache.copy_(v_cache_2)
    page_table.copy_(page_table_2)
    cache_seqlens.copy_(cache_seqlens_2)
    cu_seqlens_q.copy_(cu_seqlens_q_2)
    assert {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    } == stable_ptrs

    expected_2, expected_lse_2 = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    output.fill_(float("nan"))
    binding.scratch.lse.fill_(float("nan"))
    allocated_before_replay = torch.cuda.memory_allocated()
    reserved_before_replay = torch.cuda.memory_reserved()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before_replay
    assert torch.cuda.memory_reserved() == reserved_before_replay
    assert {
        name: tensor.data_ptr() for name, tensor in stable_tensors.items()
    } == stable_ptrs
    _assert_paged_result(
        output,
        binding.scratch.current_lse_view(),
        expected_2,
        expected_lse_2,
        fp8_kv=kv_dtype == torch.float8_e4m3fn,
    )
    assert not torch.equal(output, first_output)


@pytest.mark.parametrize(
    "kv_dtype",
    (torch.bfloat16, torch.float8_e4m3fn),
    ids=("kv-bf16", "kv-fp8"),
)
@torch.inference_mode()
def test_paged_verify_split_merge_graph_oracle(kv_dtype: torch.dtype) -> None:
    """Exercise the API-routed PagedForward plus persistent merge graph."""
    _run_paged_forward_graph_oracle(kv_dtype, disable_split_kv=False)


@torch.inference_mode()
def test_paged_decode_direct_graph_oracle() -> None:
    """Exercise the direct decode PagedForward graph without split/merge."""
    _run_paged_forward_graph_oracle(torch.bfloat16, disable_split_kv=True)


@dataclass(frozen=True)
class _RawCase:
    id: str
    family: str
    q_len: int
    fp8_kv: bool
    split_kv: bool
    cta_tile_q: int | None = None
    page_size: int = _PAGE_SIZE


_RAW_CASES = (
    _RawCase("fp8-decode-direct", "fp8_decode", 1, True, False),
    _RawCase("bf16-extend-direct", "bf16_extend", 8, False, False),
    _RawCase("bf16-extend-split", "bf16_extend", 8, False, True),
    _RawCase("fp8-extend-q32-direct", "fp8_extend", 4, True, False, 32),
    _RawCase("fp8-extend-q32-split", "fp8_extend", 4, True, True, 32),
    _RawCase("fp8-extend-q48-direct", "fp8_extend", 6, True, False, 48),
    _RawCase("fp8-extend-q48-split", "fp8_extend", 6, True, True, 48),
)


def _raw_kernel(case: _RawCase):
    if case.family == "fp8_decode":
        return PagedFp8DecodeRawForwardKernel()
    if case.family == "bf16_extend":
        return PagedBf16ExtendRawForwardKernel(
            split_kv=case.split_kv,
            page_size=case.page_size,
        )
    if case.family == "fp8_extend":
        assert case.cta_tile_q is not None
        return PagedFp8ExtendRawForwardKernel(
            split_kv=case.split_kv,
            cta_tile_q=case.cta_tile_q,
            page_size=case.page_size,
        )
    raise AssertionError(f"unknown raw family {case.family}")


def _as_kernel_tensor(tensor: torch.Tensor, dtype=None, *, align: int | None = None):
    dtype = _torch_to_cutlass_dtype(tensor.dtype) if dtype is None else dtype
    if align is None:
        return _to_kernel_tensor(tensor, dtype)
    return _to_kernel_tensor(tensor, dtype, assumed_align=align)


def _compile_raw_launch(
    case: _RawCase,
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
):
    num_chunks = 2 if case.split_kv else 1
    work_items = num_chunks
    request_indices = torch.zeros(work_items, dtype=torch.int32, device="cuda")
    qo_tile_indices = torch.zeros_like(request_indices)
    kv_tile_indices = torch.arange(
        work_items,
        dtype=torch.int32,
        device="cuda",
    )
    o_indptr = torch.tensor(
        [0, case.q_len * num_chunks],
        dtype=torch.int32,
        device="cuda",
    )
    kv_chunk_size_ptr = torch.tensor(
        [64 if case.split_kv else int(cache_seqlens[0].item())],
        dtype=torch.int32,
        device="cuda",
    )
    block_valid_mask = torch.ones(
        work_items,
        dtype=torch.int32,
        device="cuda",
    )
    if case.split_kv:
        raw_output = torch.empty(
            case.q_len * num_chunks,
            _Q_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        raw_lse = torch.empty(
            case.q_len * num_chunks,
            _Q_HEADS,
            dtype=torch.float32,
            device="cuda",
        )
    else:
        raw_output = torch.empty_like(q)
        raw_lse = torch.empty(
            _Q_HEADS,
            case.q_len,
            dtype=torch.float32,
            device="cuda",
        )

    kernel = _raw_kernel(case)
    args = (
        _as_kernel_tensor(q),
        _as_kernel_tensor(
            k_cache.view(torch.uint8) if case.fp8_kv else k_cache,
            cutlass.Uint8 if case.fp8_kv else cutlass.BFloat16,
        ),
        _as_kernel_tensor(
            v_cache.view(torch.uint8) if case.fp8_kv else v_cache,
            cutlass.Uint8 if case.fp8_kv else cutlass.BFloat16,
        ),
        _as_kernel_tensor(page_table, cutlass.Int32, align=4),
        _as_kernel_tensor(cache_seqlens, cutlass.Int32, align=4),
        _as_kernel_tensor(cu_seqlens_q, cutlass.Int32, align=4),
        _as_kernel_tensor(request_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(qo_tile_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(kv_tile_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(o_indptr, cutlass.Int32, align=4),
        _as_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, align=4),
        _as_kernel_tensor(block_valid_mask, cutlass.Int32, align=4),
        _as_kernel_tensor(raw_output),
        _as_kernel_tensor(raw_lse, cutlass.Float32),
        None if k_descale is None else _as_kernel_tensor(k_descale, cutlass.Float32),
        None if v_descale is None else _as_kernel_tensor(v_descale, cutlass.Float32),
        current_cuda_stream(),
    )
    raw_spec = KernelCompileSpec.from_facts(
        f"diagnostic.attention.paged.raw.{case.family}",
        1,
        ("case", case.id),
        ("split_kv", case.split_kv),
        ("cta_tile_q", case.cta_tile_q),
        ("page_size", case.page_size),
    )
    compiled = b12x_compile(kernel, *args, compile_spec=raw_spec)

    final_output = raw_output
    final_lse = raw_lse.transpose(0, 1) if not case.split_kv else raw_lse
    merge_compiled = None
    merge_args = None
    merge_keepalive: tuple[torch.Tensor, ...] = ()
    if case.split_kv:
        merge_indptr = torch.arange(
            0,
            (case.q_len + 1) * num_chunks,
            num_chunks,
            dtype=torch.int32,
            device="cuda",
        )
        final_output = torch.empty_like(q)
        final_lse_storage = torch.empty(
            _Q_HEADS,
            case.q_len,
            dtype=torch.float32,
            device="cuda",
        )
        total_rows_ptr = torch.tensor(
            [case.q_len],
            dtype=torch.int32,
            device="cuda",
        )
        merge_kernel = PagedPersistentMergeKernel(
            cutlass.BFloat16,
            cutlass.BFloat16,
            head_dim=_HEAD_DIM,
            persistent_ctas=2,
        )
        merge_args = (
            _as_kernel_tensor(raw_output),
            _as_kernel_tensor(raw_lse, cutlass.Float32),
            _as_kernel_tensor(merge_indptr, cutlass.Int32, align=4),
            _as_kernel_tensor(cache_seqlens, cutlass.Int32, align=4),
            _as_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, align=4),
            _as_kernel_tensor(final_output),
            _as_kernel_tensor(final_lse_storage, cutlass.Float32),
            _as_kernel_tensor(total_rows_ptr, cutlass.Int32, align=4),
            current_cuda_stream(),
        )
        merge_spec = KernelCompileSpec.from_facts(
            "attention.paged.merge",
            2,
            ("diagnostic_raw_case", case.id),
            ("head_dim", _HEAD_DIM),
            ("persistent_ctas", 2),
        )
        merge_compiled = b12x_compile(
            merge_kernel,
            *merge_args,
            compile_spec=merge_spec,
        )
        final_lse = final_lse_storage.transpose(0, 1)
        merge_keepalive = (merge_indptr, final_lse_storage, total_rows_ptr)

    keepalive = (
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        kv_chunk_size_ptr,
        block_valid_mask,
        raw_output,
        raw_lse,
        *merge_keepalive,
    )
    return (
        compiled,
        args,
        merge_compiled,
        merge_args,
        final_output,
        final_lse,
        keepalive,
    )


def _launch_compiled_pair(
    compiled,
    args,
    merge_compiled,
    merge_args,
) -> None:
    stream = current_cuda_stream()
    run_compiled(compiled, (*args[:-1], stream))
    if merge_compiled is not None:
        assert merge_args is not None
        run_compiled(merge_compiled, (*merge_args[:-1], stream))


@pytest.mark.parametrize("case", _RAW_CASES, ids=lambda case: case.id)
@torch.inference_mode()
def test_paged_unreachable_raw_body_graph_oracle(case: _RawCase) -> None:
    """Migration-only proof for raw bodies that have no serving call sites."""
    require_b12x()
    cache_len = 128 if case.split_kv else 64
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[case.q_len],
            cache_seqlens=[cache_len],
            page_size=_PAGE_SIZE,
            q_heads=_Q_HEADS,
            kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=9300 + case.q_len + (100 if case.split_kv else 0),
        )
    )
    k_descale = None
    v_descale = None
    if case.fp8_kv:
        k_cache, v_cache, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
        )
        k_descale = k_descale.reshape(-1)
        v_descale = v_descale.reshape(-1)
    expected, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    (
        compiled,
        args,
        merge_compiled,
        merge_args,
        final_output,
        final_lse,
        keepalive,
    ) = _compile_raw_launch(
        case,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    # The CuTe runtime arguments contain device pointers, so retain every
    # owning torch tensor through eager launch, capture, and replay.
    assert keepalive

    _launch_compiled_pair(compiled, args, merge_compiled, merge_args)
    torch.cuda.synchronize()
    _assert_paged_result(
        final_output,
        final_lse,
        expected,
        expected_lse,
        fp8_kv=case.fp8_kv,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_compiled_pair(compiled, args, merge_compiled, merge_args)
    final_output.fill_(float("nan"))
    final_lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_paged_result(
        final_output,
        final_lse,
        expected,
        expected_lse,
        fp8_kv=case.fp8_kv,
    )


_RAW_PAGE128_HIGH_PID_CASES = (
    _RawCase(
        id="bf16-extend-page128-high-pid",
        family="bf16_extend",
        q_len=8,
        fp8_kv=False,
        split_kv=False,
        page_size=128,
    ),
    _RawCase(
        id="fp8-extend-q32-page128-high-pid",
        family="fp8_extend",
        q_len=4,
        fp8_kv=True,
        split_kv=False,
        cta_tile_q=32,
        page_size=128,
    ),
)


@pytest.mark.parametrize(
    "case",
    _RAW_PAGE128_HIGH_PID_CASES,
    ids=lambda case: case.id,
)
@torch.inference_mode()
def test_raw_extend_page128_graph_handles_high_pool_page_id(case: _RawCase) -> None:
    """Exercise raw two/four-plane TMA helpers beyond a 2^31 byte offset."""
    device = require_b12x()
    torch.manual_seed(20260722 + case.q_len)

    kv_dtype = torch.float8_e4m3fn if case.fp8_kv else torch.bfloat16
    element_size = torch.empty((), dtype=kv_dtype).element_size()
    page_stride_bytes = (
        case.page_size * _KV_HEADS * _HEAD_DIM * element_size
    )
    int32_max = torch.iinfo(torch.int32).max
    high_page_id = int32_max // page_stride_bytes + 2
    num_cache_pages = high_page_id + 1

    # Keep the multi-GiB pool uninitialized except for the single live tail
    # page.  Allocator-recycled ids in serving routinely expose this exact
    # address range even when unit-test page tables usually start at zero.
    cache_shape = (
        num_cache_pages,
        case.page_size,
        _KV_HEADS,
        _HEAD_DIM,
    )
    k_cache = torch.empty(cache_shape, dtype=kv_dtype, device=device)
    v_cache = torch.empty(cache_shape, dtype=kv_dtype, device=device)
    assert k_cache.stride(0) * k_cache.element_size() == page_stride_bytes
    assert high_page_id * page_stride_bytes > int32_max

    live_shape = (case.page_size, _KV_HEADS, _HEAD_DIM)
    live_k = torch.randn(live_shape, dtype=torch.bfloat16, device=device) / 4
    live_v = torch.randn(live_shape, dtype=torch.bfloat16, device=device) / 4
    k_cache[high_page_id].copy_(live_k.to(kv_dtype))
    v_cache[high_page_id].copy_(live_v.to(kv_dtype))
    live_k_expected = k_cache[high_page_id].clone()
    live_v_expected = v_cache[high_page_id].clone()

    # The manual helpers flatten each page into stage-row TMA coordinates.
    # Check that the coordinate's underlying byte address—not merely the
    # allocation size—crosses the signed-Int32 boundary.
    stage_tile_rows = 64 if case.fp8_kv else 32
    page_tiles_per_page = case.page_size // stage_tile_rows
    tile_stride_bytes = stage_tile_rows * _HEAD_DIM * element_size
    assert (
        high_page_id * page_tiles_per_page * tile_stride_bytes > int32_max
    )

    q = torch.randn(
        (case.q_len, _Q_HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ) / 4
    page_table = torch.tensor(
        [[high_page_id]], dtype=torch.int32, device=device
    )
    page_table_expected = page_table.clone()
    cache_seqlens = torch.tensor(
        [case.page_size], dtype=torch.int32, device=device
    )
    cu_seqlens_q = torch.tensor(
        [0, case.q_len], dtype=torch.int32, device=device
    )
    k_descale = (
        torch.ones(1, dtype=torch.float32, device=device)
        if case.fp8_kv
        else None
    )
    v_descale = (
        torch.ones(1, dtype=torch.float32, device=device)
        if case.fp8_kv
        else None
    )
    expected, expected_lse = paged_attention_reference(
        q,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=True,
    )
    (
        compiled,
        args,
        merge_compiled,
        merge_args,
        final_output,
        final_lse,
        keepalive,
    ) = _compile_raw_launch(
        case,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert keepalive
    assert merge_compiled is None
    assert merge_args is None

    _launch_compiled_pair(compiled, args, merge_compiled, merge_args)
    torch.cuda.synchronize(device)
    _assert_paged_result(
        final_output,
        final_lse,
        expected,
        expected_lse,
        fp8_kv=case.fp8_kv,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_compiled_pair(compiled, args, merge_compiled, merge_args)
    final_output.fill_(torch.nan)
    final_lse.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize(device)
    _assert_paged_result(
        final_output,
        final_lse,
        expected,
        expected_lse,
        fp8_kv=case.fp8_kv,
    )
    assert torch.equal(k_cache[high_page_id], live_k_expected)
    assert torch.equal(v_cache[high_page_id], live_v_expected)
    assert torch.equal(page_table, page_table_expected)


def _fp8_planewords_expected(v_cache: torch.Tensor, page_idx: int) -> torch.Tensor:
    # The rank-2 TMA tile partitions each 16-byte vector into two two-word
    # planes.  Physical shared-memory order is then [word-in-plane,
    # vector-seed, row, byte], with the default Swizzle<3,4,3> XORing the
    # vector seed by the low three row bits.
    page = v_cache.view(torch.uint8)[page_idx, :, 0, :].reshape(
        _PAGE_SIZE,
        16,
        4,
        4,
    )
    rows = torch.arange(_PAGE_SIZE, dtype=torch.long, device=page.device)
    vector_seeds = (
        torch.arange(8, dtype=torch.long, device=page.device)[:, None]
        + torch.tensor([0, 8], dtype=torch.long, device=page.device)
    ).reshape(-1)
    source_vectors = vector_seeds[:, None] ^ (rows[None, :] % 8)
    planes = []
    for plane_idx in range(2):
        for word_in_vector in range(plane_idx * 2, plane_idx * 2 + 2):
            planes.append(
                page[
                    rows[None, :],
                    source_vectors,
                    word_in_vector,
                    :,
                ].reshape(-1)
            )
    return torch.cat(planes)


@torch.inference_mode()
def test_paged_fp8_planewords_atom_byte_contract_graph(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Check every PLANEWORDS byte through the supported CuTe-atom path."""
    require_b12x()
    monkeypatch.setenv("B12X_PAGED_KV_TMA", "1")
    monkeypatch.setenv("B12X_PAGED_KV_DEBUG_DUMP", "PLANEWORDS")
    monkeypatch.setenv("B12X_PAGED_KV_TMA_PLANE_SWIZZLE", "3,4,3")
    clear_attention_caches()
    request.addfinalizer(clear_attention_caches)
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q = (
        make_paged_inputs(
            q_seqlens=[4],
            cache_seqlens=[64],
            page_size=_PAGE_SIZE,
            q_heads=_Q_HEADS,
            kv_heads=_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=torch.bfloat16,
            seed=9400,
        )
    )
    k_fp8, v_fp8, k_descale, v_descale = quantize_paged_kv_cache_e4m3(
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
    )
    k_descale = k_descale.reshape(-1)
    v_descale = v_descale.reshape(-1)
    plan = create_paged_plan(
        q,
        k_fp8,
        v_fp8,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        mode="extend",
        disable_split_kv=True,
    )
    traits = select_paged_forward_traits_from_plan(plan)
    assert traits.cta_tile_q == 16
    kernel = _build_extend_forward_kernel(
        traits,
        False,
        False,
        plan.window_left,
        False,
        False,
        False,
        False,
        int(plan.page_size),
    )
    assert kernel.debug_dump_paged_kv_planewords
    # Exercise the byte-exact PLANEWORDS dump through the supported CuTe-atom
    # transport.  The dead raw tensor-map debug entrypoint was removed because
    # its instruction helper raised before IR generation.
    kernel.use_paged_kv_tma_fp8_raw_issue = False
    kernel.use_paged_k_tma = False
    kernel.use_paged_v_tma = True

    request_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    qo_tile_indices = torch.zeros_like(request_indices)
    kv_tile_indices = torch.zeros_like(request_indices)
    o_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_chunk_size_ptr = torch.tensor([64], dtype=torch.int32, device="cuda")
    kv_window_start = torch.zeros(1, dtype=torch.int32, device="cuda")
    block_valid_mask = torch.ones(1, dtype=torch.int32, device="cuda")
    attention_sink_bias = torch.empty(0, dtype=torch.float32, device="cuda")
    output = torch.empty_like(q)
    lse = torch.empty(_Q_HEADS, q.shape[0], dtype=torch.float32, device="cuda")
    k_desc = _encode_plane_tma_descriptors(
        k_fp8,
        plane_cols=kernel.kv_tma_plane_head_dim,
        tile_rows=kernel.stage_tile_rows,
    )
    v_desc = _encode_plane_tma_descriptors(
        v_fp8,
        plane_cols=kernel.kv_tma_plane_head_dim,
        tile_rows=kernel.stage_tile_rows,
    )
    k_desc_ptrs = _descriptor_row_ptrs(k_desc)
    v_desc_ptrs = _descriptor_row_ptrs(v_desc)
    args = (
        _as_kernel_tensor(q),
        _as_kernel_tensor(k_fp8.view(torch.uint8), cutlass.Uint8),
        _as_kernel_tensor(v_fp8.view(torch.uint8), cutlass.Uint8),
        _as_kernel_tensor(page_table, cutlass.Int32, align=4),
        _as_kernel_tensor(cache_seqlens, cutlass.Int32, align=4),
        _as_kernel_tensor(cu_seqlens_q, cutlass.Int32, align=4),
        _as_kernel_tensor(request_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(qo_tile_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(kv_tile_indices, cutlass.Int32, align=4),
        _as_kernel_tensor(o_indptr, cutlass.Int32, align=4),
        _as_kernel_tensor(kv_chunk_size_ptr, cutlass.Int32, align=4),
        _as_kernel_tensor(kv_window_start, cutlass.Int32, align=4),
        _as_kernel_tensor(block_valid_mask, cutlass.Int32, align=4),
        None,
        None,
        None,
        None,
        _as_kernel_tensor(attention_sink_bias, cutlass.Float32),
        None,
        _as_kernel_tensor(output),
        _as_kernel_tensor(lse, cutlass.Float32),
        _as_kernel_tensor(k_descale, cutlass.Float32),
        _as_kernel_tensor(v_descale, cutlass.Float32),
        _as_kernel_tensor(k_desc_ptrs, cutlass.Int64, align=8),
        _as_kernel_tensor(v_desc_ptrs, cutlass.Int64, align=8),
        current_cuda_stream(),
    )
    spec = KernelCompileSpec.from_facts(
        "diagnostic.paged.planewords",
        1,
        ("cta_tile_q", traits.cta_tile_q),
        ("page_size", _PAGE_SIZE),
        ("head_dim", _HEAD_DIM),
    )
    compiled = b12x_compile(kernel, *args, compile_spec=spec)
    expected = _fp8_planewords_expected(
        v_fp8,
        int(page_table[0, 0].item()),
    )

    run_compiled(compiled, args)
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8).reshape(-1), expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_compiled(compiled, (*args[:-1], current_cuda_stream()))
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.uint8).reshape(-1), expected)
