"""
Tests for FP8 output support in CUTLASS MLA paged attention (PR #2779).

Tests:
1. FP8 output matches bf16 output + separate quantization
2. Validation: o_scale without out tensor raises error
3. Validation: o_scale with non-FP8 out tensor raises error
4. Validation: bf16 out tensor without o_scale still works
"""

import math

import pytest
import torch
from tests.test_helpers.test_helpers import clear_cuda_cache

import flashinfer
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported


def _skip_if_unsupported(device):
    if not is_sm100a_supported(device) and not is_sm110a_supported(device):
        pytest.skip("CUTLASS MLA requires SM100a+ (Blackwell)")


def _setup_mla_inputs(batch_size, max_seq_len, page_size, dtype, device):
    """Create test inputs matching test_cutlass_mla pattern."""
    torch.manual_seed(42)

    num_local_heads = 128
    head_dim_ckv = 512
    head_dim_kpe = 64
    total_page_num = 8192

    q_nope = torch.randn(
        batch_size, num_local_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size, num_local_heads, head_dim_kpe, dtype=dtype, device=device
    )
    ckv_cache = torch.randn(
        total_page_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe_cache = torch.randn(
        total_page_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )
    kv_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    page_num_per_batch = math.ceil(max_seq_len / page_size)
    page_table = torch.randint(
        0,
        total_page_num,
        (batch_size, page_num_per_batch),
        dtype=torch.int32,
        device=device,
    )

    return q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table


def _planned_cutlass_backend(
    *,
    output_dtype=torch.bfloat16,
    output_scale="none",
    batch_size=1,
    page_size=1,
):
    from flashinfer.mla._batch_mla._backends.cutlass_backend import (
        _BatchMLAPagedAttentionCutlassBackend,
    )

    backend = _BatchMLAPagedAttentionCutlassBackend(torch.empty(16, dtype=torch.uint8))
    backend._batch_size = batch_size
    backend._page_size = page_size
    backend._head_dim_ckv = 512
    backend._head_dim_kpe = 64
    backend._q_data_type = torch.bfloat16
    backend._kv_data_type = torch.bfloat16
    backend._output_dtype = output_dtype
    backend._output_scale = output_scale
    backend._kv_len = torch.tensor([1], dtype=torch.int32)
    backend._page_table = torch.zeros((1, 128), dtype=torch.int32)
    backend._empty_lse = torch.empty(0, dtype=torch.float32)
    return backend


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [128, 1024])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
def test_cutlass_mla_fp8_output(batch_size, max_seq_len, page_size, fp8_dtype):
    """FP8 output should match bf16 output + manual quantization."""
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    _skip_if_unsupported(device)

    dtype = torch.bfloat16
    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        batch_size, max_seq_len, page_size, dtype, device
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Reference: bf16 output
    wrapper_ref = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace, backend="cutlass"
    )
    o_bf16 = wrapper_ref.run(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_len=kv_lens, page_table=page_table
    )

    # o_scale is dequant scale: real = quantized * o_scale.
    amax = o_bf16.float().abs().max().item()
    fp8_max = torch.finfo(fp8_dtype).max
    o_scale = amax / fp8_max if amax > 0 else 1.0

    # Manual quantization: bf16 -> fp8
    o_manual_fp8 = (o_bf16.float() / o_scale).to(fp8_dtype)

    # Fused: direct FP8 output from kernel
    wrapper_fused = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace, backend="cutlass"
    )
    o_fused_fp8 = torch.empty(q_nope.shape, dtype=fp8_dtype, device=device)
    wrapper_fused.run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out=o_fused_fp8,
        kv_len=kv_lens,
        page_table=page_table,
        o_scale=o_scale,
    )

    # Compare: dequantize both and check they match
    o_manual_dequant = o_manual_fp8.float() * o_scale
    o_fused_dequant = o_fused_fp8.float() * o_scale

    # FP8 has limited precision, so use relaxed tolerance
    torch.testing.assert_close(o_fused_dequant, o_manual_dequant, rtol=1e-1, atol=1e-1)

    # Also verify the fused output is close to the original bf16 output
    torch.testing.assert_close(o_fused_dequant, o_bf16.float(), rtol=1e-1, atol=1e-1)


def test_cutlass_mla_fp8_output_validation_no_out():
    """o_scale without out tensor should raise ValueError."""
    device = torch.device("cuda:0")
    _skip_if_unsupported(device)

    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        1, 128, 1, torch.bfloat16, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutlass")

    with pytest.raises(ValueError, match="out tensor must be provided"):
        wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            o_scale=0.1,
            kv_len=kv_lens,
            page_table=page_table,
        )


def test_cutlass_mla_fp8_output_validation_wrong_dtype():
    """o_scale with non-FP8 out tensor should raise ValueError."""
    device = torch.device("cuda:0")
    _skip_if_unsupported(device)

    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        1, 128, 1, torch.bfloat16, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutlass")

    out_bf16 = torch.empty_like(q_nope)
    with pytest.raises(ValueError, match="out must be an FP8 tensor"):
        wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            out=out_bf16,
            o_scale=0.1,
            kv_len=kv_lens,
            page_table=page_table,
        )


@pytest.mark.parametrize("o_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_cutlass_mla_fp8_output_validation_invalid_scale(o_scale):
    """o_scale must be finite and positive."""
    device = torch.device("cuda:0")
    _skip_if_unsupported(device)

    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        1, 128, 1, torch.bfloat16, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="cutlass")
    out_fp8 = torch.empty(q_nope.shape, dtype=torch.float8_e4m3fn, device=device)

    with pytest.raises(ValueError, match="o_scale must be a finite positive value"):
        wrapper.run(
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            out=out_fp8,
            o_scale=o_scale,
            kv_len=kv_lens,
            page_table=page_table,
        )


def test_cutlass_mla_bf16_output_unchanged():
    """Default bf16 path (no o_scale) should still work correctly."""
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    _skip_if_unsupported(device)

    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        2, 256, 16, torch.bfloat16, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Run without o_scale (auto-allocated output)
    wrapper1 = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace, backend="cutlass"
    )
    o1 = wrapper1.run(
        q_nope, q_pe, ckv_cache, kpe_cache, kv_len=kv_lens, page_table=page_table
    )

    # Run with pre-allocated bf16 output (no o_scale)
    wrapper2 = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace, backend="cutlass"
    )
    o2 = torch.empty_like(q_nope)
    wrapper2.run(
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        out=o2,
        kv_len=kv_lens,
        page_table=page_table,
    )

    torch.testing.assert_close(o1, o2, rtol=1e-3, atol=1e-3)


def test_planned_cutlass_fp8_output_uses_ckv_width_and_preserves_identity():
    class _FakeModule:
        def cutlass_mla_paged_attention(self, *args):
            self.args = args

    backend = _planned_cutlass_backend(
        output_dtype=torch.float8_e4m3fn,
        output_scale="per-tensor",
    )
    backend._cached_module = _FakeModule()
    query = torch.empty((1, 128, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 1, 576), dtype=torch.bfloat16)
    out = torch.empty((1, 128, 512), dtype=torch.float8_e4m3fn)

    actual = backend.run_from_wrapper(
        query=query,
        kv_cache=kv_cache,
        out=out,
        lse=None,
        return_lse=False,
        profiler_buffer=None,
        kv_len=None,
        page_table=None,
        return_lse_base_on_e=False,
        o_scale=0.25,
        ckv_scale=None,
        ckv_scale_arr=None,
        kpe_scale=None,
    )

    assert actual is out
    assert backend._cached_module.args[1] is out


@pytest.mark.parametrize(
    "field,match",
    [
        ("query_dtype", "query dtype.*planned"),
        ("kv_dtype", "KV cache dtype.*planned"),
        ("kv_len_dtype", "kv_len must have dtype torch.int32"),
        ("page_table_dtype", "page_table must have dtype torch.int32"),
    ],
)
def test_planned_cutlass_run_enforces_planned_dtypes(field, match):
    backend = _planned_cutlass_backend()
    backend._cached_module = object()
    query = torch.empty((1, 128, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 1, 576), dtype=torch.bfloat16)
    if field == "query_dtype":
        query = query.to(torch.float16)
    elif field == "kv_dtype":
        kv_cache = kv_cache.to(torch.float16)
    elif field == "kv_len_dtype":
        backend._kv_len = backend._kv_len.to(torch.int64)
    else:
        backend._page_table = backend._page_table.to(torch.int64)

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(
            query=query,
            kv_cache=kv_cache,
            out=None,
            lse=None,
            return_lse=False,
            profiler_buffer=None,
            kv_len=None,
            page_table=None,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            ckv_scale_arr=None,
            kpe_scale=None,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("short_kv_len", "kv_len.*one entry"),
        ("batch", "query batch size.*planned"),
        ("page_size", "KV cache page size.*planned"),
    ],
)
def test_planned_cutlass_run_enforces_planned_metadata_shape(mutation, match):
    backend = _planned_cutlass_backend()
    backend._cached_module = object()
    query = torch.empty((1, 128, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 1, 576), dtype=torch.bfloat16)
    kv_len = backend._kv_len
    page_table = backend._page_table
    if mutation == "short_kv_len":
        query = torch.empty((2, 128, 576), dtype=torch.bfloat16)
        page_table = torch.zeros((2, 128), dtype=torch.int32)
    elif mutation == "batch":
        query = torch.empty((2, 128, 576), dtype=torch.bfloat16)
        kv_len = torch.tensor([1, 1], dtype=torch.int32)
        page_table = torch.zeros((2, 128), dtype=torch.int32)
    else:
        kv_cache = torch.empty((1, 2, 576), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=match):
        backend.run_from_wrapper(
            query=query,
            kv_cache=kv_cache,
            out=None,
            lse=None,
            return_lse=False,
            profiler_buffer=None,
            kv_len=kv_len,
            page_table=page_table,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            ckv_scale_arr=None,
            kpe_scale=None,
        )


@pytest.mark.parametrize(
    "output_dtype,output_scale",
    [
        (torch.float8_e4m3fn, "none"),
        (torch.bfloat16, "per-tensor"),
    ],
)
def test_cutlass_plan_rejects_impossible_output_contract(output_dtype, output_scale):
    from flashinfer.mla._batch_mla._backends.cutlass_backend import (
        _BatchMLAPagedAttentionCutlassBackend,
    )

    backend = _BatchMLAPagedAttentionCutlassBackend(torch.empty(16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="output"):
        backend.plan(
            num_heads=128,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=False,
            sm_scale=1.0 / math.sqrt(192),
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            output_dtype=output_dtype,
            output_scale=output_scale,
            use_profiler=False,
            batch_size=1,
            kv_len=torch.tensor([1], dtype=torch.int32),
            page_table=torch.zeros((1, 128), dtype=torch.int32),
        )


if __name__ == "__main__":
    test_cutlass_mla_fp8_output(1, 128, 1, torch.float8_e4m3fn)
    test_cutlass_mla_fp8_output(4, 1024, 16, torch.float8_e4m3fn)
    test_cutlass_mla_fp8_output_validation_no_out()
    test_cutlass_mla_fp8_output_validation_wrong_dtype()
    test_cutlass_mla_bf16_output_unchanged()
