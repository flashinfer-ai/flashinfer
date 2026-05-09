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


def test_cutlass_mla_fp8_non_cutlass_backend_rejected():
    """o_scale with non-cutlass backend should raise ValueError.

    We directly set _backend to 'fa2' without calling plan() to avoid
    JIT compilation dependencies. The o_scale check happens before any
    module call.
    """
    device = torch.device("cuda:0")

    q_nope, q_pe, ckv_cache, kpe_cache, kv_lens, page_table = _setup_mla_inputs(
        1, 128, 1, torch.float16, device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    # Force backend without plan() to avoid JIT compilation
    wrapper._backend = "fa2"

    out_fp8 = torch.empty(1, 128, 512, dtype=torch.float8_e4m3fn, device=device)
    with pytest.raises(ValueError, match="o_scale is only supported with the cutlass"):
        wrapper.run(q_nope, q_pe, ckv_cache, kpe_cache, out=out_fp8, o_scale=0.1)


if __name__ == "__main__":
    test_cutlass_mla_fp8_output(1, 128, 1, torch.float8_e4m3fn)
    test_cutlass_mla_fp8_output(4, 1024, 16, torch.float8_e4m3fn)
    test_cutlass_mla_fp8_output_validation_no_out()
    test_cutlass_mla_fp8_output_validation_wrong_dtype()
    test_cutlass_mla_bf16_output_unchanged()
