"""
Tests for CuTe DSL FMHA prefill kernel (cubin distribution).

These tests verify the DSL FMHA kernel loaded via ExternalBinaryModule
against a PyTorch reference implementation.

Usage:
    # Set local .so directory (from compile_dsl_fmha.py output)
    export FLASHINFER_DSL_FMHA_LOCAL_DIR=/path/to/build/cute_dsl_fmha

    pytest tests/attention/test_cute_dsl_fmha_prefill.py -x -v
"""

import math
import os

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

# Skip entire module if no local .so directory and no artifactory configured
DSL_FMHA_LOCAL_DIR = os.environ.get("FLASHINFER_DSL_FMHA_LOCAL_DIR")
DSL_FMHA_AVAILABLE = DSL_FMHA_LOCAL_DIR is not None and os.path.isdir(
    DSL_FMHA_LOCAL_DIR
)

pytestmark = [
    pytest.mark.skipif(
        not DSL_FMHA_AVAILABLE,
        reason="FLASHINFER_DSL_FMHA_LOCAL_DIR not set or directory does not exist. "
        "Run compile_dsl_fmha.py first and set the env var.",
    ),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
        reason="Requires SM10x (Blackwell) GPU with CUDA 12.8+",
    ),
]


def _quantize_to_fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize a tensor to FP8 e4m3fn with the given scale.

    Convention: x_real = x_fp8 * scale, so x_fp8 = x_real / scale.
    """
    return (x / scale).to(torch.float8_e4m3fn)


# Per-tensor scales used for FP8 tests
FP8_SCALE_Q = 0.05
FP8_SCALE_K = 0.04
FP8_SCALE_V = 0.06


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """PyTorch reference implementation of multi-head attention.

    Args:
        q: (B, S_q, H_q, D)
        k: (B, S_k, H_k, D)
        v: (B, S_k, H_k, D_v)

    Returns:
        o: (B, S_q, H_q, D_v)
    """
    B, S_q, H_q, D = q.shape
    _, S_k, H_k, D_v = v.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # GQA: repeat K/V heads
    if H_q != H_k:
        repeat_factor = H_q // H_k
        k = k.repeat_interleave(repeat_factor, dim=2)
        v = v.repeat_interleave(repeat_factor, dim=2)

    # (B, H, S, D)
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()

    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if is_causal:
        # Bottom-right aligned causal mask (matches FlashInfer convention):
        # position i can attend to key positions 0..(i + S_k - S_q)
        mask = torch.triu(
            torch.ones(S_q, S_k, device=q.device, dtype=torch.bool),
            diagonal=S_k - S_q + 1,
        )
        attn.masked_fill_(mask, float("-inf"))

    attn = torch.softmax(attn, dim=-1)
    o = torch.matmul(attn, v)

    # (B, S_q, H, D_v)
    o = o.transpose(1, 2)
    return o


# =============================================================================
# Test: cute_dsl_fmha_prefill (direct API)
# =============================================================================


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16]
)  # TODO: add torch.float8_e4m3fn once pytest CUDA state issue is resolved
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "B, S_q, S_k, H_q, H_k",
    [
        (1, 128, 128, 8, 8),  # basic MHA
        (2, 256, 256, 8, 8),  # batched
        (1, 128, 256, 8, 8),  # S_q != S_k
        (2, 128, 128, 16, 4),  # GQA
        # Production-scale shapes
        (1, 8192, 8192, 8, 8),  # long context
        (1, 8192, 32768, 8, 8),  # long KV cache
        (4, 1024, 81920, 8, 8),  # batched decode-like prefill
    ],
)
def test_cute_dsl_fmha_prefill_direct(
    enable_tvm_ffi, dtype, head_dim, is_causal, B, S_q, S_k, H_q, H_k
):
    """Test cute_dsl_fmha_prefill directly with torch tensors."""

    from flashinfer.attention_dsl.cute_dsl.fmha import cute_dsl_fmha_prefill

    torch.manual_seed(42)
    D = head_dim
    D_v = D if D != 192 else 128

    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(B, S_k, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(B, S_k, H_k, D_v, dtype=dtype, device="cuda")
    o = torch.zeros(B, S_q, H_q, D_v, dtype=dtype, device="cuda")

    cute_dsl_fmha_prefill(
        q, k, v, o, is_causal=is_causal, enable_tvm_ffi=enable_tvm_ffi
    )
    torch.cuda.synchronize()

    o_ref = reference_attention(q, k, v, is_causal=is_causal).to(dtype)

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)


# =============================================================================
# Test: cute_dsl_fmha_prefill with FP8 (direct API)
# =============================================================================


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "B, S_q, S_k, H_q, H_k",
    [
        (1, 128, 128, 8, 8),
        (2, 256, 256, 8, 8),
        (2, 128, 128, 16, 4),  # GQA
    ],
)
def test_cute_dsl_fmha_prefill_fp8(
    enable_tvm_ffi, head_dim, is_causal, B, S_q, S_k, H_q, H_k
):
    """Test cute_dsl_fmha_prefill with FP8 input and FP16 output (mixed precision)."""

    from flashinfer.attention_dsl.cute_dsl.fmha import cute_dsl_fmha_prefill

    torch.manual_seed(42)
    D = head_dim

    # Create float32 reference tensors with small values (FP8 friendly range)
    q_f32 = torch.randn(B, S_q, H_q, D, dtype=torch.float32, device="cuda") * 0.1
    k_f32 = torch.randn(B, S_k, H_k, D, dtype=torch.float32, device="cuda") * 0.1
    v_f32 = torch.randn(B, S_k, H_k, D, dtype=torch.float32, device="cuda") * 0.1

    # Quantize to FP8
    q_fp8 = _quantize_to_fp8(q_f32, FP8_SCALE_Q)
    k_fp8 = _quantize_to_fp8(k_f32, FP8_SCALE_K)
    v_fp8 = _quantize_to_fp8(v_f32, FP8_SCALE_V)

    # Output in FP16 (mixed precision: e4m3 in → fp16 out)
    o = torch.zeros(B, S_q, H_q, D, dtype=torch.float16, device="cuda")

    cute_dsl_fmha_prefill(
        q_fp8,
        k_fp8,
        v_fp8,
        o,
        is_causal=is_causal,
        scale_q=FP8_SCALE_Q,
        scale_k=FP8_SCALE_K,
        scale_v=FP8_SCALE_V,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    torch.cuda.synchronize()

    # Reference: dequantize FP8 back to float32, then compute attention
    q_deq = q_fp8.float() * FP8_SCALE_Q
    k_deq = k_fp8.float() * FP8_SCALE_K
    v_deq = v_fp8.float() * FP8_SCALE_V
    o_ref = reference_attention(q_deq, k_deq, v_deq, is_causal=is_causal).to(
        torch.float16
    )

    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


# =============================================================================
# Test: single_prefill_with_kv_cache with backend="cute-dsl"
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_single_prefill_cute_dsl_backend(dtype, head_dim, is_causal):
    """Test single_prefill_with_kv_cache with backend='cute-dsl'."""

    import flashinfer

    torch.manual_seed(42)
    S_q, S_k = 128, 256
    H_q, H_k = 8, 8
    D = head_dim
    D_v = D if D != 192 else 128

    q = torch.randn(S_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(S_k, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(S_k, H_k, D_v, dtype=dtype, device="cuda")

    o = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=is_causal,
        backend="cute-dsl",
    )
    torch.cuda.synchronize()

    # Reference: add batch dim for comparison
    o_ref = (
        reference_attention(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            is_causal=is_causal,
        )
        .squeeze(0)
        .to(dtype)
    )

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)


# =============================================================================
# Test: kernel loading
# =============================================================================


def test_kernel_loading():
    """Test that get_cute_dsl_fmha_kernel can load a .so file."""

    from flashinfer.attention_dsl.cute_dsl.fmha import get_cute_dsl_fmha_kernel

    # Try to load one variant
    fn = get_cute_dsl_fmha_kernel(
        in_dtype=torch.float16,
        out_dtype=torch.float16,
        head_dim=128,
        is_causal=False,
    )
    assert callable(fn), f"Expected callable, got {type(fn)}"


# =============================================================================
# Test: cute-dsl vs FlashInfer fa3 backend
# =============================================================================


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "S_q, S_k, H_q, H_k",
    [
        (128, 128, 8, 8),
        (128, 256, 8, 8),
        (256, 256, 16, 4),  # GQA
    ],
)
def test_cute_dsl_vs_flashinfer(dtype, head_dim, is_causal, S_q, S_k, H_q, H_k):
    """Compare cute-dsl backend output against FlashInfer's fa3 backend."""

    import flashinfer

    torch.manual_seed(42)
    D = head_dim

    q = torch.randn(S_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(S_k, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(S_k, H_k, D, dtype=dtype, device="cuda")

    # cute-dsl backend
    o_dsl = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=is_causal,
        backend="cute-dsl",
    )
    torch.cuda.synchronize()

    # FlashInfer fa3 backend (auto selects fa3 on SM100)
    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=is_causal,
        backend="auto",
    )

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o_dsl, o_fa3, rtol=rtol, atol=atol)


# =============================================================================
# Test: BatchPrefillWithRaggedKVCacheWrapper with backend="cute-dsl"
# =============================================================================


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16]
)  # TODO: add torch.float8_e4m3fn once pytest CUDA state issue is resolved
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_batch_ragged_prefill_cute_dsl(dtype, head_dim, is_causal):
    """Test BatchPrefillWithRaggedKVCacheWrapper with backend='cute-dsl'."""

    import flashinfer

    torch.manual_seed(42)
    H_q, H_k = 8, 8
    D = head_dim

    # Variable-length sequences
    seq_lens_q = [64, 128, 32]
    seq_lens_k = [64, 128, 32]
    batch_size = len(seq_lens_q)

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_k)

    q = torch.randn(total_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")

    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_q).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_k).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    # cute-dsl backend
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        backend="cute-dsl",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=H_q,
        num_kv_heads=H_k,
        head_dim_qk=D,
        causal=is_causal,
        q_data_type=dtype,
    )
    o_dsl = wrapper.run(q, k, v)
    torch.cuda.synchronize()

    # Reference: per-sequence attention
    o_ref = torch.zeros_like(o_dsl)
    for i in range(batch_size):
        q_i = q[qo_indptr[i] : qo_indptr[i + 1]].unsqueeze(0)
        k_i = k[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        v_i = v[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        o_i = (
            reference_attention(q_i, k_i, v_i, is_causal=is_causal).squeeze(0).to(dtype)
        )
        o_ref[qo_indptr[i] : qo_indptr[i + 1]] = o_i

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o_dsl, o_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_batch_ragged_prefill_cute_dsl_fp8(enable_tvm_ffi, head_dim, is_causal):
    """Test BatchPrefillWithRaggedKVCacheWrapper with FP8 input and FP16 output."""

    from flashinfer.attention_dsl.cute_dsl.fmha import cute_dsl_fmha_ragged_prefill

    torch.manual_seed(42)
    H_q, H_k = 8, 8
    D = head_dim

    seq_lens_q = [64, 128, 32]
    seq_lens_k = [64, 128, 32]
    batch_size = len(seq_lens_q)

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_k)
    max_s_q = max(seq_lens_q)
    max_s_k = max(seq_lens_k)

    # Allocate with padding (max_s_q / max_s_k extra) to match DSL example convention.
    # TMA descriptors may read beyond the logical tensor boundary; the padding
    # prevents illegal-memory-access errors.
    q_f32_padded = (
        torch.randn(total_q + max_s_q, H_q, D, dtype=torch.float32, device="cuda") * 0.1
    )
    k_f32_padded = (
        torch.randn(total_kv + max_s_k, H_k, D, dtype=torch.float32, device="cuda")
        * 0.1
    )
    v_f32_padded = (
        torch.randn(total_kv + max_s_k, H_k, D, dtype=torch.float32, device="cuda")
        * 0.1
    )

    # Quantize to FP8 on padded tensors, then slice (keeps padding memory accessible)
    q_fp8 = _quantize_to_fp8(q_f32_padded, FP8_SCALE_Q)[:total_q]
    k_fp8 = _quantize_to_fp8(k_f32_padded, FP8_SCALE_K)[:total_kv]
    v_fp8 = _quantize_to_fp8(v_f32_padded, FP8_SCALE_V)[:total_kv]

    # Output in FP16 (also padded)
    o_padded = torch.zeros(
        total_q + max_s_q, H_q, D, dtype=torch.float16, device="cuda"
    )
    o = o_padded[:total_q]

    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_q).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_k).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    cute_dsl_fmha_ragged_prefill(
        q_fp8,
        k_fp8,
        v_fp8,
        o,
        qo_indptr,
        kv_indptr,
        is_causal=is_causal,
        scale_q=FP8_SCALE_Q,
        scale_k=FP8_SCALE_K,
        scale_v=FP8_SCALE_V,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    torch.cuda.synchronize()

    # Reference: dequantize then compute per-sequence attention
    q_deq = q_fp8.float() * FP8_SCALE_Q
    k_deq = k_fp8.float() * FP8_SCALE_K
    v_deq = v_fp8.float() * FP8_SCALE_V
    o_ref = torch.zeros_like(o)
    for i in range(batch_size):
        q_i = q_deq[qo_indptr[i] : qo_indptr[i + 1]].unsqueeze(0)
        k_i = k_deq[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        v_i = v_deq[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        o_i = (
            reference_attention(q_i, k_i, v_i, is_causal=is_causal)
            .squeeze(0)
            .to(torch.float16)
        )
        o_ref[qo_indptr[i] : qo_indptr[i + 1]] = o_i

    torch.testing.assert_close(o, o_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_batch_ragged_prefill_cute_dsl_asymmetric_seqlens(dtype, head_dim, is_causal):
    """Test BatchPrefillWithRaggedKVCacheWrapper with seq_lens_q != seq_lens_k."""

    import flashinfer

    torch.manual_seed(42)
    H_q, H_k = 8, 8
    D = head_dim

    # Asymmetric: Q sequences shorter than K (e.g., append/prefill with longer context)
    seq_lens_q = [32, 64, 16]
    seq_lens_k = [128, 256, 64]
    batch_size = len(seq_lens_q)

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_k)

    q = torch.randn(total_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")

    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_q).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_k).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        backend="cute-dsl",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=H_q,
        num_kv_heads=H_k,
        head_dim_qk=D,
        causal=is_causal,
        q_data_type=dtype,
    )
    o_dsl = wrapper.run(q, k, v)
    torch.cuda.synchronize()

    # Reference: per-sequence attention
    o_ref = torch.zeros_like(o_dsl)
    for i in range(batch_size):
        q_i = q[qo_indptr[i] : qo_indptr[i + 1]].unsqueeze(0)
        k_i = k[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        v_i = v[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        o_i = (
            reference_attention(q_i, k_i, v_i, is_causal=is_causal).squeeze(0).to(dtype)
        )
        o_ref[qo_indptr[i] : qo_indptr[i + 1]] = o_i

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o_dsl, o_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("is_causal", [False, True])
def test_batch_ragged_prefill_cute_dsl_vs_flashinfer(dtype, head_dim, is_causal):
    """Compare BatchPrefillWithRaggedKVCacheWrapper cute-dsl vs auto backend."""

    import flashinfer

    torch.manual_seed(42)
    H_q, H_k = 8, 8
    D = head_dim

    seq_lens_q = [64, 128, 32]
    seq_lens_k = [64, 128, 32]

    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_k)

    q = torch.randn(total_q, H_q, D, dtype=dtype, device="cuda")
    k = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")
    v = torch.randn(total_kv, H_k, D, dtype=dtype, device="cuda")

    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_q).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(seq_lens_k).cumsum(0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    # cute-dsl
    wrapper_dsl = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        backend="cute-dsl",
    )
    wrapper_dsl.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=H_q,
        num_kv_heads=H_k,
        head_dim_qk=D,
        causal=is_causal,
        q_data_type=dtype,
    )
    o_dsl = wrapper_dsl.run(q, k, v)
    torch.cuda.synchronize()

    # auto (fa3 on SM100)
    wrapper_auto = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        "NHD",
        backend="auto",
    )
    wrapper_auto.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads=H_q,
        num_kv_heads=H_k,
        head_dim_qk=D,
        causal=is_causal,
        q_data_type=dtype,
    )
    o_auto = wrapper_auto.run(q, k, v)

    rtol = 1e-2 if dtype == torch.float16 else 2e-2
    atol = 1e-2 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(o_dsl, o_auto, rtol=rtol, atol=atol)
