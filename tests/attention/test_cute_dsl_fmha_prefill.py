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


def _make_qkvo(B, S_q, S_k, H_q, H_k, D, dtype, out_dtype=None):
    """Create Q, K, V, O tensors and FP8 reference data if needed.

    Returns (q, k, v, o, q_ref, k_ref, v_ref, scale_q, scale_k, scale_v, out_dtype)
    where q_ref/k_ref/v_ref are dequantized float32 tensors for FP8, or None for others.
    """
    if out_dtype is None:
        out_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    D_v = D if D != 192 else 128
    is_fp8 = dtype == torch.float8_e4m3fn

    if is_fp8:
        q_f32 = torch.randn(B, S_q, H_q, D, dtype=torch.float32, device="cuda") * 0.1
        k_f32 = torch.randn(B, S_k, H_k, D, dtype=torch.float32, device="cuda") * 0.1
        v_f32 = torch.randn(B, S_k, H_k, D_v, dtype=torch.float32, device="cuda") * 0.1
        q = (q_f32 / FP8_SCALE_Q).to(torch.float8_e4m3fn)
        k = (k_f32 / FP8_SCALE_K).to(torch.float8_e4m3fn)
        v = (v_f32 / FP8_SCALE_V).to(torch.float8_e4m3fn)
        o = torch.zeros(B, S_q, H_q, D_v, dtype=out_dtype, device="cuda")
        # Dequantized reference
        q_ref = q.float() * FP8_SCALE_Q
        k_ref = k.float() * FP8_SCALE_K
        v_ref = v.float() * FP8_SCALE_V
        return (
            q,
            k,
            v,
            o,
            q_ref,
            k_ref,
            v_ref,
            FP8_SCALE_Q,
            FP8_SCALE_K,
            FP8_SCALE_V,
            out_dtype,
        )
    else:
        q = torch.randn(B, S_q, H_q, D, dtype=dtype, device="cuda")
        k = torch.randn(B, S_k, H_k, D, dtype=dtype, device="cuda")
        v = torch.randn(B, S_k, H_k, D_v, dtype=dtype, device="cuda")
        o = torch.zeros(B, S_q, H_q, D_v, dtype=dtype, device="cuda")
        return q, k, v, o, None, None, None, 1.0, 1.0, 1.0, dtype


def _get_tolerances(dtype):
    """Return (rtol, atol) for the given dtype."""
    if dtype == torch.float8_e4m3fn:
        return 2e-2, 2e-2
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 2e-2, 2e-2


# =============================================================================
# Test: cute_dsl_fmha_prefill (direct API)
# =============================================================================


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
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
        (4, 1024, 16384, 8, 8),  # batched long KV
    ],
)
def test_cute_dsl_fmha_prefill_direct(
    enable_tvm_ffi, dtype, head_dim, is_causal, B, S_q, S_k, H_q, H_k
):
    """Test cute_dsl_fmha_prefill directly with torch tensors."""
    from flashinfer.attention_dsl.cute_dsl.fmha import cute_dsl_fmha_prefill

    torch.manual_seed(42)
    D = head_dim
    q, k, v, o, q_ref, k_ref, v_ref, sq, sk, sv, out_dtype = _make_qkvo(
        B, S_q, S_k, H_q, H_k, D, dtype
    )

    cute_dsl_fmha_prefill(
        q,
        k,
        v,
        o,
        is_causal=is_causal,
        scale_q=sq,
        scale_k=sk,
        scale_v=sv,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    torch.cuda.synchronize()

    # Reference attention (use dequantized tensors for FP8)
    rq, rk, rv = (q_ref, k_ref, v_ref) if q_ref is not None else (q, k, v)
    o_ref = reference_attention(rq, rk, rv, is_causal=is_causal).to(out_dtype)

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)


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

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)


# =============================================================================
# Test: kernel loading
# =============================================================================


def test_kernel_loading():
    """Test that get_cute_dsl_fmha_kernel can load a .so file."""
    from flashinfer.attention_dsl.cute_dsl.fmha import get_cute_dsl_fmha_kernel

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

    o_dsl = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=is_causal,
        backend="cute-dsl",
    )
    torch.cuda.synchronize()

    o_fa3 = flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=is_causal,
        backend="auto",
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(o_dsl, o_fa3, rtol=rtol, atol=atol)


# =============================================================================
# Helper: build ragged (variable-length) tensors
# =============================================================================


def _make_ragged_qkvo(seq_lens_q, seq_lens_k, H_q, H_k, D, dtype, out_dtype=None):
    """Create ragged Q, K, V, O tensors with TMA-safe padding.

    Returns (q, k, v, o, qo_indptr, kv_indptr,
             q_ref, k_ref, v_ref, scale_q, scale_k, scale_v, out_dtype, batch_size)
    """
    if out_dtype is None:
        out_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
    D_v = D if D != 192 else 128
    is_fp8 = dtype == torch.float8_e4m3fn
    batch_size = len(seq_lens_q)
    total_q = sum(seq_lens_q)
    total_kv = sum(seq_lens_k)
    max_s_q = max(seq_lens_q)
    max_s_k = max(seq_lens_k)

    # Front-padding to match DSL example's create_and_pad_tensor convention.
    # The varlen kernel applies a negative offset (q_offset = -max_s_q * H * D)
    # to the pointer, so we must have valid GPU memory before the data start.
    # Allocate (max_s + total) and place data at [max_s:].
    if is_fp8:
        q_f32_full = (
            torch.randn(max_s_q + total_q, H_q, D, dtype=torch.float32, device="cuda")
            * 0.1
        )
        k_f32_full = (
            torch.randn(max_s_k + total_kv, H_k, D, dtype=torch.float32, device="cuda")
            * 0.1
        )
        v_f32_full = (
            torch.randn(
                max_s_k + total_kv, H_k, D_v, dtype=torch.float32, device="cuda"
            )
            * 0.1
        )
        q = (q_f32_full / FP8_SCALE_Q).to(torch.float8_e4m3fn)[max_s_q:]
        k = (k_f32_full / FP8_SCALE_K).to(torch.float8_e4m3fn)[max_s_k:]
        v = (v_f32_full / FP8_SCALE_V).to(torch.float8_e4m3fn)[max_s_k:]
        o_full = torch.zeros(
            max_s_q + total_q, H_q, D_v, dtype=out_dtype, device="cuda"
        )
        o = o_full[max_s_q:]
        q_ref = q.float() * FP8_SCALE_Q
        k_ref = k.float() * FP8_SCALE_K
        v_ref = v.float() * FP8_SCALE_V
        sq, sk, sv = FP8_SCALE_Q, FP8_SCALE_K, FP8_SCALE_V
    else:
        q_full = torch.randn(max_s_q + total_q, H_q, D, dtype=dtype, device="cuda")
        k_full = torch.randn(max_s_k + total_kv, H_k, D, dtype=dtype, device="cuda")
        v_full = torch.randn(max_s_k + total_kv, H_k, D_v, dtype=dtype, device="cuda")
        o_full = torch.zeros(max_s_q + total_q, H_q, D_v, dtype=dtype, device="cuda")
        q = q_full[max_s_q:]
        k = k_full[max_s_k:]
        v = v_full[max_s_k:]
        o = o_full[max_s_q:]
        q_ref, k_ref, v_ref = None, None, None
        sq, sk, sv = 1.0, 1.0, 1.0

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

    return (
        q,
        k,
        v,
        o,
        qo_indptr,
        kv_indptr,
        q_ref,
        k_ref,
        v_ref,
        sq,
        sk,
        sv,
        out_dtype,
        batch_size,
    )


def _ragged_reference(
    q_ref, k_ref, v_ref, qo_indptr, kv_indptr, batch_size, is_causal, out_dtype
):
    """Compute per-sequence reference attention for ragged tensors."""
    o_ref = torch.zeros(
        q_ref.shape[0],
        q_ref.shape[1],
        v_ref.shape[-1],
        dtype=out_dtype,
        device=q_ref.device,
    )
    for i in range(batch_size):
        q_i = q_ref[qo_indptr[i] : qo_indptr[i + 1]].unsqueeze(0)
        k_i = k_ref[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        v_i = v_ref[kv_indptr[i] : kv_indptr[i + 1]].unsqueeze(0)
        o_i = (
            reference_attention(q_i, k_i, v_i, is_causal=is_causal)
            .squeeze(0)
            .to(out_dtype)
        )
        o_ref[qo_indptr[i] : qo_indptr[i + 1]] = o_i
    return o_ref


# =============================================================================
# Test: BatchPrefillWithRaggedKVCacheWrapper (direct API)
# =============================================================================


@pytest.mark.parametrize("enable_tvm_ffi", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "seq_lens_q, seq_lens_k, H_q, H_k",
    [
        ([64, 128, 32], [64, 128, 32], 8, 8),  # basic MHA, symmetric
        ([64, 128, 32], [64, 128, 32], 16, 4),  # GQA
        ([4096, 8192], [4096, 8192], 8, 8),  # long context
        ([32, 64, 16], [128, 256, 64], 8, 8),  # asymmetric (S_q < S_k)
        ([512, 1024], [8192, 16384], 8, 8),  # long KV cache
    ],
)
def test_batch_ragged_prefill_cute_dsl(
    enable_tvm_ffi, dtype, head_dim, is_causal, seq_lens_q, seq_lens_k, H_q, H_k
):
    """Test cute_dsl_fmha_ragged_prefill with variable-length sequences."""
    from flashinfer.attention_dsl.cute_dsl.fmha import cute_dsl_fmha_ragged_prefill

    torch.manual_seed(42)
    D = head_dim

    (
        q,
        k,
        v,
        o,
        qo_indptr,
        kv_indptr,
        q_ref,
        k_ref,
        v_ref,
        sq,
        sk,
        sv,
        out_dtype,
        batch_size,
    ) = _make_ragged_qkvo(seq_lens_q, seq_lens_k, H_q, H_k, D, dtype)

    cute_dsl_fmha_ragged_prefill(
        q,
        k,
        v,
        o,
        qo_indptr,
        kv_indptr,
        is_causal=is_causal,
        scale_q=sq,
        scale_k=sk,
        scale_v=sv,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    torch.cuda.synchronize()

    rq, rk, rv = (q_ref, k_ref, v_ref) if q_ref is not None else (q, k, v)
    o_ref = _ragged_reference(
        rq, rk, rv, qo_indptr, kv_indptr, batch_size, is_causal, out_dtype
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(o, o_ref, rtol=rtol, atol=atol)


# =============================================================================
# Test: BatchPrefillWithRaggedKVCacheWrapper cute-dsl vs auto backend
# =============================================================================


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

    q, k, v, _, qo_indptr, kv_indptr, _, _, _, _, _, _, _, _ = _make_ragged_qkvo(
        seq_lens_q, seq_lens_k, H_q, H_k, D, dtype
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

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(o_dsl, o_auto, rtol=rtol, atol=atol)
