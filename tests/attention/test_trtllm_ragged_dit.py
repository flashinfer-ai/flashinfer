"""
Tests for DiT (Diffusion Transformer) oriented ragged attention kernels.

Covers three variants:
1. Q/K/V all FP8 E4M3 (standard case)
2. Q/K in BF16, V in FP8 E4M3 (DiT: BMM1 in BF16, BMM2 in FP8)
3. Q/K in INT8, V in FP8 E4M3, with SageAttention scaling factors

All tests run on SM100/SM103 only (Blackwell).
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


GPU_DEVICE = "cuda:0"
IS_BLACKWELL = get_compute_capability(torch.device(GPU_DEVICE))[0] == 10
WORKSPACE_SIZE = 256 * 1024 * 1024


def _get_workspace():
    return torch.zeros(WORKSPACE_SIZE, dtype=torch.uint8, device=GPU_DEVICE)


def _to_float8(x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn):
    """Quantize float tensor to FP8 and return (quantized, inv_scale)."""
    finfo = torch.finfo(dtype)
    amax = x.abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_sat.to(dtype), scale.float().reciprocal()


def _to_int8_blocked(x: torch.Tensor, block_size: int):
    """
    Quantize float tensor to INT8 with per-block scaling.

    x: [tokens, heads, head_dim]
    block_size: number of elements per block along tokens

    Returns:
        x_q: INT8 tensor same shape as x
        sfs: float32 per-block scales [heads * (tokens // block_size)]
              (inverse scales / dequant scales)
    """
    tokens, heads, head_dim = x.shape
    assert tokens % block_size == 0
    num_blocks = tokens // block_size
    x_blocks = x.reshape(num_blocks, block_size, heads, head_dim)
    amax = x_blocks.abs().amax(dim=-1, keepdim=True).amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = 127.0 / amax  # per-block quantization scale
    x_sat = (x_blocks * scale).round().clamp(-128, 127)
    x_q = x_sat.reshape(tokens, heads, head_dim).to(torch.int8)
    # inv_scale (dequant scale): 1/scale = amax / 127
    inv_scale = (amax / 127.0).reshape(num_blocks, heads).T.flatten().contiguous()
    return x_q, inv_scale


def _ragged_reference_bf16(
    q: torch.Tensor,  # [total_q, heads, head_dim_qk] in float32
    k: torch.Tensor,  # [total_kv, heads, head_dim_qk] in float32
    v: torch.Tensor,  # [total_kv, heads, head_dim_vo] in float32
    q_lens: torch.Tensor,  # [batch]
    kv_lens: torch.Tensor,  # [batch]
    scale: float,
    causal: bool,
):
    """
    Compute ragged (variable-length) attention in float32 for reference.
    Returns output [total_q, heads, head_dim_vo].
    """
    batch = q_lens.shape[0]
    q_cpu = q.float().cpu()
    k_cpu = k.float().cpu()
    v_cpu = v.float().cpu()
    q_lens_cpu = q_lens.cpu().tolist()
    kv_lens_cpu = kv_lens.cpu().tolist()

    out_chunks = []
    q_offset = 0
    kv_offset = 0
    for b in range(batch):
        sq = int(q_lens_cpu[b])
        skv = int(kv_lens_cpu[b])
        q_b = q_cpu[q_offset : q_offset + sq]   # [sq, H, Dqk]
        k_b = k_cpu[kv_offset : kv_offset + skv]  # [skv, H, Dqk]
        v_b = v_cpu[kv_offset : kv_offset + skv]  # [skv, H, Dvo]

        # [H, sq, Dqk] @ [H, Dqk, skv] -> [H, sq, skv]
        attn = (
            q_b.permute(1, 0, 2) @ k_b.permute(1, 2, 0)
        ) * scale

        if causal:
            mask = torch.ones(sq, skv, dtype=torch.bool)
            mask = torch.tril(mask, diagonal=skv - sq)
            attn.masked_fill_(~mask.unsqueeze(0), float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        # [H, sq, skv] @ [H, skv, Dvo] -> [H, sq, Dvo]
        out_b = attn @ v_b.permute(1, 0, 2)
        out_chunks.append(out_b.permute(1, 0, 2))  # [sq, H, Dvo]

        q_offset += sq
        kv_offset += skv

    return torch.cat(out_chunks, dim=0).to(GPU_DEVICE)


# ---------------------------------------------------------------------------
# Test 1: Q/K/V all FP8
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IS_BLACKWELL, reason="DiT attention tests require SM100/SM103 (Blackwell).")
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(512, 512), (256, 512)])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_ragged_dit_qkv_fp8(
    causal: bool,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_heads: int,
    head_dim: int,
):
    torch.manual_seed(42)
    device = GPU_DEVICE

    q_lens = torch.full((batch_size,), s_qo, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch_size,), s_kv, dtype=torch.int32, device=device)
    total_q = int(q_lens.sum())
    total_kv = int(kv_lens.sum())

    q_f = torch.randn(total_q, num_heads, head_dim, device=device)
    k_f = torch.randn(total_kv, num_heads, head_dim, device=device)
    v_f = torch.randn(total_kv, num_heads, head_dim, device=device)

    q_fp8, q_inv_scale = _to_float8(q_f)
    k_fp8, k_inv_scale = _to_float8(k_f)
    v_fp8, v_inv_scale = _to_float8(v_f)

    scale = 1.0 / math.sqrt(head_dim)
    bmm1_scale = scale * q_inv_scale * k_inv_scale
    bmm2_scale = v_inv_scale

    qo_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0).int()]
    )
    kv_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0).int()]
    )

    workspace = _get_workspace()

    out_trtllm, lse = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q_fp8,
        k_fp8,
        v_fp8,
        workspace,
        kv_lens,
        s_qo,
        s_kv,
        float(bmm1_scale),
        float(bmm2_scale),
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
    )

    assert out_trtllm.shape == (total_q, num_heads, head_dim)
    assert out_trtllm.dtype == torch.bfloat16

    # Reference
    out_ref = _ragged_reference_bf16(
        q_f,
        k_f,
        v_f,
        q_lens,
        kv_lens,
        scale,
        causal,
    )
    torch.testing.assert_close(
        out_trtllm.float(),
        out_ref.float(),
        atol=0.05,
        rtol=0.05,
    )


# ---------------------------------------------------------------------------
# Test 2: Q/K in BF16, V in FP8 E4M3 (DiT mixed-dtype)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IS_BLACKWELL, reason="DiT attention tests require SM100/SM103 (Blackwell).")
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(512, 512), (256, 512)])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_ragged_dit_qk_bf16_v_fp8(
    causal: bool,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_heads: int,
    head_dim: int,
):
    torch.manual_seed(42)
    device = GPU_DEVICE

    q_lens = torch.full((batch_size,), s_qo, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch_size,), s_kv, dtype=torch.int32, device=device)
    total_q = int(q_lens.sum())
    total_kv = int(kv_lens.sum())

    q_bf16 = torch.randn(total_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(total_kv, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_f = torch.randn(total_kv, num_heads, head_dim, device=device)
    v_fp8, v_inv_scale = _to_float8(v_f)

    scale = 1.0 / math.sqrt(head_dim)
    bmm1_scale = scale   # Q/K are BF16 (no quantization scale)
    bmm2_scale = v_inv_scale

    qo_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0).int()]
    )
    kv_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0).int()]
    )

    workspace = _get_workspace()

    out_trtllm, lse = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q_bf16,
        k_bf16,
        v_fp8,
        workspace,
        kv_lens,
        s_qo,
        s_kv,
        float(bmm1_scale),
        float(bmm2_scale),
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
    )

    assert out_trtllm.shape == (total_q, num_heads, head_dim)
    assert out_trtllm.dtype == torch.bfloat16

    # Reference
    out_ref = _ragged_reference_bf16(
        q_bf16,
        k_bf16,
        v_f,
        q_lens,
        kv_lens,
        scale,
        causal,
    )
    torch.testing.assert_close(
        out_trtllm.float(),
        out_ref.float(),
        atol=0.05,
        rtol=0.05,
    )


# ---------------------------------------------------------------------------
# Test 3: Q/K in INT8, V in FP8 E4M3, with SageAttention block scaling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IS_BLACKWELL, reason="DiT attention tests require SM100/SM103 (Blackwell).")
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("s_qo,s_kv", [(256, 256), (512, 2048)])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("sage_blk_q", [1])
@pytest.mark.parametrize("sage_blk_k", [4, 16])
def test_trtllm_ragged_dit_sage_qk_int8_v_fp8(
    causal: bool,
    batch_size: int,
    s_qo: int,
    s_kv: int,
    num_heads: int,
    head_dim: int,
    sage_blk_q: int,
    sage_blk_k: int,
):
    torch.manual_seed(42)
    device = GPU_DEVICE

    q_lens = torch.full((batch_size,), s_qo, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch_size,), s_kv, dtype=torch.int32, device=device)
    total_q = int(q_lens.sum())
    total_kv = int(kv_lens.sum())

    q_f = torch.randn(total_q, num_heads, head_dim, device=device)
    k_f = torch.randn(total_kv, num_heads, head_dim, device=device)
    v_f = torch.randn(total_kv, num_heads, head_dim, device=device)

    # Per-block INT8 quantization for Q and K
    q_int8, q_sfs = _to_int8_blocked(q_f.cpu(), sage_blk_q)
    k_int8, k_sfs = _to_int8_blocked(k_f.cpu(), sage_blk_k)
    q_int8 = q_int8.to(device)
    k_int8 = k_int8.to(device)
    q_sfs = q_sfs.to(device)
    k_sfs = k_sfs.to(device)

    sage_blk_v = 1
    v_fp8, v_inv_scale = _to_float8(v_f)
    v_sfs = torch.ones((num_heads * head_dim), device=device)

    # For SageAttention, bmm1_scale encodes 1/sqrt(head_dim) only;
    # per-block Q/K dequant is handled via sage_attn_sfs_q/k
    scale = 1.0 / math.sqrt(head_dim)
    # INT8 range is [-127,127] so per-block inv_scale is amax/127;
    # the effective per-block scale that the kernel uses is
    # sfs_q[i] * sfs_k[j] * (1/sqrt(head_dim)), but bmm1_scale here is 1.0
    # because the kernel multiplies by sfs_q * sfs_k internally.
    bmm1_scale = 1.0 / math.sqrt(head_dim)
    bmm2_scale = float(v_inv_scale)

    qo_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), q_lens.cumsum(0).int()]
    )
    kv_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), kv_lens.cumsum(0).int()]
    )

    workspace = _get_workspace()

    out_trtllm, lse = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        q_int8,
        k_int8,
        v_fp8,
        workspace,
        kv_lens,
        s_qo,
        s_kv,
        bmm1_scale,
        bmm2_scale,
        -1,
        batch_size,
        -1,
        qo_indptr,
        kv_indptr,
        False,
        causal,
        True,
        sage_attn_sfs=(q_sfs, k_sfs, None, v_sfs),
        num_elts_per_sage_attn_blk=(sage_blk_q, sage_blk_k, 0, sage_blk_v),
    )

    assert out_trtllm.shape == (total_q, num_heads, head_dim)
    assert out_trtllm.dtype == torch.bfloat16

    out_ref = _ragged_reference_bf16(
        q_f,
        k_f,
        v_f,
        q_lens,
        kv_lens,
        scale,
        causal,
    )
    torch.testing.assert_close(
        out_trtllm.float(),
        out_ref.float(),
        atol=0.1,
        rtol=0.1,
    )
