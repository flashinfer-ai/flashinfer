import math

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import get_compute_capability


def _init_scaled_uniform(shape, device, dtype):
    base = torch.rand(*shape, device=device)
    exp = torch.exp2(torch.rand(*shape, device=device) + 1.0)
    return (base * exp).to(dtype=dtype)


def _sdpa_ref_bs1(q, k, v, is_causal):
    # SDPA expects (B, H, S, D) and bf16 inputs, while inputs have (S, H, D)
    q_bf16 = q.to(torch.bfloat16).transpose(0, 1).unsqueeze(0)
    k_bf16 = k.to(torch.bfloat16).transpose(0, 1).unsqueeze(0)
    v_bf16 = v.to(torch.bfloat16).transpose(0, 1).unsqueeze(0)
    ref = F.scaled_dot_product_attention(q_bf16, k_bf16, v_bf16, is_causal=is_causal)
    return ref.squeeze(0).transpose(0, 1).contiguous()


def _dequant_sage(q_int8, k_int8, v_fp8, q_sfs, k_sfs, v_sfs, sage_block_k):
    # q_int8, k_int8, v_fp8: (S, H, D)
    # q_sfs: (H, S), k_sfs: (H, S/sage_block_k), v_sfs: (H*D)
    q = q_int8.to(torch.bfloat16) * q_sfs.t().unsqueeze(-1)

    num_heads = k_int8.shape[1]
    seq_len = k_int8.shape[0]
    head_dim = k_int8.shape[2]
    sfs_per_head_k = seq_len // sage_block_k
    k_perm = k_int8.transpose(0, 1)  # (H, S, D)
    k_blocks = k_perm.unflatten(
        1, (sfs_per_head_k, sage_block_k)
    )  # (H, -1, sage_block_k, D)
    k = k_blocks.to(torch.bfloat16) * k_sfs[:, :, None, None]
    k = k.flatten(1, 2).transpose(0, 1).contiguous()  # (S, H, D)

    v_scale = v_sfs.view(num_heads, head_dim).unsqueeze(0)
    v = v_fp8.to(torch.bfloat16) * v_scale
    return q, k, v


def _test_trtllm_ragged_dit_bs1(
    name: str,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    head_dim: int,
    num_heads: int,
    seq_len: int,
    use_sage: bool,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("trtllm-gen backend requires SM100/SM103.")
    if compute_capability[1] != 0 and use_sage:
        pytest.skip("trtllm-gen backend doesn't support SageAttention for SM103.")

    batch_size = 1
    device = torch.device("cuda")

    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    cum_seq_lens_q = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    cum_seq_lens_kv = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    sum_seq = int(cum_seq_lens_q[-1].item())

    query = _init_scaled_uniform((sum_seq, num_heads, head_dim), device, q_dtype)
    key = _init_scaled_uniform((sum_seq, num_heads, head_dim), device, k_dtype)
    value = _init_scaled_uniform((sum_seq, num_heads, head_dim), device, v_dtype)

    bmm1_scale = 1.0 / math.sqrt(head_dim)
    bmm2_scale = 1.0
    out = torch.empty(sum_seq, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    workspace_buffer = torch.zeros(16 * 1024 * 1024, device=device, dtype=torch.uint8)

    sage_attn_sfs = (None, None, None, None)
    num_elts_per_sage_attn_blk = (0, 0, 0, 0)
    if use_sage:
        sage_block_k = 16
        sfs_per_head_k = sum_seq // sage_block_k
        q_sfs = _init_scaled_uniform((num_heads, sum_seq), device, torch.float32)
        k_sfs = _init_scaled_uniform((num_heads, sfs_per_head_k), device, torch.float32)
        v_sfs = _init_scaled_uniform((num_heads * head_dim,), device, torch.float32)
        sage_attn_sfs = (q_sfs, k_sfs, None, v_sfs)
        num_elts_per_sage_attn_blk = (1, sage_block_k, 0, 1)

    out, lse = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=query,
        key=key,
        value=value,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens,
        max_q_len=seq_len,
        max_kv_len=seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        o_sf_scale=0.0,
        batch_size=batch_size,
        window_left=-1,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        enable_pdl=False,
        is_causal=False,
        return_lse=True,
        attention_sinks=None,
        sage_attn_sfs=sage_attn_sfs,
        num_elts_per_sage_attn_blk=num_elts_per_sage_attn_blk,
        out=out,
    )
    assert lse is not None

    if use_sage:
        q_sfs, k_sfs, _, v_sfs = sage_attn_sfs
        q_ref, k_ref, v_ref = _dequant_sage(
            query, key, value, q_sfs, k_sfs, v_sfs, num_elts_per_sage_attn_blk[1]
        )
        ref = _sdpa_ref_bs1(q_ref, k_ref, v_ref, is_causal=False)
        # SageAttention sfs brings by larger total amplitude
        # atol needs to be relaxed accordingly
        atol, rtol = 2.0, 0.05
    else:
        ref = _sdpa_ref_bs1(query, key, value, is_causal=False)
        if q_dtype == torch.bfloat16:
            atol, rtol = 0.1, 0.05
        else:
            atol, rtol = 0.5, 0.1

    out_fp32 = out.to(torch.float32)
    ref_fp32 = ref.to(torch.float32)
    diff = (out_fp32 - ref_fp32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    assert torch.allclose(out_fp32, ref_fp32, rtol=rtol, atol=atol), (
        f"{name} mismatch: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}, rtol={rtol}, atol={atol}"
    )


@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("num_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_ragged_dit_bs1_qkv_fp8(head_dim: int, num_heads: int, seq_len: int):
    _test_trtllm_ragged_dit_bs1(
        f"qkv_fp8_h{head_dim}",
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        use_sage=False,
    )


@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("num_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_ragged_dit_bs1_qk_bf16_v_fp8(
    head_dim: int, num_heads: int, seq_len: int
):
    _test_trtllm_ragged_dit_bs1(
        f"qk_bf16_v_fp8_h{head_dim}",
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        v_dtype=torch.float8_e4m3fn,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        use_sage=False,
    )


@pytest.mark.parametrize("seq_len", [256, 1024])
@pytest.mark.parametrize("num_heads", [1, 4, 16])
@pytest.mark.parametrize("head_dim", [128])
def test_trtllm_ragged_dit_bs1_sage_qk_int8_v_fp8(
    head_dim: int, num_heads: int, seq_len: int
):
    _test_trtllm_ragged_dit_bs1(
        f"sage_qk_int8_v_fp8_h{head_dim}",
        q_dtype=torch.int8,
        k_dtype=torch.int8,
        v_dtype=torch.float8_e4m3fn,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        use_sage=True,
    )
