import pytest
import torch
import math

import flashinfer
from flashinfer.prefill import fmha_v2_prefill_deepseek
from tests.utils_fp8 import to_float8
from flashinfer.utils import is_sm120a_supported


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    # tensors are (batch_size, seq_len, num_heads, head_dim)
    qo_len = q.shape[1]
    kv_len = k.shape[1]
    logits = torch.einsum("bmhd,bnhd->bhmn", q.float(), k.float()) * sm_scale

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    # LSE computation: logsumexp over the key dimension (last dim)
    # logits shape: (batch, num_heads, seq_len, seq_len)
    lse_ref = torch.logsumexp(logits, -1)  # (batch, num_heads, seq_len)
    # Transpose to match expected shape (batch, seq_len, num_heads)
    lse_ref = lse_ref.transpose(1, 2)
    p = torch.softmax(logits, dim=-1)
    o_ref = torch.einsum("bhmn,bnhd->bmhd", p, v.float()).contiguous()

    # Return LSE in natural log (no conversion needed)
    return o_ref, lse_ref


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_dim_qk", [192])
@pytest.mark.parametrize("head_dim_v", [128])
@pytest.mark.parametrize("seq_len", [1024, 4096, 8192])
@pytest.mark.parametrize(
    "qkv_dtype,o_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
    ],
)
def test_fmha_v2_prefill_deepseek(
    batch_size, num_heads, head_dim_qk, head_dim_v, seq_len, qkv_dtype, o_dtype
):
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("fmha_v2_prefill_deepseek is only supported on SM120 GPUs.")
    torch.manual_seed(42)

    def initialize_tensors(batch_size, num_heads, head_dim_qk, head_dim_v, seq_len):
        device = "cuda"
        if qkv_dtype == torch.float8_e4m3fn:
            q = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=torch.bfloat16,
                device=device,
            )
            k = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=torch.bfloat16,
                device=device,
            )
            v = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_v),
                dtype=torch.bfloat16,
                device=device,
            )

            q, q_scale = to_float8(q, dtype=torch.float8_e4m3fn)
            k, k_scale = to_float8(k, dtype=torch.float8_e4m3fn)
            v, v_scale = to_float8(v, dtype=torch.float8_e4m3fn)
            q_scale = q_scale.item()
            k_scale = k_scale.item()
            v_scale = v_scale.item()
        else:
            q = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=qkv_dtype,
                device=device,
            )
            k = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_qk),
                dtype=qkv_dtype,
                device=device,
            )
            v = torch.randn(
                (batch_size, seq_len, num_heads, head_dim_v),
                dtype=qkv_dtype,
                device=device,
            )
            # For non-FP8 case, scales are 1.0
            q_scale = 1.0
            k_scale = 1.0
            v_scale = 1.0

        # Output and statistics
        o = torch.zeros(
            batch_size, seq_len, num_heads, head_dim_v, dtype=o_dtype, device=device
        )
        lse = torch.zeros(
            batch_size, seq_len, num_heads, 2, dtype=torch.float, device=device
        )
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
        return q, k, v, o, lse, sm_scale, q_scale, k_scale, v_scale

    q, k, v, o, lse, sm_scale, q_scale, k_scale, v_scale = initialize_tensors(
        batch_size, num_heads, head_dim_qk, head_dim_v, seq_len
    )
    scale_bmm1 = q_scale * k_scale * sm_scale
    scale_bmm2 = v_scale
    scale_softmax = 1.0 if qkv_dtype == torch.float8_e4m3fn else 0.0
    out, lse = fmha_v2_prefill_deepseek(
        q,
        k,
        v,
        o,
        num_heads,
        head_dim_qk,
        seq_len,
        scale_softmax=scale_softmax,
        scale_bmm1=scale_bmm1,
        scale_bmm2=scale_bmm2,
        return_lse=True,
        lse=lse,
    )

    # implementation gives [max(s_i), sum(exp(s_i - max(s_i)))], compute lse from this
    if qkv_dtype == torch.float8_e4m3fn:
        # For E4M3 the softmax is scaled by 256 (the largest power-of-2 below E4M3_MAX=448.0)
        descale = 256
        lse = lse[:, :, :, 0] + torch.log(lse[:, :, :, 1] / descale)
    else:
        lse = lse[:, :, :, 0] + torch.log(lse[:, :, :, 1])

    if qkv_dtype == torch.float8_e4m3fn:
        q_32 = q.to(torch.float32) * q_scale
        k_32 = k.to(torch.float32) * k_scale
        v_32 = v.to(torch.float32) * v_scale
        out_ref, lse_ref = attention_ref(
            batch_size, q_32, k_32, v_32, causal=True, sm_scale=sm_scale
        )
    else:
        out_ref, lse_ref = attention_ref(
            batch_size, q, k, v, causal=True, sm_scale=sm_scale
        )
        out_ref = out_ref.to(o.dtype)

    if q.dtype == torch.float8_e4m3fn and o.dtype == torch.bfloat16:
        rtol, atol = 4e-2, 6e-2
        torch.testing.assert_close(out, out_ref.to(o.dtype), rtol=rtol, atol=atol)
    elif q.dtype == torch.bfloat16 and o.dtype == torch.bfloat16:
        rtol, atol = 1e-2, 1e-2
        torch.testing.assert_close(out, out_ref, rtol=rtol, atol=atol)
    else:
        rtol, atol = 1e-2, 1e-3

    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [8])  # Paged KV cache
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_fmha_v2_prefill_paged(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    causal,
):
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.utils import is_sm90a_supported

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")

    torch.manual_seed(42)
    device = torch.device("cuda")

    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    max_kv_len = seq_lens.max().item()
    max_num_blocks = (max_kv_len + page_size - 1) // page_size

    num_pages = batch_size * max_num_blocks
    paged_kv_cache = torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=device
    )
    for i in range(batch_size):
        num_blocks_needed = (seq_lens[i].item() + page_size - 1) // page_size
        block_tables[i, :num_blocks_needed] = torch.arange(
            i * max_num_blocks, i * max_num_blocks + num_blocks_needed, device=device
        )

    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens_q[1:] = torch.cumsum(seq_lens, dim=0)
    cum_seq_lens_kv = cum_seq_lens_q.clone()

    total_q_tokens = cum_seq_lens_q[-1].item()

    q = torch.randn(total_q_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    o = torch.zeros(total_q_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim)
    max_q_len = seq_lens.max().item()
    mask_mode_str = "causal" if causal else "padding"

    output = trtllm_fmha_v2_prefill(
        (q, paged_kv_cache),  # (Q, paged_KV) tuple
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        block_tables=block_tables,
        out=o,
        out_dtype=dtype,
        kv_layout="NHD",
        input_layout="Q_PAGED_KV",
        mask_mode=mask_mode_str,
        window_left=-1,
    )

    # cumulative page counts per sequence
    page_per_seq = (seq_lens + page_size - 1) // page_size
    paged_kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(page_per_seq, dim=0, dtype=torch.int32),
        ]
    )

    # Flatten block_tables to get paged_kv_indices
    paged_kv_indices = []
    for i in range(batch_size):
        num_pages_needed = page_per_seq[i].item()
        paged_kv_indices.append(block_tables[i, :num_pages_needed])
    paged_kv_indices = torch.cat(paged_kv_indices)

    kv_last_page_len = seq_lens % page_size
    kv_last_page_len[kv_last_page_len == 0] = page_size

    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    plan_params = {
        "qo_indptr": cum_seq_lens_q,
        "paged_kv_indptr": paged_kv_indptr,
        "paged_kv_indices": paged_kv_indices,
        "paged_kv_last_page_len": kv_last_page_len,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim,
        "page_size": page_size,
        "causal": causal,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": q.dtype,
        "kv_data_type": paged_kv_cache.dtype,
        "window_left": -1,
    }

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "NHD"
    )
    wrapper_ref.plan(**plan_params)
    output_ref = wrapper_ref.run(q, paged_kv_cache)
    rtol, atol = 1e-2, 5e-3
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_fmha_v2_prefill_packed_qkv(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    causal,
):
    """
    Packed-QKV: in the kernel, QKV are packed into [total_tokens, (H + H_kv + H_kv), D] layout.
    For standard MHA with H == H_kv, this is [total_tokens, 3*H, D].
    The memory layout per token is: [Q_data (H*D), K_data (H_kv*D), V_data (H_kv*D)].

    For this layout, Q and KV must have the same sequence length.
    """
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.utils import is_sm90a_supported

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")

    torch.manual_seed(42)
    device = torch.device("cuda")

    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    max_kv_len = seq_lens.max().item()
    max_num_blocks = (max_kv_len + page_size - 1) // page_size
    cum_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    total_tokens = cum_seq_lens[-1].item()

    assert num_qo_heads == num_kv_heads, (
        "Packed QKV test assumes MHA (num_qo_heads == num_kv_heads)"
    )
    packed_qkv = torch.randn(
        total_tokens, 3, num_qo_heads, head_dim, dtype=dtype, device=device
    )
    q_ref = packed_qkv[:, 0, :, :].contiguous()  # [total_tokens, H, D]
    k_ref = packed_qkv[:, 1, :, :].contiguous()  # [total_tokens, H, D]
    v_ref = packed_qkv[:, 2, :, :].contiguous()  # [total_tokens, H, D]
    o = torch.zeros(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim)
    max_q_len = max_kv_len
    mask_mode_str = "causal" if causal else "padding"
    output = trtllm_fmha_v2_prefill(
        packed_qkv,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        out=o,
        out_dtype=dtype,
        kv_layout="NHD",
        input_layout="PACKED_QKV",
        mask_mode=mask_mode_str,
        window_left=-1,
    )

    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    # cumulative token counts per sequence
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )

    wrapper_ref = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref, "NHD"
    )
    wrapper_ref.plan(
        qo_indptr=cum_seq_lens,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    output_ref = wrapper_ref.run(q_ref, k_ref, v_ref)
    rtol, atol = 1e-2, 5e-3
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("max_seq_len", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [8])  # Test both MHA and GQA
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_fmha_v2_prefill_separate_qkv(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    causal,
):
    """
    Separate Q, K, V: Q, K, V are provided as separate tensors in a tuple.
    Q shape: [total_tokens, num_qo_heads, head_dim]
    K shape: [total_tokens, num_kv_heads, head_dim]
    V shape: [total_tokens, num_kv_heads, head_dim]

    This layout supports both MHA (num_qo_heads == num_kv_heads) and
    GQA (num_qo_heads > num_kv_heads).
    """
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.utils import is_sm90a_supported

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")

    torch.manual_seed(42)
    device = torch.device("cuda")

    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    max_kv_len = seq_lens.max().item()
    cum_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    total_tokens = cum_seq_lens[-1].item()

    # Create separate Q, K, V tensors
    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    o = torch.zeros(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    sm_scale = 1.0 / math.sqrt(head_dim)
    max_q_len = max_kv_len
    mask_mode_str = "causal" if causal else "padding"

    # Pass (Q, K, V) tuple with input_layout="SEPARATE_Q_K_V"
    output = trtllm_fmha_v2_prefill(
        (q, k, v),  # Tuple of separate tensors
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        out=o,
        out_dtype=dtype,
        kv_layout="NHD",
        input_layout="SEPARATE_Q_K_V",
        mask_mode=mask_mode_str,
        window_left=-1,
    )
    print(output)
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    # cumulative token counts per sequence
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )

    wrapper_ref = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer_ref, "NHD"
    )
    wrapper_ref.plan(
        qo_indptr=cum_seq_lens,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    output_ref = wrapper_ref.run(q, k, v)
    rtol, atol = 1e-2, 5e-3
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)
