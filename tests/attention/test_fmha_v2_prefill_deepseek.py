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


def attention_ref_torch(
    qkv,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    sm_scale: float,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = True,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
    block_tables: torch.Tensor = None,
    cum_seq_lens_kv: torch.Tensor = None,
    return_lse: bool = False,
):
    """
    Pure-torch reference for attention supporting multiple input layouts.

    Layouts (auto-detected from qkv):
      - Packed QKV: single tensor [total_tokens, 3, num_heads, head_dim]
      - Separate Q, K, V: tuple (q, k, v)
      - Contiguous Q + KV: tuple (q, kv) where kv is 4-D
      - Paged Q + KV cache: tuple (q, paged_kv_cache) where paged_kv_cache is 5-D
        (requires block_tables)

    Returns output tensor, or (output, lse) when return_lse=True.
    LSE shape: [total_tokens, num_qo_heads].
    """
    device = seq_lens.device
    batch_size = seq_lens.shape[0]

    if cum_seq_lens_kv is None:
        cum_seq_lens_kv = cum_seq_lens_q

    # --- parse input layout ---
    is_paged = False
    if isinstance(qkv, torch.Tensor):
        # Packed QKV: [total, 3, H, D]
        q_flat = qkv[:, 0, :, :].contiguous()
        k_flat = qkv[:, 1, :, :].contiguous()
        v_flat = qkv[:, 2, :, :].contiguous()
    elif isinstance(qkv, tuple):
        if len(qkv) == 3:
            q_flat, k_flat, v_flat = qkv
        elif len(qkv) == 2:
            q_flat = qkv[0]
            second = qkv[1]
            if second.ndim == 5:
                # Paged: (q, paged_kv_cache[num_pages, 2, page_size, H_kv, D])
                is_paged = True
                paged_kv_cache = second
                page_size = paged_kv_cache.shape[2]
            else:
                # Contiguous: (q, kv[total, 2, H_kv, D])
                k_flat = second[:, 0, :, :].contiguous()
                v_flat = second[:, 1, :, :].contiguous()
        else:
            raise ValueError(f"Unexpected tuple length: {len(qkv)}")
    else:
        raise TypeError(f"Unexpected qkv type: {type(qkv)}")

    num_qo_heads = q_flat.shape[1]
    head_dim = q_flat.shape[2]
    if is_paged:
        num_kv_heads = paged_kv_cache.shape[3]
    else:
        num_kv_heads = k_flat.shape[1]
    heads_per_group = num_qo_heads // num_kv_heads

    q_float = q_flat.float() * q_scale

    outputs = []
    lse_outputs = []
    for b in range(batch_size):
        seq_len = seq_lens[b].item()
        q_start = cum_seq_lens_q[b].item()
        q_end = cum_seq_lens_q[b + 1].item()
        q_len = q_end - q_start
        q_seq = q_float[q_start:q_end]

        if is_paged:
            num_pages_needed = (seq_len + page_size - 1) // page_size
            k_pages = []
            v_pages = []
            for p in range(num_pages_needed):
                page_idx = block_tables[b, p].item()
                k_pages.append(paged_kv_cache[page_idx, 0])
                v_pages.append(paged_kv_cache[page_idx, 1])
            k_seq = torch.cat(k_pages, dim=0)[:seq_len].float() * k_scale
            v_seq = torch.cat(v_pages, dim=0)[:seq_len].float() * v_scale
        else:
            kv_start = cum_seq_lens_kv[b].item()
            kv_end = cum_seq_lens_kv[b + 1].item()
            k_seq = k_flat[kv_start:kv_end].float() * k_scale
            v_seq = v_flat[kv_start:kv_end].float() * v_scale

        o_seq = torch.zeros(
            q_len, num_qo_heads, head_dim, dtype=torch.float32, device=device
        )
        if return_lse:
            lse_seq = torch.zeros(
                q_len, num_qo_heads, dtype=torch.float32, device=device
            )

        for h in range(num_qo_heads):
            kv_h = h // heads_per_group

            q_h = q_seq[:, h, :]
            k_h = k_seq[:, kv_h, :]
            v_h = v_seq[:, kv_h, :]

            scores = torch.matmul(q_h, k_h.t()) * sm_scale

            if logits_soft_cap > 0.0:
                scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)

            if causal:
                q_indices = torch.arange(q_len, device=device).unsqueeze(1)
                kv_indices = torch.arange(seq_len, device=device).unsqueeze(0)
                offset = seq_len - q_len
                causal_mask = (q_indices + offset) >= kv_indices
                scores = scores.masked_fill(~causal_mask, float("-inf"))

            if window_left >= 0:
                q_indices = torch.arange(q_len, device=device).unsqueeze(1)
                kv_indices = torch.arange(seq_len, device=device).unsqueeze(0)
                offset = seq_len - q_len
                window_mask = kv_indices >= (q_indices + offset - window_left)
                scores = scores.masked_fill(~window_mask, float("-inf"))

            if return_lse:
                lse_seq[:, h] = torch.logsumexp(scores, dim=-1)

            attn = torch.softmax(scores, dim=-1)
            o_seq[:, h, :] = torch.matmul(attn, v_h)

        outputs.append(o_seq)
        if return_lse:
            lse_outputs.append(lse_seq)

    out = torch.cat(outputs, dim=0)
    if return_lse:
        return out, torch.cat(lse_outputs, dim=0)
    return out


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


@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("max_seq_len", [1024, 4096])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    ("dtype", "o_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
    ],
)
@pytest.mark.parametrize(
    ("input_layout", "page_size", "save_softmax_stats"),
    [
        ("PACKED_QKV", None, False),
        ("CONTIGUOUS_Q_KV", None, False),
        ("SEPARATE_Q_K_V", None, False),
        ("Q_PAGED_KV", 32, False),
        ("Q_PAGED_KV", 128, False),
        ("Q_PAGED_KV", 32, True),
        ("Q_PAGED_KV", 128, True),
    ],
)
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 512, "SLIDING_WINDOW"),
    ],
)
@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_trtllm_fmha_v2_prefill(
    input_layout,
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    o_dtype,
    causal,
    mask_mode,
    non_blocking,
    window_left,
    logits_soft_cap,
    pos_encoding_mode,
    save_softmax_stats,
):
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.utils import is_sm90a_supported

    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")

    # Skip invalid combinations
    if input_layout == "SEPARATE_Q_K_V" and dtype == torch.float8_e4m3fn:
        pytest.skip("FP8 not supported for SEPARATE_Q_K_V layout")
    if input_layout == "SEPARATE_Q_K_V" and logits_soft_cap > 0:
        pytest.skip("Logits soft capping not supported for SEPARATE_Q_K_V layout")
    # save_softmax_stats only supported for CONTIGUOUS_Q_KV (normal attention)
    if save_softmax_stats and input_layout != "CONTIGUOUS_Q_KV":
        pytest.skip(
            "For normal attention, Only CONTIGUOUS_Q_KV layout supports "
            "save_softmax_stats. For MLA only SEPARATE_Q_K_V layout supports "
            "save_softmax_stats."
        )

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

    sm_scale = 1.0 / math.sqrt(head_dim)
    max_q_len = seq_lens.max().item()
    block_tables = None

    # --- Create inputs and scales per layout ---
    if input_layout == "PACKED_QKV":
        if dtype == torch.float8_e4m3fn:
            packed_bf16 = torch.randn(
                total_tokens,
                3,
                num_qo_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            packed_qkv, qkv_scale = to_float8(packed_bf16, dtype=torch.float8_e4m3fn)
            qkv_scale = qkv_scale.item()
        else:
            packed_qkv = torch.randn(
                total_tokens,
                3,
                num_qo_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            qkv_scale = 1.0
        qkv_arg = packed_qkv
        q_scale, k_scale, v_scale = qkv_scale, qkv_scale, qkv_scale

    elif input_layout == "SEPARATE_Q_K_V":
        q = torch.randn(
            total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device
        )
        k = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v = torch.randn(
            total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        qkv_arg = (q, k, v)
        q_scale, k_scale, v_scale = 1.0, 1.0, 1.0

    elif input_layout == "CONTIGUOUS_Q_KV":
        if dtype == torch.float8_e4m3fn:
            q_bf16 = torch.randn(
                total_tokens,
                num_qo_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            q, q_scale = to_float8(q_bf16, dtype=torch.float8_e4m3fn)
            q_scale = q_scale.item()
            kv_bf16 = torch.randn(
                total_tokens,
                2,
                num_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            kv, kv_scale = to_float8(kv_bf16, dtype=torch.float8_e4m3fn)
            kv_scale = kv_scale.item()
        else:
            q = torch.randn(
                total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device
            )
            kv = torch.randn(
                total_tokens,
                2,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            q_scale, kv_scale = 1.0, 1.0
        qkv_arg = (q, kv)
        k_scale, v_scale = kv_scale, kv_scale

    elif input_layout == "Q_PAGED_KV":
        max_num_blocks = (max_kv_len + page_size - 1) // page_size
        num_pages = batch_size * max_num_blocks
        if dtype == torch.float8_e4m3fn:
            paged_bf16 = torch.randn(
                num_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            paged_kv_cache, kv_scale = to_float8(paged_bf16, dtype=torch.float8_e4m3fn)
            kv_scale = kv_scale.item()
            q_bf16 = torch.randn(
                total_tokens,
                num_qo_heads,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            q, q_scale = to_float8(q_bf16, dtype=torch.float8_e4m3fn)
            q_scale = q_scale.item()
        else:
            paged_kv_cache = torch.randn(
                num_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            q = torch.randn(
                total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device
            )
            q_scale, kv_scale = 1.0, 1.0
        block_tables = torch.zeros(
            batch_size,
            max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        for i in range(batch_size):
            num_blocks_needed = (seq_lens[i].item() + page_size - 1) // page_size
            block_tables[i, :num_blocks_needed] = torch.arange(
                i * max_num_blocks,
                i * max_num_blocks + num_blocks_needed,
                device=device,
            )
        qkv_arg = (q, paged_kv_cache)
        k_scale, v_scale = kv_scale, kv_scale

    # --- Compute BMM scales ---
    if dtype == torch.float8_e4m3fn:
        bmm1_scale = sm_scale * q_scale * k_scale
        bmm2_scale = v_scale
    else:
        bmm1_scale = sm_scale
        bmm2_scale = 1.0

    o = torch.zeros(total_tokens, num_qo_heads, head_dim, dtype=o_dtype, device=device)
    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    # --- Run kernel ---
    result = trtllm_fmha_v2_prefill(
        qkv_arg,
        workspace_buffer=workspace_buffer,
        seq_lens=seq_lens,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens,
        cum_seq_lens_kv=cum_seq_lens,
        block_tables=block_tables,
        out=o,
        out_dtype=o_dtype,
        kv_layout="NHD",
        mask_mode=mask_mode,
        window_left=window_left,
        non_blocking=non_blocking,
        logits_soft_cap_scale=logits_soft_cap if logits_soft_cap > 0 else None,
        pos_encoding_mode=pos_encoding_mode,
        save_softmax_stats=save_softmax_stats,
    )

    if save_softmax_stats:
        output, kernel_lse = result
    else:
        output = result

    # --- Reference ---
    ref_result = attention_ref_torch(
        qkv_arg,
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        block_tables=block_tables,
        return_lse=save_softmax_stats,
    )

    if save_softmax_stats:
        output_ref, lse_ref = ref_result
    else:
        output_ref = ref_result

    if dtype == torch.float8_e4m3fn:
        rtol, atol = 4e-2, 7e-2
    else:
        rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)

    if save_softmax_stats:
        # kernel_lse: [total_tokens, num_qo_heads, 2] -> [max, sum_exp] in ragged format
        # The Softmax_saver_tma stores max / sqrt(head_dim).
        # Non-softcap (exp2f path): max = max(raw_BMM_output), scores = raw * q_scale * k_scale * sm_scale
        #   lse = kernel_max * q_scale * k_scale + ln(sum)  (since sqrt(d) * sm_scale = 1)
        # Softcap (expf path): max = max(softcapped(scores * scale_bmm1)), scores already scaled
        #   lse = kernel_max / sm_scale + ln(sum)  (i.e. kernel_max * sqrt(d))
        kernel_max = kernel_lse[:, :, 0]
        kernel_sum_exp = kernel_lse[:, :, 1]
        if logits_soft_cap > 0:
            lse_kernel = kernel_max / sm_scale + torch.log(kernel_sum_exp)
        else:
            lse_kernel = kernel_max * (q_scale * k_scale) + torch.log(kernel_sum_exp)
        torch.testing.assert_close(lse_kernel, lse_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("max_seq_len", [1024, 4096])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 512, "SLIDING_WINDOW"),
    ],
)
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
def test_trtllm_fmha_v2_prefill_attention_sinks(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    non_blocking,
    causal,
    window_left,
    mask_mode,
    pos_encoding_mode,
):
    """
    Test trtllm_fmha_v2_prefill with attention sinks.
    Compares against BatchAttentionWithAttentionSinkWrapper as reference.
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

    # Create sink tensor with random values to properly test the feature
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    # Test trtllm_fmha_v2_prefill with sinks parameter
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
        sinks=sink,
        mask_mode=mask_mode,
        window_left=window_left,
        non_blocking=non_blocking,
        pos_encoding_mode=pos_encoding_mode,
    )

    # Reference: use BatchAttentionWithAttentionSinkWrapper
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # cumulative token counts per sequence
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ]
    )

    # Create kv_indices for page_size=1 (each token is a page)
    kv_indices = torch.arange(0, total_tokens, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.full(
        (batch_size,), 1, dtype=torch.int32, device=device
    )

    wrapper_ref = flashinfer.BatchAttentionWithAttentionSinkWrapper(
        workspace_buffer_ref,
        kv_layout="NHD",
        backend="fa3",
        q_data_type=dtype,
        kv_data_type=dtype,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        window_left=window_left,
    )
    wrapper_ref.plan(
        cum_seq_lens.cpu(),
        kv_indptr.cpu(),
        kv_indices.cpu(),
        paged_kv_last_page_len.cpu(),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,  # page_size
        causal=causal,
        window_left=window_left,
        q_data_type=dtype,
        kv_data_type=dtype,
        non_blocking=non_blocking,
    )
    output_ref = wrapper_ref.run(q, (k, v), sink, sm_scale)

    rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)
