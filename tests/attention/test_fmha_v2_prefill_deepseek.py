import pytest
import torch
import math


from flashinfer.prefill import fmha_v2_prefill_deepseek
from tests.utils_fp8 import to_float8
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported


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
    if not (
        is_sm120a_supported(torch.device("cuda"))
        or is_sm121a_supported(torch.device("cuda"))
    ):
        pytest.skip("fmha_v2_prefill_deepseek is only supported on SM12x GPUs.")
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
