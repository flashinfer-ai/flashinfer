from typing import Tuple

import torch

import flashinfer


def per_head_symmetric_quant(
    x: torch.Tensor, quant_dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [seq_len, num_heads, head_dim]
    assert quant_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    def get_dtype_minmax(dtype: torch.dtype) -> Tuple[float, float]:
        if dtype == torch.float8_e4m3fn:
            return -448.0, 448.0
        elif dtype == torch.float8_e5m2:
            return -57344, 57344
        else:
            raise ValueError(f"Unsupported quantization dtype: {dtype}")

    o_min_val, o_max_val = get_dtype_minmax(quant_dtype)
    x_max_val = x.abs().amax(dim=(0, 2)).to(dtype=torch.float32)

    s_out = torch.clamp(x_max_val / o_max_val, min=1e-6)
    s_out_broadcast = s_out.view(1, -1, 1)

    q_x_out = torch.clamp(
        x / s_out_broadcast,
        min=o_min_val,
        max=o_max_val,
    ).to(dtype=quant_dtype)

    assert not torch.any(torch.isnan(q_x_out))
    assert not torch.any(torch.isnan(s_out))

    return q_x_out, s_out


def test_single_prefill(seq_len, num_heads, causal, head_dim, dtype):
    o_dtype = torch.half
    num_qo_heads = num_kv_heads = num_heads

    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa3"
    )

    q_fp8, s_q = per_head_symmetric_quant(q, quant_dtype=dtype)
    k_fp8, s_k = per_head_symmetric_quant(k, quant_dtype=dtype)
    v_fp8, s_v = per_head_symmetric_quant(v, quant_dtype=dtype)
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q_fp8,
        k_fp8,
        v_fp8,
        s_q,
        s_k,
        s_v,
        causal=causal,
        backend="fa3",
        o_dtype=o_dtype,
    )

    assert not torch.any(torch.isnan(o_fp8))
    assert not torch.any(torch.isnan(o_ref))

    # MSE
    mse = torch.mean((o_ref.float() - o_fp8.float()) ** 2)
    print(
        f"test_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}, dtype={dtype}), MSE: {mse:.5f}"
    )


if __name__ == "__main__":
    for seq_len in [117, 509, 1011, 2372, 7777]:
        for num_heads in [24, 32]:
            for causal in [True, False]:
                for head_dim in [64, 128, 256]:
                    for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                        test_single_prefill(seq_len, num_heads, causal, head_dim, dtype)
