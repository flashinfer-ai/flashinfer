# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""ResInfer paged NVFP4 attention with grouped Q residue and BF16 output.

Run: python examples/resinfer/resattn.py
Requires SM100 and CUTLASS DSL 4.8 with its CUDA 13.4 PTX compiler.
This example constructs a synthetic prequantized KV cache, runs split1/split2,
and checks uniform attention against the mean of the decoded values.
"""

import torch

from flashinfer.resinfer import paged_decode_residue


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise SystemExit("ResAttn requires an SM100 GPU")
    torch.manual_seed(42)
    batch, kv_heads, group, dim, pages_per_seq = 2, 2, 4, 128, 5
    pages = batch * pages_per_seq
    q = torch.randn(batch, kv_heads * group, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randint(
        0, 256, (pages, kv_heads, 128, dim // 2), device="cuda", dtype=torch.uint8
    )
    # Both E2M1 nibbles in 0x22 represent 1, so decoded V equals its block scale.
    v = torch.full_like(k, 0x22)
    scale_shape = (pages, kv_heads, 128, dim // 16)
    k_scale = (torch.rand(scale_shape, device="cuda") * 0.1 + 0.025).to(
        torch.float8_e4m3fn
    )
    v_scale_linear = (torch.rand(scale_shape, device="cuda") * 0.2 + 0.05).to(
        torch.float8_e4m3fn
    )
    # Prepare this layout when writing the cache, outside the decode loop.
    v_scale = (
        v_scale_linear.reshape(pages, kv_heads, 32, 4, dim // 16)
        .transpose(-1, -2)
        .contiguous()
    )
    block_tables = (
        torch.randperm(pages, device="cuda").int().reshape(batch, pages_per_seq)
    )
    lengths = [601, 519]
    seq_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    value_scale = 1.25
    args = (q, k, v, k_scale, v_scale, block_tables, seq_lens)

    for splits in (1, 2):
        out = paged_decode_residue(*args, value_scale=value_scale, num_splits=splits)
        assert out.shape == q.shape and out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    # Zero Q gives uniform attention. These V values are exactly representable
    # in E4M3, so the reference also matches QMUL4's V conversion.
    q.zero_()
    expected = []
    for i, length in enumerate(lengths):
        values = v_scale_linear.float()[block_tables[i].long()]
        values = values.permute(1, 0, 2, 3).flatten(1, 2)[:, :length]
        mean = values.mean(1).repeat_interleave(16, dim=-1)
        expected.append(mean.repeat_interleave(group, dim=0) * value_scale)
    expected = torch.stack(expected).to(torch.bfloat16)
    for splits in (1, 2):
        out = paged_decode_residue(*args, value_scale=value_scale, num_splits=splits)
        torch.testing.assert_close(out, expected, rtol=0.004, atol=1e-5)
        print(
            f"ResAttn split{splits}: {tuple(out.shape)}, {out.dtype}; reference passed"
        )


if __name__ == "__main__":
    main()
