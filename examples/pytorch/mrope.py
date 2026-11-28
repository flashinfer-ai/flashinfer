# SPDX-License-Identifier: Apache-2.0
"""Cached MRoPE on packed QKV views; run from a FlashInfer development checkout."""

import torch

from flashinfer import apply_mrope_with_cos_sin_cache_inplace


def main():
    torch.manual_seed(0)
    # Two video frames with a 3x4 patch grid, using distinct THW positions.
    axes = torch.meshgrid(
        torch.arange(2, device="cuda"),
        torch.arange(3, device="cuda"),
        torch.arange(4, device="cuda"),
        indexing="ij",
    )
    positions = torch.stack([axis.reshape(-1) for axis in axes])
    tokens = positions.shape[1]
    head_size, q_heads, kv_heads = 128, 16, 8
    qkv = torch.randn(
        tokens,
        (q_heads + 2 * kv_heads) * head_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    query = qkv[:, : q_heads * head_size]
    key = qkv[:, q_heads * head_size : (q_heads + kv_heads) * head_size]
    value = qkv[:, (q_heads + kv_heads) * head_size :]
    value_before = value.clone()
    q_before, k_before = query.clone(), key.clone()
    # Illustrative frequencies; real integrations use the model's cache.
    frequencies = 10000.0 ** (
        -torch.arange(0, head_size, 2, device="cuda").float() / head_size
    )
    angles = torch.arange(256, device="cuda")[:, None] * frequencies
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    sections = (24, 20, 20)

    apply_mrope_with_cos_sin_cache_inplace(
        positions, query, key, head_size, cache, sections, mrope_interleaved=True
    )

    # Independent cache selection and rotation, in FP64 before final BF16
    # conversion. Allow BF16 rounding differences from the FP32-cache kernel.
    pairs = torch.arange(head_size // 2, device="cuda")
    axis = torch.zeros_like(pairs)
    axis[(pairs % 3 == 1) & (pairs < 3 * sections[1])] = 1
    axis[(pairs % 3 == 2) & (pairs < 3 * sections[2])] = 2
    selected_positions = positions[axis].T
    cos = cache[selected_positions, pairs].double()[:, None, :]
    sin = cache[selected_positions, pairs + head_size // 2].double()[:, None, :]
    for before, actual, heads in (
        (q_before, query, q_heads),
        (k_before, key, kv_heads),
    ):
        first, second = before.view(tokens, heads, head_size).double().chunk(2, dim=-1)
        expected = torch.cat(
            (first * cos - second * sin, first * sin + second * cos), dim=-1
        )
        torch.testing.assert_close(
            actual, expected.reshape_as(actual).to(actual.dtype), rtol=1e-2, atol=1e-2
        )
    assert torch.equal(value, value_before)
    print("MRoPE outputs match; packed V is unchanged.")


if __name__ == "__main__":
    main()
