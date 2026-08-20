# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for fp8_paged_mqa_logits and fp4_paged_mqa_logits.

Reference implementations are adapted from TRT-LLM test scripts.
"""

import pytest
import torch

from flashinfer.utils import is_sm100a_supported

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: FP8
# ──────────────────────────────────────────────────────────────────────────────


def _ceil_to_ue8m0_fp(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _make_fused_kv_fp8(
    kv_fp8: torch.Tensor, kv_scales: torch.Tensor, block_size: int, head_dim: int
) -> torch.Tensor:
    """Pack KV + per-token FP32 scales into the fused layout.

    Returns [num_blocks, block_size, 1, head_dim+4] uint8.
    """
    num_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_size * per_token_size
    scale_offset = block_size * head_dim

    fused = torch.zeros(
        num_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, block_size, 1, per_token_size)


def _ref_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    block_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pure-torch FP8 reference (adapted from TRT-LLM unit test)."""
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device
    logits = torch.full(
        (B * next_n, max_model_len), float("-inf"), device=device, dtype=out_dtype
    )
    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = int(context_lens[b].item())
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].to(out_dtype)

        for blk_idx in range((ctx_len + block_size - 1) // block_size):
            phys_blk = int(block_table[b, blk_idx].item())
            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk].to(out_dtype)

            k_positions = torch.arange(
                blk_idx * block_size, (blk_idx + 1) * block_size, device=device
            )
            mask = (k_positions[None, :] < ctx_len) & (
                k_positions[None, :] <= q_positions[:, None]
            )
            qk = torch.matmul(
                q_f32[b].permute(1, 0, 2), k_f32.T
            )  # [H, next_n, block_kv]
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk).to(out_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0)  # [next_n, block_kv]
            weighted = weighted * scales[None, :]

            start = blk_idx * block_size
            end = min(start + block_size, max_model_len)
            if start >= max_model_len:
                break
            ncol = end - start
            logits[b * next_n : (b + 1) * next_n, start:end] = torch.where(
                mask[:, :ncol],
                weighted[:, :ncol],
                torch.tensor(float("-inf"), device=device, dtype=out_dtype),
            )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: FP4
# ──────────────────────────────────────────────────────────────────────────────


def _ceil_to_ue8m0_int(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def _unpack_ue8m0_from_int(packed: torch.Tensor) -> torch.Tensor:
    return (packed.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=ax.dtype
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    return (code | (sign.to(torch.uint8) << 3)).view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float
    )
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def _per_token_cast_to_fp4(x: torch.Tensor, gran_k: int = 32):
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = _ceil_to_ue8m0_int(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), _pack_ue8m0_to_int(sf)


def _cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 32):
    m, n2 = packed.shape
    n = n2 * 2
    sf = _unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x = _dequantize_from_fp4_e2m1(unpacked)
    return x * sf[:, torch.arange(n, device=packed.device) // gran_k]


def _kv_cache_cast_to_fp4(x: torch.Tensor, is_kv_sf_interleaved: bool = False):
    num_blocks, block_size, num_heads, head_dim = x.shape
    x_scaled, sf = _per_token_cast_to_fp4(x.view(-1, head_dim), gran_k=32)
    x_back = _cast_back_from_fp4(x_scaled, sf, gran_k=32).view(
        num_blocks, block_size, 1, head_dim
    )
    x_fp4 = torch.empty(
        (num_blocks, block_size * (head_dim // 2 + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : block_size * head_dim // 2] = x_scaled.view(
        num_blocks, block_size * head_dim // 2
    ).view(torch.uint8)
    sf_per_block = sf.view(num_blocks, block_size)
    if is_kv_sf_interleaved and block_size == 128:
        sf_per_block = (
            sf_per_block.reshape(num_blocks, 4, 32)
            .transpose(-1, -2)
            .contiguous()
            .reshape(num_blocks, 128)
        )
    x_fp4[:, block_size * head_dim // 2 :] = sf_per_block.view(torch.uint8)
    return (
        x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4),
        x_back.to(x.dtype),
    )


def _ref_fp4_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Pure-torch FP4 reference (adapted from TRT-LLM test)."""
    batch_size, next_n, num_heads, dim = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    cl_list = context_lens.tolist()
    for i in range(batch_size):
        ctx = int(cl_list[i])
        q_offsets = torch.arange(ctx - next_n, ctx, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        n_blk = (ctx + block_size - 1) // block_size
        block_idxs = block_tables[i, :n_blk]
        kv_slice = kv_cache[block_idxs]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)
        qx = q[i].transpose(0, 1)
        s = torch.matmul(qx, kx).to(logits.dtype)
        total_len = n_blk * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < ctx) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        w = min(total_len, max_model_len)
        logits[i * next_n : (i + 1) * next_n, :w] = torch.where(
            k_offsets[None, :w] <= q_offsets[:, None], s[:, :w], float("-inf")
        )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Common: paged KV pool builder
# ──────────────────────────────────────────────────────────────────────────────


def _make_paged_kv(batch_size, block_size, context_lens, device):
    n_blk_per_seq = (context_lens + block_size - 1) // block_size
    # The kernel reads ceil(ctx/128) compute tiles * (128 // block_size) physical
    # blocks per row, which can exceed ceil(ctx/block_size) when ctx is not a
    # multiple of 128. Size block_table for that access pattern; the extra columns
    # default to physical index 0 (a valid pool block) since those positions are
    # beyond ctx (masked) — this avoids an out-of-bounds block_table/KV read.
    kern_blk = ((context_lens + 127) // 128) * (128 // block_size)
    total = int(n_blk_per_seq.sum().item())
    num_total_blocks = total + batch_size * 2
    max_blk = int(kern_blk.max().item())
    block_table = torch.zeros((batch_size, max_blk), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk_per_seq.tolist()):
        block_table[i, :nb] = pool[off : off + nb]
        off += nb
    return block_table, num_total_blocks


def _calc_cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _valid_causal_mask(context_lens, next_n, max_len, device):
    """Boolean [B*next_n, max_len] mask of in-context, causally-valid positions."""
    rows = context_lens.shape[0] * next_n
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(rows, -1)
    offsets = torch.arange(rows, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    return positions <= limits


# ──────────────────────────────────────────────────────────────────────────────
# GPU schedule kernel tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 4, 32, 128])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 32768])
def test_gpu_schedule_matches_cpu(batch_size, avg_ctx):
    """GPU schedule kernel must be bit-exact vs CPU numpy reference."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("GPU schedule kernel requires SM100a (B200)")

    from flashinfer.attn_scores.attn_scores import (
        _cached_num_sms,
        _compute_schedule_metadata,
        compute_paged_mqa_logits_schedule,
    )
    from flashinfer.utils import get_device_index

    device = "cuda"
    lo = max(128, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    )

    # Use the REAL device SM count so the CPU reference has the same [num_sms+1, 2]
    # shape as the GPU kernel output (hardcoding 148 breaks on non-148-SM devices).
    num_sms = _cached_num_sms(get_device_index(torch.device(device)))
    ref_cpu = _compute_schedule_metadata(context_lens.cpu(), num_sms).to(device)
    gpu = compute_paged_mqa_logits_schedule(context_lens, use_gpu_kernel=True)
    torch.cuda.synchronize()
    assert torch.equal(ref_cpu, gpu), (
        f"GPU/CPU schedule mismatch: max diff {(ref_cpu - gpu).abs().max().item()}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# FP8 tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.float32])
def test_fp8_paged_mqa_logits(batch_size, next_n, avg_ctx, block_size, output_dtype):
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(42)
    num_heads, head_dim = 64, 128
    max_model_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_model_len)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )

    # FP8 inputs
    q_f32 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8 = q_f32.to(torch.float8_e4m3fn)

    kv_f32 = torch.randn(num_total_blocks, block_size, head_dim, device=device)
    kv_amax = kv_f32.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        block_size,
        out_dtype=output_dtype,
    )

    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=output_dtype,
    )

    # Mask out padding / out-of-context positions before comparing
    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)

    atol, rtol = {
        torch.float32: (5e-5, 1e-5),
        torch.float16: (1e-3, 1e-3),
    }[output_dtype]
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=atol, rtol=rtol
        )

    diff = _calc_cosine_diff(out_clean, ref_clean)
    assert diff < 0.02, f"cosine diff {diff:.3e} too large"


@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("block_size", [64, 128])
def test_fp8_paged_mqa_logits_head_dim64(next_n, block_size):
    """head_dim != 128 (the only value every other test uses).

    The FP8 kernel is parametric over head_dim (unlike FP4, which asserts
    head_dim == 128), but nothing else in the suite exercises that. head_dim=64
    is the natural second point: a multiple of the MMA instruction K (32), a
    native TMA swizzle width (64 B row), and half the SMEM of head_dim=128.
    Catches any 128 accidentally hardcoded in the SMEM/TMA layout math.
    """
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(42)
    batch_size, avg_ctx, head_dim = 4, 1024, 64
    output_dtype = torch.float32
    max_model_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_model_len)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )

    num_heads = 64
    q_f32 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8 = q_f32.to(torch.float8_e4m3fn)

    kv_f32 = torch.randn(num_total_blocks, block_size, head_dim, device=device)
    kv_amax = kv_f32.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        block_size,
        out_dtype=output_dtype,
    )

    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)

    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=5e-5, rtol=1e-5
        )

    diff = _calc_cosine_diff(out_clean, ref_clean)
    assert diff < 0.02, f"cosine diff {diff:.3e} too large"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize("avg_ctx", [256, 2048])
def test_fp8_paged_mqa_logits_fp16(batch_size, next_n, avg_ctx):
    """FP16 output path: use integer-valued data to keep precision losses small."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(0)
    device = "cuda"
    num_heads, head_dim, block_size = 64, 128, 64
    max_model_len = max(avg_ctx * 2, 2048)
    output_dtype = torch.float16

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )

    # Integer-valued inputs to avoid fp16 accumulation drift
    q_i = torch.randint(
        -2, 3, (batch_size, next_n, num_heads, head_dim), device=device
    ).float()
    q_fp8 = q_i.to(torch.float8_e4m3fn)
    kv_i = torch.randint(
        -2, 3, (num_total_blocks, block_size, head_dim), device=device
    ).float()
    kv_amax = kv_i.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_i / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randint(
        -2, 3, (batch_size * next_n, num_heads), device=device
    ).float()
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        block_size,
        out_dtype=output_dtype,
    )
    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=output_dtype,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("avg_ctx", [256, 2048])
def test_fp8_paged_mqa_logits_next_n4(batch_size, avg_ctx):
    """FP8 natively supports next_n=4 without atom-split."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(5)
    device = "cuda"
    num_heads, head_dim, block_size, next_n = 64, 128, 64, 4
    max_model_len = max(avg_ctx * 2, 2048)

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )

    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(num_total_blocks, block_size, head_dim, device=device)
    kv_amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        block_size,
        out_dtype=torch.float32,
    )
    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_model_len,
    )

    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=5e-5, rtol=1e-5
        )
    assert _calc_cosine_diff(out_clean, ref_clean) < 0.02


# ──────────────────────────────────────────────────────────────────────────────
# FP4 tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("next_n", [1, 2, 3])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_fp4_paged_mqa_logits(batch_size, next_n, avg_ctx, block_size, output_dtype):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(42)
    device = "cuda"
    num_heads, head_dim = 64, 128
    max_model_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_model_len)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )

    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks,
        block_size,
        1,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_f32.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(
        batch_size, next_n, num_heads, head_dim // 2
    )
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_sim = (
        _cast_back_from_fp4(q_packed, sf_q_packed, gran_k=32)
        .view(batch_size, next_n, num_heads, head_dim)
        .to(torch.bfloat16)
    )

    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)

    ref = _ref_fp4_paged_mqa_logits(
        q_sim.float(), kv_sim.float(), weights, context_lens, block_table, max_model_len
    )

    out = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)

    tol = {
        torch.float32: (5e-5, 1e-5),
        torch.bfloat16: (1e-2, 1e-2),
        torch.float16: (1e-3, 1e-3),
    }
    atol, rtol = tol[output_dtype]
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=atol, rtol=rtol
        )

    diff = _calc_cosine_diff(out_clean, ref_clean)
    assert diff < 0.02, f"cosine diff {diff:.3e} too large"


def test_fp4_next_n4_rejected():
    """FP4 supports next_n up to 3; 4 is rejected rather than emulated.

    An earlier version decomposed next_n=4 into two next_n=2 atoms, which cost
    a per-call page-table duplication and context-length rebuild, and made a
    caller-supplied schedule_meta silently wrong (the split changes the batch
    the scheduler must describe).  The kernel asserts next_n in {1,2,3}; the
    API now reports that directly.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    device = "cuda"
    B, H, D, block_size, ctx = 2, 64, 128, 64, 256
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)

    for next_n, should_pass in ((3, True), (4, False), (5, False)):
        q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
        sf_q = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
        args = (q, sf_q, kv, w, context_lens, block_table, ctx)
        if should_pass:
            fp4_paged_mqa_logits(*args, output_dtype=torch.bfloat16)
        else:
            with pytest.raises(ValueError, match=r"next_n in 1\.\.3"):
                fp4_paged_mqa_logits(*args, output_dtype=torch.bfloat16)


@pytest.mark.parametrize("block_size", [64, 128])
def test_fp4_paged_mqa_logits_sf_interleaved(block_size):
    """is_kv_sf_interleaved=True path (requires block_size=128)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(3)
    device = "cuda"
    batch_size, next_n, num_heads, head_dim = 4, 2, 64, 128
    avg_ctx = 1024
    max_model_len = avg_ctx * 2

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )
    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks,
        block_size,
        1,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_f32.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(
        batch_size, next_n, num_heads, head_dim // 2
    )
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)

    # Pre-interleaved KV SF
    kv_fused_interleaved, kv_sim = _kv_cache_cast_to_fp4(
        kv_cache, is_kv_sf_interleaved=(block_size == 128)
    )

    out = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused_interleaved,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=torch.bfloat16,
        is_kv_sf_interleaved=(block_size == 128),
    )

    # Also run with the in-kernel rearrangement for cross-check
    kv_fused_online, _ = _kv_cache_cast_to_fp4(kv_cache, is_kv_sf_interleaved=False)
    out_online = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused_online,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=torch.bfloat16,
        is_kv_sf_interleaved=False,
    )

    # Mask to valid (in-context, finite) positions before comparing
    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    out_online_m = out_online.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(out_m) & torch.isfinite(out_online_m)
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_m[valid], out_online_m[valid], atol=1e-2, rtol=1e-2
        )


# ──────────────────────────────────────────────────────────────────────────────
# out= / schedule_meta= paths and input validation
# ──────────────────────────────────────────────────────────────────────────────


def _make_fp8_case(batch_size, next_n, ctx, block_size, device, seed=11, head_dim=128):
    torch.manual_seed(seed)
    num_heads = 64
    context_lens = torch.full((batch_size,), ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, block_size, context_lens, device
    )
    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(num_total_blocks, block_size, head_dim, device=device)
    kv_amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, block_size, head_dim)
    return q_fp8, kv_fused, weights, context_lens, block_table


def test_fp8_out_and_schedule_meta_paths():
    """out= (pre-allocated buffer) and schedule_meta= (pre-computed) must match the
    default path bit-for-bit and the returned tensor must alias out."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import (
        padded_context_len,
        compute_paged_mqa_logits_schedule,
        fp8_paged_mqa_logits,
    )

    B, next_n, ctx, block_size = 4, 2, 4096, 64
    max_ml = ctx + 512
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, block_size, device)

    ref = fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml)

    # Pre-allocated out= sized via padded_context_len, plus pre-computed schedule.
    padded = padded_context_len(max_ml)
    out_buf = torch.empty((B * next_n, padded), device=device, dtype=torch.float32)
    sched = compute_paged_mqa_logits_schedule(cl)
    ret = fp8_paged_mqa_logits(
        q, kv_fused, w, cl, bt, max_ml, schedule_meta=sched, out=out_buf
    )

    assert ret.data_ptr() == out_buf.data_ptr(), "returned tensor must alias out="
    # Same kernel + same schedule → bit-identical in the causal-valid region.
    # (Beyond-context padding is unwritten garbage and differs between buffers.)
    valid = _valid_causal_mask(cl, next_n, max_ml, device)
    torch.testing.assert_close(ret.float()[valid], ref.float()[valid], atol=0, rtol=0)


def test_fp4_out_and_schedule_meta_paths():
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import (
        padded_context_len,
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
    )

    torch.manual_seed(13)
    B, next_n, ctx, block_size = 4, 2, 4096, 64
    num_heads, head_dim = 64, 128
    max_ml = ctx + 512
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(B, block_size, context_lens, device)
    q_bf = torch.randn(
        B, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(B * next_n, num_heads, device=device, dtype=torch.float32)
    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_bf.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(B, next_n, num_heads, head_dim // 2)
    sf_q = sf_q_packed.view(torch.int32).view(B, next_n, num_heads)
    kv_fused, _ = _kv_cache_cast_to_fp4(kv_cache)

    ref = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_ml,
        output_dtype=torch.float32,
    )
    padded = padded_context_len(max_ml)
    out_buf = torch.empty((B * next_n, padded), device=device, dtype=torch.float32)
    sched = compute_paged_mqa_logits_schedule(context_lens)
    ret = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused,
        weights,
        context_lens,
        block_table,
        max_ml,
        output_dtype=torch.float32,
        schedule_meta=sched,
        out=out_buf,
    )
    assert ret.data_ptr() == out_buf.data_ptr()
    valid = _valid_causal_mask(context_lens, next_n, max_ml, device)
    torch.testing.assert_close(ret.float()[valid], ref.float()[valid], atol=0, rtol=0)


def test_fp8_input_validation():
    """Guard rails: bad out= size, unsupported dtype, and non-int32/CPU inputs must raise."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import padded_context_len, fp8_paged_mqa_logits

    B, next_n, ctx, block_size = 2, 1, 1024, 64
    max_ml = ctx + 300  # deliberately not a multiple of 256 so padded_ctx_len > max_ml
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, block_size, device)

    # Undersized out= (max_ml columns instead of padded) → ValueError (prevents OOB)
    bad_out = torch.empty((B * next_n, max_ml), device=device, dtype=torch.float32)
    assert padded_context_len(max_ml) > max_ml
    with pytest.raises(ValueError, match="padded_context_len"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml, out=bad_out)

    # Unsupported output dtype for FP8
    with pytest.raises(ValueError, match="float32, float16"):
        fp8_paged_mqa_logits(
            q, kv_fused, w, cl, bt, max_ml, output_dtype=torch.bfloat16
        )

    # CPU context_lens → ValueError (kernel needs int32 CUDA)
    with pytest.raises(ValueError, match="context_lens"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl.cpu(), bt, max_ml)

    # int64 block_table → ValueError
    with pytest.raises(ValueError, match="block_table"):
        fp8_paged_mqa_logits(q, kv_fused, w, cl, bt.to(torch.int64), max_ml)

    # head_dim not a multiple of the MMA instruction K (32) → ValueError.
    # Without this guard the kernel's `head_dim // 32` integer division would
    # silently truncate the QK contraction (100 -> 96) and return wrong logits.
    q_bad, kv_bad, w_bad, cl_bad, bt_bad = _make_fp8_case(
        B, next_n, ctx, block_size, device, head_dim=100
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        fp8_paged_mqa_logits(q_bad, kv_bad, w_bad, cl_bad, bt_bad, max_ml)

    # head_dim large enough to exceed the per-CTA SMEM budget → ValueError.
    # Measured on sm_100a: head_dim=256 needs 249856 B vs a 232448 B cap, and
    # without this check the launch fails with a bare cudaErrorInvalidValue.
    q_big, kv_big, w_big, cl_big, bt_big = _make_fp8_case(
        B, next_n, ctx, block_size, device, head_dim=256
    )
    with pytest.raises(ValueError, match="shared memory"):
        fp8_paged_mqa_logits(q_big, kv_big, w_big, cl_big, bt_big, max_ml)

    # block_size that does not divide the 128-token compute tile into <=4
    # sub-blocks. Measured: 16 trips "num_blocks_per_mma=8 exceeds max 4" and
    # 48 trips "block_kv=128 must be divisible by block_size=48", both bare
    # kernel assertions without this check.
    for bad_pbk in (16, 48):
        q2, kv2, w2, cl2, bt2 = _make_fp8_case(B, next_n, ctx, 64, device)
        kv_reshaped = torch.zeros(
            (kv2.shape[0], bad_pbk, 1, kv2.shape[-1]),
            dtype=kv2.dtype,
            device=kv2.device,
        )
        with pytest.raises(ValueError, match="block_size"):
            fp8_paged_mqa_logits(q2, kv_reshaped, w2, cl2, bt2, max_ml)

    # next_n * num_heads outside the UMMA N-mode range [8, 256].
    # Measured: next_n=5 at num_heads=64 gives N=320 -> opaque DSL OpError.
    q3, kv3, w3, cl3, bt3 = _make_fp8_case(B, 5, ctx, block_size, device)
    with pytest.raises(ValueError, match="N-mode"):
        fp8_paged_mqa_logits(q3, kv3, w3, cl3, bt3, max_ml)


def test_fp4_head_dim_num_heads_validation():
    """FP4 hardcodes head_dim=128 / num_heads=64; the wrapper must say so clearly."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    B, next_n, num_heads, block_size, ctx = 2, 1, 64, 64, 512
    max_ml = 2048
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)

    def _case(head_dim, n_heads):
        half_d = head_dim // 2
        q = torch.zeros(B, next_n, n_heads, half_d, dtype=torch.uint8, device=device)
        sf_q = torch.zeros(B, next_n, n_heads, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, half_d + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, n_heads, device=device, dtype=torch.float32)
        return q, sf_q, kv, w

    # Wrong head_dim (64 instead of 128)
    q, sf_q, kv, w = _case(64, num_heads)
    with pytest.raises(ValueError, match="requires head_dim == 128"):
        fp4_paged_mqa_logits(q, sf_q, kv, w, context_lens, block_table, max_ml)

    # Wrong num_heads (32 instead of 64)
    q, sf_q, kv, w = _case(128, 32)
    with pytest.raises(ValueError, match="requires num_heads == 64"):
        fp4_paged_mqa_logits(q, sf_q, kv, w, context_lens, block_table, max_ml)

    # next_n beyond what the kernel supports (1-3).
    q5 = torch.zeros(B, 5, num_heads, 64, dtype=torch.uint8, device=device)
    sf5 = torch.zeros(B, 5, num_heads, dtype=torch.int32, device=device)
    kv5 = torch.zeros(ntb, block_size, 1, 68, dtype=torch.uint8, device=device)
    w5 = torch.randn(B * 5, num_heads, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="next_n"):
        fp4_paged_mqa_logits(q5, sf5, kv5, w5, context_lens, block_table, max_ml)

    # block_size not a valid sub-tiling of the 128-token compute tile.
    q6, sf6, kv6, w6 = _case(128, num_heads)
    kv_bad = torch.zeros(ntb, 48, 1, 68, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="block_size"):
        fp4_paged_mqa_logits(q6, sf6, kv_bad, w6, context_lens, block_table, max_ml)


def test_block_table_width_contract(monkeypatch):
    """block_table must be wide enough for the kernel's compute-tile indexing.

    The kernel reads ceil(ctx/128) tiles x (128 // block_size) pages per tile,
    which exceeds ceil(ctx/block_size) when ctx is not a multiple of 128, so a
    naturally-sized table is indexed out of bounds. The bound needs the per-row
    context_lens from device memory, so the check is opt-in behind
    FLASHINFER_VALIDATE_INPUTS. Regression for PR #4365 review r3824399380.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import _validate_paged_bounds

    device = "cuda"
    B, H, D, block_size = 2, 64, 128, 64
    ctx = 257  # deliberately not a multiple of 128
    pages = -(-ctx // block_size)  # 5 -- what a caller would naturally allocate
    need = -(-ctx // 128) * (128 // block_size)  # 6 -- what the kernel indexes
    assert (pages, need) == (5, 6), "test premise: ctx exposes the tile/page gap"

    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    narrow = torch.zeros((B, pages), dtype=torch.int32, device=device)
    wide = torch.zeros((B, need), dtype=torch.int32, device=device)

    # Default (unset): no sync, no check. Exercise the validator directly rather
    # than launching -- a narrow table would genuinely read out of bounds.
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    assert _validate_paged_bounds(narrow, context_lens, ctx, block_size, "x") is None

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    ntb = B * need
    w = torch.zeros(B * 1, H, device=device, dtype=torch.float32)

    q8 = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"block_table has 5 columns.*indexes up to 6"):
        fp8_paged_mqa_logits(q8, kv8, w, context_lens, narrow, ctx)
    fp8_paged_mqa_logits(q8, kv8, w, context_lens, wide, ctx)

    q4 = torch.zeros(B, 1, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, 1, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"block_table has 5 columns.*indexes up to 6"):
        fp4_paged_mqa_logits(
            q4, sf4, kv4, w, context_lens, narrow, ctx, output_dtype=torch.bfloat16
        )
    fp4_paged_mqa_logits(
        q4, sf4, kv4, w, context_lens, wide, ctx, output_dtype=torch.bfloat16
    )

    # Lives in the API body, so skip_check=True must not bypass it.
    with pytest.raises(ValueError, match=r"block_table has 5 columns"):
        fp8_paged_mqa_logits(q8, kv8, w, context_lens, narrow, ctx, skip_check=True)

    # A padded table is accepted even though ctx is not a multiple of 128, and
    # max_context_len being much larger than ctx must NOT tighten the bound.
    fp8_paged_mqa_logits(q8, kv8, w, context_lens, wide, ctx * 8)


def test_max_context_len_bound(monkeypatch):
    """max_context_len must be >= max(context_lens), or the kernel writes OOB.

    The output row is sized from max_context_len while the schedule is derived
    from context_lens; context_lens=[257] with max_context_len=256 allocates 256
    columns but schedules splits reaching 512, and the store is unconditional.
    Checked under FLASHINFER_VALIDATE_INPUTS since the bound needs the per-row
    lengths from device memory. Regression for PR #4365 review r3824481310.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import _validate_paged_bounds

    device = "cuda"
    B, H, D, block_size = 1, 64, 128, 64
    ctx, max_ml = 257, 256  # the reviewer's example: schedule reaches 512

    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    width = -(-ctx // 128) * (128 // block_size)
    block_table = torch.zeros((B, width), dtype=torch.int32, device=device)
    ntb = B * width
    w = torch.zeros(B * 1, H, device=device, dtype=torch.float32)

    # Default (unset): no sync, no check -- exercise the validator directly
    # rather than launching, which would genuinely write out of bounds.
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    assert (
        _validate_paged_bounds(block_table, context_lens, max_ml, block_size, "x")
        is None
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    q8 = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"max_context_len \(256\) must be at least"):
        fp8_paged_mqa_logits(q8, kv8, w, context_lens, block_table, max_ml)

    q4 = torch.zeros(B, 1, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, 1, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"max_context_len \(256\) must be at least"):
        fp4_paged_mqa_logits(
            q4,
            sf4,
            kv4,
            w,
            context_lens,
            block_table,
            max_ml,
            output_dtype=torch.bfloat16,
        )

    # skip_check=True must not bypass it (silent OOB write).
    with pytest.raises(ValueError, match=r"max_context_len \(256\) must be at least"):
        fp8_paged_mqa_logits(
            q8, kv8, w, context_lens, block_table, max_ml, skip_check=True
        )

    # Raising max_context_len to the real length makes it legal again.
    fp8_paged_mqa_logits(q8, kv8, w, context_lens, block_table, ctx)

    # Ragged lengths: only the longest row matters.
    ragged = torch.tensor([64, 257], dtype=torch.int32, device=device)
    bt2 = torch.zeros((2, width), dtype=torch.int32, device=device)
    q8b = torch.zeros(2, 1, H, D, device=device).to(torch.float8_e4m3fn)
    wb = torch.zeros(2, H, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"max\(context_lens\) \(257\)"):
        fp8_paged_mqa_logits(q8b, kv8, wb, ragged, bt2, max_ml)


def test_precompile_variants():
    """precompile_paged_mqa_logits(variants=...) builds only what was asked for."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits, precompile_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import (
        _cached_compile_fp4_kernel,
        _cached_compile_fp8_kernel,
    )

    with pytest.raises(ValueError, match="unknown variants"):
        precompile_paged_mqa_logits(variants=("fp8", "int4"))

    # fp8-only must not touch the fp4 cache, and vice versa.
    fp4_before = _cached_compile_fp4_kernel.cache_info().misses
    precompile_paged_mqa_logits(variants=("fp8",))
    assert _cached_compile_fp4_kernel.cache_info().misses == fp4_before

    fp8_before = _cached_compile_fp8_kernel.cache_info().misses
    precompile_paged_mqa_logits(variants=("fp4",))
    assert _cached_compile_fp8_kernel.cache_info().misses == fp8_before

    # Every advertised fp8 config must now be a cache hit, not a fresh build.
    precompile_paged_mqa_logits(variants=("fp8",))
    misses = _cached_compile_fp8_kernel.cache_info().misses
    device = "cuda"
    for block_size in (64, 128):
        for next_n in (1, 2, 3, 4):
            ctx, max_ml = 512, 512
            context_lens = torch.full((2,), ctx, dtype=torch.int32, device=device)
            block_table, ntb = _make_paged_kv(2, block_size, context_lens, device)
            q = torch.zeros(2, next_n, 64, 128, device=device).to(torch.float8_e4m3fn)
            kv = torch.zeros(ntb, block_size, 1, 132, dtype=torch.uint8, device=device)
            w = torch.randn(2 * next_n, 64, device=device, dtype=torch.float32)
            fp8_paged_mqa_logits(q, kv, w, context_lens, block_table, max_ml)
    assert _cached_compile_fp8_kernel.cache_info().misses == misses, (
        "a precompiled fp8 config still triggered a build; the precompile "
        "config list has drifted from what the API actually compiles"
    )


def test_fp4_is_kv_sf_interleaved_guard():
    """is_kv_sf_interleaved=True is only valid at block_size=128.

    The kernel silently forces the flag back to False for other page sizes, so
    a caller who pre-arranged SF for UTCCP would get it interleaved twice
    -- wrong logits with no error. The API must reject the combination.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")
    from flashinfer import fp4_paged_mqa_logits

    device, B, next_n, H, D, ctx, max_ml = "cuda", 2, 1, 64, 128, 512, 512
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)

    for block_size, should_raise in ((32, True), (64, True), (128, False)):
        block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)
        q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
        sf_q = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
        if should_raise:
            with pytest.raises(ValueError, match="is_kv_sf_interleaved"):
                fp4_paged_mqa_logits(
                    q,
                    sf_q,
                    kv,
                    w,
                    context_lens,
                    block_table,
                    max_ml,
                    is_kv_sf_interleaved=True,
                )
        else:
            fp4_paged_mqa_logits(
                q,
                sf_q,
                kv,
                w,
                context_lens,
                block_table,
                max_ml,
                is_kv_sf_interleaved=True,
            )


@pytest.mark.parametrize(
    "sub,valid",
    [
        (1, True),
        (2, True),
        (4, True),
        (16, True),
        (0, False),
        (3, False),
        (32, False),
        (64, False),
    ],
)
def test_num_epi_subtiles_guard(sub, valid):
    """num_epi_subtiles must divide num_heads with the quotient a multiple of 4.

    Previously this escaped to a bare assertion inside kernel construction at
    JIT time; both APIs now reject it at the boundary.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires SM100a (B200)")
    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits

    device, B, next_n, H, D, block_size, ctx, max_ml = (
        "cuda",
        2,
        1,
        64,
        128,
        64,
        512,
        512,
    )
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)

    q8 = torch.zeros(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q4 = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)

    if valid:
        fp8_paged_mqa_logits(
            q8, kv8, w, context_lens, block_table, max_ml, num_epi_subtiles=sub
        )
        fp4_paged_mqa_logits(
            q4, sf4, kv4, w, context_lens, block_table, max_ml, num_epi_subtiles=sub
        )
    else:
        with pytest.raises(ValueError, match="num_epi_subtiles"):
            fp8_paged_mqa_logits(
                q8, kv8, w, context_lens, block_table, max_ml, num_epi_subtiles=sub
            )
        with pytest.raises(ValueError, match="num_epi_subtiles"):
            fp4_paged_mqa_logits(
                q4, sf4, kv4, w, context_lens, block_table, max_ml, num_epi_subtiles=sub
            )


def test_next_n_and_weights_contract():
    """weights shape/dtype are validated, and pin next_n without a next_n arg."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires SM100a (B200)")
    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits

    device, B, next_n, H, D, block_size, ctx, max_ml = (
        "cuda",
        2,
        2,
        64,
        128,
        64,
        512,
        512,
    )
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    q8 = torch.zeros(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q4 = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)

    # next_n is not an API parameter: it is q.shape[1] by definition, and a q
    # that disagrees with the other tensors is caught by the weights and sf_q
    # cross-checks without needing the caller to restate it.
    q8_bad = torch.zeros(B, next_n + 1, H, D, device=device).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weights must be"):
        fp8_paged_mqa_logits(q8_bad, kv8, w, context_lens, block_table, max_ml)
    q4_bad = torch.zeros(B, next_n + 1, H, D // 2, dtype=torch.uint8, device=device)
    # Either cross-check may fire first depending on ordering; both prove the
    # inconsistency is caught without an explicit next_n argument.
    with pytest.raises(ValueError, match="(weights|sf_q) must be"):
        fp4_paged_mqa_logits(q4_bad, sf4, kv4, w, context_lens, block_table, max_ml)

    # weights shape and dtype are now validated (previously unchecked).
    bad_shape = torch.randn(B * next_n + 1, H, device=device, dtype=torch.float32)
    bad_dtype = torch.randn(B * next_n, H, device=device, dtype=torch.bfloat16)
    for bad_w, pat in (
        (bad_shape, "weights must be"),
        (bad_dtype, "weights must be float32"),
    ):
        with pytest.raises(ValueError, match=pat):
            fp8_paged_mqa_logits(q8, kv8, bad_w, context_lens, block_table, max_ml)
        with pytest.raises(ValueError, match=pat):
            fp4_paged_mqa_logits(q4, sf4, kv4, bad_w, context_lens, block_table, max_ml)


def test_fp4_sf_vec_size_contract():
    """sf_vec_size drives both sf_q packing and the KV row's scale-factor bytes."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")
    from flashinfer import fp4_paged_mqa_logits

    device, B, next_n, H, D, block_size, ctx, max_ml = (
        "cuda",
        2,
        1,
        64,
        128,
        64,
        512,
        512,
    )
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, ntb = _make_paged_kv(B, block_size, context_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    sf_q = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)

    fp4_paged_mqa_logits(
        q, sf_q, kv, w, context_lens, block_table, max_ml, sf_vec_size=32
    )
    for bad in (16, 64):
        with pytest.raises(ValueError, match="sf_vec_size must be 32"):
            fp4_paged_mqa_logits(
                q, sf_q, kv, w, context_lens, block_table, max_ml, sf_vec_size=bad
            )

    # The KV row's scale-factor bytes are derived from sf_vec_size, so a row
    # sized for a different SF count is rejected with the arithmetic spelled out.
    kv_bad = torch.zeros(
        ntb, block_size, 1, D // 2 + 8, dtype=torch.uint8, device=device
    )
    with pytest.raises(ValueError, match="scale-factor bytes"):
        fp4_paged_mqa_logits(q, sf_q, kv_bad, w, context_lens, block_table, max_ml)
