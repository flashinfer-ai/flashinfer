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
    kv_fp8: torch.Tensor, kv_scales: torch.Tensor, phys_block_kv: int, head_dim: int
) -> torch.Tensor:
    """Pack KV + per-token FP32 scales into the fused layout.

    Returns [num_blocks, phys_block_kv, 1, head_dim+4] uint8.
    """
    num_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = phys_block_kv * per_token_size
    scale_offset = phys_block_kv * head_dim

    fused = torch.zeros(
        num_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, phys_block_kv, 1, per_token_size)


def _ref_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    phys_block_kv: int,
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

        for blk_idx in range((ctx_len + phys_block_kv - 1) // phys_block_kv):
            phys_blk = int(block_table[b, blk_idx].item())
            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk].to(out_dtype)

            k_positions = torch.arange(
                blk_idx * phys_block_kv, (blk_idx + 1) * phys_block_kv, device=device
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

            start = blk_idx * phys_block_kv
            end = start + phys_block_kv
            logits[b * next_n : (b + 1) * next_n, start:end] = torch.where(
                mask,
                weighted,
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


def _kv_cache_cast_to_fp4(x: torch.Tensor, remove_online_sf_transpose: bool = False):
    num_blocks, page_size, num_heads, head_dim = x.shape
    x_scaled, sf = _per_token_cast_to_fp4(x.view(-1, head_dim), gran_k=32)
    x_back = _cast_back_from_fp4(x_scaled, sf, gran_k=32).view(
        num_blocks, page_size, 1, head_dim
    )
    x_fp4 = torch.empty(
        (num_blocks, page_size * (head_dim // 2 + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : page_size * head_dim // 2] = x_scaled.view(
        num_blocks, page_size * head_dim // 2
    ).view(torch.uint8)
    sf_per_block = sf.view(num_blocks, page_size)
    if remove_online_sf_transpose and page_size == 128:
        sf_per_block = (
            sf_per_block.reshape(num_blocks, 4, 32)
            .transpose(-1, -2)
            .contiguous()
            .reshape(num_blocks, 128)
        )
    x_fp4[:, page_size * head_dim // 2 :] = sf_per_block.view(torch.uint8)
    return (
        x_fp4.view(num_blocks, page_size, num_heads, head_dim // 2 + 4),
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
        logits[i * next_n : (i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Common: paged KV pool builder
# ──────────────────────────────────────────────────────────────────────────────


def _make_paged_kv(batch_size, phys_block_kv, context_lens, device):
    n_blk_per_seq = (context_lens + phys_block_kv - 1) // phys_block_kv
    total = int(n_blk_per_seq.sum().item())
    num_total_blocks = total + batch_size * 2
    max_blk = int(n_blk_per_seq.max().item())
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
@pytest.mark.parametrize("phys_block_kv", [64, 128])
@pytest.mark.parametrize("output_dtype", [torch.float32])
def test_fp8_paged_mqa_logits(batch_size, next_n, avg_ctx, phys_block_kv, output_dtype):
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(42)
    num_heads, head_dim = 64, 128
    max_model_len = max(avg_ctx * 2, 2048)

    lo = max(phys_block_kv, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_model_len)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )

    # FP8 inputs
    q_f32 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8 = q_f32.to(torch.float8_e4m3fn)

    kv_f32 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    kv_amax = kv_f32.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, phys_block_kv, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        phys_block_kv,
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
    num_heads, head_dim, phys_block_kv = 64, 128, 64
    max_model_len = max(avg_ctx * 2, 2048)
    output_dtype = torch.float16

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )

    # Integer-valued inputs to avoid fp16 accumulation drift
    q_i = torch.randint(
        -2, 3, (batch_size, next_n, num_heads, head_dim), device=device
    ).float()
    q_fp8 = q_i.to(torch.float8_e4m3fn)
    kv_i = torch.randint(
        -2, 3, (num_total_blocks, phys_block_kv, head_dim), device=device
    ).float()
    kv_amax = kv_i.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_i / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randint(
        -2, 3, (batch_size * next_n, num_heads), device=device
    ).float()
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, phys_block_kv, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        phys_block_kv,
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
    num_heads, head_dim, phys_block_kv, next_n = 64, 128, 64, 4
    max_model_len = max(avg_ctx * 2, 2048)

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )

    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    kv_amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, phys_block_kv, head_dim)

    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        phys_block_kv,
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
@pytest.mark.parametrize("phys_block_kv", [32, 64, 128])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_fp4_paged_mqa_logits(batch_size, next_n, avg_ctx, phys_block_kv, output_dtype):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(42)
    device = "cuda"
    num_heads, head_dim = 64, 128
    max_model_len = max(avg_ctx * 2, 2048)

    lo = max(phys_block_kv, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    context_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_model_len)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )

    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks,
        phys_block_kv,
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


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
def test_fp4_paged_mqa_logits_next_n4(batch_size, avg_ctx):
    """next_n=4 is handled via internal atom-split."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP4 paged MQA logits requires SM100a (B200)")

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(7)
    device = "cuda"
    num_heads, head_dim, phys_block_kv = 64, 128, 64
    next_n = 4
    max_model_len = max(avg_ctx * 2, 2048)

    context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )

    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks,
        phys_block_kv,
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
        output_dtype=torch.bfloat16,
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
            out_clean[valid], ref_clean[valid], atol=1e-2, rtol=1e-2
        )
    diff = _calc_cosine_diff(out_clean, ref_clean)
    assert diff < 0.02, f"cosine diff {diff:.3e} too large"


@pytest.mark.parametrize("phys_block_kv", [64, 128])
def test_fp4_paged_mqa_logits_remove_sf_transpose(phys_block_kv):
    """remove_online_sf_transpose=True path (requires phys_block_kv=128)."""
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
        batch_size, phys_block_kv, context_lens, device
    )
    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks,
        phys_block_kv,
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

    # Pre-transposed KV SF
    kv_fused_transposed, kv_sim = _kv_cache_cast_to_fp4(
        kv_cache, remove_online_sf_transpose=(phys_block_kv == 128)
    )

    out = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused_transposed,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=torch.bfloat16,
        remove_online_sf_transpose=(phys_block_kv == 128),
    )

    # Also run with online transpose for cross-check
    kv_fused_online, _ = _kv_cache_cast_to_fp4(
        kv_cache, remove_online_sf_transpose=False
    )
    out_online = fp4_paged_mqa_logits(
        q_fp4,
        sf_q,
        kv_fused_online,
        weights,
        context_lens,
        block_table,
        max_model_len,
        output_dtype=torch.bfloat16,
        remove_online_sf_transpose=False,
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


def _make_fp8_case(batch_size, next_n, ctx, phys_block_kv, device, seed=11):
    torch.manual_seed(seed)
    num_heads, head_dim = 64, 128
    context_lens = torch.full((batch_size,), ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(
        batch_size, phys_block_kv, context_lens, device
    )
    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    kv_amax = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0_fp(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_f32 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )
    kv_fused = _make_fused_kv_fp8(kv_fp8, kv_scale, phys_block_kv, head_dim)
    return q_fp8, kv_fused, weights, context_lens, block_table


def test_fp8_out_and_schedule_meta_paths():
    """out= (pre-allocated buffer) and schedule_meta= (pre-computed) must match the
    default path bit-for-bit and the returned tensor must alias out."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip("FP8 paged MQA logits requires SM100a (B200)")

    from flashinfer import (
        aligned_context_len,
        compute_paged_mqa_logits_schedule,
        fp8_paged_mqa_logits,
    )

    B, next_n, ctx, pbk = 4, 2, 4096, 64
    max_ml = ctx + 512
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, pbk, device)

    ref = fp8_paged_mqa_logits(q, kv_fused, w, cl, bt, max_ml)

    # Pre-allocated out= sized via aligned_context_len, plus pre-computed schedule.
    aligned = aligned_context_len(max_ml)
    out_buf = torch.empty((B * next_n, aligned), device=device, dtype=torch.float32)
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
        aligned_context_len,
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
    )

    torch.manual_seed(13)
    B, next_n, ctx, pbk = 4, 2, 4096, 64
    num_heads, head_dim = 64, 128
    max_ml = ctx + 512
    context_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_table, num_total_blocks = _make_paged_kv(B, pbk, context_lens, device)
    q_bf = torch.randn(
        B, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks, pbk, 1, head_dim, device=device, dtype=torch.bfloat16
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
    aligned = aligned_context_len(max_ml)
    out_buf = torch.empty((B * next_n, aligned), device=device, dtype=torch.float32)
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

    from flashinfer import aligned_context_len, fp8_paged_mqa_logits

    B, next_n, ctx, pbk = 2, 1, 1024, 64
    max_ml = ctx + 300  # deliberately non-256-aligned so aligned_ctx > max_ml
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, pbk, device)

    # Undersized out= (max_ml columns instead of aligned) → ValueError (prevents OOB)
    bad_out = torch.empty((B * next_n, max_ml), device=device, dtype=torch.float32)
    assert aligned_context_len(max_ml) > max_ml
    with pytest.raises(ValueError, match="aligned_context_len"):
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
