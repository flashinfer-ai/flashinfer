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
"""Tests for paged FP8/FP4 MQA scoring and selective TopK.

Reference implementations are adapted from TRT-LLM test scripts.
"""

from __future__ import annotations

import gc
import importlib.util
import math
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import contextlib
import os
import shutil
import subprocess
import sys
import functools

import torch

from flashinfer import FP8PagedMQATopKWrapper
from flashinfer.attn_scores import selective_logits as sl
from flashinfer.attn_scores.selective_logits import (
    _capture_cuda_graph,
    _fp8_mqa_selective_logits,
    _SelectiveLogitsMode,
)
from flashinfer.utils import is_sm100a_supported

from tests.attn_scores._selective_engine_case import SelectiveEngineCase


_SM107_UNSUPPORTED_KERNEL_TESTS = frozenset(
    {
        "test_block_table_width_contract",
        "test_context_lens_must_be_rank_1",
        "test_empty_batch_returns_without_launching",
        "test_empty_batch_still_validates_out",
        "test_fp4_head_dim_num_heads_validation",
        "test_fp4_is_kv_sf_interleaved_guard",
        "test_fp4_next_n4_rejected",
        "test_fp4_out_and_schedule_meta_paths",
        "test_fp4_paged_mqa_logits",
        "test_fp4_paged_mqa_logits_sf_interleaved",
        "test_fp4_sf_vec_size_contract",
        "test_fp8_input_validation",
        "test_fp8_out_and_schedule_meta_paths",
        "test_fp8_paged_mqa_logits",
        "test_fp8_paged_mqa_logits_fp16",
        "test_fp8_paged_mqa_logits_head_dim64",
        "test_fp8_paged_mqa_logits_next_n4",
        "test_fp8_paged_mqa_logits_small_num_heads",
        "test_max_context_len_bound",
        "test_next_n_and_weights_contract",
        "test_num_epi_subtiles_guard",
        "test_oversized_out_returns_exact_rows",
        "test_precompile_variants",
        "test_relu_is_applied_per_head_before_weighting",
        "test_schedule_meta_graph_replay_across_split_boundary",
        "test_stale_schedule_meta_detected",
    }
)


@pytest.fixture(autouse=True)
def _skip_sm107_unsupported_kernel_tests(request):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return
    test_name = getattr(request.node, "originalname", request.node.name)
    if test_name in _SM107_UNSUPPORTED_KERNEL_TESTS:
        pytest.skip("paged MQA logits kernels support SM100 and SM103, not SM107")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: FP8
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "epi_dtype,acc_dtype,output_dtype",
    [
        (torch.float32, torch.float32, torch.float32),
        (torch.float32, torch.float16, torch.float16),
        (torch.float16, torch.float32, torch.float32),
        (torch.float16, torch.float16, torch.float16),
    ],
)
def test_fp8_math_mode_finite_and_fp16_policy(epi_dtype, acc_dtype, output_dtype):
    """Mode alternation retains FP16 policy and separate ordinary cache entries."""
    from flashinfer import fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import _cached_compile_fp8_kernel

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100-class GPU")
    torch.manual_seed(13)
    q = torch.randint(-2, 3, (1, 2, 64, 128), device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randint(-2, 3, (2, 128, 128), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((2, 128), 0.5, device="cuda")
    weights = torch.randint(-2, 3, (2, 64), device="cuda").float()
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([255], device="cuda", dtype=torch.int32)
    fused = _make_fused_kv_fp8(kv, scale, 128, 128)
    reference = _ref_fp8_paged_mqa_logits(
        q, kv, scale, weights, lengths, table, 256, 128, out_dtype=epi_dtype
    ).to(output_dtype)
    with pytest.raises(ValueError, match="math_mode"):
        fp8_paged_mqa_logits(
            q, fused, weights, table, lengths, 256, math_mode="invalid"
        )
    _cached_compile_fp8_kernel.cache_clear()
    outputs = []
    cache_states = []
    for mode in ("relu_fast", "relu_strict", "relu_fast"):
        outputs.append(
            fp8_paged_mqa_logits(
                q,
                fused,
                weights,
                table,
                lengths,
                256,
                epi_dtype=epi_dtype,
                acc_dtype=acc_dtype,
                output_dtype=output_dtype,
                math_mode=mode,
            ).clone()
        )
        cache_states.append(_cached_compile_fp8_kernel.cache_info())
    torch.cuda.synchronize()
    # The public scorer deliberately leaves future/padded columns unspecified.
    valid = _valid_causal_mask(lengths, 2, 256, "cuda")
    for output in outputs:
        torch.testing.assert_close(output[valid], reference[valid], atol=0, rtol=0)
    assert torch.equal(outputs[0][valid], outputs[2][valid])
    if epi_dtype == torch.float16:
        assert torch.equal(outputs[0][valid], outputs[1][valid])
    assert [state.currsize for state in cache_states] == [1, 2, 2]
    assert [state.misses for state in cache_states] == [1, 2, 2]
    assert cache_states[2].hits == cache_states[1].hits + 1
    default_kwargs = dict(
        epi_dtype=epi_dtype, acc_dtype=acc_dtype, output_dtype=output_dtype
    )
    default_output = fp8_paged_mqa_logits(
        q, fused, weights, table, lengths, 256, **default_kwargs
    ).clone()
    assert _cached_compile_fp8_kernel.cache_info().hits == cache_states[-1].hits + 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8_paged_mqa_logits(
            q, fused, weights, table, lengths, 256, **default_kwargs
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(default_output[valid], outputs[1][valid], atol=0, rtol=0)
    torch.testing.assert_close(captured[valid], outputs[1][valid], atol=0, rtol=0)


def test_fp8_math_mode_rejects_before_cuda_work():
    from flashinfer import FP8PagedMQATopKWrapper, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import precompile_paged_mqa_logits

    for invalid in ("auto", "", None, 1):
        with pytest.raises(ValueError, match="math_mode"):
            FP8PagedMQATopKWrapper(strategy="selective", math_mode=invalid)
        with pytest.raises(ValueError, match="math_mode"):
            # Isolate API-body validation from the independent device-capability
            # decorator: all-None tensors intentionally have no capability.
            fp8_paged_mqa_logits(
                None,
                None,
                None,
                None,
                None,
                128,
                math_mode=invalid,
                skip_check=True,
            )
        with pytest.raises(ValueError, match="math_mode"):
            precompile_paged_mqa_logits(math_mode=invalid)
    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    assert wrapper.math_mode == "relu_strict"
    with pytest.raises(AttributeError):
        wrapper.math_mode = "relu_fast"
    wrapper._math_mode = "relu_fast"
    with pytest.raises(RuntimeError, match="math_mode"):
        wrapper.run(q=None, kv_fused=None, weights=None, block_table=None)


@pytest.mark.parametrize("case", ["negative_infinity", "nan_weight", "tiny_scale"])
def test_fp8_math_mode_exceptional_values(case):
    """Observe the documented fast recipe, not a strict-equivalence promise."""
    from flashinfer import fp8_paged_mqa_logits

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100-class GPU")
    q = torch.zeros((1, 1, 64, 128), device="cuda")
    kv = torch.ones((1, 128, 128), device="cuda")
    weights = torch.zeros((1, 64), device="cuda")
    q[0, 0, 0, 0] = 1
    weights[0, 0] = 1
    scale = torch.ones((1, 128), device="cuda")
    acc_dtype = torch.float32
    if case == "negative_infinity":
        q[0, 0, 0, :] = -448
        kv.fill_(448)
        acc_dtype = torch.float16
    elif case == "nan_weight":
        weights[0, 0] = float("nan")
    else:
        scale.fill_(torch.finfo(torch.float32).tiny * torch.finfo(torch.float32).eps)
    q = q.to(torch.float8_e4m3fn)
    fused = _make_fused_kv_fp8(kv.to(torch.float8_e4m3fn), scale, 128, 128)
    table = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    outputs = {}
    for mode in ("relu_fast", "relu_strict", None):
        outputs[mode] = fp8_paged_mqa_logits(
            q,
            fused,
            weights,
            table,
            lengths,
            128,
            acc_dtype=acc_dtype,
            **({} if mode is None else {"math_mode": mode}),
        ).clone()
    torch.cuda.synchronize()
    if case == "negative_infinity":
        assert torch.isnan(outputs["relu_fast"]).all()
        assert (outputs["relu_strict"] == 0).all()
    elif case == "nan_weight":
        assert all(torch.isnan(output).all() for output in outputs.values())
    else:
        assert (outputs["relu_fast"] == 0).all()
        torch.testing.assert_close(outputs["relu_strict"], scale, atol=0, rtol=0)
    torch.testing.assert_close(
        outputs[None], outputs["relu_strict"], atol=0, rtol=0, equal_nan=True
    )


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
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    block_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pure-torch FP8 reference (adapted from TRT-LLM unit test)."""
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device
    logits = torch.full(
        (B * next_n, max_seq_len), float("-inf"), device=device, dtype=out_dtype
    )
    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = int(seq_lens[b].item())
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].to(out_dtype)

        for blk_idx in range((ctx_len + block_size - 1) // block_size):
            phys_blk = int(block_tables[b, blk_idx].item())
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
            end = min(start + block_size, max_seq_len)
            if start >= max_seq_len:
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
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Pure-torch FP4 reference (adapted from TRT-LLM test)."""
    batch_size, next_n, num_heads, dim = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_seq_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    cl_list = seq_lens.tolist()
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
        w = min(total_len, max_seq_len)
        logits[i * next_n : (i + 1) * next_n, :w] = torch.where(
            k_offsets[None, :w] <= q_offsets[:, None], s[:, :w], float("-inf")
        )
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# Common: paged KV pool builder
# ──────────────────────────────────────────────────────────────────────────────


def _make_paged_kv(batch_size, block_size, seq_lens, device):
    n_blk_per_seq = (seq_lens + block_size - 1) // block_size
    # Natural width -- ceil(max ctx / block_size), the table a serving stack
    # keeps.  The kernels predicate every block-table read on the row's own
    # block count, so no compute-tile padding is needed; a shorter row's
    # trailing columns (zeros here) are never read.
    total = int(n_blk_per_seq.sum().item())
    num_total_blocks = total + batch_size * 2
    max_blk = int(n_blk_per_seq.max().item())
    block_tables = torch.zeros((batch_size, max_blk), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk_per_seq.tolist()):
        block_tables[i, :nb] = pool[off : off + nb]
        off += nb
    return block_tables, num_total_blocks


def _calc_cosine_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _valid_causal_mask(seq_lens, next_n, max_len, device):
    """Boolean [B*next_n, max_len] mask of within-length, causally-valid positions."""
    rows = seq_lens.shape[0] * next_n
    positions = torch.arange(max_len, device=device).unsqueeze(0).expand(rows, -1)
    offsets = torch.arange(rows, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    return positions <= limits


# ──────────────────────────────────────────────────────────────────────────────
# GPU schedule kernel tests
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("batch_size", [1, 4, 32, 128])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 32768])
def test_gpu_schedule_matches_cpu(batch_size, avg_ctx):
    """GPU schedule kernel must be bit-exact vs CPU numpy reference."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "GPU schedule kernel requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer.attn_scores.attn_scores import (
        _cached_num_sms,
        _compute_schedule_metadata,
        compute_paged_mqa_logits_schedule,
    )
    from flashinfer.utils import get_device_index

    device = "cuda"
    lo = max(128, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    seq_lens = torch.randint(lo, hi, (batch_size,), dtype=torch.int32, device=device)

    # Use the REAL device SM count so the CPU reference has the same [num_sms+1, 2]
    # shape as the GPU kernel output (hardcoding 148 breaks on non-148-SM devices).
    num_sms = _cached_num_sms(get_device_index(torch.device(device)))
    ref_cpu = _compute_schedule_metadata(seq_lens.cpu(), num_sms).to(device)
    gpu = compute_paged_mqa_logits_schedule(seq_lens, use_gpu_kernel=True)
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
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(42)
    num_heads, head_dim = 64, 128
    max_seq_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    seq_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_seq_len)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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

    # The call below keeps the default FP32 accumulator and epilogue. Only
    # the final store is converted, even when output_dtype is FP16.
    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        seq_lens,
        block_tables,
        max_seq_len,
        block_size,
        out_dtype=torch.float32,
    ).to(output_dtype)

    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=output_dtype,
    )

    # Mask out padding / beyond-length positions before comparing
    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
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


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("early_tmem_copy", [False, True])
@pytest.mark.parametrize("remove_kv_wait", [False, True])
def test_fp8_scale_reads_precede_stage_reuse(
    monkeypatch, num_heads, early_tmem_copy, remove_kv_wait
):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("FP8 paged MQA logits requires SM100a")

    from flashinfer.attn_scores import attn_scores as scores
    from flashinfer.attn_scores import kernels

    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    monkeypatch.setattr(scores, "_cached_num_sms", lambda _device: 4)
    monkeypatch.setattr(
        kernels,
        "FP8MQALogitsKernel",
        functools.partial(
            kernels.FP8MQALogitsKernel,
            early_tmem_copy=early_tmem_copy,
            remove_kv_wait_in_epilogue=remove_kv_wait,
        ),
    )
    scores._cached_compile_fp8_kernel.cache_clear()
    try:
        batch_size, next_n, head_dim, block_size, kv_rows = 7, 2, 128, 128, 32768
        device = "cuda"
        q = torch.ones((batch_size, next_n, num_heads, head_dim), device=device).to(
            torch.float8_e4m3fn
        )
        kv = torch.ones(
            (kv_rows // block_size, block_size, head_dim), device=device
        ).to(torch.float8_e4m3fn)
        # Distinct stage scales make premature TMA reuse observable in logits.
        scales = torch.pow(
            2.0, (torch.arange(kv_rows, device=device) // block_size) % 7 - 3
        ).reshape(-1, block_size)
        fused = _make_fused_kv_fp8(kv, scales, block_size, head_dim)
        weights = torch.ones((batch_size * next_n, num_heads), device=device)
        context_lens = kv_rows - 129 * torch.arange(
            batch_size, dtype=torch.int32, device=device
        )
        block_table = (
            torch.arange(kv_rows // block_size, dtype=torch.int32, device=device)[
                None, :
            ]
            + 17 * torch.arange(batch_size, dtype=torch.int32, device=device)[:, None]
        ) % (kv_rows // block_size)
        expected = (
            scales[block_table.long()].reshape(batch_size, kv_rows)
            * head_dim
            * num_heads
        ).repeat_interleave(next_n, dim=0)
        ends = (
            context_lens.repeat_interleave(next_n)
            - next_n
            + torch.arange(batch_size * next_n, device=device) % next_n
        )
        valid = torch.arange(kv_rows, device=device)[None, :] <= ends[:, None]
        out = torch.empty_like(expected)

        def run():
            return scores.fp8_paged_mqa_logits(
                q,
                fused,
                weights,
                block_table,
                context_lens,
                kv_rows,
                out=out,
            )

        run()
        torch.testing.assert_close(out[valid], expected[valid], rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(5):
            graph.replay()
            torch.testing.assert_close(out[valid], expected[valid], rtol=0, atol=0)
    finally:
        scores._cached_compile_fp8_kernel.cache_clear()


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
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(42)
    batch_size, avg_ctx, head_dim = 4, 1024, 64
    output_dtype = torch.float32
    max_seq_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    seq_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_seq_len)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
        seq_lens,
        block_tables,
        max_seq_len,
        block_size,
        out_dtype=output_dtype,
    )

    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
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
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(0)
    device = "cuda"
    num_heads, head_dim, block_size = 64, 128, 64
    max_seq_len = max(avg_ctx * 2, 2048)
    output_dtype = torch.float16

    seq_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
        seq_lens,
        block_tables,
        max_seq_len,
        block_size,
        out_dtype=output_dtype,
    )
    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=output_dtype,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
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
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(5)
    device = "cuda"
    num_heads, head_dim, block_size, next_n = 64, 128, 64, 4
    max_seq_len = max(avg_ctx * 2, 2048)

    seq_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
        seq_lens,
        block_tables,
        max_seq_len,
        block_size,
        out_dtype=torch.float32,
    )
    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)
    valid = (~neginf_mask) & finite
    if valid.any():
        torch.testing.assert_close(
            out_clean[valid], ref_clean[valid], atol=5e-5, rtol=1e-5
        )
    assert _calc_cosine_diff(out_clean, ref_clean) < 0.02


# (num_heads, next_n): the epilogue weight cache holds NUM_W_IN_REG * next_n
# elements. The per-slot cap stops binding once num_heads <= it, so these small
# num_heads shapes are the ones whose footprint is limited by the register
# budget instead. (32, 4) is the control -- it stays under the budget and keeps
# the unclamped value.
@pytest.mark.parametrize("num_heads,next_n", [(32, 6), (32, 8), (32, 4)])
def test_fp8_paged_mqa_logits_small_num_heads(num_heads, next_n):
    """num_heads below the per-slot weight-cache cap, where the budget binds."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(5)
    device = "cuda"
    head_dim, block_size, batch_size, avg_ctx = 128, 64, 4, 2048
    max_seq_len = max(avg_ctx * 2, 2048)

    seq_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
        seq_lens,
        block_tables,
        max_seq_len,
        block_size,
        out_dtype=torch.float32,
    )
    out = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)
    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    out_clean = out_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)
    valid = (~neginf_mask) & finite
    assert valid.any(), "no valid positions to compare"
    torch.testing.assert_close(out_clean[valid], ref_clean[valid], atol=5e-5, rtol=1e-5)
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
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(42)
    device = "cuda"
    num_heads, head_dim = 64, 128
    max_seq_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    seq_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_seq_len)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
    q_sf = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_sim = (
        _cast_back_from_fp4(q_packed, sf_q_packed, gran_k=32)
        .view(batch_size, next_n, num_heads, head_dim)
        .to(torch.bfloat16)
    )

    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)

    ref = _ref_fp4_paged_mqa_logits(
        q_sim.float(), kv_sim.float(), weights, seq_lens, block_tables, max_seq_len
    )

    out = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
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
    """Numerical check of next_n=4 against the torch reference.

    next_n=4 raw TMEM footprint is 544 columns.  Rubin (SM107+, 576) runs it as
    one atom; Blackwell (512) decomposes it into two atoms of 2.  The
    decomposition is a fixed internal rule keyed on the device, so running this
    test on both architectures covers both executions: direct on Rubin, the
    in-kernel split on Blackwell.  The staggered per-atom causal limits are the
    easy thing to get wrong, so both a short sequence (one KV block) and a
    multi-block one are covered.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp4_paged_mqa_logits

    device = "cuda"
    next_n = 4

    torch.manual_seed(42)
    num_heads, head_dim, block_size = 64, 128, 64
    output_dtype = torch.float32
    max_seq_len = max(avg_ctx * 2, 2048)

    lo = max(block_size, int(0.7 * avg_ctx))
    hi = int(1.3 * avg_ctx) + 1
    seq_lens = torch.randint(
        lo, hi, (batch_size,), dtype=torch.int32, device=device
    ).clamp(max=max_seq_len)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
    )

    q_f32 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    )

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_f32.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(
        batch_size, next_n, num_heads, head_dim // 2
    )
    q_sf = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_sim = (
        _cast_back_from_fp4(q_packed, sf_q_packed, gran_k=32)
        .view(batch_size, next_n, num_heads, head_dim)
        .to(torch.bfloat16)
    )
    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)

    ref = _ref_fp4_paged_mqa_logits(
        q_sim.float(), kv_sim.float(), weights, seq_lens, block_tables, max_seq_len
    )
    out = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=output_dtype,
    )

    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    out_m = out.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    # A non-finite kernel value where the reference is finite is a kernel
    # failure (uninitialized read, bad predication), never an accumulation
    # difference -- assert before masking so it cannot be hidden.  Exception:
    # fp16 outputs saturate at |logit| > 65504 by contract (the fp16 epilogue
    # trades range), so overflow-to-inf there is legitimate.
    if out.dtype != torch.float16:
        assert torch.isfinite(out_m[torch.isfinite(ref_m)]).all(), (
            "kernel produced non-finite logits at positions the reference keeps finite"
        )
    finite = torch.isfinite(out_m) & torch.isfinite(ref_m)
    valid = (~neginf_mask) & finite
    assert valid.any(), "test shape produced no comparable positions"
    torch.testing.assert_close(
        out_m.masked_fill(~finite, 0)[valid],
        ref_m.masked_fill(~finite, 0)[valid],
        atol=5e-5,
        rtol=1e-5,
    )


def test_fp4_next_n_limits():
    """FP4 accepts next_n up to 4; 5 is still rejected.

    next_n=4 needs 544 TMEM columns, over SM100's 512-column cap, so Rubin
    (SM107+, 576 columns) runs it directly while Blackwell reaches it as two
    atoms of 2.  The decomposition is now done *inside* the kernel, which is
    what makes it acceptable: an earlier caller-side version duplicated the
    block table and rebuilt the sequence lengths per call, and this test
    previously asserted next_n=4 was refused outright for that reason.

    The one hazard that remains is a caller-supplied schedule_meta: a split
    makes the scheduler describe batch*num_atoms rows while the schedule's
    shape is unchanged, so a schedule built from the native seq_lens would
    pass every always-on check and still be wrong.  Such a schedule is only
    accepted when built with this call's next_n and variant="fp4" (the helper
    applies the same decomposition); a default-args schedule is caught by the
    opt-in freshness check -- see test_fp4_next_n4_caller_schedule_via_helper.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp4_paged_mqa_logits

    device = "cuda"
    B, H, D, block_size, ctx = 2, 64, 128, 64, 256
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)

    for next_n, should_pass in ((3, True), (4, True), (5, False)):
        q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
        args = (q, q_sf, kv, w, block_tables, seq_lens, ctx)
        if should_pass:
            fp4_paged_mqa_logits(*args, output_dtype=torch.bfloat16)
        else:
            with pytest.raises(ValueError, match=r"next_n in 1\.\.4"):
                fp4_paged_mqa_logits(*args, output_dtype=torch.bfloat16)


def test_fp4_next_n4_caller_schedule_via_helper(monkeypatch):
    """A caller schedule built with matching helper arguments works everywhere.

    For next_n=4 on SM100/SM103 the kernel internally restructures the problem
    (the two-pass split), so the schedule differs from the native one;
    compute_paged_mqa_logits_schedule applies the same internal policy when
    given the call's next_n and variant.  Pins: (a) the helper output equals
    the schedule the kernel computes internally, (b) the main call accepts it,
    and (c) on a split device a schedule built with the DEFAULT helper
    arguments describes the wrong problem and the opt-in freshness check
    catches it -- the hang class commit 29ca0629 documented.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import compute_paged_mqa_logits_schedule, fp4_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import (
        _expand_seq_lens,
        _fp4_atom_decomposition,
        _fp4_max_atom_for_device,
    )

    device = "cuda"
    B, H, D, block_size, ctx, next_n = 2, 64, 128, 64, 256, 4
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)

    # (a) The helper with matching arguments reproduces the internal schedule.
    atom, num_atoms = _fp4_atom_decomposition(next_n, torch.device(device))
    expected = compute_paged_mqa_logits_schedule(
        _expand_seq_lens(seq_lens, num_atoms, atom), device=device
    )
    sched = compute_paged_mqa_logits_schedule(
        seq_lens, device=device, next_n=next_n, variant="fp4"
    )
    assert torch.equal(sched, expected), (
        "helper(next_n, variant) must equal the kernel's internal schedule"
    )

    # (b) The main call accepts it (on any architecture).
    res = fp4_paged_mqa_logits(
        q,
        q_sf,
        kv,
        w,
        block_tables,
        seq_lens,
        ctx,
        output_dtype=torch.bfloat16,
        schedule_meta=sched,
    )
    torch.cuda.synchronize()
    assert res.shape == (B * next_n, ctx)

    # (c) On a split device the native (default-args) schedule is a different
    # tensor, and the opt-in freshness check refuses it before the kernel
    # could hang on the mismatched end boundary.  The error's remediation
    # recipe must itself be actionable: run it (parsed, not eval'd) into the
    # rejected buffer and the very next call must be accepted.
    if _fp4_max_atom_for_device(torch.device(device)) < next_n:
        native = compute_paged_mqa_logits_schedule(seq_lens, device=device)
        assert not torch.equal(native, sched), "test premise: split changes it"
        monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
        with pytest.raises(ValueError, match="Recompute it with") as excinfo:
            fp4_paged_mqa_logits(
                q,
                q_sf,
                kv,
                w,
                block_tables,
                seq_lens,
                ctx,
                output_dtype=torch.bfloat16,
                schedule_meta=native,
            )
        recipe_nn, recipe_variant = _parse_schedule_recipe(str(excinfo.value))
        assert (recipe_nn, recipe_variant) == (next_n, "fp4"), str(excinfo.value)
        compute_paged_mqa_logits_schedule(
            seq_lens,
            device=device,
            next_n=recipe_nn,
            variant=recipe_variant,
            out=native,
        )
        assert torch.equal(native, sched), (
            "the recipe must reproduce the accepted schedule"
        )
        fp4_paged_mqa_logits(
            q,
            q_sf,
            kv,
            w,
            block_tables,
            seq_lens,
            ctx,
            output_dtype=torch.bfloat16,
            schedule_meta=native,
        )
        torch.cuda.synchronize()


def _parse_schedule_recipe(message: str):
    """Extract (next_n, variant) from the freshness error's recommended call.

    Parsed with a regex rather than eval'd so no code from an error string is
    ever executed by the test."""
    import re

    m = re.search(
        r"compute_paged_mqa_logits_schedule\(seq_lens, next_n=(\d+), "
        r"variant='(fp[48])', out=schedule_meta\)",
        message,
    )
    assert m, f"no actionable recipe in the message: {message}"
    return int(m.group(1)), m.group(2)


@pytest.mark.parametrize("variant,next_n", [("fp8", 2), ("fp4", 4)])
def test_stale_schedule_message_recipe_is_actionable(variant, next_n, monkeypatch):
    """Following the freshness error's recipe verbatim must fix the mismatch.

    Arch-independent stale trick (a schedule built for length 257 reused at
    256) so the message is exercised on every SM100-class device, including
    Rubin where the fp4 split is inactive; leg (c) of
    test_fp4_next_n4_caller_schedule_via_helper covers the split-specific
    default-args mismatch.  Against the pre-fix message ("...schedule(seq_lens,
    out=schedule_meta)") the recipe parse fails: it named neither next_n nor
    variant, and for the fp4 split it regenerated exactly the rejected schedule.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
    )

    device = "cuda"
    B, H, D, block_size = 1, 64, 128, 64
    seq_lens = torch.full((B,), 256, dtype=torch.int32, device=device)
    stale_lens = torch.full((B,), 257, dtype=torch.int32, device=device)
    max_ml = 512
    block_tables, ntb = _make_paged_kv(B, block_size, stale_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    if variant == "fp8":
        q = torch.zeros(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
        kv = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)

        def call(sched):
            return fp8_paged_mqa_logits(
                q, kv, w, block_tables, seq_lens, max_ml, schedule_meta=sched
            )
    else:
        q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )

        def call(sched):
            return fp4_paged_mqa_logits(
                q,
                q_sf,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                output_dtype=torch.bfloat16,
                schedule_meta=sched,
            )

    stale = compute_paged_mqa_logits_schedule(
        stale_lens, device=device, next_n=next_n, variant=variant
    )
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    with pytest.raises(ValueError, match="schedule_meta does not match seq_lens") as e:
        call(stale)
    msg = str(e.value)
    assert f"next_n={next_n}" in msg and f"variant='{variant}'" in msg, msg
    recipe_nn, recipe_variant = _parse_schedule_recipe(msg)
    compute_paged_mqa_logits_schedule(
        seq_lens, device=device, next_n=recipe_nn, variant=recipe_variant, out=stale
    )
    call(stale)  # the recipe made it fresh: accepted
    torch.cuda.synchronize()


def test_fp4_caller_schedule_does_not_expand_seq_lens(monkeypatch):
    """With a caller schedule, the fp4 body must not materialise the per-atom
    lengths: that is one dead elementwise kernel (and a [B*num_atoms] alloc)
    per call -- and under CUDA-graph capture a dead node replayed on every
    launch -- on the very path schedule_meta= exists to slim down.  The
    expansion happens lazily inside the opt-in freshness check instead.
    Asserted on call count, so it runs on every arch (identity or not)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
        padded_seq_len,
    )
    from flashinfer.attn_scores import attn_scores as A

    device = "cuda"
    B, H, D, block_size, ctx, next_n = 2, 64, 128, 64, 256, 4
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    out = torch.empty(
        (B * next_n, padded_seq_len(ctx)), dtype=torch.bfloat16, device=device
    )
    sched = compute_paged_mqa_logits_schedule(
        seq_lens, device=device, next_n=next_n, variant="fp4"
    )

    def call():
        return fp4_paged_mqa_logits(
            q,
            q_sf,
            kv,
            w,
            block_tables,
            seq_lens,
            ctx,
            output_dtype=torch.bfloat16,
            schedule_meta=sched,
            out=out,
        )

    call()
    torch.cuda.synchronize()  # warm JIT before counting
    ref = call().clone()
    torch.cuda.synchronize()

    calls = []
    orig = A._expand_seq_lens

    def spy(sl, n, a):
        calls.append((n, a))
        return orig(sl, n, a)

    monkeypatch.setattr(A, "_expand_seq_lens", spy)
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)

    call()
    torch.cuda.synchronize()
    assert calls == [], f"eager caller-schedule path expanded seq_lens: {calls}"

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    assert calls == [], f"expansion was captured into the graph: {calls}"
    g.replay()
    torch.cuda.synchronize()
    valid = _valid_causal_mask(seq_lens, next_n, ctx, device)
    assert torch.equal(out[:, :ctx][valid], ref[:, :ctx][valid])

    # With validation on, the expansion happens exactly once -- inside the
    # freshness check -- and the helper-built schedule is accepted.
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    call()
    torch.cuda.synchronize()
    assert len(calls) == 1, calls


@pytest.mark.parametrize("block_size", [64, 128])
def test_fp4_paged_mqa_logits_sf_interleaved(block_size):
    """is_kv_sf_interleaved=True path (requires block_size=128)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp4_paged_mqa_logits

    torch.manual_seed(3)
    device = "cuda"
    batch_size, next_n, num_heads, head_dim = 4, 2, 64, 128
    avg_ctx = 1024
    max_seq_len = avg_ctx * 2

    seq_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
    q_sf = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)

    # Pre-interleaved KV SF
    kv_fused_interleaved, kv_sim = _kv_cache_cast_to_fp4(
        kv_cache, is_kv_sf_interleaved=(block_size == 128)
    )

    out = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused_interleaved,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=torch.bfloat16,
        is_kv_sf_interleaved=(block_size == 128),
    )

    # Also run with the in-kernel rearrangement for cross-check
    kv_fused_online, _ = _kv_cache_cast_to_fp4(kv_cache, is_kv_sf_interleaved=False)
    out_online = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused_online,
        weights,
        block_tables,
        seq_lens,
        max_seq_len,
        output_dtype=torch.bfloat16,
        is_kv_sf_interleaved=False,
    )

    # Mask to valid (within-length, finite) positions before comparing
    positions = (
        torch.arange(max_seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (seq_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
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


def _make_fp8_case(
    batch_size, next_n, ctx, block_size, device, seed=11, head_dim=128, num_heads=64
):
    torch.manual_seed(seed)
    seq_lens = torch.full((batch_size,), ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(
        batch_size, block_size, seq_lens, device
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
    return q_fp8, kv_fused, weights, seq_lens, block_tables


# Shapes the UMMA N-mode rule (next_n*num_heads a multiple of 8 in [8, 256])
# admits although num_heads is not a multiple of the epilogue unroll (4).  Only
# possible when next_n itself is a multiple of 4.
@pytest.mark.parametrize("next_n,num_heads", [(4, 10), (8, 9)])
def test_fp8_num_heads_unroll_granularity_rejected(next_n, num_heads):
    """num_heads not a multiple of 4 is rejected at the API boundary, in the
    caller's vocabulary, before the compile layer is entered.

    Previously the public supportedness checker accepted these shapes and the
    kernel constructor rejected them -- after the output had been allocated
    and the schedule kernel compiled -- with a message naming the internal
    compile-layer parameter num_epi_subtiles, which the caller cannot set.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits
    from flashinfer.attn_scores import attn_scores as A

    N = next_n * num_heads
    assert N % 8 == 0 and 8 <= N <= 256 and num_heads % 4 != 0, "test premise"
    ctx, block_size = 256, 64
    q, kv, w, cl, bt = _make_fp8_case(
        2, next_n, ctx, block_size, "cuda", num_heads=num_heads
    )

    misses = A._cached_compile_fp8_kernel.cache_info().misses
    with pytest.raises(ValueError, match=r"num_heads must be a multiple of 4") as ei:
        fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx)
    assert "num_epi_subtiles" not in str(ei.value), "internal parameter leaked"
    assert A._cached_compile_fp8_kernel.cache_info().misses == misses, (
        "the compile layer was entered before the shape was rejected"
    )
    # The public checker is the single source of supportedness: same verdict.
    with pytest.raises(ValueError, match=r"num_heads must be a multiple of 4"):
        A._check_fp8_paged_mqa_logits_supported(q, kv, w, bt, cl, ctx)
    assert fp8_paged_mqa_logits.is_compute_capability_supported(100)


@pytest.mark.parametrize("epi_dtype", [torch.float32, torch.float16])
def test_fp8_epilogue_policy_is_bit_identical_and_keyed(epi_dtype, monkeypatch):
    """The internal per-shape epilogue policy only changes speed.

    At num_heads=32, next_n=6 the default weight-cache budget (160 registers,
    24 weights per slot) spills -- Nsight Compute: ~12k local loads and ~8k
    local stores per launch -- and the policy's 96-register budget (16 per
    slot) removes every spill, 8-15% faster on B200 and 11-35% on Rubin; the
    fp16 epilogue at this shape gains 5-13% from two epilogue subtiles.
    Neither knob changes the head accumulation order, so the policy's kernel
    and the default kernel must agree bit for bit, and they must be distinct
    compile-cache entries so a policy change can never serve a stale artifact.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits
    from flashinfer.attn_scores import attn_scores as A

    H, next_n, ctx, block_size = 32, 6, 2048, 64
    policy = A._fp8_epilogue_policy(H, next_n, A._to_cutlass(epi_dtype))
    assert policy != A._FP8_EPILOGUE_DEFAULT, "test premise: shape has a policy"
    q, kv, w, cl, bt = _make_fp8_case(4, next_n, ctx, block_size, "cuda", num_heads=H)
    kw = dict(epi_dtype=epi_dtype, output_dtype=torch.float32)

    out_policy = fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, **kw).clone()
    entries = A._cached_compile_fp8_kernel.cache_info().currsize
    monkeypatch.setattr(A, "_fp8_epilogue_policy", lambda *a: A._FP8_EPILOGUE_DEFAULT)
    out_default = fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, **kw)
    assert A._cached_compile_fp8_kernel.cache_info().currsize == entries + 1, (
        "the default and the policy kernel must be distinct cache entries"
    )
    torch.cuda.synchronize()
    valid = _valid_causal_mask(cl, next_n, ctx, "cuda")
    assert torch.isfinite(out_policy[valid]).all()
    assert torch.equal(out_policy, out_default), "policy changed the numerics"


def test_fp8_kernel_w_cache_regs_knob():
    """The kernel's weight-cache register budget defaults to the module
    constant, takes any positive override, and rejects the rest (host-only)."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer.attn_scores.kernels import FP8MQALogitsKernel
    from flashinfer.attn_scores.kernels import fp8_paged_mqa_logits as K

    def make(**kw):
        return FP8MQALogitsKernel(
            num_heads=32, next_n=6, phys_block_kv=64, arch="sm_100a", **kw
        )

    assert make().w_cache_regs == K._MAX_W_CACHE_REGS == 160
    assert make(w_cache_regs=96).w_cache_regs == 96
    with pytest.raises(ValueError, match="w_cache_regs"):
        make(w_cache_regs=0)


def test_fp8_out_and_schedule_meta_paths():
    """out= (pre-allocated buffer) and schedule_meta= (pre-computed) must match the
    default path bit-for-bit and the returned tensor must alias out."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import (
        padded_seq_len,
        compute_paged_mqa_logits_schedule,
        fp8_paged_mqa_logits,
    )

    B, next_n, ctx, block_size = 4, 2, 4096, 64
    max_ml = ctx + 512
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, block_size, device)

    ref = fp8_paged_mqa_logits(q, kv_fused, w, bt, cl, max_ml)

    # Pre-allocated out= sized via padded_seq_len, plus pre-computed schedule.
    padded = padded_seq_len(max_ml)
    out_buf = torch.empty((B * next_n, padded), device=device, dtype=torch.float32)
    sched = compute_paged_mqa_logits_schedule(cl)
    ret = fp8_paged_mqa_logits(
        q, kv_fused, w, bt, cl, max_ml, schedule_meta=sched, out=out_buf
    )

    assert ret.data_ptr() == out_buf.data_ptr(), "returned tensor must alias out="
    # Same kernel + same schedule → bit-identical in the causal-valid region.
    # (Beyond-length padding is unwritten garbage and differs between buffers.)
    valid = _valid_causal_mask(cl, next_n, max_ml, device)
    torch.testing.assert_close(ret.float()[valid], ref.float()[valid], atol=0, rtol=0)


def test_fp4_out_and_schedule_meta_paths():
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import (
        padded_seq_len,
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
    )

    torch.manual_seed(13)
    B, next_n, ctx, block_size = 4, 2, 4096, 64
    num_heads, head_dim = 64, 128
    max_ml = ctx + 512
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, num_total_blocks = _make_paged_kv(B, block_size, seq_lens, device)
    q_bf = torch.randn(
        B, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_total_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(B * next_n, num_heads, device=device, dtype=torch.float32)
    q_packed, sf_q_packed = _per_token_cast_to_fp4(q_bf.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(B, next_n, num_heads, head_dim // 2)
    q_sf = sf_q_packed.view(torch.int32).view(B, next_n, num_heads)
    kv_fused, _ = _kv_cache_cast_to_fp4(kv_cache)

    ref = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_ml,
        output_dtype=torch.float32,
    )
    padded = padded_seq_len(max_ml)
    out_buf = torch.empty((B * next_n, padded), device=device, dtype=torch.float32)
    sched = compute_paged_mqa_logits_schedule(seq_lens)
    ret = fp4_paged_mqa_logits(
        q_fp4,
        q_sf,
        kv_fused,
        weights,
        block_tables,
        seq_lens,
        max_ml,
        output_dtype=torch.float32,
        schedule_meta=sched,
        out=out_buf,
    )
    assert ret.data_ptr() == out_buf.data_ptr()
    valid = _valid_causal_mask(seq_lens, next_n, max_ml, device)
    torch.testing.assert_close(ret.float()[valid], ref.float()[valid], atol=0, rtol=0)


def test_fp8_input_validation():
    """Guard rails: bad out= size, unsupported dtype, and non-int32/CPU inputs must raise."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip(
            "FP8 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import padded_seq_len, fp8_paged_mqa_logits

    B, next_n, ctx, block_size = 2, 1, 1024, 64
    max_ml = ctx + 300  # deliberately not a 256 multiple so padded_max_seq_len > max_ml
    q, kv_fused, w, cl, bt = _make_fp8_case(B, next_n, ctx, block_size, device)

    # Undersized out= (max_ml columns instead of padded) → ValueError (prevents OOB)
    bad_out = torch.empty((B * next_n, max_ml), device=device, dtype=torch.float32)
    assert padded_seq_len(max_ml) > max_ml
    with pytest.raises(ValueError, match="padded_seq_len"):
        fp8_paged_mqa_logits(q, kv_fused, w, bt, cl, max_ml, out=bad_out)

    # Unsupported output dtype for FP8
    with pytest.raises(ValueError, match="float32, float16"):
        fp8_paged_mqa_logits(
            q, kv_fused, w, bt, cl, max_ml, output_dtype=torch.bfloat16
        )

    # CPU seq_lens → ValueError (kernel needs int32 CUDA)
    with pytest.raises(ValueError, match="seq_lens"):
        fp8_paged_mqa_logits(q, kv_fused, w, bt, cl.cpu(), max_ml)

    # int64 block_tables → ValueError
    with pytest.raises(ValueError, match="block_tables"):
        fp8_paged_mqa_logits(q, kv_fused, w, bt.to(torch.int64), cl, max_ml)

    # head_dim not a multiple of the MMA instruction K (32) → ValueError.
    # Without this guard the kernel's `head_dim // 32` integer division would
    # silently truncate the QK contraction (100 -> 96) and return wrong logits.
    q_bad, kv_bad, w_bad, cl_bad, bt_bad = _make_fp8_case(
        B, next_n, ctx, block_size, device, head_dim=100
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        fp8_paged_mqa_logits(q_bad, kv_bad, w_bad, bt_bad, cl_bad, max_ml)

    # head_dim large enough to exceed the per-CTA SMEM budget → ValueError.
    # Measured on sm_100a: head_dim=256 needs 249856 B vs a 232448 B cap, and
    # without this check the launch fails with a bare cudaErrorInvalidValue.
    q_big, kv_big, w_big, cl_big, bt_big = _make_fp8_case(
        B, next_n, ctx, block_size, device, head_dim=256
    )
    with pytest.raises(ValueError, match="shared memory"):
        fp8_paged_mqa_logits(q_big, kv_big, w_big, bt_big, cl_big, max_ml)

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
            fp8_paged_mqa_logits(q2, kv_reshaped, w2, bt2, cl2, max_ml)

    # next_n * num_heads outside the UMMA N-mode range [8, 256].
    # Measured: next_n=5 at num_heads=64 gives N=320 -> opaque DSL OpError.
    q3, kv3, w3, cl3, bt3 = _make_fp8_case(B, 5, ctx, block_size, device)
    with pytest.raises(ValueError, match="N-mode"):
        fp8_paged_mqa_logits(q3, kv3, w3, bt3, cl3, max_ml)


def test_fp4_head_dim_num_heads_validation():
    """FP4 hardcodes head_dim=128 / num_heads=64; the wrapper must say so clearly."""
    device = "cuda"
    if not is_sm100a_supported(torch.device(device)):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )

    from flashinfer import fp4_paged_mqa_logits

    B, next_n, num_heads, block_size, ctx = 2, 1, 64, 64, 512
    max_ml = 2048
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)

    def _case(head_dim, n_heads):
        half_d = head_dim // 2
        q = torch.zeros(B, next_n, n_heads, half_d, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(B, next_n, n_heads, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, half_d + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, n_heads, device=device, dtype=torch.float32)
        return q, q_sf, kv, w

    # Wrong head_dim (64 instead of 128)
    q, q_sf, kv, w = _case(64, num_heads)
    with pytest.raises(ValueError, match="requires head_dim == 128"):
        fp4_paged_mqa_logits(q, q_sf, kv, w, block_tables, seq_lens, max_ml)

    # Wrong num_heads (32 instead of 64)
    q, q_sf, kv, w = _case(128, 32)
    with pytest.raises(ValueError, match="requires num_heads == 64"):
        fp4_paged_mqa_logits(q, q_sf, kv, w, block_tables, seq_lens, max_ml)

    # next_n beyond what the kernel supports (1-4).
    q5 = torch.zeros(B, 5, num_heads, 64, dtype=torch.uint8, device=device)
    sf5 = torch.zeros(B, 5, num_heads, dtype=torch.int32, device=device)
    kv5 = torch.zeros(ntb, block_size, 1, 68, dtype=torch.uint8, device=device)
    w5 = torch.randn(B * 5, num_heads, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="next_n"):
        fp4_paged_mqa_logits(q5, sf5, kv5, w5, block_tables, seq_lens, max_ml)

    # block_size not a valid sub-tiling of the 128-token compute tile.
    q6, sf6, kv6, w6 = _case(128, num_heads)
    kv_bad = torch.zeros(ntb, 48, 1, 68, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="block_size"):
        fp4_paged_mqa_logits(q6, sf6, kv_bad, w6, block_tables, seq_lens, max_ml)


def test_block_table_width_contract(monkeypatch):
    """block_tables must be at least the natural width ceil(ctx / block_size).

    ctx=257 at block_size=64 occupies 5 blocks while the kernel walks 3
    128-token compute tiles spanning 6 table columns.  The kernel predicates
    each column on the row's own block count, so the natural 5-column table
    (what a serving stack emits) is accepted and runs, and only a table
    narrower than 5 is rejected.  The bound needs the per-row seq_lens from
    device memory, so the check is opt-in behind FLASHINFER_VALIDATE_INPUTS.
    Regression for PR #4365 review r3824399380 and the block-table width
    thread on PR #4737.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
        min_block_table_width,
    )
    from flashinfer.attn_scores.attn_scores import _validate_paged_bounds

    device = "cuda"
    B, H, D, block_size = 2, 64, 128, 64
    ctx = 257  # deliberately not a multiple of the 128-token compute tile
    natural = -(-ctx // block_size)  # 5 -- what a serving stack allocates
    tile_cols = -(-ctx // 128) * (128 // block_size)  # 6 -- what the tiles span
    assert (natural, tile_cols) == (5, 6), "test premise: ctx exposes the gap"
    assert min_block_table_width(ctx, block_size) == natural

    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    narrow = torch.zeros((B, natural - 1), dtype=torch.int32, device=device)
    table = torch.zeros((B, natural), dtype=torch.int32, device=device)

    # Default (unset): no sync, no check. Exercise the validator directly rather
    # than launching -- a narrow table would genuinely read out of bounds.
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    assert (
        _validate_paged_bounds(narrow, seq_lens, ctx, block_size, B * natural, "x")
        is None
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    ntb = B * natural
    w = torch.zeros(B * 1, H, device=device, dtype=torch.float32)

    q8 = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    with pytest.raises(
        ValueError, match=r"block_tables has 4 columns.*occupies 5 blocks"
    ):
        fp8_paged_mqa_logits(q8, kv8, w, narrow, seq_lens, ctx)
    fp8_paged_mqa_logits(q8, kv8, w, table, seq_lens, ctx)  # natural width runs

    q4 = torch.zeros(B, 1, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, 1, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    with pytest.raises(
        ValueError, match=r"block_tables has 4 columns.*occupies 5 blocks"
    ):
        fp4_paged_mqa_logits(
            q4, sf4, kv4, w, narrow, seq_lens, ctx, output_dtype=torch.bfloat16
        )
    fp4_paged_mqa_logits(
        q4, sf4, kv4, w, table, seq_lens, ctx, output_dtype=torch.bfloat16
    )

    # Lives in the API body, so skip_check=True must not bypass it.
    with pytest.raises(ValueError, match=r"block_tables has 4 columns"):
        fp8_paged_mqa_logits(q8, kv8, w, narrow, seq_lens, ctx, skip_check=True)

    # max_seq_len being much larger than ctx must NOT tighten the bound.
    fp8_paged_mqa_logits(q8, kv8, w, table, seq_lens, ctx * 8)


def _natural_width_serving_case(variant, device):
    """SGLang-shaped inputs for the block-table width contract.

    seq_lens = [129, 257, 385] at block_size=64 occupy [3, 5, 7] blocks, so the
    conventional table -- ceil(max(seq_lens) / block_size) = 7 columns, what a
    serving stack's metadata emits -- is [3, 7].  The kernel walks [2, 3, 4]
    128-token compute tiles spanning [4, 6, 8] columns: every row's last tile
    spans a column past the row's blocks, and the longest row is LAST so a
    tile-granular reader would run one entry past the 84-byte allocation
    (memcheck on the unpredicated kernel: "Invalid __global__ read of size 4
    ... 1 bytes after the nearest allocation ... of size 84 bytes").  Shorter
    rows' trailing entries are an out-of-pool sentinel, so any read of them
    is caught by the debug validator (which checks exactly the entries the
    kernel reads) or by memcheck.

    Returns ``(call, ref, seq_lens, next_n, max_seq_len)``; ``call(**kw)``
    runs the public API (float32 output) and ``ref`` is the torch reference.
    """
    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits

    torch.manual_seed(7)
    H, D, block_size, next_n = 64, 128, 64, 2
    seq_lens = torch.tensor([129, 257, 385], dtype=torch.int32, device=device)
    B, max_seq_len = seq_lens.numel(), 385
    lens = seq_lens.tolist()
    n_blk = [-(-c // block_size) for c in lens]  # [3, 5, 7]
    tile_cols = [-(-c // 128) * (128 // block_size) for c in lens]  # [4, 6, 8]
    assert (n_blk, tile_cols) == ([3, 5, 7], [4, 6, 8]), "test premise"
    width = max(n_blk)
    num_blocks = sum(n_blk)
    pool = torch.randperm(num_blocks, dtype=torch.int32, device=device)
    SENTINEL = 2**30  # far outside the pool; never read by the kernel
    block_tables = torch.full((B, width), SENTINEL, dtype=torch.int32, device=device)
    off = 0
    for r, nb in enumerate(n_blk):
        block_tables[r, :nb] = pool[off : off + nb]
        off += nb
    weights = torch.randn(B * next_n, H, device=device, dtype=torch.float32)

    if variant == "fp8":
        q = torch.randn(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
        kv_f32 = torch.randn(num_blocks, block_size, D, device=device)
        scale = _ceil_to_ue8m0_fp(
            kv_f32.abs().amax(-1, keepdim=True).clamp(1e-4) / 448.0
        ).squeeze(-1)
        kv_fp8 = (kv_f32 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        kv_fused = _make_fused_kv_fp8(kv_fp8, scale, block_size, D)
        ref = _ref_fp8_paged_mqa_logits(
            q, kv_fp8, scale, weights, seq_lens, block_tables, max_seq_len, block_size
        )

        def call(**kw):
            return fp8_paged_mqa_logits(
                q, kv_fused, weights, block_tables, seq_lens, max_seq_len, **kw
            )

    else:
        q_bf = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
        qp, sfp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
        q = qp.view(torch.uint8).view(B, next_n, H, D // 2)
        q_sf = sfp.view(torch.int32).view(B, next_n, H)
        q_sim = _cast_back_from_fp4(qp, sfp, gran_k=32).view(B, next_n, H, D)
        kv_cache = torch.randn(
            num_blocks, block_size, 1, D, device=device, dtype=torch.bfloat16
        )
        kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)
        ref = _ref_fp4_paged_mqa_logits(
            q_sim.float(), kv_sim.float(), weights, seq_lens, block_tables, max_seq_len
        )

        def call(**kw):
            return fp4_paged_mqa_logits(
                q,
                q_sf,
                kv_fused,
                weights,
                block_tables,
                seq_lens,
                max_seq_len,
                output_dtype=torch.float32,
                **kw,
            )

    return call, ref, seq_lens, next_n, max_seq_len


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_natural_width_block_table_serving_layout(variant, monkeypatch):
    """A block table of the conventional serving width ceil(seq_len / block_size)
    with lengths that are not multiples of the compute tile is accepted and
    computes correctly, eagerly with FLASHINFER_VALIDATE_INPUTS=1 and under
    CUDA-graph replay.

    SGLang's DSV4 metadata emits exactly this width (raw length 1028 ->
    seq_len 257 at block size 64 -> 5 entries); the unpredicated kernel read a
    sixth.  See _natural_width_serving_case for the shape.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import padded_seq_len

    device = "cuda"
    call, ref, seq_lens, next_n, max_seq_len = _natural_width_serving_case(
        variant, device
    )
    valid = _valid_causal_mask(seq_lens, next_n, max_seq_len, device)

    # Debug validation accepts the natural width and ignores the sentinel
    # entries, because it checks exactly the entries the kernel reads.
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    eager = call().clone()
    torch.cuda.synchronize()
    assert torch.isfinite(eager[valid]).all(), "non-finite logits in the valid region"
    torch.testing.assert_close(eager[valid], ref[valid].float(), atol=5e-5, rtol=1e-5)
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS")

    # The same natural-width table captured in a CUDA graph (no validation is
    # possible there, so the kernel's own predication is all that protects it).
    out = torch.empty(
        (seq_lens.numel() * next_n, padded_seq_len(max_seq_len)),
        dtype=torch.float32,
        device=device,
    )
    call(out=out)  # warm-up outside capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        res = call(out=out)
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(res[valid], eager[valid])


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_natural_width_block_table_memcheck(variant):
    """compute-sanitizer memcheck: the kernel reads nothing past a natural-width
    block table.  Before the per-column predication it over-read 4 bytes past
    the 84-byte allocation of the longest-last row in
    _natural_width_serving_case (and the read landed on the next row's entry
    for interior rows).

    Opt-in -- FLASHINFER_MEMCHECK_TESTS=1 with compute-sanitizer available --
    because it spawns the sanitizer, which not every CI image has, and a
    regression does not fail cleanly: memcheck turns the over-read into a
    launch failure whose teardown wedges until the timeout.  With the kernels
    cached a passing run takes about ten seconds.  Every tensor is its own
    cudaMalloc (PYTORCH_NO_CUDA_MEMORY_CACHING) so the over-read is visible at
    the allocation boundary.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")
    if os.environ.get("FLASHINFER_MEMCHECK_TESTS", "0") in ("0", ""):
        pytest.skip(
            "opt-in: set FLASHINFER_MEMCHECK_TESTS=1 (spawns compute-sanitizer)"
        )
    sanitizer = (
        shutil.which("compute-sanitizer") or "/usr/local/cuda/bin/compute-sanitizer"
    )
    if not os.path.exists(sanitizer):
        pytest.skip("compute-sanitizer not found")

    import flashinfer

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(flashinfer.__file__)))
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    code = (
        f"import sys, torch; sys.path.insert(0, {tests_dir!r}); "
        "from test_attn_scores import _natural_width_serving_case; "
        f"call, *_ = _natural_width_serving_case({variant!r}, 'cuda'); "
        "call(); torch.cuda.synchronize(); print('MEMCHECK_CASE_RAN')"
    )
    env = dict(os.environ)
    env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [
            sanitizer,
            "--tool",
            "memcheck",
            "--print-limit",
            "3",
            sys.executable,
            "-c",
            code,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    report = proc.stdout + proc.stderr
    assert "MEMCHECK_CASE_RAN" in report, report[-3000:]
    assert "ERROR SUMMARY: 0 errors" in report, report[-3000:]


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
@pytest.mark.parametrize("next_n", [3, 4])
def test_defined_output_region_fully_written_at_tile_boundary(variant, next_n):
    """Every cell the API defines is written at seq_len=257, one past a compute
    tile: row b*next_n + t through position seq_lens[b] - next_n + t.

    The boundary is where the fp4 next_n=4 split on SM100/SM103 differs from
    the direct path: a request's two leading rows run as an atom with context
    255, cover two 128-token tiles instead of three, and never write column
    256 -- which lies in their causal tail (unspecified by contract), not in
    their defined region.  A NaN prefill of out= makes an unwritten defined
    cell unmistakable, where torch.empty would hide it behind stale data.  The
    benchmark's finiteness self-check relies on exactly this contract.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits, padded_seq_len

    device = "cuda"
    B, H, D, block_size, ctx = 2, 64, 128, 64, 257
    torch.manual_seed(3)
    if variant == "fp8":
        q, kv, w, seq_lens, block_tables = _make_fp8_case(
            B, next_n, ctx, block_size, device
        )
        dtype = torch.float32

        def call(o):
            return fp8_paged_mqa_logits(q, kv, w, block_tables, seq_lens, ctx, out=o)

    else:
        seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
        block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
        w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
        q_bf = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
        qp, sfp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
        q = qp.view(torch.uint8).view(B, next_n, H, D // 2)
        q_sf = sfp.view(torch.int32).view(B, next_n, H)
        kv, _ = _kv_cache_cast_to_fp4(
            torch.randn(ntb, block_size, 1, D, device=device, dtype=torch.bfloat16)
        )
        dtype = torch.bfloat16

        def call(o):
            return fp4_paged_mqa_logits(
                q, q_sf, kv, w, block_tables, seq_lens, ctx, output_dtype=dtype, out=o
            )

    out = torch.full(
        (B * next_n, padded_seq_len(ctx)), float("nan"), dtype=dtype, device=device
    )
    res = call(out)
    torch.cuda.synchronize()
    defined = _valid_causal_mask(seq_lens, next_n, ctx, device)
    assert torch.isfinite(res.float()[defined]).all(), (
        "an API-defined output cell was left unwritten or is non-finite"
    )


def test_max_seq_len_bound(monkeypatch):
    """max_seq_len must be >= max(seq_lens), or the kernel writes OOB.

    The output row is sized from max_seq_len while the schedule is derived
    from seq_lens; seq_lens=[257] with max_seq_len=256 allocates 256
    columns but schedules splits reaching 512, and the store is unconditional.
    Checked under FLASHINFER_VALIDATE_INPUTS since the bound needs the per-row
    lengths from device memory. Regression for PR #4365 review r3824481310.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import _validate_paged_bounds

    device = "cuda"
    B, H, D, block_size = 1, 64, 128, 64
    ctx, max_ml = 257, 256  # the reviewer's example: schedule reaches 512

    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    width = -(-ctx // 128) * (128 // block_size)
    block_tables = torch.zeros((B, width), dtype=torch.int32, device=device)
    ntb = B * width
    w = torch.zeros(B * 1, H, device=device, dtype=torch.float32)

    # Default (unset): no sync, no check -- exercise the validator directly
    # rather than launching, which would genuinely write out of bounds.
    monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    assert (
        _validate_paged_bounds(block_tables, seq_lens, max_ml, block_size, ntb, "x")
        is None
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    q8 = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"max_seq_len \(256\) must be at least"):
        fp8_paged_mqa_logits(q8, kv8, w, block_tables, seq_lens, max_ml)

    q4 = torch.zeros(B, 1, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, 1, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"max_seq_len \(256\) must be at least"):
        fp4_paged_mqa_logits(
            q4,
            sf4,
            kv4,
            w,
            block_tables,
            seq_lens,
            max_ml,
            output_dtype=torch.bfloat16,
        )

    # skip_check=True must not bypass it (silent OOB write).
    with pytest.raises(ValueError, match=r"max_seq_len \(256\) must be at least"):
        fp8_paged_mqa_logits(
            q8, kv8, w, block_tables, seq_lens, max_ml, skip_check=True
        )

    # Raising max_seq_len to the real length makes it legal again.
    fp8_paged_mqa_logits(q8, kv8, w, block_tables, seq_lens, ctx)

    # Ragged lengths: only the longest row matters.
    ragged = torch.tensor([64, 257], dtype=torch.int32, device=device)
    bt2 = torch.zeros((2, width), dtype=torch.int32, device=device)
    q8b = torch.zeros(2, 1, H, D, device=device).to(torch.float8_e4m3fn)
    wb = torch.zeros(2, H, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"max\(seq_lens\) \(257\)"):
        fp8_paged_mqa_logits(q8b, kv8, wb, bt2, ragged, max_ml)


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_schedule_meta_graph_replay_across_split_boundary(variant):
    """A captured graph must recompute schedule_meta when lengths cross 256.

    The schedule is a function of the whole ceil(seq_lens/256) vector, so a
    sequence moving 256 <-> 257 changes it while every shape stays identical.
    Capture the recomputation into the same static buffer ahead of the kernel,
    then replay in both directions and compare against a freshly scheduled
    reference. Regression for PR #4365 review r3824580059.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        compute_paged_mqa_logits_schedule,
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
        padded_seq_len,
    )

    device = "cuda"
    torch.manual_seed(11)
    B, next_n, H, D, block_size = 2, 1, 64, 128, 64
    max_ml = 1024

    # static buffers: addresses fixed for the lifetime of the graph
    seq_lens = torch.full((B,), 256, dtype=torch.int32, device=device)
    sched = compute_paged_mqa_logits_schedule(seq_lens, device=device)
    block_tables, ntb = _make_paged_kv(
        B,
        block_size,
        torch.full((B,), max_ml, dtype=torch.int32, device=device),
        device,
    )
    out = torch.empty(
        (B * next_n, padded_seq_len(max_ml)),
        device=device,
        dtype=torch.float32 if variant == "fp8" else torch.bfloat16,
    )
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)

    if variant == "fp8":
        q = torch.randn(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
        kv_f32 = torch.randn(ntb, block_size, D, device=device)
        scale = _ceil_to_ue8m0_fp(
            kv_f32.abs().amax(-1, keepdim=True).clamp(1e-4) / 448.0
        ).squeeze(-1)
        kv = _make_fused_kv_fp8(
            (kv_f32 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale, block_size, D
        )

        def launch():
            compute_paged_mqa_logits_schedule(seq_lens, device=device, out=sched)
            return fp8_paged_mqa_logits(
                q,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                schedule_meta=sched,
                out=out,
            )
    else:
        q_bf = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
        qp, sfp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
        q = qp.view(torch.uint8).view(B, next_n, H, D // 2)
        q_sf = sfp.view(torch.int32).view(B, next_n, H)
        kv, _ = _kv_cache_cast_to_fp4(
            torch.randn(ntb, block_size, 1, D, device=device, dtype=torch.bfloat16)
        )

        def launch():
            compute_paged_mqa_logits_schedule(seq_lens, device=device, out=sched)
            return fp4_paged_mqa_logits(
                q,
                q_sf,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                output_dtype=torch.bfloat16,
                schedule_meta=sched,
                out=out,
            )

    # capture with the schedule recomputation INSIDE the graph
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        launch()
    torch.cuda.synchronize()

    # 256 -> 257 (grow across the boundary) and 257 -> 256 (shrink back)
    for new_len in (257, 256, 257, 256):
        seq_lens.fill_(new_len)
        g.replay()
        torch.cuda.synchronize()
        replayed = out[:, :new_len].clone()

        ref_sched = compute_paged_mqa_logits_schedule(seq_lens, device=device)
        ref_out = torch.empty_like(out)
        if variant == "fp8":
            fp8_paged_mqa_logits(
                q,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                schedule_meta=ref_sched,
                out=ref_out,
            )
        else:
            fp4_paged_mqa_logits(
                q,
                q_sf,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                output_dtype=torch.bfloat16,
                schedule_meta=ref_sched,
                out=ref_out,
            )
        torch.cuda.synchronize()
        assert torch.equal(replayed, ref_out[:, :new_len]), (
            f"{variant}: replay at ctx={new_len} disagrees with a fresh schedule"
        )


def test_stale_schedule_meta_detected(monkeypatch):
    """A stale schedule_meta is reported instead of hanging the kernel.

    Reusing a schedule built for a different ceil(seq_lens/256) vector makes
    the persistent kernel's exact-equality termination unreachable. Under
    FLASHINFER_VALIDATE_INPUTS the mismatch is caught before launch. Regression
    for PR #4365 review r3824550881.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import compute_paged_mqa_logits_schedule, fp8_paged_mqa_logits

    device = "cuda"
    B, H, D, block_size, max_ml = 1, 64, 128, 64, 1024
    width = -(-max_ml // 128) * (128 // block_size)
    block_tables = torch.zeros((B, width), dtype=torch.int32, device=device)
    ntb = B * width
    q = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    w = torch.zeros(B, H, device=device, dtype=torch.float32)

    cl_257 = torch.full((B,), 257, dtype=torch.int32, device=device)
    cl_256 = torch.full((B,), 256, dtype=torch.int32, device=device)
    stale = compute_paged_mqa_logits_schedule(cl_257, device=device)
    assert not torch.equal(
        stale, compute_paged_mqa_logits_schedule(cl_256, device=device)
    ), "test premise: 256 and 257 must schedule differently"

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    with pytest.raises(ValueError, match="schedule_meta does not match seq_lens"):
        fp8_paged_mqa_logits(
            q, kv, w, block_tables, cl_256, max_ml, schedule_meta=stale
        )

    # the matching schedule is accepted
    fp8_paged_mqa_logits(
        q,
        kv,
        w,
        block_tables,
        cl_256,
        max_ml,
        schedule_meta=compute_paged_mqa_logits_schedule(cl_256, device=device),
    )


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_oversized_out_returns_exact_rows(variant):
    """An oversized out= must return exactly batch_size*next_n rows.

    A shared max-batch buffer is the CUDA-graph pattern: one address-stable
    allocation reused across captures of different batch sizes. Returning all
    capacity rows would hand a consumer that derives its batch from
    scores.shape[0] both the wrong row count and stale values. Regression for
    PR #4365 review r3824692495.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
        padded_seq_len,
    )

    device = "cuda"
    torch.manual_seed(5)
    B, next_n, H, D, block_size, ctx = 2, 1, 64, 128, 64, 512
    MAX_B = 8  # buffer sized for the largest capture
    rows = B * next_n

    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    padded = padded_seq_len(ctx)
    dtype = torch.float32 if variant == "fp8" else torch.bfloat16
    SENTINEL = -12345.0

    if variant == "fp8":
        q = torch.randn(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
        kv_f32 = torch.randn(ntb, block_size, D, device=device)
        scale = _ceil_to_ue8m0_fp(
            kv_f32.abs().amax(-1, keepdim=True).clamp(1e-4) / 448.0
        ).squeeze(-1)
        kv = _make_fused_kv_fp8(
            (kv_f32 / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale, block_size, D
        )
        call = lambda o: fp8_paged_mqa_logits(  # noqa: E731
            q, kv, w, block_tables, seq_lens, ctx, out=o
        )
    else:
        q_bf = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
        qp, sfp = _per_token_cast_to_fp4(q_bf.view(-1, D), gran_k=32)
        q = qp.view(torch.uint8).view(B, next_n, H, D // 2)
        q_sf = sfp.view(torch.int32).view(B, next_n, H)
        kv, _ = _kv_cache_cast_to_fp4(
            torch.randn(ntb, block_size, 1, D, device=device, dtype=torch.bfloat16)
        )
        call = lambda o: fp4_paged_mqa_logits(  # noqa: E731
            q,
            q_sf,
            kv,
            w,
            block_tables,
            seq_lens,
            ctx,
            output_dtype=torch.bfloat16,
            out=o,
        )

    big = torch.full((MAX_B * next_n, padded), SENTINEL, device=device, dtype=dtype)
    res = call(big)

    assert res.shape == (rows, ctx), (
        f"expected exactly {rows} rows, got {tuple(res.shape)}"
    )
    assert res.data_ptr() == big.data_ptr(), "must stay a view of the caller's buffer"

    # rows past the real batch are left alone -- nothing stale is exposed
    assert torch.all(big[rows:] == SENTINEL), "surplus rows must not be written"

    # an exactly-sized buffer gives the same values
    exact = torch.empty((rows, padded), device=device, dtype=dtype)
    assert torch.equal(res, call(exact))


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
@pytest.mark.parametrize("preallocated_out", [False, True])
@pytest.mark.parametrize("next_n", [1, 4])
def test_empty_batch_returns_without_launching(variant, preallocated_out, next_n):
    """B == 0 returns an empty result without scheduling, compiling or launching.

    The persistent kernel's grid is num_sms regardless of batch size, and each
    CTA's prologue clamps with min(start_q, batch_size - 1) -- which is -1 for an
    empty batch, an out-of-bounds read of an empty mContextLens (reproduced as a
    CUDA illegal memory access before this guard existed).

    Both "did not schedule" and "did not compile" are checked directly rather
    than by patching:
      * a well-formed but content-garbage schedule_meta is passed.  Its shape/
        dtype/device pass the always-on buffer validation (which now runs even
        at B == 0, like out=), but launching with those garbage boundaries
        would hang the persistent kernel -- returning cleanly proves the
        schedule/launch path was skipped.
      * the compile entry points are functools-cached, so cache_info() moving
        would mean the compile path was entered.

    Regression for PR #4365 review r3824960307.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import (
        fp4_paged_mqa_logits,
        fp8_paged_mqa_logits,
        padded_seq_len,
    )
    from flashinfer.attn_scores.attn_scores import (
        _cached_compile_fp4_kernel,
        _cached_compile_fp8_kernel,
        _cached_num_sms,
    )

    device = "cuda"
    H, D, block_size, ctx = 64, 128, 64, 256

    seq_lens = torch.zeros((0,), dtype=torch.int32, device=device)
    block_tables = torch.zeros((0, 4), dtype=torch.int32, device=device)
    w = torch.zeros(0, H, device=device, dtype=torch.float32)
    dtype = torch.float32 if variant == "fp8" else torch.bfloat16

    # Well-formed shape/dtype/device (so it passes the always-on buffer
    # validation) but garbage contents: launching with it would hang, so a
    # clean return proves the schedule/launch path was skipped.
    num_sms = _cached_num_sms(0)
    bad_schedule = torch.full(
        (num_sms + 1, 2), 123_456_789, dtype=torch.int32, device=device
    )
    schedule_arg = bad_schedule

    out = None
    if preallocated_out:
        out = torch.empty((0, padded_seq_len(ctx)), device=device, dtype=dtype)

    compile_fn = (
        _cached_compile_fp8_kernel if variant == "fp8" else _cached_compile_fp4_kernel
    )
    before = compile_fn.cache_info()

    if variant == "fp8":
        q = torch.zeros(0, next_n, H, D, device=device).to(torch.float8_e4m3fn)
        kv = torch.zeros(4, block_size, 1, D + 4, dtype=torch.uint8, device=device)
        res = fp8_paged_mqa_logits(
            q,
            kv,
            w,
            block_tables,
            seq_lens,
            ctx,
            schedule_meta=schedule_arg,
            out=out,
        )
    else:
        q = torch.zeros(0, next_n, H, D // 2, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(0, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(4, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
        res = fp4_paged_mqa_logits(
            q,
            q_sf,
            kv,
            w,
            block_tables,
            seq_lens,
            ctx,
            output_dtype=torch.bfloat16,
            schedule_meta=schedule_arg,
            out=out,
        )
    torch.cuda.synchronize()

    assert res.shape == (0, ctx), f"expected (0, {ctx}), got {tuple(res.shape)}"
    assert res.dtype == dtype
    assert res.device.type == "cuda"
    if preallocated_out:
        assert res.data_ptr() == out.data_ptr(), "should be a view of out="

    after = compile_fn.cache_info()
    assert (after.hits, after.misses) == (before.hits, before.misses), (
        "the compile path was entered for an empty batch"
    )


def test_empty_batch_still_validates_out():
    """The empty-batch short-circuit sits after out= validation, not before.

    A malformed out= must still be rejected when B == 0; otherwise the guard
    would silently accept buffers it never checks.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits, padded_seq_len

    device = "cuda"
    H, D, block_size, ctx = 64, 128, 64, 256
    seq_lens = torch.zeros((0,), dtype=torch.int32, device=device)
    block_tables = torch.zeros((0, 4), dtype=torch.int32, device=device)
    w = torch.zeros(0, H, device=device, dtype=torch.float32)
    q = torch.zeros(0, 1, H, D, device=device).to(torch.float8_e4m3fn)
    kv = torch.zeros(4, block_size, 1, D + 4, dtype=torch.uint8, device=device)

    wrong_dtype = torch.empty(
        (0, padded_seq_len(ctx)), device=device, dtype=torch.float16
    )
    with pytest.raises(ValueError, match="out.dtype"):
        fp8_paged_mqa_logits(q, kv, w, block_tables, seq_lens, ctx, out=wrong_dtype)

    too_narrow = torch.empty((0, 8), device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="out must be at least"):
        fp8_paged_mqa_logits(q, kv, w, block_tables, seq_lens, ctx, out=too_narrow)

    # A malformed caller-supplied schedule_meta is likewise validated at B == 0:
    # a buffer that cannot be right at any batch size fails immediately.
    bad_schedule = torch.zeros((3, 2), dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="schedule_meta must have shape"):
        fp8_paged_mqa_logits(
            q, kv, w, block_tables, seq_lens, ctx, schedule_meta=bad_schedule
        )


def test_api_rejects_malformed_ranks_and_dtypes():
    """Wrong-rank q / kv_fused and non-uint8 kv_fused get curated errors.

    Before these guards a 3-D q died on the checker's tuple unpack ("not
    enough values to unpack"), naming neither the argument nor the fix, and a
    wrong-dtype kv_fused passed every curated check only to fail post-JIT at
    the FFI boundary with a bare dtype error.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits

    device = "cuda"
    H, D, block_size, ctx = 64, 128, 64, 256
    cl = torch.full((2,), ctx, dtype=torch.int32, device=device)
    bt = torch.zeros((2, 4), dtype=torch.int32, device=device)
    w = torch.zeros(2, H, device=device, dtype=torch.float32)
    kv = torch.zeros(8, block_size, 1, D + 4, dtype=torch.uint8, device=device)

    q3 = torch.zeros(2, H, D, device=device).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"q must be 4-D.*unsqueeze"):
        fp8_paged_mqa_logits(q3, kv, w, bt, cl, ctx)

    q4_fp4 = torch.zeros(2, H, D // 2, dtype=torch.uint8, device=device)
    sf = torch.zeros(2, 1, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(8, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match=r"q must be 4-D"):
        fp4_paged_mqa_logits(q4_fp4, sf, kv4, w, bt, cl, ctx)

    q = torch.zeros(2, 1, H, D, device=device).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="kv_fused must be 4-D"):
        fp8_paged_mqa_logits(q, kv.flatten(1), w, bt, cl, ctx)
    with pytest.raises(ValueError, match="kv_fused must be uint8"):
        fp8_paged_mqa_logits(
            q,
            torch.zeros(8, block_size, 1, D + 4, dtype=torch.float16, device=device),
            w,
            bt,
            cl,
            ctx,
        )

    # A column-strided table (inner stride != 1) is rejected with the fix, not
    # a bare FFI error; row-strided views are accepted (see
    # test_index_tensors_row_strided_views).
    bt_noncontig = torch.zeros((4, 2), dtype=torch.int32, device=device).t()
    with pytest.raises(ValueError, match="innermost stride must be 1"):
        fp8_paged_mqa_logits(q, kv, w, bt_noncontig, cl, ctx)


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
@pytest.mark.parametrize("offset", [0, 1])
def test_index_tensors_row_strided_views(variant, offset):
    """Row-strided block_tables and strided seq_lens views are accepted
    zero-copy and give bit-identical logits to the compact tensors.

    This is the serving shape SGLang produces for speculative verification: a
    table with one row per draft token, de-expanded for this API with
    block_tables[::next_n] (unit inner stride, row stride next_n*W) and
    seq_lens[::next_n] (stride next_n).  The kernels read both only through
    their layouts with scalar loads, so the API must not demand a copy the
    kernel does not need.  offset=1 shifts the view's base by W*4 = 24 bytes
    (not 16-byte aligned), pinning that no stronger alignment is assumed on
    the index tensors.  Against the old code the call raised 'must be
    contiguous' before any launch; if only the Python check were relaxed and
    the compiled declaration stayed compact, request 1 would read request 0's
    blocks and the bit-exact comparison would fail on its rows.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import min_block_table_width

    device = "cuda"
    torch.manual_seed(3)
    B, next_n, H, D, block_size, ctx = 2, 2, 64, 128, 64, 300
    seq_lens = torch.tensor([257, 300], dtype=torch.int32, device=device)
    W = min_block_table_width(ctx, block_size)  # 6 -> W*4 = 24 B row shift
    num_blocks = B * W
    bt = torch.arange(num_blocks, dtype=torch.int32, device=device).view(B, W)
    w = torch.rand(B * next_n, H, device=device)
    if variant == "fp8":
        kv_vals = torch.randn(num_blocks, block_size, D, device=device).to(
            torch.float8_e4m3fn
        )
        kv_scales = torch.rand(num_blocks, block_size, device=device) + 0.5
        kv = torch.cat(
            [
                kv_vals.view(torch.uint8).flatten(1),
                kv_scales.view(torch.uint8).flatten(1),
            ],
            dim=1,
        ).view(num_blocks, block_size, 1, D + 4)
        q = torch.randn(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)

        def call(bt_arg, sl_arg):
            return fp8_paged_mqa_logits(q, kv, w, bt_arg, sl_arg, ctx)
    else:
        kv_codes = torch.randint(
            0,
            256,
            (num_blocks, block_size * (D // 2)),
            dtype=torch.uint8,
            device=device,
        )
        kv_sf = torch.full(
            (num_blocks, block_size * 4), 0x7F, dtype=torch.uint8, device=device
        )
        kv = torch.cat([kv_codes, kv_sf], dim=1).view(
            num_blocks, block_size, 1, D // 2 + 4
        )
        q = torch.randint(
            0, 256, (B, next_n, H, D // 2), dtype=torch.uint8, device=device
        )
        q_sf = torch.full((B, next_n, H), 0x7F7F7F7F, dtype=torch.int32, device=device)

        def call(bt_arg, sl_arg):
            return fp4_paged_mqa_logits(
                q, q_sf, kv, w, bt_arg, sl_arg, ctx, output_dtype=torch.bfloat16
            )

    ref = call(bt, seq_lens).clone()
    torch.cuda.synchronize()

    bt_view = bt.repeat_interleave(next_n, dim=0)[offset::next_n]  # (B, W), (2W, 1)
    sl_view = seq_lens.repeat_interleave(next_n)[offset::next_n]  # (B,), stride 2
    assert bt_view.stride() == (next_n * W, 1) and not bt_view.is_contiguous()
    assert sl_view.stride() == (next_n,) and not sl_view.is_contiguous()
    assert torch.equal(bt_view, bt) and torch.equal(sl_view, seq_lens)

    got = call(bt_view, sl_view)
    torch.cuda.synchronize()
    valid = _valid_causal_mask(seq_lens, next_n, ctx, device)
    assert torch.equal(got[:, :ctx][valid], ref[:, :ctx][valid]), (
        "row-strided index views must be bit-identical to the compact tensors"
    )
    assert torch.isfinite(ref[:, :ctx][valid]).all()

    # A column-strided table (inner stride != 1) is still rejected: that IS a
    # kernel requirement (entries of a row are read as adjacent scalars).
    bt_cols = torch.zeros((B, 2 * W), dtype=torch.int32, device=device)[:, ::2]
    assert bt_cols.shape == (B, W) and bt_cols.stride(1) == 2
    with pytest.raises(ValueError, match="innermost stride must be 1"):
        call(bt_cols, seq_lens)


def test_out_layout_rejected():
    """A column-strided or row-overlapping out= view is rejected with the
    allocation recipe.

    All three passed device/dtype/shape validation.  The column-strided view
    reached the FFI binding, which either errored bare or silently misplaced
    stores.  The overlapping views -- an as_strided() pitch narrower than
    padded_seq_len, or the zero row stride of an expand()ed row -- ran to
    completion and returned silently wrong logits, later rows overwriting
    earlier ones (every store stays inside the view's storage, so nothing
    faults).
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits, padded_seq_len

    device = "cuda"
    H, D, block_size, ctx = 64, 128, 64, 256
    cl = torch.full((2,), ctx, dtype=torch.int32, device=device)
    bt = torch.zeros((2, 4), dtype=torch.int32, device=device)
    w = torch.zeros(2, H, device=device, dtype=torch.float32)
    kv = torch.zeros(8, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q = torch.zeros(2, 1, H, D, device=device).to(torch.float8_e4m3fn)
    padded = padded_seq_len(ctx)

    base = torch.empty((2, 2 * padded), device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="innermost stride must be 1"):
        fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, out=base[:, ::2])

    # Pitched: row 1 starts halfway through row 0's padded span.
    pitched = base.flatten().as_strided((2, padded), (padded // 2, 1))
    assert pitched.shape == (2, padded) and pitched.stride() == (padded // 2, 1)
    with pytest.raises(ValueError, match="rows overlap"):
        fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, out=pitched)

    # Zero row stride: both rows alias one row of storage.
    expanded = torch.empty((1, padded), device=device, dtype=torch.float32).expand(
        2, padded
    )
    assert expanded.stride(0) == 0
    with pytest.raises(ValueError, match="rows overlap"):
        fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, out=expanded)

    # A row slice of a wider buffer has a LARGER row stride and stays accepted:
    # the check must not tighten the contract beyond distinct rows.
    wide_rows = base[:, :padded]
    assert wide_rows.stride(0) == 2 * padded
    exact = torch.empty((2, padded), device=device, dtype=torch.float32)
    ref = fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, out=exact)
    got = fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx, out=wide_rows)
    assert got.data_ptr() == wide_rows.data_ptr() and torch.equal(got, ref)

    # A single row has nothing to collide with, whatever its stride(0) reads.
    fp8_paged_mqa_logits(
        q[:1], kv, w[:1], bt[:1], cl[:1], ctx, out=base.flatten()[:padded].view(1, -1)
    )


def test_inputs_foreign_device_rejected_paged():
    """Tensor arguments on a different GPU than q are rejected by name.

    is_cuda alone admits a tensor on another device; the kernel launched on
    q's device would dereference peer pointers -- silently right or wrong
    depending on peer access.  Needs two GPUs.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")

    from flashinfer import fp8_paged_mqa_logits

    device = "cuda:0"
    H, D, block_size, ctx = 64, 128, 64, 256
    cl_far = torch.full((2,), ctx, dtype=torch.int32, device="cuda:1")
    bt = torch.zeros((2, 4), dtype=torch.int32, device=device)
    w = torch.zeros(2, H, device=device, dtype=torch.float32)
    kv = torch.zeros(8, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q = torch.zeros(2, 1, H, D, device=device).to(torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=r"seq_lens is on cuda:1 but q is on"):
        fp8_paged_mqa_logits(q, kv, w, bt, cl_far, ctx)


# The schedule helper's three internal paths: the GPU schedule kernel, the fp4
# atom-split expansion in front of it, and the CPU numpy fallback.
_SCHEDULE_HELPER_PATHS = [
    pytest.param({}, id="fp8-gpu"),
    pytest.param({"next_n": 4, "variant": "fp4"}, id="fp4-split"),
    pytest.param({"use_gpu_kernel": False}, id="cpu-fallback"),
]


@pytest.mark.parametrize("kwargs", _SCHEDULE_HELPER_PATHS)
def test_schedule_helper_validates_inputs(kwargs):
    """compute_paged_mqa_logits_schedule validates seq_lens and out= up front,
    on every internal path.

    The paths failed differently on a malformed seq_lens, two of them
    silently: the GPU kernel path errored bare in the FFI binding on a [1, B]
    tensor; the fp4 split path reshaped [1, B] through the atom expansion into
    a wrong flat vector; and the CPU fallback read shape[0] == 1 and emitted a
    one-row schedule (a hang or unwritten rows in the persistent kernel, not
    a wrong number).  An int64 seq_lens died inside the DSL or was truncated
    by numpy, and a malformed out= reached the compiled kernel or out.copy_
    bare.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer import compute_paged_mqa_logits_schedule

    dev = torch.device("cuda", torch.cuda.current_device())
    cl64 = torch.full((2,), 256, dtype=torch.int64, device=dev)
    with pytest.raises(ValueError, match="seq_lens must be int32"):
        compute_paged_mqa_logits_schedule(cl64, **kwargs)

    # [1, B] has the same numel as [B]; every path used to accept it.
    cl_2d = torch.full((1, 2), 256, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match=r"seq_lens must be 1-D"):
        compute_paged_mqa_logits_schedule(cl_2d, **kwargs)

    cl = torch.full((2,), 256, dtype=torch.int32, device=dev)
    bad_out = torch.zeros((3, 2), dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="schedule_meta must have shape"):
        compute_paged_mqa_logits_schedule(cl, out=bad_out, **kwargs)

    # Guard the new checks against over-strictness on the same path: the
    # well-formed call succeeds, and a strided view of the same lengths gives
    # the identical schedule (layout is not a constraint).
    if is_sm100a_supported(dev) or kwargs.get("use_gpu_kernel", True) is False:
        ref = compute_paged_mqa_logits_schedule(cl, **kwargs)
        strided = cl.repeat_interleave(2)[::2]
        assert strided.stride(0) == 2, "test premise: a genuinely strided view"
        assert torch.equal(compute_paged_mqa_logits_schedule(strided, **kwargs), ref)


def test_schedule_helper_rejects_foreign_device_seq_lens():
    """seq_lens on another GPU than the target device is rejected by name on
    every path instead of reaching the FFI binding (or being silently copied
    across devices by the CPU fallback).  Needs two GPUs."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")

    from flashinfer import compute_paged_mqa_logits_schedule

    cl_far = torch.full((2,), 256, dtype=torch.int32, device="cuda:1")
    for kwargs in ({}, {"next_n": 4, "variant": "fp4"}, {"use_gpu_kernel": False}):
        with pytest.raises(ValueError, match=r"seq_lens is on cuda:1"):
            compute_paged_mqa_logits_schedule(cl_far, device="cuda:0", **kwargs)


def test_precompile_dsl_guards(monkeypatch):
    """precompile warns (not silently no-ops) without a DSL, and fails with
    the curated stale-toolchain message when the DSL cannot target the device.

    Previously it returned silently with no DSL (a deployment build step
    "succeeded" and serving JIT'd later), and went straight into cute.compile
    -- deep KeyError -- on a DSL predating the device.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.attn_scores import attn_scores as A

    monkeypatch.setattr(A, "_CUTE_DSL_AVAILABLE", False)
    with pytest.warns(RuntimeWarning, match="nothing was precompiled"):
        A.precompile_paged_mqa_logits()

    monkeypatch.setattr(A, "_CUTE_DSL_AVAILABLE", True)
    monkeypatch.setattr(A, "_cached_dsl_targets_device", lambda i: False)
    with pytest.raises(RuntimeError, match="cannot target"):
        A.precompile_paged_mqa_logits()


def test_validate_inputs_rejects_out_of_pool_block_index(monkeypatch):
    """FLASHINFER_VALIDATE_INPUTS=1 catches stale/out-of-pool block indices.

    The debug validator already pays the D2H sync for the width check; the
    value check rides along.  A stale block index is the classic silent OOB
    READ this validator exists to surface.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    device = "cuda"
    H, D, block_size, ctx = 64, 128, 64, 256
    cl = torch.full((1,), ctx, dtype=torch.int32, device=device)
    bt = torch.zeros((1, 4), dtype=torch.int32, device=device)
    bt[0, 1] = 99  # only 8 blocks in the pool
    w = torch.zeros(1, H, device=device, dtype=torch.float32)
    kv = torch.zeros(8, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q = torch.zeros(1, 1, H, D, device=device).to(torch.float8_e4m3fn)

    with pytest.raises(ValueError, match=r"only 8 blocks"):
        fp8_paged_mqa_logits(q, kv, w, bt, cl, ctx)


def test_output_span_32bit_guard():
    """FP8 store width is span-derived; FP4 uses an Int32 span limit."""
    from flashinfer.attn_scores.attn_scores import _fp8_output_index_bits

    assert _fp8_output_index_bits(2, 2**30 - 256, None, 4) == 32
    assert _fp8_output_index_bits(2, 2**30, None, 4) == 64
    assert _fp8_output_index_bits(8192, 1048576, None, 4) == 64
    assert _fp8_output_index_bits(0, 256, None, 4) == 32

    # Pin pitched-span arithmetic without allocating a multi-gigabyte buffer.
    from flashinfer.attn_scores.attn_scores import _validate_output_addressable

    arena_row = torch.empty((2, 4096), dtype=torch.float32)
    assert _fp8_output_index_bits(2, 256, arena_row, 4) == 32
    assert _fp8_output_index_bits(2**19, 256, arena_row, 4) == 64
    with pytest.raises(ValueError, match="32-bit"):
        # rows * stride(0) = 2^19 * 4096 = 2^31: the stride, not the padded
        # width, is what overflows.
        _validate_output_addressable(2**19, 256, arena_row, "fp4_paged_mqa_logits")
    for rows, columns in ((2**31, 256), (1, 2**31), (-1, 256), (1, -1)):
        with pytest.raises(ValueError, match="logical output"):
            _fp8_output_index_bits(rows, columns, None, 4)
    with pytest.raises(ValueError, match="byte span"):
        _fp8_output_index_bits(2**31 - 1, 2**31 - 1, None, 4)
    assert _fp8_output_index_bits(2**31 - 1, 2**31 - 1, None, 2) == 64


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
def test_fp8_output_offset_width_lowering(monkeypatch, index_bits, output_dtype):
    """Compile both offset widths on small planes with the existing FP32 oracle.

    Small output planes exercise both offset types without large allocations;
    accesses beyond the Int32 boundary are not exercised here.
    """
    from flashinfer.attn_scores import attn_scores as api

    actual_choice = api._fp8_output_index_bits

    def force_width(*args):
        assert actual_choice(*args) == 32
        return index_bits

    monkeypatch.setattr(api, "_fp8_output_index_bits", force_width)
    test_fp8_paged_mqa_logits(2, 2, 384, 128, output_dtype)
    if output_dtype == torch.float32:
        test_out_layout_rejected()


def test_fp4_split_graph_replay_across_split_boundary():
    """CUDA-graph replay of fp4 next_n=4 with the INTERNAL schedule.

    The only other graph test uses a caller schedule_meta, which the split
    rejects, so the split's capture machinery (the expanded per-atom schedule
    computed by a captured GPU kernel; the offsets cache that refuses capture
    population) had no replay coverage. Replays cross a 256-token schedule
    boundary in BOTH directions and must be bit-exact against eager -- this
    module's failure mode for schedule mismatches is a hang, so the guard
    matters. On SM100/SM103 this exercises the two-pass split; on Rubin the
    same test covers the direct path.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, padded_seq_len

    device = "cuda"
    torch.manual_seed(7)
    B, next_n, H, D, block_size = 2, 4, 64, 128, 64
    max_len = 512

    sizing = torch.full((B,), max_len, dtype=torch.int32, device=device)
    bt, ntb = _make_paged_kv(B, block_size, sizing, device)
    q_f32 = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
    q_packed, q_sf_packed = _per_token_cast_to_fp4(q_f32.view(-1, D), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(B, next_n, H, D // 2)
    q_sf = q_sf_packed.view(torch.int32).view(B, next_n, H)
    kv_cache = torch.randn(ntb, block_size, 1, D, device=device, dtype=torch.bfloat16)
    kv_fused, _ = _kv_cache_cast_to_fp4(kv_cache)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)

    seq_lens = torch.full((B,), 256, dtype=torch.int32, device=device)  # static
    out = torch.empty(
        (B * next_n, padded_seq_len(max_len)), device=device, dtype=torch.bfloat16
    )

    def call():
        return fp4_paged_mqa_logits(
            q_fp4, q_sf, kv_fused, w, bt, seq_lens, max_len, out=out
        )

    call()  # warm: kernel + schedule-bucket compiles
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()

    # 256 -> 257 crosses the boundary upward, 300 -> 256 back down.  Compare
    # bit-exact over the causal-valid region only: columns past each row's
    # limit are documented UNSPECIFIED scratch and legitimately differ between
    # the replayed buffer and a fresh eager allocation.
    for length in (256, 257, 300, 256, 512):
        seq_lens.fill_(length)
        g.replay()
        torch.cuda.synchronize()
        replay_out = out[: B * next_n, :max_len].clone()
        eager = fp4_paged_mqa_logits(q_fp4, q_sf, kv_fused, w, bt, seq_lens, max_len)
        torch.cuda.synchronize()
        valid = _valid_causal_mask(seq_lens, next_n, max_len, device)
        assert torch.equal(
            replay_out.masked_fill(~valid, 0), eager.masked_fill(~valid, 0)
        ), f"replay != eager at seq_len={length}"


def test_atom_offsets_cache_refuses_capture_population():
    """A cache miss during CUDA-graph capture must NOT populate _ATOM_OFFSETS_CACHE.

    A tensor allocated during capture comes from the graph's private memory
    pool; caching it would leave a dangling entry after the graph is freed,
    and later eager calls would read freed memory as per-atom lengths. The
    manual dict deliberately skips population under capture (validated by hand
    in commit 4c72476b); this pins it so a future functools.cache "cleanup"
    cannot reintroduce the use-after-free. Split-only machinery, so it runs
    where the atom split is active (SM100/SM103).
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, padded_seq_len
    from flashinfer.attn_scores import attn_scores as A

    if A._fp4_max_atom_for_device(torch.device("cuda")) >= 4:
        pytest.skip("direct next_n=4 path (Rubin): the offsets cache is unused")

    device = "cuda"
    torch.manual_seed(11)
    B, next_n, H, D, block_size = 2, 4, 64, 128, 64
    max_len = 256
    sizing = torch.full((B,), max_len, dtype=torch.int32, device=device)
    bt, ntb = _make_paged_kv(B, block_size, sizing, device)
    q_f32 = torch.randn(B, next_n, H, D, device=device, dtype=torch.bfloat16)
    q_packed, q_sf_packed = _per_token_cast_to_fp4(q_f32.view(-1, D), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(B, next_n, H, D // 2)
    q_sf = q_sf_packed.view(torch.int32).view(B, next_n, H)
    kv_cache = torch.randn(ntb, block_size, 1, D, device=device, dtype=torch.bfloat16)
    kv_fused, _ = _kv_cache_cast_to_fp4(kv_cache)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    seq_lens = torch.full((B,), max_len, dtype=torch.int32, device=device)
    out = torch.empty(
        (B * next_n, padded_seq_len(max_len)), device=device, dtype=torch.bfloat16
    )

    def call():
        return fp4_paged_mqa_logits(
            q_fp4, q_sf, kv_fused, w, bt, seq_lens, max_len, out=out
        )

    call()  # warm compiles (also populates the cache eagerly)
    torch.cuda.synchronize()

    A._ATOM_OFFSETS_CACHE.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()  # cache miss under capture: compute, do NOT retain
    assert len(A._ATOM_OFFSETS_CACHE) == 0, (
        "a graph-pool tensor was cached during capture (dangles after graph free)"
    )

    g.replay()
    torch.cuda.synchronize()
    replay_out = out[: B * next_n, :max_len].clone()

    del g
    torch.cuda.empty_cache()

    eager = fp4_paged_mqa_logits(q_fp4, q_sf, kv_fused, w, bt, seq_lens, max_len)
    torch.cuda.synchronize()
    assert torch.equal(replay_out, eager), "post-graph-free eager call corrupted"
    assert len(A._ATOM_OFFSETS_CACHE) == 1, "eager call should populate the cache"


def test_schedule_meta_device_must_match():
    """A schedule_meta on another GPU must be rejected here, not by the FFI layer.

    is_cuda only asserts "some CUDA device", so before this check a cuda:1
    schedule passed to a cuda:0 call reached the generated TVM-FFI binding and
    failed there with a much less actionable error. Regression for PR #4365
    review r3824973386.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")

    from flashinfer import compute_paged_mqa_logits_schedule, fp8_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import _cached_num_sms

    H, D, block_size, ctx = 64, 128, 64, 256
    B = 2
    dev0, dev1 = torch.device("cuda", 0), torch.device("cuda", 1)

    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=dev0)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, dev0)
    q = torch.zeros(B, 1, H, D, device=dev0).to(torch.float8_e4m3fn)
    kv = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=dev0)
    w = torch.zeros(B, H, device=dev0, dtype=torch.float32)

    # correctly shaped for device 0, but living on device 1
    n0 = _cached_num_sms(0)
    wrong_dev = torch.zeros((n0 + 1, 2), dtype=torch.int32, device=dev1)
    with pytest.raises(ValueError, match=r"schedule_meta.device .* must match"):
        fp8_paged_mqa_logits(
            q, kv, w, block_tables, seq_lens, ctx, schedule_meta=wrong_dev
        )

    # the matching device is still accepted
    right_dev = compute_paged_mqa_logits_schedule(seq_lens, device=dev0)
    fp8_paged_mqa_logits(q, kv, w, block_tables, seq_lens, ctx, schedule_meta=right_dev)


@pytest.mark.parametrize("variant", ["fp8", "fp4"])
def test_seq_lens_must_be_rank_1(variant):
    """A [B,1] seq_lens must be rejected here, not by the FFI binding.

    [B] and [B,1] have the same shape[0], so the old shape-only check accepted
    both and the rank-2 tensor failed later against the rank-1 compiled fake --
    after JIT compilation, with a message that never mentions rank. [B,1] is a
    real shape: SGLang's DeepGEMM path produces it.

    Matches on the message text on purpose: the old path also raised ValueError,
    so asserting the exception type alone would pass against unfixed code.
    Regression for PR #4365 review r3825144972.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, fp8_paged_mqa_logits

    device = "cuda"
    B, H, D, block_size, ctx = 2, 64, 128, 64, 256
    good = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, good, device)
    w = torch.zeros(B, H, device=device, dtype=torch.float32)

    if variant == "fp8":
        q = torch.zeros(B, 1, H, D, device=device).to(torch.float8_e4m3fn)
        kv = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
        call = lambda cl: fp8_paged_mqa_logits(  # noqa: E731
            q, kv, w, block_tables, cl, ctx
        )
    else:
        q = torch.zeros(B, 1, H, D // 2, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(B, 1, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )
        call = lambda cl: fp4_paged_mqa_logits(  # noqa: E731
            q, q_sf, kv, w, block_tables, cl, ctx, output_dtype=torch.bfloat16
        )

    for bad in (
        torch.full((B, 1), ctx, dtype=torch.int32, device=device),  # [B,1]
        torch.full((B, 1, 1), ctx, dtype=torch.int32, device=device),  # [B,1,1]
    ):
        with pytest.raises(ValueError, match="seq_lens must be 1-D"):
            call(bad)

    # rank 1 still works
    call(good)
    torch.cuda.synchronize()


def _to_cutlass_for_test(dtype):
    from flashinfer.attn_scores.attn_scores import _to_cutlass

    return _to_cutlass(dtype)


def test_gpu_arch_resolves_from_requested_device():
    """The compile arch must follow the requested device, not the current one.

    CUTLASS's own probe queries ordinal 0 unconditionally and PyTorch's
    no-argument get_device_capability() follows the process-wide current device,
    so before this change num_sms came from the device= argument while the
    codegen target and cache tag came from elsewhere. Regression for PR #4365
    review r3824968121.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer.attn_scores.attn_scores import _cached_gpu_arch

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        got = _cached_gpu_arch(i)
        assert got.startswith(f"sm_{p.major}"), (
            f"cuda:{i} is sm_{p.major}{p.minor} but arch resolved to {got}"
        )

    # ...and it must not drift with the process-wide current device.
    if torch.cuda.device_count() > 1:
        before = [_cached_gpu_arch(i) for i in range(torch.cuda.device_count())]
        cur = torch.cuda.current_device()
        try:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                after = [_cached_gpu_arch(j) for j in range(torch.cuda.device_count())]
                assert after == before, (
                    f"arch resolution changed when current device was {i}: "
                    f"{before} -> {after}"
                )
        finally:
            torch.cuda.set_device(cur)


def test_arch_backstop_rejects_target_it_cannot_run(monkeypatch):
    """The arch backstop must still reject a target the current device can't run.

    This is defence in depth, not the primary mechanism. The launch paths run
    inside ``_on_device``, which makes the target current before compiling and
    launching, so on those paths the arch matches by construction and this check
    is unreachable. It stays because the failure it prevents is silent and
    deferred: CuTe-DSL does not reject a --gpu-arch the current device cannot
    execute -- it compiles, exports, reloads, and raises only on first
    invocation (measured on sm_100a for sm_90a, sm_103a and sm_80), and
    load_module performs no architecture check either. A future caller that
    compiles for a device without entering it would hit exactly that.

    The arch lookup is stubbed so this runs on a single-GPU host. Real
    multi-GPU behaviour -- including that a non-current target now *works*
    rather than raising -- is covered on a heterogeneous host. Regression for
    PR #4365 review r3824968121.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.attn_scores import attn_scores as A

    cur = torch.cuda.current_device()
    other = cur + 1  # need not exist: only the arch lookup is consulted

    # Same device: must never raise, regardless of what the lookup returns.
    monkeypatch.setattr(A, "_cached_gpu_arch", lambda i: "sm_100a")
    assert A._arch_for_launch(cur, "fp8_paged_mqa_logits") == "sm_100a"

    # Different device, same arch: allowed here (a same-arch device mismatch is
    # a stream-placement concern, deliberately out of this check's scope).
    assert A._arch_for_launch(other, "fp8_paged_mqa_logits") == "sm_100a"

    # Different device, different arch: rejected, and the message must name
    # both devices and both archs so the fix is obvious.
    monkeypatch.setattr(
        A, "_cached_gpu_arch", lambda i: "sm_100a" if i == other else "sm_90a"
    )
    with pytest.raises(ValueError) as e:
        A._arch_for_launch(other, "fp8_paged_mqa_logits")
    msg = str(e.value)
    assert "fp8_paged_mqa_logits" in msg
    assert "sm_100a" in msg and "sm_90a" in msg
    assert f"cuda:{other}" in msg and f"cuda:{cur}" in msg
    assert "torch.cuda.device" in msg, "the error should say how to fix it"


def test_kernel_arch_gates_follow_passed_arch():
    """Construction-time arch gates must follow the arch argument, never the
    DSL's ambient (device-0 / CUTE_DSL_ARCH) probe.

    The compile target is already threaded from the input tensor's device
    (``_arch_for_launch`` -> ``--gpu-arch``), but the kernels also make
    decisions at construction time -- the FP4 atom cap and the per-arch perf
    levers -- and ``__init__`` runs before ``cute.compile`` installs the
    target, where a DSL arch query silently reads the ambient arch. On a
    heterogeneous node (e.g. cuda:0 Blackwell + cuda:1 Rubin) that crashed
    valid next_n=4 calls with a bare AssertionError and baked the wrong
    arch's levers into artifacts cached under the target's tag. Construction
    is pure host code, so both targets are checked from whatever single GPU
    runs this test.
    """
    pytest.importorskip("cutlass")

    from flashinfer.attn_scores.kernels import FP4MQALogitsKernel, FP8MQALogitsKernel

    # FP4 atom cap: a Blackwell target rejects atom=4 and gates its levers per
    # Blackwell policy, regardless of which GPU (if any) is ambient. No DSL
    # arch table is consulted, so this works on any installed DSL release.
    with pytest.raises(AssertionError, match="requires Rubin"):
        FP4MQALogitsKernel(next_n=4, num_next_n_atoms=1, arch="sm_100a")
    k4s = FP4MQALogitsKernel(next_n=4, num_next_n_atoms=2, arch="sm_100a")
    assert k4s.next_n_atom == 2
    assert k4s.use_flat_logits_view is True  # Blackwell auto-gate: lever on
    assert k4s.use_two_level_task_loop is True
    # atom=3: both levers off everywhere (B200 register-pressure gate).
    k43 = FP4MQALogitsKernel(next_n=3, num_next_n_atoms=1, arch="sm_100a")
    assert k43.use_flat_logits_view is False
    assert k43.use_two_level_task_loop is False

    # Rubin target: atom=4 constructs directly and the levers flip to the
    # Rubin policy. FP4 construction consults no DSL arch table; FP8 does
    # (get_max_tmem_alloc_cols), so its Rubin half is gated on the installed
    # DSL knowing the part.
    k4 = FP4MQALogitsKernel(next_n=4, num_next_n_atoms=1, arch="sm_107a")
    assert k4.next_n_atom == 4 and k4.N == 256
    assert k4.use_flat_logits_view is False  # Rubin auto-gate: lever off

    k8b = FP8MQALogitsKernel(next_n=4, arch="sm_100a")
    assert k8b.use_flat_logits_view is True
    assert k8b.use_two_level_task_loop is True
    assert k8b.use_paired_pipeline_polls is True

    from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported

    if is_cute_dsl_arch_supported(10, 7, native_only=True):
        k8r = FP8MQALogitsKernel(next_n=4, arch="sm_107a")
        assert k8r.use_flat_logits_view is False
        assert k8r.use_two_level_task_loop is False
        assert k8r.use_paired_pipeline_polls is False


def test_on_device_enters_target_and_is_free_when_already_current():
    """The launch paths must make the target device current before launching.

    The kernels take the TVM-FFI environment stream, which resolves to the
    current stream of the *current* device, so launching with tensors from
    another device onto that stream is wrong regardless of architecture -- the
    case raised in PR #4365 review r3824968121. ``_on_device`` fixes it by
    entering the target, and must cost nothing when the target is already
    current, which is every single-GPU call.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    import contextlib

    from flashinfer.attn_scores.attn_scores import _on_device

    cur = torch.cuda.current_device()

    # Already current: must be a no-op context, not a device switch.
    ctx = _on_device(cur)
    assert isinstance(ctx, contextlib.nullcontext), (
        "entering the already-current device should cost nothing; "
        f"got {type(ctx).__name__}"
    )
    with ctx:
        assert torch.cuda.current_device() == cur

    # A different device: must actually switch, and restore afterwards.
    if torch.cuda.device_count() > 1:
        other = (cur + 1) % torch.cuda.device_count()
        with _on_device(other):
            assert torch.cuda.current_device() == other, (
                "the target device was not made current for the launch"
            )
        assert torch.cuda.current_device() == cur, (
            "the previous device was not restored"
        )


def test_arch_is_part_of_the_compile_cache_key():
    """Two architectures must not share a compiled-kernel cache entry.

    Without arch in the key, a kernel built for one device could be handed to
    another with the same SM count. The annotation check also guards against
    keying on cute.GPUArch, whose equality is identity-based (GPUArch(x) ==
    GPUArch(x) is False) and would silently miss every lookup.
    """
    import inspect

    from flashinfer.attn_scores.attn_scores import (
        _cached_compile_fp4_kernel,
        _cached_compile_fp8_kernel,
    )
    from flashinfer.attn_scores.kernels.schedule_kernel import _compile_schedule_kernel

    for fn in (
        _cached_compile_fp8_kernel,
        _cached_compile_fp4_kernel,
        _compile_schedule_kernel,
    ):
        inner = getattr(fn, "__wrapped__", fn)
        sig = inspect.signature(inner)
        assert "arch" in sig.parameters, (
            f"{inner.__name__} cache key omits arch: {list(sig.parameters)}"
        )
        assert sig.parameters["arch"].annotation is str, (
            f"{inner.__name__} must key on the arch string, not a cute.GPUArch"
        )


def test_precompile_targets_the_requested_device():
    """precompile(device=...) builds entries keyed on that device's arch/SM count.

    Verified by a cache *hit*: after precompiling, asking for the same config
    with the target's (num_sms, arch) must not recompile. A different arch must
    miss, which is what proves arch is really in the key.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import precompile_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import (
        _COMPUTE_BLOCK_KV,
        _cached_compile_fp8_kernel,
        _cached_gpu_arch,
        _cached_num_sms,
    )

    # Pick the highest-numbered device that can actually run this kernel: on a
    # heterogeneous box the last device may be another architecture entirely,
    # and precompiling an sm100a kernel for it correctly raises.
    capable = [
        i
        for i in range(torch.cuda.device_count())
        if is_sm100a_supported(torch.device("cuda", i))
    ]
    assert capable, "no sm100a-capable device"
    target = capable[-1]
    arch = _cached_gpu_arch(target)
    sms = _cached_num_sms(target)
    f32 = _to_cutlass_for_test(torch.float32)

    precompile_paged_mqa_logits(device=torch.device("cuda", target), variants=("fp8",))

    hits = _cached_compile_fp8_kernel.cache_info().hits
    # Same positional call shape as precompile and the launch path:
    # functools.cache keys on the literal arguments, defaults included.
    shape = (64, 64, 128, 1, sms, f32, f32, f32, 1, None)
    tail = (32, "relu_strict", _COMPUTE_BLOCK_KV, False, None, None)
    _cached_compile_fp8_kernel(*shape, arch, *tail)
    assert _cached_compile_fp8_kernel.cache_info().hits == hits + 1, (
        f"precompile(device=cuda:{target}) produced no entry for "
        f"(num_sms={sms}, arch={arch})"
    )

    # A different arch must be a distinct key, not a silent reuse.
    misses = _cached_compile_fp8_kernel.cache_info().misses
    # Compiling this kernel for sm_80 legitimately fails (no tcgen05); the cache
    # miss is what we are asserting, not the compile result.
    with contextlib.suppress(Exception):
        other_arch = "sm_90a" if arch != "sm_90a" else "sm_80"
        _cached_compile_fp8_kernel(*shape, other_arch, *tail)
    assert _cached_compile_fp8_kernel.cache_info().misses > misses, (
        "a different arch reused the cache entry -- arch is not in the key"
    )


def test_precompile_warms_fp4_float32_output():
    """precompile must cover the output dtypes a deployment actually runs.

    output_dtype is part of the compile cache key, so warming only the API
    default (bfloat16) left a first-request JIT for any consumer binding logits
    as C float. Asserted via cache hits, with an unwarmed dtype required to
    miss -- otherwise the test would pass even if nothing were warmed.
    Regression for PR #4365 review r3825087342.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import precompile_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import (
        _cached_compile_fp4_kernel,
        _cached_gpu_arch,
        _cached_num_sms,
        _to_cutlass,
    )

    dev = torch.cuda.current_device()
    arch = _cached_gpu_arch(dev)
    sms = _cached_num_sms(dev)
    f32 = _to_cutlass(torch.float32)

    # The in-process cache is shared across the whole session, and other tests
    # in this file compile fp4 with every supported output dtype. Without
    # clearing it, "float16 must miss" below is decided by test ordering rather
    # than by what precompile warmed. Clearing only drops the in-memory entries;
    # the on-disk CuTe-DSL cache still serves them, so repopulation is a load.
    _cached_compile_fp4_kernel.cache_clear()

    precompile_paged_mqa_logits(variants=("fp4",))

    # both the API default and float32 must now be hits
    for out_dtype in (torch.bfloat16, torch.float32):
        hits = _cached_compile_fp4_kernel.cache_info().hits
        _cached_compile_fp4_kernel(
            64, 64, 128, 1, sms, f32, _to_cutlass(out_dtype), 1, False, arch, 1
        )
        assert _cached_compile_fp4_kernel.cache_info().hits == hits + 1, (
            f"precompile did not warm fp4 output_dtype={out_dtype}"
        )

    # a dtype outside the warmed set must miss, or the assertions above are vacuous
    misses = _cached_compile_fp4_kernel.cache_info().misses
    _cached_compile_fp4_kernel(
        64, 64, 128, 1, sms, f32, _to_cutlass(torch.float16), 1, False, arch, 1
    )
    assert _cached_compile_fp4_kernel.cache_info().misses > misses, (
        "float16 was not warmed but did not miss -- the cache key ignores "
        "output_dtype, so the hit assertions above prove nothing"
    )


def test_precompile_warms_fp4_split_schedule_bucket():
    """precompile(batch_sizes=[B]) must warm the schedule bucket fp4 next_n=4
    actually requests: 2B rows on SM100/SM103 (two atoms), B rows on SM107.

    batch_sizes is in caller units but the schedule kernel is keyed on the
    scheduler row count, and only the launch path applied the atom
    decomposition -- so before the fix every B > 16 missed on a split device
    (bucket(B) warmed, bucket(2B) requested) and the first fp4 next_n=4 call,
    or the first CUDA-graph capture, compiled inside the hot path.  Pins:
    (a) the runtime bucket is an in-process hit, (b) the unsplit bucket stays
    warm (fp8 and fp4 next_n<=3 request it), (c) an end-to-end fp4 next_n=4
    call adds no miss, (d) an unwarmed bucket still misses so (a)-(c) are not
    vacuous.  Fails against the unfixed code at (a) on SM100/SM103 and at the
    import of the (then missing) shared derivation everywhere.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp4_paged_mqa_logits, precompile_paged_mqa_logits
    from flashinfer.attn_scores.attn_scores import (
        _SPLIT_KV,
        _cached_gpu_arch,
        _cached_num_sms,
        _fp4_atom_decomposition,
        _precompile_schedule_buckets,
        _schedule_bucket,
    )
    from flashinfer.attn_scores.kernels.schedule_kernel import _compile_schedule_kernel

    dev = torch.cuda.current_device()
    device = torch.device("cuda", dev)
    arch, sms = _cached_gpu_arch(dev), _cached_num_sms(dev)
    B = 32
    _, num_atoms = _fp4_atom_decomposition(4, device)
    expected_bucket = _schedule_bucket(B * num_atoms)  # 64 on SM100/SM103, 32 on SM107

    # The derivation itself, GPU-free: bucket(B) is always present; the split
    # adds bucket(B*num_atoms); fp8 alone never expands.
    assert _precompile_schedule_buckets([B], ("fp4",), device) == {32, expected_bucket}
    assert _precompile_schedule_buckets([B], ("fp8",), device) == {32}
    assert _precompile_schedule_buckets([1, 40], ("fp8", "fp4"), device) == (
        {32, 64} if num_atoms == 1 else {32, 64, 96}
    )

    # In-process cache is session-wide; clear it so hits below are decided by
    # what precompile warmed, not by test order (disk cache still serves).
    _compile_schedule_kernel.cache_clear()
    precompile_paged_mqa_logits(
        device=device,
        variants=("fp4",),
        output_dtypes=(torch.bfloat16,),
        batch_sizes=[B],
    )
    # (a) the bucket the runtime will request is warm
    hits = _compile_schedule_kernel.cache_info().hits
    _compile_schedule_kernel(expected_bucket, _SPLIT_KV, sms, arch)
    assert _compile_schedule_kernel.cache_info().hits == hits + 1, (
        f"precompile(batch_sizes=[{B}]) did not warm schedule bucket "
        f"{expected_bucket} (fp4 next_n=4 schedules {B}*{num_atoms} rows here)"
    )
    # (b) the unsplit bucket is warm too
    hits = _compile_schedule_kernel.cache_info().hits
    _compile_schedule_kernel(32, _SPLIT_KV, sms, arch)
    assert _compile_schedule_kernel.cache_info().hits == hits + 1

    # (c) a real fp4 next_n=4 call at B=32 must not enter the compile path
    ctx, block_size, H, D, next_n = 512, 64, 64, 128, 4
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    misses = _compile_schedule_kernel.cache_info().misses
    fp4_paged_mqa_logits(
        q, q_sf, kv, w, block_tables, seq_lens, ctx, output_dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    assert _compile_schedule_kernel.cache_info().misses == misses, (
        "fp4 next_n=4 at B=32 compiled a schedule bucket after precompile"
    )
    # (d) negative control: an unwarmed bucket misses
    misses = _compile_schedule_kernel.cache_info().misses
    _compile_schedule_kernel(expected_bucket + 128, _SPLIT_KV, sms, arch)
    assert _compile_schedule_kernel.cache_info().misses == misses + 1


def test_precompile_output_dtypes_is_honoured_and_validated():
    """An explicit output_dtypes tuple builds only those, and is checked up front."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import precompile_paged_mqa_logits

    # fp8 does not support bfloat16; reject before compiling anything
    with pytest.raises(ValueError, match="does not support output dtype"):
        precompile_paged_mqa_logits(variants=("fp8",), output_dtypes=(torch.bfloat16,))

    # fp4 does support it, so the same tuple is fine there
    precompile_paged_mqa_logits(variants=("fp4",), output_dtypes=(torch.bfloat16,))


def test_relu_is_applied_per_head_before_weighting():
    """The kernel computes Σ_h w·relu(dot), not relu(Σ_h w·dot).

    The module and trace-template docstrings previously documented the latter.
    The two disagree whenever head scores have mixed signs: with scores [1, -1]
    and weights [1, 1] the kernel returns 1 while relu-of-the-sum returns 0.
    They also disagree on range -- relu-of-the-sum cannot be negative, but the
    kernel can be with negative weights. Regression for PR #4365 review
    r3825128740; the references were always right, so only prose was wrong and
    no existing test could have caught it.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

    from flashinfer import fp8_paged_mqa_logits

    device = "cuda"
    B, next_n, H, D, block_size, ctx = 1, 1, 64, 128, 64, 64
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)

    # head 0 sees +1, head 1 sees -1 against the same K column; all others zero
    q = torch.zeros(B, next_n, H, D, device=device)
    kv = torch.zeros(ntb, block_size, D, device=device)
    kv[block_tables[0, 0].item(), 0, 0] = 1.0
    q[0, 0, 0, 0] = 1.0
    q[0, 0, 1, 0] = -1.0

    scale = _ceil_to_ue8m0_fp(
        kv.abs().amax(-1, keepdim=True).clamp(1e-4) / 448.0
    ).squeeze(-1)
    kv_fused = _make_fused_kv_fp8(
        (kv / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale, block_size, D
    )
    q8 = q.to(torch.float8_e4m3fn)

    w = torch.zeros(B * next_n, H, device=device, dtype=torch.float32)
    w[0, 0] = 1.0
    w[0, 1] = 1.0
    out = fp8_paged_mqa_logits(q8, kv_fused, w, block_tables, seq_lens, ctx)
    torch.cuda.synchronize()
    # Σ w·relu(dot) = 1·1 + 1·0 = 1 (times the KV scale, which is 1 here);
    # relu(Σ w·dot) = relu(0) = 0.
    assert out[0, 0].item() > 0.5, (
        f"expected ~1 from per-head ReLU, got {out[0, 0].item()}; "
        "relu-of-the-sum would give 0"
    )

    # negative weights: relu-of-the-sum could never be negative, the kernel can
    w_neg = torch.zeros(B * next_n, H, device=device, dtype=torch.float32)
    w_neg[0, 0] = -1.0
    out_neg = fp8_paged_mqa_logits(q8, kv_fused, w_neg, block_tables, seq_lens, ctx)
    torch.cuda.synchronize()
    assert out_neg[0, 0].item() < -0.5, (
        f"expected a negative logit with a negative weight, got {out_neg[0, 0].item()}"
    )


def test_precompile_variants():
    """precompile_paged_mqa_logits(variants=...) builds only what was asked for."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")

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
            seq_lens = torch.full((2,), ctx, dtype=torch.int32, device=device)
            block_tables, ntb = _make_paged_kv(2, block_size, seq_lens, device)
            q = torch.zeros(2, next_n, 64, 128, device=device).to(torch.float8_e4m3fn)
            kv = torch.zeros(ntb, block_size, 1, 132, dtype=torch.uint8, device=device)
            w = torch.randn(2 * next_n, 64, device=device, dtype=torch.float32)
            fp8_paged_mqa_logits(q, kv, w, block_tables, seq_lens, max_ml)
    assert _cached_compile_fp8_kernel.cache_info().misses == misses, (
        "a precompiled fp8 config still triggered a build; the precompile "
        "config list has drifted from what the API actually compiles"
    )


def test_fp4_is_kv_sf_interleaved_guard():
    """is_kv_sf_interleaved=True is only valid at block_size=128.

    The kernel silently forces the flag back to False for other block sizes, so
    a caller who pre-arranged SF for UTCCP would get it interleaved twice
    -- wrong logits with no error. The API must reject the combination.
    """
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )
    from flashinfer import fp4_paged_mqa_logits

    device, B, next_n, H, D, ctx, max_ml = "cuda", 2, 1, 64, 128, 512, 512
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)

    for block_size, should_raise in ((32, True), (64, True), (128, False)):
        block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
        q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
        q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
        kv = torch.zeros(
            ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device
        )
        w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
        if should_raise:
            with pytest.raises(ValueError, match="is_kv_sf_interleaved"):
                fp4_paged_mqa_logits(
                    q,
                    q_sf,
                    kv,
                    w,
                    block_tables,
                    seq_lens,
                    max_ml,
                    is_kv_sf_interleaved=True,
                )
        else:
            fp4_paged_mqa_logits(
                q,
                q_sf,
                kv,
                w,
                block_tables,
                seq_lens,
                max_ml,
                is_kv_sf_interleaved=True,
            )


def test_next_n_and_weights_contract():
    """weights shape/dtype are validated, and pin next_n without a next_n arg."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")
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
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    q8 = torch.zeros(B, next_n, H, D, device=device).to(torch.float8_e4m3fn)
    kv8 = torch.zeros(ntb, block_size, 1, D + 4, dtype=torch.uint8, device=device)
    q4 = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    sf4 = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv4 = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)

    # next_n is not an API parameter: it is q.shape[1] by definition, and a q
    # that disagrees with the other tensors is caught by the weights and q_sf
    # cross-checks without needing the caller to restate it.
    q8_bad = torch.zeros(B, next_n + 1, H, D, device=device).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="weights must be"):
        fp8_paged_mqa_logits(q8_bad, kv8, w, block_tables, seq_lens, max_ml)
    q4_bad = torch.zeros(B, next_n + 1, H, D // 2, dtype=torch.uint8, device=device)
    # Either cross-check may fire first depending on ordering; both prove the
    # inconsistency is caught without an explicit next_n argument.
    with pytest.raises(ValueError, match="(weights|q_sf) must be"):
        fp4_paged_mqa_logits(q4_bad, sf4, kv4, w, block_tables, seq_lens, max_ml)

    # weights shape and dtype are now validated (previously unchecked).
    bad_shape = torch.randn(B * next_n + 1, H, device=device, dtype=torch.float32)
    bad_dtype = torch.randn(B * next_n, H, device=device, dtype=torch.bfloat16)
    for bad_w, pat in (
        (bad_shape, "weights must be"),
        (bad_dtype, "weights must be float32"),
    ):
        with pytest.raises(ValueError, match=pat):
            fp8_paged_mqa_logits(q8, kv8, bad_w, block_tables, seq_lens, max_ml)
        with pytest.raises(ValueError, match=pat):
            fp4_paged_mqa_logits(q4, sf4, kv4, bad_w, block_tables, seq_lens, max_ml)


def test_fp4_sf_vec_size_contract():
    """sf_vec_size drives both q_sf packing and the KV row's scale-factor bytes."""
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip(
            "FP4 paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)"
        )
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
    seq_lens = torch.full((B,), ctx, dtype=torch.int32, device=device)
    block_tables, ntb = _make_paged_kv(B, block_size, seq_lens, device)
    w = torch.randn(B * next_n, H, device=device, dtype=torch.float32)
    q = torch.zeros(B, next_n, H, D // 2, dtype=torch.uint8, device=device)
    q_sf = torch.zeros(B, next_n, H, dtype=torch.int32, device=device)
    kv = torch.zeros(ntb, block_size, 1, D // 2 + 4, dtype=torch.uint8, device=device)

    fp4_paged_mqa_logits(q, q_sf, kv, w, block_tables, seq_lens, max_ml, sf_vec_size=32)
    for bad in (16, 64):
        with pytest.raises(ValueError, match="sf_vec_size must be 32"):
            fp4_paged_mqa_logits(
                q, q_sf, kv, w, block_tables, seq_lens, max_ml, sf_vec_size=bad
            )

    # The KV row's scale-factor bytes are derived from sf_vec_size, so a row
    # sized for a different SF count is rejected with the arithmetic spelled out.
    kv_bad = torch.zeros(
        ntb, block_size, 1, D // 2 + 8, dtype=torch.uint8, device=device
    )
    with pytest.raises(ValueError, match="scale-factor bytes"):
        fp4_paged_mqa_logits(q, q_sf, kv_bad, w, block_tables, seq_lens, max_ml)


# Selective FP8 MQA: inputs, independent oracles, kernels and wrapper lifecycle


_requires_selective_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or importlib.util.find_spec("cutlass") is None
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="selective FP8 logits requires SM100 or SM103",
)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (10, 3), (10, 7), (12, 0)])
def test_selective_constructor_device_support(monkeypatch, capability):
    """Unsupported devices reject before device-scope entry or workspace allocation."""
    from flashinfer.attn_scores import selective_capacity

    monkeypatch.setattr(sl, "get_compute_capability", lambda device: capability)
    device = torch.device("cuda:0")
    if capability in ((10, 0), (10, 3)):
        sl._validate_selective_device(device)
        return

    def unexpected_cuda_work(*args, **kwargs):
        raise AssertionError(
            "unsupported construction must not allocate or enter device scope"
        )

    monkeypatch.setattr(torch.cuda, "device", unexpected_cuda_work)
    monkeypatch.setattr(selective_capacity, "_CapacityStepInputs", unexpected_cuda_work)
    with pytest.raises(ValueError, match="requires SM100 or SM103"):
        FP8PagedMQATopKWrapper(
            strategy="selective",
            max_context_length=1024,
            max_chunk_size=1,
            max_batch_size=1,
            num_pages=8,
            device=device,
        )


@pytest.mark.parametrize(
    "device, expected", [("cuda:0", 0), ("cuda:1", 1), ("cuda", 2)]
)
def test_selective_compile_device_ordinal(monkeypatch, device, expected):
    from flashinfer.attn_scores.selective_logits import _device_ordinal

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert _device_ordinal(torch.device(device)) == expected


def test_topk_plan_explicit_signature_and_strategy_binding(monkeypatch):
    """Strategy-specific logging uses true names for both planning contracts."""
    import inspect

    for method, names in (
        (
            FP8PagedMQATopKWrapper._plan_selective,
            ("self", "qo_indptr", "kv_indptr", "block_table", "num_query_rows"),
        ),
        (
            FP8PagedMQATopKWrapper._plan_full,
            (
                "self",
                "q",
                "kv_fused",
                "weights",
                "qo_indptr",
                "kv_indptr",
                "block_table",
                "top_k",
                "num_heads",
                "head_dim",
            ),
        ),
    ):
        signature = inspect.signature(method)
        assert tuple(signature.parameters) == names
        assert all(
            parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            for parameter in signature.parameters.values()
        )
    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    tensors = [object() for _ in range(6)]
    full_calls = []
    monkeypatch.setattr(
        wrapper,
        "_plan_full_payload",
        lambda *args, **kwargs: full_calls.append((args, kwargs)),
    )
    wrapper.plan(*tensors, top_k=2048, num_heads=64)
    wrapper.plan(
        **dict(
            zip(
                ("q", "kv_fused", "weights", "qo_indptr", "kv_indptr", "block_table"),
                tensors,
                strict=True,
            )
        )
    )
    assert [call[0] for call in full_calls] == [tuple(tensors)] * 2
    assert full_calls[0][1] == dict(top_k=2048, num_heads=64, head_dim=128)
    with pytest.raises(TypeError):
        wrapper.plan(qo_indptr=tensors[0])

    step_calls = []
    wrapper._capacity_plan = SimpleNamespace(
        plan=lambda *args, **kwargs: step_calls.append((args, kwargs))
    )
    prefixes = tensors[:3]
    wrapper.plan(*prefixes, num_query_rows=7)
    wrapper.plan(
        qo_indptr=prefixes[0],
        kv_indptr=prefixes[1],
        block_table=prefixes[2],
        num_query_rows=7,
    )
    wrapper.plan(*prefixes[:2], block_table=prefixes[2], num_query_rows=7)
    assert step_calls == [(tuple(prefixes), {"num_query_rows": 7})] * 3
    with pytest.raises(TypeError, match="multiple values"):
        wrapper.plan(*prefixes, qo_indptr=prefixes[0])
    with pytest.raises(TypeError, match="multiple values"):
        wrapper.plan(*prefixes, block_table=prefixes[2])
    with pytest.raises(TypeError, match="required positional"):
        wrapper.plan()
    with pytest.raises(TypeError, match="unexpected keyword"):
        wrapper.plan(*prefixes, top_k=1024)


def _compressed_wrapper_fixture(q_lengths, token_lengths, heads, page_size):
    physical_lengths = [length // 4 for length in token_lengths]
    q, kv, scale, weights, cu_q, cu_physical = _request_inputs(
        q_lengths, physical_lengths, heads
    )
    token_prefix = [0]
    for tokens in token_lengths:
        token_prefix.append(token_prefix[-1] + tokens)
    cu_tokens = torch.tensor(token_prefix, device=q.device, dtype=torch.int32)
    if kv.shape[0]:
        fused, table = _paged_inputs(
            (q, kv, scale, weights, cu_q, cu_physical), page_size
        )
        for request, length in enumerate(physical_lengths):
            live_pages = math.ceil(length / page_size)
            table[request, live_pages:].fill_(2**31 - 1)
            if length % page_size:
                page = int(table[request, live_pages - 1].item())
                fused.flatten(1)[page, : page_size * 128].view(page_size, 128)[
                    length % page_size :
                ].fill_(0x7F)
    else:
        fused = torch.empty((0, page_size, 1, 132), device=q.device, dtype=torch.uint8)
        table = torch.empty((len(q_lengths), 0), device=q.device, dtype=torch.int32)
    return q, fused, weights, cu_q, cu_tokens, table


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["selective"])
def test_compression_ratio_one_keeps_original_plan_and_schema(strategy):
    q, kv, scales, weights, _, _ = _inputs(17, 8192, 8)
    cu_q = torch.tensor([0, 17], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 8192], device=q.device, dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv))
    capacities = dict(
        max_context_length=8192, max_chunk_size=17, max_batch_size=1, kv_layout=fused
    )
    wrappers = [
        FP8PagedMQATopKWrapper(strategy=strategy, **capacities),
        FP8PagedMQATopKWrapper(strategy=strategy, compression_ratio=1, **capacities),
    ]
    outputs = []
    for wrapper in wrappers:
        wrapper.plan(cu_q, cu_kv, table)
        outputs.append(wrapper.run(q, weights, fused).clone())
        assert wrapper.compression_ratio == 1
    torch.cuda.synchronize()
    for output in outputs:
        _check_selection(
            output, _oracle_rows((q, kv, scales, weights, cu_q, cu_kv)), 512
        )
    from flashinfer.trace.templates.attn_scores import (
        fp8_paged_mqa_selective_topk_trace,
        fp8_paged_mqa_topk_trace_dispatch,
    )

    assert all(
        fp8_paged_mqa_topk_trace_dispatch(self=wrapper)
        is fp8_paged_mqa_selective_topk_trace
        for wrapper in wrappers
    )
    assert "compression_ratio" not in fp8_paged_mqa_selective_topk_trace.inputs
    for ratio in (0, 2, True, 4.0):
        with pytest.raises(ValueError, match="compression_ratio must be"):
            FP8PagedMQATopKWrapper(
                strategy=strategy, compression_ratio=ratio, **capacities
            )
    with pytest.raises(ValueError, match="only supported by selective"):
        FP8PagedMQATopKWrapper(strategy="full", compression_ratio=4)


def _check_compressed_wrapper(output, payload, top_k):
    q, fused, weights, cu_q, cu_tokens, table = payload
    page_size = fused.shape[1]
    packed = fused.flatten(1)
    q_prefix, token_prefix = cu_q.tolist(), cu_tokens.tolist()
    for request in range(len(q_prefix) - 1):
        qb, qe = q_prefix[request : request + 2]
        tokens = token_prefix[request + 1] - token_prefix[request]
        physical = tokens // 4
        if physical:
            raw = packed[table[request, : math.ceil(physical / page_size)].long()]
            keys = (
                raw[:, : page_size * 128]
                .contiguous()
                .view(q.dtype)
                .reshape(-1, 128)[:physical]
                .float()
            )
            scales = (
                raw[:, page_size * 128 :]
                .contiguous()
                .view(torch.float32)
                .reshape(-1)[:physical]
            )
        for local, row in enumerate(range(qb, qe)):
            end = (tokens - (qe - qb) + local + 1) // 4
            count = min(top_k, end)
            assert (output[row, count:] == -1).all()
            ids = output[row, :count].long()
            assert ((ids >= 0) & (ids < end)).all()
            assert ids.unique().numel() == count
            if count:
                scores = (
                    (q[row].float() @ keys[:end].T).relu_() * weights[row, :, None]
                ).sum(0) * scales[:end]
                torch.testing.assert_close(
                    scores[ids].sort().values,
                    scores.topk(count).values.sort().values,
                    atol=5e-5,
                    rtol=1e-5,
                )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "page_size", "top_k", "q_lengths", "token_lengths"),
    [
        (8, 32, 512, [1, 2, 3], [1, 9, 11]),
        (64, 64, 2048, [37], [37]),
        (8, 128, 512, [1, 17], [1, 40001]),
        (8, 128, 512, [1, 2, 3], [1, 2, 3]),
        (64, 128, 512, [1, 17], [1, 40001]),
        (64, 128, 512, [9], [40001]),
        (64, 128, 2048, [9], [40001]),
        (64, 128, 2048, [1, 37], [1, 37]),
    ],
)
def test_engine_compressed_original_token_bounds_and_graph(
    heads, page_size, top_k, q_lengths, token_lengths, monkeypatch
):
    """Nonzero-Q empty requests, per-request floors, poisoned pages and repair."""
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    payload = _compressed_wrapper_fixture(q_lengths, token_lengths, heads, page_size)
    q, fused, weights, cu_q, cu_tokens, table = payload
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=heads,
        num_sms=4,
        compression_ratio=4,
    )
    output = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    torch.cuda.synchronize()
    _check_compressed_wrapper(output, payload, top_k)
    if page_size == 32:
        from flashinfer.trace.templates.attn_scores import (
            _compressed_topk_reference,
            _compressed_topk_check,
        )

        trace_args = dict(
            q=q,
            kv_fused=fused,
            weights=weights,
            cu_q=cu_q,
            cu_kv=cu_tokens,
            block_table=table,
            top_k=top_k,
            compression_ratio=4,
        )
        reference = _compressed_topk_reference(**trace_args)
        assert _compressed_topk_check(reference, output, **trace_args)
        bad = output.clone()
        bad[0, 0] = 0  # The first nonzero-Q request has no physical key0.
        assert not _compressed_topk_check(reference, bad, **trace_args)
    expected_prefix = [0]
    for length in token_lengths:
        expected_prefix.append(expected_prefix[-1] + length // 4)
    assert wrapper._physical_cu_kv.tolist() == expected_prefix
    if expected_prefix[-1] == 0:
        assert wrapper._candidate_prepared is None
        assert wrapper._empty_physical
    if heads == 64 and q_lengths == [9]:
        engine = wrapper
        # R4 single-pool prefill is published by the H64 candidate variant.
        assert engine._candidate_schedule.h64_candidate
        assert engine._candidate_schedule.publication == "warp_striped"
        assert engine._candidate_prepared.query_tile == 4
        assert engine._candidate_prepared.num_umma_stages == 1
        assert engine._candidate_prepared.split_kv == 1
        assert engine._candidate_counts.shape[1] == 5
        assert engine._candidate_schedule.sample_count_segments == 32
    force_repair = max(token_lengths) == 40001
    if force_repair:
        engine = wrapper
        threshold, threshold_args = engine._sample_threshold_launch

        def forced_threshold(*args):
            threshold(*args)
            engine._thresholds.fill_(float("inf"))

        engine._sample_threshold_launch = forced_threshold, threshold_args
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    for _ in range(2):
        q.copy_((q.float() * 0.5).to(q.dtype))
        weights.mul_(0.75)
        for request, length in enumerate(token_lengths):
            if length // 4 >= 2 * page_size:
                table[request, :2].copy_(table[request, :2].flip(0))
        graph.replay()
        torch.cuda.synchronize()
        _check_compressed_wrapper(captured, payload, top_k)
        assert captured.data_ptr() == output.data_ptr()
    if force_repair:
        repair_engine = wrapper
        assert repair_engine._repair_flags.count_nonzero().item() > 0
        assert repair_engine._exact_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["selective"])
@pytest.mark.parametrize("heads", [8, 64])
def test_compressed_wrapper_empty_live_empty_replan(strategy, monkeypatch, heads):
    """One bound pool supports empty/live steps; a failed pool binding retries."""
    empty = _compressed_wrapper_fixture([1], [1], heads, 128)
    live = _compressed_wrapper_fixture([1], [9], heads, 128)
    q, fused, weights, cu_q, cu_tokens, table = live
    specialization = dict(strategy=strategy, num_heads=heads, compression_ratio=4)
    capacities = dict(max_context_length=9, max_chunk_size=1, max_batch_size=1)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    wrapper.prepare_pool(fused)
    pool_graph = wrapper._capacity_plan._pool_graphs.copy()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, cu_tokens, table)
        output = wrapper.run(q, weights, fused)
    for tokens in (1, 9, 1):
        cu_tokens[-1] = tokens
        graph.replay()
        torch.cuda.synchronize()
        _check_compressed_wrapper(
            output, (q, fused, weights, cu_q, cu_tokens, table), 512
        )
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert wrapper._capacity_plan._pool_graphs == pool_graph
    del graph
    # Capacity changes construct a new wrapper rather than re-reserving one.
    empty_wrapper = FP8PagedMQATopKWrapper(
        **specialization,
        max_context_length=1,
        max_chunk_size=1,
        max_batch_size=1,
        kv_layout=empty[1],
    )
    empty_wrapper.plan(empty[3], empty[4], empty[5])
    result = empty_wrapper.run(empty[0], empty[2], empty[1])
    torch.cuda.synchronize()
    assert (result == -1).all()
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    with monkeypatch.context() as patch:
        patch.setattr(
            torch.cuda,
            "CUDAGraph",
            lambda: (_ for _ in ()).throw(
                RuntimeError("injected compressed capture failure")
            ),
        )
        with pytest.raises(RuntimeError, match="injected compressed capture failure"):
            wrapper.prepare_pool(fused)
    assert not wrapper._capacity_plan._pool_graphs
    with pytest.raises(RuntimeError, match="plan must be called"):
        wrapper.run(q, weights, fused)
    wrapper.prepare_pool(fused)
    cu_tokens[-1] = 9
    wrapper.plan(cu_q, cu_tokens, table, num_query_rows=q.shape[0])
    output = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    _check_compressed_wrapper(output, (q, fused, weights, cu_q, cu_tokens, table), 512)


_requires_selective_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cutlass") is None,
    reason="CUDA and CuTe DSL are required",
)

_PAGED_INPUT_CACHE: dict[
    tuple[torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@_requires_selective_sm100
def test_candidate_publication_rejects_layout_from_another_split():
    inputs = _inputs(q_rows=1, kv_rows=128)
    prepared = _prepare(inputs, split_kv=1)
    values = torch.empty((16, 6144), device="cuda", dtype=torch.int64)
    split_two_lane_counts = torch.empty((16, 65), device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="split_kv=1 candidate publication requires"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            6144,
            values,
            split_two_lane_counts,
            prepared,
            inputs,
            torch.full((1,), float("-inf"), device="cuda"),
        )


@_requires_selective_sm100
def test_sample_pack_kernel_uses_64_bit_source_coordinates():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveLogitsSamplePackKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    head_dim = 128
    page_size = 64
    sample_page_size = 128
    source_page_bytes = page_size * (head_dim + 4)
    first_block_past_i32_byte_offset = (2**31 // source_page_bytes) + 1

    source = torch.empty(
        (first_block_past_i32_byte_offset + 1, source_page_bytes),
        dtype=torch.uint8,
        device="cuda",
    )
    expected_kv = torch.arange(head_dim, dtype=torch.uint8, device="cuda")
    expected_scale = torch.tensor([0.125], dtype=torch.float32, device="cuda").view(
        torch.uint8
    )
    source[first_block_past_i32_byte_offset, :head_dim] = expected_kv
    source[
        first_block_past_i32_byte_offset,
        page_size * head_dim : page_size * head_dim + 4,
    ] = expected_scale

    destination = torch.full(
        (1, sample_page_size * (head_dim + 4)),
        0xCD,
        dtype=torch.uint8,
        device="cuda",
    )
    block_table = torch.tensor(
        [[first_block_past_i32_byte_offset]], dtype=torch.int32, device="cuda"
    )
    sample_block_table = torch.zeros((1, 1), dtype=torch.int32, device="cuda")
    sample_request_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    sample_request_starts = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    sample_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = _current_stream(source.device)
    args = (
        from_dlpack(source, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(block_table, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(sample_block_table, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(sample_request_ids, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(sample_request_starts, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(sample_indices, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        ),
        from_dlpack(destination, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        cutlass.Int32(1),
    )
    compiled = cute.compile(
        SelectiveLogitsSamplePackKernel(head_dim, page_size, sample_page_size),
        *args,
        stream,
    )
    compiled(*args, stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(destination[0, :head_dim], expected_kv)
    scale_offset = sample_page_size * head_dim
    torch.testing.assert_close(
        destination[0, scale_offset : scale_offset + 4], expected_scale
    )


@_requires_selective_sm100
def test_unified_wrapper_requires_explicit_supported_strategy():
    from flashinfer import FP8PagedMQATopKWrapper

    assert FP8PagedMQATopKWrapper(strategy="full").strategy == "full"
    with pytest.raises(ValueError, match="strategy must be 'full' or 'selective'"):
        FP8PagedMQATopKWrapper(strategy="auto")


def _inputs(
    q_rows: int = 17,
    kv_rows: int = 387,
    num_heads: int = 8,
    head_dim: int = 128,
):
    torch.manual_seed(1)
    device = "cuda"
    q = (torch.randn(q_rows, num_heads, head_dim, device=device) * 0.1).to(
        torch.float8_e4m3fn
    )
    kv_f32 = torch.randn(kv_rows, head_dim, device=device) * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, num_heads, device=device, dtype=torch.float32)
    if q_rows == 1:
        cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, kv_rows], device=device, dtype=torch.int32)
    else:
        first_kv = min(132, kv_rows // 2)
        cu_q = torch.tensor([0, 1, q_rows], device=device, dtype=torch.int32)
        cu_kv = torch.tensor([0, first_kv, kv_rows], device=device, dtype=torch.int32)
    return q, kv, scale, weights, cu_q, cu_kv


def _request_inputs(q_lengths, kv_lengths, heads=8, head_dim=128):
    """Fresh raw owners and explicit request prefixes; no wrapper or graph state."""
    q, kv, scales, weights, _, _ = _inputs(
        sum(q_lengths), sum(kv_lengths), heads, head_dim
    )
    q_prefix, kv_prefix = [0], [0]
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_prefix.append(q_prefix[-1] + q_length)
        kv_prefix.append(kv_prefix[-1] + kv_length)
    return (
        q,
        kv,
        scales,
        weights,
        torch.tensor(q_prefix, device=q.device, dtype=torch.int32),
        torch.tensor(kv_prefix, device=q.device, dtype=torch.int32),
    )


def _valid_lengths(cu_q: torch.Tensor, cu_kv: torch.Tensor) -> list[int]:
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_len = q_prefix[batch + 1] - q_prefix[batch]
        kv_len = kv_prefix[batch + 1] - kv_prefix[batch]
        result.extend(kv_len - q_len + row + 1 for row in range(q_len))
    return result


def _fuse_paged_kv(
    kv: torch.Tensor, scale: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """Pack contiguous KV and scales for the public paged-logits API."""
    head_dim = kv.shape[1]
    num_blocks = kv.shape[0] // block_size
    fused = torch.zeros(
        num_blocks,
        block_size * (head_dim + 4),
        device=kv.device,
        dtype=torch.uint8,
    )
    kv_blocks = kv.view(num_blocks, block_size, head_dim)
    scale_blocks = scale.view(num_blocks, block_size)
    scale_offset = block_size * head_dim
    for block in range(num_blocks):
        fused[block, :scale_offset] = kv_blocks[block].view(torch.uint8).reshape(-1)
        fused[block, scale_offset:] = (
            scale_blocks[block].contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_blocks, block_size, 1, head_dim + 4)


def _paged_inputs(inputs, block_size: int = 128):
    """Build reusable request-local pages with a reversed physical mapping."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    key = (kv, scale, block_size)
    cached = _PAGED_INPUT_CACHE.get(key)
    if cached is not None:
        return cached
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    pages_per_compute_tile = 128 // block_size
    max_pages = max(
        ((kv_prefix[i + 1] - kv_prefix[i] + 127) // 128 * pages_per_compute_tile)
        for i in range(len(kv_prefix) - 1)
    )
    table = torch.zeros(
        (len(q_prefix) - 1, max_pages), device=kv.device, dtype=torch.int32
    )
    page_values = []
    page_scales = []
    for batch in range(len(q_prefix) - 1):
        begin, end = kv_prefix[batch : batch + 2]
        for local_page, offset in enumerate(range(begin, end, block_size)):
            valid = min(block_size, end - offset)
            page = torch.zeros(
                (block_size, kv.shape[1]), device=kv.device, dtype=kv.dtype
            )
            page_scale = torch.ones(block_size, device=scale.device, dtype=scale.dtype)
            page[:valid] = kv[offset : offset + valid]
            page_scale[:valid] = scale[offset : offset + valid]
            table[batch, local_page] = len(page_values)
            page_values.append(page)
            page_scales.append(page_scale)
    paged_kv = torch.stack(page_values)
    paged_scale = torch.stack(page_scales)
    fused = _fuse_paged_kv(
        paged_kv.reshape(-1, kv.shape[1]), paged_scale.reshape(-1), block_size
    )
    # Exercise a non-identity physical layout in every selective test while
    # preserving each request's logical page order through the block table.
    fused = fused.flip(0).contiguous()
    table = fused.shape[0] - 1 - table
    _PAGED_INPUT_CACHE[key] = (fused, table)
    return fused, table


@pytest.mark.parametrize("heads,query_tile", [(8, 16), (64, 4)])
@pytest.mark.parametrize("ratio", [1, 4])
def test_fp8_grouped_public_scorer_live_maps(heads, query_tile, ratio, monkeypatch):
    """Public grouped scoring preserves request bounds and live graph inputs."""
    from flashinfer.attn_scores import (
        FP8MQAKernelConfig,
        FP8QueryGroups,
        fp8_paged_mqa_logits,
    )

    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("grouped FP8 scorer requires SM100 or SM103")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q_lengths = [query_tile + 1, 0, 1]
    token_lengths = [ratio * 385 + ratio - 1, ratio * 129, 1]
    physical_lengths = [length // ratio for length in token_lengths]
    inputs = _request_inputs(q_lengths, physical_lengths, heads)
    q, kv, scales, weights, cu_q, cu_kv = inputs
    token_prefix = [0]
    for length in token_lengths:
        token_prefix.append(token_prefix[-1] + length)
    groups = FP8QueryGroups.build(cu_q.cpu().tolist(), token_prefix, ratio, query_tile)
    fused, table = _paged_inputs(inputs)
    # Neither an empty-query request nor a zero-key row may dereference its map.
    table[1].fill_(2**30)
    if physical_lengths[2] == 0:
        table[2].fill_(2**30)
    gather = torch.tensor(groups.gather_rows, device=q.device, dtype=torch.int64)
    requests = torch.tensor(groups.request_ids, device=q.device, dtype=torch.int64)
    invalid = torch.tensor(groups.invalid_queries, device=q.device)
    grouped_q = torch.empty(
        (len(groups.rows), query_tile, heads, 128), device=q.device, dtype=q.dtype
    )
    grouped_weights = torch.empty(
        (len(gather), heads), device=q.device, dtype=weights.dtype
    )
    grouped_table = torch.empty(
        (len(groups.rows), table.shape[1]), device=q.device, dtype=table.dtype
    )
    metadata = torch.tensor(groups.rows, device=q.device, dtype=torch.int32)
    ends = torch.tensor(groups.group_ends, device=q.device, dtype=torch.int32)
    config = FP8MQAKernelConfig(128, 1, 160, query_tile, 3, 3)
    output = torch.full(
        (q.shape[0], config.padded_columns(max(physical_lengths))),
        float("nan"),
        device=q.device,
    )

    def run():
        torch.index_select(
            q.view(torch.uint8),
            0,
            gather,
            out=grouped_q.view(-1, heads, 128).view(torch.uint8),
        )
        torch.index_select(weights, 0, gather, out=grouped_weights)
        grouped_q.view(-1, heads, 128).view(torch.uint8).masked_fill_(
            invalid[:, None, None], 0
        )
        grouped_weights.masked_fill_(invalid[:, None], 0)
        torch.index_select(table, 0, requests, out=grouped_table)
        return fp8_paged_mqa_logits(
            grouped_q,
            fused,
            grouped_weights,
            grouped_table,
            ends,
            max(physical_lengths),
            kernel_config=config,
            query_metadata=metadata,
            output_rows=q.shape[0],
            out=output,
        )

    def check():
        row = 0
        physical_prefix = cu_kv.cpu().tolist()
        for req, (queries, tokens) in enumerate(
            zip(q_lengths, token_lengths, strict=True)
        ):
            base = physical_prefix[req]
            for local in range(queries):
                end = (tokens - queries + local + 1) // ratio
                dots = q[row].float() @ kv[base : base + end].float().T
                expected = (dots.relu() * weights[row, :, None]).sum(0) * scales[
                    base : base + end
                ]
                torch.testing.assert_close(
                    output[row, :end], expected, atol=1e-4, rtol=1e-4
                )
                row += 1

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    check()
    old_output = output.clone()
    q.copy_((-q.float()).to(q.dtype))
    weights.mul_(0.5)
    # Keep logical keys unchanged while changing the physical map and payload.
    fused.copy_(fused.flip(0))
    valid_pages = table < fused.shape[0]
    table[valid_pages] = fused.shape[0] - 1 - table[valid_pages]
    graph.replay()
    check()
    assert not torch.allclose(old_output[0, :128], output[0, :128])
    run()
    check()
    if heads == 8 and ratio == 4:
        config = FP8MQAKernelConfig(128, 1, 160, query_tile, 2, 2)
        run()
        check()


def _prepare(inputs, **kwargs):
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    return _prepare_fp8_mqa_selective_logits(
        q, kv_fused, weights, cu_q, cu_kv, block_table, **kwargs
    )


def _oracle_rows(inputs) -> list[torch.Tensor]:
    """Evaluate the reference per-head-ReLU FP32 score definition."""
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q_prefix = cu_q.cpu().tolist()
    kv_prefix = cu_kv.cpu().tolist()
    result = []
    for batch in range(len(q_prefix) - 1):
        q_begin, q_end = q_prefix[batch : batch + 2]
        kv_begin, kv_end = kv_prefix[batch : batch + 2]
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        for local_row in range(q_len):
            valid = kv_len - q_len + local_row + 1
            dots = torch.matmul(
                q[q_begin + local_row].float(),
                kv[kv_begin : kv_begin + valid].float().T,
            )
            weighted = (torch.relu(dots) * weights[q_begin + local_row, :, None]).sum(
                dim=0
            )
            result.append(weighted * scale[kv_begin : kv_begin + valid])
    return result


def _launch(
    mode,
    capacity,
    values,
    index_counts,
    prepared,
    inputs,
    policy_values=None,
    radix_shift=0,
):
    from flashinfer.attn_scores.selective_logits import _fp8_mqa_selective_logits

    q, _kv, _scale, weights, _cu_q, _cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=mode,
        capacity=capacity,
        values=values,
        index_counts=index_counts,
        policy_values=policy_values,
        radix_shift=radix_shift,
    )


@_requires_selective_sm100
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("capacity", [7, 512])
@pytest.mark.parametrize("ties", [False, True])
def test_fused_exact_emission_banks(num_heads, capacity, ties):
    inputs = _inputs(q_rows=33, num_heads=num_heads)
    if ties:
        inputs[3].zero_()
    prepared = _prepare(inputs, num_sms=4, num_heads=num_heads)
    padded = (
        (33 + prepared.query_tile - 1) // prepared.query_tile
    ) * prepared.query_tile
    bank_rows = ((padded + 3) // 4) * 4
    sampled = torch.zeros((padded, 512), device="cuda", dtype=torch.float32)
    unused = torch.empty((padded, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 512, sampled, unused, prepared, inputs)
    lengths = _valid_lengths(inputs[4], inputs[5])
    thresholds = torch.stack(
        [sampled[row, valid // 2] for row, valid in enumerate(lengths)]
    )
    values = torch.full((2 * bank_rows, capacity), -7, device="cuda", dtype=torch.int32)
    counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        thresholds,
    )
    for row, valid in enumerate(lengths):
        for bank in (0, 1):
            expected = (
                torch.nonzero(
                    (sampled[row, :valid] > thresholds[row])
                    if bank == 0
                    else (sampled[row, :valid] == thresholds[row])
                )
                .flatten()
                .to(torch.int32)
            )
            count = int(counts[row + bank * bank_rows, 0])
            assert count == expected.numel()
            actual = values[row + bank * bank_rows, : min(count, capacity)]
            assert actual.unique().numel() == actual.numel()
            assert torch.isin(actual, expected).all()
            if count <= capacity:
                torch.testing.assert_close(actual.sort().values, expected.sort().values)
    assert (counts[33:bank_rows] == 0).all()
    assert (counts[bank_rows + 33 :] == 0).all()


@_requires_selective_sm100
def test_selective_sample_matches_public_paged_logits_gemm():
    """Prove both logits implementations agree on identical causal inputs."""
    from flashinfer import fp8_paged_mqa_logits

    torch.manual_seed(7)
    q_rows, kv_rows = 16, 256
    q = (torch.randn(q_rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(q_rows, 8, device="cuda", dtype=torch.float32)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs)
    selective = torch.full((q_rows, kv_rows), float("nan"), device="cuda")
    unused_counts = torch.empty((q_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        kv_rows,
        selective,
        unused_counts,
        prepared,
        inputs,
    )

    paged = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        _fuse_paged_kv(kv, scale),
        weights,
        torch.arange(2, device="cuda", dtype=torch.int32).unsqueeze(0),
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        kv_rows,
    )
    torch.cuda.synchronize()

    for row in range(q_rows):
        causal_end = kv_rows - q_rows + row + 1
        torch.testing.assert_close(
            selective[row, :causal_end],
            paged[row, :causal_end],
            atol=5e-5,
            rtol=1e-5,
        )
        assert torch.isnan(selective[row, causal_end:]).all()


@_requires_selective_sm100
def test_selective_h64_reads_arbitrary_physical_page_mapping():
    """Exercise the selective route directly with H64 and reordered pages."""
    from flashinfer.attn_scores.selective_logits import (
        _fp8_mqa_selective_logits,
        _prepare_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=4, kv_rows=256, num_heads=64, head_dim=128)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=64)
    prepared = _prepare_fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=64,
        head_dim=128,
    )
    assert prepared.query_tile == 2
    values = torch.full((4, 256), float("nan"), device="cuda")
    counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.SAMPLE,
        capacity=256,
        values=values,
        index_counts=counts,
    )
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            values[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(values[row, expected.numel() :]).all()

    candidates = torch.empty((4, 256), device="cuda", dtype=torch.int64)
    candidate_counts = torch.empty((4, 1), device="cuda", dtype=torch.int32)
    thresholds = torch.full((4,), float("-inf"), device="cuda")
    _fp8_mqa_selective_logits(
        q,
        kv_fused,
        weights,
        block_table,
        prepared,
        mode=_SelectiveLogitsMode.CANDIDATE,
        capacity=256,
        values=candidates,
        index_counts=candidate_counts,
        policy_values=thresholds,
    )
    torch.cuda.synchronize()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        assert int(candidate_counts[row, 0]) == valid
        indices = (candidates[row, :valid] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            indices, torch.arange(valid, device="cuda", dtype=torch.int64)
        )


@_requires_selective_sm100
def test_unified_wrapper_matches_ordinary_logits_topk():
    """Public capacity replay agrees with the ordinary full scorer."""
    from flashinfer import fp8_paged_mqa_logits

    q, kv, scale, weights, _, _ = _inputs(q_rows=1024, kv_rows=2048)
    cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device=q.device, dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(
        kv.shape[0] // 128, device=q.device, dtype=torch.int32
    ).unsqueeze(0)
    batches = q.shape[0] // 16
    ordinary = fp8_paged_mqa_logits(
        q.reshape(batches, 16, 8, 128),
        fused,
        weights,
        table.expand(batches, -1).contiguous(),
        torch.full((batches,), kv.shape[0], device=q.device, dtype=torch.int32),
        kv.shape[0],
    ).reshape(q.shape[0], -1)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=2048,
        max_chunk_size=q.shape[0],
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table)
    for _ in range(2):
        selected = wrapper.run(q, weights, fused)
        torch.cuda.synchronize()
        assert selected.shape == (q.shape[0], 512) and selected.dtype == torch.int32
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(
            selected,
            [ordinary[r, :end] for r, end in enumerate(_valid_lengths(cu_q, cu_kv))],
            512,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_unified_wrapper_full_strategy_ragged_causal_topk(monkeypatch, top_k):
    """Full and selective strategies share the exact public result contract."""
    from flashinfer import FP8PagedMQATopKWrapper

    inputs = _request_inputs((2, 3), (4096, 4352))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    torch_topk = torch.topk

    def reject_dense_topk(*args, **kwargs):
        raise AssertionError(
            "full strategy must use the native variable-length selector"
        )

    monkeypatch.setattr(torch, "topk", reject_dense_topk)

    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
    )
    first = wrapper.run(
        q=q, kv_fused=kv_fused, weights=weights, block_table=block_table
    )
    # Unwritten causal tails are not part of the selector's input domain.
    # Poison them to catch accidental dense selection after future refactors.
    invalid = (
        torch.arange(wrapper._full_scores.shape[1], device=q.device)[None, :]
        >= wrapper._full_row_ends[:, None]
    )
    wrapper._full_scores.masked_fill_(invalid, float("inf"))
    second = wrapper.run(
        q=q, kv_fused=kv_fused, weights=weights, block_table=block_table
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            q=q, kv_fused=kv_fused, weights=weights, block_table=block_table
        )
    graph.replay()
    torch.cuda.synchronize()
    monkeypatch.setattr(torch, "topk", torch_topk)

    assert first.data_ptr() == second.data_ptr()
    assert first.data_ptr() == captured.data_ptr()
    assert first.shape == (q.shape[0], top_k)
    assert first.dtype == torch.int32
    _check_selection(first, oracle, top_k)

    # Retiring a full plan must release its graph without cyclic collection,
    # which may otherwise destroy graph-owned storage during another capture.
    owner = weakref.ref(wrapper)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        del wrapper
        assert owner() is None
    finally:
        if gc_enabled:
            gc.enable()


@_requires_selective_sm100
def test_full_wrapper_plan_rejects_capture_without_invalidating_plan():
    """Reject host-reading full planning before capture or plan state is changed."""
    inputs = _request_inputs((2,), (1024,))
    q, _, _, weights, qo_indptr, kv_indptr = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    wrapper.plan(q, fused, weights, qo_indptr, kv_indptr, table)
    oracle = _oracle_rows(inputs)
    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(graph),
        pytest.raises(
            RuntimeError,
            match="full strategy plan must be called outside CUDA graph capture",
        ),
    ):
        wrapper.plan(q, fused, weights, qo_indptr, kv_indptr, table)
    selected = wrapper.run(q, weights, fused, block_table=table)
    torch.cuda.synchronize()
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_full_wrapper_gvr2_workspace_capture_and_replan(monkeypatch):
    """The full prefill route owns scratch across eager and graph streams."""
    from flashinfer.topk_varlen.kernels import gvr2_topk_host

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    q_rows, kv_rows = 8192, 65536
    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=kv_rows)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scale, weights, cu_q, cu_kv))
    calls = []
    real_run = gvr2_topk_host.run_varlen

    def observe_gvr2(*args, **kwargs):
        capturing = torch.cuda.is_current_stream_capturing()
        workspace = kwargs.get("workspace")
        calls.append((capturing, workspace))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(gvr2_topk_host, "run_varlen", observe_gvr2)
    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
    # Capture records operations without executing them. Preserve the eager
    # plan result before any public run can overwrite it by graph replay.
    planned_output = wrapper._selected.clone()
    slab = wrapper._full_gvr2_workspace
    assert slab is not None and slab.dtype == torch.int32
    assert slab.numel() * slab.element_size() == gvr2_topk_host.workspace_bytes()
    assert slab.device == q.device and slab.data_ptr() % 16 == 0
    assert any(capturing for capturing, _ in calls)
    assert any(not capturing for capturing, _ in calls)
    assert all(workspace is slab for _, workspace in calls)

    def check_rows(output, logical_kv, logical_scale):
        ends = kv_rows - q_rows + torch.arange(q_rows, device=q.device) + 1
        assert ((output >= 0) & (output < ends[:, None])).all()
        # The faithful capture geometry is large. Independent score/TopK
        # checks cover representative boundary/interior rows; small ragged
        # tests separately cover every row and all supported output widths.
        for row in (0, 15, 16, q_rows // 2, q_rows - 1):
            end = kv_rows - q_rows + row + 1
            dots = q[row].float() @ logical_kv[:end].float().T
            scores = (torch.relu(dots) * weights[row, :, None]).sum(0)
            scores *= logical_scale[:end]
            indices = output[row].long()
            assert indices.unique().numel() == 512
            torch.testing.assert_close(
                scores[indices].sort().values,
                scores.topk(512).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )

    check_rows(planned_output, kv, scale)
    with pytest.raises(ValueError, match="address-stable plan"):
        wrapper.run(q=q.view_as(q), kv_fused=fused, weights=weights, block_table=table)
    stream = torch.cuda.Stream(device=q.device)
    stream.wait_stream(torch.cuda.current_stream(q.device))
    with torch.cuda.stream(stream):
        first = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    stream.synchronize()
    check_rows(first, kv, scale)
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer, stream=stream):
        captured = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    outer.replay()
    torch.cuda.synchronize()
    assert captured.data_ptr() == first.data_ptr()
    check_rows(captured, kv, scale)

    q.copy_(q.float().roll(1, dims=1).to(q.dtype))
    weights.copy_(weights.flip(1))
    kv.copy_(kv.float().roll(1, dims=0).to(kv.dtype))
    scale.copy_(scale.roll(1))
    fused.copy_(_fuse_paged_kv(kv, scale).flip(0))
    table.copy_(table.flip(1))
    outer.replay()
    torch.cuda.synchronize()
    reversed_kv = kv.float().view(-1, 128, kv.shape[1]).flip(0).reshape_as(kv)
    reversed_scale = scale.view(-1, 128).flip(0).reshape_as(scale)
    check_rows(captured, reversed_kv, reversed_scale)
    assert wrapper._full_gvr2_workspace is slab
    assert sum(capturing for capturing, _ in calls) >= 2
    assert all(workspace is slab for _, workspace in calls)

    one_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    one_query = q[:1]
    one_weights = weights[:1]
    other = FP8PagedMQATopKWrapper(strategy="full")
    other.plan(one_query, fused, one_weights, one_q, cu_kv, table)
    assert other._full_gvr2_workspace.data_ptr() != slab.data_ptr()
    other_slab = other._full_gvr2_workspace
    with pytest.raises(ValueError, match="requires top_k in"):
        other.plan(one_query, fused, one_weights, one_q, cu_kv, table, top_k=8)
    assert other._full_gvr2_workspace is other_slab
    assert other.run(
        q=one_query, kv_fused=fused, weights=one_weights, block_table=table
    ).shape == (1, 512)

    def fail_capture():
        raise RuntimeError("injected full GVR capture failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "CUDAGraph", fail_capture)
        with pytest.raises(RuntimeError, match="injected full GVR capture failure"):
            wrapper.plan(one_query, fused, one_weights, one_q, cu_kv, table)
    assert wrapper._full_gvr2_workspace is None
    with pytest.raises(RuntimeError, match="plan must be called"):
        wrapper.run(q=one_query, kv_fused=fused, weights=one_weights, block_table=table)
    wrapper.plan(one_query, fused, one_weights, one_q, cu_kv, table)
    assert wrapper._full_gvr2_workspace is not None
    recovered = wrapper.run(
        q=one_query, kv_fused=fused, weights=one_weights, block_table=table
    )
    torch.cuda.synchronize()
    assert recovered.shape == (1, 512)


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [1, 8, 17, 32, 128, 511, 512, 2048])
def test_unified_wrapper_ragged_batch_and_top_k(top_k):
    """Public TopK supports ragged requests across the selected K domain."""
    inputs = _request_inputs((2, 3), (7168, 9216))
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        top_k=top_k,
        max_context_length=9216,
        max_chunk_size=5,
        max_batch_size=2,
        kv_layout=fused,
    )
    wrapper.plan(cu_q, cu_kv, table)
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("strategy", ["full", "selective"])
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_unified_wrapper_underfilled_causal_rows(strategy, top_k):
    """Every short row keeps all keys and pads its suffix with -1."""
    from flashinfer.trace.templates.attn_scores import (
        fp8_paged_mqa_topk_trace,
        fp8_paged_mqa_selective_topk_trace,
    )

    inputs = _request_inputs((2, 3), (2, 4))
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    if strategy == "selective":
        wrapper = FP8PagedMQATopKWrapper(
            strategy=strategy,
            top_k=top_k,
            max_context_length=4,
            max_chunk_size=5,
            max_batch_size=2,
            kv_layout=fused,
        )
        wrapper.plan(cu_q, cu_kv, table)
        selected = wrapper.run(q, weights, fused)
        template = fp8_paged_mqa_selective_topk_trace
        trace_args = dict(
            q=q,
            weights=weights,
            qo_indptr=cu_q,
            kv_indptr=cu_kv,
            kv_fused=fused,
            block_table=table,
            top_k=top_k,
        )
    else:
        wrapper = FP8PagedMQATopKWrapper(strategy=strategy)
        wrapper.plan(q, fused, weights, cu_q, cu_kv, table, top_k=top_k)
        selected = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
        template = fp8_paged_mqa_topk_trace
        trace_args = dict(
            q=q,
            kv_fused=fused,
            weights=weights,
            block_table=table,
            qo_indptr=cu_q,
            kv_indptr=cu_kv,
            top_k=top_k,
        )
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    reference = template.reference(**trace_args)
    assert template.check(reference, selected, **trace_args)
    bad = selected.clone()
    bad[0, _valid_lengths(cu_q, cu_kv)[0]] = 0
    assert not template.check(reference, bad, **trace_args)


@_requires_selective_sm100
def test_engine_compact_repair_queue_changes_cuda_graph():
    """Replaying a growing then empty repair queue cannot consume stale TopK rows."""
    from flashinfer import FP8PagedMQATopKWrapper

    rows, kv_rows = 33, 32768
    q, kv, scale, weights, _, _ = _inputs(q_rows=rows, kv_rows=kv_rows)
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scale, weights, cu_q, cu_kv))
    payload = q, fused, weights, table
    full = FP8PagedMQATopKWrapper(strategy="full")
    full.plan(q, fused, weights, cu_q, cu_kv, table)
    reference = full.run(
        payload[0], payload[2], payload[1], block_table=payload[3]
    ).long()
    scores = full._full_scores[:, :kv_rows]
    expected = scores.gather(1, reference).sort(1).values
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
    assert wrapper._compact_repair

    force_rows = torch.zeros(1, device="cuda", dtype=torch.int32)
    forced_thresholds = torch.full_like(wrapper._thresholds, float("inf"))
    row_ids = torch.arange(rows, device="cuda")
    threshold, threshold_args = wrapper._sample_threshold_launch

    def controlled_threshold(*args):
        threshold(*args)
        torch.where(
            row_ids < force_rows,
            forced_thresholds,
            wrapper._thresholds,
            out=wrapper._thresholds,
        )

    wrapper._sample_threshold_launch = controlled_threshold, threshold_args
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        selected = wrapper.run(*payload)
    counts = []
    for forced in (0, 1, rows, 0):
        force_rows.fill_(forced)
        graph.replay()
        torch.cuda.synchronize()
        counts.append(int(wrapper._repair_active_count.item()))
        assert counts[-1] >= forced
        assert int(wrapper._exact_error.item()) == 0
        indices = selected.long()
        assert ((indices >= 0) & (indices < wrapper._row_ends[:, None])).all()
        ordered = indices.sort(1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()
        torch.testing.assert_close(
            scores.gather(1, indices).sort(1).values, expected, atol=1e-4, rtol=1e-4
        )
    assert counts[0] == counts[-1] == 0


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("heads", [8, 64])
def test_engine_underfilled_causal_rows_skip_repair(top_k, heads):
    """Complete short rows skip both selected repair layouts."""

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=16, kv_rows=128, num_heads=heads
    )
    cu_q = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=heads,
        num_sms=4,
    )
    _assert_repair_position_storage(wrapper)
    for _ in range(3):
        # Every row is shorter than TopK and publishes all of its keys, so no
        # row is flagged and the conditional repair body must not run: the
        # stale ranks of a previous compact turn stay untouched.
        wrapper._ranks.fill_(top_k)
        selected = wrapper.run(q, kv_fused, weights, block_table).long()
        torch.cuda.synchronize()
        assert bool((wrapper._ranks == top_k).all())

    assert wrapper._compact_repair == (heads == 8)
    assert int(wrapper._exact_error.item()) == 0
    assert torch.count_nonzero(wrapper._repair_flags).item() == 0
    if heads == 8:
        assert int(wrapper._second_active_count.item()) == 0
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        indices = selected[row, :valid]
        assert ((indices >= 0) & (indices < valid)).all()
        assert (selected[row, valid:] == -1).all()
        assert torch.equal(
            torch.unique(indices).cpu(), torch.arange(valid, dtype=torch.int64)
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "query_tile"),
    [(128, 1), (64, 2), (32, 4), (8, 16)],
)
def test_engine_segmented_publication_query_tiles(num_heads, query_tile):
    """Q1/Q2/Q4/Q16 use the same segmented public candidate path."""
    from flashinfer import fp8_paged_mqa_logits

    q_rows = query_tile + 1
    kv_rows = 131072
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        block_table,
        torch.tensor([kv_rows], device="cuda", dtype=torch.int32),
        kv_rows,
    )

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=2,
    )
    assert wrapper._candidate_prepared.query_tile == query_tile
    warp_ranked = num_heads == 64
    assert wrapper._candidate_schedule.publication == (
        "warp_striped" if warp_ranked else "lane_local"
    )
    assert wrapper._candidate_counts.shape[1] == (5 if warp_ranked else 33)
    assert wrapper._candidate_schedule.sample_count_segments == 32
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if wrapper._padded_rows > q_rows:
        assert (wrapper._candidate_counts[q_rows:] == 0).all()
    for row in range(q_rows):
        row_end = kv_rows - q_rows + row + 1
        assert ((selected[row] >= 0) & (selected[row] < row_end)).all()
        actual_scores = ordinary[row, selected[row]]
        expected_scores = torch.topk(ordinary[row, :row_end], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "q_rows", "expected_split"),
    [(8, 32, 1), (8, 64, 2), (64, 6, 1), (64, 12, 1)],
)
def test_engine_selects_split_kv_from_sm_waves(num_heads, q_rows, expected_split):
    """The public plan selects the smallest layout with best SM-wave use."""

    kv_rows = 4096
    inputs = _request_inputs((q_rows,), (kv_rows,), num_heads)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == expected_split
    assert (
        wrapper._candidate_schedule.query_tile == wrapper._candidate_prepared.query_tile
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    warp_ranked = num_heads == 64
    # Split-1 H64 prefill uses the H64 candidate variant's four warp segments.
    assert wrapper._candidate_schedule.h64_candidate == (
        warp_ranked and expected_split == 1
    )
    assert wrapper._candidate_schedule.publication == (
        "warp_striped" if warp_ranked else "lane_local"
    )
    assert wrapper._candidate_counts.shape[1] == (
        1 + (4 if warp_ranked else 32) * expected_split
    )
    assert wrapper._candidate_schedule.sample_count_segments == 32 * expected_split
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("num_heads", "kv_rows", "top_k", "expected_split", "force_repair"),
    (
        (8, 4096, 8, 16, False),
        (64, 8192, 17, 32, True),
        (8, 16384, 512, 64, False),
        (8, 16384, 512, 64, True),
    ),
)
def test_engine_automatic_high_split_cuda_graph(
    num_heads, kv_rows, top_k, expected_split, force_repair
):
    """Public decode planning exercises every pooled split and exact repair."""

    inputs = _inputs(q_rows=1, kv_rows=kv_rows, num_heads=num_heads, head_dim=128)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    if force_repair:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        top_k=top_k,
        num_heads=num_heads,
        head_dim=128,
        num_sms=148,
    )
    assert wrapper._candidate_schedule.split_kv == expected_split
    assert wrapper._candidate_schedule.publication == "warp_pooled"
    assert wrapper._candidate_counts.shape[1] == expected_split + 1
    if num_heads == 8:
        assert wrapper._candidate_schedule.logical_n == 8
        for prepared in (
            wrapper._candidate_prepared,
            wrapper._sample_prepared,
            wrapper._repair_prepared,
            wrapper._prepared,
        ):
            assert prepared.query_tile == 1
            assert prepared.num_epi_subtiles == 2

    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert int(wrapper._exact_error.item()) == 0
    if force_repair:
        assert (wrapper._repair_flags != 0).all()
    indices = captured[0].long()
    scores = oracle[0]
    assert ((indices >= 0) & (indices < scores.numel())).all()
    assert indices.unique().numel() == top_k
    torch.testing.assert_close(
        scores[indices].sort().values,
        torch.topk(scores, top_k).values.sort().values,
        atol=5e-5,
        rtol=1e-5,
    )


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", (32, 64, 128))
@pytest.mark.parametrize("math_mode", ("relu_strict", "relu_fast"))
def test_engine_decode_uses_n8_request_tile(page_size, math_mode):
    """Packed H8 Q1 requests use unpadded N8 in eager and graph execution."""

    batches = 4
    kv_per_request = 2048
    inputs = _request_inputs((1,) * batches, (kv_per_request,) * batches)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=page_size)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase(math_mode=math_mode)
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=4,
    )
    assert wrapper._candidate_schedule.logical_n == 8
    for prepared in (
        wrapper._candidate_prepared,
        wrapper._sample_prepared,
        wrapper._repair_prepared,
        wrapper._prepared,
    ):
        assert prepared.query_tile == 1
        assert prepared.num_epi_subtiles == 2
        assert prepared.num_umma_stages == 2
    eager = wrapper.run(q, kv_fused, weights, block_table).clone().long()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    for selected in (eager, captured.long()):
        _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_engine_split_kv_handles_heterogeneous_batch():
    """Ragged request costs share one automatic candidate publication plan."""

    inputs = _request_inputs((16, 48), (2048, 4096))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_sms=3,
    )
    assert wrapper._candidate_prepared.split_kv == 2
    assert wrapper._candidate_schedule.publication == "lane_local"
    assert wrapper._candidate_counts.shape[1] == 65
    selected = wrapper.run(q, kv_fused, weights, block_table).long()
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    _check_selection(selected, oracle, 512)


@_requires_selective_sm100
def test_engine_q2_segmented_overflow_repair_cuda_graph():
    """Inactive store warps retain lifecycle while Q2 repairs exact overflow."""

    q_rows = 3
    kv_rows = 131072
    num_heads = 64
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=q_rows, kv_rows=kv_rows, num_heads=num_heads, head_dim=128
    )
    q.zero_()
    kv.zero_()
    scale.fill_(1.0)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    kv_fused = _fuse_paged_kv(kv, scale)
    block_table = torch.arange(
        kv_rows // 128 - 1, -1, -1, device="cuda", dtype=torch.int32
    ).unsqueeze(0)

    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        num_heads=num_heads,
        head_dim=128,
        num_sms=4,
    )
    eager = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, kv_fused, weights, block_table)
    graph.replay()
    torch.cuda.synchronize()

    assert eager.data_ptr() == captured.data_ptr()
    assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    assert (wrapper._candidate_counts[q_rows:] == 0).all()
    row_ends = torch.arange(
        kv_rows - q_rows + 1, kv_rows + 1, device="cuda", dtype=torch.int32
    )
    selected = captured
    assert ((selected >= 0) & (selected < row_ends[:, None])).all()
    assert all(row.unique().numel() == 512 for row in selected)


@_requires_selective_sm100
@pytest.mark.parametrize("force_ties", [False, True])
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict", None])
def test_engine_sampled_common_and_exact_repair(force_ties, math_mode):
    """The public run owns sampling and fail-closed repair on arbitrary pages."""
    from flashinfer import fp8_paged_mqa_logits

    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(q_rows=16, kv_rows=8192)
    if force_ties:
        q.zero_()
        kv.zero_()
        scale.fill_(1.0)
    cu_q = torch.tensor([0, q.shape[0]], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    ordinary = fp8_paged_mqa_logits(
        q.unsqueeze(0),
        kv_fused,
        weights,
        block_table,
        torch.tensor([kv.shape[0]], device="cuda", dtype=torch.int32),
        kv.shape[0],
        **({} if math_mode is None else {"math_mode": math_mode}),
    )

    wrapper = SelectiveEngineCase(
        **{} if math_mode is None else {"math_mode": math_mode}
    )
    assert wrapper.math_mode == ("relu_strict" if math_mode is None else math_mode)
    wrapper.plan(q, kv_fused, weights, cu_q, cu_kv, block_table, num_sms=4)
    selected = wrapper.run(q, kv_fused, weights, block_table)
    torch.cuda.synchronize()

    assert int(wrapper._exact_error.item()) == 0
    if force_ties:
        assert (wrapper._repair_flags != 0).all()
    for row, valid in enumerate(_valid_lengths(cu_q, cu_kv)):
        actual_indices = selected[row].long()
        assert ((actual_indices >= 0) & (actual_indices < valid)).all()
        assert actual_indices.unique().numel() == 512
        actual_scores = ordinary[row, actual_indices]
        expected_scores = torch.topk(ordinary[row, :valid], 512).values
        torch.testing.assert_close(
            actual_scores.sort().values,
            expected_scores.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("ties", [True, False], ids=["tied", "distinct"])
@pytest.mark.parametrize("q_lengths", [[2], [2, 2]], ids=["compact", "standard"])
def test_unified_wrapper_exact_repair_negative_threshold(q_lengths, ties):
    """Exact repair keys a negative TopK threshold in both repair layouts."""

    # Positive dots with negative weights give negative scores. Tied rows
    # overflow both bounded passes; varying scales also tests strict ordering.
    inputs = _request_inputs(q_lengths, [9001] * len(q_lengths))
    q, kv, scale, weights, cu_q, cu_kv = inputs
    q.copy_(torch.ones(q.shape, device=q.device).to(q.dtype))
    kv.copy_(torch.ones(kv.shape, device=kv.device).to(kv.dtype))
    scale.fill_(1.0)
    if not ties:
        # Distinct negative scores exercise ordering, not just sign mapping.
        scale.copy_(torch.linspace(1.0, 2.0, scale.numel(), device=scale.device))
    weights.fill_(-1.0)
    kv_fused, block_table = _paged_inputs(inputs)

    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=9001,
        max_chunk_size=q.shape[0],
        max_batch_size=len(q_lengths),
        kv_layout=kv_fused,
    )
    wrapper.prepare_pool(kv_fused)
    engine = wrapper._capacity_plan._engine
    assert engine._compact_repair == (len(q_lengths) == 1)
    wrapper.plan(cu_q, cu_kv, block_table, num_query_rows=q.shape[0])
    selected = wrapper.run(q, weights, kv_fused)
    torch.cuda.synchronize()

    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    if ties:
        assert int(engine._second_repair_flags.count_nonzero()) == q.shape[0]
    expected = _oracle_rows(inputs)
    assert all(bool((row < 0).all()) for row in expected)
    _check_selection(selected, expected, 512)
    if not ties:
        outside = torch.topk(expected[0], 513).indices[-1]
        corrupted = selected.clone()
        corrupted[0, 0] = outside
        with pytest.raises(AssertionError):
            _check_selection(corrupted, expected, 512)


@_requires_selective_sm100
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_sample_candidate_overflow_and_exact_modes(math_mode):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
        _shared_engine_source_token,
    )

    inputs = _inputs()
    q, kv, scale, weights, cu_q, cu_kv = inputs
    immutable = tuple(tensor.clone() for tensor in (q, kv, scale, weights))
    prepared = _prepare(inputs, math_mode=math_mode)
    rows = q.shape[0]
    width = 257
    sampled = torch.full((32, width), float("nan"), device="cuda")
    unused = torch.empty((32, width + 1), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="_SelectiveLogitsMode"):
        _launch("sample", width, sampled, unused, prepared, inputs)
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches
    cache_key = (
        _SelectiveLogitsMode.SAMPLE,
        width,
        0,
        1,
        False,
        prepared.math_mode,
        _shared_engine_source_token(),
    )
    cached_sample_launch = prepared._launches[cache_key]
    unused.zero_()
    _launch(_SelectiveLogitsMode.SAMPLE, width, sampled, unused, prepared, inputs)
    assert prepared._launches[cache_key] is cached_sample_launch
    torch.cuda.synchronize()

    valid_lengths = _valid_lengths(cu_q, cu_kv)
    assert valid_lengths[0] == 132
    assert valid_lengths[-1] == 255
    assert torch.isfinite(sampled[0, :132]).all()
    assert torch.isfinite(sampled[rows - 1, :255]).all()
    assert torch.isnan(sampled[0, 132:]).all()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )

    thresholds = torch.stack(
        [sampled[row, valid_lengths[row] // 2] for row in range(rows)]
    )
    capacity = width
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    candidate_cache_key = (
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        0,
        1,
        False,
        prepared.math_mode,
        _shared_engine_source_token(),
    )
    cached_candidate_launch = prepared._launches[candidate_cache_key]
    thresholds.add_(0.0)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        index_counts,
        prepared,
        inputs,
        thresholds,
    )
    assert prepared._launches[candidate_cache_key] is cached_candidate_launch
    torch.cuda.synchronize()
    for row, valid in enumerate(valid_lengths):
        expected = torch.nonzero(sampled[row, :valid] >= thresholds[row]).flatten()
        count = int(index_counts[row, 0])
        packed = values[row, :count]
        actual_indices = (packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        actual_indices = actual_indices[order]
        assert count == expected.numel(), (
            f"row {row}: published {count}, expected {expected.numel()}, "
            f"unique={actual_indices.unique().numel()}"
        )
        torch.testing.assert_close(
            actual_indices, expected.to(torch.int32).sort().values
        )
        actual_score_bits = ((packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        expected_score_bits = sampled[row, actual_indices.long()].view(torch.int32)
        torch.testing.assert_close(actual_score_bits, expected_score_bits)

    tiny_values = torch.full((32, 5), -7, device="cuda", dtype=torch.int64)
    tiny_indices = torch.full((32, 1), -11, device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.REPAIR,
        3,
        tiny_values,
        tiny_indices,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    assert torch.equal(tiny_indices[:rows, 0], index_counts[:rows, 0])
    assert (tiny_indices[:rows, 0] > 3).any()
    assert (tiny_values[:, 3:] == -7).all()
    lengths = torch.full((rows + 2,), -13, device="cuda", dtype=torch.int32)
    repair_flags = torch.full((rows + 2,), -17, device="cuda", dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        tiny_indices,
        tiny_values,
        lengths,
        repair_flags,
        rows=rows,
        capacity=3,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths[:rows], tiny_indices[:rows, 0].clamp(max=3))
    assert (repair_flags[:rows] == 1).all()
    assert (lengths[rows:] == -13).all()
    assert (repair_flags[rows:] == -17).all()

    # Bank 0 holds > T and bank 1 holds == T. Banks taller than the 32
    # padded rows check the epilogue's `out_row + bank_rows` addressing for
    # every row and leave both padded tails untouched.
    bank_rows = 40
    exact_values = torch.full(
        (2 * bank_rows, width), -7, device="cuda", dtype=torch.int32
    )
    exact_counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        width,
        exact_values,
        exact_counts,
        prepared,
        inputs,
        thresholds,
    )
    torch.cuda.synchronize()
    for bank, compare in ((0, torch.gt), (1, torch.eq)):
        for row, valid in enumerate(valid_lengths):
            expected = torch.nonzero(
                compare(sampled[row, :valid], thresholds[row])
            ).flatten()
            count = int(exact_counts[bank * bank_rows + row, 0])
            actual = exact_values[bank * bank_rows + row, :count].sort().values
            torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)
        tail = slice(bank * bank_rows + rows, (bank + 1) * bank_rows)
        assert (exact_counts[tail] == 0).all()
        assert (exact_values[tail] == -7).all()

    with pytest.raises(ValueError, match="too narrow"):
        _launch(
            _SelectiveLogitsMode.SAMPLE,
            width,
            sampled,
            torch.empty((32, 0), device="cuda", dtype=torch.int32),
            prepared,
            inputs,
        )
    for before, after in zip(immutable, (q, kv, scale, weights), strict=True):
        torch.testing.assert_close(after, before)


@_requires_selective_sm100
@pytest.mark.parametrize("shift", [24, 16, 8, 0])
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict"])
def test_radix_histogram_matches_sampled_score_bits(shift, math_mode):
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, math_mode=math_mode)
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()

    bits = sampled[0].view(torch.int32)
    ordered_i32 = bits ^ torch.where(
        bits < 0,
        torch.full_like(bits, -1),
        torch.full_like(bits, -2147483648),
    )
    ordered = ordered_i32.to(torch.int64) & 0xFFFFFFFF
    if shift == 24:
        prefix = torch.zeros(1, device="cuda", dtype=torch.int32)
        selected = ordered
    else:
        prefix = (ordered[0] >> (shift + 8)).to(torch.int32).reshape(1)
        selected = ordered[(ordered >> (shift + 8)) == prefix[0]]
    expected = torch.bincount(((selected >> shift) & 0xFF).long(), minlength=256)

    hist = torch.empty((16, 256), device="cuda", dtype=torch.int32)
    dummy = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.RADIX_HISTOGRAM,
        1,
        dummy,
        hist,
        prepared,
        inputs,
        prefix,
        radix_shift=shift,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(hist[0], expected.to(torch.int32))


@_requires_selective_sm100
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize(
    "mode", [_SelectiveLogitsMode.SAMPLE, _SelectiveLogitsMode.RADIX_HISTOGRAM]
)
def test_selective_scale_stage_survives_persistent_epilogue(num_heads, mode):
    """Non-publisher epilogues must finish reading scales before stage reuse."""
    q_rows, kv_rows = 33, 32768
    q, kv, scale, weights, _, _ = _inputs(q_rows, kv_rows, num_heads)
    q.copy_(torch.ones(q.shape, device="cuda").to(q.dtype))
    kv.copy_(torch.ones(kv.shape, device="cuda").to(kv.dtype))
    weights.fill_(1)
    # Every K128 tile has a different exact power-of-two scale. A reused
    # stage must not silently substitute a later tile's scale in the epilogue.
    scale.copy_(torch.pow(2.0, (torch.arange(kv_rows, device="cuda") // 128) % 7 - 3))
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv_rows], device="cuda", dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    prepared = _prepare(inputs, num_heads=num_heads, num_sms=4)
    reference = torch.matmul(q[0].float(), kv.float().T).relu().sum(0) * scale
    ends = _valid_lengths(cu_q, cu_kv)
    is_sample = mode == _SelectiveLogitsMode.SAMPLE
    capacity = kv_rows if is_sample else 1
    values = torch.empty(
        (48, capacity), device="cuda", dtype=torch.float32 if is_sample else torch.int32
    )
    counts = torch.empty((48, 256), device="cuda", dtype=torch.int32)
    prefixes = torch.zeros(q_rows, device="cuda", dtype=torch.int32)

    def run():
        _launch(
            mode, capacity, values, counts, prepared, inputs, prefixes, radix_shift=24
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for row, end in enumerate(ends):
            if mode == _SelectiveLogitsMode.SAMPLE:
                torch.testing.assert_close(
                    values[row, :end], reference[:end], rtol=0, atol=0
                )
            else:
                keys = reference[:end].view(torch.int32).to(torch.int64) ^ 0x80000000
                expected = torch.bincount((keys >> 24) & 255, minlength=256).int()
                torch.testing.assert_close(counts[row], expected, rtol=0, atol=0)


@_requires_selective_sm100
def test_explicit_nonzero_row_bounds_predicate_every_store():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.full((16, 130), float("nan"), device="cuda")
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, sampled, unused, prepared, inputs)
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :17]).all()
    assert torch.isfinite(sampled[0, 17:119]).all()
    assert torch.isnan(sampled[0, 119:]).all()


@_requires_selective_sm100
def test_explicit_atomic_modes_split_one_q16_tile_across_ctas():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=1024)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 1024, device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        num_sms=4,
        kv_chunk_size=256,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        prepared.schedule_meta.cpu(),
        torch.tensor([[0, 0], [0, 1], [0, 2], [0, 3], [1, 0]], dtype=torch.int32),
    )

    capacity = 1024
    values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    lengths = torch.empty(1, device="cuda", dtype=torch.int32)
    flags = torch.empty(1, device="cuda", dtype=torch.int32)
    threshold = torch.full((1,), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.REPAIR,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts,
        values,
        lengths,
        flags,
        rows=1,
        capacity=capacity,
    )
    torch.cuda.synchronize()
    assert int(lengths[0]) == capacity
    actual_indices = (values[0, :capacity] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices, torch.arange(capacity, device="cuda", dtype=torch.int64)
    )

    segmented_counts = torch.empty((16, 5), device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="requires exclusive query tiles"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            segmented_counts,
            prepared,
            inputs,
            threshold,
        )


def _append_candidate_segments(inputs, prepared, threshold, *, capacity, count_columns):
    """Launch the H8 component fixture with fresh, 16-row-padded storage."""
    padded_rows = math.ceil(inputs[0].shape[0] / 16) * 16
    values = torch.empty(
        (padded_rows, capacity), device=inputs[0].device, dtype=torch.int64
    )
    counts = torch.empty(
        (padded_rows, count_columns), device=inputs[0].device, dtype=torch.int32
    )
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    return values, counts


def _finalize_candidate_segments(values, counts, *, rows):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    lengths = torch.empty(rows, device=values.device, dtype=torch.int32)
    flags = torch.empty(rows, device=values.device, dtype=torch.int32)
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=rows, capacity=values.shape[1]
    )
    torch.cuda.synchronize()
    return lengths, flags


@_requires_selective_sm100
@pytest.mark.parametrize(
    "split_kv,q_rows,kv_rows,capacity,overflow_capacity",
    [(2, 17, 1025, 6144, 256), (4, 17, 1025, 8192, 512), (4, 1, 641, 8192, None)],
)
def test_split_kv_uses_disjoint_segments(
    split_kv, q_rows, kv_rows, capacity, overflow_capacity
):
    inputs = _inputs(q_rows=q_rows, kv_rows=kv_rows)
    prepared = _prepare(inputs, num_sms=4, split_kv=split_kv)
    padded_rows = math.ceil(q_rows / 16) * 16
    columns = 4 * split_kv + 1
    threshold = torch.full((q_rows,), float("-inf"), device="cuda")
    expected = torch.tensor(
        _valid_lengths(inputs[4], inputs[5]), device="cuda", dtype=torch.int32
    )

    def append_and_finalize(capacity):
        values, counts = _append_candidate_segments(
            inputs, prepared, threshold, capacity=capacity, count_columns=columns
        )
        raw_counts = counts[:q_rows, 1:].clone()
        lengths, flags = _finalize_candidate_segments(values, counts, rows=q_rows)
        torch.testing.assert_close(raw_counts.sum(dim=1, dtype=torch.int32), expected)
        return values, counts, lengths, flags, raw_counts

    values, counts, lengths, flags, raw_counts = append_and_finalize(capacity)
    torch.testing.assert_close(lengths, expected)
    # The first request from _inputs has only 132 keys, below the default K512.
    # Complete short rows still require the finalizer's underfill flag.
    torch.testing.assert_close(flags, (expected < 512).to(torch.int32))
    owner_counts = raw_counts.reshape(q_rows, split_kv, 4).sum(dim=2)
    for row, length in enumerate(expected.tolist()):
        active_owners = min(math.ceil(length / 256), split_kv)
        assert (owner_counts[row, :active_owners] > 0).all()
        assert (owner_counts[row, active_owners:] == 0).all()
        actual = (values[row, :length] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual, torch.arange(length, device="cuda", dtype=torch.int64)
        )

    wrong_counts = torch.empty(
        (padded_rows, 2 * split_kv + 1), device="cuda", dtype=torch.int32
    )
    with pytest.raises(ValueError, match=f"requires Qx{columns}"):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            wrong_counts,
            prepared,
            inputs,
            threshold,
        )

    if overflow_capacity is None:
        return  # The 641-key case specifically checks three unequal active owners.
    _, _, lengths, flags, _ = append_and_finalize(overflow_capacity)
    per_owner = overflow_capacity // split_kv
    retained = []
    for length in expected.tolist():
        pairs = math.ceil(length / 256)
        owners = min(pairs, split_kv)
        total = 0
        for owner in range(owners):
            first = math.ceil(owner * pairs / owners) * 256
            last = min(math.ceil((owner + 1) * pairs / owners) * 256, length)
            total += min(max(last - first, 0), per_owner)
        retained.append(total)
    torch.testing.assert_close(
        lengths, torch.tensor(retained, device="cuda", dtype=torch.int32)
    )
    assert (flags == 1).all()


@_requires_selective_sm100
def test_split_kv_eight_uses_disjoint_qx33_warp_segments():
    inputs = _inputs(q_rows=1, kv_rows=2049)
    _q, _kv, _scale, _weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, num_sms=8, split_kv=8)
    capacity = 6144
    threshold = torch.full((1,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=33
    )
    raw_counts = counts[0, 1:].clone()
    lengths, _flags = _finalize_candidate_segments(values, counts, rows=1)

    expected_length = _valid_lengths(cu_q, cu_kv)[0]
    assert int(raw_counts.sum()) == expected_length
    assert all(
        int(raw_counts[split_idx * 4 : (split_idx + 1) * 4].sum()) > 0
        for split_idx in range(8)
    )
    assert int(lengths[0]) == expected_length
    actual_indices = (values[0, :expected_length] & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(expected_length, device="cuda", dtype=torch.int64),
    )


@_requires_selective_sm100
@pytest.mark.parametrize("split_kv", (16, 32, 64))
def test_high_split_kv_pools_one_ballot_segment_per_owner(split_kv: int):
    inputs = _inputs(q_rows=1, kv_rows=16385)
    prepared = _prepare(inputs, num_sms=64, split_kv=split_kv)
    capacity = 32768
    threshold = torch.full((1,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=split_kv + 1
    )
    torch.cuda.synchronize()

    raw_counts = counts[0, 1:].to(torch.int64)
    assert (raw_counts > 0).all()
    assert int(raw_counts.sum()) == 16385
    segment_capacity = capacity // split_kv
    published = torch.cat(
        [
            values[0, segment * segment_capacity :][: int(raw_counts[segment].item())]
            for segment in range(split_kv)
        ]
    )
    actual_indices = (published & 0xFFFFFFFF).sort().values
    torch.testing.assert_close(
        actual_indices,
        torch.arange(16385, device="cuda", dtype=torch.int64),
    )


@_requires_selective_sm100
def test_split_kv_eight_resets_warp_cursors_between_persistent_work_items():
    batches = 32
    kv_per_request = 2049
    q, kv, scale, weights, _cu_q, _cu_kv = _inputs(
        q_rows=batches, kv_rows=batches * kv_per_request
    )
    cu_q = torch.arange(batches + 1, device="cuda", dtype=torch.int32)
    cu_kv = cu_q * kv_per_request
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs, num_sms=16, split_kv=8)
    capacity = 6144
    threshold = torch.full((batches,), float("-inf"), device="cuda")
    values, counts = _append_candidate_segments(
        inputs, prepared, threshold, capacity=capacity, count_columns=33
    )
    lengths, _flags = _finalize_candidate_segments(values, counts, rows=batches)

    assert (lengths == kv_per_request).all()
    expected = torch.arange(kv_per_request, device="cuda", dtype=torch.int64)
    for row in range(batches):
        actual = (values[row, :kv_per_request] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(actual, expected)


@_requires_selective_sm100
def test_all_partial_q16_request_tails_and_packed_origins():
    torch.manual_seed(2)
    q_lengths = torch.arange(1, 16, device="cuda", dtype=torch.int32)
    kv_lengths = torch.full_like(q_lengths, 260)
    kv_lengths[-1] = 257
    cu_q = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), q_lengths))
    cu_kv = torch.cat((torch.zeros(1, device="cuda", dtype=torch.int32), kv_lengths))
    cu_q.cumsum_(0)
    cu_kv.cumsum_(0)
    rows = int(q_lengths.sum())
    kv_rows = int(kv_lengths.sum())
    q = (torch.randn(rows, 8, 128, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    kv_f32 = torch.randn(kv_rows, 128, device="cuda") * 0.1
    amax = kv_f32.abs().amax(dim=1).clamp_min(1e-6)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    kv = (kv_f32 / scale[:, None]).to(torch.float8_e4m3fn)
    weights = torch.randn(rows, 8, device="cuda", dtype=torch.float32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    sampled = torch.full((128, 260), float("nan"), device="cuda")
    counts = torch.empty((128, 1), device="cuda", dtype=torch.int32)
    _launch(_SelectiveLogitsMode.SAMPLE, 260, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()

    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


def _capture_refreshed_sample(
    prepared, inputs, sampled, counts, *, row_starts=None, row_ends=None
):
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    cu_q, cu_kv = inputs[4:]
    capacity = sampled.shape[1]
    _launch(_SelectiveLogitsMode.SAMPLE, capacity, sampled, counts, prepared, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled.fill_(float("nan"))
        _refresh_fp8_mqa_selective_logits(
            prepared, cu_q, cu_kv, row_starts=row_starts, row_ends=row_ends
        )
        _launch(
            _SelectiveLogitsMode.SAMPLE, capacity, sampled, counts, prepared, inputs
        )
    return graph


@_requires_selective_sm100
def test_cuda_graph_refreshes_explicit_bounds_without_reallocation():
    inputs = _inputs(q_rows=1, kv_rows=130)
    starts = torch.tensor([17], device="cuda", dtype=torch.int32)
    ends = torch.tensor([119], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(
        prepared, inputs, sampled, unused, row_starts=starts, row_ends=ends
    )

    starts.fill_(23)
    ends.fill_(77)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0, :23]).all()
    assert torch.isfinite(sampled[0, 23:77]).all()
    assert torch.isnan(sampled[0, 77:]).all()


@_requires_selective_sm100
def test_cuda_graph_refreshes_row_end_base_and_inactive_tail():
    inputs = _inputs(q_rows=2, kv_rows=130)
    starts = torch.zeros(2, device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    prepared = _prepare(
        inputs,
        row_starts=starts,
        row_ends=ends,
        row_end_base=64,
    )
    sampled = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    unused = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(
        prepared, inputs, sampled, unused, row_starts=starts, row_ends=ends
    )

    ends[0] = 0
    ends[1] = 3
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(sampled[0]).all()
    assert torch.isfinite(sampled[1, :67]).all()
    assert torch.isnan(sampled[1, 67:]).all()


@_requires_selective_sm100
def test_cuda_graph_refreshes_request_prefixes_without_reallocation():
    inputs = _inputs()
    cu_q = inputs[4]
    prepared = _prepare(inputs)
    sampled = torch.empty((32, 257), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    graph = _capture_refreshed_sample(prepared, inputs, sampled, counts)

    cu_q[1] = 2
    graph.replay()
    torch.cuda.synchronize()
    for row, expected in enumerate(_oracle_rows(inputs)):
        torch.testing.assert_close(
            sampled[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
        )
        assert torch.isnan(sampled[row, expected.numel() :]).all()


@_requires_selective_sm100
def test_zero_ties_use_bit_exact_equal_policy():
    inputs = list(_inputs(q_rows=1, kv_rows=130))
    inputs[3].zero_()
    inputs = tuple(inputs)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    bank_rows = 16
    values = torch.empty((2 * bank_rows, 130), device="cuda", dtype=torch.int32)
    counts = torch.empty((2 * bank_rows, 1), device="cuda", dtype=torch.int32)
    positive_zero = torch.zeros(1, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        130,
        values,
        counts,
        prepared,
        inputs,
        positive_zero,
    )
    torch.cuda.synchronize()
    assert int(counts[bank_rows, 0]) == 130
    assert int(counts[0, 0]) == 0

    # -0.0 keys strictly below +0.0, so no score ties it; the greater bank
    # holds every score whose ordered key exceeds the -0.0 key.
    negative_zero = torch.full((1,), -0.0, device="cuda", dtype=torch.float32)
    _launch(
        _SelectiveLogitsMode.EXACT,
        130,
        values,
        counts,
        prepared,
        inputs,
        negative_zero,
    )
    torch.cuda.synchronize()
    assert int(counts[bank_rows, 0]) == 0

    def ordered_keys(scores):
        bits = scores.view(torch.int32).long() & 0xFFFFFFFF
        return torch.where(bits >= 1 << 31, bits ^ 0xFFFFFFFF, bits ^ (1 << 31))

    scores = _oracle_rows(inputs)[0]
    expected = torch.nonzero(
        ordered_keys(scores) > ordered_keys(negative_zero.to(scores.device))
    ).flatten()
    count = int(counts[0, 0])
    actual = values[0, :count].sort().values
    torch.testing.assert_close(actual, expected.to(torch.int32).sort().values)


@_requires_selective_sm100
def test_cuda_graph_replays_into_caller_owned_buffers():
    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    threshold = torch.zeros(1, device="cuda", dtype=torch.float32)
    values = torch.empty((16, 130), device="cuda", dtype=torch.int64)
    index_counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        130,
        values,
        index_counts,
        prepared,
        inputs,
        threshold,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            130,
            values,
            index_counts,
            prepared,
            inputs,
            threshold,
        )
    first = None
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        count = int(index_counts[0, 0])
        current = values[0, :count].clone()
        if first is None:
            first = current
        else:
            torch.testing.assert_close(current.sort().values, first.sort().values)
    threshold.fill_(float("inf"))
    graph.replay()
    torch.cuda.synchronize()
    assert int(index_counts[0, 0]) == 0
    threshold.zero_()
    graph.replay()
    torch.cuda.synchronize()
    restored_count = int(index_counts[0, 0])
    torch.testing.assert_close(
        values[0, :restored_count].sort().values,
        first.sort().values,
    )


@_requires_selective_sm100
@pytest.mark.parametrize(
    "metadata_name", ["context_lens", "tile_meta", "schedule_meta"]
)
def test_cached_launch_rebinds_prepared_metadata_storage(metadata_name):
    from flashinfer.attn_scores.selective_logits import (
        _refresh_fp8_mqa_selective_logits,
    )

    inputs = _inputs(q_rows=1, kv_rows=130)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs)
    values = torch.empty((16, 130), device="cuda", dtype=torch.float32)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)

    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()
    cache_key = (
        _SelectiveLogitsMode.SAMPLE,
        130,
        0,
        1,
        False,
        prepared.math_mode,
        sl._shared_engine_source_token(),
    )
    cached = prepared._launches[cache_key]

    metadata = getattr(prepared, metadata_name)
    replacement = torch.empty_like(metadata)
    assert replacement.data_ptr() != metadata.data_ptr()
    metadata.set_(replacement)
    _refresh_fp8_mqa_selective_logits(prepared, cu_q, cu_kv)
    _launch(_SelectiveLogitsMode.SAMPLE, 130, values, counts, prepared, inputs)
    torch.cuda.synchronize()

    assert prepared._launches[cache_key] is not cached
    expected = _oracle_rows(inputs)[0]
    torch.testing.assert_close(
        values[0, : expected.numel()], expected, atol=5e-5, rtol=1e-5
    )


@_requires_selective_sm100
def test_segmented_candidate_store_skips_empty_request_with_sparse_scores(monkeypatch):
    """One CTA crosses an empty request through the score-plane store path."""
    q, kv, scale, weights, _, _ = _inputs(q_rows=3, kv_rows=388)
    cu_q = torch.tensor([0, 1, 2, 3], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 129, 129, 388], device=q.device, dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    # The empty request is legal only with explicit inactive-row bounds;
    # causal Q-length-one/KV-length-zero would violate the caller contract.
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    starts = torch.zeros(3, device=q.device, dtype=torch.int32)
    ends = torch.tensor([129, 0, 259], device=q.device, dtype=torch.int32)
    prepared = _prepare(inputs, num_sms=1, row_starts=starts, row_ends=ends)
    torch.cuda.synchronize()
    assert prepared.schedule_meta[:, 0].tolist() == [0, prepared.tiles]
    assert prepared.schedule_meta[:, 1].count_nonzero().item() == 0
    oracle = _oracle_rows(inputs)
    assert [row.numel() for row in oracle] == [129, 0, 259]
    padded = math.ceil(3 / prepared.query_tile) * prepared.query_tile
    capacity = 2048
    values = torch.empty((padded, capacity), device=q.device, dtype=torch.int64)
    counts = torch.empty((padded, 33), device=q.device, dtype=torch.int32)
    lengths = torch.empty(3, device=q.device, dtype=torch.int32)
    flags = torch.empty_like(lengths)
    for threshold in (float("-inf"), 0.0, float("inf")):
        thresholds = torch.full((3,), threshold, device=q.device)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            thresholds,
        )
        sl._finalize_fp8_mqa_packed_candidates(
            counts, values, lengths, flags, rows=3, capacity=capacity
        )
        torch.cuda.synchronize()
        for row, expected_scores in enumerate(oracle):
            expected = torch.nonzero(expected_scores >= threshold).flatten()
            count = int(lengths[row])
            assert count == expected.numel()
            packed = values[row, :count]
            indices = (packed & 0xFFFFFFFF).long()
            order = indices.argsort()
            torch.testing.assert_close(indices[order], expected)
            actual_scores = (
                ((packed[order] >> 32) & 0xFFFFFFFF).int().view(torch.float32)
            )
            torch.testing.assert_close(
                actual_scores, expected_scores[expected], atol=5e-5, rtol=1e-5
            )
        assert counts[1].count_nonzero().item() == 0
        assert counts[3:].count_nonzero().item() == 0
        if threshold == 0.0:
            assert 0 < int(lengths[0] + lengths[2]) < 388

    # A full Q tile has distinct bounds inside an interior KV pair. Another
    # row keeps the tile context longer, so tail-only masking cannot pass.
    q, kv, scale, weights, _, _ = _inputs(q_rows=18, kv_rows=2050)
    cu_q = torch.tensor([0, 16, 17, 18], device=q.device, dtype=torch.int32)
    cu_kv = torch.tensor([0, 1025, 1025, 2050], device=q.device, dtype=torch.int32)
    inputs = q, kv, scale, weights, cu_q, cu_kv
    starts = torch.tensor(
        [31 + row for row in range(16)] + [0, 129], device=q.device, dtype=torch.int32
    )
    ends = torch.tensor(
        [129 + 32 * row for row in range(16)] + [0, 1025],
        device=q.device,
        dtype=torch.int32,
    )
    oracle = _oracle_rows(inputs)
    thresholds = torch.full((18,), float("-inf"), device=q.device)
    capacity = 6144
    # Lane-local, striped ballot, and pooled ballot share the changed bounded
    # append sites; no algorithm or tuning choice varies within a layout.
    for segments, split_kv in ((32, 1), (4, 1), (16, 16)):
        prepared = _prepare(
            inputs, num_sms=1, split_kv=split_kv, row_starts=starts, row_ends=ends
        )
        assert prepared.query_tile == 16
        padded = math.ceil(18 / prepared.query_tile) * prepared.query_tile
        values = torch.empty((padded, capacity), device=q.device, dtype=torch.int64)
        counts = torch.empty((padded, segments + 1), device=q.device, dtype=torch.int32)
        lengths = torch.empty(18, device=q.device, dtype=torch.int32)
        flags = torch.empty_like(lengths)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            thresholds,
        )
        raw_counts = counts.clone()
        sl._finalize_fp8_mqa_packed_candidates(
            counts, values, lengths, flags, rows=18, capacity=capacity
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(lengths, ends - starts)
        torch.testing.assert_close(
            raw_counts[:18, 1:].sum(dim=1, dtype=torch.int32), ends - starts
        )
        assert counts[16].count_nonzero().item() == 0
        assert counts[18:].count_nonzero().item() == 0
        for row, expected_scores in enumerate(oracle):
            expected = torch.arange(int(starts[row]), int(ends[row]), device=q.device)
            packed = values[row, : int(lengths[row])]
            indices = (packed & 0xFFFFFFFF).long()
            order = indices.argsort()
            torch.testing.assert_close(indices[order], expected)
            actual_scores = (
                ((packed[order] >> 32) & 0xFFFFFFFF).int().view(torch.float32)
            )
            torch.testing.assert_close(
                actual_scores, expected_scores[expected], atol=5e-5, rtol=1e-5
            )


@_requires_selective_sm100
@pytest.mark.parametrize(("num_segments", "split_kv"), [(32, 1), (64, 2), (128, 4)])
def test_lane_segmented_candidates_preserve_partial_q_and_tail_rows(
    num_segments, split_kv
):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=1025)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(inputs, split_kv=split_kv)
    threshold = torch.full((17,), float("-inf"), device="cuda", dtype=torch.float32)
    expected_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    lengths = torch.empty(17, device="cuda", dtype=torch.int32)
    flags = torch.empty(17, device="cuda", dtype=torch.int32)

    capacity = 6144
    values = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    counts = torch.empty((32, num_segments + 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        values,
        counts,
        prepared,
        inputs,
        threshold,
    )
    _finalize_fp8_mqa_packed_candidates(
        counts, values, lengths, flags, rows=17, capacity=capacity
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(lengths, expected_lengths)
    for row in range(17):
        count = int(lengths[row])
        actual_indices = (values[row, :count] & 0xFFFFFFFF).sort().values
        torch.testing.assert_close(
            actual_indices, torch.arange(count, device="cuda", dtype=torch.int64)
        )

    overflow_capacity = 256
    overflow_values = torch.empty(
        (32, overflow_capacity), device="cuda", dtype=torch.int64
    )
    overflow_counts = torch.empty(
        (32, num_segments + 1), device="cuda", dtype=torch.int32
    )
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        overflow_capacity,
        overflow_values,
        overflow_counts,
        prepared,
        inputs,
        threshold,
    )
    raw_overflow_counts = overflow_counts[:17].clone()
    _finalize_fp8_mqa_packed_candidates(
        overflow_counts,
        overflow_values,
        lengths,
        flags,
        rows=17,
        capacity=overflow_capacity,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        raw_overflow_counts[:, 1:].sum(dim=1, dtype=torch.int32), expected_lengths
    )
    segment_capacity = overflow_capacity // num_segments
    segment_overflow = (raw_overflow_counts[:, 1:] > segment_capacity).any(dim=1)
    assert segment_overflow.any()
    expected_published = torch.where(
        segment_overflow,
        torch.full_like(expected_lengths, overflow_capacity + 1),
        expected_lengths,
    )
    torch.testing.assert_close(overflow_counts[:17, 0], expected_published)
    torch.testing.assert_close(
        lengths,
        raw_overflow_counts[:, 1:]
        .clamp(max=segment_capacity)
        .sum(dim=1, dtype=torch.int32),
    )
    assert (flags == 1).all()


@_requires_selective_sm100
def test_adapter_rejects_unsafe_layouts_before_launch():
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    q, kv, scale, weights, cu_q, cu_kv = _inputs(q_rows=1, kv_rows=130)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    kv_fused, block_table = _paged_inputs(inputs)
    raw = torch.empty(kv_fused.numel() + 1, device="cuda", dtype=torch.uint8)[1:]
    raw.copy_(kv_fused.reshape(-1))
    misaligned_kv = raw.view_as(kv_fused)
    with pytest.raises(ValueError, match="16-byte aligned"):
        _prepare_fp8_mqa_selective_logits(
            q, misaligned_kv, weights, cu_q, cu_kv, block_table
        )

    strided_weights = torch.empty((1, 16), device="cuda", dtype=torch.float32)[:, ::2]
    strided_weights.copy_(weights)
    with pytest.raises(ValueError, match="contiguous"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, strided_weights, cu_q, cu_kv, block_table
        )

    with pytest.raises(ValueError, match="split_kv must be 1, 2, 4, 8, 16, 32, or 64"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, cu_kv, block_table, split_kv=3
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _prepare_fp8_mqa_selective_logits(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            kv_chunk_size=256,
            split_kv=2,
        )

    q, kv, scale, weights, cu_q, _cu_kv = _inputs(q_rows=17, kv_rows=387)
    cu_kv = torch.tensor([0, 130, 387], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)
    prepared = _prepare(inputs)
    assert prepared.batches == 2


@_requires_selective_sm100
def test_prepared_problem_shape_configuration_reaches_compiled_kernel():
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=17, kv_rows=387, num_heads=4, head_dim=64)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    prepared = _prepare(
        inputs,
        num_heads=4,
        head_dim=64,
    )
    assert prepared.num_heads == 4
    assert prepared.head_dim == 64

    values = torch.empty((32, 387), device="cuda", dtype=torch.float32)
    counts = torch.empty((32, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        387,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    expected_rows = _oracle_rows(inputs)
    for row, expected in enumerate(expected_rows):
        torch.testing.assert_close(values[row, : expected.numel()], expected)

    capacity = 1024
    packed = torch.empty((32, capacity), device="cuda", dtype=torch.int64)
    segmented_counts = torch.empty((32, 33), device="cuda", dtype=torch.int32)
    thresholds = torch.full((q.shape[0],), float("-inf"), device="cuda")
    _launch(
        _SelectiveLogitsMode.CANDIDATE,
        capacity,
        packed,
        segmented_counts,
        prepared,
        inputs,
        thresholds,
    )
    raw_counts = segmented_counts.clone()
    lengths = torch.empty(q.shape[0], device="cuda", dtype=torch.int32)
    flags = torch.empty_like(lengths)
    _finalize_fp8_mqa_packed_candidates(
        segmented_counts,
        packed,
        lengths,
        flags,
        rows=q.shape[0],
        capacity=capacity,
    )
    torch.cuda.synchronize()
    valid_lengths = torch.tensor(
        _valid_lengths(cu_q, cu_kv), device="cuda", dtype=torch.int32
    )
    torch.testing.assert_close(
        raw_counts[: q.shape[0], 1:].sum(dim=1), valid_lengths.to(torch.int64)
    )
    torch.testing.assert_close(lengths, valid_lengths)
    for row, valid in enumerate(valid_lengths.tolist()):
        row_packed = packed[row, :valid]
        actual_indices = (row_packed & 0xFFFFFFFF).to(torch.int32)
        order = actual_indices.argsort()
        torch.testing.assert_close(
            actual_indices[order],
            torch.arange(valid, device="cuda", dtype=torch.int32),
        )
        actual_score_bits = ((row_packed[order] >> 32) & 0xFFFFFFFF).to(torch.int32)
        torch.testing.assert_close(
            actual_score_bits,
            values[row, :valid].view(torch.int32),
        )


@_requires_selective_sm100
def test_prepared_problem_shape_supports_logical_n_256():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=256, head_dim=32)
    q, _kv, _scale, _weights, _cu_q, _cu_kv = inputs
    prepared = _prepare(inputs, num_heads=256, head_dim=32)
    assert prepared.query_tile == 1
    assert prepared.num_umma_stages == 1

    values = torch.empty((1, 128), device="cuda", dtype=torch.float32)
    counts = torch.empty((1, 1), device="cuda", dtype=torch.int32)
    _launch(
        _SelectiveLogitsMode.SAMPLE,
        128,
        values,
        counts,
        prepared,
        inputs,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(values[0], _oracle_rows(inputs)[0])


@_requires_selective_sm100
def test_metadata_schedule_has_no_fixed_tile_limit():
    """Runtime global scratch supports schedules beyond the old 8192-tile cap."""
    q_rows = 8193
    inputs = _inputs(q_rows=q_rows, kv_rows=8320, num_heads=256, head_dim=32)
    q, kv, scale, weights, _cu_q, _cu_kv = inputs
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    inputs = (q, kv, scale, weights, cu_q, cu_kv)

    prepared = _prepare(inputs, num_heads=256, head_dim=32, num_sms=4)
    torch.cuda.synchronize()

    assert prepared.query_tile == 1
    assert prepared.tiles == q_rows
    assert prepared.schedule_prefix.shape == (q_rows,)
    assert int(prepared.schedule_prefix[-1]) == q_rows
    assert torch.equal(
        prepared.schedule_meta[-1].cpu(), torch.tensor([q_rows, 0], dtype=torch.int32)
    )


@_requires_selective_sm100
def test_prepared_problem_shape_rejects_unsupported_specializations():
    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=4, head_dim=96)
    q, kv, scale, weights, cu_q, cu_kv = inputs
    with pytest.raises(ValueError, match="head_dim must be a multiple"):
        _prepare(
            (
                q[:, :, :65].contiguous(),
                kv[:, :65].contiguous(),
                scale,
                weights,
                cu_q,
                cu_kv,
            ),
            num_heads=4,
            head_dim=65,
        )

    inputs = _inputs(q_rows=1, kv_rows=128, num_heads=257, head_dim=32)
    with pytest.raises(ValueError, match="no legal selective FP8 query tile"):
        _prepare(inputs, num_heads=257, head_dim=32)


@_requires_selective_sm100
def test_opt_in_validation_rejects_invalid_device_metadata(monkeypatch):
    from flashinfer.attn_scores.selective_logits import (
        _prepare_fp8_mqa_selective_logits,
    )

    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    inputs = _inputs(q_rows=17, kv_rows=387)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    kv_fused, block_table = _paged_inputs(inputs, block_size=32)

    bad_cu_q = cu_q.clone()
    bad_cu_q[-1] = q.shape[0] - 1
    with pytest.raises(ValueError, match=r"cu_q\[-1\]"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, bad_cu_q, cu_kv, block_table
        )

    bad_cu_kv = cu_kv.clone()
    bad_cu_kv[1] = -1
    with pytest.raises(ValueError, match="monotonically"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, bad_cu_kv, block_table
        )

    short_causal_kv = cu_kv.clone()
    short_causal_kv[1] = 0
    with pytest.raises(ValueError, match="greater than"):
        _prepare_fp8_mqa_selective_logits(
            q, kv_fused, weights, cu_q, short_causal_kv, block_table
        )

    too_short = block_table[:, :-1].contiguous()
    with pytest.raises(ValueError, match="requires at least"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, too_short)

    bad_table = block_table.clone()
    bad_table[0, 0] = kv_fused.shape[0]
    with pytest.raises(ValueError, match="physical block index"):
        _prepare_fp8_mqa_selective_logits(q, kv_fused, weights, cu_q, cu_kv, bad_table)

    single = _inputs(q_rows=1, kv_rows=128)
    q1, _kv1, _scale1, weights1, cu_q1, cu_kv1 = single
    fused1, table1 = _paged_inputs(single)
    starts = torch.zeros(1, device="cuda", dtype=torch.int32)
    ends = torch.full((1,), 129, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="explicit row"):
        _prepare_fp8_mqa_selective_logits(
            q1,
            fused1,
            weights1,
            cu_q1,
            cu_kv1,
            table1,
            row_starts=starts,
            row_ends=ends,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("q_rows", [1, 33])
@pytest.mark.parametrize("strategy", ["full", "selective"])
def test_wrapper_page_map_updates_in_eager_and_graph(strategy, q_rows):
    """Both execution modes consume the current physical mapping."""
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=8192)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(fused.shape[0], device="cuda", dtype=torch.int32)[None, :]
    top_k = 512 if strategy == "full" else 8
    selective = dict(
        top_k=top_k,
        max_context_length=8192,
        max_chunk_size=q_rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy, **(selective if strategy == "selective" else {})
    )
    if strategy == "full":
        wrapper.plan(q, fused, weights, cu_q, cu_kv, table, top_k=top_k)
    else:
        wrapper.prepare_pool(fused)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if strategy == "selective":
            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        output = (
            wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
            if strategy == "full"
            else wrapper.run(q, weights, fused)
        )
    for shift in (1, 3):
        # Replay must consume new payload values as well as the new page map.
        q.copy_((-q.float()).to(q.dtype))
        weights.copy_(weights.roll(1, dims=1))
        kv.copy_(kv.float().roll(shift, dims=0).to(kv.dtype))
        scale.copy_(scale.roll(shift))
        fused.copy_(_fuse_paged_kv(kv, scale))
        order = torch.arange(fused.shape[0], device="cuda").roll(shift)
        table.copy_(order[None, :])
        mapped_kv = kv.reshape(-1, 128, 128)[order].reshape_as(kv)
        mapped_scale = scale.reshape(-1, 128)[order].reshape_as(scale)
        oracle = _oracle_rows((q, mapped_kv, mapped_scale, weights, cu_q, cu_kv))
        if strategy == "selective":
            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        eager = (
            wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
            if strategy == "full"
            else wrapper.run(q, weights, fused)
        ).clone()
        graph.replay()
        torch.cuda.synchronize()
        for actual in (eager, output):
            _check_selection(actual, oracle, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("q_rows", [1, 33])
def test_full_wrapper_failed_capture_rejects_run_and_retries(monkeypatch, q_rows):
    from flashinfer import FP8PagedMQATopKWrapper

    q, kv, scale, weights, _, _ = _inputs(q_rows=q_rows, kv_rows=2048)
    cu_q = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused = _fuse_paged_kv(kv, scale)
    table = torch.arange(fused.shape[0], device="cuda", dtype=torch.int32)[None, :]
    wrapper = FP8PagedMQATopKWrapper(strategy="full")
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table)

    def fail_capture():
        raise RuntimeError("injected capture failure")

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "CUDAGraph", fail_capture)
        with pytest.raises(RuntimeError, match="injected capture failure"):
            wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
    with pytest.raises(RuntimeError, match="plan must be called"):
        wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)

    wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
    output = wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
    torch.cuda.synchronize()
    assert output.shape == (q_rows, 512)
    for row, scores in enumerate(_oracle_rows((q, kv, scale, weights, cu_q, cu_kv))):
        torch.testing.assert_close(
            scores[output[row].long()].sort().values,
            scores.topk(512).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_selective_math_mode_cache_alternation(heads):
    """Prepared and process caches cannot reuse a different score recipe."""
    from dataclasses import FrozenInstanceError

    inputs = _inputs(3, 259, heads)
    rows = _oracle_rows(inputs)
    prepared = [
        _prepare(inputs, num_heads=heads, num_sms=4, math_mode=mode)
        for mode in ("relu_fast", "relu_strict")
    ]
    prepared.append(_prepare(inputs, num_heads=heads, num_sms=4))
    assert prepared[2].math_mode == "relu_strict"
    cached = []
    # Hold runtime storage fixed: rebinding an output correctly rebuilds the
    # prepared launch object independently of the numerical compile policy.
    values = torch.empty((16, 259), device="cuda", dtype=torch.float32)
    counts = torch.empty((16, 1), device="cuda", dtype=torch.int32)
    for policy in (*prepared, prepared[0]):
        _launch(_SelectiveLogitsMode.SAMPLE, 259, values, counts, policy, inputs)
        torch.cuda.synchronize()
        for row, expected in enumerate(rows):
            torch.testing.assert_close(
                values[row, : expected.numel()], expected, atol=5e-5, rtol=1e-5
            )
        assert all(key[-2] == policy.math_mode for key in policy._launches)
        cached.append(next(iter(policy._launches.values())))
    assert cached[0] is cached[3]
    assert cached[0] is not cached[1]
    assert cached[0].compiled is cached[3].compiled
    assert cached[0].compiled is not cached[1].compiled
    assert cached[1].compiled is cached[2].compiled
    with pytest.raises(FrozenInstanceError):
        prepared[0].math_mode = "relu_strict"


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("last_head", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_selective_sample_and_repair_range_safe_relu(heads, last_head, sign):
    inputs = _inputs(17, 387, heads)
    q, kv, scales, weights, _, _ = inputs
    active = heads - 1 if last_head else 0
    q.zero_()
    kv.zero_()
    weights.zero_()
    q[:, active, 0] = 1
    kv[:, 0] = 1
    scales.fill_(0.5)
    weights[:, active] = sign * torch.finfo(torch.float32).max
    prepared = _prepare(inputs, num_heads=heads, num_sms=4, math_mode="relu_strict")
    fused, table = _paged_inputs(inputs)
    tile = prepared.query_tile
    padded = (q.shape[0] + tile - 1) // tile * tile
    samples = torch.empty((padded, 512), device=q.device, dtype=torch.float32)
    records = torch.empty((padded, 6144), device=q.device, dtype=torch.int64)
    counts = torch.empty((padded, 1), device=q.device, dtype=torch.int32)
    unused = torch.empty_like(counts)
    thresholds = torch.full((q.shape[0],), -torch.inf, device=q.device)

    def sample():
        _fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=_SelectiveLogitsMode.SAMPLE,
            capacity=512,
            values=samples,
            index_counts=unused,
        )

    sample()
    _fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        table,
        prepared,
        mode=_SelectiveLogitsMode.REPAIR,
        capacity=6144,
        values=records,
        index_counts=counts,
        policy_values=thresholds,
    )
    expected = sign * torch.finfo(torch.float32).max / 2
    for row, length in enumerate(_valid_lengths(inputs[-2], inputs[-1])):
        assert bool(torch.isfinite(samples[row, :length]).all())
        torch.testing.assert_close(
            samples[row, :length],
            torch.full_like(samples[row, :length], expected),
            atol=0,
            rtol=0,
        )
        count = int(counts[row, 0])
        assert count == length
        packed = records[row, :count]
        indices = packed.bitwise_and(0xFFFFFFFF)
        assert indices.unique().numel() == length
        assert bool(((indices >= 0) & (indices < length)).all())
        values = (packed >> 32).to(torch.int32).view(torch.float32)
        assert bool(torch.isfinite(values).all())
        torch.testing.assert_close(
            values, torch.full_like(values, expected), atol=0, rtol=0
        )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sample()
    q[:, active, 0] = -1
    graph.replay()
    for row, length in enumerate(_valid_lengths(inputs[-2], inputs[-1])):
        assert bool((samples[row, :length] == 0).all())


@contextmanager
def _preserve_cuda_device_streams(devices):
    original = torch.cuda.current_device()
    saved_streams = [torch.cuda.current_stream(device) for device in devices]

    def stream_handles():
        return [torch.cuda.current_stream(device).cuda_stream for device in devices]

    try:
        yield stream_handles
    finally:
        for stream in saved_streams:
            with torch.cuda.device(stream.device):
                torch.cuda.set_stream(stream)
        torch.cuda.set_device(original)


@_requires_selective_sm100
@pytest.mark.parametrize("route", ["full", "selective"])
def test_public_wrapper_on_non_current_device_after_warm_cache(route):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    wrappers = []
    handles = []
    with _preserve_cuda_device_streams((0, 1)) as stream_handles:
        torch.cuda.set_device(0)
        caller = torch.cuda.Stream(device=0)
        torch.cuda.set_stream(caller)
        expected_streams = stream_handles()
        # Compile on device0 first, then reuse same-CC caches on device1.
        for target in (0, 1):
            with torch.cuda.device(target):
                inputs = _inputs(5, 8192)
                q, _, _, weights, cu_q, cu_kv = inputs
                cu_q.copy_(torch.tensor([0, 1, 5], device=q.device, dtype=torch.int32))
                fused, table = _paged_inputs(inputs, 32)
            selective = dict(
                page_size=32,
                max_context_length=8192,
                max_chunk_size=5,
                max_batch_size=2,
                kv_layout=fused,
            )
            wrapper = FP8PagedMQATopKWrapper(
                strategy="full" if route == "full" else "selective",
                **(selective if route == "selective" else {}),
            )
            if route == "full":
                wrapper.plan(q, fused, weights, cu_q, cu_kv, table)
            else:
                wrapper.prepare_pool(fused)
            assert torch.cuda.current_device() == 0
            assert stream_handles() == expected_streams
            wrappers.append(wrapper)
            if route != "full":
                engine = wrapper._capacity_plan._engine
                assert engine._repair_body_stream.device == q.device
                handles.append(engine._repair_body_stream.cuda_stream)
            for tied in (False, True):
                if tied:
                    weights.zero_()
                if route == "selective":
                    wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
                output = (
                    wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
                    if route == "full"
                    else wrapper.run(q, weights, fused)
                )
                assert torch.cuda.current_device() == 0
                assert stream_handles() == expected_streams
                with torch.cuda.device(target):
                    graph = torch.cuda.CUDAGraph()
                    capture_stream = torch.cuda.Stream(device=q.device)
                    with torch.cuda.graph(graph, stream=capture_stream):
                        if route == "selective":
                            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
                        captured = (
                            wrapper.run(
                                q=q, kv_fused=fused, weights=weights, block_table=table
                            )
                            if route == "full"
                            else wrapper.run(q, weights, fused)
                        )
                    assert stream_handles() == expected_streams
                    graph.replay()
                    for indices in (output, captured):
                        for row, scores in enumerate(_oracle_rows(inputs)):
                            count = min(512, scores.numel())
                            selected = indices[row, :count].long()
                            assert bool(
                                ((selected >= 0) & (selected < scores.numel())).all()
                            )
                            assert selected.unique().numel() == count
                            torch.testing.assert_close(
                                scores[selected].sort().values,
                                scores.topk(count).values.sort().values,
                                atol=5e-5,
                                rtol=1e-5,
                            )
                assert torch.cuda.current_device() == 0
                assert stream_handles() == expected_streams
        if handles:
            assert len(set(handles)) == 2


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("route", ["full", "selective"])
def test_natural_page_table_replay_and_replan(page_size, heads, route, monkeypatch):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    q, kv, scales, weights, _, _ = _inputs(5, 257 + 9001, heads)
    cu_q = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 257, 257 + 9001], device="cuda", dtype=torch.int32)
    fused, padded_table = _paged_inputs(
        (q, kv, scales, weights, cu_q, cu_kv), page_size
    )
    table = padded_table[:, : math.ceil(9001 / page_size)].contiguous()
    table[0, math.ceil(257 / page_size) :] = 2**31 - 1
    packed = fused.flatten(1)
    page_keys = packed[:, : page_size * 128].contiguous().view(kv.dtype)
    page_keys = page_keys.reshape(-1, page_size, 128).float()
    page_scales = packed[:, page_size * 128 :].contiguous().view(torch.float32)
    strategy = "full" if route == "full" else "selective"
    wrapper = FP8PagedMQATopKWrapper(
        strategy=strategy,
        **(
            {
                "num_heads": heads,
                "page_size": page_size,
                "max_context_length": 9001,
                "max_chunk_size": 5,
                "max_batch_size": 2,
                "kv_layout": fused,
            }
            if strategy == "selective"
            else {}
        ),
    )

    def check(indices, lengths):
        for request, length in enumerate(lengths):
            q_begin, q_end = (0, 3) if request == 0 else (3, 5)
            physical = table[request, : math.ceil(length / page_size)].long()
            keys = page_keys[physical].reshape(-1, 128)[:length]
            scale = page_scales[physical].reshape(-1)[:length]
            scores = torch.einsum("qhd,kd->qhk", q[q_begin:q_end].float(), keys)
            scores.relu_()
            scores = (scores * weights[q_begin:q_end, :, None]).sum(1) * scale
            for local_row, row in enumerate(range(q_begin, q_end)):
                end = length - (q_end - q_begin) + local_row + 1
                count = min(end, 512)
                selected = indices[row, :count].long()
                assert bool(((selected >= 0) & (selected < end)).all())
                assert selected.unique().numel() == count
                torch.testing.assert_close(
                    scores[local_row, selected].sort().values,
                    scores[local_row, :end].topk(count).values.sort().values,
                    atol=2e-5,
                    rtol=2e-5,
                )
                assert bool((indices[row, count:] == -1).all())

    for generation, length in enumerate((9001, 8999)):
        if generation:
            weights.zero_()
        cu_kv[-1] = 257 + length
        if strategy == "full":
            wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_heads=heads)
        else:
            wrapper.prepare_pool(fused)
        if strategy == "selective":
            wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
        check(
            (
                wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
                if strategy == "full"
                else wrapper.run(q, weights, fused)
            ),
            (257, length),
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if strategy == "selective":
                wrapper.plan(cu_q, cu_kv, table, num_query_rows=q.shape[0])
            output = (
                wrapper.run(q=q, kv_fused=fused, weights=weights, block_table=table)
                if strategy == "full"
                else wrapper.run(q, weights, fused)
            )
        for _ in range(2):
            for request, request_length in enumerate((257, length)):
                pages = math.ceil(request_length / page_size)
                table[request, :pages].copy_(table[request, :pages].flip(0))
            graph.replay()
            check(output, (257, length))
        if generation and strategy == "selective":
            assert (
                wrapper._capacity_plan._engine._second_repair_flags[3:5]
                .count_nonzero()
                .item()
                == 2
            )
            assert wrapper.repair_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize(
    "page_size,heads,internal,lengths",
    [
        pytest.param(
            page, heads, internal, (0, 129, 259), id=f"{internal}-{heads}-{page}"
        )
        for internal in (False, True)
        for heads in (8, 64)
        for page in (32, 64, 128)
    ]
    + [
        pytest.param(
            128, heads, True, (129, 0, 9001), id=f"interspersed-True-{heads}-128"
        )
        for heads in (8, 64)
    ],
)
def test_inactive_sample_rows_and_logical_page_clamp(
    page_size, heads, internal, lengths, monkeypatch
):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    active_requests = [request for request, length in enumerate(lengths) if length]
    inactive_request = next(
        request for request, length in enumerate(lengths) if not length
    )
    interspersed = inactive_request == 1
    pages = [math.ceil(length / page_size) for length in lengths]
    physical_lengths = [2 * count * page_size for count in pages]
    q, kv, scales, weights, _, _ = _inputs(3, sum(physical_lengths), heads)
    cu_q = torch.tensor([0, 1, 2, 3], device=q.device, dtype=torch.int32)
    original_prefix = torch.tensor(
        [0, physical_lengths[0], sum(physical_lengths[:2]), sum(physical_lengths)],
        device=q.device,
        dtype=torch.int32,
    )
    fused, table = _paged_inputs(
        (q, kv, scales, weights, cu_q, original_prefix), page_size
    )
    table[inactive_request].fill_(2**31 - 1)
    sample_pages = torch.full(
        (3, max(pages)), 2**31 - 1, device=q.device, dtype=torch.int64
    )
    for request in active_requests:
        sample_pages[request, : pages[request]] = 2 * torch.arange(
            pages[request], device=q.device
        )
        table[request, 2 * pages[request] :].fill_(2**31 - 1)
    sample_prefix = torch.tensor(
        [0, lengths[0], sum(lengths[:2]), sum(lengths)],
        device=q.device,
        dtype=torch.int32,
    )
    starts = torch.zeros(3, device=q.device, dtype=torch.int32)
    ends = torch.tensor(lengths, device=q.device, dtype=torch.int32)
    prepared = sl._prepare_fp8_mqa_selective_logits(
        q,
        fused,
        weights,
        cu_q,
        sample_prefix,
        table,
        row_starts=starts,
        row_ends=ends,
        num_heads=heads,
        # One CTA forces Q lookahead across the empty middle request rather
        # than assigning each live request to a separate producer role.
        num_sms=1 if interspersed else 4,
    )
    if interspersed:
        assert prepared.context_lens[:3].tolist() == list(lengths)
        # The final request owns tile 2; padded tiles have no work.
        assert prepared.schedule_meta[:, 0].tolist() == [0, 3]
        assert prepared.schedule_meta[:, 1].count_nonzero().item() == 0
    padded = math.ceil(3 / prepared.query_tile) * prepared.query_tile
    capacity = math.ceil(max(lengths) / 128) * 128
    values = torch.full((padded, capacity), float("nan"), device=q.device)
    counts = torch.empty((padded, 1), device=q.device, dtype=torch.int32)

    def run():
        sl._fp8_mqa_selective_logits(
            q,
            fused,
            weights,
            table,
            prepared,
            mode=sl._SelectiveLogitsMode.SAMPLE,
            capacity=capacity,
            values=values,
            index_counts=counts,
            sample_page_indices=sample_pages if internal else None,
        )

    packed = fused.flatten(1)
    page_keys = (
        packed[:, : page_size * 128]
        .contiguous()
        .view(kv.dtype)
        .reshape(-1, page_size, 128)
        .float()
    )
    page_scales = packed[:, page_size * 128 :].contiguous().view(torch.float32)

    def check():
        assert torch.isnan(values[inactive_request]).all()
        for request in active_requests:
            positions = torch.arange(lengths[request], device=q.device)
            logical_pages = positions // page_size
            if internal:
                logical_pages = sample_pages[request, logical_pages]
            physical_pages = table[request, logical_pages].long()
            offsets = positions % page_size
            keys = page_keys[physical_pages, offsets]
            scale = page_scales[physical_pages, offsets]
            scores = torch.einsum("hd,kd->hk", q[request].float(), keys).relu_()
            expected = (scores * weights[request, :, None]).sum(0) * scale
            torch.testing.assert_close(
                values[request, : lengths[request]], expected, atol=2e-5, rtol=2e-5
            )
            assert torch.isnan(values[request, lengths[request] :]).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(2):
        for request in active_requests:
            count = 2 * pages[request]
            table[request, :count].copy_(table[request, :count].flip(0))
            sample_pages[request, : pages[request]].copy_(
                sample_pages[request, : pages[request]].flip(0)
            )
        graph.replay()
        check()


@pytest.fixture(autouse=True)
def _clear_paged_input_cache():
    # Reuse pages within one case without keeping its GPU tensors across cases.
    _PAGED_INPUT_CACHE.clear()
    try:
        yield
    finally:
        _PAGED_INPUT_CACHE.clear()


# Sample threshold and parallel ladder


@_requires_selective_cuda
@pytest.mark.parametrize("signed_bins", [False, True])
def test_sample_threshold_negative_bins(signed_bins):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveSampleThresholdKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    samples = torch.stack(
        [
            torch.linspace(-32, -1, 512, device="cuda"),
            torch.linspace(1, 32, 512, device="cuda"),
            torch.zeros(512, device="cuda"),
            torch.full((512,), -1e6, device="cuda"),
        ]
    )
    valid = torch.full((4,), 512, device="cuda", dtype=torch.int32)
    ranks = torch.full((4,), 32, device="cuda", dtype=torch.int32)
    ends = torch.full((4,), 4096, device="cuda", dtype=torch.int32)
    thresholds = torch.empty(4, device="cuda", dtype=torch.float32)
    bins = torch.empty(4, device="cuda", dtype=torch.int32)
    args = (
        from_dlpack(samples).mark_layout_dynamic(leading_dim=1),
        *(
            from_dlpack(t).mark_compact_shape_dynamic(mode=0, divisibility=1)
            for t in (valid, ranks, ends, thresholds, bins)
        ),
        cutlass.Int32(4),
    )
    stream = _current_stream(samples.device)
    compiled = cute.compile(
        SelectiveSampleThresholdKernel(8, signed_bins), *args, stream
    )
    compiled(*args, stream)
    torch.cuda.synchronize()
    bits = samples.half().view(torch.uint16).long()
    ordered = torch.where(bits & 0x8000 != 0, bits, (bits ^ 0xFFFF) & 0x7FFF)
    expected_bins = torch.stack(
        [
            torch.searchsorted(torch.bincount(row >> 5, minlength=2048).cumsum(0), 32)
            for row in ordered
        ]
    ).int()
    torch.testing.assert_close(bins, expected_bins, atol=0, rtol=0)
    expected_key = ((expected_bins.to(torch.int64) << 5) + 31).clamp(
        max=0xFC00 if signed_bins else 0x7FFF
    )
    expected_bits = torch.where(
        (expected_key & 0x8000) != 0, expected_key, (expected_key ^ 0xFFFF) & 0x7FFF
    )
    expected = expected_bits.to(torch.uint16).view(torch.float16).float()
    torch.testing.assert_close(thresholds, expected, atol=0, rtol=0)
    if signed_bins:
        assert thresholds[0] < 0
        assert (samples[0] >= thresholds[0]).sum() >= 32
        assert torch.isneginf(thresholds[3])
    else:
        assert thresholds[0] == 0
        assert (samples[0] >= thresholds[0]).sum() == 0
    assert thresholds[1] > 0
    assert thresholds[2] <= 0


def _sample_ladder_reference(samples, valid, ranks, ends, top_k, segments):
    """Independent histogram crossings and bank estimates; no production helper."""
    rows, device = valid.numel(), samples.device
    expected_ladder = torch.empty((rows, 3), device=device)
    expected_populations = torch.zeros((rows, 3), dtype=torch.int32, device=device)
    expected_bins = torch.full((rows,), 2047, dtype=torch.int32, device=device)
    expected_thresholds = torch.empty(rows, device=device)
    for row in range(rows):
        count = int(valid[row])
        bits = samples[row, :count].half().view(torch.uint16).long()
        keys = torch.where(bits & 0x8000 != 0, bits, (bits ^ 0xFFFF) & 0x7FFF)
        prefix = torch.bincount(keys >> 5, minlength=2048).cumsum(0)
        bins = []
        for rung in range(3):
            target = max(1, min(count, int(ranks[row])) // (4 >> rung))
            index = int(torch.searchsorted(prefix, target)) if count else 2047
            bins.append(index)
            key = min(0xFC00, (index << 5) + 31)
            floor_bits = key if key & 0x8000 else (key ^ 0xFFFF) & 0x7FFF
            expected_ladder[row, rung] = (
                torch.tensor(floor_bits, dtype=torch.uint16, device=device)
                .view(torch.float16)
                .float()
            )
            if count:
                expected_populations[row, rung] = prefix[index]
        chosen = 2
        for rung in (2, 1, 0):
            hits = int(expected_populations[row, rung])
            sigma = math.sqrt(max(hits, 1))
            scale = int(ends[row]) / max(count, 1)
            lower = max(0, hits - 3 * sigma) * scale
            bank_mean = (hits + 3 * sigma) * scale / segments
            bank_upper = bank_mean + 3 * math.sqrt(max(bank_mean, 1))
            if lower >= top_k and bank_upper < 6144 // segments:
                chosen = rung
        expected_thresholds[row] = float("-inf")
        if count and ends[row] >= top_k:
            expected_thresholds[row] = expected_ladder[row, chosen]
            expected_bins[row] = bins[chosen]

    return expected_thresholds, expected_bins, expected_ladder, expected_populations


@_requires_selective_cuda
@pytest.mark.parametrize("segments", [4, 8, 16, 32, 64, 128])
def test_native_ladder_matches_signed_histogram(segments):
    """Crossings, inclusive populations, and bank policy match a host oracle."""
    from flashinfer.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    gen = torch.Generator(device="cuda").manual_seed(1)
    samples = torch.randn((8, 1024), generator=gen, device="cuda") * 16
    samples[1].zero_()
    samples[2].fill_(-1e6)
    samples[3].fill_(1e6)
    samples[4].abs_().neg_()
    valid = torch.tensor(
        [0, 1024, 1024, 1024, 1, 511, 768, 1024], dtype=torch.int32, device="cuda"
    )
    ranks = torch.tensor(
        [1, 128, 256, 512, 1, 64, 192, 256], dtype=torch.int32, device="cuda"
    )
    ends = torch.tensor(
        [16, 8192, 16384, 16384, 8, 32768, 8192, 65536],
        dtype=torch.int32,
        device="cuda",
    )
    thresholds = torch.empty(8, device="cuda")
    bins = torch.empty(8, dtype=torch.int32, device="cuda")
    ladder = torch.empty((8, 3), device="cuda")
    populations = torch.empty((8, 3), dtype=torch.int32, device="cuda")
    actual = thresholds, bins, ladder, populations
    get_selective_sample_ladder_module().run(
        samples, valid, ranks, ends, *actual, 512, 6144, segments
    )
    expected = _sample_ladder_reference(samples, valid, ranks, ends, 512, segments)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


# Native sample ladder


@_requires_selective_cuda
@pytest.mark.parametrize("top_k", [1, 31, 128, 512])
@pytest.mark.parametrize("segments", [4, 8, 12, 16, 32, 64, 128])
def test_cuda_sample_ladder_policy_and_graph(top_k, segments):
    from flashinfer.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    torch.manual_seed(1)
    rows, width = 33, 3840
    # The scorer pads its row storage independently of logical policy rows.
    samples = torch.randn((rows + 3, width + 16), device="cuda")[:, :width]
    valid = torch.empty(rows, device="cuda", dtype=torch.int32)
    ranks = torch.empty_like(valid)
    ends = torch.empty_like(valid)

    def outputs():
        return (
            torch.empty(rows, device="cuda"),
            torch.empty_like(valid),
            torch.empty((rows, 3), device="cuda"),
            torch.empty((rows, 3), dtype=torch.int32, device="cuda"),
        )

    actual = outputs()
    module = get_selective_sample_ladder_module()

    def run():
        module.run(samples, valid, ranks, ends, *actual, top_k, 6144, segments)

    valid.fill_(width)
    ranks.fill_(128)
    ends.fill_(32768)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case in range(5):
        if case == 0:
            samples.zero_()
        elif case == 1:
            samples.normal_()
        elif case == 2:
            samples.fill_(7.5)
            samples[:, ::7] = -3.0
        elif case == 3:
            samples.uniform_(-1e-7, 1e-7)
        else:
            # Values around half-precision bin boundaries, including infinities.
            values = torch.tensor(
                [
                    -float("inf"),
                    -65504,
                    -1.001,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    1.001,
                    65504,
                    float("inf"),
                ],
                device="cuda",
            )
            samples.copy_(values[torch.arange(width, device="cuda") % len(values)])
        valid.copy_(torch.linspace(0, width, rows, device="cuda").int())
        ranks.copy_(torch.clamp(valid // 7, min=1))
        ends.copy_(torch.arange(rows, device="cuda", dtype=torch.int32) * 32768)
        ends[1] = top_k - 1
        ends[2] = top_k
        expected = _sample_ladder_reference(
            samples, valid, ranks, ends, top_k, segments
        )
        graph.replay()
        for want, got in zip(expected, actual, strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@_requires_selective_cuda
def test_cuda_sample_ladder_refined_rank_and_graph():
    from flashinfer.jit.selective_sample_ladder import (
        get_selective_sample_ladder_module,
    )

    # Include a partial CTA loop and the actual scorer's full sample capacity.
    counts = [0, 1, 127, 128, 129, 3839, 3840, 3840, 129, 3840, 3840, 3840]
    rows, width = len(counts), 3840
    samples = torch.full((rows, width), 0.9999, device="cuda")
    valid = torch.tensor(counts, dtype=torch.int32, device="cuda")
    ranks = torch.clamp(valid // 3, min=1)
    ends = torch.full_like(valid, 1048576)
    ends[-1] = 511  # Rows shorter than TopK use no threshold.

    def outputs():
        return (
            torch.empty(rows, device="cuda"),
            torch.empty_like(valid),
            torch.empty((rows, 3), device="cuda"),
            torch.empty((rows, 3), dtype=torch.int32, device="cuda"),
        )

    coarse, refined = outputs(), outputs()
    module = get_selective_sample_ladder_module()

    def run():
        module.run(samples, valid, ranks, ends, *coarse, 512, 6144, 1)
        module.run_refined(samples, valid, ranks, ends, *refined, 512, 6144, 1)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for tied in (False, True):
        samples.fill_(float("nan"))
        for row, count in enumerate(counts):
            # All finite values round into the SAME half bin. In particular,
            # 0.9998999834 rounds to half 1.0 but is below the FP16 bin floor.
            values = torch.linspace(0.9999, 1.0002, max(count, 1), device="cuda")
            if tied:
                values.fill_(0.9998999834060669)
            if row in (7, 8):
                # Earlier positive bins force a nonzero residual-rank
                # subtraction; the selected negative bin tests signed keys.
                values = torch.linspace(-1.02, -0.99, count, device="cuda")
                if tied:
                    values.fill_(-1.0001)
                values[: count // 8] = 2.0
            samples[row, :count].copy_(values[:count])
        samples[-3, 0] = float("inf")
        samples[-2, 0] = float("nan")
        graph.replay()
        # The chooser and repair ABI remain coarse, even when the literal
        # effective threshold moves down rather than up.
        for want, got in zip(coarse[1:], refined[1:], strict=True):
            torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)
        for row, count in enumerate(counts):
            values = samples[row, :count]
            if count == 0 or row >= rows - 3:
                torch.testing.assert_close(
                    refined[0][row], coarse[0][row], rtol=0, atol=0, equal_nan=True
                )
                continue
            # Even the smallest rung exceeds capacity at this physical length,
            # so the unchanged chooser retains rung 2.
            if row not in (7, 8):
                assert coarse[3][row].tolist() == [count] * 3
            rank = max(count // 3, 1)
            expected = values.sort(descending=True).values[rank - 1]
            torch.testing.assert_close(refined[0][row], expected, rtol=0, atol=0)
            assert int((values > refined[0][row]).sum()) < rank
            assert int((values >= refined[0][row]).sum()) >= rank
        if tied:
            assert refined[0][1] < coarse[0][1]


# Segmented candidate selection


@pytest.mark.parametrize("capacity", [6144, 8192])
@pytest.mark.parametrize("segments", [1, 4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("top_k", [1, 8, 32, 128, 512, 2048])
def test_segmented_packed_topk(segments, top_k, capacity):
    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a")
    # One pool is H64-only at 8192; lane-local banks use 8192 split 32..128 ways.
    if capacity % segments or (segments == 1 and capacity != 8192):
        pytest.skip("not a production capacity/segment layout")
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectivePackedTopKKernel,
    )

    torch.manual_seed(1)
    rows = 9
    bank = capacity // segments
    scores = torch.randn(rows, capacity, device="cuda")
    scores[5].zero_()
    indices = torch.arange(capacity, device="cuda", dtype=torch.int64).expand(rows, -1)
    values = (scores.view(torch.int32).long() << 32) | indices
    counts = torch.zeros((rows, segments + 1), dtype=torch.int32, device="cuda")
    counts[1, 1] = min(top_k - 1, bank)
    counts[2, 1:] = bank
    counts[3, 1] = bank + 1  # Segment overflow even below global capacity.
    counts[4, 1:] = bank // 2
    counts[5, 1:] = bank
    counts[6, 1:] = 1  # Inactive slots must never participate.
    short = min(top_k - 1, bank)
    counts[7, 1] = short  # Short causal row that published all of its keys.
    counts[8, 1] = short  # Short causal row missing its last key.
    row_ends = torch.full((rows,), 1 << 30, dtype=torch.int32, device="cuda")
    row_ends[7] = short
    row_ends[8] = short + 1
    lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
    flags = torch.empty_like(lengths)
    selected = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    kernel = SelectivePackedTopKKernel(top_k, capacity, segments)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        from_dlpack(values, assumed_align=16),
        from_dlpack(lengths, assumed_align=16),
        from_dlpack(selected, assumed_align=16),
        cutlass.Int32(rows),
        from_dlpack(counts, assumed_align=16),
        from_dlpack(flags, assumed_align=16),
        from_dlpack(row_ends, assumed_align=16),
    )
    compiled = cute.compile(kernel.segmented, *args, stream)
    compiled(*args, stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    for replay in range(2):
        if replay:
            counts[:, 1:].zero_()
        graph.replay()
        keep = counts[:, 1:].clamp(max=bank)
        total = counts[:, 1:].sum(1)
        overflow = (counts[:, 1:] > bank).any(1)
        torch.testing.assert_close(lengths.long(), keep.sum(1))
        torch.testing.assert_close(
            counts[:, 0].long(), torch.where(overflow, capacity + 1, total)
        )
        # Only rows short of min(top_k, row_end) need repair.
        required = row_ends.long().clamp(max=top_k)
        expected_flags = (total < required) | (total > capacity) | overflow
        torch.testing.assert_close(flags.bool(), expected_flags)
        if not replay:
            # total < top_k alone would send the complete short row to repair.
            assert int(flags[7]) == 0 and int(flags[8]) == 1
        for row in range(rows):
            valid = (torch.arange(capacity, device="cuda") % bank) < keep[
                row
            ].repeat_interleave(bank)
            valid_indices = valid.nonzero().flatten()
            n = min(top_k, valid_indices.numel())
            got = selected[row, :n].long()
            assert got.unique().numel() == n
            assert valid[got].all()
            torch.testing.assert_close(
                scores[row, got].sort().values,
                scores[row, valid].topk(n).values.sort().values,
                atol=0,
                rtol=0,
            )
            assert (selected[row, n:] == -1).all()


@_requires_selective_sm100
def test_h64_candidate_variant_requires_128_token_pages():
    """Smaller pages keep the pooled layout and its exact selection."""
    inputs = list(_inputs(q_rows=37, kv_rows=16384, num_heads=64))
    q, kv, _scale, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, 37], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs(tuple(inputs), block_size=64)
    engine = SelectiveEngineCase(math_mode="relu_strict")
    engine.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=512,
        num_heads=64,
        head_dim=128,
        num_sms=2,
    )
    assert not engine._candidate_schedule.h64_candidate
    selected = engine.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("h64_candidate", [False, True])
def test_single_pool8192_changed_map_tail_graph(
    top_k, ratio, h64_candidate, monkeypatch
):
    """Split-1 H64 candidates, partial Q/page and mutable map replay.

    The pooled layout publishes one repair-wide pool; the specialized H64
    candidate variant splits the same backing into four warp segments.
    """
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    physical = 16387
    q, kv, scales, weights, _, _ = _inputs(9, physical, 64)
    cu_q = torch.tensor([0, 9], device="cuda", dtype=torch.int32)
    cu_physical = torch.tensor([0, physical], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_physical))
    tokens = physical * ratio + ratio - 1
    cu_tokens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    raw = fused.flatten(1)
    last = int(table[0, -1].item())
    raw[last, : 128 * 128].view(128, 128)[physical % 128 :].fill_(0x7F)
    wrapper = SelectiveEngineCase()
    _force_h64_candidate_layout(monkeypatch, h64_candidate)
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=64,
        num_sms=4,
        compression_ratio=ratio,
    )
    assert wrapper._candidate_schedule.candidate_capacity == 8192
    assert wrapper._candidate_schedule.sample_count_segments == 32
    assert wrapper._candidate_schedule.h64_candidate is h64_candidate
    assert wrapper._candidate_counts.shape[1] == (5 if h64_candidate else 2)
    assert wrapper._candidate_values.shape[1] == 8192
    assert wrapper._sample_threshold_launch[1][-2:] == (
        wrapper._candidate_schedule.candidate_capacity,
        wrapper._candidate_schedule.count_segments,
    )
    assert not wrapper._compact_repair
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, fused, weights, table)
    for changed in (False, True):
        if changed:
            table[:, :2].copy_(table[:, :2].flip(1))
            weights.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        mapped = raw[table[0].long()]
        keys = mapped[:, : 128 * 128].contiguous().view(q.dtype).reshape(-1, 128)
        scale = mapped[:, 128 * 128 :].contiguous().view(torch.float32).reshape(-1)
        for row in range(9):
            end = (tokens - 9 + row + 1) // ratio
            scores = (
                (q[row].float() @ keys[:end].float().T).relu_() * weights[row, :, None]
            ).sum(0) * scale[:end]
            ids = output[row].long()
            assert ((ids >= 0) & (ids < end)).all() and ids.unique().numel() == top_k
            torch.testing.assert_close(
                scores[ids].sort().values,
                scores.topk(top_k).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
@pytest.mark.parametrize("ratio", [1, 4])
def test_lane_local8192_changed_map_tail_graph(top_k, ratio, monkeypatch):
    """K>=1024 H8 prefill lane banks use the repair-wide backing; K512 keeps 6144."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    physical, rows = 16387, 33
    q, kv, scales, weights, _, _ = _inputs(rows, physical, 8)
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_physical = torch.tensor([0, physical], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_physical))
    tokens = physical * ratio + ratio - 1
    cu_tokens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    raw = fused.flatten(1)
    last = int(table[0, -1].item())
    raw[last, : 128 * 128].view(128, 128)[physical % 128 :].fill_(0x7F)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_tokens,
        table,
        top_k=top_k,
        num_heads=8,
        num_sms=4,
        compression_ratio=ratio,
    )
    schedule = wrapper._candidate_schedule
    assert schedule.publication == "lane_local"
    assert schedule.candidate_capacity == (8192 if top_k >= 1024 else 6144)
    assert wrapper._candidate_counts.shape[1] == 1 + 32 * schedule.split_kv
    assert wrapper._candidate_values.shape[1] == 8192
    assert wrapper._sample_threshold_launch[1][-2:] == (
        schedule.candidate_capacity,
        schedule.count_segments,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = wrapper.run(q, fused, weights, table)
    for changed in (False, True):
        if changed:
            table[:, :2].copy_(table[:, :2].flip(1))
            weights.mul_(0.75)
        graph.replay()
        torch.cuda.synchronize()
        mapped = raw[table[0].long()]
        keys = mapped[:, : 128 * 128].contiguous().view(q.dtype).reshape(-1, 128)
        scale = mapped[:, 128 * 128 :].contiguous().view(torch.float32).reshape(-1)
        for row in range(rows):
            end = (tokens - rows + row + 1) // ratio
            scores = (
                (q[row].float() @ keys[:end].float().T).relu_() * weights[row, :, None]
            ).sum(0) * scale[:end]
            ids = output[row].long()
            assert ((ids >= 0) & (ids < end)).all() and ids.unique().numel() == top_k
            torch.testing.assert_close(
                scores[ids].sort().values,
                scores.topk(top_k).values.sort().values,
                atol=5e-5,
                rtol=1e-5,
            )


@_requires_selective_sm100
@pytest.mark.parametrize("segments", [0, 4])
@pytest.mark.parametrize("top_k", [7, 512, 2048])
def test_finalizer_flags_only_incomplete_rows(segments, top_k):
    """Single-row plans finalize candidates; short complete rows stay unflagged."""
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    from flashinfer.attn_scores._selective_contract import _REPAIR_CAPACITY

    capacity, rows = _REPAIR_CAPACITY, 6
    span = capacity // max(segments, 1)
    totals = [0, top_k - 1, top_k - 1, top_k, top_k + 3, 0]
    ends = [0, top_k - 1, top_k, 1 << 30, 1 << 30, 1 << 30]
    counts = torch.zeros((rows, segments + 1), dtype=torch.int32, device="cuda")
    for row, total in enumerate(totals):
        if segments:
            counts[row, 1] = min(total, span)
            counts[row, 2] = total - min(total, span)
        else:
            counts[row, 0] = total
    row_ends = torch.tensor(ends, dtype=torch.int32, device="cuda")
    expected_new = [t < min(top_k, e) for t, e in zip(totals, ends, strict=True)]
    expected_old = [t < top_k for t in totals]
    # The short complete row distinguishes the two conditions.
    assert expected_new[1] != expected_old[1]
    for ends_arg, expected in ((row_ends, expected_new), (None, expected_old)):
        values = torch.zeros((rows, capacity), dtype=torch.int64, device="cuda")
        lengths = torch.empty(rows, dtype=torch.int32, device="cuda")
        flags = torch.empty(rows, dtype=torch.int32, device="cuda")
        _finalize_fp8_mqa_packed_candidates(
            counts.clone(),
            values,
            lengths,
            flags,
            rows=rows,
            capacity=capacity,
            top_k=top_k,
            row_ends=ends_arg,
        )
        torch.cuda.synchronize()
        assert flags.bool().tolist() == expected


# Exact repair


@_requires_selective_cuda
def test_repair_threshold_negative_overflow_is_not_nan():
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveRepairThresholdKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    counts = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
    flags = torch.ones(3, dtype=torch.int32, device="cuda")
    thresholds = torch.full((3,), -60000.0, device="cuda")
    bins = torch.tensor([2008, 2016, 2047], dtype=torch.int32, device="cuda")
    ends = torch.full((3,), 8192, dtype=torch.int32, device="cuda")
    repaired_thresholds = torch.empty_like(thresholds)
    repaired_ends = torch.empty_like(ends)
    tensors = (
        counts,
        flags,
        thresholds,
        bins,
        ends,
        repaired_thresholds,
        repaired_ends,
    )
    args = tuple(
        from_dlpack(t).mark_layout_dynamic(leading_dim=1)
        if t.ndim == 2
        else from_dlpack(t).mark_compact_shape_dynamic(mode=0, divisibility=1)
        for t in tensors
    ) + (cutlass.Int32(3),)
    stream = _current_stream(counts.device)
    kernel = cute.compile(SelectiveRepairThresholdKernel(512), *args, stream)
    kernel(*args, stream)
    assert torch.isneginf(repaired_thresholds).all()
    torch.testing.assert_close(repaired_ends, ends)


def _prepare_radix_pass(
    histogram, prefixes, ranks, thresholds, *, clear_histogram, shift
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveTopKRadixSelectKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    args = (
        from_dlpack(histogram).mark_layout_dynamic(leading_dim=1),
        *(
            from_dlpack(t).mark_compact_shape_dynamic(mode=0, divisibility=1)
            for t in (prefixes, ranks, thresholds)
        ),
        cutlass.Int32(histogram.shape[0]),
    )
    stream = _current_stream(histogram.device)
    compiled = cute.compile(
        SelectiveTopKRadixSelectKernel(clear_histogram=clear_histogram),
        *args,
        cutlass.Int32(shift),
        stream,
    )
    return compiled, args, stream


@_requires_selective_cuda
@pytest.mark.parametrize("clear_histogram", [False, True])
def test_exact_radix_starts_a_new_prefix_on_each_first_pass(clear_histogram):
    """Compact repair can reuse a slot whose previous score key was nonzero."""
    import cutlass

    histogram = torch.zeros((1, 256), dtype=torch.int32, device="cuda")
    prefixes = torch.full((1,), -559038737, dtype=torch.int32, device="cuda")
    ranks = torch.full((1,), 8, dtype=torch.int32, device="cuda")
    thresholds = torch.empty_like(prefixes)
    compiled, args, stream = _prepare_radix_pass(
        histogram,
        prefixes,
        ranks,
        thresholds,
        clear_histogram=clear_histogram,
        shift=24,
    )
    for key in (0xC4800000, 0x3B7FFFFF, 0xBF800000, 0x407FFFFF, 0xC4800000):
        ranks.fill_(8)
        for shift in (24, 16, 8, 0):
            if not clear_histogram:
                histogram.zero_()
            histogram[0, (key >> shift) & 255] = 512
            compiled(*args, cutlass.Int32(shift), stream)
            observed = int(prefixes.item()) & 0xFFFFFFFF
            assert observed == key >> shift
            if clear_histogram:
                assert torch.count_nonzero(histogram).item() == 0
        expected_bits = key ^ (0x80000000 if key & 0x80000000 else 0xFFFFFFFF)
        assert (int(thresholds.item()) & 0xFFFFFFFF) == expected_bits


@_requires_selective_cuda
@pytest.mark.parametrize("rows", [33, 257])
def test_exact_radix_consume_reset_matches_histogram_reference(rows):
    import cutlass

    torch.manual_seed(1)
    histogram = torch.zeros((rows, 256), dtype=torch.int32, device="cuda")
    prefixes = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ranks = torch.zeros_like(prefixes)
    thresholds = torch.full_like(prefixes, -1)
    compiled, args, stream = _prepare_radix_pass(
        histogram, prefixes, ranks, thresholds, clear_histogram=True, shift=0
    )
    for active in (rows, 0, 1, rows // 2, rows):
        histogram.random_(0, 17)
        histogram[active:].zero_()
        ranks.copy_(histogram.sum(1).clamp(min=1).div(2, rounding_mode="floor"))
        initial_ranks = ranks.clone()
        descending = histogram.flip(1).cumsum(1)
        offset = (descending >= ranks[:, None]).to(torch.int32).argmax(1)
        expected_bins = 255 - offset
        before = torch.where(
            offset > 0,
            descending.gather(1, (offset - 1).clamp(min=0)[:, None]).flatten(),
            0,
        )
        prefixes.fill_(0x123456)
        compiled(*args, cutlass.Int32(0), stream)
        torch.testing.assert_close(
            prefixes[:active].long(), (0x123456 << 8) | expected_bins[:active]
        )
        torch.testing.assert_close(
            ranks[:active].long(), initial_ranks[:active] - before[:active]
        )
        torch.testing.assert_close(thresholds[:active], ~prefixes[:active])
        assert torch.count_nonzero(histogram).item() == 0
        assert (prefixes[active:] == 0x123456).all()


# Repair metadata and queue reset


@_requires_selective_sm100
@pytest.mark.parametrize("query_tile", [1, 2, 4, 16])
@pytest.mark.parametrize("use_active_count", [False, True])
def test_active_metadata_graph_replay_and_prefix_bounds(query_tile, use_active_count):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flashinfer.attn_scores.kernels.selective_logits_metadata import (
        SelectiveLogitsExplicitMetadataScheduleKernel,
    )
    from flashinfer.attn_scores.selective_logits import _current_stream

    rows, sms = 33, 4
    tiles = (rows + query_tile - 1) // query_tile
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 8192], dtype=torch.int32, device="cuda")
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full_like(starts, 8192)
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    contexts = torch.full((tiles,), -1, dtype=torch.int32, device="cuda")
    metadata = torch.full(
        (tiles, 4 + 2 * query_tile), -1, dtype=torch.int32, device="cuda"
    )
    schedule = torch.empty((sms + 1, 2), dtype=torch.int32, device="cuda")
    prefix_storage = torch.full((tiles + 32,), -777, dtype=torch.int32, device="cuda")
    prefix = prefix_storage[:tiles]
    args = (
        from_dlpack(cu_q),
        from_dlpack(cu_kv),
        from_dlpack(starts),
        from_dlpack(ends),
        cutlass.Int32(0),
        from_dlpack(count) if use_active_count else None,
        from_dlpack(contexts),
        from_dlpack(metadata),
        from_dlpack(schedule),
        from_dlpack(prefix),
        cutlass.Int32(1),
    )
    stream = _current_stream(cu_q.device)
    compiled = cute.compile(
        SelectiveLogitsExplicitMetadataScheduleKernel(sms, query_tile, 256, paged=True),
        *args,
        stream,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args, _current_stream(cu_q.device))
    for requested in (-1, 0, 1, 17, 33, 40, 2, 0):
        count.fill_(requested)
        graph.replay()
        active = min(max(requested, 0), rows) if use_active_count else rows
        live_tiles = (active + query_tile - 1) // query_tile
        assert bool((prefix_storage[tiles:] == -777).all())
        assert bool((contexts[:live_tiles] == 8192).all())
        expected = []
        total = live_tiles * 32
        for sm in range(sms + 1):
            work = sm * (total // sms) + min(sm, total % sms)
            expected.append((work // 32, work % 32))
        torch.testing.assert_close(
            schedule, torch.tensor(expected, device="cuda", dtype=torch.int32)
        )
        if live_tiles:
            assert (
                int(metadata[live_tiles - 1, 2])
                == active - (live_tiles - 1) * query_tile
            )
            assert int(prefix[live_tiles - 1]) == total


# Conditional repair


@_requires_selective_sm100
@pytest.mark.parametrize("compact", [False, True])
def test_repair_if_skips_body_and_resets_visible_state(compact):
    from flashinfer.jit.selective_repair_if import get_selective_repair_if_module

    module = get_selective_repair_if_module()
    flags = torch.zeros(33, dtype=torch.int32, device="cuda")
    second = torch.empty_like(flags)
    error = torch.empty(1, dtype=torch.int32, device="cuda")
    first_count = torch.empty_like(error) if compact else None
    second_count = torch.empty_like(error) if compact else None
    counter = torch.zeros_like(error)
    observed = torch.empty_like(counter)
    body_owner = module.create_body_stream(flags.device.index)
    body_stream = torch.cuda.ExternalStream(body_owner(), device=flags.device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        module.preflight_body(flags, body_stream.cuda_stream)
        state = module.begin_if(flags, second, error, first_count, second_count)
        module.begin_body(flags, body_stream.cuda_stream, state[2])
        with torch.cuda.stream(body_stream):
            counter.add_(1)
        module.end_body(flags, body_stream.cuda_stream, state[2])
        module.finish_if(flags, state)
        # This consumer must depend on the conditional body, not its predicate.
        observed.copy_(counter)
    for active in (0, 1, 33, 0):
        flags.zero_()
        flags[:active].fill_(1)
        second.fill_(17)
        error.fill_(19)
        counter.zero_()
        if compact:
            first_count.fill_(23)
            second_count.fill_(29)
        graph.replay()
        assert observed.item() == (active > 0)
        assert error.item() == 0
        assert not second.any().item()
        if compact:
            assert first_count.item() == second_count.item() == 0


# Sampling skew and fused repair


def _assert_repair_position_storage(engine):
    """Check exact views without inspecting expired scratch payloads."""
    positions = engine._repair_topk_positions_i32
    whole, greater, equal = (
        engine._exact_indices,
        engine._greater_indices,
        engine._equal_indices,
    )
    rows, top_k = engine._rows, engine._top_k
    bank_rows = greater.shape[0]
    assert positions.shape == (rows, top_k)
    assert positions.dtype == greater.dtype == equal.dtype == torch.int32
    assert positions.is_contiguous() and positions.stride() == (top_k, 1)
    assert greater.is_contiguous() and equal.is_contiguous()
    assert greater.stride() == equal.stride() == (top_k, 1)
    assert greater.shape == equal.shape == (bank_rows, top_k)
    assert whole.shape == (2 * bank_rows, top_k)
    assert bank_rows >= rows and bank_rows % 4 == 0
    assert greater.storage_offset() == whole.storage_offset() == 0
    assert equal.storage_offset() == bank_rows * top_k
    assert equal.data_ptr() - greater.data_ptr() == bank_rows * top_k * 4
    assert greater.untyped_storage().data_ptr() == equal.untyped_storage().data_ptr()
    h64 = (
        engine._candidate_prepared.num_heads == 64
        and engine._candidate_prepared.head_dim == 128
    )
    if h64:
        assert (
            positions.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
        )
        assert positions.data_ptr() == greater.data_ptr()
        assert positions.storage_offset() == greater.storage_offset()
        assert positions.numel() * positions.element_size() <= bank_rows * top_k * 4
        assert positions.untyped_storage().nbytes() == whole.numel() * 4
    else:
        assert (
            positions.untyped_storage().data_ptr() != whole.untyped_storage().data_ptr()
        )


@_requires_selective_sm100
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("heads", [4, 8, 64])
def test_internal_sample_ragged_page_lookup(page_size, heads):
    import flashinfer.attn_scores.selective_logits as sl

    head_dim = 64 if heads == 4 else 128
    data = _inputs(17, 9001, heads, head_dim)
    q, kv, scales, weights, cu_q, cu_kv = data
    fused, table = _paged_inputs(data, page_size)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_kv,
        table,
        num_heads=heads,
        head_dim=head_dim,
        num_sms=4,
    )
    for _ in range(2):
        wrapper.run(q, fused, weights, table)
        assert wrapper._exact_error.item() == 0
    # Independent eager sample oracle for a short request and a long request.
    for q_begin, q_end, kv_begin, kv_end in ((0, 1, 0, 132), (1, 17, 132, 9001)):
        length = kv_end - kv_begin
        pages = math.ceil(length / page_size)
        count = min(pages, sl._SAMPLE_CAPACITY // page_size)
        sampled = [
            page * page_size + offset
            for page in (i * (pages - 1) // max(count - 1, 1) for i in range(count))
            for offset in range(min(page_size, length - page * page_size))
        ]
        offsets = torch.tensor(sampled, device=q.device, dtype=torch.long)
        scores = torch.einsum(
            "qhd,kd->qhk", q[q_begin:q_end].float(), kv[kv_begin + offsets].float()
        ).relu_()
        scores = (scores * weights[q_begin:q_end, :, None]).sum(1)
        scores *= scales[kv_begin + offsets][None]
        ends = wrapper._sample_ends[q_begin:q_end]
        valid = torch.arange(len(sampled), device=q.device)[None] < ends[:, None]
        torch.testing.assert_close(
            wrapper._sample_scores[q_begin:q_end, : len(sampled)][valid],
            scores[valid],
            atol=2e-5,
            rtol=2e-5,
        )


@_requires_selective_sm100
@pytest.mark.parametrize(
    "page_size,top_k,high_on_sampled",
    [
        (32, 512, False),
        (32, 8, True),
        (64, 128, False),
        (64, 32, True),
        (128, 8, False),
        (128, 512, True),
    ],
)
def test_engine_page_correlated_sampling(page_size, top_k, high_on_sampled):
    rows, length, heads, dim = 17, 9001, 8, 128
    pages = math.ceil(length / page_size)
    sample_pages = min(pages, 3840 // page_size)
    selected_pages = {
        i * (pages - 1) // max(sample_pages - 1, 1) for i in range(sample_pages)
    }
    values = torch.tensor(
        [
            float((i // page_size in selected_pages) == high_on_sampled)
            for i in range(length)
        ],
        device="cuda",
    )
    q = torch.ones((rows, heads, dim), device="cuda").to(torch.float8_e4m3fn)
    kv = values[:, None].expand(length, dim).contiguous().to(torch.float8_e4m3fn)
    scales = torch.ones(length, device="cuda")
    weights = torch.ones((rows, heads), device="cuda")
    cu_q = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv), page_size)
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table, top_k=top_k, num_sms=4)
    # Independent eager FP32 score oracle, not the sampler or either selector.
    scores = torch.einsum("qhd,kd->qhk", q.float(), kv.float()).relu_()
    scores = (scores * weights[:, :, None]).sum(1) * scales[None]
    for _ in range(3):
        indices = wrapper.run(q, fused, weights, table).long()
        assert wrapper._exact_error.item() == 0
        for row in range(rows):
            end = length - rows + row + 1
            assert ((indices[row] >= 0) & (indices[row] < end)).all()
            assert indices[row].unique().numel() == top_k
            torch.testing.assert_close(
                scores[row, indices[row]].sort().values,
                scores[row, :end].topk(top_k).values.sort().values,
                atol=0,
                rtol=0,
            )


# Chain replanning


# Selective payload lifecycle


# Capture and stream ownership


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("query_tile", [1, 16])
def test_benchmark_full_step_refreshes_lengths_and_page_maps(ratio, query_tile):
    """The separated layer consumes the last device-planned lengths and map."""
    from flashinfer.attn_scores.fp8_kernel_config import FP8MQAKernelConfig
    from flashinfer.attn_scores.query_group_metadata import FP8QueryGroups

    spec = importlib.util.spec_from_file_location(
        "bench_fp8_mqa_selective_logits",
        os.path.join(
            os.path.dirname(__file__),
            "../../benchmarks/bench_fp8_mqa_selective_logits.py",
        ),
    )
    assert spec is not None and spec.loader is not None
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    inputs = _request_inputs([query_tile, query_tile], [1024, 1024])
    q, keys, scales, weights, qo_indptr, physical_kv = inputs
    pool, table = _paged_inputs(inputs)
    kv_indptr = physical_kv * ratio
    groups = FP8QueryGroups.build(
        qo_indptr.cpu().tolist(), kv_indptr.cpu().tolist(), ratio, query_tile
    )
    full = benchmark._FullGVR(
        q,
        pool,
        weights,
        table,
        groups,
        FP8MQAKernelConfig(128, 1, 160, query_tile, num_q_stages=3, num_kv_stages=3),
        512,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
    )
    assert full.scorer_mode == (
        "ordinary-causal-envelope"
        if ratio == 4 and query_tile > 1
        else "ordinary-causal"
    )
    assert full.scorer_width == 1024 + (
        query_tile - 1 if ratio == 4 and query_tile > 1 else 0
    )
    full.step()
    full.layer()
    torch.cuda.synchronize()
    step_graph, layer_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(step_graph):
        full.step()
    with torch.cuda.graph(layer_graph):
        full.layer()
    for lengths in ([1024, 1024], [768, 512]):
        if lengths[0] == 768:
            # Keep the pool address but replace its data, so a private stale
            # pool copy cannot satisfy the captured layer's reference check.
            key_bytes = keys.view(torch.uint8)
            key_bytes.copy_(key_bytes.flip(0))
            pool.copy_(_fuse_paged_kv(keys, scales).flip(0))
        kv_indptr.copy_(
            torch.tensor(
                [0, lengths[0] * ratio, sum(lengths) * ratio],
                device=q.device,
                dtype=torch.int32,
            )
        )
        table.copy_(table.roll(1, dims=1))
        step_graph.replay()
        layer_graph.replay()
        torch.cuda.synchronize()
        oracle = []
        for request, length in enumerate(lengths):
            for local in range(query_tile):
                end = (length * ratio - query_tile + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                page = pool.shape[0] - 1 - table[request, slots // 128].long()
                physical = page * 128 + slots % 128
                row = request * query_tile + local
                dots = q[row].float() @ keys[physical].float().T
                oracle.append(
                    (dots.relu() * weights[row, :, None]).sum(0) * scales[physical]
                )
        _check_selection(full.output, oracle, 512)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_plan_capture_requires_prepared_pool(ratio):
    """A new wrapper must not capture a step before its first pool binding."""
    inputs = _request_inputs([2], [1024])
    q, _, _, weights, qo_indptr, physical_kv = inputs
    pool, table = _paged_inputs(inputs)
    kv_indptr = physical_kv * ratio
    for capacity_rows, layout in (
        (2, dict(kv_layout=pool)),
        (4, dict(num_pages=pool.shape[0], device=pool.device)),
    ):
        kv_indptr.copy_(physical_kv * ratio)
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            compression_ratio=ratio,
            max_context_length=1024 * ratio,
            max_chunk_size=capacity_rows,
            max_batch_size=1,
            **layout,
        )
        rejected_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(rejected_graph),
            pytest.raises(RuntimeError, match="prepare_pool.*capturing plan"),
        ):
            wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=2)

        wrapper.prepare_pool(pool)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.plan(qo_indptr, kv_indptr, table, num_query_rows=2)
            selected = wrapper.run(q, weights, pool)
        for length in (1024, 768):
            kv_indptr[1] = length * ratio
            graph.replay()
            torch.cuda.synchronize()
            expected_inputs = (*inputs[:5], kv_indptr // ratio)
            _check_selection(selected, _oracle_rows(expected_inputs), 512)
            assert wrapper.error.item() == wrapper.repair_error.item() == 0
        del graph, rejected_graph


@_requires_selective_sm100
def test_capacity_step_shape_writes_skip_repeats_until_outer_capture():
    """Repeated steps skip shape writes only until a capture can rewrite them."""
    from flashinfer.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    capacity = _SelectiveCapacity(8, 128, 32, 1, 65536, 8, 4)
    state = _CapacityStepInputs(capacity, device="cuda", num_pages=8)
    table = torch.zeros_like(state.block_table)
    shape_a = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    shape_b = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

    def stage(rows, prefix):
        state.stage_step(prefix, prefix, table, num_query_rows=rows)

    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]
    state.step_shape.fill_(-7)
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [-7, -7]
    stage(3, shape_b)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [3, 3]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stage(3, shape_b)
    stage(5, shape_a)
    graph.replay()
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [3, 3]
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]
    state.step_shape.fill_(-7)
    stage(5, shape_a)
    torch.cuda.synchronize()
    assert state.step_shape.tolist() == [5, 2]


@_requires_selective_sm100
def test_capacity_constructor_contract(monkeypatch):
    """Selective construction owns capacities and one pool layout; full takes none."""
    inputs = _request_inputs([2], [1024])
    q, _, _, weights, qo, kv = inputs
    pool, table = _paged_inputs(inputs)
    capacities = dict(max_context_length=1024, max_chunk_size=2, max_batch_size=1)
    with pytest.raises(ValueError, match="only supported by selective"):
        FP8PagedMQATopKWrapper(strategy="full", **capacities)
    with pytest.raises(ValueError, match="only supported by selective"):
        FP8PagedMQATopKWrapper(strategy="full", kv_layout=pool)
    with pytest.raises(ValueError, match="requires max_context_length"):
        FP8PagedMQATopKWrapper(strategy="selective", kv_layout=pool)
    with pytest.raises(ValueError, match="kv_layout or num_pages and device"):
        FP8PagedMQATopKWrapper(strategy="selective", **capacities)
    with pytest.raises(ValueError, match="either kv_layout or num_pages"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool, num_pages=1
        )
    with pytest.raises(ValueError, match="num_pages must be"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=-1, device=pool.device
        )
    with pytest.raises(ValueError, match="device must be CUDA"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=1, device="cpu"
        )
    with pytest.raises(ValueError, match="kv_layout must be"):
        FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool.view(torch.int8)
        )
    rejected_graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(rejected_graph),
        pytest.raises(RuntimeError, match="outside capture"),
    ):
        FP8PagedMQATopKWrapper(strategy="selective", **capacities, kv_layout=pool)
    del rejected_graph

    from flashinfer.attn_scores import selective_logits

    def reject_compile(*args, **kwargs):
        raise AssertionError("construction and initial staging must not compile")

    with monkeypatch.context() as cold:
        cold.setattr(selective_logits, "_compile_for_device", reject_compile)
        exemplar = FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, kv_layout=pool
        )
        paged = FP8PagedMQATopKWrapper(
            strategy="selective", **capacities, num_pages=pool.shape[0], device="cuda"
        )
        for wrapper in (exemplar, paged):
            wrapper.plan(qo, kv, table)
    for wrapper in (exemplar, paged):
        plan = wrapper._capacity_plan
        assert wrapper.strategy == "selective"
        assert wrapper.math_mode == "relu_strict"
        assert plan._pool_layout == exemplar._capacity_plan._pool_layout
        assert plan._engine is None  # The scoring chain compiles lazily.
        # A bad initial step must be validated when the first pool is bound,
        # through either explicit preparation or eager run's automatic binding.
        wrapper.plan(qo, torch.zeros_like(kv), table)
        if wrapper is exemplar:
            wrapper.prepare_pool(pool)
        invalid = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert wrapper.error.item() & 2
        assert bool((invalid == -1).all())
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert plan._engine is not None
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
def test_capacity_wrappers_isolate_caller_graphs():
    """Live wrappers keep separate workspaces across alternating caller graphs."""
    inputs = _request_inputs([2, 1], [9001, 7000])
    q, _, _, weights, cu_q, cu_kv = inputs
    pool, table = _paged_inputs(inputs)
    wrappers, graphs, outputs, expected = [], [], [], []
    for top_k in (512, 2048, 512):
        # Each wrapper stays referenced: a caller graph addresses its workspace.
        w = FP8PagedMQATopKWrapper(
            strategy="selective",
            top_k=top_k,
            max_context_length=9001,
            max_chunk_size=3,
            max_batch_size=2,
            kv_layout=pool,
        )
        w.plan(cu_q, cu_kv, table)
        expected.append(w.run(q, weights, pool).clone())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            w.plan(cu_q, cu_kv, table)
            outputs.append(w.run(q, weights, pool))
        wrappers.append(w)
        graphs.append(graph)
    torch.cuda.synchronize()
    for _ in range(3):
        for w, graph, output, want in zip(
            wrappers, graphs, outputs, expected, strict=True
        ):
            graph.replay()
            torch.cuda.synchronize()
            assert w.error.item() == w.repair_error.item() == 0
            # Tie-aware: each wrapper's own width and oracle, not bitwise sets.
            assert output.shape == want.shape
            _check_selection(output, _oracle_rows(inputs), want.shape[1])


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_capture_binds_unprepared_pool(ratio):
    """After one preparation, caller capture binds an unseen equal-layout pool."""
    inputs = _request_inputs([1, 1], [9001, 9001])
    q, keys, scales, weights, cu_q, physical_kv = inputs
    pool_a, table = _paged_inputs(inputs)
    inputs_b = (q, keys, scales.flip(0).contiguous(), weights, cu_q, physical_kv)
    pool_b, table_b = _paged_inputs(inputs_b)
    assert torch.equal(table, table_b)
    original_kv = physical_kv * ratio

    def wrapper():
        return FP8PagedMQATopKWrapper(
            strategy="selective",
            compression_ratio=ratio,
            max_context_length=9001 * ratio,
            max_chunk_size=2,
            max_batch_size=2,
            kv_layout=pool_a,
        )

    # Nothing compiled yet: capture cannot bind any pool.
    cold = wrapper()
    cold.plan(cu_q, original_kv, table)
    rejected = torch.cuda.CUDAGraph()
    with (
        torch.cuda.graph(rejected),
        pytest.raises(RuntimeError, match="must precede caller capture"),
    ):
        cold.run(q, weights, pool_a)
    del rejected

    w = wrapper()
    w.prepare_pool(pool_a)
    out_a = torch.empty((2, 512), device=q.device, dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.plan(cu_q, original_kv, table)
        w.run(q, weights, pool_a, out=out_a)
        w.run(q, weights, pool_b, out=out_b)
    # A pool seen only under capture keeps no owned eager graph.
    assert len(w._capacity_plan._pool_graphs) == 1
    # The oracle is tie-aware; index sets may differ only among equal scores.
    for factor in (1.0, 0.5):
        weights.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        assert w.error.item() == w.repair_error.item() == 0
        _check_selection(out_a, _oracle_rows(inputs), 512)
        _check_selection(out_b, _oracle_rows(inputs_b), 512)
    # Eager use still builds the second pool's owned graph on demand.
    w.plan(cu_q, original_kv, table)
    _check_selection(w.run(q, weights, pool_b), _oracle_rows(inputs_b), 512)
    torch.cuda.synchronize()
    assert len(w._capacity_plan._pool_graphs) == 2


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_shared_layer_workspace(heads, ratio):
    """Distinct pools share scratch without stale repair bindings or lost output."""
    import weakref

    inputs = _request_inputs([1, 1], [9001, 9001], heads=heads)
    q, keys, scales, weights, cu_q, physical_kv = inputs
    q.copy_(torch.ones(q.shape, device=q.device).to(q.dtype))
    keys.copy_(torch.ones(keys.shape, device=keys.device).to(keys.dtype))
    weights.fill_(-1)
    scales.copy_(torch.linspace(1.00001, 1.0001, scales.numel(), device=q.device))
    pool_a, table = _paged_inputs(inputs)
    reversed_scales = scales.flip(0).contiguous()
    inputs_b = (q, keys, reversed_scales, weights, cu_q, physical_kv)
    pool_b, table_b = _paged_inputs(inputs_b)
    assert torch.equal(table, table_b)
    exemplar = pool_a.clone()
    exemplar_ref = weakref.ref(exemplar)
    # Layout inspection must not rely on collection of import/decorator frames
    # to release the exemplar, even on the first construction in a process.
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            num_heads=heads,
            compression_ratio=ratio,
            max_context_length=9001 * ratio,
            max_chunk_size=2,
            max_batch_size=2,
            kv_layout=exemplar,
        )
        del exemplar
        assert exemplar_ref() is None
    finally:
        if gc_enabled:
            gc.enable()
    wrapper.prepare_pool(pool_a)
    engine = wrapper._capacity_plan._engine
    workspace = engine._candidate_values
    wrapper.prepare_pool(pool_b)
    assert wrapper._capacity_plan._engine is engine
    assert engine._candidate_values is workspace
    original_kv = physical_kv * ratio
    wrapper.plan(cu_q, original_kv, table)
    out_a = torch.empty((2, 512), device=q.device, dtype=torch.int32)
    out_b = torch.empty_like(out_a)
    wrapper.run(q, weights, pool_a, out=out_a)
    torch.cuda.synchronize()
    assert engine._second_repair_flags.count_nonzero().item() == 2
    saved_a = out_a.clone()
    wrapper.run(q, weights, pool_b, out=out_b)
    torch.cuda.synchronize()
    _check_selection(out_a, _oracle_rows(inputs), 512)
    _check_selection(out_b, _oracle_rows(inputs_b), 512)
    assert torch.equal(out_a, saved_a)
    with pytest.raises(AssertionError):
        _check_selection(out_a, _oracle_rows(inputs_b), 512)
    assert wrapper.repair_error.item() == 0
    wrapper.repair_error.fill_(8)
    wrapper.run(q, weights, pool_b)
    torch.cuda.synchronize()
    assert wrapper.repair_error.item() == 8

    # A later layer must not re-read a caller map changed after plan.
    invalid = torch.full_like(table, -1)
    wrapper.plan(cu_q, original_kv, invalid)
    invalid.copy_(table)
    for pool in (pool_a, pool_b):
        result = wrapper.run(q, weights, pool)
        torch.cuda.synchronize()
        assert wrapper.error.item() == 8
        assert (result == -1).all()
    with pytest.raises(ValueError, match="layout"):
        wrapper.run(q, weights, pool_a[:-1])

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(cu_q, original_kv, table)
        with pytest.raises(RuntimeError, match="clear_pools must run outside"):
            wrapper.clear_pools()
        wrapper.run(q, weights, pool_a, out=out_a)
        wrapper.run(q, weights, pool_b, out=out_b)
    for factor in (0.5, 1.5):
        weights.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        _check_selection(out_a, _oracle_rows(inputs), 512)
        _check_selection(out_b, _oracle_rows(inputs_b), 512)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

    # Retiring pool graphs must release tensor owners without rebuilding scratch.
    del graph
    retired_pool = pool_a.clone()
    retired_ref = weakref.ref(retired_pool)
    wrapper.prepare_pool(retired_pool)
    wrapper.run(q, weights, retired_pool, out=out_a)
    wrapper.run(q, weights, pool_b, out=out_b)
    del retired_pool
    assert retired_ref() is not None
    wrapper.clear_pools()
    assert retired_ref() is None
    assert wrapper._capacity_plan._engine is engine
    assert engine._candidate_values is workspace
    assert not wrapper._capacity_plan._pool_graphs
    wrapper.run(q, weights, pool_b, out=out_b)
    torch.cuda.synchronize()
    _check_selection(out_b, _oracle_rows(inputs_b), 512)


@_requires_selective_sm100
def test_capacity_first_compact_pool_precompiles_metadata(monkeypatch, record_property):
    """A cold compact-repair pool never compiles metadata during capture."""
    import time

    from flashinfer.attn_scores import selective_logits

    # Preserve existing compiled owners while giving this plan a cold metadata
    # cache, so the assertion does not depend on earlier tests' specializations.
    monkeypatch.setattr(selective_logits, "_METADATA_COMPILED", {})
    real_compile = selective_logits._compile_for_device
    metadata_compiles = []

    def compile_outside_capture(device, kernel, *args):
        if type(kernel).__name__ in (
            "SelectiveLogitsMetadataScheduleKernel",
            "SelectiveLogitsExplicitMetadataScheduleKernel",
        ):
            assert not torch.cuda.is_current_stream_capturing()
            metadata_compiles.append(type(kernel).__name__)
        return real_compile(device, kernel, *args)

    monkeypatch.setattr(
        selective_logits, "_compile_for_device", compile_outside_capture
    )
    inputs = _request_inputs([2], [9001])
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=9001,
        max_chunk_size=2,
        max_batch_size=1,
        kv_layout=fused,
    )
    started = time.perf_counter()
    wrapper.prepare_pool(fused)
    record_property("first_prepare_pool_seconds", time.perf_counter() - started)
    assert wrapper._capacity_plan._engine._compact_repair
    assert metadata_compiles
    assert any(key[-1] for key in selective_logits._METADATA_COMPILED)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, fused)
    graph.replay()
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    _check_selection(selected, _oracle_rows(inputs), 512)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("physical_capacity", [6143, 6144, 6145])
def test_capacity_sampling_physical_boundary(ratio, physical_capacity, monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _request_inputs([3], [physical_capacity])
    q, keys, scales, weights, qo, _ = inputs
    fused, table = _paged_inputs(inputs)
    original_capacity = physical_capacity * ratio + ratio - 1
    kv = torch.tensor([0, original_capacity], dtype=torch.int32, device=q.device)
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        compression_ratio=ratio,
        max_context_length=original_capacity,
        max_chunk_size=3,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    assert engine._needs_sampling == (physical_capacity > 6144)
    if not engine._needs_sampling:
        assert bool((engine._selected_bins == 2047).all())
    wrapper.plan(qo, kv, table)
    wrapper.run(q, weights, fused)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table)
        selected = wrapper.run(q, weights, fused)
    for original_length in (
        original_capacity,
        512 * ratio + ratio - 1,
        original_capacity,
    ):
        kv[1] = original_length
        q.copy_((-q.float()).to(q.dtype))
        weights.mul_(-0.75)
        fused.copy_(fused.flip(0))
        table.copy_(fused.shape[0] - 1 - table)
        engine._candidate_counts.fill_(2**30)
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert engine._needs_sampling == (physical_capacity > 6144)
        if ratio == 4:
            _check_compressed_wrapper(selected, (q, fused, weights, qo, kv, table), 512)
        else:
            _check_selection(
                selected, _oracle_rows((q, keys, scales, weights, qo, kv)), 512
            )


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_sampling_on_off_same_bucket(ratio, monkeypatch):
    from flashinfer.attn_scores.selective_capacity import _CapacitySelectiveTopKPlan

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    inputs = _request_inputs([2], [6144])
    q, keys, scales, weights, qo, _ = inputs
    fused, table = _paged_inputs(inputs)
    kv = torch.tensor([0, 6144 * ratio], dtype=torch.int32, device=q.device)
    graphs, outputs, wrappers = [], [], []
    for force_sampling in (False, True):
        # Compare sampling and all-key publication within identical capacities.
        # The forced control is private to this test, not a public routing knob.
        with monkeypatch.context() as patch:
            if force_sampling:
                patch.setattr(
                    _CapacitySelectiveTopKPlan,
                    "_needs_sampling",
                    property(lambda _: True),
                )
            wrapper = FP8PagedMQATopKWrapper(
                strategy="selective",
                compression_ratio=ratio,
                max_context_length=6144 * ratio,
                max_chunk_size=2,
                max_batch_size=1,
                kv_layout=fused,
            )
            wrapper.prepare_pool(fused)
            assert wrapper._capacity_plan._engine._needs_sampling == force_sampling
            wrapper.plan(qo, kv, table)
            wrapper.run(q, weights, fused)
            graph = torch.cuda.CUDAGraph()
            with _capture_cuda_graph(graph):
                wrapper.plan(qo, kv, table)
                selected = wrapper.run(q, weights, fused)
            graphs.append(graph)
            outputs.append(selected)
            wrappers.append(wrapper)
    for tied in (False, True, False):
        q.copy_(torch.randn(q.shape, device=q.device).mul_(0.1).to(q.dtype))
        if tied:
            q.zero_()
        for wrapper, graph, selected in zip(wrappers, graphs, outputs, strict=True):
            graph.replay()
            torch.cuda.synchronize()
            assert wrapper.error.item() == wrapper.repair_error.item() == 0
            if ratio == 4:
                _check_compressed_wrapper(
                    selected, (q, fused, weights, qo, kv, table), 512
                )
            else:
                _check_selection(
                    selected, _oracle_rows((q, keys, scales, weights, qo, kv)), 512
                )


@_requires_selective_sm100
@pytest.mark.parametrize("heads,ratio", [(8, 1), (8, 4), (64, 1), (64, 4)])
def test_capacity_selective_chain_live_steps_and_repair(heads, ratio):
    inputs = _inputs(8, 8192, num_heads=heads)
    q, keys, scales, weights, _, _ = inputs
    fused = _fuse_paged_kv(keys, scales)
    table = (
        torch.arange(3 * (65536 // ratio // 128), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view(3, -1)
    )
    top_k = 512 if heads == 8 else 2048
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=65536,
        max_chunk_size=8,
        max_batch_size=3,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    plan = wrapper._capacity_plan
    device_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    assert plan._engine._candidate_prepared.num_sms == device_sms
    assert plan._engine._sample_prepared.num_sms == device_sms
    graph = plan._pool_graphs[plan._pool_key(fused)][1]
    wrapper.prepare_pool(fused.view_as(fused))

    def prefixes(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=q.device,
        )

    def check(q_lengths, kv_lengths, *, tied=False, observed=None):
        rows = sum(q_lengths)
        step_weights = torch.zeros_like(weights[:rows]) if tied else weights[:rows]
        qo, kv = prefixes(q_lengths), prefixes(kv_lengths)
        if observed is None:
            wrapper.plan(qo, kv, table, num_query_rows=q[:rows].shape[0])
        selected = (
            wrapper.run(q[:rows], step_weights, fused) if observed is None else observed
        )
        torch.cuda.synchronize()
        assert wrapper.error.item() == 0
        assert wrapper.repair_error.item() == 0
        expected, offset = [], 0
        for request, (q_length, kv_length) in enumerate(
            zip(q_lengths, kv_lengths, strict=True)
        ):
            for local in range(q_length):
                end = (kv_length - q_length + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                physical = table[request, slots // 128].long() * 128 + slots % 128
                dots = q[offset + local].float() @ keys[physical].float().T
                expected.append(
                    (dots.relu() * step_weights[offset + local, :, None]).sum(0)
                    * scales[physical]
                )
            offset += q_length
        _check_selection(selected, expected, top_k)
        assert plan._pool_graphs[plan._pool_key(fused)][1] is graph
        return selected

    check([1, 0, 3], [65535, 7, 8195])
    check([1, 0, 3], [65535, 7, 8195], tied=True)
    assert plan._engine._repair_flags.count_nonzero().item() > 0
    assert plan._engine._second_repair_flags.count_nonzero().item() > 0
    table.copy_(table.roll(1, dims=1))
    check([2, 0, 3], [9, 0, 17])
    check([1, 1], [1000, 1001])
    check([8], [777])
    check([], [])

    # Invalid device metadata suppresses scoring, then a valid step recovers.
    table.fill_(-1)
    wrapper.plan(prefixes([1]), prefixes([128]), table, num_query_rows=q[:1].shape[0])
    selected = wrapper.run(q[:1], weights[:1], fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == 8
    assert (selected == -1).all()
    table.copy_(
        torch.arange(table.numel(), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view_as(table)
    )
    check([2], [2049])

    # Equal-layout pools share the workspace but keep separate captured bindings.
    # Each physical row must exceed K, or every index remains a valid selection
    # after rebinding and cannot detect execution against the retired map.
    rebind_lengths = [8191 * ratio, 4095 * ratio + 1]
    previous = check([1, 2], rebind_lengths).clone()
    old_owners = fused, table
    fused, table = fused.clone(), table.roll(1, dims=1)
    wrapper.prepare_pool(fused)
    graph = plan._pool_graphs[plan._pool_key(fused)][1]
    with pytest.raises(AssertionError):
        check([1, 2], rebind_lengths, observed=previous)
    check([1, 2], rebind_lengths)
    assert old_owners[0].data_ptr() != fused.data_ptr()
    buffers = wrapper.input_buffers()
    buffers["q"][:3].copy_(q[:3])
    buffers["weights"][:3].copy_(weights[:3])
    buffers["qo_indptr"][:3].copy_(prefixes([1, 2]))
    buffers["kv_indptr"][:3].copy_(prefixes(rebind_lengths))
    destination = torch.empty((3, top_k), device=q.device, dtype=torch.int32)
    wrapper.plan(
        buffers["qo_indptr"][:3],
        buffers["kv_indptr"][:3],
        table,
        num_query_rows=buffers["q"][:3].shape[0],
    )
    assert (
        wrapper.run(buffers["q"][:3], buffers["weights"][:3], fused, out=destination)
        is destination
    )
    check([1, 2], rebind_lengths, observed=destination)
    with pytest.raises(ValueError, match="cross-alias"):
        wrapper.plan(
            buffers["kv_indptr"][:3],
            buffers["qo_indptr"][:3],
            table,
            num_query_rows=q[:3].shape[0],
        )
        wrapper.run(q[:3], weights[:3], fused)
    protected_output = (
        plan._engine._candidate_values.view(torch.int32)
        .flatten()[:top_k]
        .view(1, top_k)
    )
    with pytest.raises(ValueError, match="overlap"):
        wrapper.plan(
            prefixes([1]), prefixes([8192]), table, num_query_rows=q[:1].shape[0]
        )
        wrapper.run(q[:1], weights[:1], fused, out=protected_output)
    qo, kv = prefixes([2, 1]), prefixes([3073, 1031])
    outer = torch.cuda.CUDAGraph()
    with torch.cuda.graph(outer):
        wrapper.plan(qo, kv, table, num_query_rows=q[:3].shape[0])
        output = wrapper.run(q[:3], weights[:3], fused)
    for q_values, kv_values in (([2, 1], [3073, 1031]), ([1, 2], [999, 1003])):
        qo.copy_(prefixes(q_values))
        kv.copy_(prefixes(kv_values))
        outer.replay()
        torch.cuda.synchronize()
        check(q_values, kv_values, observed=output)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("bucket", [2, 8])
def test_capacity_decode_bucket_multirow_steps(heads, ratio, bucket):
    q, keys, scales, weights, _, _ = _inputs(bucket, 8192, num_heads=heads)
    fused = _fuse_paged_kv(keys, scales)
    table = torch.arange(8192 // 128, device=q.device, dtype=torch.int32).repeat(
        bucket, 1
    )
    top_k = 512 if heads == 8 else 2048
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=8192 * ratio,
        max_chunk_size=bucket,
        max_batch_size=bucket,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)

    def prefixes(lengths):
        return torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device=q.device,
        )

    q_lengths = [2, 0] if bucket == 2 else [1, 2, 0, 3]
    rows = sum(q_lengths)
    lengths = [8192 * ratio - request for request in range(len(q_lengths))]
    qo, kv = prefixes(q_lengths), prefixes(lengths)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=q[:rows].shape[0])
        selected = wrapper.run(q[:rows], weights[:rows], fused)
    for step_lengths in (q_lengths, list(reversed(q_lengths))):
        qo.copy_(prefixes(step_lengths))
        table.copy_(table.roll(1, dims=1))
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == 0
        assert wrapper.repair_error.item() == 0
        oracle, offset = [], 0
        for request, (q_length, kv_length) in enumerate(
            zip(step_lengths, lengths, strict=True)
        ):
            for local in range(q_length):
                end = (kv_length - q_length + local + 1) // ratio
                slots = torch.arange(end, device=q.device)
                physical = table[request, slots // 128].long() * 128 + slots % 128
                dots = q[offset + local].float() @ keys[physical].float().T
                oracle.append(
                    (dots.relu() * weights[offset + local, :, None]).sum(0)
                    * scales[physical]
                )
            offset += q_length
        _check_selection(selected, oracle, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_reused_output_clears_inactive_rows(ratio):
    """Repeated layers rewrite caller-modified output, including unscored rows."""
    rows = ratio + 2
    q, keys, scales, weights, _, _ = _inputs(rows, 128, num_heads=8)
    fused = _fuse_paged_kv(keys, scales)
    table = torch.zeros((1, 1), device=q.device, dtype=torch.int32)

    def prefix(values):
        return torch.tensor(values, device=q.device, dtype=torch.int32)

    for empty in (False, True):
        pool = fused[:0] if empty else fused
        wrapper = FP8PagedMQATopKWrapper(
            strategy="selective",
            top_k=512,
            compression_ratio=ratio,
            max_context_length=ratio,
            max_chunk_size=rows,
            max_batch_size=1,
            kv_layout=pool,
        )
        wrapper.prepare_pool(pool)
        wrapper.plan(
            prefix([0, 0 if empty else 1]),
            prefix([0, 0 if empty else ratio]),
            table,
            num_query_rows=rows,
        )
        selected = wrapper.run(q, weights, pool)
        expected = torch.full_like(selected, -1)
        if not empty:
            expected[0, 0] = 0
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, weights, pool, out=selected)
        selected.fill_(17)
        wrapper.run(q, weights, pool, out=selected)
        torch.testing.assert_close(selected, expected)
        selected.fill_(17)
        graph.replay()
        torch.testing.assert_close(selected, expected)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0

        wrapper.plan(prefix([1, 1]), prefix([0, 0]), table, num_query_rows=rows)
        selected.fill_(17)
        graph.replay()
        assert wrapper.error.item() == 2
        assert (selected == -1).all()


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_selective_public_validation(ratio):
    q, keys, scales, weights, _, _ = _inputs(2, 1024, num_heads=8)
    fused = _fuse_paged_kv(keys, scales)
    table = (
        torch.arange(2 * (8192 // ratio // 128), device=q.device, dtype=torch.int32)
        .remainder(fused.shape[0])
        .view(2, -1)
    )
    specialization = dict(strategy="selective", top_k=512, compression_ratio=ratio)
    capacities = dict(max_context_length=8192, max_chunk_size=2, max_batch_size=2)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=fused)
    wrapper.prepare_pool(fused)

    def prefix(values):
        return torch.tensor(values, device=q.device, dtype=torch.int32)

    qo, kv = prefix([0, 1, 2]), prefix([0, 1024, 2048])
    for queries, tokens, code in (
        ([1, 1, 2], [0, 1024, 2048], 2),
        ([0, 2, 1], [0, 1024, 2048], 2),
        ([0, 1, 3], [0, 1024, 2048], 4),
        ([0, 1, 2], [0, 8193, 9217], 4),
    ):
        wrapper.plan(prefix(queries), prefix(tokens), table, num_query_rows=q.shape[0])
        selected = wrapper.run(q, weights, fused)
        torch.cuda.synchronize()
        assert wrapper.error.item() == code
        assert (selected == -1).all()
    wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    count = min(512, 1024 // ratio)
    assert (selected[:, :count] >= 0).all()
    assert (selected[:, count:] == -1).all()

    with pytest.raises(ValueError, match="fit the planned"):
        wrapper.run(
            torch.empty((3, 8, 128), device=q.device, dtype=q.dtype),
            torch.empty((3, 8), device=q.device),
            fused,
        )
    with pytest.raises(ValueError, match="indptrs"):
        wrapper.plan(
            prefix([0, 1, 1, 2]),
            prefix([0, 1024, 1024, 2048]),
            table,
            num_query_rows=q.shape[0],
        )
        wrapper.run(q, weights, fused)
    with pytest.raises(ValueError, match="out must be"):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
        wrapper.run(
            q,
            weights,
            fused,
            out=torch.empty((2, 513), device=q.device, dtype=torch.int32),
        )

    q_owner = torch.empty(2 * 512 * 4, device=q.device, dtype=torch.uint8)
    aliased_q = q_owner[: q.numel()].view(q.dtype).view_as(q)
    aliased_q.copy_(q)
    with pytest.raises(ValueError, match="out must not overlap"):
        wrapper.plan(qo, kv, table, num_query_rows=aliased_q.shape[0])
        wrapper.run(
            aliased_q, weights, fused, out=q_owner.view(torch.int32).view(2, 512)
        )
    weight_owner = torch.empty((2, 512), device=q.device)
    aliased_weights = weight_owner.flatten()[: weights.numel()].view_as(weights)
    aliased_weights.copy_(weights)
    with pytest.raises(ValueError, match="out must not overlap"):
        wrapper.plan(qo, kv, table, num_query_rows=q.shape[0])
        wrapper.run(q, aliased_weights, fused, out=weight_owner.view(torch.int32))

    with pytest.raises(ValueError, match="sample virtual prefixes"):
        FP8PagedMQATopKWrapper(
            **specialization,
            max_context_length=8192,
            max_chunk_size=2,
            max_batch_size=(2**31 - 1) // 3840 + 1,
            kv_layout=fused,
        )

    # Contiguity does not fix singleton-axis strides. Such a layout is a
    # different pool identity: the constructed wrapper rejects it at run, and
    # only an exemplar (not num_pages/device) can describe it for a new wrapper.
    strides = list(fused.stride())
    strides[2] = 0
    variant = fused.as_strided(fused.shape, strides)
    assert variant.is_contiguous() and variant.stride() != fused.stride()
    with pytest.raises(ValueError, match="layout"):
        wrapper.run(q, weights, variant)
    paged = FP8PagedMQATopKWrapper(
        **specialization, **capacities, num_pages=fused.shape[0], device=fused.device
    )
    paged.prepare_pool(fused)
    with pytest.raises(ValueError, match="layout"):
        paged.prepare_pool(variant)
    wrapper = FP8PagedMQATopKWrapper(**specialization, **capacities, kv_layout=variant)
    wrapper.plan(qo, kv, table)
    selected = wrapper.run(q, weights, variant)
    torch.cuda.synchronize()
    oracle = []
    for row in range(2):
        slots = torch.arange(1024 // ratio, device=q.device)
        physical = table[row, slots // 128].long() * 128 + slots % 128
        dots = q[row].float() @ keys[physical].float().T
        oracle.append((dots.relu() * weights[row, :, None]).sum(0) * scales[physical])
    _check_selection(selected, oracle, 512)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_step_inputs_refresh_and_graph_replay(ratio):
    from flashinfer.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    capacity = _SelectiveCapacity(8, 128, 32, ratio, 65536, 8, 4)
    fused = torch.empty((8, 32, 1, 132), dtype=torch.uint8, device="cuda")
    columns = 2048 // ratio
    table = (torch.arange(4 * columns, dtype=torch.int32, device="cuda") % 8).view(
        4, columns
    )
    state = _CapacityStepInputs(capacity, device=fused.device, num_pages=fused.shape[0])
    state.prepare_sampling(512)
    q = torch.zeros((8, 8, 128), dtype=torch.float8_e4m3fn, device="cuda")
    weights = torch.ones((8, 8), dtype=torch.float32, device="cuda")

    def step(q_prefix, kv_prefix):
        # New prefix objects must never replace the graph's input addresses.
        qo = torch.tensor(q_prefix, dtype=torch.int32, device="cuda")
        kv = torch.tensor(kv_prefix, dtype=torch.int32, device="cuda")
        state.stage_payload(q, weights)
        state.stage_step(qo, kv, table, num_query_rows=q.shape[0])
        graph.replay()
        torch.cuda.synchronize()

    def valid(q_prefix, kv_prefix):
        assert state.error.item() == 0
        lengths = [b - a for a, b in zip(kv_prefix[:-1], kv_prefix[1:], strict=True)]
        physical = [0]
        for length in lengths:
            physical.append(physical[-1] + length // ratio)
        expected_q = q_prefix + [q_prefix[-1]] * (5 - len(q_prefix))
        expected_kv = physical + [physical[-1]] * (5 - len(physical))
        assert state.safe_qo_indptr.tolist() == expected_q
        assert state.physical_kv_indptr.tolist() == expected_kv
        assert state.active_shape.tolist() == [q_prefix[-1], len(lengths)]
        ends, owners = [], []
        for request, (qb, qe, length) in enumerate(
            zip(q_prefix[:-1], q_prefix[1:], lengths, strict=True)
        ):
            ends.extend(
                (length - (qe - qb) + row + 1) // ratio for row in range(qe - qb)
            )
            owners.extend([request] * (qe - qb))
        assert state.row_ends.tolist() == ends + [0] * (8 - len(ends))
        assert state.row_requests.tolist() == owners + [-1] * (8 - len(owners))
        expected_counts, expected_ranks = [], []
        pages_per_request = []
        for length in lengths:
            physical_length = length // ratio
            pages = math.ceil(physical_length / 32)
            chosen = min(pages, 3840 // 32)
            indices = [i * (pages - 1) // max(chosen - 1, 1) for i in range(chosen)]
            pages_per_request.append(indices)
        for row_end, owner in zip(ends, owners, strict=True):
            coordinates = [
                page * 32 + offset
                for page in pages_per_request[owner]
                for offset in range(min(32, lengths[owner] // ratio - page * 32))
            ]
            count = sum(index < row_end for index in coordinates)
            expected_counts.append(count)
            expected_ranks.append(
                0
                if row_end == 0
                else max(1, min(count, math.ceil(2048 * count / row_end)))
            )
        assert state.sample_ends.tolist() == expected_counts + [0] * (8 - len(ends))
        assert state.sample_ranks.tolist() == expected_ranks + [0] * (8 - len(ends))
        padded_pages = [pages + [0] * (120 - len(pages)) for pages in pages_per_request]
        padded_pages.extend([[0] * 120 for _ in range(4 - len(padded_pages))])
        assert state.sample_page_indices.tolist() == padded_pages

    state.stage_payload(q, weights)
    state.stage_step(
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        table,
        num_query_rows=q.shape[0],
    )
    state.refresh()
    state.refresh_sampling()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.refresh()
        state.refresh_sampling()
    for qo, kv in (
        ([0, 2, 2, 5], [0, 9, 9, 26]),
        ([0, 2, 2, 5], [0, 9, 14, 31]),
        ([0, 2, 2, 5], [0, 3, 3, 10]),
        ([0, 2, 2, 5], [0, 32767, 32768, 65535]),
        ([0, 1], [0, 64]),
        ([0], [0]),
    ):
        step(qo, kv)
        valid(qo, kv)

    table.fill_(-1)
    step([0, 3, 1, 5], [0, 9, 9, 26])
    assert state.error.item() == 2  # Invalid prefixes suppress page-map reads.
    step([0, 2, 2, 5], [0, 9, 9, 26])
    assert state.error.item() == 8
    assert state.safe_qo_indptr.count_nonzero().item() == 0
    assert state.physical_kv_indptr.count_nonzero().item() == 0
    assert state.active_shape.count_nonzero().item() == 0
    assert state.row_ends.count_nonzero().item() == 0
    assert (state.row_requests == -1).all()
    table.copy_(
        (torch.arange(table.numel(), dtype=torch.int32, device="cuda") % 8).view_as(
            table
        )
    )
    step([0, 1], [0, 65537])
    assert state.error.item() == 4
    step([0, 2, 2, 5], [0, 3, 3, 10])
    valid([0, 2, 2, 5], [0, 3, 3, 10])

    # A host shape failure must precede even the first staging copy.
    old_q = state.q.clone()
    with pytest.raises(ValueError, match="fit the planned"):
        state.stage_payload(
            torch.empty((9, 8, 128), dtype=q.dtype, device=q.device),
            torch.empty((9, 8), dtype=weights.dtype, device=weights.device),
        )
    assert torch.equal(state.q.view(torch.uint8), old_q.view(torch.uint8))


@_requires_selective_sm100
@pytest.mark.parametrize(
    "topology,heads,rows,length,top_k",
    [
        ("high_split", 8, 1, 131072, 512),
        ("owner_pool", 8, 1, 1024, 512),
        ("segmented", 8, 2, 32768, 512),
        ("h64_standard", 64, 9, 9001, 2048),
    ],
)
def test_capacity_chain_topologies_replay(
    topology, heads, rows, length, top_k, monkeypatch
):
    if topology == "h64_standard":
        # Fill one unsplit Q4 wave with the actual device geometry. A tiny
        # request chooses split-KV instead of the H64 candidate variant.
        rows = 4 * torch.cuda.get_device_properties("cuda").multi_processor_count
    inputs = _request_inputs([rows], [length], heads)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    if topology == "h64_standard":
        q.zero_()  # Overflow publication and exercise standard H64 recovery.
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        max_context_length=length,
        max_chunk_size=rows,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    schedule = engine._candidate_schedule
    if topology == "high_split":
        assert schedule.split_kv > 8
    elif topology == "owner_pool":
        assert schedule.split_kv <= 8
        assert schedule.count_segments == schedule.split_kv
    elif topology == "segmented":
        assert schedule.count_segments > 1
    else:
        assert not engine._compact_repair
    wrapper.plan(qo, kv, table, num_query_rows=rows)
    _check_selection(wrapper.run(q, weights, fused), _oracle_rows(inputs), top_k)
    if topology == "h64_standard":
        assert engine._repair_flags.count_nonzero().item() > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=rows)
        selected = wrapper.run(q, weights, fused)
    for tied in (False, True):
        q.copy_(torch.randn(q.shape, device=q.device).mul_(0.1).to(q.dtype))
        if tied:
            q.zero_()
        weights.mul_(-0.75)
        fused.copy_(fused.flip(0))
        table.copy_(fused.shape[0] - 1 - table)
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_capacity_page_checks_stride_past_grid(ratio):
    from flashinfer.attn_scores.selective_capacity import (
        _CapacityStepInputs,
        _SelectiveCapacity,
    )

    check_ctas = 2 * torch.cuda.get_device_properties("cuda").multi_processor_count
    batches = check_ctas // 2 + 1
    columns, page_size = 2049, 32
    physical_length = columns * page_size
    capacity = _SelectiveCapacity(
        8, 128, page_size, ratio, physical_length * ratio, 2, batches
    )
    state = _CapacityStepInputs(capacity, device="cuda", num_pages=2)
    qo = torch.zeros(batches + 1, dtype=torch.int32, device="cuda")
    kv = torch.zeros_like(qo)
    qo[-1] = 1
    kv[-1] = physical_length * ratio
    table = torch.full((batches, columns), -1, dtype=torch.int32, device="cuda")
    table[-1].zero_()  # Invalid padding in empty requests must never be read.
    state.stage_step(qo, kv, table, num_query_rows=2)
    state.refresh()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.refresh()
    graph.replay()
    torch.cuda.synchronize()
    assert state.error.item() == 0
    assert state.active_shape.tolist() == [1, batches]
    assert state.row_ends.tolist() == [physical_length, 0]
    assert state.row_requests.tolist() == [batches - 1, -1]

    # Unit check_ctas+1 is CTA1's second iteration, in the second page chunk.
    # CTA0 and a single-pass check both miss this live invalid page.
    state.block_table[-1, 2048] = 2
    graph.replay()
    torch.cuda.synchronize()
    assert state.page_flags[0].item() == 0
    assert state.page_flags[1].item() == 8
    assert state.page_flags.count_nonzero().item() == 1
    assert state.error.item() == 8
    assert state.safe_qo_indptr.count_nonzero().item() == 0
    assert state.physical_kv_indptr.count_nonzero().item() == 0
    assert state.active_shape.count_nonzero().item() == 0
    assert state.row_ends.count_nonzero().item() == 0
    assert (state.row_requests == -1).all()
    state.block_table[-1, 2048] = 0
    graph.replay()
    torch.cuda.synchronize()
    assert state.error.item() == 0
    assert state.page_flags.count_nonzero().item() == 0
    assert state.row_requests.tolist() == [batches - 1, -1]


@_requires_selective_sm100
def test_capacity_schedule_large_batch_owner_walk():
    batches = 4097
    kv_length = 1024
    owners = [0, 2048, 4096]
    q_lengths = [int(request in owners) for request in range(batches)]
    kv_lengths = [kv_length * length for length in q_lengths]
    inputs = _request_inputs(q_lengths, kv_lengths)
    q, _, _, weights, qo, kv = inputs
    fused, table = _paged_inputs(inputs)
    pages = table[owners].clone()
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        max_context_length=kv_length,
        max_chunk_size=3,
        max_batch_size=batches,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    wrapper.plan(qo, kv, table, num_query_rows=3)
    _check_selection(wrapper.run(q, weights, fused), _oracle_rows(inputs), 512)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.plan(qo, kv, table, num_query_rows=3)
        selected = wrapper.run(q, weights, fused)

    # More than 4096 requests select the walk fallback; empty requests change
    # the owners without changing query storage, pool layout or graph bindings.
    for owners in ([1, 2047, 4095], [0, 2048, 4096]):
        q_prefix, kv_prefix = [0], [0]
        for request in range(batches):
            length = int(request in owners)
            q_prefix.append(q_prefix[-1] + length)
            kv_prefix.append(kv_prefix[-1] + kv_length * length)
        qo.copy_(torch.tensor(q_prefix, dtype=qo.dtype, device=qo.device))
        kv.copy_(torch.tensor(kv_prefix, dtype=kv.dtype, device=kv.device))
        table.zero_()
        table[owners] = pages
        graph.replay()
        torch.cuda.synchronize()
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        assert wrapper._capacity_plan._state.row_requests.tolist() == owners
        _check_selection(selected, _oracle_rows(inputs), 512)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_capture_restores_gc_state(monkeypatch, enabled, fail):
    events = []

    class Stream:
        def __init__(self):
            self.waits = []

        def wait_stream(self, other):
            self.waits.append(other)

    capture_stream, caller_stream = Stream(), Stream()
    active_stream = caller_stream

    def make_stream(*, device):
        assert device == 7
        return capture_stream

    def reject_collection_or_flush(*args, **kwargs):
        raise AssertionError("capture must not collect or empty allocator caches")

    @contextmanager
    def stream_context(stream):
        nonlocal active_stream
        assert stream is capture_stream
        active_stream = stream
        try:
            yield
        finally:
            active_stream = caller_stream

    class Graph:
        def capture_begin(self):
            assert active_stream is capture_stream
            assert not gc.isenabled()
            events.append("begin")

        def capture_end(self):
            assert active_stream is capture_stream
            assert not gc.isenabled()
            events.append("end")

    monkeypatch.setattr(gc, "collect", reject_collection_or_flush)
    monkeypatch.setattr(torch.cuda, "graph", reject_collection_or_flush)
    monkeypatch.setattr(torch.cuda, "empty_cache", reject_collection_or_flush)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: active_stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    original = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        if fail:
            with (
                pytest.raises(RuntimeError, match="injected capture failure"),
                _capture_cuda_graph(Graph()),
            ):
                raise RuntimeError("injected capture failure")
        else:
            with _capture_cuda_graph(Graph()):
                assert not gc.isenabled()
        assert gc.isenabled() == enabled
        assert active_stream is caller_stream
        assert events == ["begin", "end"]
        assert capture_stream.waits == [caller_stream]
        assert caller_stream.waits == ([] if fail else [capture_stream])
    finally:
        (gc.enable if original else gc.disable)()


def _case():
    q, kv, scales, weights, _, _ = _inputs(16, 8192)
    cu_q = torch.tensor([0, 16], dtype=torch.int32, device=q.device)
    cu_kv = torch.tensor([0, 8192], dtype=torch.int32, device=q.device)
    inputs = q, kv, scales, weights, cu_q, cu_kv
    fused, table = _paged_inputs(inputs)
    return inputs, fused, table


@_requires_selective_sm100
def test_owned_body_stream_survives_pool_wrap_replan_and_multiple_wrappers():
    inputs, fused, table = _case()
    q, _, _, weights, cu_q, cu_kv = inputs
    pooled = [torch.cuda.Stream(device=q.device) for _ in range(96)]
    wrappers = [SelectiveEngineCase() for _ in range(2)]
    handles = []
    for length in (1024, 8192, 1024):
        cu_kv[-1] = length
        # Bounded ties fit the candidate buffer; long ties overflow it.
        weights.zero_()
        for wrapper_index, wrapper in enumerate(wrappers):
            wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_sms=4)
            body = wrapper._repair_body_stream
            assert body.cuda_stream not in {stream.cuda_stream for stream in pooled}
            if length == 1024 and len(handles) < 2:
                handles.append(body.cuda_stream)
            assert body.cuda_stream == handles[wrapper_index]
            _check_selection(
                wrapper.run(q, fused, weights, table), _oracle_rows(inputs), 512
            )
            assert bool(wrapper._repair_flags.any()) == (length == 8192)
            for parent in (pooled[0], pooled[-1]):
                graph = torch.cuda.CUDAGraph()
                copied = torch.empty_like(wrapper._selected)
                with torch.cuda.graph(graph, stream=parent):
                    copied.copy_(wrapper.run(q, fused, weights, table))
                graph.replay()
                _check_selection(copied, _oracle_rows(inputs), 512)
                with torch.cuda.stream(body):
                    assert not torch.cuda.is_current_stream_capturing()
    assert len(set(handles)) == 2


@_requires_selective_sm100
@pytest.mark.parametrize("failure_site", ["begin", "body", "end"])
def test_body_capture_failure_preserves_exception_and_recovers(
    failure_site, monkeypatch
):
    inputs, fused, table = _case()
    q, _, _, weights, cu_q, cu_kv = inputs
    wrapper = SelectiveEngineCase()
    wrapper.plan(q, fused, weights, cu_q, cu_kv, table, num_sms=4)
    module = wrapper._repair_if_module
    body = wrapper._repair_body_stream
    original_repair = wrapper._run_repair
    finish_calls = []

    def finish(*args):
        finish_calls.append(True)
        return module.finish_if(*args)

    def fail_body(*args):
        wrapper._exact_error.zero_()
        raise RuntimeError("injected repair body failure")

    def fail_begin(*args):
        module.begin_body(*args)
        raise RuntimeError("injected repair begin failure")

    def fail_end(*args):
        module.end_body(*args)
        raise RuntimeError("injected repair end failure")

    proxy = SimpleNamespace(
        begin_if=module.begin_if,
        begin_body=fail_begin if failure_site == "begin" else module.begin_body,
        end_body=fail_end if failure_site == "end" else module.end_body,
        finish_if=finish,
        preflight_body=module.preflight_body,
        body_capture_status=module.body_capture_status,
        abort_body=module.abort_body,
    )
    monkeypatch.setattr(wrapper, "_repair_if_module", proxy)
    if failure_site == "body":
        monkeypatch.setattr(wrapper, "_run_repair", fail_body)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match=f"injected repair {failure_site} failure"),
        torch.cuda.graph(graph),
    ):
        wrapper.run(q, fused, weights, table)
    # Only the outer context ends its caller-owned capture; the body is idle.
    assert not torch.cuda.is_current_stream_capturing()
    assert module.body_capture_status(q, body.cuda_stream) == 0
    assert not finish_calls
    monkeypatch.setattr(wrapper, "_repair_if_module", module)
    monkeypatch.setattr(wrapper, "_run_repair", original_repair)
    _check_selection(wrapper.run(q, fused, weights, table), _oracle_rows(inputs), 512)
    recovered = torch.cuda.CUDAGraph()
    with torch.cuda.graph(recovered):
        output = wrapper.run(q, fused, weights, table)
    recovered.replay()
    _check_selection(output, _oracle_rows(inputs), 512)


# Selective row tails and storage replanning


def _force_h64_candidate_layout(monkeypatch, variant: bool) -> None:
    """Pin single-pool H64 publication to the variant or the pooled layout."""
    import flashinfer.attn_scores.selective_logits as selective_logits

    monkeypatch.setattr(
        selective_logits,
        "_h64_candidate_variant_serves",
        lambda page_size: variant and page_size == 128,
    )


def _check_selection(selected, expected_rows, top_k):
    for row, scores in enumerate(expected_rows):
        count = min(top_k, scores.numel())
        indices = selected[row, :count].long()
        assert ((indices >= 0) & (indices < scores.numel())).all()
        assert indices.unique().numel() == count
        torch.testing.assert_close(
            scores[indices].sort().values,
            scores.topk(count).values.sort().values,
            atol=5e-5,
            rtol=1e-5,
        )
        assert (selected[row, count:] == -1).all()


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("split_kv", [1, 4])
def test_q1_owner_pool_raw_counts_and_capacity_boundaries(
    heads, split_kv, record_property
):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    capacity = 6144
    top_k = 512 if heads == 8 else 2048
    for kv_rows in (31, capacity, capacity + 1, 8193):
        inputs = _inputs(q_rows=1, kv_rows=kv_rows, num_heads=heads)
        prepared = _prepare(
            inputs,
            num_sms=8,
            num_heads=heads,
            query_tile=1,
            num_umma_stages=2,
            split_kv=split_kv,
        )
        values = torch.empty((16, capacity), device="cuda", dtype=torch.int64)
        counts = torch.empty((16, split_kv + 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty(1, device="cuda", dtype=torch.int32)
        flags = torch.empty_like(lengths)
        threshold = torch.full((1,), float("-inf"), device="cuda")
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        raw = counts[0, 1:].clone()
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=capacity,
            top_k=top_k,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert int(raw.sum()) == kv_rows
        overflow = bool((raw > capacity // split_kv).any())
        assert int(counts[0, 0]) == (capacity + 1 if overflow else kv_rows)
        assert bool(flags[0]) == (overflow or kv_rows < top_k)
        record_property(
            f"raw_counts_{kv_rows}",
            dict(
                logical_row=0,
                owners=raw.cpu().tolist(),
                published_count=int(counts[0, 0]),
                overflow=overflow,
            ),
        )
        if kv_rows > capacity + 1:
            assert int(counts[0, 0]) != int(raw.sum())
        if not overflow:
            ids = (values[0, :kv_rows] & 0xFFFFFFFF).sort().values
            torch.testing.assert_close(
                ids, torch.arange(kv_rows, device="cuda", dtype=torch.int64)
            )
        # Raw owner counts are copied before the overflow sentinel is used.
        # Reusing the same normal launch must also reset every owner cursor.
        threshold.fill_(float("inf"))
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            capacity,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=capacity,
            top_k=top_k,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert not bool(counts[0].any()) and int(lengths[0]) == 0
        assert int(flags[0]) == 1


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_q1_owner_pool_removes_only_local_vector_overflow(heads, record_property):
    from flashinfer.attn_scores.selective_logits import (
        _finalize_fp8_mqa_packed_candidates,
    )

    inputs = _inputs(q_rows=1, kv_rows=8192, num_heads=heads)
    q, kv, scales, weights, _, _ = inputs
    q.fill_(1)
    scales.fill_(1)
    weights.fill_(1)
    signs = torch.where(torch.arange(8192, device="cuda") % 4 == 0, 1.0, -1.0)
    kv.copy_(signs[:, None].expand(-1, 128).to(kv.dtype))
    prepared = _prepare(
        inputs, num_sms=4, num_heads=heads, query_tile=1, num_umma_stages=2, split_kv=1
    )
    threshold = torch.full((1,), 0.5, device="cuda")
    for segments in (4, 1):
        values = torch.empty((16, 6144), device="cuda", dtype=torch.int64)
        counts = torch.empty((16, segments + 1), device="cuda", dtype=torch.int32)
        lengths = torch.empty(1, device="cuda", dtype=torch.int32)
        flags = torch.empty_like(lengths)
        _launch(
            _SelectiveLogitsMode.CANDIDATE,
            6144,
            values,
            counts,
            prepared,
            inputs,
            threshold,
        )
        raw = counts[0, 1:].clone()
        _finalize_fp8_mqa_packed_candidates(
            counts,
            values,
            lengths,
            flags,
            rows=1,
            capacity=6144,
            top_k=2048,
            _candidate_prepared=prepared,
        )
        torch.cuda.synchronize()
        assert int(raw.sum()) == 2048
        assert int(flags[0]) == (1 if segments == 4 else 0)
        assert int(counts[0, 0]) == (6145 if segments == 4 else 2048)
        record_property(
            f"segments_{segments}",
            dict(
                logical_row=0,
                raw_counts=raw.cpu().tolist(),
                published_count=int(counts[0, 0]),
            ),
        )
        if segments == 1:
            ids = (values[0, :2048] & 0xFFFFFFFF).sort().values
            torch.testing.assert_close(
                ids, torch.arange(0, 8192, 4, device="cuda", dtype=torch.int64)
            )


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "q_len", "kv_len"),
    [
        (8, 4, 257),
        (8, 16, 257),
        (8, 16, 8193),
        (64, 8, 513),
        (64, 8, 514),
        (64, 8, 8193),
    ],
)
@pytest.mark.parametrize("top_k", [64, 512])
def test_full_query_tile_causal_end_inside_penultimate_kv_pair(
    heads, q_len, kv_len, top_k
):
    """Full Q tiles must bound every KV pair that crosses a row's causal end.

    With an odd tile KV-block count and a context just past an even block
    boundary, early rows end inside the second-to-last 256-token KV pair.
    """
    inputs = _request_inputs([q_len], [kv_len], heads=heads)
    q, _, _, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        q,
        fused,
        weights,
        cu_q,
        cu_kv,
        table,
        top_k=top_k,
        num_heads=heads,
        head_dim=128,
    )
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)


@_requires_selective_sm100
@pytest.mark.parametrize(
    ("heads", "q_lengths", "kv_lengths"),
    [
        (8, [1, 1, 1], [600, 9000, 20000]),
        (8, [37], [20037]),
        (64, [9, 1], [12009, 3001]),
    ],
)
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("top_k", [513, 1024, 2047])
def test_selective_top_k_between_512_and_2048(
    heads, q_lengths, kv_lengths, ratio, top_k
):
    """Every TopK width up to 2048 returns exact request-local selections."""
    if ratio == 1:
        inputs = _request_inputs(q_lengths, kv_lengths, heads=heads)
        q, _, _, weights, cu_q, cu_kv = inputs
        fused, table = _paged_inputs(inputs)
        payload = (q, fused, weights, cu_q, cu_kv, table)
    else:
        payload = _compressed_wrapper_fixture(
            q_lengths, [4 * length for length in kv_lengths], heads, 128
        )
        q, fused, weights, _, _, table = payload
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=top_k,
        compression_ratio=ratio,
        max_context_length=max(kv_lengths) * ratio,
        max_chunk_size=q.shape[0],
        max_batch_size=len(q_lengths),
        kv_layout=fused,
    )
    wrapper.plan(payload[3], payload[4], table)
    selected = wrapper.run(q, weights, fused)
    torch.cuda.synchronize()
    assert selected.shape == (q.shape[0], top_k)
    assert wrapper.error.item() == wrapper.repair_error.item() == 0
    if ratio == 1:
        _check_selection(selected, _oracle_rows(inputs), top_k)
    else:
        _check_compressed_wrapper(selected, payload, top_k)


@_requires_selective_sm100
@pytest.mark.parametrize("ratio", [1, 4])
def test_policy_row_tails_forced_repair(ratio):
    """Q9/H64 leaves a padded Q2 lane in repair, radix and exact emission."""
    rows, length, top_k = 9, 9001, 512
    if ratio == 4:
        payload = _compressed_wrapper_fixture([rows], [length * 4 + 1], 64, 128)
        expected = None
    else:
        inputs = _inputs(rows, length, 64)
        q, kv, scales, weights, cu_q, cu_kv = inputs
        cu_q = torch.tensor([0, rows], device=q.device, dtype=torch.int32)
        cu_kv = torch.tensor([0, length], device=q.device, dtype=torch.int32)
        fused, table = _paged_inputs((q, kv, scales, weights, cu_q, cu_kv))
        payload = q, fused, weights, cu_q, cu_kv, table
    q, fused, weights, cu_q, cu_kv, table = payload
    # Ties overflow both bounded arenas, forcing all terminal radix passes and
    # EXACT rather than merely compiling the affected helpers.
    weights.zero_()
    if ratio == 1:
        expected = _oracle_rows((q, kv, scales, weights, cu_q, cu_kv))
    wrapper = SelectiveEngineCase()
    wrapper.plan(
        *payload, top_k=top_k, num_heads=64, num_sms=4, compression_ratio=ratio
    )
    assert wrapper._repair_prepared.query_tile == wrapper._prepared.query_tile == 2
    for policy in (
        wrapper._repair_thresholds,
        wrapper._prefixes,
        wrapper._threshold_bits,
    ):
        # Logical-size buffers cannot mask an out-of-bounds padded-row read.
        assert policy.numel() == rows

    def check(output):
        torch.cuda.synchronize()
        if ratio == 4:
            _check_compressed_wrapper(output, payload, top_k)
        else:
            _check_selection(output, expected, top_k)
        assert (wrapper._repair_flags != 0).all()
        assert (wrapper._second_repair_flags != 0).all()
        assert wrapper._exact_error.item() == 0

    check(wrapper._run_impl(q, fused, weights, table))
    check(wrapper.run(q, fused, weights, table))
    graph = torch.cuda.CUDAGraph()
    with _capture_cuda_graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    for _ in range(2):
        graph.replay()
        check(captured)


# H64 query-tile scheduling and graph replay


@_requires_selective_sm100
@pytest.mark.parametrize(
    "q_rows,top_k,ties,num_sms",
    [
        # Fallback/query4 boundaries, publication layouts and persistent tails.
        (2, 512, False, 2),
        (2, 2048, True, 4),
        (4, 2048, False, 4),
        (4, 512, True, 2),
        (5, 512, False, 2),
        (5, 2048, True, 2),
        (5, 512, True, 4),
        (5, 2048, False, 4),
        (7, 512, False, 4),
        (7, 2048, True, 2),
        (33, 512, True, 2),
        (33, 2048, False, 4),
    ],
)
def test_h64_query4_eager_graph_and_tails(q_rows, top_k, ties, num_sms):
    inputs = list(_inputs(q_rows=q_rows, kv_rows=16384, num_heads=64))
    q, kv, scale, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, q_rows], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, kv.shape[0]], device="cuda", dtype=torch.int32)
    if ties:
        q.zero_()
    fused, table = _paged_inputs(tuple(inputs))
    wrapper = SelectiveEngineCase(math_mode="relu_strict")
    wrapper.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=top_k,
        num_heads=64,
        head_dim=128,
        num_sms=num_sms,
    )
    candidate = wrapper._candidate_prepared
    assert candidate.query_tile == (4 if q_rows > 4 else 2)
    assert candidate.num_umma_stages == (1 if q_rows > 4 else 2)
    if q_rows == 5:
        assert candidate.split_kv == (1 if num_sms == 2 else 2)
        # Split-1 H64 prefill uses the H64 candidate variant's 4 segments.
        assert wrapper._candidate_schedule.h64_candidate == (num_sms == 2)
        assert wrapper._candidate_schedule.publication == "warp_striped"
        assert wrapper._candidate_schedule.count_segments == (4 if num_sms == 2 else 8)
        assert wrapper._candidate_schedule.sample_count_segments == (
            32 if num_sms == 2 else 64
        )
    for prepared in (
        wrapper._sample_prepared,
        wrapper._repair_prepared,
        wrapper._prepared,
    ):
        assert prepared.query_tile == 2
        assert prepared.num_umma_stages == 2
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    if ties:
        assert (wrapper._repair_flags != 0).all()
    assert int(wrapper._exact_error.item()) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert selected.data_ptr() == captured.data_ptr()
    # Change physical placement and logical Q/scale/weight values in place.
    fused.copy_(fused.flip(0))
    table.copy_(fused.shape[0] - 1 - table)
    q.copy_(q.float().neg().to(q.dtype))
    weights[:, ::2].mul_(-1)
    scale[::2].mul_(2)
    fused.flatten(1)[:, 128 * 128 :].view(torch.float32)[:, ::2].mul_(2)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert int(wrapper._exact_error.item()) == 0


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths"),
    [([1, 36], [132, 16252]), ([40, 7, 64], [5000, 300, 9000])],
)
def test_h64_candidate_variant_matches_pooled_layout(
    q_lengths, kv_lengths, top_k, monkeypatch
):
    """The H64 candidate variant publishes the pooled layout's candidates.

    Covers ragged multi-request prefill, partial query tiles and short rows.
    """
    inputs = _request_inputs(q_lengths, kv_lengths, heads=64)
    q, _kv, _scale, weights, cu_q, cu_kv = inputs
    fused, table = _paged_inputs(inputs)
    oracle = _oracle_rows(inputs)
    rows = sum(q_lengths)
    selected, lengths = {}, {}
    for h64_candidate in (False, True):
        _force_h64_candidate_layout(monkeypatch, h64_candidate)
        engine = SelectiveEngineCase(math_mode="relu_strict")
        engine.plan(
            q,
            fused,
            weights,
            cu_q,
            cu_kv,
            table,
            top_k=top_k,
            num_heads=64,
            head_dim=128,
            num_sms=2,
        )
        assert engine._candidate_schedule.h64_candidate is h64_candidate
        selected[h64_candidate] = engine.run(q, fused, weights, table).clone()
        torch.cuda.synchronize()
        lengths[h64_candidate] = engine._candidate_lengths[:rows].clone()
        assert int(engine._exact_error.item()) == 0
        _check_selection(selected[h64_candidate], oracle, top_k)
    # Identical scores and thresholds publish identical candidate populations.
    torch.testing.assert_close(lengths[True], lengths[False], atol=0, rtol=0)
    torch.testing.assert_close(
        selected[True].sort(1).values, selected[False].sort(1).values, atol=0, rtol=0
    )


@_requires_selective_sm100
@pytest.mark.parametrize("top_k", [512, 2048])
def test_query4_mixed_empty_short_and_tail_request(top_k):
    inputs = list(_inputs(q_rows=6, kv_rows=16384, num_heads=64))
    q, _, _, weights, _, _ = inputs
    inputs[4] = torch.tensor([0, 0, 1, 6], device="cuda", dtype=torch.int32)
    inputs[5] = torch.tensor([0, 0, 64, 16384], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs(tuple(inputs))
    table[0].fill_(fused.shape[0] + 17)
    wrapper = SelectiveEngineCase(math_mode="relu_strict")
    wrapper.plan(
        q,
        fused,
        weights,
        inputs[4],
        inputs[5],
        table,
        top_k=top_k,
        num_heads=64,
        head_dim=128,
        num_sms=4,
    )
    assert wrapper._candidate_prepared.query_tile == 4
    selected = wrapper.run(q, fused, weights, table)
    torch.cuda.synchronize()
    _check_selection(selected, _oracle_rows(inputs), top_k)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(q, fused, weights, table)
    graph.replay()
    torch.cuda.synchronize()
    _check_selection(captured, _oracle_rows(inputs), top_k)
    assert int(wrapper._exact_error.item()) == 0


# K2048 population targets and rank refresh


def _check_rank_bounds(wrapper, top_k):
    valid = wrapper._sample_ends.cpu().tolist()
    ends = wrapper._row_ends.cpu().tolist()
    # Independent expected population targets for K512 and K2048.
    target = {512: 2048, 2048: 4096}[top_k]
    expected = [
        max(1, min(count, math.ceil(target * count / end)))
        for count, end in zip(valid, ends, strict=True)
    ]
    assert wrapper._sample_ranks.cpu().tolist() == expected
    assert max(expected) <= wrapper._sample_rank_capacity
    assert max(expected) <= sl._SAMPLE_CAPACITY


@_requires_selective_sm100
@pytest.mark.parametrize("heads", [8, 64])
def test_k2048_changed_lengths_refresh_ranks_and_repair(heads):
    inputs = list(_inputs(q_rows=5, kv_rows=16384, num_heads=heads))
    q, _, _, weights, cu_q, _ = inputs
    cu_q = torch.tensor([0, 5], device="cuda", dtype=torch.int32)
    inputs[4] = cu_q
    inputs[5] = torch.tensor([0, 16384], device="cuda", dtype=torch.int32)
    fused, table = _paged_inputs(tuple(inputs))
    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective",
        num_heads=heads,
        top_k=2048,
        max_context_length=16384,
        max_chunk_size=5,
        max_batch_size=1,
        kv_layout=fused,
    )
    wrapper.prepare_pool(fused)
    engine = wrapper._capacity_plan._engine
    rank_pointer = None
    prior_ranks = None
    retained = None
    for generation, length in enumerate((12288, 12289, 16384, 12288)):
        inputs[5] = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        if generation == 1:
            weights[0].zero_()  # One row needs the exact overflow path.
        wrapper.plan(cu_q, inputs[5], table)
        selected = wrapper.run(q, weights, fused)
        _check_rank_bounds(engine, 2048)
        ranks = engine._sample_ranks.cpu().tolist()
        if rank_pointer is None:
            rank_pointer = engine._sample_ranks.data_ptr()
            retained = (
                engine._sample_scores,
                engine._candidate_values,
                engine._selected,
            )
        else:
            assert engine._sample_ranks.data_ptr() == rank_pointer
            assert ranks != prior_ranks
            assert all(
                current is original
                for current, original in zip(
                    (
                        engine._sample_scores,
                        engine._candidate_values,
                        engine._selected,
                    ),
                    retained,
                    strict=True,
                )
            )
        prior_ranks = ranks
        torch.cuda.synchronize()
        _check_selection(selected, _oracle_rows(inputs), 2048)
        assert wrapper.error.item() == wrapper.repair_error.item() == 0
        if generation:
            assert int(engine._repair_flags[0]) != 0
