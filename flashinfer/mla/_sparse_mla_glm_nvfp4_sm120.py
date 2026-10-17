# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""GLM sparse MLA with NVFP4 KV storage and the existing SM120 FP8 MMA kernels.

Each contiguous cache row holds 256 packed E2M1 bytes, 32 E4M3 block-16
scales, and 128 BF16 RoPE bytes. ``kv_global_scale`` is a positive finite
device FP32 scalar; a latent value is ``E2M1 * E4M3_scale * global_scale``.
The consumer expands selected rows in shared memory to the GLM FP8 tile.
No persistent FP8 cache or global conversion workspace is materialized.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from ..utils import register_custom_op, register_fake_op
from ..jit.mla import gen_sparse_mla_glm_nvfp4_sm120_module


@functools.cache
def get_sparse_mla_glm_nvfp4_sm120_module():
    module = gen_sparse_mla_glm_nvfp4_sm120_module().build_and_load()

    @register_custom_op(
        "flashinfer::sparse_mla_glm_nvfp4_sm120_paged_attention",
        mutates_args=("output", "out_lse", "mid_out", "mid_lse"),
    )
    def paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        kv_global_scale: torch.Tensor,
        sm_scale: float,
        chunks_per_block: int,
        topk_length: Optional[torch.Tensor],
        attn_sink: Optional[torch.Tensor],
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
    ) -> None:
        module.sparse_mla_glm_nvfp4(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            kv_global_scale,
            sm_scale,
            chunks_per_block,
            topk_length,
            attn_sink,
            mid_out,
            mid_lse,
        )

    @register_fake_op("flashinfer::sparse_mla_glm_nvfp4_sm120_paged_attention")
    def fake(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(paged_attention=paged_attention)


def _cache_page_size(kv_cache: torch.Tensor) -> int:
    if kv_cache.dtype != torch.uint8 or not kv_cache.is_contiguous():
        raise ValueError("GLM NVFP4 cache must be contiguous uint8 packed rows")
    if kv_cache.ndim == 2:
        if kv_cache.shape[1] % 416:
            raise ValueError("packed page byte width must be divisible by 416")
        page_size = kv_cache.shape[1] // 416
    elif kv_cache.ndim == 3 and kv_cache.shape[-1] == 416:
        page_size = kv_cache.shape[1]
    elif kv_cache.ndim == 4 and kv_cache.shape[-1] == 416:
        if kv_cache.shape[1] == 1:
            page_size = kv_cache.shape[2]
        elif kv_cache.shape[2] == 1:
            page_size = kv_cache.shape[1]
        else:
            raise ValueError("GLM NVFP4 cache requires one KV head")
    else:
        raise ValueError("GLM NVFP4 cache must contain 416-byte rows in paged layout")
    if page_size != 64:
        raise ValueError("GLM NVFP4 SM120 kernels require page_size=64")
    return page_size


@functools.cache
def _heuristic_decode_cpb(
    num_tokens: int, num_heads: int, total_chunks: int, sm_count: int
) -> int:
    """Materialize the existing decode launcher's three-wave tail heuristic."""
    if sm_count < 1 or total_chunks < 1:
        raise ValueError("decode needs a positive SM count and candidate count")
    per_token_head = num_tokens * ((num_heads + 15) // 16)
    best_cpb, best_gap = 1, 2 * sm_count
    for cpb in range(1, total_chunks + 1):
        active = per_token_head * ((total_chunks + cpb - 1) // cpb)
        ceil_waves = (active + sm_count - 1) // sm_count
        if ceil_waves > 3:
            continue
        # An integer numerator preserves the C++ gap ordering and ties without
        # floating-point rounding: gap = ceil_waves - active / sm_count.
        gap = ceil_waves * sm_count - active
        if gap < best_gap or (gap == best_gap and cpb > best_cpb):
            best_cpb, best_gap = cpb, gap
    return best_cpb


@functools.cache
def _automatic_decode_cpb(
    num_tokens: int, num_heads: int, total_chunks: int, sm_count: int
) -> int:
    """Reuse the wave heuristic with NVFP4 residency and serial-work caps."""
    if sm_count < 1 or total_chunks < 1:
        raise ValueError("decode needs a positive SM count and candidate count")
    head_groups = num_tokens * ((num_heads + 15) // 16)
    if num_heads == 8:
        # H8 BI32 uses two resident CTAs/SM.
        slots = 2 * sm_count
        cap = 4 if num_tokens <= 64 else 8
    else:
        slots = sm_count
        # Once query/head groups alone occupy more than half the SMs, shorter
        # serial loops retain useful split parallelism. These caps reflect
        # measured NVFP4 conversion and FP32 merge costs, not stock FP8 fits.
        cap = 5 if 2 * head_groups > sm_count else 7
    if head_groups > 3 * slots:
        # No candidate can meet the old three-wave envelope, even at one
        # split/query. Avoid its CPB1 fallback; cap the fewest-split choice.
        cpb = total_chunks
    else:
        cpb = _heuristic_decode_cpb(num_tokens, num_heads, total_chunks, slots)
    return min(cap, cpb)


@functools.cache
def _device_sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


class SparseMLAGlmNvfp4Sm120Wrapper:
    """SM120/SM121 GLM DSA: NVFP4 storage with FP8 latent QK/PV.

    Q is BF16 [T, H, 576], the output is BF16 [T, H, 512], and the
    cache contains packed 416-byte rows in 64-token pages. H may be
    8, 16, 32, 64, or 128. RoPE QK uses BF16 and accumulation uses FP32.
    The baseline two-pass P quantization is retained. Q/K/V each have one
    FP8 compute representation. FP32 split outputs avoid extra BF16 rounding.

    Warm every shape before graph capture. All split and LSE buffers remain
    owned by this wrapper for the lifetime of captured graphs. The optional
    route and chunks_per_block arguments support controlled tuning; a chunk
    covers 32 candidates for H=8 and 64 candidates otherwise.

    The caller must supply a positive finite device FP32 kv_global_scale
    and valid cache indices (negative values mask candidates). Scale values
    are not copied to the host during graph capture or replay.
    """

    def __init__(self, max_num_tokens=None, max_num_heads=None, *, device=None):
        self._device = torch.device(device or "cuda")
        if self._device.type != "cuda":
            raise ValueError("GLM NVFP4 sparse MLA requires a CUDA device")
        if self._device.index is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(self._device)[0] != 12:
            raise ValueError("GLM NVFP4 sparse MLA requires SM120/SM121")
        self._max_num_tokens = max_num_tokens
        self._max_num_heads = max_num_heads
        self._scratch_by_shape = {}
        self._lse_by_shape = {}
        self.last_plan = None
        self.last_decode_tile_size = None
        self.last_num_splits = None

    def _get_out_lse(self, t, h, out_lse):
        if out_lse is not None:
            if (
                out_lse.shape != (t, h)
                or out_lse.dtype != torch.float32
                or out_lse.device != self._device
                or out_lse.stride(-1) != 1
                or out_lse.stride(0) < h
            ):
                raise ValueError("out_lse must be FP32 [T, H] on the query device")
            return out_lse
        key = (t, h)
        if key not in self._lse_by_shape:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm GLM NVFP4 LSE shape before graph capture")
            self._lse_by_shape[key] = torch.empty(
                key, dtype=torch.float32, device=self._device
            )
        return self._lse_by_shape[key]

    def _scratch(
        self,
        q: torch.Tensor,
        topk: int,
        mid_out: Optional[torch.Tensor],
        mid_lse: Optional[torch.Tensor],
        *,
        chunks_per_block: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (mid_out is None) != (mid_lse is None):
            raise ValueError("mid_out and mid_lse must be provided together")
        tile_bi = 32 if q.shape[1] == 8 else 64
        total_chunks = (topk + tile_bi - 1) // tile_bi
        if not 1 <= chunks_per_block <= total_chunks:
            raise ValueError(
                "chunks_per_block must be within the candidate chunk count"
            )
        num_splits = (total_chunks + chunks_per_block - 1) // chunks_per_block
        shape = (q.shape[0], q.shape[1], num_splits)
        mid_dtype = torch.float32
        key = (*shape, mid_dtype)
        if mid_out is None:
            if key not in self._scratch_by_shape:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "warm this GLM NVFP4 decode shape before graph capture"
                    )
                self._scratch_by_shape[key] = (
                    torch.empty((*shape, 512), dtype=mid_dtype, device=q.device),
                    torch.empty(shape, dtype=torch.float32, device=q.device),
                )
            return self._scratch_by_shape[key]
        needed = shape[0] * shape[1] * shape[2]
        for name, tensor, dtype, count in (
            ("mid_out", mid_out, mid_dtype, needed * 512),
            ("mid_lse", mid_lse, torch.float32, needed),
        ):
            if (
                tensor.device != q.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.numel() < count
            ):
                raise ValueError(
                    f"{name} must be a contiguous {dtype} workspace with {count} elements"
                )
        # The C++ kernel uses packed scratch strides; flattening the reusable
        # allocation avoids noncontiguous views when a larger shape is supplied.
        return (
            mid_out.view(-1)[: needed * 512].view(*shape, 512),
            mid_lse.view(-1)[:needed].view(*shape),
        )

    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        sm_scale: float,
        *,
        kv_global_scale: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        out_lse: Optional[torch.Tensor] = None,
        mid_out: Optional[torch.Tensor] = None,
        mid_lse: Optional[torch.Tensor] = None,
        kernel_variant: Optional[str] = None,
        chunks_per_block: Optional[int] = None,
        return_lse: bool = False,
    ) -> Optional[torch.Tensor]:
        for name, tensor in (("q", q), ("output", output), ("indices", indices)):
            if tensor.ndim == (4 if name != "indices" else 3) and tensor.shape[1] != 1:
                raise ValueError(f"{name} singleton query axis must have size 1")
        if q.ndim == 4:
            q = q.squeeze(1)
        if output.ndim == 4:
            output = output.squeeze(1)
        if indices.ndim == 3:
            indices = indices.squeeze(1)
        if q.ndim != 3 or q.shape[-1] != 576 or q.dtype != torch.bfloat16:
            raise ValueError("q must be BF16 [T, H, 576]")
        t, h, _ = q.shape
        if h not in (8, 16, 32, 64, 128):
            raise ValueError("supported query heads: 8, 16, 32, 64, 128")
        if indices.shape[-1] <= 0:
            raise ValueError("topk must be positive")
        if chunks_per_block is not None and chunks_per_block < 1:
            raise ValueError("chunks_per_block must be positive")
        if not q.is_contiguous() or q.device != self._device:
            raise ValueError(f"q must be contiguous on {self._device}")
        if self._max_num_tokens is not None and t > self._max_num_tokens:
            raise ValueError("num_tokens exceeds max_num_tokens")
        if self._max_num_heads is not None and h > self._max_num_heads:
            raise ValueError("num_heads exceeds max_num_heads")
        if (
            output.shape != (t, h, 512)
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous BF16 [T, H, 512]")
        if (
            indices.ndim != 2
            or indices.shape[0] != t
            or indices.dtype != torch.int32
            or indices.stride(-1) != 1
        ):
            raise ValueError(
                "indices must be int32 [T, topk] with contiguous last dimension"
            )
        if kv_global_scale.dtype != torch.float32 or kv_global_scale.numel() != 1:
            raise ValueError(
                "kv_global_scale must contain one positive finite device FP32 value"
            )
        _cache_page_size(kv_cache)
        if t and kv_cache.shape[0] == 0:
            raise ValueError("nonempty GLM NVFP4 queries require a KV cache page")
        for name, tensor in (
            ("kv_cache", kv_cache),
            ("indices", indices),
            ("output", output),
            ("kv_global_scale", kv_global_scale),
            ("topk_length", topk_length),
            ("attn_sink", attn_sink),
        ):
            if tensor is not None and tensor.device != q.device:
                raise ValueError(f"{name} must be on {q.device}")
        if topk_length is not None and (
            topk_length.shape != (t,)
            or topk_length.dtype != torch.int32
            or not topk_length.is_contiguous()
        ):
            raise ValueError("topk_length must be contiguous int32 [T]")
        if attn_sink is not None and (
            attn_sink.shape != (h,)
            or attn_sink.dtype != torch.float32
            or not attn_sink.is_contiguous()
        ):
            raise ValueError("attn_sink must be contiguous FP32 [H]")
        lse = self._get_out_lse(t, h, out_lse)
        self.last_decode_tile_size = None
        self.last_num_splits = None
        if t == 0:
            return lse if return_lse else None
        topk = indices.shape[-1]
        route = kernel_variant or "auto"
        if route not in ("auto", "decode", "prefill", "sg"):
            raise ValueError("kernel_variant must be auto, decode, prefill, or sg")
        decode = route == "decode" or (route == "auto" and (t <= 512 or topk % 64 != 0))
        if decode:
            tile = 32 if h == 8 else 64
            total_chunks = (topk + tile - 1) // tile
            cpb = chunks_per_block or _automatic_decode_cpb(
                t, h, total_chunks, _device_sm_count(q.device)
            )
            mid_out, mid_lse = self._scratch(
                q, topk, mid_out, mid_lse, chunks_per_block=cpb
            )
            self.last_decode_tile_size = tile
            self.last_num_splits = mid_out.shape[2]
        else:
            if chunks_per_block is not None:
                raise ValueError("chunks_per_block applies only to decode")
            if not indices.is_contiguous() or topk % 64:
                raise ValueError(
                    "prefill requires contiguous indices and topk divisible by 64"
                )
            cpb = 0
            mid_out = mid_lse = None
        self.last_plan = ("decode" if decode else "sg", cpb)
        get_sparse_mla_glm_nvfp4_sm120_module().paged_attention(
            q,
            kv_cache,
            indices,
            output,
            lse,
            kv_global_scale,
            sm_scale,
            cpb,
            topk_length,
            attn_sink,
            mid_out,
            mid_lse,
        )
        return lse if return_lse else None
