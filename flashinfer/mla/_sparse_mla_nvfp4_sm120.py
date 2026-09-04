# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVFP4 cache primitives for DeepSeek-V4 sparse MLA on SM120."""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..jit.mla import (
    gen_sparse_mla_nvfp4_sm120_module,
    gen_sparse_mla_nvfp4_sm120_tile_module,
)
from ..utils import (
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


_D_NOPE = 448
_D_ROPE = 64
_D_LATENT = _D_NOPE + _D_ROPE
_PACKED_NOPE_BYTES = _D_NOPE // 2
_ROPE_BYTES = _D_ROPE * 2
_DATA_BYTES_PER_TOKEN = _PACKED_NOPE_BYTES + _ROPE_BYTES
_SCALE_BYTES_PER_TOKEN = 32
_BYTES_PER_TOKEN = _DATA_BYTES_PER_TOKEN + _SCALE_BYTES_PER_TOKEN
_CANDIDATES_PER_CHUNK = 64
_DECODE_TOKEN_BUCKETS = (1, 4, 8, 16, 32, 64)
_DECODE_AUTOTUNE_OP = "sparse_mla_sm120_nvfp4_decode"
_decode_hot_cache: dict[tuple[Any, ...], int] = {}


def _decode_token_buckets(*_args, **_kwargs) -> tuple[int, ...]:
    return _DECODE_TOKEN_BUCKETS


def _decode_token_bucket(num_tokens: int) -> int:
    for bucket in _DECODE_TOKEN_BUCKETS:
        if num_tokens <= bucket:
            return bucket
    return _DECODE_TOKEN_BUCKETS[-1]


def _init_decode_q(shapes, dtype, device):
    return (
        (torch.randn(shapes, device=device, dtype=torch.float32) / 10.0)
        .clamp(-1, 1)
        .to(dtype)
    )


def _init_decode_indices(shapes, dtype, device):
    # The live paged pools used for serving contain far more than 256 rows.
    # The autotuner profiles latency only, so arbitrary legal rows are enough.
    return torch.randint(0, 256, shapes, dtype=dtype, device=device)


def _init_decode_topk_length(shapes, dtype, device):
    return torch.full(shapes, 1 << 30, dtype=dtype, device=device)


def _decode_inputs_pre_hook(inputs):
    inputs = list(inputs)
    indices = inputs[1] if len(inputs) > 1 else None
    topk_length = inputs[6] if len(inputs) > 6 else None
    extra_indices = inputs[8] if len(inputs) > 8 else None
    extra_topk_length = inputs[9] if len(inputs) > 9 else None
    if topk_length is not None and indices is not None:
        inputs[6] = torch.full_like(topk_length, indices.shape[-1])
    if extra_topk_length is not None and extra_indices is not None:
        inputs[9] = torch.full_like(extra_topk_length, extra_indices.shape[-1])
    return inputs


@functools.cache
def _decode_tuning_config() -> TuningConfig:
    # Keep the bucket and scratch-shape contract aligned with the existing FP8
    # DSv4 sparse-MLA tuner so paired serving warms the same live batch shapes.
    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 6, 8, 9),
                dim_idx=(0, 0, 0, 0, 0),
                gen_tuning_buckets=_decode_token_buckets,
                map_to_tuning_buckets=_decode_token_bucket,
            ),
        ),
        tensor_initializers=(
            (0, _init_decode_q),
            (1, _init_decode_indices),
            (6, _init_decode_topk_length),
            (8, _init_decode_indices),
            (9, _init_decode_topk_length),
        ),
        inputs_pre_hook=_decode_inputs_pre_hook,
        constraint_specs=(
            ConstraintSpec(2, 0, lambda shapes: shapes[0][0]),
            ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),
            ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),
            ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),
        ),
    )


def _cache_page_size(cache: torch.Tensor | None) -> int:
    if cache is None:
        return 0
    if cache.ndim == 3:
        return int(cache.shape[1])
    if cache.ndim == 4 and cache.shape[1] == 1:
        return int(cache.shape[2])
    if cache.ndim == 4 and cache.shape[2] == 1:
        return int(cache.shape[1])
    raise ValueError(f"unsupported NVFP4 cache shape for decode tuning: {cache.shape}")


class _SparseMlaNvfp4DecodeRunner(TunableRunner):
    """Tune chunks-per-block while preserving the native streaming kernel."""

    def __init__(
        self,
        module,
        primary_page_size: int,
        extra_page_size: int,
    ) -> None:
        self.module = module
        self.primary_page_size = primary_page_size
        self.extra_page_size = extra_page_size

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[Any, ...]:
        topk_length = inputs[6] if len(inputs) > 6 else None
        attn_sink = inputs[7] if len(inputs) > 7 else None
        extra_indices = inputs[8] if len(inputs) > 8 else None
        extra_topk_length = inputs[9] if len(inputs) > 9 else None
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        return (
            topk_length is not None,
            attn_sink is not None,
            int(extra_topk),
            extra_topk_length is not None,
            self.primary_page_size,
            self.extra_page_size,
        )

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
    ) -> list[int]:
        del profile
        indices = inputs[1]
        extra_indices = inputs[8] if len(inputs) > 8 else None
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        num_splits = (
            indices.shape[-1] + _CANDIDATES_PER_CHUNK - 1
        ) // _CANDIDATES_PER_CHUNK + (
            extra_topk + _CANDIDATES_PER_CHUNK - 1
        ) // _CANDIDATES_PER_CHUNK
        return list(range(1, num_splits + 1))

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del do_preparation
        q, indices, mid_out, mid_lse, output, out_lse = inputs[:6]
        topk_length, attn_sink, extra_indices, extra_topk_length = inputs[6:10]
        extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
        num_splits = (
            indices.shape[-1] + _CANDIDATES_PER_CHUNK - 1
        ) // _CANDIDATES_PER_CHUNK + (
            extra_topk + _CANDIDATES_PER_CHUNK - 1
        ) // _CANDIDATES_PER_CHUNK
        self.module.sparse_mla_sm120_nvfp4_decode(
            q,
            kwargs["kv_cache"],
            indices,
            mid_out,
            mid_lse,
            output,
            out_lse,
            num_splits,
            kwargs["sm_scale"],
            topk_length,
            attn_sink,
            kwargs.get("extra_kv_cache"),
            extra_indices,
            extra_topk_length,
            int(tactic) if int(tactic) > 0 else 0,
            False,
        )
        return output


def _decode_hot_key(
    runner: _SparseMlaNvfp4DecodeRunner,
    inputs: list[torch.Tensor],
) -> tuple[Any, ...]:
    q, indices = inputs[:2]
    extra_indices = inputs[8] if len(inputs) > 8 else None
    extra_topk = extra_indices.shape[-1] if extra_indices is not None else 0
    num_splits = (
        indices.shape[-1] + _CANDIDATES_PER_CHUNK - 1
    ) // _CANDIDATES_PER_CHUNK + (
        extra_topk + _CANDIDATES_PER_CHUNK - 1
    ) // _CANDIDATES_PER_CHUNK
    return (
        _decode_token_bucket(q.shape[0]),
        q.shape[1],
        indices.shape[-1],
        extra_topk,
        num_splits,
        runner.get_cache_key_extras(inputs),
    )


def _run_nvfp4_decode(
    runner: _SparseMlaNvfp4DecodeRunner,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    mid_out: torch.Tensor,
    mid_lse: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_kv_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    chunks_per_block_override: int,
) -> None:
    inputs = [
        q,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        topk_length,
        attn_sink,
        extra_indices,
        extra_topk_length,
    ]
    forward_kwargs = {
        "sm_scale": sm_scale,
        "kv_cache": kv_cache,
        "extra_kv_cache": extra_kv_cache,
    }
    if chunks_per_block_override > 0:
        runner(
            inputs=inputs,
            tactic=chunks_per_block_override,
            **forward_kwargs,
        )
        return

    tuner = AutoTuner.get()
    if not tuner.is_tuning_mode:
        cached_tactic = _decode_hot_cache.get(_decode_hot_key(runner, inputs))
        if cached_tactic is not None:
            runner(inputs=inputs, tactic=cached_tactic, **forward_kwargs)
            return

    chosen, tactic = tuner.choose_one(
        _DECODE_AUTOTUNE_OP,
        [runner],
        _decode_tuning_config(),
        inputs,
        **forward_kwargs,
    )
    if int(tactic) > 0:
        _decode_hot_cache[_decode_hot_key(runner, inputs)] = int(tactic)
    chosen(inputs=inputs, tactic=tactic, **forward_kwargs)


@functools.cache
def get_sparse_mla_nvfp4_sm120_module():
    module = gen_sparse_mla_nvfp4_sm120_module().build_and_load()
    decode_runners: dict[tuple[int, int], _SparseMlaNvfp4DecodeRunner] = {}

    def get_decode_runner(
        kv_cache: torch.Tensor,
        extra_kv_cache: torch.Tensor | None,
    ) -> _SparseMlaNvfp4DecodeRunner:
        page_key = (
            _cache_page_size(kv_cache),
            _cache_page_size(extra_kv_cache),
        )
        runner = decode_runners.get(page_key)
        if runner is None:
            runner = _SparseMlaNvfp4DecodeRunner(module, *page_key)
            decode_runners[page_key] = runner
        return runner

    @register_custom_op(
        "flashinfer::sparse_mla_nvfp4_sm120_paged_attention",
        mutates_args=(
            "mid_out",
            "mid_lse",
            "output",
            "out_lse",
        ),
    )
    def _paged_attention(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        mid_out: torch.Tensor | None,
        mid_lse: torch.Tensor | None,
        output: torch.Tensor,
        out_lse: torch.Tensor,
        sm_scale: float,
        topk_length: torch.Tensor | None,
        attn_sink: torch.Tensor | None,
        extra_kv_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_topk_length: torch.Tensor | None,
        use_prefill: bool,
        chunks_per_block_override: int,
    ) -> None:
        if not use_prefill:
            if mid_out is None or mid_lse is None:
                raise ValueError("NVFP4 decode requires mid_out and mid_lse workspace")
            _run_nvfp4_decode(
                get_decode_runner(kv_cache, extra_kv_cache),
                q,
                kv_cache,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
                chunks_per_block_override,
            )
        else:
            module.sparse_mla_sm120_nvfp4_prefill(
                q,
                kv_cache,
                indices,
                output,
                out_lse,
                sm_scale,
                topk_length,
                attn_sink,
                extra_kv_cache,
                extra_indices,
                extra_topk_length,
            )

    @register_fake_op("flashinfer::sparse_mla_nvfp4_sm120_paged_attention")
    def _fake_paged_attention(*_args, **_kwargs) -> None:
        return None

    return SimpleNamespace(
        paged_attention=_paged_attention,
        sparse_mla_sm120_nvfp4_quantize_pack=module.sparse_mla_sm120_nvfp4_quantize_pack,
        sparse_mla_sm120_nvfp4_quantize_append=module.sparse_mla_sm120_nvfp4_quantize_append,
        sparse_mla_sm120_nvfp4_decode=module.sparse_mla_sm120_nvfp4_decode,
        sparse_mla_sm120_nvfp4_prefill=module.sparse_mla_sm120_nvfp4_prefill,
    )


@functools.cache
def get_sparse_mla_nvfp4_sm120_tile_module():
    """Build the test-only MMA-layout validation module."""
    return gen_sparse_mla_nvfp4_sm120_tile_module().build_and_load()


@supported_compute_capability([120, 121])
def _sparse_mla_nvfp4_sm120_paged_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    use_prefill: bool | None = None,
    chunks_per_block_override: int = 0,
) -> None:
    """Run the allocation-free NVFP4 sparse-MLA custom op."""
    get_sparse_mla_nvfp4_sm120_module().paged_attention(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        q.shape[0] > 64 if use_prefill is None else use_prefill,
        chunks_per_block_override,
    )


def _check_latent_kv(latent_kv: torch.Tensor, *, expected_rows: int | None) -> int:
    if not latent_kv.is_cuda:
        raise ValueError(f"latent_kv must be a CUDA tensor, got {latent_kv.device}")
    if latent_kv.dtype != torch.bfloat16:
        raise ValueError(
            f"DeepSeek-V4 latent_kv must have dtype torch.bfloat16, got {latent_kv.dtype}"
        )
    if latent_kv.ndim not in (2, 3, 4):
        raise ValueError(
            f"latent_kv must be 2D, 3D, or 4D, got shape={tuple(latent_kv.shape)}"
        )
    if latent_kv.shape[-1] != _D_LATENT:
        raise ValueError(
            f"latent_kv last dimension must be {_D_LATENT}, got {latent_kv.shape[-1]}"
        )
    if not latent_kv.is_contiguous():
        raise ValueError("latent_kv must be contiguous")
    rows = latent_kv.numel() // _D_LATENT
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(
            f"latent_kv contains {rows} rows, expected {expected_rows} rows"
        )
    return rows


def _cache_shape(cache: torch.Tensor) -> tuple[int, int, str]:
    if not cache.is_cuda:
        raise ValueError(f"cache must be a CUDA tensor, got {cache.device}")
    if cache.dtype != torch.uint8:
        raise ValueError(f"cache must have dtype torch.uint8, got {cache.dtype}")
    if cache.ndim not in (3, 4) or cache.shape[-1] != _BYTES_PER_TOKEN:
        raise ValueError(
            "cache must be [num_pages, page_size, 384], HND "
            "[num_pages, 1, page_size, 384], or NHD "
            f"[num_pages, page_size, 1, 384], got shape={tuple(cache.shape)}"
        )
    if cache.ndim == 3:
        num_pages, page_size, layout = cache.shape[0], cache.shape[1], "NHD"
    elif cache.shape[1] == 1:
        num_pages, page_size, layout = cache.shape[0], cache.shape[2], "HND"
    elif cache.shape[2] == 1:
        num_pages, page_size, layout = cache.shape[0], cache.shape[1], "NHD"
    else:
        raise ValueError(
            "cache must have a singleton latent-head dimension at axis 1 or 2"
        )
    page_dim = 1 if cache.ndim == 3 or layout == "NHD" else 2
    if cache.stride(-1) != 1 or cache.stride(page_dim) != _BYTES_PER_TOKEN:
        raise ValueError(
            "cache entries must be contiguous inside each page with strides "
            f"(..., {_BYTES_PER_TOKEN}, 1), got {cache.stride()}"
        )
    if cache.stride(0) < page_size * _BYTES_PER_TOKEN:
        raise ValueError(
            "cache page stride must cover the logical page payload, got "
            f"stride(0)={cache.stride(0)} for page_size={page_size}"
        )
    return int(num_pages), int(page_size), layout


@supported_compute_capability([120, 121])
def nvfp4_quantize_pack_sparse_mla_cache(
    latent_kv: torch.Tensor,
    *,
    kv_layout: str = "HND",
) -> torch.Tensor:
    r"""Quantize complete DeepSeek-V4 latent-KV pages to the NVFP4 cache ABI.

    ``latent_kv`` has shape ``[num_pages, page_size, 512]``. A singleton
    latent-head axis is also accepted in HND or NHD position. The first 448
    values are quantized in groups of 16 to packed E2M1 with E4M3 scales; the
    final 64 BF16 RoPE values are copied bit-for-bit.

    The returned uint8 tensor is an opaque paged cache with logical shape
    ``[num_pages, 1, page_size, 384]`` for HND or
    ``[num_pages, page_size, 1, 384]`` for NHD. Within each physical page it
    stores ``page_size * 352`` data bytes followed by ``page_size * 32`` scale
    bytes. Consumers must not interpret the last dimension as a contiguous
    per-token record.
    """
    if kv_layout not in ("HND", "NHD"):
        raise ValueError(f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}")
    _check_latent_kv(latent_kv, expected_rows=None)

    if latent_kv.ndim == 2:
        raise ValueError(
            "full-page pack requires a page dimension; use shape "
            "[num_pages, page_size, 512]"
        )
    if latent_kv.ndim == 3:
        num_pages, page_size = latent_kv.shape[:2]
    elif latent_kv.shape[1] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[2]
    elif latent_kv.shape[2] == 1:
        num_pages, page_size = latent_kv.shape[0], latent_kv.shape[1]
    else:
        raise ValueError(
            "4D latent_kv must have a singleton latent-head dimension at axis 1 or 2"
        )

    _check_latent_kv(latent_kv, expected_rows=int(num_pages) * int(page_size))
    cache_shape = (
        (num_pages, 1, page_size, _BYTES_PER_TOKEN)
        if kv_layout == "HND"
        else (num_pages, page_size, 1, _BYTES_PER_TOKEN)
    )
    cache = torch.empty(cache_shape, dtype=torch.uint8, device=latent_kv.device)
    if int(num_pages) == 0 or int(page_size) == 0:
        return cache
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_quantize_pack(
        latent_kv, cache
    )
    return cache


@supported_compute_capability([120, 121])
def nvfp4_quantize_append_sparse_mla_cache(
    latent_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    r"""Quantize and append DeepSeek-V4 latent KV by physical cache slot.

    ``cache`` accepts the 3D shorthand ``[num_pages, page_size, 384]`` in
    addition to the public 4D HND/NHD layouts. ``slot_mapping[i]`` is
    ``page_id * page_size + entry_id``. Page-strided views are accepted so the
    cache can live inside vLLM's packed physical block allocation. Negative and
    out-of-range slots are padding and are ignored. Only the addressed data row
    and scale slot are written, so prefill history is reused directly by decode.
    """
    num_pages, page_size, _ = _cache_shape(cache)
    if not slot_mapping.is_cuda:
        raise ValueError(
            f"slot_mapping must be a CUDA tensor, got {slot_mapping.device}"
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"slot_mapping must have dtype torch.int32 or torch.int64, got {slot_mapping.dtype}"
        )
    if slot_mapping.ndim != 1 or not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be a contiguous 1D tensor")
    if latent_kv.device != cache.device or slot_mapping.device != cache.device:
        raise ValueError(
            "latent_kv, slot_mapping, and cache must be on the same device"
        )
    _check_latent_kv(latent_kv, expected_rows=slot_mapping.numel())
    if num_pages * page_size == 0 and slot_mapping.numel() != 0:
        raise ValueError("cannot append to an empty cache")

    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_quantize_append(
        latent_kv, slot_mapping, cache
    )


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_m16n32k64(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    iterations: int = 1,
) -> torch.Tensor:
    """Run the internal M16N32K64 NVFP4 validation tile."""
    output = torch.empty((16, 32), dtype=torch.float32, device=a.device)
    get_sparse_mla_nvfp4_sm120_tile_module().sparse_mla_sm120_nvfp4_m16n32k64(
        a, b, sfa, sfb, output, iterations
    )
    return output


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_m16n8k64_candidate_major(
    a: torch.Tensor,
    b: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    *,
    iterations: int = 1,
) -> torch.Tensor:
    """Run the internal candidate-major PV validation tile."""
    output = torch.empty((16, 8), dtype=torch.float32, device=a.device)
    get_sparse_mla_nvfp4_sm120_tile_module().sparse_mla_sm120_nvfp4_m16n8k64_candidate_major(
        a, b, sfa, sfb, output, iterations
    )
    return output


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    chunks_per_block_override: int = 0,
    stage1_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the internal DeepSeek-V4 NVFP4 sparse-MLA decode prototype."""
    if q.ndim != 3 or q.shape[-1] != _D_LATENT:
        raise ValueError(f"q must be [T, H, {_D_LATENT}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be a 2D int32 tensor")
    num_tokens, num_heads, _ = q.shape
    topk = indices.shape[1]
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    extra_topk = extra_indices.shape[1] if extra_indices is not None else 0
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    mid_out = torch.empty(
        (num_tokens, num_heads, num_splits, _D_LATENT),
        dtype=torch.bfloat16,
        device=q.device,
    )
    mid_lse = torch.empty(
        (num_tokens, num_heads, num_splits),
        dtype=torch.float32,
        device=q.device,
    )
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_decode(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        num_splits,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
        chunks_per_block_override,
        stage1_only,
    )
    return output, out_lse


@supported_compute_capability([120, 121])
def _nvfp4_sparse_mla_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_kv_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the single-launch streaming DeepSeek-V4 NVFP4 prefill kernel."""
    if q.ndim != 3 or q.shape[-1] != _D_LATENT:
        raise ValueError(f"q must be [T, H, {_D_LATENT}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16 or not q.is_cuda or not q.is_contiguous():
        raise ValueError("q must be a contiguous CUDA bfloat16 tensor")
    if indices.ndim != 2 or indices.dtype != torch.int32:
        raise ValueError("indices must be a 2D int32 tensor")
    num_tokens, num_heads, _ = q.shape
    if (extra_kv_cache is None) != (extra_indices is None):
        raise ValueError("extra_kv_cache and extra_indices must be provided together")
    if extra_topk_length is not None and extra_indices is None:
        raise ValueError("extra_topk_length requires extra_indices")
    output = torch.empty_like(q)
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    get_sparse_mla_nvfp4_sm120_module().sparse_mla_sm120_nvfp4_prefill(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        topk_length,
        attn_sink,
        extra_kv_cache,
        extra_indices,
        extra_topk_length,
    )
    return output, out_lse


__all__ = [
    "nvfp4_quantize_append_sparse_mla_cache",
    "nvfp4_quantize_pack_sparse_mla_cache",
]
