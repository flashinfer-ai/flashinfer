"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Backend of :mod:`flashinfer.cake_fused_qk_rope_append`: the Cake fused QK
RMSNorm + NeoX RoPE + BF16 Q output + paged KV append kernel (SM90 / SM100 /
SM103) as one self-contained package (the host API body, the ``cake_jit``
registry with the stage policy, the ``cake_build`` loader and the generated
CUDA sources under ``csrc/``).  The stable, ``@flashinfer_api``-decorated
entry point and the full user-facing documentation live in
:mod:`flashinfer.cake_fused_qk_rope_append`; import that module.

The submit path allocates nothing on the device, never synchronises the host
and is CUDA-graph capturable: every launch argument is a by-value scalar or a
tensor pointer, and the stage (the generated build) is selected from the
architecture, the SM count and the problem shape only, so a captured graph
replays the same build.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    HEAD_DIM,
    KNOB_KEYS,
    MODULES,
    STAGE_POLICY,
    SUPPORTED_HEADS,
    arch_for_capability,
    clear_units_per_request,
    stage_for,
    stages_for,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The kernel file's default clearing granularity (16-byte vectors per clearing
# unit); the public ``launch_plan`` predates the knob and keeps it implicit.
_DEFAULT_CLEAR_UNIT_CHUNKS = 256


def launch_plan(
    num_rows: int,
    num_requests: int,
    page_size: int,
    num_kv_heads: int,
    warps_per_row: int,
    clear_units_per_warp: int = 1,
) -> dict[str, Any]:
    """Grid and host-derived scalars (mirrors the Cake kernel's ``launch_plan``).

    ``warps_per_row`` / ``clear_units_per_warp`` are the selected stage's build
    knobs; the registry's :func:`cake_jit.launch_plan` evaluates the same formula
    from a module record.
    """
    record = {
        "knobs": {
            "num_kv_heads": int(num_kv_heads),
            "warps_per_row": int(warps_per_row),
            "clear_units_per_warp": int(clear_units_per_warp),
            "clear_unit_chunks": _DEFAULT_CLEAR_UNIT_CHUNKS,
        }
    }
    return _loader.launch_plan(
        record,
        num_rows=int(num_rows),
        num_requests=int(num_requests),
        page_size=int(page_size),
    )


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
    qkv: torch.Tensor,
    cos_sin: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indptr: torch.Tensor,
    page_indices: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    qk_norm_policy: int = 0,
    q_norm_weight: Optional[torch.Tensor] = None,
    k_norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    out_q: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None,
    out_v: Optional[torch.Tensor] = None,
    clear_unused_last_page_rows: bool = True,
) -> torch.Tensor:
    """Body of :func:`flashinfer.cake_fused_qk_rope_append.cake_fused_qk_rmsnorm_rope_append_paged_kv_cache`."""
    _check(qk_norm_policy in (0, 1, 2), "qk_norm_policy must be 0, 1 or 2")
    _check(
        qkv.dim() == 2 and qkv.dtype == torch.bfloat16, "qkv must be a 2-D bf16 tensor"
    )
    _check(qkv.is_cuda, "qkv must be a CUDA tensor")
    device = qkv.device
    num_rows = int(qkv.shape[0])
    width = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    _check(
        int(qkv.shape[1]) == width,
        f"qkv must have {width} columns for this head configuration",
    )
    _check(
        cos_sin.dtype == torch.float32
        and cos_sin.dim() == 2
        and int(cos_sin.shape[1]) == HEAD_DIM,
        "cos_sin must be f32 [max_position, 128]",
    )
    for name, t in (
        ("seq_lens", seq_lens),
        ("q_indptr", q_indptr),
        ("page_indices", page_indices),
    ):
        _check(t.dtype == torch.int32, f"{name} must be int32")
    num_requests = int(seq_lens.shape[0])
    _check(
        seq_lens.dim() == 1
        and q_indptr.dim() == 1
        and int(q_indptr.shape[0]) == num_requests + 1,
        "q_indptr must be [B + 1] and seq_lens [B]",
    )
    _check(
        page_indices.dim() == 2 and int(page_indices.shape[0]) == num_requests,
        "page_indices must be [B, max_pages]",
    )
    for name, t in (("key_cache", key_cache), ("value_cache", value_cache)):
        _check(
            t.dtype == torch.bfloat16
            and t.dim() == 4
            and int(t.shape[2]) == num_kv_heads
            and int(t.shape[3]) == HEAD_DIM,
            f"{name} must be bf16 [num_pages, page_size, num_kv_heads, 128]",
        )
    _check(
        tuple(key_cache.shape) == tuple(value_cache.shape),
        "key_cache and value_cache must have equal shapes",
    )
    page_size = int(key_cache.shape[1])
    if out_q is None:
        out_q = torch.empty(
            (num_rows, num_q_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    _check(
        out_q.dtype == torch.bfloat16
        and tuple(out_q.shape) == (num_rows, num_q_heads, HEAD_DIM),
        "out_q must be bf16 [T, num_q_heads, 128]",
    )
    for name, t in (("out_k", out_k), ("out_v", out_v)):
        if t is not None:
            _check(
                t.dtype == torch.bfloat16
                and tuple(t.shape) == (num_rows, num_kv_heads, HEAD_DIM),
                f"{name} must be bf16 [T, num_kv_heads, 128]",
            )
    if qk_norm_policy != 0:
        _check(
            q_norm_weight is not None and k_norm_weight is not None,
            "q_norm_weight and k_norm_weight are required when qk_norm_policy != 0",
        )
    if q_norm_weight is None or k_norm_weight is None:
        # qk_norm_policy == 0 never reads the weights: hand the kernel an existing f32 [128] row
        # instead of allocating and filling a tensor on every call (no extra kernel, graph-safe).
        placeholder = cos_sin[0, :HEAD_DIM]
        q_norm_weight = placeholder if q_norm_weight is None else q_norm_weight
        k_norm_weight = placeholder if k_norm_weight is None else k_norm_weight
    for name, w in (("q_norm_weight", q_norm_weight), ("k_norm_weight", k_norm_weight)):
        _check(
            w.dtype == torch.float32 and tuple(w.shape) == (HEAD_DIM,),
            f"{name} must be f32 [128]",
        )
    tensors = {
        "qkv": qkv,
        "cos_sin": cos_sin,
        "q_indptr": q_indptr,
        "seq_lens": seq_lens,
        "page_table": page_indices,
        "k_cache": key_cache,
        "v_cache": value_cache,
        "out_q": out_q,
        "out_k": out_k if out_k is not None else key_cache,
        "out_v": out_v if out_v is not None else value_cache,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
    }
    for name, t in tensors.items():
        _check(t.is_contiguous(), f"{name} must be contiguous")
        _check(t.device == device, f"{name} must be on {device}")

    device_index = _device_index(device)
    arch = _loader.device_arch(device_index)
    stage = _loader.stage_for(
        num_q_heads,
        num_kv_heads,
        arch=arch,
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
        num_sms=_loader.sm_count(device_index),
        qk_norm_policy=int(qk_norm_policy),
    )
    plan = _loader.launch_plan(
        MODULES[stage],
        num_rows=num_rows,
        num_requests=num_requests,
        page_size=page_size,
    )
    values: dict[str, Any] = {
        **tensors,
        "num_rows": num_rows,
        "num_requests": num_requests,
        "max_pages": int(page_indices.shape[1]),
        "page_size": page_size,
        "k_page_stride": int(key_cache.stride(0)),
        "v_page_stride": int(value_cache.stride(0)),
        "num_row_ctas": int(plan["num_row_ctas"]),
        "clear_units_per_request": int(plan["clear_units_per_request"]),
        "qk_norm_policy": int(qk_norm_policy),
        "eps": float(eps),
        "use_out_k": 0 if out_k is None else 1,
        "use_out_v": 0 if out_v is None else 1,
        "clear_last_page": 1 if clear_unused_last_page_rows else 0,
    }
    with torch.cuda.device(device):
        _loader.run_module(stage, values, device_index=device_index)
    return out_q


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "KNOB_KEYS",
    "MODULES",
    "STAGE_POLICY",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SUPPORTED_HEADS",
    "arch_for_capability",
    "cake_fused_qk_rmsnorm_rope_append_paged_kv_cache",
    "clear_units_per_request",
    "launch_plan",
    "stage_for",
    "stages_for",
]
