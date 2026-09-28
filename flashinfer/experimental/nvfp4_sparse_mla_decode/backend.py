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
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Optional

import torch

from .jit import gen_nvfp4_sparse_mla_decode_module

NUM_HEADS = 16
HEAD_DIM = 576  # 512 NoPE + 64 RoPE
V_HEAD_DIM = 512
ROW_BYTES = 352  # one nvfp4_ds_mla row
STAGE_KEYS = 32
MAX_KEYS_PER_CTA = 1024
RING_SLOTS = 3
# Cluster sizes the automatic plan considers, largest first. 7 fits no more clusters per wave than 8.
PLAN_CTAS = (8, 6, 5, 4, 3)
# Cluster sizes accepted from callers: the ones validated on SM100 (2 fits the kernel but was never validated).
VALID_CTAS = range(3, 9)
LOG2E = math.log2(math.e)


@functools.cache
def get_module():
    return gen_nvfp4_sparse_mla_decode_module().build_and_load()


def is_valid_config(topk_width: int, num_ctas: int) -> bool:
    """Whether every CTA of a ``num_ctas`` cluster gets 3 to 32 stages of 32 keys (the kernel's own check)."""
    if topk_width <= 0 or topk_width % STAGE_KEYS or num_ctas not in VALID_CTAS:
        return False
    stages = topk_width // STAGE_KEYS
    return (
        -(-stages // num_ctas) <= MAX_KEYS_PER_CTA // STAGE_KEYS
        and stages // num_ctas >= RING_SLOTS
    )


def select_num_ctas(num_tokens: int, topk_width: int, capacity: Dict[int, int]) -> int:
    """The largest valid cluster size whose ``num_tokens`` clusters fit on the device in one wave.

    ``capacity[c]`` is the number of ``c``-CTA clusters the device runs at once. When no size fits in one wave,
    the smallest valid size is used and the launch runs in several waves.
    """
    valid = [c for c in PLAN_CTAS if is_valid_config(topk_width, c)]
    if not valid:
        raise ValueError(
            f"NVFP4 sparse MLA decode does not support topk_width={topk_width}"
        )
    for c in valid:
        if num_tokens <= capacity.get(c, 0):
            return c
    return valid[-1]


@functools.cache
def _capacity(device_index: int) -> Dict[int, int]:
    module = get_module()
    with torch.cuda.device(device_index):
        return {c: int(module.max_active_clusters(c)) for c in PLAN_CTAS}


def _check_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    out: Optional[torch.Tensor],
) -> None:
    tensors = {"query": query, "kv_cache": kv_cache, "indices": indices}
    if out is not None:
        tensors["out"] = out
    for name, t in tensors.items():
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.device != query.device:
            raise ValueError(f"{name} is on {t.device}, query on {query.device}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if torch.cuda.get_device_capability(query.device) != (10, 0):
        raise RuntimeError(
            "NVFP4 sparse MLA decode requires compute capability 10.0 (SM100), got "
            f"{torch.cuda.get_device_capability(query.device)}"
        )
    if (
        query.dtype != torch.float8_e4m3fn
        or query.dim() != 3
        or query.shape[1:] != (NUM_HEADS, HEAD_DIM)
    ):
        raise ValueError(
            f"query must be [num_tokens, {NUM_HEADS}, {HEAD_DIM}] float8_e4m3fn, got "
            f"{list(query.shape)} {query.dtype}"
        )
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.dim() < 2
        or kv_cache.shape[-1] != ROW_BYTES
    ):
        raise ValueError(
            f"kv_cache must be uint8 [..., {ROW_BYTES}] (nvfp4_ds_mla rows), got "
            f"{list(kv_cache.shape)} {kv_cache.dtype}"
        )
    if (
        indices.dtype != torch.int32
        or indices.dim() != 2
        or indices.shape[0] != query.shape[0]
    ):
        raise ValueError(
            f"indices must be int32 [num_tokens, topk], got {list(indices.shape)} {indices.dtype}"
        )
    if out is not None and (
        out.dtype != torch.bfloat16
        or out.shape != (query.shape[0], NUM_HEADS, V_HEAD_DIM)
    ):
        raise ValueError(
            f"out must be [num_tokens, {NUM_HEADS}, {V_HEAD_DIM}] bfloat16, got "
            f"{list(out.shape)} {out.dtype}"
        )
    for name in ("query", "kv_cache"):
        if tensors[name].data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")


def run(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    out: Optional[torch.Tensor],
    num_ctas_per_token: Optional[int],
) -> torch.Tensor:
    _check_inputs(query, kv_cache, indices, out)
    num_tokens, topk_width = query.shape[0], indices.shape[1]
    if out is None:
        out = torch.empty(
            (num_tokens, NUM_HEADS, V_HEAD_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        )
    if num_tokens == 0:
        return out
    if num_ctas_per_token is None:
        num_ctas = select_num_ctas(
            num_tokens, topk_width, _capacity(query.device.index)
        )
    else:
        num_ctas = int(num_ctas_per_token)
        if not is_valid_config(topk_width, num_ctas):
            raise ValueError(
                f"num_ctas_per_token={num_ctas} is not valid for topk_width={topk_width}: each CTA needs "
                f"{RING_SLOTS} to {MAX_KEYS_PER_CTA // STAGE_KEYS} stages of {STAGE_KEYS} keys "
                f"(sizes {VALID_CTAS.start} to {VALID_CTAS.stop - 1})"
            )
    get_module().run(
        kv_cache,
        query,
        indices,
        out,
        num_ctas,
        float(bmm1_scale) * LOG2E,
        float(bmm2_scale),
    )
    return out
