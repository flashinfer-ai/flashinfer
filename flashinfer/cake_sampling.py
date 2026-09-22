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

Fused radix top-k -> sparse top-p -> sampling for Blackwell (sm_100a / sm_103a).

Two frozen kernels per row of probabilities:

1. ``radix_topk``: one thread-block cluster per row selects the exact top-k by a three-pass
   radix select (11/11/10 bits of the float32 key), reducing the 2048-bucket histograms across
   the cluster through distributed shared memory, and writes a ``[batch, 1024]`` slab of
   (value, index) pairs.
2. ``sparse_topp_sample``: one CTA per row sorts the slab prefix, keeps the shortest prefix
   whose exclusive mass is below ``top_p * mass(top-k)``, renormalizes, and draws one token by
   inverse CDF from ``curand_init(seed, row, offset)``.  It is launched with programmatic
   dependent launch so its prologue overlaps the tail of stage 1.

Semantics match :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` (same support, same Philox stream advancement).  Requests
the kernels cannot serve (top-k disabled, ``k > 1024``, non-float32 rows, very large
``batch * vocab``, or an unsupported GPU) are dispatched to that function.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from .api_logging import flashinfer_api
from .jit.cake_sampling import (
    arch_dir_for_capability,
    load_cake_sampling_module,
    load_manifest,
)
from .sampling import get_seed_and_offset

_THREADS = 512
_MAX_CLUSTER = 8
_TOPK_SCALAR, _TOPK_PER_ROW = 1, 2
_TOPP_SCALAR, _TOPP_PER_ROW = 1, 2
# Clusters are co-scheduled inside one GPC; measured B200 single-wave CTA capacity per cluster size.
_WAVE_CTAS = {1: 148, 2: 144, 4: 128, 8: 64}
_PREFERRED_MIN_EPT = 16
# batch * vocab beyond which the persistent FlashInfer radix top-k is faster than per-row clusters.
_LARGE_BATCH_ELEMENTS = 1 << 24

_WORKSPACES: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _arch(device: torch.device) -> Optional[str]:
    return arch_dir_for_capability(torch.cuda.get_device_capability(device))


def _stage1_variants(arch: str) -> list[tuple[int, int]]:
    return [(v["cluster"], v["ept"]) for v in load_manifest(arch)["stage1"]]


def _stage23_variants(arch: str) -> list[tuple[int, int]]:
    return [(v["threads"], v["items"]) for v in load_manifest(arch)["stage23"]]


def _slab(arch: str) -> int:
    return int(load_manifest(arch)["slab_entries"])


def choose_stage1(arch: str, batch: int, vocab: int) -> tuple[int, int]:
    """``(cluster, ept)`` for ``batch`` rows of ``vocab`` entries: fewest waves, then a register
    chunk of at least 16 entries, then the larger cluster."""
    epts = sorted({e for _, e in _stage1_variants(arch)})
    available = set(_stage1_variants(arch))
    candidates = []
    for cluster in (1, 2, 4, 8):
        need = math.ceil(vocab / (_THREADS * cluster))
        ept = next((e for e in epts if e >= need), None)
        if ept is None or (cluster, ept) not in available:
            continue
        if cluster > 1 and _THREADS * ept * (cluster // 2) >= vocab:
            continue
        candidates.append((cluster, ept))
    if not candidates:
        raise ValueError(f"vocab={vocab} exceeds the frozen stage-1 capacity")

    def waves(c: int) -> int:
        return -(-(batch * c) // _WAVE_CTAS[c])

    return min(
        candidates,
        key=lambda ce: (waves(ce[0]), 0 if ce[1] >= _PREFERRED_MIN_EPT else 1, -ce[0]),
    )


def choose_stage23(arch: str, top_k_max: int) -> tuple[int, int]:
    """Smallest frozen ``(threads, items)`` slab that holds ``top_k_max`` entries."""
    for threads, items in sorted(_stage23_variants(arch), key=lambda ti: ti[0] * ti[1]):
        if threads * items >= top_k_max:
            return threads, items
    raise ValueError(f"top_k_max={top_k_max} exceeds the frozen stage-2/3 capacity")


def cake_sampling_route(
    probs: torch.Tensor,
    top_k: Optional[Union[int, torch.Tensor]],
    top_k_max: Optional[int] = None,
) -> str:
    """``"pipeline"`` when the frozen kernels serve this request, else ``"fallback:<reason>"``.

    ``top_k_max`` avoids a device sync when ``top_k`` is a tensor.
    """
    if not probs.is_cuda:
        return "fallback:device"
    if probs.dtype != torch.float32:
        return "fallback:dtype"
    if probs.dim() != 2 or probs.stride(1) != 1 or probs.stride(0) != probs.size(1):
        return "fallback:layout"
    arch = _arch(probs.device)
    if arch is None:
        return "fallback:arch"
    batch, vocab = probs.shape
    if top_k is None:
        return "fallback:no_top_k"
    if isinstance(top_k, int):
        kmax = top_k
    else:
        kmax = int(top_k_max) if top_k_max is not None else int(top_k.max().item())
    if kmax <= 0 or kmax >= vocab:
        return "fallback:top_k_disabled"
    if kmax > _slab(arch):
        return "fallback:top_k_gt_slab"
    try:
        choose_stage1(arch, batch, vocab)
    except ValueError:
        return "fallback:vocab_too_large"
    if batch * vocab >= _LARGE_BATCH_ELEMENTS:
        return "fallback:large_batch"
    return "pipeline"


def _workspace(batch: int, slab: int, device: torch.device):
    key = (batch, slab, device.index or 0)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = (
            torch.empty(batch, slab, device=device, dtype=torch.float32),
            torch.empty(batch, slab, device=device, dtype=torch.int32),
            torch.empty(batch, device=device, dtype=torch.int32),
        )
        _WORKSPACES[key] = ws
    return ws


@flashinfer_api
def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    top_p: Union[float, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    renorm_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    r"""Fused top-k-then-top-p sampling from probabilities (Blackwell radix pipeline).

    Parameters
    ----------
    probs: torch.Tensor
        ``float32 [batch, vocab]`` probabilities, contiguous.
    top_k: Union[int, torch.Tensor]
        Number of candidates kept per row (``int`` or ``int32 [batch]``), ``1 <= k <= 1024``.
    top_p: Union[float, torch.Tensor]
        Nucleus threshold in ``(0, 1]`` (``float`` or ``float32 [batch]``).
    top_k_max: Optional[int]
        Upper bound of ``top_k`` when it is a tensor (avoids a device synchronization).
    deterministic: bool
        Break probability ties by vocabulary index so equal inputs replay bit-identically.
    generator: Optional[torch.Generator]
        Source of the Philox seed/offset (default CUDA generator when omitted), advanced exactly
        like :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`.
    philox_seed, philox_offset: Optional[int]
        Explicit Philox parameters (both required together); ``generator`` is then not touched.
    out: Optional[torch.Tensor]
        ``int32 [batch]`` output buffer (allocated when omitted); ``renorm_out`` optionally
        receives ``float32 [batch, 1024]`` renormalized kept probabilities in slab order.
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        Stage-1 slab buffers ``(values float32 [batch, 1024], indices int32 [batch, 1024],
        counts int32 [batch])``. When given, the exact top-k slab of this call is left in them
        (the slab order matches ``renorm_out``); otherwise a cached per-shape workspace is used.
    enable_pdl: bool
        Launch stage 2/3 with programmatic dependent launch.

    Returns
    -------
    samples: torch.Tensor
        ``int32 [batch]`` sampled token ids.
    """
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route != "pipeline":
        from .sampling import top_k_top_p_sampling_from_probs as _fallback

        res = _fallback(
            probs,
            top_k,
            top_p,
            filter_apply_order="top_k_first",
            deterministic=deterministic,
            generator=generator,
        )
        if out is not None:
            out.copy_(res.to(torch.int32))
            return out
        return res.to(torch.int32)

    arch = _arch(probs.device)
    batch, vocab = probs.shape
    kmax = (
        top_k
        if isinstance(top_k, int)
        else (int(top_k_max) if top_k_max is not None else int(top_k.max().item()))
    )
    cluster, ept = choose_stage1(arch, batch, vocab)
    threads, items = choose_stage23(arch, kmax)
    slab = _slab(arch)
    vals, idxs, cnt = (
        workspace if workspace is not None else _workspace(batch, slab, probs.device)
    )
    if out is None:
        out = torch.empty(batch, device=probs.device, dtype=torch.int32)
    if (philox_seed is None) != (philox_offset is None):
        raise ValueError("philox_seed and philox_offset must be given together")
    if philox_seed is None:
        philox_seed, philox_offset = get_seed_and_offset(batch, generator, probs.device)
    if isinstance(top_k, int):
        k_arr, k_scalar, k_kind = cnt, int(top_k), _TOPK_SCALAR
    else:
        k_arr, k_scalar, k_kind = top_k, 0, _TOPK_PER_ROW
    if isinstance(top_p, (int, float)):
        p_arr, p_scalar, p_kind = probs, float(top_p), _TOPP_SCALAR
    else:
        p_arr, p_scalar, p_kind = top_p, 0.0, _TOPP_PER_ROW
    renorm = renorm_out if renorm_out is not None else vals
    module = load_cake_sampling_module(arch)
    stream = torch.cuda.current_stream(device=probs.device).cuda_stream
    module.radix_topk(
        probs, k_arr, k_scalar, k_kind, vals, idxs, cnt, cluster, ept, stream
    )
    module.sparse_topp_sample(
        vals,
        idxs,
        cnt,
        p_arr,
        p_scalar,
        p_kind,
        out,
        renorm,
        int(philox_seed) & 0xFFFFFFFFFFFFFFFF,
        int(philox_offset) & 0xFFFFFFFFFFFFFFFF,
        1 if deterministic else 0,
        1 if renorm_out is not None else 0,
        threads,
        items,
        1 if enable_pdl else 0,
        stream,
    )
    return out


@flashinfer_api
def top_k_probs_to_slab(
    probs: torch.Tensor,
    top_k: Union[int, torch.Tensor],
    *,
    top_k_max: Optional[int] = None,
    out_vals: Optional[torch.Tensor] = None,
    out_idx: Optional[torch.Tensor] = None,
    out_count: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Stage 1 alone: exact per-row top-k into ``(values [batch, 1024], indices [batch, 1024],
    counts [batch])``; entries beyond ``count`` are undefined.  Same dispatch conditions as
    :func:`top_k_top_p_sampling_from_probs` except that large batches are accepted."""
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route not in ("pipeline", "fallback:large_batch"):
        raise ValueError(f"frozen radix top-k cannot serve this request ({route})")
    arch = _arch(probs.device)
    batch, vocab = probs.shape
    slab = _slab(arch)
    cluster, ept = choose_stage1(arch, batch, vocab)
    vals = (
        out_vals
        if out_vals is not None
        else torch.empty(batch, slab, device=probs.device, dtype=torch.float32)
    )
    idxs = (
        out_idx
        if out_idx is not None
        else torch.empty(batch, slab, device=probs.device, dtype=torch.int32)
    )
    cnt = (
        out_count
        if out_count is not None
        else torch.empty(batch, device=probs.device, dtype=torch.int32)
    )
    if isinstance(top_k, int):
        k_arr, k_scalar, k_kind = cnt, int(top_k), _TOPK_SCALAR
    else:
        k_arr, k_scalar, k_kind = top_k, 0, _TOPK_PER_ROW
    stream = torch.cuda.current_stream(device=probs.device).cuda_stream
    load_cake_sampling_module(arch).radix_topk(
        probs, k_arr, k_scalar, k_kind, vals, idxs, cnt, cluster, ept, stream
    )
    return vals, idxs, cnt


__all__ = [
    "cake_sampling_route",
    "choose_stage1",
    "choose_stage23",
    "top_k_probs_to_slab",
    "top_k_top_p_sampling_from_probs",
]
