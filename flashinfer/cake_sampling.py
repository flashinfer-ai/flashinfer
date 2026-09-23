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

Fused radix top-k -> sparse top-p -> sampling for Hopper and newer (compute capability 9.x,
10.x, 11.x, 12.x; one frozen source compiled once into a fatbin for the CompilationContext
targets).

Two frozen kernels per row of probabilities:

1. ``radix_topk``: one thread-block cluster per row selects the exact top-k by a three-pass
   radix select (11/11/10 bits of the float32 key), reducing the 2048-bucket histograms across
   the cluster through distributed shared memory, and writes a ``[batch, 1024]`` slab of
   (value, index) pairs.
2. ``sparse_topp_sample``: one CTA per row sorts the slab (descending probability, ascending
   index), keeps the shortest prefix whose exclusive mass is below ``top_p * mass(top-k)``,
   renormalizes, draws one token by inverse CDF from ``curand_init(seed, row, offset)``, and
   rewrites the slab in sorted order.  It is launched with programmatic dependent launch so its
   prologue overlaps the tail of stage 1.

Semantics follow :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs` with
``filter_apply_order="top_k_first"`` (same support, same Philox stream advancement) with these
guarantees on top:

* **Strict determinism.** Ties at the top-k boundary are resolved toward the lower vocabulary
  index (the support is exactly the first ``k`` entries of ``lexsort(-prob, index)``), every
  top-p / sampling decision uses exact 64-bit fixed-point prefix sums, and no atomic decides an
  output.  Identical inputs give bitwise identical samples, ``renorm_out`` and slab for every
  kernel variant, stream launch or CUDA-graph replay.
* **NaN / Inf.** NaN, negative and ``-0.0`` probabilities are treated as ``+0`` and never
  sampled.  A row containing ``+inf`` keeps exactly its ``+inf`` entries, samples uniformly among
  them and renormalizes them to ``1/m`` (``top_p`` is ignored for that row).  A row whose top-k
  mass is zero returns its smallest slab index with an all-zero ``renorm_out``.  Entries below
  ``2**-53`` times the row maximum carry zero mass.  Every output is finite.
* ``top_p`` is clamped to ``(0, 1]`` per row (``p <= 0`` selects the argmax, ``p >= 1`` or NaN
  keeps every entry with mass); ``top_k`` is clamped to ``[1, min(vocab, 1024)]`` per row.

Requests the kernels cannot serve (top-k disabled, ``k > 1024``, non-float32 rows, or an
unsupported GPU) are dispatched to :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`
(``deterministic=True``); large ``batch * vocab`` launches run on the streaming stage-1 variants.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Union

import torch

from .api_logging import flashinfer_api
from .jit.cake_sampling import (
    load_cake_sampling_module,
    load_manifest,
    supported_capability,
)
from .sampling import get_seed_and_offset

_THREADS = 512
_MAX_CLUSTER = 8
_TOPK_SCALAR, _TOPK_PER_ROW = 1, 2
_TOPP_SCALAR, _TOPP_PER_ROW = 1, 2
# Clusters are co-scheduled inside one GPC, so the number of stage-1 CTAs that run in a single
# wave depends on the cluster size and on the device's SM / GPC layout.  Measured single-wave CTA
# capacity per cluster size, keyed by SM count: 148 = B200 / GB300 (B300 tracks it), 132 = H100
# SXM (64 cluster-4 CTAs run in one wave and 128 take two, so the GPC layout bounds the capacity
# to 112-127; 120 is used).  A device with another SM count uses the table of the nearest SM count.
_WAVE_CTAS_BY_SM_COUNT: dict[int, dict[int, int]] = {
    148: {1: 148, 2: 144, 4: 128, 8: 64},
    132: {1: 132, 2: 132, 4: 120, 8: 64},
}
_DEFAULT_SM_COUNT = 148
_PREFERRED_MIN_EPT = 16
# Stage-1 cost model (fitted on B200 stage-1 CUPTI microseconds, k = 50, 49 (vocab, batch)
# cells): a register-resident wave costs _RESIDENT_BASE_US + _RESIDENT_PER_EPT_US per register
# entry; a streaming wave costs _STREAM_WAVE_BASE_US plus _STREAM_CHUNK_US per 512 x 16-entry
# chunk each CTA walks.  The constants minimise the dispatcher's regret against the measured
# best variant per cell (0.07 summed relative regret); they only rank the frozen variants.
_RESIDENT_BASE_US = 3.0
_RESIDENT_PER_EPT_US = 0.2
_STREAM_WAVE_BASE_US = 8.0
_STREAM_CHUNK_US = 0.8

_WORKSPACES: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _capability(device: torch.device) -> Optional[tuple[int, int]]:
    return supported_capability(torch.cuda.get_device_capability(device))


def _stage1_variants(smem_limit: Optional[int] = None) -> list[tuple[int, int, bool]]:
    """Frozen stage-1 variants whose dynamic shared memory fits ``smem_limit`` (all when None).

    The frozen variants were sized for the 227 KB opt-in limit of 9.x-11.x devices; 12.x devices
    opt in to 99 KB, so the larger register-resident variants are not candidates there."""
    return [
        (v["cluster"], v["ept"], bool(v["stream"]))
        for v in load_manifest()["stage1"]
        if smem_limit is None or int(v["dynamic_smem_bytes"]) <= smem_limit
    ]


def _stage23_variants() -> list[tuple[int, int]]:
    return [(v["threads"], v["items"]) for v in load_manifest()["stage23"]]


def _slab() -> int:
    return int(load_manifest()["slab_entries"])


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def _smem_optin(device_index: int) -> int:
    return int(
        torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin
    )


def _wave_ctas(sm_count: int) -> dict[int, int]:
    nearest = min(_WAVE_CTAS_BY_SM_COUNT, key=lambda n: (abs(n - sm_count), -n))
    return _WAVE_CTAS_BY_SM_COUNT[nearest]


def choose_stage1(
    batch: int,
    vocab: int,
    sm_count: Optional[int] = None,
    smem_limit: Optional[int] = None,
) -> tuple[int, int, bool]:
    """``(cluster, ept, stream)`` for ``batch`` rows of ``vocab`` entries.

    Register-resident candidates: fewest waves, then a register chunk of at least 16 entries,
    then the larger cluster.  That resident choice is compared with every streaming variant
    through the fitted cost model (resident ``waves * (_RESIDENT_BASE_US + _RESIDENT_PER_EPT_US *
    ept)`` against streaming ``waves * (_STREAM_WAVE_BASE_US + _STREAM_CHUNK_US * chunks)``); the
    resident variant wins ties and streaming ties prefer the smaller cluster.  The wave table is
    selected by ``sm_count`` (the current device's SM count when omitted; B200 148 and H100 132
    are measured), the cost constants were fitted on B200 and rank the frozen variants on every
    device.  Variants needing more dynamic shared memory than ``smem_limit`` (the current
    device's opt-in limit when omitted) are not candidates."""
    if sm_count is None:
        sm_count = (
            _sm_count(torch.cuda.current_device())
            if torch.cuda.is_available()
            else _DEFAULT_SM_COUNT
        )
    if smem_limit is None and torch.cuda.is_available():
        smem_limit = _smem_optin(torch.cuda.current_device())
    wave_ctas = _wave_ctas(int(sm_count))
    variants = _stage1_variants(smem_limit)
    epts = sorted({e for _, e, st in variants if not st})
    available = {(c, e) for c, e, st in variants if not st}
    streaming = [(c, e) for c, e, st in variants if st]
    candidates = []
    for cluster in (1, 2, 4, 8):
        need = math.ceil(vocab / (_THREADS * cluster))
        ept = next((e for e in epts if e >= need), None)
        if ept is None or (cluster, ept) not in available:
            continue
        if cluster > 1 and _THREADS * ept * (cluster // 2) >= vocab:
            continue
        candidates.append((cluster, ept))

    def waves(c: int) -> int:
        return -(-(batch * c) // wave_ctas[c])

    resident = None
    if candidates:
        resident = min(
            candidates,
            key=lambda ce: (
                waves(ce[0]),
                0 if ce[1] >= _PREFERRED_MIN_EPT else 1,
                -ce[0],
            ),
        )
    if not streaming:
        if resident is None:
            raise ValueError(f"vocab={vocab} exceeds the frozen stage-1 capacity")
        return resident[0], resident[1], False

    def stream_cost(ce: tuple[int, int]) -> float:
        chunks = math.ceil(vocab / (ce[0] * _THREADS * ce[1]))
        return waves(ce[0]) * (_STREAM_WAVE_BASE_US + _STREAM_CHUNK_US * chunks)

    best = min(streaming, key=lambda ce: (stream_cost(ce), ce[0]))
    if resident is not None:
        resident_cost = waves(resident[0]) * (
            _RESIDENT_BASE_US + _RESIDENT_PER_EPT_US * resident[1]
        )
        if resident_cost <= stream_cost(best):
            return resident[0], resident[1], False
    return best[0], best[1], True


def choose_stage23(top_k_max: int) -> tuple[int, int]:
    """Smallest frozen ``(threads, items)`` slab that holds ``top_k_max`` entries."""
    for threads, items in sorted(_stage23_variants(), key=lambda ti: ti[0] * ti[1]):
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
    if _capability(probs.device) is None:
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
    if kmax > _slab():
        return "fallback:top_k_gt_slab"
    try:
        choose_stage1(batch, vocab)
    except ValueError:
        return "fallback:vocab_too_large"
    return "pipeline"


def _per_row_param(
    value: torch.Tensor, batch: int, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Per-request sampling parameter as a contiguous ``[batch]`` tensor of the kernel dtype.

    Same contract as the tensor form accepted by :mod:`flashinfer.sampling` (any integer or
    floating dtype, one entry per row); the kernels read ``int32`` / ``float32`` only.
    """
    if value.dim() != 1 or value.shape[0] != batch:
        raise ValueError(
            f"{name}: expected a 1D tensor of shape (batch_size,), got {tuple(value.shape)}"
        )
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value.to(dtype=dtype).contiguous()


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
    generator: Optional[torch.Generator] = None,
    philox_seed: Optional[int] = None,
    philox_offset: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    renorm_out: Optional[torch.Tensor] = None,
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    enable_pdl: bool = True,
) -> torch.Tensor:
    r"""Fused top-k-then-top-p sampling from probabilities (thread-block-cluster radix pipeline).

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
    generator: Optional[torch.Generator]
        Source of the Philox seed/offset (default CUDA generator when omitted), advanced exactly
        like :func:`flashinfer.sampling.top_k_top_p_sampling_from_probs`.
    philox_seed, philox_offset: Optional[int]
        Explicit Philox parameters (both required together); ``generator`` is then not touched.
    out: Optional[torch.Tensor]
        ``int32 [batch]`` output buffer (allocated when omitted).
    renorm_out: Optional[torch.Tensor]
        Optional ``float32 [batch, 1024]`` buffer that receives the renormalized kept
        probabilities in sorted slab order (descending probability, ascending index; zeros for
        dropped entries; only the first ``count`` entries of a row are written).  Served by the
        pipeline route only.
    workspace: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        Stage-1 slab buffers ``(values float32 [batch, 1024], indices int32 [batch, 1024],
        counts int32 [batch])``. When given, the sorted top-k slab of this call is left in them
        (aligned with ``renorm_out``); otherwise a cached per-shape workspace is used.
    enable_pdl: bool
        Launch stage 2/3 with programmatic dependent launch.

    Returns
    -------
    samples: torch.Tensor
        ``int32 [batch]`` sampled token ids.
    """
    if (philox_seed is None) != (philox_offset is None):
        raise ValueError("philox_seed and philox_offset must be given together")
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route != "pipeline":
        if renorm_out is not None:
            raise ValueError(
                f"renorm_out is served by the pipeline route only; this request falls back ({route})"
            )
        from .sampling import top_k_top_p_sampling_from_probs as _fallback

        res = _fallback(
            probs,
            top_k,
            top_p,
            filter_apply_order="top_k_first",
            deterministic=True,
            generator=generator,
            seed=philox_seed,
            offset=philox_offset,
        )
        if out is not None:
            out.copy_(res.to(torch.int32))
            return out
        return res.to(torch.int32)

    capability = _capability(probs.device)
    batch, vocab = probs.shape
    if isinstance(top_k, torch.Tensor):
        top_k = _per_row_param(top_k, batch, torch.int32, "top_k")
    if isinstance(top_p, torch.Tensor):
        top_p = _per_row_param(top_p, batch, torch.float32, "top_p")
    kmax = (
        top_k
        if isinstance(top_k, int)
        else (int(top_k_max) if top_k_max is not None else int(top_k.max().item()))
    )
    cluster, ept, stream_variant = choose_stage1(batch, vocab)
    threads, items = choose_stage23(kmax)
    slab = _slab()
    vals, idxs, cnt = (
        workspace if workspace is not None else _workspace(batch, slab, probs.device)
    )
    if out is None:
        out = torch.empty(batch, device=probs.device, dtype=torch.int32)
    if philox_seed is None:
        # Same stride as top_p_sampling_from_probs (32 reserved draws per row): a generator shared with
        # the top_k_first route stays in lockstep.
        philox_seed, philox_offset = get_seed_and_offset(
            batch * 32, generator, probs.device
        )
    if isinstance(top_k, int):
        k_arr, k_scalar, k_kind = cnt, int(top_k), _TOPK_SCALAR
    else:
        k_arr, k_scalar, k_kind = top_k, 0, _TOPK_PER_ROW
    if isinstance(top_p, (int, float)):
        p_arr, p_scalar, p_kind = probs, float(top_p), _TOPP_SCALAR
    else:
        p_arr, p_scalar, p_kind = top_p, 0.0, _TOPP_PER_ROW
    renorm = renorm_out if renorm_out is not None else vals
    module = load_cake_sampling_module()
    stream = torch.cuda.current_stream(device=probs.device).cuda_stream
    module.radix_topk(
        probs,
        k_arr,
        k_scalar,
        k_kind,
        vals,
        idxs,
        cnt,
        cluster,
        ept,
        1 if stream_variant else 0,
        stream,
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
    r"""Stage 1 alone: exact per-row top-k into a ``[batch, 1024]`` slab.

    The slab holds the first ``k`` entries of ``lexsort(-prob, index)`` (NaN/negative
    probabilities sanitized to ``+0``).  Its layout is deterministic (a pure function of the
    input) but *not* sorted; entries beyond ``count`` are undefined.  Same dispatch conditions
    as :func:`top_k_top_p_sampling_from_probs`; requests the frozen kernels cannot serve raise
    ``ValueError`` with the :func:`cake_sampling_route` reason.

    Parameters
    ----------
    probs: torch.Tensor
        ``float32 [batch, vocab]`` probabilities, contiguous.
    top_k: Union[int, torch.Tensor]
        Number of candidates kept per row (``int`` or ``int32 [batch]``), ``1 <= k <= 1024``.
    top_k_max: Optional[int]
        Upper bound of ``top_k`` when it is a tensor (avoids a device synchronization).
    out_vals: Optional[torch.Tensor]
        ``float32 [batch, 1024]`` slab values buffer (allocated when omitted).
    out_idx: Optional[torch.Tensor]
        ``int32 [batch, 1024]`` slab indices buffer (allocated when omitted).
    out_count: Optional[torch.Tensor]
        ``int32 [batch]`` per-row kept counts buffer (allocated when omitted).

    Returns
    -------
    values, indices, counts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The slab ``(values [batch, 1024], indices [batch, 1024], counts [batch])``.
    """
    route = cake_sampling_route(probs, top_k, top_k_max)
    if route != "pipeline":
        raise ValueError(f"frozen radix top-k cannot serve this request ({route})")
    capability = _capability(probs.device)
    batch, vocab = probs.shape
    slab = _slab()
    cluster, ept, stream_variant = choose_stage1(batch, vocab)
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
    load_cake_sampling_module().radix_topk(
        probs,
        k_arr,
        k_scalar,
        k_kind,
        vals,
        idxs,
        cnt,
        cluster,
        ept,
        1 if stream_variant else 0,
        stream,
    )
    return vals, idxs, cnt


__all__ = [
    "cake_sampling_route",
    "choose_stage1",
    "choose_stage23",
    "top_k_probs_to_slab",
    "top_k_top_p_sampling_from_probs",
]
