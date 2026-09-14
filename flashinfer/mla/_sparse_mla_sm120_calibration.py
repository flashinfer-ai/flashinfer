# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Analytical ``chunks_per_block`` model for SM120 sparse-MLA decode kernels.

The old per-shape sweep profiled with synthetic indices drawn from a tiny
pool, so the working set was L2-resident and the tuned cpb was distorted on
some GPUs. Instead, six fixed measurements calibrate three hardware
constants once per device (in ``autotune()`` tuning mode), and a closed-form
model picks cpb per call, with an L2-footprint guard rail for the head-tile
reuse window (see :func:`select_cpb`). Without calibrated constants the
launcher's built-in heuristic is used.

The model prices a call as the exact list-scheduling makespan of its active
blocks on ``sm_count`` SMs (see :func:`predict_time_s`): blocks of a split
are identical, splits launch in z order, and the full ``num_splits`` grid's
inactive splits early-exit at negligible cost. This replaces the previous
ceil-wave form (``ceil(G/S) * (cpb*t_c + c0) + beta*splits``), which
over-charged imbalanced mid-cpb candidates — at e.g. T=8/H=128/N=16 the
7+7+2 split finishes in one heavy round while 8+8 pays a full extra chunk
round, the sawtooth the ceil form cannot see — and whose ``beta`` merge term
was unidentifiable from the measurement grid (it fit to zero on every
family, and non-zero values poisoned the latency-regime picks).

The same tuning-mode pass also measures the decode/prefill crossover per
decode-instantiated ``(num_heads, topk)`` config (:func:`calibrate_crossover`)
and persists it as ``decode_max_tokens`` in the same JSON document (schema
version 3; only current-schema files load — files at any other version count
as absent, so their families recalibrate on the next tuning-mode pass). The
runtime decode/prefill routing in :mod:`._sparse_mla_sm120` consults it;
absent entries keep the historical decode-first policy.

Finally, tuning-mode decode-form calls refine the model's cpb pick for the
exact shape being warmed (:func:`refine_cpb`): the pick +/- a small candidate
window is timed and the measured best persists as a per-shape override. The
model remains the proposal and the fallback for every shape never warmed
(off-grid ``num_heads``, dual-cache calls, non-tuning processes).

DSV4.1 and NVFP4 skip the analytical model: each exact cache configuration
stores one measured per-bucket profile (:func:`_calibrate_dsv41`). Tuning-mode
DSV4.1 selection of an off-grid token count measures that count once
(:func:`refine_dsv41`), and the exact entry then takes priority over
nearest-up bucket interpolation.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import logging
import math
import os
import pathlib
import time
import threading
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from filelock import FileLock
from ._sparse_mla_sm120_execution import MODEL_FAMILIES, FormatValues, format_info

logger = logging.getLogger(__name__)

_BI = 64  # chunk width in candidates (BLOCK_SIZE_N)
_HPB = 16  # head tile per block

_SCHEMA_VERSION = 3
# v3: NVFP4 selections moved from phase-encoded crossover/override entries to
# per-bucket profile records, and DSV4.1 profiles may carry refined exact-T
# entries. Only current-schema files load; stale families recalibrate.

_CALIBRATION_MODELS = {
    family: model for model, family in MODEL_FAMILIES.items() if family != "glm_nsa"
}
_BYTES_PER_TOKEN = FormatValues(_CALIBRATION_MODELS, "bytes_per_token")
_D_QK = FormatValues(_CALIBRATION_MODELS, "query_dim")
_D_V = FormatValues(_CALIBRATION_MODELS, "value_dim")
_CHUNK_WIDTH = FormatValues(_CALIBRATION_MODELS, "chunk_width")

# Device-level key in the JSON payload holding the crossover table.
_DECODE_MAX_TOKENS_KEY = "decode_max_tokens"
# Device-level key holding measured per-shape cpb picks (refine_cpb).
_CPB_OVERRIDES_KEY = "cpb_overrides"
# Probe grid and decode-wins margin for crossover calibration.
_CROSSOVER_PROBED_T = (4, 8, 16, 24, 32, 48, 64)
_CROSSOVER_MARGIN = 0.95
# refine_cpb times the model pick +- this many cpb candidates and keeps the
# measured best; the window covers every model-vs-oracle gap observed in the
# kernel-bench sweep matrix (max distance 6 at mid-T wave-quantization rows).
_REFINE_WINDOW = 6


def _model_type_for_family(family: str) -> int:
    """FFI model_type for one calibration family (the decode-dsv4 FFI needs the
    explicit selector: DSV4_1 shares d_qk=512 with DSV4)."""
    from ._sparse_mla_sm120_policy import (
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_DSV4,
        _MODEL_TYPE_DSV4_1,
        _MODEL_TYPE_DOTS3_SWA,
        _MODEL_TYPE_GLM53_NOPE,
    )

    return {
        "dsv4": _MODEL_TYPE_DSV4,
        "dsv4_1": _MODEL_TYPE_DSV4_1,
        "dots3_swa": _MODEL_TYPE_DOTS3_SWA,
        "glm53_nope": _MODEL_TYPE_GLM53_NOPE,
        # dsv3_2 and glm_nsa share the dsv3_2-kernel call path, which re-maps
        # glm_nsa at the builder; calibrate() only ever passes dsv3_2 here.
    }.get(family, _MODEL_TYPE_DSV3_2)


# (num_tokens, num_heads, topk, chunks_per_block); see calibrate().
_MEASUREMENTS = (
    (64, 128, 128, 1),
    (64, 128, 1024, 8),
    (64, 128, 1024, 1),
    (64, 128, 512, 1),
    (1, 8, 1024, 16),
    # Second saturated mid-size point at a different wave count: keeps the
    # streaming/overhead directions non-collinear for the LM fit.
    (32, 128, 512, 1),
)
# glm53_nope decode is instantiated at topk=2176 only (N=34 chunks, fixed),
# so M1/M2 isolate the streaming term by varying cpb (17 vs 33) at identical
# token/head counts instead of varying N. The wide cpb gap keeps the signal
# well above min-of-iters timing noise. M5 is the latency point.
_MEASUREMENTS_GLM53_NOPE = (
    (64, 64, 2176, 17),
    (64, 64, 2176, 33),
    (64, 64, 2176, 1),
    (64, 32, 2176, 1),
    (1, 32, 2176, 34),
    # Half the waves of M3 at a different split count, keeping the
    # streaming/overhead directions non-collinear for the LM fit.
    (32, 64, 2176, 2),
)
# DOTS3_SWA decode is instantiated at topk=576 only (N=18 chunks at the 32-wide
# tile, fixed). Its per-block fixed cost dwarfs the marginal per-chunk cost, so
# a narrow same-shape cpb pair (like glm53_nope's) lands under timing noise;
# M1 (cpb=1) vs M2 (cpb=17) instead spans the full cpb range. M5 is the
# latency point. The num_chunks basis is topk (the launched split grid), not
# the 513-token window: the kernel clamps the scan to WINDOW inside each block
# but the grid — and therefore the makespan structure the model prices — is
# sized from TOPK.
_MEASUREMENTS_DOTS3_SWA = (
    (64, 64, 576, 1),
    (64, 64, 576, 17),
    (64, 32, 576, 1),
    (32, 64, 576, 2),
    (1, 32, 576, 18),
    # Same shape as M2 at a different cpb and split count, keeping the
    # streaming/overhead directions non-collinear for the LM fit.
    (64, 64, 576, 9),
)

_POOL_BYTES_TARGET = 2 << 30  # >> L2, so calibration traffic is DRAM-faithful
_POOL_BYTES_MIN = 512 << 20
_WARMUP_ITERS = 3
# Rotation grows with device L2 and is bounded by index memory, not call count.
_TIMED_BATCHES = 5
_MIN_BATCH_CALLS = 8


class CalibrationError(RuntimeError):
    """Calibration measurements were unusable (OOM or implausible constants)."""


def _target_calibration(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        device = bound.arguments.get("device")
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else torch.device(device)
        )
        bound.arguments["device"] = device
        with (
            torch.cuda.device(device)
            if device.type == "cuda"
            else contextlib.nullcontext()
        ):
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise CalibrationError(
                    "calibration must not run under CUDA graph capture"
                )
            return function(*bound.args, **bound.kwargs)

    return wrapped


def _target_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True)
class CpbConstants:
    """Calibrated hardware constants for one (device, kernel family).

    Attributes
    ----------
    inv_bw : float
        s/byte; inverse aggregate DRAM bandwidth.
    inv_rsm : float
        s/byte; inverse single-SM streaming rate (latency-bound regime).
    c0 : float
        s; fixed per-block overhead (Q load, epilogue).
    sm_count : int
        Device SM count at calibration time.
    bytes_per_chunk : int
        Bytes per family-specific chunk (``chunk_width * bytes_per_token``).
    l2_cache_bytes : int
        Device L2 size; bounds the head-tile reuse window in
        :func:`select_cpb`. ``0`` disables the guard rail. Read from device
        properties, not measured.
    """

    inv_bw: float
    inv_rsm: float
    c0: float
    sm_count: int
    bytes_per_chunk: int
    l2_cache_bytes: int


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _num_chunks(topk: int, extra_topk: int, chunk_width: int = _BI) -> int:
    return _ceil_div(topk, chunk_width) + (
        _ceil_div(extra_topk, chunk_width) if extra_topk else 0
    )


def _graham_makespan(sm_count: int, batches: tuple[tuple[int, float], ...]) -> float:
    """Exact Graham list-scheduling makespan for ``sm_count`` SMs.

    ``batches`` are ``(job_count, job_duration)`` pairs in queue order, jobs
    identical within a batch; a freed SM always takes the next queued block
    (the GPU work distributor's behavior). Compressed per level: SMs tied at
    the minimum availability rise in lockstep, so each iteration raises the
    whole min pool instead of stepping job by job. Iterations are bounded by
    the level-merge count (<= ~2 per batch here: the batches start from at
    most two distinct availability levels).
    """
    avail = np.zeros(sm_count)
    for k, d in batches:
        if k <= 0 or d <= 0:
            continue
        while k > 0:
            avail.sort()
            v = avail[0]
            c = int(np.searchsorted(avail, v, side="right"))
            nxt = avail[c] if c < sm_count else np.inf
            t_jobs = -(-k // c)  # rounds to exhaust the queue on this pool
            t_reach = (
                np.inf if np.isinf(nxt) else max(1, math.ceil((nxt - v) / d - 1e-9))
            )
            if t_jobs <= t_reach:
                # Jobs run out while the pool is still the minimum: spread
                # round-robin; the first k % c machines take one extra.
                q, rem = divmod(k, c)
                avail[:c] += q * d
                avail[:rem] += d
                k = 0
            else:
                avail[:c] += int(t_reach) * d
                k -= c * int(t_reach)
    return float(avail.max())


def predict_time_s(
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    cpb: int,
    c: CpbConstants,
    chunk_width: int = _BI,
) -> float:
    """Predicted wall time (seconds) of one decode call at ``cpb``.

    The launcher always fires the full ``T * H_b * num_splits`` grid; split z
    owns chunks ``[z*cpb, z*cpb+cpb)`` and splits past ``ceil(N/cpb)``
    early-exit. Blocks of one split are identical and launch in z order, so
    the stage-1 makespan is the Graham list schedule of the active blocks on
    ``sm_count`` SMs: ``m*(s-1)`` full blocks of ``cpb`` chunks followed by
    ``m`` tail blocks of the short last split, each block paying the fixed
    per-block overhead ``c0`` plus per-chunk time ``t_c``. ``t_c`` is the
    larger of the bandwidth-bound term (``g`` concurrent blocks share DRAM)
    and the single-SM latency-bound term. Early-exit blocks only write their
    LSE sentinel and retire; their scheduling churn is neglected (charging
    them ``c0`` each measurably over-predicts mid-cpb candidates).
    """
    h_b = _ceil_div(num_heads, _HPB)
    n = _num_chunks(topk, extra_topk, chunk_width)
    m = num_tokens * h_b
    s = _ceil_div(n, cpb)
    g = m * s
    t_c = max(
        c.bytes_per_chunk * min(g, c.sm_count) * c.inv_bw,
        c.bytes_per_chunk * c.inv_rsm,
    )
    c_last = n - (s - 1) * cpb
    return _graham_makespan(
        c.sm_count,
        ((m * (s - 1), cpb * t_c + c.c0), (m, c_last * t_c + c.c0)),
    )


def select_cpb(
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    c: CpbConstants,
    chunk_width: int = _BI,
) -> int:
    """Argmin of :func:`predict_time_s` over cpb in 1..N; ties prefer larger cpb.

    Guard rail: each token's candidate set is re-read once per ``_HPB``-wide
    head tile, and the re-reads hit L2 only while the concurrent streaming
    footprint ``min(G, S) * cpb * W`` fits in L2. Beyond that, measured per-
    chunk cost degrades ~45% (L2 hit 87%->70%, DRAM re-reads 2x compulsory at
    the N=50 dual-cache shape), which the closed-form terms do not capture —
    so candidates past the L2 footprint are excluded. Falls back to the
    unconstrained argmin if nothing fits (e.g. unknown L2 size).
    """
    n = _num_chunks(topk, extra_topk, chunk_width)
    h_b = _ceil_div(num_heads, _HPB)
    best_cpb, best_t = 1, float("inf")
    for cpb in range(1, n + 1):
        t = predict_time_s(num_tokens, num_heads, topk, extra_topk, cpb, c, chunk_width)
        if t <= best_t:
            best_cpb, best_t = cpb, t
    if not c.l2_cache_bytes:
        return best_cpb
    capped_cpb, capped_t = 0, float("inf")
    for cpb in range(1, n + 1):
        g = num_tokens * h_b * _ceil_div(n, cpb)
        if min(g, c.sm_count) * cpb * c.bytes_per_chunk > c.l2_cache_bytes:
            continue
        t = predict_time_s(num_tokens, num_heads, topk, extra_topk, cpb, c, chunk_width)
        if t <= capped_t:
            capped_cpb, capped_t = cpb, t
    return capped_cpb or best_cpb


_SAMPLE_CHUNK_BYTES = 8 << 20


def _device_l2(device: torch.device) -> int:
    l2 = int(getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0) or 0)
    if l2 <= 0:
        raise CalibrationError("calibration requires a measured device L2 capacity")
    return l2


def _check_pool_capacity(device: torch.device, pool_bytes: int) -> None:
    if pool_bytes <= 2 * _device_l2(device):
        raise CalibrationError(
            "calibration KV pool must exceed twice the device L2 capacity"
        )


def _initialize_fp8_pool(cache: torch.Tensor, family: str, page_size: int) -> None:
    """Fill finite random encoded rows in bounded chunks, outside timed calls."""
    facts = format_info(_CALIBRATION_MODELS[family])
    nope, rope = facts["nope_dim"], facts["rope_dim"]
    count, scale_bytes = facts["num_scales"], facts["scale_bytes"]
    generator = torch.Generator(device=cache.device).manual_seed(0)
    pages_per_chunk = max(1, _SAMPLE_CHUNK_BYTES // (page_size * (nope + rope) * 8))
    for start in range(0, cache.shape[0], pages_per_chunk):
        block = cache[start : start + pages_per_chunk].view(
            -1, page_size * facts["bytes_per_token"]
        )
        block.zero_()
        pages = block.shape[0]
        if facts["inline_scale"]:
            data = block.view(pages, page_size, facts["bytes_per_token"])
            scales = data[..., nope : nope + scale_bytes]
        else:
            offset = page_size * facts["data_bytes"]
            data = block[:, :offset].view(pages, page_size, facts["data_bytes"])
            scales = block[:, offset:].view(pages, page_size, scale_bytes)
        values = torch.randn(
            (pages, page_size, nope), device=cache.device, generator=generator
        ).clamp_(-4, 4)
        data[..., :nope].copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
        exponents = torch.randint(
            -3, 1, (pages, page_size, count), device=cache.device, generator=generator
        )
        if facts["inline_scale"]:
            scales.copy_(torch.exp2(exponents.float()).view(torch.uint8))
        else:
            scales[..., :count].copy_((exponents + 127).to(torch.uint8))
        if rope:
            values = (
                torch.randn(
                    (pages, page_size, rope), device=cache.device, generator=generator
                )
                .clamp_(-4, 4)
                .to(torch.bfloat16)
            )
            offset = facts["rope_offset"]
            data[..., offset : offset + rope * 2].copy_(values.view(torch.uint8))


def _allocate_kv_pool(family: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """Allocate and initialize a finite 64-token-page pool, halving on OOM."""
    w = 64 * _BYTES_PER_TOKEN[family]
    pool_bytes = _POOL_BYTES_TARGET
    while True:
        kv_cache = None
        _check_pool_capacity(device, pool_bytes // w * w)
        try:
            kv_cache = torch.empty(pool_bytes // w, w, dtype=torch.uint8, device=device)
            _initialize_fp8_pool(kv_cache, family, 64)
            return kv_cache, kv_cache.shape[0] * 64
        except torch.cuda.OutOfMemoryError:
            del kv_cache
            if pool_bytes <= _POOL_BYTES_MIN:
                raise CalibrationError(
                    f"cannot allocate and initialize a >= {_POOL_BYTES_MIN >> 20} MiB KV pool "
                    "for sparse-MLA cpb calibration"
                ) from None
            pool_bytes //= 2
            with (
                torch.cuda.device(device)
                if device.type == "cuda"
                else contextlib.nullcontext()
            ):
                torch.cuda.empty_cache()


def calibration_batch_count(
    num_tokens: int,
    total_topk: int,
    bytes_per_token: int,
    device: torch.device,
    *,
    max_batch_calls: int | None = None,
) -> int:
    """Number of rotating index sets needed to evict one call's KV footprint."""
    l2 = _device_l2(device)
    footprint = max(1, num_tokens * total_topk * bytes_per_token)
    count = max(_MIN_BATCH_CALLS, l2 // footprint + 2)
    if max_batch_calls is not None and count > max_batch_calls:
        raise CalibrationError("calibration batch cap cannot satisfy L2 reuse distance")
    if count * num_tokens * total_topk * 4 > 64 << 20:
        raise CalibrationError("calibration rotation exceeds 64 MiB index budget")
    return count


def time_calibration_calls(
    call: Callable[..., None],
    argument_sets: list[tuple[Any, ...]],
    device: torch.device,
) -> float:
    """Return steady-state seconds/call on the target device's current stream."""
    if not argument_sets:
        raise ValueError("calibration requires at least one argument set")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise CalibrationError(
                "calibration timing must not run under CUDA graph capture"
            )
        stream = torch.cuda.current_stream(device)
        for i in range(_WARMUP_ITERS):
            call(*argument_sets[i % len(argument_sets)])
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = float("inf")
        for _ in range(_TIMED_BATCHES):
            start.record(stream)
            for args in argument_sets:
                call(*args)
            end.record(stream)
            torch.cuda.synchronize(device)
            best = min(best, start.elapsed_time(end) / 1e3 / len(argument_sets))
        return best


def _time_call_fresh_indices(
    call: Callable[[torch.Tensor], None],
    num_tokens: int,
    topk: int,
    num_slots: int,
    device: torch.device,
    bytes_per_token: int,
) -> float:
    """Steady-state time (seconds) of one call: min over ``_TIMED_BATCHES``
    batches of the per-call mean, each batch enqueuing its calls back-to-back
    under a single sync.

    The queued batch keeps the GPU busy across call boundaries, so per-call
    launch latency overlaps execution — the regime production runs in (a
    graph-replayed decode step pays no per-kernel launch gap). A per-rep
    sync instead exposes one launch gap per call, which distorted small-T
    measurements by a fixed ~7us.

    L2 fidelity: every call in a batch gathers under a different pre-drawn
    full-pool uniform index set. Consecutive sets overlap by
    ~footprint/pool, and the batch length is sized so a set's reuse distance
    ((K-1) calls x per-call gather footprint) exceeds L2 — reusing one index
    set across reps makes the working set L2-resident after warmup and
    understates the DRAM-bound steady state (this tainted earlier calibration
    rounds: decode looked artificially fast). The draws run before timing.
    """
    # A set recurs after K-1 intervening calls. Unknown L2 capacity or an
    # insufficient batch cap is a measurement failure, not a warm-cache fallback.
    k = calibration_batch_count(
        num_tokens,
        topk,
        bytes_per_token,
        device,
    )
    generator = torch.Generator(device=device).manual_seed(0)
    sets = [
        (
            torch.randint(
                0,
                num_slots,
                (num_tokens, topk),
                dtype=torch.int32,
                device=device,
                generator=generator,
            ),
        )
        for _ in range(k)
    ]
    return time_calibration_calls(call, sets, device)


def _make_decode_call_builder(
    module: Any, family: str, device: torch.device, kv_cache: torch.Tensor
) -> Callable[[int, int, int, int, int], Callable[[torch.Tensor], None]]:
    """Decode-call constructor shared by calibrate() and calibrate_crossover().

    Returns a builder mapping ``(num_tokens, num_heads, topk, model_type,
    cpb)`` to a ``call(indices) -> None`` closure that drives the family's
    decode kernel over ``kv_cache``, so the two calibration passes' FFI
    argument lists cannot drift apart. ``model_type`` only reaches the
    dsv3_2-kernel families (where one family hosts several model types); the
    decode-dsv4 branch derives it from the family itself (DSV4_1 shares
    d_qk=512 with DSV4, so width alone cannot resolve it).
    """
    d_qk = _D_QK[family]
    d_v = _D_V[family]
    bi = _CHUNK_WIDTH[family]
    sm_scale = d_qk**-0.5
    from ._sparse_mla_sm120_policy import _decode_scratch_heads

    def build(
        num_tokens: int, num_heads: int, topk: int, model_type: int, cpb: int
    ) -> Callable[[torch.Tensor], None]:
        num_splits = _ceil_div(topk, bi)
        q = (
            (
                torch.randn(
                    num_tokens,
                    num_heads,
                    d_qk,
                    device=device,
                    dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(1),
                )
                / 10.0
            )
            .clamp(-1, 1)
            .to(torch.bfloat16)
        )
        mid_out = torch.empty(
            num_tokens,
            _decode_scratch_heads(num_heads),
            num_splits,
            d_v,
            dtype=torch.bfloat16,
            device=device,
        )
        mid_lse = torch.empty(
            num_tokens,
            _decode_scratch_heads(num_heads),
            num_splits,
            dtype=torch.float32,
            device=device,
        )
        output = torch.empty(
            num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
        )
        out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
        if family in ("dsv4", "dots3_swa", "dsv4_1"):

            def call(indices: torch.Tensor) -> None:
                module.sparse_mla_sm120_decode_dsv4(
                    q,
                    kv_cache,
                    indices,
                    mid_out,
                    mid_lse,
                    output,
                    out_lse,
                    num_splits,
                    sm_scale,
                    None,
                    None,
                    None,
                    None,
                    None,
                    _model_type_for_family(family),
                    cpb,
                    False,  # extra_fp4: calibration times single-cache shapes
                )

        else:

            def call(indices: torch.Tensor) -> None:
                module.sparse_mla_sm120_decode_dsv3_2(
                    q,
                    kv_cache,
                    indices,
                    mid_out,
                    mid_lse,
                    output,
                    out_lse,
                    num_splits,
                    sm_scale,
                    None,
                    None,
                    model_type,
                    cpb,
                )

        return call

    return build


@_target_calibration
def calibrate(
    module_getter: Callable[[], Any], family: str, device: torch.device
) -> CpbConstants:
    """Calibrate the cpb model constants for ``family`` on ``device``.

    Drives the real decode kernel over a ~2 GiB KV pool (halved on OOM down
    to 512 MiB), timing queued batches over rotating fresh full-pool uniform
    index sets (:func:`_time_call_fresh_indices`) so the measured working
    set stays DRAM-resident and launch latency stays off the clock, then fits
    the three constants to six fixed shapes by Levenberg-Marquardt on
    relative residuals.
    ``module_getter`` returns the loaded TVM-FFI kernel module.
    """
    if family not in _BYTES_PER_TOKEN:
        raise ValueError(f"unknown sparse-MLA family {family!r}")
    if torch.cuda.is_current_stream_capturing():
        # Capture-fatal: calibration synchronizes, empties the cache, and
        # allocates GiB-scale pools.
        raise CalibrationError(
            "sparse-MLA SM120 calibration must not run under CUDA graph capture"
        )

    device = torch.device(device)
    props = torch.cuda.get_device_properties(device)
    sm_count = int(props.multi_processor_count)
    l2_cache_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    # On integrated-memory devices (GB10) the L2 is shared with the CPU fabric
    # and the effective streaming window measures ~half the reported size.
    if int(getattr(props, "is_integrated", 0) or 0):
        l2_cache_bytes //= 2
    bi = _CHUNK_WIDTH[family]
    w = bi * _BYTES_PER_TOKEN[family]
    # Families whose decode is instantiated at a single topk have a fixed N,
    # so the bandwidth term is identified from a cpb pair instead of an N pair.
    _CPB_PAIR_MEASUREMENTS = {
        "glm53_nope": _MEASUREMENTS_GLM53_NOPE,
        "dots3_swa": _MEASUREMENTS_DOTS3_SWA,
    }
    measurements = _CPB_PAIR_MEASUREMENTS.get(family, _MEASUREMENTS)
    model_type = _model_type_for_family(family)

    kv_cache, num_slots = _allocate_kv_pool(family, device)

    module = module_getter()
    build_call = _make_decode_call_builder(module, family, device, kv_cache)

    def measure(num_tokens: int, num_heads: int, topk: int, cpb: int) -> float:
        call = build_call(num_tokens, num_heads, topk, model_type, cpb)
        return _time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, _BYTES_PER_TOKEN[family]
        )

    t = [measure(*m) for m in measurements]

    # Fit (inv_bw, inv_rsm, c0) to the six points by Levenberg-Marquardt on
    # relative residuals, in log space (positivity by construction). The
    # scheduling-makespan model is piecewise-linear in the constants, so a
    # closed-form solve does not exist; LM converges in <10 iterations from
    # the fixed inits below on every family. When the bandwidth term is
    # shadowed by the single-SM latency floor at all six points, inv_bw is
    # unidentifiable and stays near its init — harmless, because predictions
    # are then insensitive to it.
    def predict_with(x: np.ndarray, m: tuple[int, int, int, int]) -> float:
        num_tokens, num_heads, topk, cpb = m
        return predict_time_s(
            num_tokens,
            num_heads,
            topk,
            0,
            cpb,
            CpbConstants(
                inv_bw=float(x[0]),
                inv_rsm=float(x[1]),
                c0=float(x[2]),
                sm_count=sm_count,
                bytes_per_chunk=w,
                l2_cache_bytes=l2_cache_bytes,
            ),
            chunk_width=bi,
        )

    def resid(theta: np.ndarray) -> np.ndarray:
        x = np.exp(theta)
        return np.array(
            [
                (predict_with(x, m) - t_i) / t_i
                for m, t_i in zip(measurements, t, strict=True)
            ]
        )

    theta = np.log(np.array([5e-13, 1.5e-10, 6e-6]))
    r = resid(theta)
    cost = 0.5 * float(r @ r)
    lam = 1e-3
    for _ in range(64):
        jac = np.empty((len(measurements), 3))
        for j in range(3):
            h = 1e-4
            theta_p = theta.copy()
            theta_p[j] += h
            theta_m = theta.copy()
            theta_m[j] -= h
            jac[:, j] = (resid(theta_p) - resid(theta_m)) / (2 * h)
        grad = jac.T @ r
        step_matrix = jac.T @ jac + lam * np.diag(
            np.maximum(np.diag(jac.T @ jac), 1e-24)
        )
        try:
            delta = np.linalg.solve(step_matrix, -grad)
        except np.linalg.LinAlgError:
            break
        theta_new = theta + delta
        if not np.all(np.isfinite(theta_new)):
            break
        r_new = resid(theta_new)
        cost_new = 0.5 * float(r_new @ r_new)
        if np.isfinite(cost_new) and cost_new < cost:
            theta, r, cost = theta_new, r_new, cost_new
            lam = max(lam / 4.0, 1e-12)
            if np.max(np.abs(delta)) < 1e-6:
                break
        else:
            lam *= 8.0
            if lam > 1e12:
                break

    inv_bw, inv_rsm, c0 = (float(v) for v in np.exp(theta))
    rel_rms = float(np.sqrt(2.0 * cost / len(measurements)))
    if (
        not all(np.isfinite([inv_bw, inv_rsm, c0]))
        or inv_bw <= 0
        or inv_rsm <= 0
        or c0 <= 0
        or rel_rms > 0.25
    ):
        raise CalibrationError(
            f"implausible cpb calibration constants for {family}: inv_bw={inv_bw}, "
            f"inv_rsm={inv_rsm}, c0={c0} (relative rms residual {rel_rms:.3f})"
        )
    return CpbConstants(
        inv_bw=inv_bw,
        inv_rsm=inv_rsm,
        c0=c0,
        sm_count=sm_count,
        bytes_per_chunk=w,
        l2_cache_bytes=l2_cache_bytes,
    )


@_target_calibration
def calibrate_crossover(
    module: Any,
    device: torch.device,
    family: str,
    c: CpbConstants,
    grid_override: Optional[list[tuple[int, int]]] = None,
) -> dict[str, int]:
    """Measure the decode/prefill crossover for the decode-instantiated
    configs of ``family`` on ``device``.

    For every ``(num_heads, topk)`` pair on the family's calibration grid (the
    dedicated-H corner plus the power-of-2 head counts; runtime-H shapes off
    the grid keep the decode-first default) — or on ``grid_override`` when
    given (the public calibration API's arbitrary-shape entries) — both paths
    are timed at
    each probed T with the DRAM-faithful protocol of
    :func:`_time_call_fresh_indices`: the decode kernel runs with the model's
    ``select_cpb`` pick; the prefill orchestrator runs with
    ``prefill_impl=auto`` variant choice (swapAB preferred where
    instantiated). Family
    ``"dsv3_2"`` covers both the ``dsv3_2`` and ``glm_nsa`` key spaces because
    the scale format changes prefill speed; the decode kernel is timed with
    the matching ``model_type`` too. ``"glm53_nope"`` covers its own key
    space at topk=2176, ``"dots3_swa"`` its own at topk=576. A config the
    prefill envelope does not serve (e.g. an off-envelope ``num_heads``, or a
    ``topk`` that is not a whole number of 64-wide index tiles) records
    ``decode_max_tokens=64``.

    Returns a flat ``{"<family>|<num_heads>|<topk>": decode_max_tokens}``
    table: the largest probed T with ``decode_time <= 0.95 * prefill_time``,
    ``0`` when decode never wins, ``64`` when it wins everywhere probed.
    """
    from ._sparse_mla_sm120_policy import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
        _DECODE_DSV4_1_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
        _PREFILL_IMPL_AUTO,
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_DSV4,
        _MODEL_TYPE_DSV4_1,
        _MODEL_TYPE_GLM_NSA,
        _MODEL_TYPE_GLM53_NOPE,
        _MODEL_TYPE_DOTS3_SWA,
        prefill_variant,
    )

    device = torch.device(device)
    if torch.cuda.is_current_stream_capturing():
        # Same capture-fatal profile as calibrate().
        raise CalibrationError(
            "sparse-MLA SM120 crossover calibration must not run under CUDA "
            "graph capture"
        )
    grid: Optional[list[tuple[int, int]]] = (
        sorted(grid_override) if grid_override is not None else None
    )
    if family == "dsv4":
        # (key prefix, calibration grid, FFI model_type)
        spaces = [
            ("dsv4", grid or sorted(_DECODE_DSV4_CALIBRATION_GRID), _MODEL_TYPE_DSV4)
        ]
    elif family == "dsv4_1":
        spaces = [
            (
                "dsv4_1",
                grid or sorted(_DECODE_DSV4_1_CALIBRATION_GRID),
                _MODEL_TYPE_DSV4_1,
            )
        ]
    elif family == "dsv3_2":
        pairs = grid or sorted(_DECODE_DSV3_2_CALIBRATION_GRID)
        spaces = [
            ("dsv3_2", pairs, _MODEL_TYPE_DSV3_2),
            ("glm_nsa", pairs, _MODEL_TYPE_GLM_NSA),
        ]
    elif family == "glm53_nope":
        spaces = [
            (
                "glm53_nope",
                grid or sorted(_DECODE_GLM53_NOPE_CALIBRATION_GRID),
                _MODEL_TYPE_GLM53_NOPE,
            )
        ]
    elif family == "dots3_swa":
        spaces = [
            (
                "dots3_swa",
                grid or sorted(_DECODE_DOTS3_SWA_CALIBRATION_GRID),
                _MODEL_TYPE_DOTS3_SWA,
            )
        ]
    else:
        raise ValueError(f"unknown sparse-MLA family {family!r}")

    d_qk = _D_QK[family]
    d_v = _D_V[family]
    bi = _CHUNK_WIDTH[family]
    sm_scale = d_qk**-0.5
    kv_cache, num_slots = _allocate_kv_pool(family, device)
    build_call = _make_decode_call_builder(module, family, device, kv_cache)

    def time_decode(
        num_tokens: int, num_heads: int, topk: int, model_type: int
    ) -> float:
        cpb = select_cpb(num_tokens, num_heads, topk, 0, c, chunk_width=bi)
        call = build_call(num_tokens, num_heads, topk, model_type, cpb)
        return _time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, _BYTES_PER_TOKEN[family]
        )

    def time_prefill(
        num_tokens: int, num_heads: int, topk: int, model_type: int
    ) -> float:
        # The prefill variant the auto policy would pick; None when the
        # prefill envelope does not serve the shape (e.g. an off-envelope
        # num_heads, or a ragged topk).
        variant = prefill_variant(
            model_type, num_heads, topk, 64, False, _PREFILL_IMPL_AUTO
        )
        if variant is None:
            return float("inf")
        q = (
            (
                torch.randn(
                    num_tokens,
                    num_heads,
                    d_qk,
                    device=device,
                    dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(1),
                )
                / 10.0
            )
            .clamp(-1, 1)
            .to(torch.bfloat16)
        )
        output = torch.empty(
            num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
        )
        out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

        def call(indices: torch.Tensor) -> None:
            module.sparse_mla_sm120_paged_attention(
                q,
                kv_cache,
                indices,
                output,
                out_lse,
                sm_scale,
                model_type,
                int(variant),
                None,
                None,
                None,
                None,
                None,
                False,
            )

        return _time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, _BYTES_PER_TOKEN[family]
        )

    table: dict[str, int] = {}
    for prefix, pairs, model_type in spaces:
        for num_heads, topk in pairs:
            best = 0
            for num_tokens in _CROSSOVER_PROBED_T:
                t_dec = time_decode(num_tokens, num_heads, topk, model_type)
                t_pre = time_prefill(num_tokens, num_heads, topk, model_type)
                if t_dec <= _CROSSOVER_MARGIN * t_pre:
                    best = num_tokens
            table[f"{prefix}|{num_heads}|{topk}"] = best
    return table


@_target_calibration
def refine_cpb(
    module_getter: Callable[[], Any],
    family: str,
    device: torch.device,
    c: CpbConstants,
    num_tokens: int,
    num_heads: int,
    topk: int,
) -> int:
    """Measured best cpb around the model pick for one single-cache shape.

    Times ``select_cpb``'s pick +/- ``_REFINE_WINDOW`` candidates (clamped to
    1..N) with the calibration timing protocol and returns the measured
    argmin. The analytical model proposes; this closes its residual pick
    error — largest at mid-T wave-quantization shapes (up to 1.3x kernel
    time in kernel-bench sweeps) — for exactly the shapes the caller warms
    up. Dual-cache (extra_topk > 0) shapes stay on the model: their measured
    pick error stays within ~6%.
    """
    if family not in _BYTES_PER_TOKEN:
        raise ValueError(f"unknown sparse-MLA family {family!r}")
    if torch.cuda.is_current_stream_capturing():
        # Same contract as calibrate(): synchronizes and allocates GiB pools.
        raise CalibrationError(
            "sparse-MLA SM120 cpb refinement must not run under CUDA graph capture"
        )
    bi = _CHUNK_WIDTH[family]
    n = _ceil_div(topk, bi)
    center = select_cpb(num_tokens, num_heads, topk, 0, c, chunk_width=bi)
    kv_cache, num_slots = _allocate_kv_pool(family, device)
    build_call = _make_decode_call_builder(module_getter(), family, device, kv_cache)
    model_type = _model_type_for_family(family)
    best_cpb, best_t = center, float("inf")
    lo = max(1, center - _REFINE_WINDOW)
    hi = min(n, center + _REFINE_WINDOW)
    for cpb in range(lo, hi + 1):
        call = build_call(num_tokens, num_heads, topk, model_type, cpb)
        t = _time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, _BYTES_PER_TOKEN[family]
        )
        if t < best_t:
            best_cpb, best_t = cpb, t
    return best_cpb


def default_cache_path() -> pathlib.Path:
    """Default disk path for the calibrated cpb constants.

    Override via the ``FLASHINFER_AUTOTUNE_DIR`` env var.
    """
    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = pathlib.Path(override)
    else:
        from ..jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    return base / "sparse_mla_sm120_cpb.json"


_constants: dict[tuple[str, str], CpbConstants] = {}
_failed: set[tuple[str, str]] = set()
# dev_key -> flat {"<family>|<num_heads>|<topk>": decode_max_tokens} table.
_crossover: dict[str, dict[str, int]] = {}
_crossover_failed: set[tuple[str, str]] = set()
# dev_key -> flat {"<family>|<num_heads>|<topk>|<num_tokens>": cpb} table of
# per-shape measured picks written by refine_cpb at tuning time.
_cpb_overrides: dict[str, dict[str, int]] = {}
# Bumped whenever new constants enter the process (disk load or save), so
# select_cpb memoization keyed on it never serves stale picks.
_constants_version: int = 0
_device_key_cache: dict[torch.device, str] = {}


def _device_key(device: torch.device) -> str:
    device = torch.device(device)
    if device.index is None:
        # Resolve against the CURRENT device on every call: caching under the
        # unindexed object would keep serving the first-resolved index after a
        # set_device switch.
        device = torch.device(device.type, torch.cuda.current_device())
    key = _device_key_cache.get(device)
    if key is None:
        name = torch.cuda.get_device_properties(device.index).name
        key = f"{device.index}:{name}"
        _device_key_cache[device] = key
    return key


def _parse_payload_devices(devices: dict) -> tuple[dict, dict, dict]:
    """Parse a cache document's ``devices`` mapping into process-cache entries.

    Raises on malformed entries; callers must publish the returned dicts
    atomically (a mid-document failure must not publish a prefix).
    """
    new_constants: dict = {}
    new_crossover: dict = {}
    new_overrides: dict = {}
    for dev_key, families in devices.items():
        if not isinstance(families, dict):
            continue
        for family, raw in families.items():
            if family == "profiles":
                canonical = {str(t) for t in _PROFILE_T}
                for profile in raw.values():
                    buckets = profile["buckets"]
                    # Refined exact-T entries extend the canonical grid.
                    if not canonical <= set(buckets):
                        raise ValueError("incomplete calibration profile")
                    variants = (
                        ("decode_splitk", "prefill_streaming")
                        if profile.get("request", {}).get("family") == "dsv4_nvfp4"
                        else (0, 1)
                    )
                    for bucket in buckets.values():
                        if bucket["variant"] not in variants or bucket["cpb"] < 1:
                            raise ValueError("invalid profile selection")
                        for name in ("decode_s", "prefill_s"):
                            latency = bucket.get(name)
                            if latency is not None and (
                                not math.isfinite(latency) or latency <= 0
                            ):
                                raise ValueError("invalid profile latency")
                continue
            if family in (_DECODE_MAX_TOKENS_KEY, _CPB_OVERRIDES_KEY):
                minimum = 1 if family == _CPB_OVERRIDES_KEY else 0
                if not isinstance(raw, dict) or any(
                    type(v) is not int or v < minimum for v in raw.values()
                ):
                    raise ValueError("invalid calibration selection")
                target = new_overrides if minimum else new_crossover
                target[dev_key] = dict(raw)
                continue
            value = CpbConstants(**raw)
            if (
                not all(math.isfinite(v) and v >= 0 for v in asdict(value).values())
                or value.sm_count <= 0
                or value.bytes_per_chunk <= 0
            ):
                raise ValueError("invalid calibration constants")
            new_constants[(dev_key, family)] = value
    return new_constants, new_crossover, new_overrides


_store_states: dict = {}
_active_scope = None
_store_lock = threading.RLock()


def _file_token(path: pathlib.Path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino
    except FileNotFoundError:
        return None


def _activate_store():
    global _active_scope, _constants, _crossover, _cpb_overrides
    global _failed, _crossover_failed, _constants_version
    path = default_cache_path().expanduser().resolve()
    # One store state per cache file: every family lives in the same payload,
    # so splitting scope by family would only bounce _constants_version when a
    # run alternates families.
    scope = str(path)
    state = _store_states.setdefault(
        scope,
        {
            "token": object(),
            "disk": {},
            "overlay": [],
            "constants": {},
            "crossover": {},
            "overrides": {},
            "failed": set(),
            "cross_failed": set(),
        },
    )
    if _active_scope != scope:
        _active_scope = scope
        _constants_version += 1
    _constants, _crossover, _cpb_overrides = (
        state["constants"],
        state["crossover"],
        state["overrides"],
    )
    _failed, _crossover_failed = state["failed"], state["cross_failed"]
    return path, state


def _apply_unit(
    devices: dict,
    dev_key: str,
    constants: dict,
    crossover: dict,
    overrides: dict,
    prefixes: tuple[str, ...],
    profiles: dict | None = None,
    profile_buckets: dict | None = None,
) -> None:
    dev = devices.setdefault(dev_key, {})
    if prefixes:
        for section in (_DECODE_MAX_TOKENS_KEY, _CPB_OVERRIDES_KEY):
            dev[section] = {
                k: v
                for k, v in dev.get(section, {}).items()
                if not k.startswith(prefixes)
            }
    dev.update(constants)
    for section, entries in (
        (_DECODE_MAX_TOKENS_KEY, crossover),
        (_CPB_OVERRIDES_KEY, overrides),
    ):
        dev.setdefault(section, {}).update(entries)
    if profiles:
        stored = dev.setdefault("profiles", {})
        for key, profile in profiles.items():
            if profile_buckets and key in profile_buckets and key in stored:
                for tokens, entry in profile_buckets[key].items():
                    stored[key]["buckets"].setdefault(tokens, entry)
            else:
                stored[key] = json.loads(json.dumps(profile))


def _replay_unit(devices: dict, unit) -> None:
    """Apply one stored unit, re-evaluating conditional replacement.

    A conditional unit (recorded by replace_family_if_absent) is destructive
    only while the family is still absent; once any publisher has landed the
    family's constants, the unit merges instead of discarding entries."""
    (
        dev_key,
        constants,
        crossover,
        overrides,
        prefixes,
        profiles,
        profile_buckets,
        conditional,
    ) = unit
    if conditional and any(f in devices.get(dev_key, {}) for f in conditional):
        prefixes = ()
    _apply_unit(
        devices,
        dev_key,
        constants,
        crossover,
        overrides,
        prefixes,
        profiles,
        profile_buckets,
    )


def _materialize_store(state: dict) -> None:
    global _constants_version
    devices = json.loads(json.dumps(state["disk"]))
    for unit in state["overlay"]:
        _replay_unit(devices, unit)
    parsed = _parse_payload_devices(devices)
    state["profiles"] = {dev: data.get("profiles", {}) for dev, data in devices.items()}
    for name, values in zip(
        ("constants", "crossover", "overrides"), parsed, strict=True
    ):
        state[name].clear()
        state[name].update(values)
    _constants_version += 1


def refresh_store() -> None:
    """One stat per eager refresh; parse only changed complete snapshots."""
    with _store_lock:
        path, state = _activate_store()
        try:
            token = _file_token(path)
        except OSError:
            return
        if token == state["token"]:
            return
        payload = _read_payload_for_merge(path)
        try:
            devices = payload.get("devices", {})
            _parse_payload_devices(devices)
        except (AttributeError, TypeError, ValueError, KeyError):
            devices = {}
        state["disk"], state["token"] = devices, token
        state["failed"].clear()
        state["cross_failed"].clear()
        _materialize_store(state)


def _maybe_load_disk() -> None:
    refresh_store()


def get_constants(device: torch.device, family: str) -> Optional[CpbConstants]:
    with _store_lock:
        refresh_store()
        return _constants.get((_device_key(device), family))


def _read_payload_for_merge(path: pathlib.Path) -> dict:
    """Existing cache content to merge into. Only current-schema files merge;
    anything else starts fresh (stale entries recalibrate on the next
    tuning-mode pass)."""
    try:
        existing = json.loads(path.read_text())
    except (OSError, ValueError):
        existing = None
    if isinstance(existing, dict) and existing.get("schema_version") == (
        _SCHEMA_VERSION
    ):
        return existing
    return {"schema_version": _SCHEMA_VERSION, "devices": {}}


def publish_calibration(
    device: torch.device,
    family: str,
    *,
    constants: CpbConstants | None = None,
    crossover: dict[str, int] | None = None,
    overrides: dict[str, int] | None = None,
    replace_family: bool = False,
    replace_family_if_absent: bool = False,
    profiles: dict | None = None,
    profile_buckets: dict | None = None,
) -> bool:
    """Publish one measured unit; write failures retain a scope-local overlay.

    ``replace_family`` discards the family's stale crossover/override entries
    unconditionally (force re-measurement). ``replace_family_if_absent`` does
    so only when the family is still absent under the publish lock; when a
    concurrent publisher landed in between, entries merge instead.

    ``profiles`` replaces complete profiles. Refinement supplies
    ``profile_buckets`` to insert only newly measured buckets into an existing
    profile; its complete profile is used only if no profile remains on disk.
    The same incremental intent is retained when replaying a failed write."""
    with _store_lock:
        refresh_store()
        path, state = _activate_store()
        dev_key = _device_key(device)
        aliases = (
            ("dsv3_2", "glm_nsa") if family in ("dsv3_2", "glm_nsa") else (family,)
        )
        replace = replace_family or replace_family_if_absent
        prefixes = tuple(f + "|" for f in aliases) if replace else ()
        unit = (
            dev_key,
            {family: asdict(constants)} if constants is not None else {},
            crossover or {},
            overrides or {},
            prefixes,
            profiles,
            profile_buckets,
            aliases if (replace_family_if_absent and not replace_family) else (),
        )
        candidate: dict = {}
        _apply_unit(candidate, *unit[:7])
        _parse_payload_devices(candidate)
        persisted = False
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(path) + ".lock"):
                payload = _read_payload_for_merge(path)
                devices = payload.setdefault("devices", {})
                for pending in state["overlay"]:
                    _replay_unit(devices, pending)
                _replay_unit(devices, unit)
                _parse_payload_devices(devices)
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=path.parent, delete=False
                ) as handle:
                    tmp = pathlib.Path(handle.name)
                    handle.write(json.dumps(payload, indent=2) + "\n")
                os.replace(tmp, path)
                state["disk"] = devices
                state["overlay"].clear()
                state["token"] = _file_token(path)
                persisted = True
        except OSError as error:
            logger.warning("SM120 calibration not persisted to %s (%s)", path, error)
        finally:
            if tmp is not None:
                with contextlib.suppress(FileNotFoundError):
                    tmp.unlink()
        if not persisted:
            state["overlay"].append(unit)
        for alias in aliases:
            state["failed"].discard((dev_key, alias))
            state["cross_failed"].discard((dev_key, alias))
        _materialize_store(state)
        return persisted


def save_constants(device: torch.device, family: str, c: CpbConstants) -> bool:
    return publish_calibration(device, family, constants=c)


def _store_projection(family: str, section: str):
    with _store_lock:
        refresh_store()
        return {
            "crossover": _crossover,
            "overrides": _cpb_overrides,
            "failed": _failed,
            "cross_failed": _crossover_failed,
        }[section]


def mark_calibration_failed(device: torch.device, family: str) -> None:
    """Suppress further calibration attempts for (device, family) in-process."""
    failed = _store_projection(family, "failed")
    failed.add((_device_key(device), family))


def is_calibration_failed(device: torch.device, family: str) -> bool:
    """True iff calibration already failed for (device, family) in-process."""
    failed = _store_projection(family, "failed")
    return (_device_key(device), family) in failed


def save_crossover(device: torch.device, table: dict[str, int]) -> bool:
    family = next(iter(table), "dsv4").split("|")[0]
    return publish_calibration(device, family, crossover=table)


def get_decode_max_tokens(
    device: torch.device, family: str, num_heads: int, topk: int
) -> Optional[int]:
    """Calibrated crossover for one config; None when absent (default policy:
    decode-form calls always take the decode kernel)."""
    projection = _store_projection(family, "crossover")
    dev_key = _device_key(device)
    table = projection.get(dev_key)
    if table is None:
        return None
    return table.get(f"{family}|{num_heads}|{topk}")


def has_crossover(device: torch.device, family: str) -> bool:
    """True iff the process/disk cache holds crossover entries for ``family``.
    The ``dsv3_2`` calibration covers both the ``dsv3_2`` and ``glm_nsa`` key
    spaces, so both must be present. Loose any-key predicate;
    :func:`crossover_grid_complete` is the full-sweep gate."""
    projection = _store_projection(family, "crossover")
    dev_key = _device_key(device)
    table = projection.get(dev_key)
    if not table:
        return False
    prefixes = ("dsv3_2|", "glm_nsa|") if family == "dsv3_2" else (f"{family}|",)
    return all(any(k.startswith(p) for k in table) for p in prefixes)


def crossover_grid_complete(device: torch.device, family: str) -> bool:
    """True iff the cache holds a crossover entry for every (num_heads, topk)
    pair on the family's full calibration grid (both key spaces for
    ``dsv3_2``). Targeted off-grid calibrations do not count, so a one-off
    ``calibrate_sparse_mla_sm120(heads=..., topks=...)`` cannot suppress the
    full tuning-mode sweep."""
    from ._sparse_mla_sm120_policy import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
        _DECODE_DSV4_1_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
    )

    key_spaces = {
        "dsv4": (("dsv4", _DECODE_DSV4_CALIBRATION_GRID),),
        "dsv4_1": (("dsv4_1", _DECODE_DSV4_1_CALIBRATION_GRID),),
        "dsv3_2": (
            ("dsv3_2", _DECODE_DSV3_2_CALIBRATION_GRID),
            ("glm_nsa", _DECODE_DSV3_2_CALIBRATION_GRID),
        ),
        "glm53_nope": (("glm53_nope", _DECODE_GLM53_NOPE_CALIBRATION_GRID),),
        "dots3_swa": (("dots3_swa", _DECODE_DOTS3_SWA_CALIBRATION_GRID),),
    }.get(family)
    if key_spaces is None:
        return False
    projection = _store_projection(family, "crossover")
    dev_key = _device_key(device)
    table = projection.get(dev_key)
    if not table:
        return False
    return all(
        f"{prefix}|{h}|{k}" in table for prefix, grid in key_spaces for h, k in grid
    )


def mark_crossover_failed(device: torch.device, family: str) -> None:
    """Suppress further crossover calibration for (device, family) in-process."""
    failed = _store_projection(family, "cross_failed")
    failed.add((_device_key(device), family))


def is_crossover_failed(device: torch.device, family: str) -> bool:
    """True iff crossover calibration already failed for (device, family)."""
    failed = _store_projection(family, "cross_failed")
    return (_device_key(device), family) in failed


def get_cpb_override(
    device: torch.device, family: str, num_heads: int, topk: int, num_tokens: int
) -> Optional[int]:
    """Measured per-shape cpb from :func:`refine_cpb`; None when absent (the
    analytical model's pick governs)."""
    projection = _store_projection(family, "overrides")
    dev_key = _device_key(device)
    table = projection.get(dev_key)
    if table is None:
        return None
    return table.get(f"{family}|{num_heads}|{topk}|{num_tokens}")


def save_cpb_override(
    device: torch.device,
    family: str,
    num_heads: int,
    topk: int,
    num_tokens: int,
    cpb: int,
) -> bool:
    entry = {f"{family}|{num_heads}|{topk}|{num_tokens}": int(cpb)}
    return publish_calibration(device, family, overrides=entry)


_PROFILE_T = (1, 4, 8, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class _Dsv41Request:
    heads: int
    topk: int
    compute_precision: str = "default"
    primary_page_size: int = 64
    extra_topk: int = 0
    extra_page_size: int = 0
    extra_kv_fp4: bool = False
    has_topk_length: bool = False
    has_extra_topk_length: bool = False
    has_attn_sink: bool = False

    @property
    def effective(self) -> dict:
        values = asdict(self)
        for name in (
            "extra_kv_fp4",
            "has_topk_length",
            "has_extra_topk_length",
            "has_attn_sink",
        ):
            values[name] = bool(values[name])
        values["compute_precision"] = (
            "bf16" if self.compute_precision == "bf16" else "fp8"
        )
        return {"family": "dsv4_1", "strategy": "queued_rotation_v1", **values}

    @property
    def key(self) -> str:
        return json.dumps(self.effective, sort_keys=True)

    def validate(self) -> None:
        if self.compute_precision not in ("default", "fp8", "bf16"):
            raise ValueError("unsupported compute_precision")
        if not 1 <= self.heads <= 128 or self.topk <= 0 or self.primary_page_size <= 0:
            raise ValueError("invalid DSV4.1 heads/topk/page size")
        if self.extra_topk < 0 or (self.extra_topk > 0 and self.extra_page_size <= 0):
            raise ValueError("extra cache requires positive topk and page size")
        if not self.extra_topk and (
            self.extra_page_size or self.extra_kv_fp4 or self.has_extra_topk_length
        ):
            raise ValueError("extra configuration requires extra_topk")


def _profile_bucket(profile: dict, tokens: int) -> dict | None:
    """Exact refined entry first, else the nearest-up canonical bucket."""
    buckets = profile["buckets"]
    exact = buckets.get(str(tokens))
    if exact is not None:
        return exact
    bucket = next((t for t in _PROFILE_T if tokens <= t), None)
    return None if bucket is None else buckets[str(bucket)]


def get_profile(key: str, device: torch.device) -> dict | None:
    with _store_lock:
        refresh_store()
        _, state = _activate_store()
        value = state.get("profiles", {}).get(_device_key(device), {}).get(key)
        return None if value is None else json.loads(json.dumps(value))


def get_dsv41_profile(request: _Dsv41Request, device: torch.device) -> dict | None:
    return get_profile(request.key, device)


def _profile_pool(
    request: _Dsv41Request, device: torch.device, page: int, pool_bytes: int, fp4: bool
) -> tuple[torch.Tensor, int]:
    from ._sparse_mla_sm120 import dsv41_fp4_quantize_pack_sparse_mla_cache

    facts = format_info(5)
    bpt = facts["fp4_bytes_per_token"] if fp4 else facts["bytes_per_token"]
    while True:
        pages = max(1, pool_bytes // (page * bpt))
        _check_pool_capacity(device, pages * page * bpt)
        cache = None
        try:
            cache = torch.empty((pages, page * bpt), dtype=torch.uint8, device=device)
            if not fp4:
                _initialize_fp8_pool(cache, "dsv4_1", page)
            else:
                generator = torch.Generator(device=device).manual_seed(2)
                chunk = max(1, _SAMPLE_CHUNK_BYTES // (page * facts["query_dim"] * 8))
                for start in range(0, pages, chunk):
                    count = min(chunk, pages - start)
                    latent = (
                        torch.randn(
                            (count, page, facts["query_dim"]),
                            device=device,
                            generator=generator,
                        )
                        .mul_(0.1)
                        .clamp_(-1, 1)
                        .to(torch.bfloat16)
                    )
                    packed = dsv41_fp4_quantize_pack_sparse_mla_cache(latent)
                    cache[start : start + count].copy_(packed.view(count, page * bpt))
                    del latent, packed
            return cache, pages * page
        except torch.cuda.OutOfMemoryError:
            del cache
            if pool_bytes <= _POOL_BYTES_MIN:
                raise CalibrationError(
                    "cannot initialize DSV4.1 profile pool"
                ) from None
            pool_bytes //= 2
            torch.cuda.empty_cache()


@dataclass(frozen=True)
class _Dsv41MeasureContext:
    """Device-resident pools and capabilities shared across profile buckets."""

    device: torch.device
    main: torch.Tensor
    slots: int
    extra: torch.Tensor | None
    extra_slots: int
    caps: tuple[int, int]
    l2: int
    module: Any


def _dsv41_measure_context(
    request: _Dsv41Request, device: torch.device
) -> _Dsv41MeasureContext:
    from ._sparse_mla_sm120_execution import get_sparse_mla_sm120_module

    props = torch.cuda.get_device_properties(device)
    caps = (props.multi_processor_count, props.shared_memory_per_block_optin)
    facts = format_info(5)
    main_bytes = request.topk * facts["bytes_per_token"]
    extra_bpt = (
        facts["fp4_bytes_per_token"]
        if request.extra_kv_fp4
        else facts["bytes_per_token"]
    )
    extra_bytes = request.extra_topk * extra_bpt
    main, slots = _profile_pool(
        request,
        device,
        request.primary_page_size,
        max(
            _POOL_BYTES_MIN,
            _POOL_BYTES_TARGET * main_bytes // (main_bytes + extra_bytes),
        ),
        False,
    )
    extra, extra_slots = (None, 0)
    if request.extra_topk:
        extra, extra_slots = _profile_pool(
            request,
            device,
            request.extra_page_size,
            max(
                _POOL_BYTES_MIN,
                _POOL_BYTES_TARGET * extra_bytes // (main_bytes + extra_bytes),
            ),
            request.extra_kv_fp4,
        )
    return _Dsv41MeasureContext(
        device=device,
        main=main,
        slots=slots,
        extra=extra,
        extra_slots=extra_slots,
        caps=caps,
        l2=_device_l2(device),
        module=get_sparse_mla_sm120_module(),
    )


def _measure_dsv41_bucket(
    request: _Dsv41Request, ctx: _Dsv41MeasureContext, tokens: int
) -> dict:
    from ._sparse_mla_sm120_execution import (
        AttentionMetadata,
        metadata_candidates,
        resolve_attention,
    )

    device = ctx.device
    facts = format_info(5)
    main_bytes = request.topk * facts["bytes_per_token"]
    extra_bpt = (
        facts["fp4_bytes_per_token"]
        if request.extra_kv_fp4
        else facts["bytes_per_token"]
    )
    extra_bytes = request.extra_topk * extra_bpt
    generator = torch.Generator(device=device).manual_seed(3)
    q = (
        torch.randn(
            (tokens, request.heads, facts["query_dim"]),
            device=device,
            generator=generator,
        )
        .mul_(0.1)
        .clamp_(-1, 1)
        .to(torch.bfloat16)
    )
    output = torch.empty_like(q)
    lse = torch.empty((tokens, request.heads), device=device)
    lengths = (
        torch.full((tokens,), request.topk, dtype=torch.int32, device=device)
        if request.has_topk_length
        else None
    )
    extra_lengths = (
        torch.full((tokens,), request.extra_topk, dtype=torch.int32, device=device)
        if request.has_extra_topk_length
        else None
    )
    sink = (
        torch.zeros((request.heads,), device=device) if request.has_attn_sink else None
    )
    metadata = AttentionMetadata(
        5,
        tokens,
        request.heads,
        request.topk,
        request.extra_topk,
        request.primary_page_size,
        request.extra_page_size,
        ctx.main.stride(0),
        0 if ctx.extra is None else ctx.extra.stride(0),
        facts["bytes_per_token"],
        request.topk,
        request.extra_topk,
        request.heads,
        lengths is not None,
        extra_lengths is not None,
        sink is not None,
        request.extra_kv_fp4,
        0,
    )
    legal = metadata_candidates(
        metadata, request.effective["compute_precision"], *ctx.caps
    )
    if 0 not in legal:
        raise CalibrationError("no legal DSV4.1 decode candidate for profile")
    footprint = tokens * (main_bytes + extra_bytes)
    count = max(_MIN_BATCH_CALLS, 4 * ctx.l2 // footprint + 2)
    index_bytes = count * tokens * (request.topk + request.extra_topk) * 4
    if index_bytes > 64 << 20:
        raise CalibrationError("profile rotation exceeds bounded 64 MiB index budget")
    indices = torch.randint(
        ctx.slots,
        (count, tokens, request.topk),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    extra_indices = (
        torch.randint(
            ctx.extra_slots,
            (count, tokens, request.extra_topk),
            device=device,
            dtype=torch.int32,
            generator=generator,
        )
        if ctx.extra is not None
        else None
    )
    distinct = torch.unique(indices).numel() * facts["bytes_per_token"]
    if extra_indices is not None:
        distinct += torch.unique(extra_indices).numel() * extra_bpt
    if distinct - footprint <= 2 * ctx.l2:
        raise CalibrationError(
            "profile distinct rotation footprint does not exceed 2xL2"
        )
    arguments = [
        (indices[i], None if extra_indices is None else extra_indices[i])
        for i in range(count)
    ]

    def timed(variant: int, cpb: int) -> float:
        plan = resolve_attention(
            **metadata._replace(variant=variant)._asdict(),
            precision=request.effective["compute_precision"],
            cpb=cpb,
            sm_count=ctx.caps[0],
            max_shared_bytes=ctx.caps[1],
        )
        workspace = [
            torch.empty(tuple(shape), dtype=getattr(torch, str(dtype)), device=device)
            for shape, dtype, _, _ in plan.workspace()
        ]

        def call(ix, ex):
            ctx.module.execute_attention(
                plan,
                q,
                ctx.main,
                ix,
                workspace[0],
                workspace[1],
                output,
                lse,
                facts["query_dim"] ** -0.5,
                lengths,
                sink,
                ctx.extra,
                ex,
                extra_lengths,
            )

        return time_calibration_calls(call, arguments, device)

    decoded = {cpb: timed(0, cpb) for cpb in range(1, legal[0] + 1)}
    best = min(decoded, key=lambda cpb: (decoded[cpb], -cpb))
    prefill = timed(1, 1) if 1 in legal else None
    variant = (
        0 if prefill is None or decoded[best] <= _CROSSOVER_MARGIN * prefill else 1
    )
    return {
        "variant": variant,
        "cpb": best,
        "decode_s": decoded[best],
        "prefill_s": prefill,
        "decode_candidates_s": decoded,
        "prefill_absent": None if prefill is not None else "metadata_ineligible",
        "rotation_calls": count,
        "index_bytes": index_bytes,
        "distinct_rotation_bytes": distinct,
    }


@_target_calibration
def _measure_dsv41(request: _Dsv41Request, device: torch.device) -> dict:
    request.validate()
    ctx = _dsv41_measure_context(request, device)
    buckets = {
        str(tokens): _measure_dsv41_bucket(request, ctx, tokens)
        for tokens in _PROFILE_T
    }
    return {
        "request": request.effective,
        "sample_kind": "full_capacity_canonical_layout",
        "caps": {
            "sm_count": ctx.caps[0],
            "shared_bytes": ctx.caps[1],
            "l2_bytes": ctx.l2,
        },
        "pool_bytes": [
            ctx.main.numel(),
            0 if ctx.extra is None else ctx.extra.numel(),
        ],
        "buckets": buckets,
    }


@_target_calibration
def refine_dsv41(
    request: _Dsv41Request, device: torch.device, tokens: int
) -> dict | None:
    """Measure one exact token count and merge it into the stored profile.

    Returns the refined bucket entry, or ``None`` when refinement does not
    apply (outside the profiled range, no stored profile, or the exact entry
    already measured).
    """
    if not 1 <= tokens <= _PROFILE_T[-1]:
        return None
    request.validate()
    profile = get_profile(request.key, device)
    if profile is None or str(tokens) in profile["buckets"]:
        return None
    ctx = _dsv41_measure_context(request, device)
    entry = _measure_dsv41_bucket(request, ctx, tokens)
    entry["provenance"] = "refined"
    with _store_lock:
        profile = get_profile(request.key, device)
        if profile is None or str(tokens) in profile["buckets"]:
            return None
        profile["buckets"][str(tokens)] = entry
        persisted = publish_calibration(
            device,
            "dsv4_1",
            profiles={request.key: profile},
            profile_buckets={request.key: {str(tokens): entry}},
        )
    return {**entry, "persisted": persisted}


def _calibrate_dsv41(request: _Dsv41Request, device: torch.device, force: bool) -> dict:
    with _store_lock:
        old = get_dsv41_profile(request, device)
        if old is not None and not force:
            _, state = _activate_store()
            return {
                **old,
                "status": "reused",
                "persisted": not state["overlay"],
            }
    started = time.monotonic()
    try:
        profile = _measure_dsv41(request, device)
    except (CalibrationError, RuntimeError) as error:
        mark_calibration_failed(device, request.key)
        return {
            "request": request.effective,
            "status": "failed",
            "error": str(error),
            "old_profile_retained": old is not None,
            "persisted": False,
        }
    profile["elapsed_s"] = time.monotonic() - started
    persisted = publish_calibration(device, "dsv4_1", profiles={request.key: profile})
    failed = _store_projection("dsv4_1", "failed")
    failed.discard((_device_key(device), request.key))
    return {**profile, "status": "measured", "persisted": persisted}


# ── Public calibration entry point ─────────────────────────────────────────


# ("<grid heads>", "<grid topks>", min_topk)
def _family_specs() -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]]:
    from ._sparse_mla_sm120_policy import (
        _CALIBRATION_HEADS,
        _DECODE_DSV3_2_TOPKS,
        _DECODE_DSV4_TOPKS,
        _DECODE_DSV4_1_TOPK,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_TOPK,
        _DECODE_DOTS3_SWA_TOPK,
    )

    v32_topks = tuple(sorted(_DECODE_DSV3_2_TOPKS))
    return {
        "dsv4": (_CALIBRATION_HEADS, tuple(sorted(_DECODE_DSV4_TOPKS)), 1),
        "dsv4_1": (_CALIBRATION_HEADS, (_DECODE_DSV4_1_TOPK,), 1),
        "dsv3_2": (_CALIBRATION_HEADS, v32_topks, 1),
        "glm_nsa": (_CALIBRATION_HEADS, v32_topks, 1),
        "glm53_nope": (
            tuple(sorted({h for h, _ in _DECODE_GLM53_NOPE_CALIBRATION_GRID})),
            (_DECODE_GLM53_NOPE_TOPK,),
            1,
        ),
        "dots3_swa": (
            tuple(sorted({h for h, _ in _DECODE_DOTS3_SWA_CALIBRATION_GRID})),
            (_DECODE_DOTS3_SWA_TOPK,),
            513,
        ),
    }


@dataclass(frozen=True)
class SparseMLASm120CalibrationReport:
    """Outcome of one :func:`calibrate_sparse_mla_sm120` call.

    Attributes
    ----------
    device : str
        The device key the entries were recorded under.
    constants_calibrated : tuple[str, ...]
        Families whose cpb constants were (re)measured this call.
    constants_present : tuple[str, ...]
        Families whose cpb constants were already on disk (skipped).
    entries_calibrated : int
        Crossover ``(family, num_heads, topk)`` entries newly measured.
    entries_skipped : int
        Requested crossover entries already present (idempotent default).
    failed : tuple[str, ...]
        Human-readable per-family/per-entry failures, if any. Calibration
        failures are collected here instead of raised so a multi-family call
        still records the families that did succeed.
    cache_path : str
        The JSON document the entries were merged into.
    elapsed_s : float
        Wall-clock seconds for the whole call.
    """

    device: str
    constants_calibrated: tuple[str, ...]
    constants_present: tuple[str, ...]
    entries_calibrated: int
    entries_skipped: int
    failed: tuple[str, ...]
    cache_path: str
    elapsed_s: float
    persisted: bool = False
    profiles: tuple[dict, ...] = ()


@_target_calibration
def calibrate_sparse_mla_sm120(
    device: Optional[torch.device] = None,
    *,
    heads: Optional[tuple[int, ...]] = None,
    topks: Optional[tuple[int, ...]] = None,
    families: Optional[tuple[str, ...]] = None,
    force: bool = False,
    compute_precision: str = "default",
    primary_page_size: int = 64,
    extra_topk: int = 0,
    extra_page_size: int = 0,
    extra_kv_fp4: bool = False,
    has_topk_length: bool = False,
    has_extra_topk_length: bool = False,
    has_attn_sink: bool = False,
) -> SparseMLASm120CalibrationReport:
    """Calibrate the SM120 sparse-MLA decode model on ``device`` and persist it.

    One call does both layers: the per-family cpb constants (measured when
    absent, or always when ``force=True``) and then the decode/prefill
    crossover entry for every requested ``(family, num_heads, topk)``
    combination. Results merge into the JSON cache (see
    :func:`default_cache_path`) and take effect in-process immediately (the
    ``_constants_version`` bump self-invalidates the plan memoization).

    The default is idempotent skip-existing: frameworks may call this
    unconditionally on every startup warmup. ``force=True`` re-measures even
    present entries — the escape hatch after a kernel upgrade changes the
    measured optimum.

    Calibration also runs lazily on the first decode call under
    ``autotune(tuning_mode=True)`` when entries are absent;
    ``autotune(..., skip_ops={"sparse_mla_sm120"})`` opts out of those
    passes. Neither entry point may run under CUDA graph capture.

    Measure on an idle GPU (the protocol is timing-sensitive), and calibrate
    per machine — the constants are device-local. A full default sweep (all
    families, grid heads x grid topks) takes on the order of minutes; a
    single ``(family, heads, topks)`` combination is seconds.

    Parameters
    ----------
    device : Optional[torch.device]
        Target device; defaults to the current CUDA device.
    heads : Optional[tuple[int, ...]]
        Head counts to calibrate. Defaults to the family's crossover grid
        (``{8,16,32,64,128}`` for the DeepSeek families, ``{32,64}`` for
        glm53_nope, ``{8,16,32,64}`` for dots3_swa). Any count in
        ``[1, 128]`` is accepted — off-grid counts ride the runtime-H
        instantiation.
    topks : Optional[tuple[int, ...]]
        Top-k widths to calibrate; defaults to the family's calibrated
        values. Any width above the family minimum is accepted (topk is a
        runtime kernel argument).
    families : Optional[tuple[str, ...]]
        Subset of ``{"dsv4", "dsv3_2", "glm_nsa", "glm53_nope",
        "dots3_swa", "dsv4_1"}``; defaults to all, including DSV4.1 single
        FP8 only. ``dsv3_2`` and ``glm_nsa`` share constants and one sweep.
    force : bool
        Re-measure complete units; failed measurements retain old profiles.
    compute_precision : str
        DSV4.1 ``default``/``fp8`` share FP8 profiles; ``bf16`` is independent.
    primary_page_size, extra_page_size : int
        DSV4.1 main and extra page sizes. Extra page size is zero without extra KV.
    extra_topk : int
        Extra-cache candidate capacity, zero for a single cache.
    extra_kv_fp4 : bool
        Quantize the extra DSV4.1 cache with the existing FP4 packer.
    has_topk_length, has_extra_topk_length, has_attn_sink : bool
        Presence flags. Samples use full-capacity lengths and a finite zero sink.
        Nondefault extended configuration requires only the DSV4.1 family.

    DSV4.1 profiles measure all eight token buckets, independent phase decisions
    and complete decode CPB sweeps. Each exact H/K/cache configuration counts as
    one entry. Profiles do not certify arbitrary ragged distributions. Runtime
    token counts use an exact refined entry when tuning measured one, else the
    next larger bucket; counts above 64 require legal prefill.

    Returns
    -------
    SparseMLASm120CalibrationReport
        Counts and the cache path, for framework logging.

    Raises
    ------
    ValueError
        If any requested ``(family, num_heads, topk)`` is outside the decode
        envelope (e.g. dots3_swa topk < 513); all invalid combinations are
        listed.
    """
    from ._sparse_mla_sm120_execution import (
        get_sparse_mla_sm120_module as _get_sparse_mla_sm120_decode_module,
    )
    from ._sparse_mla_sm120_policy import _CPB_FAMILY_ALIAS

    t0 = time.monotonic()
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device = torch.device(device)

    specs = _family_specs()
    fams = tuple(families) if families is not None else tuple(specs)
    unknown = [f for f in fams if f not in specs]
    if unknown:
        raise ValueError(
            f"unknown sparse-MLA families: {unknown}; available: {sorted(specs)}"
        )

    extended = (
        compute_precision,
        primary_page_size,
        extra_topk,
        extra_page_size,
        extra_kv_fp4,
        has_topk_length,
        has_extra_topk_length,
        has_attn_sink,
    )
    if extended != ("default", 64, 0, 0, False, False, False, False) and any(
        f != "dsv4_1" for f in fams
    ):
        raise ValueError("extended calibration configuration requires only DSV4.1")
    requests = []
    if "dsv4_1" in fams:
        grid_heads, grid_topks, _ = specs["dsv4_1"]
        for h in heads if heads is not None else grid_heads:
            for k in topks if topks is not None else grid_topks:
                request = _Dsv41Request(h, k, *extended)
                request.validate()
                requests.append(request)

    # Validate every requested combination up front; nothing is measured or
    # written when any combination is out of envelope.
    invalid = []
    for fam in fams:
        grid_heads, grid_topks, min_topk = specs[fam]
        for h in tuple(heads) if heads is not None else grid_heads:
            for k in tuple(topks) if topks is not None else grid_topks:
                if not (1 <= h <= 128 and k >= min_topk):
                    invalid.append(
                        f"{fam}(num_heads={h}, topk={k}) "
                        f"[need 1<=num_heads<=128, topk>={min_topk}]"
                    )
    if invalid:
        raise ValueError(
            "calibrate_sparse_mla_sm120: combinations outside the decode "
            "envelope: " + "; ".join(invalid)
        )

    constants_calibrated: list[str] = []
    constants_present: list[str] = []
    failed: list[str] = []
    entries_calibrated = 0
    entries_skipped = 0

    persisted = True
    profiles = tuple(_calibrate_dsv41(request, device, force) for request in requests)
    for profile in profiles:
        entries_calibrated += profile["status"] == "measured"
        entries_skipped += profile["status"] == "reused"
        if profile["status"] == "failed":
            failed.append(f"dsv4_1: {profile['error']}")
        persisted = persisted and profile["persisted"]
    cpb_families = sorted({_CPB_FAMILY_ALIAS.get(f, f) for f in fams if f != "dsv4_1"})
    for cpb_family in cpb_families:
        existing = None if force else get_constants(device, cpb_family)
        requested = [f for f in fams if _CPB_FAMILY_ALIAS.get(f, f) == cpb_family]
        pairs: set[tuple[int, int]] = set()
        for fam in requested:
            grid_heads, grid_topks, _ = specs[fam]
            for h in tuple(heads) if heads is not None else grid_heads:
                for k in tuple(topks) if topks is not None else grid_topks:
                    if (
                        existing is not None
                        and get_decode_max_tokens(device, fam, h, k) is not None
                    ):
                        entries_skipped += 1
                    else:
                        pairs.add((h, k))
        try:
            c = (
                existing
                if existing is not None
                else calibrate(_get_sparse_mla_sm120_decode_module, cpb_family, device)
            )
            table = (
                calibrate_crossover(
                    _get_sparse_mla_sm120_decode_module(),
                    device,
                    cpb_family,
                    c,
                    grid_override=sorted(pairs),
                )
                if pairs
                else {}
            )
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as error:
            failed.append(f"{cpb_family}: {error}")
            mark_calibration_failed(device, cpb_family)
            continue
        if existing is None or table:
            persisted = (
                publish_calibration(
                    device,
                    cpb_family,
                    constants=c if existing is None else None,
                    crossover=table,
                    replace_family=force,
                    replace_family_if_absent=existing is None and not force,
                )
                and persisted
            )
        else:
            _, state = _activate_store()
            persisted = persisted and not state["overlay"]
        if existing is None:
            constants_calibrated.append(cpb_family)
        else:
            constants_present.append(cpb_family)
        for fam in requested:
            for h, k in pairs:
                if f"{fam}|{h}|{k}" in table:
                    entries_calibrated += 1

    return SparseMLASm120CalibrationReport(
        device=_device_key(device),
        constants_calibrated=tuple(constants_calibrated),
        constants_present=tuple(constants_present),
        entries_calibrated=entries_calibrated,
        entries_skipped=entries_skipped,
        failed=tuple(failed),
        cache_path=str(default_cache_path()),
        elapsed_s=time.monotonic() - t0,
        persisted=persisted,
        profiles=profiles,
    )
