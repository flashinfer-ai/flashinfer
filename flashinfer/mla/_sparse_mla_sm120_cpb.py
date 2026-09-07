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
version 1; only current-schema files load — files at any other version count
as absent, so their families recalibrate on the next tuning-mode pass). The
runtime decode/prefill routing in :mod:`._sparse_mla_sm120` consults it;
absent entries keep the historical decode-first policy.

Finally, tuning-mode decode-form calls refine the model's cpb pick for the
exact shape being warmed (:func:`refine_cpb`): the pick +/- a small candidate
window is timed and the measured best persists as a per-shape override. The
model remains the proposal and the fallback for every shape never warmed
(off-grid ``num_heads``, dual-cache calls, non-tuning processes).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from filelock import FileLock

logger = logging.getLogger(__name__)

_BI = 64  # chunk width in candidates (BLOCK_SIZE_N)
_HPB = 16  # head tile per block

_SCHEMA_VERSION = 1
# Only current-schema files load; any other version counts as absent and the
# families recalibrate on the next tuning-mode pass.
_BYTES_PER_TOKEN = {"dsv4": 584, "dsv3_2": 656, "glm53_nope": 656, "dots3_swa": 1160}
_D_QK = {"dsv4": 512, "dsv3_2": 576, "glm53_nope": 512, "dots3_swa": 1088}
_D_V = {"dsv4": 512, "dsv3_2": 512, "glm53_nope": 512, "dots3_swa": 1024}
# Kernel candidate-tile width per family: DOTS3_SWA decodes at BI=32 (its
# 1040-byte KV smem stride does not fit BI=64); the others run 64. The head
# tile is HPB=16 for every family.
_CHUNK_WIDTH = {"dsv4": 64, "dsv3_2": 64, "glm53_nope": 64, "dots3_swa": 32}

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
# Batched timing (_time_call_fresh_indices): batches per measurement, and the
# per-batch call-count floor/cap around the L2 reuse-distance sizing. The cap
# must stay above L2/min-footprint (T=4, topk=128 ~ 0.3 MiB -> ~400 calls)
# or small-footprint probes would go L2-warm across batches.
_TIMED_BATCHES = 5
_MIN_BATCH_CALLS = 8
_MAX_BATCH_CALLS = 1024


class CalibrationError(RuntimeError):
    """Calibration measurements were unusable (OOM or implausible constants)."""


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
        Bytes per ``_BI``-wide chunk (``_BI * bytes_per_token``).
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


def _allocate_kv_pool(family: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """Allocate a ~2 GiB paged KV pool for ``family`` (halved on OOM down to
    512 MiB) and return it with its slot count. The 2-D ``[blocks, bytes]``
    form is accepted by the FFI binding, which derives the block stride from
    the tensor metadata. The row is one 64-token page for every family — the
    64 here is the page block size, not the family's chunk width (DOTS3_SWA
    chunks 32 candidates per tile inside the same 64-token page)."""
    w = 64 * _BYTES_PER_TOKEN[family]
    pool_bytes = _POOL_BYTES_TARGET
    while True:
        try:
            kv_cache = torch.empty(pool_bytes // w, w, dtype=torch.uint8, device=device)
            break
        except torch.cuda.OutOfMemoryError:
            if pool_bytes <= _POOL_BYTES_MIN:
                raise CalibrationError(
                    f"cannot allocate a >= {_POOL_BYTES_MIN >> 20} MiB KV pool "
                    "for sparse-MLA cpb calibration"
                ) from None
            pool_bytes //= 2
            torch.cuda.empty_cache()
    return kv_cache, kv_cache.shape[0] * 64


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
    # A set recurs after K-1 intervening calls; require the reuse distance to
    # cover L2 so no gathered page survives between visits. Without a known
    # L2 size, the minimum batch still hides launch latency.
    l2 = int(getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0) or 0)
    footprint = max(1, num_tokens * topk * bytes_per_token)
    k = (
        _MIN_BATCH_CALLS
        if not l2
        else min(_MAX_BATCH_CALLS, max(_MIN_BATCH_CALLS, l2 // footprint + 2))
    )
    sets = [
        torch.randint(
            0, num_slots, (num_tokens, topk), dtype=torch.int32, device=device
        )
        for _ in range(k)
    ]
    for i in range(_WARMUP_ITERS):
        call(sets[i % k])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(_TIMED_BATCHES):
        start.record()
        for indices in sets:
            call(indices)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3 / k)
    return best


def _make_decode_call_builder(
    module: Any, family: str, device: torch.device, kv_cache: torch.Tensor
) -> Callable[[int, int, int, int, int], Callable[[torch.Tensor], None]]:
    """Decode-call constructor shared by calibrate() and calibrate_crossover().

    Returns a builder mapping ``(num_tokens, num_heads, topk, model_type,
    cpb)`` to a ``call(indices) -> None`` closure that drives the family's
    decode kernel over ``kv_cache``, so the two calibration passes' FFI
    argument lists cannot drift apart. ``model_type`` only reaches the
    dsv3_2-kernel families; the decode-dsv4 FFI resolves the model type from
    ``d_qk`` itself (512 -> DSV4, 1088 -> DOTS3_SWA).
    """
    d_qk = _D_QK[family]
    d_v = _D_V[family]
    bi = _CHUNK_WIDTH[family]
    sm_scale = d_qk**-0.5
    from ._sparse_mla_sm120_plan import _decode_scratch_heads

    def build(
        num_tokens: int, num_heads: int, topk: int, model_type: int, cpb: int
    ) -> Callable[[torch.Tensor], None]:
        num_splits = _ceil_div(topk, bi)
        q = (
            (
                torch.randn(
                    num_tokens, num_heads, d_qk, device=device, dtype=torch.float32
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
        if family in ("dsv4", "dots3_swa"):

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
                    cpb,
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
    from ._sparse_mla_sm120_plan import (
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_GLM53_NOPE,
    )

    device = torch.device(device)
    props = torch.cuda.get_device_properties(device)
    sm_count = int(props.multi_processor_count)
    l2_cache_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    bi = _CHUNK_WIDTH[family]
    w = bi * _BYTES_PER_TOKEN[family]
    # Families whose decode is instantiated at a single topk have a fixed N,
    # so the bandwidth term is identified from a cpb pair instead of an N pair.
    _CPB_PAIR_MEASUREMENTS = {
        "glm53_nope": _MEASUREMENTS_GLM53_NOPE,
        "dots3_swa": _MEASUREMENTS_DOTS3_SWA,
    }
    measurements = _CPB_PAIR_MEASUREMENTS.get(family, _MEASUREMENTS)
    model_type = (
        _MODEL_TYPE_GLM53_NOPE if family == "glm53_nope" else _MODEL_TYPE_DSV3_2
    )

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
    from ._sparse_mla_sm120_plan import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
        _PREFILL_IMPL_AUTO,
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_DSV4,
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
                    num_tokens, num_heads, d_qk, device=device, dtype=torch.float32
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
            )

        try:
            # One probe launch to surface any launch-time failure early.
            call(
                torch.randint(
                    0, num_slots, (num_tokens, topk), dtype=torch.int32, device=device
                )
            )
            torch.cuda.synchronize()
        except RuntimeError:
            return float("inf")
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
    from ._sparse_mla_sm120_plan import (
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_GLM53_NOPE,
    )

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
    model_type = (
        _MODEL_TYPE_GLM53_NOPE if family == "glm53_nope" else _MODEL_TYPE_DSV3_2
    )
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


_cache_mtime: float = -1.0
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
            if family == _DECODE_MAX_TOKENS_KEY:
                if isinstance(raw, dict):
                    new_crossover[dev_key] = {str(k): int(v) for k, v in raw.items()}
                continue
            if family == _CPB_OVERRIDES_KEY:
                if isinstance(raw, dict):
                    new_overrides[dev_key] = {str(k): int(v) for k, v in raw.items()}
                continue
            new_constants[(dev_key, family)] = CpbConstants(**raw)
    return new_constants, new_crossover, new_overrides


def _maybe_load_disk() -> None:
    """Mtime-gated lazy load; corrupt or mismatched files count as absent."""
    global _cache_mtime, _constants_version
    path = default_cache_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return
    if mtime <= _cache_mtime:
        return
    try:
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _SCHEMA_VERSION
        ):
            return
        devices = payload["devices"]
        if not isinstance(devices, dict):
            return
        # Parse fully into locals first: a mid-document failure must not
        # publish a prefix of the entries while leaving mtime/version stale.
        new_constants, new_crossover, new_overrides = _parse_payload_devices(devices)
    except (OSError, ValueError, TypeError, KeyError):
        # Keep mtime unchanged so the next cold call retries.
        return
    _constants.update(new_constants)
    _crossover.update(new_crossover)
    _cpb_overrides.update(new_overrides)
    _cache_mtime = mtime
    _constants_version += 1


def get_constants(device: torch.device, family: str) -> Optional[CpbConstants]:
    """Process-cached or mtime-gated disk-loaded constants; None if absent."""
    key = (_device_key(device), family)
    cached = _constants.get(key)
    if cached is not None:
        return cached
    _maybe_load_disk()
    return _constants.get(key)


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


def _merge_into_cache(dev_key: str, section: str, entries: dict) -> dict:
    """Read-modify-write ``entries`` into ``section`` of ``dev_key``'s device
    record and return the merged document ({"devices": {}} when the write
    fails — callers still publish to the process cache, losing only
    cross-process sharing).

    The read-modify-write is FileLock-serialized across processes (multi-GPU
    tuning saves sibling device keys into the same file); reads stay
    lock-free because the write is an atomic replace.
    """
    global _cache_mtime
    path = default_cache_path()
    try:
        with FileLock(str(path.with_name(path.name + ".lock"))):
            payload = _read_payload_for_merge(path)
            if not isinstance(payload.get("devices"), dict):
                payload["devices"] = {}
            dev = payload["devices"].setdefault(dev_key, {})
            if not isinstance(dev, dict):
                dev = payload["devices"][dev_key] = {}
            sec = dev.setdefault(section, {})
            if not isinstance(sec, dict):
                sec = dev[section] = {}
            sec.update(entries)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, path)
    except OSError as e:
        logger.warning(
            "SM120 sparse-MLA cpb cache not persisted to %s (%s); "
            "using entries in-process only.",
            path,
            e,
        )
        payload = {"devices": {}}
    with contextlib.suppress(OSError):
        _cache_mtime = path.stat().st_mtime
    return payload


def _publish_payload(payload: dict) -> None:
    """Re-parse a merged document into the process caches and bump the
    constants version. Publishes the whole document, not only the section
    just saved: the merge may have carried on-disk sibling entries written by
    other processes, and _cache_mtime now asserts we hold the file's content.
    """
    global _constants_version
    try:
        new_constants, new_crossover, new_overrides = _parse_payload_devices(
            payload["devices"]
        )
    except (ValueError, TypeError, KeyError):
        new_constants, new_crossover, new_overrides = {}, {}, {}
    _constants.update(new_constants)
    _crossover.update(new_crossover)
    _cpb_overrides.update(new_overrides)
    _constants_version += 1


def save_constants(device: torch.device, family: str, c: CpbConstants) -> None:
    """Merge ``c`` into the disk cache (read-modify-write, atomic replace) and
    the process cache. A failed disk write only loses cross-process sharing;
    the in-process constants still take effect."""
    key = (_device_key(device), family)
    _publish_payload(_merge_into_cache(key[0], family, asdict(c)))
    _constants[key] = c


def mark_calibration_failed(device: torch.device, family: str) -> None:
    """Suppress further calibration attempts for (device, family) in-process."""
    _failed.add((_device_key(device), family))


def is_calibration_failed(device: torch.device, family: str) -> bool:
    """True iff calibration already failed for (device, family) in-process."""
    return (_device_key(device), family) in _failed


def save_crossover(device: torch.device, table: dict[str, int]) -> None:
    """Merge a crossover table into the disk cache (read-modify-write, atomic
    replace) and the process cache. Same failure semantics as
    :func:`save_constants`."""
    dev_key = _device_key(device)
    _publish_payload(_merge_into_cache(dev_key, _DECODE_MAX_TOKENS_KEY, table))
    _crossover.setdefault(dev_key, {}).update(table)


def get_decode_max_tokens(
    device: torch.device, family: str, num_heads: int, topk: int
) -> Optional[int]:
    """Calibrated crossover for one config; None when absent (default policy:
    decode-form calls always take the decode kernel)."""
    dev_key = _device_key(device)
    table = _crossover.get(dev_key)
    if table is None:
        _maybe_load_disk()
        table = _crossover.get(dev_key)
    if table is None:
        return None
    return table.get(f"{family}|{num_heads}|{topk}")


def has_crossover(device: torch.device, family: str) -> bool:
    """True iff the process/disk cache holds crossover entries for ``family``.
    The ``dsv3_2`` calibration covers both the ``dsv3_2`` and ``glm_nsa`` key
    spaces, so both must be present. Loose any-key predicate;
    :func:`crossover_grid_complete` is the full-sweep gate."""
    dev_key = _device_key(device)
    table = _crossover.get(dev_key)
    if table is None:
        _maybe_load_disk()
        table = _crossover.get(dev_key)
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
    from ._sparse_mla_sm120_plan import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
    )

    key_spaces = {
        "dsv4": (("dsv4", _DECODE_DSV4_CALIBRATION_GRID),),
        "dsv3_2": (
            ("dsv3_2", _DECODE_DSV3_2_CALIBRATION_GRID),
            ("glm_nsa", _DECODE_DSV3_2_CALIBRATION_GRID),
        ),
        "glm53_nope": (("glm53_nope", _DECODE_GLM53_NOPE_CALIBRATION_GRID),),
        "dots3_swa": (("dots3_swa", _DECODE_DOTS3_SWA_CALIBRATION_GRID),),
    }.get(family)
    if key_spaces is None:
        return False
    dev_key = _device_key(device)
    table = _crossover.get(dev_key)
    if table is None:
        _maybe_load_disk()
        table = _crossover.get(dev_key)
    if not table:
        return False
    return all(
        f"{prefix}|{h}|{k}" in table for prefix, grid in key_spaces for h, k in grid
    )


def mark_crossover_failed(device: torch.device, family: str) -> None:
    """Suppress further crossover calibration for (device, family) in-process."""
    _crossover_failed.add((_device_key(device), family))


def is_crossover_failed(device: torch.device, family: str) -> bool:
    """True iff crossover calibration already failed for (device, family)."""
    return (_device_key(device), family) in _crossover_failed


def get_cpb_override(
    device: torch.device, family: str, num_heads: int, topk: int, num_tokens: int
) -> Optional[int]:
    """Measured per-shape cpb from :func:`refine_cpb`; None when absent (the
    analytical model's pick governs)."""
    dev_key = _device_key(device)
    table = _cpb_overrides.get(dev_key)
    if table is None:
        _maybe_load_disk()
        table = _cpb_overrides.get(dev_key)
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
) -> None:
    """Merge one refined pick into the disk cache (read-modify-write, atomic
    replace) and the process cache. Same failure semantics as
    :func:`save_constants`."""
    dev_key = _device_key(device)
    entry = {f"{family}|{num_heads}|{topk}|{num_tokens}": int(cpb)}
    _publish_payload(_merge_into_cache(dev_key, _CPB_OVERRIDES_KEY, entry))
    _cpb_overrides.setdefault(dev_key, {}).update(entry)


# ── Public calibration entry point ─────────────────────────────────────────


# Per-family calibration defaults and legality bounds, derived from the
# plan-layer envelope constants. ("<grid heads>", "<grid topks>", min_topk)
def _family_specs() -> dict[str, tuple[tuple[int, ...], tuple[int, ...], int]]:
    from ._sparse_mla_sm120_plan import (
        _CALIBRATION_HEADS,
        _DECODE_DSV3_2_TOPKS,
        _DECODE_DSV4_TOPKS,
        _DECODE_GLM53_NOPE_CALIBRATION_GRID,
        _DECODE_DOTS3_SWA_CALIBRATION_GRID,
        _DECODE_GLM53_NOPE_TOPK,
        _DECODE_DOTS3_SWA_TOPK,
    )

    v32_topks = tuple(sorted(_DECODE_DSV3_2_TOPKS))
    return {
        "dsv4": (_CALIBRATION_HEADS, tuple(sorted(_DECODE_DSV4_TOPKS)), 1),
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


def calibrate_sparse_mla_sm120(
    device: Optional[torch.device] = None,
    *,
    heads: Optional[tuple[int, ...]] = None,
    topks: Optional[tuple[int, ...]] = None,
    families: Optional[tuple[str, ...]] = None,
    force: bool = False,
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
        "dots3_swa"}``; defaults to all. ``dsv3_2`` and ``glm_nsa`` share
        constants and are measured in one sweep (requesting either calibrates
        both key spaces).
    force : bool
        Re-measure entries already present in the cache.

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
    from ._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
    from ._sparse_mla_sm120_plan import _CPB_FAMILY_ALIAS

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

    # Phase 1: cpb constants per cpb family (glm_nsa aliases dsv3_2).
    cpb_families = sorted({_CPB_FAMILY_ALIAS.get(f, f) for f in fams})
    constants: dict[str, CpbConstants] = {}
    for cpb_family in cpb_families:
        existing = None if force else get_constants(device, cpb_family)
        if existing is not None:
            constants_present.append(cpb_family)
            constants[cpb_family] = existing
            continue
        try:
            c = calibrate(_get_sparse_mla_sm120_decode_module, cpb_family, device)
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "SM120 sparse-MLA %s cpb calibration failed (%s)", cpb_family, e
            )
            failed.append(f"{cpb_family} constants: {e}")
            mark_calibration_failed(device, cpb_family)
            continue
        save_constants(device, cpb_family, c)
        constants[cpb_family] = c
        constants_calibrated.append(cpb_family)

    # Phase 2: crossover entries, one sweep per cpb family over the requested
    # pairs that still need measuring. The dsv3_2 sweep writes both the dsv3_2
    # and glm_nsa key spaces, so a request for either covers both.
    for cpb_family in cpb_families:
        c = constants.get(cpb_family)
        if c is None:
            continue  # constants failed above; crossover entries need them
        requested = [f for f in fams if _CPB_FAMILY_ALIAS.get(f, f) == cpb_family]
        pairs: set[tuple[int, int]] = set()
        for fam in requested:
            grid_heads, grid_topks, _ = specs[fam]
            for h in tuple(heads) if heads is not None else grid_heads:
                for k in tuple(topks) if topks is not None else grid_topks:
                    if (
                        not force
                        and get_decode_max_tokens(device, fam, h, k) is not None
                    ):
                        entries_skipped += 1
                    else:
                        pairs.add((h, k))
        if not pairs:
            continue
        try:
            table = calibrate_crossover(
                _get_sparse_mla_sm120_decode_module(),
                device,
                cpb_family,
                c,
                grid_override=sorted(pairs),
            )
        except (CalibrationError, torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                "SM120 sparse-MLA %s crossover calibration failed (%s)",
                cpb_family,
                e,
            )
            failed.append(f"{cpb_family} crossover: {e}")
            mark_crossover_failed(device, cpb_family)
            continue
        save_crossover(device, table)
        # Count a requested entry as calibrated iff its key landed in the
        # table written above (the dsv3_2 sweep writes both key spaces).
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
    )
