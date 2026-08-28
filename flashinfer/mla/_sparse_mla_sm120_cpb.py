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
some GPUs. Instead, six fixed measurements calibrate four hardware constants
once per device (in ``autotune()`` tuning mode), and a closed-form model
picks cpb per call, with an L2-footprint guard rail for the head-tile reuse
window (see :func:`select_cpb`). Without calibrated constants the launcher's
built-in heuristic is used.

The same tuning-mode pass also measures the decode/prefill crossover per
decode-instantiated ``(num_heads, topk)`` config (:func:`calibrate_crossover`)
and persists it as ``decode_max_tokens`` in the same JSON document (schema
version 2; v1 files load constants only). The runtime decode/prefill routing
in :mod:`._sparse_mla_sm120` consults it; absent entries keep the historical
decode-first policy.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

_BI = 64  # chunk width in candidates (BLOCK_SIZE_N)
_HPB = 16  # head tile per block

_SCHEMA_VERSION = 2
# v1 files predate the crossover table: constants load, crossover counts as
# absent and the runtime falls back to the default decode-first policy.
_LOADABLE_SCHEMA_VERSIONS = (1, 2)
_BYTES_PER_TOKEN = {"dsv4": 584, "dsv3_2": 656}
_D_QK = {"dsv4": 512, "dsv3_2": 576}

# Device-level key in the JSON payload holding the crossover table.
_DECODE_MAX_TOKENS_KEY = "decode_max_tokens"
# Probe grid and decode-wins margin for crossover calibration.
_CROSSOVER_PROBED_T = (4, 8, 16, 24, 32, 48, 64)
_CROSSOVER_MARGIN = 0.95

# (num_tokens, num_heads, topk, chunks_per_block); see calibrate().
_MEASUREMENTS = (
    (64, 128, 128, 1),
    (64, 128, 1024, 8),
    (64, 128, 1024, 1),
    (64, 128, 512, 1),
    (1, 8, 1024, 16),
    # Same split count as M4 at half the waves: breaks the (waves, s)
    # collinearity that makes the c0/beta split unidentifiable over M1..M4.
    (32, 128, 512, 1),
)

_POOL_BYTES_TARGET = 2 << 30  # >> L2, so calibration traffic is HBM-faithful
_POOL_BYTES_MIN = 512 << 20
_WARMUP_ITERS = 3
_TIMED_ITERS = 10


class CalibrationError(RuntimeError):
    """Calibration measurements were unusable (OOM or implausible constants)."""


@dataclass(frozen=True)
class CpbConstants:
    """Calibrated hardware constants for one (device, kernel family).

    Attributes
    ----------
    inv_bw : float
        s/byte; inverse aggregate HBM bandwidth.
    inv_rsm : float
        s/byte; inverse single-SM streaming rate (latency-bound regime).
    c0 : float
        s; fixed per-block overhead (Q load, epilogue).
    beta : float
        s per split; merge-kernel cost.
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
    beta: float
    sm_count: int
    bytes_per_chunk: int
    l2_cache_bytes: int


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _num_chunks(topk: int, extra_topk: int) -> int:
    return _ceil_div(topk, _BI) + (_ceil_div(extra_topk, _BI) if extra_topk else 0)


def predict_time_s(
    num_tokens: int,
    num_heads: int,
    topk: int,
    extra_topk: int,
    cpb: int,
    c: CpbConstants,
) -> float:
    """Predicted wall time (seconds) of one decode call at ``cpb``.

    The grid is ``G = T * H_b * ceil(N / cpb)`` blocks, run in
    ``ceil(G / sm_count)`` waves. Each block streams ``cpb`` chunks, pays the
    fixed per-block overhead ``c0``, and each split pays the merge cost
    ``beta``. Per-chunk time is the larger of the bandwidth-bound term (``g``
    concurrent SMs share HBM) and the single-SM latency-bound term.
    """
    h_b = _ceil_div(num_heads, _HPB)
    n = _num_chunks(topk, extra_topk)
    g = num_tokens * h_b * _ceil_div(n, cpb)
    waves = _ceil_div(g, c.sm_count)
    t_c = max(
        c.bytes_per_chunk * min(g, c.sm_count) * c.inv_bw,
        c.bytes_per_chunk * c.inv_rsm,
    )
    return waves * (cpb * t_c + c.c0) + c.beta * _ceil_div(n, cpb)


def select_cpb(
    num_tokens: int, num_heads: int, topk: int, extra_topk: int, c: CpbConstants
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
    n = _num_chunks(topk, extra_topk)
    h_b = _ceil_div(num_heads, _HPB)
    best_cpb, best_t = 1, float("inf")
    for cpb in range(1, n + 1):
        t = predict_time_s(num_tokens, num_heads, topk, extra_topk, cpb, c)
        if t <= best_t:
            best_cpb, best_t = cpb, t
    if not c.l2_cache_bytes:
        return best_cpb
    capped_cpb, capped_t = 0, float("inf")
    for cpb in range(1, n + 1):
        g = num_tokens * h_b * _ceil_div(n, cpb)
        if min(g, c.sm_count) * cpb * c.bytes_per_chunk > c.l2_cache_bytes:
            continue
        t = predict_time_s(num_tokens, num_heads, topk, extra_topk, cpb, c)
        if t <= capped_t:
            capped_cpb, capped_t = cpb, t
    return capped_cpb or best_cpb


def _allocate_kv_pool(family: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """Allocate a ~2 GiB paged KV pool for ``family`` (halved on OOM down to
    512 MiB) and return it with its slot count. The 2-D ``[blocks, bytes]``
    form is accepted by the FFI binding, which derives the block stride from
    the tensor metadata."""
    w = _BI * _BYTES_PER_TOKEN[family]
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
    return kv_cache, kv_cache.shape[0] * _BI


def _time_call(call: Callable[[], None]) -> float:
    """Min CUDA-event wall time (seconds) over warmup + timed iterations."""
    for _ in range(_WARMUP_ITERS):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(_TIMED_ITERS):
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


def _time_call_fresh_indices(
    call: Callable[[torch.Tensor], None],
    num_tokens: int,
    topk: int,
    num_slots: int,
    device: torch.device,
) -> float:
    """Like :func:`_time_call`, but every rep (warmup included) gets a freshly
    drawn full-pool uniform index set.

    ``num_tokens * topk * bytes_per_token`` fits in L2 at these sizes, so
    reusing one index set across reps makes the KV working set L2-resident
    after warmup and understates the HBM-bound steady state — this tainted
    earlier calibration rounds (decode looked artificially fast). The redraw
    runs outside the timed region.
    """

    def fresh() -> torch.Tensor:
        return torch.randint(
            0, num_slots, (num_tokens, topk), dtype=torch.int32, device=device
        )

    for _ in range(_WARMUP_ITERS):
        call(fresh())
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(_TIMED_ITERS):
        indices = fresh()
        start.record()
        call(indices)
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1e3)
    return best


def calibrate(
    module_getter: Callable[[], Any], family: str, device: torch.device
) -> CpbConstants:
    """Calibrate the cpb model constants for ``family`` on ``device``.

    Drives the real decode kernel over a ~2 GiB KV pool (halved on OOM down
    to 512 MiB) with fresh full-pool uniform indices per rep so the measured
    working set stays HBM-resident, then solves the four constants from six
    fixed shapes. ``module_getter`` returns the loaded TVM-FFI kernel module.
    """
    if family not in _BYTES_PER_TOKEN:
        raise ValueError(f"unknown sparse-MLA family {family!r}")
    from ._sparse_mla_sm120 import _MODEL_TYPE_DSV3_2

    device = torch.device(device)
    props = torch.cuda.get_device_properties(device)
    sm_count = int(props.multi_processor_count)
    l2_cache_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    w = _BI * _BYTES_PER_TOKEN[family]
    d_qk = _D_QK[family]

    kv_cache, num_slots = _allocate_kv_pool(family, device)

    module = module_getter()

    def measure(num_tokens: int, num_heads: int, topk: int, cpb: int) -> float:
        num_splits = _ceil_div(topk, _BI)
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
            num_tokens, num_heads, num_splits, 512, dtype=torch.bfloat16, device=device
        )
        mid_lse = torch.empty(
            num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
        )
        output = torch.empty(
            num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device
        )
        out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
        sm_scale = d_qk**-0.5
        if family == "dsv4":

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
                    _MODEL_TYPE_DSV3_2,
                    cpb,
                )

        # Fresh full-pool uniform indices per rep: one fixed index set goes
        # L2-resident after warmup and understates the HBM-bound steady state
        # (the crossover calibration's protocol).
        return _time_call_fresh_indices(call, num_tokens, topk, num_slots, device)

    t = [measure(*m) for m in _MEASUREMENTS]

    def shape_terms(num_tokens: int, num_heads: int, topk: int, cpb: int):
        h_b = _ceil_div(num_heads, _HPB)
        n = _ceil_div(topk, _BI)
        splits = _ceil_div(n, cpb)
        waves = _ceil_div(num_tokens * h_b * splits, sm_count)
        return h_b, n, splits, waves

    # M1 and M2 share waves, split count, and overheads, so their difference
    # isolates the bandwidth term: t2 - t1 = (N2 - N1) * T * H_b * W * inv_bw.
    # The bytes_i * inv_bw residuals below carry a wave-quantization bias
    # (measured grid size g vs ceil-quantized waves) that partially cancels in
    # predict_time_s, which applies the same ceil.
    h_b1, n1, _, _ = shape_terms(*_MEASUREMENTS[0])
    _, n2, _, _ = shape_terms(*_MEASUREMENTS[1])
    inv_bw = (t[1] - t[0]) / ((n2 - n1) * _MEASUREMENTS[0][0] * h_b1 * w)

    # Overheads over the saturated-regime points M1..M4 + M6 (M5 is the
    # latency point): t_i - bytes_i * inv_bw = c0 * waves_i + beta * s_i.
    # M6 shares M4's split count at half its waves, so the (waves, s) rows
    # are not proportional and the c0/beta split is identifiable by design.
    sat = list(range(4)) + [5]
    a_rows, b_rows = [], []
    for i in sat:
        num_tokens, num_heads, topk, cpb = _MEASUREMENTS[i]
        h_b, n, splits, waves = shape_terms(num_tokens, num_heads, topk, cpb)
        a_rows.append((waves, splits))
        b_rows.append(t[i] - num_tokens * h_b * n * w * inv_bw)
    (c0, beta), *_ = np.linalg.lstsq(np.array(a_rows), np.array(b_rows), rcond=None)
    if beta < 0:
        # NNLS active-set step; only a fallback for measurement noise, now
        # that the measurement set is identifiable by design.
        waves_arr = np.array([r[0] for r in a_rows])
        c0 = float(waves_arr @ np.array(b_rows) / (waves_arr @ waves_arr))
        beta = 0.0

    # M5 launches a single block (latency-bound): t5 = waves * (cpb * W *
    # inv_rsm + c0) + beta * splits.
    num_tokens, num_heads, topk, cpb = _MEASUREMENTS[4]
    _, _, splits5, waves5 = shape_terms(num_tokens, num_heads, topk, cpb)
    inv_rsm = (t[4] - waves5 * c0 - beta * splits5) / (waves5 * cpb * w)

    if inv_bw <= 0 or inv_rsm <= 0 or c0 <= 0 or beta < 0:
        raise CalibrationError(
            f"implausible cpb calibration constants for {family}: inv_bw={inv_bw}, "
            f"inv_rsm={inv_rsm}, c0={c0}, beta={beta}"
        )
    return CpbConstants(
        inv_bw=float(inv_bw),
        inv_rsm=float(inv_rsm),
        c0=float(c0),
        beta=float(beta),
        sm_count=sm_count,
        bytes_per_chunk=w,
        l2_cache_bytes=l2_cache_bytes,
    )


def calibrate_crossover(
    module: Any, device: torch.device, family: str, c: CpbConstants
) -> dict[str, int]:
    """Measure the decode/prefill crossover for the decode-instantiated
    configs of ``family`` on ``device``.

    For every instantiated ``(num_heads, topk)`` pair, both paths are timed at
    each probed T with the HBM-faithful protocol of
    :func:`_time_call_fresh_indices`: the decode kernel runs with the model's
    ``select_cpb`` pick; the prefill orchestrator runs with
    ``prefill_impl=auto`` variant choice (swapAB preferred where
    instantiated). Family
    ``"dsv3_2"`` covers both the ``dsv3_2`` and ``glm_nsa`` key spaces because
    the scale format changes prefill speed; the decode kernel is timed with
    the matching ``model_type`` too. A config the prefill envelope does not
    serve (e.g. DSV3_2-family topk != 2048) records ``decode_max_tokens=64``.

    Returns a flat ``{"<family>|<num_heads>|<topk>": decode_max_tokens}``
    table: the largest probed T with ``decode_time <= 0.95 * prefill_time``,
    ``0`` when decode never wins, ``64`` when it wins everywhere probed.
    """
    from ._sparse_mla_sm120 import (
        _DECODE_DSV3_2_DISPATCH,
        _DECODE_DSV4_DISPATCH,
        _MODEL_TYPE_DSV3_2,
        _MODEL_TYPE_DSV4,
        _MODEL_TYPE_GLM_NSA,
    )
    from ._sparse_mla_sm120_plan import (
        _PREFILL_IMPL_AUTO,
        prefill_variant,
    )

    device = torch.device(device)
    if family == "dsv4":
        # (key prefix, instantiation set, FFI model_type)
        spaces = [("dsv4", sorted(_DECODE_DSV4_DISPATCH), _MODEL_TYPE_DSV4)]
    elif family == "dsv3_2":
        spaces = [
            ("dsv3_2", sorted(_DECODE_DSV3_2_DISPATCH), _MODEL_TYPE_DSV3_2),
            ("glm_nsa", sorted(_DECODE_DSV3_2_DISPATCH), _MODEL_TYPE_GLM_NSA),
        ]
    else:
        raise ValueError(f"unknown sparse-MLA family {family!r}")

    d_qk = _D_QK[family]
    sm_scale = d_qk**-0.5
    kv_cache, num_slots = _allocate_kv_pool(family, device)

    def time_decode(
        num_tokens: int, num_heads: int, topk: int, model_type: int
    ) -> float:
        num_splits = _ceil_div(topk, _BI)
        cpb = select_cpb(num_tokens, num_heads, topk, 0, c)
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
            num_tokens, num_heads, num_splits, 512, dtype=torch.bfloat16, device=device
        )
        mid_lse = torch.empty(
            num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
        )
        output = torch.empty(
            num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device
        )
        out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
        if family == "dsv4":

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

        return _time_call_fresh_indices(call, num_tokens, topk, num_slots, device)

    def time_prefill(
        num_tokens: int, num_heads: int, topk: int, model_type: int
    ) -> float:
        # The prefill variant the auto policy would pick; None when the
        # prefill envelope does not serve the shape (e.g. DSV3_2-family
        # topk != 2048).
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
            num_tokens, num_heads, 512, dtype=torch.bfloat16, device=device
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
        return _time_call_fresh_indices(call, num_tokens, topk, num_slots, device)

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
            or payload.get("schema_version") not in _LOADABLE_SCHEMA_VERSIONS
        ):
            return
        devices = payload["devices"]
        if not isinstance(devices, dict):
            return
        # Parse fully into locals first: a mid-document failure must not
        # publish a prefix of the entries while leaving mtime/version stale.
        new_constants: dict = {}
        new_crossover: dict = {}
        for dev_key, families in devices.items():
            if not isinstance(families, dict):
                continue
            for family, raw in families.items():
                if family == _DECODE_MAX_TOKENS_KEY:
                    if isinstance(raw, dict):
                        new_crossover[dev_key] = {
                            str(k): int(v) for k, v in raw.items()
                        }
                    continue
                new_constants[(dev_key, family)] = CpbConstants(**raw)
    except (OSError, ValueError, TypeError, KeyError):
        # Keep mtime unchanged so the next cold call retries.
        return
    _constants.update(new_constants)
    _crossover.update(new_crossover)
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


def save_constants(device: torch.device, family: str, c: CpbConstants) -> None:
    """Merge ``c`` into the disk cache (read-modify-write, atomic replace) and
    the process cache. A failed disk write only loses cross-process sharing;
    the in-process constants still take effect."""
    global _cache_mtime, _constants_version
    key = (_device_key(device), family)
    path = default_cache_path()
    payload: dict = {"schema_version": _SCHEMA_VERSION, "devices": {}}
    try:
        existing = json.loads(path.read_text())
        if isinstance(existing, dict) and existing.get("schema_version") in (
            _LOADABLE_SCHEMA_VERSIONS
        ):
            payload = existing
            payload["schema_version"] = _SCHEMA_VERSION
    except (OSError, ValueError):
        pass
    if not isinstance(payload.get("devices"), dict):
        payload["devices"] = {}
    if not isinstance(payload["devices"].get(key[0]), dict):
        payload["devices"][key[0]] = {}
    payload["devices"][key[0]][family] = asdict(c)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(
            "SM120 sparse-MLA cpb constants not persisted to %s (%s); "
            "using them in-process only.",
            path,
            e,
        )
    _constants[key] = c
    _constants_version += 1
    with contextlib.suppress(OSError):
        _cache_mtime = path.stat().st_mtime


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
    global _cache_mtime, _constants_version
    dev_key = _device_key(device)
    path = default_cache_path()
    payload: dict = {"schema_version": _SCHEMA_VERSION, "devices": {}}
    try:
        existing = json.loads(path.read_text())
        if isinstance(existing, dict) and existing.get("schema_version") in (
            _LOADABLE_SCHEMA_VERSIONS
        ):
            payload = existing
            payload["schema_version"] = _SCHEMA_VERSION
    except (OSError, ValueError):
        pass
    if not isinstance(payload.get("devices"), dict):
        payload["devices"] = {}
    dev = payload["devices"].setdefault(dev_key, {})
    if not isinstance(dev, dict):
        dev = payload["devices"][dev_key] = {}
    xo = dev.setdefault(_DECODE_MAX_TOKENS_KEY, {})
    if not isinstance(xo, dict):
        xo = dev[_DECODE_MAX_TOKENS_KEY] = {}
    xo.update(table)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(
            "SM120 sparse-MLA crossover table not persisted to %s (%s); "
            "using it in-process only.",
            path,
            e,
        )
    _crossover.setdefault(dev_key, {}).update(table)
    _constants_version += 1
    with contextlib.suppress(OSError):
        _cache_mtime = path.stat().st_mtime


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
    spaces, so both must be present."""
    dev_key = _device_key(device)
    table = _crossover.get(dev_key)
    if table is None:
        _maybe_load_disk()
        table = _crossover.get(dev_key)
    if not table:
        return False
    prefixes = ("dsv3_2|", "glm_nsa|") if family == "dsv3_2" else (f"{family}|",)
    return all(any(k.startswith(p) for k in table) for p in prefixes)


def mark_crossover_failed(device: torch.device, family: str) -> None:
    """Suppress further crossover calibration for (device, family) in-process."""
    _crossover_failed.add((_device_key(device), family))


def is_crossover_failed(device: torch.device, family: str) -> bool:
    """True iff crossover calibration already failed for (device, family)."""
    return (_device_key(device), family) in _crossover_failed
