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

"""Tests for the calibrated analytical chunks_per_block model (SM120 sparse MLA)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from flashinfer.mla import _sparse_mla_sm120_dsv4_nvfp4_policy as native

import pytest
import torch

from flashinfer.mla import _sparse_mla_sm120_calibration as cpb_mod
from flashinfer.mla._sparse_mla_sm120_calibration import (
    CpbConstants,
    predict_time_s,
    select_cpb,
)
from flashinfer.utils import is_sm12x_supported

requires_sm12x = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x.",
)

# Plausible RTX-PRO-6000-class constants for the pure-model tests.
_C = CpbConstants(
    inv_bw=1e-12,
    inv_rsm=1e-10,
    c0=1e-6,
    sm_count=148,
    bytes_per_chunk=37376,
    l2_cache_bytes=0,  # guard rail disabled unless a test sets it
)


def test_model_regime_behavior() -> None:
    """Small grids prefer modest cpb (latency-bound); saturated grids prefer
    larger cpb (bandwidth amortizes per-block overhead)."""
    small = select_cpb(1, 16, 1024, 0, _C)
    large = select_cpb(64, 128, 1024, 0, _C)
    assert small < 16
    assert large > small


def test_select_cpb_tail_imbalance_sawtooth() -> None:
    """Mid-cpb sawtooth: at T=8/H=128/topk=1024 (N=16) the 7+7+2 split must
    beat 8+8 — one heavy round plus a short tail that fills freed SMs beats a
    second full chunk round. The retired ceil-wave form priced this backwards
    (kernel-bench v6: the model picked 8, the sweep's best is 7)."""
    c = CpbConstants(
        inv_bw=5e-13,
        inv_rsm=1.2e-10,
        c0=6.4e-6,
        sm_count=188,
        bytes_per_chunk=37376,
        l2_cache_bytes=0,
    )
    assert predict_time_s(8, 128, 1024, 0, 7, c) < predict_time_s(8, 128, 1024, 0, 8, c)
    assert select_cpb(8, 128, 1024, 0, c) == 7


def test_select_cpb_bounds() -> None:
    for num_tokens, num_heads, topk, extra_topk in [
        (8, 64, 512, 0),
        (64, 128, 512, 256),
    ]:
        n = -(-topk // 64) + (-(-extra_topk // 64) if extra_topk else 0)
        cpb = select_cpb(num_tokens, num_heads, topk, extra_topk, _C)
        assert 1 <= cpb <= n


def test_select_cpb_tie_prefers_larger() -> None:
    """Exactly tied predictions resolve to the larger cpb."""
    c = CpbConstants(
        inv_bw=0.0,
        inv_rsm=0.5,
        c0=0.0,
        sm_count=148,
        bytes_per_chunk=1,
        l2_cache_bytes=0,
    )
    # T=148, H=16 (one head tile), topk=128 (N=2): cpb=1 runs two 0.5 s
    # rounds of 148 blocks, cpb=2 one 1.0 s round — both exactly 1.0 s.
    t1 = predict_time_s(148, 16, 128, 0, 1, c)
    t2 = predict_time_s(148, 16, 128, 0, 2, c)
    assert t1 == t2
    assert select_cpb(148, 16, 128, 0, c) == 2


def test_select_cpb_l2_guard_rail() -> None:
    """Candidates whose concurrent streaming footprint exceeds L2 are
    excluded; if none fit, fall back to the unconstrained argmin."""
    shape = (64, 128, 3200, 0)  # N=50, saturated grid
    uncapped = select_cpb(*shape, _C)
    assert uncapped == 24
    # L2 = 10 * S * W: allowed footprint caps cpb at 10, and the best
    # fitting candidate is 8.
    capped = CpbConstants(**{**_C.__dict__, "l2_cache_bytes": 148 * 10 * 37376})
    assert select_cpb(*shape, capped) == 8
    # Cap above the unconstrained pick's footprint: no change.
    loose = CpbConstants(**{**_C.__dict__, "l2_cache_bytes": 148 * 30 * 37376})
    assert select_cpb(*shape, loose) == uncapped
    # Nothing fits: fall back to the unconstrained argmin.
    tiny = CpbConstants(**{**_C.__dict__, "l2_cache_bytes": 1})
    assert select_cpb(*shape, tiny) == uncapped


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path))
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: str(device))
    monkeypatch.setattr(cpb_mod, "_store_states", {})
    monkeypatch.setattr(cpb_mod, "_active_scope", None)
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version)
    for name in (
        "_constants",
        "_crossover",
        "_cpb_overrides",
        "_failed",
        "_crossover_failed",
    ):
        monkeypatch.setattr(
            cpb_mod, name, set() if name in ("_failed", "_crossover_failed") else {}
        )
    return torch.device("cpu:0")


@pytest.fixture
def isolated_cpb(clean_cpb_state, monkeypatch):
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: "0:Test GPU")
    return cpb_mod.default_cache_path()


@pytest.fixture
def clean_cpb_state(store, monkeypatch, tmp_path):
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: "0:Fake GPU")
    return tmp_path


def _skip_if_low_vram(needed_gib: int) -> None:
    """Skip when the GPU cannot fit the multi-GiB KV pool (mirrors the
    torch.cuda.mem_get_info precedent in test_mla_decode_kernel.py)."""
    if torch.cuda.mem_get_info(torch.device("cuda"))[0] < needed_gib * (1 << 30):
        pytest.skip(f"needs >= {needed_gib} GiB free VRAM for the KV pool")


@pytest.fixture(scope="module")
def dsv4_constants() -> CpbConstants:
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
    )

    _skip_if_low_vram(3)  # 2 GiB calibration pool plus headroom
    return cpb_mod.calibrate(
        _get_sparse_mla_sm120_decode_module, "dsv4", torch.device("cuda")
    )


@pytest.fixture(scope="module", params=["dsv4", "glm53_nope", "dots3_swa"])
def family_constants(request: pytest.FixtureRequest) -> tuple[str, CpbConstants]:
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
    )

    _skip_if_low_vram(3)  # 2 GiB calibration pool plus headroom
    return request.param, cpb_mod.calibrate(
        _get_sparse_mla_sm120_decode_module, request.param, torch.device("cuda")
    )


@requires_sm12x
def test_calibration_smoke(family_constants: tuple[str, CpbConstants]) -> None:
    family, c = family_constants
    assert c.inv_bw > 0 and c.inv_rsm > 0 and c.c0 > 0
    assert (
        c.sm_count
        == torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    )
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    if getattr(props, "L2_cache_size", None):
        expected_l2 = props.L2_cache_size
        if getattr(props, "is_integrated", 0):
            expected_l2 //= 2  # calibrate() halves the rail window on SoCs
        assert c.l2_cache_bytes == expected_l2
    bw_gbps = 1.0 / c.inv_bw / 1e9
    print(f"\ncalibrated {family} constants: {c}")
    print(f"implied aggregate DRAM bandwidth: {bw_gbps:.0f} GB/s")
    # Loose physical-plausibility band around modern datacenter GPUs.
    assert 100 < bw_gbps < 20000


@requires_sm12x
@pytest.mark.parametrize("topk", [128, 1024])
@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_model_cpb_accuracy_guard(
    dsv4_constants: CpbConstants, num_tokens, topk
) -> None:
    """Model-picked cpb is within 1.25x of the best swept cpb, measured with
    the calibration timing protocol (queued batches, L2-cold indices) so the
    guard certifies the regime production calibration runs in."""
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
    )

    _skip_if_low_vram(3)  # 2 GiB pool plus headroom
    device = torch.device("cuda")
    c = dsv4_constants
    num_heads = 128
    num_splits = -(-topk // 64)
    kv_cache, num_slots = cpb_mod._allocate_kv_pool("dsv4", device)
    build = cpb_mod._make_decode_call_builder(
        _get_sparse_mla_sm120_decode_module(), "dsv4", device, kv_cache
    )

    def run(cpb_override: int) -> float:
        # model_type is unused by the dsv4 FFI branch of the call builder.
        call = build(num_tokens, num_heads, topk, 0, cpb_override)
        return cpb_mod._time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, c.bytes_per_chunk // 64
        )

    swept = {cpb: run(cpb) for cpb in range(1, num_splits + 1)}
    heuristic_t = run(-1)
    model_cpb = select_cpb(num_tokens, num_heads, topk, 0, c)
    model_t = swept[model_cpb]
    best_cpb = min(swept, key=swept.get)
    best_t = swept[best_cpb]
    print(
        f"\nT={num_tokens} H={num_heads} topk={topk}: "
        f"model cpb={model_cpb} {model_t * 1e6:.1f} us | "
        f"best cpb={best_cpb} {best_t * 1e6:.1f} us | "
        f"heuristic {heuristic_t * 1e6:.1f} us"
    )
    assert model_t <= 1.25 * best_t

    # Release the 2 GiB pool before the next parametrized case.
    kv_cache = None
    torch.cuda.empty_cache()


@requires_sm12x
@pytest.mark.parametrize(
    "topk,extra_topk,num_tokens",
    [(128, 2176, 1), (128, 2176, 64), (1024, 2176, 64)],
)
def test_model_cpb_accuracy_guard_dual_cache(
    dsv4_constants: CpbConstants, topk, extra_topk, num_tokens
) -> None:
    """Dual-cache (C128A: pbs_extra=2): model cpb within 1.25x of the best
    swept cpb. The (1024, 2176, 64) case is the kernel-bench v2 regression
    (N=50 chunks; L2-thrash at cpb=N)."""
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
    )

    _skip_if_low_vram(5)  # 2 GiB main pool + 2 GiB extra pool plus headroom
    device = torch.device("cuda")
    module = _get_sparse_mla_sm120_decode_module()
    c = dsv4_constants
    d_qk, d_v = 512, 512
    num_heads, pbs_extra = 128, 2

    num_splits = -(-topk // 64) + -(-extra_topk // 64)
    w = c.bytes_per_chunk
    bpt = w // 64
    kv_cache = torch.empty((2 << 30) // w, w, dtype=torch.uint8, device=device)
    extra_kv_cache = torch.empty(
        (2 << 30) // (pbs_extra * bpt),
        pbs_extra * bpt,
        dtype=torch.uint8,
        device=device,
    )
    cpb_mod._initialize_fp8_pool(kv_cache, "dsv4", 64)
    cpb_mod._initialize_fp8_pool(extra_kv_cache, "dsv4", pbs_extra)

    q = (torch.randn(num_tokens, num_heads, d_qk, device=device) / 10.0).to(
        torch.bfloat16
    )
    extra_indices = torch.randint(
        0,
        extra_kv_cache.shape[0] * pbs_extra,
        (num_tokens, extra_topk),
        dtype=torch.int32,
        device=device,
    )
    mid_out = torch.empty(
        num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
    )
    output = torch.empty(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)
    sm_scale = d_qk**-0.5

    def run(cpb_override: int) -> float:
        # The timing helper rotates fresh main-cache indices per call; the
        # extra-cache set stays fixed (within-row uniform across candidates).
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
                extra_kv_cache,
                extra_indices,
                None,
                -1,  # model_type: legacy width inference (d_qk=512 -> DSV4)
                cpb_override,
                False,  # extra_fp4
            )

        return cpb_mod._time_call_fresh_indices(
            call, num_tokens, topk, kv_cache.shape[0] * 64, device, bpt
        )

    swept = {cpb: run(cpb) for cpb in range(1, num_splits + 1)}
    heuristic_t = run(-1)
    model_cpb = select_cpb(num_tokens, num_heads, topk, extra_topk, c)
    model_t = swept[model_cpb]
    best_cpb = min(swept, key=swept.get)
    best_t = swept[best_cpb]
    print(
        f"\nT={num_tokens} H={num_heads} topk={topk} extra_topk={extra_topk}: "
        f"model cpb={model_cpb} {model_t * 1e6:.1f} us | "
        f"best cpb={best_cpb} {best_t * 1e6:.1f} us | "
        f"heuristic {heuristic_t * 1e6:.1f} us"
    )
    assert model_t <= 1.25 * best_t

    # Release the pools before the next parametrized case.
    kv_cache = extra_kv_cache = None
    torch.cuda.empty_cache()


@requires_sm12x
def test_refine_cpb_beats_or_matches_model(
    monkeypatch, tmp_path, dsv4_constants: CpbConstants
) -> None:
    """refine_cpb's measured pick never loses to the model pick under the same
    timing protocol, persists to disk, and _resolve_cpb serves the override
    ahead of the model (tuning mode off)."""
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
        _resolve_cpb,
    )

    _skip_if_low_vram(3)  # 2 GiB pool plus headroom
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path))
    device = torch.device("cuda")
    c = dsv4_constants
    num_tokens, num_heads, topk = 64, 128, 1024

    refined = cpb_mod.refine_cpb(
        _get_sparse_mla_sm120_decode_module,
        "dsv4",
        device,
        c,
        num_tokens,
        num_heads,
        topk,
    )
    model_cpb = select_cpb(num_tokens, num_heads, topk, 0, c)

    module = _get_sparse_mla_sm120_decode_module()
    kv_cache, num_slots = cpb_mod._allocate_kv_pool("dsv4", device)
    build = cpb_mod._make_decode_call_builder(module, "dsv4", device, kv_cache)

    def timed(cpb: int) -> float:
        # model_type is unused by the dsv4 FFI branch of the call builder.
        call = build(num_tokens, num_heads, topk, 0, cpb)
        return cpb_mod._time_call_fresh_indices(
            call, num_tokens, topk, num_slots, device, c.bytes_per_chunk // 64
        )

    t_refined, t_model = timed(refined), timed(model_cpb)
    print(
        f"\nrefine: model cpb={model_cpb} {t_model * 1e6:.1f} us | "
        f"refined cpb={refined} {t_refined * 1e6:.1f} us"
    )
    assert t_refined <= t_model * 1.05

    cpb_mod.save_constants(device, "dsv4", c)
    cpb_mod.save_cpb_override(device, "dsv4", num_heads, topk, num_tokens, refined)
    cpb_mod._cpb_overrides.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert (
        cpb_mod.get_cpb_override(device, "dsv4", num_heads, topk, num_tokens) == refined
    )
    assert _resolve_cpb(device, "dsv4", num_tokens, num_heads, topk, 0) == refined
    # An unrefined shape still falls through to the model pick.
    assert _resolve_cpb(
        device, "dsv4", num_tokens + 1, num_heads, topk, 0
    ) == select_cpb(num_tokens + 1, num_heads, topk, 0, c)

    # Do not leak process state into later tests.
    kv_cache = None
    torch.cuda.empty_cache()
    cpb_mod._constants.clear()
    cpb_mod._cpb_overrides.clear()


@requires_sm12x
def test_model_path_dual_cache_wiring(monkeypatch, tmp_path) -> None:
    """Public wrapper + extra cache + injected constants: select_cpb's cpb
    (covering the extra chunks) reaches the kernel as cpb_override, with
    num_splits spanning both index sets."""
    from types import SimpleNamespace

    from flashinfer.mla import _sparse_mla_sm120 as sm

    device = torch.device("cuda")
    num_tokens, num_heads = 2, 16
    topk, extra_topk = 128, 2176
    d_qk, d_v = 512, 512
    num_splits = -(-topk // 64) + -(-extra_topk // 64)

    kv_cache = torch.empty(256, 64 * 584, dtype=torch.uint8, device=device)
    extra_kv_cache = torch.empty(
        (extra_topk + 1) // 2, 2 * 584, dtype=torch.uint8, device=device
    )
    cpb_mod._initialize_fp8_pool(kv_cache, "dsv4", 64)
    cpb_mod._initialize_fp8_pool(extra_kv_cache, "dsv4", 2)
    q = torch.randn(num_tokens, num_heads, d_qk, device=device).to(torch.bfloat16)
    indices = torch.randint(
        0, kv_cache.shape[0] * 64, (num_tokens, topk), dtype=torch.int32, device=device
    )
    extra_indices = torch.randint(
        0,
        extra_kv_cache.shape[0] * 2,
        (num_tokens, extra_topk),
        dtype=torch.int32,
        device=device,
    )
    mid_out = torch.empty(
        num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
    )
    output = torch.empty(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    # Inject calibrated constants so the wrapper takes the model path (no
    # tuning mode active); save_constants also bumps the memo version.
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path))
    cpb_mod.save_constants(device, "dsv4", _C)

    real_module = sm._get_sparse_mla_sm120_decode_module()
    real_call = real_module.sparse_mla_sm120_decode_dsv4
    recorded = {}

    def spy(*args):
        recorded["num_splits"] = args[7]
        recorded["cpb_override"] = args[-2]  # trailing arg is extra_fp4
        return real_call(*args)

    monkeypatch.setattr(
        sm,
        "_get_sparse_mla_sm120_decode_module",
        lambda: SimpleNamespace(sparse_mla_sm120_decode_dsv4=spy),
    )

    sm.sparse_mla_sm120_decode_dsv4(
        q,
        kv_cache,
        indices,
        mid_out,
        mid_lse,
        output,
        out_lse,
        d_qk**-0.5,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
    )

    assert recorded["num_splits"] == num_splits
    assert recorded["cpb_override"] == select_cpb(
        num_tokens, num_heads, topk, extra_topk, _C
    )


@requires_sm12x
def test_glm_nsa_decode_uses_dsv3_2_cpb_family(clean_cpb_state, monkeypatch) -> None:
    """GLM_NSA decode shares the dsv3_2 cpb constants (same kernel and ABI):
    non-tuning picks up injected dsv3_2 constants, and tuning mode calibrates
    under the "dsv3_2" family key instead of raising ValueError on the
    unknown "glm_nsa" key. Crossover keys stay glm_nsa-flavored (produced by
    the dsv3_2 crossover calibration, which covers both key spaces)."""
    from types import SimpleNamespace

    from flashinfer.autotuner import AutoTuner
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _MODEL_TYPE_GLM_NSA

    device = torch.device("cuda")
    num_tokens, num_heads, topk = 2, 64, 2048
    d_qk, d_v = 576, 512
    num_splits = -(-topk // 64)

    kv_cache = torch.empty(256, 64 * 656, dtype=torch.uint8, device=device)
    cpb_mod._initialize_fp8_pool(kv_cache, "dsv3_2", 64)
    q = torch.randn(num_tokens, num_heads, d_qk, device=device).to(torch.bfloat16)
    indices = torch.randint(
        0, kv_cache.shape[0] * 64, (num_tokens, topk), dtype=torch.int32, device=device
    )
    mid_out = torch.empty(
        num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
    )
    output = torch.empty(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    real_module = sm._get_sparse_mla_sm120_decode_module()
    real_call = real_module.sparse_mla_sm120_decode_dsv3_2
    recorded = {}

    def spy(*args):
        recorded["cpb_override"] = args[-1]
        return real_call(*args)

    monkeypatch.setattr(
        sm,
        "_get_sparse_mla_sm120_decode_module",
        lambda: SimpleNamespace(sparse_mla_sm120_decode_dsv3_2=spy),
    )

    def call() -> None:
        sm.sparse_mla_sm120_decode_dsv3_2(
            q,
            kv_cache,
            indices,
            mid_out,
            mid_lse,
            output,
            out_lse,
            d_qk**-0.5,
            model_type=_MODEL_TYPE_GLM_NSA,
        )

    expected_cpb = select_cpb(num_tokens, num_heads, topk, 0, _C)

    # Non-tuning: injected dsv3_2 constants drive the GLM_NSA decode call.
    cpb_mod.save_constants(device, "dsv3_2", _C)
    call()
    assert recorded["cpb_override"] == expected_cpb

    # Tuning mode with an empty cache: calibrate + crossover calibrate run
    # under the dsv3_2 family key, not the unknown glm_nsa key.
    cpb_mod._constants.clear()
    cpb_mod._crossover.clear()
    seen = {}

    def fake_calibrate(module_getter, family, dev):
        seen["calibrate"] = family
        return _C

    def fake_calibrate_crossover(module, dev, family, c):
        seen["calibrate_crossover"] = family
        return {"dsv3_2|64|2048": 32, "glm_nsa|64|2048": 32}

    def fake_refine(module_getter, family, dev, c, t, h, k):
        seen["refine_cpb"] = family
        return select_cpb(t, h, k, 0, c)

    monkeypatch.setattr(cpb_mod, "calibrate", fake_calibrate)
    monkeypatch.setattr(cpb_mod, "calibrate_crossover", fake_calibrate_crossover)
    monkeypatch.setattr(cpb_mod, "refine_cpb", fake_refine)
    monkeypatch.setattr(AutoTuner.get(), "is_tuning_mode", True)
    call()
    assert seen == {
        "calibrate": "dsv3_2",
        "calibrate_crossover": "dsv3_2",
        "refine_cpb": "dsv3_2",
    }
    assert recorded["cpb_override"] == expected_cpb


# ── Public calibration API (calibrate_sparse_mla_sm120) ────────────────────


def test_public_calibrate_validates_envelope(clean_cpb_state) -> None:
    """Unknown families raise before any measurement (envelope misses are
    covered by the listing test below)."""
    import flashinfer.mla

    calibrate = flashinfer.mla.calibrate_sparse_mla_sm120  # lazy export resolves
    with pytest.raises(ValueError, match="unknown sparse-MLA families") as exc:
        calibrate(
            torch.device("cpu"),
            families=("dsv4", "dots3_swa", "bogus_family"),
        )
    assert "bogus_family" in str(exc.value)


def test_public_calibrate_lists_all_invalid_combinations(clean_cpb_state) -> None:
    import flashinfer.mla

    with pytest.raises(ValueError) as exc:
        flashinfer.mla.calibrate_sparse_mla_sm120(
            torch.device("cpu"),
            families=("dots3_swa",),
            heads=(64, 256),
            topks=(512, 576),
        )
    msg = str(exc.value)
    assert "num_heads=256" in msg and "topk=512" in msg and "topk>=513" in msg
    # The legal combination (64, 576) must not be blamed.
    assert "(num_heads=64, topk=576)" not in msg


@requires_sm12x
def test_public_calibrate_offgrid_shape_idempotent_force(clean_cpb_state) -> None:
    """End-to-end on GPU: calibrate an off-grid shape, check the disk entry,
    idempotent second call, force re-measure, and in-process plan() pickup."""
    import json

    import flashinfer.mla
    from flashinfer.mla import _sparse_mla_sm120_policy as plan_mod

    device = torch.device("cuda")
    calibrate = flashinfer.mla.calibrate_sparse_mla_sm120

    # A grid shape both envelopes serve, so a measured crossover < 64 must
    # flip plan() routing in-process (no restart, no manual memo clear).
    t_probe, h_probe, k_probe = 48, 64, 512
    planned = plan_mod.plan(
        t_probe,
        h_probe,
        k_probe,
        plan_mod._MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        device,
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.DECODE_SPLITK  # decode-first

    report = calibrate(device, families=("dsv4",), heads=(64, 80), topks=(384, 512))
    assert report.failed == ()
    assert report.constants_calibrated == ("dsv4",)
    assert report.entries_calibrated == 4
    assert report.entries_skipped == 0
    assert report.elapsed_s > 0

    # Disk: the JSON document carries every requested entry, incl. the
    # off-grid (80, 384) one.
    payload = json.loads((clean_cpb_state / "sparse_mla_sm120_cpb.json").read_text())
    xo = payload["devices"]["0:Fake GPU"]["decode_max_tokens"]
    for h in (64, 80):
        for k in (384, 512):
            assert f"dsv4|{h}|{k}" in xo
    # Off-grid H=80: the prefill envelope has no H=80 instantiation, so
    # decode always wins by construction. Off-grid topk=384 at H=64 is
    # prefill-served (topk is a runtime kernel argument), so its entry is a
    # measured crossover in [0, 64], not forced.
    assert xo["dsv4|80|384"] == 64
    assert xo["dsv4|80|512"] == 64
    assert 0 <= xo["dsv4|64|384"] <= 64

    # The (64, 512) entry must be a real measured crossover below 64 on this
    # GPU class, and plan() must read the new crossover on this call.
    dmt = cpb_mod.get_decode_max_tokens(device, "dsv4", 64, 512)
    assert dmt is not None and dmt < 64
    planned = plan_mod.plan(
        t_probe,
        h_probe,
        k_probe,
        plan_mod._MODEL_TYPE_DSV4,
        64,
        False,
        plan_mod._PREFILL_IMPL_AUTO,
        device,
    )
    assert planned is not None
    assert planned.variant is plan_mod.KernelVariant.PREFILL_MG

    # Idempotent second call: everything already present.
    report2 = calibrate(device, families=("dsv4",), heads=(64, 80), topks=(384, 512))
    assert report2.failed == ()
    assert report2.constants_present == ("dsv4",)
    assert report2.constants_calibrated == ()
    assert report2.entries_calibrated == 0
    assert report2.entries_skipped == 4

    # force re-measures present entries.
    report3 = calibrate(
        device, families=("dsv4",), heads=(64, 80), topks=(384, 512), force=True
    )
    assert report3.failed == ()
    assert report3.constants_calibrated == ("dsv4",)
    assert report3.entries_calibrated == 4
    assert report3.entries_skipped == 0


def test_public_calibrate_report_type_is_public(clean_cpb_state) -> None:
    """The report class rides the same lazy export as the other public names."""
    import flashinfer.mla

    assert "calibrate_sparse_mla_sm120" in dir(flashinfer.mla)
    assert "SparseMLASm120CalibrationReport" in dir(flashinfer.mla)
    from flashinfer.mla._sparse_mla_sm120_calibration import (
        SparseMLASm120CalibrationReport,
    )

    assert (
        flashinfer.mla.SparseMLASm120CalibrationReport
        is SparseMLASm120CalibrationReport
    )


@requires_sm12x
def test_tuning_calibration_honors_skip_ops(clean_cpb_state, monkeypatch) -> None:
    """autotune(skip_ops={"sparse_mla_sm120"}) opts out of the lazy
    calibration passes, not only of choose_one."""
    from flashinfer.autotuner import autotune
    from flashinfer.mla import _sparse_mla_sm120 as sm

    device = torch.device("cuda")
    num_tokens, num_heads, topk = 2, 64, 512
    d_qk, d_v = 512, 512
    num_splits = -(-topk // 64)

    kv_cache = torch.empty(256, 64 * 584, dtype=torch.uint8, device=device)
    cpb_mod._initialize_fp8_pool(kv_cache, "dsv4", 64)
    q = torch.randn(num_tokens, num_heads, d_qk, device=device).to(torch.bfloat16)
    indices = torch.randint(
        0, kv_cache.shape[0] * 64, (num_tokens, topk), dtype=torch.int32, device=device
    )
    mid_out = torch.empty(
        num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
    )
    output = torch.empty(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=device)

    seen = {}

    def fake_calibrate(module_getter, family, dev):
        seen["calibrate"] = family
        return _C

    def fake_calibrate_crossover(module, dev, family, c):
        seen["calibrate_crossover"] = family
        return {}

    def fake_refine(module_getter, family, dev, c, t, h, k):
        seen["refine_cpb"] = family
        return 1

    monkeypatch.setattr(cpb_mod, "calibrate", fake_calibrate)
    monkeypatch.setattr(cpb_mod, "calibrate_crossover", fake_calibrate_crossover)
    monkeypatch.setattr(cpb_mod, "refine_cpb", fake_refine)

    def call() -> None:
        sm.sparse_mla_sm120_decode_dsv4(
            q, kv_cache, indices, mid_out, mid_lse, output, out_lse, d_qk**-0.5
        )

    with autotune(True, skip_ops={"sparse_mla_sm120"}):
        call()
    assert seen == {}

    with autotune(True):
        call()
    assert seen == {
        "calibrate": "dsv4",
        "calibrate_crossover": "dsv4",
        "refine_cpb": "dsv4",
    }


def test_single_cache_override_does_not_leak_into_dual(monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_policy as plan

    constants = cpb_mod.CpbConstants(
        inv_bw=1e-12,
        inv_rsm=1e-10,
        c0=1e-6,
        sm_count=148,
        l2_cache_bytes=0,
        bytes_per_chunk=64 * 584,
    )
    monkeypatch.setattr(cpb_mod, "get_constants", lambda *args: constants)
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: "scope-test")
    monkeypatch.setattr(plan, "_cpb_hot_cache", {})
    reads, picks = [], []

    def override(*args):
        reads.append(args)
        return 7

    def model_pick(*args, **kwargs):
        picks.append(args[3])
        return 2

    monkeypatch.setattr(cpb_mod, "get_cpb_override", override)
    monkeypatch.setattr(cpb_mod, "select_cpb", model_pick)
    for _ in range(2):
        assert plan._resolve_cpb(torch.device("cpu"), "dsv4", 4, 32, 512, 0) == 7
        assert plan._resolve_cpb(torch.device("cpu"), "dsv4", 4, 32, 512, 128) == 2
    assert len(reads) == 1 and picks == [128]


def test_finite_synthetic_inline_and_footer(monkeypatch):
    for inline, nope, rope, count, width in [
        (1, 512, 64, 4, 656),
        (0, 448, 64, 7, 584),
        (0, 512, 0, 16, 528),
    ]:
        facts = dict(
            inline_scale=inline,
            nope_dim=nope,
            rope_dim=rope,
            num_scales=count,
            scale_bytes=count * 4 if inline else (count + 7) // 8 * 8,
            data_bytes=nope if inline else nope + rope * 2,
            rope_offset=nope + count * 4 if inline else nope,
            bytes_per_token=width,
        )
        monkeypatch.setattr(cpb_mod, "format_info", lambda model: facts)
        monkeypatch.setattr(cpb_mod, "_SAMPLE_CHUNK_BYTES", 65536)
        cache = torch.full((9, 64 * width), 255, dtype=torch.uint8)
        state = torch.random.get_rng_state().clone()
        cpb_mod._initialize_fp8_pool(cache, "dsv4", 64)
        torch.testing.assert_close(state, torch.random.get_rng_state())
        data = (
            cache.view(9, 64, width)
            if inline
            else cache[:, : 64 * facts["data_bytes"]].reshape(
                9, 64, facts["data_bytes"]
            )
        )
        values = data[..., :nope].contiguous().view(torch.float8_e4m3fn).float()
        assert torch.isfinite(values).all() and values.std() > 0.1
        assert not torch.equal(data[0], data[1])
        scales = (
            data[..., nope : nope + count * 4].contiguous().view(torch.float32)
            if inline
            else cache[:, 64 * facts["data_bytes"] :].reshape(
                9, 64, facts["scale_bytes"]
            )[..., :count]
        )
        assert torch.isfinite(scales.float()).all() and (scales > 0).all()
        if rope:
            r = (
                data[..., facts["rope_offset"] : facts["rope_offset"] + rope * 2]
                .contiguous()
                .view(torch.bfloat16)
            )
            assert torch.isfinite(r).all() and r.float().std() > 0.1
        if not inline and facts["scale_bytes"] > count:
            assert (
                cache[:, 64 * facts["data_bytes"] :].reshape(
                    9, 64, facts["scale_bytes"]
                )[..., count:]
                == 0
            ).all()


def test_memory_regime_requires_known_l2_and_sufficient_pool(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(L2_cache_size=1024),
    )
    with pytest.raises(cpb_mod.CalibrationError, match="L2"):
        cpb_mod._check_pool_capacity(torch.device("cpu"), 1024)
    cpb_mod._check_pool_capacity(torch.device("cpu"), 4096)
    with pytest.raises(cpb_mod.CalibrationError, match="reuse"):
        cpb_mod.calibration_batch_count(1, 1, 1, torch.device("cpu"), max_batch_calls=8)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda dev: SimpleNamespace()
    )
    with pytest.raises(cpb_mod.CalibrationError, match="L2"):
        cpb_mod.calibration_batch_count(1, 128, 584, torch.device("cpu"))


def test_initializer_oom_retries_but_non_oom_propagates(monkeypatch):
    monkeypatch.setattr(cpb_mod, "_BYTES_PER_TOKEN", {"dsv4": 8})
    monkeypatch.setattr(cpb_mod, "_POOL_BYTES_TARGET", 4096)
    monkeypatch.setattr(cpb_mod, "_POOL_BYTES_MIN", 1024)
    monkeypatch.setattr(cpb_mod, "_check_pool_capacity", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = []

    def initialize(cache, *args):
        calls.append(cache.numel())
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("synthetic generation")
        cache.zero_()

    monkeypatch.setattr(cpb_mod, "_initialize_fp8_pool", initialize)
    result, slots = cpb_mod._allocate_kv_pool("dsv4", torch.device("cpu"))
    assert calls == [4096, 2048] and result.numel() == 2048 and slots == 256
    monkeypatch.setattr(
        cpb_mod, "_initialize_fp8_pool", Mock(side_effect=RuntimeError("bad pack"))
    )
    with pytest.raises(RuntimeError, match="bad pack"):
        cpb_mod._allocate_kv_pool("dsv4", torch.device("cpu"))


def test_all_calibration_entries_reject_target_capture_before_work(monkeypatch):
    state = [0]

    @contextmanager
    def device(target):
        previous = state[0]
        state[0] = target.index
        try:
            yield
        finally:
            state[0] = previous

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state[0] == 1
    )
    monkeypatch.setattr(
        cpb_mod, "get_constants", Mock(side_effect=AssertionError("cache read"))
    )
    target = torch.device("cuda:1")
    for call in [
        lambda: cpb_mod.calibrate(None, "dsv4", target),
        lambda: cpb_mod.calibrate_crossover(None, target, "dsv4", None),
        lambda: cpb_mod.refine_cpb(None, "dsv4", target, None, 1, 8, 128),
        lambda: cpb_mod.calibrate_sparse_mla_sm120(target),
        lambda: native.calibrate_nvfp4_sparse_mla_sm120(target, num_heads=16, topk=128),
    ]:
        with pytest.raises(cpb_mod.CalibrationError, match="capture"):
            call()
        assert state == [0]

    from flashinfer.mla import _sparse_mla_sm120_policy as policy

    tuner = SimpleNamespace(is_tuning_mode=True, _get_skip_ops_stack=lambda: [])
    monkeypatch.setattr(policy.AutoTuner, "get", lambda: tuner)
    monkeypatch.setattr(cpb_mod, "get_constants", lambda *args: None)
    monkeypatch.setattr(
        cpb_mod, "calibrate", Mock(side_effect=AssertionError("lazy launch"))
    )
    assert policy._resolve_cpb(target, "dsv4", 4, 16, 128, 0) == -1
    monkeypatch.setattr(native, "_autotune_skipped", lambda: False)
    native._maybe_calibrate(
        device=target,
        family="native",
        num_heads=16,
        topk=128,
        primary_page_size=64,
        extra_topk=0,
        extra_page_size=0,
        has_topk_length=False,
        has_extra_topk_length=False,
        has_attn_sink=False,
    )
    assert state == [0]


@pytest.fixture
def fake_cuda(monkeypatch):
    state = {"current": 0, "capture": False, "operations": []}

    @contextmanager
    def device(target):
        previous = state["current"]
        state["current"] = torch.device(target).index
        try:
            yield
        finally:
            state["current"] = previous

    def record(operation):
        state["operations"].append((operation, state["current"]))

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing
            record("event")

        def record(self, stream):
            assert stream == ("stream", 1)
            record("record")

        def elapsed_time(self, other):
            record("elapsed")
            return 2.0

    def synchronize(target):
        assert torch.device(target).index == 1
        record("synchronize")

    def capturing():
        record("capture")
        return state["capture"]

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda target: ("stream", 1))
    monkeypatch.setattr(cpb_mod, "_WARMUP_ITERS", 2)
    monkeypatch.setattr(cpb_mod, "_TIMED_BATCHES", 2)
    return state


def test_timer_uses_target_device_and_restores_current(fake_cuda):
    def call(value):
        fake_cuda["operations"].append((f"call:{value}", fake_cuda["current"]))

    elapsed = cpb_mod.time_calibration_calls(call, [(1,), (2,)], torch.device("cuda:1"))
    assert elapsed == 0.001
    assert fake_cuda["current"] == 0
    assert fake_cuda["operations"][0] == ("capture", 1)
    assert all(device == 1 for _, device in fake_cuda["operations"])
    assert sum(name.startswith("call:") for name, _ in fake_cuda["operations"]) == 6


def test_timer_capture_refuses_before_warmup(fake_cuda):
    fake_cuda["capture"] = True
    with pytest.raises(cpb_mod.CalibrationError, match="capture"):
        cpb_mod.time_calibration_calls(
            lambda: pytest.fail("capture must not launch"), [()], torch.device("cuda:1")
        )
    assert fake_cuda["operations"] == [("capture", 1)]
    assert fake_cuda["current"] == 0


def test_timer_propagates_launch_failure_and_restores_current(fake_cuda):
    failure = RuntimeError("launch failed")

    def call():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        cpb_mod.time_calibration_calls(call, [()], torch.device("cuda:1"))
    assert caught.value is failure
    assert fake_cuda["current"] == 0


def test_ordinary_and_native_timers_pass_target_device(monkeypatch):
    target = torch.device("cpu")
    seen = []

    def timer(call, arguments, device):
        seen.append((arguments, device))
        return 0.002

    monkeypatch.setattr(cpb_mod, "time_calibration_calls", timer)
    monkeypatch.setattr(cpb_mod, "calibration_batch_count", lambda *args: 2)
    before = torch.random.get_rng_state().clone()
    assert (
        cpb_mod._time_call_fresh_indices(lambda _: None, 2, 3, 32, target, 1) == 0.002
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert len(seen[0][0]) == 2
    assert seen[0][1] == target
    sets = [(torch.zeros(1), None)]
    assert native._time_indexed_calls(lambda *_: None, sets, target) == 2000
    assert seen[1] == (sets, target)


def test_native_index_sets_preserve_global_rng(monkeypatch):
    monkeypatch.setattr(cpb_mod, "calibration_batch_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(
        native, "dsv4_nvfp4_format_info", lambda: {"bytes_per_token": 384}
    )
    before = torch.random.get_rng_state().clone()
    arguments = dict(
        num_tokens=2,
        topk=3,
        primary_slots=32,
        extra_topk=2,
        extra_slots=16,
        device=torch.device("cpu"),
    )
    first = native._make_index_sets(**arguments)
    second = native._make_index_sets(**arguments)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    for actual, repeated in zip(first, second):
        for tensor, other in zip(actual, repeated):
            torch.testing.assert_close(tensor, other)
    assert not torch.equal(first[0][0], first[1][0])


@pytest.mark.parametrize("eligible", [True, False])
def test_crossover_distinguishes_ineligible_prefill_from_launch_failure(
    monkeypatch, eligible
):
    from flashinfer.mla import _sparse_mla_sm120_policy as policy

    failure = RuntimeError("prefill launch failed")

    def prefill(*args):
        raise failure

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(cpb_mod, "_D_QK", {"dsv4": 8})
    monkeypatch.setattr(cpb_mod, "_D_V", {"dsv4": 8})
    monkeypatch.setattr(cpb_mod, "_CHUNK_WIDTH", {"dsv4": 64})
    monkeypatch.setattr(cpb_mod, "_BYTES_PER_TOKEN", {"dsv4": 16})
    monkeypatch.setattr(cpb_mod, "_CROSSOVER_PROBED_T", (4,))
    monkeypatch.setattr(
        cpb_mod,
        "_allocate_kv_pool",
        lambda *args: (torch.zeros(1, 1024, dtype=torch.uint8), 64),
    )
    monkeypatch.setattr(
        cpb_mod,
        "_make_decode_call_builder",
        lambda *args: lambda *args: lambda indices: None,
    )
    monkeypatch.setattr(cpb_mod, "select_cpb", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        policy, "prefill_variant", lambda *args: 1 if eligible else None
    )

    def time_call(call, *args):
        call(torch.zeros((4, 64), dtype=torch.int32))
        return 0.001

    monkeypatch.setattr(cpb_mod, "_time_call_fresh_indices", time_call)
    arguments = (
        SimpleNamespace(sparse_mla_sm120_paged_attention=prefill),
        torch.device("cpu"),
        "dsv4",
        None,
    )
    if eligible:
        with pytest.raises(RuntimeError) as caught:
            cpb_mod.calibrate_crossover(*arguments, grid_override=[(8, 64)])
        assert caught.value is failure
    else:
        assert cpb_mod.calibrate_crossover(*arguments, grid_override=[(8, 64)]) == {
            "dsv4|8|64": 4
        }


# ── persistence store: publish/merge/atomicity/failure ──


def test_scope_switch_same_mtime_delete_and_entry_remove(store, monkeypatch, tmp_path):
    assert cpb_mod.save_constants(store, "dsv4", _C)
    first = cpb_mod.default_cache_path()
    stamp = first.stat()
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(other))
    assert cpb_mod.get_constants(store, "dsv4") is None
    cpb_mod.save_constants(store, "dsv4", replace(_C, c0=2e-6))
    os.utime(cpb_mod.default_cache_path(), ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert cpb_mod.get_constants(store, "dsv4").c0 == 2e-6
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path))
    assert cpb_mod.get_constants(store, "dsv4") == _C
    payload = json.loads(first.read_text())
    payload["devices"][str(store)].pop("dsv4")
    replacement = first.with_suffix(".new")
    replacement.write_text(json.dumps(payload))
    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    os.replace(replacement, first)
    assert cpb_mod.get_constants(store, "dsv4") is None
    cpb_mod.save_constants(store, "dsv4", _C)
    first.unlink()
    assert cpb_mod.get_constants(store, "dsv4") is None


def test_atomic_alias_invalidation_and_writer_merge(store):
    cpb_mod.save_constants(store, "dsv3_2", _C)
    cpb_mod.save_crossover(
        store, {"dsv3_2|16|128": 8, "glm_nsa|32|256": 16, "dsv4|16|128": 32}
    )
    cpb_mod.save_cpb_override(store, "dsv3_2", 16, 128, 4, 2)
    cpb_mod.save_cpb_override(store, "glm_nsa", 32, 256, 4, 3)
    assert cpb_mod.publish_calibration(
        store,
        "dsv3_2",
        constants=_C,
        crossover={"dsv3_2|16|128": 4},
        replace_family=True,
    )
    assert cpb_mod.get_decode_max_tokens(store, "glm_nsa", 32, 256) is None
    assert cpb_mod.get_cpb_override(store, "dsv3_2", 16, 128, 4) is None
    assert cpb_mod.get_cpb_override(store, "glm_nsa", 32, 256, 4) is None
    assert cpb_mod.get_decode_max_tokens(store, "dsv4", 16, 128) == 32
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda index: cpb_mod.save_constants(torch.device(f"cpu:{index}"), "dsv4", _C),
                (1, 2),
            )
        )
    assert all(results)
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    for index in (1, 2):
        assert cpb_mod.get_constants(torch.device(f"cpu:{index}"), "dsv4") == _C


def test_write_failure_overlay_does_not_cross_path(store, monkeypatch, tmp_path):
    cpb_mod.save_constants(store, "dsv4", _C)
    monkeypatch.setattr(
        cpb_mod.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("read only"))
    )
    assert not cpb_mod.save_constants(store, "dsv4", replace(_C, c0=4e-6))
    cpb_mod.default_cache_path().unlink()
    assert cpb_mod.get_constants(store, "dsv4").c0 == 4e-6
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path / "other"))
    assert cpb_mod.get_constants(store, "dsv4") is None


def test_public_force_keeps_old_unit_on_measurement_failure(store, monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_execution as execution

    cpb_mod.publish_calibration(store, "dsv4", constants=_C, crossover={"dsv4|16|128": 8})
    before = cpb_mod.default_cache_path().read_bytes()
    monkeypatch.setattr(cpb_mod, "_family_specs", lambda: {"dsv4": ((16,), (128,), 1)})
    monkeypatch.setattr(execution, "get_sparse_mla_sm120_module", lambda: object())
    monkeypatch.setattr(cpb_mod, "calibrate", lambda *args: replace(_C, c0=9e-6))

    def failed(*args, **kwargs):
        raise cpb_mod.CalibrationError("measurement failed")

    monkeypatch.setattr(cpb_mod, "calibrate_crossover", failed)
    report = cpb_mod.calibrate_sparse_mla_sm120(store, families=("dsv4",), force=True)
    assert report.failed and report.constants_calibrated == ()
    assert cpb_mod.default_cache_path().read_bytes() == before
    assert cpb_mod.get_constants(store, "dsv4") == _C
    monkeypatch.setattr(
        cpb_mod, "calibrate_crossover", lambda *args, **kwargs: {"dsv4|16|128": 4}
    )
    report = cpb_mod.calibrate_sparse_mla_sm120(store, families=("dsv4",), force=True)
    assert not report.failed and report.persisted
    assert cpb_mod.get_constants(store, "dsv4").c0 == 9e-6
    assert cpb_mod.get_decode_max_tokens(store, "dsv4", 16, 128) == 4
    monkeypatch.setattr(
        cpb_mod.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("read only"))
    )
    report = cpb_mod.calibrate_sparse_mla_sm120(store, families=("dsv4",), force=True)
    assert not report.failed and not report.persisted
    assert report.constants_calibrated == ("dsv4",)


def test_native_phase_and_cpb_publish_once(store, monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_dsv4_nvfp4_policy as native
    from flashinfer.mla import _sparse_mla_sm120_execution as execution

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(native, "_eligible", lambda *args: (True, True))
    monkeypatch.setattr(native, "_CROSSOVER_PROBED_T", (4, 8))
    monkeypatch.setattr(native, "dsv4_nvfp4_format_info", lambda: {"chunk_width": 64})
    monkeypatch.setattr(execution, "get_sparse_mla_dsv4_nvfp4_module", lambda: object())
    monkeypatch.setattr(
        native, "_allocate_calibration_pools", lambda *args: (None, 1024, None, 0)
    )
    monkeypatch.setattr(native, "_make_index_sets", lambda **kwargs: [])
    monkeypatch.setattr(
        native, "_make_calibration_calls", lambda **kwargs: (lambda cpb: cpb, 0)
    )
    monkeypatch.setattr(
        native, "_time_indexed_calls", lambda call, *args: 1.0 if call else 2.0
    )
    writes = []
    original = cpb_mod.os.replace

    def replace_file(*args):
        writes.append(args)
        original(*args)

    monkeypatch.setattr(cpb_mod.os, "replace", replace_file)
    report = native.calibrate_nvfp4_sparse_mla_sm120(
        store, num_heads=16, topk=128, force=True
    )
    assert report.persisted and len(writes) == 1
    before = cpb_mod.default_cache_path().read_bytes()

    def fail(*args):
        raise cpb_mod.CalibrationError("native launch failed")

    monkeypatch.setattr(native, "_time_indexed_calls", fail)
    with pytest.raises(cpb_mod.CalibrationError):
        native.calibrate_nvfp4_sparse_mla_sm120(
            store, num_heads=16, topk=128, force=True
        )
    assert cpb_mod.default_cache_path().read_bytes() == before and len(writes) == 1


def test_crossover_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    """Crossover tables merge into the JSON document and survive reload."""
    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv4|64|512": 32, "dsv4|64|1024": 64})
    cpb_mod.save_crossover(device, {"dsv3_2|64|2048": 16, "glm_nsa|64|2048": 8})
    cpb_mod._crossover.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 64, 512) == 32
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 64, 1024) == 64
    assert cpb_mod.get_decode_max_tokens(device, "glm_nsa", 64, 2048) == 8
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 8, 128) is None
    assert cpb_mod.has_crossover(device, "dsv4")
    assert cpb_mod.has_crossover(device, "dsv3_2")


def test_dsv3_2_crossover_requires_glm_nsa_entries(
    clean_cpb_state, monkeypatch
) -> None:
    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv3_2|64|2048": 16})
    assert not cpb_mod.has_crossover(device, "dsv3_2")
    cpb_mod.save_crossover(device, {"glm_nsa|64|2048": 8})
    assert cpb_mod.has_crossover(device, "dsv3_2")


def test_cpb_override_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    device = torch.device("cpu")
    cpb_mod.save_cpb_override(device, "dsv4", 128, 1024, 64, 12)
    cpb_mod.save_cpb_override(device, "dsv4", 128, 1024, 8, 3)
    cpb_mod.save_cpb_override(device, "dsv3_2", 64, 2048, 32, 9)
    cpb_mod._cpb_overrides.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 64) == 12
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 8) == 3
    assert cpb_mod.get_cpb_override(device, "dsv3_2", 64, 2048, 32) == 9
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 16) is None
    cpb_mod.save_constants(device, "dsv4", _C)
    cpb_mod.save_crossover(device, {"dsv4|128|1024": 16})
    cpb_mod._cpb_overrides.clear()
    cpb_mod._constants.clear()
    cpb_mod._crossover.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 64) == 12
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 128, 1024) == 16


def test_crossover_grid_complete_gates_full_sweep(clean_cpb_state) -> None:
    from flashinfer.mla._sparse_mla_sm120_policy import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
    )

    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv4|48|256": 8})
    assert cpb_mod.has_crossover(device, "dsv4")
    assert not cpb_mod.crossover_grid_complete(device, "dsv4")
    cpb_mod.save_crossover(
        device, {f"dsv4|{h}|{k}": 32 for h, k in _DECODE_DSV4_CALIBRATION_GRID}
    )
    assert cpb_mod.crossover_grid_complete(device, "dsv4")
    cpb_mod.save_crossover(
        device, {f"dsv3_2|{h}|{k}": 16 for h, k in _DECODE_DSV3_2_CALIBRATION_GRID}
    )
    assert not cpb_mod.crossover_grid_complete(device, "dsv3_2")
    cpb_mod.save_crossover(
        device, {f"glm_nsa|{h}|{k}": 16 for h, k in _DECODE_DSV3_2_CALIBRATION_GRID}
    )
    assert cpb_mod.crossover_grid_complete(device, "dsv3_2")


def test_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    device = torch.device("cpu")
    cpb_mod.save_constants(device, "dsv4", _C)
    cpb_mod._constants.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_constants(device, "dsv4") == _C
    other = replace(_C, bytes_per_chunk=41984)
    cpb_mod.save_constants(device, "dsv3_2", other)
    cpb_mod._constants.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_constants(device, "dsv3_2") == other


def test_save_publishes_on_disk_sibling_entries(clean_cpb_state) -> None:
    device = torch.device("cpu")
    sibling = replace(_C, bytes_per_chunk=41984)
    path = cpb_mod.default_cache_path()
    path.write_text(
        json.dumps(
            {
                "schema_version": cpb_mod._SCHEMA_VERSION,
                "devices": {
                    cpb_mod._device_key(device): {"dsv3_2": cpb_mod.asdict(sibling)}
                },
            }
        )
    )
    cpb_mod.save_constants(device, "dsv4", _C)
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_constants(device, "dsv3_2") == sibling


def test_missing_or_corrupt_cache_falls_back(clean_cpb_state, tmp_path) -> None:
    from flashinfer.mla._sparse_mla_sm120 import _resolve_cpb

    device = torch.device("cpu")
    assert cpb_mod.get_constants(device, "dsv4") is None
    assert _resolve_cpb(device, "dsv4", 1, 16, 1024, 0) == -1
    path = cpb_mod.default_cache_path()
    path.write_text("{not json")
    assert cpb_mod.get_constants(device, "dsv4") is None
    path.write_text('{"schema_version": 999, "devices": {}}')
    assert cpb_mod.get_constants(device, "dsv4") is None
    assert _resolve_cpb(device, "dsv4", 1, 16, 1024, 0) == -1


def test_stale_schema_read_once_until_changed(isolated_cpb):
    from pathlib import Path
    from unittest.mock import patch

    path = isolated_cpb
    path.write_text(json.dumps({"schema_version": -1, "devices": {}}))
    old = path.stat().st_mtime
    original = Path.read_text
    device = torch.device("cpu")
    with patch.object(Path, "read_text", autospec=True, side_effect=original) as read:
        for _ in range(3):
            assert cpb_mod.get_constants(device, "glm53_nope") is None
            assert cpb_mod.get_cpb_override(device, "glm53_nope", 32, 512, 4) is None
            assert cpb_mod.get_decode_max_tokens(device, "glm53_nope", 32, 512) is None
        assert read.call_count == 1
        constants = replace(_C, l2_cache_bytes=0, bytes_per_chunk=64 * 528)
        path.write_text(
            json.dumps(
                {
                    "schema_version": cpb_mod._SCHEMA_VERSION,
                    "devices": {
                        "0:Test GPU": {"glm53_nope": cpb_mod.asdict(constants)}
                    },
                }
            )
        )
        os.utime(path, (old + 1, old + 1))
        assert cpb_mod.get_constants(device, "glm53_nope") == constants
        assert read.call_count == 2


def test_legacy_glm_calibration_invalidated(isolated_cpb):
    constants = replace(_C, l2_cache_bytes=0, bytes_per_chunk=64 * 656)
    isolated_cpb.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "devices": {
                    "0:Test GPU": {
                        "glm53_nope": cpb_mod.asdict(constants),
                        cpb_mod._CPB_OVERRIDES_KEY: {"glm53_nope|32|512|4": 7},
                        cpb_mod._DECODE_MAX_TOKENS_KEY: {"glm53_nope|32|512": 16},
                    }
                },
            }
        )
    )
    device = torch.device("cpu")
    assert cpb_mod.get_constants(device, "glm53_nope") is None
    assert cpb_mod.get_cpb_override(device, "glm53_nope", 32, 512, 4) is None
    assert cpb_mod.get_decode_max_tokens(device, "glm53_nope", 32, 512) is None
    current = replace(constants, bytes_per_chunk=64 * 528)
    cpb_mod.save_constants(device, "glm53_nope", current)
    cpb_mod._constants.clear()
    cpb_mod._store_states.clear()
    cpb_mod._active_scope = None
    assert cpb_mod.get_constants(device, "glm53_nope") == current
    assert json.loads(isolated_cpb.read_text())["schema_version"] == 2


def test_public_profile_reuse_force_and_failure(store, monkeypatch):
    calls = []

    def measure(request, device):
        calls.append(request)
        return {
            "request": request.effective,
            "sample_kind": "full_capacity_canonical_layout",
            "buckets": {
                str(t): {"variant": 0, "cpb": 1, "decode_s": 1e-5, "prefill_s": None}
                for t in cpb_mod._PROFILE_T
            },
        }

    monkeypatch.setattr(cpb_mod, "_measure_dsv41", measure)
    kwargs = dict(
        families=("dsv4_1",),
        heads=(16,),
        topks=(128,),
        compute_precision="bf16",
        extra_topk=77,
        extra_page_size=53,
        extra_kv_fp4=True,
    )
    report = cpb_mod.calibrate_sparse_mla_sm120(store, **kwargs)
    assert report.entries_calibrated == 1 and report.constants_calibrated == ()
    assert report.profiles[0]["status"] == "measured" and report.persisted
    repeated = cpb_mod.calibrate_sparse_mla_sm120(store, **kwargs)
    assert repeated.entries_skipped == 1 and len(calls) == 1
    before = cpb_mod.default_cache_path().read_bytes()

    def fail(*args):
        raise cpb_mod.CalibrationError("launch failed")

    monkeypatch.setattr(cpb_mod, "_measure_dsv41", fail)
    failed = cpb_mod.calibrate_sparse_mla_sm120(store, force=True, **kwargs)
    assert failed.failed and failed.entries_calibrated == 0
    assert cpb_mod.default_cache_path().read_bytes() == before


def test_public_preflight_and_default_six_families(store, monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_execution as execution

    with pytest.raises(ValueError, match="DSV4.1"):
        cpb_mod.calibrate_sparse_mla_sm120(
            store, families=("dsv4",), compute_precision="bf16"
        )
    with pytest.raises(ValueError, match="extra"):
        cpb_mod.calibrate_sparse_mla_sm120(store, families=("dsv4_1",), extra_kv_fp4=True)
    requested = set()
    monkeypatch.setattr(execution, "get_sparse_mla_sm120_module", lambda: object())

    def profile(request, device):
        requested.add("dsv4_1")
        assert request.effective["compute_precision"] == "fp8"
        assert request.extra_topk == 0
        return {
            "request": request.effective,
            "buckets": {str(t): {"variant": 0, "cpb": 1} for t in cpb_mod._PROFILE_T},
        }

    def crossover_lookup(device, family, heads, topk):
        requested.add(family)
        return None

    monkeypatch.setattr(cpb_mod, "get_constants", lambda *args: _C)
    monkeypatch.setattr(cpb_mod, "get_decode_max_tokens", crossover_lookup)
    monkeypatch.setattr(cpb_mod, "calibrate_crossover", lambda *args, **kwargs: {})
    monkeypatch.setattr(cpb_mod, "_measure_dsv41", profile)
    report = cpb_mod.calibrate_sparse_mla_sm120(store)
    assert not report.failed
    assert requested == {
        "dsv4",
        "dsv3_2",
        "glm_nsa",
        "glm53_nope",
        "dots3_swa",
        "dsv4_1",
    }


def test_runtime_exact_lookup_unknown_persistence_and_precision(store, monkeypatch):
    from flashinfer.mla import _sparse_mla_sm120_policy as policy
    from flashinfer.mla._sparse_mla_sm120_execution import AttentionMetadata

    request = cpb_mod._Dsv41Request(16, 128)
    profile = {
        "request": request.effective,
        "buckets": {
            str(t): {
                "variant": int(t in (4, 16)),
                "cpb": 2,
                "decode_s": 1e-5,
                "prefill_s": 1e-5,
            }
            for t in cpb_mod._PROFILE_T
        },
    }
    cpb_mod.publish_calibration(store, "dsv4_1", profiles={request.key: profile})
    m = AttentionMetadata(
        5,
        5,
        16,
        128,
        0,
        64,
        0,
        64 * 528,
        0,
        528,
        128,
        0,
        16,
        False,
        False,
        False,
        False,
        0,
    )
    assert policy.profile_selection(m, store, "default").cpb == 2
    assert policy.profile_selection(m, store, "fp8").cpb == 2
    assert (
        policy.profile_selection(
            m._replace(extra_fp4=0, has_lengths=0), store, "fp8"
        ).cpb
        == 2
    )
    assert policy.profile_selection(m, store, "bf16") is None
    assert policy.profile_selection(m._replace(heads=17), store, "fp8") is None
    assert (
        policy.profile_selection(
            m._replace(extra_topk=77, extra_page_size=53), store, "fp8"
        )
        is None
    )
    assert cpb_mod._profile_bucket(cpb_mod.get_dsv41_profile(request, store), 65) is None
    assert policy.profile_selection(m._replace(tokens=65), store, "fp8").variant == 1
    assert policy.profile_selection(m._replace(tokens=4), store, "fp8").variant == 1


def test_dsv41_fallback_does_not_start_legacy_measurement(store, monkeypatch):
    from types import SimpleNamespace
    from flashinfer.mla import _sparse_mla_sm120_policy as policy
    from flashinfer.mla import _sparse_mla_sm120_execution as execution

    monkeypatch.setattr(execution, "get_sparse_mla_sm120_module", lambda: object())
    monkeypatch.setattr(
        policy.AutoTuner,
        "get",
        lambda: SimpleNamespace(is_tuning_mode=True, _get_skip_ops_stack=lambda: []),
    )
    monkeypatch.setattr(cpb_mod, "_target_capturing", lambda device: False)
    monkeypatch.setattr(cpb_mod, "get_constants", lambda *args: None)
    monkeypatch.setattr(cpb_mod, "calibrate", lambda *args: pytest.fail("legacy measurement"))
    assert policy._resolve_cpb(store, "dsv4_1", 5, 16, 128, 0) == -1
