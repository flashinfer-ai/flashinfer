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

import pytest
import torch

from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
from flashinfer.mla._sparse_mla_sm120_cpb import (
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
def clean_cpb_state(monkeypatch, tmp_path):
    """Isolated cache dir, fake device key, and cleared process state."""
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_DIR", str(tmp_path))
    monkeypatch.setattr(cpb_mod, "_device_key", lambda device: "0:Fake GPU")
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    cpb_mod._constants.clear()
    cpb_mod._failed.clear()
    cpb_mod._crossover.clear()
    cpb_mod._crossover_failed.clear()
    cpb_mod._cpb_overrides.clear()
    yield tmp_path
    cpb_mod._constants.clear()
    cpb_mod._failed.clear()
    cpb_mod._crossover.clear()
    cpb_mod._crossover_failed.clear()
    cpb_mod._cpb_overrides.clear()


def test_crossover_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    """Crossover tables merge into the JSON document and survive reload."""
    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv4|64|512": 32, "dsv4|64|1024": 64})
    cpb_mod.save_crossover(device, {"dsv3_2|64|2048": 16, "glm_nsa|64|2048": 8})
    cpb_mod._crossover.clear()
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 64, 512) == 32
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 64, 1024) == 64
    assert cpb_mod.get_decode_max_tokens(device, "glm_nsa", 64, 2048) == 8
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 8, 128) is None
    assert cpb_mod.has_crossover(device, "dsv4")
    assert cpb_mod.has_crossover(device, "dsv3_2")


def test_dsv3_2_crossover_requires_glm_nsa_entries(
    clean_cpb_state, monkeypatch
) -> None:
    """has_crossover('dsv3_2') needs both the dsv3_2 and glm_nsa key spaces."""
    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv3_2|64|2048": 16})
    assert not cpb_mod.has_crossover(device, "dsv3_2")
    cpb_mod.save_crossover(device, {"glm_nsa|64|2048": 8})
    assert cpb_mod.has_crossover(device, "dsv3_2")


def test_cpb_override_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    """Refined per-shape picks merge into the JSON document alongside the
    constants and crossover entries, and survive a process-state reload."""
    device = torch.device("cpu")
    cpb_mod.save_cpb_override(device, "dsv4", 128, 1024, 64, 12)
    cpb_mod.save_cpb_override(device, "dsv4", 128, 1024, 8, 3)
    cpb_mod.save_cpb_override(device, "dsv3_2", 64, 2048, 32, 9)
    cpb_mod._cpb_overrides.clear()
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 64) == 12
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 8) == 3
    assert cpb_mod.get_cpb_override(device, "dsv3_2", 64, 2048, 32) == 9
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 16) is None
    # Coexists with constants and crossover entries in the same document.
    cpb_mod.save_constants(device, "dsv4", _C)
    cpb_mod.save_crossover(device, {"dsv4|128|1024": 16})
    cpb_mod._cpb_overrides.clear()
    cpb_mod._constants.clear()
    cpb_mod._crossover.clear()
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    assert cpb_mod.get_cpb_override(device, "dsv4", 128, 1024, 64) == 12
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_decode_max_tokens(device, "dsv4", 128, 1024) == 16


def test_crossover_grid_complete_gates_full_sweep(clean_cpb_state) -> None:
    """A targeted off-grid crossover entry must not count as a completed full
    sweep (it would silently suppress the tuning-mode grid calibration)."""
    from flashinfer.mla._sparse_mla_sm120_plan import (
        _DECODE_DSV3_2_CALIBRATION_GRID,
        _DECODE_DSV4_CALIBRATION_GRID,
    )

    device = torch.device("cpu")
    cpb_mod.save_crossover(device, {"dsv4|48|256": 8})  # targeted one-off
    assert cpb_mod.has_crossover(device, "dsv4")
    assert not cpb_mod.crossover_grid_complete(device, "dsv4")

    cpb_mod.save_crossover(
        device, {f"dsv4|{h}|{k}": 32 for h, k in _DECODE_DSV4_CALIBRATION_GRID}
    )
    assert cpb_mod.crossover_grid_complete(device, "dsv4")

    # dsv3_2 completeness needs both key spaces on the shared grid.
    cpb_mod.save_crossover(
        device,
        {f"dsv3_2|{h}|{k}": 16 for h, k in _DECODE_DSV3_2_CALIBRATION_GRID},
    )
    assert not cpb_mod.crossover_grid_complete(device, "dsv3_2")
    cpb_mod.save_crossover(
        device,
        {f"glm_nsa|{h}|{k}": 16 for h, k in _DECODE_DSV3_2_CALIBRATION_GRID},
    )
    assert cpb_mod.crossover_grid_complete(device, "dsv3_2")


def test_persistence_round_trip(clean_cpb_state, monkeypatch) -> None:
    device = torch.device("cpu")
    cpb_mod.save_constants(device, "dsv4", _C)
    # Drop process state to force a disk reload.
    cpb_mod._constants.clear()
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    loaded = cpb_mod.get_constants(device, "dsv4")
    assert loaded == _C
    # A second family merges into the same file without clobbering dsv4.
    other = CpbConstants(**{**_C.__dict__, "bytes_per_chunk": 41984})
    cpb_mod.save_constants(device, "dsv3_2", other)
    cpb_mod._constants.clear()
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_constants(device, "dsv3_2") == other


def test_save_publishes_on_disk_sibling_entries(clean_cpb_state) -> None:
    """A targeted save as the process's first cpb activity must not hide
    on-disk families written by other processes for the process lifetime."""
    device = torch.device("cpu")
    sibling = CpbConstants(**{**_C.__dict__, "bytes_per_chunk": 41984})
    path = cpb_mod.default_cache_path()
    path.write_text(
        cpb_mod.json.dumps(
            {
                "schema_version": cpb_mod._SCHEMA_VERSION,
                "devices": {
                    cpb_mod._device_key(device): {"dsv3_2": cpb_mod.asdict(sibling)}
                },
            }
        )
        + "\n"
    )
    cpb_mod.save_constants(device, "dsv4", _C)
    assert cpb_mod.get_constants(device, "dsv4") == _C
    assert cpb_mod.get_constants(device, "dsv3_2") == sibling


def test_missing_or_corrupt_cache_falls_back(clean_cpb_state, tmp_path) -> None:
    from flashinfer.mla._sparse_mla_sm120 import _resolve_cpb

    device = torch.device("cpu")
    # Missing file.
    assert cpb_mod.get_constants(device, "dsv4") is None
    assert _resolve_cpb(device, "dsv4", 1, 16, 1024, 0) == -1

    # Corrupt JSON.
    path = cpb_mod.default_cache_path()
    path.write_text("{not json")
    assert cpb_mod.get_constants(device, "dsv4") is None

    # Schema mismatch.
    path.write_text('{"schema_version": 999, "devices": {}}')
    assert cpb_mod.get_constants(device, "dsv4") is None
    assert _resolve_cpb(device, "dsv4", 1, 16, 1024, 0) == -1


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
        assert c.l2_cache_bytes == props.L2_cache_size
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
                cpb_override,
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
    monkeypatch.setattr(cpb_mod, "_cache_mtime", -1.0)
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
        recorded["cpb_override"] = args[-1]
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
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

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
    # GPU class, and plan() must pick it up in-process via the
    # _constants_version invalidation (deliberately no _plan_memo.clear()).
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
    from flashinfer.mla._sparse_mla_sm120_cpb import (
        SparseMLASm120CalibrationReport,
    )

    assert (
        flashinfer.mla.SparseMLASm120CalibrationReport
        is SparseMLASm120CalibrationReport
    )


def test_calibration_refused_under_graph_capture(monkeypatch) -> None:
    """Calibration synchronizes and allocates GiB-scale pools; under CUDA
    graph capture it must refuse loudly instead of corrupting the capture."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(cpb_mod.CalibrationError, match="capture"):
        cpb_mod.calibrate(None, "dsv4", torch.device("cuda"))
    with pytest.raises(cpb_mod.CalibrationError, match="capture"):
        cpb_mod.calibrate_crossover(None, torch.device("cuda"), "dsv4", _C)


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
