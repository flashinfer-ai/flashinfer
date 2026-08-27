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
    beta=5e-6,
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


def test_beta_penalizes_splits() -> None:
    """A larger per-split merge cost shifts the optimum toward fewer splits
    (weakly larger cpb); with beta=0 the merge term vanishes and cpb drops."""
    shape = (1, 16, 1024, 0)
    base = select_cpb(*shape, _C)
    zero_beta = select_cpb(*shape, CpbConstants(**{**_C.__dict__, "beta": 0.0}))
    high_beta = select_cpb(
        *shape, CpbConstants(**{**_C.__dict__, "beta": _C.beta * 20})
    )
    assert zero_beta <= base <= high_beta


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
        beta=0.5,
        sm_count=148,
        bytes_per_chunk=1,
        l2_cache_bytes=0,
    )
    # T=1, H=16 (one head tile), topk=128 (N=2): both cpb=1 and cpb=2
    # predict exactly 2.0 s.
    t1 = predict_time_s(1, 16, 128, 0, 1, c)
    t2 = predict_time_s(1, 16, 128, 0, 2, c)
    assert t1 == t2
    assert select_cpb(1, 16, 128, 0, c) == 2


def test_select_cpb_l2_guard_rail() -> None:
    """Candidates whose concurrent streaming footprint exceeds L2 are
    excluded; if none fit, fall back to the unconstrained argmin."""
    shape = (64, 128, 3200, 0)  # N=50, saturated grid
    uncapped = select_cpb(*shape, _C)
    assert uncapped == 25
    # L2 = 10 * S * W: allowed footprint caps cpb at 10.
    capped = CpbConstants(**{**_C.__dict__, "l2_cache_bytes": 148 * 10 * 37376})
    assert select_cpb(*shape, capped) == 10
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
    yield tmp_path
    cpb_mod._constants.clear()
    cpb_mod._failed.clear()


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


@requires_sm12x
def test_calibration_smoke(dsv4_constants: CpbConstants) -> None:
    c = dsv4_constants
    assert c.inv_bw > 0 and c.inv_rsm > 0 and c.c0 > 0 and c.beta >= 0
    assert (
        c.sm_count
        == torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    )
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    if getattr(props, "L2_cache_size", None):
        assert c.l2_cache_bytes == props.L2_cache_size
    bw_gbps = 1.0 / c.inv_bw / 1e9
    print(f"\ncalibrated dsv4 constants: {c}")
    print(f"implied aggregate HBM bandwidth: {bw_gbps:.0f} GB/s")
    # Loose physical-plausibility band around modern datacenter GPUs.
    assert 100 < bw_gbps < 20000


@requires_sm12x
@pytest.mark.parametrize("topk", [128, 1024])
@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_model_cpb_accuracy_guard(
    dsv4_constants: CpbConstants, num_tokens, topk
) -> None:
    """Model-picked cpb is within 1.25x of the best swept cpb."""
    from flashinfer.mla._sparse_mla_sm120 import (
        _get_sparse_mla_sm120_decode_module,
    )

    _skip_if_low_vram(3)  # 2 GiB pool plus headroom
    device = torch.device("cuda")
    module = _get_sparse_mla_sm120_decode_module()
    c = dsv4_constants
    d_qk, d_v = 512, 512
    num_heads = 128

    num_splits = -(-topk // 64)
    w = c.bytes_per_chunk
    num_blocks = (2 << 30) // w
    kv_cache = torch.empty(num_blocks, w, dtype=torch.uint8, device=device)
    num_slots = num_blocks * 64

    q = (torch.randn(num_tokens, num_heads, d_qk, device=device) / 10.0).to(
        torch.bfloat16
    )
    indices = torch.randint(
        0, num_slots, (num_tokens, topk), dtype=torch.int32, device=device
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
        def call() -> None:
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
                cpb_override,
            )

        return cpb_mod._time_call(call)

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
    indices = torch.randint(
        0, kv_cache.shape[0] * 64, (num_tokens, topk), dtype=torch.int32, device=device
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
        def call() -> None:
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

        return cpb_mod._time_call(call)

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
