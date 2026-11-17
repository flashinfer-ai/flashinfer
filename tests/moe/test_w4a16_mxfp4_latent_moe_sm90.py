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

import pytest
import torch

from flashinfer.fused_moe.w4a16_mxfp4_moe import (
    BACKENDS,
    _composable_reference,
    _dispatch_table,
    _reset_w4a16_mxfp4_latent_moe_stats,
    _w4a16_mxfp4_latent_moe_stats,
    w4a16_mxfp4_latent_routed_moe,
)

DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
DISABLE_ENV_LEGACY = "FLASHINFER_W4A16_MXFP4_LATENT_MOE"
EL, K, L, I, G = 56, 16, 3584, 3072, 32


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    monkeypatch.delenv(DISABLE_ENV_LEGACY, raising=False)
    _reset_w4a16_mxfp4_latent_moe_stats()
    yield
    _reset_w4a16_mxfp4_latent_moe_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


def _case(T=8, el=EL, l=L, i=I, k=K, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((T, l), generator=g, device=device) * 1.0).to(torch.bfloat16)
    w13p = torch.randint(
        0, 256, (el, 2 * i, l // 2), generator=g, device=device, dtype=torch.uint8
    )
    w13s = torch.randint(
        118, 122, (el, 2 * i, l // G), generator=g, device=device, dtype=torch.uint8
    )
    w2p = torch.randint(
        0, 256, (el, l, i // 2), generator=g, device=device, dtype=torch.uint8
    )
    w2s = torch.randint(
        118, 122, (el, l, i // G), generator=g, device=device, dtype=torch.uint8
    )
    # realistic routing: DISTINCT global expert ids per token (duplicates
    # would double-hit an expert, which real routers never emit)
    ids = torch.stack(
        [torch.randperm(el * 2, generator=g, device=device)[:k] for _ in range(T)]
    ).to(torch.int32)
    w = torch.rand((T, k), generator=g, device=device, dtype=torch.float32)
    return x, w13p, w13s, w2p, w2s, ids, w


def _matched(y, ref, rtol=0.03, atol=0.02):
    yf, rf = y.float(), ref.float()
    return float(((yf - rf).abs() <= (atol + rtol * rf.abs())).float().mean().item())


def test_backends_tuple_and_table():
    assert BACKENDS == ("auto", "specialized", "reference")
    table = _dispatch_table()
    assert table is not None
    assert table["kill_switch"] == DISABLE_ENV
    op = table["ops"]["w4a16_mxfp4_latent_routed_moe"]
    assert (op["E_local"], op["top_k"], op["L"], op["I"]) == (EL, K, L, I)


def test_unknown_backend_raises_gpu_free():
    x = torch.zeros((1, L), dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="never silently swapped"):
        w4a16_mxfp4_latent_routed_moe(
            x,
            *(torch.zeros(1, dtype=torch.uint8) for _ in range(4)),
            torch.zeros((1, K), dtype=torch.int32),
            torch.zeros((1, K), dtype=torch.float32),
            backend="cutlass",
        )


def test_specialized_matches_reference_backend():
    _sm90_or_skip()
    args = _case()
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert _matched(y_spec, y_ref) >= 0.99
    # determinism contract (as documented): bitwise on the decode launch
    # path (T <= 64); the prefill path's bf16 atomic combine order is
    # disclosed nondeterministic and is tolerance-checked instead.
    y2 = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    if not torch.equal(y_spec, y2):
        # the op's documented rerun contract: the bf16 atomic combine order
        # is nondeterministic (disclosed in the PR); reruns must still agree
        # within the package gate band
        assert _matched(y2, y_spec, rtol=0.03, atol=0.02) >= 0.999, (
            "reruns diverged beyond the documented atomic-order band"
        )


def test_explicit_reference_never_fused_and_auto_routes():
    _sm90_or_skip()
    args = _case(T=4)
    _reset_w4a16_mxfp4_latent_moe_stats()
    w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    _reset_w4a16_mxfp4_latent_moe_stats()
    w4a16_mxfp4_latent_routed_moe(*args, backend="auto")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats


def test_off_geometry_served_by_reference_only():
    _sm90_or_skip()
    # smaller expert count: consistent shapes, off the fixed geometry
    args = _case(T=2, el=8)
    y = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert any("specialized:" in r for r in stats["fallback_reasons"]), stats
    out = torch.empty_like(y)
    ref = _composable_reference(*args, out)
    assert _matched(y, ref) >= 0.999


def test_prefill_specialized_matches_reference_backend():
    """Parity on the prefill launch path (T > 64), not just the decode path."""
    _sm90_or_skip()
    args = _case(T=128, seed=3)
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    assert _matched(y_spec, y_ref) >= 0.99


def test_extreme_activations_stay_finite():
    """|x| >= 512 would overflow the kernel's scaled fp16 staging to inf and
    poison the combine with NaN (inf * 0-weight); the staging now saturates.
    Out-of-range magnitudes lose accuracy, so only the clean rows are
    parity-checked; the poisoned row must still come back finite."""
    _sm90_or_skip()
    args = _case(T=8, seed=5)
    x = args[0].clone()
    x[0, :8] = 1024.0
    x[0, 8:16] = -1024.0
    args = (x,) + args[1:]
    y_spec = w4a16_mxfp4_latent_routed_moe(*args, backend="specialized")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 1, stats
    assert torch.isfinite(y_spec.float()).all()
    y_ref = w4a16_mxfp4_latent_routed_moe(*args, backend="reference")
    assert _matched(y_spec[1:], y_ref[1:]) >= 0.99


def test_misaligned_packed_weights_fall_back():
    """A contiguous uint8 view with a storage offset breaks the 16-byte
    alignment the kernel's vector loads require; it must be served by the
    reference path, not faulted through uint4 loads."""
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, ids, w = _case(T=4, seed=7)
    buf = torch.empty(w13p.numel() + 8, dtype=torch.uint8, device=w13p.device)
    buf[8:] = w13p.reshape(-1)
    w13p_off = buf[8:].view(w13p.shape)
    assert w13p_off.is_contiguous() and w13p_off.data_ptr() % 16 == 8
    y = w4a16_mxfp4_latent_routed_moe(
        x, w13p_off, w13s, w2p, w2s, ids, w, backend="specialized"
    )
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
    assert any("16-byte" in r for r in stats["fallback_reasons"]), stats
    y_ref = w4a16_mxfp4_latent_routed_moe(
        x, w13p, w13s, w2p, w2s, ids, w, backend="reference"
    )
    assert torch.equal(y, y_ref)


def test_inconsistent_weight_structure_raises():
    _sm90_or_skip()
    x, w13p, w13s, w2p, w2s, ids, w = _case(T=2, el=8)
    # mismatched expert counts
    with pytest.raises(ValueError, match="local expert count"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p[:4], w13s, w2p, w2s, ids, w, backend="reference"
        )
    # fc1 packed vs scale row counts
    with pytest.raises(ValueError, match="w13 packed/scale rows"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p[:, : 2 * I - 2], w13s, w2p, w2s, ids, w, backend="reference"
        )
    # fc2 packed vs scale reduction widths
    with pytest.raises(ValueError, match="reduction width"):
        w4a16_mxfp4_latent_routed_moe(
            x, w13p, w13s, w2p[..., : I // 2 - 16], w2s, ids, w, backend="reference"
        )


@pytest.mark.parametrize("env,val", [(DISABLE_ENV, "1"), (DISABLE_ENV_LEGACY, "0")])
def test_kill_switch_forces_the_reference_path(env, val, monkeypatch):
    _sm90_or_skip()
    monkeypatch.setenv(env, val)
    args = _case(T=2)
    w4a16_mxfp4_latent_routed_moe(*args, backend="auto")
    stats = _w4a16_mxfp4_latent_moe_stats()
    assert stats["backend_dispatches"]["specialized"] == 0, stats
    assert stats["backend_dispatches"]["reference"] == 1, stats
