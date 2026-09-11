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

"""Backend selection under ``backend="auto"`` for ragged prefill.

``determine_attention_backend`` only ever answers ``"fa3"`` or ``"fa2"``, and
FA3 is gated on ``is_sm90a_supported()``, so on Blackwell ``auto`` lands on FA2.
``plan()`` then upgrades that answer where a faster kernel is known to be
equivalent -- ``fmha_v2`` on SM120, ``cutlass`` on SM100a/SM110a.

Those upgrades are the only place a caller's backend is chosen for them, and
the CUTLASS run path (``fmha_varlen``) silently ignores ``window_left``,
``logits_soft_cap``, custom masks and the multi-item-scoring pointers -- it
receives only ``causal`` and the scales. An upgrade that fired on any of those
would return quietly wrong numbers, so the negative cases below matter more
than the positive one and are tested first-class rather than as an afterthought.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm100a_supported, is_sm110a_supported

DTYPE = torch.bfloat16


def _cutlass_upgrade_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    dev = torch.device("cuda")
    return is_sm100a_supported(dev) or is_sm110a_supported(dev)


requires_cutlass_arch = pytest.mark.skipif(
    not _cutlass_upgrade_arch(),
    reason="the auto->cutlass upgrade targets SM100a/SM110a",
)


def _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, seed=1234):
    """Seeded so two backends can be compared over byte-identical inputs."""
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    q = torch.randn(batch * s_q, h_qo, d_qk, dtype=DTYPE, device=dev)
    k = torch.randn(batch * s_kv, h_kv, d_qk, dtype=DTYPE, device=dev)
    v = torch.randn(batch * s_kv, h_kv, d_vo, dtype=DTYPE, device=dev)
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_q
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=dev) * s_kv
    return q, k, v, qo_indptr, kv_indptr


def _plan_only(
    backend, batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, kv_layout="NHD", **plan_kwargs
):
    """Resolve the backend without launching a kernel.

    The choice is made in ``plan()``, so the selection tests stop there. That
    keeps a negative case from passing for the wrong reason -- e.g. an HND
    layout would make ``run()`` fail on tensor shapes, which would look like a
    pass while telling us nothing about which backend was picked.
    """
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout, backend=backend
    )
    _, _, _, qo_indptr, kv_indptr = _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    return wrapper._backend


def _plan_and_run(
    backend, batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo, kv_layout="NHD", **plan_kwargs
):
    """Returns ``(resolved_backend, output)``."""
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, kv_layout, backend=backend
    )
    q, k, v, qo_indptr, kv_indptr = _inputs(batch, s_q, s_kv, h_qo, h_kv, d_qk, d_vo)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        h_qo,
        h_kv,
        d_qk,
        head_dim_vo=d_vo,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        **plan_kwargs,
    )
    out = wrapper.run(q, k, v)
    torch.cuda.synchronize()
    return wrapper._backend, out


# ---------------------------------------------------------------------------
# The upgrade fires where it should
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo",
    [
        (64, 8, 128, 128),  # GQA, square d128
        (32, 32, 128, 128),  # MHA, square d128
        (128, 128, 192, 128),  # MLA-style rectangular (DeepSeek-V3 prefill dims)
    ],
)
def test_auto_upgrades_to_cutlass(h_qo, h_kv, d_qk, d_vo):
    """On SM100a/SM110a, ``auto`` must reach CUTLASS rather than settling for FA2."""
    resolved = _plan_only("auto", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    assert resolved == "cutlass", (
        f"auto resolved to {resolved!r}; expected the CUTLASS upgrade for "
        f"h_qo={h_qo} h_kv={h_kv} d_qk={d_qk} d_vo={d_vo}"
    )


# ---------------------------------------------------------------------------
# The upgrade must NOT fire where CUTLASS would drop semantics
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "tag,plan_kwargs",
    [
        # fmha_varlen takes no window_left: upgrading would compute full
        # attention and silently ignore the window.
        ("sliding_window", {"window_left": 256}),
        # ... and no logits_soft_cap.
        ("logits_soft_cap", {"logits_soft_cap": 30.0}),
    ],
)
def test_auto_declines_when_semantics_would_be_dropped(tag, plan_kwargs):
    resolved = _plan_only("auto", 4, 1024, 1024, 64, 8, 128, 128, **plan_kwargs)
    assert resolved != "cutlass", (
        f"auto upgraded to cutlass with {tag} requested; the CUTLASS run path "
        f"never receives it, so the result would be silently wrong"
    )


@requires_cutlass_arch
@pytest.mark.parametrize("d_qk,d_vo", [(256, 256), (64, 64)])
def test_auto_declines_outside_cutlass_head_dims(d_qk, d_vo):
    """``get_fmha_module`` serves square d128 and d192 only."""
    resolved = _plan_only("auto", 4, 1024, 1024, 32, 32, d_qk, d_vo)
    assert resolved != "cutlass", (
        f"auto upgraded to cutlass at d_qk={d_qk}/d_vo={d_vo}, which "
        f"get_fmha_module does not serve"
    )


@requires_cutlass_arch
def test_auto_declines_on_hnd_layout():
    """The CUTLASS path is only wired for NHD here."""
    resolved = _plan_only("auto", 4, 1024, 1024, 64, 8, 128, 128, kv_layout="HND")
    assert resolved != "cutlass"


# ---------------------------------------------------------------------------
# Numerics: the upgrade must not change the answer
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize(
    "h_qo,h_kv,d_qk,d_vo", [(64, 8, 128, 128), (128, 128, 192, 128)]
)
def test_auto_matches_fa2_numerically(h_qo, h_kv, d_qk, d_vo):
    """What ``auto`` returns now must match what it returned before (FA2).

    Compared over byte-identical seeded inputs; the tolerance is the usual bf16
    accumulation slack between two independent kernels, not an exactness claim.
    """
    resolved, out_auto = _plan_and_run("auto", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    _, out_fa2 = _plan_and_run("fa2", 4, 1024, 1024, h_qo, h_kv, d_qk, d_vo)
    assert resolved == "cutlass", "precondition: this shape should have upgraded"

    diff = (out_auto.float() - out_fa2.float()).abs()
    denom = out_fa2.float().abs().max().clamp_min(1e-6)
    rel_max = (diff.max() / denom).item()
    assert rel_max < 2e-2, (
        f"auto({resolved}) diverges from fa2: rel_max={rel_max:.3e}, "
        f"max_abs={diff.max().item():.3e}, mean_abs={diff.mean().item():.3e}"
    )


# ---------------------------------------------------------------------------
# An explicit backend is never second-guessed
# ---------------------------------------------------------------------------


@requires_cutlass_arch
@pytest.mark.parametrize("backend", ["fa2", "cutlass"])
def test_explicit_backend_is_respected(backend):
    """The upgrade is scoped to ``auto``; it must not rewrite an explicit choice.

    Pinning this matters because the gate lives inside ``plan()`` next to the
    resolver call rather than inside ``determine_attention_backend``.
    """
    resolved = _plan_only(backend, 4, 1024, 1024, 64, 8, 128, 128)
    assert resolved == backend
