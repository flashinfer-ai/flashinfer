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

Tests for flashinfer.gdn_fused_decode_step and its specialized kernels.

The op takes NO backend argument and NO environment gate: which
implementation runs is decided from the workload registry
(flashinfer/gdn_kernels/experimental/gdn_fused_decode_registry.json) and the
device, and that is a SUPPORT question only -- whether to use the op at all
belongs to the calling framework.  Semantics under test (full matrix:
flashinfer/gdn_kernels/experimental/README.md):

- the public surface: the op is reached as flashinfer.*, its routing probe
  keeps the full keyword geometry consumers gate on, it rejects a backend=
  keyword, and it exposes no on/off variable, so no removed surface can
  come back unnoticed;
- correctness of EACH shipped impl against the composable path on every
  registered geometry, including the vLLM-style padded fp32 state-pool row
  stride, with the conv-state pool in the registered SD layout (the vLLM
  default: physical (state_len, dim) rows passed as their transposed view).
  An impl is isolated by restricting the registry, not by a call argument;
- the gate numerics where random inputs never go: a decay-gate input past
  the fp32 exp() overflow point must still track the composable path, so no
  impl can ship a softplus that silently collapses the gate to zero;
- the batch SHAPE random inputs never produce either: padded rows carrying
  vLLM's PAD_SLOT_ID (-1), which a CUDA-graph replay hands to every
  registered batch size.  Every path must skip such a row rather than use
  it as a pool offset, and the pools are allocated behind a guard region so
  "wrote in front of the pool base" is an assertion rather than corruption
  of an unrelated allocation;
- the internal preference order (cute_dsl before cuda) and its fallthrough
  to the composable path;
- no environment gate: the retired kill switch (and any other
  FLASHINFER_* variable) changes nothing about dispatch or the probe;
- the exact per-geometry, per-implementation registry surface and the batches
  deliberately left off each measured-win window;
- DS-dense conv pools staying kernel-supported (registry-gated: dispatch
  requires a DS row, correctness checked with a patched registry) and
  unrecognized conv-state stride patterns falling back cleanly;
- CUDA-graph capture after per-variant warmup (specialized path captured,
  replay correct) and capture with cold caches (clean composable bake, no
  specialized dispatch, nothing compiled under capture);
- the kernel-failure latch (warn once, stock path serves, the impl stays
  off), the attestation that makes it visible to a measurement, and the
  probe memo it has to invalidate;
- the CuTe-DSL impl staying inside the DSL surface its DEPLOYED version
  offers, not the one a dev box happens to have, and that documented floor
  staying consistent with the nvidia-cutlass-dsl version the repo pins;
- the PDL contract of any impl that launches with programmatic stream
  serialization: every such kernel must reach a griddepcontrol wait on every
  path before it can read predecessor-produced data, and must not release its
  dependents before that wait.  Asserted structurally (AST), because the race
  it prevents is not reproducible by timing;
- multi-device dispatch: a call whose tensors live on a device other than
  the ambient one must still reach the kernel (TP > 1 serving does exactly
  that) and must leave the caller's ambient device alone, and no impl may
  take its launch stream from an unnamed device;
- the probe memo more generally: it must answer a repeated question
  cheaply without ever outliving the registry it was derived from.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import NamedTuple

import pytest
import torch

import flashinfer.gdn_kernels.experimental.gdn_fused_decode as gfd
from flashinfer.gdn_kernels.experimental import (
    gdn_fused_decode_specialized as specialized_gdn,
)
from flashinfer.gdn_kernels.experimental.gdn_fused_decode import gdn_fused_decode_step
from flashinfer.utils import is_sm120a_supported

_CUDA_IMPL = "cuda_sm120_persistent"
_CUTEDSL_IMPL = "cutedsl_sm120_pdl"
SHIPPED_IMPLS = (_CUTEDSL_IMPL, _CUDA_IMPL)  # in dispatch preference order

# The oldest nvidia-cutlass-dsl release the CuTe-DSL impl has to RUN on.
#
# This is deliberately BELOW the version FlashInfer itself pins
# (``requirements.txt`` / the ``cu12``/``cu13`` extras in ``pyproject.toml``),
# and the two are not in conflict: the pin says which DSL a
# ``pip install flashinfer-python`` brings along, while the floor says which
# DSL the shipped kernel must still compile under.  Those differ because this
# op is consumed from serving stacks that resolve the DSL themselves -- the
# vLLM nightly image the end-to-end arms of this work ran in downgrades to
# 4.5.2 (vLLM's own pin) after FlashInfer is installed, and the DGX/pt2605
# containers ship 4.5.0.  So "the pin is 4.7.0" is not a licence to use a 4.6+
# primitive here.
#
# Authoritative statement of the floor for this package: this constant plus
# PORTABLE_CUTE_MATH_PRIMITIVES below.  Raising it is a deliberate act -- edit
# both, and say so in ``flashinfer/gdn_kernels/experimental/README.md``.
CUTE_DSL_RUNTIME_FLOOR = (4, 5)

# The ``cute.math`` surface of nvidia-cutlass-dsl 4.5.x.  4.5 ships a
# hand-written ``cute/math.py`` exporting exactly these nineteen names; 4.6
# replaced it with a re-export of ``cutlass._mlir_helpers.math``, which
# exports 54 (``max``, ``min``, ``clamp``, ``log1p``, ``fma`` ... ).
#
# That difference is a deployment trap, not a style question: a primitive
# added in 4.6 resolves fine on a dev box or in the kernel-benchmark image
# and raises ``AttributeError: module 'cutlass.cute.math' has no attribute
# ...`` inside ``cute.compile`` wherever an older DSL is pinned -- where the
# dispatch layer catches it, latches the impl off and silently serves a
# different kernel.  Pin the surface here so the trap is caught by a test
# that needs neither a GPU nor cutlass installed.
PORTABLE_CUTE_MATH_PRIMITIVES = frozenset(
    {
        "absf",
        "acos",
        "asin",
        "atan",
        "atan2",
        "copysign",
        "cos",
        "erf",
        "exp",
        "exp2",
        "floor",
        "log",
        "log10",
        "log2",
        "rsqrt",
        "sin",
        "sqrt",
        "tan",
        "tanh",
    }
)

# Default Qwen3.6-27B surface used by the unparameterized guard tests below.
REGISTERED_BATCHES = (1, 2, 4, 8)
# Batches the kernels handle for that geometry but the registry omits: faster
# in a kernel A/B, no end-to-end serving win, so they keep the stock path
# (rationale in flashinfer/gdn_kernels/experimental/README.md).
UNREGISTERED_BATCHES = (16, 24, 32)


class Geometry(NamedTuple):
    """One registered layer geometry.

    The geometry is a compile-time parameter of both impls, so the tests
    below are parameterized over this table rather than over module-level
    constants: adding a model means adding registry rows and an entry here.
    ``h_q`` is derived exactly as the dispatch guard derives it, from the
    q/k/v split of ``qkv_dim``.
    """

    name: str
    hidden: int
    n_ba: int
    qkv_dim: int
    hv: int
    d: int
    conv_width: int
    conv_state_len: int
    impls: tuple
    batches: tuple

    @property
    def h_q(self) -> int:
        return (self.qkv_dim - self.hv * self.d) // (2 * self.d)

    def key(self) -> tuple:
        """The geometry key the kernel modules and the registry agree on."""
        return (
            self.hidden,
            self.n_ba,
            self.qkv_dim,
            self.h_q,
            self.hv,
            self.d,
            self.conv_width,
            self.conv_state_len,
        )


QWEN_27B = Geometry(
    "qwen3.6-27b", 5120, 96, 10240, 48, 128, 4, 3, SHIPPED_IMPLS, (1, 2, 4, 8)
)
QWEN_35B_A3B = Geometry(
    "qwen3.6-35b-a3b", 2048, 64, 8192, 32, 128, 4, 3, SHIPPED_IMPLS, (1, 2, 4)
)
QWEN35_27B_TP2 = Geometry(
    "qwen3.5-27b-tp2",
    5120,
    48,
    5120,
    24,
    128,
    4,
    3,
    (_CUDA_IMPL,),
    (1, 2, 4, 8, 16),
)
QWEN35_35B_A3B_TP2 = Geometry(
    "qwen3.5-35b-a3b-tp2",
    2048,
    32,
    4096,
    16,
    128,
    4,
    3,
    (_CUDA_IMPL,),
    (1, 2, 4, 8, 16, 24, 32),
)
GEOMETRIES = (QWEN_27B, QWEN_35B_A3B, QWEN35_27B_TP2, QWEN35_35B_A3B_TP2)
TP2_GEOMETRIES = (QWEN35_27B_TP2, QWEN35_35B_A3B_TP2)

# The geometry the un-parameterized tests below build inputs for.  It is the
# first registered one; the per-geometry tests cover the rest.
HIDDEN = QWEN_27B.hidden
N_BA = QWEN_27B.n_ba
QKV_DIM = QWEN_27B.qkv_dim
HV = QWEN_27B.hv
D = QWEN_27B.d
CONV_WIDTH = QWEN_27B.conv_width
CONV_STATE_LEN = QWEN_27B.conv_state_len
# One distinct pool slot per row of the largest batch any test builds inputs
# for. The default Qwen3.6-27B 16/24/32 guard rows never dispatch, but they
# still run the composable path over `state_indices`, and `_make_inputs` walks
# the slots downwards from POOL-1. A pool that only just fits (or does not fit)
# the batch runs off the bottom, where torch's negative-index wrap silently
# aliases two batch rows onto one state slot and makes the comparison
# meaningless instead of failing.
POOL = max(REGISTERED_BATCHES + UNREGISTERED_BATCHES) + 1
PADDED_ROW_STRIDE = HV * D * D + 4096

# bf16 output / fp32 state tolerance family for the GDN decode tests. The
# specialized kernels round the fp32 b/a GEMV sums through bf16 exactly like
# the composable path, but accumulate the fp32 partials in a different order,
# so gate inputs can differ by one bf16 ulp on rare boundary values.
ATOL, RTOL = 5e-3, 5e-3


def _skip_if_no_cuda() -> None:
    """Skip when the node has no CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _skip_if_no_specialized() -> None:
    """Skip unless this node can run the registered specialized kernels."""
    _skip_if_no_cuda()
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("the registered specialized fused GDN kernels target SM120")


def _impl(name: str):
    """The impl module ``name``, or skip when it cannot load here.

    An impl may be legitimately unavailable (the CuTe-DSL one needs the
    optional nvidia-cutlass-dsl package), which is a skip, not a failure.
    """
    module = specialized_gdn._load_impl(name)
    if module is None:
        pytest.skip(f"impl {name!r} unavailable on this node")
    return module


def _restrict_registry_to(monkeypatch, impl_name: str) -> None:
    """Make ``impl_name`` the only registered impl for this test.

    With no backend argument, restricting the registry is how a test pins
    which implementation serves a call -- and it exercises exactly the
    mechanism production dispatch uses.
    """
    rows = tuple(
        row
        for row in specialized_gdn.load_gdn_fused_decode_registry()
        if row["impl"] == impl_name
    )
    assert rows, f"no registry rows for impl {impl_name!r}"
    monkeypatch.setattr(specialized_gdn, "load_gdn_fused_decode_registry", lambda: rows)


def _signature_for(inputs: dict) -> dict:
    """The dispatch signature of an input set built by :func:`_make_inputs`.

    ``ready_for_graph_capture`` takes the matched signature, so readiness is
    answered about the exact variant -- including its layer geometry -- that
    the dispatcher would run.
    """
    signature = specialized_gdn.signature_from_tensors(*_guard_args(inputs))
    assert signature is not None, "test inputs must satisfy the op contract"
    return signature


def _make_conv_state(
    conv_layout: str, device, geometry: Geometry = QWEN_27B
) -> torch.Tensor:
    """Conv-state pool as the logical [P, qkv_dim, conv_state_len] view the
    op consumes: SD = physical (state_len, dim) rows passed transposed (the
    vLLM default allocation), DS = dense (dim, state_len) rows."""
    qkv_dim, state_len = geometry.qkv_dim, geometry.conv_state_len
    if conv_layout == "SD":
        pool = torch.randn(POOL, state_len, qkv_dim, device=device).bfloat16() * 0.5
        return pool.transpose(-1, -2)
    assert conv_layout == "DS"
    return torch.randn(POOL, qkv_dim, state_len, device=device).bfloat16() * 0.5


def _make_inputs(
    B: int,
    *,
    padded_pool: bool,
    seed: int,
    conv_layout: str = "SD",
    device="cuda",
    saturate_gate: bool = False,
    geometry: Geometry = QWEN_27B,
) -> dict:
    """Build one input set for a registered layer geometry.

    Scales are chosen so the gates stay in their ordinary range; pass
    ``saturate_gate`` for the overflow regime random inputs never reach.
    ``padded_pool`` reproduces vLLM's padded ssm-pool row stride, and
    ``conv_layout`` picks the physical conv-pool layout.  Each batch row gets
    its own pool slot, walking downwards from ``POOL - 1``.
    """
    assert B < POOL, "every batch row needs its own state-pool slot"
    hidden, n_ba = geometry.hidden, geometry.n_ba
    qkv_dim, hv, d = geometry.qkv_dim, geometry.hv, geometry.d
    conv_width = geometry.conv_width
    padded_row_stride = hv * d * d + 4096
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    inputs = {
        "hidden_states": torch.randn(B, hidden, device=device).bfloat16() * 0.5,
        "w_ba": torch.randn(hidden, n_ba, device=device).bfloat16() * 0.02,
        # Serving passes mixed_qkv as a row-strided view into the wider fused
        # qkvz projection; reproduce that layout (values live in [:, :qkv_dim]).
        "mixed_qkv": (torch.randn(B, qkv_dim + 2048, device=device).bfloat16() * 0.5)[
            :, :qkv_dim
        ],
        "conv_weight": torch.randn(qkv_dim, conv_width, device=device).bfloat16() * 0.3,
        "conv_bias": torch.randn(qkv_dim, device=device).bfloat16() * 0.1,
        "conv_state": _make_conv_state(conv_layout, device, geometry),
        "A_log": torch.randn(hv, device=device).float() * 0.5,
        "dt_bias": torch.randn(hv, device=device).bfloat16() * 0.1,
        "scale": 1.0 / math.sqrt(d),
        "state_indices": torch.arange(POOL - 1, POOL - 1 - B, -1, device=device).int(),
        "use_qk_l2norm": True,
    }
    if padded_pool:
        backing = torch.randn(POOL * padded_row_stride, device=device).float() * 0.05
        inputs["ssm_state"] = backing.as_strided(
            (POOL, hv, d, d), (padded_row_stride, d * d, d, 1)
        )
    else:
        inputs["ssm_state"] = torch.randn(POOL, hv, d, d, device=device).float() * 0.05
    if saturate_gate:
        # Decay gate g = exp(-exp(A_log) * softplus(a + dt_bias)).  Push the
        # softplus argument to ~100, past the point where exp() overflows in
        # fp32 (~88.7), while keeping exp(A_log) = exp(-6) small enough that
        # the true gate stays O(1): exp(-exp(-6) * 100) = 0.78.  A softplus
        # written as log(1 + exp(x)) returns +inf here and collapses g to 0.
        inputs["dt_bias"] = torch.full((hv,), 100.0, device=device).bfloat16()
        inputs["A_log"] = torch.full((hv,), -6.0, device=device).float()
    return inputs


def _run_reference(
    B: int,
    *,
    padded_pool: bool,
    seed: int,
    conv_layout: str = "SD",
    saturate_gate: bool = False,
    device="cuda",
    geometry: Geometry = QWEN_27B,
):
    """Composable-path result on an identically-seeded fresh input set."""
    ref_inputs = _make_inputs(
        B,
        padded_pool=padded_pool,
        seed=seed,
        conv_layout=conv_layout,
        saturate_gate=saturate_gate,
        device=device,
        geometry=geometry,
    )
    out = gfd._gdn_fused_decode_step_fallback(
        ref_inputs["hidden_states"],
        ref_inputs["w_ba"],
        ref_inputs["mixed_qkv"],
        ref_inputs["conv_weight"],
        ref_inputs["conv_bias"],
        ref_inputs["conv_state"],
        ref_inputs["A_log"],
        ref_inputs["dt_bias"],
        ref_inputs["scale"],
        ref_inputs["ssm_state"],
        ref_inputs["state_indices"],
        ref_inputs["use_qk_l2norm"],
    )
    return out


def _guard_args(inputs: dict) -> tuple:
    """Positional arguments of specialized_gdn.signature_from_tensors."""
    return (
        inputs["hidden_states"],
        inputs["w_ba"],
        inputs["mixed_qkv"],
        inputs["conv_weight"],
        inputs["conv_bias"],
        inputs["conv_state"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["ssm_state"],
        inputs["state_indices"],
        inputs["use_qk_l2norm"],
    )


def _matched_rows(inputs: dict):
    """Registry rows this input set dispatches to (empty when none)."""
    signature = specialized_gdn.signature_from_tensors(*_guard_args(inputs))
    if signature is None:
        return []
    return specialized_gdn.match_gdn_fused_decode_signature(
        signature, inputs["hidden_states"].device
    )


def _make_impls_look_cold(monkeypatch) -> tuple:
    """Present both impls as freshly imported, undoably.

    Returns ``(cuda_module, cutedsl_module_or_None)``.

    The CUDA impl memoizes its loaded JIT modules in a module-level dict.
    Clearing that process-global dict would make every later test (and call
    in the session) pay a module reload. Swapping in a fresh dict is
    equivalent for this test and lets monkeypatch restore the real cache.
    """
    cuda = _impl(_CUDA_IMPL)
    monkeypatch.setattr(cuda, "_modules", {})
    monkeypatch.setattr(cuda, "_barrier_cache", {})
    monkeypatch.setattr(cuda, "_scratch_cache", {})
    cutedsl = specialized_gdn._load_impl(_CUTEDSL_IMPL)
    if cutedsl is not None:
        monkeypatch.setattr(cutedsl, "_compiled", {})
        monkeypatch.setattr(cutedsl, "_workspace_cache", {})
    return cuda, cutedsl


@pytest.fixture(autouse=True)
def _clean_default_state(monkeypatch):
    """Each test starts from a clean failure latch and a cold probe memo.

    The memo is process-global and answers from the registry that was
    installed when it was filled, so a test that restricts the registry
    must not inherit another test's answers.  (Production does not need
    this fixture: _registry_index() drops the memo whenever the registry
    object changes.  Clearing here keeps a test failure readable.)
    """
    monkeypatch.setattr(specialized_gdn, "_failed_impls", set())
    monkeypatch.setattr(specialized_gdn, "_served_impls", set())
    specialized_gdn._probe_memo.clear()
    yield


@pytest.mark.parametrize("impl_name", SHIPPED_IMPLS)
@pytest.mark.parametrize("B", REGISTERED_BATCHES)
@pytest.mark.parametrize("padded_pool", [False, True])
def test_each_impl_matches_composable(
    impl_name: str, B: int, padded_pool: bool, monkeypatch
):
    """Every shipped impl serves every registered geometry correctly.

    The impl is pinned by restricting the registry to its rows -- the op
    has no backend argument, and this is the same mechanism dispatch uses.
    """
    _skip_if_no_specialized()
    impl = _impl(impl_name)
    seed = 20260711 + B + (1000 if padded_pool else 0)

    inputs = _make_inputs(B, padded_pool=padded_pool, seed=seed)
    assert {row["impl"] for row in _matched_rows(inputs)} >= {impl_name}, (
        "test geometry must be registered for this impl"
    )
    _restrict_registry_to(monkeypatch, impl_name)

    launches_before = impl.launch_count()
    out, conv_state, ssm_state = gdn_fused_decode_step(**inputs)
    assert impl.launch_count() == launches_before + 1
    ref_out, ref_conv, ref_ssm = _run_reference(B, padded_pool=padded_pool, seed=seed)

    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    # The conv-state update is exact bf16 data movement on both paths.
    assert torch.equal(conv_state, ref_conv)
    torch.testing.assert_close(ssm_state, ref_ssm, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("impl_name", SHIPPED_IMPLS)
@pytest.mark.parametrize("B", (1, 4))
def test_saturated_decay_gate_matches_composable(impl_name: str, B: int, monkeypatch):
    """A large decay-gate input must not collapse the gate to zero.

    The gate is ``g = exp(-exp(A_log) * softplus(a + dt_bias))``.  Spelled
    ``log(1 + exp(x))``, the softplus overflows to ``+inf`` once ``x`` passes
    ~88.7 in fp32 and takes ``g`` to exactly 0, while the composable path
    (:func:`torch.nn.functional.softplus`, threshold 20) returns ``x``.  The
    random-input tests never reach that range, so the difference is invisible
    to them; here ``exp(A_log)`` is small enough that the true gate is O(1),
    which makes the whole gated state contribution the error.  Both shipped
    impls must use an overflow-free softplus.
    """
    _skip_if_no_specialized()
    impl = _impl(impl_name)
    seed = 20260812 + B

    inputs = _make_inputs(B, padded_pool=False, seed=seed, saturate_gate=True)
    assert {row["impl"] for row in _matched_rows(inputs)} >= {impl_name}
    _restrict_registry_to(monkeypatch, impl_name)

    launches_before = impl.launch_count()
    out, _, ssm_state = gdn_fused_decode_step(**inputs)
    assert impl.launch_count() == launches_before + 1
    ref_out, _, ref_ssm = _run_reference(
        B, padded_pool=False, seed=seed, saturate_gate=True
    )

    # The gate is O(1) here, so a collapsed gate is not a rounding difference.
    assert ref_ssm.abs().max() > 1e-2, "reference gate must not be degenerate"
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ssm_state, ref_ssm, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Padded batch rows (a negative state index == vLLM's PAD_SLOT_ID)
# ===========================================================================
# A CUDA-graph replay carries the captured batch size, not the live request
# count, and vLLM fills the rows in between with PAD_SLOT_ID = -1.  The
# registry ships exactly the capture sizes (1/2/4/8), so this shape reaches
# the kernels in production -- while every random-input test above hands out
# one distinct, valid slot per row and therefore never produces it.
#
# The pools are allocated with unused slots IN FRONT of the base, because the
# failure this pins is not a wrong number: a negative index multiplies the
# page stride, so the read and the write land *below* the pool.  One guard
# slot is exactly enough -- the most negative offset either kernel can form is
# -1 * page_stride -- and it turns "corrupts whatever the allocator handed out
# next" into an assertion this file can make.
PAD_GUARD_SLOTS = 1
PAD_POOL = 8  # >= the largest batch the padded cases below use
PAD_CONV_SLOT_ELEMS = CONV_STATE_LEN * QKV_DIM
PAD_SSM_SLOT_ELEMS = HV * D * D

PAD_CASES = {
    # The minimal reproducer: one live row, one padded row.
    "one_live_one_pad": (2, [4, -1]),
    # The shape serving actually produces: a graph captured at batch 8
    # replaying with three live requests.
    "graph_b8_three_live": (8, [5, 2, 7, -1, -1, -1, -1, -1]),
}


def _pad_pool_views(conv_backing: torch.Tensor, ssm_backing: torch.Tensor):
    """The (conv_state, ssm_state) op views over guard-prefixed backings."""
    return (
        conv_backing[PAD_GUARD_SLOTS:].transpose(-1, -2),
        ssm_backing.as_strided(
            (PAD_POOL, HV, D, D),
            (PADDED_ROW_STRIDE, D * D, D, 1),
            PAD_GUARD_SLOTS * PADDED_ROW_STRIDE,
        ),
    )


def _make_padded_inputs(B: int, slots, *, seed: int, device="cuda"):
    """One input set for the registered geometry with an explicit index vector.

    ``slots`` is the exact per-row ``state_indices`` content (``-1`` marking a
    padded row).  Returns the op kwargs plus the two backing allocations, whose
    leading ``PAD_GUARD_SLOTS`` slots are not part of the pools and must stay
    untouched.  The ssm pool keeps vLLM's padded page stride, which is also
    what makes the out-of-bounds offset land squarely inside the guard.
    """
    assert len(slots) == B
    assert all(s < PAD_POOL for s in slots)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    conv_backing = (
        torch.randn(
            PAD_GUARD_SLOTS + PAD_POOL, CONV_STATE_LEN, QKV_DIM, device=device
        ).bfloat16()
        * 0.5
    )
    ssm_backing = (
        torch.randn(
            (PAD_GUARD_SLOTS + PAD_POOL) * PADDED_ROW_STRIDE, device=device
        ).float()
        * 0.05
    )
    conv_state, ssm_state = _pad_pool_views(conv_backing, ssm_backing)
    inputs = {
        "hidden_states": torch.randn(B, HIDDEN, device=device).bfloat16() * 0.5,
        "w_ba": torch.randn(HIDDEN, N_BA, device=device).bfloat16() * 0.02,
        "mixed_qkv": (torch.randn(B, QKV_DIM + 2048, device=device).bfloat16() * 0.5)[
            :, :QKV_DIM
        ],
        "conv_weight": torch.randn(QKV_DIM, CONV_WIDTH, device=device).bfloat16() * 0.3,
        "conv_bias": torch.randn(QKV_DIM, device=device).bfloat16() * 0.1,
        "conv_state": conv_state,
        "A_log": torch.randn(HV, device=device).float() * 0.5,
        "dt_bias": torch.randn(HV, device=device).bfloat16() * 0.1,
        "scale": 1.0 / math.sqrt(D),
        "ssm_state": ssm_state,
        "state_indices": torch.tensor(slots, dtype=torch.int32, device=device),
        "use_qk_l2norm": True,
    }
    return inputs, conv_backing, ssm_backing


def _run_padded_reference(inputs: dict, conv_backing, ssm_backing):
    """Composable-path result on cloned pools, so the fused call can be
    compared against it without either run seeing the other's mutations."""
    conv_clone, ssm_clone = conv_backing.clone(), ssm_backing.clone()
    conv_state, ssm_state = _pad_pool_views(conv_clone, ssm_clone)
    out, _, _ = gfd._gdn_fused_decode_step_fallback(
        inputs["hidden_states"],
        inputs["w_ba"],
        inputs["mixed_qkv"],
        inputs["conv_weight"],
        inputs["conv_bias"],
        conv_state,
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["scale"],
        ssm_state,
        inputs["state_indices"],
        inputs["use_qk_l2norm"],
    )
    return out, conv_clone, ssm_clone


def _assert_only_live_slots_changed(
    backing, before, slots, *, slot_elems: int, slot_stride: int, name: str
) -> None:
    """Every element outside the live slots' own rows is byte-identical.

    Covers the guard region in front of the pool, the slots no batch row
    named, and -- for the ssm pool -- the padding between page strides.  The
    check is on BYTES, not on a tolerance: this is the property that actually
    failed, and nothing here is supposed to be recomputed.
    """
    base = PAD_GUARD_SLOTS * slot_stride
    flat, flat_before = backing.reshape(-1), before.reshape(-1)
    assert torch.equal(flat[:base], flat_before[:base]), (
        f"{name}: memory BEFORE the pool base was written -- a negative "
        f"state index was used as a pool offset"
    )
    restored = flat.clone()
    for slot in sorted({int(s) for s in slots if s >= 0}):
        lo = base + slot * slot_stride
        restored[lo : lo + slot_elems] = flat_before[lo : lo + slot_elems]
    assert torch.equal(restored, flat_before), (
        f"{name}: a slot outside the live set changed"
    )


def _assert_padded_call_is_correct(inputs, conv_backing, ssm_backing, slots) -> None:
    """Run the op over ``inputs`` and check the whole padding contract."""
    live = [i for i, slot in enumerate(slots) if slot >= 0]
    padded = [i for i, slot in enumerate(slots) if slot < 0]
    assert live and padded, "the case must have both live and padded rows"

    ref_out, ref_conv, ref_ssm = _run_padded_reference(
        inputs, conv_backing, ssm_backing
    )
    conv_before, ssm_before = conv_backing.clone(), ssm_backing.clone()
    out, _, _ = gdn_fused_decode_step(**inputs)

    # (a) live rows agree with the composable path, pools included.
    torch.testing.assert_close(out[live], ref_out[live], atol=ATOL, rtol=RTOL)
    for slot in {int(slots[i]) for i in live}:
        assert torch.equal(
            conv_backing[PAD_GUARD_SLOTS + slot], ref_conv[PAD_GUARD_SLOTS + slot]
        )
    ref_ssm_view = _pad_pool_views(ref_conv, ref_ssm)[1]
    for slot in {int(slots[i]) for i in live}:
        torch.testing.assert_close(
            inputs["ssm_state"][slot], ref_ssm_view[slot], atol=ATOL, rtol=RTOL
        )

    # (b) padded rows are exactly zero, not merely small or finite.
    assert torch.count_nonzero(out[padded]) == 0, (
        "padded batch rows must produce a zero output row"
    )

    # (c)+(d) neither pool changed outside the live slots, and nothing was
    # written in front of either pool base.
    _assert_only_live_slots_changed(
        conv_backing,
        conv_before,
        slots,
        slot_elems=PAD_CONV_SLOT_ELEMS,
        slot_stride=PAD_CONV_SLOT_ELEMS,
        name="conv_state pool",
    )
    _assert_only_live_slots_changed(
        ssm_backing,
        ssm_before,
        slots,
        slot_elems=PAD_SSM_SLOT_ELEMS,
        slot_stride=PADDED_ROW_STRIDE,
        name="ssm_state pool",
    )
    # The composable path is held to the same contract, on its own clones.
    _assert_only_live_slots_changed(
        ref_conv,
        conv_before,
        slots,
        slot_elems=PAD_CONV_SLOT_ELEMS,
        slot_stride=PAD_CONV_SLOT_ELEMS,
        name="conv_state pool (composable path)",
    )
    _assert_only_live_slots_changed(
        ref_ssm,
        ssm_before,
        slots,
        slot_elems=PAD_SSM_SLOT_ELEMS,
        slot_stride=PADDED_ROW_STRIDE,
        name="ssm_state pool (composable path)",
    )
    assert torch.count_nonzero(ref_out[padded]) == 0


@pytest.mark.parametrize("impl_name", SHIPPED_IMPLS)
@pytest.mark.parametrize("case", sorted(PAD_CASES))
def test_padded_state_indices_are_skipped(impl_name: str, case: str, monkeypatch):
    """A negative state index must skip the row, not index below the pool.

    ``state_indices`` carries no in-band "inactive" value other than a
    negative one, and both pools are addressed as ``index * page_stride``, so
    an unguarded negative index reads AND writes memory in front of the pool
    -- silently, with a finite-looking output and no CUDA error.  Every
    shipped impl must instead skip such a row entirely (no pool read, no pool
    write) and write its output row as zero, which is the contract
    ``gated_delta_rule_decode_pretranspose``'s float32 path already documents.

    Asserted here: live rows still match the composable path; padded rows are
    exactly zero; and -- the property that actually failed -- no byte outside
    the live slots changes, including the guard region allocated in front of
    each pool base.
    """
    _skip_if_no_specialized()
    impl = _impl(impl_name)
    B, slots = PAD_CASES[case]

    inputs, conv_backing, ssm_backing = _make_padded_inputs(B, slots, seed=20260819 + B)
    assert {row["impl"] for row in _matched_rows(inputs)} >= {impl_name}, (
        "test geometry must be registered for this impl"
    )
    _restrict_registry_to(monkeypatch, impl_name)

    launches_before = impl.launch_count()
    _assert_padded_call_is_correct(inputs, conv_backing, ssm_backing, slots)
    assert impl.launch_count() == launches_before + 1, (
        "the specialized impl must have served the call, or this test proves "
        "nothing about the kernel"
    )


@pytest.mark.parametrize("case", sorted(PAD_CASES))
def test_padded_state_indices_on_the_composable_path(case: str, monkeypatch):
    """The composable path owes the caller the same padding contract.

    It serves every geometry the registry does not list, so a consumer that
    passes PAD_SLOT_ID must not get one answer from the kernels and another
    (here: an unrecoverable device-side assert out of ``index_select``) from
    the fallback.  The registry is emptied rather than the batch changed, so
    the case is identical to the dispatched one.
    """
    _skip_if_no_cuda()
    B, slots = PAD_CASES[case]
    monkeypatch.setattr(specialized_gdn, "load_gdn_fused_decode_registry", lambda: ())
    inputs, conv_backing, ssm_backing = _make_padded_inputs(B, slots, seed=20260819 + B)
    assert not _matched_rows(inputs)
    _assert_padded_call_is_correct(inputs, conv_backing, ssm_backing, slots)


def test_padded_rows_do_not_disturb_the_live_rows():
    """Padding rows onto a batch must not change what the live rows compute.

    The statement a serving stack actually depends on: a graph captured at 8
    replaying with three live requests must give those three requests the same
    answer -- output and both pool slots -- as a batch of 8 in which the same
    three rows are live and the rest happen to hold real slots.  Same batch
    size, so the same compiled kernel variant serves both, and the rows are
    independent by construction; only the padding differs.  A kernel that
    "handled" padding by folding the padded rows onto some real slot would
    pass the zero-output check above and fail this one.
    """
    _skip_if_no_specialized()
    live = [5, 2, 7]
    padded_slots = live + [-1] * 5
    # The same batch with the trailing rows pointed at their own real slots.
    all_live_slots = live + [0, 1, 3, 4, 6]

    padded_inputs = _make_padded_inputs(8, padded_slots, seed=20260819)[0]
    out_padded, padded_conv, padded_ssm = gdn_fused_decode_step(**padded_inputs)

    dense_inputs = _make_padded_inputs(8, all_live_slots, seed=20260819)[0]
    out_dense, dense_conv, dense_ssm = gdn_fused_decode_step(**dense_inputs)

    torch.testing.assert_close(out_padded[:3], out_dense[:3], atol=ATOL, rtol=RTOL)
    for slot in live:
        assert torch.equal(padded_conv[slot], dense_conv[slot])
        torch.testing.assert_close(
            padded_ssm[slot], dense_ssm[slot], atol=ATOL, rtol=RTOL
        )


def test_internal_preference_is_cutedsl_then_cuda(monkeypatch):
    """Dispatch selects among the signature's registered impls in the
    internal preference order: cute_dsl first, cuda when cute_dsl has no
    row.  Nothing about this is caller-visible except which kernel runs."""
    _skip_if_no_specialized()
    cutedsl = _impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)
    seed = 20260722

    inputs = _make_inputs(2, padded_pool=False, seed=seed)
    dsl_before, cuda_before = cutedsl.launch_count(), cuda.launch_count()
    gdn_fused_decode_step(**inputs)
    assert cutedsl.launch_count() == dsl_before + 1
    assert cuda.launch_count() == cuda_before

    # Registry rows drive the selection: with only the cuda rows left for
    # the signature, dispatch takes the cuda impl.
    _restrict_registry_to(monkeypatch, _CUDA_IMPL)
    inputs = _make_inputs(2, padded_pool=False, seed=seed)
    dsl_before, cuda_before = cutedsl.launch_count(), cuda.launch_count()
    gdn_fused_decode_step(**inputs)
    assert cutedsl.launch_count() == dsl_before
    assert cuda.launch_count() == cuda_before + 1


@pytest.mark.parametrize("impl_name", SHIPPED_IMPLS)
def test_ds_layout_kernel_supported(impl_name: str, monkeypatch):
    """DS-dense conv pools stay kernel-supported (the addressing is
    stride-parameterized): with DS registry rows, dispatch and results
    match the composable path — a DS deployment only needs rows added."""
    _skip_if_no_specialized()
    impl = _impl(impl_name)
    ds_rows = tuple(
        {**row, "conv_layout": "DS"}
        for row in specialized_gdn.load_gdn_fused_decode_registry()
        if row["impl"] == impl_name
    )
    assert ds_rows, "registry must not be empty"
    monkeypatch.setattr(
        specialized_gdn, "load_gdn_fused_decode_registry", lambda: ds_rows
    )

    seed = 20260714
    inputs = _make_inputs(2, padded_pool=False, seed=seed, conv_layout="DS")
    assert _matched_rows(inputs)
    launches_before = impl.launch_count()
    out, conv_state, ssm_state = gdn_fused_decode_step(**inputs)
    assert impl.launch_count() == launches_before + 1
    ref_out, ref_conv, ref_ssm = _run_reference(
        2, padded_pool=False, seed=seed, conv_layout="DS"
    )
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    assert torch.equal(conv_state, ref_conv)
    torch.testing.assert_close(ssm_state, ref_ssm, atol=ATOL, rtol=RTOL)


def test_ds_layout_not_registered_falls_back():
    """The shipped registry pins the observed SD serving layout: a DS-dense
    pool takes the composable path, stays bit-identical to it, and is
    reported unsupported by the probe."""
    _skip_if_no_specialized()
    inputs = _make_inputs(1, padded_pool=False, seed=13, conv_layout="DS")
    assert not _matched_rows(inputs)
    assert not gfd.gdn_fused_decode_step_supported(1, conv_state_layout="DS")
    out, _, _ = gdn_fused_decode_step(**inputs)
    ref_out, _, _ = _run_reference(1, padded_pool=False, seed=13, conv_layout="DS")
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)


def test_unrecognized_conv_state_strides_fall_back():
    """A conv_state view that is neither DS-dense nor a transposed SD pool
    must be refused by the guard and served by the composable path."""
    _skip_if_no_specialized()
    cutedsl = specialized_gdn._load_impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)

    def _make():
        """Inputs whose conv-state pool has an unrecognized stride pattern."""
        inputs = _make_inputs(1, padded_pool=False, seed=17)
        backing = (
            torch.randn(POOL, QKV_DIM, 2 * CONV_STATE_LEN, device="cuda").bfloat16()
            * 0.5
        )
        inputs["conv_state"] = backing[:, :, :CONV_STATE_LEN]  # stride(1) == 6
        return inputs

    inputs = _make()
    assert specialized_gdn.conv_state_layout(inputs["conv_state"]) is None
    assert specialized_gdn.signature_from_tensors(*_guard_args(inputs)) is None

    before = (cutedsl.launch_count() if cutedsl is not None else 0, cuda.launch_count())
    out, _, _ = gdn_fused_decode_step(**_make())
    after = (cutedsl.launch_count() if cutedsl is not None else 0, cuda.launch_count())
    assert after == before, "a guard-refused call must not dispatch a kernel"
    assert out.shape == (1, 1, HV, D)


def test_supported_probe():
    """The routing probe answers the default Qwen3.6-27B surface as shipped.

    It is the consumer's gating decision, so it must accept the registered
    batches and layout and decline everything else -- unregistered batches,
    a geometry that is not ours, and unknown layout tags.
    """
    _skip_if_no_specialized()
    for batch in REGISTERED_BATCHES:
        assert gfd.gdn_fused_decode_step_supported(batch)
    # The probe is the consumer's routing decision, so it must report the
    # registry as shipped: batches above the registered surface are declined
    # exactly like an unregistered geometry.
    for batch in UNREGISTERED_BATCHES:
        assert not gfd.gdn_fused_decode_step_supported(batch)
    assert not gfd.gdn_fused_decode_step_supported(3)  # not registered
    assert not gfd.gdn_fused_decode_step_supported(1, hidden_size=4096)
    # Layout is part of the registered surface: SD (the vLLM default pool
    # layout) is served, DS has no shipped row, unknown tags are refused.
    assert gfd.gdn_fused_decode_step_supported(1, conv_state_layout="SD")
    assert not gfd.gdn_fused_decode_step_supported(1, conv_state_layout="DS")
    assert not gfd.gdn_fused_decode_step_supported(1, conv_state_layout="sd")


def test_registry_drives_dispatch_both_ways(monkeypatch):
    """Shipped registry -> specialized dispatch; empty registry -> pure
    composable.  The registry is the ONLY thing that decides, so this is
    the whole gating story now that there is no environment variable."""
    _skip_if_no_specialized()
    cutedsl = specialized_gdn._load_impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)

    def _counts():
        """(cutedsl, cuda) launch counters, for before/after comparisons."""
        return (
            cutedsl.launch_count() if cutedsl is not None else 0,
            cuda.launch_count(),
        )

    ref_out, _, _ = _run_reference(1, padded_pool=False, seed=7)

    # Shipped registry: the op dispatches a specialized kernel, and the two
    # paths agree within the bf16 tolerance family.
    before = _counts()
    inputs = _make_inputs(1, padded_pool=False, seed=7)
    out_enabled, _, _ = gdn_fused_decode_step(**inputs)
    assert sum(_counts()) == sum(before) + 1
    torch.testing.assert_close(out_enabled, ref_out, atol=ATOL, rtol=RTOL)
    assert gfd.gdn_fused_decode_step_supported(1)

    # Empty registry: nothing is served, so the op is bit-exactly the
    # composable path and the probe declines.  The probe answer must flip
    # even though it was memoized a line ago -- substituting the registry
    # invalidates the memo.
    monkeypatch.setattr(specialized_gdn, "load_gdn_fused_decode_registry", tuple)
    before = _counts()
    inputs = _make_inputs(1, padded_pool=False, seed=7)
    out_stock, _, _ = gdn_fused_decode_step(**inputs)
    assert _counts() == before
    torch.testing.assert_close(out_stock, ref_out, atol=0.0, rtol=0.0)
    assert not gfd.gdn_fused_decode_step_supported(1)


def test_no_environment_gate(monkeypatch):
    """The package exposes no on/off variable, and retired gates are inert.

    A brand-new API has no in-FlashInfer alternative to fall back to, so an
    environment gate here would be a second policy surface that nobody
    measures: support is this library's answer, policy is the framework's.
    Pinned negatively so the retired variables cannot reappear without this
    test failing.
    """
    _skip_if_no_specialized()
    from flashinfer.gdn_kernels import experimental as experimental_pkg

    for module in (experimental_pkg, gfd, specialized_gdn):
        offenders = [
            name
            for name in dir(module)
            if "DISABLE" in name.upper()
            or "ENABLE_ENV" in name.upper()
            or name.endswith("_ENV")
        ]
        assert not offenders, f"{module.__name__} exposes a gate: {offenders}"

    # Retired variables must not change dispatch, the probe, or the numbers.
    cutedsl = specialized_gdn._load_impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)

    def _counts():
        """(cutedsl, cuda) launch counters, for before/after comparisons."""
        return (
            cutedsl.launch_count() if cutedsl is not None else 0,
            cuda.launch_count(),
        )

    for name, value in (
        ("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1"),
        ("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "true"),
        ("FLASHINFER_ENABLE_EXPERIMENTAL_FEATURES", "0"),
    ):
        monkeypatch.setenv(name, value)
        specialized_gdn._probe_memo.clear()  # force a real re-evaluation
        assert gfd.gdn_fused_decode_step_supported(1), name
        before = _counts()
        out, _, _ = gdn_fused_decode_step(**_make_inputs(1, padded_pool=False, seed=19))
        assert sum(_counts()) == sum(before) + 1, name
        assert out.shape == (1, 1, HV, D)
        monkeypatch.delenv(name)


def test_probe_memo_is_consistent_and_scoped(monkeypatch):
    """The probe memoizes the answer serving repeats every layer, every
    step -- without ever surviving the registry it came from.

    Cost matters here: a framework consumer calls this once per layer per
    decode step, outside the CUDA graph, so a linear registry scan per call
    is a one-directional tax at exactly the shapes the registry declines.
    """
    _skip_if_no_specialized()
    specialized_gdn._probe_memo.clear()
    assert gfd.gdn_fused_decode_step_supported(1)
    filled = len(specialized_gdn._probe_memo)
    assert filled == 1
    # Repeating the question must not grow the memo or change the answer.
    for _ in range(5):
        assert gfd.gdn_fused_decode_step_supported(1)
    assert len(specialized_gdn._probe_memo) == filled
    # A different geometry is a different question.
    assert not gfd.gdn_fused_decode_step_supported(3)
    assert len(specialized_gdn._probe_memo) == filled + 1

    # Substituting the registry drops every memoized answer: a stale True
    # would route a framework into an op that no longer serves the shape.
    _restrict_registry_to(monkeypatch, _CUDA_IMPL)
    assert gfd.gdn_fused_decode_step_supported(1)
    monkeypatch.setattr(specialized_gdn, "load_gdn_fused_decode_registry", tuple)
    assert not gfd.gdn_fused_decode_step_supported(1)


def test_no_backend_argument():
    """The op exposes no implementation selector.

    This is the API contract, not an implementation detail: a fused op has
    one behaviour, and which internal kernel serves a call is decided from
    the registry and the device.  Pinned so the selector cannot reappear.
    """
    _skip_if_no_cuda()
    assert "backend" not in inspect.signature(gdn_fused_decode_step).parameters
    inputs = _make_inputs(1, padded_pool=False, seed=23)
    with pytest.raises(TypeError, match="backend"):
        gdn_fused_decode_step(backend="cuda", **inputs)


def test_public_names_are_top_level():
    """The consumer-facing names are exported at the top level, like the
    other GDN APIs -- "experimental" is where the code lives, not how it is
    imported.  The old model namespace must not linger: a consumer probing
    for the op reads an ATTRIBUTE off flashinfer, so a stale
    flashinfer.qwen alias would make two spellings of one API."""
    import flashinfer

    assert flashinfer.gdn_fused_decode_step is gdn_fused_decode_step
    assert (
        flashinfer.gdn_fused_decode_step_supported
        is gfd.gdn_fused_decode_step_supported
    )
    assert not hasattr(flashinfer, "qwen")


def test_importing_flashinfer_does_not_import_a_kernel():
    """The capability check a consumer runs must cost nothing.

    ``getattr(flashinfer, "gdn_fused_decode_step", None)`` has to answer
    without importing the dispatch module, an impl module, the JIT toolchain
    or the optional CuTe-DSL dependency -- otherwise a consumer's fail-closed
    probe pays a compile, or fails on an install that is merely missing an
    optional dependency.  Pinned here because the lazy imports that make it
    true are easy to "simplify" away.
    """
    # No trailing dot on the kernel-package prefix: the impl package itself
    # (``...experimental.kernel``) is as much an eager import as any module
    # under it -- importing it runs the package __init__ -- and a trailing dot
    # would only match the submodules.
    probe = (
        "import sys, flashinfer;"
        "assert callable(flashinfer.gdn_fused_decode_step);"
        "assert callable(flashinfer.gdn_fused_decode_step_supported);"
        "eager = [m for m in sys.modules"
        " if 'gdn_fused_decode_specialized' in m"
        " or '.experimental.kernel' in m];"
        "print(eager)"
    )
    # Hand the child this interpreter's search path so it imports the same
    # flashinfer this test did (source checkout or installed wheel alike).
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_routing_probe_keeps_its_keyword_geometry():
    """The probe's parameters ARE the consumer-facing gating contract.

    Frameworks call it with the full layer geometry by keyword and resolve
    the route only when the signature carries ``conv_state_layout`` (a probe
    revision predating pool-layout awareness would mis-gate vLLM's default
    state-first pool).  Renaming, re-ordering or collapsing these into
    ``**kwargs`` would not break a build -- it would silently change which
    calls dispatch -- so pin the names, the order and the absence of a
    catch-all.
    """
    parameters = inspect.signature(gfd.gdn_fused_decode_step_supported).parameters
    assert list(parameters) == [
        "batch_size",
        "hidden_size",
        "n_ba",
        "qkv_dim",
        "num_qk_heads",
        "num_v_heads",
        "head_dim",
        "conv_width",
        "conv_state_len",
        "device",
        "conv_state_layout",
    ]
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    # Every geometry argument is reachable by keyword: the consumer passes
    # only ``batch_size`` positionally.
    assert all(
        p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values()
    )


def test_non_registered_batch_falls_back():
    """An unregistered batch size runs the composable path and stays correct."""
    _skip_if_no_specialized()
    inputs = _make_inputs(3, padded_pool=False, seed=11)  # B=3 not registered
    out, _, _ = gdn_fused_decode_step(**inputs)
    ref_out, _, _ = _run_reference(3, padded_pool=False, seed=11)
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("B", UNREGISTERED_BATCHES)
def test_batch_above_registered_surface_does_not_dispatch(B: int):
    """Batches off the Qwen3.6-27B shipped surface keep the stock path.

    The kernels handle 16/24/32 — they are benchmarked there and beat the
    stock chain in isolation — but the registry lists only the batches with
    a measured end-to-end win, so serving must not route them here.  Pinned
    from three angles: no registry row matches, the probe declines (which
    is what keeps a framework consumer on its own path), and the call is
    bit-identical to the composable path — not merely close, which is all a
    dispatched kernel could promise — with no specialized launch.
    """
    _skip_if_no_specialized()
    seed = 20260807 + B

    inputs = _make_inputs(B, padded_pool=False, seed=seed)
    assert not _matched_rows(inputs)
    assert not gfd.gdn_fused_decode_step_supported(B)

    impls = [
        impl
        for impl in (
            specialized_gdn._load_impl(_CUTEDSL_IMPL),
            specialized_gdn._load_impl(_CUDA_IMPL),
        )
        if impl is not None
    ]
    counts_before = [impl.launch_count() for impl in impls]
    out, _, _ = gdn_fused_decode_step(**inputs)
    ref_out, _, _ = _run_reference(B, padded_pool=False, seed=seed)
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
    assert [impl.launch_count() for impl in impls] == counts_before


def test_kernel_failure_latches_impl_off(monkeypatch):
    """A kernel failure warns, serves the call on the composable path, and
    stops dispatching that impl for the rest of the process.  With no
    explicit-request path there is no way for a kernel failure to surface
    as an exception from this op."""
    _skip_if_no_specialized()
    cutedsl = _impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)

    def _boom(*args, **kwargs):
        """Stand-in for a kernel that fails at launch."""
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(cutedsl, "execute", _boom)
    monkeypatch.setattr(cuda, "execute", _boom)

    seed = 20260723
    # Ask the probe BEFORE the failure so its True is memoized: the latch
    # has to invalidate that answer, or a consumer would keep routing into
    # an impl that has been turned off.
    assert gfd.gdn_fused_decode_step_supported(1)
    inputs = _make_inputs(1, padded_pool=False, seed=seed)
    out, _, _ = gdn_fused_decode_step(**inputs)  # served by the fallback
    ref_out, _, _ = _run_reference(1, padded_pool=False, seed=seed)
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)
    assert specialized_gdn._failed_impls == {_CUTEDSL_IMPL, _CUDA_IMPL}
    # Latched impls stay off (and the probe reports the surface as gone).
    assert not gfd.gdn_fused_decode_step_supported(1)
    # A second call still succeeds on the composable path, without
    # re-entering the failing kernels.
    out, _, _ = gdn_fused_decode_step(**_make_inputs(1, padded_pool=False, seed=seed))
    torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("impl_name", SHIPPED_IMPLS)
def test_graph_capture_after_warmup(impl_name: str, monkeypatch):
    """After a warmup call, the specialized impl must be capturable: the
    composable fallback is not consulted during capture and the replayed
    graph produces correct results.

    Every assertion here is on the impl HAVING SERVED, never on the call
    merely returning: dispatch is allowed to fall through to another impl or
    to the composable path at any point, so a test that only checks the
    result would pass while the impl under test never ran.
    """
    _skip_if_no_specialized()
    impl = _impl(impl_name)
    _restrict_registry_to(monkeypatch, impl_name)
    seed = 20260712

    # Warmup on scratch inputs: compiles the variant and creates the
    # persistent per-device workspace/barrier.  This is where a kernel that
    # cannot be compiled by the installed CuTe-DSL gets latched off, so the
    # attestation is checked here as well as after capture.
    warm = _make_inputs(1, padded_pool=True, seed=seed + 1)
    gdn_fused_decode_step(**warm)
    torch.cuda.synchronize()
    assert specialized_gdn.gdn_fused_decode_stats()["failed_impls"] == [], (
        "the impl was latched off during warmup -- read the logged reason"
    )
    assert specialized_gdn.gdn_fused_decode_stats()["served_impls"] == [impl_name]
    assert impl.compiled_variant_keys()

    inputs = _make_inputs(1, padded_pool=True, seed=seed)
    assert impl.ready_for_graph_capture(
        _signature_for(inputs),
        inputs["hidden_states"],
        inputs["conv_state"],
        inputs["scale"],
    )

    def _boom(*args, **kwargs):
        """Fail loudly if the composable path is reached at all."""
        raise AssertionError("composable fallback used during capture of a warm impl")

    real_fallback = gfd._gdn_fused_decode_step_fallback
    monkeypatch.setattr(gfd, "_gdn_fused_decode_step_fallback", _boom)
    # Framework-style pre-allocated output buffer (written in place).
    out_buf = torch.zeros((1, 1, HV, D), dtype=torch.bfloat16, device="cuda")
    launches_before = impl.launch_count()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _, _ = gdn_fused_decode_step(out=out_buf, **inputs)
    monkeypatch.setattr(gfd, "_gdn_fused_decode_step_fallback", real_fallback)
    assert out is out_buf
    # The specialized path (not the fallback) was recorded: its host-side
    # dispatch ran exactly once, at capture time.
    assert impl.launch_count() == launches_before + 1
    # ... and it is still the impl under test that served it. A kernel that
    # raises under capture would be latched off here, the graph would capture
    # someone else's work, and the replay below would still be correct.
    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["failed_impls"] == [], "an impl was latched off during capture"
    assert stats["served_impls"] == [impl_name]

    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_conv, ref_ssm = _run_reference(1, padded_pool=True, seed=seed)
    torch.testing.assert_close(out_buf, ref_out, atol=ATOL, rtol=RTOL)
    assert torch.equal(inputs["conv_state"], ref_conv)
    torch.testing.assert_close(inputs["ssm_state"], ref_ssm, atol=ATOL, rtol=RTOL)


def test_graph_capture_with_cold_caches_bakes_composable(monkeypatch):
    """With cold impl caches, capture must cleanly take the composable path
    (no compilation and no persistent allocation under capture, no
    specialized dispatch recorded into the graph)."""
    _skip_if_no_specialized()
    seed = 20260713

    # Make both impls look cold without touching the real caches.
    cuda, cutedsl = _make_impls_look_cold(monkeypatch)

    calls = {"fallback": 0}
    real_fallback = gfd._gdn_fused_decode_step_fallback

    def _spy(*args, **kwargs):
        """Count composable-path calls, then run the real one."""
        calls["fallback"] += 1
        return real_fallback(*args, **kwargs)

    monkeypatch.setattr(gfd, "_gdn_fused_decode_step_fallback", _spy)

    # Warm up the composable path outside capture (cuBLAS/kernel init is not
    # capture-safe on first use).
    warm = _make_inputs(1, padded_pool=False, seed=seed + 1)
    real_fallback(
        warm["hidden_states"],
        warm["w_ba"],
        warm["mixed_qkv"],
        warm["conv_weight"],
        warm["conv_bias"],
        warm["conv_state"],
        warm["A_log"],
        warm["dt_bias"],
        warm["scale"],
        warm["ssm_state"],
        warm["state_indices"],
        warm["use_qk_l2norm"],
    )
    torch.cuda.synchronize()

    inputs = _make_inputs(1, padded_pool=False, seed=seed)
    launches_before = (
        cutedsl.launch_count() if cutedsl is not None else 0,
        cuda.launch_count(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _, _ = gdn_fused_decode_step(**inputs)
    assert calls["fallback"] == 1
    assert (
        cutedsl.launch_count() if cutedsl is not None else 0,
        cuda.launch_count(),
    ) == launches_before, "a cold impl must not be recorded into the graph"

    graph.replay()
    torch.cuda.synchronize()
    ref_out, _, _ = _run_reference(1, padded_pool=False, seed=seed)
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)


def test_capture_never_compiles_a_cold_impl(monkeypatch):
    """Nothing may compile under capture: with a cold impl and capture in
    progress, dispatch declines and the composable path serves the call.

    Capture is simulated by patching is_current_stream_capturing so the CUDA
    context is not disturbed by an aborted real capture.
    """
    _skip_if_no_specialized()
    cuda, _ = _make_impls_look_cold(monkeypatch)
    _restrict_registry_to(monkeypatch, _CUDA_IMPL)
    inputs = _make_inputs(1, padded_pool=False, seed=29)
    assert _matched_rows(inputs), "the geometry must be registered for this test"
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    launches_before = cuda.launch_count()
    calls = {"fallback": 0}
    real_fallback = gfd._gdn_fused_decode_step_fallback

    def _spy(*args, **kwargs):
        """Count composable-path calls, then run the real one."""
        calls["fallback"] += 1
        return real_fallback(*args, **kwargs)

    monkeypatch.setattr(gfd, "_gdn_fused_decode_step_fallback", _spy)
    gdn_fused_decode_step(**inputs)
    assert calls["fallback"] == 1
    assert cuda.launch_count() == launches_before
    assert not cuda._modules, "nothing may compile during capture"


def _cute_math_primitives_used_by_the_cutedsl_impl() -> set:
    """``cute.math.*`` names the CuTe-DSL impl actually calls.

    Parsed with ``ast``, not grep: the module is READ rather than imported
    (importing it needs cutlass, and this check has to work on a node that
    does not have it), and the AST ignores the comments and docstrings that
    necessarily discuss these very primitives.
    """
    path = (
        pathlib.Path(specialized_gdn.__file__).parent
        / "kernel"
        / f"gdn_fused_decode_{_CUTEDSL_IMPL}.py"
    )
    assert path.is_file(), f"CuTe-DSL impl not found at {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "math"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "cute"
    }


def test_cutedsl_impl_only_uses_portable_cute_math_primitives():
    """The CuTe-DSL kernel may only use ``cute.math`` primitives that exist
    in the OLDEST supported nvidia-cutlass-dsl, not the newest installed one.

    Regression test.  ``cute.math.max`` exists from 4.6 on and does not exist
    in 4.5, so a kernel using it compiles on a 4.6+ box, passes the kernel
    A/B, and then raises AttributeError under a 4.5 pin -- where the failure
    latch catches it, quietly hands the workload to the other impl, and every
    downstream gate still reports green on the wrong kernel.  Nothing else in
    this suite can see that: it needs neither a GPU nor cutlass, so it is
    also the only check that runs everywhere.
    """
    used = _cute_math_primitives_used_by_the_cutedsl_impl()
    assert used, "expected the CuTe-DSL impl to call cute.math primitives"
    unportable = sorted(used - PORTABLE_CUTE_MATH_PRIMITIVES)
    assert not unportable, (
        f"the CuTe-DSL impl uses cute.math.{{{','.join(unportable)}}}, which "
        "nvidia-cutlass-dsl 4.5 does not export. Express it with the "
        "primitives in PORTABLE_CUTE_MATH_PRIMITIVES (e.g. max(x, 0) as "
        "0.5*x + 0.5*absf(x)), or raise the pinned floor deliberately and "
        "update this set."
    )


def test_portable_cute_math_primitives_exist_in_the_installed_dsl():
    """The pinned floor must also be real: every name in it has to resolve on
    whatever nvidia-cutlass-dsl is installed here.

    Guards the other direction -- a future DSL that drops one of these would
    otherwise leave the allow-list quietly lying.
    """
    # Import the SUBMODULE, not the package: ``cutlass.cute.math`` is only an
    # attribute of ``cutlass.cute`` because that package happens to import it
    # eagerly today.  Asking for the submodule by name is what actually pins
    # its contents; the attribute spelling the kernel uses is then checked to
    # resolve to the same module, so both halves of the assumption are tested.
    cute_math = pytest.importorskip(
        "cutlass.cute.math", reason="nvidia-cutlass-dsl not installed"
    )
    cute = importlib.import_module("cutlass.cute")
    assert getattr(cute, "math", None) is cute_math, (
        "the kernel reaches these primitives as cute.math.<name>, so "
        "cutlass.cute must expose the math submodule as an attribute"
    )
    missing = sorted(
        name for name in PORTABLE_CUTE_MATH_PRIMITIVES if not hasattr(cute_math, name)
    )
    assert not missing, f"installed cute.math is missing {missing}"


def test_the_documented_dsl_floor_is_below_the_repo_pin():
    """The DSL floor this package targets and the version the repo installs
    must tell one story.

    ``requirements.txt`` constrains the nvidia-cutlass-dsl a FlashInfer install
    brings along; ``CUTE_DSL_RUNTIME_FLOOR`` is the oldest release the shipped
    CuTe-DSL kernel must still compile under, because serving stacks resolve
    the DSL themselves and routinely resolve older (vLLM's image downgrades to
    4.5.2).  The floor therefore has to stay at or below the LOWEST version
    the requirement admits -- a floor above it would mean the repo's own CI
    never exercises the version the kernel claims to support.

    The operator is deliberately not assumed.  This test previously matched
    ``==`` only and broke when upstream relaxed the line from ``==4.7.0`` to
    ``>=4.6.2a0``: a requirement getting *looser* made the check fail while the
    property it guards still held.  What matters is the low end of whatever the
    line admits, which for ``==``, ``>=`` and ``~=`` alike is the version named.
    Cheap, needs nothing installed.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "requirements.txt"
    if not path.is_file():
        pytest.skip("requirements.txt is only present in a source checkout")
    requirements = path.read_text(encoding="utf-8")
    match = re.search(
        r"^nvidia-cutlass-dsl\s*(==|>=|~=)\s*(\d+)\.(\d+)",
        requirements,
        re.MULTILINE,
    )
    assert match, (
        "requirements.txt has no nvidia-cutlass-dsl requirement this test "
        "recognizes (expected ==, >= or ~=). The documented floor now has "
        "nothing to be checked against -- re-derive it rather than deleting "
        "the check"
    )
    operator = match.group(1)
    lowest_admitted = (int(match.group(2)), int(match.group(3)))
    assert lowest_admitted >= CUTE_DSL_RUNTIME_FLOOR, (
        f"the CuTe-DSL impl documents a floor of {CUTE_DSL_RUNTIME_FLOOR} but "
        f"requirements.txt admits {operator}{lowest_admitted[0]}."
        f"{lowest_admitted[1]} at the low end; lower the floor or raise the "
        "requirement, and update PORTABLE_CUTE_MATH_PRIMITIVES to match "
        "whichever moved"
    )


def _bare_current_stream_calls(path: pathlib.Path) -> list:
    """Lines calling ``...current_stream()`` with no explicit device."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "current_stream"
        and not node.args
        and not node.keywords
    ]


def test_impl_modules_name_the_device_they_take_a_stream_from():
    """An impl must read the launch stream off the tensors' device.

    ``torch.cuda.current_stream()`` with no argument answers for the AMBIENT
    device.  Dispatch makes the tensors' device current before calling an
    impl, so the two agree today -- but an impl that relies on that silently
    launches onto another device's stream the moment it is called from
    anywhere else (a bare ``impl.execute`` in a harness, or a future dispatch
    path).  Cheap to keep right, invisible on a single-GPU box, so it is
    pinned here.  AST-parsed, so it needs neither a GPU nor cutlass.
    """
    kernel_dir = pathlib.Path(specialized_gdn.__file__).parent / "kernel"
    offenders = {}
    for impl_name in SHIPPED_IMPLS:
        path = kernel_dir / f"gdn_fused_decode_{impl_name}.py"
        assert path.is_file(), f"impl module not found at {path}"
        lines = _bare_current_stream_calls(path)
        if lines:
            offenders[path.name] = lines
    assert not offenders, (
        f"impl modules call current_stream() with no device: {offenders}. "
        "Pass the tensors' device explicitly, e.g. "
        "torch.cuda.current_stream(hidden_states.device)."
    )


_PDL_WAIT = "griddepcontrol_wait"
_PDL_TRIGGER = "griddepcontrol_launch_dependents"


def _pdl_launched_kernels(tree: ast.AST) -> dict:
    """Kernel functions the module launches with ``use_pdl=True``.

    Matches the ``<kernel>(...).launch(..., use_pdl=True)`` shape the CuTe-DSL
    impls use and resolves the kernel name back to its ``FunctionDef``.
    """
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    launched = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "launch":
            continue
        if not any(
            kw.arg == "use_pdl"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        ):
            continue
        inner = node.func.value
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
            if inner.func.id in functions:
                launched[inner.func.id] = functions[inner.func.id]
    return launched


def _griddepcontrol_lines(node: ast.AST, primitive: str) -> list:
    """Line numbers of every ``...<primitive>()`` call inside ``node``."""
    return sorted(
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == primitive
    )


def _every_path_calls(body: list, primitive: str) -> bool:
    """True when every control-flow path through ``body`` runs ``primitive``.

    A statement-level call counts; an ``if`` counts only when BOTH arms do.
    Deliberately conservative: loops and ``try`` bodies are not credited,
    because neither is guaranteed to execute.  Conservative in the safe
    direction -- it can refuse a kernel that is in fact fine (say the wait
    with a real ``else``-less guard), never accept one that is not.
    """
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == primitive:
                return True
        if isinstance(stmt, ast.If) and stmt.orelse:
            if _every_path_calls(stmt.body, primitive) and _every_path_calls(
                stmt.orelse, primitive
            ):
                return True
    return False


def _impl_module_pdl_kernels() -> list:
    """``(module_name, kernel_name, FunctionDef)`` for every PDL-launched
    kernel across the shipped impl modules."""
    kernel_dir = pathlib.Path(specialized_gdn.__file__).parent / "kernel"
    found = []
    for impl_name in SHIPPED_IMPLS:
        path = kernel_dir / f"gdn_fused_decode_{impl_name}.py"
        assert path.is_file(), f"impl module not found at {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for kernel_name, node in sorted(_pdl_launched_kernels(tree).items()):
            found.append((path.name, kernel_name, node))
    return found


def test_every_pdl_kernel_waits_on_all_paths():
    """A kernel launched with PDL must call ``griddepcontrol_wait()`` on every
    path before it can read anything, and every block must reach it.

    ``use_pdl=True`` sets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
    which (CUDA programming guide, *Programmatic Dependent Launch*) tells the
    driver it "is safe to launch the secondary kernel early and not wait on the
    completion and memory flush of the primary before launching the secondary",
    and that consequently "secondary thread blocks might launch before data
    written by the primary kernel is visible".  Every input of this op is
    produced by a stream predecessor -- the layer's projections, the caller's
    metadata prep, an earlier decode step's pool update, and the workspace the
    host memsets immediately before the launch -- so a PDL kernel with no wait
    is reading them unordered.

    This is a STATIC guarantee, not a runtime race test, and that is on
    purpose.  The window is short and the data usually already there, so a
    timing test would pass on broken code nearly always and fail nowhere
    reproducibly; whether the race fires also depends on whether some
    *upstream* kernel fires a PDL trigger, which is outside this repo.  What
    can be proved here is the structural property the contract actually asks
    for: the barrier exists and is unconditionally reached.  Needs neither a
    GPU nor cutlass.
    """
    kernels = _impl_module_pdl_kernels()
    assert kernels, (
        "no use_pdl=True launch found in any shipped impl module -- either the "
        "PDL impl was removed (then delete this test) or _pdl_launched_kernels "
        "no longer matches how impls launch (then fix the matcher)"
    )
    offenders = []
    for module_name, kernel_name, node in kernels:
        if not _every_path_calls(node.body, _PDL_WAIT):
            offenders.append(f"{module_name}::{kernel_name}")
    assert not offenders, (
        f"PDL-launched kernels with no unconditional {_PDL_WAIT}(): "
        f"{offenders}. Add the wait above the kernel's first global load and "
        "above any block-role split so every block runs it (or drop use_pdl "
        "from that launch, which also costs the overlap it buys)."
    )


def test_pdl_kernels_wait_before_their_first_launch_dependents():
    """A PDL kernel that releases its dependents must be ordered itself first.

    ``griddepcontrol_launch_dependents()`` is a SCHEDULING gate -- PTX: the
    designated dependents "can be scheduled as soon as all other CTAs in the
    grid issue the same instruction or have completed" -- so a kernel that
    triggers before its own ``griddepcontrol_wait()`` puts the dependent's CTAs
    on the SMs while it is itself still unordered against its predecessors.
    Any load the dependent issues *before* its own wait (this op's
    ``delta_kernel`` deliberately prefetches ``ssm_state`` and reads
    ``state_indices`` there) then has no ordering either.  Waiting first is
    what hands the dependent an already-ordered state.

    Note what this does NOT assert: that the trigger sits after the kernel's
    stores.  It need not.  A dependent's wait blocks until every prerequisite
    grid "in flight has completed and all the memory operations from the
    prerequisite grids are performed and made visible to the current grid"
    (PTX), so the trigger publishes nothing and cannot release a dependent
    early onto unwritten data.  Where the trigger sits between the wait and
    the end of the kernel is a performance choice -- earlier means more
    overlap and more SM competition -- and pinning it would forbid the
    entry-fire pattern this impl and ``gdn_decode_bf16_wy_ucache_flush.py``
    both rely on.  Static check, no GPU or cutlass needed.
    """
    kernels = _impl_module_pdl_kernels()
    assert kernels, "no use_pdl=True launch found in any shipped impl module"
    offenders = {}
    for module_name, kernel_name, node in kernels:
        triggers = _griddepcontrol_lines(node, _PDL_TRIGGER)
        if not triggers:
            continue
        waits = _griddepcontrol_lines(node, _PDL_WAIT)
        if not waits or waits[0] > triggers[0]:
            offenders[f"{module_name}::{kernel_name}"] = {
                "first_wait_line": waits[0] if waits else None,
                "first_trigger_line": triggers[0],
            }
    assert not offenders, (
        f"PDL kernels that call {_PDL_TRIGGER}() before their own "
        f"{_PDL_WAIT}(): {offenders}. Move the wait above the trigger."
    )


def test_dispatch_serves_a_call_on_a_non_ambient_device():
    """A call whose tensors live on a device other than the current one must
    still dispatch, and must not disturb the caller's ambient device.

    TP > 1 serving drives rank r's layers on ``cuda:r``; the ambient device of
    the calling thread is not guaranteed to be that one.  The impls take their
    launch stream (and, through tvm_ffi, their launch context) from the
    current device, so dispatch sets it.  Without that the launch picks up
    ``cuda:0``'s stream: at best the kernel raises and the failure latch turns
    the impl off *for the whole process* -- every later call on every rank
    quietly runs the composable path -- at worst it runs unordered against the
    stream the caller is actually synchronizing.  Single-GPU CI cannot see any
    of that.
    """
    _skip_if_no_specialized()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    if not is_sm120a_supported(torch.device("cuda:1")):
        pytest.skip("the registered specialized fused GDN kernels target SM120")

    B, seed = 4, 20260818
    device = torch.device("cuda:1")
    inputs = _make_inputs(B, padded_pool=False, seed=seed, device=device)
    assert _matched_rows(inputs), "test geometry must be registered on cuda:1"

    with torch.cuda.device(0):
        assert torch.cuda.current_device() == 0
        out, conv_state, ssm_state = gdn_fused_decode_step(**inputs)
        # Dispatch may make the tensors' device current; it must put the
        # caller's back.
        assert torch.cuda.current_device() == 0
    torch.cuda.synchronize(device)

    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["served_impls"], (
        "the call on cuda:1 did not reach a specialized impl -- it fell "
        f"through to the composable path (failed_impls={stats['failed_impls']})"
    )
    assert stats["failed_impls"] == []

    ref_out, ref_conv, ref_ssm = _run_reference(
        B, padded_pool=False, seed=seed, device=device
    )
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    assert torch.equal(conv_state, ref_conv)
    torch.testing.assert_close(ssm_state, ref_ssm, atol=ATOL, rtol=RTOL)


def test_dispatch_attests_which_impl_served(monkeypatch):
    """A served call must be attributable to a named impl.

    The failure latch means "the op returned a result" does not identify the
    kernel that produced it, so dispatch records the impl that served and
    reports it in the stats hook.  A measurement run reads that instead of
    inferring the impl from the registry.
    """
    _skip_if_no_specialized()
    for impl_name in SHIPPED_IMPLS:
        _impl(impl_name)
    assert specialized_gdn.gdn_fused_decode_stats()["served_impls"] == []

    _restrict_registry_to(monkeypatch, _CUTEDSL_IMPL)
    gdn_fused_decode_step(**_make_inputs(1, padded_pool=False, seed=20260818))
    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["served_impls"] == [_CUTEDSL_IMPL]
    assert stats["failed_impls"] == []


def test_a_latched_impl_never_attests_as_served(monkeypatch):
    """The attestation must record the impl that RAN, not the preferred one.

    This is the shape of the regression the latch hid: the preferred impl
    fails, the next one serves, the op returns fine.  ``served_impls`` has to
    name the second one and ``failed_impls`` the first, so a harness that
    pinned the first can tell it did not get it.
    """
    _skip_if_no_specialized()
    cutedsl = _impl(_CUTEDSL_IMPL)
    cuda = _impl(_CUDA_IMPL)

    def _boom(*args, **kwargs):
        """Reproduce the DSL-surface failure shape: a missing attribute."""
        raise AttributeError("injected: module has no attribute 'max'")

    monkeypatch.setattr(cutedsl, "execute", _boom)
    seed = 20260818
    inputs = _make_inputs(1, padded_pool=False, seed=seed)
    cuda_launches = cuda.launch_count()
    out, _, _ = gdn_fused_decode_step(**inputs)

    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["failed_impls"] == [_CUTEDSL_IMPL]
    assert stats["served_impls"] == [_CUDA_IMPL]
    assert cuda.launch_count() == cuda_launches + 1
    ref_out, _, _ = _run_reference(1, padded_pool=False, seed=seed)
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)


def test_registry_shape_and_stats():
    """The shipped registry pins the traced SD surface for both impls, and
    the stats hook reports per-impl introspection.

    The batch set is the measured end-to-end win surface, not the set of
    batches the kernels can run: this assertion is what makes widening it a
    deliberate act.
    """
    rows = specialized_gdn.load_gdn_fused_decode_registry()
    assert rows, "packaged registry must load"
    by_impl: dict = {}
    for row in rows:
        assert row["conv_layout"] == "SD"
        assert row["cc"] == 120
        by_impl.setdefault(row["impl"], {}).setdefault(
            tuple(row[field] for field in specialized_gdn._GEOMETRY_FIELDS), set()
        ).add(row["b"])
    expected = {
        impl_name: {
            geometry.key(): set(geometry.batches)
            for geometry in GEOMETRIES
            if impl_name in geometry.impls
        }
        for impl_name in SHIPPED_IMPLS
    }
    assert by_impl == expected
    assert not set(UNREGISTERED_BATCHES) & set(QWEN_27B.batches)
    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["registry_entries"] == len(rows)
    assert set(stats["impls"]) == {_CUTEDSL_IMPL, _CUDA_IMPL}
    cuda_stats = stats["impls"][_CUDA_IMPL]
    if cuda_stats["distinct_kernels_for_registry"] is not None:
        # One B-dynamic CUDA module per layer geometry (the geometry is a
        # compile-time parameter; batch, scale and conv strides are not).
        assert cuda_stats["distinct_kernels_for_registry"] == len(GEOMETRIES)


def test_registry_geometries_are_the_documented_surface():
    """``registry_geometries()`` is exactly the distinct geometries the
    registry lists, and matches the table these tests are written against.

    That set is what a geometry-parameterized build (or a JIT-disabled
    deployment restoring an AOT entry) would have to cover: one CUDA module
    per geometry.  Needs no GPU.
    """
    assert specialized_gdn.registry_geometries() == [
        geometry.key() for geometry in GEOMETRIES
    ]


def test_registered_geometry_tiling_is_exact():
    """Every registered geometry satisfies the relations both kernels tile
    on, exactly.

    These are the divisibility facts the kernel bodies assume -- the CUDA
    kernel ``static_assert``s them and the CuTe-DSL impl re-checks them at
    dispatch -- so a registry row that broke one would mis-tile rather than
    fail.  Asserting them here guards the registry in plain CI, with no GPU
    and no compiler.
    """
    cutedsl = _impl(_CUTEDSL_IMPL)
    for geometry in GEOMETRIES:
        hidden, n_ba, qkv_dim, h_q, hv, d, conv_width, state_len = geometry.key()
        why = f"geometry {geometry.name}"
        # Shared by both impls.
        assert n_ba == 2 * hv, f"{why}: w_ba columns are [b gates | a decays]"
        assert hv % h_q == 0, f"{why}: each qk-head serves whole v-heads"
        assert qkv_dim == (2 * h_q + hv) * d, f"{why}: qkv_dim matches the head split"
        assert d == 4 * 32, f"{why}: one D-wide row per warp, 4 channels per lane"
        assert d & (d - 1) == 0, f"{why}: B=1 row indexing uses D as a power of two"
        assert (conv_width, state_len) == (4, 3), (
            f"{why}: conv taps unrolled as width 4"
        )
        # CUDA impl: warps own whole groups of state rows, block is 256 threads.
        assert d % 8 == 0, f"{why}: warps own whole groups of state rows"
        assert n_ba <= 256, f"{why}: the gate reduction fits one block"
        if _CUTEDSL_IMPL in geometry.impls:
            # CuTe-DSL impl: the K-split and conv tile must divide exactly.
            assert hidden % cutedsl.KS == 0, f"{why}: hidden divisible by the K-split"
            assert qkv_dim % cutedsl.CONV_TILE == 0, (
                f"{why}: qkv_dim divisible by the tile"
            )
            assert d % cutedsl.RPB == 0, f"{why}: rows-per-block divides D"
            # And the derived tile counts are the exact integers the kernel uses.
            gqa, nrb, kchunk, nconv = cutedsl._derived(hidden, qkv_dim, hv, h_q, d)
            assert (gqa, nrb, kchunk, nconv) == (
                hv // h_q,
                d // cutedsl.RPB,
                hidden // cutedsl.KS,
                qkv_dim // cutedsl.CONV_TILE,
            )
            # The impl's own guard must accept every geometry we ship for it.
            cutedsl._check_geometry(hidden, n_ba, qkv_dim, h_q, hv, d)


def test_cutedsl_geometry_guard_rejects_a_mis_tiling_geometry():
    """The CuTe-DSL geometry guard is a detector, not decoration.

    A geometry whose v-heads do not divide into its qk-heads would silently
    mis-map heads; the guard must raise so the dispatch layer falls back.
    """
    cutedsl = _impl(_CUTEDSL_IMPL)
    with pytest.raises(RuntimeError, match="unsupported fused GDN decode geometry"):
        # h_q=48 does not divide hv=32, and n_ba/qkv_dim no longer agree.
        cutedsl._check_geometry(2048, 64, 8192, 48, 32, 128)


def test_geometry_is_a_compile_time_parameter_of_the_cuda_module(monkeypatch):
    """The CUDA JIT spec bakes the geometry in, so each geometry gets its own
    module name and its own ``-D`` defines.

    Two geometries sharing a module name would collide in the on-disk JIT
    cache and serve one model's kernel to the other.  Needs no GPU: the spec
    is built, not compiled, and the target arch is pinned here so the check
    does not depend on what the node happens to have.
    """
    from flashinfer.jit import core as jit_core
    from flashinfer.jit.gdn_fused_decode import gen_gdn_fused_decode_module

    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(12, "0a")}
    )
    names = set()
    for geometry in GEOMETRIES:
        spec = gen_gdn_fused_decode_module(*geometry.key())
        names.add(spec.name)
        flags = " ".join(spec.extra_cuda_cflags)
        for key, value in zip(
            ("HIDDEN", "N_BA", "QKV_DIM", "H_Q", "HV", "D"),
            geometry.key(),
            strict=False,
        ):
            assert f"-DFI_GDN_{key}={value}" in flags
    assert len(names) == len(GEOMETRIES), "each geometry needs its own module name"


def test_cuda_scratch_cache_is_scoped_to_geometry():
    """Two geometries cannot share one launch-scratch cache entry."""
    cuda = _impl(_CUDA_IMPL)
    hidden_states = torch.empty((1, 1))
    conv_state = torch.empty((1, 1, 1))
    keys = [
        cuda._scratch_key(geometry.key(), hidden_states, conv_state)
        for geometry in GEOMETRIES
    ]
    assert len(set(keys)) == len(keys)


@pytest.mark.parametrize(
    "impl_name,geometry,B",
    [
        pytest.param(
            impl_name, geometry, B, id=f"{impl_name}-{geometry.name}-batch-{B}"
        )
        for geometry in GEOMETRIES
        for impl_name in geometry.impls
        for B in geometry.batches
    ],
)
def test_every_registered_geometry_matches_composable(
    impl_name: str, geometry: Geometry, B: int, monkeypatch
):
    """Each impl reproduces the composable path at every registered geometry.

    Run at every registered batch with the padded pool stride, which is the
    serving layout.  This is what makes the geometry a parameter rather than
    a claim.
    """
    _skip_if_no_specialized()
    _restrict_registry_to(monkeypatch, impl_name)
    impl = _impl(impl_name)
    seed = 1234
    inputs = _make_inputs(B, padded_pool=True, seed=seed, geometry=geometry)
    launches_before = impl.launch_count()
    out, conv_state, ssm_state = gdn_fused_decode_step(**inputs)
    torch.cuda.synchronize()
    assert impl.launch_count() > launches_before, (
        f"{impl_name} did not serve geometry {geometry.name} -- "
        "the registry row or the dispatch guard declined it"
    )
    assert specialized_gdn.gdn_fused_decode_stats()["failed_impls"] == []
    ref_out, ref_conv_state, ref_ssm_state = _run_reference(
        B, padded_pool=True, seed=seed, geometry=geometry
    )
    torch.testing.assert_close(out, ref_out, atol=ATOL, rtol=RTOL)
    assert torch.equal(conv_state, ref_conv_state)
    torch.testing.assert_close(ssm_state, ref_ssm_state, atol=ATOL, rtol=RTOL)


TP2_CASES = [
    pytest.param(geometry, B, id=f"{geometry.name}-batch-{B}")
    for geometry in TP2_GEOMETRIES
    for B in geometry.batches
]


def _clone_inputs(inputs: dict) -> dict:
    """Clone tensors so candidate and reference own independent state pools."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def _assert_result_close(candidate, reference) -> None:
    torch.testing.assert_close(candidate[0], reference[0], atol=ATOL, rtol=RTOL)
    assert torch.equal(candidate[1], reference[1])
    torch.testing.assert_close(candidate[2], reference[2], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize(
    "geometry,B",
    [
        pytest.param(QWEN35_27B_TP2, 24, id="qwen3.5-27b-tp2-batch-24"),
        pytest.param(QWEN35_27B_TP2, 32, id="qwen3.5-27b-tp2-batch-32"),
        pytest.param(QWEN35_35B_A3B_TP2, 3, id="qwen3.5-35b-a3b-tp2-batch-3"),
    ],
)
def test_tp2_unregistered_shapes_fail_closed(geometry: Geometry, B: int):
    """Unregistered TP2 shapes use the bit-exact composable path."""
    _skip_if_no_specialized()
    inputs = _make_inputs(B, padded_pool=False, seed=20264824 + B, geometry=geometry)
    reference = _clone_inputs(inputs)
    assert not _matched_rows(inputs)
    assert not gfd.gdn_fused_decode_step_supported(
        B,
        hidden_size=geometry.hidden,
        n_ba=geometry.n_ba,
        qkv_dim=geometry.qkv_dim,
        num_qk_heads=geometry.h_q,
        num_v_heads=geometry.hv,
        head_dim=geometry.d,
        conv_width=geometry.conv_width,
        conv_state_len=geometry.conv_state_len,
    )

    impls = [
        impl
        for impl in (
            specialized_gdn._load_impl(_CUTEDSL_IMPL),
            specialized_gdn._load_impl(_CUDA_IMPL),
        )
        if impl is not None
    ]
    counts_before = [impl.launch_count() for impl in impls]
    candidate_result = gdn_fused_decode_step(**inputs)
    reference_result = gfd._gdn_fused_decode_step_fallback(**reference)
    assert [impl.launch_count() for impl in impls] == counts_before
    for candidate_tensor, reference_tensor in zip(
        candidate_result, reference_result, strict=True
    ):
        assert torch.equal(candidate_tensor, reference_tensor)


@pytest.mark.parametrize("geometry,B", TP2_CASES)
def test_tp2_cuda_stateful_sequence_matches_composable(geometry: Geometry, B: int):
    """Every TP2 row preserves recurrent state over changing inputs."""
    _skip_if_no_specialized()
    cuda = _impl(_CUDA_IMPL)
    candidate = _make_inputs(B, padded_pool=False, seed=20260824 + B, geometry=geometry)
    reference = _clone_inputs(candidate)
    launches_before = cuda.launch_count()

    for step in range(4):
        if step:
            fresh = _make_inputs(
                B,
                padded_pool=False,
                seed=20260824 + B + 1000 * step,
                geometry=geometry,
            )
            for field in ("hidden_states", "mixed_qkv"):
                candidate[field].copy_(fresh[field])
                reference[field].copy_(fresh[field])
        _assert_result_close(
            gdn_fused_decode_step(**candidate),
            gfd._gdn_fused_decode_step_fallback(**reference),
        )

    assert cuda.launch_count() == launches_before + 4
    assert specialized_gdn.gdn_fused_decode_stats()["failed_impls"] == []


@pytest.mark.parametrize("geometry,B", TP2_CASES)
def test_tp2_cuda_changing_input_graph_replay(geometry: Geometry, B: int):
    """Every TP2 row remains correct across changing-input graph replays."""
    _skip_if_no_specialized()
    cuda = _impl(_CUDA_IMPL)
    warm = _make_inputs(B, padded_pool=False, seed=20261824 + B, geometry=geometry)
    gdn_fused_decode_step(**warm)
    torch.cuda.synchronize()
    assert specialized_gdn.gdn_fused_decode_stats()["served_impls"] == [_CUDA_IMPL]

    candidate = _make_inputs(B, padded_pool=False, seed=20262824 + B, geometry=geometry)
    reference = _clone_inputs(candidate)
    initial_conv = candidate["conv_state"].clone()
    initial_ssm = candidate["ssm_state"].clone()
    out = torch.empty(
        (B, 1, geometry.hv, geometry.d), dtype=torch.bfloat16, device="cuda"
    )
    assert cuda.ready_for_graph_capture(
        _signature_for(candidate),
        candidate["hidden_states"],
        candidate["conv_state"],
        candidate["scale"],
    )
    launches_before = cuda.launch_count()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gdn_fused_decode_step(out=out, **candidate)
    assert cuda.launch_count() == launches_before + 1
    candidate["conv_state"].copy_(initial_conv)
    candidate["ssm_state"].copy_(initial_ssm)

    for step in range(3):
        fresh = _make_inputs(
            B,
            padded_pool=False,
            seed=20262824 + B + 1000 * step,
            geometry=geometry,
        )
        for field in ("hidden_states", "mixed_qkv"):
            candidate[field].copy_(fresh[field])
            reference[field].copy_(fresh[field])
        graph.replay()
        reference_result = gfd._gdn_fused_decode_step_fallback(**reference)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference_result[0], atol=ATOL, rtol=RTOL)
        assert torch.equal(candidate["conv_state"], reference["conv_state"])
        torch.testing.assert_close(
            candidate["ssm_state"], reference["ssm_state"], atol=ATOL, rtol=RTOL
        )

    assert specialized_gdn.gdn_fused_decode_stats()["failed_impls"] == []


@pytest.mark.parametrize("geometry", TP2_GEOMETRIES, ids=lambda geometry: geometry.name)
def test_tp2_cuda_padded_rows_do_not_update_state(geometry: Geometry):
    """TP2 graph padding writes zero output without touching either state pool."""
    _skip_if_no_specialized()
    cuda = _impl(_CUDA_IMPL)
    B = 4
    candidate = _make_inputs(B, padded_pool=False, seed=20263824, geometry=geometry)
    candidate["state_indices"].copy_(
        torch.tensor([POOL - 1, POOL - 3, -1, -1], device="cuda", dtype=torch.int32)
    )
    reference = _clone_inputs(candidate)
    before = _clone_inputs(candidate)

    launches_before = cuda.launch_count()
    candidate_result = gdn_fused_decode_step(**candidate)
    assert cuda.launch_count() == launches_before + 1
    _assert_result_close(
        candidate_result, gfd._gdn_fused_decode_step_fallback(**reference)
    )
    assert torch.count_nonzero(candidate_result[0][2:]) == 0
    unchanged = [slot for slot in range(POOL) if slot not in {POOL - 1, POOL - 3}]
    assert torch.equal(
        candidate["conv_state"][unchanged], before["conv_state"][unchanged]
    )
    assert torch.equal(
        candidate["ssm_state"][unchanged], before["ssm_state"][unchanged]
    )


if __name__ == "__main__":
    pytest.main([__file__])
