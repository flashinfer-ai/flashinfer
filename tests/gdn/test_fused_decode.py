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
- the internal preference order (cute_dsl before cuda) and its fallthrough
  to the composable path;
- no environment gate: the retired kill switch (and any other
  FLASHINFER_* variable) changes nothing about dispatch or the probe;
- the shape of the shipped registry (decode batches 1/2/4/8 per impl) and
  the batches deliberately left off it: 16/24/32 must not dispatch, must
  not be reported supported by the routing probe, and must stay
  bit-identical to the composable path;
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
- multi-device dispatch: a call whose tensors live on a device other than
  the ambient one must still reach the kernel (TP > 1 serving does exactly
  that) and must leave the caller's ambient device alone, and no impl may
  take its launch stream from an unnamed device;
- the probe memo more generally: it must answer a repeated question
  cheaply without ever outliving the registry it was derived from.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import math
import os
import pathlib
import re
import subprocess
import sys

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

REGISTERED_BATCHES = (1, 2, 4, 8)
# Batches the kernels handle but the registry deliberately does not list:
# faster in a kernel A/B, no end-to-end serving win, so they keep the stock
# path (rationale in flashinfer/gdn_kernels/experimental/README.md).
UNREGISTERED_BATCHES = (16, 24, 32)

HIDDEN = 5120
N_BA = 96
QKV_DIM = 10240
HV = 48
D = 128
CONV_WIDTH = 4
CONV_STATE_LEN = 3
# One distinct pool slot per row of the largest batch any test builds inputs
# for -- the 16/24/32 guard rows included: they never dispatch, but they still
# run the composable path over `state_indices`, and `_make_inputs` walks the
# slots downwards from POOL-1.  A pool that only just fits (or does not fit)
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


def _make_conv_state(conv_layout: str, device) -> torch.Tensor:
    """Conv-state pool as the logical [P, QKV_DIM, CONV_STATE_LEN] view the
    op consumes: SD = physical (state_len, dim) rows passed transposed (the
    vLLM default allocation), DS = dense (dim, state_len) rows."""
    if conv_layout == "SD":
        pool = (
            torch.randn(POOL, CONV_STATE_LEN, QKV_DIM, device=device).bfloat16() * 0.5
        )
        return pool.transpose(-1, -2)
    assert conv_layout == "DS"
    return torch.randn(POOL, QKV_DIM, CONV_STATE_LEN, device=device).bfloat16() * 0.5


def _make_inputs(
    B: int,
    *,
    padded_pool: bool,
    seed: int,
    conv_layout: str = "SD",
    device="cuda",
    saturate_gate: bool = False,
) -> dict:
    """Build one input set for the registered layer geometry.

    Scales are chosen so the gates stay in their ordinary range; pass
    ``saturate_gate`` for the overflow regime random inputs never reach.
    ``padded_pool`` reproduces vLLM's padded ssm-pool row stride, and
    ``conv_layout`` picks the physical conv-pool layout.  Each batch row gets
    its own pool slot, walking downwards from ``POOL - 1``.
    """
    assert B < POOL, "every batch row needs its own state-pool slot"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    inputs = {
        "hidden_states": torch.randn(B, HIDDEN, device=device).bfloat16() * 0.5,
        "w_ba": torch.randn(HIDDEN, N_BA, device=device).bfloat16() * 0.02,
        # Serving passes mixed_qkv as a row-strided view into the wider fused
        # qkvz projection; reproduce that layout (values live in [:, :QKV_DIM]).
        "mixed_qkv": (torch.randn(B, QKV_DIM + 2048, device=device).bfloat16() * 0.5)[
            :, :QKV_DIM
        ],
        "conv_weight": torch.randn(QKV_DIM, CONV_WIDTH, device=device).bfloat16() * 0.3,
        "conv_bias": torch.randn(QKV_DIM, device=device).bfloat16() * 0.1,
        "conv_state": _make_conv_state(conv_layout, device),
        "A_log": torch.randn(HV, device=device).float() * 0.5,
        "dt_bias": torch.randn(HV, device=device).bfloat16() * 0.1,
        "scale": 1.0 / math.sqrt(D),
        "state_indices": torch.arange(POOL - 1, POOL - 1 - B, -1, device=device).int(),
        "use_qk_l2norm": True,
    }
    if padded_pool:
        backing = torch.randn(POOL * PADDED_ROW_STRIDE, device=device).float() * 0.05
        inputs["ssm_state"] = backing.as_strided(
            (POOL, HV, D, D), (PADDED_ROW_STRIDE, D * D, D, 1)
        )
    else:
        inputs["ssm_state"] = torch.randn(POOL, HV, D, D, device=device).float() * 0.05
    if saturate_gate:
        # Decay gate g = exp(-exp(A_log) * softplus(a + dt_bias)).  Push the
        # softplus argument to ~100, past the point where exp() overflows in
        # fp32 (~88.7), while keeping exp(A_log) = exp(-6) small enough that
        # the true gate stays O(1): exp(-exp(-6) * 100) = 0.78.  A softplus
        # written as log(1 + exp(x)) returns +inf here and collapses g to 0.
        inputs["dt_bias"] = torch.full((HV,), 100.0, device=device).bfloat16()
        inputs["A_log"] = torch.full((HV,), -6.0, device=device).float()
    return inputs


def _run_reference(
    B: int,
    *,
    padded_pool: bool,
    seed: int,
    conv_layout: str = "SD",
    saturate_gate: bool = False,
    device="cuda",
):
    """Composable-path result on an identically-seeded fresh input set."""
    ref_inputs = _make_inputs(
        B,
        padded_pool=padded_pool,
        seed=seed,
        conv_layout=conv_layout,
        saturate_gate=saturate_gate,
        device=device,
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

    The CUDA impl memoizes its loaded JIT module in a module-level
    ``functools.cache``.  Calling ``_get_module.cache_clear()`` would empty
    that PROCESS-GLOBAL memo, and monkeypatch cannot put the entry back at
    teardown -- so every later test (and every later call in the session)
    pays a module reload, and a test that happens to assert on residency
    inherits whichever order pytest chose.  Swapping in a fresh cache over
    the same underlying function is equivalent for the test and is undone
    automatically, because monkeypatch restores the attribute itself.
    """
    cuda = _impl(_CUDA_IMPL)
    monkeypatch.setattr(
        cuda, "_get_module", functools.cache(cuda._get_module.__wrapped__)
    )
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
    """The routing probe answers exactly the registry as shipped.

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
    """The package exposes no on/off variable, and the retired kill switch
    is inert.

    A brand-new API has no in-FlashInfer alternative to fall back to, so an
    environment gate here would be a second policy surface that nobody
    measures: support is this library's answer, policy is the framework's.
    Pinned negatively so neither the retired variable nor a replacement can
    reappear without this test failing.
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

    # The retired variable (and a plausible replacement) must not change
    # dispatch, the probe, or the numbers.
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
        ("FLASHINFER_QWEN_GDN_FUSED_DECODE_DISABLE", "1"),
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
    """Batches off the shipped surface keep the stock path.

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
        inputs["hidden_states"], inputs["conv_state"], inputs["scale"]
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
    assert not cuda._module_is_resident(), "nothing may compile during capture"


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

    ``requirements.txt`` pins the nvidia-cutlass-dsl a FlashInfer install
    brings along; ``CUTE_DSL_RUNTIME_FLOOR`` is the oldest release the shipped
    CuTe-DSL kernel must still compile under, because serving stacks resolve
    the DSL themselves and routinely pin older (vLLM's image downgrades to
    4.5.2).  The floor therefore has to stay at or below the pin -- a floor
    ABOVE it would mean the repo's own CI never exercises the version the
    kernel claims to support, and a floor equal to it would mean the
    portability allow-list can be widened.  Cheap, needs nothing installed.
    """
    path = pathlib.Path(__file__).resolve().parents[2] / "requirements.txt"
    if not path.is_file():
        pytest.skip("requirements.txt is only present in a source checkout")
    requirements = path.read_text(encoding="utf-8")
    match = re.search(
        r"^nvidia-cutlass-dsl\s*==\s*(\d+)\.(\d+)", requirements, re.MULTILINE
    )
    assert match, "requirements.txt no longer pins nvidia-cutlass-dsl with =="
    pinned = (int(match.group(1)), int(match.group(2)))
    assert pinned >= CUTE_DSL_RUNTIME_FLOOR, (
        f"the CuTe-DSL impl documents a floor of {CUTE_DSL_RUNTIME_FLOOR} but "
        f"the repo pins {pinned}; lower the floor or raise the pin, and update "
        "PORTABLE_CUTE_MATH_PRIMITIVES to match whichever moved"
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
        by_impl.setdefault(row["impl"], set()).add(row["b"])
    assert by_impl == {
        _CUTEDSL_IMPL: set(REGISTERED_BATCHES),
        _CUDA_IMPL: set(REGISTERED_BATCHES),
    }
    assert not set(UNREGISTERED_BATCHES) & set(REGISTERED_BATCHES)
    stats = specialized_gdn.gdn_fused_decode_stats()
    assert stats["registry_entries"] == len(rows)
    assert set(stats["impls"]) == {_CUTEDSL_IMPL, _CUDA_IMPL}
    cuda_stats = stats["impls"][_CUDA_IMPL]
    if cuda_stats["distinct_kernels_for_registry"] is not None:
        # One B-dynamic CUDA module serves every row.
        assert cuda_stats["distinct_kernels_for_registry"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
