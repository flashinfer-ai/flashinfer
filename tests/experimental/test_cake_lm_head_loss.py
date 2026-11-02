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

Backend suite of the non-deterministic chunked LM-head + loss package
``flashinfer.experimental.cake.cake_lm_head_loss``.

The package is the ``deterministic=False`` form of the deterministic package
``flashinfer.experimental.cake_lm_head_det_loss`` (``test_cake_lm_head_det_loss.py``): the same
precision boundaries, with the dX GEMM's K-slice partials of a chunk of more than one slice added by one
launch into the ``-0.0``-filled FP32 accumulator in an unordered FP32 reduction (a one-slice chunk keeps the
deterministic plain store).
Its outputs are held to the deterministic suite's accuracy budgets against the FP64 oracle and the
unchunked PyTorch path (the same ``GATE_MARGIN`` / ``GATE_TINY`` / ``PRODUCTION_B0``, checked
against that suite's source below), over repeated runs (worst case), and compared with the
deterministic package on the same inputs: ``loss`` / ``logp`` / ``lse`` and ``dW`` keep their fixed
order and equal its bits; ``dX`` meets the budgets and its distance to the deterministic ``dX`` is
reported.  The host layer (mode contract, PyTorch's deterministic-algorithms alert, the reference
backend) runs anywhere; the device tests need an SM100 / SM103 GPU and the package's rendered
programs.
"""

import ast
import contextlib
import dataclasses
import functools
import gc
import inspect
import math
import os
import subprocess
import sys
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint
from torch.utils._python_dispatch import TorchDispatchMode

from tests.test_helpers.cake_lm_head_loss_reference import (
    error_report,
    make_inputs,
    reference_fp64,
    reference_unchunked,
    rel_l2,
)
from tests.test_helpers.cake_package import cake_package

# The package under test, named once through ``cake_package``; the deterministic package is the comparison arm.
pkg = cake_package("cake.cake_lm_head_loss")
cake_backend = cake_package("cake.cake_lm_head_loss", "cake_backend")
opt_in = cake_package("cake.cake_lm_head_loss", "cake_backend_opt_in")
cake_jit = cake_package("cake.cake_lm_head_loss", "cake_jit")
cake_build = cake_package("cake.cake_lm_head_loss", "cake_build")
det_backend = cake_package("cake_lm_head_det_loss", "cake_backend")
det_opt_in = cake_package("cake_lm_head_det_loss", "cake_backend_opt_in")

PACKAGE_DIR = Path(pkg.__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = torch.device("cuda")
SEED = 20261031
# Host-layer shapes (the reference backend runs at any multiple of 256).
H_HOST, V_HOST = 256, 512
# The deterministic suite's accuracy budgets, unchanged (``test_tolerances_are_the_deterministic_suites``):
# at most 1.05x the error of the unchunked PyTorch path on the same inputs plus a floor per metric, and below
# twice the production-geometry errors of that path.
GATE_MARGIN = 1.05
GATE_TINY = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
PRODUCTION_B0 = dict(dX_rel_l2=1.85e-3, dW_rel_l2=1.88e-3, logp_max_abs=4.1e-3)
# Repeated runs of the worst-case tests (at least 20).
REPEATS = 20
# The non-deterministic mode is the only one this package serves.
ND = dict(deterministic=False)
# Seconds a two-thread test waits for a barrier or a thread before it fails
TIMEOUT_S = 120


def _det_suite_constants() -> dict:
    """``GATE_MARGIN`` / ``GATE_TINY`` / ``PRODUCTION_B0`` as the deterministic suite's source assigns them (read, not
    imported: importing that module would run its module-level package lookups)."""
    path = Path(__file__).with_name("test_cake_lm_head_det_loss.py")
    found = {}
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("GATE_MARGIN", "GATE_TINY", "PRODUCTION_B0")
        ):
            found[node.targets[0].id] = eval(  # noqa: S307 -- literal constants of a repository file
                compile(ast.Expression(node.value), str(path), "eval"), {"dict": dict}
            )
    return found


@pytest.fixture(autouse=True)
def _zero_weight_split_at_every_size():
    """The zero-weight split at every call size: the suite's small calls exercise it (the shipped floor,
    :data:`cake_backend.ZERO_WEIGHT_MIN_ROWS_DEFAULT`, keeps calls below 4096 rows unsplit; its own test sets it)."""
    previous = cake_backend.set_zero_weight_min_rows(0)
    try:
        yield
    finally:
        cake_backend.set_zero_weight_min_rows(previous)


@contextlib.contextmanager
def _slice_policy(policy: str):
    """Run under the dX slice rule's ``policy`` (:func:`cake_backend.set_dx_slice_policy`), then restore the previous."""
    previous = cake_backend.set_dx_slice_policy(policy)
    try:
        yield
    finally:
        cake_backend.set_dx_slice_policy(previous)


@contextlib.contextmanager
def _torch_deterministic(enabled: bool, warn_only: bool = False):
    was, was_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=was_warn)


def _host_inputs(T, objective="ce", *, seed=SEED, device=DEV, **kw):
    return make_inputs(
        T, objective=objective, seed=seed + T, H=H_HOST, V=V_HOST, device=device, **kw
    )


# The mode and alert tests run on the CPU: under torch's deterministic algorithms some CUDA operators of the
# reference backend raise their own alert, which is not the one under test.
CPU = torch.device("cpu")


_OPT_IN_KEYWORDS = {"memory_mode", "num_valid_rows", "check_num_valid_rows"}


def _run(
    inp,
    C,
    *,
    package="nd",
    entry="loss",
    train_x=True,
    train_w=True,
    backend="reference",
    scale=None,
    keep=False,
    **switches,
):
    """One forward + backward through the autograd entry of ``package`` (``"nd"``: this package with
    ``deterministic=False``; ``"det"``: the deterministic package) on ``backend``.  ``switches`` pass through
    (``memory_mode`` / ``num_valid_rows`` / ``check_num_valid_rows`` select the opt-in module); ``keep`` calls the
    log-probability entry with the kept logits (``chunked_lm_head_logprob_keep`` /
    ``chunked_lm_head_det_logprob_keep``).  Returns ``loss`` / ``logp`` / ``dX`` / ``dW`` and the leaves."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    opt = bool(_OPT_IN_KEYWORDS & switches.keys())
    if package == "nd":
        be = opt_in if opt else cake_backend
        loss_fn, logprob_fn = be.chunked_lm_head_loss, be.chunked_lm_head_logprob
        mode = ND
    else:
        be = det_opt_in if opt else det_backend
        loss_fn, logprob_fn = (
            be.chunked_lm_head_det_loss,
            be.chunked_lm_head_det_logprob,
        )
        mode = {}
    if keep:
        logprob_fn = (
            cake_backend.chunked_lm_head_logprob_keep
            if package == "nd"
            else det_backend.chunked_lm_head_det_logprob_keep
        )
    if entry == "loss":
        loss, logp = loss_fn(
            X,
            W,
            inp.labels,
            objective=inp.objective,
            loss_div=inp.loss_div if inp.objective == "ce" else None,
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=C,
            return_logp=True,
            backend=backend,
            **mode,
            **switches,
        )
        if train_x or train_w:
            (loss if scale is None else loss * scale).backward()
    else:
        logp = logprob_fn(
            X, W, inp.labels, chunk_size=C, backend=backend, **mode, **switches
        )
        loss = (logp.detach() * inp.dlogp)[inp.valid].sum()
        if train_x or train_w:
            logp.backward(inp.dlogp if scale is None else inp.dlogp * scale)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad, X=X, W=W)


def _check_against_references(result, inp, *, entry="loss", ceiling=False):
    """The deterministic suite's gates, unchanged: every metric at most ``GATE_MARGIN`` times the unchunked path's
    error against the FP64 oracle plus ``GATE_TINY``; with ``ceiling`` also below twice ``PRODUCTION_B0``; exact zeros
    on the ignored rows."""
    oracle = reference_fp64(inp, entry=entry)
    b0 = reference_unchunked(inp, entry=entry)
    errors = error_report(result, oracle, inp.labels)
    b0_errors = error_report(b0, oracle, inp.labels)
    assert not errors["nan"]
    for key, tiny in GATE_TINY.items():
        if key in errors:
            assert errors[key] <= GATE_MARGIN * b0_errors[key] + tiny, (
                f"{key}: {errors[key]:.3e} vs unchunked {b0_errors[key]:.3e}"
            )
    if ceiling:
        for key, measured in PRODUCTION_B0.items():
            if key in errors:
                assert errors[key] <= 2.0 * measured, (
                    f"{key}: {errors[key]:.3e} above twice the production-geometry unchunked error"
                )
    assert errors["logp_ignored_zero"]
    if result.get("dX") is not None:
        assert errors["dX_ignored_zero"]
    return errors, b0_errors


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a float tensor as integers (``torch.equal`` on floats equates +0 and -0)."""
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _ulp_flips(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    """Fraction of BF16 elements that differ, and the largest difference in units in the last place."""
    ia, ib = _bits(a).to(torch.int32), _bits(b).to(torch.int32)
    # ordered integer view of the BF16 bit patterns (sign-magnitude -> two's complement order)
    oa = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    diff = (oa - ob).abs()
    return float((diff != 0).float().mean()), int(diff.max()) if diff.numel() else 0


# ---------------------------------------------------------------------------
# Host layer (any device; the reference backend)
# ---------------------------------------------------------------------------


def test_tolerances_are_the_deterministic_suites():
    """The accuracy budgets of this suite are the deterministic suite's (the same budgets for both paths)."""
    assert _det_suite_constants() == dict(
        GATE_MARGIN=GATE_MARGIN, GATE_TINY=GATE_TINY, PRODUCTION_B0=PRODUCTION_B0
    )


# The names of the package without ``det`` and the deterministic package's name of each.
NAMES = {
    "cake_backend": {
        "prepare_lm_head_loss": "prepare_lm_head_det_loss",
        "validate_lm_head_inputs": "validate_lm_head_det_inputs",
        "lm_head_loss_workspace_size": "lm_head_det_loss_workspace_size",
        "LmHeadLossRunner": "LmHeadDetLossRunner",
        "ChunkedLmHeadLossFunction": "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadLogprobFunction": "ChunkedLmHeadDetLogprobFunction",
        "chunked_lm_head_loss": "chunked_lm_head_det_loss",
        "chunked_lm_head_logprob": "chunked_lm_head_det_logprob",
        "load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module",
        "forward_loss": "forward_loss",
        "backward_loss": "backward_loss",
        "forward_logprob": "forward_logprob",
        "backward_logprob": "backward_logprob",
        "ChunkedLmHeadLogprobKeepFunction": "ChunkedLmHeadDetLogprobKeepFunction",
        "chunked_lm_head_logprob_keep": "chunked_lm_head_det_logprob_keep",
        "forward_logprob_keep": "forward_logprob_keep",
        "backward_logprob_keep": "backward_logprob_keep",
        "KeptForwardResult": "KeptForwardResult",
        "kept_memory_report": "kept_memory_report",
        "check_kept_logits": "check_kept_logits",
    },
    "cake_backend_opt_in": {
        "prepare_lm_head_loss": "prepare_lm_head_det_loss",
        "LmHeadLossRunner": "LmHeadDetLossRunner",
        "ChunkedLmHeadLossFunction": "ChunkedLmHeadDetLossFunction",
        "ChunkedLmHeadLogprobFunction": "ChunkedLmHeadDetLogprobFunction",
        "chunked_lm_head_loss": "chunked_lm_head_det_loss",
        "chunked_lm_head_logprob": "chunked_lm_head_det_logprob",
        "preload": "preload",
        "memory_plan": "memory_plan",
    },
    "cake_support": {
        "SupportLogprobFunction": "SupportDetLogprobFunction",
        "chunked_lm_head_support_logprob": "chunked_lm_head_det_support_logprob",
        "chunked_lm_head_support_kl": "chunked_lm_head_det_support_kl",
    },
    "cake_jit": {"load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module"},
}


def test_names_are_the_deterministic_names_without_det():
    """Every name of the package is the deterministic package's name without ``det``, defined by this package (not
    the deterministic object), and no name of the package carries ``det``."""
    for module_name, names in NAMES.items():
        nd = cake_package("cake.cake_lm_head_loss", module_name)
        det = cake_package("cake_lm_head_det_loss", module_name)
        for new, old in names.items():
            obj = getattr(nd, new)
            assert obj is not getattr(det, old), (module_name, new)
            # defined in this package (``cake_backend`` re-exports ``cake_jit``'s loader)
            assert getattr(obj, "__module__", nd.__name__).startswith(
                pkg.__name__ + "."
            ), (module_name, new)
        for name in vars(nd):
            assert "_det_" not in name and "Det" not in name, (module_name, name)
    assert cake_jit.library_stem("x") == "cake_lm_head_loss_x"
    for record_name in cake_jit.MODULES:
        assert record_name.startswith("cake_lm_head_loss_"), record_name


def test_loader_is_the_deterministic_packages_bytes():
    """``cake_build.py`` is the shared ``cake-build/1`` loader, byte for byte the deterministic package's."""
    det_dir = Path(det_backend.__file__).resolve().parent
    assert (PACKAGE_DIR / "cake_build.py").read_bytes() == (
        det_dir / "cake_build.py"
    ).read_bytes()


def _mode_calls(X, W, labels):
    """Every entry of the package that takes ``deterministic``, as ``name -> call(deterministic)`` on host tensors
    (the reference backend, so a call that passes the mode check computes on the CPU)."""
    common = dict(chunk_size=4, backend="reference")
    loss_kw = dict(common, loss_div=1.0)
    return {
        "validate_lm_head_inputs": lambda d: cake_backend.validate_lm_head_inputs(
            X, W, labels, loss_div=1.0, deterministic=d
        ),
        "forward_loss": lambda d: cake_backend.forward_loss(
            X, W, labels, deterministic=d, **loss_kw
        ),
        "forward_logprob": lambda d: cake_backend.forward_logprob(
            X, W, labels, deterministic=d, **common
        ),
        "backward_logprob": lambda d: cake_backend.backward_logprob(
            X,
            W,
            labels,
            torch.zeros(labels.shape[0]),
            torch.zeros(labels.shape[0]),
            deterministic=d,
            **common,
        ),
        "prepare_lm_head_loss": lambda d: cake_backend.prepare_lm_head_loss(
            X, W, labels, deterministic=d, **loss_kw
        ),
        "chunked_lm_head_loss": lambda d: cake_backend.chunked_lm_head_loss(
            X, W, labels, deterministic=d, **loss_kw
        ),
        "chunked_lm_head_logprob": lambda d: cake_backend.chunked_lm_head_logprob(
            X, W, labels, deterministic=d, **common
        ),
        "opt_in.forward_loss": lambda d: opt_in.forward_loss(
            X, W, labels, deterministic=d, **loss_kw
        ),
        "opt_in.prepare_lm_head_loss": lambda d: opt_in.prepare_lm_head_loss(
            X, W, labels, deterministic=d, **loss_kw
        ),
        "opt_in.chunked_lm_head_loss": lambda d: opt_in.chunked_lm_head_loss(
            X, W, labels, deterministic=d, memory_mode="fast", **loss_kw
        ),
        "opt_in.chunked_lm_head_logprob": lambda d: opt_in.chunked_lm_head_logprob(
            X, W, labels, deterministic=d, memory_mode="fast", **common
        ),
    }


@pytest.mark.parametrize("value", [True, None, 1, 0, "False"])
def test_every_entry_refuses_any_mode_but_false(value):
    """``deterministic`` other than ``False`` (the default ``True`` included; ``0`` and other falsy values too) is a
    ``ValueError`` that names the deterministic package, raised before any work."""
    inp = _host_inputs(8, device=CPU)
    for name, call in _mode_calls(inp.X, inp.W, inp.labels).items():
        with pytest.raises(ValueError, match="cake_lm_head_det_loss") as info:
            call(value)
        assert "deterministic=False" in str(info.value), name
    # the defaults refuse too: every entry of this package must be asked for the non-deterministic mode
    with pytest.raises(ValueError, match="deterministic=False"):
        cake_backend.chunked_lm_head_loss(
            inp.X, inp.W, inp.labels, loss_div=1.0, backend="reference"
        )


def test_every_entry_accepts_false():
    inp = _host_inputs(8, device=CPU)
    for call in _mode_calls(inp.X, inp.W, inp.labels).values():
        call(False)  # no exception: the reference backend computes on the host


def test_autograd_records_the_mode():
    inp = _host_inputs(8, device=CPU)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss = cake_backend.chunked_lm_head_loss(
        X, W, inp.labels, loss_div=1.0, chunk_size=4, backend="reference", **ND
    )
    assert loss.grad_fn.deterministic is False
    logp = cake_backend.chunked_lm_head_logprob(
        X, W, inp.labels, chunk_size=4, backend="reference", **ND
    )
    assert logp.grad_fn.deterministic is False


def _alerting_calls(inp):
    """``name -> call()`` of every path that launches the gradient GEMMs (each raises PyTorch's alert under torch's
    deterministic algorithms), on the reference backend."""
    X, W, labels = inp.X, inp.W, inp.labels
    kw = dict(chunk_size=4, backend="reference", **ND)

    def deferred_backward():
        with _torch_deterministic(False):
            fwd = cake_backend.forward_loss(
                X, W, labels, loss_div=1.0, fuse_dw_cast=True, **kw
            )
        assert (
            fwd.dz_last is not None
        )  # the plan deferred the last chunk's weight-gradient GEMM
        return fwd.backward()

    def runner_forward():
        with _torch_deterministic(False):
            runner = cake_backend.prepare_lm_head_loss(X, W, labels, loss_div=1.0, **kw)
        return runner.forward()

    def runner_backward_logprob():
        with _torch_deterministic(False):
            runner = cake_backend.prepare_lm_head_loss(
                X, W, labels, entry="logprob", **kw
            )
            runner.forward()
        return runner.backward()

    def opt_in_runner_forward():
        with _torch_deterministic(False):
            runner = opt_in.prepare_lm_head_loss(X, W, labels, loss_div=1.0, **kw)
        return runner.forward()

    def opt_in_runner_backward_logprob():
        with _torch_deterministic(False):
            runner = opt_in.prepare_lm_head_loss(X, W, labels, entry="logprob", **kw)
            runner.forward()
        return runner.backward()

    return {
        "forward_loss": lambda: cake_backend.forward_loss(
            X, W, labels, loss_div=1.0, **kw
        ),
        "opt_in.forward_loss": lambda: opt_in.forward_loss(
            X, W, labels, loss_div=1.0, **kw
        ),
        "backward_logprob": lambda: cake_backend.backward_logprob(
            X,
            W,
            labels,
            torch.zeros(labels.shape[0]),
            torch.zeros(labels.shape[0]),
            **kw,
        ),
        "backward_loss": deferred_backward,
        "LmHeadLossRunner.forward": runner_forward,
        "LmHeadLossRunner.backward": runner_backward_logprob,
        "opt_in.LmHeadLossRunner.forward": opt_in_runner_forward,
        "opt_in.LmHeadLossRunner.backward": opt_in_runner_backward_logprob,
        "chunked_lm_head_loss": lambda: cake_backend.chunked_lm_head_loss(
            X.detach().requires_grad_(True), W, labels, loss_div=1.0, **kw
        ),
    }


def test_torch_deterministic_mode_raises_the_alert():
    """Under ``torch.use_deterministic_algorithms(True)`` every gradient path raises PyTorch's ``RuntimeError`` (its
    wording, naming the deterministic package); under ``warn_only=True`` it warns and computes."""
    inp = _host_inputs(8, device=CPU)
    for name, call in _alerting_calls(inp).items():
        with (
            _torch_deterministic(True),
            pytest.raises(
                RuntimeError, match="does not have a deterministic implementation"
            ) as info,
        ):
            call()
        assert "cake_lm_head_det_loss" in str(info.value), name
        with (
            _torch_deterministic(True, warn_only=True),
            pytest.warns(
                UserWarning, match="does not have a deterministic implementation"
            ),
        ):
            call()


def test_paths_without_gradient_gemms_do_not_alert():
    """The log-probability forward and a loss forward without trainable inputs launch no gradient GEMM: no alert."""
    inp = _host_inputs(8, device=CPU)
    kw = dict(chunk_size=4, backend="reference", **ND)
    with _torch_deterministic(True), warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*deterministic implementation.*")
        cake_backend.forward_logprob(inp.X, inp.W, inp.labels, **kw)
        cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, loss_div=1.0, need_dx=False, need_dw=False, **kw
        )


@pytest.mark.parametrize("T, C", [(1, 16), (37, 16), (200, 64)])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_reference_matches_the_deterministic_reference_and_the_oracle(objective, T, C):
    """The reference backend (PyTorch operators at the kernels' rounding boundaries, fixed order) is the
    deterministic package's bit for bit and passes the deterministic suite's gates."""
    inp = _host_inputs(T, objective)
    nd = _run(inp, C)
    det = _run(inp, C, package="det")
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(nd[key], det[key]), key
    _check_against_references(nd, inp)


def test_reference_logprob_entry_matches_the_deterministic_reference():
    inp = _host_inputs(37)
    nd = _run(inp, 16, entry="logprob")
    det = _run(inp, 16, entry="logprob", package="det")
    for key in ("logp", "dX", "dW"):
        assert _same_bits(nd[key], det[key]), key
    _check_against_references(nd, inp, entry="logprob")


# The B200 (sm_100a, 148 SMs, 74 co-resident dX clusters) geometry of the GLM-class head the slice rules were measured
# on (H 6144, V 154880).
GLM_H, GLM_V, B200_SMS, B200_DX_RESIDENT = 6144, 154880, 148, 74


def _plan(
    rows,
    *,
    backend=None,
    policy=None,
    chunk=4096,
    hidden=GLM_H,
    vocab=GLM_V,
    entry="loss",
    need_dx=True,
    tuning=None,
    dx_max_slices=None,
    red_min_rows=None,
):
    """The ``sm_100a`` B200 plan of ``rows`` chunk-loop rows of the cross-entropy loss (``entry="logprob"``: the
    log-probability entry) built by ``backend``'s ``make_plan`` (default this package's ``cake_backend``; the
    deterministic package takes no policy and no reduction floor)."""
    backend = cake_backend if backend is None else backend
    problem = backend.Problem(
        num_rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        ld_x=hidden,
        x_copy=False,
        objective="ce",
        loss_div=1.0,
        grad_weight_dtype=torch.bfloat16,
        entry=entry,
    )
    kw = dict(
        need_dx=need_dx,
        need_dw=True,
        geometry=backend.Geometry.from_record(
            {"geometry": {"hidden": hidden, "vocab": vocab}}
        ),
        dx_max_slices=(
            dx_max_slices
            if dx_max_slices is not None
            else (cake_backend.DX_RED_SLICES_MAX if backend is not det_backend else 4)
        ),
        num_sms=B200_SMS,
        dx_resident=B200_DX_RESIDENT,
        fuse_dw_cast=True,
        arch="sm_100a",
        gemm_tuning=tuning,
        dx_finalize=True,
    )
    if backend is not det_backend:
        kw["dx_slice_policy"] = policy
        kw["dx_red_min_rows"] = red_min_rows
    return backend.make_plan(problem, **kw)


def test_dx_launches_come_from_the_dx_helper():
    """Every dX launch of a plan is planned by ``_dx_keys``: one launch per chunk -- the K-sliced reduction
    (``gemm_dx_red*``) for a chunk of more than one slice, which adds every slice itself (no slab, no slab sum, no host
    reduction chain) -- and the call's one ``dX_acc`` fill heads the launch sequence."""
    plan = _plan(16231, hidden=7168, vocab=129280, tuning={"dx": {"k_slices": 2}})
    dx = [key for i in range(plan.num_chunks) for key in cake_backend._dx_keys(plan, i)]
    assert len(dx) == plan.num_chunks
    assert all(stage.startswith(cake_backend.DX_RED_STAGE) for stage, _ in dx), dx
    keys = cake_backend.forward_keys(plan)
    assert [key for key in keys if key[0].startswith("gemm_dx")] == dx
    assert keys[0] == ("dx_fill", "dx") and keys.count(("dx_fill", "dx")) == 1
    assert plan.dx_fill_rows == (0, 16231)


def _det_sliced(stage: str) -> bool:
    """A deterministic K-sliced dX stage name (``gemm_dx_s`` and its forms; ``gemm_dx_st3`` is the plain store)."""
    return stage == "gemm_dx_s" or stage.startswith("gemm_dx_s_")


def test_no_slab_path_remains():
    """The registered stage names carry the reduction forms and no deterministic sliced form, the host has no slab
    step, and no workspace, memory report or workspace size counts a slab."""
    stages = set(cake_jit.STAGES)
    assert "slab_sum" not in stages
    assert not any(_det_sliced(s) for s in stages), stages
    red = {
        "gemm_dx_red",
        "gemm_dx_red_tn256",
        "gemm_dx_red_st3",
        "gemm_dx_red_split",
        "gemm_dx_red_st3_split",
        "gemm_dx_red_swab",
    }
    assert red <= stages and red <= set(cake_jit.GEMM_STAGES)
    assert set(cake_backend.HOST_STAGES) == {"dx_fill", "gather_rows"}
    assert "dx_fill" in opt_in.HOST_STAGES and "dx_reduce" not in opt_in.HOST_STAGES
    for name in ("DX_REDUCE_STAGES", "DX_SLICED_STAGE", "_dx_reduce"):
        assert not hasattr(cake_backend, name), name
    assert "slab_sum" not in cake_backend.STAGE_TENSORS
    for compact in (False, True):
        layout = cake_backend.workspace_layout(
            16231, GLM_V, 4096, hidden=GLM_H, compact=compact
        )
        assert not any("dx" in name for name in layout), layout
    report = cake_backend.memory_report(16231, GLM_H, GLM_V, 4096, valid_rows=15419)
    assert not any("dx_ws" in name or "slab" in name for name in report["temporary"])
    assert report["peak_temporary_bytes"] == report["temporary_bytes"]
    sizes = {
        need_dx: cake_backend.lm_head_loss_workspace_size(
            16231, GLM_V, 4096, backend="reference", hidden=GLM_H, need_dx=need_dx
        )
        for need_dx in (True, False)
    }
    assert sizes[True] == sizes[False]
    plan = _plan(16231)
    assert not any(stage in ("slab_sum", "dx_reduce") for stage in plan.stages)


# The model picks of the B200 measurement (sm_100a, H 6144, V 154880, cross-entropy loss): per chunk-loop row count
# and chunk size, the ``"measured"`` policy's slice count of each chunk.  Only the 4096-row chunks of a call of three or
# more chunks differ from the deterministic package's rule (3 instead of 4).
MEASURED_PICKS = [
    (16231, 4096, (3, 3, 3, 4)),  # 3943-row tail
    (15419, 4096, (3, 3, 3, 4)),  # 3131-row tail
    (8192, 4096, (4, 4)),  # 4096-row chunks of a two-chunk call
    (7711, 4096, (4, 1)),  # 3615-row tail: the plain store
    (2168, 4096, (1,)),
    (1946, 4096, (3,)),
    (1083, 4096, (3,)),
    (339, 4096, (3,)),
    (1, 4096, (3,)),
    (15419, 1024, (3,) * 16),  # 1024-row chunks, 59-row tail
]


@pytest.mark.parametrize("rows, chunk, picks", MEASURED_PICKS, ids=str)
def test_slice_policies(rows, chunk, picks):
    """The dX slice rule: ``"det"`` (the default) gives the deterministic package's slice count per chunk, ``"measured"``
    the B200 model picks (among the measured slice counts); an explicit ``k_slices`` knob wins over either."""
    assert cake_backend.dx_slice_policy() == "det"
    assert cake_backend.DX_SLICE_POLICIES == ("det", "measured")
    det = _plan(rows, backend=det_backend, chunk=chunk)
    for policy in (None, "det"):
        assert _plan(rows, chunk=chunk, policy=policy).dx_slices == det.dx_slices
    measured = _plan(rows, chunk=chunk, policy="measured")
    assert measured.dx_slices == picks
    assert set(picks) <= set(cake_backend.DX_MEASURED_SLICE_CANDIDATES)
    with _slice_policy("measured"):
        assert cake_backend.dx_slice_policy() == "measured"
        assert _plan(rows, chunk=chunk).dx_slices == picks
        assert _plan(rows, chunk=chunk, policy="det").dx_slices == det.dx_slices
    assert cake_backend.dx_slice_policy() == "det"
    for plan in (_plan(rows, chunk=chunk), measured):
        for i in range(plan.num_chunks):
            stage = plan.dx_stage_of(i)
            assert stage.startswith(cake_backend.DX_RED_STAGE) == (
                plan.dx_slices_of(i) > 1
            ), (i, stage)
    pinned = _plan(rows, chunk=chunk, policy="measured", tuning={"dx": {"k_slices": 8}})
    assert set(pinned.dx_slices) == {8}


def test_slice_policy_switch_and_fit():
    """``set_dx_slice_policy`` returns the previous policy and refuses an unknown one; ``dx_slices_fit`` keeps every
    slice inside the K steps (the reduction kernel's ``k_slice_iters * (k_slices - 1) < k_iters``); a pinned slice
    count that leaves a slice empty, or more slices than the program serves, is refused."""
    assert cake_backend.set_dx_slice_policy("measured") == "det"
    try:
        with pytest.raises(ValueError, match="policy"):
            cake_backend.set_dx_slice_policy("fastest")
        assert cake_backend.dx_slice_policy() == "measured"
    finally:
        assert cake_backend.set_dx_slice_policy("det") == "measured"
    for k_iters in (1, 2, 3, 5, 7, 2420, 2020):
        for ks in range(1, 9):
            fit = cake_backend.dx_slices_fit(ks, k_iters)
            per = -(-k_iters // fit)
            assert 1 <= fit <= ks and per * (fit - 1) < k_iters, (ks, k_iters, fit)
    with pytest.raises(ValueError, match="k_slices"):
        _plan(4096, tuning={"dx": {"k_slices": 9}})
    with pytest.raises(ValueError, match="k_slices"):
        _plan(
            4096, vocab=256 * 7, tuning={"dx": {"k_slices": 8}}
        )  # 28 K steps: 7 slices of 4 reach them, 8 do not
    assert _plan(4096, tuning={"dx": {"k_slices": None}}).dx_slices == (1,)


def _holds_a_k_step(slices: int, k_iters: int) -> bool:
    """Every slice of ``ceil(k_iters / slices)`` K steps holds a K step: the last one runs
    ``k_iters - (slices - 1) * ceil(k_iters / slices)`` steps (at zero or fewer it would add an unaccumulated tile)."""
    return slices == 1 or (slices - 1) * -(-k_iters // slices) < k_iters


def _slices_per_chunk(plan) -> tuple[int, ...]:
    """The dX slice count of every chunk of ``plan`` (an empty ``dx_slices``: one slice everywhere)."""
    return tuple(plan.dx_slices) or (1,) * plan.num_chunks


def test_det_policy_runs_the_deterministic_count_fitted_at_v256():
    """At ``V = 256`` (four K steps) this package's ``"det"`` policy runs, per chunk, the deterministic package's slice
    count fitted to the K steps (:func:`dx_slices_fit`: the largest count at most it whose slices all hold a K step),
    so no slice is empty -- whether that package plans the deterministic rule's raw count or fits it the same way.  The
    rule's 3 slices (``T`` = 1000 and the 1-row tail chunk of ``T`` = 4097, ``H`` 6144 on 148 SMs) run as 2."""
    for rows in (*range(1, 8193, 37), 1000, 4096, 4097):
        nd = _slices_per_chunk(_plan(rows, vocab=256, hidden=6144))
        det_plan = _plan(rows, backend=det_backend, vocab=256, hidden=6144)
        det = _slices_per_chunk(det_plan)
        fitted = tuple(cake_backend.dx_slices_fit(s, 4) for s in det)
        assert nd == fitted, (rows, nd, det)
        assert all(_holds_a_k_step(s, 4) for s in nd), (rows, nd)
    geometry = cake_backend.Geometry.from_record(None)
    rule = cake_backend.recommended_k_slices(
        1000, 6144, B200_SMS, cake_backend.DX_K_SLICES_MAX, geometry, B200_DX_RESIDENT
    )
    assert rule == 3 and not _holds_a_k_step(rule, 4)
    assert _plan(1000, vocab=256, hidden=6144).dx_slices == (2,)
    assert _plan(4097, vocab=256, hidden=6144).dx_slices == (4, 2)


def test_dx_slices_fit_every_k_step_count():
    """``dx_slices_fit`` over every K step count up to 512 and every requested count the reduction form serves: the
    fitted count is the largest count at most the request whose ceil-sized slices all hold a K step, and equals the
    request exactly when the request holds one in every slice; ``gemm_scalars`` takes exactly the holding counts of the
    K-sliced reduction form (its last slice then runs 1 .. ``k_slice_iters`` steps) and rejects the others."""
    fit = cake_backend.dx_slices_fit
    g = cake_backend.Geometry.from_record(None)
    empty = []
    for k in range(1, 513):
        for ks in range(1, cake_backend.DX_RED_SLICES_MAX + 1):
            f = fit(ks, k)
            assert 1 <= f <= ks and _holds_a_k_step(f, k), (k, ks, f)
            best = max(s for s in range(1, ks + 1) if _holds_a_k_step(s, k))
            assert f == best, (k, ks, f)
            assert (f == ks) == _holds_a_k_step(ks, k), (k, ks, f)
            if not _holds_a_k_step(ks, k):
                empty.append((k, ks))
            if ks == 1:
                continue
            kw = dict(hidden=6144, vocab=k * g.k_block, rows_c=4096, k_slices=ks)
            if _holds_a_k_step(ks, k):
                sc = cake_backend.gemm_scalars(cake_backend.DX_RED_STAGE, g, **kw)
                last = sc["k_iters"] - (ks - 1) * sc["k_slice_iters"]
                assert sc["k_iters"] == k and 1 <= last <= sc["k_slice_iters"], (k, ks)
            else:
                with pytest.raises(ValueError, match="K step"):
                    cake_backend.gemm_scalars(cake_backend.DX_RED_STAGE, g, **kw)
    # 3 slices over 4 K steps (2, 2, 0) and 4 slices over 5, 6 or 9 K steps
    assert {(4, 3), (5, 4), (6, 4), (9, 4)} <= set(empty)
    cases = ((4, 3), (5, 4), (6, 4), (9, 4), (8, 3), (2420, 4), (8, 6), (12, 8))
    assert [fit(ks, k) for k, ks in cases] == [2, 3, 3, 3, 3, 4, 4, 6]
    # V is a multiple of vocab_multiple: of the deterministic rule's counts (the "det" policy) only 3 slices over
    # V = 256 leave a slice without a K step
    step = g.vocab_multiple // g.k_block
    reachable = {
        (k, ks)
        for k, ks in empty
        if k % step == 0 and ks <= cake_backend.DX_K_SLICES_MAX
    }
    assert reachable == {(4, 3)}


def test_dx_slice_rule_and_pin_fit_the_k_steps():
    """Both slice policies fit their count to V's K steps on every row count, H and SM count (132 / 148 / 152 / 160),
    with the default geometry and every architecture record's geometry, co-resident cluster count and architecture:
    the planned count is :func:`dx_slices_fit` of the policy's own count (``"det"``: the deterministic rule
    :func:`recommended_k_slices`; ``"measured"``: its count at a V whose K steps hold every candidate) -- the largest
    count at most it whose slices all hold a K step -- so no slice is empty.  Under ``"det"`` only V = 256 changes a
    count (3 runs as 2).  Both planners plan those counts per chunk (``T`` = 1000 at V = 256: 2; at V = 512: 3; the
    4097-row plan's 1-row tail chunk at V = 256: 2), never plan an empty slice, and reject an explicit ``k_slices``
    pin that leaves a slice without a K step."""
    # 2420 K steps: every count up to DX_RED_SLICES_MAX holds a K step in every slice
    wide_v = 154880
    det_max = min(cake_backend.DX_RED_SLICES_MAX, cake_backend.DX_K_SLICES_MAX)
    records = [None, *cake_jit.MODULES.values()]
    assert len(records) >= 3
    for record in records:
        geo = cake_backend.Geometry.from_record(record)
        arch = None if record is None else record["arch"]
        assert all(
            _holds_a_k_step(s, wide_v // geo.k_block)
            for s in range(1, cake_backend.DX_RED_SLICES_MAX + 1)
        )
        for sms in (132, 148, 152, 160):
            resident = cake_backend.cluster_resident(None, geo.dx_cluster_ctas, sms)
            for H in (256, 512, 1024, 2048, 4096, 6144, 7168, 8192):
                for rows in range(1, 8193, 127):
                    det_raw = cake_backend.recommended_k_slices(
                        rows, H, sms, det_max, geo, resident
                    )
                    raw = {"det": det_raw}
                    # the measured model on every other row count (it costs more per call)
                    if arch is not None and rows % 2 == 1:
                        raw["measured"] = cake_backend.dx_slices(
                            rows,
                            H,
                            wide_v,
                            sms,
                            geo,
                            arch,
                            max_slices=cake_backend.DX_RED_SLICES_MAX,
                            resident=resident,
                            policy="measured",
                        )
                    for policy, want in raw.items():
                        for V in (256, 512, 768, 1024, wide_v):
                            k = V // geo.k_block
                            got = cake_backend.dx_slices(
                                rows,
                                H,
                                V,
                                sms,
                                geo,
                                arch,
                                max_slices=cake_backend.DX_RED_SLICES_MAX,
                                resident=resident,
                                policy=policy,
                            )
                            want_k = cake_backend.dx_slices_fit(want, k)
                            where = (arch, sms, H, rows, policy, V, got, want)
                            assert got == want_k and _holds_a_k_step(got, k), where
                            if policy == "det":
                                assert got == want or (V, want, got) == (256, 3, 2)
    for backend in (cake_backend, opt_in):
        assert _plan(1000, backend=backend, vocab=256).dx_slices == (2,)
        assert _plan(1000, backend=backend, vocab=512).dx_slices == (3,)
        assert _plan(4097, backend=backend, vocab=256).dx_slices == (4, 2)
        for policy in ("det", "measured"):
            for rows in range(1, 8193, 97):
                for V in (256, 512, 768):
                    plan = _plan(rows, backend=backend, vocab=V, policy=policy)
                    slices = _slices_per_chunk(plan)
                    where = (backend.__name__, policy, rows, V, slices)
                    assert all(_holds_a_k_step(s, V // 64) for s in slices), where
        for k in (1, 2, 4):
            tuning = {"dx": {"k_slices": k}}
            pinned = _plan(1000, backend=backend, vocab=256, tuning=tuning)
            assert pinned.dx_slices == (k,)
        tuning = {"dx": {"k_slices": 3}}
        with pytest.raises(ValueError, match="K steps"):
            _plan(1000, backend=backend, vocab=256, tuning=tuning)
        assert _plan(1000, backend=backend, vocab=512, tuning=tuning).dx_slices == (3,)


def test_red_min_rows_keeps_small_chunks_on_the_plain_store():
    """``set_dx_red_min_rows``: a chunk of fewer rows than the floor keeps one slice (the plain store, no fill); the
    rule's slices elsewhere; an explicit ``k_slices`` knob wins; the floor is part of the slice rule's settings the
    binding keys and the plans take (``dx_slice_rule``)."""
    assert cake_backend.dx_red_min_rows() == 0
    assert cake_backend.dx_slice_rule() == ("det", 0)
    rule = _plan(16231).dx_slices
    assert rule == (4, 4, 4, 4)
    assert _plan(16231, red_min_rows=4000).dx_slices == (4, 4, 4, 1)
    assert _plan(16231, red_min_rows=4000).dx_fill_rows == (0, 12288)
    assert _plan(339, red_min_rows=340).dx_slices == (1,)
    assert _plan(339, red_min_rows=340).dx_fill_rows is None
    pinned = _plan(339, red_min_rows=340, tuning={"dx": {"k_slices": 4}})
    assert pinned.dx_slices == (4,)
    previous = cake_backend.set_dx_red_min_rows(4000)
    try:
        assert previous == 0 and cake_backend.dx_slice_rule() == ("det", 4000)
        assert _plan(16231).dx_slices == (4, 4, 4, 1)
        for bad in (-1, 1.5, True):
            with pytest.raises(ValueError, match="min rows"):
                cake_backend.set_dx_red_min_rows(bad)
    finally:
        assert cake_backend.set_dx_red_min_rows(0) == 4000


@pytest.mark.parametrize("backend", ["reference", "cake"])
def test_entries_read_the_slice_rule_once_and_key_and_plan_with_it(
    monkeypatch, request, backend
):
    """Each entry call reads the dX slice rule's settings once -- through ``dx_slice_rule``, never through
    ``dx_slice_policy`` / ``dx_red_min_rows`` on the side -- and gives the same pair to its binding key and its plan: a
    setting that changes between two reads (another thread's switch) can never file a binding under settings it was
    not planned with.  On the cake backend the remembered binding's key carries the pair the plan was made with."""
    if backend == "cake":
        inp = _device_inputs(339, "ce", W=request.getfixturevalue("glm_weight"))
        chunk = 128
    else:
        inp = _host_inputs(40, "ce")
        chunk = 16
    reads, prepared, side_reads = [], [], []
    rules = iter([("det", 0), ("measured", 7), ("det", 3), ("measured", 0)] * 4)

    def changing_rule():
        reads.append(next(rules))
        return reads[-1]

    original = cake_backend.prepare_lm_head_loss

    def recording(*a, **k):
        prepared.append(k.get("dx_rule"))
        return original(*a, **k)

    monkeypatch.setattr(cake_backend, "dx_slice_rule", changing_rule)
    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss", recording)
    for name in ("dx_slice_policy", "dx_red_min_rows"):
        monkeypatch.setattr(
            cake_backend, name, lambda name=name: side_reads.append(name) or 0
        )
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    kw = dict(chunk_size=chunk, backend=backend, **ND)
    cake_backend.forward_loss(inp.X, inp.W, inp.labels, loss_div=1.0, **kw)
    assert len(reads) == 1 and prepared == [reads[0]], (reads, prepared)
    if backend == "cake":
        assert next(reversed(cache._bindings))[-1] == reads[0]
    fwd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, **kw)
    reads.clear()
    prepared.clear()
    cake_backend.backward_logprob(
        inp.X, inp.W, inp.labels, fwd.lse, inp.dlogp, row_index=fwd.row_index, **kw
    )
    assert len(reads) == 1 and prepared == [reads[0]], (reads, prepared)
    if backend == "cake":
        newest = next(reversed(cache._bindings))
        assert newest[:2] == ("bwd", "logprob") and newest[-1] == reads[0]
        cache.clear()
    assert side_reads == [], side_reads
    key = cake_backend.forward_binding_key(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=1.0,
        infer_logp=None,
        loss_weights=None,
        chunk_size=16,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.bfloat16,
        entry="loss",
        dx_rule=("measured", 7),
    )
    assert key[-1] == ("measured", 7)
    assert (
        key[:-1]
        == cake_backend.forward_binding_key(
            inp.X,
            inp.W,
            inp.labels,
            objective="ce",
            loss_div=1.0,
            infer_logp=None,
            loss_weights=None,
            chunk_size=16,
            need_dx=True,
            need_dw=True,
            grad_weight_dtype=torch.bfloat16,
            entry="loss",
            dx_rule=("det", 0),
        )[:-1]
    )


def test_binding_keys_cover_the_options_of_the_plan():
    """Every option that shapes a binding's plan or argument plans is in its key (the D:3203 analog): the entry, the
    tensors' addresses / shapes / strides / dtypes, the objective and its scalars, the switches, the explicit GEMM knobs
    and the slice rule's settings; two calls that differ in any of them never share a binding.  The log-probability
    backward's ``lse`` / ``dlogp`` enter by layout only: the binding slots them per call and no plan reads their
    addresses, so the same layout at another address shares the binding and another layout does not."""
    inp = _host_inputs(40, "policy")
    base = dict(
        objective="policy",
        loss_div=None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=16,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.bfloat16,
        entry="loss",
        dx_rule=("det", 0),
    )
    for module in (cake_backend, opt_in):
        key = module.forward_binding_key(inp.X, inp.W, inp.labels, **base)
        changes = dict(
            objective="ce",
            loss_div=2.0,
            infer_logp=inp.infer_logp.clone(),
            loss_weights=inp.loss_weights.clone(),
            chunk_size=32,
            need_dx=False,
            need_dw=False,
            grad_weight_dtype=torch.float32,
            entry="logprob",
            valid_rows=39,
            fuse_dw_cast=True,
            gemm_tuning={"dx": {"k_slices": 2}},
            dx_finalize=True,
            hidden_count=True,
            logprob_lbl=True,
            dx_rule=("measured", 0),
        )
        for name, value in changes.items():
            other = module.forward_binding_key(
                inp.X, inp.W, inp.labels, **dict(base, **{name: value})
            )
            assert other != key, (module.__name__, name)
        assert (
            module.forward_binding_key(
                inp.X, inp.W, inp.labels, **dict(base, dx_rule=("det", 4000))
            )
            != key
        )
        for name, tensors in (
            ("X", (inp.X.clone(), inp.W, inp.labels)),
            ("W", (inp.X, inp.W.clone(), inp.labels)),
            ("labels", (inp.X, inp.W, inp.labels.clone())),
        ):
            assert module.forward_binding_key(*tensors, **base) != key, name
        bwd = dict(chunk_size=16, need_dx=True, need_dw=True, dx_rule=("det", 0))
        lse, dlogp = torch.zeros(40), inp.dlogp.clone()
        k0 = module.logprob_backward_binding_key(
            inp.X, inp.W, inp.labels, lse, dlogp, **bwd
        )
        assert k0[-1] == ("det", 0)
        # the same layouts at other addresses: the same binding
        assert (
            module.logprob_backward_binding_key(
                inp.X, inp.W, inp.labels, lse.clone(), dlogp.clone(), **bwd
            )
            == k0
        )
        for name, args in (
            ("X", (inp.X.clone(), inp.W, inp.labels, lse, dlogp)),
            ("W", (inp.X, inp.W.clone(), inp.labels, lse, dlogp)),
            ("labels", (inp.X, inp.W, inp.labels.clone(), lse, dlogp)),
            ("lse shape", (inp.X, inp.W, inp.labels, torch.zeros(39), dlogp)),
            ("lse dtype", (inp.X, inp.W, inp.labels, lse.double(), dlogp)),
            ("dlogp stride", (inp.X, inp.W, inp.labels, lse, torch.zeros(80)[::2])),
        ):
            assert module.logprob_backward_binding_key(*args, **bwd) != k0, name
        bwd_changes = dict(
            chunk_size=32,
            need_dx=False,
            need_dw=False,
            valid_rows=39,
            fuse_dw_cast=True,
            gemm_tuning={"dx": {"k_slices": 2}},
            dx_finalize=True,
            hidden_count=True,
            logprob_dz=True,
            dx_rule=("det", 64),
        )
        for name, value in bwd_changes.items():
            other = module.logprob_backward_binding_key(
                inp.X, inp.W, inp.labels, lse, dlogp, **dict(bwd, **{name: value})
            )
            assert other != k0, (module.__name__, name)


def test_package_reads_no_environment_variable():
    """The package takes every switch as a keyword or a programmatic setter; no module of it reads the
    environment (the public entries own the ``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` fallbacks)."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("environ", "getenv", "putenv"), (
                    path.name,
                    node.lineno,
                )
            if isinstance(node, ast.Name):
                assert node.id not in ("environ", "getenv"), (path.name, node.lineno)


def test_fill_placement_and_zero_weight_floor_switches():
    """The programmatic switches of the ``dx_fill`` placement and of the zero-weight split's smallest call: defaults
    (the caller's stream; calls of at least 4096 rows -- this suite runs at 0, every compacted policy call),
    the previous value returned, bad values refused, and the floor applied to the split decision (no binding depends
    on either)."""
    assert cake_backend.ZERO_WEIGHT_MIN_ROWS_DEFAULT == 4096
    assert cake_backend.dx_fill_side_stream() is False
    assert cake_backend.set_dx_fill_side_stream(True) is False
    try:
        assert cake_backend.dx_fill_side_stream() is True
        with pytest.raises(ValueError, match="bool"):
            cake_backend.set_dx_fill_side_stream(1)
    finally:
        assert cake_backend.set_dx_fill_side_stream(False) is True
    inp = _host_inputs(200, "policy")
    problem = cake_backend.validate_lm_head_inputs(
        inp.X,
        inp.W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        deterministic=False,
    )
    assert cake_backend.zero_weight_min_rows() == 0
    assert cake_backend.zero_weight_split(problem, True)
    assert not cake_backend.zero_weight_split(problem, True, min_rows=201)
    assert cake_backend.zero_weight_split(problem, True, min_rows=200)
    previous = cake_backend.set_zero_weight_min_rows(256)
    try:
        assert previous == 0 and not cake_backend.zero_weight_split(problem, True)
        for bad in (-1, 2.5, False):
            with pytest.raises(ValueError, match="min rows"):
                cake_backend.set_zero_weight_min_rows(bad)
        zero = _zero_weight_set(inp, ("random", 0.3))
        weighted = _with_zero_weights(inp, zero)
        kw = dict(
            objective="policy",
            infer_logp=weighted.infer_logp,
            loss_weights=weighted.loss_weights,
            chunk_size=64,
            backend="reference",
            **ND,
        )
        below = cake_backend.forward_loss(inp.X, inp.W, weighted.labels, **kw)
        assert "zero_weight_rows" not in below.memory
        assert below.row_index.numel() == int((inp.labels >= 0).sum())
    finally:
        assert cake_backend.set_zero_weight_min_rows(0) == 256
    split = cake_backend.forward_loss(inp.X, inp.W, weighted.labels, **kw)
    assert split.memory["zero_weight_rows"] == int(zero.sum())
    torch.testing.assert_close(split.logp, below.logp, atol=1e-5, rtol=0)


def test_dx_fill_covers_the_sliced_chunks_once():
    """The ``dX_acc`` fill: one host step at the head of the launch sequence that adds into the accumulator (the loss
    forward, the log-probability backward's recompute), over the rows of the chunks of more than one slice; none when
    every chunk stores (one slice) or without ``dX``.  A one-slice chunk keeps the plain-store form."""
    mixed = _plan(7711)
    assert mixed.dx_slices == (4, 1)
    assert mixed.dx_fill_rows == (0, 4096)
    assert mixed.dx_stage_of(0).startswith(cake_backend.DX_RED_STAGE)
    assert not mixed.dx_stage_of(1).startswith(cake_backend.DX_RED_STAGE)
    keys = cake_backend.forward_keys(mixed)
    assert keys[0] == ("dx_fill", "dx") and keys.count(("dx_fill", "dx")) == 1
    plain = _plan(2168)
    assert plain.dx_slices == (1,) and plain.dx_fill_rows is None
    assert ("dx_fill", "dx") not in cake_backend.forward_keys(plain)
    assert _plan(7711, need_dx=False).dx_fill_rows is None
    assert _plan(16231).dx_fill_rows == (0, 16231)
    logprob = _plan(7711, entry="logprob")
    assert ("dx_fill", "dx") not in cake_backend.forward_keys(logprob)
    assert cake_backend.recompute_keys(logprob)[0] == ("dx_fill", "dx")
    opt_plan = opt_in.make_plan(
        mixed.problem,
        need_dx=True,
        need_dw=True,
        geometry=mixed.geometry,
        dx_max_slices=cake_backend.DX_RED_SLICES_MAX,
        num_sms=B200_SMS,
        dx_resident=B200_DX_RESIDENT,
        fuse_dw_cast=False,
        arch="sm_100a",
        dx_finalize=True,
        dw_rows=None,
        keep_row_grad=True,
    )
    assert opt_plan.dx_slices == mixed.dx_slices
    assert opt_in.forward_keys(opt_plan)[0] == ("dx_fill", "dx")


def test_zero_weight_rows_report_the_combined_peak():
    """A loss call with zero-weight rows reports ``peak_temporary_bytes`` as the larger of the chunk loop's
    temporaries and the logp-only pass's phase -- its own temporaries plus the compaction outputs it coexists with,
    plus chunk 0's buffers when it ran while the hidden valid-row count held them."""
    temporary = dict(logits=1000, stats=100, x_c=200, row_index=8, row_valid=4, d=16)
    memory = dict(temporary=temporary, temporary_bytes=1328, peak_temporary_bytes=1328)
    zero_index = torch.tensor([1, 3])

    def report(zero_temp, **kw):
        result = SimpleNamespace(memory=dict(memory), logp=torch.zeros(5))
        logp = torch.tensor([-1.0, -2.0])
        out = cake_backend._with_zero_weight_rows(
            result, zero_index, (logp, {"temporary_bytes": zero_temp}), **kw
        )
        assert out.logp.tolist() == [0.0, -1.0, 0.0, -2.0, 0.0]
        return out.memory

    after = report(1250)
    assert after["zero_weight_logp_temporary_bytes"] == 1250
    assert after["zero_weight_logp_phase_bytes"] == 1250 + 12
    assert after["peak_temporary_bytes"] == 1328 and after["temporary_bytes"] == 1328
    assert report(2000)["peak_temporary_bytes"] == 2012
    first = report(1250, chunk0_alive=True)
    assert first["zero_weight_logp_phase_bytes"] == 1250 + 12 + 1300
    assert first["peak_temporary_bytes"] == 2562
    skipped = cake_backend._with_zero_weight_rows(
        SimpleNamespace(memory=dict(memory), logp=torch.zeros(5)), zero_index, None
    ).memory
    assert skipped["zero_weight_logp_temporary_bytes"] == 0
    assert skipped["peak_temporary_bytes"] == 1328
    assert "zero_weight_logp_phase_bytes" not in skipped


def _assert_bindings_own_no_buffer(modules) -> int:
    """No stage of ``modules`` declares kernel scratch, a descriptor workspace or a descriptor-preparation entry, and
    the record-to-binding path gives no owned workspace region (:data:`cake_backend._OWNED_VALUES`).  Returns the
    number of stages checked."""
    stages = 0
    for name, record in modules.items():
        for stage in record["stages"]:
            physical = record[stage]
            assert physical["workspace_bytes"] == 0, (name, stage)
            assert physical["tma_workspace_bytes"] == 0, (name, stage)
            assert not physical.get("tma_prepare_entry"), (name, stage)
            stages += 1
        scratch = cake_backend._record_scratch_bytes(record, record["stages"])
        tma = cake_backend._record_tma_bytes(record, record["stages"])
        layout = cake_backend.workspace_layout(
            16231,
            GLM_V,
            4096,
            hidden=GLM_H,
            compact=True,
            scratch_bytes=scratch,
            tma_workspace_bytes=tma,
        )
        assert not set(layout) & set(cake_backend._OWNED_VALUES), (name, sorted(layout))
    return stages


def test_bindings_own_no_buffer():
    """No registered stage declares kernel scratch or a descriptor workspace (``workspace_bytes`` /
    ``tma_workspace_bytes`` 0, no ``tma_prepare_entry``), so a remembered binding owns no buffer: calls with the same
    inputs in flight at once on two streams share one binding (its key holds no stream) and write only their own
    per-call allocations.  A stage that declares one needs the stream in the binding key first.  Controls: a copy of
    a registered record whose one stage declares 64 bytes of scratch, 128 bytes of descriptor workspace or a
    preparation entry fails the same check (the first two also give the binding an owned workspace region)."""
    assert _assert_bindings_own_no_buffer(cake_jit.MODULES) > 0
    name, record = next(iter(cake_jit.MODULES.items()))
    stage = record["stages"][0]
    for field, value in (
        ("workspace_bytes", 64),
        ("tma_workspace_bytes", 128),
        ("tma_prepare_entry", "prepare"),
    ):
        declared = dict(record)
        declared[stage] = dict(record[stage], **{field: value})
        with pytest.raises(AssertionError):
            _assert_bindings_own_no_buffer({name: declared})
    declared = dict(record)
    declared[stage] = dict(record[stage], workspace_bytes=64)
    scratch = cake_backend._record_scratch_bytes(declared, declared["stages"])
    layout = cake_backend.workspace_layout(16231, GLM_V, 4096, scratch_bytes=scratch)
    assert layout["workspace"][1] == 64 and "workspace" in cake_backend._OWNED_VALUES


def test_hidden_count_state_is_per_thread():
    """The hidden valid-row count's pinned cells and events, its count constant and the dW side stream with its
    fork event are the calling thread's: two threads never share one in flight."""
    if not torch.cuda.is_available():
        pytest.skip("pinned cells and streams need a CUDA device")
    here = (
        cake_backend._hidden_count_cell(0),
        cake_backend._hidden_count_pair_cell(0),
        cake_backend._dw_stream(0),
        cake_backend._fill_stream(0),
    )
    assert cake_backend._hidden_count_cell(0) is here[0]
    other = []

    def worker():
        other.append(
            (
                cake_backend._hidden_count_cell(0),
                cake_backend._hidden_count_pair_cell(0),
                cake_backend._dw_stream(0),
                cake_backend._fill_stream(0),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    for mine, theirs in zip(here, other[0], strict=True):
        assert mine[0] is not theirs[0] and mine[1] is not theirs[1]
        if isinstance(mine[0], torch.Tensor):
            assert mine[0].data_ptr() != theirs[0].data_ptr()


def test_hidden_count_arange_records_each_stream_once(monkeypatch):
    """The count constant is recorded once per stream it is used on, again on a new stream, and afresh after it grew
    (a fresh thread on its own stream: its own constant); ``record_stream`` runs once per recorded stream, and the
    grown constant keeps its values."""
    if not torch.cuda.is_available():
        pytest.skip("the count constant needs a CUDA device")
    records: list[int] = []
    record_stream = torch.Tensor.record_stream

    def counting(self, stream):
        records.append(stream.cuda_stream)
        return record_stream(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", counting)
    index = torch.cuda.current_device()
    got = {}
    errors: list[BaseException] = []

    def run():
        try:
            with (
                torch.cuda.device(index),
                torch.cuda.stream(torch.cuda.Stream(index)),
            ):
                main = torch.cuda.current_stream(index).cuda_stream
                recorded = cake_backend._thread_state("arange_streams")
                ar = cake_backend._hidden_count_arange(index, 8)
                cake_backend._hidden_count_arange(index, 8)
                got["one"] = (len(recorded[index]), list(records))
                side = torch.cuda.Stream(index)
                with torch.cuda.stream(side):
                    assert cake_backend._hidden_count_arange(index, 8) is ar
                    cake_backend._hidden_count_arange(index, 8)
                assert cake_backend._hidden_count_arange(index, 8) is ar
                got["two"] = (len(recorded[index]), list(records))
                grown = cake_backend._hidden_count_arange(index, 2 * ar.numel() + 1)
                got["grown"] = (grown is not ar, len(recorded[index]), list(records))
                got["values"] = torch.equal(
                    grown.cpu(),
                    torch.arange(1, grown.numel() + 1, dtype=torch.int32),
                )
                got["streams"] = (main, side.cuda_stream)
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the main thread
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "the thread did not finish"
    torch.cuda.synchronize()
    if errors:
        raise errors[0]
    main, side = got["streams"]
    assert got["one"] == (1, [main])
    assert got["two"] == (2, [main, side])
    assert got["grown"] == (True, 1, [main, side, main])
    assert got["values"]


@pytest.mark.parametrize("module", ["backend", "opt_in"])
def test_binding_cache_skips_what_another_threads_eviction_removed(module):
    """Threads share the binding cache without a lock: a lookup whose key another thread evicts between its ``get``
    and its recency update returns the binding (or eager launch) it found, and an eviction that another thread's
    eviction emptied between its size check and its ``popitem`` stops -- no ``KeyError`` (the races, run
    deterministically through the cache's ordered dicts)."""

    # another thread evicts the key right after this thread's get
    class EvictedAfterGet(OrderedDict):
        def get(self, key, default=None):
            value = super().get(key, default)
            self.pop(key, None)
            return value

    # another thread's eviction empties it before this thread's popitem
    class EmptiedBeforePop(OrderedDict):
        def popitem(self, last=True):
            self.clear()
            return super().popitem(last=last)

    be = {"backend": cake_backend, "opt_in": opt_in}[module]
    binding, launch, record = object(), object(), object()
    cache = be.BindingCache(capacity=1)
    cache._bindings = EvictedAfterGet({("k",): binding})
    assert cache.lookup(("k",)) is binding and cache.hits == 1
    assert cache.lookup(("k",)) is None and cache.misses == 1
    cache._eager = EvictedAfterGet({("f",): (record, launch)})
    assert cache.eager_lookup(("f",), record) is launch
    cache._bindings = EmptiedBeforePop({("a",): binding})
    assert cache.remember(("b",), binding) is binding and len(cache) == 0
    cache._bindings = EmptiedBeforePop({("a",): binding, ("b",): binding})
    assert cache.configure(capacity=1) is cache and len(cache) == 0
    cache.EAGER_CAPACITY = 1
    cache._eager = EmptiedBeforePop({("a",): (record, launch)})
    cache.eager_remember(("b",), record, launch)
    assert len(cache._eager) == 0


@pytest.mark.parametrize("module", ["backend", "opt_in"])
def test_binding_cache_owned_bytes_sums_a_copy(module):
    """``owned_bytes`` sums a copy of the bindings: another thread that empties the cache during the sum (run
    deterministically: the bindings' ``owned_bytes`` empties it) raises no ``RuntimeError``."""
    cache = {"backend": cake_backend, "opt_in": opt_in}[module].BindingCache()

    class Evicting:
        @property
        def owned_bytes(self):
            cache._bindings.clear()
            return 7

    cache._bindings = OrderedDict({("x",): Evicting(), ("y",): Evicting()})
    assert cache.owned_bytes == 14


@pytest.mark.parametrize("module", ["backend", "opt_in"])
def test_binding_cache_owned_bytes_while_another_thread_remembers(module):
    """``owned_bytes`` sums a copy of the bindings: another thread that remembers a binding while it sums (run
    deterministically from inside the read of a binding's size) raises no ``OrderedDict mutated during iteration``."""
    be = {"backend": cake_backend, "opt_in": opt_in}[module]
    cache = be.BindingCache(capacity=4)

    class Binding:
        def __init__(self, nbytes):
            self.nbytes = nbytes

        @property
        def owned_bytes(self):
            # another thread remembers a binding
            cache.remember(("other", self.nbytes), SimpleNamespace(owned_bytes=0))
            return self.nbytes

    cache.remember(("a",), Binding(1))
    cache.remember(("b",), Binding(2))
    assert cache.owned_bytes == 3
    assert len(cache) == 4


# ---------------------------------------------------------------------------
# Device (SM100 / SM103 with the package's rendered programs)
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, entry: str = "loss"):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip(
            "the package's programs are not rendered in this checkout (no build.json)"
        )
    if not cake_backend.generated_program_available(CUDA, entry=entry):
        pytest.skip(
            f"generated chunked LM-head program ({entry} entry) not registered for this device"
        )


@pytest.fixture(scope="module")
def glm_weight():
    _require_program(entry="loss")
    return make_inputs(1, seed=SEED, device=CUDA, ignore_frac=0.0).W


def _device_inputs(T, objective="ce", *, W, seed=SEED, ignore_frac=None):
    return make_inputs(
        T,
        objective=objective,
        seed=seed + T,
        device=CUDA,
        W=W,
        ignore_frac=(0.0 if T < 16 else 0.05) if ignore_frac is None else ignore_frac,
    )


def _poison_allocator(nbytes: int = 1 << 31) -> None:
    """Fill free caching-allocator memory with 0xFF bytes (a NaN pattern in BF16 and FP32), so a buffer the package
    forgets to write -- or a zero fill of ``dX_acc`` it skips -- cannot inherit zeros."""
    torch.cuda.synchronize()
    junk = torch.empty((int(nbytes),), dtype=torch.uint8, device=CUDA).fill_(0xFF)
    del junk
    torch.cuda.synchronize()


def _det_tuning(kw: dict) -> dict:
    """The deterministic package's arm of a call's ``gemm_tuning``: a pinned dX slice count above its four slices (the
    reduction serves up to eight) falls back to its own rule -- ``loss`` / ``logp`` / ``dW`` do not depend on it."""
    tuning = kw.get("gemm_tuning")
    k = (tuning or {}).get("dx", {}).get("k_slices")
    if k is None or int(k) <= 4:
        return kw
    dx = {key: v for key, v in tuning["dx"].items() if key != "k_slices"}
    rest = {op: v for op, v in tuning.items() if op != "dx"}
    return dict(kw, gemm_tuning=dict(rest, **({"dx": dx} if dx else {})) or None)


def _compare_with_det(nd, det, *, record=None):
    """``loss`` / ``logp`` and ``dW`` are the deterministic package's bits (fixed-order stages); ``dX`` is
    reported: max |ND - det| and the BF16 flip rate."""
    for key in ("loss", "logp", "dW"):
        assert _same_bits(nd[key], det[key]), key
    if nd["dX"] is not None:
        diff = float((nd["dX"].float() - det["dX"].float()).abs().max())
        flips, max_ulp = _ulp_flips(nd["dX"], det["dX"])
        if record is not None:
            record.append(
                dict(dX_max_abs_vs_det=diff, dX_flip_rate=flips, dX_max_ulp=max_ulp)
            )
        return diff, flips, max_ulp
    return None


@contextlib.contextmanager
def _launch_record():
    """The launch keys both backend modules run (each key as ``_run_keys`` runs it), in order: the record of what a
    call launched (the hidden valid-row count's own chunk-0 launches are not keys).  A remembered binding fills
    ``dX_acc`` as it allocates it (``_filled_dx_acc``) instead of running the ``dx_fill`` key: that fill is recorded
    as ``("dx_fill", "dx")`` at its place in the sequence."""
    launched: list = []
    originals = (cake_backend._run_keys, opt_in._run_keys)
    fills = (cake_backend._filled_dx_acc, opt_in._filled_dx_acc)

    def recording(original):
        def run_keys(plan, device_index, keys, run_key, *args, **kwargs):
            def run(key):
                launched.append(key)
                return run_key(key)

            return original(plan, device_index, keys, run, *args, **kwargs)

        return run_keys

    def filling(original):
        def filled_dx_acc(rows, hidden, fill_rows, device):
            if fill_rows is not None:
                launched.append(("dx_fill", "dx"))
            return original(rows, hidden, fill_rows, device)

        return filled_dx_acc

    cake_backend._run_keys = recording(originals[0])
    opt_in._run_keys = recording(originals[1])
    cake_backend._filled_dx_acc = filling(fills[0])
    opt_in._filled_dx_acc = filling(fills[1])
    try:
        yield launched
    finally:
        cake_backend._run_keys, opt_in._run_keys = originals
        cake_backend._filled_dx_acc, opt_in._filled_dx_acc = fills


# (T, objective, entry, gemm_tuning): a one-row call, a multi-chunk call with a one-row tail, a long call with a
# ragged tail (each chunk at the slice rule's count: the reduction above one slice), a representative four-chunk call,
# and forced slice counts (one: the plain store everywhere; four and eight: the reduction).
DEVICE_CASES = [
    (1, "ce", "loss", None),
    (4097, "ce", "loss", None),
    (4097, "policy", "loss", None),
    (8191, "policy", "loss", None),
    (16231, "ce", "loss", None),
    (4097, "ce", "logprob", None),
    (4097, "ce", "loss", {"dx": {"k_slices": None}}),
    (4097, "ce", "loss", {"dx": {"k_slices": 4}}),
    (4097, "ce", "loss", {"dx": {"k_slices": 8}}),
    (4097, "ce", "logprob", {"dx": {"k_slices": 4}}),
]


@pytest.mark.parametrize(
    "T, objective, entry, tuning",
    DEVICE_CASES,
    ids=lambda v: "-" if v is None else str(v).replace(" ", ""),
)
def test_device_matches_references_and_the_deterministic_package(
    glm_weight, T, objective, entry, tuning
):
    _require_program(entry=entry)
    inp = _device_inputs(T, objective, W=glm_weight)
    kw = {} if tuning is None else dict(gemm_tuning=tuning)
    nd = _run(inp, 4096, entry=entry, backend="cake", **kw)
    det = _run(inp, 4096, entry=entry, backend="cake", package="det", **_det_tuning(kw))
    errors, b0 = _check_against_references(nd, inp, entry=entry, ceiling=True)
    _check_against_references(det, inp, entry=entry, ceiling=True)
    diff, flips, max_ulp = _compare_with_det(nd, det)
    print(
        f"\nND T={T} {objective} {entry} {tuning}: dX_rel_l2 {errors['dX_rel_l2']:.3e} (b0 {b0['dX_rel_l2']:.3e}) "
        f"dW_rel_l2 {errors['dW_rel_l2']:.3e} max|ND-det| dX {diff:.3e} flip {flips:.2e} max_ulp {max_ulp}"
    )


@pytest.mark.parametrize(
    "T, objective, entry, tuning, policy",
    [
        (4097, "policy", "loss", None, "det"),
        (4097, "ce", "loss", {"dx": {"k_slices": 4}}, "det"),
        (4097, "ce", "loss", {"dx": {"k_slices": 8}}, "det"),
        (8191, "ce", "logprob", None, "det"),
        (16231, "ce", "loss", None, "det"),
        (16231, "ce", "loss", None, "measured"),
        (16231, "policy", "loss", None, "measured"),
    ],
    ids=lambda v: "-" if v is None else str(v).replace(" ", ""),
)
def test_device_worst_case_over_repeated_runs(
    glm_weight, T, objective, entry, tuning, policy
):
    """``REPEATS`` runs on the same inputs (fresh allocator state each time, 0xFF-poisoned): every run passes the
    gates against the FP64 oracle and the unchunked FP32 path; ``loss`` / ``logp`` / ``dW`` never change and are the
    deterministic package's bits; the worst ``dX`` error, the run-to-run ``dX`` spread and the worst max |ND - det| and
    BF16 flip rate of ``dX`` are reported."""
    _require_program(entry=entry)
    inp = _device_inputs(T, objective, W=glm_weight)
    kw = {} if tuning is None else dict(gemm_tuning=tuning)
    oracle = reference_fp64(inp, entry=entry)
    b0 = error_report(reference_unchunked(inp, entry=entry), oracle, inp.labels)
    det = _run(inp, 4096, entry=entry, backend="cake", package="det", **_det_tuning(kw))
    first, worst, spread, versus = None, {}, 0.0, []
    for _ in range(REPEATS):
        _poison_allocator(1 << 30)
        with _slice_policy(policy):
            run = _run(inp, 4096, entry=entry, backend="cake", **kw)
        _compare_with_det(run, det, record=versus)
        errors = error_report(run, oracle, inp.labels)
        assert (
            not errors["nan"]
            and errors["logp_ignored_zero"]
            and errors["dX_ignored_zero"]
        )
        for key, tiny in GATE_TINY.items():
            if key in errors:
                assert errors[key] <= GATE_MARGIN * b0[key] + tiny, key
                worst[key] = max(worst.get(key, 0.0), errors[key])
        if first is None:
            first = run
        else:
            for key in ("loss", "logp", "dW"):
                assert _same_bits(run[key], first[key]), key
            spread = max(
                spread, float((run["dX"].float() - first["dX"].float()).abs().max())
            )
    print(
        f"\nND worst over {REPEATS} runs T={T} {objective} {entry} {tuning} policy={policy}: {worst}; dX run-to-run "
        f"max {spread:.3e}; max|ND-det| dX {max(v['dX_max_abs_vs_det'] for v in versus):.3e} flip "
        f"{max(v['dX_flip_rate'] for v in versus):.2e} max_ulp {max(v['dX_max_ulp'] for v in versus)}"
    )


def test_device_changing_shapes_reuse_nonzero_scratch(glm_weight):
    """Ragged and changing row counts on one weight, the binding cache and the allocator's blocks reused between
    calls after a 0xFF fill: every call passes the gates and repeats its own ``loss`` / ``logp`` / ``dW``."""
    _require_program(entry="loss")
    sizes = (4097, 1, 12289, 4095, 37, 4097, 8191, 1)
    inputs = {T: _device_inputs(T, "policy", W=glm_weight) for T in sorted(set(sizes))}
    seen = {}
    for T in sizes:
        _poison_allocator(1 << 30)
        run = _run(inputs[T], 4096, backend="cake")
        if T in seen:
            for key in ("loss", "logp", "dW"):
                assert _same_bits(run[key], seen[T][key]), (T, key)
        else:
            _check_against_references(run, inputs[T], ceiling=True)
            seen[T] = run


def test_device_checkpoint_replay(glm_weight):
    """``torch.utils.checkpoint`` replays the forward in the backward: ``loss`` / ``logp`` / ``dW`` are the
    uncheckpointed call's bits, ``dX`` passes the gates."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "ce", W=glm_weight)

    def head(X, W):
        return cake_backend.chunked_lm_head_loss(
            X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, **ND
        )

    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
    loss.backward()
    plain = _run(inp, 4096, backend="cake")
    assert _same_bits(loss.detach(), plain["loss"])
    assert _same_bits(W.grad, plain["dW"])
    _check_against_references(
        dict(loss=loss.detach(), logp=plain["logp"], dX=X.grad, dW=W.grad),
        inp,
        ceiling=True,
    )


def test_device_gradient_accumulation(glm_weight):
    """Two backward passes into the same leaves: each leaf's ``.grad`` is the sum of the two incoming gradients (the
    autograd engine's BF16 add), and each incoming gradient passes the gates."""
    _require_program(entry="loss")
    a = _device_inputs(4097, "ce", W=glm_weight)
    b = _device_inputs(4095, "ce", W=glm_weight, seed=SEED + 7)
    X = a.X.detach().requires_grad_(True)
    W = glm_weight.detach().requires_grad_(True)
    seen_w = []
    W.register_hook(lambda g: seen_w.append(g.detach().clone()))
    for inp in (a, b):
        loss = cake_backend.chunked_lm_head_loss(
            X if inp is a else inp.X.detach().requires_grad_(False),
            W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=4096,
            **ND,
        )
        loss.backward()
    assert len(seen_w) == 2
    assert _same_bits(W.grad, seen_w[0] + seen_w[1])
    single_a, single_b = (
        _run(a, 4096, backend="cake"),
        _run(b, 4096, backend="cake", train_x=False),
    )
    assert _same_bits(seen_w[0], single_a["dW"]) and _same_bits(
        seen_w[1], single_b["dW"]
    )
    _check_against_references(dict(single_a, dX=X.grad), a, ceiling=True)


def test_device_retained_graph(glm_weight):
    """A retained graph runs the backward again from the saved accumulators: the same bits both times."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "policy", W=glm_weight)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss = cake_backend.chunked_lm_head_loss(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
        **ND,
    )
    dx1, dw1 = torch.autograd.grad(loss, (X, W), retain_graph=True)
    dx2, dw2 = torch.autograd.grad(loss, (X, W))
    assert _same_bits(dx1, dx2) and _same_bits(dw1, dw2)


def test_device_concurrent_streams(glm_weight):
    """Calls on two streams at once (different row counts, one weight): each gives the serial call's ``loss`` /
    ``logp`` / ``dW`` and passes the gates."""
    _require_program(entry="loss")
    inputs = [_device_inputs(T, "ce", W=glm_weight) for T in (4097, 8191)]
    serial = [_run(inp, 4096, backend="cake") for inp in inputs]
    streams = [torch.cuda.Stream() for _ in inputs]
    results = [None, None]
    torch.cuda.synchronize()
    for i, (inp, stream) in enumerate(zip(inputs, streams, strict=True)):
        with torch.cuda.stream(stream):
            X = inp.X.detach().requires_grad_(True)
            W = inp.W.detach().requires_grad_(True)
            loss, logp = cake_backend.chunked_lm_head_loss(
                X,
                W,
                inp.labels,
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                **ND,
            )
            loss.backward()
            results[i] = dict(loss=loss.detach(), logp=logp, dX=X.grad, dW=W.grad)
    torch.cuda.synchronize()
    for inp, got, want in zip(inputs, results, serial, strict=True):
        for key in ("loss", "logp", "dW"):
            assert _same_bits(got[key], want[key]), key
        _check_against_references(got, inp, ceiling=True)


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_prepared_runner_and_cuda_graph(glm_weight, entry):
    """A prepared runner launches without allocating; a CUDA graph captured from it replays for new values written
    into the bound tensors and gives the runner's own results on those values."""
    _require_program(entry=entry)
    inp = _device_inputs(4097, "ce", W=glm_weight, ignore_frac=0.0)
    new = _device_inputs(4097, "ce", W=glm_weight, seed=SEED + 11, ignore_frac=0.0)
    X = inp.X.clone()
    labels = inp.labels.clone()
    kw = dict(chunk_size=4096, backend="cake", entry=entry, **ND)
    if entry == "loss":
        kw.update(loss_div=inp.loss_div)
    runner = cake_backend.prepare_lm_head_loss(X, inp.W, labels, **kw)
    if entry == "logprob":
        runner.tensors["d_in"].copy_(inp.dlogp)
    runner.step()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    runner.step()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        runner.step()  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.step()
    X.copy_(new.X)
    labels.copy_(new.labels)
    if entry == "logprob":
        runner.tensors["d_in"].copy_(new.dlogp)

    def outputs():
        return {
            name: (None if t is None else t.clone())
            for name, t in (
                ("logp", runner.logp),
                ("dX", runner.dx_out),
                ("dW", runner.dw_out),
                ("loss", runner.loss if entry == "loss" else None),
            )
        }

    assert (
        runner.plan.dx_fill_rows is not None
    )  # the graph holds the fill and the reduction
    graph.replay()
    torch.cuda.synchronize()
    replayed = outputs()
    # the bound accumulators and chunk buffers hold NaN before the second replay: the graph's own fill restores the
    # additive identity under every reduction
    for name in ("dx_acc", "logits", "stats"):
        runner.tensors[name].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    again = outputs()
    for key in ("loss", "logp", "dW"):
        if replayed[key] is not None:
            assert _same_bits(again[key], replayed[key]), key
    assert bool(torch.isfinite(again["dX"].float()).all())
    print(
        f"\nND CUDA graph {entry}: second replay max|dX - first| "
        f"{float((again['dX'].float() - replayed['dX'].float()).abs().max()):.3e}"
    )
    runner.step()
    torch.cuda.synchronize()
    eager = dict(
        logp=runner.logp,
        dX=runner.dx_out,
        dW=runner.dw_out,
        loss=runner.loss if entry == "loss" else None,
    )
    for key in ("loss", "logp", "dW"):
        if replayed[key] is not None:
            assert _same_bits(replayed[key], eager[key]), key
    result = dict(
        loss=replayed["loss"].reshape(())
        if entry == "loss"
        else (replayed["logp"] * new.dlogp).sum(),
        logp=replayed["logp"],
        dX=replayed["dX"],
        dW=replayed["dW"],
    )
    if entry == "loss":
        _check_against_references(result, new, ceiling=True)
    else:
        _check_against_references(
            dict(result, loss=None), new, entry="logprob", ceiling=True
        )


def test_device_frozen_inputs(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4095, W=glm_weight)
    both = _run(inp, 4096, backend="cake")
    x_only = _run(inp, 4096, backend="cake", train_w=False)
    w_only = _run(inp, 4096, backend="cake", train_x=False)
    assert x_only["dW"] is None and w_only["dX"] is None
    assert _same_bits(w_only["dW"], both["dW"])
    for r in (x_only, w_only):
        assert _same_bits(r["loss"], both["loss"]) and _same_bits(
            r["logp"], both["logp"]
        )
    _check_against_references(x_only, inp, ceiling=True)


@pytest.mark.parametrize("memory_mode", ["fast", "budget"])
def test_device_opt_in_forms(glm_weight, memory_mode):
    """The opt-in forms (``memory_mode``, the caller's valid-row count) of this package pass the gates and keep the
    deterministic package's ``loss`` / ``logp`` / ``dW``."""
    _require_program(entry="loss")
    inp = _device_inputs(8191, "ce", W=glm_weight)
    count = int(inp.valid.sum())
    nd = _run(inp, 4096, backend="cake", memory_mode=memory_mode, num_valid_rows=count)
    det = _run(
        inp,
        4096,
        backend="cake",
        package="det",
        memory_mode=memory_mode,
        num_valid_rows=count,
    )
    _check_against_references(nd, inp, ceiling=True)
    _compare_with_det(nd, det)


@pytest.mark.parametrize("policy", ["det", "measured"])
@pytest.mark.parametrize(
    "T, entry", [(16231, "loss"), (4097, "loss"), (8191, "logprob")], ids=str
)
def test_device_unordered_reduce_is_exercised(glm_weight, T, entry, policy):
    """On the representative rows the eager entries launch the K-sliced reduction (``gemm_dx_red*``) for every
    chunk the slice rule gives more than one slice, behind one ``dX_acc`` fill, and no slab: the device's registered
    program serves the reduction forms and no deterministic sliced form, the bound runner holds no slab workspace."""
    _require_program(entry=entry)
    module_name, record = cake_backend.record_for(CUDA)
    registered = set(cake_jit.registered_stages(module_name))
    assert cake_backend.DX_RED_STAGE in registered
    assert "slab_sum" not in registered and not any(_det_sliced(s) for s in registered)
    assert record["abi"] == cake_backend.ABI_CONTRACT
    inp = _device_inputs(T, "ce", W=glm_weight)
    runners = []
    original = cake_backend.prepare_lm_head_loss

    def recording(*a, **k):
        runner = original(*a, **k)
        runners.append(runner)
        return runner

    cake_backend.BINDING_CACHE.clear()
    cake_backend.prepare_lm_head_loss = recording
    try:
        with _slice_policy(policy), _launch_record() as launched:
            nd = _run(inp, 4096, entry=entry, backend="cake")
    finally:
        cake_backend.prepare_lm_head_loss = original
        cake_backend.BINDING_CACHE.clear()
    ran = [key for key in launched if key[0].startswith(cake_backend.DX_RED_STAGE)]
    runner = runners[-1]
    plan = runner.plan
    order = runner.forward_order + runner.backward_order
    red = [key for key in order if key[0].startswith(cake_backend.DX_RED_STAGE)]
    sliced = [i for i in range(plan.num_chunks) if plan.dx_slices_of(i) > 1]
    assert sliced and [key[1] for key in red] == sliced, (plan.dx_slices, red)
    assert [key[1] for key in ran] == sliced, (
        plan.dx_slices,
        ran,
    )  # launched, not only planned
    assert launched.count(("dx_fill", "dx")) == 1
    assert order.count(("dx_fill", "dx")) == 1
    assert order.index(("dx_fill", "dx")) < order.index(red[0])
    assert not any(key[0] in ("slab_sum", "dx_reduce") for key in order)
    assert "dx_ws" not in runner.tensors and "dx_ws" not in runner.layout
    _check_against_references(nd, inp, entry=entry, ceiling=True)
    print(
        f"\nND reduction T={T} {entry} policy={policy}: slices {plan.dx_slices} red {[k[0] for k in red]}"
    )


def _device_slices(
    rows: int, policy: str, W: torch.Tensor, *, chunk: int = 4096, red_min_rows: int = 0
) -> tuple[int, ...]:
    """The running device's dX slices of a ``rows``-row cross-entropy call (every row valid) on the weight ``W`` under
    ``policy`` and the plain-store floor ``red_min_rows``: the planner's rule on this device's SM count, co-resident
    clusters and architecture (no allocation, no launch)."""
    hidden, vocab = int(W.shape[1]), int(W.shape[0])
    module_name, record = cake_backend.record_for(CUDA)
    stages = cake_jit.registered_stages(module_name)
    geometry = cake_backend.Geometry.from_record(record)
    num_sms = int(torch.cuda.get_device_properties(CUDA).multi_processor_count)
    problem = cake_backend.Problem(
        num_rows=rows,
        hidden=hidden,
        vocab=vocab,
        chunk=chunk,
        ld_x=hidden,
        x_copy=False,
        objective="ce",
        loss_div=1.0,
        grad_weight_dtype=torch.bfloat16,
    )
    return cake_backend.plan_dx_slices(
        problem,
        cake_backend.plan_chunks(rows, chunk),
        need_dx=True,
        tuning={},
        dx_max_slices=cake_backend.dx_max_slices(stages),
        num_sms=num_sms,
        geometry=geometry,
        dx_resident=cake_backend.cluster_resident(
            CUDA, geometry.dx_cluster_ctas, num_sms
        ),
        arch=record["arch"],
        policy=policy,
        red_min_rows=red_min_rows,
    )


@pytest.mark.parametrize("policy", ["det", "measured"])
def test_device_one_slice_chunks_are_the_deterministic_bits(glm_weight, policy):
    """A call whose plan on the RUNNING device mixes a chunk of more than one K slice (the reduction) with a chunk of
    one (the plain store) -- the row count is searched on this device's plan (its SM count and architecture), never
    assumed: the one-slice chunk's ``dX`` rows are the deterministic package's one-slice bits, ``loss`` / ``logp`` /
    ``dW`` its bits under its own rule, the call passes the gates, and the
    sliced rows' max |ND - det| and BF16 flip rate are reported.  Skipped only when no two-chunk call of this device
    has such a plan."""
    _require_program(entry="loss")
    T = next(
        (
            rows
            for rows in range(4097, 8192)
            if 1 in (s := _device_slices(rows, policy, glm_weight)) and max(s) > 1
        ),
        None,
    )
    if T is None:
        pytest.skip(
            f"no two-chunk call on this device plans a one-slice chunk next to a sliced one ({policy} policy)"
        )
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
    with _slice_policy(policy):
        runner = cake_backend.prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=4096,
            **ND,
        )
        assert runner.plan.compact is False or runner.plan.rows == T
        slices, chunks = runner.plan.dx_slices, runner.plan.chunks
        del runner
        nd = _run(inp, 4096, backend="cake")
    assert slices == _device_slices(T, policy, glm_weight), slices
    det = _run(inp, 4096, backend="cake", package="det")
    one = _run(
        inp, 4096, backend="cake", package="det", gemm_tuning={"dx": {"k_slices": None}}
    )
    _check_against_references(nd, inp, ceiling=True)
    for key in ("loss", "logp", "dW"):
        assert _same_bits(nd[key], det[key]), key
    sliced = torch.zeros(T, dtype=torch.bool, device=CUDA)
    for (row0, rows_c), s in zip(chunks, slices, strict=True):
        if s > 1:
            sliced[row0 : row0 + rows_c] = True
    # the one-slice rows: the deterministic package's one-slice dX bits (one slice sums the same products in the same
    # K order in every dX form, so this holds whatever slice count the deterministic rule plans for that chunk)
    assert _same_bits(nd["dX"][~sliced], one["dX"][~sliced])
    diff = float((nd["dX"][sliced].float() - det["dX"][sliced].float()).abs().max())
    flips, max_ulp = _ulp_flips(nd["dX"][sliced], det["dX"][sliced])
    print(
        f"\nND mixed slices T={T} {slices} policy={policy}: S1 rows bitwise det; sliced rows max|ND-det| {diff:.3e} "
        f"flip {flips:.2e} max_ulp {max_ulp}"
    )


def test_device_logp_statistics_never_change(glm_weight):
    """``loss`` / ``logp`` / ``lse`` keep their fixed order in this package: the deterministic package's bits."""
    _require_program(entry="logprob")
    inp = _device_inputs(8191, "ce", W=glm_weight)
    nd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096, **ND)
    det = det_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
    assert _same_bits(nd.logp, det.logp) and _same_bits(nd.lse, det.lse)
    assert math.isfinite(float(nd.logp.sum()))


# ---------------------------------------------------------------------------
# Zero-weight policy rows
# ---------------------------------------------------------------------------


def _zero_weight_set(inp, kind, *, seed=SEED):
    """bool ``[T]`` of the valid rows whose weight becomes zero: ``"none"``, ``"one"`` (the first valid row),
    ``("group", f)`` (whole groups of 512 consecutive rows, ``round(f * groups)`` of them), ``("random", f)``
    (``round(f * T_v)`` random valid rows) or ``"all"``."""
    valid = inp.labels >= 0
    T = int(valid.numel())
    zs = torch.zeros(T, dtype=torch.bool, device=valid.device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + T)
    if kind == "none":
        return zs
    if kind == "all":
        return valid.clone()
    if kind == "one":
        zs[int(valid.nonzero()[0])] = True
        return zs
    pattern, frac = kind
    if pattern == "group":
        groups = -(-T // 512)
        for g in torch.randperm(groups, generator=gen)[
            : int(round(frac * groups))
        ].tolist():
            zs[g * 512 : min(T, (g + 1) * 512)] = True
    else:
        rows = valid.nonzero().squeeze(1).cpu()
        pick = rows[
            torch.randperm(rows.numel(), generator=gen)[
                : int(round(frac * rows.numel()))
            ]
        ]
        zs[pick.to(zs.device)] = True
    return zs & valid


def _with_zero_weights(inp, zero, *, negative_zero=False):
    """``inp`` with ``loss_weights`` 0 on ``zero`` (``-0.0`` on every other zero row with ``negative_zero``)."""
    w = inp.loss_weights.clone()
    w[zero] = 0.0
    if negative_zero:
        rows = zero.nonzero().squeeze(1)
        w[rows[::2]] = -0.0
    return dataclasses.replace(inp, loss_weights=w.contiguous())


def _loss_call(
    inp,
    C,
    *,
    package="nd",
    backend="cake",
    return_logp=True,
    train_x=True,
    train_w=True,
    **kw,
):
    """One autograd loss call (forward + backward) of ``package``; returns loss / logp (``None`` without
    ``return_logp``) / dX / dW."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    if package == "nd":
        fn, mode = cake_backend.chunked_lm_head_loss, ND
    else:
        fn, mode = det_backend.chunked_lm_head_det_loss, {}
    out = fn(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=C,
        return_logp=return_logp,
        backend=backend,
        **mode,
        **kw,
    )
    loss, logp = out if return_logp else (out, None)
    if train_x or train_w:
        loss.backward()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return dict(
        loss=loss.detach(),
        logp=None if logp is None else logp.detach(),
        dX=X.grad,
        dW=W.grad,
    )


def test_gradient_rows_split_the_valid_rows():
    """``gradient_rows``: the gradient rows (weight != 0, NaN included) and the zero-weight rows (+0 and -0), both
    ascending, ``None`` for an empty side; every row in ``G``: the uncompacted path."""
    labels = torch.tensor([3, -100, 5, 7, 1, 2, -100, 4, 0], dtype=torch.int64)
    w = torch.tensor([1.0, 0.0, 0.0, -0.0, float("nan"), -2.0, 3.0, 0.0, 0.5])
    g, mask, z = cake_backend.gradient_rows(labels, w, mask=True)
    assert g.tolist() == [0, 4, 5, 8] and z.tolist() == [2, 3, 7]
    assert mask.tolist() == [True, False, False, False, True, True, False, False, True]
    g, mask, z = cake_backend.gradient_rows(labels, torch.ones(9))
    assert g.tolist() == [0, 2, 3, 4, 5, 7, 8] and mask is None and z is None
    assert cake_backend.gradient_rows(torch.arange(4), torch.ones(4)) == (
        None,
        None,
        None,
    )
    g, _, z = cake_backend.gradient_rows(torch.arange(4), torch.zeros(4))
    assert g.numel() == 0 and z.tolist() == [0, 1, 2, 3]


def test_zero_weight_split_applies_to_compacted_policy_loss_calls_only():
    inp = _host_inputs(8, "policy", device=CPU)
    problem = cake_backend.validate_lm_head_inputs(
        inp.X,
        inp.W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        **ND,
    )
    assert cake_backend.zero_weight_split(problem, True)
    assert not cake_backend.zero_weight_split(problem, False)
    ce = dataclasses.replace(problem, objective="ce", loss_div=1.0)
    assert not cake_backend.zero_weight_split(ce, True)
    lp = dataclasses.replace(problem, objective="ce", loss_div=None, entry="logprob")
    assert not cake_backend.zero_weight_split(lp, True)


ZERO_KINDS_HOST = ["one", ("random", 0.166), ("group", 0.5), "all"]


@pytest.mark.parametrize("kind", ZERO_KINDS_HOST, ids=str)
@pytest.mark.parametrize("T, C", [(37, 16), (1200, 256)])
def test_reference_zero_weight_rows(kind, T, C):
    """The reference backend with zero-weight rows (ignored rows mixed in, ragged multi-chunk calls): the
    deterministic suite's gates against the FP64 oracle; ``logp`` of every valid row (the zero-weight rows' from the
    logp-only pass) matches the deterministic reference; the zero-weight rows' ``dX`` rows are exact zeros; ``loss``
    and the gradients agree with the deterministic reference within the gates."""
    base = _host_inputs(T, "policy")
    if kind == ("group", 0.5) and T < 1024:
        kind = ("random", 0.5)  # one 512-row group: the fraction rounds to no group
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero, negative_zero=True)
    nd = _run(inp, C)
    det = _run(inp, C, package="det")
    valid = inp.labels >= 0
    torch.testing.assert_close(nd["logp"], det["logp"], atol=1e-5, rtol=0)
    assert bool((nd["logp"][~valid] == 0).all())
    assert bool((nd["dX"][zero] == 0).all())
    if bool(zero[valid].all()):
        assert float(nd["loss"]) == 0.0
        assert not bool(nd["dX"].any()) and not bool(nd["dW"].any())
    else:
        _check_against_references(nd, inp)
        assert (
            abs(float(nd["loss"]) - float(det["loss"]))
            <= 1e-6 * abs(float(det["loss"])) + 1e-6
        )


def test_reference_zero_weight_rows_without_return_logp():
    """``return_logp=False``: the logp-only pass does not run (``forward_loss`` leaves the zero-weight rows' ``logp``
    at 0); ``loss`` and the gradients are the ``return_logp=True`` call's bits."""
    base = _host_inputs(200, "policy")
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="reference",
        **ND,
    )
    with_logp = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    without = cake_backend.forward_loss(
        inp.X, inp.W, inp.labels, return_logp=False, **kw
    )
    assert bool((without.logp[zero] == 0).all()) and bool(
        (with_logp.logp[zero] != 0).all()
    )
    assert _same_bits(without.logp[~zero], with_logp.logp[~zero])
    assert _same_bits(without.loss, with_logp.loss)
    assert with_logp.memory["zero_weight_rows"] == int(zero.sum())
    assert without.memory["zero_weight_logp_temporary_bytes"] == 0
    a = _loss_call(inp, 64, backend="reference")
    b = _loss_call(inp, 64, backend="reference", return_logp=False)
    for key in ("loss", "dX", "dW"):
        assert _same_bits(a[key], b[key]), key


def test_reference_zero_weight_rows_do_not_propagate_non_finite_values():
    """A non-finite ``X`` row or ``infer_logp`` in a zero-weight row does not reach ``loss`` / ``dX`` / ``dW`` (the
    row is treated like an ignored row for the gradients); a NaN weight keeps its row in the gradient path."""
    base = _host_inputs(200, "policy")
    zero = _zero_weight_set(base, ("random", 0.2))
    clean = _with_zero_weights(base, zero)
    rows = zero.nonzero().squeeze(1)
    X = clean.X.clone()
    X[rows[0]] = float("inf")
    infer = clean.infer_logp.clone()
    infer[rows[1]] = float("nan")
    poisoned = dataclasses.replace(clean, X=X, infer_logp=infer)
    a = _loss_call(clean, 64, backend="reference")
    b = _loss_call(poisoned, 64, backend="reference")
    for key in ("loss", "dW"):
        assert _same_bits(a[key], b[key]), key
    assert _same_bits(a["dX"][~zero], b["dX"][~zero]) and not bool(b["dX"][zero].any())
    assert _same_bits(a["logp"][rows[2:]], b["logp"][rows[2:]])
    w = clean.loss_weights.clone()
    w[int((~zero & (clean.labels >= 0)).nonzero()[0])] = float("nan")
    nan_row = _loss_call(
        dataclasses.replace(clean, loss_weights=w), 64, backend="reference"
    )
    assert math.isnan(float(nan_row["loss"]))


# (T, zero rows, chunk): one row, 16.6 % in groups of 512 and at random, 30 / 50 % at random and in groups, every
# valid row; multi-chunk calls (C 4096) with ragged tails, 5 % ignored rows mixed in, T > 3 chunks on the hidden
# valid-row count path, T <= 1024 on the host-mask path.
ZERO_DEVICE_CASES = [
    (1000, "one"),
    (4097, "one"),
    (12289, ("group", 0.166)),
    (12289, ("random", 0.166)),
    (8191, ("random", 0.3)),
    (12289, ("group", 0.5)),
    (4097, ("random", 0.5)),
    (12289, "all"),
]


@pytest.mark.parametrize("return_logp", [True, False])
@pytest.mark.parametrize("T, kind", ZERO_DEVICE_CASES, ids=str)
def test_device_zero_weight_rows_match_the_deterministic_package(
    glm_weight, T, kind, return_logp
):
    """Zero-weight policy rows on the device: the deterministic suite's gates against the FP64 oracle; ``logp`` of
    every valid row -- the zero-weight rows' from the logp-only pass -- is the deterministic package's bits (with
    ``return_logp``); the zero-weight rows' ``dX`` rows are exact zeros; ``loss`` / ``dW`` / ``dX`` are reported
    against the deterministic package.  Every valid row of zero weight: loss 0, zero gradients, no gradient GEMM."""
    _require_program(entry="loss")
    base = _device_inputs(T, "policy", W=glm_weight)
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero, negative_zero=True)
    launched = []
    original = cake_backend.prepare_lm_head_loss

    def recording(*a, **k):
        launched.append(k.get("entry", "loss"))
        return original(*a, **k)

    cake_backend.BINDING_CACHE.clear()
    cake_backend.prepare_lm_head_loss = recording
    try:
        nd = _loss_call(inp, 4096, return_logp=return_logp)
    finally:
        cake_backend.prepare_lm_head_loss = original
    det = _loss_call(inp, 4096, package="det")
    valid = inp.labels >= 0
    assert bool((nd["dX"][zero] == 0).all())
    if return_logp:
        assert _same_bits(nd["logp"], det["logp"])
    every = bool(zero[valid].all())
    if every:
        assert launched == (["logprob"] if return_logp else [])
        assert float(nd["loss"]) == 0.0
        assert not bool(nd["dX"].any()) and not bool(nd["dW"].any())
        return
    assert launched.count("loss") == 1
    # without return_logp the gates read the deterministic logp (the gradients and the loss are this call's)
    _check_against_references(
        nd if return_logp else dict(nd, logp=det["logp"]), inp, ceiling=True
    )
    loss_rel = abs(float(nd["loss"]) - float(det["loss"])) / max(
        abs(float(det["loss"])), 1e-30
    )
    dw_rel = float(
        (nd["dW"].float() - det["dW"].float()).norm() / det["dW"].float().norm()
    )
    dx_diff = float((nd["dX"].float() - det["dX"].float()).abs().max())
    dw_flips, dw_ulp = _ulp_flips(nd["dW"], det["dW"])
    assert loss_rel <= 1e-6, loss_rel
    print(
        f"\nzero-weight T={T} {kind} zero {int(zero.sum())}/{int(valid.sum())} logp={return_logp}: loss_rel vs det "
        f"{loss_rel:.2e}, dW rel-L2 vs det {dw_rel:.2e} (flip {dw_flips:.2e} max_ulp {dw_ulp}), "
        f"dX max|ND-det| {dx_diff:.2e}"
    )


def test_device_zero_weight_rows_repeat_and_both_count_paths(glm_weight):
    """``REPEATS`` runs with 16.6 % zero-weight rows on the hidden valid-row count path and on the host count path:
    every run passes the gates and repeats the first run's ``loss`` / ``logp`` / ``dW``; both paths give the same
    ``loss`` / ``logp`` / ``dW`` bits; one synchronizing readback per call on the host path (PyTorch's
    synchronization debug mode), as the plain path."""
    _require_program(entry="loss")
    base = _device_inputs(12289, "policy", W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("random", 0.166)))
    oracle = reference_fp64(inp)
    b0 = error_report(reference_unchunked(inp), oracle, inp.labels)
    firsts = {}
    for hidden in (True, False):
        first = None
        for _ in range(REPEATS):
            _poison_allocator(1 << 30)
            run = _loss_call(inp, 4096, hidden_count=hidden)
            errors = error_report(run, oracle, inp.labels)
            assert (
                not errors["nan"]
                and errors["logp_ignored_zero"]
                and errors["dX_ignored_zero"]
            )
            for key, tiny in GATE_TINY.items():
                if key in errors:
                    assert errors[key] <= GATE_MARGIN * b0[key] + tiny, key
            if first is None:
                first = run
            else:
                for key in ("loss", "logp", "dW"):
                    assert _same_bits(run[key], first[key]), key
        firsts[hidden] = first
    for key in ("loss", "logp", "dW"):
        assert _same_bits(firsts[True][key], firsts[False][key]), key

    def syncs(forward, hidden, **mode):
        counts = []
        for _ in range(3):
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    forward(
                        inp.X,
                        inp.W,
                        inp.labels,
                        objective="policy",
                        infer_logp=inp.infer_logp,
                        loss_weights=inp.loss_weights,
                        chunk_size=4096,
                        hidden_count=hidden,
                        **mode,
                    )
            finally:
                torch.cuda.set_sync_debug_mode("default")
            counts.append(sum("synchroniz" in str(w.message) for w in rec))
        return min(counts)

    for hidden in (False, True):
        nd_syncs = syncs(cake_backend.forward_loss, hidden, **ND)
        det_syncs = syncs(det_backend.forward_loss, hidden)
        print(
            f"\nzero-weight synchronizations hidden_count={hidden}: ND {nd_syncs} det {det_syncs}"
        )
        assert nd_syncs == det_syncs


def test_device_zero_weight_rows_checkpoint_accumulation_retained_graph(glm_weight):
    """With 30 % zero-weight rows: a checkpoint replay gives the plain call's ``loss`` / ``dW`` bits; a retained graph
    repeats its backward bit for bit; two backward passes accumulate into ``W.grad`` (the autograd engine's add)."""
    _require_program(entry="loss")
    base = _device_inputs(8191, "policy", W=glm_weight)
    inp = _with_zero_weights(base, _zero_weight_set(base, ("group", 0.3)))
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
        **ND,
    )

    def head(X, W):
        return cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)

    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
    loss.backward()
    plain = _loss_call(inp, 4096)
    assert _same_bits(loss.detach(), plain["loss"]) and _same_bits(W.grad, plain["dW"])
    _check_against_references(dict(plain, dX=X.grad), inp, ceiling=True)
    X_retained = inp.X.detach().requires_grad_(True)
    W_retained = inp.W.detach().requires_grad_(True)
    loss2 = head(X_retained, W_retained)
    dx1, dw1 = torch.autograd.grad(loss2, (X_retained, W_retained), retain_graph=True)
    dx2, dw2 = torch.autograd.grad(loss2, (X_retained, W_retained))
    assert _same_bits(dx1, dx2) and _same_bits(dw1, dw2)
    W_hooked = inp.W.detach().requires_grad_(True)
    seen = []
    W_hooked.register_hook(lambda g: seen.append(g.detach().clone()))
    for _ in range(2):
        head(inp.X, W_hooked).backward()
    assert len(seen) == 2 and _same_bits(W_hooked.grad, seen[0] + seen[1])


def test_device_zero_weight_rows_do_not_propagate_non_finite_values(glm_weight):
    """An ``inf`` ``X`` row and a NaN ``infer_logp`` in zero-weight rows leave ``loss`` / ``dX`` / ``dW`` finite and
    equal to the clean call's (the deterministic package gives a non-finite ``dW``); a NaN weight keeps its row in
    the gradient path (NaN loss)."""
    _require_program(entry="loss")
    base = _device_inputs(4097, "policy", W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.2))
    clean = _with_zero_weights(base, zero)
    rows = zero.nonzero().squeeze(1)
    X = clean.X.clone()
    X[rows[0]] = float("inf")
    infer = clean.infer_logp.clone()
    infer[rows[1]] = float("nan")
    poisoned = dataclasses.replace(clean, X=X, infer_logp=infer)
    a = _loss_call(clean, 4096)
    b = _loss_call(poisoned, 4096)
    for key in ("loss", "dW"):
        assert _same_bits(a[key], b[key]), key
    assert bool(torch.isfinite(b["dX"].float()).all()) and not bool(b["dX"][zero].any())
    det = _loss_call(poisoned, 4096, package="det")
    assert not bool(torch.isfinite(det["dW"].float()).all())
    w = clean.loss_weights.clone()
    w[int((~zero & (clean.labels >= 0)).nonzero()[0])] = float("nan")
    assert math.isnan(
        float(_loss_call(dataclasses.replace(clean, loss_weights=w), 4096)["loss"])
    )


@pytest.mark.parametrize("memory_mode", [None, "budget"])
@pytest.mark.parametrize("grads", ["w_frozen", "no_grad"])
def test_device_zero_weight_pass_adds_no_peak_without_a_weight_gradient(
    glm_weight, grads, memory_mode
):
    """Without a weight gradient (``W`` frozen, or no gradient at all) on the hidden valid-row count path
    (four chunks), the zero-weight rows' logp-only pass runs after the chunk loop once chunk 0's buffers are released:
    the call's measured peak with ``return_logp=True`` stays within the same call's without the split (every weight
    nonzero) plus the split's extra ``O(T)`` row vectors; the memory report's ``peak_temporary_bytes`` covers the
    loop and the pass; ``logp`` is the deterministic package's bits and the zero-weight rows' ``dX`` rows are zeros.
    ``memory_mode="budget"`` runs the opt-in module's forward: without a weight gradient its budget form does not
    apply (``budget = need_dw``), so the call keeps the hidden valid-row count and the opt-in module's own pass order --
    after the chunk loop (the recorded order), with the same peaks as the default rows."""
    _require_program(entry="loss")
    T = 16231
    module = cake_backend if memory_mode is None else opt_in
    order: list = []
    in_pass: list = []
    original_pass, original_forward = module._zero_weight_logp, module._Binding.forward

    def pass_recorded(*a, **k):
        order.append("zero_pass")
        in_pass.append(True)
        try:
            return original_pass(*a, **k)
        finally:
            in_pass.pop()

    def forward_recorded(self, *a, **k):
        if not in_pass:  # the pass's own log-probability binding is not the chunk loop
            order.append("loop")
        return original_forward(self, *a, **k)

    base = _device_inputs(T, "policy", W=glm_weight)
    zero = _zero_weight_set(base, ("group", 0.5))
    split = _with_zero_weights(base, zero)
    weights = base.loss_weights.clone()
    weights[weights == 0] = 1.0
    plain = dataclasses.replace(base, loss_weights=weights)
    train_x = grads == "w_frozen"

    def call(inp):
        X = inp.X.detach().requires_grad_(train_x)
        W = inp.W.detach()
        loss, logp = (opt_in if memory_mode else cake_backend).chunked_lm_head_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **({} if memory_mode is None else dict(memory_mode=memory_mode)),
            **ND,
        )
        if train_x:
            loss.backward()
        return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad)

    def peak(inp):
        call(
            inp
        )  # binds (a cache miss), then the measured calls take the remembered binding
        peaks, out = [], None
        for _ in range(2):
            out = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
            out = call(inp)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - start)
        return max(peaks), out

    plain_peak, _ = peak(plain)
    module._zero_weight_logp, module._Binding.forward = pass_recorded, forward_recorded
    try:
        split_peak, nd = peak(split)
    finally:
        module._zero_weight_logp, module._Binding.forward = (
            original_pass,
            original_forward,
        )
    assert order[-2:] == ["loop", "zero_pass"], (
        order
    )  # the measured call: the pass after the loop
    slack = (
        (1 << 20) + 64 * T
    )  # the split's extra O(T) vectors (both counts' masks and scans) and allocator rounding
    print(
        f"\nzero-weight pass peak {grads} {memory_mode} T={T} zero {int(zero.sum())}: peak with the split {split_peak} vs without {plain_peak} "
        f"(+{(split_peak - plain_peak) / 2**20:.2f} MiB)"
    )
    assert split_peak <= plain_peak + slack, (split_peak, plain_peak)
    det = _loss_call(split, 4096, package="det", train_w=False, train_x=train_x)
    assert _same_bits(nd["logp"], det["logp"])
    if train_x:
        assert not bool(nd["dX"][zero].any())
        _check_against_references(dict(nd, dW=None), split, ceiling=True)
    if memory_mode is not None:
        return
    result = cake_backend.forward_loss(
        split.X,
        split.W,
        split.labels,
        objective="policy",
        infer_logp=split.infer_logp,
        loss_weights=split.loss_weights,
        chunk_size=4096,
        need_dx=train_x,
        need_dw=False,
        **ND,
    )
    memory = result.memory
    assert memory.get("hidden_count") and memory["zero_weight_rows"] == int(zero.sum())
    assert memory["peak_temporary_bytes"] == max(
        memory["temporary_bytes"], memory["zero_weight_logp_phase_bytes"]
    )
    temporary = memory["temporary"]
    assert memory["zero_weight_logp_phase_bytes"] == memory[
        "zero_weight_logp_temporary_bytes"
    ] + temporary.get("row_index", 0) + temporary.get("row_valid", 0)


def _zero_pass_phase(monkeypatch, module) -> dict:
    """Measure the zero-weight rows' logp-only pass of ``module``'s loss forward: the bytes live at its entry above the
    call's start (``phase["start"]``, set by the caller) and the pass's own peak above that."""
    phase: dict = {}
    original = module._zero_weight_logp

    def measured(*a, **k):
        torch.cuda.synchronize()
        phase["live_at_entry"] = torch.cuda.memory_allocated() - phase["start"]
        torch.cuda.reset_peak_memory_stats()
        at = torch.cuda.memory_allocated()
        out = original(*a, **k)
        torch.cuda.synchronize()
        phase["pass_peak"] = torch.cuda.max_memory_allocated() - at
        return out

    monkeypatch.setattr(module, "_zero_weight_logp", measured)
    return phase


# (zero-weight rows, gradients, module, where the pass runs): with a weight gradient (the deferred last chunk) the pass
# runs ahead of the loop beside the hidden count's chunk-0 buffers; without one, after the loop; with every valid row
# of zero weight there is no loop and the pass runs once chunk 0's buffers are released (both modules).
ZERO_PHASE_CASES = [
    (("group", 0.5), "trainable", "nd", "ahead"),
    (("group", 0.5), "w_frozen", "nd", "after"),
    (("group", 0.5), "no_grad", "nd", "after"),
    ("all", "trainable", "nd", "alone"),
    ("all", "w_frozen", "nd", "alone"),
    ("all", "trainable", "opt_in", "alone"),
]
ZERO_PHASE_SLACK = (
    16 << 20
)  # allocator rounding and unsplit large-pool block remainders of the phase's tensors


@pytest.mark.parametrize("kind, grads, module, where", ZERO_PHASE_CASES, ids=str)
def test_device_zero_weight_pass_phase_matches_the_report(
    glm_weight, monkeypatch, kind, grads, module, where
):
    """The logp-only pass's phase measured directly -- the bytes live at its entry plus its own peak, above
    the call's start -- equals the report's ``zero_weight_logp_phase_bytes`` plus the accumulators alive at that
    point (after the loop: the FP32 accumulators the loop filled; ahead of it or without a loop: none) within
    ``ZERO_PHASE_SLACK``, in both directions; ``peak_temporary_bytes`` is the larger phase.  Every valid row of zero
    weight on the hidden valid-row count path with a trainable ``W``: the pass runs once chunk 0's buffers are released
    (no overlap: the bytes live at its entry are far below them)."""
    _require_program(entry="loss")
    backend = cake_backend if module == "nd" else opt_in
    base = _device_inputs(16231, "policy", W=glm_weight)
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero)
    phase = _zero_pass_phase(monkeypatch, backend)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
        need_dx=grads != "no_grad",
        need_dw=grads == "trainable",
        **ND,
    )
    result = None
    for _ in range(2):  # binds, then the measured call from the remembered binding
        result = None
        gc.collect()
        torch.cuda.synchronize()
        phase.clear()
        phase["start"] = torch.cuda.memory_allocated()
        result = backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
        torch.cuda.synchronize()
    memory = result.memory
    measured = phase["live_at_entry"] + phase["pass_peak"]
    live_acc = int(memory.get("accumulator_bytes", 0)) if where == "after" else 0
    expected = memory["zero_weight_logp_phase_bytes"] + live_acc
    rows0, V = min(4096, 16231), int(inp.V)
    chunk0 = rows0 * V * 2  # the hidden count's chunk-0 logits buffer alone
    print(
        f"\nzero-weight pass phase {kind} {grads} {module} ({where}): measured phase {measured} (live at entry {phase['live_at_entry']}, "
        f"pass peak {phase['pass_peak']}) reported phase {memory['zero_weight_logp_phase_bytes']} + live accumulators "
        f"{live_acc} = {expected} (diff {measured - expected:+d}); peak temporary {memory['peak_temporary_bytes']}"
    )
    assert abs(measured - expected) <= ZERO_PHASE_SLACK, (measured, expected, phase)
    assert memory["peak_temporary_bytes"] == max(
        int(memory.get("temporary_bytes", 0)), memory["zero_weight_logp_phase_bytes"]
    )
    temporary = memory.get("temporary") or {}
    chunk0_reported = sum(temporary.get(name, 0) for name in ("logits", "stats", "x_c"))
    if where == "ahead":
        assert phase["live_at_entry"] >= chunk0 and chunk0_reported >= chunk0
    else:
        assert phase["live_at_entry"] < chunk0 // 4, phase
    if kind == "all":
        assert result.row_index is not None and result.row_index.numel() == 0
        assert float(result.loss) == 0.0
        torch.testing.assert_close(
            result.logp,
            det_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096).logp,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("T", [1000, 4097, 8191, 12289, 16231])
@pytest.mark.parametrize("objective", ["ce", "policy_zero"])
def test_device_no_added_synchronization(glm_weight, T, objective):
    """The forward + backward's host waits on both valid-row count paths equal the deterministic package's on the same
    inputs, counted per kind: ``torch.cuda.Event.synchronize`` (the hidden count's readback, which PyTorch's
    synchronization debug mode does not report), ``torch.cuda.Stream.synchronize``, ``torch.cuda.synchronize`` and the
    debug mode's synchronizing operations.  On the hidden count path (three chunks or more) the call waits exactly once,
    and the forward's memory report records that path; the ``dX_acc`` fill, the K-sliced reduction and the zero-weight
    split (``policy_zero``: 16.6 % zero-weight rows) add none."""
    _require_program(entry="loss")
    base = _device_inputs(T, "policy" if objective != "ce" else "ce", W=glm_weight)
    inp = (
        _with_zero_weights(base, _zero_weight_set(base, ("random", 0.166)))
        if objective == "policy_zero"
        else base
    )
    kw = (
        dict(loss_div=inp.loss_div)
        if objective == "ce"
        else dict(
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
        )
    )

    def call(package, hidden):
        fn, mode = (
            (cake_backend.chunked_lm_head_loss, ND)
            if package == "nd"
            else (det_backend.chunked_lm_head_det_loss, {})
        )
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        loss, _ = fn(
            X,
            W,
            inp.labels,
            chunk_size=4096,
            return_logp=True,
            hidden_count=hidden,
            **kw,
            **mode,
        )
        loss.backward()

    original = (
        torch.cuda.Event.synchronize,
        torch.cuda.Stream.synchronize,
        torch.cuda.synchronize,
    )

    def waits(package, hidden):
        call(package, hidden)  # binds
        best = None
        for _ in range(3):
            counts = dict(event=0, stream=0, device=0, debug=0)

            def event_sync(self, _counts=counts):
                _counts["event"] += 1
                return original[0](self)

            def stream_sync(self, _counts=counts):
                _counts["stream"] += 1
                return original[1](self)

            def device_sync(*a, _counts=counts, **k):
                _counts["device"] += 1
                return original[2](*a, **k)

            torch.cuda.synchronize()
            torch.cuda.Event.synchronize = event_sync
            torch.cuda.Stream.synchronize = stream_sync
            torch.cuda.synchronize = device_sync
            torch.cuda.set_sync_debug_mode("warn")
            try:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter("always")
                    call(package, hidden)
            finally:
                torch.cuda.set_sync_debug_mode("default")
                (
                    torch.cuda.Event.synchronize,
                    torch.cuda.Stream.synchronize,
                    torch.cuda.synchronize,
                ) = original
            counts["debug"] = sum("synchroniz" in str(w.message) for w in rec)
            if best is None or sum(counts.values()) < sum(best.values()):
                best = counts
        return best

    for hidden in (False, True):
        nd, det = waits("nd", hidden), waits("det", hidden)
        print(f"\nsyncs T={T} {objective} hidden_count={hidden}: ND {nd} det {det}")
        assert nd == det, (T, objective, hidden, nd, det)
        if hidden and -(-T // 4096) >= cake_backend.HIDDEN_COUNT_MIN_CHUNKS:
            assert sum(nd.values()) == 1 and nd["event"] == 1, nd
            fwd = cake_backend.forward_loss(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=4096,
                hidden_count=True,
                **kw,
                **ND,
            )
            assert fwd.memory.get("hidden_count"), fwd.memory


def test_device_zero_weight_rows_concurrent_threads(glm_weight):
    """Zero-weight splits on the hidden valid-row count path (four chunks, the dW side stream) from two threads
    at once, each on its own stream, several calls each: every call gives its serial call's ``loss`` / ``logp`` /
    ``dW`` bits and zero ``dX`` rows on the zero-weight rows (each thread reads its own counts and forks its own side
    stream), and passes the gates."""
    _require_program(entry="loss")
    inputs, zeros = [], []
    for T, kind in ((12289, ("random", 0.166)), (16231, ("group", 0.5))):
        base = _device_inputs(T, "policy", W=glm_weight)
        zero = _zero_weight_set(base, kind)
        inputs.append(_with_zero_weights(base, zero))
        zeros.append(zero)
    serial = [_loss_call(inp, 4096) for inp in inputs]
    runs: list[list] = [[], []]
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(i):
        try:
            stream = torch.cuda.Stream()
            inp = inputs[i]
            with torch.cuda.stream(stream):
                barrier.wait()
                for _ in range(4):
                    X = inp.X.detach().requires_grad_(True)
                    W = inp.W.detach().requires_grad_(True)
                    loss, logp = cake_backend.chunked_lm_head_loss(
                        X,
                        W,
                        inp.labels,
                        objective="policy",
                        infer_logp=inp.infer_logp,
                        loss_weights=inp.loss_weights,
                        chunk_size=4096,
                        return_logp=True,
                        **ND,
                    )
                    loss.backward()
                    runs[i].append(
                        dict(
                            loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad
                        )
                    )
            stream.synchronize()
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()
    assert not errors, errors
    for inp, zero, got, want in zip(inputs, zeros, runs, serial, strict=True):
        assert len(got) == 4
        for run in got:
            for key in ("loss", "logp", "dW"):
                assert _same_bits(run[key], want[key]), key
            assert not bool(run["dX"][zero].any())
        _check_against_references(got[-1], inp, ceiling=True)


def test_device_zero_weight_rows_under_a_non_finite_upstream_gradient(glm_weight):
    """A non-finite upstream loss gradient leaves the zero-weight rows' ``dX`` rows exact zeros (as an ignored
    row's); the deterministic package's are ``bf16(g * 0)``."""
    _require_program(entry="loss")
    base = _device_inputs(4097, "policy", W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    out = {}
    for package in ("nd", "det"):
        X = inp.X.detach().requires_grad_(True)
        fn, mode = (
            (cake_backend.chunked_lm_head_loss, ND)
            if package == "nd"
            else (det_backend.chunked_lm_head_det_loss, {})
        )
        loss = fn(
            X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=False,
            **mode,
        )
        loss.backward(torch.tensor(float("inf"), device=CUDA))
        torch.cuda.synchronize()
        out[package] = X.grad
    nd_zero = out["nd"][zero]
    assert bool((nd_zero == 0).all())
    assert not bool(out["nd"][inp.labels < 0].any())
    det_zero = out["det"][zero].float()
    print(
        f"\nnon-finite grad_loss=inf: ND zero-weight dX rows all +0; det non-finite share "
        f"{float((~torch.isfinite(det_zero)).float().mean()):.3f}"
    )


def test_opt_in_forms_split_zero_weight_rows_unless_the_count_is_given():
    """The opt-in module's forward leaves the zero-weight rows out of its chunk loop when it forms the count itself
    (the same bits as :mod:`cake_backend`'s), and keeps every valid row with a caller-given ``num_valid_rows``."""
    base = _host_inputs(200, "policy")
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="reference",
        **ND,
    )
    main = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    opt = opt_in.forward_loss(inp.X, inp.W, inp.labels, **kw)
    valid = int((inp.labels >= 0).sum())
    assert (
        opt.memory["zero_weight_rows"]
        == int(zero.sum())
        == main.memory["zero_weight_rows"]
    )
    assert opt.row_index.numel() == valid - int(zero.sum())
    for key in ("loss", "logp", "dx_acc", "dw_acc"):
        assert _same_bits(getattr(opt, key), getattr(main, key)), key
    given = opt_in.forward_loss(inp.X, inp.W, inp.labels, num_valid_rows=valid, **kw)
    assert "zero_weight_rows" not in given.memory and given.row_index.numel() == valid
    torch.testing.assert_close(given.logp, main.logp, atol=1e-5, rtol=0)


def test_memory_plan_counts_the_zero_weight_logp_pass():
    """``memory_plan(zero_rows=...)``: the budget still counts every valid row (``budget_rows``) while the chunk loop
    plans on the gradient rows; the zero-weight rows' logp-only pass is a phase of the predicted peak.  The guard holds
    ``budget_rows <= rows + zero_total`` (``zero_total``: every zero-weight row, also when the pass does not run)."""
    kw = dict(num_rows=16384, hidden=6144, vocab=154880, chunk=4096)
    plain = opt_in.memory_plan("loss", rows=10000, **kw)
    split = opt_in.memory_plan(
        "loss", rows=6000, budget_rows=10000, zero_rows=4000, **kw
    )
    assert split.budget_bytes == plain.budget_bytes
    assert split.predicted_bytes >= 2 * 4000 * 154880
    no_pass = opt_in.memory_plan(
        "loss", rows=6000, budget_rows=10000, zero_total=4000, **kw
    )
    assert no_pass.budget_bytes == plain.budget_bytes
    with pytest.raises(ValueError, match="zero_rows"):
        opt_in.memory_plan("loss", rows=6000, budget_rows=10000, zero_rows=12000, **kw)
    # The budget rows are at most the chunk loop's rows plus every zero-weight row of the call
    with pytest.raises(ValueError, match="budget_rows"):
        opt_in.memory_plan("loss", rows=6000, budget_rows=10000, **kw)
    with pytest.raises(ValueError, match="budget_rows"):
        opt_in.memory_plan("loss", rows=10000, budget_rows=10001, **kw)
    with pytest.raises(ValueError, match="zero_total"):
        opt_in.memory_plan(
            "loss", rows=6000, budget_rows=10000, zero_rows=4000, zero_total=3000, **kw
        )
    with pytest.raises(ValueError, match="num_rows"):
        opt_in.memory_plan("loss", rows=12000, budget_rows=12000, zero_total=4385, **kw)
    assert "dx_ws_slabs" not in inspect.signature(opt_in.memory_plan).parameters


@pytest.mark.parametrize(
    "T, kind", [(4096, ("random", 0.3)), (16384, ("group", 0.5))], ids=str
)
def test_device_zero_weight_rows_budget_form(glm_weight, T, kind):
    """``memory_mode="budget"`` with zero-weight rows (a third of the labels ignored): ``loss`` / ``logp`` / ``dW``
    are the fast form's bits, ``dX`` passes the gates, and the step's peak -- the logp-only pass included, of the call
    that binds and of one from the remembered binding -- stays within the bytes of a head that materializes the BF16
    logits of the valid rows."""
    _require_program(entry="loss")
    base = make_inputs(
        T,
        objective="policy",
        seed=SEED + 7545 + T,
        device=CUDA,
        W=glm_weight,
        ignore_frac=0.33,
    )
    zero = _zero_weight_set(base, kind)
    inp = _with_zero_weights(base, zero)
    R, H, V = int(inp.valid.sum()), int(inp.H), int(inp.V)
    head = 2 * R * V + 2 * V * H + 2 * T * H
    fast = _loss_call(inp, 4096)
    peaks = []
    for _ in range(2):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        loss, logp = opt_in.chunked_lm_head_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            memory_mode="budget",
            **ND,
        )
        mplan = loss.grad_fn.memory_plan
        loss.backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - start)
        result = dict(loss=loss.detach(), logp=logp, dX=X.grad, dW=W.grad)
        for key in ("loss", "logp", "dW"):
            assert _same_bits(result[key], fast[key]), key
        assert not bool(result["dX"][zero].any())
        _check_against_references(result, inp, ceiling=True)
        del result, X, W, loss, logp
    print(
        f"\nzero-weight budget T={T} {kind}: form {mplan.form} predicted {mplan.predicted_bytes} budget {mplan.budget_bytes} "
        f"peaks {peaks} head {head}"
    )
    assert mplan.fits and max(peaks) <= head, (mplan, peaks, head)
    assert max(peaks) <= mplan.predicted_bytes, (mplan, peaks)


def test_zero_weight_split_is_not_applied_to_the_excluded_forms():
    """``compact_rows=False`` and the prepared runner keep every valid row in the chunk loop (the plain path); so does
    the cross-entropy objective, which has no weights."""
    base = _host_inputs(200, "policy")
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="reference",
        **ND,
    )
    flat = cake_backend.forward_loss(inp.X, inp.W, inp.labels, compact_rows=False, **kw)
    assert "zero_weight_rows" not in flat.memory and flat.row_index is None
    assert tuple(flat.dx_acc.shape) == (200, inp.H)
    runner = cake_backend.prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, compact_rows=True, **kw
    )
    assert runner.plan.rows == int((inp.labels >= 0).sum())
    split = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    assert split.row_index.numel() == int((inp.labels >= 0).sum()) - int(zero.sum())
    torch.testing.assert_close(split.logp, flat.logp, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Every reduction form, the FP32 accumulator's run-to-run spread, and the contract rows of the coverage matrix
# ---------------------------------------------------------------------------

RED_STAGES = (
    "gemm_dx_red",
    "gemm_dx_red_tn256",
    "gemm_dx_red_st3",
    "gemm_dx_red_split",
    "gemm_dx_red_st3_split",
    "gemm_dx_red_swab",
)


def _red_stage_pins(stage: str, k_slices: int = 4) -> dict:
    """The ``gemm_tuning`` dX knobs that select the reduction form ``stage`` at ``k_slices`` slices: its tile, ring
    depth, split tail (one half-item, valid for every chunk) and swap, every other knob pinned to the default form."""
    _, knobs = cake_backend.parse_stage(stage)
    return {
        "k_slices": k_slices,
        "tile_n": knobs["tile_n"],
        "stages": knobs["stages"],
        "split": 1 if knobs["split"] else None,
        "swap_ab": bool(knobs["swab"]),
    }


@pytest.mark.parametrize("stage", RED_STAGES)
def test_device_every_reduction_form_runs_within_the_budgets(glm_weight, stage):
    """Every ``gemm_dx_red*`` form the running architecture registers (``gemm_dx_red_swab`` on ``sm_103a`` only), pinned
    at four K slices on a call of two whole 4096-row chunks: the launch record shows it launched on both, the outputs
    pass the deterministic suite's gates against the FP64 oracle, and ``loss`` / ``logp`` / ``dW`` are the
    deterministic package's bits."""
    _require_program(entry="loss")
    module_name, _ = cake_backend.record_for(CUDA)
    if stage not in cake_jit.registered_stages(module_name):
        pytest.skip(f"{stage} is not registered for this architecture")
    inp = _device_inputs(8192, "ce", W=glm_weight, ignore_frac=0.0)
    tuning = {"dx": _red_stage_pins(stage)}
    cake_backend.BINDING_CACHE.clear()
    with _launch_record() as launched:
        nd = _run(inp, 4096, backend="cake", gemm_tuning=tuning)
    count = sum(1 for key in launched if key[0] == stage)
    assert count == 2, (stage, sorted({key[0] for key in launched}))
    det = _run(inp, 4096, backend="cake", package="det")
    errors, b0 = _check_against_references(nd, inp, ceiling=True)
    diff, flips, max_ulp = _compare_with_det(nd, det)
    print(
        f"\nND form {stage} x{count}: dX_rel_l2 {errors['dX_rel_l2']:.3e} (b0 {b0['dX_rel_l2']:.3e}) "
        f"max|ND-det| {diff:.3e} flip {flips:.2e} max_ulp {max_ulp}"
    )


@pytest.mark.parametrize("policy", ["det", "measured"])
def test_device_reduction_on_a_four_chunk_cross_entropy_call(glm_weight, policy):
    """A cross-entropy call of four compacted chunks (``T = 16231``, every row valid) under each slice policy: exactly
    the chunks the device's rule slices launch one ``gemm_dx_red*`` form each (launch record) behind one fill, and the
    call passes the gates."""
    _require_program(entry="loss")
    T = 16231
    slices = _device_slices(T, policy, glm_weight)
    sliced = [i for i, s in enumerate(slices) if s > 1]
    if not sliced:
        pytest.skip(
            f"the {policy} policy slices no chunk of a {T}-row call on this device"
        )
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
    with _slice_policy(policy), _launch_record() as launched:
        nd = _run(inp, 4096, backend="cake")
    red = [key for key in launched if key[0].startswith(cake_backend.DX_RED_STAGE)]
    assert [key[1] for key in red] == sliced, (slices, red)
    assert launched.count(("dx_fill", "dx")) == 1, launched
    _check_against_references(nd, inp, ceiling=True)
    print(
        f"\nND 4-chunk ce policy={policy} slices {slices}: red launches {[key[0] for key in red]}"
    )


def test_device_fp32_dx_acc_spread_over_repeated_runs(glm_weight):
    """The FP32 ``dX_acc`` of ``REPEATS`` forwards at one and 339 rows with three and eight K slices and at 4097
    rows with eight: every run is within the deterministic suite's ``dX`` budget against the FP64 oracle, and the number
    of distinct bit patterns and the largest run-to-run difference are reported.  Every element of ``dX_acc`` gets one
    nonzero partial per slice (``dz`` is dense over the vocabulary), so three or more slices make its sum depend on the
    order the slices finish in (two would not: ``a + b == b + a``); the eight-slice cases -- the most addition orders,
    up to 4097 x H elements -- must show more than one result together (the unordered reduction is exercised; bitwise
    repeatability is never asserted for this package)."""
    _require_program(entry="loss")
    distinct: dict = {}
    for T, slices in ((1, (3, 8)), (339, (3, 8)), (4097, (8,))):
        inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
        oracle = reference_fp64(inp, need_dw=False)
        b0 = error_report(reference_unchunked(inp, need_dw=False), oracle, inp.labels)
        for k in slices:
            runs = []
            for _ in range(REPEATS):
                fwd = cake_backend.forward_loss(
                    inp.X,
                    inp.W,
                    inp.labels,
                    loss_div=inp.loss_div,
                    chunk_size=4096,
                    need_dw=False,
                    gemm_tuning={"dx": {"k_slices": k}},
                    **ND,
                )
                ref = (
                    oracle["dX"]
                    if fwd.row_index is None
                    else oracle["dX"].index_select(
                        0, fwd.row_index.to(oracle["dX"].device)
                    )
                )
                assert tuple(fwd.dx_acc.shape) == (T, inp.H)  # every row valid
                err = rel_l2(fwd.dx_acc, ref)  # dX = bf16(g * dx_acc), g = 1
                assert err <= GATE_MARGIN * b0["dX_rel_l2"] + GATE_TINY["dX_rel_l2"], (
                    T,
                    k,
                    err,
                )
                runs.append(fwd.dx_acc.clone())
            unique: list = []
            for run in runs:
                if not any(_same_bits(run, seen) for seen in unique):
                    unique.append(run)
            spread = max(float((run - runs[0]).abs().max()) for run in runs)
            changed = (
                int((_bits(unique[1]) != _bits(unique[0])).sum())
                if len(unique) > 1
                else 0
            )
            distinct[(T, k)] = len(unique)
            print(
                f"\nND FP32 dX_acc T={T} S{k}: {len(unique)} distinct of {REPEATS}, run-to-run max {spread:.3e}, "
                f"{changed} of {runs[0].numel()} elements differ between the first two patterns"
            )
            del runs, unique
    assert max(n for (T, k), n in distinct.items() if k == 8) > 1, distinct


def _budget_step(inp, entry, **kw):
    """One training step of this package's opt-in autograd entry with its peak allocated bytes above the bytes before
    it (outputs and gradients count): ``(result, peak, memory_plan)`` (``memory_plan`` of a loss call's opt-in form)."""
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    mplan = None
    if entry == "loss":
        loss, logp = opt_in.chunked_lm_head_loss(
            X,
            W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=4096,
            return_logp=True,
            **kw,
            **ND,
        )
        mplan = getattr(loss.grad_fn, "memory_plan", None)
        loss.backward()
        loss = loss.detach()
    else:
        logp = opt_in.chunked_lm_head_logprob(
            X, W, inp.labels, chunk_size=4096, **kw, **ND
        )
        logp.backward(inp.dlogp)
        loss = None
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return dict(loss=loss, logp=logp.detach(), dX=X.grad, dW=W.grad), peak, mplan


@pytest.mark.parametrize("entry", ["loss", "logprob"])
@pytest.mark.parametrize("T", [4096, 16384])
def test_device_budget_step_peak_within_the_prediction(glm_weight, entry, T):
    """With the dX slabs gone, the budget step of the cross-entropy loss and of the log-probability entry (a
    third of the labels ignored), of the call that binds and of one from the remembered binding, peaks within the
    bytes of a head that materializes the BF16 logits of the valid rows and -- the loss entry's recorded plan -- within
    the planner's prediction; ``loss`` / ``logp`` / ``dW`` are the fast step's bits and ``dX`` passes the gates."""
    _require_program(entry=entry)
    inp = make_inputs(
        T, seed=SEED + 7545 + T, device=CUDA, W=glm_weight, ignore_frac=0.33
    )
    R, H, V = int(inp.valid.sum()), int(inp.H), int(inp.V)
    head = 2 * R * V + 2 * V * H + 2 * T * H
    fast, _, _ = _budget_step(inp, entry)
    peaks, mplan = [], None
    for _ in range(2):
        result, peak, mplan = _budget_step(inp, entry, memory_mode="budget")
        for key in ("loss", "logp", "dW") if entry == "loss" else ("logp", "dW"):
            assert _same_bits(result[key], fast[key]), key
        _check_against_references(result, inp, entry=entry, ceiling=True)
        peaks.append(peak)
        del result
    print(
        f"\nbudget {entry} T={T}: peaks {peaks} head {head} predicted "
        f"{None if mplan is None else mplan.predicted_bytes}"
    )
    assert max(peaks) <= head, (peaks, head)
    if entry == "loss":
        assert mplan is not None and mplan.fits
        assert max(peaks) <= mplan.predicted_bytes, (peaks, mplan)


def test_device_checkpointed_forward_keeps_no_more_than_the_deterministic_package(
    glm_weight,
):
    """After a non-reentrant checkpointed forward (https://github.com/flashinfer-ai/flashinfer/issues/7039) the bytes
    the graph keeps alive, and after its backward (graph released) the bytes left beyond the gradients, are no more
    than the deterministic package's for the same call, whatever each package forms in that forward (the FP32 gradient
    accumulators, kept until the backward, or the loss only); the outputs keep the deterministic package's ``loss`` /
    ``dW`` bits.  The bytes are the ones the live tensors requested from the caching allocator, not its allocated
    blocks: a block can exceed its request by up to 1 MiB, by an amount set by the free blocks the process's earlier
    tests left, so the allocated bytes of the same call differ with the test order."""
    _require_program(entry="loss")
    inp = _device_inputs(8191, "ce", W=glm_weight)

    def requested():
        return torch.cuda.memory_stats()["requested_bytes.all.current"]

    def retained(package):
        fn, mode = (
            (cake_backend.chunked_lm_head_loss, ND)
            if package == "nd"
            else (det_backend.chunked_lm_head_det_loss, {})
        )

        def head(X, W):
            return fn(X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, **mode)

        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        head(X, W).backward()  # binds
        X.grad = W.grad = None
        gc.collect()
        torch.cuda.synchronize()
        before = requested()
        loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
        torch.cuda.synchronize()
        kept = requested() - before
        loss.backward()
        out = dict(loss=loss.detach(), dW=W.grad, dX=X.grad)
        del loss
        gc.collect()
        torch.cuda.synchronize()
        grads = sum(t.numel() * t.element_size() for t in (X.grad, W.grad))
        after = requested() - before - grads
        return kept, after, out

    nd_kept, nd_after, nd = retained("nd")
    det_kept, det_after, det = retained("det")
    print(
        f"\ncheckpoint T=8191 (requested bytes): kept after forward ND {nd_kept} det {det_kept}; "
        f"after backward ND {nd_after} det {det_after}"
    )
    assert nd_kept <= det_kept + (1 << 20), (nd_kept, det_kept)
    assert nd_after <= max(det_after, 0) + (1 << 20), (nd_after, det_after)
    assert _same_bits(nd["loss"], det["loss"]) and _same_bits(nd["dW"], det["dW"])


@pytest.mark.parametrize("fuse", [True, False])
def test_device_fp32_weight_gradient(glm_weight, fuse):
    """FP32 ``dW``: ``forward_loss`` / ``backward_loss`` with ``grad_weight_dtype=torch.float32``
    (fused and unfused last-chunk cast) give an FP32 ``dW`` with the deterministic package's bits, within the gates."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "ce", W=glm_weight)
    kw = dict(
        loss_div=inp.loss_div,
        chunk_size=4096,
        grad_weight_dtype=torch.float32,
        fuse_dw_cast=fuse,
    )
    nd_fwd = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw, **ND)
    dx, dw = nd_fwd.backward(grad_weight_dtype=torch.float32)
    det_fwd = det_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    det_dx, det_dw = det_fwd.backward(grad_weight_dtype=torch.float32)
    assert dw.dtype == torch.float32 and _same_bits(dw, det_dw)
    assert _same_bits(nd_fwd.loss, det_fwd.loss) and _same_bits(
        nd_fwd.logp, det_fwd.logp
    )
    _check_against_references(
        dict(loss=nd_fwd.loss, logp=nd_fwd.logp, dX=dx, dW=dw), inp, ceiling=True
    )


def test_device_policy_rows_at_the_clip_knee(glm_weight):
    """The policy objective at ``ratio = 2``: rows whose inference log-probability puts the ratio at the
    clip knee and one FP32 step on either side: ``loss`` / ``logp`` / ``dW`` are the deterministic package's bits
    (the same row kernel decides the same side), the rows the deterministic package gives zero ``dX`` get zero ``dX``,
    and ``dX`` is the deterministic package's within the dX budget's floor (an FP64 oracle decides a knee row's side in
    FP64, so it is no reference at the knee)."""
    _require_program(entry="loss")
    base = _device_inputs(4097, "policy", W=glm_weight)
    logp = det_backend.forward_logprob(
        base.X, base.W, base.labels, chunk_size=4096
    ).logp
    rows = (base.labels >= 0).nonzero().squeeze(1)
    knee = logp - math.log(2.0)
    infer = base.infer_logp.clone()
    third = rows.numel() // 3
    infer[rows[:third]] = knee[rows[:third]]
    infer[rows[third : 2 * third]] = torch.nextafter(
        knee[rows[third : 2 * third]],
        torch.full_like(knee[rows[third : 2 * third]], -1e30),
    )
    infer[rows[2 * third :]] = torch.nextafter(
        knee[rows[2 * third :]], torch.full_like(knee[rows[2 * third :]], 1e30)
    )
    inp = dataclasses.replace(base, infer_logp=infer.contiguous())
    nd = _loss_call(inp, 4096)
    det = _loss_call(inp, 4096, package="det")
    for key in ("loss", "logp", "dW"):
        assert _same_bits(nd[key], det[key]), key
    clipped = (det["dX"] == 0).all(dim=1) & (inp.labels >= 0)
    assert bool((nd["dX"][clipped] == 0).all())
    err = rel_l2(nd["dX"], det["dX"].double())
    print(
        f"\nknee rows: {int(clipped.sum())} zero-dX of {rows.numel()}; dX rel-L2 vs det {err:.3e}"
    )
    assert err <= GATE_TINY["dX_rel_l2"], err


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_every_row_ignored_and_no_rows(glm_weight, entry):
    """Every label ``-100`` gives loss 0, ``logp`` 0 and zero gradients (the deterministic package's
    bits); ``T = 0`` gives loss 0, an empty ``logp`` and zero gradients."""
    _require_program(entry=entry)
    inp = _device_inputs(37, "ce", W=glm_weight, ignore_frac=1.0)
    assert not bool((inp.labels >= 0).any())
    nd = _run(inp, 4096, entry=entry, backend="cake")
    det = _run(inp, 4096, entry=entry, backend="cake", package="det")
    for key in ("logp", "dX", "dW") + (("loss",) if entry == "loss" else ()):
        assert _same_bits(nd[key], det[key]), key
        assert not bool(nd[key].any()), key
    empty = dataclasses.replace(
        inp,
        X=inp.X[:0],
        labels=inp.labels[:0],
        dlogp=inp.dlogp[:0],
        regime=None if inp.regime is None else inp.regime[:0],
    )
    nd0 = _run(empty, 4096, entry=entry, backend="cake")
    assert nd0["logp"].numel() == 0 and float(nd0["loss"]) == 0.0
    assert tuple(nd0["dX"].shape) == (0, inp.H) and not bool(nd0["dW"].any())


def test_device_upstream_gradient_scale(glm_weight):
    """A loss upstream gradient other than 1 (``loss * 3``) scales ``dX`` / ``dW`` through the same
    single cast: ``loss`` / ``logp`` / ``dW`` are the deterministic package's bits and ``dX`` passes the gates against
    three times the oracle's."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "ce", W=glm_weight)
    nd = _run(inp, 4096, backend="cake", scale=3.0)
    det = _run(inp, 4096, backend="cake", package="det", scale=3.0)
    _compare_with_det(nd, det)
    oracle = reference_fp64(inp)
    b0 = error_report(reference_unchunked(inp), oracle, inp.labels)
    for name in ("dX", "dW"):
        err = rel_l2(nd[name], 3.0 * oracle[name])
        assert (
            err <= GATE_MARGIN * b0[f"{name}_rel_l2"] + GATE_TINY[f"{name}_rel_l2"]
        ), (
            name,
            err,
        )


def test_device_uncompacted_path(glm_weight):
    """``compact_rows=False``: the chunk loop over every row gives the deterministic package's
    uncompacted ``loss`` / ``logp`` / ``dW`` bits, ``dX`` within the gates, and the compacted call's ``logp`` bits."""
    _require_program(entry="loss")
    inp = _device_inputs(8191, "ce", W=glm_weight)
    nd = _run(inp, 4096, backend="cake", compact_rows=False)
    det = _run(inp, 4096, backend="cake", package="det", compact_rows=False)
    _check_against_references(nd, inp, ceiling=True)
    _compare_with_det(nd, det)
    assert _same_bits(nd["logp"], _run(inp, 4096, backend="cake")["logp"])


@pytest.mark.parametrize("ld_pad", [256, 1])
def test_device_strided_x(glm_weight, ld_pad):
    """Strided ``X``: the ``[:, :H]`` view of a ``[T, H + ld_pad]`` buffer -- read in place at a 16-byte
    row pitch (``ld_pad`` 256), copied to a contiguous tensor otherwise (``ld_pad`` 1) -- gives the deterministic
    package's ``loss`` / ``logp`` / ``dW`` bits and ``dX`` within the gates."""
    _require_program(entry="loss")
    inp = make_inputs(4097, seed=SEED + 4097, device=CUDA, W=glm_weight, ld_pad=ld_pad)
    assert inp.X.stride(0) == inp.H + ld_pad
    nd = _run(inp, 4096, backend="cake")
    det = _run(inp, 4096, backend="cake", package="det")
    _check_against_references(nd, inp, ceiling=True)
    _compare_with_det(nd, det)


def test_device_wrong_valid_row_count_raises_on_every_path(glm_weight):
    """A wrong caller ``num_valid_rows`` with ``check_num_valid_rows=True`` raises ``ValueError`` on
    every path of this package: the compacted loss and log-probability entries, their budget forms, the uncompacted
    calls and the calls without any gradient (frozen ``X`` and ``W``)."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "ce", W=glm_weight)
    wrong = int(inp.valid.sum()) + 1
    X = inp.X.detach().requires_grad_(True)
    frozen = inp.X.detach()
    common = dict(chunk_size=4096, num_valid_rows=wrong, check_num_valid_rows=True)
    calls = {
        "loss-no-grad": lambda: opt_in.chunked_lm_head_loss(
            frozen, inp.W, inp.labels, loss_div=inp.loss_div, **common, **ND
        ),
        "logprob-no-grad": lambda: opt_in.chunked_lm_head_logprob(
            frozen, inp.W, inp.labels, **common, **ND
        ),
        "logprob-budget": lambda: opt_in.chunked_lm_head_logprob(
            X, inp.W, inp.labels, memory_mode="budget", **common, **ND
        ),
        "loss": lambda: opt_in.chunked_lm_head_loss(
            X, inp.W, inp.labels, loss_div=inp.loss_div, **common, **ND
        ),
        "loss-budget": lambda: opt_in.chunked_lm_head_loss(
            X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            memory_mode="budget",
            **common,
            **ND,
        ),
        "loss-uncompacted": lambda: opt_in.chunked_lm_head_loss(
            X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            compact_rows=False,
            **common,
            **ND,
        ),
        "logprob": lambda: opt_in.chunked_lm_head_logprob(
            X, inp.W, inp.labels, **common, **ND
        ),
        "logprob-uncompacted": lambda: opt_in.chunked_lm_head_logprob(
            X, inp.W, inp.labels, compact_rows=False, **common, **ND
        ),
    }
    for name, call in calls.items():
        try:
            call()
        except ValueError as exc:
            assert "num_valid_rows" in str(exc), (name, exc)
        else:
            pytest.fail(f"{name}: a wrong num_valid_rows did not raise")
        torch.cuda.synchronize()


def test_device_dw_stream_modes(glm_weight):
    """``dw_stream``: off, on and ``"auto"`` on a four-chunk call give the same ``loss`` / ``logp`` /
    ``dW`` bits (the deterministic package's), and ``dX`` within the gates after a 0xFF-poisoned allocator: the
    ``-0.0`` fill stays ordered before the reductions when the weight-gradient GEMMs run on the side stream."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    oracle = reference_fp64(inp)
    b0 = error_report(reference_unchunked(inp), oracle, inp.labels)
    det = _run(inp, 4096, backend="cake", package="det")
    for mode in (False, True, "auto"):
        _poison_allocator(1 << 30)
        nd = _run(inp, 4096, backend="cake", dw_stream=mode)
        _compare_with_det(nd, det)
        errors = error_report(nd, oracle, inp.labels)
        assert not errors["nan"] and errors["dX_ignored_zero"]
        for key, tiny in GATE_TINY.items():
            if key in errors:
                assert errors[key] <= GATE_MARGIN * b0[key] + tiny, (mode, key)


def test_device_fresh_thread_and_preload(glm_weight):
    """A call from a thread that never called the package before forms that thread's state (side
    streams, count cells) and gives the main thread's ``loss`` / ``logp`` / ``dW`` bits; ``chunked_lm_head_preload(
    deterministic=False)`` forms the calling thread's state ahead of its first call."""
    _require_program(entry="loss")
    import flashinfer.chunked_lm_head as public

    inp = _device_inputs(12289, "ce", W=glm_weight)
    main = _run(inp, 4096, backend="cake")
    out: dict = {}

    def fresh():
        try:
            out["state_before"] = (
                getattr(cake_backend._THREAD_STATE, "dw_streams", None),
                getattr(cake_backend._THREAD_STATE, "fill_streams", None),
            )
            out["result"] = _run(inp, 4096, backend="cake")
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            out["error"] = exc

    def preloaded():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                public.chunked_lm_head_preload(CUDA, deterministic=False)
            index = torch.cuda.current_device()
            out["preloaded"] = (
                index in cake_backend._thread_state("dw_streams"),
                index in cake_backend._thread_state("fill_streams"),
                index in cake_backend._thread_state("cells"),
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            out["error"] = exc

    for target in (fresh, preloaded):
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()
        if "error" in out:
            raise out["error"]
    assert out["state_before"] == (None, None)
    for key in ("loss", "logp", "dW"):
        assert _same_bits(out["result"][key], main[key]), key
    _check_against_references(out["result"], inp, ceiling=True)
    assert out["preloaded"] == (True, True, True)


# about half a second of the fill stream at B200 / GB300 clocks: a side fill lands long after any dX GEMM of the call
# that did not wait for it
FILL_DELAY_CYCLES = 1 << 30


@contextlib.contextmanager
def _side_fill_probe(*, drop_join: bool = False):
    """The side-stream ``dx_fill`` made late and observable.  The pool of every thread's fill stream (``dX_acc`` comes
    from it; autograd runs a backward on its own device thread, which has its own fill stream) is 0xFF-poisoned when
    the probe first sees the stream, the fill stream spins ``FILL_DELAY_CYCLES`` in front of every side fill, and the
    yielded record holds, in host order, ``"fill"`` per side fill, ``"join"`` per wait of the caller's stream on a
    fill's join event and every launch key the call ran.  A call whose dX GEMMs do not wait for the join adds into the
    poison (or has its sums overwritten by the late fill) and fails the gates; ``drop_join=True`` skips those waits
    (the negative control)."""
    record: list = []
    seen: list = []  # the (fill stream, join event) pairs handed out, each poisoned once
    fill, wait_event = cake_backend._dx_fill, torch.cuda.Stream.wait_event
    fill_stream = cake_backend._fill_stream
    run_keys = (cake_backend._run_keys, opt_in._run_keys)

    def poisoned_stream(index):
        pair = fill_stream(index)
        if not any(pair is other for other in seen):
            seen.append(pair)
            with torch.cuda.stream(pair[0]):
                _poison_allocator(1 << 30)
        return pair

    # The side fill only: the dx_fill launch key runs HOST_STAGES' own reference.
    def late_fill(values):
        torch.cuda._sleep(FILL_DELAY_CYCLES)
        record.append("fill")
        fill(values)

    def waiting(self, event):
        if any(event is pair[1] for pair in seen):
            record.append("join")
            if drop_join:
                return None
        return wait_event(self, event)

    def recording(original):
        def run_keys_(plan, device_index, keys, run_key, *args, **kwargs):
            def run(key):
                record.append(key)
                return run_key(key)

            return original(plan, device_index, keys, run, *args, **kwargs)

        return run_keys_

    # the calling thread's fill stream, before the first call
    poisoned_stream(torch.cuda.current_device())
    cake_backend._dx_fill, cake_backend._fill_stream = late_fill, poisoned_stream
    torch.cuda.Stream.wait_event = waiting
    cake_backend._run_keys, opt_in._run_keys = (recording(f) for f in run_keys)
    try:
        yield record
    finally:
        torch.cuda.synchronize()
        cake_backend._dx_fill, cake_backend._fill_stream = fill, fill_stream
        torch.cuda.Stream.wait_event = wait_event
        cake_backend._run_keys, opt_in._run_keys = run_keys


def _joined_before_dx(record: list) -> None:
    """One side fill and one join in the call, the join after the fill and in front of the call's first ``gemm_dx*``
    key."""
    first_dx = next(
        i
        for i, e in enumerate(record)
        if isinstance(e, tuple) and e[0].startswith("gemm_dx")
    )
    assert record.count("fill") == 1 and record.count("join") == 1, record
    assert record.index("fill") < record.index("join") < first_dx, record


@pytest.mark.parametrize(
    "entry, hidden", [("loss", True), ("loss", False), ("logprob", None)]
)
def test_device_side_stream_fill(glm_weight, entry, hidden):
    """The side-stream ``dx_fill`` (``set_dx_fill_side_stream(True)``, off by default) under :func:`_side_fill_probe`
    (poisoned fill pool, a fill stream half a second late): on the hidden and the host valid-row count paths of the
    loss entry and in the autograd log-probability backward, the call from the remembered binding fills on the fill
    stream, the caller's stream waits for the join in front of the first dX GEMM, and every output matches the
    caller-stream fill's (``loss`` / ``logp`` / ``dW`` bitwise, ``dX`` within the gates).  The same call with the
    join wait skipped fails the gates (negative control)."""
    T = 12289
    _require_program(entry=entry)
    inp = _device_inputs(T, "ce", W=glm_weight)
    switches = {} if hidden is None else dict(hidden_count=hidden)
    plain = _run(inp, 4096, entry=entry, backend="cake", **switches)
    previous = cake_backend.set_dx_fill_side_stream(True)
    try:
        side = None
        with _side_fill_probe() as record:
            # the call that binds fills on the caller's stream; the next one runs from the remembered binding
            for _ in range(3):
                record.clear()
                _poison_allocator(1 << 30)
                result = _run(inp, 4096, entry=entry, backend="cake", **switches)
                if "fill" in record:
                    side = result
                    break
        assert side is not None, "no call filled on the fill stream"
        _joined_before_dx(record)
        keys = ("loss", "logp", "dW") if entry == "loss" else ("logp", "dW")
        for key in keys:
            assert _same_bits(side[key], plain[key]), key
        _check_against_references(side, inp, entry=entry, ceiling=True)
        with _side_fill_probe(drop_join=True) as record:
            _poison_allocator(1 << 30)
            broken = _run(inp, 4096, entry=entry, backend="cake", **switches)
        assert "fill" in record and "join" in record, record
        with pytest.raises(AssertionError):
            _check_against_references(broken, inp, entry=entry, ceiling=True)
    finally:
        cake_backend.set_dx_fill_side_stream(previous)


def test_device_side_stream_fill_in_the_log_probability_backward(glm_weight):
    """The side-stream ``dx_fill`` in ``backward_logprob`` itself under :func:`_side_fill_probe`: the second backward
    call on one forward runs from the remembered binding, fills on the fill stream, joins in front of its first dX
    GEMM and gives the caller-stream fill's ``dW`` bits and ``dX`` within the gates."""
    _require_program(entry="logprob")
    inp = _device_inputs(8191, "ce", W=glm_weight)
    fwd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096, **ND)

    def backward():
        return cake_backend.backward_logprob(
            inp.X,
            inp.W,
            inp.labels,
            fwd.lse,
            inp.dlogp,
            chunk_size=4096,
            row_index=fwd.row_index,
            **ND,
        )

    _, plain_dw = backward()
    previous = cake_backend.set_dx_fill_side_stream(True)
    try:
        with _side_fill_probe() as record:
            _poison_allocator(1 << 30)
            dx, dw = backward()
    finally:
        cake_backend.set_dx_fill_side_stream(previous)
    _joined_before_dx(record)
    assert _same_bits(dw, plain_dw)
    _check_against_references(
        dict(logp=fwd.logp, dX=dx, dW=dw), inp, entry="logprob", ceiling=True
    )


def _forget_device_constants():
    """Drop the device's constant cells: the next call forms them."""
    cake_backend._device_constants.cache_clear()
    cake_backend._DEVICE_CELLS.clear()


def _poison_free_small_blocks():
    """Leave NaN bytes at the start of every free small block of the current stream (the caching allocator's pool of
    allocations up to 1 MiB): 512-byte NaN cells are allocated until the pool reserves a new segment, then freed --
    a small cell allocated next and read before its fill reads NaN."""
    torch.cuda.synchronize()
    key = "reserved_bytes.small_pool.current"
    reserved = torch.cuda.memory_stats()[key]
    junk = []
    while torch.cuda.memory_stats()[key] == reserved:
        junk.append(torch.full((128,), float("nan"), device=CUDA))
    torch.cuda.current_stream().synchronize()
    del junk


def test_device_constant_cells_are_filled_before_another_stream_reads_them():
    """A device's constant cells are formed once and shared by every stream: a read on another stream right after
    they are formed sees their values, never the bytes before their fills -- with a long kernel queued ahead on the
    stream that formed them (whose wait covers the fills only, not that kernel) and PyTorch's synchronization debug
    mode at ``"error"`` (the wait is no synchronization it flags; the mode is not changed).  The read is a copy into
    a buffer allocated ahead: no allocation or first kernel launch on the reading stream that could wait for the
    device."""
    _require_program(entry="loss")
    index = torch.cuda.current_device()
    _forget_device_constants()
    _poison_free_small_blocks()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        got = torch.full((18,), -1.0, device=CUDA)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda._sleep(2_000_000_000)  # about a second of the forming stream
    torch.cuda.set_sync_debug_mode("error")
    try:
        cells = cake_backend._device_constants(index)
        busy = not torch.cuda.current_stream().query()
        with torch.cuda.stream(side):
            got[0:1].copy_(cells["grad_scale"])
            got[1:2].copy_(cells["unit_scale"])
            got[2:].copy_(cells["f32_dummy"])
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert busy, "the cells waited for the forming stream's queued kernel"
    assert torch.cuda.get_sync_debug_mode() == previous
    assert got.tolist() == [1.0, 1.0] + [0.0] * 16, got.tolist()
    assert cake_backend._device_constants(index) is cells


def test_device_constant_cells_of_two_threads_first_calls_are_one(monkeypatch):
    """The first calls of two threads, each on its own stream, that both miss the device's constant cells (each
    held in its forming until the other arrives) return the same cells, the ones every later call returns: no thread
    binds cells that are freed when its call returns while its stream's kernels still read them."""
    _require_program(entry="loss")
    index = torch.cuda.current_device()
    zeros = torch.zeros
    barrier = threading.Barrier(2)

    def zeros_when_both_form(*args, **kwargs):
        # the f32_dummy cell, formed in both threads' first calls
        if args and args[0] == (16,):
            barrier.wait(TIMEOUT_S)
        return zeros(*args, **kwargs)

    streams = [torch.cuda.Stream() for _ in range(2)]
    results: list = [None, None]
    errors: list = []

    def worker(i):
        try:
            with torch.cuda.stream(streams[i]):
                results[i] = cake_backend._device_constants(index)
        except BaseException as exc:  # noqa: BLE001 - re-raised by the test thread
            errors.append(exc)

    torch.cuda.synchronize()
    _forget_device_constants()
    monkeypatch.setattr(torch, "zeros", zeros_when_both_form)
    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(TIMEOUT_S)
        alive = any(thread.is_alive() for thread in threads)
    finally:
        monkeypatch.setattr(torch, "zeros", zeros)
    torch.cuda.synchronize()
    assert not alive, "a thread did not finish"
    assert not errors, errors
    cells = cake_backend._device_constants(index)
    assert results[0] is cells and results[1] is cells, "threads got different cells"
    assert cells["grad_scale"].tolist() == [1.0]
    assert cells["unit_scale"].tolist() == [1.0]
    assert cells["f32_dummy"].tolist() == [0.0] * 16


def test_device_constant_cells_wait_for_their_own_fills_only():
    """The constant cells are filled on a private stream and their first use waits for those
    fills only -- with a long kernel queued on the caller's stream and PyTorch's synchronization debug mode at
    ``"error"``, creating them neither raises nor waits for the caller's kernel -- and they hold their values.  A
    side stream's first fills are run once before the timed check: as a process's first CUDA work they load their
    kernels and the stream pool, which can outlast the caller's kernel."""
    _require_program(entry="loss")
    _forget_device_constants()
    index = torch.cuda.current_device()
    with torch.cuda.stream(torch.cuda.Stream(index)):
        cake_backend._constant_cells(torch.device("cuda", index))
    torch.cuda.synchronize()
    torch.cuda._sleep(1 << 31)  # about a second of the caller's stream
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cells = cake_backend._device_constants(index)
        busy = not torch.cuda.current_stream().query()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert busy, "the cells waited for the caller's stream"
    assert torch.cuda.get_sync_debug_mode() == previous
    torch.cuda.synchronize()
    assert cells["grad_scale"].tolist() == [1.0]
    assert cells["unit_scale"].tolist() == [1.0]
    assert not bool(cells["f32_dummy"].any())
    assert cake_backend._device_constants(index) is cells


# A process's first call, run by a fresh interpreter: argv = repository root, T, seed.
_FIRST_CALL_OF_A_PROCESS = """
import sys

import torch

sys.path.insert(0, sys.argv[1])
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs  # noqa: E402
from tests.test_helpers.cake_package import cake_package  # noqa: E402

backend = cake_package("cake.cake_lm_head_loss", "cake_backend")
modes = []
set_mode = torch.cuda.set_sync_debug_mode
torch.cuda.set_sync_debug_mode = lambda mode: (modes.append(mode), set_mode(mode))[1]
inp = make_inputs(
    int(sys.argv[2]), seed=int(sys.argv[3]), device=torch.device("cuda"), ignore_frac=0.0
)
X = inp.X.detach().requires_grad_(True)
W = inp.W.detach().requires_grad_(True)
assert backend._device_constants.cache_info().currsize == 0
loss, _ = backend.chunked_lm_head_loss(
    X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, return_logp=True, deterministic=False
)
loss.backward()
torch.cuda.synchronize()
assert backend._device_constants.cache_info().currsize == 1
assert modes == [] and torch.cuda.get_sync_debug_mode() == 0, modes
print("FIRST_CALL_OK", float(loss.detach()))
"""


def test_device_first_call_of_a_process_leaves_the_sync_debug_mode_alone():
    """A process that never set PyTorch's synchronization debug mode: its first call -- a fresh interpreter run with
    ``-W error::UserWarning`` -- forms the constant cells without setting the mode.  Setting it, even to its default,
    emits PyTorch's one-time prototype-feature warning, which would raise there.  This process's cached blocks are
    released first: earlier tests can leave its caching allocator holding nearly the whole device, and the fresh
    interpreter needs memory for its own context."""
    _require_program(entry="loss")
    root = Path(__file__).resolve().parents[2]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::UserWarning",
            "-c",
            _FIRST_CALL_OF_A_PROCESS,
            str(root),
            "257",
            str(SEED),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0 and "FIRST_CALL_OK" in proc.stdout, (
        proc.returncode,
        proc.stdout[-2000:],
        proc.stderr[-4000:],
    )


@pytest.mark.parametrize("mode", ["warn", "error"])
def test_device_first_calls_of_two_threads_keep_the_sync_debug_mode(glm_weight, mode):
    """With a synchronization debug mode set, the constant cells' one wait -- an event recorded behind their fills --
    neither triggers nor changes the mode and takes no lock.  The first calls of two threads race into the cells'
    creation (cleared cache; every event wait held open for 0.5 s): their waits overlap, each sees ``mode``, the
    package never sets the mode, the calls raise nothing and give the serial call's ``loss`` / ``logp`` / ``dW`` bits
    (``dX`` of both within the gates), and the mode is ``mode`` after both.  ``T`` = 12289 (``ceil(T / 4096)`` = 4, at
    least :data:`cake_backend.HIDDEN_COUNT_MIN_CHUNKS`): the call takes the hidden valid-row count, whose wait is an
    event wait too."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    # binds: the threads' calls are binding-cache hits
    serial = _run(inp, 4096, backend="cake")
    original = torch.cuda.Event.synchronize
    guard = threading.Lock()
    inside, entered, seen, modes = [0], [], [], []

    def held_open(self):
        with guard:
            inside[0] += 1
            entered.append(inside[0])
        seen.append(torch.cuda.get_sync_debug_mode())
        time.sleep(0.5)
        try:
            return original(self)
        finally:
            with guard:
                inside[0] -= 1

    results: list = [None, None]
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker(i):
        try:
            barrier.wait(TIMEOUT_S)
            X = inp.X.detach().requires_grad_(True)
            W = inp.W.detach().requires_grad_(True)
            loss, logp = cake_backend.chunked_lm_head_loss(
                X,
                W,
                inp.labels,
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
                **ND,
            )
            loss.backward()
            results[i] = dict(
                loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the main thread
            errors.append(exc)

    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    _forget_device_constants()
    torch.cuda.set_sync_debug_mode(mode)
    set_mode = torch.cuda.set_sync_debug_mode
    torch.cuda.set_sync_debug_mode = lambda m: (modes.append(m), set_mode(m))[1]
    torch.cuda.Event.synchronize = held_open
    try:
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(TIMEOUT_S)
        alive = any(thread.is_alive() for thread in threads)
        after = torch.cuda.get_sync_debug_mode()
    finally:
        torch.cuda.Event.synchronize = original
        torch.cuda.set_sync_debug_mode = set_mode
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    level = {"warn": 1, "error": 2}[mode]
    assert not alive, "a thread did not finish"
    assert not errors, errors
    assert modes == [], modes
    assert max(entered) >= 2, entered
    assert seen and set(seen) == {level}, seen
    assert after == level, after
    for got in results:
        for key in ("loss", "logp", "dW"):
            assert _same_bits(got[key], serial[key]), key
        _check_against_references(got, inp, ceiling=True)


# Cycles of the delay that holds both backwards of the two-stream tests, and of the shared-buffer control's delay
# behind the first fill (about a second at the B200 boost clock)
GATE_CYCLES = 1 << 31
# Runs of the two-stream sequence a test makes until one overlaps (or, in the control, races); with none it skips
TWO_STREAM_ATTEMPTS = 4


def _same_inputs_on_two_streams(
    inp, entry: str, *, shared_dx_acc: bool = False, tuning=None
):
    """A serial call of ``entry`` that binds, then two calls with the same ``X`` / ``W`` / ``labels`` (and the same
    ``gemm_tuning`` ``tuning``) on two streams of this thread: the forwards on the two streams, then both backwards --
    the autograd engine runs each on its forward's stream -- queued behind one delay on a third stream.  A timing event
    after every remembered launch records when each call's launches complete (per call and phase).  ``shared_dx_acc``
    (a control, test-only patch): both calls get one FP32 ``dX_acc`` -- each still fills its rows with ``-0.0`` first
    -- as they would share a buffer the binding owned; behind its fill each call's stream waits for one delay of
    :data:`GATE_CYCLES` on a fourth stream that starts behind the first call's fill, and a timing event behind each
    fill and one at the delay's end record when they complete (``fill_ends``, ``hold_end``).  Host clock readings give
    ``host_gap`` (ms from queuing the first call to queuing the second), ``host_bwd`` (ms from queuing the backwards'
    delay to both backwards queued) and, with ``shared_dx_acc``, ``fill_gap`` (ms between queuing the two fills)."""
    tune = {} if tuning is None else dict(gemm_tuning=tuning)
    serial = _run(inp, 4096, entry=entry, backend="cake", **tune)  # binds here
    cache = cake_backend.BINDING_CACHE
    bound = len(cache)
    assert bound > 0 and cache.owned_bytes == 0
    streams = [torch.cuda.Stream() for _ in range(2)]
    handles = [stream.cuda_stream for stream in streams]
    gate = torch.cuda.Stream()
    hold_stream = torch.cuda.Stream()
    done: dict = {(i, phase): [] for i in (0, 1) for phase in ("fwd", "bwd")}
    phase = ["fwd"]
    prepared: list = []
    fills: list = []
    filled: list = []
    hold: list = []
    starts: list = []  # host clock (s) as each call is queued
    fill_host: list = []  # host clock (s) as each fill is queued
    run_keys, prepare = cake_backend._run_keys, cake_backend.prepare_lm_head_loss
    filled_dx_acc = cake_backend._filled_dx_acc
    one: dict = {}

    def recording(plan, device_index, keys, run_key, *args, **kwargs):
        events = done[
            (handles.index(torch.cuda.current_stream().cuda_stream), phase[0])
        ]

        def run(key):
            run_key(key)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append(event)

        return run_keys(plan, device_index, keys, run, *args, **kwargs)

    def counting(*a, **k):
        prepared.append(k.get("entry"))
        return prepare(*a, **k)

    def shared(rows, hidden, fill_rows, device):
        fills.append(fill_rows)
        fill_host.append(time.perf_counter())
        acc = one.get((rows, hidden))
        if acc is None:
            acc = one[(rows, hidden)] = filled_dx_acc(rows, hidden, fill_rows, device)
        elif fill_rows is not None:
            acc[fill_rows[0] : fill_rows[1]].fill_(cake_backend.DX_FILL_VALUE)
        current = torch.cuda.current_stream()
        filled.append(torch.cuda.Event(enable_timing=True))
        filled[-1].record()
        if not hold:  # the delay starts behind this first fill
            hold_stream.wait_stream(current)
            with torch.cuda.stream(hold_stream):
                torch.cuda._sleep(GATE_CYCLES)
                hold.append(torch.cuda.Event(enable_timing=True))
                hold[0].record()
        current.wait_event(hold[0])  # this call's adds into acc wait for the delay
        return acc

    calls = []
    torch.cuda.synchronize()
    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    cake_backend._run_keys, cake_backend.prepare_lm_head_loss = recording, counting
    if shared_dx_acc:
        cake_backend._filled_dx_acc = shared
    try:
        for stream in streams:
            starts.append(time.perf_counter())
            with torch.cuda.stream(stream):
                X = inp.X.detach().requires_grad_(True)
                W = inp.W.detach().requires_grad_(True)
                if entry == "loss":
                    out, logp = cake_backend.chunked_lm_head_loss(
                        X,
                        W,
                        inp.labels,
                        objective=inp.objective,
                        loss_div=inp.loss_div,
                        infer_logp=inp.infer_logp,
                        loss_weights=inp.loss_weights,
                        chunk_size=4096,
                        return_logp=True,
                        backend="cake",
                        **ND,
                        **tune,
                    )
                    upstream = None
                else:
                    out = logp = cake_backend.chunked_lm_head_logprob(
                        X,
                        W,
                        inp.labels,
                        chunk_size=4096,
                        backend="cake",
                        **ND,
                        **tune,
                    )
                    upstream = inp.dlogp
            calls.append((X, W, out, logp, upstream))
        phase[0] = "bwd"
        gate_host = time.perf_counter()
        with torch.cuda.stream(gate):
            torch.cuda._sleep(GATE_CYCLES)
            released = torch.cuda.Event(enable_timing=True)
            released.record()
        ends, grads = [], []
        for stream, (X, W, out, _, upstream) in zip(streams, calls, strict=True):
            stream.wait_stream(gate)
            with torch.cuda.stream(stream):
                grads.append(torch.autograd.grad(out, (X, W), grad_outputs=upstream))
                ends.append(torch.cuda.Event(enable_timing=True))
                ends[-1].record()
        queued = not released.query()
        host_bwd = time.perf_counter() - gate_host
    finally:
        cake_backend._run_keys, cake_backend.prepare_lm_head_loss = run_keys, prepare
        cake_backend._filled_dx_acc = filled_dx_acc
    torch.cuda.synchronize()
    times = {
        key: [origin.elapsed_time(event) for event in events]
        for key, events in done.items()
    }
    results = []
    for (_, _, out, logp, _), (dx, dw) in zip(calls, grads, strict=True):
        got = dict(logp=logp.detach(), dX=dx, dW=dw)
        if entry == "loss":
            got["loss"] = out.detach()
        results.append(got)
    spans = {}
    for (i, ph), t in times.items():
        span = (len(t), round(min(t), 3), round(max(t), 3)) if t else (0,)
        spans[f"{'AB'[i]} {ph}"] = span
    return SimpleNamespace(
        serial=serial,
        results=results,
        times=times,
        spans=spans,
        released=origin.elapsed_time(released),
        ends=[round(origin.elapsed_time(e), 3) for e in ends],
        queued=queued,
        prepared=prepared,
        bound=bound,
        after=len(cache),
        fills=fills,
        fill_ends=[round(origin.elapsed_time(e), 3) for e in filled],
        hold_end=round(origin.elapsed_time(hold[0]), 3) if hold else None,
        host_gap=round((starts[1] - starts[0]) * 1e3, 3),
        host_bwd=round(host_bwd * 1e3, 3),
        fill_gap=(
            round((fill_host[1] - fill_host[0]) * 1e3, 3)
            if len(fill_host) == 2
            else None
        ),
    )


def _spans_overlap(first: list, second: list) -> bool:
    """Two calls' launch completion times overlap: each call has a launch that completes before the other's last."""
    if not first or not second:
        return False
    return min(second) < max(first) and min(first) < max(second)


def _overlapping(run, entry: str) -> bool:
    """A run of :func:`_same_inputs_on_two_streams` overlaps: both backwards queued before the delay ended, the spans
    of the two forwards' launch completions overlapping and, for the log-probability entry, the backwards' spans
    too.  The forward reads its row count back before it returns, so a host slowed by other work can queue the second
    forward after the first one's launches have all completed: such a run does not overlap."""
    fwd = _spans_overlap(run.times[(0, "fwd")], run.times[(1, "fwd")])
    bwd = entry != "logprob" or _spans_overlap(
        run.times[(0, "bwd")], run.times[(1, "bwd")]
    )
    return bool(run.queued and fwd and bwd)


def _racing(run) -> bool:
    """A run of the shared-buffer control races: both calls' ``-0.0`` fills completed before the delay that starts
    behind the first fill ended, so every add of both calls into the shared accumulator comes after both fills."""
    return (
        len(run.fill_ends) == 2
        and run.hold_end is not None
        and max(run.fill_ends) < run.hold_end
    )


def _two_stream_attempts(inp, entry: str, ready, what: str, node, **kwargs):
    """Runs of :func:`_same_inputs_on_two_streams` (at most :data:`TWO_STREAM_ATTEMPTS`) until ``ready(run)`` (the
    run is ``what``).  Each run prints its times; the number of runs is recorded as the test item ``node``'s
    ``two_stream_runs`` user property (``request.node.user_properties``, core pytest; the junit XML report lists it)
    and, with a ready run, in a warning (shown under xdist, which keeps no printed output of a passing test).
    Returns every run and the ready one (``None`` if none was ready)."""
    runs = []
    for attempt in range(TWO_STREAM_ATTEMPTS):
        run = _same_inputs_on_two_streams(inp, entry, **kwargs)
        runs.append(run)
        print(
            f"\ntwo streams {entry} T=16231 attempt {attempt}: ready {ready(run)}, queued {run.queued}; host gap "
            f"{run.host_gap} ms, backwards queued in {run.host_bwd} ms; delay ends {run.released:.3f} ms, "
            f"backwards end {run.ends} ms; fills end {run.fill_ends} ms (queued {run.fill_gap} ms apart), their "
            f"delay ends {run.hold_end} ms; remembered launches (count, first, last ms) {run.spans}"
        )
        if ready(run):
            break
    else:
        run = None
    node.user_properties.append(("two_stream_runs", len(runs)))
    if run is not None:
        warnings.warn(
            f"two streams {entry}: run {len(runs)} of at most {TWO_STREAM_ATTEMPTS} was {what}",
            stacklevel=2,
        )
    return runs, run


def _skip_without(run, runs, what: str) -> None:
    """Skip (never fail) a two-stream test none of whose runs had the timing it needs.  The reason carries every run's
    measured times -- the host gap between queuing the two calls and the forwards' launch completion spans, the host
    time to queue both backwards and the end of their delay, the two fills' host gap and completion times and the end
    of their delay -- so a skip on every run can be checked against them (a host slowed by other work, or a call that
    waits for the device)."""
    if run is None:
        times = [
            f"run {i + 1}: host gap {r.host_gap} ms, forwards (count, first, last ms) A {r.spans['A fwd']} "
            f"B {r.spans['B fwd']}; backwards queued {r.queued} in {r.host_bwd} ms, delay ends {r.released:.3f} ms, "
            f"backwards (count, first, last ms) A {r.spans['A bwd']} B {r.spans['B bwd']}; fills end {r.fill_ends} ms "
            f"(queued {r.fill_gap} ms apart), their delay ends {r.hold_end} ms"
            for i, r in enumerate(runs)
        ]
        pytest.skip(
            f"no {what} run in {len(runs)} runs (the second call was not queued in time): "
            + "; ".join(times)
        )


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_same_inputs_on_two_streams_at_once(glm_weight, entry, request):
    """Two calls with the same ``X`` / ``W`` / ``labels`` on two streams of one thread share the serial call's one
    remembered binding (binding-cache hits; no new binding; it owns no buffer, :func:`test_bindings_own_no_buffer`).
    The test repeats the sequence until one run overlaps (:func:`_overlapping`): the spans of the two forwards' launch
    completions overlap (both entries), both backwards are queued behind one delay on a third stream, so both are in
    flight on the device when it ends (the delay's event had not completed when both were queued), and in the
    log-probability backward -- the loss backward runs only eager casts -- the spans of the two calls' launch
    completions overlap.  Each call of EVERY run, overlapping or not, gives the serial call's ``loss`` / ``logp`` /
    ``dW`` bits and ``dX`` within the gates, and prepares no binding.  With no overlapping run in
    :data:`TWO_STREAM_ATTEMPTS` (a host too slow to queue the second call in time) the test skips with the runs' times
    after those checks; it never fails on timing.  That no call writes a buffer of the shared binding rests on the
    binding owning none (``owned_bytes == 0`` here, the registry check of :func:`test_bindings_own_no_buffer`): with the
    same inputs, two calls storing into one buffer would write the same values.  The control
    :func:`test_device_two_streams_detect_a_shared_dx_buffer` shows that these checks catch two such calls adding into
    one shared accumulator."""
    _require_program(entry=entry)
    inp = _device_inputs(16231, "ce", W=glm_weight)
    runs, run = _two_stream_attempts(
        inp, entry, lambda r: _overlapping(r, entry), "overlapping", request.node
    )
    for r in runs:
        counts = (r.prepared, r.after, r.bound)
        assert r.prepared == [] and r.after == r.bound, counts
        for got in r.results:
            for key in got.keys() - {"dX"}:
                assert _same_bits(got[key], r.serial[key]), key
            _check_against_references(got, inp, entry=entry, ceiling=True)
    _skip_without(run, runs, "overlapping")


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_two_streams_detect_a_shared_dx_buffer(glm_weight, entry, request):
    """The control of :func:`test_device_same_inputs_on_two_streams_at_once`: the same two calls with the same inputs,
    every chunk at two K slices (``gemm_tuning``, so every chunk adds into its rows), but a test-only patch hands both
    one FP32 ``dX_acc`` (the K-sliced dX reduction's target), as a buffer owned by their shared binding would be.  Each
    call fills the rows with ``-0.0``; behind its fill, its stream waits for one delay that starts behind the first
    call's fill (the forward for the loss entry, the backward for the log-probability entry).  In a racing run
    (:func:`_racing`: timing events show both fills complete before that delay ends) every add of both calls lands in
    the same rows after both fills.  Each call reads the accumulator after its own adds (the loss entry in its
    backward), so the call that reads it last reads both calls' sums, twice its ``dX``: at least one call's ``dX`` must
    fail the gates, although both have the same inputs (identical plain stores could not be told apart).  The test
    repeats the sequence until one run races; with none in :data:`TWO_STREAM_ATTEMPTS` (the second fill completed
    after the delay, a host too slow to queue it in time) it skips with the runs' times instead of passing without a
    race or failing on timing.  Every run has two fills and prepares no binding."""
    _require_program(entry=entry)
    inp = _device_inputs(16231, "ce", W=glm_weight)
    tuning = {"dx": {"k_slices": 2}}
    runs, run = _two_stream_attempts(
        inp,
        entry,
        _racing,
        "racing (shared dX_acc control)",
        request.node,
        shared_dx_acc=True,
        tuning=tuning,
    )
    for r in runs:
        assert len(r.fills) == 2 and None not in r.fills, r.fills
        assert r.prepared == [] and r.after == r.bound, (r.prepared, r.after, r.bound)
    _skip_without(run, runs, "racing (shared dX_acc control)")
    failed = []
    for got in run.results:
        try:
            _check_against_references(got, inp, entry=entry, ceiling=True)
        except AssertionError as exc:
            failed.append(repr(exc)[:200])
    ref = run.serial["dX"].float()
    diffs = [float((got["dX"].float() - ref).abs().max()) for got in run.results]
    worst = max(diffs)
    print(
        f"\nshared dX_acc control {entry} T=16231: fills {run.fills}; max|dX - serial dX| {worst:.3e} "
        f"(serial max {float(ref.abs().max()):.3e}); gate failures {len(failed)} of 2 {failed}; fills end "
        f"{run.fill_ends} ms, their delay ends {run.hold_end} ms"
    )
    assert failed, "two calls sharing one dX accumulator passed the gates"


def test_device_logprob_backward_binding_holds_no_lse_or_dlogp_address(glm_weight):
    """The log-probability backward's binding key holds the layout of ``lse`` / ``dlogp``, not their addresses.
    Backward calls on one forward whose ``lse`` and ``dlogp`` sit at a new address every time (copies kept alive) bind
    once: the later calls are binding-cache hits that prepare nothing and give the first call's ``dW`` bits and ``dX``
    within the gates.  The autograd entry's second step prepares nothing either, after a 1 GiB allocator poison
    between the steps (each forward allocates a fresh ``lse``)."""
    _require_program(entry="logprob")
    inp = _device_inputs(4097, "ce", W=glm_weight)
    fwd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096, **ND)
    prepared: list = []
    original = cake_backend.prepare_lm_head_loss

    def counting(*a, **k):
        prepared.append(k.get("entry"))
        return original(*a, **k)

    cache = cake_backend.BINDING_CACHE
    cache.clear()
    kept, results, hits = [], [], []
    cake_backend.prepare_lm_head_loss = counting
    try:
        for _ in range(3):
            lse, dlogp = fwd.lse.clone(), inp.dlogp.clone()
            kept.append((lse, dlogp))
            before = cache.hits
            results.append(
                cake_backend.backward_logprob(
                    inp.X,
                    inp.W,
                    inp.labels,
                    lse,
                    dlogp,
                    chunk_size=4096,
                    row_index=fwd.row_index,
                    **ND,
                )
            )
            hits.append(cache.hits - before)
        torch.cuda.synchronize()
    finally:
        cake_backend.prepare_lm_head_loss = original
    assert len({lse.data_ptr() for lse, _ in kept}) == 3
    assert len({dlogp.data_ptr() for _, dlogp in kept}) == 3
    assert prepared == ["logprob"] and hits == [0, 1, 1], (prepared, hits)
    for dx, dw in results[1:]:
        assert _same_bits(dw, results[0][1])
        _check_against_references(
            dict(logp=fwd.logp, dX=dx, dW=dw), inp, entry="logprob", ceiling=True
        )
    _run(inp, 4096, entry="logprob", backend="cake")
    prepared.clear()
    cake_backend.prepare_lm_head_loss = counting
    try:
        _poison_allocator(1 << 30)
        second = _run(inp, 4096, entry="logprob", backend="cake")
    finally:
        cake_backend.prepare_lm_head_loss = original
    assert prepared == [], prepared
    _check_against_references(second, inp, entry="logprob", ceiling=True)


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_side_stream_fill_reserved_memory(glm_weight, entry):
    """What the side-stream fill costs in reserved memory: ``dX_acc`` comes from the fill stream's pool, which the
    caller's stream does not share, so a call from the remembered binding started on an emptied cache reserves -- at
    its peak and after it, ``dX_acc``'s block staying cached in that pool -- at most ``T * H * 4`` bytes (``dX_acc``'s
    size bound) plus the allocator's rounding more with the fill on than with it off; the allocated bytes are the
    same up to block rounding.  Printed as the ``side-fill reserved`` line."""
    T = 12289
    _require_program(entry=entry)
    inp = _device_inputs(T, "ce", W=glm_weight)
    rows, H = int(inp.valid.sum()), int(inp.X.shape[1])
    reserved, peak, allocated = {}, {}, {}
    for side in (False, True):
        previous = cake_backend.set_dx_fill_side_stream(side)
        try:
            for _ in range(2):  # bind, then one call from the binding
                _run(inp, 4096, entry=entry, backend="cake")
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _run(inp, 4096, entry=entry, backend="cake")
            gc.collect()
            torch.cuda.synchronize()
            reserved[side] = torch.cuda.memory_reserved()
            peak[side] = torch.cuda.max_memory_reserved()
            allocated[side] = torch.cuda.max_memory_allocated()
        finally:
            cake_backend.set_dx_fill_side_stream(previous)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    extra = {k: v[True] - v[False] for k, v in (("after", reserved), ("peak", peak))}
    print(
        f"\nside-fill reserved {entry} T={T} rows={rows} H={H}: dX_acc {rows * H * 4} B (bound T*H*4 "
        f"{T * H * 4} B); reserved on - off after the call {extra['after']} B, at the peak {extra['peak']} B; "
        f"peak allocated off {allocated[False]} B, on {allocated[True]} B"
    )
    assert max(extra.values()) <= T * H * 4 + (2 << 20), (reserved, peak)
    # the same tensors: block rounding only
    assert abs(allocated[True] - allocated[False]) <= 2 << 20, allocated


def test_device_red_min_rows_keeps_small_chunks_on_the_plain_store(glm_weight):
    """``set_dx_red_min_rows``: a two-chunk call whose one-row tail falls below the floor runs that chunk with the
    plain store -- its ``dX`` rows are the deterministic package's one-slice bits -- while the 4096-row chunk keeps
    the reduction (launch record); the call passes the gates."""
    _require_program(entry="loss")
    policy = cake_backend.dx_slice_policy()
    if _device_slices(4097, policy, glm_weight)[0] == 1:
        pytest.skip(
            "the 4096-row chunk plans one slice on this device: nothing to keep on the reduction"
        )
    assert _device_slices(4097, policy, glm_weight, red_min_rows=2)[1] == 1
    inp = _device_inputs(4097, "ce", W=glm_weight, ignore_frac=0.0)
    previous = cake_backend.set_dx_red_min_rows(2)
    try:
        cake_backend.BINDING_CACHE.clear()
        with _launch_record() as launched:
            nd = _run(inp, 4096, backend="cake")
    finally:
        cake_backend.set_dx_red_min_rows(previous)
    dx_keys = [key for key in launched if key[0].startswith("gemm_dx")]
    assert [key[1] for key in dx_keys] == [0, 1], dx_keys
    assert dx_keys[0][0].startswith(cake_backend.DX_RED_STAGE)
    assert not dx_keys[1][0].startswith(cake_backend.DX_RED_STAGE)
    plain = _run(
        inp, 4096, backend="cake", package="det", gemm_tuning={"dx": {"k_slices": None}}
    )
    assert _same_bits(nd["dX"][4096:], plain["dX"][4096:])
    _check_against_references(nd, inp, ceiling=True)


def test_device_zero_weight_floor_keeps_small_calls_unsplit(glm_weight):
    """``set_zero_weight_min_rows``: a policy call below the floor keeps its zero-weight rows on the gradient path (no
    split): ``loss`` / ``logp`` / ``dW`` are the deterministic package's bits and the zero-weight rows' ``dX`` rows are
    zeros."""
    _require_program(entry="loss")
    base = _device_inputs(4097, "policy", W=glm_weight)
    zero = _zero_weight_set(base, ("random", 0.3))
    inp = _with_zero_weights(base, zero)
    previous = cake_backend.set_zero_weight_min_rows(8192)
    try:
        nd = _loss_call(inp, 4096)
        fwd = cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            **ND,
        )
    finally:
        cake_backend.set_zero_weight_min_rows(previous)
    assert "zero_weight_rows" not in (fwd.memory or {})
    det = _loss_call(inp, 4096, package="det")
    for key in ("loss", "logp", "dW"):
        assert _same_bits(nd[key], det[key]), key
    assert not bool(nd["dX"][zero].any())


# ---------------------------------------------------------------------------------------------------------------------
# Gradient placement (``set_gradient_placement``): no gradient GEMM without an autograd graph, and the gradient GEMMs
# once per checkpointed step (the non-reentrant checkpoint's recomputation forms them, not the checkpointed forward)


@contextlib.contextmanager
def _placement(mode: str):
    previous = cake_backend.set_gradient_placement(mode)
    try:
        yield
    finally:
        cake_backend.set_gradient_placement(previous)


def _gradient_count(seen) -> int:
    """Calls in ``seen`` (:func:`_gradient_forwards`) that run the gradient GEMMs."""
    return sum(1 for dx, dw in seen if dx or dw)


def _assert_step_bits(run, plain, keys=("loss", "logp", "dX", "dW")):
    for key in keys:
        assert _same_bits(run[key], plain[key]), key


@contextlib.contextmanager
def _gradient_forwards():
    """``(need_dx, need_dw)`` of every ``forward_loss`` call of both backend modules, in order (a call with either
    flag set runs the gradient GEMMs)."""
    seen: list = []
    originals = (cake_backend.forward_loss, opt_in.forward_loss)

    def recording(original):
        def forward_loss(*args, **kwargs):
            seen.append(
                (bool(kwargs.get("need_dx", True)), bool(kwargs.get("need_dw", True)))
            )
            return original(*args, **kwargs)

        return forward_loss

    cake_backend.forward_loss = recording(originals[0])
    opt_in.forward_loss = recording(originals[1])
    try:
        yield seen
    finally:
        cake_backend.forward_loss, opt_in.forward_loss = originals


def _gradient_gemm_keys(launched) -> int:
    return sum(1 for key in launched if str(key[0]).startswith(("gemm_dx", "gemm_dw")))


CHECKPOINT_KINDS = (
    "reentrant",
    "nonreentrant",
    "nonreentrant_no_early_stop",
    "nonreentrant_no_determinism_check",
    "nested",
    "offload_outside",
    "copies_outside",
    "offload_inside",
)


class _PassThroughMode(TorchDispatchMode):
    """A torch dispatch mode that runs every operator unchanged."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


_MM_OPS = (
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
)


def _selective(policy):
    """``context_fn`` of a selective activation checkpoint: ``"recompute"`` recomputes every operator, ``"save_mm"``
    saves the matrix products, ``"save_all"`` saves every operator."""
    CheckpointPolicy = torch.utils.checkpoint.CheckpointPolicy

    def policy_fn(ctx, func, *args, **kwargs):
        if policy == "save_all" or (policy == "save_mm" and func in _MM_OPS):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return functools.partial(
        torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
    )


def _checkpointed(kind, fn, X, W):
    """``fn(X, W)`` under ``torch.utils.checkpoint`` of the given kind (``plain``: no checkpoint).  ``offload_outside``:
    the checkpoint inside ``save_on_cpu`` (the checkpoint's hooks on top); ``copies_outside``: likewise inside hooks that
    save copies (the recomputation's inputs are copies on any device); ``offload_inside``: ``save_on_cpu`` inside the
    checkpointed function (its hooks on top: the call keeps its fused forward).  ``selective_recompute`` /
    ``selective_save_mm`` / ``selective_save_all``: a selective activation checkpoint (its dispatch modes in both
    passes); ``dispatch_mode``: a non-reentrant checkpoint under a pass-through dispatch mode (the checkpointed forward
    only)."""
    ckpt = torch.utils.checkpoint.checkpoint
    if kind == "plain":
        return fn(X, W)
    if kind in ("selective_recompute", "selective_save_mm", "selective_save_all"):
        return ckpt(
            fn,
            X,
            W,
            use_reentrant=False,
            context_fn=_selective(kind.removeprefix("selective_")),
        )
    if kind == "dispatch_mode":
        with _PassThroughMode():
            return ckpt(fn, X, W, use_reentrant=False)
    if kind == "reentrant":
        return ckpt(fn, X, W, use_reentrant=True)
    if kind == "nonreentrant":
        return ckpt(fn, X, W, use_reentrant=False)
    if kind == "nonreentrant_no_early_stop":
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            return ckpt(fn, X, W, use_reentrant=False)
    if kind == "nonreentrant_no_determinism_check":
        return ckpt(fn, X, W, use_reentrant=False, determinism_check="none")
    if kind == "nested":
        return ckpt(
            lambda a, b: ckpt(fn, a, b, use_reentrant=False) * 1.0,
            X,
            W,
            use_reentrant=False,
        )
    if kind == "offload_outside":
        with torch.autograd.graph.save_on_cpu():
            return ckpt(fn, X, W, use_reentrant=False)
    if kind == "copies_outside":
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: t.detach().clone(), lambda t: t
        ):
            return ckpt(fn, X, W, use_reentrant=False)
    if kind == "offload_inside":

        def offloaded(a, b):
            with torch.autograd.graph.save_on_cpu():
                return fn(a, b)

        return ckpt(offloaded, X, W, use_reentrant=False)
    raise ValueError(kind)


def _placement_step(inp, C, kind, *, module=None, memory=None, **switches):
    """One step of the loss entry (``return_logp=True``) behind a ``tanh`` of ``X`` (an op that saves a tensor ahead of
    the head) under ``kind``; returns ``loss`` / ``logp`` / ``dX`` / ``dW``.  ``memory`` (a dict, CUDA): filled with
    the bytes the forward keeps until the backward and the step's peak, both above the bytes allocated before."""
    be = cake_backend if module is None else module
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logps = []
    if memory is not None:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()

    def head(a, b):
        loss, logp = be.chunked_lm_head_loss(
            torch.tanh(a),
            b,
            inp.labels,
            objective=inp.objective,
            loss_div=inp.loss_div if inp.objective == "ce" else None,
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=C,
            return_logp=True,
            **ND,
            **switches,
        )
        logps.append(logp)
        return loss * 1.0

    loss = _checkpointed(kind, head, X, W)
    if memory is not None:
        torch.cuda.synchronize()
        memory["kept"] = torch.cuda.memory_allocated() - base
    loss.backward()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    if memory is not None:
        memory["peak"] = torch.cuda.max_memory_allocated() - base
    return dict(loss=loss.detach(), logp=logps[0].detach(), dX=X.grad, dW=W.grad)


def test_gradient_placement_switch():
    """The programmatic gradient placement: default ``"auto"``, the previous value returned, a bad value refused; the
    opt-in module reads the same setting."""
    assert cake_backend.GRADIENT_PLACEMENTS == ("auto", "forward")
    assert cake_backend.gradient_placement() == "auto"
    assert cake_backend.set_gradient_placement("forward") == "auto"
    try:
        assert cake_backend.gradient_placement() == "forward"
        assert opt_in._main.gradient_placement() == "forward"
        for bad in ("fused", None, True):
            with pytest.raises(ValueError, match="gradient placement"):
                cake_backend.set_gradient_placement(bad)
        assert cake_backend.gradient_placement() == "forward"
    finally:
        assert cake_backend.set_gradient_placement("auto") == "forward"


@pytest.mark.parametrize("graph", ["no_grad", "inference_mode", "frozen_inputs"])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_no_graph_runs_no_gradient_forward(objective, graph):
    """A loss entry call no backward can follow (grad mode off, inference mode, no input requiring a gradient) runs the
    loss-only forward; ``loss`` / ``logp`` are the fused forward's bits.  ``"forward"`` placement: the fused forward."""
    inp = _host_inputs(200, objective)
    kw = dict(
        objective=objective,
        loss_div=inp.loss_div if objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        return_logp=True,
        backend="reference",
        **ND,
    )
    X = inp.X.detach().requires_grad_(graph != "frozen_inputs")
    W = inp.W.detach().requires_grad_(graph != "frozen_inputs")
    fused = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)

    def call():
        if graph == "no_grad":
            with torch.no_grad():
                return cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)
        if graph == "inference_mode":
            with torch.inference_mode():
                return cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)
        return cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)

    with _gradient_forwards() as seen:
        loss, logp = call()
    assert seen and _gradient_count(seen) == 0, seen
    assert _same_bits(loss, fused.loss) and _same_bits(logp, fused.logp)
    assert not loss.requires_grad
    if graph != "frozen_inputs":
        with _placement("forward"), _gradient_forwards() as seen:
            loss2, logp2 = call()
        assert _gradient_count(seen) == 1, seen
        assert _same_bits(loss2, fused.loss) and _same_bits(logp2, fused.logp)


def test_frozen_head_with_a_trainable_policy_operand_keeps_the_autograd_entry():
    """Frozen ``X`` / ``W`` with an ``infer_logp`` / ``loss_weights`` that requires a gradient: the call keeps the
    autograd entry (the loss stays in the graph, a backward returns no gradient for them, as before) and forms no
    gradient."""
    inp = _host_inputs(200, "policy")
    for name in ("infer_logp", "loss_weights"):
        ops = dict(infer_logp=inp.infer_logp, loss_weights=inp.loss_weights)
        ops[name] = ops[name].detach().clone().requires_grad_(True)
        with _gradient_forwards() as seen:
            loss = cake_backend.chunked_lm_head_loss(
                inp.X,
                inp.W,
                inp.labels,
                objective="policy",
                chunk_size=64,
                backend="reference",
                **ops,
                **ND,
            )
        assert seen and _gradient_count(seen) == 0, seen
        assert loss.requires_grad and loss.grad_fn is not None, name
        loss.backward()
        assert ops[name].grad is None, name


@pytest.mark.parametrize("kind", CHECKPOINT_KINDS)
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_checkpointed_step_forms_the_gradients_once(objective, kind):
    """A checkpointed step runs one gradient forward, as an uncheckpointed step (the non-reentrant checkpointed forward
    and the reentrant first pass run the loss only; the recomputation forms the gradients; with offload hooks on top
    the call keeps its fused forward); ``loss`` / ``logp`` / ``dX`` / ``dW`` are the uncheckpointed step's bits.
    ``"forward"`` placement: two gradient forwards, the same bits."""
    inp = _host_inputs(200, objective)
    plain = _placement_step(inp, 64, "plain", backend="reference")
    with _gradient_forwards() as seen:
        run = _placement_step(inp, 64, kind, backend="reference")
    grads = [p for p in seen if p[0] or p[1]]
    assert len(grads) == 1, seen
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(run[key], plain[key]), key
    if kind != "offload_inside":
        with _placement("forward"), _gradient_forwards() as seen:
            again = _placement_step(inp, 64, kind, backend="reference")
        assert len([p for p in seen if p[0] or p[1]]) == 2, seen
        for key in ("loss", "logp", "dX", "dW"):
            assert _same_bits(again[key], plain[key]), key
    gc.collect()
    assert len(cake_backend._CHECKPOINT_CALLS) == 0


def test_checkpointed_forward_alerts_in_the_forward_under_deterministic_algorithms():
    """Under torch's deterministic algorithms a checkpointed call keeps its fused forward: the alert stays raised in the
    checkpointed forward, before anything is bound or launched, as without a checkpoint.  Under ``torch.no_grad`` the
    call launches no gradient GEMM and does not alert."""
    inp = _host_inputs(37, device=CPU)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def head(a, b):
        return cake_backend.chunked_lm_head_loss(
            a, b, inp.labels, loss_div=1.0, chunk_size=16, backend="reference", **ND
        )

    with (
        _torch_deterministic(True),
        pytest.raises(
            RuntimeError, match="does not have a deterministic implementation"
        ),
    ):
        torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
    with _torch_deterministic(True), warnings.catch_warnings(), torch.no_grad():
        warnings.filterwarnings("error", message=".*deterministic implementation.*")
        head(X, W)


@pytest.mark.parametrize("module", ["default", "count"])
@pytest.mark.parametrize(
    "kind",
    [
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
        "dispatch_mode_step",
    ],
)
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_selective_checkpoint_and_dispatch_modes_form_the_gradients_once(
    objective, kind, module
):
    """Selective activation checkpointing replays the checkpointed forward's operators in the recomputation by
    position; the call runs outside its dispatch modes, so the checkpointed forward computes the loss only and the
    recomputation forms the gradients, as in a plain non-reentrant checkpoint (also with a policy that saves every
    operator).  Likewise under another dispatch mode (``dispatch_mode``: around the checkpointed forward only;
    ``dispatch_mode_step``: around the forward and the backward).  One gradient forward, the uncheckpointed step's
    bits, no frame record left; the ``"forward"`` placement forms them in both passes, with the same bits.  ``count``:
    the opt-in count form."""
    inp = _host_inputs(200, objective)
    kw = dict(backend="reference")
    if module == "count":
        kw.update(module=opt_in, num_valid_rows=int((inp.labels >= 0).sum()))
    plain = _placement_step(inp, 64, "plain", **kw)
    for mode, gradient_forwards in (("auto", 1), ("forward", 2)):
        with _placement(mode), _gradient_forwards() as seen:
            if kind == "dispatch_mode_step":
                with _PassThroughMode():
                    run = _placement_step(inp, 64, "nonreentrant", **kw)
            else:
                run = _placement_step(inp, 64, kind, **kw)
        assert _gradient_count(seen) == gradient_forwards, (mode, seen)
        _assert_step_bits(run, plain)
    gc.collect()
    assert len(cake_backend._CHECKPOINT_CALLS) == 0


def test_checkpointed_retained_graph_forms_the_gradients_per_backward():
    """Two backward passes through one retained checkpointed graph: each recomputes and forms the gradients once,
    each pair of gradients is the uncheckpointed step's bits."""
    inp = _host_inputs(200, "policy")
    plain = _placement_step(inp, 64, "plain", backend="reference")
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def head(a, b):
        return cake_backend.chunked_lm_head_loss(
            torch.tanh(a),
            b,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=64,
            backend="reference",
            **ND,
        )

    with _gradient_forwards() as seen:
        loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
        first = torch.autograd.grad(loss, (X, W), retain_graph=True)
        second = torch.autograd.grad(loss, (X, W))
    assert len([p for p in seen if p[0] or p[1]]) == 2, seen
    for grads in (first, second):
        assert _same_bits(grads[0], plain["dX"]) and _same_bits(grads[1], plain["dW"])


def test_checkpointed_call_without_a_handed_result_forms_its_gradients_in_the_backward():
    """When the recomputation cannot hand its result to the checkpointed call (its frame record dropped), the
    checkpointed call's backward runs the fused forward itself, with a warning: the same bits."""
    inp = _host_inputs(200, "ce")
    plain = _placement_step(inp, 64, "plain", backend="reference")
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def head(a, b):
        return cake_backend.chunked_lm_head_loss(
            torch.tanh(a),
            b,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=64,
            backend="reference",
            **ND,
        )

    loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
    assert len(cake_backend._CHECKPOINT_CALLS) == 1
    cake_backend._CHECKPOINT_CALLS.clear()
    with (
        _gradient_forwards() as seen,
        pytest.warns(RuntimeWarning, match="no gradient state from its recomputation"),
    ):
        loss.backward()
    assert len([p for p in seen if p[0] or p[1]]) == 2, (
        seen
    )  # the recomputation's and the backward's own
    assert _same_bits(X.grad, plain["dX"]) and _same_bits(W.grad, plain["dW"])


def test_checkpointed_steps_on_concurrent_threads():
    """Non-reentrant checkpointed steps on two threads at once: each forms the gradients once, with the uncheckpointed
    step's bits, and no frame record is left behind."""
    inps = [_host_inputs(200, "ce"), _host_inputs(201, "policy")]
    plains = [_placement_step(inp, 64, "plain", backend="reference") for inp in inps]
    barrier = threading.Barrier(2)
    runs: list = [None, None]
    errors: list = []

    def work(i):
        try:
            barrier.wait(TIMEOUT_S)
            for _ in range(3):
                runs[i] = _placement_step(
                    inps[i], 64, "nonreentrant", backend="reference"
                )
        except BaseException as exc:  # noqa: BLE001 -- reported below
            errors.append(exc)

    with _gradient_forwards() as seen:
        threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    assert not errors, errors
    assert _gradient_count(seen) == 6, seen
    for run, plain in zip(runs, plains, strict=True):
        _assert_step_bits(run, plain)
    gc.collect()
    assert len(cake_backend._CHECKPOINT_CALLS) == 0


@pytest.mark.parametrize("kind", ["no_grad", "nonreentrant", "reentrant"])
def test_opt_in_count_form_places_the_gradients_alike(kind):
    """The opt-in count form (``num_valid_rows``) follows the gradient placement: no gradient forward without a graph,
    one per checkpointed step, the uncheckpointed step's bits."""
    inp = _host_inputs(200, "policy")
    count = int((inp.labels >= 0).sum())
    plain = _placement_step(
        inp, 64, "plain", module=opt_in, backend="reference", num_valid_rows=count
    )
    if kind == "no_grad":
        with torch.no_grad(), _gradient_forwards() as seen:
            # the plain step's head input (``tanh`` of X), trainable
            loss, logp = opt_in.chunked_lm_head_loss(
                torch.tanh(inp.X).detach().requires_grad_(True),
                inp.W.detach().requires_grad_(True),
                inp.labels,
                objective="policy",
                infer_logp=inp.infer_logp,
                loss_weights=inp.loss_weights,
                chunk_size=64,
                return_logp=True,
                backend="reference",
                num_valid_rows=count,
                **ND,
            )
        assert seen and _gradient_count(seen) == 0, seen
        _assert_step_bits(dict(loss=loss, logp=logp), plain, ("loss", "logp"))
        return
    with _gradient_forwards() as seen:
        run = _placement_step(
            inp, 64, kind, module=opt_in, backend="reference", num_valid_rows=count
        )
    assert _gradient_count(seen) == 1, seen
    _assert_step_bits(run, plain)


def test_deterministic_true_is_one_call_of_the_deterministic_entry_under_no_grad_and_checkpoint(
    monkeypatch,
):
    """The public ``deterministic=True`` path stays one call of the deterministic entry with the caller's arguments
    (no gradient placement of its own), under ``torch.no_grad`` and inside a checkpoint (its forward and its
    recomputation)."""
    from flashinfer import chunked_lm_head as pub

    calls = []

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        # saves its input: a checkpoint's backward unpacks it and so runs the recomputation
        return args[0].float().square().sum() * 0.0

    monkeypatch.setattr(pub, "_det_loss", recording)
    inp = _host_inputs(64, "ce")
    kw = dict(loss_div=inp.loss_div, chunk_size=32)
    X = inp.X.detach().requires_grad_(True)
    with torch.no_grad():
        pub.chunked_lm_head_loss(X, inp.W, inp.labels, **kw)
    assert len(calls) == 1 and calls[0][0][0] is X and calls[0][1] == kw
    torch.utils.checkpoint.checkpoint(
        lambda a: pub.chunked_lm_head_loss(a, inp.W, inp.labels, **kw),
        X,
        use_reentrant=False,
    ).backward()
    assert len(calls) == 3 and all(c[1] == kw for c in calls)


@pytest.mark.parametrize("graph", ["no_grad", "inference_mode"])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_no_graph_launches_no_gradient_gemm(glm_weight, objective, graph):
    """A three-chunk loss entry call under ``torch.no_grad`` / ``torch.inference_mode`` launches no dX / dW GEMM (launch
    record); ``loss`` / ``logp`` are the fused forward's bits."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, objective, W=glm_weight)
    kw = dict(
        objective=objective,
        loss_div=inp.loss_div if objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
        return_logp=True,
        **ND,
    )
    fused = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    mode = torch.no_grad() if graph == "no_grad" else torch.inference_mode()
    with mode, _launch_record() as launched:
        loss, logp = cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)
    torch.cuda.synchronize()
    assert _gradient_gemm_keys(launched) == 0, sorted({k[0] for k in launched})
    assert any(k[0] == "row_finalize" for k in launched)
    assert _same_bits(loss, fused.loss) and _same_bits(logp, fused.logp)


@pytest.mark.parametrize(
    "kind",
    [
        "reentrant",
        "nonreentrant",
        "nonreentrant_no_early_stop",
        "nonreentrant_no_determinism_check",
        "nested",
        "offload_outside",
        "copies_outside",
        "offload_inside",
    ],
)
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_checkpointed_step_launches_the_gradient_gemms_once(
    glm_weight, objective, kind
):
    """A checkpointed three-chunk step launches as many dX / dW GEMMs as the uncheckpointed step (launch record);
    with one dX slice pinned (the same plan, the same bits run to run) ``loss`` / ``logp`` / ``dX`` / ``dW`` are the
    uncheckpointed step's bits and the ``"forward"`` placement's (which launches them twice -- the reentrant first
    pass: its dW GEMMs only -- and once with the offload hooks innermost)."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, objective, W=glm_weight)
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})
    _placement_step(inp, 4096, "plain", **one)  # binds
    with _launch_record() as launched:
        plain = _placement_step(inp, 4096, "plain", **one)
    count = _gradient_gemm_keys(launched)
    assert count >= 5, sorted({k[0] for k in launched})
    with _launch_record() as launched:
        run = _placement_step(inp, 4096, kind, **one)
    assert _gradient_gemm_keys(launched) == count
    _assert_step_bits(run, plain)
    with _placement("forward"), _launch_record() as launched:
        again = _placement_step(inp, 4096, kind, **one)
    forward = _gradient_gemm_keys(launched)
    if kind == "offload_inside":  # nothing is recomputed: one fused forward
        assert forward == count
    elif kind == "reentrant":
        # the first pass runs under no_grad, where the head's input (tanh of X) is no leaf of the graph: dW only
        assert count < forward < 2 * count, (forward, count)
    else:
        assert forward == 2 * count, (forward, count)
    _assert_step_bits(again, plain)


@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_checkpointed_step_default_plan_and_memory(glm_weight, objective):
    """Under the default dX slice rule a non-reentrant checkpointed three-chunk step gives the uncheckpointed step's
    ``loss`` / ``logp`` / ``dW`` bits and a ``dX`` within the gates; it keeps fewer bytes from the forward to the
    backward than the ``"forward"`` placement (which keeps the FP32 gradient accumulators) and peaks no higher."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, objective, W=glm_weight)
    plain = _placement_step(inp, 4096, "plain")
    mem = {}
    for mode in ("forward", "auto"):
        with _placement(mode):
            _placement_step(inp, 4096, "nonreentrant")  # binds
            mem[mode] = {}
            run = _placement_step(inp, 4096, "nonreentrant", memory=mem[mode])
        for key in ("loss", "logp", "dW"):
            assert _same_bits(run[key], plain[key]), (mode, key)
        assert (
            float((run["dX"].float() - plain["dX"].float()).abs().max())
            <= float(plain["dX"].float().abs().max()) * 2.0**-7
        ), mode
    print(
        f"\ncheckpoint T=12289 {objective}: kept/peak forward {mem['forward']} auto {mem['auto']}"
    )
    acc = (
        int((inp.labels >= 0).sum()) * inp.X.shape[1] * 4
    )  # the FP32 dX accumulator alone
    assert mem["auto"]["kept"] + acc <= mem["forward"]["kept"], mem
    assert mem["auto"]["peak"] <= mem["forward"]["peak"], mem


def test_device_checkpointed_logprob_step_launches_the_gradient_gemms_once(glm_weight):
    """The log-probability entry forms no gradient in its forward: a checkpointed step launches the uncheckpointed
    step's dX / dW GEMMs and gives its bits (one dX slice pinned)."""
    _require_program(entry="logprob")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})

    def step(kind):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)

        def head(a, b):
            logp = cake_backend.chunked_lm_head_logprob(
                a, b, inp.labels, chunk_size=4096, **ND, **one
            )
            return (logp * inp.dlogp).sum()

        _checkpointed(kind, head, X, W).backward()
        torch.cuda.synchronize()
        return X.grad, W.grad

    step("plain")
    with _launch_record() as launched:
        plain = step("plain")
    count = _gradient_gemm_keys(launched)
    assert count >= 5, sorted({k[0] for k in launched})
    for kind in (
        "reentrant",
        "nonreentrant",
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
    ):
        with _launch_record() as launched:
            run = step(kind)
        assert _gradient_gemm_keys(launched) == count, kind
        assert _same_bits(run[0], plain[0]) and _same_bits(run[1], plain[1]), kind


@pytest.mark.parametrize("memory_mode", [None, "budget"])
@pytest.mark.parametrize(
    "kind",
    [
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
    ],
)
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_selective_checkpoint_step_launches_the_gradient_gemms_once(
    glm_weight, objective, kind, memory_mode
):
    """A three-chunk step under selective activation checkpointing (or a non-reentrant checkpoint under a dispatch
    mode) launches the uncheckpointed step's dX / dW GEMMs (the recomputation forms the gradients) and gives its
    bits (one dX slice pinned)."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, objective, W=glm_weight)
    kw = {} if memory_mode is None else dict(module=opt_in, memory_mode=memory_mode)
    kw.update(gemm_tuning={"dx": {"k_slices": 1}})
    _placement_step(inp, 4096, "plain", **kw)  # binds
    with _launch_record() as launched:
        plain = _placement_step(inp, 4096, "plain", **kw)
    count = _gradient_gemm_keys(launched)
    assert count >= 5, sorted({k[0] for k in launched})
    with _launch_record() as launched:
        run = _placement_step(inp, 4096, kind, **kw)
    assert _gradient_gemm_keys(launched) == count, kind
    _assert_step_bits(run, plain)


# A process whose first gradient step of an entry is a selective-checkpoint step, run by a fresh interpreter (one dX
# slice pinned: the selective and the plain step are compared by their bits).
_SELECTIVE_FIRST_CALL_SCRIPT = """
import functools, sys, torch
import torch.utils.checkpoint as tuc
sys.path.insert(0, {root!r})
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.test_helpers.cake_package import cake_package
backend = cake_package("cake.cake_lm_head_loss", {module!r})
W = make_inputs(1, seed={seed_w}, device="cuda", ignore_frac=0.0).W
inp = make_inputs({T}, seed={seed}, device="cuda", W=W, ignore_frac=0.05)
kw = dict(chunk_size={C}, backend="cake", deterministic=False, gemm_tuning={{"dx": {{"k_slices": 1}}}}, **{switches!r})


def head(a, b):
    if {entry!r} == "loss":
        return backend.chunked_lm_head_loss(a, b, inp.labels, loss_div=inp.loss_div, **kw)
    return (backend.chunked_lm_head_logprob(a, b, inp.labels, **kw) * inp.dlogp).sum()


def policy(ctx, func, *args, **kwargs):
    return tuc.CheckpointPolicy.{policy}


def step(selective):
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    if selective:
        context_fn = functools.partial(tuc.create_selective_checkpoint_contexts, policy)
        loss = tuc.checkpoint(head, X, W, use_reentrant=False, context_fn=context_fn)
    else:
        loss = head(X, W)
    loss.backward()
    torch.cuda.synchronize()
    return [loss.detach(), X.grad, W.grad]


if {warm!r}:  # the main thread's state formed first: the recomputation (autograd thread) forms its own
    with torch.no_grad():
        head(inp.X, inp.W)
first = step(True)  # the entry's first gradient step: inside a selective checkpoint
plain = step(False)
for a, b in zip(first, plain):
    assert a.dtype == b.dtype and torch.equal(a.reshape(-1).view(torch.int8), b.reshape(-1).view(torch.int8))
print("SELECTIVE_FIRST_CALL_OK")
"""


@pytest.mark.parametrize("policy", ["PREFER_RECOMPUTE", "MUST_SAVE"])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "entry,memory_mode",
    [
        ("loss", None),
        ("loss", "budget"),
        ("logprob", None),
        ("logprob", "budget"),
        ("logprob", "keep_logits"),
    ],
)
def test_device_selective_checkpoint_first_call_of_a_process(
    entry, memory_mode, warm, policy
):
    """A fresh process whose first gradient step of the entry is a selective-checkpoint step (a policy that
    recomputes every operator, and one that saves every operator): the process-wide constant cells (first used inside
    its recomputation) and the autograd thread's own state (the valid-row constant and count cells, formed in the
    recomputation; ``warm``: after the main thread's were formed by a ``no_grad`` call) stay outside the checkpoint's
    operator record.  The step runs and gives the uncheckpointed step's bits.  This process's cached blocks are
    released first (the fresh interpreter needs memory for its own context)."""
    _require_program(entry=entry)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    T = 12289
    script = _SELECTIVE_FIRST_CALL_SCRIPT.format(
        root=root,
        module="cake_backend" if memory_mode is None else "cake_backend_opt_in",
        seed_w=SEED,
        T=T,
        seed=SEED + T,
        C=4096,
        entry=entry,
        switches={} if memory_mode is None else dict(memory_mode=memory_mode),
        warm=warm,
        policy=policy,
    )
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
        timeout=600,
    )
    assert proc.returncode == 0 and "SELECTIVE_FIRST_CALL_OK" in proc.stdout, (
        f"fresh process failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )


def _region_entry(entry, inp, X, W):
    """The scalar a selective-checkpoint region test differentiates: one entry of this package on ``X`` / ``W`` (one
    dX slice pinned)."""
    one = dict(gemm_tuning={"dx": {"k_slices": 1}}, **ND)
    if entry in ("loss", "loss_budget"):
        mod, kw = (
            (cake_backend, {})
            if entry == "loss"
            else (opt_in, dict(memory_mode="budget"))
        )
        return mod.chunked_lm_head_loss(
            X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, **kw, **one
        )
    if entry == "support":
        # eight distinct candidates per row, the label first (an ignored row: candidates of token 0)
        K, T, V = 8, X.shape[0], W.shape[0]
        base = inp.labels.clamp_min(0)[:, None]
        ids = (base + 997 * torch.arange(K, device=X.device)[None, :]) % V
        offsets = torch.arange(0, T * K + 1, K, device=X.device)
        support = cake_package("cake.cake_lm_head_loss", "cake_support")
        logp = support.chunked_lm_head_support_logprob(
            X, W, ids.reshape(-1), offsets, inp.labels, k_cap=K, **ND
        )
        return (logp * inp.dlogp).sum()
    if entry == "keep_backend":
        logp = cake_backend.chunked_lm_head_logprob_keep(
            X, W, inp.labels, chunk_size=4096, **one
        )
    else:
        memory_mode = {
            "logprob": None,
            "logprob_budget": "budget",
            "keep": "keep_logits",
        }[entry]
        logp = opt_in.chunked_lm_head_logprob(
            X, W, inp.labels, chunk_size=4096, memory_mode=memory_mode, **one
        )
    return (logp * inp.dlogp).sum()


@pytest.mark.parametrize("kind", ["hold", "user_mode", "nested", "save_all"])
@pytest.mark.parametrize(
    "entry",
    [
        "loss",
        "loss_budget",
        "logprob",
        "logprob_budget",
        "keep",
        "keep_backend",
        "support",
    ],
)
def test_device_selective_checkpoint_region_input(glm_weight, entry, kind):
    """The entry's input formed inside the selective-checkpoint region (``X = Hin @ P``, ``P`` trainable): a plain
    step binds at ``X``'s address, the checkpointed forward finds that binding, and memory held between the forward
    and the backward moves the recomputed ``X`` to another address (a binding miss in the recomputation only).  Also
    with a user dispatch mode around the whole step (``user_mode``), a non-reentrant checkpoint nested inside the
    selective one (``nested``) and a policy that saves every operator (``save_all``).  ``loss`` / ``dHin`` / ``dP`` /
    ``dW`` are the uncheckpointed step's bits."""
    _require_program(entry="loss" if entry.startswith("loss") else "logprob")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    H = inp.X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    Hin0 = (torch.randn(inp.X.shape, generator=g) * 0.5).to(torch.bfloat16).cuda()
    P0 = (torch.randn(H, H, generator=g) / H**0.5).to(torch.bfloat16).cuda()

    def region(h, p, w):
        return _region_entry(entry, inp, torch.mm(h, p), w)

    def step(selective):
        h, p, w = (t.detach().requires_grad_(True) for t in (Hin0, P0, inp.W))
        mode = (
            _PassThroughMode()
            if (selective and kind == "user_mode")
            else contextlib.nullcontext()
        )
        with mode:
            if not selective:
                out = region(h, p, w)
            else:
                ctx = _selective("save_all" if kind == "save_all" else "recompute")
                fn = region
                if kind == "nested":

                    def fn(a, b, c):
                        return torch.utils.checkpoint.checkpoint(
                            region, a, b, c, use_reentrant=False
                        )

                out = torch.utils.checkpoint.checkpoint(
                    fn, h, p, w, use_reentrant=False, context_fn=ctx
                )
            held = [torch.empty_like(Hin0) for _ in range(2)] if selective else []
            out.backward()
        torch.cuda.synchronize()
        del held
        return [out.detach(), h.grad, p.grad, w.grad]

    step(False)  # binds
    plain = step(False)
    for _ in range(2):
        # a plain step right before (results dropped): its X's block is the one the checkpointed forward's X reuses
        step(False)
        run = step(True)
        for name, a, b in zip(("loss", "dHin", "dP", "dW"), run, plain, strict=True):
            assert _same_bits(a, b), name


@pytest.mark.parametrize("entry", ["backend", "opt_in"])
def test_device_checkpointed_logprob_keep_step_is_bitwise(glm_weight, entry):
    """The kept logits (``chunked_lm_head_logprob_keep``; the opt-in ``memory_mode="keep_logits"``) under a
    non-reentrant checkpoint, selective activation checkpointing and a dispatch mode (around the checkpointed forward;
    around the whole step) give the uncheckpointed default entry's ``dX`` / ``dW`` bits (one dX slice pinned)."""
    _require_program(entry="logprob")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    one = dict(gemm_tuning={"dx": {"k_slices": 1}}, **ND)

    def head_default(a, b):
        return (
            cake_backend.chunked_lm_head_logprob(
                a, b, inp.labels, chunk_size=4096, **one
            )
            * inp.dlogp
        ).sum()

    def head_keep(a, b):
        if entry == "backend":
            logp = cake_backend.chunked_lm_head_logprob_keep(
                a, b, inp.labels, chunk_size=4096, **one
            )
        else:
            logp = opt_in.chunked_lm_head_logprob(
                a, b, inp.labels, chunk_size=4096, memory_mode="keep_logits", **one
            )
        return (logp * inp.dlogp).sum()

    def step(kind, head):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        if kind == "dispatch_mode_step":
            with _PassThroughMode():
                _checkpointed("nonreentrant", head, X, W).backward()
        else:
            _checkpointed(kind, head, X, W).backward()
        torch.cuda.synchronize()
        return X.grad, W.grad

    step("plain", head_keep)  # binds
    plain = step("plain", head_default)
    for kind in (
        "plain",
        "nonreentrant",
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
        "dispatch_mode_step",
    ):
        run = step(kind, head_keep)
        assert _same_bits(run[0], plain[0]), kind
        assert _same_bits(run[1], plain[1]), kind


@pytest.mark.parametrize("memory_mode", ["budget", None])
def test_device_opt_in_forms_place_the_gradients_alike(glm_weight, memory_mode):
    """The opt-in forms (``memory_mode="budget"`` / ``num_valid_rows``) follow the gradient placement on the device:
    no dX / dW GEMM under ``torch.no_grad``; a non-reentrant checkpointed step launches the uncheckpointed step's
    count; the same bits (one dX slice pinned)."""
    _require_program(entry="loss")
    inp = _device_inputs(12289, "policy", W=glm_weight)
    sw = dict(gemm_tuning={"dx": {"k_slices": 1}})
    if memory_mode is None:
        sw["num_valid_rows"] = int((inp.labels >= 0).sum())
    else:
        sw["memory_mode"] = memory_mode
    _placement_step(inp, 4096, "plain", module=opt_in, **sw)
    with _launch_record() as launched:
        plain = _placement_step(inp, 4096, "plain", module=opt_in, **sw)
    count = _gradient_gemm_keys(launched)
    with _launch_record() as launched:
        run = _placement_step(inp, 4096, "nonreentrant", module=opt_in, **sw)
    assert _gradient_gemm_keys(launched) == count
    _assert_step_bits(run, plain)
    # trainable inputs (the plain step's head input, ``tanh`` of X): only the grad mode keeps the call from forming
    # gradients
    X = torch.tanh(inp.X).detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    with torch.no_grad(), _launch_record() as launched:
        loss, logp = opt_in.chunked_lm_head_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **ND,
            **sw,
        )
    torch.cuda.synchronize()
    assert _gradient_gemm_keys(launched) == 0
    _assert_step_bits(dict(loss=loss, logp=logp), plain, ("loss", "logp"))


# ---------------------------------------------------------------------------
# Kept log-probability logits (``memory_mode="keep_logits"``; the deterministic package's kept logits, this
# package's names)
# ---------------------------------------------------------------------------

# One dX slice pinned: the plain store in every chunk, so a dX comparison is a comparison of bits.
ONE_SLICE = {"dx": {"k_slices": 1}}


@contextlib.contextmanager
def _cache(enabled: bool):
    cache = cake_backend.BINDING_CACHE
    was = cache.enabled
    cache.enabled = enabled
    try:
        yield cache
    finally:
        cache.enabled = was


def _keep_problem(T, H, V, C, entry="loss"):
    return cake_backend.Problem(
        num_rows=T,
        hidden=H,
        vocab=V,
        chunk=C,
        ld_x=H,
        x_copy=False,
        objective="ce",
        loss_div=1.0,
        grad_weight_dtype=torch.bfloat16,
        entry=entry,
    )


def _same_dx(dx, base, inp, plan, case):
    """``dX`` of the kept path against the default call ``base``: the same bits when every chunk of ``plan`` (the kept
    backward's) runs one dX slice; else the same products in the unordered K-slice reduction, held to the suite's
    budgets (``base``'s ``loss`` / ``logp`` / ``dW``, the kept path's bits, with this ``dX``)."""
    if dx is None or base["dX"] is None:
        assert dx is None and base["dX"] is None, case
        return
    if max(plan.dx_slices or (1,)) == 1:
        assert _same_bits(dx, base["dX"]), case
        return
    flips, ulps = _ulp_flips(dx, base["dX"])
    print(
        f"\nkept dX {case}: {flips:.2e} of elements differ from the default call's, max {ulps} ulp"
    )
    beyond = inp.labels >= inp.V
    if not bool(beyond.any()):
        _check_against_references(dict(base, dX=dx), inp, entry="logprob")
        return
    # labels beyond the last vocabulary column: no oracle describes those rows, so their dX is held to the default
    # call's within the dX budget; the oracles see them as ignored rows and the gate holds the dX of the other rows
    masked = dataclasses.replace(
        inp, labels=torch.where(beyond, torch.full_like(inp.labels, -100), inp.labels)
    )
    oracle = reference_fp64(masked, entry="logprob", need_dw=False)
    b0 = reference_unchunked(masked, entry="logprob", need_dw=False)
    rows = ~beyond
    err = rel_l2(dx[rows], oracle["dX"][rows])
    b0_err = rel_l2(b0["dX"][rows], oracle["dX"][rows])
    assert err <= GATE_MARGIN * b0_err + GATE_TINY["dX_rel_l2"], (case, err, b0_err)
    assert rel_l2(dx[beyond], base["dX"][beyond]) <= GATE_TINY["dX_rel_l2"], case


def test_logprob_keep_plan_keys_and_memory():
    """The kept-logits plan of the log-probability entry (``logprob_keep``): the forward runs the plain ``gemm_logits``
    per chunk (never the label-column arm), the backward recomputes nothing -- the recompute path's keys without its
    logits GEMMs: ``row_grad`` over each chunk's kept logits, then the dX / weight-gradient GEMMs, the chunk's ``X``
    rows gathered for the weight gradient only -- and the dW side stream forks after ``row_grad``; the arms are off
    whatever their switches; the loss entry and a plan without a registered program reject it; the kept form is its
    own plan class (``_KeptPlan``, the same fields), so the default plan and cache keys are unchanged;
    ``memory_report`` counts the kept logits."""
    g = cake_backend.Geometry.from_record(
        {"geometry": {"hidden": 6144, "vocab": 154880}}
    )
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        dx_finalize=True,
        arch="sm_100a",
    )
    problem = _keep_problem(8700, 6144, 154880, 4096, entry="logprob")
    plain = cake_backend.make_plan(problem, valid_rows=8265, **common)
    keep = cake_backend.make_plan(
        problem,
        valid_rows=8265,
        logprob_dz=True,
        logprob_lbl=True,
        logprob_keep=True,
        **common,
    )
    assert keep.logprob_keep and not (keep.logprob_dz or keep.logprob_lbl)
    assert type(keep) is cake_backend._KeptPlan and type(plain) is cake_backend.Plan
    assert plain.logprob_keep is False
    assert "logprob_keep" not in {f.name for f in dataclasses.fields(cake_backend.Plan)}
    assert dataclasses.fields(keep) == dataclasses.fields(plain)
    assert type(dataclasses.replace(keep, hidden_count=True)) is cake_backend._KeptPlan
    assert keep.chunks == plain.chunks and keep.num_chunks == 3
    assert keep.dx_slices == plain.dx_slices
    assert cake_backend.forward_keys(keep) == cake_backend.forward_keys(plain)
    assert {k[0] for k in cake_backend.forward_keys(keep)} >= {"gemm_logits"}
    bwd = cake_backend.recompute_keys(keep)
    assert list(bwd) == [
        k for k in cake_backend.recompute_keys(plain) if k[0] != "gemm_logits_nostats"
    ]
    assert [k for k in bwd if k[0] == "row_grad"] == [("row_grad", i) for i in range(3)]
    assert [k for k in bwd if k[0] == "gather_rows"] == [("gather_rows", "d_in")] + [
        ("gather_rows", i) for i in range(3)
    ]
    assert {"gemm_logits", "row_grad"} <= set(keep.stages)
    assert not {"gemm_logits_nostats", "gemm_logits_dz", "gemm_logits_lbl"} & set(
        keep.stages
    )
    _, forks, sides = cake_backend.side_stream_schedule(keep, bwd, "1")
    assert sorted(forks) == [i for i, k in enumerate(bwd) if k[0] == "row_grad"]
    assert len(sides) == 2
    # a frozen W: no weight gradient, so no chunk gather (the kept logits replace the logits GEMM's X operand)
    frozen = cake_backend.make_plan(
        problem, valid_rows=8265, logprob_keep=True, **dict(common, need_dw=False)
    )
    rk = cake_backend.recompute_keys(frozen)
    assert [k for k in rk if k[0] == "gather_rows"] == [("gather_rows", "d_in")]
    assert [k for k in rk if k[0] == "row_grad"] == [("row_grad", i) for i in range(3)]
    with pytest.raises(ValueError, match="log-probability"):
        cake_backend.make_plan(
            _keep_problem(8700, 6144, 154880, 4096), logprob_keep=True, **common
        )
    with pytest.raises(ValueError, match="arch required"):
        cake_backend.make_plan(problem, logprob_keep=True, **dict(common, arch=None))
    # the default cache keys take no kept-logits element (the kept entry points append ``"keep"`` themselves)
    for fn in (
        cake_backend.forward_binding_key,
        cake_backend.logprob_backward_binding_key,
    ):
        assert "logprob_keep" not in inspect.signature(fn).parameters
    # memory: the chunks after the first are a new saved buffer, chunk 0 stays in the chunk buffer; the default
    # report (memory_report, unchanged) has no kept-logits key
    V = 154880
    for hidden in (False, True):
        rep0 = cake_backend.memory_report(
            8700, 6144, V, 4096, entry="logprob", valid_rows=8265, hidden_count=hidden
        )
        assert "saved_logits_bytes" not in rep0
        assert "saved_logits" not in rep0["accumulators"]
        rep1 = cake_backend.kept_memory_report(rep0, keep)
        assert rep1["saved_logits_bytes"] == 8265 * V * 2
        assert rep1["accumulators"]["saved_logits"] == (8265 - 4096) * V * 2
        assert (
            rep1["accumulator_bytes"] - rep0["accumulator_bytes"]
            == (8265 - 4096) * V * 2
        )
        assert rep1["temporary"] == rep0["temporary"]
        assert {
            k: v
            for k, v in rep1.items()
            if k not in ("saved_logits_bytes", "accumulators", "accumulator_bytes")
        } == {
            k: v
            for k, v in rep0.items()
            if k not in ("accumulators", "accumulator_bytes")
        }
    assert (
        "logprob_keep" not in inspect.signature(cake_backend.memory_report).parameters
    )


def test_logprob_keep_entry_points_and_opt_in_mode(monkeypatch):
    """The kept logits have their own entry points (``chunked_lm_head_logprob_keep``, ``forward_logprob_keep`` /
    ``backward_logprob_keep``); the default entry points take no kept-logits keyword.  The kept autograd entry runs
    the default entry wherever no backward can follow -- grad mode off (``torch.no_grad``, ``torch.inference_mode``),
    no input requiring a gradient -- and on the reference backend; the explicit pair and the prepared runner of the
    kept plan belong to the ``cake`` backend.  The opt-in form ``memory_mode="keep_logits"`` belongs to the
    log-probability entry without ``num_valid_rows`` and runs the kept entry; elsewhere it is a ``ValueError``.  The
    deterministic package's kept entry takes the same keywords."""
    default_params = inspect.signature(cake_backend.chunked_lm_head_logprob).parameters
    assert (
        inspect.signature(cake_backend.chunked_lm_head_logprob_keep).parameters
        == default_params
    )
    assert list(
        inspect.signature(cake_backend.chunked_lm_head_logprob_keep).parameters
    ) == list(
        inspect.signature(det_backend.chunked_lm_head_det_logprob_keep).parameters
    )
    for fn in (
        cake_backend.chunked_lm_head_logprob,
        cake_backend.forward_logprob,
        cake_backend.backward_logprob,
    ):
        params = inspect.signature(fn).parameters
        assert "logprob_keep" not in params and "logits" not in params, fn.__name__
    assert "logits" not in {
        f.name for f in dataclasses.fields(cake_backend.ForwardResult)
    }
    param = inspect.signature(cake_backend.prepare_lm_head_loss).parameters[
        "logprob_keep"
    ]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is False
    inp = _host_inputs(37)
    with pytest.raises(ValueError, match="log-probability"):
        cake_backend.prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=16,
            backend="reference",
            logprob_keep=True,
            **ND,
        )
    with pytest.raises(ValueError, match="cake backend"):
        cake_backend.prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=16,
            entry="logprob",
            backend="reference",
            logprob_keep=True,
            **ND,
        )
    with pytest.raises(ValueError, match="cake backend"):
        cake_backend.forward_logprob_keep(
            inp.X, inp.W, inp.labels, chunk_size=16, backend="reference", **ND
        )
    with pytest.raises(ValueError, match="cake backend"):
        cake_backend.backward_logprob_keep(
            inp.X,
            inp.W,
            inp.labels,
            torch.zeros(37, dtype=torch.float32, device=DEV),
            inp.dlogp,
            (inp.X,),
            chunk_size=16,
            backend="reference",
            **ND,
        )
    # the reference backend: the default entry, the same outputs
    ref = _run(inp, 16, entry="logprob")
    kept = _run(inp, 16, entry="logprob", keep=True)
    for key in ("logp", "dX", "dW"):
        assert _same_bits(kept[key], ref[key]), key
    # where no backward can follow, the default entry (the label-column forward); else the kept Function
    calls = []

    def default_spy(*args, **kw):
        calls.append(("default", kw))
        return torch.zeros(())

    def keep_spy(*args):
        calls.append(("keep", args))
        return torch.zeros(())

    monkeypatch.setattr(cake_backend, "chunked_lm_head_logprob", default_spy)
    monkeypatch.setattr(
        cake_backend.ChunkedLmHeadLogprobKeepFunction, "apply", keep_spy
    )
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    X0, W0 = inp.X.detach(), inp.W.detach()
    cases = (
        ("grad", X, W, contextlib.nullcontext, "cake", "keep"),
        ("X only", X, W0, contextlib.nullcontext, "cake", "keep"),
        ("W only", X0, W, contextlib.nullcontext, "cake", "keep"),
        ("no_grad", X, W, torch.no_grad, "cake", "default"),
        ("inference_mode", X, W, torch.inference_mode, "cake", "default"),
        ("frozen", X0, W0, contextlib.nullcontext, "cake", "default"),
        ("reference", X, W, contextlib.nullcontext, "reference", "default"),
    )
    for tag, x, w, ctx, backend, expect in cases:
        calls.clear()
        with ctx():
            cake_backend.chunked_lm_head_logprob_keep(
                x, w, inp.labels, chunk_size=16, backend=backend, **ND
            )
        assert [c[0] for c in calls] == [expect], tag
        if expect == "default":
            assert calls[0][1]["backend"] == backend, tag
            assert calls[0][1]["chunk_size"] == 16, tag
            assert calls[0][1]["deterministic"] is False, tag
        else:
            assert calls[0][1][4] is False, tag  # the mode reaches the Function
    monkeypatch.undo()
    # the opt-in form
    # the resolver (every entry's) rejects it; the log-probability entry maps that rejection to the kept form
    with pytest.raises(ValueError, match="memory_mode must be"):
        opt_in._resolve_memory_mode("keep_logits")
    assert opt_in._resolve_memory_mode(None) == "fast"
    assert opt_in.KEEP_LOGITS == det_opt_in.KEEP_LOGITS == "keep_logits"
    assert opt_in.MEMORY_MODES == det_opt_in.MEMORY_MODES
    assert not hasattr(opt_in, "_resolve_logprob_memory_mode")
    with pytest.raises(ValueError, match="memory_mode must be"):
        _run(inp, 16, entry="loss", memory_mode="keep_logits")
    with pytest.raises(ValueError, match="memory_mode must be"):
        _run(inp, 16, entry="logprob", memory_mode="keep")
    n = int(inp.valid.sum())
    with pytest.raises(ValueError, match="num_valid_rows"):
        _run(inp, 16, entry="logprob", memory_mode="keep_logits", num_valid_rows=n)
    with pytest.raises(ValueError, match="cake backend"):
        _run(inp, 16, entry="logprob", memory_mode="keep_logits")
    seen = {}

    def spy(*args, **kw):
        seen.update(kw)
        return torch.zeros(())

    monkeypatch.setattr(opt_in._main, "chunked_lm_head_logprob_keep", spy)
    opt_in.chunked_lm_head_logprob(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=16,
        backend="cake",
        memory_mode="keep_logits",
        **ND,
    )
    assert seen["backend"] == "cake" and seen["chunk_size"] == 16
    assert seen["deterministic"] is False
    assert set(seen) <= set(default_params)


def test_device_logprob_keep_is_bitwise(glm_weight):
    """The kept logits against the default log-probability path on the generated program: the kept autograd entry
    (``chunked_lm_head_logprob_keep``, first call and remembered binding; also through the opt-in
    ``memory_mode="keep_logits"``) and the explicit pair (``forward_logprob_keep`` keeps one BF16 ``[rows_c, V]`` view
    per chunk, ``backward_logprob_keep`` on the forward's row index and on a recount) give bitwise the same ``logp`` /
    ``dW`` -- and ``dX`` wherever the call runs one dX slice per chunk (pinned or by the rule; else the same products
    in the unordered K-slice reduction, held to the budgets) -- over one-, two- and many-chunk calls, compacted and
    uncompacted calls, every row valid, the hidden valid-row count on and off, the dW side stream, the unfused dW
    cast and dX finalize, labels at and beyond the last vocabulary column and an ``X`` with a padded leading stride;
    the kept bindings are their own (``_KeptBinding``, keys ending in ``"keep"``) and the kept backward bindings
    launch no logits GEMM; with one trainable input (a frozen ``X``: no ``dX``; a frozen ``W``: no ``dW`` and no
    chunk row gather) the kept path gives the default path's gradient."""
    _require_program(entry="logprob")
    V = int(glm_weight.shape[0])
    edge_labels = [-1, -7, V - 1, 0, V, V + 5, 2 * V]
    cases = [
        # (T, chunk, ignored fraction, switches, X pitch padding, label edits)
        (1500, 4096, 0.05, {}, 0, False),
        (1500, 4096, 0.0, {}, 0, False),
        (1500, 4096, 0.05, dict(compact_rows=False), 0, False),
        (6000, 4096, 0.05, {}, 0, False),
        (8700, 4096, 0.05, {}, 0, False),
        (8700, 4096, 0.05, dict(gemm_tuning=ONE_SLICE), 0, False),
        (8700, 4096, 0.05, dict(hidden_count=False), 0, False),
        (8700, 4096, 0.0, {}, 0, False),
        (3100, 1024, 0.5, dict(dw_stream=True), 0, False),
        (1500, 4096, 0.05, {}, 8, False),
        (2500, 2048, 0.05, dict(compact_rows=False), 0, True),
        (5000, 2048, 0.05, dict(fuse_dw_cast=False, dx_finalize=False), 0, True),
        (
            5000,
            2048,
            0.05,
            dict(fuse_dw_cast=False, dx_finalize=False, gemm_tuning=ONE_SLICE),
            0,
            True,
        ),
    ]
    logits_gemm = {
        s
        for s in cake_backend.GEMM_STAGES
        if cake_backend.base_stage(s) in cake_backend.LOGITS_BASES
    }
    for T, C, ignore, switches, ld_pad, edit in cases:
        inp = make_inputs(
            T,
            seed=SEED + 90 + T,
            device=CUDA,
            W=glm_weight,
            ignore_frac=ignore,
            ld_pad=ld_pad,
        )
        if edit:
            labels = inp.labels.clone()
            labels[: len(edge_labels)] = torch.tensor(edge_labels, device=CUDA)
            inp = dataclasses.replace(inp, labels=labels)
        case = (T, C, ignore, switches, ld_pad, edit)
        with _cache(True) as cache:
            cache.clear()
            base = [
                _run(inp, C, entry="logprob", backend="cake", **switches)
                for _ in range(2)
            ]
            keep = [
                _run(inp, C, entry="logprob", backend="cake", keep=True, **switches)
                for _ in range(2)
            ]
            forms = {
                (key[0], b.plan.logprob_keep) for key, b in cache._bindings.items()
            }
            for key, b in cache._bindings.items():
                kept_binding = isinstance(b, cake_backend._KeptBinding)
                assert kept_binding == b.plan.logprob_keep == (key[-1] == "keep"), case
            for b in cache._bindings.values():
                if b.plan.logprob_keep and b.backward_order:
                    assert not {k[0] for k in b.backward_order} & logits_gemm, case
            (plan,) = [
                b.plan
                for key, b in cache._bindings.items()
                if key[0] == "bwd" and b.plan.logprob_keep
            ]
        assert forms == {
            ("fwd", True),
            ("fwd", False),
            ("bwd", True),
            ("bwd", False),
        }, (case, forms)
        mode = _run(
            inp,
            C,
            entry="logprob",
            backend="cake",
            memory_mode="keep_logits",
            **switches,
        )
        for key in ("logp", "dW"):
            for i, out in enumerate((*keep, mode)):
                assert _same_bits(out[key], base[0][key]), (case, key, i)
            assert _same_bits(base[1][key], base[0][key]), (case, key)
        for i, out in enumerate((*keep, mode)):
            _same_dx(out["dX"], base[0], inp, plan, (case, i))
        fwd_kw = {
            k: v
            for k, v in switches.items()
            if k in ("compact_rows", "hidden_count", "gemm_tuning")
        }
        dlogp = torch.where(inp.valid, inp.dlogp, torch.zeros_like(inp.dlogp))
        bwd_kw = {k: v for k, v in switches.items() if k != "hidden_count"}
        for tag in ("index", "recount"):
            fr = cake_backend.forward_logprob_keep(
                inp.X, inp.W, inp.labels, chunk_size=C, **fwd_kw, **ND
            )
            rows = int((inp.labels >= 0).sum()) if fr.row_index is not None else T
            assert len(fr.logits) == len(cake_backend.plan_chunks(rows, C)), case
            assert [int(v.shape[0]) for v in fr.logits] == [
                r for _, r in cake_backend.plan_chunks(rows, C)
            ], case
            assert all(v.dtype == torch.bfloat16 and v.shape[1] == V for v in fr.logits)
            assert _same_bits(fr.logp, base[0]["logp"]), case
            extra = dict(row_index=fr.row_index) if tag == "index" else {}
            dx, dw = cake_backend.backward_logprob_keep(
                inp.X,
                inp.W,
                inp.labels,
                fr.lse,
                dlogp,
                fr.logits,
                chunk_size=C,
                **bwd_kw,
                **extra,
                **ND,
            )
            _same_dx(dx, base[0], inp, plan, (case, tag))
            assert _same_bits(dw, base[0]["dW"]), (case, tag)
        assert torch.all(keep[0]["logp"][~inp.valid] == 0)
        assert torch.all(keep[0]["dX"][~inp.valid] == 0)
        del base, keep, mode, fr
    # one trainable input: the kept backward without dX (need_dx False) and without dW (need_dw False; no chunk row
    # gather, only the dlogp rows), first call and remembered binding
    for T, C, ignore, switches in (
        (1500, 4096, 0.05, {}),
        (8700, 4096, 0.05, {}),
        (8700, 4096, 0.05, dict(hidden_count=False)),
        (2500, 2048, 0.05, dict(compact_rows=False)),
        (3100, 1024, 0.5, dict(dw_stream=True)),
        (5000, 2048, 0.05, dict(fuse_dw_cast=False, dx_finalize=False)),
    ):
        inp = make_inputs(
            T, seed=SEED + 190 + T, device=CUDA, W=glm_weight, ignore_frac=ignore
        )
        for train_x, train_w in ((False, True), (True, False)):
            case = (T, C, ignore, switches, train_x, train_w)
            frozen = dict(train_x=train_x, train_w=train_w)
            with _cache(True) as cache:
                cache.clear()
                base = _run(
                    inp, C, entry="logprob", backend="cake", **frozen, **switches
                )
                keep = [
                    _run(
                        inp,
                        C,
                        entry="logprob",
                        backend="cake",
                        keep=True,
                        **frozen,
                        **switches,
                    )
                    for _ in range(2)
                ]
                kept = [
                    b
                    for key, b in cache._bindings.items()
                    if key[0] == "bwd" and isinstance(b, cake_backend._KeptBinding)
                ]
                assert len(kept) == 1, case
                plan = kept[0].plan
                assert (plan.need_dx, plan.need_dw) == (train_x, train_w), case
                order = kept[0].backward_order
                assert not {k[0] for k in order} & logits_gemm, case
                chunk_gathers = [
                    k for k in order if k[0] == "gather_rows" and isinstance(k[1], int)
                ]
                assert bool(chunk_gathers) == (train_w and plan.compact), case
                assert [k for k in order if k[0] == "row_grad"] == [
                    ("row_grad", i) for i in range(plan.num_chunks)
                ], case
            for out in keep:
                assert _same_bits(out["logp"], base["logp"]), case
                if train_x:
                    _same_dx(out["dX"], base, inp, plan, case)
                    assert out["dW"] is None, case
                else:
                    assert _same_bits(out["dW"], base["dW"]), case
                    assert out["dX"] is None, case
            del base, keep


def test_device_logprob_keep_lifetimes(glm_weight, monkeypatch):
    """Where the kept logits live and what they cost (one dX slice pinned: every gradient is compared by its bits): a
    kept-entry call no backward can follow (``torch.no_grad``, ``torch.inference_mode``, frozen inputs) is the default
    call -- the same launches (the label-column forward), bitwise the same ``logp``, no kept binding; the first
    backward through a retained graph spends the kept logits and a second one recomputes (both bitwise the default
    path's); a non-reentrant and a reentrant ``torch.utils.checkpoint`` give the default gradients; the bytes held
    from the forward to the backward grow by the kept bindings' ``memory["saved_logits_bytes"]`` (chunk 0 lives in the
    chunk buffer the default forward frees), the step peak by no more than that, and nothing stays allocated after the
    backward; a kept views tuple of the wrong plan is a ``ValueError``; the prepared runner of a kept plan does not
    launch it."""
    _require_program(entry="logprob")
    C, T = 4096, 8700
    inp = make_inputs(T, seed=SEED + 97, device=CUDA, W=glm_weight, ignore_frac=0.05)
    T_v = int(inp.valid.sum())
    V = int(glm_weight.shape[0])
    ref = _run(inp, C, entry="logprob", backend="cake", gemm_tuning=ONE_SLICE)

    def leaves(x=True, w=True):
        return inp.X.detach().requires_grad_(x), inp.W.detach().requires_grad_(w)

    def call(X, W, keep=True):
        fn = (
            cake_backend.chunked_lm_head_logprob_keep
            if keep
            else cake_backend.chunked_lm_head_logprob
        )
        return fn(X, W, inp.labels, chunk_size=C, gemm_tuning=ONE_SLICE, **ND)

    launched = []
    run_keys = cake_backend._run_keys

    def recording(plan, device_index, keys, run_key, *args, **kw):
        def run(key):
            launched.append(key)
            return run_key(key)

        return run_keys(plan, device_index, keys, run, *args, **kw)

    monkeypatch.setattr(cake_backend, "_run_keys", recording)
    lbl = {
        s
        for s in cake_backend.GEMM_STAGES
        if cake_backend.base_stage(s) == cake_backend.LOGPROB_LBL_STAGE
    }
    with _cache(True) as cache:
        # no backward can follow: the default call (its launches, the label-column forward), nothing kept
        for tag, ctx, x, w in (
            ("no_grad", torch.no_grad, True, True),
            ("inference_mode", torch.inference_mode, True, True),
            ("frozen", contextlib.nullcontext, False, False),
        ):
            cache.clear()
            X, W = leaves(x, w)
            runs = {}
            # the second pair: remembered bindings
            for keep in (False, True, False, True):
                launched.clear()
                with ctx():
                    lp = call(X, W, keep=keep)
                torch.cuda.synchronize()
                assert _same_bits(lp, ref["logp"]), (tag, keep)
                assert lp.grad_fn is None, (tag, keep)
                runs[keep] = list(launched)
            assert runs[True] == runs[False] and runs[True], tag
            assert any(k[0] in lbl for k in runs[True]) == any(
                b.plan.logprob_lbl for b in cache._bindings.values()
            ), tag
            assert not any(
                isinstance(b, cake_backend._KeptBinding)
                for b in cache._bindings.values()
            ), tag
            assert all(key[-1] != "keep" for key in cache._bindings), tag
        # one trainable input: the kept backward without dX (no X gradient) or without dW (no chunk gather)
        for x, w in ((False, True), (True, False)):
            base = _run(
                inp,
                C,
                entry="logprob",
                backend="cake",
                train_x=x,
                train_w=w,
                gemm_tuning=ONE_SLICE,
            )
            X, W = leaves(x, w)
            lp = call(X, W)
            assert lp.grad_fn.kept is True
            lp.backward(inp.dlogp)
            assert _same_bits(lp.detach(), ref["logp"]), (x, w)
            if x:
                assert _same_bits(X.grad, base["dX"]) and W.grad is None
            else:
                assert _same_bits(W.grad, base["dW"]) and X.grad is None
            del lp, base
        # a retained graph: the first backward spends the kept logits, the second recomputes
        cache.clear()
        X, W = leaves()
        lp = call(X, W)
        node = lp.grad_fn
        assert len(node.saved_tensors) == 5 + len(cake_backend.plan_chunks(T_v, C))
        assert node.kept is True  # kept, not yet spent
        lp.backward(inp.dlogp, retain_graph=True)
        assert node.kept is False  # spent
        assert _same_bits(X.grad, ref["dX"]) and _same_bits(W.grad, ref["dW"])
        X.grad = W.grad = None
        lp.backward(inp.dlogp)
        assert _same_bits(X.grad, ref["dX"]) and _same_bits(W.grad, ref["dW"])
        del lp, node
        # checkpointed: both the non-reentrant and the reentrant form
        for reentrant in (False, True):
            X, W = leaves()
            lp = torch.utils.checkpoint.checkpoint(call, X, W, use_reentrant=reentrant)
            lp.backward(inp.dlogp)
            assert _same_bits(lp.detach(), ref["logp"]), reentrant
            assert _same_bits(X.grad, ref["dX"]), reentrant
            assert _same_bits(W.grad, ref["dW"]), reentrant
            del lp
        monkeypatch.undo()
        # memory, after both forms are bound
        for _ in range(2):
            _run(inp, C, entry="logprob", backend="cake", gemm_tuning=ONE_SLICE)
            _run(
                inp,
                C,
                entry="logprob",
                backend="cake",
                keep=True,
                gemm_tuning=ONE_SLICE,
            )

        def step(keep):
            X, W = leaves()
            gc.collect()
            torch.cuda.synchronize()
            start = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            lp = call(X, W, keep=keep)
            torch.cuda.synchronize()
            held = torch.cuda.memory_allocated() - start
            lp.backward(inp.dlogp)
            del lp
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - start
            grads = sum(t.numel() * t.element_size() for t in (X.grad, W.grad))
            left = torch.cuda.memory_allocated() - start - grads
            return held, peak, left

        held0, peak0, left0 = step(False)
        held1, peak1, left1 = step(True)
        reports = [
            b.memory
            for b in cache._bindings.values()
            if isinstance(b, cake_backend._KeptBinding)
        ]
        assert reports and all(r["saved_logits_bytes"] == T_v * V * 2 for r in reports)
        assert not any(
            "saved_logits_bytes" in b.memory
            for b in cache._bindings.values()
            if not isinstance(b, cake_backend._KeptBinding)
        )
    saved = T_v * V * 2
    slack = 8 << 20  # allocator rounding of a few tensors
    print(
        f"\nkept logits T={T}: held {held1 - held0:+d} B (saved {saved}), peak {peak1 - peak0:+d} B, "
        f"left {left0} / {left1} B"
    )
    assert abs((held1 - held0) - saved) <= slack, (held0, held1, saved)
    assert peak1 - peak0 <= saved + slack, (peak0, peak1, saved)
    assert abs(left1 - left0) <= slack and left1 <= slack, (left0, left1)
    # a kept views tuple that does not match the plan
    kw = dict(chunk_size=C, gemm_tuning=ONE_SLICE, **ND)
    fr = cake_backend.forward_logprob_keep(inp.X, inp.W, inp.labels, **kw)
    dlogp = torch.where(inp.valid, inp.dlogp, torch.zeros_like(inp.dlogp))
    for bad in (fr.logits[:-1], (fr.logits[0].view(torch.int16),) + fr.logits[1:]):
        with pytest.raises(ValueError, match="kept chunk view"):
            cake_backend.backward_logprob_keep(
                inp.X,
                inp.W,
                inp.labels,
                fr.lse,
                dlogp,
                bad,
                row_index=fr.row_index,
                **kw,
            )
    # the explicit pair: one backward per kept views tuple (row_grad overwrote them), a second is a ValueError
    fr = cake_backend.forward_logprob_keep(inp.X, inp.W, inp.labels, **kw)
    dx, dw = cake_backend.backward_logprob_keep(
        inp.X, inp.W, inp.labels, fr.lse, dlogp, fr.logits, row_index=fr.row_index, **kw
    )
    assert _same_bits(dx, ref["dX"]) and _same_bits(dw, ref["dW"])
    for again in (fr.logits, tuple(fr.logits), list(fr.logits)):
        with pytest.raises(ValueError, match="consumed by an earlier backward"):
            cake_backend.backward_logprob_keep(
                inp.X,
                inp.W,
                inp.labels,
                fr.lse,
                dlogp,
                again,
                row_index=fr.row_index,
                **kw,
            )
    del fr, dx, dw
    runner = cake_backend.prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        entry="logprob",
        need_dx=False,
        need_dw=False,
        logprob_keep=True,
        **ND,
    )
    assert runner.plan.logprob_keep
    with pytest.raises(NotImplementedError, match="eager entry points"):
        runner.forward()


# ---------------------------------------------------------------------------
# Zero-gradient row skip of the log-probability backward
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _zero_dlogp_floor(rows):
    """Run under the zero-gradient row skip's floor ``rows`` (:func:`cake_backend.set_zero_dlogp_min_rows`; ``None``:
    never), then restore the previous."""
    previous = cake_backend.set_zero_dlogp_min_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_zero_dlogp_min_rows(previous)


def _with_zero_dlogp(inp, zero, *, negative_zero=True):
    """``inp`` with ``dlogp`` 0 on ``zero`` (``-0.0`` on every other zero row with ``negative_zero``)."""
    d = inp.dlogp.clone()
    d[zero] = 0.0
    if negative_zero:
        rows = zero.nonzero().squeeze(1)
        d[rows[::2]] = -0.0
    return dataclasses.replace(inp, dlogp=d.contiguous())


def _zero_dlogp_set(inp, kind):
    """:func:`_zero_weight_set` plus ``"single"`` (every valid row zero but one in the middle) and ``"last"`` (every
    valid row zero but those of the last 4096-row chunk of the valid rows)."""
    valid = inp.labels >= 0
    rows = valid.nonzero().squeeze(1)
    if kind == "single":
        zero = valid.clone()
        zero[rows[rows.numel() // 2]] = False
        return zero
    if kind == "last":
        zero = valid.clone()
        zero[rows[(rows.numel() - 1) // 4096 * 4096 :]] = False
        return zero
    return _zero_weight_set(inp, kind)


def _logprob_call(inp, C=4096, *, package="nd"):
    """One autograd log-probability call (forward + ``backward(dlogp)``) of ``package`` on the generated programs."""
    return _run(inp, C, package=package, entry="logprob", backend="cake")


def test_zero_dlogp_partition_orders_the_live_rows_first():
    """``zero_dlogp_partition``: the valid rows of a nonzero ``dlogp`` (NaN included) first, then the zero-gradient
    rows (``+0`` and ``-0``), each group in ascending order; ``lse`` in that order; the inclusive scan of the live rows
    over ``[T]``; the per-chunk live counts, at least one in the first chunk."""
    labels = torch.tensor([3, -100, 5, 7, 1, 2, -100, 4, 0], dtype=torch.int64)
    d = torch.tensor([1.0, 0.0, 0.0, -0.0, float("nan"), -2.0, 3.0, 0.0, 0.5])
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.arange(7, dtype=torch.float32)
    order, lse_o, scan, counts = cake_backend.zero_dlogp_partition(
        labels, d, index, lse, 4
    )
    assert order.tolist() == [0, 4, 5, 8, 2, 3, 7]
    assert lse_o.tolist() == [0.0, 3.0, 4.0, 6.0, 1.0, 2.0, 5.0]
    assert scan.dtype == torch.int32
    assert scan.tolist() == [1, 1, 1, 1, 2, 3, 3, 3, 4]
    assert counts.dtype == torch.int32 and counts.tolist() == [4, 0]
    order, _, scan, counts = cake_backend.zero_dlogp_partition(
        labels, torch.zeros(9), index, lse, 4
    )
    assert order.tolist() == index.tolist() and int(scan[-1]) == 0
    # the first chunk's weight-gradient GEMM stores one K step (of zeros): its count is at least one
    assert counts.tolist() == [1, 0]
    order, lse_o, _, counts = cake_backend.zero_dlogp_partition(
        labels, torch.ones(9), index, lse, 3
    )
    assert order.tolist() == index.tolist() and lse_o.tolist() == lse.tolist()
    assert counts.tolist() == [3, 3, 1]


def test_zero_dlogp_floor_switch():
    """``set_zero_dlogp_min_rows`` returns the previous floor; ``None`` turns the skip off; a negative or non-integer
    floor is refused."""
    default = cake_backend.zero_dlogp_min_rows()
    assert default == cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT
    with _zero_dlogp_floor(8192):
        assert cake_backend.zero_dlogp_min_rows() == 8192
        with _zero_dlogp_floor(None):
            assert cake_backend.zero_dlogp_min_rows() is None
        assert cake_backend.zero_dlogp_min_rows() == 8192
    assert cake_backend.zero_dlogp_min_rows() == default
    for bad in (-1, 1.5, True, "8192"):
        with pytest.raises((TypeError, ValueError)):
            cake_backend.set_zero_dlogp_min_rows(bad)
    assert cake_backend.zero_dlogp_min_rows() == default


def test_dcnt_stage_names_round_trip():
    """The ``_dcnt`` forms: the dz arm, the K-sliced dX reduction (not its swapped tile; the one-slice plain store has
    none), the weight-gradient accumulate (plain and blocked) and the bf16 fused cast (not its TMA store); every
    registered ``_dcnt`` name parses back to its base and knobs."""
    sv = cake_backend.stage_variant
    assert sv("gemm_logits_dz", dcnt=True) == cake_backend.LOGPROB_DZ_DCNT_STAGE
    assert sv("gemm_dx_red", tile_n=256, dcnt=True) == "gemm_dx_red_tn256_dcnt"
    assert (
        sv("gemm_dx_red", stages=3, split=True, dcnt=True)
        == "gemm_dx_red_st3_split_dcnt"
    )
    assert sv("gemm_dw_acc", blocked=True, dcnt=True) == "gemm_dw_acc_gn_dcnt"
    assert sv("gemm_dw_cast_bf16", dcnt=True) == "gemm_dw_cast_bf16_dcnt"
    for base, kw in (
        ("gemm_dx_red", dict(swab=True)),
        ("gemm_dw_cast_bf16", dict(epi_store="tma")),
        ("gemm_dw_acc", dict(epi_store="tma")),
        ("gemm_dx", {}),
        ("gemm_logits", {}),
    ):
        with pytest.raises(ValueError):
            sv(base, dcnt=True, **kw)
    assert cake_backend.dx_stage(1, dcnt=True) == "gemm_dx"
    assert cake_backend.dx_stage(3, dcnt=True) == "gemm_dx_red_dcnt"
    counted = [stage for stage in cake_jit.STAGES if stage.endswith("_dcnt")]
    assert len(counted) == 9
    for stage in counted:
        base, knobs = cake_backend.parse_stage(stage)
        assert base in cake_backend.DCNT_BASES and knobs["dcnt"]
        assert sv(base, **knobs) == stage
        assert stage.removesuffix("_dcnt") in cake_jit.STAGES


def test_plan_runs_the_counted_forms():
    """``make_plan(dcnt=True)`` (a log-probability plan with the dz arm on a registered architecture): the recompute,
    the K-sliced dX chunks and the weight-gradient GEMMs take their ``_dcnt`` forms, a one-slice dX chunk its plain
    store; without the dz arm, or with a chunk on a form without a counted one, the plan keeps the shipped forms; the
    binding key tells the two apart."""
    g = cake_backend.Geometry.from_record(
        {"geometry": {"hidden": 6144, "vocab": 154880}}
    )
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        dx_finalize=True,
        logprob_dz=True,
    )
    # 12000 rows: three chunks, the last of 3808 rows (above the TMA fused cast's last-chunk bound)
    problem = _keep_problem(12000, 6144, 154880, 4096, entry="logprob")
    counted = 0
    for arch in ("sm_100a", "sm_103a"):
        kw = dict(common, arch=arch)
        plain = cake_backend.make_plan(problem, valid_rows=12000, **kw)
        plan = cake_backend.make_plan(problem, valid_rows=12000, dcnt=True, **kw)
        if any(
            v.dx_swab or v.dw_epi_store not in (None, "redsm") or v.dw_cast_tma
            for v in plan.variants
        ):
            assert not plan.dcnt and plan.stages == plain.stages
            continue
        assert plan.dcnt and not plain.dcnt
        counted += 1
        assert plan.chunks == plain.chunks and plan.dx_slices == plain.dx_slices
        for i in range(plan.num_chunks):
            assert plan.recompute_stage(i) == "gemm_logits_dz_dcnt"
            assert plan.dx_stage_of(i).endswith("_dcnt") == (
                plan.dx_slices_of(i) > 1
            ), plan.dx_stage_of(i)
            assert plan.dx_stage_of(i).removesuffix("_dcnt") == plain.dx_stage_of(i)
        assert all(s.endswith("_dcnt") for s in plan.stages if s.startswith("gemm_dw"))
        assert {s.removesuffix("_dcnt") for s in plan.stages} == set(plain.stages)
        assert not cake_backend.make_plan(
            problem, valid_rows=12000, dcnt=True, **dict(kw, logprob_dz=False)
        ).dcnt
    assert counted, "no architecture ran the counted forms"
    assert not cake_backend.make_plan(
        problem, valid_rows=12000, dcnt=True, **common
    ).dcnt  # the reference path (no arch)
    loss = _keep_problem(12000, 6144, 154880, 4096)
    assert not cake_backend.make_plan(
        loss,
        valid_rows=12000,
        dcnt=True,
        **dict(common, logprob_dz=False, arch="sm_100a"),
    ).dcnt
    key = functools.partial(
        cake_backend.logprob_backward_binding_key,
        *(torch.empty(s) for s in ((8, 64), (128, 64), (8,), (8,), (8,))),
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
    )
    assert key(zero_dlogp=True) != key() == key(zero_dlogp=False)


ZERO_DLOGP_DEVICE_CASES = [
    (4097, "none"),
    (4097, "one"),
    (4097, "single"),
    (8117, ("random", 0.166)),
    (8117, ("random", 0.5)),
    (12289, ("group", 0.5)),
    (12289, "last"),
    (12289, "all"),
    (2048, ("random", 0.3)),
]


@pytest.mark.parametrize("T, kind", ZERO_DLOGP_DEVICE_CASES, ids=str)
def test_device_zero_dlogp_rows_skip(glm_weight, T, kind):
    """The zero-gradient row skip on the device against the same backward without it (``set_zero_dlogp_min_rows(None)``)
    and the FP64 oracle: ``logp`` the same bits; the deterministic suite's gates; the zero rows' and the ignored rows'
    ``dX`` rows exact ``+0``; the other ``dX`` rows equal to the backward without the skip within one BF16 rounding
    (the K-sliced reductions add in any order); a second call from the remembered binding gives the first's ``dW``
    within the FP32 reduction order.  Every valid row zero: zero gradients."""
    _require_program(entry="logprob")
    base = _device_inputs(T, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, kind)
    inp = _with_zero_dlogp(base, zero)
    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(0):
        nd = _logprob_call(inp)
        again = _logprob_call(inp)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    valid = inp.labels >= 0
    live = valid & ~zero
    assert _same_bits(nd["logp"], plain["logp"])
    for run in (nd, again):
        assert not bool(_bits(run["dX"][zero | ~valid]).any())
    if not bool(live.any()):
        assert not bool(_bits(nd["dX"]).any()) and not bool(nd["dW"].float().any())
        return
    scale = float(plain["dX"][live].float().abs().max())
    dx_diff = float((nd["dX"][live].float() - plain["dX"][live].float()).abs().max())
    assert dx_diff <= scale * 2**-7, (dx_diff, scale)
    _check_against_references(nd, inp, entry="logprob", ceiling=True)
    _check_against_references(again, inp, entry="logprob", ceiling=True)
    dw_rel = float(
        (nd["dW"].float() - plain["dW"].float()).norm() / plain["dW"].float().norm()
    )
    print(
        f"\nzero-dlogp skip T={T} {kind} zero {int(zero.sum())}/{int(valid.sum())}: dW rel-L2 vs no skip "
        f"{dw_rel:.2e}, dX max|skip - no skip| {dx_diff:.2e}"
    )


def test_device_zero_dlogp_rows_repeat(glm_weight):
    """``REPEATS`` runs with 16.6 % zero-gradient rows (T 12289, chunks of 4096), each after poisoning the free
    allocator memory: every run passes the deterministic suite's gates and keeps the zero rows' ``dX`` exact ``+0``; the
    worst error per metric is printed beside the deterministic package's on the same inputs."""
    _require_program(entry="logprob")
    base = _device_inputs(12289, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.166))
    inp = _with_zero_dlogp(base, zero)
    oracle = reference_fp64(inp, entry="logprob")
    b0 = error_report(reference_unchunked(inp, entry="logprob"), oracle, inp.labels)
    det = error_report(_logprob_call(inp, package="det"), oracle, inp.labels)
    worst = {}
    with _zero_dlogp_floor(0):
        for _ in range(REPEATS):
            _poison_allocator(1 << 30)
            run = _logprob_call(inp)
            assert not bool(_bits(run["dX"][zero]).any())
            errors = error_report(run, oracle, inp.labels)
            assert not errors["nan"] and errors["logp_ignored_zero"]
            assert errors["dX_ignored_zero"]
            for key, tiny in GATE_TINY.items():
                if key in errors:
                    assert errors[key] <= GATE_MARGIN * b0[key] + tiny, key
                    worst[key] = max(worst.get(key, 0.0), errors[key])
    print(
        "\nzero-dlogp skip repeat worst: "
        + ", ".join(
            f"{k} {v:.3e} (det {det.get(k, float('nan')):.3e}, b0 {b0[k]:.3e})"
            for k, v in worst.items()
        )
    )


def test_device_zero_dlogp_floor_keeps_small_calls_unskipped(glm_weight):
    """A backward of fewer valid rows than the floor runs without the skip: ``logp`` / ``dW`` the bits of the backward
    with the skip off, the zero rows' ``dX`` rows zeros either way."""
    _require_program(entry="logprob")
    base = _device_inputs(4097, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero, negative_zero=False)
    with _zero_dlogp_floor(8192):
        nd = _logprob_call(inp)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    for key in ("logp", "dW"):
        assert _same_bits(nd[key], plain[key]), key
    assert not bool(nd["dX"][zero].float().any())


def test_device_zero_dlogp_rows_keep_a_nan_upstream_row_live(glm_weight):
    """A NaN ``dlogp`` row is a live row: NaN in its ``dX`` row and in ``dW``, as without the skip."""
    _require_program(entry="logprob")
    base = _device_inputs(4097, "ce", W=glm_weight)
    rows = (base.labels >= 0).nonzero().squeeze(1)
    zero = torch.zeros_like(base.labels, dtype=torch.bool)
    zero[rows[1:300]] = True
    inp = _with_zero_dlogp(base, zero)
    d = inp.dlogp.clone()
    d[rows[0]] = float("nan")
    with _zero_dlogp_floor(0):
        nd = _logprob_call(dataclasses.replace(inp, dlogp=d))
    assert bool(torch.isnan(nd["dX"][rows[0]].float()).all())
    assert bool(torch.isnan(nd["dW"].float()).any())
    assert not bool(_bits(nd["dX"][zero]).any())


def test_device_zero_dlogp_skip_adds_no_synchronization(glm_weight):
    """The backward with the skip runs under PyTorch's synchronization debug mode ``"error"``, binding (first call) and
    from the remembered binding (second call)."""
    _require_program(entry="logprob")
    base = _device_inputs(8117, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero)
    cake_backend.BINDING_CACHE.clear()

    def backward_under_error_mode():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logp = cake_backend.chunked_lm_head_logprob(
            X, W, inp.labels, chunk_size=4096, backend="cake", **ND
        )
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            logp.backward(inp.dlogp)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
        return X.grad

    with _zero_dlogp_floor(0):
        for _ in range(2):
            dx = backward_under_error_mode()
            assert not bool(_bits(dx[zero]).any())


def test_device_zero_dlogp_backward_cuda_graph(glm_weight):
    """The log-probability backward with the skip captured in a CUDA graph (after a warm-up that binds it on the
    capture stream): replays with other ``dlogp`` values -- other zero rows and live counts per chunk, none and every
    row zero -- written into the captured input give the eager backward's results on those values (``dW`` within
    the FP32 reduction order, the zero rows' ``dX`` exact ``+0``)."""
    _require_program(entry="logprob")
    base = _device_inputs(12289, "ce", W=glm_weight)
    kw = dict(chunk_size=4096, **ND)
    fr = cake_backend.forward_logprob(base.X, base.W, base.labels, **kw)
    torch.cuda.synchronize()
    assert fr.row_index is not None
    dlogp = base.dlogp.clone()

    def backward(d):
        return cake_backend.backward_logprob(
            base.X, base.W, base.labels, fr.lse, d, row_index=fr.row_index, **kw
        )

    with _zero_dlogp_floor(0):
        cake_backend.BINDING_CACHE.clear()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            backward(dlogp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dx_g, dw_g = backward(dlogp)
        valid = base.labels >= 0
        for kind in (("random", 0.166), ("group", 0.5), "none", "last", "all"):
            zero = _zero_dlogp_set(base, kind)
            new = _with_zero_dlogp(base, zero).dlogp
            dlogp.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            got_dx, got_dw = dx_g.clone(), dw_g.clone()
            want_dx, want_dw = backward(new.clone())
            torch.cuda.synchronize()
            assert not bool(_bits(got_dx[zero | ~valid]).any()), kind
            live = valid & ~zero
            if not bool(live.any()):
                assert not bool(got_dw.float().any()), kind
                continue
            scale = float(want_dx[live].float().abs().max())
            assert (
                float((got_dx[live].float() - want_dx[live].float()).abs().max())
                <= scale * 2**-7
            ), kind
            dw_rel = float(
                (got_dw.float() - want_dw.float()).norm() / want_dw.float().norm()
            )
            assert dw_rel <= 1e-3, (kind, dw_rel)
