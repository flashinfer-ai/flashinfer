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
``flashinfer.experimental.cake.cake_lm_head_nondet_train`` (the backend of ``flashinfer.cake_lm_head_nondet_train``).

The package is the non-deterministic counterpart of the deterministic package
``flashinfer.experimental.cake.cake_lm_head_det_train`` (``test_cake_lm_head_det_train.py``): the same
precision boundaries, with the dX GEMM's K-slice partials of a chunk of more than one slice summed in an
ordered FP32 reduction: each slice stores its tile into a slab and the slab sum launched right after the GEMM
stores the slices' sum, taken in slice order, into the FP32 accumulator rows (a one-slice chunk keeps the
deterministic plain store).
Its outputs are held to the deterministic suite's accuracy budgets against the FP64 oracle and the
unchunked PyTorch path (the same ``GATE_MARGIN`` / ``GATE_TINY`` / ``PRODUCTION_B0``, checked
against that suite's source below), over repeated runs (worst case), and compared with the
deterministic package on the same inputs: ``loss`` / ``logp`` / ``lse`` and ``dW`` keep their fixed
order and equal its bits; ``dX`` meets the budgets, repeats its own bits from run to run, and its
distance to the deterministic ``dX`` is reported.  A log-probability backward of more than one chunk runs
vocabulary-major by default (``cake_backend._set_logprob_vocab_major``): its ``dX`` / ``dW`` meet the same budgets
and repeat bit for bit; the tests of the chunk loop's own machinery and bits pin the chunk loop
(``logprob_chunk_loop``).  The host layer (no mode keyword, PyTorch's deterministic-algorithms alert, the reference
backend) runs anywhere; the device tests need an SM100 / SM103 / SM107 GPU and the package's
rendered programs.
"""

import ast
import contextlib
import copy
import dataclasses
import functools
import gc
import inspect
import json
import math
import os
import pickle
import re
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

from tests.experimental.test_helpers.cake_lm_head_loss_reference import (
    DEFAULT_H,
    DEFAULT_V,
    error_report,
    make_inputs,
    make_temperature,
    reference_fp64,
    reference_unchunked,
    rel_l2,
)
from tests.cake.test_helpers.cake_package import cake_package

# The package under test, named once through ``cake_package``; the deterministic package is the comparison arm.
pkg = cake_package("cake.cake_lm_head_nondet_train")
cake_backend = cake_package("cake.cake_lm_head_nondet_train", "cake_backend")
opt_in = cake_package("cake.cake_lm_head_nondet_train", "cake_backend_opt_in")
cake_jit = cake_package("cake.cake_lm_head_nondet_train", "cake_jit")
cake_build = cake_package("cake.cake_lm_head_nondet_train", "cake_build")
det_backend = cake_package("cake.cake_lm_head_det_train", "cake_backend")
det_opt_in = cake_package("cake.cake_lm_head_det_train", "cake_backend_opt_in")

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
# Seconds a two-thread test waits for a barrier or a thread before it fails
TIMEOUT_S = 120


def _det_suite_constants() -> dict:
    """``GATE_MARGIN`` / ``GATE_TINY`` / ``PRODUCTION_B0`` as the deterministic suite's source assigns them (read, not
    imported: importing that module would run its module-level package lookups)."""
    path = Path(__file__).with_name("test_cake_lm_head_det_train.py")
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


# A zero-weight floor above every call's row count: the split never applies
NO_SPLIT = 1 << 62


@contextlib.contextmanager
def _det_unsplit():
    """Run the deterministic package without its own zero-weight split (its floor raised past every call,
    ``det_backend.set_zero_weight_min_rows``), then restore its previous floor: its plain path, every valid row in the
    chunk loop."""
    previous = det_backend.set_zero_weight_min_rows(NO_SPLIT)
    try:
        yield
    finally:
        det_backend.set_zero_weight_min_rows(previous)


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
    """One forward + backward through the autograd entry of ``package`` (``"nd"``: this package; ``"det"``: the
    deterministic package) on ``backend``.  ``switches`` pass through
    (``memory_mode`` / ``num_valid_rows`` / ``check_num_valid_rows`` select the opt-in module); ``keep`` calls the
    log-probability entry with the kept logits (``cake_lm_head_nondet_train_logprob_keep`` /
    ``cake_lm_head_det_train_logprob_keep``); the log-probability entry also takes the inputs' ``temperature`` unless
    ``switches`` names one, and a call with ``temperature`` or ``check_temperature`` goes to the module's
    ``cake_lm_head_nondet_train_logprob_temperature`` / ``cake_lm_head_det_train_logprob_temperature``.  Returns ``loss`` /
    ``logp`` / ``dX`` / ``dW`` and the leaves."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    if entry != "loss" and inp.temperature is not None:
        switches.setdefault("temperature", inp.temperature)
    opt = bool(_OPT_IN_KEYWORDS & switches.keys())
    tau = bool(TEMPERATURE_KEYWORDS & switches.keys())
    if package == "nd":
        be = opt_in if opt else cake_backend
        loss_fn, logprob_fn = (
            be.cake_lm_head_nondet_train_loss,
            be.cake_lm_head_nondet_train_logprob_temperature
            if tau
            else be.cake_lm_head_nondet_train_logprob,
        )
    else:
        be = det_opt_in if opt else det_backend
        loss_fn, logprob_fn = (
            be.cake_lm_head_det_train_loss,
            be.cake_lm_head_det_train_logprob_temperature
            if tau
            else be.cake_lm_head_det_train_logprob,
        )
    if keep:
        logprob_fn = (
            cake_backend.cake_lm_head_nondet_train_logprob_keep
            if package == "nd"
            else det_backend.cake_lm_head_det_train_logprob_keep
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
            **switches,
        )
        if train_x or train_w:
            (loss if scale is None else loss * scale).backward()
    else:
        logp = logprob_fn(X, W, inp.labels, chunk_size=C, backend=backend, **switches)
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
        "CakeLMHeadNondetLossFunction": "CakeLMHeadDetLossFunction",
        "CakeLMHeadNondetLogprobFunction": "CakeLMHeadDetLogprobFunction",
        "cake_lm_head_nondet_train_loss": "cake_lm_head_det_train_loss",
        "cake_lm_head_nondet_train_logprob": "cake_lm_head_det_train_logprob",
        "load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module",
        "forward_loss": "forward_loss",
        "backward_loss": "backward_loss",
        "forward_logprob": "forward_logprob",
        "backward_logprob": "backward_logprob",
        "CakeLMHeadNondetLogprobKeepFunction": "CakeLMHeadDetLogprobKeepFunction",
        "cake_lm_head_nondet_train_logprob_keep": "cake_lm_head_det_train_logprob_keep",
        "forward_logprob_keep": "forward_logprob_keep",
        "backward_logprob_keep": "backward_logprob_keep",
        "KeptForwardResult": "KeptForwardResult",
        "kept_memory_report": "kept_memory_report",
        "check_kept_logits": "check_kept_logits",
        "prepare_lm_head_loss_temperature": "prepare_lm_head_det_loss_temperature",
        "CakeLMHeadNondetLogprobTemperatureFunction": "CakeLMHeadDetLogprobTemperatureFunction",
        "cake_lm_head_nondet_train_logprob_temperature": "cake_lm_head_det_train_logprob_temperature",
        "forward_logprob_temperature": "forward_logprob_temperature",
        "backward_logprob_temperature": "backward_logprob_temperature",
        "memory_report_temperature": "memory_report_temperature",
    },
    "cake_backend_opt_in": {
        "prepare_lm_head_loss": "prepare_lm_head_det_loss",
        "LmHeadLossRunner": "LmHeadDetLossRunner",
        "CakeLMHeadNondetLossFunction": "CakeLMHeadDetLossFunction",
        "CakeLMHeadNondetLogprobFunction": "CakeLMHeadDetLogprobFunction",
        "cake_lm_head_nondet_train_loss": "cake_lm_head_det_train_loss",
        "cake_lm_head_nondet_train_logprob": "cake_lm_head_det_train_logprob",
        "cake_lm_head_nondet_train_preload": "cake_lm_head_det_train_preload",
        "memory_plan": "memory_plan",
        "prepare_lm_head_loss_temperature": "prepare_lm_head_det_loss_temperature",
        "CakeLMHeadNondetLogprobTemperatureFunction": "CakeLMHeadDetLogprobTemperatureFunction",
        "cake_lm_head_nondet_train_logprob_temperature": "cake_lm_head_det_train_logprob_temperature",
        "forward_logprob_temperature": "forward_logprob_temperature",
        "backward_logprob_temperature": "backward_logprob_temperature",
    },
    "cake_support": {
        "CakeLMHeadNondetSupportFunction": "CakeLMHeadDetSupportFunction",
        "cake_lm_head_nondet_train_support_logprob": "cake_lm_head_det_train_support_logprob",
        "cake_lm_head_nondet_train_support_kl": "cake_lm_head_det_train_support_kl",
    },
    "cake_jit": {"load_cake_lm_head_loss_module": "load_cake_lm_head_det_loss_module"},
}


def test_names_pair_with_the_deterministic_names():
    """Every listed name of the package pairs with the deterministic package's name, is defined by this package (not
    the deterministic object), and no name of the package carries ``_det_`` or ``Det``."""
    for module_name, names in NAMES.items():
        nd = cake_package("cake.cake_lm_head_nondet_train", module_name)
        det = cake_package("cake.cake_lm_head_det_train", module_name)
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


def test_each_supported_architecture_has_its_record_and_libraries():
    """The supported compute capabilities name exactly the architectures of the registry's records (``(10, 7)``
    only with an ``sm_107a`` record), ``build.json`` builds exactly those architectures, and every registered stage
    of each record has its ``<library>_<arch>`` entry.  The ``sm_107a`` record registers the ``sm_103a`` record's
    stages and programs, the zero-gradient row skip's (the ``_dcnt`` forms and the partition kernel) and the cast at the
    last live chunk (the ``_lc`` forms) included."""
    assert cake_backend.SUPPORTED_COMPUTE_CAPABILITIES == {
        (10, 0): "sm_100a",
        (10, 3): "sm_103a",
        (10, 7): "sm_107a",
    }
    arches = set(cake_backend.SUPPORTED_COMPUTE_CAPABILITIES.values())
    by_arch = {record["arch"]: name for name, record in cake_jit.MODULES.items()}
    assert set(by_arch) == arches and len(by_arch) == len(cake_jit.MODULES)
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip(
            "the package's programs are not rendered in this checkout (no build.json)"
        )
    manifest = json.loads((PACKAGE_DIR / "build.json").read_text())
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert set(manifest["arch_flags"]) == arches
    assert {entry["arch"] for entry in entries.values()} == arches
    for arch, name in by_arch.items():
        stages = cake_jit.registered_stages(name)
        assert stages, name
        for stage in stages:
            library = cake_jit.library_name(name, stage)
            assert library.endswith("_" + arch), (name, stage)
            assert library in entries and entries[library]["arch"] == arch, (
                name,
                stage,
            )
    sm107, sm103 = by_arch["sm_107a"], by_arch["sm_103a"]
    for name in (sm103, sm107):
        assert cake_backend.LOGPROB_DZ_DCNT_STAGE in cake_jit.registered_stages(name)
        assert cake_backend.ZERO_DLOGP_PARTITION_STAGE in cake_jit.registered_stages(
            name
        )
    # every stage of sm_103a, the cast at the last live chunk (the ``_lc`` forms) included
    assert cake_jit.registered_stages(sm107) == cake_jit.registered_stages(sm103)
    assert any(
        stage.endswith("_lc_dcnt") for stage in cake_jit.registered_stages(sm107)
    )
    for stage in cake_jit.registered_stages(sm107):
        assert (
            cake_jit.MODULES[sm107][stage]["library"]
            == cake_jit.MODULES[sm103][stage]["library"]
        ), stage
    assert cake_jit.MODULES[sm107]["geometry"] == cake_jit.MODULES[sm103]["geometry"]


def test_sm107_takes_the_sm103_rule_tables_except_the_long_chunk_ring():
    """``sm_107a`` (R200, the ``sm_103a`` programs) takes the ``sm_103a`` entry of every per-architecture GEMM tile rule
    table except ``DX_LONG_CHUNK_STAGES`` and ``LOGITS_LONG_RASTER_GROUPS``: its chunks of ``>= DW_LONG_CHUNK_MIN_ROWS``
    rows keep the 512-wide dX tile's table ring, and its logits GEMM keeps the default raster height (32) at the chunks
    of ``>= LOGITS_LONG_RASTER_MIN_ROWS`` rows below ``RASTER_RULE_MIN_HIDDEN``.  At the same SM count (B300 148, R200
    212 and 216) its plans are the ``sm_103a`` plans but for that ring depth and that logits height (the C 4096 plans
    run the same stages), and its record registers every form they launch.  The zero-gradient
    row skip is not a GEMM tile rule: ``ZERO_DLOGP_MIN_ROWS_BY_ARCH`` gives ``sm_107a`` its own floor (1946 valid
    rows; ``sm_103a``: 3891), and ``DX_COUNTED_MIN_SLICES_BY_ARCH`` the ``sm_103a`` two-slice counted dX, so its
    counted plans take the ``sm_103a`` dX slices and its record registers every counted form they launch."""
    b = cake_backend
    for table in (
        b.RASTER_WIDE_GROUPS,
        b.DW_LONG_CHUNK_GROUPS,
        b.DW_BLOCK_GROUPS,
        b.DW_CAST_TMA_MAX_ROWS,
    ):
        assert table["sm_107a"] == table["sm_103a"], table
    for archs in (
        b.DX_SPLIT_ARCHES,
        b.DX_SPLIT_LONG_RING_ARCHES,
        b.DX_SPLIT_WHOLE_ARCHES,
        b.DX_SWAB_ARCHES,
    ):
        assert {"sm_103a", "sm_107a"} <= set(archs), archs
    for table in (b.DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES, b.DX_WIDE_MIN_EFF):
        assert "sm_103a" not in table and "sm_107a" not in table, table
    assert b.DX_LONG_CHUNK_STAGES == {"sm_100a": 3, "sm_103a": 3}
    # the logits GEMM's long-chunk raster: the default height on sm_107a, 16 on sm_103a
    assert b.LOGITS_LONG_RASTER_GROUPS["sm_103a"] == 16
    assert (
        b.LOGITS_LONG_RASTER_GROUPS["sm_107a"]
        == b.GEOMETRY_DEFAULTS["logits_group_m"]
        == 32
    )
    assert b.ZERO_DLOGP_MIN_ROWS_BY_ARCH["sm_107a"] == 1946
    assert (
        b.DX_COUNTED_MIN_SLICES_BY_ARCH["sm_107a"]
        == b.DX_COUNTED_MIN_SLICES_BY_ARCH["sm_103a"]
        == 2
    )
    assert b.ZERO_DLOGP_MIN_ROWS_DEFAULT == 4097
    sm107 = next(n for n, r in cake_jit.MODULES.items() if r["arch"] == "sm_107a")
    registered = set(cake_jit.registered_stages(sm107))
    g = b.Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        fuse_dw_cast=True,
        dx_finalize=True,
    )
    long_ring = 0
    for num_sms in (148, 212, 216):
        for T, C, entry in (
            (16231, 4096, "loss"),
            (16231, 4096, "logprob"),
            (65031, 4096, "logprob"),
            (16231, 8192, "loss"),
            (16231, 8192, "logprob"),
        ):
            case = (num_sms, T, C, entry)
            problem = _keep_problem(T, 6144, 154880, C, entry=entry)
            kw = dict(
                common,
                num_sms=num_sms,
                dx_resident=num_sms // 2,
                valid_rows=T,
                logprob_dz=entry == "logprob",
            )
            p103 = b.make_plan(problem, arch="sm_103a", **kw)
            p107 = b.make_plan(problem, arch="sm_107a", **kw)
            assert p107.chunks == p103.chunks, case
            assert p107.dx_slices == p103.dx_slices, case
            assert not any(v.dx_stages for v in p107.variants), case
            # the chunks of >= LOGITS_LONG_RASTER_MIN_ROWS rows: logits height 16 on sm_103a, 32 on sm_107a
            assert [v.logits_group_m for v in p107.variants] == [
                32 if rows >= b.LOGITS_LONG_RASTER_MIN_ROWS else None
                for _start, rows in p107.chunks
            ], case
            assert [v.logits_group_m for v in p103.variants] == [
                16 if rows >= b.LOGITS_LONG_RASTER_MIN_ROWS else None
                for _start, rows in p103.chunks
            ], case
            assert p107.variants == tuple(
                dataclasses.replace(v, dx_stages=None, logits_group_m=w.logits_group_m)
                for v, w in zip(p103.variants, p107.variants, strict=True)
            ), case
            if C <= b.DW_BLOCK_MAX_ROWS:
                assert p107.variants == tuple(
                    dataclasses.replace(v, logits_group_m=w.logits_group_m)
                    for v, w in zip(p103.variants, p107.variants, strict=True)
                ), case
                assert p107.stages == p103.stages, case
            long_ring += sum(
                v.dx_stages == b.DX_LONG_CHUNK_STAGES["sm_103a"] for v in p103.variants
            )
            assert not [s for s in p107.stages if "_st3" in s], case
            assert set(p107.stages) <= registered, (
                case,
                sorted(set(p107.stages) - registered),
            )
            if entry == "logprob":
                c103 = b.make_plan(problem, arch="sm_103a", dcnt=True, **kw)
                c107 = b.make_plan(problem, arch="sm_107a", dcnt=True, **kw)
                assert c107.dcnt == c103.dcnt, case
                assert c107.dx_slices == c103.dx_slices, case
                assert set(c107.stages) <= registered, (
                    case,
                    sorted(set(c107.stages) - registered),
                )
    assert long_ring, "no C 8192 chunk took the sm_103a long-chunk ring"


def _former_mode_calls(X, W, labels):
    """Every entry of the package that took ``deterministic`` before issue #8928, as ``name -> call(**mode)`` on host
    tensors (the reference backend; the ``cake`` backend for the kept-logits pair): ``call()`` runs the entry,
    ``call(deterministic=False)`` passes the former mode keyword."""
    common = dict(chunk_size=4, backend="reference")
    loss_kw = dict(common, loss_div=1.0)
    rows = labels.shape[0]
    return {
        "validate_lm_head_inputs": lambda **mode: cake_backend.validate_lm_head_inputs(
            X, W, labels, loss_div=1.0, **mode
        ),
        "forward_loss": lambda **mode: cake_backend.forward_loss(
            X, W, labels, **mode, **loss_kw
        ),
        "forward_logprob": lambda **mode: cake_backend.forward_logprob(
            X, W, labels, **mode, **common
        ),
        "backward_logprob": lambda **mode: cake_backend.backward_logprob(
            X,
            W,
            labels,
            torch.zeros(labels.shape[0]),
            torch.zeros(labels.shape[0]),
            **mode,
            **common,
        ),
        "prepare_lm_head_loss": lambda **mode: cake_backend.prepare_lm_head_loss(
            X, W, labels, **mode, **loss_kw
        ),
        "cake_lm_head_nondet_train_loss": lambda **mode: cake_backend.cake_lm_head_nondet_train_loss(
            X, W, labels, **mode, **loss_kw
        ),
        "cake_lm_head_nondet_train_logprob": lambda **mode: cake_backend.cake_lm_head_nondet_train_logprob(
            X, W, labels, **mode, **common
        ),
        "opt_in.forward_loss": lambda **mode: opt_in.forward_loss(
            X, W, labels, **mode, **loss_kw
        ),
        "opt_in.prepare_lm_head_loss": lambda **mode: opt_in.prepare_lm_head_loss(
            X, W, labels, **mode, **loss_kw
        ),
        "opt_in.cake_lm_head_nondet_train_loss": lambda **mode: opt_in.cake_lm_head_nondet_train_loss(
            X, W, labels, **mode, memory_mode="fast", **loss_kw
        ),
        "opt_in.cake_lm_head_nondet_train_logprob": lambda **mode: opt_in.cake_lm_head_nondet_train_logprob(
            X, W, labels, **mode, memory_mode="fast", **common
        ),
        "forward_logprob_temperature": lambda **mode: (
            cake_backend.forward_logprob_temperature(
                X, W, labels, **mode, temperature=0.7, **common
            )
        ),
        "backward_logprob_temperature": lambda **mode: (
            cake_backend.backward_logprob_temperature(
                X,
                W,
                labels,
                torch.zeros(labels.shape[0]),
                torch.zeros(labels.shape[0]),
                **mode,
                temperature=0.7,
                **common,
            )
        ),
        "prepare_lm_head_loss_temperature": lambda **mode: (
            cake_backend.prepare_lm_head_loss_temperature(
                X,
                W,
                labels,
                **mode,
                entry="logprob",
                temperature=torch.full((labels.shape[0],), 0.7),
                **common,
            )
        ),
        "cake_lm_head_nondet_train_logprob_temperature": lambda **mode: (
            cake_backend.cake_lm_head_nondet_train_logprob_temperature(
                X, W, labels, **mode, temperature=0.7, **common
            )
        ),
        "opt_in.forward_logprob_temperature": lambda **mode: (
            opt_in.forward_logprob_temperature(
                X, W, labels, **mode, temperature=0.7, **common
            )
        ),
        "opt_in.prepare_lm_head_loss_temperature": lambda **mode: (
            opt_in.prepare_lm_head_loss_temperature(
                X,
                W,
                labels,
                **mode,
                entry="logprob",
                temperature=torch.full((labels.shape[0],), 0.7),
                **common,
            )
        ),
        "opt_in.cake_lm_head_nondet_train_logprob_temperature": lambda **mode: (
            opt_in.cake_lm_head_nondet_train_logprob_temperature(
                X,
                W,
                labels,
                **mode,
                num_valid_rows=int((labels >= 0).sum()),
                temperature=0.7,
                **common,
            )
        ),
        "cake_lm_head_nondet_train_logprob_keep": lambda **mode: (
            cake_backend.cake_lm_head_nondet_train_logprob_keep(
                X, W, labels, **mode, **common
            )
        ),
        "forward_logprob_keep": lambda **mode: cake_backend.forward_logprob_keep(
            X, W, labels, **mode, chunk_size=4, backend="cake"
        ),
        "backward_logprob_keep": lambda **mode: cake_backend.backward_logprob_keep(
            X,
            W,
            labels,
            torch.zeros(rows),
            torch.zeros(rows),
            (),
            **mode,
            chunk_size=4,
            backend="cake",
        ),
        "opt_in.forward_logprob": lambda **mode: opt_in.forward_logprob(
            X, W, labels, **mode, **common
        ),
        "opt_in.backward_logprob": lambda **mode: opt_in.backward_logprob(
            X,
            W,
            labels,
            torch.zeros(rows),
            torch.zeros(rows),
            **mode,
            **common,
        ),
        "opt_in.backward_logprob_temperature": lambda **mode: (
            opt_in.backward_logprob_temperature(
                X,
                W,
                labels,
                torch.zeros(rows),
                torch.zeros(rows),
                **mode,
                temperature=0.7,
                **common,
            )
        ),
    }


def test_no_entry_takes_a_mode_keyword():
    """The mode is part of the package name (issue #8928): passing ``deterministic`` to any entry that took it before
    is Python's own ``TypeError``, raised before any work, and no autograd ``Function.forward`` of the two backends
    binds a ``deterministic`` keyword."""
    inp = _host_inputs(8, device=CPU)
    calls = _former_mode_calls(inp.X, inp.W, inp.labels)
    assert len(calls) == 24
    for call in calls.values():
        with pytest.raises(TypeError, match="deterministic"):
            call(deterministic=False)
    functions = {
        f"{module.__name__}.{name}": obj
        for module in (cake_backend, opt_in)
        for name, obj in vars(module).items()
        if isinstance(obj, type)
        and issubclass(obj, torch.autograd.Function)
        and obj.__module__ == module.__name__
    }
    assert sorted(name.rsplit(".", 1)[1] for name in functions) == sorted(
        [
            "CakeLMHeadNondetLogprobFunction",
            "CakeLMHeadNondetLogprobFunction",
            "CakeLMHeadNondetLogprobKeepFunction",
            "CakeLMHeadNondetLogprobTemperatureFunction",
            "CakeLMHeadNondetLogprobTemperatureFunction",
            "CakeLMHeadNondetLossFunction",
            "CakeLMHeadNondetLossFunction",
            "_SmallCallStage",
        ]
    ), sorted(functions)
    for fn in functions.values():
        with pytest.raises(TypeError):
            inspect.signature(fn.forward).bind_partial(deterministic=False)


def test_former_mode_entries_run_without_a_mode_keyword():
    """Every entry of :func:`_former_mode_calls` on the reference backend runs on the host without a mode keyword
    (the kept-logits forward / backward pair takes the ``cake`` backend only and is left out)."""
    inp = _host_inputs(8, device=CPU)
    calls = _former_mode_calls(inp.X, inp.W, inp.labels)
    ran = []
    for name, call in calls.items():
        if name in ("forward_logprob_keep", "backward_logprob_keep"):
            continue
        call()  # the reference backend computes on the host
        ran.append(name)
    assert len(ran) == 22, ran


def _det_counterpart(name: str) -> str:
    """The deterministic counterpart the alert of the call ``name`` (a key of :func:`_dx_calls`) names: the loss
    entry's for the loss forwards, else the log-probability entry's."""
    kind = "loss" if name.endswith(("forward_loss", ".forward", "_loss")) else "logprob"
    return f"flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_{kind}"


def _dx_calls(inp):
    """``name -> call()`` of every path that launches the dX GEMMs, on the reference backend: each raises PyTorch's
    alert under torch's deterministic algorithms when its plan has a K-sliced dX chunk (:func:`_forced_dx_slices`)
    and runs without one when every chunk is one slice (the reference backend's own plan)."""
    X, W, labels = inp.X, inp.W, inp.labels
    kw = dict(chunk_size=4, backend="reference")

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

    def runner_backward_temperature():
        with _torch_deterministic(False):
            runner = cake_backend.prepare_lm_head_loss_temperature(
                X,
                W,
                labels,
                entry="logprob",
                temperature=torch.full((labels.shape[0],), 0.7),
                **kw,
            )
            runner.forward()
        return runner.backward()

    def autograd_temperature(mod, **extra):
        def call():
            with _torch_deterministic(False):
                logp = mod.cake_lm_head_nondet_train_logprob_temperature(
                    X.detach().requires_grad_(True),
                    W,
                    labels,
                    temperature=0.7,
                    **extra,
                    **kw,
                )
            return logp.backward(torch.ones_like(logp))

        return call

    def autograd_logprob(mod, **extra):
        def call():
            with _torch_deterministic(False):
                logp = mod.cake_lm_head_nondet_train_logprob(
                    X.detach().requires_grad_(True), W, labels, **extra, **kw
                )
            return logp.backward(torch.ones_like(logp))

        return call

    valid_rows = int((labels >= 0).sum())
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
        "opt_in.backward_logprob": lambda: opt_in.backward_logprob(
            X,
            W,
            labels,
            torch.zeros(labels.shape[0]),
            torch.zeros(labels.shape[0]),
            **kw,
        ),
        "opt_in.CakeLMHeadNondetLogprobFunction.backward": autograd_logprob(
            opt_in, num_valid_rows=valid_rows
        ),
        "LmHeadLossRunner.forward": runner_forward,
        "LmHeadLossRunner.backward": runner_backward_logprob,
        "opt_in.LmHeadLossRunner.forward": opt_in_runner_forward,
        "opt_in.LmHeadLossRunner.backward": opt_in_runner_backward_logprob,
        "cake_lm_head_nondet_train_loss": lambda: cake_backend.cake_lm_head_nondet_train_loss(
            X.detach().requires_grad_(True), W, labels, loss_div=1.0, **kw
        ),
        "backward_logprob_temperature": lambda: (
            cake_backend.backward_logprob_temperature(
                X,
                W,
                labels,
                torch.zeros(labels.shape[0]),
                torch.zeros(labels.shape[0]),
                temperature=0.7,
                **kw,
            )
        ),
        "opt_in.backward_logprob_temperature": lambda: (
            opt_in.backward_logprob_temperature(
                X,
                W,
                labels,
                torch.zeros(labels.shape[0]),
                torch.zeros(labels.shape[0]),
                temperature=0.7,
                **kw,
            )
        ),
        "LmHeadLossRunner.backward (temperature)": runner_backward_temperature,
        "CakeLMHeadNondetLogprobTemperatureFunction.backward": autograd_temperature(
            cake_backend
        ),
        "opt_in.CakeLMHeadNondetLogprobTemperatureFunction.backward": autograd_temperature(
            opt_in, num_valid_rows=valid_rows
        ),
    }


def _fixed_order_backwards(inp):
    """``name -> call()`` of the backward-only launches of a loss call whose forward ran outside torch's
    deterministic algorithms -- the deferred last-chunk weight gradient (``backward_loss``), the loss runner's
    backward and the loss autograd backward, both modules: casts and weight-gradient GEMMs of one fixed order, also
    after a forward that ran a K-sliced dX (on the reference backend)."""
    X, W, labels = inp.X, inp.W, inp.labels
    kw = dict(chunk_size=4, backend="reference")

    def deferred_backward(mod):
        def call():
            with _torch_deterministic(False):
                fwd = mod.forward_loss(
                    X, W, labels, loss_div=1.0, fuse_dw_cast=True, **kw
                )
            assert (
                fwd.dz_last is not None
            )  # the plan deferred the last chunk's weight-gradient GEMM
            return fwd.backward()

        return call

    def runner_backward(mod):
        def call():
            with _torch_deterministic(False):
                runner = mod.prepare_lm_head_loss(X, W, labels, loss_div=1.0, **kw)
                runner.forward()
            return runner.backward()

        return call

    def autograd_backward(mod):
        def call():
            with _torch_deterministic(False):
                loss = mod.cake_lm_head_nondet_train_loss(
                    X.detach().requires_grad_(True),
                    W.detach().requires_grad_(True),
                    labels,
                    loss_div=1.0,
                    **kw,
                )
            loss.backward()

        return call

    return {
        "backward_loss": deferred_backward(cake_backend),
        "opt_in.backward_loss": deferred_backward(opt_in),
        "LmHeadLossRunner.backward (loss)": runner_backward(cake_backend),
        "opt_in.LmHeadLossRunner.backward (loss)": runner_backward(opt_in),
        "CakeLMHeadNondetLossFunction.backward": autograd_backward(cake_backend),
        "opt_in.CakeLMHeadNondetLossFunction.backward": autograd_backward(opt_in),
    }


@contextlib.contextmanager
def _forced_dx_slices(slices: int = 2):
    """Every plan of both backend modules -- the reference backend's included, which plans one dX slice -- runs its
    dX GEMM at ``slices`` K slices: the K-sliced reduction (``gemm_dx_red``; the reference engine adds the slices in
    ascending order), so the host tests reach the alert of a K-sliced dX on the CPU."""
    originals = (cake_backend.plan_dx_slices, opt_in.plan_dx_slices)

    def forced(problem, chunks, *, need_dx, **_):
        return tuple(slices for _ in chunks) if need_dx else ()

    cake_backend.plan_dx_slices = opt_in.plan_dx_slices = forced
    try:
        yield
    finally:
        cake_backend.plan_dx_slices, opt_in.plan_dx_slices = originals


def _record_reference_dx_red(monkeypatch) -> list:
    """The reference engine's K-sliced dX launches (``gemm_dx_red``) of the calls that follow, in order."""
    launched: list = []
    reduce = cake_backend.ReferenceEngine.gemm_dx_red

    def recording(values):
        launched.append(int(values["k_slices"]))
        return reduce(values)

    monkeypatch.setattr(
        cake_backend.ReferenceEngine, "gemm_dx_red", staticmethod(recording)
    )
    return launched


_ALERT = "does not have a deterministic implementation"


def test_torch_deterministic_mode_alerts_before_a_k_sliced_dx(monkeypatch):
    """Under ``torch.use_deterministic_algorithms(True)`` every dX path whose plan has a K-sliced dX chunk (forced: the
    reference backend plans one slice) raises PyTorch's ``RuntimeError`` (its wording, naming the entry and its
    deterministic counterpart) before its first K-sliced dX launch; under ``warn_only=True`` it warns once per call, then
    runs the reduction."""
    inp = _host_inputs(8, device=CPU)
    launched = _record_reference_dx_red(monkeypatch)
    with _forced_dx_slices():
        for name, call in _dx_calls(inp).items():
            launched.clear()
            with (
                _torch_deterministic(True),
                pytest.raises(RuntimeError, match=_ALERT) as info,
            ):
                call()
            assert _det_counterpart(name) in str(info.value), name
            assert not launched, name
            with (
                _torch_deterministic(True, warn_only=True),
                warnings.catch_warnings(record=True) as record,
            ):
                warnings.simplefilter("always")
                call()
            alerts = [w for w in record if _ALERT in str(w.message)]
            assert len(alerts) == 1, (name, [str(w.message) for w in alerts])
            assert issubclass(alerts[0].category, UserWarning), name
            assert launched and set(launched) == {2}, (name, launched)


def test_warn_only_alert_names_the_callers_frame():
    """Under ``warn_only=True`` the alert of an autograd entry's forward is attributed to the caller's frame: the
    package's, flashinfer's and torch's frames (``Function.apply``) are skipped (forced K slices, reference
    backend)."""
    inp = _host_inputs(8, device=CPU)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    with (
        _forced_dx_slices(),
        _torch_deterministic(True, warn_only=True),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        cake_backend.cake_lm_head_nondet_train_loss(
            X, W, inp.labels, loss_div=1.0, chunk_size=4, backend="reference"
        )
    alerts = [w for w in record if _ALERT in str(w.message)]
    assert len(alerts) == 1, [str(w.message) for w in record]
    assert os.path.abspath(alerts[0].filename) == os.path.abspath(__file__), alerts[
        0
    ].filename


def test_one_slice_and_fixed_order_launches_do_not_alert(monkeypatch):
    """A plan whose dX chunks all have one slice (the reference backend's) runs every dX path under strict
    ``torch.use_deterministic_algorithms(True)`` without an alert, and the backward-only launches of a loss call (the
    deferred weight gradient, the casts, the runner's and the autograd backward) run without one even after a forward
    that ran a K-sliced dX: their order is fixed."""
    inp = _host_inputs(8, device=CPU)
    launched = _record_reference_dx_red(monkeypatch)
    with _torch_deterministic(True), warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
        for call in _dx_calls(inp).values():
            call()
        for call in _fixed_order_backwards(inp).values():
            call()
    assert not launched
    with (
        _forced_dx_slices(),
        _torch_deterministic(True),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
        for name, call in _fixed_order_backwards(inp).items():
            launched.clear()
            call()
            assert launched, name  # the forward (outside the mode) ran the K-sliced dX


def test_paths_without_gradient_gemms_do_not_alert():
    """The log-probability forward and a loss forward without trainable inputs launch no gradient GEMM: no alert,
    also with every dX plan forced to K slices."""
    inp = _host_inputs(8, device=CPU)
    kw = dict(chunk_size=4, backend="reference")
    with (
        _forced_dx_slices(),
        _torch_deterministic(True),
        warnings.catch_warnings(),
    ):
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
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
    tail_fold=False,
):
    """The ``sm_100a`` B200 plan of ``rows`` chunk-loop rows of the cross-entropy loss (``entry="logprob"``: the
    log-probability entry) built by ``backend``'s ``make_plan`` (default this package's ``cake_backend``; the
    deterministic package takes no policy, no reduction floor and no short-tail fold).  ``tail_fold`` (this package):
    the default plan's short-tail fold; off by default, so the slice-rule tests compare the deterministic package's
    chunks (the fold has its own tests)."""
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
    if backend is cake_backend:
        kw["tail_fold"] = tail_fold  # the opt-in module's plans do not fold
    return backend.make_plan(problem, **kw)


@pytest.mark.parametrize(
    "rows, chunk, folded",
    [
        (4097, 4096, ((0, 4097),)),  # a one-row tail
        (4370, 4096, ((0, 4370),)),  # V07: T 4600 with 5 % ignored rows
        (6143, 4096, ((0, 6143),)),  # a 2047-row tail: the longest folded one
        (6144, 4096, ((0, 4096), (4096, 2048))),  # a C / 2 tail stays a chunk
        (8192, 4096, ((0, 4096), (4096, 4096))),
        # three chunks: the shipped plan
        (8193, 4096, ((0, 4096), (4096, 4096), (8192, 1))),
        (4096, 4096, ((0, 4096),)),
        (37, 16, ((0, 16), (16, 16), (32, 5))),
        (20, 16, ((0, 20),)),
        # a folded chunk would exceed CHUNK_MAX_ROWS rows
        (70000, 60000, ((0, 60000), (60000, 10000))),
        (0, 4096, ()),
    ],
    ids=str,
)
def test_plan_chunks_short_tail_fold(rows, chunk, folded):
    """``plan_chunks(fold=True)``: a two-chunk plan whose last chunk has fewer than ``chunk / 2`` rows runs as one chunk
    of fewer than ``1.5 * chunk`` rows (at most ``CHUNK_MAX_ROWS``); every other plan keeps its chunks.
    ``chunk_rows_max`` is the largest chunk (the chunk buffers' rows)."""
    assert cake_backend.plan_chunks(rows, chunk, fold=True) == folded
    shipped = cake_backend.plan_chunks(rows, chunk)
    assert sum(r for _, r in folded) == sum(r for _, r in shipped) == rows
    if len(folded) < len(shipped):
        assert len(shipped) == 2 and 2 * shipped[1][1] < chunk
        assert chunk < folded[0][1] < 1.5 * chunk
        assert folded[0][1] <= cake_backend.CHUNK_MAX_ROWS
    assert cake_backend.chunk_rows_max(rows, chunk, fold=True) == max(
        [r for _, r in folded] or [min(1, chunk)]
    )
    assert cake_backend.chunk_rows_max(rows, chunk) == min(chunk, max(rows, 1))


def test_tail_fold_plan_rules():
    """The default plan folds the short tail of a call of fewer than three chunks of ``T`` (the compacted loop rows
    decide the tail), the kept-logits plan and the zero-gradient row skip's counted plan included (one rule,
    ``loop_chunks``; the counted plan's live counts follow the folded chunk: ``chunk_stride``); a hidden-count plan, a
    call of three chunks or more of ``T`` and ``tail_fold=False`` keep the shipped chunks.  A plan folded to one chunk with the fused cast has no FP32
    ``dW_acc``, and the memory report of the folded plan says so."""
    assert not cake_backend.tail_fold_eligible(0, 4096)
    assert cake_backend.tail_fold_eligible(4600, 4096)
    assert cake_backend.tail_fold_eligible(8192, 4096)
    assert not cake_backend.tail_fold_eligible(8193, 4096)
    assert cake_backend.HIDDEN_COUNT_MIN_CHUNKS == 3
    for entry in ("loss", "logprob"):
        problem = _problem(4600, GLM_H, GLM_V, 4096, entry=entry)
        base = dict(
            need_dx=True,
            need_dw=True,
            valid_rows=4370,
            fuse_dw_cast=True,
            arch="sm_100a",
            num_sms=B200_SMS,
            dx_max_slices=cake_backend.DX_RED_SLICES_MAX,
            dx_resident=B200_DX_RESIDENT,
        )
        plan = cake_backend.make_plan(problem, **base)
        assert plan.chunks == ((0, 4370),)
        assert plan.dw_deferred and not plan.dw_acc_needed
        assert cake_backend.make_plan(problem, **base, tail_fold=False).chunks == (
            (0, 4096),
            (4096, 274),
        )
        # every row valid: the 504-row tail folds too (one chunk of 4600 rows)
        assert cake_backend.make_plan(
            problem, **dict(base, valid_rows=None)
        ).chunks == ((0, 4600),)
        if entry == "logprob":
            kept = cake_backend.make_plan(problem, **base, logprob_keep=True)
            assert kept.logprob_keep and kept.chunks == ((0, 4370),)
            assert cake_backend.make_plan(
                problem, **base, logprob_keep=True, tail_fold=False
            ).chunks == ((0, 4096), (4096, 274))
            counted = cake_backend.make_plan(
                problem, **base, dcnt=True, logprob_dz=True
            )
            assert counted.dcnt and counted.chunks == ((0, 4370),)
            assert cake_backend.chunk_stride(counted.chunks, 4096) == 4370
            assert cake_backend.loop_chunks(4600, 4370, 4096) == ((0, 4370),)
    # an unfolded plan's stride is the chunk size; the chunk count follows the rule of the plan
    assert cake_backend.chunk_stride(((0, 4096), (4096, 274)), 4096) == 4096
    assert cake_backend.chunk_stride(((0, 1000),), 4096) == 4096
    for T, rows, want in (
        (8193, 4300, 2),
        (4600, 4000, 1),
        (12289, 12289, 4),
        (6144, 6144, 2),
        (6143, 6143, 1),
    ):
        assert len(cake_backend.loop_chunks(T, rows, 4096)) == want, (T, rows)
    unfolded = ((0, 4096), (4096, 274))
    assert cake_backend.loop_chunks(4600, 4370, 4096, hidden_count=True) == unfolded
    assert cake_backend.loop_chunks(4600, 4370, 4096, fold=False) == unfolded
    # the workspace helper's chunk buffers hold the largest folded chunk of any loop-row count up to T
    for T, rows_max in (
        (4600, 4600),
        (7000, 6143),
        (6143, 6143),
        (8193, 4096),
        (3000, 3000),
    ):
        assert cake_backend.fold_rows_max(T, 4096) == rows_max, T
        assert all(
            rows_c <= rows_max
            for rows in range(1, T + 1, 37)
            for _, rows_c in cake_backend.loop_chunks(T, rows, 4096)
        ), T
        for compact in (False, True):
            assert (
                cake_backend.lm_head_loss_workspace_size(
                    T,
                    GLM_V,
                    4096,
                    backend="reference",
                    hidden=GLM_H,
                    compact_rows=compact,
                )
                == cake_backend.workspace_layout(
                    T, GLM_V, rows_max, hidden=GLM_H, compact=compact
                )["total"]
            ), (T, compact)
    # a folded call prepared on a workspace of the helper's size (20 rows at chunk 16: one chunk of 20 rows)
    inp = _host_inputs(20, device=CPU, ignore_frac=0.0)
    size = cake_backend.lm_head_loss_workspace_size(
        20, V_HOST, 16, backend="reference", hidden=H_HOST
    )
    runner = cake_backend.prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        loss_div=1.0,
        chunk_size=16,
        backend="reference",
        workspace_buffer=torch.empty(size, dtype=torch.uint8),
    )
    assert runner.plan.chunks == ((0, 20),)
    long = _problem(8193, GLM_H, GLM_V, 4096)
    assert cake_backend.make_plan(
        long, need_dx=True, need_dw=True, valid_rows=4300
    ).chunks == ((0, 4096), (4096, 204))
    for fold, dw_acc, rows_max, chunks in (
        (True, False, 4370, 1),
        (False, True, 4096, 2),
    ):
        report = cake_backend.memory_report(
            4600,
            GLM_H,
            GLM_V,
            4096,
            entry="logprob",
            valid_rows=4370,
            fuse_dw_cast=True,
            tail_fold=fold,
        )
        assert ("dW_acc" in report["accumulators"]) == dw_acc
        assert report["vocab_rows_max"] == rows_max and report["num_chunks"] == chunks
        assert report["temporary"]["logits"] == rows_max * GLM_V * 2


def test_dx_launches_come_from_the_dx_helper():
    """Every dX launch of a plan is planned by ``_dx_keys``: for a chunk of more than one slice the K-sliced reduction
    (``gemm_dx_red*``), which stores its slices into the call's slab, and right after it the slab sum
    (``dx_slab_sum``), which stores their sum into the chunk's rows (no host reduction chain); a chunk of one slice is
    one plain-store launch.  Nothing fills ``dX_acc``: the sequence starts with chunk 0's logits GEMM and holds no host
    step."""
    plan = _plan(16231, hidden=7168, vocab=129280, tuning={"dx": {"k_slices": 2}})
    for i in range(plan.num_chunks):
        stage = plan.dx_stage_of(i)
        assert stage.startswith(cake_backend.DX_RED_STAGE), (i, stage)
        assert plan.dx_slab_sum_stage_of(i) == cake_backend.DX_SLAB_SUM_STAGE
        assert cake_backend._dx_keys(plan, i) == [
            (stage, i),
            (cake_backend.DX_SLAB_SUM_STAGE, i),
        ]
    dx = [key for i in range(plan.num_chunks) for key in cake_backend._dx_keys(plan, i)]
    keys = cake_backend.forward_keys(plan)
    slab_sums = cake_backend.DX_SLAB_SUM_STAGES
    assert [
        key for key in keys if key[0].startswith("gemm_dx") or key[0] in slab_sums
    ] == dx
    for i in range(plan.num_chunks):
        # the slab sum runs right behind its chunk's K-sliced launch
        pos = keys.index((plan.dx_stage_of(i), i))
        assert keys[pos + 1] == (cake_backend.DX_SLAB_SUM_STAGE, i), keys
    assert set(slab_sums) & set(plan.stages) == {cake_backend.DX_SLAB_SUM_STAGE}
    assert cake_backend.DX_SLAB_SUM_ACC_STAGE not in plan.stages
    assert keys[0] == (plan.forward_logits_stage(0), 0), keys
    assert not any(key[0] in cake_backend.HOST_STAGES for key in keys), keys
    assert plan.dx_sliced
    plain = _plan(16231, hidden=7168, vocab=129280, tuning={"dx": {"k_slices": None}})
    for i in range(plain.num_chunks):
        assert plain.dx_slab_sum_stage_of(i) is None
        assert cake_backend._dx_keys(plain, i) == [(plain.dx_stage_of(i), i)]
    assert not set(cake_backend.DX_SLAB_SUM_STAGES) & set(plain.stages)
    assert not plain.dx_sliced


def test_slab_sum_binds_the_k_sliced_launch_slab():
    """The slab sum of a K-sliced chunk (forced two slices, reference backend) is bound right behind the chunk's
    ``gemm_dx_red`` launch to the chunk's rows of ``dX_acc`` and the call's slab that launch writes, at its
    ``m_tiles`` / ``ldc``: ``ws_slab = m_tiles * 128 * ldc``, ``num_vecs = rows_c * H / cast_vec``, ``n_slabs = 2``.
    The slab sum of a chunk the plan does not slice is refused, and the reference engine's store-then-sum ``dX``
    passes the gates."""
    inp = _host_inputs(37)
    kw = dict(loss_div=1.0, chunk_size=16, backend="reference")
    with _forced_dx_slices():
        runner = cake_backend.prepare_lm_head_loss(inp.X, inp.W, inp.labels, **kw)
        nd = _run(inp, 16)
    plan, t = runner.plan, runner.tensors
    assert plan.dx_slices == (2,) * plan.num_chunks
    order = runner.forward_order
    for i, (row0, rows_c) in enumerate(plan.chunks):
        gemm_key = (plan.dx_stage_of(i), i)
        sum_key = (cake_backend.DX_SLAB_SUM_STAGE, i)
        assert order[order.index(gemm_key) + 1] == sum_key, order
        gemm, v = runner.values[gemm_key], runner.values[sum_key]
        assert v["ws"] is t["dx_slab"]
        assert gemm["SLAB"].data_ptr() == t["dx_slab"].data_ptr()
        assert v["dx"].data_ptr() == t["dx_acc"][row0:].data_ptr()
        assert tuple(v["dx"].shape) == (rows_c, H_HOST)
        assert v["ws_slab"] == gemm["m_tiles"] * 128 * gemm["ldc"]
        assert v["num_vecs"] == rows_c * H_HOST // plan.geometry.cast_vec
        assert v["n_slabs"] == 2
    plain = cake_backend.prepare_lm_head_loss(inp.X, inp.W, inp.labels, **kw)
    assert plain.plan.dx_slices_of(0) == 1
    with pytest.raises(ValueError, match="slab sum"):
        cake_backend.stage_values(
            cake_backend.DX_SLAB_SUM_STAGE, plain.tensors, plain.plan, 0
        )
    _check_against_references(nd, inp)


def test_reference_slab_sum_stores_without_reading_dx():
    """The reference engine's slab sum (plain and counted form) stores the slabs' sum, taken in slice order, into the
    chunk's rows without reading them: rows that held NaN hold ``(s_0 + s_1) + s_2`` bit for bit."""
    rows, cols, slabs, pitch = 5, 8, 3, 64
    ws = torch.randn(
        slabs * pitch, generator=torch.Generator(device="cpu").manual_seed(SEED)
    )
    want = None
    for s in range(slabs):
        part = ws[s * pitch : s * pitch + rows * cols].view(rows, cols)
        want = part if want is None else want + part
    for stage in cake_backend.DX_SLAB_SUM_STAGES:
        dx = torch.full((rows, cols), float("nan"))
        getattr(cake_backend.ReferenceEngine, stage)(
            dict(dx=dx, ws=ws, ws_slab=pitch, n_slabs=slabs)
        )
        assert _same_bits(dx, want), stage


def _det_sliced(stage: str) -> bool:
    """A deterministic K-sliced dX stage name (``gemm_dx_s`` and its forms; ``gemm_dx_st3`` is the plain store)."""
    return stage == "gemm_dx_s" or stage.startswith("gemm_dx_s_")


def test_no_slab_path_remains():
    """The registered stage names carry the reduction forms, their slab sum (a row kernel, plain and counted) and no
    deterministic sliced form or slab sum, the host has no slab step and no tile counters, and no workspace, workspace
    size or memory report without the ordered slab's bytes (``dx_slab_bytes`` 0) counts a slab."""
    stages = set(cake_jit.STAGES)
    assert "slab_sum" not in stages
    assert not any(_det_sliced(s) for s in stages), stages
    slab_sums = set(cake_backend.DX_SLAB_SUM_STAGES)
    assert slab_sums == {"dx_slab_sum", "dx_slab_sum_dcnt"}
    assert slab_sums <= stages and slab_sums <= set(cake_jit.ROW_STAGES)
    assert not slab_sums & set(cake_jit.GEMM_STAGES)
    assert cake_backend.STAGE_TENSORS["dx_slab_sum"] == ("dx", "ws")
    assert cake_backend.STAGE_TENSORS["dx_slab_sum_dcnt"] == ("dx", "ws", "CNT")
    # the accumulating slab sum of the support path's later vocabulary blocks: not one of the plans' slab sums
    assert cake_backend.DX_SLAB_SUM_ACC_STAGE == "dx_slab_sum_acc"
    assert cake_backend.DX_SLAB_SUM_ACC_STAGE not in slab_sums
    assert cake_backend.STAGE_TENSORS["dx_slab_sum_acc"] == ("dx", "ws")
    for name in ("_DX_TILE_COUNTERS", "_dx_tile_counters", "DX_TILE_COLS"):
        assert not hasattr(cake_backend, name), name
    red = {
        "gemm_dx_red",
        "gemm_dx_red_tn256",
        "gemm_dx_red_st3",
        "gemm_dx_red_split",
        "gemm_dx_red_st3_split",
        "gemm_dx_red_swab",
    }
    assert red <= stages and red <= set(cake_jit.GEMM_STAGES)
    assert set(cake_backend.HOST_STAGES) == {"gather_rows"}
    assert set(opt_in.HOST_STAGES) == {"gather_rows", "save_row_grad"}
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
    """``set_dx_red_min_rows``: a chunk of fewer rows than the floor keeps one slice (the plain store, no slab sum); the
    rule's slices elsewhere; an explicit ``k_slices`` knob wins; the floor is part of the slice rule's settings the
    binding keys and the plans take (``dx_slice_rule``)."""
    assert cake_backend.dx_red_min_rows() == 0
    assert cake_backend.dx_slice_rule() == ("det", 0)
    rule = _plan(16231).dx_slices
    assert rule == (4, 4, 4, 4)
    assert _plan(16231, red_min_rows=4000).dx_slices == (4, 4, 4, 1)
    assert _plan(16231, red_min_rows=4000).dx_sliced
    assert _plan(339, red_min_rows=340).dx_slices == (1,)
    assert not _plan(339, red_min_rows=340).dx_sliced
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
    logprob_chunk_loop, monkeypatch, request, backend
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
    kw = dict(chunk_size=chunk, backend=backend)
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


def test_zero_weight_floor_switch():
    """The programmatic switch of the zero-weight split's smallest call: default (calls of at least 4096 rows -- this
    suite runs at 0, every compacted policy call), the previous value returned, bad values refused, and the floor
    applied to the split decision (no binding depends on it)."""
    assert cake_backend.ZERO_WEIGHT_MIN_ROWS_DEFAULT == 4096
    inp = _host_inputs(200, "policy")
    problem = cake_backend.validate_lm_head_inputs(
        inp.X,
        inp.W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
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
        )
        below = cake_backend.forward_loss(inp.X, inp.W, weighted.labels, **kw)
        assert "zero_weight_rows" not in below.memory
        assert below.row_index.numel() == int((inp.labels >= 0).sum())
    finally:
        assert cake_backend.set_zero_weight_min_rows(0) == 256
    split = cake_backend.forward_loss(inp.X, inp.W, weighted.labels, **kw)
    assert split.memory["zero_weight_rows"] == int(zero.sum())
    torch.testing.assert_close(split.logp, below.logp, atol=1e-5, rtol=0)


def test_k_sliced_plans_start_their_sequences_with_chunk_0():
    """``Plan.dx_sliced``: the call needs ``dX`` and has a chunk of more than one slice.  The slab sum is the only
    writer of a sliced chunk's ``dX_acc`` rows, so no launch sequence of a K-sliced plan holds a step ahead of its
    chunk loop: the loss forward, the log-probability backward's recompute and the opt-in forward start with chunk 0's
    launches and (no compaction) hold no host step.  A one-slice chunk keeps the plain-store form and has no slab
    sum."""
    mixed = _plan(7711)
    assert mixed.dx_slices == (4, 1) and mixed.dx_sliced
    assert mixed.dx_stage_of(0).startswith(cake_backend.DX_RED_STAGE)
    assert not mixed.dx_stage_of(1).startswith(cake_backend.DX_RED_STAGE)
    assert mixed.dx_slab_sum_stage_of(0) == cake_backend.DX_SLAB_SUM_STAGE
    assert mixed.dx_slab_sum_stage_of(1) is None
    keys = cake_backend.forward_keys(mixed)
    assert keys[0] == (mixed.forward_logits_stage(0), 0), keys
    assert not any(key[0] in cake_backend.HOST_STAGES for key in keys), keys
    plain = _plan(2168)
    assert plain.dx_slices == (1,) and not plain.dx_sliced
    assert not _plan(7711, need_dx=False).dx_sliced
    assert _plan(16231).dx_sliced
    logprob = _plan(7711, entry="logprob")
    assert logprob.dx_sliced
    recompute = cake_backend.recompute_keys(logprob)
    assert recompute[0][1] == 0, recompute
    assert not any(key[0] in cake_backend.HOST_STAGES for key in recompute), recompute
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
    assert opt_plan.dx_slices == mixed.dx_slices and opt_plan.dx_sliced
    opt_keys = opt_in.forward_keys(opt_plan)
    assert opt_keys[0] == (opt_plan.forward_logits_stage(0), 0), opt_keys


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
    )
    assert cake_backend._hidden_count_cell(0) is here[0]
    other = []

    def worker():
        other.append(
            (
                cake_backend._hidden_count_cell(0),
                cake_backend._hidden_count_pair_cell(0),
                cake_backend._dw_stream(0),
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
    grown constant keeps its values.  The counter keeps only the test thread's ``record_stream`` calls, and a mutant
    that records the constant on every call (each call sees a new raw stream handle) fails the first check."""
    if not torch.cuda.is_available():
        pytest.skip("the count constant needs a CUDA device")
    records: list[tuple[int, int]] = []
    record_stream = torch.Tensor.record_stream

    def counting(self, stream):
        records.append((threading.get_ident(), stream.cuda_stream))
        return record_stream(self, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", counting)
    index = torch.cuda.current_device()

    def scenario():
        got = {}
        errors: list[BaseException] = []
        records.clear()  # a finished thread's identifier can be reused

        def run():
            me = threading.get_ident()

            def mine():
                return [stream for ident, stream in records if ident == me]

            try:
                with (
                    torch.cuda.device(index),
                    torch.cuda.stream(torch.cuda.Stream(index)),
                ):
                    main = torch.cuda.current_stream(index).cuda_stream
                    recorded = cake_backend._thread_state("arange_streams")
                    ar = cake_backend._hidden_count_arange(index, 8)
                    cake_backend._hidden_count_arange(index, 8)
                    got["one"] = (len(recorded[index]), mine())
                    side = torch.cuda.Stream(index)
                    with torch.cuda.stream(side):
                        assert cake_backend._hidden_count_arange(index, 8) is ar
                        cake_backend._hidden_count_arange(index, 8)
                    assert cake_backend._hidden_count_arange(index, 8) is ar
                    got["two"] = (len(recorded[index]), mine())
                    grown = cake_backend._hidden_count_arange(index, 2 * ar.numel() + 1)
                    got["grown"] = (grown is not ar, len(recorded[index]), mine())
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
        return got

    got = scenario()
    main, side = got["streams"]
    assert got["one"] == (1, [main])
    assert got["two"] == (2, [main, side])
    assert got["grown"] == (True, 1, [main, side, main])
    assert got["values"]
    # the mutant: every call sees a new raw stream handle, so the constant is recorded on every call
    monkeypatch.setattr(cake_backend, "_current_raw_stream", lambda index: object())
    mutant = scenario()
    assert mutant["one"] != (1, [mutant["streams"][0]])
    assert mutant["one"][1] == [mutant["streams"][0]] * 2


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
# Device (SM100 / SM103 / SM107 with the package's rendered programs)
# ---------------------------------------------------------------------------


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, entry: str = "loss"):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 / 10.7 device")
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


@contextlib.contextmanager
def _vocab_major(mode):
    """Run under the vocabulary-major backward's ``mode`` (:func:`cake_backend._set_logprob_vocab_major`), then restore."""
    previous = cake_backend._set_logprob_vocab_major(mode)
    try:
        yield
    finally:
        cake_backend._set_logprob_vocab_major(previous)


@pytest.fixture
def logprob_chunk_loop():
    """The log-probability backward of every call in the test runs the chunk loop (``_set_logprob_vocab_major("0")``):
    for the tests of the chunk loop's plans, bindings, fills and slices, and of its bits against the kept-logits,
    budget, one-slice and deterministic forms (the vocabulary-major form is tested by its own tests below)."""
    with _vocab_major("0"):
        yield


@pytest.fixture(params=["0", "auto"], ids=["chunk_loop", "vocab_major"])
def logprob_backward_form(request):
    """Both forms of a multi-chunk log-probability backward: the chunk loop and the vocabulary-major form."""
    with _vocab_major(request.param):
        yield request.param


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
    forgets to write -- such as ``dX_acc`` rows no dX GEMM or slab sum stores -- cannot inherit zeros."""
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


def _folds(inp, chunk) -> bool:
    """Whether this package's plan of the call folds its short tail (``cake_backend.loop_chunks``) over the valid rows
    (compacted) or every row: the deterministic package's chunks then differ."""
    T = int(inp.labels.numel())
    return any(
        len(cake_backend.loop_chunks(T, rows, chunk))
        < len(cake_backend.plan_chunks(rows, chunk))
        for rows in {T, int((inp.labels >= 0).sum())}
        if rows
    )


def _compare_with_det(nd, det, *, record=None, folded=False):
    """``loss`` / ``logp`` and ``dW`` are the deterministic package's bits (fixed-order stages); ``dX`` is
    reported: max |ND - det| and the BF16 flip rate.  ``folded`` (a call whose short tail folds, :func:`_folds`): its
    chunks are not the deterministic package's, ``logp`` only (``loss`` / ``dW`` are held to the gates)."""
    for key in ("logp",) if folded else ("loss", "logp", "dW"):
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
def _launch_record(*, temperature: bool = False):
    """The launch keys both backend modules run (each key as ``_run_keys`` runs it), in order: the record of what a
    call launched (the hidden valid-row count's own chunk-0 launches are not keys).  ``temperature``: also the keys of
    the calls with a per-token temperature (``_run_keys_temperature``)."""
    launched: list = []
    originals = (cake_backend._run_keys, opt_in._run_keys)
    tau_originals = (cake_backend._run_keys_temperature, opt_in._run_keys_temperature)

    def recording(original):
        def run_keys(plan, device_index, keys, run_key, *args, **kwargs):
            def run(key):
                launched.append(key)
                return run_key(key)

            return original(plan, device_index, keys, run, *args, **kwargs)

        return run_keys

    cake_backend._run_keys = recording(originals[0])
    opt_in._run_keys = recording(originals[1])
    if temperature:
        cake_backend._run_keys_temperature = recording(tau_originals[0])
        opt_in._run_keys_temperature = recording(tau_originals[1])
    try:
        yield launched
    finally:
        cake_backend._run_keys, opt_in._run_keys = originals
        cake_backend._run_keys_temperature, opt_in._run_keys_temperature = tau_originals


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
    with _one_pass_launches() as one_pass:
        nd = _run(inp, 4096, entry=entry, backend="cake", **kw)
    det = _run(inp, 4096, entry=entry, backend="cake", package="det", **_det_tuning(kw))
    errors, b0 = _check_against_references(nd, inp, entry=entry, ceiling=True)
    _check_against_references(det, inp, entry=entry, ceiling=True)
    if one_pass:
        # a one-pass call (one_pass_rows) forms loss / logp / dW in its own fixed-order stages: the same bits run to
        # run, not the deterministic package's; the chunked path of the same call keeps the package's bits
        again = _run(inp, 4096, entry=entry, backend="cake", **kw)
        for key in ("loss", "logp", "dW"):
            assert _same_bits(again[key], nd[key]), key
        with _one_pass_rows(()):
            nd = _run(inp, 4096, entry=entry, backend="cake", **kw)
        _check_against_references(nd, inp, entry=entry, ceiling=True)
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
    logprob_chunk_loop, glm_weight, T, objective, entry, tuning, policy
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
        _compare_with_det(run, det, record=versus, folded=_folds(inp, 4096))
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


@pytest.mark.parametrize("T", [1, 256])
@pytest.mark.parametrize("form", ["plain", "temperature"])
def test_device_logprob_backward_makes_its_gradient_buffers_behind_the_recompute(
    glm_weight, T, form
):
    """A remembered log-probability backward binding (plain and with a per-token temperature) makes ``dX_acc``,
    ``dX`` and ``dW`` right before chunk 0's dX GEMM -- behind the row gathers and the recompute launch, the launches
    a call of a few rows waits for on the host -- and the call repeats the first (prepared) call's ``logp`` / ``dW``
    bits (plain: and passes the gates; temperature: ``dX`` within :func:`_assert_tau_close`)."""
    if form == "temperature":
        _require_tau_program()
        inp = _tau_inputs(T, W=glm_weight)
    else:
        _require_program(entry="logprob")
        inp = _device_inputs(T, "ce", W=glm_weight)
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    # the default Function's recompute (a small call where a backward follows would run the kept logits).  A small
    # call's bindings hold its staged rows' address (one allocation per call): the discarded call in between leaves the
    # allocator as the recorded call finds it (``first``'s outputs stay alive), so the recorded call meets the binding
    # the discarded call remembered
    with _small_t_keep(False):
        first = _run(inp, 4096, entry="logprob", backend="cake")
        _run(inp, 4096, entry="logprob", backend="cake")
        misses = cache.misses
        with _launch_record(temperature=form == "temperature") as launched:
            again = _run(inp, 4096, entry="logprob", backend="cake")
    assert cache.misses == misses, "the recorded call runs remembered bindings"
    backward = [
        b for b in cache._bindings.values() if b.plan.need_dx and b.backward_order
    ]
    assert backward and all(b.backward_gradients_at_dx for b in backward), backward
    stages = [key[0] for key in launched]
    dz = next(
        i for i, s in enumerate(stages) if s.startswith(cake_backend.LOGPROB_DZ_STAGE)
    )
    dx = next(i for i, s in enumerate(stages) if s.startswith("gemm_dx"))
    assert dz < dx, stages
    for key in ("logp", "dW"):
        assert _same_bits(again[key], first[key]), key
    if form == "temperature":
        _assert_tau_close(again, first, (T, form), keys=("dX",))
    else:
        _check_against_references(again, inp, entry="logprob", ceiling=True)


def test_device_checkpoint_replay(glm_weight):
    """``torch.utils.checkpoint`` replays the forward in the backward: ``loss`` / ``logp`` / ``dW`` are the
    uncheckpointed call's bits, ``dX`` passes the gates."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, "ce", W=glm_weight)

    def head(X, W):
        return cake_backend.cake_lm_head_nondet_train_loss(
            X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096
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
        loss = cake_backend.cake_lm_head_nondet_train_loss(
            X if inp is a else inp.X.detach().requires_grad_(False),
            W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=4096,
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
    loss = cake_backend.cake_lm_head_nondet_train_loss(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
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
            loss, logp = cake_backend.cake_lm_head_nondet_train_loss(
                X,
                W,
                inp.labels,
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
            )
            loss.backward()
            results[i] = dict(loss=loss.detach(), logp=logp, dX=X.grad, dW=W.grad)
    torch.cuda.synchronize()
    for inp, got, want in zip(inputs, results, serial, strict=True):
        for key in ("loss", "logp", "dW"):
            assert _same_bits(got[key], want[key]), key
        _check_against_references(got, inp, ceiling=True)


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_prepared_runner_and_cuda_graph(glm_weight, entry, shipped_chunks):
    """A prepared runner launches without allocating; a CUDA graph captured from it replays for new values written
    into the bound tensors and gives the runner's own results on those values.  The shipped chunks (``shipped_chunks``:
    no short-tail fold; the 4097-row call keeps its 1-row tail chunk)."""
    _require_program(entry=entry)
    inp = _device_inputs(4097, "ce", W=glm_weight, ignore_frac=0.0)
    new = _device_inputs(4097, "ce", W=glm_weight, seed=SEED + 11, ignore_frac=0.0)
    X = inp.X.clone()
    labels = inp.labels.clone()
    kw = dict(chunk_size=4096, backend="cake", entry=entry)
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

    # the graph holds the K-sliced reduction and its slab sum
    assert runner.plan.dx_sliced
    graph.replay()
    torch.cuda.synchronize()
    replayed = outputs()
    # the bound accumulators and chunk buffers hold NaN before the second replay: the slab sums store the K-sliced
    # chunks' dX_acc rows without reading them, the one-slice chunks store theirs
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
def test_device_unordered_reduce_is_exercised(
    logprob_chunk_loop, glm_weight, T, entry, policy
):
    """On the representative rows the eager entries launch the K-sliced reduction (``gemm_dx_red*``), each followed
    right away by its slab sum (``dx_slab_sum*``, never the support path's accumulating form), for every chunk the
    slice rule gives more than one slice: the device's registered program serves the reduction forms and their slab
    sum and no deterministic sliced form (no ``slab_sum`` / ``dx_reduce`` launch, no ``dx_ws`` workspace); the plan
    sizes the ordered form's slab."""
    _require_program(entry=entry)
    module_name, record = cake_backend.record_for(CUDA)
    registered = set(cake_jit.registered_stages(module_name))
    assert cake_backend.DX_RED_STAGE in registered
    assert cake_backend.DX_SLAB_SUM_STAGE in registered
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
    # the chunk's slab sum right behind its K-sliced launch, planned and launched
    for seq in (order, launched):
        for pos, key in enumerate(seq):
            if key[0].startswith(cake_backend.DX_RED_STAGE):
                want = (plan.dx_slab_sum_stage_of(key[1]), key[1])
                assert seq[pos + 1] == want, seq
    assert plan.dx_sliced
    acc = cake_backend.DX_SLAB_SUM_ACC_STAGE
    assert not any(key[0] == acc for key in launched), launched
    assert not any(key[0] in ("slab_sum", "dx_reduce") for key in order)
    assert "dx_ws" not in runner.tensors and "dx_ws" not in runner.layout
    assert plan.dx_slab_bytes > 0
    _check_against_references(nd, inp, entry=entry, ceiling=True)
    print(
        f"\nND reduction T={T} {entry} policy={policy}: slices {plan.dx_slices} red {[k[0] for k in red]}"
    )


@pytest.mark.parametrize(
    "T, entry, tuning",
    [
        (16231, "loss", None),
        (4097, "loss", {"dx": {"k_slices": 4}}),
        (8191, "logprob", {"dx": {"k_slices": 3, "tile_n": 256}}),
    ],
    ids=str,
)
def test_device_ordered_dx_repeats_bitwise(glm_weight, T, entry, tuning):
    """The K-sliced dX reduction is ordered: five runs of one call (the slice rule's slices, or forced ones) launch
    ``gemm_dx_red*`` and its slab sum, give the same ``loss`` / ``logp`` / ``dX`` / ``dW`` bits and pass the
    gates."""
    _require_program(entry=entry)
    inp = _device_inputs(T, "ce", W=glm_weight)
    kw = {} if tuning is None else {"gemm_tuning": tuning}
    with _launch_record() as launched:
        first = _run(inp, 4096, entry=entry, backend="cake", **kw)
    assert any(key[0].startswith(cake_backend.DX_RED_STAGE) for key in launched), (
        launched
    )
    assert any(key[0] in cake_backend.DX_SLAB_SUM_STAGES for key in launched), launched
    for _ in range(4):
        again = _run(inp, 4096, entry=entry, backend="cake", **kw)
        for key in ("loss", "logp", "dX", "dW"):
            assert _same_bits(again[key], first[key]), key
    _check_against_references(first, inp, entry=entry, ceiling=True)


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
def test_device_one_slice_chunks_are_the_deterministic_bits(
    glm_weight, policy, shipped_chunks
):
    """A call whose plan on the RUNNING device mixes a chunk of more than one K slice (the reduction) with a chunk of
    one (the plain store) -- the row count is searched on this device's plan (its SM count and architecture), never
    assumed: the one-slice chunk's ``dX`` rows are the deterministic package's one-slice bits, ``loss`` / ``logp`` /
    ``dW`` its bits under its own rule, the call passes the gates, and the
    sliced rows' max |ND - det| and BF16 flip rate are reported.  Skipped only when no two-chunk call of this device
    has such a plan.  The shipped chunks (``shipped_chunks``: no short-tail fold)."""
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
    nd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
    det = det_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
    assert _same_bits(nd.logp, det.logp) and _same_bits(nd.lse, det.lse)
    assert math.isfinite(float(nd.logp.sum()))


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_one_slice_call_runs_under_strict_deterministic_algorithms(
    glm_weight, entry
):
    """Under strict ``torch.use_deterministic_algorithms(True)`` a call whose dX chunks all have one slice (pinned; every
    row valid, uncompacted) raises no alert: its launches have one order, so two runs give the same ``dX`` / ``dW``
    bits and ``dX`` is the deterministic entry's (the same one-slice plain store).  This checks on the device the
    fixed order of the weight gradient (chunk order, no atomics) that the package's docstrings state."""
    _require_program(entry=entry)
    inp = _device_inputs(8191, "ce", W=glm_weight, ignore_frac=0.0)
    kw = dict(gemm_tuning={"dx": {"k_slices": None}}, compact_rows=False)
    runs = []
    for _ in range(2):
        with _torch_deterministic(True), warnings.catch_warnings():
            warnings.filterwarnings("error", message=f".*{_ALERT}.*")
            runs.append(_run(inp, 4096, entry=entry, backend="cake", **kw))
    det = _run(inp, 4096, entry=entry, backend="cake", package="det", **kw)
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(runs[0][key], runs[1][key]), key
    assert _same_bits(runs[0]["dX"], det["dX"])
    _check_against_references(runs[0], inp, entry=entry)


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_k_sliced_call_alerts_before_its_dx_and_leaves_no_state(
    glm_weight, entry
):
    """Under strict ``torch.use_deterministic_algorithms(True)`` a call whose plan has K-sliced dX chunks (two slices
    pinned; the default compacted path with the hidden valid-row count) raises PyTorch's ``RuntimeError`` before any dX
    launch -- the launch record holds no ``gemm_dx*`` key and no slab sum -- on a call that forms its binding
    and on a remembered one; the next call outside the mode gives the reference call's ``loss`` / ``logp`` / ``dW``
    bits and a ``dX`` within the budgets (the raise leaves no half-built binding or thread state).  Under
    ``warn_only=True`` the call warns and runs."""
    _require_program(entry=entry)
    inp = _device_inputs(8191, "ce", W=glm_weight)
    kw = dict(gemm_tuning={"dx": {"k_slices": 2}})
    name = "forward_loss" if entry == "loss" else "backward_logprob"
    ref = _run(inp, 4096, entry=entry, backend="cake", **kw)
    cake_backend.BINDING_CACHE.clear()
    for attempt in ("binds", "remembered"):
        with (
            _torch_deterministic(True),
            _launch_record() as launched,
            pytest.raises(
                RuntimeError,
                match=f"{name} {_ALERT}.*cake_lm_head_det_train.cake_lm_head_det_train_{entry}",
            ),
        ):
            _run(inp, 4096, entry=entry, backend="cake", **kw)
        torch.cuda.synchronize()
        dx_keys = [
            k for k in launched if str(k[0]).startswith(("gemm_dx", "dx_slab_sum"))
        ]
        assert not dx_keys, (attempt, launched)
        again = _run(inp, 4096, entry=entry, backend="cake", **kw)
        for key in ("loss", "logp", "dW"):
            assert _same_bits(again[key], ref[key]), (attempt, key)
        _check_against_references(again, inp, entry=entry)
    with (
        _torch_deterministic(True, warn_only=True),
        pytest.warns(UserWarning, match=f"{name} {_ALERT}"),
    ):
        warned = _run(inp, 4096, entry=entry, backend="cake", **kw)
    for key in ("loss", "logp", "dW"):
        assert _same_bits(warned[key], ref[key]), key
    _check_against_references(warned, inp, entry=entry)


def test_device_kept_backward_alerts_before_its_dx_and_keeps_the_logits(glm_weight):
    """``backward_logprob_keep`` under strict ``torch.use_deterministic_algorithms(True)`` with K-sliced dX chunks (two
    slices pinned) raises PyTorch's ``RuntimeError`` before any dX launch (no ``gemm_dx*`` key, no slab sum)
    and leaves the kept logits unconsumed: the same views then serve a backward outside the mode, which gives the
    default call's ``dW`` bits and a ``dX`` within the budgets.  Under ``warn_only=True`` it warns once and runs."""
    _require_program(entry="logprob")
    inp = _device_inputs(8191, "ce", W=glm_weight)
    tuning = {"dx": {"k_slices": 2}}
    kw = dict(chunk_size=4096, gemm_tuning=tuning)
    dlogp = torch.where(inp.valid, inp.dlogp, torch.zeros_like(inp.dlogp))
    ref = _run(inp, 4096, entry="logprob", backend="cake", gemm_tuning=tuning)

    def forward():
        return cake_backend.forward_logprob_keep(inp.X, inp.W, inp.labels, **kw)

    def backward(fr):
        X, W, labels = inp.X, inp.W, inp.labels
        return cake_backend.backward_logprob_keep(
            X, W, labels, fr.lse, dlogp, fr.logits, row_index=fr.row_index, **kw
        )

    def check(dx, dw, tag):
        torch.cuda.synchronize()
        assert _same_bits(dw, ref["dW"]), tag
        _check_against_references(dict(ref, dX=dx, dW=dw), inp, entry="logprob")

    fr = forward()
    with (
        _torch_deterministic(True),
        _launch_record() as launched,
        pytest.raises(
            RuntimeError,
            match=f"backward_logprob_keep {_ALERT}.*cake_lm_head_det_train.cake_lm_head_det_train_logprob",
        ),
    ):
        backward(fr)
    torch.cuda.synchronize()
    dx_keys = [k for k in launched if str(k[0]).startswith(("gemm_dx", "dx_slab_sum"))]
    assert not dx_keys, launched
    check(*backward(fr), "after the strict raise")  # the same kept views
    fr = forward()
    with (
        _torch_deterministic(True, warn_only=True),
        warnings.catch_warnings(record=True) as record,
    ):
        warnings.simplefilter("always")
        dx, dw = backward(fr)
    alerts = [w for w in record if _ALERT in str(w.message)]
    assert len(alerts) == 1, [str(w.message) for w in alerts]
    assert "backward_logprob_keep" in str(alerts[0].message)
    check(dx, dw, "warn_only")


def test_device_budget_recompute_backward_does_not_alert(glm_weight, monkeypatch):
    """The opt-in loss entry's recompute form of ``memory_mode="budget"`` (forced) finishes ``dW`` in the backward by
    the vocabulary-block recompute: weight-gradient GEMMs of one fixed order.  After a forward outside torch's
    deterministic algorithms (two dX slices pinned), the autograd backward and ``ForwardResult.backward`` run under
    strict ``torch.use_deterministic_algorithms(True)`` without an alert and give the same ``dW`` bits."""
    _require_program(entry="loss")
    step = _require_recompute()
    inp = _device_inputs(8191, "ce", W=glm_weight)
    calls: list = []
    _force_budget_plan(monkeypatch, ("recompute", 0, step), calls)
    kw = dict(
        loss_div=inp.loss_div,
        chunk_size=4096,
        memory_mode="budget",
        gemm_tuning={"dx": {"k_slices": 2}},
    )
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    loss = opt_in.cake_lm_head_nondet_train_loss(X, W, inp.labels, **kw)
    fwd = opt_in.forward_loss(inp.X, inp.W, inp.labels, **kw)
    assert calls and set(calls) == {"loss"}, calls
    assert loss.grad_fn.memory_plan.recompute and fwd.memory_plan.recompute
    with _torch_deterministic(True), warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
        loss.backward()
        dx, dw = fwd.backward()
    torch.cuda.synchronize()
    assert X.grad is not None and dx is not None
    assert _same_bits(dw, W.grad)


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
    module=None,
    **kw,
):
    """One autograd loss call (forward + backward) of ``package`` (``module``: this package's opt-in module for its
    keywords); returns loss / logp (``None`` without ``return_logp``) / dX / dW."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    if package == "nd":
        fn = (cake_backend if module is None else module).cake_lm_head_nondet_train_loss
    else:
        fn = det_backend.cake_lm_head_det_train_loss
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

    def syncs(forward, hidden):
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
                    )
            finally:
                torch.cuda.set_sync_debug_mode("default")
            counts.append(sum("synchroniz" in str(w.message) for w in rec))
        return min(counts)

    for hidden in (False, True):
        nd_syncs = syncs(cake_backend.forward_loss, hidden)
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
    )

    def head(X, W):
        return cake_backend.cake_lm_head_nondet_train_loss(X, W, inp.labels, **kw)

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
    equal to the clean call's (the deterministic package's plain path, without its own zero-weight split, gives a
    non-finite ``dW``); a NaN weight keeps its row in the gradient path (NaN loss)."""
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
    with _det_unsplit():
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
        loss, logp = (
            opt_in if memory_mode else cake_backend
        ).cake_lm_head_nondet_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **({} if memory_mode is None else dict(memory_mode=memory_mode)),
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
    and the forward's memory report records that path; the K-sliced reduction with its slab sum and the zero-weight
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
        fn = (
            cake_backend.cake_lm_head_nondet_train_loss
            if package == "nd"
            else det_backend.cake_lm_head_det_train_loss
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
                    loss, logp = cake_backend.cake_lm_head_nondet_train_loss(
                        X,
                        W,
                        inp.labels,
                        objective="policy",
                        infer_logp=inp.infer_logp,
                        loss_weights=inp.loss_weights,
                        chunk_size=4096,
                        return_logp=True,
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
        fn = (
            cake_backend.cake_lm_head_nondet_train_loss
            if package == "nd"
            else det_backend.cake_lm_head_det_train_loss
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
        loss, logp = opt_in.cake_lm_head_nondet_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            memory_mode="budget",
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
# Calls without a gradient GEMM (``T == 0``, every row ignored, every valid row of zero weight): no FP32 ``[V, H]``
# weight-gradient accumulator from the forward to the backward, the call's backend kept, ``dW = cast(0 * g)``
# ---------------------------------------------------------------------------


def _all_plus_zero(t: torch.Tensor) -> bool:
    """Every element is ``+0`` (bit pattern zero)."""
    return not bool(_bits(t).any())


@contextlib.contextmanager
def _zero_weight_floor(rows):
    """Run under the split's floor ``rows`` (:data:`NO_SPLIT`: never), then restore the previous setting."""
    previous = cake_backend.set_zero_weight_min_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_zero_weight_min_rows(previous)


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
def test_all_zero_weight_split_result(module_name):
    """A split whose valid rows all have zero weight (no gradient GEMM): the result keeps the call's backend (the
    reference backend's backward runs on any device), its ``dw_acc`` holds no FP32 ``[V, H]`` storage and the memory
    report counts that accumulator's one FP32 element, and :meth:`ForwardResult.backward` forms ``dW`` in the output
    dtype with the bits :func:`scale_cast` gives for a zero FP32 accumulator (``+0``; ``-0`` for a negative upstream
    scalar, also beyond the BF16 range; NaN for a non-finite one); the autograd entry's ``dX`` / ``dW`` are ``+0``."""
    module = cake_backend if module_name == "main" else opt_in
    base = _host_inputs(1200, "policy")
    inp = _with_zero_weights(base, base.labels >= 0)
    with _zero_weight_floor(0):
        fr = module.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=256,
            backend="reference",
        )
        out = _loss_call(inp, 256, backend="reference", module=module)
    assert "zero_weight_rows" in fr.memory and int(fr.row_index.numel()) == 0
    assert fr.backend == "reference"
    assert tuple(fr.dw_acc.shape) == (V_HOST, H_HOST)
    assert fr.dw_acc.untyped_storage().nbytes() <= 4
    assert fr.memory["accumulators"]["dW_acc"] == 4
    assert fr.memory["accumulator_bytes"] == sum(fr.memory["accumulators"].values())
    zero_acc = torch.zeros((V_HOST, H_HOST), dtype=torch.float32, device=DEV)
    f32_max = torch.finfo(torch.float32).max  # finite, beyond the BF16 range
    for g in (None, 2.0, -2.0, f32_max, -f32_max, math.inf, math.nan):
        grad = None if g is None else torch.tensor(g, dtype=torch.float32, device=DEV)
        dx, dw = fr.backward(grad)
        assert dw.dtype == torch.bfloat16 and tuple(dw.shape) == (V_HOST, H_HOST)
        ref = cake_backend.scale_cast(
            zero_acc, grad, torch.bfloat16, backend="reference"
        )
        if g is None or math.isfinite(g):
            negative = g is not None and g < 0
            assert _same_bits(dw, ref), g
            assert bool((dw == 0).all()), g
            assert bool((torch.signbit(dw) == negative).all()), g
            assert _all_plus_zero(dx)
        else:
            assert bool(torch.isnan(dw).all() and torch.isnan(ref).all()), g
    _, dw32 = fr.backward(grad_weight_dtype=torch.float32)
    assert dw32.dtype == torch.float32 and _all_plus_zero(dw32)
    assert _all_plus_zero(out["loss"])
    assert out["dW"].dtype == torch.bfloat16 and _all_plus_zero(out["dW"])
    assert _all_plus_zero(out["dX"])


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
@pytest.mark.parametrize("backend", ["reference", "cake"])
def test_zero_row_result(module_name, backend):
    """A call without rows (no binding, no launch): the result keeps the call's backend, its memory report counts the
    zero ``dw_acc``'s one FP32 element (not an FP32 ``[V, H]`` buffer), and the backward's ``dW`` is the FP32 product
    ``0 * g`` cast once (``+0``; ``-0`` for a negative ``g``, also beyond the BF16 range)."""
    module = cake_backend if module_name == "main" else opt_in
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=DEV)
    X0 = torch.zeros(0, H_HOST, dtype=torch.bfloat16, device=DEV)
    labels0 = torch.zeros(0, dtype=torch.int64, device=DEV)
    fr = module.forward_loss(
        X0, W, labels0, objective="ce", loss_div=1.0, backend=backend
    )
    assert fr.backend == backend
    assert fr.dw_acc.untyped_storage().nbytes() <= 4
    assert fr.memory["accumulators"] == {"dX_acc": 0, "dW_acc": 4}
    assert fr.memory["accumulator_bytes"] == 4
    f32_max = torch.finfo(torch.float32).max
    for g in (None, 2.0, f32_max, -f32_max):
        grad = None if g is None else torch.tensor(g, dtype=torch.float32, device=DEV)
        dx, dw = fr.backward(grad)
        assert tuple(dx.shape) == (0, H_HOST) and dx.dtype == torch.bfloat16
        assert dw.dtype == torch.bfloat16 and tuple(dw.shape) == (V_HOST, H_HOST)
        if g is None or g > 0:
            assert _all_plus_zero(dw), g
        else:
            assert bool((dw == 0).all() and torch.signbit(dw).all()), g


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(V_HOST, H_HOST), (2, 8), (2, 6), (7, 13)])
def test_zero_gradient_bits(shape, out_dtype):
    """:func:`cake_backend._zero_gradient` writes the FP32 product ``0 * g`` cast once into every element -- through
    the 16-byte view when the element count is a multiple of the copies one 16-byte element holds (8 BF16, 4 FP32),
    element by element otherwise -- with the bits of a broadcast copy of that cast value, for finite, huge,
    signed-zero and non-finite ``g``."""
    acc = cake_backend._zero_accumulator(shape, DEV)
    f32_max = torch.finfo(torch.float32).max
    finite = (None, 1.0, -2.0, f32_max, -f32_max, 0.0, -0.0)
    for g in finite + (math.inf, -math.inf, math.nan):
        grad = None if g is None else torch.tensor(g, dtype=torch.float32, device=DEV)
        dw = cake_backend._zero_gradient(acc, grad, out_dtype)
        zero = torch.zeros((), dtype=torch.float32, device=DEV)
        if grad is not None:
            zero = zero * grad
        ref = torch.empty(shape, dtype=out_dtype, device=DEV).copy_(zero.to(out_dtype))
        assert dw.dtype == out_dtype and tuple(dw.shape) == shape and dw.is_contiguous()
        assert _same_bits(dw, ref), (shape, out_dtype, g)


@pytest.mark.parametrize("module_name", ["main", "opt_in"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_empty_autograd_scales_the_zero_weight_gradient(module_name, checkpointed):
    """A loss call without gradient rows (``T == 0``; every row ignored), plain or under a non-reentrant checkpoint:
    the autograd backward's ``dW`` has the bits :meth:`ForwardResult.backward` gives for the same upstream scalar (the
    FP32 product ``0 * g`` cast once: ``+0``; ``-0`` for a negative ``g``, also beyond the BF16 range; NaN for a
    non-finite one), and ``dX`` is ``+0``.  The opt-in arm passes ``num_valid_rows=0`` so its own autograd Function
    runs (without an opt-in keyword the call is ``cake_backend``'s)."""
    from torch.utils.checkpoint import checkpoint

    module = cake_backend if module_name == "main" else opt_in
    extra = {} if module_name == "main" else dict(num_valid_rows=0)
    W0 = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=DEV)
    f32_max = torch.finfo(torch.float32).max
    for T in (0, 37):
        X0 = torch.zeros(T, H_HOST, dtype=torch.bfloat16, device=DEV)
        labels = torch.full((T,), -100, dtype=torch.int64, device=DEV)
        kw = dict(objective="ce", loss_div=1.0, backend="reference", **extra)
        fr = module.forward_loss(X0, W0, labels, **kw)

        def call(X, W):
            return module.cake_lm_head_nondet_train_loss(X, W, labels, **kw)

        for g in (2.0, -2.0, f32_max, -f32_max, math.inf, math.nan):
            grad = torch.tensor(g, dtype=torch.float32, device=DEV)
            X = X0.clone().requires_grad_()
            W = W0.clone().requires_grad_()
            loss = (
                checkpoint(call, X, W, use_reentrant=False)
                if checkpointed
                else call(X, W)
            )
            assert loss.grad_fn._forward_cls is module.CakeLMHeadNondetLossFunction
            loss.backward(grad)
            _, ref = fr.backward(grad)
            assert W.grad.dtype == ref.dtype == torch.bfloat16, (T, g)
            if math.isfinite(g):
                assert _same_bits(W.grad, ref), (T, g)
                assert bool((torch.signbit(W.grad) == (g < 0)).all()), (T, g)
            else:
                both_nan = torch.isnan(W.grad).all() and torch.isnan(ref).all()
                assert bool(both_nan), (T, g)
            assert tuple(X.grad.shape) == (T, H_HOST), (T, g)
            assert _all_plus_zero(X.grad), (T, g)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_all_zero_weight_split_autograd_scales_the_zero_weight_gradient(checkpointed):
    """Every valid row of zero weight under the split (a policy call, plain or under a non-reentrant checkpoint, whose
    backward then reads the recomputation's state): the autograd backward's ``dW`` is the FP32 product ``0 * g`` cast
    once, as :meth:`ForwardResult.backward` forms it (``-0`` for a negative ``g``, NaN for a non-finite one), and
    ``dX`` is ``+0``."""
    from torch.utils.checkpoint import checkpoint

    base = _host_inputs(1200, "policy")
    inp = _with_zero_weights(base, base.labels >= 0)
    kw = dict(
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=256,
        backend="reference",
    )
    with _zero_weight_floor(0):
        fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
        assert "zero_weight_rows" in fr.memory and int(fr.row_index.numel()) == 0

        def call(X, W):
            return cake_backend.cake_lm_head_nondet_train_loss(X, W, inp.labels, **kw)

        f32_max = torch.finfo(torch.float32).max
        for g in (-2.0, -f32_max, math.inf):
            grad = torch.tensor(g, dtype=torch.float32, device=DEV)
            X = inp.X.detach().clone().requires_grad_()
            W = inp.W.detach().clone().requires_grad_()
            loss = (
                checkpoint(call, X, W, use_reentrant=False)
                if checkpointed
                else call(X, W)
            )
            loss.backward(grad)
            _, ref = fr.backward(grad)
            if math.isfinite(g):
                assert _same_bits(W.grad, ref), g
                assert bool((W.grad == 0).all() and torch.signbit(W.grad).all()), g
            else:
                assert bool(torch.isnan(W.grad).all() and torch.isnan(ref).all()), g
            assert _all_plus_zero(X.grad), g


def _peaks(call):
    """``(allocated, reserved)`` peak bytes of ``call()`` above its start, after one warm-up call (the binding is
    remembered) with the caching allocator emptied before each measured call."""
    call()
    out = []
    for _ in range(2):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_res = torch.cuda.memory_reserved()
        call()
        torch.cuda.synchronize()
        out.append(
            (
                torch.cuda.max_memory_allocated() - start_alloc,
                torch.cuda.max_memory_reserved() - start_res,
            )
        )
    return max(a for a, _ in out), max(r for _, r in out)


def _cold_peaks(call, modules):
    """``(allocated, reserved)`` peak bytes of ``call()`` above its start with every binding cache of ``modules``
    cleared before each measured call (each call prepares its bindings), after one warm-up call."""
    call()
    out = []
    for _ in range(2):
        for mod in modules:
            mod.BINDING_CACHE.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        start_res = torch.cuda.memory_reserved()
        call()
        torch.cuda.synchronize()
        out.append(
            (
                torch.cuda.max_memory_allocated() - start_alloc,
                torch.cuda.max_memory_reserved() - start_res,
            )
        )
    return max(a for a, _ in out), max(r for _, r in out)


def _step_call(inp, module, **kw):
    """One forward + backward of the loss entry (``X`` and ``W`` trainable, ``return_logp``), outputs dropped."""

    def call():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        loss, logp = module.cake_lm_head_nondet_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
            **kw,
        )
        loss.backward()
        del loss, logp, X, W

    return call


@pytest.mark.parametrize(
    "module_name, mode", [("main", None), ("opt_in", "fast"), ("opt_in", "budget")]
)
@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("T", [4096, 16231])
def test_device_all_zero_weight_peak_within_the_plain_path(
    glm_weight, T, cold, module_name, mode
):
    """Every valid row of zero weight (one chunk at ``T`` 4096, several at 16231): the split launches no gradient
    GEMM and holds no FP32 weight-gradient accumulator into the backward, so its measured step peak stays within the
    same call's without the split -- allocated with the binding caches warm or cleared before each call, reserved
    with them cleared (a warm call's reserved peak depends on which allocator segments the remembered bindings of
    earlier calls leave partly used, so it is compared from the same cleared state); ``dX`` and ``dW`` are ``+0``."""
    _require_program(entry="loss")
    module = cake_backend if module_name == "main" else opt_in
    base = _device_inputs(T, "policy", W=glm_weight)
    inp = _with_zero_weights(base, base.labels >= 0)
    kw = {} if mode is None else dict(memory_mode=mode)
    call = _step_call(inp, module, **kw)
    peaks = (lambda c: _cold_peaks(c, (cake_backend, opt_in))) if cold else _peaks
    with _zero_weight_floor(0):
        split_alloc, split_res = peaks(call)
        out = _loss_call(inp, 4096, module=module, **kw)
    with _zero_weight_floor(NO_SPLIT):
        plain_alloc, plain_res = peaks(call)
    print(
        f"\nall-zero-weight peak {module_name}/{mode} T={T} cold={cold}: allocated split {split_alloc} "
        f"plain {plain_alloc}; reserved split {split_res} plain {plain_res}"
    )
    assert _all_plus_zero(out["dX"]) and _all_plus_zero(out["dW"])
    assert split_alloc <= plain_alloc
    if cold:
        assert split_res <= plain_res


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


def test_device_loads_the_libraries_of_its_architecture():
    """On a registered device every stage of the device's record loads the library ``build.json`` builds for the
    device's architecture (``<library>_sm_107a`` on compute capability 10.7): the prebuilt one under ``aot/`` when
    the package ships it, else FlashInfer's JIT build."""
    _require_program(entry="loss")
    name, record = cake_backend.record_for(CUDA)
    arch = record["arch"]
    capability = torch.cuda.get_device_capability(0)
    assert arch == cake_backend.SUPPORTED_COMPUTE_CAPABILITIES[capability]
    libraries = []
    for stage in cake_jit.registered_stages(name):
        library = cake_jit.library_name(name, stage)
        assert library.endswith("_" + arch), (stage, library)
        assert cake_jit.load_cake_lm_head_loss_module(name, stage) is not None
        libraries.append(library)
    prebuilt = sum(cake_build.prebuilt_path(library).is_file() for library in libraries)
    print(
        f"\n{arch} (capability {capability}): {len(libraries)} stages loaded "
        f"({len(set(libraries))} libraries, {prebuilt} stages prebuilt); "
        f"first {cake_build.library_path(libraries[0])}"
    )


@pytest.mark.parametrize("stage", RED_STAGES)
def test_device_every_reduction_form_runs_within_the_budgets(glm_weight, stage):
    """Every ``gemm_dx_red*`` form the running architecture registers and its tuning accepts (``gemm_dx_red_swab`` on
    the ``DX_SWAB_ARCHES`` only: ``sm_103a`` and ``sm_107a``, which registers it with the ``sm_103a`` stages), pinned
    at four K slices on a call of two whole 4096-row chunks: the launch record shows it launched on both, the
    outputs pass the deterministic suite's gates against the FP64 oracle, and ``loss`` / ``logp`` / ``dW`` are the
    deterministic package's bits."""
    _require_program(entry="loss")
    module_name, record = cake_backend.record_for(CUDA)
    if stage not in cake_jit.registered_stages(module_name):
        pytest.skip(f"{stage} is not registered for this architecture")
    if stage.endswith("_swab") and record["arch"] not in cake_backend.DX_SWAB_ARCHES:
        pytest.skip(
            f"{stage}: the swapped dX tile is pinned on {', '.join(cake_backend.DX_SWAB_ARCHES)} only"
        )
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
    the chunks the device's rule slices launch one ``gemm_dx_red*`` form each (launch record), each followed right away
    by its slab sum, and the call passes the gates."""
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
    for key in red:
        after = launched[launched.index(key) + 1]
        assert after[0] in cake_backend.DX_SLAB_SUM_STAGES, launched
        assert after[1] == key[1], launched
    _check_against_references(nd, inp, ceiling=True)
    print(
        f"\nND 4-chunk ce policy={policy} slices {slices}: red launches {[key[0] for key in red]}"
    )


def test_device_fp32_dx_acc_is_bitwise_over_repeated_runs(glm_weight):
    """The FP32 ``dX_acc`` of ``REPEATS`` forwards at one and 339 rows with three and eight K slices and at 4097
    rows with eight: every run is within the deterministic suite's ``dX`` budget against the FP64 oracle, the number
    of distinct bit patterns and the largest run-to-run difference are reported, and every case has one bit pattern.
    Every element of ``dX_acc`` gets one nonzero partial per slice (``dz`` is dense over the vocabulary), so three or
    more slices would make a sum in finishing order depend on which slice finishes first; the K-sliced GEMM stores each
    slice's slab and the slab sum adds the slabs in slice order, so the result does not depend on it."""
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
    assert all(n == 1 for n in distinct.values()), distinct


def _budget_step(inp, entry, chunk_size=4096, **kw):
    """One training step of this package's opt-in autograd entry with its peak allocated bytes above the bytes before
    it (outputs and gradients count): ``(result, peak, memory_plan)`` (``memory_plan`` of a loss call's opt-in form)."""
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    mplan = None
    if entry == "loss":
        loss, logp = opt_in.cake_lm_head_nondet_train_loss(
            X,
            W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=chunk_size,
            return_logp=True,
            **kw,
        )
        mplan = getattr(loss.grad_fn, "memory_plan", None)
        loss.backward()
        loss = loss.detach()
    else:
        entry_fn = (
            opt_in.cake_lm_head_nondet_train_logprob_temperature
            if TEMPERATURE_KEYWORDS & kw.keys()
            else opt_in.cake_lm_head_nondet_train_logprob
        )
        logp = entry_fn(X, W, inp.labels, chunk_size=chunk_size, **kw)
        logp.backward(inp.dlogp)
        loss = None
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    return dict(loss=loss, logp=logp.detach(), dX=X.grad, dW=W.grad), peak, mplan


@pytest.mark.parametrize("entry", ["loss", "logprob"])
@pytest.mark.parametrize("T", [4096, 16384])
def test_device_budget_step_peak_within_the_prediction(
    logprob_chunk_loop, glm_weight, entry, T
):
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


def test_memory_plan_vocab_major_form():
    """``memory_plan(vocab_major=True)`` of the log-probability entry picks the fastest form that fits the budget, with
    the ``"vocab_major"`` form right after the chunk loop's fused form, ahead of its unfused and recompute forms: at a
    budget that fits the fused form it picks that form; at a budget just below the fused peak it picks
    ``"vocab_major"`` (no chunk-loop recompute, ``dw_rows`` 0, the vocabulary-major block) where the chunk loop alone
    takes its unfused or a recompute form, and below the fused and unfused peaks where the chunk loop falls back to a
    recompute form -- every multi-chunk GLM call below, at 2 to 16 chunks; never for the loss entry."""
    geometry = cake_backend.DEFAULT_GEOMETRY
    step = opt_in.dw_block_step(geometry)
    common = dict(hidden=GLM_H, vocab=GLM_V, chunk=4096, geometry=geometry)
    for rows in (8117, 15420, 30840, 61779):
        kw = dict(common, num_rows=rows, rows=rows)
        big = opt_in.memory_plan(
            "logprob", vocab_major=True, budget_bytes=1 << 45, **kw
        )
        assert big.form == "fused" and big.fits, (rows, big)
        unfused = opt_in.memory_plan(
            "logprob", fuse_dw_cast=False, budget_bytes=1 << 45, **kw
        )
        assert unfused.form == "unfused", (rows, unfused)
        # just below the fused peak: the vocabulary-major form ahead of the chunk loop's unfused form
        under = big.predicted_bytes - 1
        first = opt_in.memory_plan(
            "logprob", vocab_major=True, budget_bytes=under, **kw
        )
        assert first.form == "vocab_major" and first.fits, (rows, first)
        loop_under = opt_in.memory_plan("logprob", budget_bytes=under, **kw)
        assert loop_under.form != "fused", (rows, loop_under)
        below = min(big.predicted_bytes, unfused.predicted_bytes) - 1
        vm = opt_in.memory_plan("logprob", vocab_major=True, budget_bytes=below, **kw)
        assert vm.form == "vocab_major" and not vm.recompute and vm.fits, (rows, vm)
        assert vm.dw_rows == 0, (rows, vm)
        assert vm.block_rows == cake_backend._logprob_vm_block_rows(
            rows, GLM_V, 4096, step
        ), (rows, vm)
        assert vm.budget_bytes == below, (rows, vm)
        plain = opt_in.memory_plan("logprob", budget_bytes=below, **kw)
        assert plain.form in ("partial", "recompute"), (rows, plain)
        assert vm.predicted_bytes < big.predicted_bytes, (rows, vm, big)
        # the head rule's budget: the fused form where it fits, else the vocabulary-major form where it fits, else the
        # chunk loop's pick
        default = opt_in.memory_plan("logprob", vocab_major=True, **kw)
        loop = opt_in.memory_plan("logprob", **kw)
        if loop.form == "fused":
            expected = "fused"
        elif vm.predicted_bytes <= default.budget_bytes:
            expected = "vocab_major"
        elif loop.fits:
            expected = loop.form
        else:
            expected = min((loop, vm), key=lambda p: p.predicted_bytes).form
        assert default.form == expected, (rows, default, loop)
    loss = opt_in.memory_plan(
        "loss",
        num_rows=30840,
        rows=30840,
        vocab_major=True,
        budget_bytes=1 << 20,
        **common,
    )
    assert loss.form != "vocab_major", loss
    assert opt_in._memory_budget_bytes() is None
    previous = opt_in._set_memory_budget_bytes(1 << 33)
    try:
        assert previous is None and opt_in._memory_budget_bytes() == 1 << 33
    finally:
        opt_in._set_memory_budget_bytes(previous)
    with pytest.raises(ValueError):
        opt_in._set_memory_budget_bytes(0)


# The forms of the loss entry's explicit byte budget (:func:`opt_in._set_memory_budget_bytes`), in pick order
_LOSS_BUDGET_FORMS = ("fused", "unfused", "partial", "recompute")


@pytest.mark.parametrize(
    "T, R, H, V, C, zero",
    [
        (16231, 15419, GLM_H, GLM_V, 4096, 0),
        (16231, 15419, GLM_H, GLM_V, 4096, 2560),
        (16231, 15419, GLM_H, GLM_V, 2048, 0),
        (32768, 31130, 5120, 248320, 4096, 0),
        (4097, 3892, GLM_H, GLM_V, 4096, 0),
    ],
    ids=["4chunks", "zero_weight", "8chunks", "h5120_v248320", "1chunk"],
)
def test_memory_plan_explicit_budget_loss_forms(T, R, H, V, C, zero):
    """An explicit budget (``memory_plan(budget_bytes=)``, the internal override of the autograd entries) gives the loss
    entry the first of ``"fused"``, ``"unfused"``, ``"partial"`` and ``"recompute"`` whose predicted step peak fits it.
    Sweeping the budget down from the fused form's peak to the smallest peak: the form never moves back up that order,
    it is ``"fused"`` exactly while the fused peak fits and ``"unfused"`` exactly while the unfused peak fits and the
    fused one does not, every pick fits (``predicted_bytes <= budget_bytes`` = the budget), and ``dw_rows`` /
    ``block_rows`` are whole granules (:func:`opt_in.dw_block_step`).  The smallest peak is the recompute form at one
    granule (at that budget a partial form of the same peak may come first); one byte below it no form fits (``fits``
    false, the recompute form returned: the entries then raise).  Without a budget the head rule's bytes hold.
    Shapes: GLM calls of 4 chunks (with and without zero-weight policy rows), of 8 chunks of 2048 rows and of one
    chunk, and a call at ``H`` 5120 / ``V`` 248320."""
    geometry = cake_backend.DEFAULT_GEOMETRY
    step = opt_in.dw_block_step(geometry)
    rows = R - zero
    kw = dict(
        num_rows=T,
        rows=rows,
        hidden=H,
        vocab=V,
        chunk=C,
        buffer_rows=min(C, rows),
        budget_rows=R,
        zero_rows=zero,
        zero_total=zero,
        geometry=geometry,
    )
    fused = opt_in.memory_plan("loss", budget_bytes=1 << 45, **kw)
    assert fused.form == "fused" and fused.fits and fused.dw_rows == V, fused
    unfused = opt_in.memory_plan("loss", fuse_dw_cast=False, budget_bytes=1 << 45, **kw)
    assert unfused.form == "unfused" and unfused.dw_rows == V, unfused
    floor = opt_in.memory_plan("loss", budget_bytes=1, **kw)
    assert not floor.fits and floor.form == "recompute", floor
    assert floor.dw_rows == 0 and floor.block_rows == step, floor
    assert floor.predicted_bytes < min(fused.predicted_bytes, unfused.predicted_bytes)
    head_bytes = 2 * R * V + 2 * V * H + 2 * T * H
    head = opt_in.memory_plan("loss", **kw)
    assert head.budget_bytes == head_bytes - math.ceil(
        head_bytes * opt_in.MEMORY_BUDGET_MARGIN
    ), head
    lo, hi = floor.predicted_bytes, fused.predicted_bytes
    budgets = {hi, hi - 1, unfused.predicted_bytes, unfused.predicted_bytes - 1, lo}
    budgets.update(lo + (hi - lo) * i // 24 for i in range(1, 24))
    last, seen = None, set()
    for budget in sorted((b for b in budgets if lo <= b <= hi), reverse=True):
        p = opt_in.memory_plan("loss", budget_bytes=budget, **kw)
        assert p.budget_bytes == budget and p.fits, (budget, p)
        assert p.predicted_bytes <= budget, (budget, p)
        if budget >= fused.predicted_bytes:
            assert p.form == "fused", (budget, p)
        elif budget >= unfused.predicted_bytes:
            assert p.form == "unfused", (budget, p)
        else:
            assert p.form in ("partial", "recompute"), (budget, p)
        if p.form == "partial":
            assert 0 < p.dw_rows < V and p.dw_rows % step == 0, (budget, p)
        if p.recompute:
            assert p.block_rows % step == 0 or p.block_rows == V - p.dw_rows, (
                budget,
                p,
            )
            assert 0 < p.block_rows <= V - p.dw_rows, (budget, p)
        if last is not None:
            assert _LOSS_BUDGET_FORMS.index(p.form) >= _LOSS_BUDGET_FORMS.index(
                last.form
            ), (last, p)
        last = p
        seen.add(p.form)
    assert "fused" in seen and last.form in ("partial", "recompute"), (seen, last)
    assert last.predicted_bytes == lo, (last, lo)
    below = opt_in.memory_plan("loss", budget_bytes=lo - 1, **kw)
    assert not below.fits and below.predicted_bytes == lo, below
    # the loss entry never takes the vocabulary-major form, whatever the switch: the same pick as without it
    for budget in (lo, hi - 1):
        vm = opt_in.memory_plan("loss", vocab_major=True, budget_bytes=budget, **kw)
        plain = opt_in.memory_plan("loss", budget_bytes=budget, **kw)
        assert vm.form != "vocab_major" and vm == plain, (budget, vm, plain)


def _sm100a_record(monkeypatch):
    """Plan on the CPU with the sm_100a record (:func:`opt_in.record_for` patched; the dX slab counts one SM)."""
    name = cake_jit.select_module("sm_100a")
    monkeypatch.setattr(
        opt_in, "record_for", lambda device=None: (name, cake_jit.MODULES[name])
    )


def _smallest_peak(excinfo) -> int:
    """The smallest predicted step peak an explicit-budget ``ValueError`` names (:func:`opt_in._explicit_budget`)."""
    found = re.search(
        r"smallest predicted step peak is (\d+) bytes", str(excinfo.value)
    )
    assert found, str(excinfo.value)
    return int(found.group(1))


@pytest.mark.parametrize(
    "entry, helper",
    [
        ("loss", "_budget_plan"),
        ("logprob", "_budget_plan"),
        ("logprob", "_budget_plan_temperature"),
    ],
)
def test_budget_plan_explicit_budget_is_one_rule(monkeypatch, entry, helper):
    """The internal explicit budget (:func:`opt_in._set_memory_budget_bytes`) reaches the plans of every budget entry
    alike -- the loss entry's, the log-probability entry's and its temperature form's (:func:`opt_in._budget_plan` /
    :func:`opt_in._budget_plan_temperature`): ``None`` keeps the head rule; a budget the fused form fits gives it; the
    smallest predicted peak (named by the ``ValueError`` of a budget below every form) fits exactly; one byte below it
    raises again, naming the same peak; and the setter rejects a budget that is not positive."""
    _sm100a_record(monkeypatch)
    plan = getattr(opt_in, helper)
    T, R = 16231, 15419
    kw = dict(
        num_rows=T,
        rows=R,
        hidden=GLM_H,
        vocab=GLM_V,
        chunk=4096,
        compact=True,
        need_dx=True,
        fuse_dw_cast=True,
        buffer_rows=4096,
        x_copy=False,
        dx_finalize=True,
        budget_rows=R,
        vocab_major=entry == "logprob",
    )
    assert opt_in._memory_budget_bytes() is None
    head = plan(entry, CPU, **kw)
    head_bytes = 2 * R * GLM_V + 2 * GLM_V * GLM_H + 2 * T * GLM_H
    assert head.budget_bytes == head_bytes - math.ceil(
        head_bytes * opt_in.MEMORY_BUDGET_MARGIN
    )
    previous = opt_in._set_memory_budget_bytes(1 << 45)
    try:
        big = plan(entry, CPU, **kw)
        assert big.form == "fused" and big.fits and big.budget_bytes == 1 << 45, big
        opt_in._set_memory_budget_bytes(1)
        with pytest.raises(
            ValueError, match=f"no {entry} form fits the budget of 1 bytes"
        ) as excinfo:
            plan(entry, CPU, **kw)
        smallest = _smallest_peak(excinfo)
        assert 0 < smallest < big.predicted_bytes
        opt_in._set_memory_budget_bytes(smallest)
        floor = plan(entry, CPU, **kw)
        assert floor.fits and floor.predicted_bytes == smallest, floor
        assert floor.form in ("partial", "recompute", "vocab_major"), floor
        opt_in._set_memory_budget_bytes(smallest - 1)
        with pytest.raises(ValueError, match="no .* form fits") as excinfo:
            plan(entry, CPU, **kw)
        assert _smallest_peak(excinfo) == smallest
        for bad in (0, -1):
            with pytest.raises(ValueError, match="positive"):
                opt_in._set_memory_budget_bytes(bad)
        assert opt_in._memory_budget_bytes() == smallest - 1
    finally:
        opt_in._set_memory_budget_bytes(previous)
    assert opt_in._memory_budget_bytes() is None
    # the head rule never raises, even where its budget fits no form: the smallest form runs (a 1-row head)
    small = plan(
        entry, CPU, **dict(kw, num_rows=1, rows=1, buffer_rows=1, budget_rows=1)
    )
    assert not small.fits, small


class _Planned(Exception):
    """Raised by the patched :func:`opt_in.memory_plan`: the forward stops at its plan."""


@pytest.mark.parametrize(
    "case",
    ["ce", "ce_count", "ce_uncompacted", "policy_zero", "policy_zero_count"],
)
def test_forward_loss_explicit_budget_reaches_the_plan(monkeypatch, case):
    """The loss entry's forward (:func:`opt_in.forward_loss`, ``memory_mode="budget"``) plans its call against the
    internal explicit budget: ``memory_plan`` gets those bytes, the chunk loop's rows, the valid rows of the budget and
    the zero-weight rows -- with the call's own count (the hidden valid-row count off, whatever ``hidden_count``), a
    caller's ``num_valid_rows`` (no zero-weight split), an uncompacted call (every row in the chunk loop, the valid rows
    in the budget) and the zero-weight policy rows (a chunk loop over the gradient rows, a budget over every valid
    row).  Host only: the forward stops at the plan (the sm_100a record, :func:`_sm100a_record`)."""
    _sm100a_record(monkeypatch)
    seen = {}

    def planned(entry, **kw):
        seen.update(kw, entry=entry)
        raise _Planned

    hidden = []
    real_hidden = opt_in._hidden_count

    def spy_hidden(*args, **kw):
        hidden.append(kw["hidden_count"])
        return real_hidden(*args, **kw)

    monkeypatch.setattr(opt_in, "memory_plan", planned)
    monkeypatch.setattr(opt_in, "_hidden_count", spy_hidden)
    objective = "policy" if case.startswith("policy") else "ce"
    inp = _host_inputs(200, objective, device=CPU)
    zero = None
    if objective == "policy":
        zero = _zero_weight_set(inp, ("random", 0.3))
        inp = _with_zero_weights(inp, zero)
    T = inp.T
    R = int((inp.labels >= 0).sum())
    Z = 0 if zero is None else int(zero.sum())
    kw = dict(
        objective=objective,
        loss_div=inp.loss_div if objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=64,
        backend="cake",
        memory_mode="budget",
        hidden_count=True,
        return_logp=False,
    )
    if case.endswith("_count"):
        kw["num_valid_rows"] = R
    if case == "ce_uncompacted":
        kw["compact_rows"] = False
    previous = opt_in._set_memory_budget_bytes(123 << 20)
    try:
        with pytest.raises(_Planned):
            opt_in.forward_loss(inp.X, inp.W, inp.labels, **kw)
    finally:
        opt_in._set_memory_budget_bytes(previous)
    assert seen["entry"] == "loss" and seen["budget_bytes"] == 123 << 20, seen
    assert hidden in ([], [False]), hidden
    split = case == "policy_zero" and Z > 0
    rows = T if case == "ce_uncompacted" else R - (Z if split else 0)
    assert seen["num_rows"] == T and seen["rows"] == rows, (seen, T, R, Z)
    assert seen["budget_rows"] == R, (seen, R)
    assert seen["compact"] == (case != "ce_uncompacted"), seen
    assert seen["buffer_rows"] == min(64, rows), seen
    # without logp the zero-weight rows' logp-only pass does not run: counted in zero_total only
    assert seen["zero_rows"] == 0 and seen["zero_total"] == (Z if split else 0), seen
    assert seen["fuse_dw_cast"] and seen["need_dx"], seen


def _explicit_budget_step(inp, C, *, checkpoint=False, **kw):
    """One training step of the opt-in loss entry with its peak allocated bytes above the bytes before it (outputs and
    gradients count): ``(result, peak, memory_plan)``; ``checkpoint``: the call inside a non-reentrant
    ``torch.utils.checkpoint`` (its plan is read from the recomputed forward's handed result)."""
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    call = dict(
        objective=inp.objective,
        loss_div=inp.loss_div if inp.objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=C,
        return_logp=True,
        **kw,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    if checkpoint:
        loss, logp = torch.utils.checkpoint.checkpoint(
            lambda x, w: opt_in.cake_lm_head_nondet_train_loss(
                x, w, inp.labels, **call
            ),
            X,
            W,
            use_reentrant=False,
        )
    else:
        loss, logp = opt_in.cake_lm_head_nondet_train_loss(X, W, inp.labels, **call)
    grad_fn = loss.grad_fn
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    mplan = getattr(grad_fn, "memory_plan", None)
    result = dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad)
    return result, peak, mplan


@contextlib.contextmanager
def _explicit_budget_bytes(nbytes):
    """Run under the internal explicit budget ``nbytes`` (:func:`opt_in._set_memory_budget_bytes`), then restore."""
    previous = opt_in._set_memory_budget_bytes(nbytes)
    try:
        yield
    finally:
        opt_in._set_memory_budget_bytes(previous)


@pytest.mark.parametrize(
    "objective, T, C, count",
    [
        ("ce", 16384, 4096, False),
        ("policy", 16384, 4096, False),
        ("ce", 16384, 2048, False),
        ("ce", 4097, 4096, False),
        ("ce", 16384, 4096, True),
    ],
    ids=["ce", "policy", "ce_chunk2048", "ce_1chunk", "ce_num_valid_rows"],
)
def test_device_loss_explicit_budget_forms(glm_weight, objective, T, C, count):
    """The loss entry under the internal explicit budget (:func:`opt_in._set_memory_budget_bytes`) at the fused form's
    peak, one byte below it, a budget between the unfused and the smallest peak and the smallest peak itself runs
    ``"fused"``, then the next form in the order (``"unfused"`` where its peak is below the fused one, else a recompute
    form), a recompute form, and a recompute form at the smallest peak: each step -- the call that binds and one from
    the remembered binding -- peaks within the plan's prediction, itself within the budget; ``loss`` / ``logp`` / ``dW`` are the default step's
    bits and ``dX`` passes the gates.  The same step inside a non-reentrant checkpoint gives the same bits within the
    budget.  One byte below the smallest peak the call raises ``ValueError`` before its chunk loop, and the next call
    runs.  GLM ``H`` / ``V`` (5 % of the labels ignored); ``count``: the caller's ``num_valid_rows``."""
    _require_program(entry="loss")
    inp = make_inputs(
        T,
        objective=objective,
        seed=SEED + 6992 + T + C,
        device=CUDA,
        W=glm_weight,
        ignore_frac=0.05,
    )
    R = int((inp.labels >= 0).sum())
    extra = dict(num_valid_rows=R) if count else {}
    fast, _, _ = _explicit_budget_step(inp, C)
    with _explicit_budget_bytes(1 << 45):
        _, _, fused = _explicit_budget_step(inp, C, memory_mode="budget", **extra)
        _, _, unfused = _explicit_budget_step(
            inp, C, memory_mode="budget", fuse_dw_cast=False, **extra
        )
    assert fused.form == "fused" and unfused.form == "unfused", (fused, unfused)
    with (
        _explicit_budget_bytes(1),
        pytest.raises(ValueError, match="no loss form fits") as excinfo,
    ):
        _explicit_budget_step(inp, C, memory_mode="budget", **extra)
    smallest = _smallest_peak(excinfo)
    top = min(fused.predicted_bytes, unfused.predicted_bytes)
    cases = [
        (fused.predicted_bytes, ("fused",)),
        (
            fused.predicted_bytes - 1,
            ("unfused",)
            if unfused.predicted_bytes < fused.predicted_bytes
            else ("partial", "recompute"),
        ),
        ((smallest + top) // 2, ("partial", "recompute")),
        (smallest, ("partial", "recompute")),
    ]
    report = []
    for budget, forms in cases:
        peaks = []
        with _explicit_budget_bytes(budget):
            for _ in range(2):
                result, peak, mplan = _explicit_budget_step(
                    inp, C, memory_mode="budget", **extra
                )
                assert mplan is not None and mplan.form in forms, (budget, mplan)
                assert mplan.fits and mplan.budget_bytes == budget, (budget, mplan)
                for key in ("loss", "logp", "dW"):
                    assert _same_bits(result[key], fast[key]), (budget, mplan.form, key)
                _check_against_references(result, inp, ceiling=True)
                peaks.append(peak)
                del result
            assert max(peaks) <= mplan.predicted_bytes <= budget, (budget, mplan, peaks)
            # the checkpointed step: its recomputation runs the same form, released as early as the plain step's
            ckpt, ckpt_peak, _ = _explicit_budget_step(
                inp, C, checkpoint=True, memory_mode="budget", **extra
            )
            plain, _, _ = _explicit_budget_step(inp, C, memory_mode="budget", **extra)
            for key in ("loss", "logp", "dX", "dW"):
                assert _same_bits(ckpt[key], plain[key]), (budget, mplan.form, key)
            assert ckpt_peak <= budget, (budget, mplan, ckpt_peak, peaks)
            peaks.append(ckpt_peak)
            del ckpt, plain
        report.append(
            (
                budget,
                mplan.form,
                mplan.dw_rows,
                mplan.block_rows,
                mplan.predicted_bytes,
                peaks,
            )
        )
    with (
        _explicit_budget_bytes(smallest - 1),
        pytest.raises(ValueError, match="no loss form fits"),
    ):
        _explicit_budget_step(inp, C, memory_mode="budget", **extra)
    with _explicit_budget_bytes(smallest):
        result, peak, mplan = _explicit_budget_step(
            inp, C, memory_mode="budget", **extra
        )
    assert mplan.form in ("partial", "recompute") and peak <= smallest, (mplan, peak)
    assert _same_bits(result["dW"], fast["dW"])
    print(
        f"\nexplicit budget {objective} T={T} C={C} count={count} R={R}: fused {fused.predicted_bytes} unfused "
        f"{unfused.predicted_bytes} smallest {smallest}; "
        f"(budget, form, dw_rows, block_rows, predicted, peaks) {report}"
    )


@pytest.mark.parametrize(
    "entry, H, V, C, R, slab",
    [
        ("loss", GLM_H, GLM_V, 6144, 10240, 4 * 32 * 128 * GLM_H * 4),
        ("loss", 4096, 152064, 4096, 7680, 4 * 28 * 128 * 4096 * 4),
        ("logprob", 4096, 152064, 4096, 7680, 4 * 28 * 128 * 4096 * 4),
    ],
    ids=["loss_glm_chunk6144", "loss_h4096", "logprob_h4096"],
)
def test_budget_dx_slab_is_the_plans(monkeypatch, entry, H, V, C, R, slab):
    """The budget model's ordered dX slab (:func:`opt_in._budget_dx_slab`) is :attr:`Plan.dx_slab_bytes` of the plan
    the call's chunk loop binds: the largest slab over its chunks.  On B200's slice rule a shorter tail chunk takes
    more K slices than the full one (6144 rows: 1 slice, the 4096-row tail: 4; 4096 rows: 2, the 3584-row tail: 4), so
    the largest chunk's slab alone is smaller.  An explicit ``gemm_tuning`` slice count is the plan's too, and a call
    without dX has no slab.  Host only: the sm_100a record at B200's 148 SMs."""
    name = cake_jit.select_module("sm_100a")
    record = cake_jit.MODULES[name]
    geometry = cake_backend.Geometry.from_record(record)
    stages = opt_in.registered_stages(name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=B200_SMS),
    )
    device = torch.device("cuda", 0)
    T = R + 7  # a compacted call: the chunk loop runs over the valid rows

    def model(gemm_tuning=None, need_dx=True):
        return opt_in._budget_dx_slab(
            entry,
            device,
            record,
            stages,
            geometry,
            num_rows=T,
            rows=R,
            hidden=H,
            vocab=V,
            chunk=C,
            compact=True,
            need_dx=need_dx,
            gemm_tuning=gemm_tuning,
        )

    def binding(gemm_tuning=None):
        problem = cake_backend.Problem(
            num_rows=T,
            hidden=H,
            vocab=V,
            chunk=C,
            ld_x=H,
            x_copy=False,
            objective="ce",
            loss_div=None,
            grad_weight_dtype=torch.bfloat16,
            entry=entry,
        )
        return opt_in.make_plan(
            problem,
            need_dx=True,
            need_dw=True,
            geometry=geometry,
            dx_max_slices=cake_backend.dx_max_slices(stages),
            num_sms=B200_SMS,
            valid_rows=R,
            dx_resident=B200_DX_RESIDENT,
            fuse_dw_cast=True,
            arch=record["arch"],
            gemm_tuning=gemm_tuning,
        )

    plan = binding()
    first = cake_backend.dx_slab_bytes(
        plan.dx_slices_of(0),
        geometry.row_tiles(plan.chunks[0][1], geometry.dx_cluster_ctas),
        H,
    )
    print(
        f"\ndx slab {entry} H={H} C={C} R={R}: chunks {plan.chunks} slices {plan.dx_slices} model {model()} "
        f"largest chunk alone {first}"
    )
    assert model() == plan.dx_slab_bytes == slab, (model(), plan.dx_slab_bytes, slab)
    assert first < slab, (first, slab)
    tuning = {"dx": {"k_slices": 2}}
    assert model(tuning) == binding(tuning).dx_slab_bytes > 0
    assert model(need_dx=False) == 0


@pytest.mark.parametrize("budget_case", ["below_top", "smallest"])
def test_device_loss_budget_dx_slab_of_a_tail_chunk(glm_weight, budget_case):
    """A loss call whose shorter tail chunk takes more K slices than its full chunk (GLM ``H`` / ``V``, 10240 valid
    rows in chunks of 6144: on B200 one slice for the 6144-row chunk, four for the 4096-row tail) under the internal
    explicit budget one byte below the fused / unfused peaks and at the smallest peak: the plan counts the tail's
    ordered dX slab (:func:`opt_in._budget_dx_slab`), so the step -- the call that binds, one from the remembered
    binding and the same step in a non-reentrant checkpoint -- peaks within the plan's prediction, itself within the
    budget; ``loss`` / ``logp`` / ``dW`` are the default step's bits, ``dX`` passes the gates and the checkpointed step
    gives the plain step's bits."""
    _require_program(entry="loss")
    T, C = 10779, 6144  # round(0.05 * 10779) = 539 ignored rows: 10240 valid rows
    inp = make_inputs(T, seed=SEED + 6993, device=CUDA, W=glm_weight, ignore_frac=0.05)
    R = int((inp.labels >= 0).sum())
    assert R == 10240, R
    fast, _, _ = _explicit_budget_step(inp, C)
    with _explicit_budget_bytes(1 << 45):
        _, _, fused = _explicit_budget_step(inp, C, memory_mode="budget")
        _, _, unfused = _explicit_budget_step(
            inp, C, memory_mode="budget", fuse_dw_cast=False
        )
    with (
        _explicit_budget_bytes(1),
        pytest.raises(ValueError, match="no loss form fits") as excinfo,
    ):
        _explicit_budget_step(inp, C, memory_mode="budget")
    smallest = _smallest_peak(excinfo)
    budget = (
        min(fused.predicted_bytes, unfused.predicted_bytes) - 1
        if budget_case == "below_top"
        else smallest
    )
    peaks = []
    with _explicit_budget_bytes(budget):
        for _ in range(2):
            result, peak, mplan = _explicit_budget_step(inp, C, memory_mode="budget")
            assert mplan is not None and mplan.form in ("partial", "recompute"), mplan
            assert mplan.fits and mplan.budget_bytes == budget, (budget, mplan)
            for key in ("loss", "logp", "dW"):
                assert _same_bits(result[key], fast[key]), (budget, mplan.form, key)
            _check_against_references(result, inp, ceiling=True)
            peaks.append(peak)
            plain = result
            del result
        ckpt, ckpt_peak, _ = _explicit_budget_step(
            inp, C, checkpoint=True, memory_mode="budget"
        )
        for key in ("loss", "logp", "dX", "dW"):
            assert _same_bits(ckpt[key], plain[key]), (budget, mplan.form, key)
    print(
        f"\nloss budget dx slab of a tail chunk ({budget_case}): fused {fused.predicted_bytes} unfused "
        f"{unfused.predicted_bytes} smallest {smallest} budget {budget} form {mplan.form} dw_rows {mplan.dw_rows} "
        f"predicted {mplan.predicted_bytes} peaks {peaks} checkpointed {ckpt_peak}"
    )
    assert max(peaks) <= mplan.predicted_bytes <= budget, (budget, mplan, peaks)
    assert ckpt_peak <= budget, (budget, mplan, ckpt_peak)


@pytest.mark.parametrize("placement", ["auto", "forward"])
def test_device_loss_explicit_budget_raises_where_the_call_plans(glm_weight, placement):
    """An explicit budget that no loss form fits raises ``ValueError`` where the call plans: in the forward of a plain
    call; inside a non-reentrant ``torch.utils.checkpoint`` under the gradient placement ``"auto"`` the checkpointed
    forward computes the loss only and plans nothing, so the backward raises (its recomputation runs the forward that
    plans), and under ``"forward"`` the checkpointed forward raises.  The next checkpointed step under a budget that
    fits runs and gives the plain step's bits."""
    _require_program(entry="loss")
    inp = make_inputs(
        8192, seed=SEED + 6994, device=CUDA, W=glm_weight, ignore_frac=0.05
    )
    call = dict(loss_div=inp.loss_div, chunk_size=4096, memory_mode="budget")

    def head(x, w):
        return opt_in.cake_lm_head_nondet_train_loss(x, w, inp.labels, **call)

    previous = cake_backend.set_gradient_placement(placement)
    try:
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        with _explicit_budget_bytes(1):
            with pytest.raises(ValueError, match="no loss form fits"):
                head(X, W)
            if placement == "auto":
                loss = torch.utils.checkpoint.checkpoint(
                    head, X, W, use_reentrant=False
                )
                assert torch.isfinite(loss).all()
                with pytest.raises(ValueError, match="no loss form fits"):
                    loss.backward()
                del loss
            else:
                with pytest.raises(ValueError, match="no loss form fits"):
                    torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
        X.grad = W.grad = None
        with _explicit_budget_bytes(1 << 45):
            plain = head(X, W)
            plain.backward()
            want = dict(loss=plain.detach(), dX=X.grad, dW=W.grad)
            X.grad = W.grad = None
            loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
            loss.backward()
            got = dict(loss=loss.detach(), dX=X.grad, dW=W.grad)
    finally:
        cake_backend.set_gradient_placement(previous)
    for key in ("loss", "dX", "dW"):
        assert _same_bits(got[key], want[key]), (placement, key)


def test_logprob_vm_auto_rows():
    """The auto rule's row counts (:func:`cake_backend._logprob_vm_auto_rows`): 2 .. 4 chunks, except 2 chunks whose
    second chunk holds at least half a chunk."""
    f = cake_backend._logprob_vm_auto_rows
    assert cake_backend.VM_AUTO_MAX_CHUNKS == 4
    assert [f(r, 4096) for r in (4096, 4097, 6143, 6144, 8192, 8193, 16384, 16385)] == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
    ]


def test_logprob_vm_dx_tuning():
    """The vocabulary-major dX knobs (:func:`cake_backend._logprob_vm_dx_tuning`): on sm_100a (148 SMs) the one launch
    per block of a 7693-row GLM call (372 wide items on 74 SM pairs: a last wave of 2 items) would run the long-chunk
    ring, which takes no split there -- it runs the table ring with the split instead; a call whose last wave is full
    enough (15420 rows: 66 items) keeps the rule, as does a record without the split forms and sm_103a (its long ring
    splits)."""
    name = cake_jit.select_module("sm_100a")
    geometry = cake_backend.Geometry.from_record(cake_jit.MODULES[name])
    registered = set(cake_jit.registered_stages(name))
    tune = cake_backend._logprob_vm_dx_tuning(
        7693, GLM_H, B200_SMS, geometry, "sm_100a", 2, registered
    )
    assert tune == {"dx": {"stages": None}}, tune
    base = cake_backend.chunk_variants(
        7693, GLM_H, B200_SMS, 1, geometry, "sm_100a", None, num_chunks=2
    )
    assert base.dx_stages == 3 and base.dx_split is None, base
    tuned = cake_backend.chunk_variants(
        7693, GLM_H, B200_SMS, 1, geometry, "sm_100a", tune, num_chunks=2
    )
    assert tuned.dx_stages is None and tuned.dx_split == 2, tuned
    assert tuned.dx_tile_n is None and not tuned.dx_swab, tuned
    assert (
        cake_backend._logprob_vm_dx_tuning(
            15420, GLM_H, B200_SMS, geometry, "sm_100a", 4, registered
        )
        is None
    )
    assert (
        cake_backend._logprob_vm_dx_tuning(
            7693, GLM_H, B200_SMS, geometry, "sm_100a", 2, set()
        )
        is None
    )
    name103 = cake_jit.select_module("sm_103a")
    assert (
        cake_backend._logprob_vm_dx_tuning(
            7693,
            GLM_H,
            B200_SMS,
            cake_backend.Geometry.from_record(cake_jit.MODULES[name103]),
            "sm_103a",
            2,
            set(cake_jit.registered_stages(name103)),
        )
        is None
    )


@pytest.mark.parametrize("tau", [None, "scalar", "hetero"])
@pytest.mark.parametrize("T", [12000, 24000])
def test_device_budget_vocab_major_logprob(glm_weight, monkeypatch, T, tau):
    """``memory_mode="budget"`` of the log-probability entry (no temperature, a scalar and a per-token one) at a budget
    below the chunk loop's fused peak (the internal override) runs the vocabulary-major backward, at any chunk count
    (here 3 and 6 chunks of 4096 rows; the default form takes it for 2 .. 4 chunks only): ``logp`` / ``dX`` / ``dW``
    are the vocabulary-major backward's bits (mode ``"1"`` on the default entry), the gates pass (the production
    ceiling where every temperature is at least 1), and the step peaks within the plan's prediction, itself within the
    budget."""
    _require_program(entry="logprob")
    calls = []
    real = cake_backend._backward_logprob_vm

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_backward_logprob_vm", counted)
    inp = make_inputs(
        T,
        seed=SEED + 7600 + T,
        device=CUDA,
        W=glm_weight,
        ignore_frac=0.05,
        temperature=tau,
    )
    tkw = {} if tau is None else dict(temperature=inp.temperature)
    # the production ceiling is a temperature-1 figure: only where every temperature is at least 1 (the suite's rule)
    ceiling = tau is None or bool(torch.as_tensor(inp.temperature).min() >= 1.0)
    R, H, V = int(inp.valid.sum()), int(inp.H), int(inp.V)
    head = 2 * R * V + 2 * V * H + 2 * T * H
    plan_kw = dict(
        num_rows=T,
        rows=R,
        hidden=H,
        vocab=V,
        chunk=4096,
        buffer_rows=min(4096, R),
        budget_rows=R,
        geometry=cake_backend.Geometry.from_record(cake_backend.record_for(CUDA)[1]),
        vocab_major=True,
    )
    # a budget below the chunk loop's fused / unfused peaks: the vocabulary-major form is the fastest that fits
    fused = opt_in.memory_plan("logprob", budget_bytes=1 << 45, **plan_kw)
    budget = fused.predicted_bytes - (64 << 20)
    predicted = opt_in.memory_plan("logprob", budget_bytes=budget, **plan_kw)
    with _vocab_major("1"):
        forced, _, _ = _budget_step(inp, "logprob", **tkw)
    calls.clear()
    peaks = []
    previous = opt_in._set_memory_budget_bytes(budget)
    try:
        for _ in range(2):
            result, peak, _ = _budget_step(inp, "logprob", memory_mode="budget", **tkw)
            for key in ("logp", "dX", "dW"):
                assert _same_bits(result[key], forced[key]), (T, tau, key)
            _check_against_references(result, inp, entry="logprob", ceiling=ceiling)
            peaks.append(peak)
            del result
    finally:
        opt_in._set_memory_budget_bytes(previous)
    assert len(calls) == 2, (T, tau, calls)
    print(
        f"\nbudget vocab-major logprob T={T} tau={tau}: peaks {peaks} head {head} budget {budget} predicted "
        f"{predicted.predicted_bytes} ({predicted.form}) fused {fused.predicted_bytes}"
    )
    assert predicted.form == "vocab_major", predicted
    assert max(peaks) <= predicted.predicted_bytes <= budget, (peaks, predicted, budget)


def _spy_budget_plans(monkeypatch):
    """Record every plan :func:`opt_in._budget_plan` / :func:`opt_in._budget_plan_temperature` returns (the plan a
    budget call runs)."""
    plans = []
    for name in ("_budget_plan", "_budget_plan_temperature"):
        real = getattr(opt_in, name)

        def spy(*args, _real=real, **kw):
            mplan = _real(*args, **kw)
            plans.append(mplan)
            return mplan

        monkeypatch.setattr(opt_in, name, spy)
    return plans


@pytest.mark.parametrize("tau", ["int", "cpu_tensor"])
def test_device_budget_logprob_temperature_one_is_the_budget_call(
    glm_weight, monkeypatch, tau
):
    """A log-probability call with ``memory_mode="budget"`` and a temperature that resolves to none (an FP32 value of
    1.0 given as an int or a 0-dim CPU tensor) is the budget call without a temperature: at an explicit budget below
    the chunk loop's fused peak it runs the vocabulary-major backward its plan picks -- here for 2 chunks whose tail
    holds more than half a chunk, which the default form's auto rule leaves to the chunk loop --, with the budget call's
    ``logp`` / ``dX`` / ``dW`` bits (the vocabulary-major backward's), the gates and a peak within the plan's
    prediction, itself within the budget."""
    _require_program(entry="logprob")
    calls = []
    real = cake_backend._backward_logprob_vm

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_backward_logprob_vm", counted)
    T = 7368  # round(0.05 * 7368) = 368 ignored rows: 7000 valid rows, chunks of 4096 and 2904
    inp = make_inputs(T, seed=SEED + 7700, device=CUDA, W=glm_weight, ignore_frac=0.05)
    R = int(inp.valid.sum())
    assert R == 7000 and not cake_backend._logprob_vm_auto_rows(R, 4096), R
    value = 1 if tau == "int" else torch.tensor(1.0)
    plans = _spy_budget_plans(monkeypatch)
    with _explicit_budget_bytes(1 << 45):
        _budget_step(inp, "logprob", memory_mode="budget")
    fused = plans[-1]
    assert fused.form == "fused", fused
    budget = fused.predicted_bytes - (64 << 20)
    with _explicit_budget_bytes(budget):
        want, _, _ = _budget_step(inp, "logprob", memory_mode="budget")
        assert plans[-1].form == "vocab_major", plans[-1]
        calls.clear()
        del plans[:]
        peaks = []
        for _ in range(2):
            result, peak, _ = _budget_step(
                inp, "logprob", memory_mode="budget", temperature=value
            )
            for key in ("logp", "dX", "dW"):
                assert _same_bits(result[key], want[key]), (tau, key)
            _check_against_references(result, inp, entry="logprob", ceiling=True)
            peaks.append(peak)
            del result
    assert len(calls) == 2, (tau, calls)
    assert [p.form for p in plans] == ["vocab_major"] * 2, plans
    predicted = plans[-1]
    print(
        f"\nbudget logprob temperature-one ({tau}) R={R}: peaks {peaks} budget {budget} predicted "
        f"{predicted.predicted_bytes} fused {fused.predicted_bytes}"
    )
    assert max(peaks) <= predicted.predicted_bytes <= budget, (peaks, predicted)


def test_device_budget_vocab_major_block_labels(glm_weight, monkeypatch):
    """The vocabulary-major budget model counts the block-shifted labels at their build peak (the labels' int64 dtype:
    the shifted labels, the block test's mask, the foreign-label fill and the result live at once): at an explicit
    budget equal to the vocabulary-major form's predicted peak of a call of many vocabulary blocks (62259 valid rows at
    a chunk of 1024 rows, GLM ``H`` / ``V``) the call runs that form, with the forced vocabulary-major backward's
    ``logp`` / ``dX`` / ``dW`` bits, and peaks within the budget."""
    _require_program(entry="logprob")
    T, C = 65536, 1024
    inp = make_inputs(T, seed=SEED + 7701, device=CUDA, W=glm_weight, ignore_frac=0.05)
    R = int(inp.valid.sum())
    with _vocab_major("1"):
        forced, _, _ = _budget_step(inp, "logprob", chunk_size=C)
    plans = _spy_budget_plans(monkeypatch)
    with _explicit_budget_bytes(1 << 45):
        _budget_step(inp, "logprob", chunk_size=C, memory_mode="budget")
    fused = plans[-1]
    with _explicit_budget_bytes(fused.predicted_bytes - 1):
        _budget_step(inp, "logprob", chunk_size=C, memory_mode="budget")
    vm = plans[-1]
    assert vm.form == "vocab_major", vm
    budget = vm.predicted_bytes
    peaks = []
    with _explicit_budget_bytes(budget):
        for _ in range(2):
            result, peak, _ = _budget_step(
                inp, "logprob", chunk_size=C, memory_mode="budget"
            )
            assert plans[-1].form == "vocab_major" and plans[-1].fits, plans[-1]
            for key in ("logp", "dX", "dW"):
                assert _same_bits(result[key], forced[key]), key
            peaks.append(peak)
            del result
    print(
        f"\nbudget vocab-major block labels R={R} C={C}: peaks {peaks} budget = vm predicted {budget} fused "
        f"{fused.predicted_bytes}"
    )
    assert max(peaks) <= budget, (peaks, budget)


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
        fn = (
            cake_backend.cake_lm_head_nondet_train_loss
            if package == "nd"
            else det_backend.cake_lm_head_det_train_loss
        )

        def head(X, W):
            return fn(X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096)

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
    nd_fwd = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
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
        "loss-no-grad": lambda: opt_in.cake_lm_head_nondet_train_loss(
            frozen, inp.W, inp.labels, loss_div=inp.loss_div, **common
        ),
        "logprob-no-grad": lambda: opt_in.cake_lm_head_nondet_train_logprob(
            frozen, inp.W, inp.labels, **common
        ),
        "logprob-budget": lambda: opt_in.cake_lm_head_nondet_train_logprob(
            X, inp.W, inp.labels, memory_mode="budget", **common
        ),
        "loss": lambda: opt_in.cake_lm_head_nondet_train_loss(
            X, inp.W, inp.labels, loss_div=inp.loss_div, **common
        ),
        "loss-budget": lambda: opt_in.cake_lm_head_nondet_train_loss(
            X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            memory_mode="budget",
            **common,
        ),
        "loss-uncompacted": lambda: opt_in.cake_lm_head_nondet_train_loss(
            X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            compact_rows=False,
            **common,
        ),
        "logprob": lambda: opt_in.cake_lm_head_nondet_train_logprob(
            X, inp.W, inp.labels, **common
        ),
        "logprob-uncompacted": lambda: opt_in.cake_lm_head_nondet_train_logprob(
            X, inp.W, inp.labels, compact_rows=False, **common
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
    ``dW`` bits (the deterministic package's), and ``dX`` within the gates after a 0xFF-poisoned allocator: every
    ``dX_acc`` row is stored (a one-slice chunk's GEMM, a K-sliced chunk's slab sum) when the weight-gradient GEMMs run
    on the side stream."""
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
    streams, count cells) and gives the main thread's ``loss`` / ``logp`` / ``dW`` bits;
    ``cake_lm_head_nondet_train_preload`` forms the calling thread's state ahead of its first call."""
    _require_program(entry="loss")
    import flashinfer.cake_lm_head_nondet_train as public

    inp = _device_inputs(12289, "ce", W=glm_weight)
    main = _run(inp, 4096, backend="cake")
    out: dict = {}

    def fresh():
        try:
            out["state_before"] = getattr(
                cake_backend._THREAD_STATE, "dw_streams", None
            )
            out["result"] = _run(inp, 4096, backend="cake")
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            out["error"] = exc

    def preloaded():
        try:
            public.cake_lm_head_nondet_train_preload(CUDA)
            index = torch.cuda.current_device()
            out["preloaded"] = (
                index in cake_backend._thread_state("dw_streams"),
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
    assert out["state_before"] is None
    for key in ("loss", "logp", "dW"):
        assert _same_bits(out["result"][key], main[key]), key
    _check_against_references(out["result"], inp, ceiling=True)
    assert out["preloaded"] == (True, True)


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
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs  # noqa: E402
from tests.cake.test_helpers.cake_package import cake_package  # noqa: E402

backend = cake_package("cake.cake_lm_head_nondet_train", "cake_backend")
modes = []
set_mode = torch.cuda.set_sync_debug_mode
torch.cuda.set_sync_debug_mode = lambda mode: (modes.append(mode), set_mode(mode))[1]
inp = make_inputs(
    int(sys.argv[2]), seed=int(sys.argv[3]), device=torch.device("cuda"), ignore_frac=0.0
)
X = inp.X.detach().requires_grad_(True)
W = inp.W.detach().requires_grad_(True)
assert backend._device_constants.cache_info().currsize == 0
loss, _ = backend.cake_lm_head_nondet_train_loss(
    X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, return_logp=True
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
            loss, logp = cake_backend.cake_lm_head_nondet_train_loss(
                X,
                W,
                inp.labels,
                loss_div=inp.loss_div,
                chunk_size=4096,
                return_logp=True,
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


# Cycles of the delay that holds both backwards of the two-stream tests (about a second at the B200 boost clock)
GATE_CYCLES = 1 << 31
# Runs of the two-stream sequence a test makes until one overlaps; with none it skips
TWO_STREAM_ATTEMPTS = 4


def _same_inputs_on_two_streams(inp, entry: str, *, tuning=None):
    """A serial call of ``entry`` that binds, then two calls with the same ``X`` / ``W`` / ``labels`` (and the same
    ``gemm_tuning`` ``tuning``) on two streams of this thread: the forwards on the two streams, then both backwards --
    the autograd engine runs each on its forward's stream -- queued behind one delay on a third stream.  A timing event
    after every remembered launch records when each call's launches complete (per call and phase).  Host clock
    readings give ``host_gap`` (ms from queuing the first call to queuing the second) and ``host_bwd`` (ms from queuing
    the backwards' delay to both backwards queued)."""
    tune = {} if tuning is None else dict(gemm_tuning=tuning)
    serial = _run(inp, 4096, entry=entry, backend="cake", **tune)  # binds here
    cache = cake_backend.BINDING_CACHE
    bound = len(cache)
    assert bound > 0 and cache.owned_bytes == 0
    streams = [torch.cuda.Stream() for _ in range(2)]
    handles = [stream.cuda_stream for stream in streams]
    gate = torch.cuda.Stream()
    done: dict = {(i, phase): [] for i in (0, 1) for phase in ("fwd", "bwd")}
    phase = ["fwd"]
    prepared: list = []
    starts: list = []  # host clock (s) as each call is queued
    run_keys, prepare = cake_backend._run_keys, cake_backend.prepare_lm_head_loss

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

    calls = []
    torch.cuda.synchronize()
    origin = torch.cuda.Event(enable_timing=True)
    origin.record()
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    cake_backend._run_keys, cake_backend.prepare_lm_head_loss = recording, counting
    try:
        for stream in streams:
            starts.append(time.perf_counter())
            with torch.cuda.stream(stream):
                X = inp.X.detach().requires_grad_(True)
                W = inp.W.detach().requires_grad_(True)
                if entry == "loss":
                    out, logp = cake_backend.cake_lm_head_nondet_train_loss(
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
                        **tune,
                    )
                    upstream = None
                else:
                    out = logp = cake_backend.cake_lm_head_nondet_train_logprob(
                        X,
                        W,
                        inp.labels,
                        chunk_size=4096,
                        backend="cake",
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
        host_gap=round((starts[1] - starts[0]) * 1e3, 3),
        host_bwd=round(host_bwd * 1e3, 3),
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
            f"backwards end {run.ends} ms; remembered launches (count, first, last ms) {run.spans}"
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
    time to queue both backwards and the end of their delay -- so a skip on every run can be checked against them (a
    host slowed by other work, or a call that waits for the device)."""
    if run is None:
        times = [
            f"run {i + 1}: host gap {r.host_gap} ms, forwards (count, first, last ms) A {r.spans['A fwd']} "
            f"B {r.spans['B fwd']}; backwards queued {r.queued} in {r.host_bwd} ms, delay ends {r.released:.3f} ms, "
            f"backwards (count, first, last ms) A {r.spans['A bwd']} B {r.spans['B bwd']}"
            for i, r in enumerate(runs)
        ]
        pytest.skip(
            f"no {what} run in {len(runs)} runs (the second call was not queued in time): "
            + "; ".join(times)
        )


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_same_inputs_on_two_streams_at_once(
    logprob_chunk_loop, glm_weight, entry, request
):
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
    same inputs, two calls storing into one buffer would write the same values, and the dX path holds no accumulator
    two calls could add into: every ``dX_acc`` row is stored (a one-slice chunk's GEMM, a K-sliced chunk's slab sum)."""
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


def test_device_logprob_backward_binding_holds_no_lse_or_dlogp_address(glm_weight):
    """The log-probability backward's binding key holds the layout of ``lse`` / ``dlogp``, not their addresses.
    Backward calls on one forward whose ``lse`` and ``dlogp`` sit at a new address every time (copies kept alive) bind
    once: the later calls are binding-cache hits that prepare nothing and give the first call's ``dW`` bits and ``dX``
    within the gates.  The autograd entry's second step prepares nothing either, after a 1 GiB allocator poison
    between the steps (each forward allocates a fresh ``lse``)."""
    _require_program(entry="logprob")
    inp = _device_inputs(4097, "ce", W=glm_weight)
    fwd = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
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


@pytest.fixture
def shipped_chunks(monkeypatch):
    """The shipped chunk plan (no short-tail fold, :func:`cake_backend.tail_fold_eligible` patched off) for a test of
    another feature on a two-chunk call with a short last chunk; remembered bindings are dropped before and after."""
    cake_backend.BINDING_CACHE.clear()
    monkeypatch.setattr(
        cake_backend, "tail_fold_eligible", lambda num_rows, chunk: False
    )
    yield
    cake_backend.BINDING_CACHE.clear()


@pytest.mark.parametrize(
    "objective, entry", [("ce", "loss"), ("policy", "loss"), ("ce", "logprob")]
)
def test_device_short_tail_fold(logprob_chunk_loop, glm_weight, objective, entry):
    """The short-tail fold on the device: T 4600 at C 4096 with 5 % ignored rows (the V07 shape: 4370 loop rows, a
    274-row tail) runs ONE chunk of 4370 rows -- one launch of each GEMM (launch record; the log-probability backward
    under the zero-gradient row skip's counted forms too, its chunk loop pinned: the vocabulary-major auto rule takes
    this two-chunk call otherwise) and, with the fused cast, no FP32 ``dW_acc`` -- passes the
    deterministic suite's gates against the FP64 oracle, and ``loss`` / ``logp`` / ``dW`` repeat bit for bit."""
    _require_program(entry=entry)
    inp = _device_inputs(4600, objective, W=glm_weight)
    rows = int((inp.labels >= 0).sum())
    assert 4096 < rows < 6144, rows
    cake_backend.BINDING_CACHE.clear()
    with _launch_record() as launched:
        nd = _run(inp, 4096, entry=entry, backend="cake")
    gemms = [key for key in launched if key[0].startswith("gemm_")]
    assert gemms and {key[1] for key in gemms} == {0}, gemms
    again = _run(inp, 4096, entry=entry, backend="cake")
    for key in ("loss", "logp", "dW"):
        assert _same_bits(nd[key], again[key]), key
    _check_against_references(nd, inp, entry=entry, ceiling=True)
    if objective == "ce" and entry == "loss":
        fr = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, loss_div=inp.loss_div, chunk_size=4096
        )
        assert fr.dw_acc is None and "dW_acc" not in fr.memory["accumulators"]
        assert fr.memory["num_chunks"] == 1 and fr.memory["vocab_rows_max"] == rows


def test_device_red_min_rows_keeps_small_chunks_on_the_plain_store(
    glm_weight, shipped_chunks
):
    """``set_dx_red_min_rows``: a two-chunk call whose one-row tail falls below the floor runs that chunk with the
    plain store -- its ``dX`` rows are the deterministic package's one-slice bits -- while the 4096-row chunk keeps
    the reduction (launch record); the call passes the gates.  The shipped chunks (no short-tail fold)."""
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
    split): ``loss`` / ``logp`` / ``dW`` are the bits of the deterministic package's plain path (its own zero-weight
    split off) and the zero-weight rows' ``dX`` rows are zeros."""
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
        )
    finally:
        cake_backend.set_zero_weight_min_rows(previous)
    assert "zero_weight_rows" not in (fwd.memory or {})
    with _det_unsplit():
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
        loss, logp = be.cake_lm_head_nondet_train_loss(
            torch.tanh(a),
            b,
            inp.labels,
            objective=inp.objective,
            loss_div=inp.loss_div if inp.objective == "ce" else None,
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=C,
            return_logp=True,
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
    )
    X = inp.X.detach().requires_grad_(graph != "frozen_inputs")
    W = inp.W.detach().requires_grad_(graph != "frozen_inputs")
    fused = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)

    def call():
        if graph == "no_grad":
            with torch.no_grad():
                return cake_backend.cake_lm_head_nondet_train_loss(
                    X, W, inp.labels, **kw
                )
        if graph == "inference_mode":
            with torch.inference_mode():
                return cake_backend.cake_lm_head_nondet_train_loss(
                    X, W, inp.labels, **kw
                )
        return cake_backend.cake_lm_head_nondet_train_loss(X, W, inp.labels, **kw)

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
            loss = cake_backend.cake_lm_head_nondet_train_loss(
                inp.X,
                inp.W,
                inp.labels,
                objective="policy",
                chunk_size=64,
                backend="reference",
                **ops,
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
    """Under torch's deterministic algorithms a checkpointed call keeps its fused forward: the alert of a K-sliced dX
    (forced: the reference backend plans one slice) stays raised in the checkpointed forward, before its dX launch, as
    without a checkpoint.  With one-slice dX chunks (the reference backend's plan) the checkpointed step, its
    recomputation and its backward run without an alert.  Under ``torch.no_grad`` the call launches no gradient GEMM
    and does not alert."""
    inp = _host_inputs(37, device=CPU)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def head(a, b):
        return cake_backend.cake_lm_head_nondet_train_loss(
            a, b, inp.labels, loss_div=1.0, chunk_size=16, backend="reference"
        )

    with (
        _forced_dx_slices(),
        _torch_deterministic(True),
        pytest.raises(RuntimeError, match=f"forward_loss {_ALERT}"),
    ):
        torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
    with _torch_deterministic(True), warnings.catch_warnings():
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
        torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False).backward()
    assert X.grad is not None and W.grad is not None
    with (
        _forced_dx_slices(),
        _torch_deterministic(True),
        warnings.catch_warnings(),
        torch.no_grad(),
    ):
        warnings.filterwarnings("error", message=f".*{_ALERT}.*")
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
        return cake_backend.cake_lm_head_nondet_train_loss(
            torch.tanh(a),
            b,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=64,
            backend="reference",
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
        return cake_backend.cake_lm_head_nondet_train_loss(
            torch.tanh(a),
            b,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=64,
            backend="reference",
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
            loss, logp = opt_in.cake_lm_head_nondet_train_loss(
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
    )
    fused = cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    mode = torch.no_grad() if graph == "no_grad" else torch.inference_mode()
    with mode, _launch_record() as launched:
        loss, logp = cake_backend.cake_lm_head_nondet_train_loss(X, W, inp.labels, **kw)
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
            logp = cake_backend.cake_lm_head_nondet_train_logprob(
                a, b, inp.labels, chunk_size=4096, **one
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


@contextlib.contextmanager
def _logprob_forwards():
    """The log-probability forwards (``forward_logprob`` / ``forward_logprob_temperature``) the main module's autograd
    entries run, in order."""
    seen: list = []
    originals = (cake_backend.forward_logprob, cake_backend.forward_logprob_temperature)

    def recording(original, name):
        def forward(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return forward

    cake_backend.forward_logprob = recording(originals[0], "logprob")
    cake_backend.forward_logprob_temperature = recording(originals[1], "temperature")
    try:
        yield seen
    finally:
        cake_backend.forward_logprob, cake_backend.forward_logprob_temperature = (
            originals
        )


def _logprob_temperature(inp, form):
    if form is None:
        return None
    if form == "scalar":
        return 0.7
    return make_temperature(inp.T, "hetero", seed=inp.seed, device=inp.X.device)


def _logprob_step(inp, kind, form, *, C=64, between=None, **kw):
    """One step of the log-probability entry (``form``: no temperature, a scalar or a per-token one) under ``kind``;
    ``between(X, W, logp)`` runs after the forward, before the backward.  Returns ``logp`` / ``dX`` / ``dW``."""
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    tau = _logprob_temperature(inp, form)
    logps = []

    def head(a, b):
        if tau is None:
            logp = cake_backend.cake_lm_head_nondet_train_logprob(
                a, b, inp.labels, chunk_size=C, **kw
            )
        else:
            logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
                a, b, inp.labels, chunk_size=C, temperature=tau, **kw
            )
        logps.append(logp)
        return (logp * inp.dlogp).sum()

    loss = _checkpointed(kind, head, X, W)
    if between is not None:
        between(X, W, logps[0])
    loss.backward()
    if X.is_cuda:
        torch.cuda.synchronize()
    return dict(logp=logps[0].detach(), dX=X.grad, dW=W.grad)


# Forwards a checkpointed log-probability step runs.  The recomputation takes the checkpointed call's row results
# when the checkpoint's hooks are innermost and it gets the same input tensors; the reentrant first pass has no graph
# (nothing kept) and ``copies_outside`` hands the recomputation copies (both run the forward again);
# ``offload_inside`` puts other hooks innermost, which save the call's tensors themselves (no recomputation of the
# call).  ``nested`` and ``offload_outside`` (the same tensors on the CPU only) run one or two.
_LOGPROB_FORWARDS = dict(
    reentrant=2,
    nonreentrant=1,
    nonreentrant_no_early_stop=1,
    nonreentrant_no_determinism_check=1,
    copies_outside=2,
    offload_inside=1,
)


@pytest.mark.parametrize("form", [None, "scalar", "tensor"])
@pytest.mark.parametrize("kind", CHECKPOINT_KINDS)
def test_checkpointed_logprob_step_runs_the_forward_once(kind, form):
    """A non-reentrant checkpointed log-probability step runs the logits forward once: the recomputation takes the
    checkpointed call's ``logp`` and row results instead of running it again.  ``logp`` / ``dX`` / ``dW`` are the
    uncheckpointed step's bits for every checkpoint kind; ``"forward"`` placement keeps the second forward, with the
    same bits.  No frame record is left behind."""
    inp = _host_inputs(200, "ce")
    plain = _logprob_step(inp, "plain", form, backend="reference")
    with _logprob_forwards() as seen:
        run = _logprob_step(inp, kind, form, backend="reference")
    taken = len(seen)
    want = _LOGPROB_FORWARDS.get(kind)
    assert taken == want if want is not None else 1 <= taken <= 2, seen
    _assert_step_bits(run, plain, keys=("logp", "dX", "dW"))
    if kind not in ("reentrant", "offload_inside"):
        with _placement("forward"), _logprob_forwards() as seen:
            again = _logprob_step(inp, kind, form, backend="reference")
        if want is None:
            assert len(seen) >= taken, seen
        else:
            assert len(seen) == 2, seen
        _assert_step_bits(again, plain, keys=("logp", "dX", "dW"))
    gc.collect()
    assert len(cake_backend._LOGPROB_CHECKPOINT_CALLS) == 0


@pytest.mark.parametrize("form", [None, "tensor"])
@pytest.mark.parametrize("change", ["labels", "logp", "options"])
def test_checkpointed_logprob_recomputation_runs_the_forward_after_a_change(
    change, form
):
    """The recomputation runs the forward again when an input of the checkpointed call (here the labels, which the
    checkpointed function reads from outside the region) or its kept ``logp`` changed in place after the forward, or
    when it calls the entry with other options: ``dX`` / ``dW`` are then those of an uncheckpointed step on the changed
    labels (the options of the checkpointed call, which its backward runs), never from a stale row result."""
    inp = _host_inputs(200, "ce")
    if change == "options":
        # a checkpointed function whose recomputation passes another chunk size than its first pass; the backward
        # takes the recomputation's row results (``lse``), the first pass's bits where the logits GEMM's bits do not
        # depend on the chunk's row count: the cake kernels and the CUDA reference, not torch's multi-threaded CPU BF16
        # GEMM (one row's ``lse`` 1 ulp apart at chunks 64 / 32 for these inputs), so the CPU runs it on one thread
        threads = torch.get_num_threads()
        if inp.X.device.type == "cpu":
            torch.set_num_threads(1)
        try:
            sizes = iter((64, 32))
            X = inp.X.detach().requires_grad_(True)
            W = inp.W.detach().requires_grad_(True)
            tau = _logprob_temperature(inp, form)

            def head(a, b):
                kw = dict(chunk_size=next(sizes), backend="reference")
                if tau is not None:
                    kw["temperature"] = tau
                    entry = cake_backend.cake_lm_head_nondet_train_logprob_temperature
                else:
                    entry = cake_backend.cake_lm_head_nondet_train_logprob
                return (entry(a, b, inp.labels, **kw) * inp.dlogp).sum()

            with _logprob_forwards() as seen:
                torch.utils.checkpoint.checkpoint(
                    head, X, W, use_reentrant=False, determinism_check="none"
                ).backward()
            assert len(seen) == 2, seen
            # the backward is the checkpointed call's (chunk size 64); the recomputed row results are the same bits
            plain = _logprob_step(inp, "plain", form, C=64, backend="reference")
        finally:
            torch.set_num_threads(threads)
        assert _same_bits(X.grad, plain["dX"])
        assert _same_bits(W.grad, plain["dW"])
        return

    own = dataclasses.replace(inp, labels=inp.labels.clone())
    ignored = own.labels.clone()
    ignored[: own.T // 2] = -100

    def between(X, W, logp):
        with torch.no_grad():
            if change == "labels":
                own.labels.copy_(ignored)
            else:
                logp.mul_(1.0)

    with _logprob_forwards() as seen:
        # the changed labels change the valid-row index's shape: no metadata check on the recomputation
        run = _logprob_step(
            own,
            "nonreentrant_no_determinism_check",
            form,
            between=between,
            backend="reference",
        )
    assert len(seen) == 2, seen
    changed = own if change == "logp" else dataclasses.replace(inp, labels=ignored)
    plain = _logprob_step(changed, "plain", form, backend="reference")
    assert _same_bits(run["dX"], plain["dX"])
    assert _same_bits(run["dW"], plain["dW"])


def test_checkpointed_logprob_retained_graph_takes_the_row_results_per_backward():
    """Two backward passes through one retained checkpointed log-probability graph: each recomputation takes the kept
    row results (one forward in all), each pair of gradients is the uncheckpointed step's bits."""
    inp = _host_inputs(200, "ce")
    plain = _logprob_step(inp, "plain", None, backend="reference")
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def head(a, b):
        logp = cake_backend.cake_lm_head_nondet_train_logprob(
            a, b, inp.labels, chunk_size=64, backend="reference"
        )
        return (logp * inp.dlogp).sum()

    with _logprob_forwards() as seen:
        loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
        first = torch.autograd.grad(loss, (X, W), retain_graph=True)
        second = torch.autograd.grad(loss, (X, W))
    assert len(seen) == 1, seen
    for grads in (first, second):
        assert _same_bits(grads[0], plain["dX"])
        assert _same_bits(grads[1], plain["dW"])


def _logits_gemm_keys(launched) -> int:
    return sum(1 for key in launched if str(key[0]).startswith("gemm_logits"))


@pytest.mark.parametrize("form", [None, "scalar", "tensor"])
def test_device_checkpointed_logprob_step_runs_the_logits_gemms_once(glm_weight, form):
    """A non-reentrant (plain, selective, under a dispatch mode) checkpointed log-probability step runs the logits
    forward once, as the uncheckpointed step (the same logits and gradient GEMM launches, no more; without a
    temperature both counted), and gives its bits (one dX slice pinned)."""
    if form is None:
        _require_program(entry="logprob")
    else:
        _require_tau_program()
    inp = _device_inputs(12289, "ce", W=glm_weight)
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})
    _logprob_step(inp, "plain", form, C=4096, **one)
    with _launch_record(temperature=True) as launched, _logprob_forwards() as seen:
        plain = _logprob_step(inp, "plain", form, C=4096, **one)
    logits, grads = _logits_gemm_keys(launched), _gradient_gemm_keys(launched)
    assert len(seen) == 1, seen
    if form is None:
        assert logits > 0 and grads > 0, sorted({k[0] for k in launched})
    for kind in (
        "nonreentrant",
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
    ):
        with _launch_record(temperature=True) as launched, _logprob_forwards() as seen:
            run = _logprob_step(inp, kind, form, C=4096, **one)
        assert len(seen) == 1, (kind, seen)
        assert _logits_gemm_keys(launched) == logits, kind
        assert _gradient_gemm_keys(launched) == grads, kind
        _assert_step_bits(run, plain, keys=("logp", "dX", "dW"))


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
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.cake.test_helpers.cake_package import cake_package
backend = cake_package("cake.cake_lm_head_nondet_train", {module!r})
W = make_inputs(1, seed={seed_w}, device="cuda", ignore_frac=0.0).W
inp = make_inputs({T}, seed={seed}, device="cuda", W=W, ignore_frac=0.05)
kw = dict(chunk_size={C}, backend="cake", gemm_tuning={{"dx": {{"k_slices": 1}}}}, **{switches!r})


def head(a, b):
    if {entry!r} == "loss":
        return backend.cake_lm_head_nondet_train_loss(a, b, inp.labels, loss_div=inp.loss_div, **kw)
    return (backend.cake_lm_head_nondet_train_logprob(a, b, inp.labels, **kw) * inp.dlogp).sum()


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
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})
    if entry in ("loss", "loss_budget"):
        mod, kw = (
            (cake_backend, {})
            if entry == "loss"
            else (opt_in, dict(memory_mode="budget"))
        )
        return mod.cake_lm_head_nondet_train_loss(
            X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096, **kw, **one
        )
    if entry == "support":
        # eight distinct candidates per row, the label first (an ignored row: candidates of token 0)
        K, T, V = 8, X.shape[0], W.shape[0]
        base = inp.labels.clamp_min(0)[:, None]
        ids = (base + 997 * torch.arange(K, device=X.device)[None, :]) % V
        offsets = torch.arange(0, T * K + 1, K, device=X.device)
        support = cake_package("cake.cake_lm_head_nondet_train", "cake_support")
        logp = support.cake_lm_head_nondet_train_support_logprob(
            X, W, ids.reshape(-1), offsets, inp.labels, k_cap=K
        )
        return (logp * inp.dlogp).sum()
    if entry == "keep_backend":
        logp = cake_backend.cake_lm_head_nondet_train_logprob_keep(
            X, W, inp.labels, chunk_size=4096, **one
        )
    else:
        memory_mode = {
            "logprob": None,
            "logprob_budget": "budget",
            "keep": "keep_logits",
        }[entry]
        logp = opt_in.cake_lm_head_nondet_train_logprob(
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
def test_device_checkpointed_logprob_keep_step_is_bitwise(
    glm_weight, entry, chunked_keep
):
    """The kept logits (``cake_lm_head_nondet_train_logprob_keep``; the opt-in ``memory_mode="keep_logits"``; the
    chunked kept form, :data:`CHUNKED_KEEP`) under a non-reentrant checkpoint, selective activation checkpointing and a
    dispatch mode (around the checkpointed forward; around the whole step) give the uncheckpointed default entry's
    ``dX`` / ``dW`` bits (one dX slice pinned)."""
    _require_program(entry="logprob")
    inp = _device_inputs(12289, "ce", W=glm_weight)
    one = dict(gemm_tuning={"dx": {"k_slices": 1}})

    def head_default(a, b):
        return (
            cake_backend.cake_lm_head_nondet_train_logprob(
                a, b, inp.labels, chunk_size=4096, **one
            )
            * inp.dlogp
        ).sum()

    def head_keep(a, b):
        if entry == "backend":
            logp = cake_backend.cake_lm_head_nondet_train_logprob_keep(
                a, b, inp.labels, chunk_size=4096, **one
            )
        else:
            logp = opt_in.cake_lm_head_nondet_train_logprob(
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
        loss, logp = opt_in.cake_lm_head_nondet_train_loss(
            X,
            W,
            inp.labels,
            objective="policy",
            infer_logp=inp.infer_logp,
            loss_weights=inp.loss_weights,
            chunk_size=4096,
            return_logp=True,
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
def _cache(enabled: bool, switches=()):
    """Switch the binding cache of the module whose entry points ``_run`` calls with ``switches`` (the opt-in
    module's own cache for an opt-in keyword) and yield it."""
    be = opt_in if _OPT_IN_KEYWORDS & set(switches) else cake_backend
    cache = be.BINDING_CACHE
    was = cache.enabled
    cache.enabled = enabled
    try:
        yield cache
    finally:
        cache.enabled = was


@contextlib.contextmanager
def _kept_form(**form):
    """The kept-logits form switches (``cake_backend.set_kept_logits_form``) set to ``form`` for the block, restored
    after it.  ``_kept_form(one_pass=False)`` (``tail_fold`` on: the default plan's fold, ``cake_backend.loop_chunks``):
    the chunked kept backward on the default path's chunks (the default path's ``dW`` bits)."""
    previous = cake_backend.set_kept_logits_form(**form)
    try:
        yield
    finally:
        cake_backend.set_kept_logits_form(**previous)


# the kept-logits form whose dW is the default path's bits (the chunked kept backward on the default chunks: the kept
# plan folds by the default plan's rule, ``cake_backend.loop_chunks``; the valid rows counted on the host at every
# size, so the kept views are row slices of one buffer)
CHUNKED_KEEP = dict(tail_fold=True, one_pass=False, hidden_count=False)
# the kept-logits form with the valid rows counted on the host at every size (the default form runs the hidden count
# where it is eligible)
HOST_COUNT = dict(hidden_count=False)


@pytest.fixture
def chunked_keep():
    """The chunked kept backward on the default path's chunks for the test (:data:`CHUNKED_KEEP`): the kept path's
    ``dW`` is then the default path's bits (the default kept form sums ``dW`` in another grouping)."""
    with _kept_form(**CHUNKED_KEEP):
        yield


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
    """The kept logits have their own entry points (``cake_lm_head_nondet_train_logprob_keep``, ``forward_logprob_keep`` /
    ``backward_logprob_keep``); the default entry points take no kept-logits keyword.  The kept autograd entry runs
    the default entry wherever no backward can follow -- grad mode off (``torch.no_grad``, ``torch.inference_mode``),
    no input requiring a gradient -- and on the reference backend; the explicit pair and the prepared runner of the
    kept plan belong to the ``cake`` backend.  The opt-in form ``memory_mode="keep_logits"`` belongs to the
    log-probability entry without ``num_valid_rows`` and runs the kept entry; elsewhere it is a ``ValueError``.  The
    deterministic package's kept entry takes the same keywords."""
    default_params = inspect.signature(
        cake_backend.cake_lm_head_nondet_train_logprob
    ).parameters
    assert (
        inspect.signature(
            cake_backend.cake_lm_head_nondet_train_logprob_keep
        ).parameters
        == default_params
    )
    # the deterministic entry's parameters without its mode keyword, if it still has one
    keep = inspect.signature(
        cake_backend.cake_lm_head_nondet_train_logprob_keep
    ).parameters
    det = inspect.signature(det_backend.cake_lm_head_det_train_logprob_keep).parameters
    assert list(keep) == [n for n in det if n != "deterministic"]
    for fn in (
        cake_backend.cake_lm_head_nondet_train_logprob,
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
        )
    with pytest.raises(ValueError, match="cake backend"):
        cake_backend.forward_logprob_keep(
            inp.X, inp.W, inp.labels, chunk_size=16, backend="reference"
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

    monkeypatch.setattr(cake_backend, "cake_lm_head_nondet_train_logprob", default_spy)
    monkeypatch.setattr(
        cake_backend.CakeLMHeadNondetLogprobKeepFunction, "apply", keep_spy
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
            cake_backend.cake_lm_head_nondet_train_logprob_keep(
                x, w, inp.labels, chunk_size=16, backend=backend
            )
        assert [c[0] for c in calls] == [expect], tag
        if expect == "default":
            assert calls[0][1]["backend"] == backend, tag
            assert calls[0][1]["chunk_size"] == 16, tag
        else:
            assert calls[0][1][3] == 16, tag  # the call's arguments reach the Function
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

    monkeypatch.setattr(opt_in._main, "cake_lm_head_nondet_train_logprob_keep", spy)
    opt_in.cake_lm_head_nondet_train_logprob(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=16,
        backend="cake",
        memory_mode="keep_logits",
    )
    assert seen["backend"] == "cake" and seen["chunk_size"] == 16
    assert "deterministic" not in seen
    assert set(seen) <= set(default_params)


def test_device_logprob_keep_is_bitwise(logprob_chunk_loop, glm_weight, chunked_keep):
    """The kept logits (the chunked kept form, :data:`CHUNKED_KEEP`) against the chunk loop of the default
    log-probability path (pinned: the default backward of a call of more than one chunk is vocabulary-major and rounds
    ``dX`` / ``dW`` differently from the kept chunk loop, see :func:`test_device_vocab_major_logprob_backward`) on the
    generated program: the kept autograd entry
    (``cake_lm_head_nondet_train_logprob_keep``, first call and remembered binding; also through the opt-in
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
                inp.X, inp.W, inp.labels, chunk_size=C, **fwd_kw
            )
            rows = int((inp.labels >= 0).sum()) if fr.row_index is not None else T
            assert len(fr.logits) == len(cake_backend.loop_chunks(T, rows, C)), case
            assert [int(v.shape[0]) for v in fr.logits] == [
                r for _, r in cake_backend.loop_chunks(T, rows, C)
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


def test_device_logprob_keep_lifetimes(glm_weight, monkeypatch, chunked_keep):
    """Where the kept logits live and what they cost (the chunked kept form, :data:`CHUNKED_KEEP`; one dX slice pinned:
    every gradient is compared by its bits): a
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
            cake_backend.cake_lm_head_nondet_train_logprob_keep
            if keep
            else cake_backend.cake_lm_head_nondet_train_logprob
        )
        return fn(X, W, inp.labels, chunk_size=C, gemm_tuning=ONE_SLICE)

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
        # the kept views are row slices of one contiguous buffer, saved alone
        assert len(node.saved_tensors) == 5 + 1
        assert node.kept_chunks == tuple(r for _, r in cake_backend.plan_chunks(T_v, C))
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
    kw = dict(chunk_size=C, gemm_tuning=ONE_SLICE)
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
    )
    assert runner.plan.logprob_keep
    with pytest.raises(NotImplementedError, match="eager entry points"):
        runner.forward()


def test_device_logprob_keep_hidden_count_memory_report(glm_weight):
    """The default kept form (one pass; the hidden valid-row count where eligible, here 4 chunks of ``T``) keeps every
    chunk's logits in the hidden count's ``[T, V]`` buffer, allocated before the count: ``forward_logprob_keep``'s
    report counts all of it (``saved_logits_bytes = 2 * T * V``, the ignored rows included, the storage of the kept
    views; the later rows in ``accumulators["saved_logits"]``), and the forward leaves at least that many bytes
    allocated.  With the valid rows counted on the host the report and the storage are ``2 * T_v * V``."""
    _require_program(entry="logprob")
    T, C = 16384, 4096
    inp = make_inputs(T, seed=SEED + 6995, device=CUDA, W=glm_weight, ignore_frac=0.5)
    T_v, V = int(inp.valid.sum()), int(glm_weight.shape[0])
    slack = 8 << 20  # allocator rounding and the [T] / [T_v] row vectors
    seen = []
    for hidden_count, rows in ((True, T), (False, T_v)):

        def forward():
            return cake_backend.forward_logprob_keep(
                inp.X, inp.W, inp.labels, chunk_size=C, hidden_count=hidden_count
            )

        forward()  # binds (the binding's own tensors stay cached; this result is dropped)
        gc.collect()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        res = forward()
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        mem = res.memory
        seen.append((hidden_count, held, mem["saved_logits_bytes"]))
        assert bool(mem.get("hidden_count")) == hidden_count, mem.get("hidden_count")
        assert mem["saved_logits_bytes"] == 2 * rows * V, (hidden_count, mem)
        assert mem["accumulators"]["saved_logits"] == 2 * rows * V - int(
            mem["temporary"]["logits"]
        ), (hidden_count, mem)
        assert mem["accumulator_bytes"] == sum(mem["accumulators"].values())
        storage = {v.untyped_storage().data_ptr() for v in res.logits}
        assert len(storage) == 1, storage  # one contiguous kept buffer
        kept_bytes = res.logits[0].untyped_storage().nbytes()
        assert kept_bytes == mem["saved_logits_bytes"], seen
        assert held >= mem["saved_logits_bytes"] - slack, seen
        del res, mem
    print(
        f"\nkept report T={T} T_v={T_v}: (hidden_count, held, saved_logits_bytes) {seen}"
    )


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


@contextlib.contextmanager
def _tau_dz_counted(counted):
    """Run with the zero-gradient row skip's ``dz`` temperature form counted or not
    (:func:`cake_backend.set_zero_dlogp_temperature_dz_counted`), then restore the previous."""
    previous = cake_backend.set_zero_dlogp_temperature_dz_counted(counted)
    try:
        yield
    finally:
        cake_backend.set_zero_dlogp_temperature_dz_counted(previous)


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


def test_kept_logits_form_plan_and_switches():
    """The kept-logits form on the host: the kept plan folds by the default plan's one rule (:func:`loop_chunks`: a
    last chunk of fewer than ``chunk / 2`` rows of a call of fewer than three chunks of ``T``, never above
    :data:`KEPT_CHUNK_MAX_ROWS` = :data:`CHUNK_MAX_ROWS` rows) and is :func:`plan_chunks` otherwise; on a call of
    three chunks or more the default plan, the kept plan and a hidden-count kept plan keep the shipped chunks, and on
    a two-chunk call the kept plan folds unless ``tail_fold=False``;
    :func:`kept_rows_of` finds the contiguous buffer behind consecutive row views from its first row only (the buffer
    itself when they cover it, else its first rows); the switches default to ``tail_fold`` / ``one_pass`` /
    ``hidden_count`` on and ``dx_one_pass`` / ``dx_side`` off, return the previous settings and take bools only; the one-pass row
    limit defaults to :data:`KEPT_CHUNK_MAX_ROWS`."""
    pc = cake_backend.plan_chunks

    # the kept plan's chunks of a call of ``rows`` rows, every row valid
    def pk(rows, chunk):
        return cake_backend.loop_chunks(rows, rows, chunk)

    assert cake_backend.KEPT_CHUNK_MAX_ROWS == cake_backend.CHUNK_MAX_ROWS == 65535
    for rows, chunk in (
        (1, 4096),
        (4096, 4096),
        (4370, 4096),
        (6143, 4096),
        (6144, 4096),
        (8265, 4096),
        (7711, 4096),
        (61780, 4096),
        (3000, 1024),
        (65535, 65535),
        (100000, 40000),
    ):
        got, base = pk(rows, chunk), pc(rows, chunk)
        assert sum(r for _, r in got) == rows and got[0][0] == 0
        assert all(a[0] + a[1] == b[0] for a, b in zip(got, got[1:], strict=False))
        assert all(r <= cake_backend.KEPT_CHUNK_MAX_ROWS for _, r in got)
        fold = len(base) == 2 and 2 * base[-1][1] < chunk and rows <= 65535
        assert got == (((0, rows),) if fold else base), (rows, chunk)
    assert pk(4370, 4096) == ((0, 4370),)
    assert pk(8265, 4096) == pc(8265, 4096)  # three chunks of T: the shipped chunks
    assert pk(7711, 4096) == pc(7711, 4096)  # a 3615-row tail is not short
    assert pk(100000, 40000) == pc(100000, 40000)  # 40000 + 20000: not below half
    assert pk(65000 + 20000, 65000) == pc(
        85000, 65000
    )  # the fold would pass 65535 rows
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
    # three chunks of T: no plan folds (the hidden count serves these calls)
    problem = _keep_problem(8700, 6144, 154880, 4096, entry="logprob")
    plain = cake_backend.make_plan(problem, valid_rows=8265, **common)
    kept = cake_backend.make_plan(problem, valid_rows=8265, logprob_keep=True, **common)
    hidden = cake_backend.make_plan(
        problem,
        valid_rows=8265,
        logprob_keep=True,
        hidden_count=True,
        **common,
    )
    assert plain.chunks == kept.chunks == hidden.chunks == pc(8265, 4096)
    # two chunks of T: the kept plan folds the 274-row tail like the default plan, unless tail_fold=False
    problem2 = _keep_problem(4600, 6144, 154880, 4096, entry="logprob")
    folded = cake_backend.make_plan(
        problem2, valid_rows=4370, logprob_keep=True, **common
    )
    unfolded = cake_backend.make_plan(
        problem2, valid_rows=4370, logprob_keep=True, tail_fold=False, **common
    )
    assert (
        folded.chunks
        == cake_backend.make_plan(problem2, valid_rows=4370, **common).chunks
        == ((0, 4370),)
        and type(folded) is cake_backend._KeptPlan
    )
    assert unfolded.chunks == pc(4370, 4096)
    assert len(folded.dx_slices or (1,)) == 1 and folded.rows == 4370
    one = cake_backend.make_plan(
        _keep_problem(8700, 6144, 154880, 8265, entry="logprob"),
        valid_rows=8265,
        logprob_keep=True,
        **common,
    )
    assert one.chunks == ((0, 8265),) and not one.dw_acc_needed
    # kept_rows_of: consecutive row views of one contiguous buffer from its first row
    buf = torch.zeros((10, 4), dtype=torch.bfloat16)
    assert cake_backend.kept_rows_of((buf[0:3], buf[3:10])) is buf
    assert cake_backend.kept_rows_of((buf[0:10],)) is buf
    assert cake_backend.kept_rows_of((buf[0:3], buf[4:10])) is None  # a gap
    assert cake_backend.kept_rows_of((buf[3:10], buf[0:3])) is None  # out of order
    assert cake_backend.kept_rows_of((buf[3:6], buf[6:10])) is None  # not from row 0
    # the hidden count's [T, V]
    prefix = cake_backend.kept_rows_of((buf[0:3], buf[3:9]))
    assert prefix is not None and tuple(prefix.shape) == (9, 4)
    assert prefix.data_ptr() == buf.data_ptr() and prefix.is_contiguous()
    assert (
        cake_backend.kept_rows_of((buf[0:3], torch.zeros((7, 4), dtype=torch.bfloat16)))
        is None
    )
    assert cake_backend.kept_rows_of((buf,)) is None  # not a view
    assert (
        cake_backend.kept_rows_of(()) is None
        and cake_backend.kept_rows_of(None) is None
    )
    # the switches
    default = dict(
        tail_fold=True,
        one_pass=True,
        one_pass_rows=0,
        dx_one_pass=False,
        dx_side=False,
        hidden_count=True,
    )
    assert cake_backend.kept_logits_form() == default
    with _kept_form(one_pass=False):
        assert cake_backend.kept_logits_form() == dict(default, one_pass=False)
        assert cake_backend.set_kept_logits_form(tail_fold=False)["tail_fold"] is True
        assert cake_backend.kept_logits_form()["tail_fold"] is False
        for name in ("one_pass", "dx_one_pass", "dx_side", "hidden_count"):
            with pytest.raises(ValueError, match="must be a bool"):
                cake_backend.set_kept_logits_form(**{name: 1})
    assert cake_backend.kept_logits_form() == default
    with _kept_form(**HOST_COUNT, dx_side=True, dx_one_pass=True):
        assert cake_backend.kept_logits_form() == dict(
            default, hidden_count=False, dx_side=True, dx_one_pass=True
        )
    assert cake_backend.kept_logits_form() == default
    # the one-pass row limit: KEPT_ONE_PASS_ROWS (= KEPT_CHUNK_MAX_ROWS) unless set, never above KEPT_CHUNK_MAX_ROWS
    assert (
        cake_backend.kept_one_pass_rows()
        == cake_backend.KEPT_ONE_PASS_ROWS
        == cake_backend.KEPT_CHUNK_MAX_ROWS
    )
    with _kept_form(one_pass_rows=100000):
        assert cake_backend.kept_one_pass_rows() == cake_backend.KEPT_CHUNK_MAX_ROWS
    with _kept_form(one_pass_rows=8192):
        assert cake_backend.kept_one_pass_rows() == 8192
        assert cake_backend.kept_logits_form()["one_pass_rows"] == 8192
    assert cake_backend.kept_logits_form()["one_pass_rows"] == 0
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="int >= 0"):
            cake_backend.set_kept_logits_form(one_pass_rows=bad)


def _kept_bwd_bindings(cache):
    """The kept backward bindings of the binding cache (:class:`cake_backend._KeptBinding` under a ``bwd`` key)."""
    return [
        b
        for key, b in cache._bindings.items()
        if key[0] == "bwd" and isinstance(b, cake_backend._KeptBinding)
    ]


def _kept_forward_chunks(inp, T_v, C, form, switches):
    """The chunks a kept forward of ``inp`` (``T_v`` rows planned) plans under the kept form ``form`` (over the default
    switches) and the call's ``switches``, and whether it runs the hidden valid-row count: the shipped chunks under
    the hidden count (a compacted call the program finds eligible, :func:`hidden_count_eligible`), else the folded
    chunks under ``tail_fold`` (the default plan's rule, :func:`loop_chunks`)."""
    form = dict(cake_backend.kept_logits_form(), **form)
    hidden = False
    if form["hidden_count"] and switches.get("compact_rows", True):
        module_name, record = cake_backend.record_for(inp.X.device)
        problem = cake_backend.validate_lm_head_inputs(
            inp.X, inp.W, inp.labels, chunk_size=C, entry="logprob"
        )
        hidden = cake_backend.hidden_count_eligible(
            problem, inp.X, record, module_name, stats=True
        )
    if hidden or not form["tail_fold"]:
        return cake_backend.plan_chunks(T_v, C), hidden
    return cake_backend.loop_chunks(int(inp.labels.shape[0]), T_v, C), hidden


@pytest.mark.parametrize(
    "T, C, ignore, form, switches",
    [
        # a 274-row tail folded: one chunk forward and backward
        (4600, 4096, 0.05, {}, {}),
        (3000, 4096, 0.05, {}, {}),  # one chunk
        (1, 4096, 0.0, {}, {}),  # one row
        (8117, 4096, 0.05, {}, {}),  # two chunks, no fold (LK08's shape class)
        (8117, 4096, 0.05, {}, dict(gemm_tuning=ONE_SLICE)),
        # three chunks of T: the hidden count, a 73-row last chunk
        (8700, 4096, 0.05, {}, {}),
        # the host count: three chunks of T keep the shipped chunks (loop_chunks)
        (8700, 4096, 0.05, dict(hidden_count=False), {}),
        (12289, 4096, 0.05, {}, {}),
        (12289, 4096, 0.05, {}, dict(gemm_tuning=ONE_SLICE)),
        (12289, 4096, 0.05, dict(dx_side=True), {}),
        (12289, 4096, 0.05, dict(hidden_count=False), {}),
        (12289, 4096, 0.05, dict(dx_one_pass=True), {}),  # one dX GEMM over M = T_v
        (6000, 2048, 0.0, {}, dict(compact_rows=False)),
        (8700, 4096, 0.05, {}, dict(fuse_dw_cast=False, dx_finalize=False)),
    ],
    ids=str,
)
def test_device_logprob_keep_one_pass(glm_weight, T, C, ignore, form, switches):
    """The kept-logits form (:func:`set_kept_logits_form`) on the generated program: the forward keeps one contiguous
    buffer (``[T_v, V]``, or the hidden count's ``[T, V]`` whose first ``T_v`` rows the node saves) on its chunks (the
    shipped chunks under the hidden count, else the folded ones); the first backward runs ``row_grad`` and dX in those
    chunks -- a kept binding without the weight gradient -- and the weight gradient as ONE fused-cast GEMM over
    ``K = T_v`` -- a one-chunk kept binding without dX or an FP32 accumulator (``dx_one_pass``: one binding of one
    chunk), a one-chunk call too (there its ``dW`` is the ``one_pass=False`` bits).  ``logp`` / ``lse`` are the
    default path's bits; ``dX`` is the chunked kept backward's bits on the same
    chunks (``one_pass=False``) wherever those repeat (at most two dX slices per chunk; with one slice pinned also
    the default path's bits); ``dX`` and ``dW`` pass the suite's FP64 gates (budgets unchanged) and ``dW`` repeats bit
    for bit; the side-stream dX (``dx_side``) gives the in-order bits.  A non-reentrant and a reentrant
    ``torch.utils.checkpoint`` give the uncheckpointed kept step's bits (one dX slice pinned)."""
    _require_program(entry="logprob")
    inp = make_inputs(
        T, seed=SEED + 290 + T, device=CUDA, W=glm_weight, ignore_frac=ignore
    )
    compact = switches.get("compact_rows", True)
    T_v = int(inp.valid.sum()) if compact else T
    V = int(glm_weight.shape[0])
    case = (T, C, ignore, form, switches)
    chunks, hidden = _kept_forward_chunks(inp, T_v, C, form, switches)
    full = dict(cake_backend.kept_logits_form(), **form)
    with _cache(True) as cache:
        cache.clear()
        # the default Function's path: a small call (T <= SMALL_T_KEEP_MAX_ROWS) with a backward would run the kept
        # logits (set_small_t_keep) on its padded rows and add their kept bindings; padding and that routing off
        with _small_pad_rows(0), _small_t_keep(False):
            base = _run(inp, C, entry="logprob", backend="cake", **switches)
        with _kept_form(**form):
            keep = [
                _run(inp, C, entry="logprob", backend="cake", keep=True, **switches)
                for _ in range(2)
            ]
            bwd = _kept_bwd_bindings(cache)
        with _kept_form(**dict(form, one_pass=False)):
            chunked = _run(
                inp, C, entry="logprob", backend="cake", keep=True, **switches
            )
        with _kept_form(**dict(form, dx_side=not full["dx_side"])):
            other_side = _run(
                inp, C, entry="logprob", backend="cake", keep=True, **switches
            )
    one_chunk = len(chunks) == 1
    if full["dx_one_pass"]:
        assert len(bwd) == 1 and bwd[0].plan.chunks == ((0, T_v),), case
        dx_plan = bwd[0].plan
    else:
        assert len(bwd) == 2, case
        (dx_binding,) = [b for b in bwd if b.plan.need_dx]
        (dw_binding,) = [b for b in bwd if b.plan.need_dw]
        assert dx_binding.plan.chunks == chunks and not dx_binding.plan.need_dw, case
        assert dw_binding.plan.chunks == ((0, T_v),), case
        assert not dw_binding.plan.need_dx, case
        if switches.get("fuse_dw_cast", True):
            assert not dw_binding.plan.dw_acc_needed, case
        dx_plan = dx_binding.plan
    deterministic_dx = max(dx_plan.dx_slices or (1,)) <= 2
    for out in keep + [chunked, other_side]:
        assert _same_bits(out["logp"], base["logp"]), case
    _check_against_references(
        dict(base, dX=keep[0]["dX"], dW=keep[0]["dW"]), inp, entry="logprob"
    )
    assert _same_bits(keep[1]["dW"], keep[0]["dW"]), case
    assert _same_bits(other_side["dW"], keep[0]["dW"]), case
    if one_chunk:  # one GEMM over K = T_v either way: the chunked kept backward's dW
        assert _same_bits(chunked["dW"], keep[0]["dW"]), case
    if not full["dx_one_pass"]:
        if deterministic_dx:
            # dX in the forward's chunks: the chunked kept backward's bits
            for out in (keep[1], chunked, other_side):
                assert _same_bits(out["dX"], keep[0]["dX"]), case
        else:  # the unordered K-slice reduction: every dX is held to the gates
            for out in (keep[1], chunked, other_side):
                _check_against_references(
                    dict(base, dX=out["dX"], dW=keep[0]["dW"]), inp, entry="logprob"
                )
    if switches.get("gemm_tuning") == ONE_SLICE:
        for out in keep:
            assert _same_bits(out["dX"], base["dX"]), case
    # the node saves the contiguous buffer alone (its first T_v rows)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    fw = {
        k: v
        for k, v in switches.items()
        if k in ("compact_rows", "gemm_tuning", "fuse_dw_cast", "dx_finalize")
    }
    with _kept_form(**form):
        lp = cake_backend.cake_lm_head_nondet_train_logprob_keep(
            X, W, inp.labels, chunk_size=C, **fw
        )
    saved = lp.grad_fn.saved_tensors
    assert len(saved) == 6, case
    assert tuple(saved[5].shape) == (T_v, V), case
    assert lp.grad_fn.kept_chunks == tuple(r for _, r in chunks), case
    if hidden:  # the hidden count's buffer: sized by T
        assert saved[5].untyped_storage().nbytes() == T * V * 2, case
    del lp, X, W, saved
    # checkpointed steps: the uncheckpointed kept step's bits (one dX slice pinned)
    one = dict(fw, gemm_tuning=ONE_SLICE)

    def step(reentrant=None):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)

        def head(a, b):
            return (
                cake_backend.cake_lm_head_nondet_train_logprob_keep(
                    a, b, inp.labels, chunk_size=C, **one
                )
                * inp.dlogp
            ).sum()

        if reentrant is None:
            head(X, W).backward()
        else:
            torch.utils.checkpoint.checkpoint(
                head, X, W, use_reentrant=reentrant
            ).backward()
        torch.cuda.synchronize()
        return X.grad, W.grad

    with _kept_form(**form):
        plain = step()
        for reentrant in (False, True):
            got = step(reentrant)
            assert _same_bits(got[0], plain[0]) and _same_bits(got[1], plain[1]), (
                case,
                reentrant,
            )


@pytest.mark.parametrize("form", [{}, dict(hidden_count=False)], ids=str)
def test_device_logprob_keep_one_pass_row_limit(glm_weight, form):
    """Above the one-pass row limit (:func:`kept_one_pass_rows`; here ``one_pass_rows=8192``) the first kept backward
    runs the forward's chunks with its FP32 ``dW`` accumulator: the bits of the chunked kept backward
    (``one_pass=False``), held to the suite's FP64 gates."""
    _require_program(entry="logprob")
    C, T = 4096, 12289
    inp = make_inputs(T, seed=SEED + 411, device=CUDA, W=glm_weight, ignore_frac=0.05)
    T_v = int(inp.valid.sum())
    assert T_v > 8192
    chunks, _ = _kept_forward_chunks(inp, T_v, C, form, {})
    with _cache(True) as cache:
        cache.clear()
        with _kept_form(one_pass_rows=8192, **form):
            capped = _run(inp, C, entry="logprob", backend="cake", keep=True)
        bwd = _kept_bwd_bindings(cache)
        with _kept_form(one_pass=False, **form):
            chunked = _run(inp, C, entry="logprob", backend="cake", keep=True)
    assert len(bwd) == 1
    assert bwd[0].plan.chunks == chunks
    assert bwd[0].plan.dw_acc_needed and bwd[0].plan.need_dx
    for key in ("logp", "dW"):
        assert _same_bits(capped[key], chunked[key]), key
    base = _run(inp, C, entry="logprob", backend="cake")
    _check_against_references(
        dict(base, dX=capped["dX"], dW=capped["dW"]), inp, entry="logprob"
    )


@pytest.mark.parametrize("form", [{}, dict(hidden_count=False)], ids=str)
def test_device_logprob_keep_one_pass_frees_the_dw_accumulator(glm_weight, form):
    """The one-pass kept backward holds no FP32 ``dW`` accumulator: its step peak (forward + backward, the kept entry,
    allocator requested bytes) is at least ``4 * V * H`` bytes below the chunked kept form's (:data:`CHUNKED_KEEP`)
    minus the one-chunk weight-gradient binding's gathered rows (``T_v`` rows; the kept backward allocates no row
    statistics) and, under the hidden count, the ``T - T_v`` unused rows of the ``[T, V]`` kept buffer; nothing stays
    allocated after the backward."""
    _require_program(entry="logprob")
    C, T = 4096, 12289
    inp = make_inputs(T, seed=SEED + 397, device=CUDA, W=glm_weight, ignore_frac=0.05)
    V, H = int(glm_weight.shape[0]), int(glm_weight.shape[1])
    T_v = int(inp.valid.sum())
    _, hidden = _kept_forward_chunks(inp, T_v, C, form, {})

    def step():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        gc.collect()
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        start = int(stats["requested_bytes.all.current"])
        torch.cuda.reset_peak_memory_stats()
        lp = cake_backend.cake_lm_head_nondet_train_logprob_keep(
            X, W, inp.labels, chunk_size=C
        )
        lp.backward(inp.dlogp)
        del lp
        torch.cuda.synchronize()
        stats = torch.cuda.memory_stats()
        peak = int(stats["requested_bytes.all.peak"]) - start
        X.grad = W.grad = None
        gc.collect()
        left = int(torch.cuda.memory_stats()["requested_bytes.all.current"]) - start
        return peak, left

    with _cache(True):
        for each in (form, CHUNKED_KEEP):  # bind both forms first
            with _kept_form(**each):
                step()
        with _kept_form(**form):
            peak1, left1 = step()
        with _kept_form(**CHUNKED_KEEP):
            peak0, left0 = step()
    # the weight-gradient binding's gathered rows
    growth = T_v * 2 * H
    if hidden:
        growth += (T - T_v) * V * 2
    print(
        f"\none-pass kept step T={T} {form}: peak {peak1} vs chunked {peak0} B ({peak1 - peak0:+d}), "
        f"dW_acc {4 * V * H} B"
    )
    assert peak1 <= peak0 - 4 * V * H + growth + (8 << 20), (peak0, peak1)
    assert left1 <= (8 << 20) and left0 <= (8 << 20), (left0, left1)


def _logprob_call(inp, C=4096, *, package="nd"):
    """One autograd log-probability call (forward + ``backward(dlogp)``) of ``package`` on the generated programs."""
    return _run(inp, C, package=package, entry="logprob", backend="cake")


def _has_zero_dlogp_skip():
    """Whether this device's record registers the zero-gradient row skip: the counted dz arm
    (``LOGPROB_DZ_DCNT_STAGE``) and the partition kernel (``ZERO_DLOGP_PARTITION_STAGE``); every shipped record
    (``sm_100a``, ``sm_103a``, ``sm_107a``) has both."""
    module_name, _record = cake_backend.record_for(CUDA)
    stages = cake_jit.registered_stages(module_name)
    return (
        cake_backend.LOGPROB_DZ_DCNT_STAGE in stages
        and cake_backend.ZERO_DLOGP_PARTITION_STAGE in stages
    )


def _require_zero_dlogp_skip():
    """Skip a test of the skip itself on a device whose record does not register it
    (:func:`test_device_zero_dlogp_skip_is_off_without_the_counted_forms` covers such a record)."""
    _require_program(entry="logprob")
    if not _has_zero_dlogp_skip():
        pytest.skip(
            "this device's record registers no zero-gradient row skip (no counted forms, no partition kernel)"
        )


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


# The partition kernel against its PyTorch reference: (T, chunk, the zero-gradient rows).  700 rows: a call below the
# host-count row bound (``HOST_COUNT_MAX_ROWS``) in one partial tile; 4096 / 4097: one whole tile and one row past it;
# 262147 rows: 65 tiles, also with 128-row chunks (2049 counts).
ZERO_DLOGP_PARTITION_CASES = [
    (1, 4096, "none"),
    (700, 4096, ("random", 0.3)),
    (4096, 4096, "all"),
    (4097, 1024, "last"),
    (8117, 4096, ("random", 0.166)),
    (8117, 4096, "single"),
    (12289, 4096, "last"),
    (16231, 4096, ("group", 0.5)),
    (65031, 4096, ("random", 0.166)),
    (4600, 4600, ("random", 0.166)),
    (65031, 4096, "none"),
    (262147, 4096, ("random", 0.5)),
    (262147, 128, ("group", 0.25)),
]


@pytest.mark.parametrize("T, chunk, kind", ZERO_DLOGP_PARTITION_CASES, ids=str)
def test_device_zero_dlogp_partition_kernel_is_the_reference(T, chunk, kind):
    """``zero_dlogp_partition`` on the device is one launch of the partition kernel and gives
    ``zero_dlogp_partition_reference``'s outputs bit for bit -- the order, the reordered ``lse``, the ``[T]`` scan and the
    per-chunk counts (at least one in the first chunk) -- over random, 512-row group, all-zero, single-live,
    last-chunk-live and no-zero gradients, with signed zeros, NaN gradients (live on a valid row, ignored rows
    included) and NaN statistics among them; the first call binds the launch, the second runs the remembered one."""
    _require_zero_dlogp_skip()
    module_name, _record = cake_backend.record_for(CUDA)
    assert cake_backend.ZERO_DLOGP_PARTITION_STAGE in cake_jit.registered_stages(
        module_name
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + T)
    labels = torch.randint(0, 1000, (T,), generator=gen)
    if T > 1:
        labels[torch.rand(T, generator=gen) < 0.1] = -100
    labels = labels.to(CUDA)
    dlogp = torch.randn(T, generator=gen).to(CUDA)
    dlogp[::97] = float("nan")
    zero = _zero_dlogp_set(SimpleNamespace(labels=labels), kind)
    dlogp[zero] = 0.0
    dlogp[zero.nonzero().squeeze(1)[::2]] = -0.0
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.randn(int(index.numel()), generator=gen).to(CUDA)
    lse[::131] = float("nan")
    launched = []
    original = cake_backend._eager_launch

    def recording(record, module_name, stage, *args, **kwargs):
        launched.append(stage)
        return original(record, module_name, stage, *args, **kwargs)

    cake_backend.BINDING_CACHE.clear()
    cake_backend._eager_launch = recording
    try:
        runs = [
            cake_backend.zero_dlogp_partition(labels, dlogp, index, lse, chunk)
            for _ in range(2)
        ]
    finally:
        cake_backend._eager_launch = original
    assert launched == [cake_backend.ZERO_DLOGP_PARTITION_STAGE]
    want = cake_backend.zero_dlogp_partition_reference(labels, dlogp, index, lse, chunk)
    torch.cuda.synchronize()
    for got in runs:
        _assert_partition_is(got, want)
        assert int(got[3][0]) >= 1


def _assert_partition_is(got, want):
    """``got`` (the partition's four outputs) equals ``want`` (the reference's) bit for bit, dtype, shape and device."""
    for name, g, w in zip(
        ("row_index", "lse", "scan", "counts"), got, want, strict=True
    ):
        assert g.dtype == w.dtype and g.shape == w.shape and g.device == w.device, name
        same = _same_bits(g, w) if g.is_floating_point() else torch.equal(g, w)
        assert same, name


def _partition_inputs(T, seed, zero_frac=0.3):
    """Partition inputs of ``T`` rows on the device: ``(labels, dlogp, row_index, lse)`` with about 10 % ignored rows
    and ``zero_frac`` of the rows of a zero ``dlogp``."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    labels = torch.randint(0, 1000, (T,), generator=gen)
    labels[torch.rand(T, generator=gen) < 0.1] = -100
    dlogp = torch.randn(T, generator=gen)
    dlogp[torch.rand(T, generator=gen) < zero_frac] = 0.0
    labels, dlogp = labels.to(CUDA), dlogp.to(CUDA)
    index = (labels >= 0).nonzero().squeeze(1)
    lse = torch.randn(int(index.numel()), generator=gen).to(CUDA)
    return labels, dlogp, index, lse


def test_device_zero_dlogp_partition_remembers_its_launch(monkeypatch):
    """The partition resolves the device's record once (no ``record_for`` per call), binds its launch once per
    ``(device, T, chunk)`` form while ``BINDING_CACHE`` is enabled (a later call of the form slots its tensors into the
    remembered launch: no binding, the same bits), binds again for another form and on every call with the cache
    disabled, and returns its four outputs as disjoint, aligned views of one allocation."""
    _require_zero_dlogp_skip()
    stage = cake_backend.ZERO_DLOGP_PARTITION_STAGE
    binds, resolved = [], []
    eager_launch, record_for = cake_backend._eager_launch, cake_backend.record_for

    def recording(record, module_name, name, *args, **kwargs):
        binds.append(name)
        return eager_launch(record, module_name, name, *args, **kwargs)

    def counting(*args, **kwargs):
        resolved.append(args)
        return record_for(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_eager_launch", recording)
    monkeypatch.setattr(cake_backend, "record_for", counting)
    monkeypatch.setattr(cake_backend, "_SKIP_RECORDS", {})
    cake_backend.BINDING_CACHE.clear()
    a, b = _partition_inputs(9001, SEED + 1), _partition_inputs(12289, SEED + 2)

    def check(args, chunk):
        got = cake_backend.zero_dlogp_partition(*args, chunk)
        _assert_partition_is(
            got, cake_backend.zero_dlogp_partition_reference(*args, chunk)
        )
        return got

    got = check(a, 4096)
    assert binds == [stage] and len(resolved) == 1
    spans = sorted(
        (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in got
    )
    assert len({t.untyped_storage().data_ptr() for t in got}) == 1
    assert all(t.data_ptr() % cake_backend.WORKSPACE_ALIGN == 0 for t in got)
    assert all(e <= s for (_, e), (s, _) in zip(spans, spans[1:], strict=False))
    for _ in range(3):
        check(a, 4096)
    assert binds == [stage]
    check(b, 4096)
    check(a, 1024)
    check(a, 4096)
    check(b, 4096)
    assert binds == [stage] * 3
    cake_backend.BINDING_CACHE.configure(enabled=False)
    try:
        check(a, 4096)
        check(a, 4096)
    finally:
        cake_backend.BINDING_CACHE.configure(enabled=True)
    assert binds == [stage] * 5 and len(resolved) == 1
    torch.cuda.synchronize()


def test_device_zero_dlogp_partition_concurrent_threads(monkeypatch):
    """Threads that run the partition at once, each on its own stream -- resolving the device's record and binding the
    launch together, on the same and on different forms, while another thread clears the binding cache between their
    calls -- all get the reference's bits."""
    _require_zero_dlogp_skip()
    monkeypatch.setattr(cake_backend, "_SKIP_RECORDS", {})
    cake_backend.BINDING_CACHE.clear()
    index = torch.cuda.current_device()
    cases = [
        _partition_inputs(T, SEED + i) for i, T in enumerate((4097, 9001, 9001, 12289))
    ]
    torch.cuda.synchronize()
    errors: list[BaseException] = []
    done = threading.Event()
    start = threading.Barrier(len(cases) + 1)

    def worker(args):
        try:
            with (
                torch.cuda.device(index),
                torch.cuda.stream(torch.cuda.Stream(index)),
            ):
                start.wait(60)
                runs = [
                    cake_backend.zero_dlogp_partition(*args, 4096) for _ in range(16)
                ]
                want = cake_backend.zero_dlogp_partition_reference(*args, 4096)
                torch.cuda.current_stream(index).synchronize()
                for got in runs:
                    _assert_partition_is(got, want)
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the main thread
            errors.append(exc)

    def clearer():
        start.wait(60)
        while not done.is_set():
            cake_backend.BINDING_CACHE.clear()
            time.sleep(0.0005)

    threads = [threading.Thread(target=worker, args=(c,)) for c in cases]
    other = threading.Thread(target=clearer)
    for t in threads + [other]:
        t.start()
    for t in threads:
        t.join(120)
    done.set()
    other.join(60)
    assert not any(t.is_alive() for t in threads + [other]), "a thread did not finish"
    torch.cuda.synchronize()
    if errors:
        raise errors[0]


@pytest.mark.parametrize("remembered", (False, True), ids=("binds", "remembered"))
def test_device_zero_dlogp_partition_cuda_graph(remembered, monkeypatch):
    """The partition captured in a CUDA graph -- binding its launch and resolving the device's record inside the capture
    (both forgotten after an eager warm-up that loaded the program) or slotting its tensors into the remembered launch
    -- replays with other ``dlogp`` values written into the captured input (random, every and no row zero) and gives the
    reference's bits for them."""
    _require_zero_dlogp_skip()
    labels, dlogp, index, lse = _partition_inputs(12289, SEED + 3)
    cake_backend.zero_dlogp_partition(labels, dlogp, index, lse, 4096)
    torch.cuda.synchronize()
    if not remembered:
        cake_backend.BINDING_CACHE.clear()
        monkeypatch.setattr(cake_backend, "_SKIP_RECORDS", {})
    static = dlogp.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = cake_backend.zero_dlogp_partition(labels, static, index, lse, 4096)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SEED + 4)
    for frac in (0.166, 1.0, 0.0):
        new = torch.randn(static.numel(), generator=gen)
        new[torch.rand(static.numel(), generator=gen) < frac] = 0.0
        static.copy_(new.to(CUDA))
        graph.replay()
        want = cake_backend.zero_dlogp_partition_reference(
            labels, static, index, lse, 4096
        )
        torch.cuda.synchronize()
        _assert_partition_is(got, want)


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


def test_zero_dlogp_floor_per_architecture(monkeypatch):
    """The skip's default floor is per architecture: ``zero_dlogp_min_rows(index)`` reads the device's architecture
    once per device index and returns its floor in ``ZERO_DLOGP_MIN_ROWS_BY_ARCH`` (sm_100a: 1946 valid rows,
    sm_103a: 3891, sm_107a: 1946), the generic ``ZERO_DLOGP_MIN_ROWS_DEFAULT`` (4097) for any other architecture and
    without a device.  An override of ``set_zero_dlogp_min_rows`` (an integer or ``None``) applies to every device;
    the setter returns ``ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`` when no override was set and takes it back, so every
    returned value passed back restores the previous state exactly."""
    sentinel = cake_backend.ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT
    assert cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT == 4097
    assert cake_backend.ZERO_DLOGP_MIN_ROWS_BY_ARCH == {
        "sm_100a": 1946,
        "sm_103a": 3891,
        "sm_107a": 1946,
    }
    assert repr(sentinel) == "ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT"
    archs = {0: "sm_100a", 1: "sm_103a", 2: "sm_107a", 3: None}
    resolved = []

    def arch_for(device=None):
        resolved.append(device.index)
        return archs[device.index]

    monkeypatch.setattr(cake_backend, "arch_for", arch_for)
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {})
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_MIN_ROWS", {"rows": sentinel})
    defaults = {0: 1946, 1: 3891, 2: 1946, 3: 4097}

    def floors():
        return {index: cake_backend.zero_dlogp_min_rows(index) for index in archs}

    assert floors() == defaults and floors() == defaults
    assert sorted(resolved) == [0, 1, 2, 3]  # once per device index
    assert cake_backend.zero_dlogp_min_rows() == 4097
    previous = cake_backend.set_zero_dlogp_min_rows(0)
    assert previous is sentinel
    assert floors() == dict.fromkeys(archs, 0)
    assert cake_backend.zero_dlogp_min_rows() == 0
    assert cake_backend.set_zero_dlogp_min_rows(None) == 0
    assert floors() == dict.fromkeys(archs)
    assert cake_backend.zero_dlogp_min_rows() is None
    assert cake_backend.set_zero_dlogp_min_rows(previous) is None
    assert floors() == defaults and cake_backend.zero_dlogp_min_rows() == 4097
    with _zero_dlogp_floor(2049):
        assert floors() == dict.fromkeys(archs, 2049)
        with _zero_dlogp_floor(sentinel):
            assert floors() == defaults
        assert floors() == dict.fromkeys(archs, 2049)
    assert floors() == defaults
    assert cake_backend.set_zero_dlogp_min_rows(sentinel) is sentinel
    assert sorted(resolved) == [0, 1, 2, 3]


def test_zero_dlogp_floor_sentinel_survives_copy_and_pickle(monkeypatch):
    """``ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT`` is one object per process: ``copy.copy``, ``copy.deepcopy`` (also inside a
    container) and a pickle round trip give that object back, and ``set_zero_dlogp_min_rows`` stores it for any value
    of its type, so a previous setting the caller copied or unpickled restores the per-architecture default floor (the
    backward's ``rows < floor`` compare never sees another object of the type)."""
    sentinel = cake_backend.ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT
    assert copy.copy(sentinel) is sentinel
    assert copy.deepcopy(sentinel) is sentinel
    assert copy.deepcopy({"floor": [sentinel]})["floor"][0] is sentinel
    assert pickle.loads(pickle.dumps(sentinel)) is sentinel
    stray = object.__new__(type(sentinel))
    assert stray is not sentinel
    assert copy.copy(stray) is sentinel and copy.deepcopy(stray) is sentinel
    monkeypatch.setattr(cake_backend, "arch_for", lambda device=None: "sm_100a")
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_ARCH", {})
    monkeypatch.setattr(cake_backend, "_ZERO_DLOGP_MIN_ROWS", {"rows": 0})
    copies = (stray, copy.deepcopy(sentinel), pickle.loads(pickle.dumps(sentinel)))
    for value in copies:
        assert cake_backend.set_zero_dlogp_min_rows(value) == 0
        assert cake_backend._ZERO_DLOGP_MIN_ROWS["rows"] is sentinel
        assert cake_backend.zero_dlogp_min_rows(0) == 1946
        assert cake_backend.zero_dlogp_min_rows() == 4097
        assert cake_backend.set_zero_dlogp_min_rows(0) is sentinel


def test_dlogp_rows_takes_a_ready_gradient_as_is():
    """The log-probability backward binds ``dlogp`` as is when it already is a contiguous FP32 ``[T]`` tensor that
    needs no detach (the autograd gradient of ``logp``); any other one (another dtype, shape or stride, or one that
    requires a gradient) is normalized to the same values as before (``detach().reshape(T).to(float32).contiguous()``)."""
    rows = cake_backend._dlogp_rows
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    d = torch.randn(37, generator=gen)
    assert rows(d, 37) is d
    for other in (
        d.double(),
        d.half(),
        d.reshape(1, 37),
        torch.randn(74, generator=gen)[::2],
        d.clone().requires_grad_(),
    ):
        got = rows(other, 37)
        assert got is not other
        assert got.dtype == torch.float32 and tuple(got.shape) == (37,)
        assert got.is_contiguous() and not got.requires_grad
        assert torch.equal(got, other.detach().reshape(37).to(torch.float32))


def test_dcnt_stage_names_round_trip():
    """The ``_dcnt`` forms: the dz arm, the K-sliced dX reduction (not its swapped tile; the one-slice plain store has
    none) and its slab sum, the weight-gradient accumulate (plain and blocked) and the bf16 fused cast (its TMA store
    too); every registered ``_dcnt`` name parses back to its base and knobs."""
    sv = cake_backend.stage_variant
    assert sv("gemm_logits_dz", dcnt=True) == cake_backend.LOGPROB_DZ_DCNT_STAGE
    assert sv("gemm_dx_red", tile_n=256, dcnt=True) == "gemm_dx_red_tn256_dcnt"
    assert (
        sv("gemm_dx_red", stages=3, split=True, dcnt=True)
        == "gemm_dx_red_st3_split_dcnt"
    )
    assert sv("gemm_dw_acc", blocked=True, dcnt=True) == "gemm_dw_acc_gn_dcnt"
    assert sv("gemm_dw_cast_bf16", dcnt=True) == "gemm_dw_cast_bf16_dcnt"
    assert (
        sv("gemm_dw_cast_bf16", epi_store="tma", dcnt=True)
        == cake_backend.DW_CAST_TMA_DCNT_STAGE
        == "gemm_dw_cast_bf16_tma_dcnt"
    )
    for base, kw in (
        ("gemm_dx_red", dict(swab=True)),
        ("gemm_dw_acc", dict(epi_store="tma")),
        ("gemm_dx", {}),
        ("gemm_logits", {}),
    ):
        with pytest.raises(ValueError):
            sv(base, dcnt=True, **kw)
    assert cake_backend.dx_stage(1, dcnt=True) == "gemm_dx"
    assert cake_backend.dx_stage(3, dcnt=True) == "gemm_dx_red_dcnt"
    assert sv(cake_backend.DX_SLAB_SUM_STAGE, dcnt=True) == "dx_slab_sum_dcnt"
    assert (
        cake_backend.dx_slab_sum_stage("gemm_dx_red_tn256_dcnt")
        == cake_backend.DX_SLAB_SUM_DCNT_STAGE
    )
    assert (
        cake_backend.dx_slab_sum_stage("gemm_dx_red_swab")
        == cake_backend.DX_SLAB_SUM_STAGE
    )
    counted = [stage for stage in cake_jit.STAGES if stage.endswith("_dcnt")]
    assert len(counted) == 11 + len(_LC_STAGES)
    for stage in counted:
        base, knobs = cake_backend.parse_stage(stage)
        assert base in cake_backend.DCNT_BASES and knobs["dcnt"]
        assert not knobs["temperature"]
        assert cake_backend.stage_form(base, **knobs) == stage
        assert stage.removesuffix("_dcnt").removesuffix("_lc") in cake_jit.STAGES


def _not_dx(stages):
    """The stages of ``stages`` other than the dX GEMM's forms and their slab sum."""
    return {
        s
        for s in stages
        if not s.startswith("gemm_dx") and s not in cake_backend.DX_SLAB_SUM_STAGES
    }


def _counted_slices(plain, arch):
    """The dX slices of the zero-gradient row skip's plan of ``plain``'s call on ``arch`` (a record serving four
    slices, a vocabulary of many K steps, no dX slice pin): each count below the architecture's
    ``DX_COUNTED_MIN_SLICES_BY_ARCH`` entry raised to it (no entry: the slice rule's counts)."""
    least = cake_backend.DX_COUNTED_MIN_SLICES_BY_ARCH.get(arch, 1)
    return tuple(max(s, least) for s in plain.dx_slices)


def _dx_free(v):
    """``v`` (:class:`ChunkVariants`) without its dX knobs."""
    return dataclasses.replace(
        v, dx_tile_n=None, dx_stages=None, dx_split=None, dx_swab=False
    )


def _assert_counted_twin(plan, plain, arch, case=None):
    """``plan`` (counted) against ``plain`` (the same call's plan without the skip): the same chunks and the dX slices
    of :func:`_counted_slices`; a chunk at the slice rule's count keeps ``plain``'s instance knobs and dX form (its
    counted form when sliced, followed by the counted slab sum); a raised chunk (one slice to two) runs the counted
    K-sliced reduction with ``plain``'s knobs outside the dX GEMM."""
    assert plan.dcnt and not plain.dcnt, case
    assert plan.chunks == plain.chunks, case
    assert plan.dx_slices == _counted_slices(plain, arch), case
    for i in range(plan.num_chunks):
        stage = plan.dx_stage_of(i)
        assert stage.endswith("_dcnt") == (plan.dx_slices_of(i) > 1), (case, i, stage)
        if plan.dx_slices_of(i) > 1:
            # the counted reduction's slab sum is counted too, right behind it
            assert cake_backend._dx_keys(plan, i) == [
                (stage, i),
                (cake_backend.DX_SLAB_SUM_DCNT_STAGE, i),
            ], (case, i)
        if plan.dx_slices_of(i) == plain.dx_slices_of(i):
            assert plan.variants[i] == plain.variants[i], (case, i)
            assert stage.removesuffix("_dcnt") == plain.dx_stage_of(i), (case, i)
        else:
            assert (plain.dx_slices_of(i), plan.dx_slices_of(i)) == (1, 2), (case, i)
            assert stage.startswith(cake_backend.DX_RED_STAGE), (case, i, stage)
            assert _dx_free(plan.variants[i]) == _dx_free(plain.variants[i]), (case, i)


def test_counted_plan_runs_no_one_slice_dx_chunk():
    """Under the zero-gradient row skip (``make_plan(dcnt=True)``) a plan on an architecture of
    ``DX_COUNTED_MIN_SLICES_BY_ARCH`` (sm_100a, sm_103a and sm_107a: two slices) runs no one-slice dX chunk: each chunk
    the slice rule gives one slice (naturally, or below ``dx_red_min_rows``) runs the counted two-slice K-sliced
    reduction, every other chunk, slice count and knob as the slice rule plans it, and the architecture's record
    registers every form of the plan.  The plan without the skip keeps the slice rule's counts
    (:func:`plan_dx_slices`) on every architecture.  The counts stay the slice rule's under an explicit dX ``k_slices``
    (a count, or ``None`` = one slice), on a record that serves one slice and at a vocabulary of one K step; the
    backward binding key tells the skip and a dX slice pin apart."""
    b = cake_backend
    assert b.DX_COUNTED_MIN_SLICES_BY_ARCH == {"sm_100a": 2, "sm_103a": 2, "sm_107a": 2}
    g = b.Geometry.from_record({"geometry": {"hidden": GLM_H, "vocab": GLM_V}})
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=B200_SMS,
        dx_resident=B200_DX_RESIDENT,
        fuse_dw_cast=True,
        dx_finalize=True,
        logprob_dz=True,
    )
    raised = dict.fromkeys(("sm_100a", "sm_103a", "sm_107a"), 0)
    for arch in raised:
        module = next(n for n, r in cake_jit.MODULES.items() if r["arch"] == arch)
        registered = set(cake_jit.registered_stages(module))
        for T, rows, red_min_rows in (
            (2048, 1946, None),
            (2048, 1946, 4097),
            (4097, 4097, None),
            (8117, 7711, None),
            (8117, 8117, None),
            (12289, 12289, 4097),
            (13500, 13500, None),
            (16231, 16231, None),
            (16231, 16231, 4000),
            (65031, 15 * 4096 + 339, None),
        ):
            case = (arch, T, rows, red_min_rows)
            problem = _keep_problem(T, GLM_H, GLM_V, 4096, entry="logprob")
            kw = dict(common, arch=arch, valid_rows=rows, dx_red_min_rows=red_min_rows)
            plain = b.make_plan(problem, **kw)
            plan = b.make_plan(problem, dcnt=True, **kw)
            assert plain.dx_slices == b.plan_dx_slices(
                problem,
                plain.chunks,
                need_dx=True,
                tuning={},
                dx_max_slices=4,
                num_sms=B200_SMS,
                geometry=g,
                dx_resident=B200_DX_RESIDENT,
                arch=arch,
                red_min_rows=red_min_rows,
            ), case
            _assert_counted_twin(plan, plain, arch, case)
            assert set(plan.stages) <= registered, (
                case,
                sorted(set(plan.stages) - registered),
            )
            if arch in b.DX_COUNTED_MIN_SLICES_BY_ARCH:
                assert 1 not in plan.dx_slices, case
            else:
                assert plan.dx_slices == plain.dx_slices, case
            raised[arch] += sum(
                x != y for x, y in zip(plan.dx_slices, plain.dx_slices, strict=True)
            )
    assert raised["sm_100a"] and raised["sm_103a"] and raised["sm_107a"], raised
    problem = _keep_problem(8117, GLM_H, GLM_V, 4096, entry="logprob")
    tiny_g = b.Geometry.from_record({"geometry": {"hidden": GLM_H, "vocab": g.k_block}})
    tiny = _keep_problem(8117, GLM_H, g.k_block, 4096, entry="logprob")
    for arch in b.DX_COUNTED_MIN_SLICES_BY_ARCH:
        kw = dict(common, arch=arch, valid_rows=8117, dx_red_min_rows=4097)
        assert b.make_plan(problem, dcnt=True, **kw).dx_slices == (2, 2), arch
        for k, want in ((None, 1), (1, 1), (3, 3)):
            pin = dict(kw, gemm_tuning={"dx": {"k_slices": k}})
            pinned = b.make_plan(problem, dcnt=True, **pin)
            assert pinned.dcnt and pinned.dx_slices == (want, want), (arch, k)
            assert pinned.dx_slices == b.make_plan(problem, **pin).dx_slices, (arch, k)
            assert all(
                pinned.dx_stage_of(i).endswith("_dcnt") == (want > 1) for i in range(2)
            ), (arch, k)
        one = b.make_plan(problem, dcnt=True, **dict(kw, dx_max_slices=1))
        assert one.dcnt and one.dx_slices == (), arch
        short = b.make_plan(tiny, dcnt=True, **dict(kw, geometry=tiny_g))
        assert short.dcnt and short.dx_slices == (1, 1), arch
    key = functools.partial(
        b.logprob_backward_binding_key,
        *(torch.empty(s) for s in ((8, 64), (128, 64), (8,), (8,), (8,))),
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        zero_dlogp=True,
    )
    assert key() != key(zero_dlogp=False)
    assert key() != key(gemm_tuning={"dx": {"k_slices": 1}})


def test_plan_runs_the_counted_forms():
    """``make_plan(dcnt=True)`` (a log-probability plan with the dz arm on a registered architecture): the recompute,
    the K-sliced dX chunks and the weight-gradient GEMMs take their ``_dcnt`` forms, the dX chunks at least the
    architecture's counted slice count (``DX_COUNTED_MIN_SLICES_BY_ARCH``; a one-slice chunk elsewhere runs its plain
    store); without the dz arm, or with a chunk on a form without a counted one, the plan keeps the shipped forms; the
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
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        kw = dict(common, arch=arch)
        plain = cake_backend.make_plan(problem, valid_rows=12000, **kw)
        plan = cake_backend.make_plan(problem, valid_rows=12000, dcnt=True, **kw)
        if any(
            v.dx_swab or v.dw_epi_store not in (None, "redsm") for v in plan.variants
        ):
            assert not plan.dcnt and plan.stages == plain.stages
            continue
        counted += 1
        _assert_counted_twin(plan, plain, arch)
        for i in range(plan.num_chunks):
            assert plan.recompute_stage(i) == "gemm_logits_dz_dcnt"
        assert all(s.endswith("_dcnt") for s in plan.stages if s.startswith("gemm_dw"))
        assert {s.removesuffix("_dcnt") for s in _not_dx(plan.stages)} == _not_dx(
            plain.stages
        )
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


def test_live_cast_stage_names():
    """The cast at the last live chunk (``_lc`` before ``_dcnt``): the counted weight-gradient accumulate on its 1-D and
    2-D raster and the counted bf16 fused cast in both stores; each name round-trips through ``parse_stage``; the
    accumulate's form also reads the next chunk's count cell (``CNT_NEXT``), the casts' forms the counted casts'
    operands; the sm_100a, sm_103a and sm_107a records register all four (the grids of their counted twins)."""
    sv = cake_backend.stage_variant
    assert sv("gemm_dw_acc", live_cast=True, dcnt=True) == "gemm_dw_acc_lc_dcnt"
    assert (
        sv("gemm_dw_acc", blocked=True, live_cast=True, dcnt=True)
        == "gemm_dw_acc_gn_lc_dcnt"
    )
    assert (
        sv("gemm_dw_cast_bf16", live_cast=True, dcnt=True)
        == "gemm_dw_cast_bf16_lc_dcnt"
    )
    assert (
        sv("gemm_dw_cast_bf16", epi_store="tma", live_cast=True, dcnt=True)
        == "gemm_dw_cast_bf16_tma_lc_dcnt"
    )
    for base, kw in (
        ("gemm_dw_acc", dict(dcnt=False)),
        ("gemm_dw_acc", dict(epi_store="tma", dcnt=True)),
        ("gemm_dw_cast_bf16", dict(dcnt=False)),
        ("gemm_dw_cast_f32", dict(dcnt=True)),
        ("gemm_dx_red", dict(dcnt=True)),
        ("gemm_logits_dz", dict(dcnt=True)),
    ):
        with pytest.raises(ValueError):
            sv(base, live_cast=True, **kw)
    assert sorted(_LC_STAGES) == sorted(
        [
            "gemm_dw_acc_lc_dcnt",
            "gemm_dw_acc_gn_lc_dcnt",
            "gemm_dw_cast_bf16_lc_dcnt",
            "gemm_dw_cast_bf16_tma_lc_dcnt",
        ]
    )
    for stage in _LC_STAGES:
        base, knobs = cake_backend.parse_stage(stage)
        assert knobs["live_cast"] and knobs["dcnt"] and not knobs["temperature"]
        assert cake_backend.stage_form(base, **knobs) == stage
        twin = stage.replace("_lc_dcnt", "_dcnt")
        assert twin in cake_jit.STAGES
        assert cake_jit.STAGES.index(stage) > cake_jit.STAGES.index(twin)
        assert cake_backend.STAGE_TENSORS[stage] == cake_backend.STAGE_TENSORS[twin] + (
            ("CNT_NEXT",) if base == "gemm_dw_acc" else ()
        )
    for bad in (
        "gemm_dw_acc_lc",
        "gemm_dw_cast_bf16_lc",
        "gemm_dw_acc_tma_lc_dcnt",
        "gemm_dw_acc_dcnt_lc",
        "gemm_dw_cast_f32_lc_dcnt",
        "gemm_dx_red_lc_dcnt",
        "gemm_logits_dz_lc_dcnt",
    ):
        with pytest.raises(ValueError):
            cake_backend.parse_stage(bad)
    for name, record in cake_jit.MODULES.items():
        present = {stage for stage in _LC_STAGES if stage in record}
        assert record["arch"] in ("sm_100a", "sm_103a", "sm_107a"), name
        assert present == set(_LC_STAGES), name
        for stage in present:
            twin = record[stage.replace("_lc_dcnt", "_dcnt")]
            assert record[stage]["grid"] == twin["grid"], (name, stage)
            assert record[stage]["launch"] == twin["launch"], (name, stage)
            plan = [slot for _kind, slot in record[stage]["arg_plan"]]
            want = [slot for _kind, slot in twin["arg_plan"]]
            if stage.startswith("gemm_dw_acc"):
                # the accumulate's form takes the deferred cast's operands plus the next chunk's count
                cast = record["gemm_dw_cast_bf16_dcnt"]["arg_plan"]
                want = [slot for _kind, slot in cast]
                want.insert(want.index("CNT") + 1, "CNT_NEXT")
            assert plan == want, (name, stage)


def test_plan_runs_the_live_cast():
    """``make_plan(dcnt=True, live_cast=True)``: a counted plan of two chunks or more with the fused bf16 cast runs the
    accumulate of every chunk but the last and the deferred cast in their ``_lc`` forms, the rest of the plan as the
    counted plan; one chunk, no counted forms or an unfused cast keep it off; the temperature plans carry it and the
    twin of the mixed plan (dcnt off) drops it."""
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
        logprob_lbl=True,
    )
    checked = 0
    # three chunks (the last of 3808 rows), two (the last of 650: the TMA cast; the shipped chunks, tail_fold=False)
    # and the T = 65031 timing row
    for T, rows in ((12000, 12000), (4746, 4746), (65031, 15 * 4096 + 339)):
        problem = _keep_problem(T, 6144, 154880, 4096, entry="logprob")
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            kw = dict(common, arch=arch, valid_rows=rows, tail_fold=False)
            counted = cake_backend.make_plan(problem, dcnt=True, **kw)
            lc = cake_backend.make_plan(problem, dcnt=True, live_cast=True, **kw)
            if not counted.dcnt:
                assert not lc.dcnt and not lc.live_cast
                continue
            checked += 1
            assert lc.live_cast and not counted.live_cast, (T, arch)
            assert dataclasses.replace(lc, live_cast=False) == counted
            for i in range(lc.num_chunks):
                assert lc.dw_acc_stage(i) == _lc_form(counted.dw_acc_stage(i), lc)
                assert lc.dx_stage_of(i) == counted.dx_stage_of(i)
                assert lc.recompute_stage(i) == counted.recompute_stage(i)
            assert lc.dw_cast_stage == _lc_form(counted.dw_cast_stage, lc)
            assert {s.replace("_lc_dcnt", "_dcnt") for s in lc.stages} == set(
                counted.stages
            )
            assert list(lc.stages) == [s for s in cake_jit.STAGES if s in lc.stages]
            assert [
                (s.replace("_lc_dcnt", "_dcnt"), i)
                for s, i in cake_backend.recompute_keys(lc)
            ] == list(cake_backend.recompute_keys(counted))
            # the accumulate of the last chunk is not a stage of a plan with the deferred cast
            last = lc.dw_acc_stage(lc.num_chunks - 1)
            assert all(
                key != (last, lc.num_chunks - 1)
                for key in cake_backend.recompute_keys(lc)
            )
            tau_problem = cake_backend._with_temperature(problem)
            for counted_dz in (True, False):
                tau = cake_backend.make_plan_temperature(
                    tau_problem, dcnt=True, live_cast=True, counted_dz=counted_dz, **kw
                )
                assert tau.live_cast and tau.dw_cast_stage == lc.dw_cast_stage
                twin = tau.without_temperature()
                assert twin.live_cast == counted_dz and twin.dcnt == counted_dz
    assert checked, "no architecture ran the counted forms"
    for T, kw in (
        (3000, dict(common, arch="sm_100a", valid_rows=3000)),  # one chunk
        (12000, dict(common, arch="sm_100a", valid_rows=12000, fuse_dw_cast=False)),
        (12000, dict(common, arch=None, valid_rows=12000)),  # the reference path
    ):
        problem = _keep_problem(T, 6144, 154880, 4096, entry="logprob")
        assert not cake_backend.make_plan(
            problem, dcnt=True, live_cast=True, **kw
        ).live_cast
    problem = _keep_problem(12000, 6144, 154880, 4096, entry="logprob")
    assert not cake_backend.make_plan(
        problem, live_cast=True, **dict(common, arch="sm_100a", valid_rows=12000)
    ).live_cast  # no counted forms


def test_plan_counts_the_short_last_chunk_of_a_long_call():
    """A log-probability call whose last chunk takes the TMA-epilogue bf16 cast in the shipped plan (61779 valid rows:
    fifteen chunks of 4096 and one of 339, the T = 65031 timing row) keeps the counted forms: its counted plan runs the
    shipped plan's chunk forms (the dX slices raised per ``DX_COUNTED_MIN_SLICES_BY_ARCH``), the last chunk's cast
    through the TMA cast's counted form, and no chunk takes the
    swapped dX tile (the swap rule serves the cross-entropy loss only); with ``counted_tma_cast`` off (a record without
    that form) it casts the last chunk through the counted shipped cast, every other chunk's forms unchanged.  An
    explicitly pinned weight-gradient TMA epilogue, or swapped dX tile, keeps the shipped plan."""
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
    rows = 15 * 4096 + 339
    problem = _keep_problem(65031, 6144, 154880, 4096, entry="logprob")
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        kw = dict(common, arch=arch)
        plain = cake_backend.make_plan(problem, valid_rows=rows, **kw)
        plan = cake_backend.make_plan(problem, valid_rows=rows, dcnt=True, **kw)
        old = cake_backend.make_plan(
            problem, valid_rows=rows, dcnt=True, counted_tma_cast=False, **kw
        )
        assert plain.num_chunks == 16 and plain.chunks[-1] == (15 * 4096, 339)
        assert not plain.dcnt and plain.variants[-1].dw_cast_tma
        assert plain.dw_cast_stage == "gemm_dw_cast_bf16_tma"
        assert plan.dcnt and old.dcnt, arch
        assert not any(v.dx_swab or v.dw_cast_tma for v in plain.variants[:-1])
        assert not any(v.dx_swab for v in plan.variants)
        _assert_counted_twin(plan, plain, arch)
        assert plan.variants[-1].dw_cast_tma
        assert plan.dw_cast_stage == cake_backend.DW_CAST_TMA_DCNT_STAGE
        assert {s.removesuffix("_dcnt") for s in _not_dx(plan.stages)} == _not_dx(
            plain.stages
        )
        assert old.variants[:-1] == plan.variants[:-1]
        assert (
            dataclasses.replace(old.variants[-1], dw_cast_tma=True) == plan.variants[-1]
        )
        assert old.dw_cast_stage == "gemm_dw_cast_bf16_dcnt"
        assert set(old.stages) == (
            set(plan.stages) - {cake_backend.DW_CAST_TMA_DCNT_STAGE}
        ) | {"gemm_dw_cast_bf16_dcnt"}
        for counted_plan in (plan, old):
            assert counted_plan.chunks == plain.chunks
            assert counted_plan.dx_slices == _counted_slices(plain, arch)
            for i in range(counted_plan.num_chunks):
                assert counted_plan.recompute_stage(i) == "gemm_logits_dz_dcnt"
                assert counted_plan.dx_stage_of(i).endswith("_dcnt") == (
                    counted_plan.dx_slices_of(i) > 1
                ), counted_plan.dx_stage_of(i)
            assert all(
                s.endswith("_dcnt")
                for s in counted_plan.stages
                if s.startswith("gemm_dw")
            )
        pins = [{"dw": {"epi_store": "tma"}}]
        if arch in cake_backend.DX_SWAB_ARCHES:
            pins.append({"dx": {"swap_ab": True, "tile_n": 512}})
        for pin in pins:
            pinned = dict(kw, gemm_tuning=pin)
            shipped = cake_backend.make_plan(problem, valid_rows=rows, **pinned)
            counted = cake_backend.make_plan(
                problem, valid_rows=rows, dcnt=True, **pinned
            )
            assert not counted.dcnt and counted == shipped, pin


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
    # the short-tail fold: one counted chunk of 4370 / 5700 valid rows (the live count follows the folded chunk)
    (4600, ("random", 0.166)),
    (4600, "last"),
    (6000, ("random", 0.5)),
]


@pytest.mark.parametrize("T, kind", ZERO_DLOGP_DEVICE_CASES, ids=str)
def test_device_zero_dlogp_rows_skip(glm_weight, request, T, kind):
    """The zero-gradient row skip on the device against the same backward without it (``set_zero_dlogp_min_rows(None)``)
    and the FP64 oracle: ``logp`` the same bits; the deterministic suite's gates; the zero rows' and the ignored rows'
    ``dX`` rows exact ``+0``; the other ``dX`` rows equal to the backward without the skip within one BF16 rounding
    (the K-sliced reductions add in any order); a second call from the remembered binding gives the first's ``dW``
    within the FP32 reduction order.  Every valid row zero: zero gradients.  The short-tail fold cases (T 4600 / 6000)
    run the chunk loop (``logprob_chunk_loop``): by default a call of 2 chunks with a short tail takes the
    vocabulary-major backward, which never runs the folded chunk."""
    _require_program(entry="logprob")
    if T in (4600, 6000):
        request.getfixturevalue("logprob_chunk_loop")
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


# One-chunk calls (1 to 4096 valid rows) under the skip at floor 0: an all-live chunk ("none"), an all-zero chunk
# ("all"), one live row ("single"), one zero row ("one") and random / grouped zero rows; 1946 valid rows = T 2048 with
# 5 % ignored rows, 3891 = T 4096.
ZERO_DLOGP_ONE_CHUNK_CASES = [
    (1, "none"),
    (1, "all"),
    (2, "one"),
    (31, ("random", 0.5)),
    (255, ("random", 0.166)),
    (1025, "none"),
    (1025, ("random", 0.166)),
    (1100, ("random", 0.5)),
    (1946, "none"),
    (1946, "all"),
    (1946, "single"),
    (1946, "one"),
    (1946, ("random", 0.166)),
    (1946, ("random", 0.5)),
    (1946, ("group", 0.5)),
    (3891, ("random", 0.166)),
    (4096, "none"),
    (4096, ("random", 0.5)),
]


@pytest.mark.parametrize("rows, kind", ZERO_DLOGP_ONE_CHUNK_CASES, ids=str)
def test_device_zero_dlogp_one_chunk_skip(glm_weight, monkeypatch, rows, kind):
    """At floor 0 a one-chunk backward of ``rows`` valid rows (one ignored row makes the call compacted) runs the skip:
    the partition runs on every call (binding and remembered binding) over every valid row, the backward plan has the
    counted forms and one chunk, ``logp`` keeps the bits of the backward without the skip, the zero rows' and the
    ignored rows' ``dX`` rows are exact ``+0``, the live ``dX`` rows are within one BF16 rounding of the backward
    without the skip, ``dW`` keeps its bits from run to run and the outputs pass the deterministic suite's gates
    against the FP64 oracle.  Without zero rows ``dW`` is bitwise the backward without the skip's (the partition keeps
    the forward's row order); every valid row zero gives zero gradients."""
    _require_zero_dlogp_skip()
    # the counted path of the default Function: a call of at most uncompacted_max_rows() rows would run uncompacted
    # (no row skip), one of at most SMALL_T_KEEP_MAX_ROWS rows the kept logits
    monkeypatch.setitem(cake_backend._UNCOMPACTED_MAX_ROWS, "rows", 0)
    monkeypatch.setitem(cake_backend._SMALL_T_KEEP, "on", False)
    T = rows + 1
    base = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
    labels = base.labels.clone()
    labels[T // 2] = -100
    dlogp = base.dlogp
    base = dataclasses.replace(
        base,
        labels=labels,
        dlogp=torch.where(dlogp == 0, torch.ones_like(dlogp), dlogp).contiguous(),
    )
    zero = _zero_dlogp_set(base, kind)
    inp = _with_zero_dlogp(base, zero)
    valid = inp.labels >= 0
    live = valid & ~zero
    assert int(valid.sum()) == rows
    assert bool((inp.dlogp[live] != 0).all())
    case = (rows, kind)
    calls = _counted_partition(monkeypatch)
    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(0):
        nd = _logprob_call(inp)
        again = _logprob_call(inp)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    assert calls == [rows, rows], (case, calls)
    (skip_plan, _), (shipped, _) = _logprob_bwd_plans()
    assert skip_plan.dcnt and not shipped.dcnt, case
    assert skip_plan.num_chunks == 1 and shipped.num_chunks == 1, case
    for run in (nd, again):
        assert _same_bits(run["logp"], plain["logp"]), case
        assert not bool(_bits(run["dX"][zero | ~valid]).any()), case
    assert _same_bits(again["dW"], nd["dW"]), case
    if not bool(live.any()):
        for run in (nd, again):
            assert not bool(_bits(run["dX"]).any()), case
            assert not bool(run["dW"].float().any()), case
        return
    if not bool(zero.any()):
        assert _same_bits(nd["dW"], plain["dW"]), case
    scale = float(plain["dX"][live].float().abs().max())
    dx_diff = 0.0
    for run in (nd, again):
        diff = float((run["dX"][live].float() - plain["dX"][live].float()).abs().max())
        assert diff <= scale * 2**-7, (case, diff, scale)
        dx_diff = max(dx_diff, diff)
        _check_against_references(run, inp, entry="logprob", ceiling=True)
    dw_rel = float(rel_l2(nd["dW"].float(), plain["dW"].float()))
    assert dw_rel <= GATE_TINY["dW_rel_l2"], (case, dw_rel)
    print(
        f"\nzero-dlogp one-chunk skip rows={rows} {kind} zero {int(zero.sum())}/{rows}: dW rel-L2 vs no skip "
        f"{dw_rel:.2e}, dX max|skip - no skip| {dx_diff:.2e}"
    )


def _one_chunk_skip_inputs(W, T, kind, tau):
    """``T`` rows (5 % ignored: a compacted one-chunk call) with no zero ``dlogp`` outside ``kind``'s zero rows
    (:func:`_zero_dlogp_set`) and, with ``tau="distinct"``, a distinct per-token temperature."""
    base = _device_inputs(T, "ce", W=W)
    dlogp = base.dlogp
    base = dataclasses.replace(
        base, dlogp=torch.where(dlogp == 0, torch.ones_like(dlogp), dlogp).contiguous()
    )
    zero = _zero_dlogp_set(base, kind)
    inp = _with_zero_dlogp(base, zero)
    if tau is not None:
        inp = dataclasses.replace(inp, temperature=_distinct_tau(T))
    return inp, zero


@pytest.mark.parametrize("slices", ["one", "default", "raised"])
@pytest.mark.parametrize("warm", [1, 2])
@pytest.mark.parametrize("tau", [None, "distinct"])
def test_device_zero_dlogp_one_chunk_backward_cuda_graph(
    glm_weight, monkeypatch, tau, warm, slices
):
    """A one-chunk log-probability backward under the zero-gradient row skip (plain and with a distinct per-token
    temperature) captured in a CUDA graph through its remembered binding -- the first warm-up call (the runner's) makes
    and remembers the binding, so the captured call is the binding's first remembered call after one warm-up and its
    second after two -- replays with other ``dlogp`` values written into the captured input
    (other zero rows and live counts, none, a single live row and every row zero) and gives the eager backward's bits
    on those values: ``dW`` bitwise; ``dX`` bitwise with one dX slice pinned and, with the default slices (an
    unordered split-K reduction) or ``"raised"`` ones (the chunk below the K-sliced reduction's row floor, so the slice
    rule gives it one slice and the counted plan the architecture's ``DX_COUNTED_MIN_SLICES_BY_ARCH`` count), exact
    ``+0`` on the zero and ignored rows and the eager values within the reduction order on the live rows.  The replays leave the forward's ``lse`` bits unchanged, and every captured and eager call
    ran the partition over every valid row."""
    _require_zero_dlogp_skip()
    if tau is not None:
        _require_tau_program()
    inp, _zero = _one_chunk_skip_inputs(glm_weight, 2048, "none", tau)
    kw = dict(chunk_size=4096)
    if tau is None:
        fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, **kw)
        entry, extra = cake_backend.backward_logprob, {}
    else:
        fr = cake_backend.forward_logprob_temperature(
            inp.X, inp.W, inp.labels, temperature=inp.temperature, **kw
        )
        form = cake_backend._resolve_temperature(
            inp.temperature, int(inp.labels.numel()), inp.X.device
        )
        entry = cake_backend.backward_logprob_temperature
        extra = dict(temperature=form, temperature_rows=fr.temperature_rows)
    if slices == "one":
        extra.update(_TAU_ONE)
    if slices == "raised":
        monkeypatch.setitem(cake_backend._DX_RED_MIN_ROWS, "rows", 4097)
    torch.cuda.synchronize()
    assert fr.row_index is not None
    lse_bits = _bits(fr.lse).clone()
    dlogp = inp.dlogp.clone()

    def backward(d):
        return entry(
            inp.X, inp.W, inp.labels, fr.lse, d, row_index=fr.row_index, **extra, **kw
        )

    valid = inp.labels >= 0
    rows = int(valid.sum())
    calls = _counted_partition(monkeypatch)
    with _zero_dlogp_floor(0):
        cake_backend.BINDING_CACHE.clear()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warm):
                backward(dlogp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dx_g, dw_g = backward(dlogp)
        for kind in (("random", 0.166), ("random", 0.5), "none", "single", "all"):
            zero = _zero_dlogp_set(inp, kind)
            new = _with_zero_dlogp(inp, zero).dlogp
            dlogp.copy_(new)
            graph.replay()
            torch.cuda.synchronize()
            got_dx, got_dw = dx_g.clone(), dw_g.clone()
            want_dx, want_dw = backward(new.clone())
            torch.cuda.synchronize()
            assert _same_bits(got_dw, want_dw), kind
            assert not bool(_bits(got_dx[zero | ~valid]).any()), kind
            if slices == "one":
                assert _same_bits(got_dx, want_dx), kind
                continue
            live = valid & ~zero
            if bool(live.any()):
                scale = float(want_dx[live].float().abs().max())
                err = float((got_dx[live].float() - want_dx[live].float()).abs().max())
                assert err <= scale * 2**-7, (kind, err, scale)
    assert calls == [rows] * (warm + 6), calls
    bindings = [
        b
        for k, b in cake_backend.BINDING_CACHE._bindings.items()
        if k[:2] == ("bwd", "logprob") and b.plan.dcnt
    ]
    assert len(bindings) == 1
    assert bindings[0].holds_no_tensor()
    if slices == "raised":
        _module_name, record = cake_backend.record_for(CUDA)
        least = cake_backend.DX_COUNTED_MIN_SLICES_BY_ARCH.get(record["arch"], 1)
        assert bindings[0].plan.dx_slices == (least,), bindings[0].plan.dx_slices
    assert torch.equal(_bits(fr.lse), lse_bits)


@pytest.mark.parametrize("tau", [None, "distinct"])
def test_device_logprob_backward_reuses_the_forward_problem(
    glm_weight, monkeypatch, tau
):
    """The autograd log-probability backward takes the forward's validated problem back (``ForwardResult.problem``,
    the internal ``_problem`` keyword) instead of validating ``X`` / ``W`` / ``labels`` again: one validation fewer per
    call.  Under saved-tensor hooks (which may unpack another ``X``) the backward gets none and validates its own.  The
    bits are the same either way (plain and with a temperature)."""
    _require_program(entry="logprob")
    if tau is not None:
        _require_tau_program()
    inp, _zero = _one_chunk_skip_inputs(glm_weight, 2048, "none", tau)
    name = "backward_logprob" if tau is None else "backward_logprob_temperature"
    validate, backward = (
        cake_backend.validate_lm_head_inputs,
        getattr(cake_backend, name),
    )
    validated, problems = [], []

    def counting(*args, **kwargs):
        validated.append(kwargs.get("entry"))
        return validate(*args, **kwargs)

    def recording(*args, **kwargs):
        problems.append(kwargs.get("_problem"))
        return backward(*args, **kwargs)

    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    cake_backend.BINDING_CACHE.clear()
    first = _run(inp, 4096, **kw)  # remembers the bindings
    monkeypatch.setattr(cake_backend, "validate_lm_head_inputs", counting)
    monkeypatch.setattr(cake_backend, name, recording)
    plain = _run(inp, 4096, **kw)
    unhooked = len(validated)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        hooked = _run(inp, 4096, **kw)
    assert len(validated) == 2 * unhooked + 1, validated
    assert len(problems) == 2, problems
    assert problems[0] is not None and problems[1] is None
    assert problems[0].num_rows == 2048 and not problems[0].temperature
    for key in ("logp", "dX", "dW"):
        assert _same_bits(plain[key], first[key]), key
        assert _same_bits(hooked[key], first[key]), key


class _ScratchProbe(dict):
    """The host values of a remembered binding's call with only the scratch slots ``allowed``
    (``cake_backend._SCRATCH_SLOTS``): an absent slot raises ``KeyError`` (``t[name]``) or reads ``None``
    (``t.get(name)``); every other name is the placeholder tensor."""

    def __init__(self, allowed, placeholder):
        super().__init__()
        self.allowed, self.placeholder = frozenset(allowed), placeholder

    def _absent(self, name):
        return name in cake_backend._SCRATCH_SLOTS and name not in self.allowed

    def __getitem__(self, name):
        if self._absent(name):
            raise KeyError(name)
        return self.placeholder

    def get(self, name, default=None):
        return default if self._absent(name) else self.placeholder


# Plans whose bindings :func:`test_device_binding_scratch_slots_cover_every_launch` checks: the loss entry (CE, policy),
# the log-probability entry, under the zero-gradient row skip and with a temperature, at one chunk (T 2048) and three
# (T 9000, the hidden valid-row count).
SCRATCH_SLOT_CASES = [
    (T, case)
    for T in (2048, 9000)
    for case in ("loss_ce", "loss_policy", "logprob", "logprob_skip", "logprob_tau")
]


@pytest.mark.parametrize("T, case", SCRATCH_SLOT_CASES, ids=str)
def test_device_binding_scratch_slots_cover_every_launch(glm_weight, T, case):
    """A remembered binding allocates per call only the scratch slots (``d`` / ``term`` / ``loss_acc`` / ``loss``) its
    sequences bind or return: the log-probability forward ``d`` and ``term`` (``row_finalize``), its backward none,
    the loss entry's forward all four.  Every launch and host step of a binding's forward / backward sequence finds
    every value it binds when only that sequence's scratch slots are present, and the remembered calls give the first
    call's bits.  The log-probability cases set the zero-gradient row skip's floor: ``logprob_skip`` runs it (``0``),
    ``logprob`` and ``logprob_tau`` run the shipped backward (``None``) on every architecture -- on sm_100a and
    sm_107a the default floor (1946 valid rows) would put both sizes on the skip."""
    _require_program(entry="loss" if case.startswith("loss") else "logprob")
    objective = "policy" if case == "loss_policy" else "ce"
    inp = _device_inputs(T, objective, W=glm_weight)
    floor = None
    if case == "logprob_skip":
        _require_zero_dlogp_skip()
        inp = _with_zero_dlogp(inp, _zero_dlogp_set(inp, ("random", 0.166)))
        floor = 0
    if case == "logprob_tau":
        _require_tau_program()
        inp = dataclasses.replace(inp, temperature=_distinct_tau(T))
    entry = "loss" if case.startswith("loss") else "logprob"
    kw = dict(entry=entry, backend="cake", **_TAU_ONE)
    cake_backend.BINDING_CACHE.clear()
    with contextlib.ExitStack() as stack:
        if entry == "logprob":
            stack.enter_context(_zero_dlogp_floor(floor))
        runs = [_run(inp, 4096, **kw) for _ in range(3)]
    for run in runs[1:]:
        for key in ("loss", "logp", "dX", "dW"):
            assert _same_bits(run[key], runs[0][key]), (case, key)
    placeholder = torch.empty((0,), device=CUDA)
    checked = {"fwd": 0, "bwd": 0}
    for key, binding in list(cake_backend.BINDING_CACHE._bindings.items()):
        kind, binding_entry = key[0], key[1]
        if kind not in checked:
            continue
        slot_values = (
            cake_backend._slot_values_temperature
            if isinstance(binding, cake_backend._TemperatureBinding)
            else cake_backend._slot_values
        )
        if kind == "fwd":
            run, allowed = binding.forward_order, binding.forward_scratch
            if binding_entry == "loss":
                assert allowed == cake_backend._SCRATCH_SLOTS, (case, allowed)
            else:
                assert set(allowed) <= {"d", "term"}, (case, allowed)
        else:
            run, allowed = binding.backward_order, binding.backward_scratch
            if binding_entry == "logprob":
                assert allowed == (), (case, allowed)
        probe = _ScratchProbe(allowed, placeholder)
        for launch_key in run:
            values = slot_values(launch_key[0], probe, binding.plan, launch_key[1])
            launch = binding.launches.get(launch_key)
            if launch is not None:
                launch.arguments_for(values)
        checked[kind] += 1
    assert checked["fwd"] >= 1, (case, checked)
    assert entry == "loss" or checked["bwd"] >= 1, (case, checked)
    if entry == "logprob":
        skip = {
            bool(b.plan.dcnt)
            for k, b in cake_backend.BINDING_CACHE._bindings.items()
            if k[0] == "bwd"
        }
        assert skip == {floor == 0}, (case, skip)


def test_device_zero_dlogp_default_floor_per_architecture(glm_weight, monkeypatch):
    """Without an override a one-chunk log-probability backward runs the skip from this device's default floor on
    (``zero_dlogp_min_rows(index)``: 1946 valid rows on sm_100a and sm_107a, 3891 on sm_103a, else 4097): a backward
    of one row fewer runs without it (the zero rows' ``dX`` zeros either way)."""
    _require_zero_dlogp_skip()
    index = torch.cuda.current_device()
    _module_name, record = cake_backend.record_for(CUDA)
    floor = cake_backend.ZERO_DLOGP_MIN_ROWS_BY_ARCH.get(
        record["arch"], cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT
    )
    calls = _counted_partition(monkeypatch)
    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(cake_backend.ZERO_DLOGP_MIN_ROWS_ARCH_DEFAULT):
        assert cake_backend.zero_dlogp_min_rows(index) == floor
        for rows in (floor - 1, floor):
            T = rows + 1
            base = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
            labels = base.labels.clone()
            # one ignored row: a compacted call of T - 1 valid rows
            labels[T // 2] = -100
            base = dataclasses.replace(base, labels=labels)
            zero = _zero_dlogp_set(base, ("random", 0.3))
            run = _logprob_call(_with_zero_dlogp(base, zero, negative_zero=False))
            assert not bool(run["dX"][zero].float().any()), rows
    assert calls == [floor], (floor, calls)


def test_device_zero_dlogp_skip_runs_the_partition_kernel(glm_weight, monkeypatch):
    """The backward with the skip forms its row order with the partition kernel, never the PyTorch reference, and gives
    the backward that forms it with the reference the same ``dW`` bits (the same row order) and the same exact ``+0``
    zero rows of ``dX``."""
    _require_zero_dlogp_skip()
    module_name, _record = cake_backend.record_for(CUDA)
    assert cake_backend.ZERO_DLOGP_PARTITION_STAGE in cake_jit.registered_stages(
        module_name
    )
    base = _device_inputs(8117, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero)
    reference = cake_backend.zero_dlogp_partition_reference
    partition = cake_backend.zero_dlogp_partition
    calls = []

    def refused(*args, **kwargs):
        raise AssertionError("the skip ran the PyTorch reference partition")

    def counted(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return partition(*args, **kwargs)

    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(0):
        monkeypatch.setattr(cake_backend, "zero_dlogp_partition_reference", refused)
        monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counted)
        kernel = _logprob_call(inp)
        assert calls, "the backward did not run the zero-gradient row skip"
        monkeypatch.setattr(cake_backend, "zero_dlogp_partition_reference", reference)
        monkeypatch.setattr(
            cake_backend, "zero_dlogp_partition", lambda *a, **k: reference(*a)
        )
        torch_order = _logprob_call(inp)
    valid = inp.labels >= 0
    assert _same_bits(kernel["logp"], torch_order["logp"])
    assert _same_bits(kernel["dW"], torch_order["dW"])
    for run in (kernel, torch_order):
        assert not bool(_bits(run["dX"][zero | ~valid]).any())


def _logprob_bwd_plans():
    """The plans of the remembered log-probability backward bindings, oldest first."""
    return [
        (b.plan, b.backward_order)
        for k, b in cake_backend.BINDING_CACHE._bindings.items()
        if k[:2] == ("bwd", "logprob")
    ]


# The cast at the last live chunk (``_lc`` before ``_dcnt``, :attr:`cake_backend.Plan.live_cast`)
_LC_STAGES = tuple(stage for stage in cake_jit.STAGES if stage.endswith("_lc_dcnt"))


def _lc_form(stage, plan):
    """``stage`` (a counted weight-gradient form) as ``plan`` runs it: its ``_lc`` form when the plan casts at the
    last live chunk."""
    return stage.replace("_dcnt", "_lc_dcnt") if plan.live_cast else stage


def _lc_registered(module_name):
    """Whether record ``module_name`` registers the cast at the last live chunk (all four ``_lc`` forms, or none)."""
    stages = cake_jit.registered_stages(module_name)
    present = [stage in stages for stage in _LC_STAGES]
    assert all(present) or not any(present), present
    return all(present)


@contextlib.contextmanager
def _without_live_cast(monkeypatch):
    """Run with the records' ``_lc`` forms hidden from the plan builders (the skip then runs the counted forms)."""
    plain = cake_backend.registered_stages
    with_tau = cake_backend.registered_stages_with_temperature
    with monkeypatch.context() as m:
        m.setattr(
            cake_backend,
            "registered_stages",
            lambda name: tuple(s for s in plain(name) if s not in _LC_STAGES),
        )
        m.setattr(
            cake_backend,
            "registered_stages_with_temperature",
            lambda name: tuple(s for s in with_tau(name) if s not in _LC_STAGES),
        )
        yield


def test_device_zero_dlogp_plan_without_counted_forms_runs_the_shipped_backward(
    logprob_chunk_loop, glm_weight, monkeypatch
):
    """An eligible backward whose plan has no counted forms (forced here: ``make_plan`` resolves ``dcnt`` off, as it
    does for an explicitly pinned swapped dX tile or weight-gradient TMA epilogue) launches no partition -- binding
    (first call) and from the remembered binding (second call) -- and runs the backward without the skip: the plan and
    launch sequence of the backward with the skip off, the same ``logp`` and ``dW`` bits, ``dX`` within its unordered
    K-slice reduction, the zero rows' ``dX`` zeros."""
    _require_zero_dlogp_skip()
    base = _device_inputs(8117, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero, negative_zero=False)
    make_plan = cake_backend.make_plan

    def uncounted(*args, **kwargs):
        return make_plan(*args, **dict(kwargs, dcnt=False))

    def refused(*args, **kwargs):
        raise AssertionError(
            "a backward whose plan has no counted forms ran the partition"
        )

    cake_backend.BINDING_CACHE.clear()
    monkeypatch.setattr(cake_backend, "make_plan", uncounted)
    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", refused)
    with _zero_dlogp_floor(0):
        runs = [_logprob_call(inp), _logprob_call(inp)]
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    (eligible, eligible_run), (shipped, shipped_run) = _logprob_bwd_plans()
    assert not eligible.dcnt and eligible == shipped and eligible_run == shipped_run
    valid = inp.labels >= 0
    live = valid & ~zero
    scale = float(plain["dX"][live].float().abs().max())
    for run in runs:
        for key in ("logp", "dW"):
            assert _same_bits(run[key], plain[key]), key
        assert not bool(run["dX"][zero | ~valid].float().any())
        dx_diff = float(
            (run["dX"][live].float() - plain["dX"][live].float()).abs().max()
        )
        assert dx_diff <= scale * 2**-7, (dx_diff, scale)


def test_device_zero_dlogp_skip_is_off_without_the_counted_forms(
    logprob_chunk_loop, glm_weight, monkeypatch
):
    """On a record that registers no zero-gradient row skip (no counted forms, no partition kernel; every shipped
    record registers them, so the device's record is read here as one without them: its ``_skip_record`` entry) a
    backward that would be eligible elsewhere (floor 0, compacted, the dz arm) is not eligible and runs the shipped
    backward: no partition, binding (first call) and from the remembered binding (second call); every remembered plan
    without counted forms; the same ``logp`` and ``dW`` bits as the backward with the skip off, ``dX`` within its
    unordered K-slice reduction, the zero rows' ``dX`` zeros."""
    _require_program(entry="logprob")
    module_name, record = cake_backend.record_for(CUDA)
    device = CUDA.index if CUDA.index is not None else torch.cuda.current_device()
    monkeypatch.setattr(
        cake_backend, "_SKIP_RECORDS", {device: (module_name, record, False, False)}
    )
    base = _device_inputs(8117, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero, negative_zero=False)
    valid = inp.labels >= 0
    index = valid.nonzero().squeeze(1)
    with _zero_dlogp_floor(0):
        assert not cake_backend._zero_dlogp_skip(CUDA, index, "cake", True, True, True)

    def refused(*args, **kwargs):
        raise AssertionError("a record without the counted forms ran the partition")

    cake_backend.BINDING_CACHE.clear()
    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", refused)
    with _zero_dlogp_floor(0):
        runs = [_logprob_call(inp), _logprob_call(inp)]
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    plans = _logprob_bwd_plans()
    assert plans and not any(plan.dcnt for plan, _run in plans)
    live = valid & ~zero
    scale = float(plain["dX"][live].float().abs().max())
    for run in runs:
        for key in ("logp", "dW"):
            assert _same_bits(run[key], plain[key]), key
        assert not bool(run["dX"][zero | ~valid].float().any())
        dx_diff = float(
            (run["dX"][live].float() - plain["dX"][live].float()).abs().max()
        )
        assert dx_diff <= scale * 2**-7, (dx_diff, scale)
    print(
        f"\nzero-dlogp skip on {record['arch']} (record read without it): off; "
        f"{len(plans)} remembered backward plan(s), none counted"
    )


def test_device_zero_dlogp_short_last_chunk_keeps_the_counted_cast(
    logprob_chunk_loop, glm_weight, monkeypatch, shipped_chunks
):
    """A backward whose last chunk the shipped plan casts through the TMA-epilogue bf16 cast (two chunks, the second of
    about 650 rows) keeps the skip: the partition runs (binding and remembered binding), the remembered plan casts the
    last chunk through the TMA cast's counted form where the record registers it (else the counted shipped cast), and
    the results pass the skip suite's checks against the backward without the skip and the FP64 oracle.  The shipped
    chunks (``shipped_chunks``: no short-tail fold)."""
    _require_zero_dlogp_skip()
    _module_name, record = cake_backend.record_for(CUDA)
    base = _device_inputs(5000, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.5))
    inp = _with_zero_dlogp(base, zero)
    partition = cake_backend.zero_dlogp_partition
    calls = []

    def counted(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return partition(*args, **kwargs)

    cake_backend.BINDING_CACHE.clear()
    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counted)
    with _zero_dlogp_floor(0):
        nd = _logprob_call(inp)
        again = _logprob_call(inp)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    valid = inp.labels >= 0
    rows = int(valid.sum())
    assert calls == [rows, rows]
    (skip_plan, _), (shipped, _) = _logprob_bwd_plans()
    assert skip_plan.dcnt and not shipped.dcnt
    assert skip_plan.num_chunks == 2 and skip_plan.chunks[-1][1] == rows - 4096
    tma = cake_backend.dw_cast_tma_rule(2, rows - 4096, record["arch"])
    assert shipped.dw_cast_stage == (
        "gemm_dw_cast_bf16_tma" if tma else "gemm_dw_cast_bf16"
    )
    stages = cake_jit.registered_stages(_module_name)
    assert skip_plan.live_cast == _lc_registered(_module_name)
    assert skip_plan.dw_cast_stage == _lc_form(
        cake_backend.DW_CAST_TMA_DCNT_STAGE
        if tma and cake_backend.DW_CAST_TMA_DCNT_STAGE in stages
        else "gemm_dw_cast_bf16_dcnt",
        skip_plan,
    )
    live = valid & ~zero
    assert _same_bits(nd["logp"], plain["logp"])
    scale = float(plain["dX"][live].float().abs().max())
    for run in (nd, again):
        assert not bool(_bits(run["dX"][zero | ~valid]).any())
        dx_diff = float(
            (run["dX"][live].float() - plain["dX"][live].float()).abs().max()
        )
        assert dx_diff <= scale * 2**-7, (dx_diff, scale)
        _check_against_references(run, inp, entry="logprob", ceiling=True)


@pytest.mark.parametrize("last", [650, 1792, 1793])
def test_device_zero_dlogp_short_last_chunk_cast_bits_without_zero_rows(
    logprob_chunk_loop, glm_weight, monkeypatch, last, shipped_chunks
):
    """Without zero-gradient rows (z = 0) a two-chunk backward under the skip gives the bits of the backward without
    it: the partition runs, every row stays live, and ``dW`` and ``logp`` are bitwise those of the skip-off run.  One
    ignored row makes the call compacted (the skip needs the forward's row index).  A last chunk of 650 or 1792 valid
    rows (``dw_cast_tma_rule`` holds on an architecture of ``DW_CAST_TMA_MAX_ROWS``) casts through the TMA cast's
    counted form ``gemm_dw_cast_bf16_tma_dcnt``; 1793 rows (one row above the rule's bound, the negative control) and
    every last chunk on an architecture without the rule take the counted per-thread cast, as the plan without the
    skip takes the per-thread cast.  The shipped chunks (``shipped_chunks``: no short-tail fold)."""
    _require_zero_dlogp_skip()
    module_name, record = cake_backend.record_for(CUDA)
    rows = cake_backend.DEFAULT_CHUNK_SIZE + last
    inp = _device_inputs(rows + 1, "ce", W=glm_weight, ignore_frac=0.0)
    labels = inp.labels.clone()
    # one ignored row: a compacted call of ``rows`` valid rows
    labels[rows // 2] = -100
    dlogp = inp.dlogp
    inp = dataclasses.replace(
        inp,
        labels=labels,
        dlogp=torch.where(dlogp == 0, torch.ones_like(dlogp), dlogp).contiguous(),
    )
    valid = inp.labels >= 0
    assert int(valid.sum()) == rows and bool((inp.dlogp[valid] != 0).all())
    partition = cake_backend.zero_dlogp_partition
    calls = []

    def counted(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return partition(*args, **kwargs)

    cake_backend.BINDING_CACHE.clear()
    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counted)
    with _zero_dlogp_floor(0):
        nd = _logprob_call(inp)
    with _zero_dlogp_floor(None):
        plain = _logprob_call(inp)
    assert calls == [rows]
    (skip_plan, _), (shipped, _) = _logprob_bwd_plans()
    assert skip_plan.dcnt and not shipped.dcnt
    assert skip_plan.num_chunks == 2 and skip_plan.chunks[-1][1] == last
    tma = cake_backend.dw_cast_tma_rule(2, last, record["arch"])
    bound = cake_backend.DW_CAST_TMA_MAX_ROWS.get(record["arch"])
    assert tma == (bound is not None and last <= bound)
    assert shipped.dw_cast_stage == (
        "gemm_dw_cast_bf16_tma" if tma else "gemm_dw_cast_bf16"
    )
    assert skip_plan.live_cast == _lc_registered(module_name)
    if tma:
        stages = cake_jit.registered_stages(module_name)
        assert cake_backend.DW_CAST_TMA_DCNT_STAGE in stages
        assert skip_plan.dw_cast_stage == _lc_form(
            cake_backend.DW_CAST_TMA_DCNT_STAGE, skip_plan
        )
    else:
        assert skip_plan.dw_cast_stage == _lc_form("gemm_dw_cast_bf16_dcnt", skip_plan)
    for key in ("logp", "dW"):
        assert _same_bits(nd[key], plain[key]), key
    assert not bool(_bits(nd["dX"][~valid]).any())
    scale = float(plain["dX"].float().abs().max())
    dx_diff = float((nd["dX"].float() - plain["dX"].float()).abs().max())
    assert dx_diff <= scale * 2**-7, (dx_diff, scale)
    print(
        f"\nzero-dlogp skip on {record['arch']}, z = 0, last chunk {last} rows: "
        f"cast {skip_plan.dw_cast_stage} (skip off: {shipped.dw_cast_stage}); "
        f"dW and logp bitwise equal, max |dX - skip off| {dx_diff:.3g}"
    )


# Backwards whose plan without the skip has one-slice dX chunks: "forced" puts every chunk below the K-sliced
# reduction's row floor (``set_dx_red_min_rows(T)``: one slice on every chunk, any architecture and SM count), "rule"
# keeps the slice rule.
ZERO_DLOGP_RAISED_CASES = [
    ("forced", 4097, ("random", 0.5)),
    ("forced", 8117, "none"),
    ("forced", 8117, ("random", 0.166)),
    ("forced", 12289, ("group", 0.5)),
    ("forced", 12289, "last"),
    ("forced", 12289, "all"),
    ("rule", 8117, ("random", 0.5)),
    ("rule", 13500, ("random", 0.5)),
]


@pytest.mark.parametrize("mode, T, kind", ZERO_DLOGP_RAISED_CASES, ids=str)
def test_device_zero_dlogp_raised_dx_slices(
    logprob_chunk_loop, glm_weight, mode, T, kind
):
    """Under the skip the backward's plan runs each dX chunk at the architecture's counted slice count at least
    (``DX_COUNTED_MIN_SLICES_BY_ARCH``: two on sm_100a, sm_103a and sm_107a) and the backward without the skip the
    slice rule's counts; every chunk of more than one slice launches its counted form.
    The results pass the skip suite's checks on the binding and the remembered binding: ``logp`` the bits of the
    backward without the skip; the zero and ignored rows' ``dX`` exact ``+0``; the live rows' ``dX`` within one BF16
    rounding of the backward without the skip; ``dW`` bitwise without zero rows (the forward's row order); the
    deterministic suite's gates against the FP64 oracle.  Every valid row zero: zero gradients."""
    _require_zero_dlogp_skip()
    _module_name, record = cake_backend.record_for(CUDA)
    least = cake_backend.DX_COUNTED_MIN_SLICES_BY_ARCH.get(record["arch"], 1)
    base = _device_inputs(T, "ce", W=glm_weight)
    dlogp = base.dlogp
    base = dataclasses.replace(
        base, dlogp=torch.where(dlogp == 0, torch.ones_like(dlogp), dlogp).contiguous()
    )
    zero = _zero_dlogp_set(base, kind)
    inp = _with_zero_dlogp(base, zero)
    case = (mode, T, kind)
    previous = cake_backend.set_dx_red_min_rows(T) if mode == "forced" else None
    try:
        cake_backend.BINDING_CACHE.clear()
        with _zero_dlogp_floor(0):
            nd = _logprob_call(inp)
            again = _logprob_call(inp)
        with _zero_dlogp_floor(None):
            plain = _logprob_call(inp)
        (skip_plan, skip_run), (shipped, _) = _logprob_bwd_plans()
    finally:
        if previous is not None:
            cake_backend.set_dx_red_min_rows(previous)
    assert skip_plan.dcnt and not shipped.dcnt, case
    assert skip_plan.chunks == shipped.chunks, case
    if mode == "forced":
        assert shipped.dx_slices == (1,) * shipped.num_chunks, (case, shipped.dx_slices)
    assert skip_plan.dx_slices == tuple(max(s, least) for s in shipped.dx_slices), (
        case,
        skip_plan.dx_slices,
        shipped.dx_slices,
    )
    dx_run = sorted(k[0] for k in skip_run if k[0].startswith("gemm_dx"))
    assert dx_run == sorted(
        skip_plan.dx_stage_of(i) for i in range(skip_plan.num_chunks)
    ), (case, dx_run)
    for i in range(skip_plan.num_chunks):
        assert skip_plan.dx_stage_of(i).endswith("_dcnt") == (
            skip_plan.dx_slices_of(i) > 1
        ), (case, i)
    valid = inp.labels >= 0
    live = valid & ~zero
    for run in (nd, again):
        assert _same_bits(run["logp"], plain["logp"]), case
        assert not bool(_bits(run["dX"][zero | ~valid]).any()), case
    if not bool(live.any()):
        for run in (nd, again):
            assert not bool(_bits(run["dX"]).any()), case
            assert not bool(run["dW"].float().any()), case
        return
    if not bool(zero.any()):
        assert _same_bits(nd["dW"], plain["dW"]), case
    scale = float(plain["dX"][live].float().abs().max())
    dx_diff = 0.0
    for run in (nd, again):
        diff = float((run["dX"][live].float() - plain["dX"][live].float()).abs().max())
        assert diff <= scale * 2**-7, (case, diff, scale)
        dx_diff = max(dx_diff, diff)
        _check_against_references(run, inp, entry="logprob", ceiling=True)
    dw_rel = float(rel_l2(nd["dW"].float(), plain["dW"].float()))
    print(
        f"\nzero-dlogp raised dX slices {case} on {record['arch']}: slices {skip_plan.dx_slices} (skip off "
        f"{shipped.dx_slices}), zero {int(zero.sum())}/{int(valid.sum())}: dW rel-L2 vs no skip {dw_rel:.2e}, "
        f"dX max|skip - no skip| {dx_diff:.2e}"
    )


def _require_live_cast():
    """Skip unless this device's record registers the skip and the cast at the last live chunk (sm_100a, sm_103a,
    sm_107a)."""
    _require_zero_dlogp_skip()
    module_name, _record = cake_backend.record_for(CUDA)
    if not _lc_registered(module_name):
        pytest.skip(
            "this device's record does not register the cast at the last live chunk (_lc forms)"
        )
    return module_name


@pytest.mark.parametrize(
    "kind, dw_stream",
    [
        (("random", 0.5), "0"),  # trailing chunks without a live row
        (("random", 0.5), "1"),  # the same on the dW side stream
        ("last", "auto"),  # only the first chunk after the partition holds live rows
        ("single", "1"),  # one live row
        (
            "none",
            "0",
        ),  # every chunk live: the accumulates add, the deferred cast writes
        (
            "all",
            "auto",
        ),  # no live row (the partition keeps one row live in the first chunk)
        (("group", 0.166), "1"),
    ],
)
def test_device_zero_dlogp_live_cast(
    logprob_chunk_loop, glm_weight, monkeypatch, kind, dw_stream
):
    """The cast at the last live chunk on the eager skip path: on a record with the ``_lc`` forms a backward of two
    chunks or more runs every accumulate but the last chunk's and the deferred cast in their ``_lc`` forms, and its
    ``dW`` is bitwise that of the counted forms (the same FP32 operations: the tile added into the sum with the
    accumulate's flush to zero, then the scale and the bf16 cast) -- with trailing chunks of no live row (no cast-only
    pass), with only the first chunk live, with every chunk live, on the dW side stream and without it; ``dX`` within
    the unordered K-slice reduction of the counted run and exact zeros on the zero rows; the public runner keeps the
    counted forms."""
    module_name = _require_live_cast()
    base = _device_inputs(13500, "ce", W=glm_weight)
    zero = _zero_dlogp_set(base, kind)
    inp = _with_zero_dlogp(base, zero)
    kw = dict(chunk_size=4096, dw_stream=dw_stream)
    fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=4096)
    torch.cuda.synchronize()
    assert fr.row_index is not None

    def backward():
        cake_backend.BINDING_CACHE.clear()
        dx, dw = cake_backend.backward_logprob(
            inp.X,
            inp.W,
            inp.labels,
            fr.lse,
            inp.dlogp.clone(),
            row_index=fr.row_index,
            **kw,
        )
        torch.cuda.synchronize()
        ((plan, keys),) = _logprob_bwd_plans()
        return dx, dw, plan, [k[0] for k in keys]

    with _zero_dlogp_floor(0):
        dx_lc, dw_lc, plan_lc, run_lc = backward()
        with _without_live_cast(monkeypatch):
            dx_c, dw_c, plan_c, run_c = backward()
    assert plan_lc.dcnt and plan_c.dcnt and plan_lc.num_chunks >= 2
    assert plan_lc.live_cast and not plan_c.live_cast
    assert _lc_registered(module_name)
    # the same launches, the weight-gradient GEMMs in their ``_lc`` forms
    assert [s.replace("_lc_dcnt", "_dcnt") for s in run_lc] == run_c
    assert sum(s.endswith("_lc_dcnt") for s in run_lc) == plan_lc.num_chunks
    assert not any(s.endswith("_lc_dcnt") for s in run_c)
    assert _same_bits(dw_lc, dw_c)
    assert bool(torch.isfinite(dw_lc.float()).all())
    valid = inp.labels >= 0
    live = valid & ~zero
    assert not bool(_bits(dx_lc[~live]).any())
    if bool(live.any()):
        scale = float(dx_c[live].float().abs().max())
        diff = float((dx_lc[live].float() - dx_c[live].float()).abs().max())
        assert diff <= scale * 2**-7, (diff, scale)
    else:
        assert not bool(dw_lc.float().any())
    # the public runner of the same call keeps the counted forms (its ``dw_acc`` holds the sum before the last chunk)
    rows = int(fr.row_index.numel())
    with _torch_deterministic(False):
        runner = cake_backend.prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=4096,
            entry="logprob",
            lse=fr.lse,
            compact_rows=fr.row_index,
            logprob_dz=True,
            dcnt=torch.empty((-(-rows // 4096),), dtype=torch.int32, device=CUDA),
        )
    assert runner.plan.dcnt and not runner.plan.live_cast
    assert not any(s.endswith("_lc_dcnt") for s in runner.plan.stages)


def test_device_zero_dlogp_live_cast_temperature(glm_weight, monkeypatch):
    """The temperature backward under the skip takes the cast at the last live chunk too (its own plan search: the
    ``_lc`` forms first, then the counted forms), with ``dW`` bitwise that of its counted forms."""
    module_name = _require_live_cast()
    _require_tau_program()
    base = _tau_inputs(13500, W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.5))
    inp = dataclasses.replace(_with_zero_dlogp(base, zero), temperature=0.7)
    stages = cake_jit.registered_stages_with_temperature(module_name)
    assert all(s in stages for s in _LC_STAGES)
    calls = {}
    with _zero_dlogp_floor(0):
        cake_backend.BINDING_CACHE.clear()
        calls["lc"] = _run(inp, 4096, entry="logprob", backend="cake", **_TAU_ONE)
        lc_plans = [p for p, _k in _logprob_bwd_plans()]
        with _without_live_cast(monkeypatch):
            cake_backend.BINDING_CACHE.clear()
            calls["counted"] = _run(
                inp, 4096, entry="logprob", backend="cake", **_TAU_ONE
            )
            counted_plans = [p for p, _k in _logprob_bwd_plans()]
    assert lc_plans and all(p.temperature and p.dcnt and p.live_cast for p in lc_plans)
    assert counted_plans and all(p.dcnt and not p.live_cast for p in counted_plans)
    assert _same_bits(calls["lc"]["dW"], calls["counted"]["dW"])
    assert _same_bits(calls["lc"]["logp"], calls["counted"]["logp"])


def test_device_zero_dlogp_generic_floor_is_one_row_above_a_chunk(
    glm_weight, monkeypatch
):
    """The generic floor (``ZERO_DLOGP_MIN_ROWS_DEFAULT``) is one row above a default 4096-row chunk: a backward of 4096
    valid rows runs without the skip, one of 4097 valid rows runs it (the zero rows' ``dX`` zeros either way)."""
    _require_zero_dlogp_skip()
    assert (
        cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT
        == cake_backend.DEFAULT_CHUNK_SIZE + 1
        == 4097
    )
    partition = cake_backend.zero_dlogp_partition
    calls = []

    def counted(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return partition(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counted)
    with _zero_dlogp_floor(cake_backend.ZERO_DLOGP_MIN_ROWS_DEFAULT):
        for T in (4097, 4098):
            base = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
            labels = base.labels.clone()
            # one ignored row: a compacted call of T - 1 valid rows
            labels[T // 2] = -100
            base = dataclasses.replace(base, labels=labels)
            zero = _zero_dlogp_set(base, ("random", 0.3))
            run = _logprob_call(_with_zero_dlogp(base, zero, negative_zero=False))
            assert not bool(run["dX"][zero].float().any()), T
    assert calls == [4097]


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


def test_device_zero_dlogp_skip_adds_no_synchronization(
    logprob_backward_form, glm_weight
):
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
        logp = cake_backend.cake_lm_head_nondet_train_logprob(
            X, W, inp.labels, chunk_size=4096, backend="cake"
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


def test_device_zero_dlogp_backward_cuda_graph(logprob_backward_form, glm_weight):
    """The log-probability backward with the skip captured in a CUDA graph (after a warm-up that binds it on the
    capture stream): replays with other ``dlogp`` values -- other zero rows and live counts per chunk, none and every
    row zero -- written into the captured input give the eager backward's results on those values (``dW`` within
    the FP32 reduction order, the zero rows' ``dX`` exact ``+0``)."""
    _require_program(entry="logprob")
    base = _device_inputs(12289, "ce", W=glm_weight)
    kw = dict(chunk_size=4096)
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


# ---------------------------------------------------------------------------
# Temperature of the log-probability entry (``*_temperature``; the deterministic suite's temperature tests on this
# package: ``logp`` / ``dW`` keep the deterministic package's bits, ``dX`` of a chunk of more than one slice comes from
# the unordered K-slice reduction)
# ---------------------------------------------------------------------------

# The keywords of the log-probability entries' temperature forms (``*_temperature``).
TEMPERATURE_KEYWORDS = frozenset({"temperature", "check_temperature"})
# One dX slice per chunk: the plain dX store, so a test of the temperature's plumbing compares ``dX`` bitwise too
# (``_region_entry``'s pin); the default slices are compared by :func:`_assert_tau_close`.
_TAU_ONE = dict(gemm_tuning={"dx": {"k_slices": 1}})
_KERNEL_SOURCE_RE = re.compile(
    r"^csrc/cake_lm_head_loss/cake_lm_head_loss_([0-9a-f]{20})_kernel\.cu$"
)
_PROGRAM_KERNEL_DEFINITION_RE = re.compile(
    r"^-DCAKE_PROGRAM_KERNEL=(kernel_cake_lm_head_loss_[0-9a-f]{20})$"
)
_PROGRAM_KERNEL_DEFAULT_RE = re.compile(
    r"^#ifndef CAKE_PROGRAM_KERNEL\n"
    r"#define CAKE_PROGRAM_KERNEL (kernel_cake_lm_head_loss_[0-9a-f]{20})\n"
    r"#endif$",
    re.MULTILINE,
)


def _check_dtypes(result, inp, *, grad_weight_dtype=torch.bfloat16):
    assert result["loss"].dtype == torch.float32 and result["loss"].shape == ()
    assert result["logp"].dtype == torch.float32 and tuple(result["logp"].shape) == (
        inp.T,
    )
    if result["dX"] is not None:
        assert result["dX"].dtype == torch.bfloat16 and tuple(result["dX"].shape) == (
            inp.T,
            inp.H,
        )
    if result["dW"] is not None:
        assert result["dW"].dtype == grad_weight_dtype and tuple(
            result["dW"].shape
        ) == (inp.V, inp.H)


def _kernel_source_symbol(entry, package):
    """The unit whose kernel translation unit a library compiles and the kernel symbol
    that source defines for it: the library's own unit spelling its symbol literally, or a
    shared unit whose source defaults ``CAKE_PROGRAM_KERNEL`` to its own symbol and takes
    a sibling form's symbol from the library's compile definitions."""
    kernel_unit = _KERNEL_SOURCE_RE.match(entry["sources"][0]).group(1)
    text = (package / entry["sources"][0]).read_text()
    own = f"kernel_cake_lm_head_loss_{kernel_unit}"
    defaults = _PROGRAM_KERNEL_DEFAULT_RE.findall(text)
    carried = [
        m.group(1)
        for m in map(_PROGRAM_KERNEL_DEFINITION_RE.match, entry["nvcc_flags"])
        if m is not None
    ]
    assert len(defaults) <= 1 and len(carried) <= 1, entry["name"]
    if not defaults:  # a unit of its own: the source spells its symbol
        assert not carried and own in text, entry["name"]
        return kernel_unit, own
    assert defaults[0] == own and "CAKE_PROGRAM_KERNEL(" in text, entry["name"]
    return kernel_unit, carried[0] if carried else own


def _valid_call(T=4):
    X = torch.zeros(T, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(T, dtype=torch.int64)
    return dict(X=X, W=W, labels=labels, objective="ce", loss_div=1.0)


def _problem(T, H, V, C, objective="ce", entry="loss", dtype=torch.bfloat16):
    return cake_backend.Problem(
        num_rows=T,
        hidden=H,
        vocab=V,
        chunk=C,
        ld_x=H,
        x_copy=False,
        objective=objective,
        loss_div=1.0 if objective == "ce" else None,
        grad_weight_dtype=dtype,
        entry=entry,
    )


_ROW_GRAD_PLAN = [
    ["buffer", "z"],
    ["buffer", "labels"],
    ["buffer", "lse"],
    ["buffer", "d"],
    ["parameter", "row0"],
    ["parameter", "d_off"],
    ["parameter", "V"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]


def _fake_record(arg_plan):
    return {
        "arch": "sm_100a",
        "abi": cake_backend.ABI_CONTRACT,
        "stages": ["row_grad"],
        "row_grad": {
            "module": "fake",
            "library": "cake_lm_head_loss_fake",
            "ffi_entry": "run",
            "arg_plan": arg_plan,
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["rows_c", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
    }


def _row_grad_values():
    return dict(
        z=torch.zeros(4, V_HOST, dtype=torch.bfloat16),
        labels=torch.zeros(4, dtype=torch.int64),
        lse=torch.zeros(4),
        d=torch.zeros(4),
        row0=0,
        d_off=0,
        V=V_HOST,
        rows_c=4,
        T=4,
        workspace=None,
        tma_descriptor_workspace=None,
    )


def _assert_same_bits(a, b, keys=("loss", "logp", "dX", "dW")):
    for key in keys:
        x, y = a[key], b[key]
        assert (x is None) == (y is None), key
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape, key
            assert torch.equal(_bits(x), _bits(y)), key


def _assert_tau_close(a, b, case=None, keys=("logp", "dX", "dW")):
    """Two calls of the same problem on this package: ``logp`` / ``dW`` (and ``loss``) bitwise; ``dX`` bitwise when
    both are, else the unordered K-slice reduction's distance -- the same zero rows and a relative L2 distance within
    the dX budget floor (``GATE_TINY["dX_rel_l2"]``); max |a - b| and the BF16 flip rate are printed."""
    for key in keys:
        x, y = a[key], b[key]
        assert (x is None) == (y is None), (case, key)
        if x is None:
            continue
        assert x.dtype == y.dtype and x.shape == y.shape, (case, key)
        if key != "dX" or torch.equal(_bits(x), _bits(y)):
            assert torch.equal(_bits(x), _bits(y)), (case, key)
            continue
        assert torch.equal((x == 0).all(dim=-1), (y == 0).all(dim=-1)), (case, key)
        err = rel_l2(x.float(), y.float())
        flips, ulps = _ulp_flips(x, y)
        diff = float((x.float() - y.float()).abs().max())
        print(
            f"\ntemperature dX {case}: max |diff| {diff:.3e}, {flips:.2e} of elements flip, max {ulps} ulp, rel L2 {err:.2e}"
        )
        assert err <= GATE_TINY["dX_rel_l2"], (case, err)


def _require_recompute():
    module_name, record = cake_backend.record_for(CUDA)
    if cake_backend.LOGPROB_DZ_STAGE not in cake_jit.registered_stages(module_name):
        pytest.skip(
            "the registered program has no dz arm of the logits GEMM (no recompute forms)"
        )
    return opt_in.dw_block_step(cake_backend.Geometry.from_record(record))


def _budget_forms(step: int, V: int = DEFAULT_V) -> tuple:
    """``(form, dw_rows, block_rows)`` of every budget form at vocabulary ``V``: several blocks with a short last one,
    one granule recomputed, a whole-vocabulary block."""
    g = V // step
    return (
        ("fused", V, None),
        ("unfused", V, None),
        ("partial", (g // 2) * step, (g // 4 + 1) * step),
        ("partial", (g - 1) * step, step),
        ("recompute", 0, (g // 3 + 1) * step),
        ("recompute", 0, g * step),
    )


def _force_budget_plan(monkeypatch, form, calls: list) -> None:
    """Route ``memory_mode="budget"`` to the form ``(form, dw_rows, block_rows)`` instead of the rule's choice."""
    plan = opt_in.MemoryPlan(*form, predicted_bytes=0, budget_bytes=0)

    def forced(entry, device, **kw):
        calls.append(entry)
        return plan

    monkeypatch.setattr(opt_in, "_budget_plan", forced)


# Host values every stage may name (the kernels' own argument names, see ``stage_values``).
_GEMM_VALUES = {"A", "B", "C", "STATS_OUT", "WS", *cake_backend.GEMM_SCALARS}
_STAGE_VALUES = {
    stage: set(tensors)
    | set(cake_backend.COMMON_TENSORS)
    | set(cake_backend.COMMON_SCALARS)
    | {"num_vecs"}
    | (_GEMM_VALUES if stage.startswith("gemm") else set())
    | (
        {"ld_x_words", "row_vecs", "num_rows"}
        if stage == cake_backend.HIDDEN_COUNT_STAGE
        else set()
    )
    for stage, tensors in cake_backend.STAGE_TENSORS.items()
}


def test_registry_temperature_forms_are_well_formed():
    """The temperature forms (:func:`cake_jit.registered_stages_with_temperature`; outside ``STAGES``) pass the
    deterministic suite's per-stage registry checks: every record (``sm_107a`` too) registers the five (the ``dz``
    arm's counted form among them), each with a ``build.json`` entry of its own, an argument plan of its twin's host
    values plus ``temperature``, kernel and binding sources that exist and name its symbol, and its twin's cluster in
    the geometry a call with a temperature binds with (``_TemperatureGeometry``, which declares every other stage's
    cluster as :class:`Geometry` does)."""
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    package = Path(cake_jit.__file__).resolve().parent
    twin = {
        "gemm_logits_mcnt_tau": "gemm_logits_mcnt",
        "gemm_logits_dz_tau": "gemm_logits_dz",
        "gemm_logits_dz_dcnt_tau": "gemm_logits_dz_dcnt",
        "gemm_logits_lbl_tau": "gemm_logits_lbl",
        "row_finalize_tau": "row_finalize",
    }
    for name, record in cake_jit.MODULES.items():
        stages = cake_jit.registered_stages_with_temperature(name)
        assert set(TAU_STAGES) <= set(stages) <= set(cake_jit.STAGES_WITH_TEMPERATURE)
        geometry = cake_backend.Geometry.from_record(record)
        bound = cake_backend._TemperatureGeometry.of(geometry)
        for stage in TAU_STAGES:
            physical = record[stage]
            assert physical["module"] and physical["ffi_entry"]
            unit = physical["module"].removeprefix("cake_lm_head_loss_")
            assert physical["library"] == cake_jit.library_stem(unit)
            entry = libraries[cake_jit.library_name(name, stage)]
            assert entry["arch"] == record["arch"]
            assert entry["path"] == f"aot/{entry['name']}.so"
            assert entry["entry"] == physical["ffi_entry"]
            assert len(entry["closure_sha256"]) == 64
            assert len(entry["sources"]) == 2
            assert set(entry["sha256"]) == set(entry["sources"])
            names = [n for _, n in physical["arg_plan"]]
            assert len(names) == len(set(names)) and "temperature" in names
            allowed = _STAGE_VALUES[twin[stage]] | {"temperature"}
            for kind, arg in physical["arg_plan"]:
                assert kind in {
                    "buffer",
                    "tma_buffer",
                    "workspace",
                    "parameter",
                    "grid",
                }
                if kind == "grid":
                    assert arg in ("grid_x", "grid_y", "grid_z")
                else:
                    assert cake_backend.CONTRACT_ALIASES.get(arg, arg) in allowed, (
                        stage,
                        arg,
                    )
            assert int(physical.get("tma_workspace_bytes", 0)) >= 0
            assert int(physical.get("workspace_bytes", 0)) >= 0
            kernel, binding = (s.removeprefix("csrc/") for s in entry["sources"])
            kernel_unit, defined = _kernel_source_symbol(entry, package)
            assert (
                kernel == f"cake_lm_head_loss/cake_lm_head_loss_{kernel_unit}_kernel.cu"
            )
            assert defined == f"kernel_{physical['module']}"
            assert (package / "csrc" / kernel).is_file()
            assert (package / "csrc" / binding).is_file()
            assert bound.cluster_ctas_of(stage) == geometry.cluster_ctas_of(twin[stage])
        for stage in cake_jit.STAGES:
            assert bound.cluster_ctas_of(stage) == geometry.cluster_ctas_of(stage)


TAU_STAGES = (
    "gemm_logits_mcnt_tau",
    "gemm_logits_dz_tau",
    "gemm_logits_dz_dcnt_tau",
    "gemm_logits_lbl_tau",
    "row_finalize_tau",
)
# the stages a temperature call never binds on the generated program (their temperature forms replace them; under
# the zero-gradient row skip the dz arm runs a temperature form, counted or not, never the counted form without one)
_UNTAU_STAGES = frozenset(
    (
        "gemm_logits",
        "gemm_logits_mcnt",
        "gemm_logits_nostats",
        "gemm_logits_nostats_mcnt",
        "gemm_logits_dz",
        "gemm_logits_dz_dcnt",
        "gemm_logits_lbl",
        "row_finalize",
        "row_grad",
    )
)
_LOGPROB_KEYS = ("logp", "dX", "dW")


def _tau_with(inp, value, *, ignored):
    """The heterogeneous temperature of ``inp``'s problem with ``value`` at its first valid (``ignored=False``) or
    first ignored row."""
    tau = make_temperature(inp.T, "hetero", seed=inp.seed, device=inp.X.device).clone()
    rows = (~inp.valid if ignored else inp.valid).nonzero().reshape(-1)
    tau[int(rows[0])] = value
    return tau


def _cuda_temperature():
    if not torch.cuda.is_available():
        pytest.skip("a CUDA scalar needs a CUDA device")
    return torch.tensor(0.7, device="cuda")


def _other_device():
    return "cpu" if DEV.type == "cuda" else "meta"


# ``case -> (kind, make(inp))``: "host" -- rejected from the host values alone (type, shape, dtype, device,
# ``requires_grad``, a scalar's value); "value" -- a per-token tensor's values (read directly on the host; on a CUDA
# device a device assertion, so these run with ``check_temperature=True`` there)
_TAU_REJECTS = {
    "zero": ("host", lambda inp: 0.0),
    "negative": ("host", lambda inp: -1.0),
    "nan": ("host", lambda inp: float("nan")),
    "inf": ("host", lambda inp: float("inf")),
    "negative_inf": ("host", lambda inp: float("-inf")),
    "fp32_underflow": ("host", lambda inp: 1e-50),
    "fp32_overflow": ("host", lambda inp: 1e39),
    "int_overflow": ("host", lambda inp: 10**400),
    "bool": ("host", lambda inp: True),
    "int_zero": ("host", lambda inp: 0),
    "string": ("host", lambda inp: "0.7"),
    "list": ("host", lambda inp: [0.7] * inp.T),
    "zero_dim_zero": ("host", lambda inp: torch.tensor(0.0)),
    "zero_dim_nan": ("host", lambda inp: torch.tensor(float("nan"))),
    "zero_dim_requires_grad": (
        "host",
        lambda inp: torch.tensor(0.7, requires_grad=True),
    ),
    "zero_dim_cuda": ("host", lambda inp: _cuda_temperature()),
    "fp64": (
        "host",
        lambda inp: torch.full((inp.T,), 0.7, dtype=torch.float64, device=DEV),
    ),
    "bf16": (
        "host",
        lambda inp: torch.full((inp.T,), 0.7, dtype=torch.bfloat16, device=DEV),
    ),
    "int64": ("host", lambda inp: torch.ones(inp.T, dtype=torch.int64, device=DEV)),
    "column": ("host", lambda inp: torch.full((inp.T, 1), 0.7, device=DEV)),
    "longer": ("host", lambda inp: torch.full((inp.T + 1,), 0.7, device=DEV)),
    "shorter": ("host", lambda inp: torch.full((inp.T - 1,), 0.7, device=DEV)),
    "wrong_device": (
        "host",
        lambda inp: torch.full((inp.T,), 0.7, device=_other_device()),
    ),
    "requires_grad": (
        "host",
        lambda inp: torch.full((inp.T,), 0.7, device=DEV, requires_grad=True),
    ),
    "zero_at_valid_row": ("value", lambda inp: _tau_with(inp, 0.0, ignored=False)),
    "negative_zero_at_valid_row": (
        "value",
        lambda inp: _tau_with(inp, -0.0, ignored=False),
    ),
    "negative_at_valid_row": (
        "value",
        lambda inp: _tau_with(inp, -0.5, ignored=False),
    ),
    "nan_at_valid_row": (
        "value",
        lambda inp: _tau_with(inp, float("nan"), ignored=False),
    ),
    "inf_at_valid_row": (
        "value",
        lambda inp: _tau_with(inp, float("inf"), ignored=False),
    ),
    # the issue: nonpositive / nonfinite temperatures are errors, ignored rows included
    "zero_at_ignored_row": ("value", lambda inp: _tau_with(inp, 0.0, ignored=True)),
    "nan_at_ignored_row": (
        "value",
        lambda inp: _tau_with(inp, float("nan"), ignored=True),
    ),
}


@pytest.mark.parametrize("case", list(_TAU_REJECTS))
def test_temperature_rejects(case):
    """Every invalid ``temperature`` is a ``ValueError`` naming it, from the autograd entry and the eager forward,
    with and without ``check_temperature`` (a per-token tensor's values on a CUDA device: with it; without it they are
    a device assertion, ``test_device_temperature_assertion_stops_the_stream``): a nonpositive / nonfinite value,
    one that rounds to FP32 zero or infinity, an int past the float range, a bool, a non-number, a 0-dim CUDA scalar
    (reading it would synchronize), a tensor that is not FP32 ``[T]`` on the inputs' device or that requires a
    gradient (the temperature is not differentiable), a nonpositive / nonfinite entry at a valid or an ignored row."""
    kind, make = _TAU_REJECTS[case]
    inp = _host_inputs(37, ignore_frac=0.2)
    value = make(inp)
    for check in (False, True) if kind == "host" or DEV.type != "cuda" else (True,):
        with pytest.raises(ValueError, match="temperature"):
            cake_backend.cake_lm_head_nondet_train_logprob_temperature(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                backend="reference",
                temperature=value,
                check_temperature=check,
            )
        with pytest.raises(ValueError, match="temperature"):
            cake_backend.forward_logprob_temperature(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                backend="reference",
                temperature=value,
                check_temperature=check,
            )


def test_temperature_accepts_numbers_and_strided_tensors():
    """A Python number, a 0-dim CPU tensor (any float dtype: rounded once to FP32) and a non-contiguous FP32 ``[T]``
    tensor (a stride-0 expanded scalar, a stride-2 view) give the bits of the contiguous constant tensor; a Python
    int is that number."""
    inp = _host_inputs(37, ignore_frac=0.2)
    s = 0.7
    full = _run(
        inp,
        16,
        entry="logprob",
        temperature=torch.full((37,), s, dtype=torch.float32, device=DEV),
    )
    for value in (
        s,
        torch.tensor(s),
        torch.tensor(s, dtype=torch.float64),
        torch.tensor(s, dtype=torch.float32, device=DEV).expand(37),
        torch.full((74,), s, dtype=torch.float32, device=DEV)[::2],
    ):
        _assert_same_bits(
            full, _run(inp, 16, entry="logprob", temperature=value), _LOGPROB_KEYS
        )
    _assert_same_bits(
        _run(inp, 16, entry="logprob", temperature=2),
        _run(inp, 16, entry="logprob", temperature=2.0),
        _LOGPROB_KEYS,
    )


@pytest.mark.parametrize("compact", [True, False])
def test_temperature_one_runs_the_default_path(compact, monkeypatch):
    """``None``, ``1.0``, ``1``, ``tensor(1.0)`` and a value that rounds to FP32 1.0 run the call without a
    temperature -- the temperature-rows helper is never called -- and a ones tensor runs the temperature path; all of
    them give the bits of the call without the keyword, with and without ``check_temperature``."""
    inp = _host_inputs(37, ignore_frac=0.2)
    base = _run(inp, 16, entry="logprob", compact_rows=compact)
    made: list = []
    real = cake_backend._temperature_rows

    def spy(*args, **kwargs):
        made.append(args)  # every call, also one that would return None
        return real(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_temperature_rows", spy)
    for value in (None, 1.0, 1, torch.tensor(1.0), 1.0 + 1e-9):
        made.clear()
        for check in (False, True):
            result = _run(
                inp,
                16,
                entry="logprob",
                compact_rows=compact,
                temperature=value,
                check_temperature=check,
            )
            _assert_same_bits(base, result, _LOGPROB_KEYS)
        assert not made, value
    ones = _run(
        inp,
        16,
        entry="logprob",
        compact_rows=compact,
        temperature=torch.ones(37, dtype=torch.float32, device=DEV),
    )
    assert made
    _assert_same_bits(base, ones, _LOGPROB_KEYS)


@pytest.mark.parametrize("s", [0.7, 2.0])
def test_temperature_scalar_equals_the_constant_tensor(s):
    """A scalar temperature gives the bits of the constant FP32 tensor of its value (the same values reach the same
    stages), differs from the call without one and passes the accuracy gates of its references."""
    inp = _host_inputs(37, ignore_frac=0.2)
    scalar = _run(inp, 16, entry="logprob", temperature=s)
    tensor = _run(
        inp,
        16,
        entry="logprob",
        temperature=torch.full((37,), s, dtype=torch.float32, device=DEV),
    )
    _assert_same_bits(scalar, tensor, _LOGPROB_KEYS)
    plain = _run(inp, 16, entry="logprob")
    assert not torch.equal(scalar["logp"], plain["logp"])
    _check_against_references(
        scalar, dataclasses.replace(inp, temperature=s), entry="logprob"
    )


@pytest.mark.parametrize(
    "T, C, frac",
    [
        (1, 256, 0.0),
        (255, 256, 0.05),
        (255, 512, 0.5),
        (513, 256, 0.5),
        (513, 512, 0.05),
        (1031, 256, 0.05),
        (1031, 512, "all_but_one"),
    ],
)
def test_temperature_heterogeneous_matches_the_references(T, C, frac):
    """A heterogeneous per-token temperature (log-uniform in [0.25, 4], some rows exactly 1.0) on irregular ``T``,
    with no / some / half / all but one of the rows ignored: ``logp`` / ``dX`` / ``dW`` within 1.05x the unchunked
    PyTorch path's error against the FP64 oracle (both with the temperature) plus the floor, zeros on ignored rows."""
    ignore = (T - 1) / T if frac == "all_but_one" else float(frac)
    inp = _host_inputs(T, ignore_frac=ignore, temperature="hetero")
    result = _run(inp, C, entry="logprob")
    _check_dtypes(result, inp)
    _check_against_references(result, inp, entry="logprob")
    assert torch.all(result["logp"][~inp.valid] == 0)
    assert torch.all(result["dX"][~inp.valid] == 0)


@pytest.mark.parametrize("frac", ["none", "five_percent", "half", "all_but_one"])
def test_temperature_compaction_matches_uncompacted(frac):
    """The temperature follows its rows through the compaction (as ``test_compaction_matches_uncompacted``): the
    compacted and uncompacted calls with a heterogeneous temperature are bitwise when no row is ignored and at the
    host's precision level otherwise; the caller's count gives the bits of the counted compacted call; a temperature
    moved by one row breaks the comparison (the check is not vacuous)."""
    T, C = 37, 16
    ignore = {
        "none": 0.0,
        "five_percent": 0.05,
        "half": 0.5,
        "all_but_one": (T - 1) / T,
    }[frac]
    inp = _host_inputs(T, ignore_frac=ignore, temperature="hetero")
    num_ignored = int((~inp.valid).sum())
    compact = _run(inp, C, entry="logprob", compact_rows=True)
    plain = _run(inp, C, entry="logprob", compact_rows=False)
    for result in (compact, plain):
        _check_against_references(result, inp, entry="logprob")
        assert torch.all(result["logp"][~inp.valid] == 0)
        assert torch.all(result["dX"][~inp.valid] == 0)
    if num_ignored == 0:
        for key in _LOGPROB_KEYS:
            assert torch.equal(compact[key], plain[key]), key
    else:
        assert torch.allclose(compact["logp"], plain["logp"], rtol=1e-5, atol=2e-5)
        assert rel_l2(compact["dX"].float(), plain["dX"].float()) <= 1e-3
        assert rel_l2(compact["dW"].float(), plain["dW"].float()) <= 1e-3
    given = _run(
        inp, C, entry="logprob", compact_rows=True, num_valid_rows=T - num_ignored
    )
    _assert_same_bits(compact, given, _LOGPROB_KEYS)
    shifted = _run(
        dataclasses.replace(inp, temperature=torch.roll(inp.temperature, 1)),
        C,
        entry="logprob",
        compact_rows=True,
    )
    assert (shifted["logp"] - compact["logp"])[inp.valid].abs().max() > 1e-3


def test_temperature_upstream_gradients():
    """Arbitrary per-row upstream gradients with a heterogeneous temperature: random (nonzero on ignored rows too),
    zeros, negated, large -- each within the gates of its references; zeros give zero gradients, the negated upstream
    the negated gradients and twice the upstream twice the gradients (exact operations), and a retained graph
    repeats the backward bitwise."""
    inp = _host_inputs(65, ignore_frac=0.1, temperature="hetero")
    base = _run(inp, 32, entry="logprob")
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED + 7637)
    upstream = dict(
        random=torch.randn(65, generator=gen, device=DEV),
        zeros=torch.zeros(65, device=DEV),
        negated=-inp.dlogp,
        large=inp.dlogp * 1e4,
    )
    results = {}
    for name, g in upstream.items():
        g_inp = dataclasses.replace(inp, dlogp=g.contiguous())
        results[name] = _run(g_inp, 32, entry="logprob")
        _check_against_references(results[name], g_inp, entry="logprob")
        assert torch.equal(results[name]["logp"], base["logp"]), name
    assert torch.all(results["zeros"]["dX"] == 0) and torch.all(
        results["zeros"]["dW"] == 0
    )
    assert torch.equal(results["negated"]["dX"], -base["dX"])
    assert torch.equal(results["negated"]["dW"], -base["dW"])
    twice = _run(inp, 32, entry="logprob", scale=2.0)
    assert torch.equal(twice["dX"], (2.0 * base["dX"].float()).to(torch.bfloat16))
    assert torch.equal(twice["dW"], (2.0 * base["dW"].float()).to(torch.bfloat16))
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
        X,
        W,
        inp.labels,
        chunk_size=32,
        backend="reference",
        temperature=inp.temperature,
    )
    first = torch.autograd.grad(logp, (X, W), inp.dlogp, retain_graph=True)
    second = torch.autograd.grad(logp, (X, W), inp.dlogp)
    for a, b, ref in zip(first, second, (base["dX"], base["dW"]), strict=True):
        assert torch.equal(a, b) and torch.equal(a, ref)


def test_temperature_frozen_inputs():
    """Frozen ``X``, frozen ``W`` or both with a heterogeneous temperature: the one-sided gradients equal the
    two-sided ones, ``logp`` is unchanged."""
    inp = _host_inputs(37, ignore_frac=0.1, temperature="hetero")
    both = _run(inp, 16, entry="logprob")
    x_only = _run(inp, 16, entry="logprob", train_w=False)
    w_only = _run(inp, 16, entry="logprob", train_x=False)
    neither = _run(inp, 16, entry="logprob", train_x=False, train_w=False)
    assert x_only["dW"] is None and w_only["dX"] is None
    assert neither["dX"] is None and neither["dW"] is None
    assert torch.equal(x_only["dX"], both["dX"])
    assert torch.equal(w_only["dW"], both["dW"])
    for r in (x_only, w_only, neither):
        assert torch.equal(r["logp"], both["logp"])


def test_temperature_keywords():
    """``temperature`` and ``check_temperature`` are keyword-only arguments of the log-probability temperature
    entries (the default resolves to no temperature; ``check_temperature`` defaults to ``False``) and of no entry
    without the suffix (the log-probability entries keep their signatures) nor any loss entry; the checked call of a
    valid temperature gives the bits of the unchecked one."""
    for fn in (
        cake_backend.cake_lm_head_nondet_train_logprob_temperature,
        cake_backend.forward_logprob_temperature,
    ):
        params = inspect.signature(fn).parameters
        for name in ("temperature", "check_temperature"):
            assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (
                fn.__name__,
                name,
            )
        assert params["check_temperature"].default is False, fn.__name__
        assert (
            cake_backend._resolve_temperature(
                params["temperature"].default, 4, torch.device("cpu")
            )
            is None
        ), fn.__name__
    for fn in (
        cake_backend.cake_lm_head_nondet_train_loss,
        cake_backend.forward_loss,
        cake_backend.backward_loss,
        cake_backend.cake_lm_head_nondet_train_logprob,
        cake_backend.forward_logprob,
        cake_backend.backward_logprob,
    ):
        params = inspect.signature(fn).parameters
        assert "temperature" not in params and "check_temperature" not in params
    inp = _host_inputs(37, ignore_frac=0.2, temperature="hetero")
    _assert_same_bits(
        _run(inp, 16, entry="logprob"),
        _run(inp, 16, entry="logprob", check_temperature=True),
        _LOGPROB_KEYS,
    )


def test_temperature_resolution():
    """``_resolve_temperature(temperature, T, device, *, check=False)`` (shared with flashinfer#7636): ``None`` for no
    temperature and every value that rounds to FP32 1.0; ``("scalar", s)`` with ``s`` the FP32 value of a positive
    finite number or CPU scalar; ``("tensor", tau)`` with ``tau`` FP32 ``[T]`` holding the caller's values."""
    resolve = cake_backend._resolve_temperature
    cpu = torch.device("cpu")
    for value in (None, 1.0, 1, torch.tensor(1.0), 1.0 + 1e-9):
        assert resolve(value, 5, cpu) is None, value
    fp32 = float(torch.tensor(0.7, dtype=torch.float32))
    for value in (0.7, torch.tensor(0.7), torch.tensor(0.7, dtype=torch.float64)):
        form, s = resolve(value, 5, cpu)
        assert form == "scalar" and float(s) == fp32, value
    assert resolve(1.0 + 2.0**-23, 5, cpu)[0] == "scalar"  # the next FP32 value above 1
    tau = torch.tensor([0.5, 1.0, 2.0, 3.0, 0.25])
    form, t = resolve(tau, 5, cpu)
    assert form == "tensor" and t.dtype == torch.float32 and tuple(t.shape) == (5,)
    assert torch.equal(t, tau)
    form, t = resolve(torch.tensor(0.7).expand(5), 5, cpu)
    assert form == "tensor" and torch.equal(t, torch.full((5,), 0.7))
    form, t = resolve(torch.empty(0), 0, cpu)
    assert form == "tensor" and t.numel() == 0
    with pytest.raises(ValueError, match="temperature"):
        resolve(torch.zeros(5), 5, cpu, check=True)


def test_temperature_all_ignored_rows(monkeypatch):
    """Every row ignored with a temperature: zeros without a binding or a launch, and the temperature is still
    validated; ``T == 0`` with an empty temperature returns an empty ``logp``."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a call whose every row is ignored must neither bind nor launch"
        )

    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss", refuse)
    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss_temperature", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    inp = _host_inputs(37, ignore_frac=1.0, temperature="hetero")
    cache = cake_backend.BINDING_CACHE
    hits, misses = cache.hits, cache.misses
    for value in (inp.temperature, 0.7):
        result = _run(inp, 16, entry="logprob", compact_rows=True, temperature=value)
        _check_dtypes(result, inp)
        assert torch.all(result["logp"] == 0)
        assert torch.all(result["dX"] == 0) and torch.all(result["dW"] == 0)
    assert (cache.hits, cache.misses) == (hits, misses)
    with pytest.raises(ValueError, match="temperature"):
        _run(
            inp,
            16,
            entry="logprob",
            compact_rows=True,
            temperature=torch.zeros(37, device=DEV),
            check_temperature=True,
        )
    empty = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
        inp.X[:0],
        inp.W,
        inp.labels[:0],
        backend="reference",
        temperature=torch.empty(0, device=DEV),
    )
    assert tuple(empty.shape) == (0,)
    empty = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
        inp.X[:0],
        inp.W,
        inp.labels[:0],
        backend="reference",
        temperature=0.7,
    )
    assert tuple(empty.shape) == (0,)


def test_temperature_reaches_the_reference_engine():
    """The reference backend applies the temperature: a heterogeneous temperature moves ``logp`` / ``dX`` / ``dW``
    well away from the call without one and matches the references with it (a reference engine that ignored the
    temperature would fail both)."""
    inp = _host_inputs(200, temperature="hetero")
    tau = _run(inp, 64, entry="logprob")
    plain = _run(dataclasses.replace(inp, temperature=None), 64, entry="logprob")
    assert (tau["logp"] - plain["logp"])[inp.valid].abs().max() > 1e-2
    assert rel_l2(tau["dX"], plain["dX"]) > 1e-2
    assert rel_l2(tau["dW"], plain["dW"]) > 1e-2
    _check_against_references(tau, inp, entry="logprob")


def test_temperature_through_the_opt_in_fast_entry(monkeypatch):
    """The opt-in log-probability entry without a count in the fast memory mode (``memory_mode=None`` or ``"fast"``)
    hands the temperature to ``cake_backend``'s entry: the graph node is ``cake_backend``'s autograd function,
    ``logp`` / ``dX`` / ``dW`` are the bits of ``cake_backend``'s call with the same temperature (tensor and scalar)
    and differ from the call without one, and ``temperature`` / ``check_temperature`` arrive unchanged (a tensor as
    the same object)."""
    inp = _host_inputs(37, ignore_frac=0.2, temperature="hetero")
    plain = _run(dataclasses.replace(inp, temperature=None), 16, entry="logprob")
    for tau in (inp.temperature, 0.7):
        main = _run(inp, 16, entry="logprob", temperature=tau)
        assert not torch.equal(main["logp"], plain["logp"])
        for mode in (None, "fast"):
            fell = _run(inp, 16, entry="logprob", temperature=tau, memory_mode=mode)
            _assert_same_bits(fell, main, _LOGPROB_KEYS)
            logp = opt_in.cake_lm_head_nondet_train_logprob_temperature(
                inp.X.detach().requires_grad_(),
                inp.W,
                inp.labels,
                chunk_size=16,
                backend="reference",
                memory_mode=mode,
                temperature=tau,
            )
            assert (
                logp.grad_fn._forward_cls
                is cake_backend.CakeLMHeadNondetLogprobTemperatureFunction
            )
    seen = {}

    def spy(*args, **kw):
        seen.update(kw)
        return torch.zeros(())

    monkeypatch.setattr(
        cake_backend, "cake_lm_head_nondet_train_logprob_temperature", spy
    )
    for mode in (None, "fast"):
        for tau, check in ((inp.temperature, True), (0.7, False)):
            seen.clear()
            opt_in.cake_lm_head_nondet_train_logprob_temperature(
                inp.X,
                inp.W,
                inp.labels,
                backend="reference",
                memory_mode=mode,
                temperature=tau,
                check_temperature=check,
            )
            assert seen["temperature"] is tau and seen["check_temperature"] is check


def test_temperature_through_the_opt_in_entry_with_a_count(monkeypatch):
    """The opt-in log-probability entry with a count (``num_valid_rows``) applies its autograd function of the path
    without a temperature for ``temperature=None`` / ``1.0`` -- which resolves no temperature, forward or backward,
    and gives the same bits for both -- and its temperature form otherwise; the default path's binding keys take no
    temperature."""
    inp = _host_inputs(37, ignore_frac=0.2, temperature="hetero")
    n = int(inp.valid.sum())
    real = opt_in._resolve_temperature

    def refuse(*args, **kw):
        raise AssertionError("the path without a temperature resolves no temperature")

    plain = opt_in.CakeLMHeadNondetLogprobFunction
    bits = []
    for tau, cls in (
        (None, plain),
        (1.0, plain),
        (0.7, opt_in.CakeLMHeadNondetLogprobTemperatureFunction),
        (inp.temperature, opt_in.CakeLMHeadNondetLogprobTemperatureFunction),
    ):
        monkeypatch.setattr(
            opt_in, "_resolve_temperature", refuse if cls is plain else real
        )
        X = inp.X.detach().requires_grad_()
        entry = (
            opt_in.cake_lm_head_nondet_train_logprob
            if tau is None
            else opt_in.cake_lm_head_nondet_train_logprob_temperature
        )
        logp = entry(
            X,
            inp.W,
            inp.labels,
            chunk_size=16,
            backend="reference",
            num_valid_rows=n,
            **({} if tau is None else dict(temperature=tau)),
        )
        assert logp.grad_fn._forward_cls is cls, tau
        logp.sum().backward()
        if cls is plain:
            bits.append((_bits(logp.detach()), _bits(X.grad)))
    assert all(torch.equal(a, b) for a, b in zip(*bits, strict=True))
    for key in (opt_in.forward_binding_key, opt_in.logprob_backward_binding_key):
        assert "temperature" not in inspect.signature(key).parameters


def test_device_temperature_through_the_opt_in_entry_with_a_count(
    glm_weight, monkeypatch
):
    """:func:`test_temperature_through_the_opt_in_entry_with_a_count` on the generated program through the opt-in
    module's binding cache: ``temperature=None`` / ``1.0`` take the autograd function without a temperature, resolve
    no temperature, reuse the forward binding of the first call (a backward binding is keyed by the addresses of the
    forward's saved row tensors, so a later call may bind a backward again; no temperature binding remembered) and
    give the same bits; ``0.7`` and per-token rows take the temperature form, which remembers temperature bindings of
    its own.  One dX slice pinned (``_TAU_ONE``): the calls' ``dX`` compare bitwise."""
    _require_tau_program()
    T, C = 5000, 2048
    inp = _device_inputs(T, W=glm_weight)
    n = int(inp.valid.sum())
    rows = make_temperature(T, "hetero", seed=SEED + 31, device=CUDA)
    real = opt_in._resolve_temperature

    def refuse(*args, **kw):
        raise AssertionError("the path without a temperature resolves no temperature")

    plain = opt_in.CakeLMHeadNondetLogprobFunction
    tempered = opt_in.CakeLMHeadNondetLogprobTemperatureFunction
    calls = (
        (None, plain),
        (1.0, plain),
        (None, plain),
        (1.0, plain),
        (0.7, tempered),
        (rows, tempered),
    )
    bits = []
    with _cache(True, ("num_valid_rows",)) as cache:
        cache.clear()
        for i, (tau, cls) in enumerate(calls):
            monkeypatch.setattr(
                opt_in, "_resolve_temperature", refuse if cls is plain else real
            )
            before = set(cache._bindings)
            X = inp.X.detach().requires_grad_()
            entry = (
                opt_in.cake_lm_head_nondet_train_logprob
                if tau is None
                else opt_in.cake_lm_head_nondet_train_logprob_temperature
            )
            logp = entry(
                X,
                inp.W,
                inp.labels,
                chunk_size=C,
                backend="cake",
                num_valid_rows=n,
                **({} if tau is None else dict(temperature=tau)),
                **_TAU_ONE,
            )
            assert logp.grad_fn._forward_cls is cls, tau
            (logp * inp.dlogp)[inp.valid].sum().backward()
            torch.cuda.synchronize()
            remembered = list(cache._bindings.values())
            if cls is plain:
                new = [key for key in cache._bindings if key not in before]
                assert i == 0 or all(key[0] == "bwd" for key in new), (i, tau, new)
                assert remembered and not any(
                    isinstance(b, opt_in._TemperatureBinding) for b in remembered
                )
                bits.append((_bits(logp.detach()), _bits(X.grad)))
            else:
                assert any(
                    isinstance(b, opt_in._TemperatureBinding) for b in remembered
                )
    for other in bits[1:]:
        assert all(torch.equal(a, b) for a, b in zip(bits[0], other, strict=True))


def test_temperature_plan_keys():
    """A log-probability plan of a registered program whose call has a temperature runs the temperature forms in
    the slots of the stages they replace -- the label-column forward, ``row_finalize``, the ``dz`` recompute and, on
    the hidden valid-row count path, chunk 0's device-count GEMM -- with the same chunks and the same other keys; the
    dW side stream forks behind each chunk's ``dz`` temperature form and a schedule whose side launch has no ``dz``
    producer before it in its chunk is refused; the reference plan keeps the names without the suffix; the
    temperature forms need a registered program, a call with a temperature and both logits GEMM arms, and have no
    kept-logits plan; a plan without the zero-gradient row skip has no ``_dcnt`` stage (the skip's mixed plan:
    :func:`test_temperature_plan_runs_the_counted_dx_dw_forms`)."""
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
    arms = dict(logprob_dz=True, logprob_lbl=True)
    problem = _problem(8700, 6144, 154880, 4096, entry="logprob")
    tau_problem = cake_backend._with_temperature(problem)
    plain = cake_backend.make_plan(problem, valid_rows=8265, **arms, **common)
    tau = cake_backend.make_plan_temperature(
        tau_problem, valid_rows=8265, **arms, **common
    )
    assert tau.temperature and not plain.temperature
    assert (
        type(tau) is cake_backend._TemperaturePlan and type(plain) is cake_backend.Plan
    )
    assert tau.without_temperature() == dataclasses.replace(plain, problem=tau_problem)
    assert tau.chunks == plain.chunks and tau.num_chunks == 3
    rename = {
        "gemm_logits_lbl": "gemm_logits_lbl_tau",
        "gemm_logits_dz": "gemm_logits_dz_tau",
        "row_finalize": "row_finalize_tau",
    }
    fwd, bwd = (
        cake_backend._forward_keys_temperature(tau),
        cake_backend._recompute_keys_temperature(tau),
    )
    assert fwd == tuple(
        (rename.get(s, s), i) for s, i in cake_backend.forward_keys(plain)
    )
    assert bwd == tuple(
        (rename.get(s, s), i) for s, i in cake_backend.recompute_keys(plain)
    )
    assert cake_backend.cast_keys(tau) == cake_backend.cast_keys(plain)
    assert set(tau.stages) == {rename.get(s, s) for s in plain.stages}
    assert list(tau.stages) == [
        s for s in cake_jit.STAGES_WITH_TEMPERATURE if s in tau.stages
    ]
    assert not set(tau.stages) & _UNTAU_STAGES
    assert not tau.dcnt and not any(s.endswith("_dcnt") for s in tau.stages)
    with pytest.raises(ValueError, match="temperature forms have no"):
        cake_backend.make_plan_temperature(
            tau_problem, valid_rows=8265, logprob_keep=True, **arms, **common
        )
    assert cake_backend.forward_early_keys(tau)[-1] == ("gemm_logits_lbl_tau", 0)
    _, forks, sides = cake_backend._side_stream_schedule_temperature(tau, bwd, True)
    assert sorted(forks) == [
        i for i, k in enumerate(bwd) if k[0] == "gemm_logits_dz_tau"
    ]
    assert len(sides) == 2 and all(p > min(forks) for p in sides)
    broken = tuple(k for k in bwd if k != ("gemm_logits_dz_tau", 1))
    with pytest.raises(RuntimeError, match="dz"):
        cake_backend._side_stream_schedule_temperature(tau, broken, True)
    hidden = dataclasses.replace(tau, hidden_count=True)
    assert hidden.hidden_logits_stage() == "gemm_logits_mcnt_tau"
    assert "gemm_logits_mcnt_tau" in hidden.stages
    # no untempered stage either: the hidden count's recount backward (chunk 0's ``row_grad``) never runs with one
    assert not set(hidden.stages) & _UNTAU_STAGES
    assert "row_grad" in dataclasses.replace(plain, hidden_count=True).stages
    assert cake_backend._forward_keys_temperature(hidden)[0] == ("row_finalize_tau", 0)
    # the reference path (no registered program): the names without the suffix, the problem carries the temperature
    ref = cake_backend.make_plan(
        tau_problem, need_dx=True, need_dw=True, valid_rows=8265
    )
    assert not ref.temperature and ref.problem.temperature
    assert ("gemm_logits", 0) in cake_backend.forward_keys(ref)
    assert ("row_finalize", 0) in cake_backend.forward_keys(ref)
    assert not any(s.endswith("_tau") for s in ref.stages)
    for bad_problem, kw in (
        (problem, arms),  # the call has no temperature
        (tau_problem, dict(logprob_dz=True)),  # one arm off
        (tau_problem, dict(logprob_lbl=True)),
        (tau_problem, {}),
    ):
        with pytest.raises(ValueError, match="temperature"):
            cake_backend.make_plan_temperature(
                bad_problem, valid_rows=8265, **kw, **common
            )
    with pytest.raises(ValueError, match="temperature"):
        cake_backend.make_plan_temperature(
            tau_problem, need_dx=True, need_dw=True, **arms
        )  # no registered program


@pytest.mark.parametrize("counted_dz", (True, False), ids=("counted_dz", "mixed"))
def test_temperature_plan_runs_the_counted_dx_dw_forms(counted_dz):
    """``make_plan_temperature(dcnt=True)`` (the zero-gradient row skip with a temperature): the K-sliced dX chunks
    and the weight-gradient GEMMs run the counted forms of the skip's plan without a temperature -- the same chunks,
    slices and instance knobs (against the plan without the skip, the dX slices raised per
    ``DX_COUNTED_MIN_SLICES_BY_ARCH``); a short last chunk the TMA cast's counted form, the counted per-thread cast with
    ``counted_tma_cast`` off -- and an explicitly pinned weight-gradient TMA epilogue keeps the plan uncounted.  Every
    chunk's ``dz`` recompute runs the counted temperature form ``gemm_logits_dz_dcnt_tau``
    (:class:`_CountedTemperaturePlan`, the default) or, with ``counted_dz=False`` (the mixed plan), the uncounted
    ``gemm_logits_dz_tau`` -- never ``gemm_logits_dz_dcnt``.  The twin (``without_temperature``) is the skip's plan
    with the counted ``dz`` arm, or the plan with the uncounted one; the launch keys are the counted plan's with the
    temperature forms, and the dW side stream forks behind the ``dz`` temperature form."""
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
        logprob_lbl=True,
    )
    dz_tau = "gemm_logits_dz_dcnt_tau" if counted_dz else "gemm_logits_dz_tau"
    rename = {
        "gemm_logits_dz_dcnt": dz_tau,
        "gemm_logits_lbl": "gemm_logits_lbl_tau",
        "row_finalize": "row_finalize_tau",
    }
    kind = (
        cake_backend._CountedTemperaturePlan
        if counted_dz
        else cake_backend._TemperaturePlan
    )
    counted = 0
    # three chunks (the last of 3808 rows), and the T = 65031 timing row: fifteen chunks of 4096 and one of 339
    for T, rows in ((12000, 12000), (65031, 15 * 4096 + 339)):
        problem = _keep_problem(T, 6144, 154880, 4096, entry="logprob")
        tau_problem = cake_backend._with_temperature(problem)
        for arch in ("sm_100a", "sm_103a", "sm_107a"):
            kw = dict(common, arch=arch, valid_rows=rows)
            skip = cake_backend.make_plan(problem, dcnt=True, **kw)
            shipped = cake_backend.make_plan_temperature(tau_problem, **kw)
            tau = cake_backend.make_plan_temperature(
                tau_problem, dcnt=True, counted_dz=counted_dz, **kw
            )
            assert type(tau) is kind and not shipped.dcnt
            assert type(shipped) is cake_backend._TemperaturePlan
            assert tau.dcnt == skip.dcnt, (T, arch)
            if not tau.dcnt:
                continue
            counted += 1
            assert tau.chunks == skip.chunks == shipped.chunks
            assert tau.dx_slices == skip.dx_slices == _counted_slices(shipped, arch)
            assert tau.variants == skip.variants
            _assert_counted_twin(skip, shipped, arch, (T, arch))
            for i in range(tau.num_chunks):
                assert tau.recompute_stage(i) == dz_tau
                assert shipped.recompute_stage(i) == "gemm_logits_dz_tau"
                assert tau.dx_stage_of(i) == skip.dx_stage_of(i)
                assert tau.dx_stage_of(i).endswith("_dcnt") == (
                    tau.dx_slices_of(i) > 1
                ), tau.dx_stage_of(i)
            assert tau.dw_cast_stage == skip.dw_cast_stage
            assert all(
                s.endswith("_dcnt") for s in tau.stages if s.startswith("gemm_dw")
            )
            assert set(tau.stages) == {rename.get(s, s) for s in skip.stages}
            assert list(tau.stages) == [
                s for s in cake_jit.STAGES_WITH_TEMPERATURE if s in tau.stages
            ]
            assert not set(tau.stages) & _UNTAU_STAGES
            assert {s.replace("_dcnt", "") for s in _not_dx(tau.stages)} == _not_dx(
                shipped.stages
            )
            twin = tau.without_temperature()
            assert type(twin) is cake_backend.Plan and twin.dcnt == counted_dz
            assert twin == dataclasses.replace(
                skip, problem=tau_problem, dcnt=counted_dz
            )
            assert {twin.recompute_stage(i) for i in range(twin.num_chunks)} == {
                "gemm_logits_dz_dcnt" if counted_dz else "gemm_logits_dz"
            }
            bwd = cake_backend._recompute_keys_temperature(tau)
            assert bwd == tuple(
                (rename.get(s, s), i) for s, i in cake_backend.recompute_keys(skip)
            )
            assert cake_backend.cast_keys(tau) == cake_backend.cast_keys(skip)
            _, forks, sides = cake_backend._side_stream_schedule_temperature(
                tau, bwd, True
            )
            assert sorted(forks) == [i for i, k in enumerate(bwd) if k[0] == dz_tau]
            assert sides and all(bwd[p][0].endswith("_dcnt") for p in sides)
            if T == 65031:
                assert shipped.dw_cast_stage == "gemm_dw_cast_bf16_tma"
                assert tau.dw_cast_stage == cake_backend.DW_CAST_TMA_DCNT_STAGE
                old = cake_backend.make_plan_temperature(
                    tau_problem,
                    dcnt=True,
                    counted_tma_cast=False,
                    counted_dz=counted_dz,
                    **kw,
                )
                assert old.dcnt and old.dw_cast_stage == "gemm_dw_cast_bf16_dcnt"
                assert type(old) is kind
            pin = dict(kw, gemm_tuning={"dw": {"epi_store": "tma"}})
            pinned = cake_backend.make_plan_temperature(
                tau_problem, dcnt=True, counted_dz=counted_dz, **pin
            )
            assert not pinned.dcnt and type(pinned) is cake_backend._TemperaturePlan
            assert pinned == cake_backend.make_plan_temperature(tau_problem, **pin)
    assert counted, "no architecture ran the counted forms"


def test_default_route_carries_no_temperature_state():
    """A call without a temperature builds and reads what it did before the temperature landed: :class:`Plan` (both
    modules), :class:`ForwardResult` (both) and ``_HiddenCount`` have no temperature field (the temperature flag of a
    plan is a class attribute of :class:`_TemperaturePlan`, a forward's compact temperature rows and the hidden count's
    ``tau_c`` are typed class defaults set by the temperature path); :func:`make_plan` has no temperature keyword
    (:func:`make_plan_temperature` plans the temperature forms, in each module of its own plan class);
    :func:`stage_values`, :func:`make_plan` and the stage methods of :class:`Plan` read no temperature, and
    :func:`stage_values_temperature` refuses a temperature form on a plan without it and a stage it replaces on a
    plan with it."""
    for cls in (
        cake_backend.Plan,
        opt_in.Plan,
        cake_backend.ForwardResult,
        opt_in.ForwardResult,
    ):
        names = {f.name for f in dataclasses.fields(cls)}
        assert not {"temperature", "temperature_rows"} & names, cls
    assert "tau_c" not in {
        f.name for f in dataclasses.fields(cake_backend._HiddenCount)
    }
    assert cake_backend.ForwardResult.temperature_rows is None
    assert cake_backend._HiddenCount.tau_c is None
    for module in (cake_backend, opt_in):
        assert "temperature" not in inspect.signature(module.make_plan).parameters
        for fn in (module.make_plan, module.stage_values):
            assert "temperature" not in inspect.getsource(fn), (module, fn)
    for fn in (
        cake_backend.Plan.hidden_logits_stage,
        cake_backend.Plan.forward_logits_stage,
        cake_backend.Plan.recompute_stage,
        cake_backend.Plan.stages.fget,
    ):
        assert "temperature" not in inspect.getsource(fn), fn
    assert not cake_backend.Plan.temperature and not opt_in.Plan.temperature
    g = cake_backend.Geometry.from_record(
        {"geometry": {"hidden": 6144, "vocab": 154880}}
    )
    kw = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        num_sms=148,
        arch="sm_100a",
        logprob_dz=True,
        logprob_lbl=True,
    )
    problem = _problem(8700, 6144, 154880, 4096, entry="logprob")
    tau_problem = cake_backend._with_temperature(problem)
    for module in (cake_backend, opt_in):
        tau = module.make_plan_temperature(tau_problem, **kw)
        plain = module.make_plan(tau_problem, **kw)
        assert isinstance(tau, module.Plan) and tau.temperature
        assert type(tau) is module._TemperaturePlan and type(plain) is module.Plan
        assert tau.without_temperature() == plain and tau != plain
        assert (
            type(dataclasses.replace(tau, dx_finalize=True)) is module._TemperaturePlan
        )
        assert tau.row_finalize_stage == "row_finalize_tau"
        assert plain.row_finalize_stage == "row_finalize"
        with pytest.raises(ValueError, match="temperature form"):
            module.stage_values_temperature("row_finalize_tau", {}, plain, 0)
        with pytest.raises(ValueError, match="row kernel"):
            module.stage_values_temperature("row_finalize", {}, tau, 0)
    assert opt_in.make_plan_temperature(tau_problem, **kw).dw_rows is None


def test_registry_declares_only_known_stages():
    """Every stage a registry record declares is a name of ``STAGES_WITH_TEMPERATURE``, of the candidate-support
    programs (``cake_support.SUPPORT_STAGES``), of the support path's dense pass
    (``cake_support.SUPPORT_DENSE_STAGES``: its accumulating slab sum ``dx_slab_sum_acc`` is outside ``STAGES``) or of
    the one-pass cross-entropy programs (``ONE_PASS_FWD_STAGES``, ``ONE_PASS_COMBINE_STAGE``, ``ONE_PASS_DW_STAGE``,
    looked up by name by the one-pass call): :func:`cake_jit.registered_stages_with_temperature` keeps the names of
    ``STAGES_WITH_TEMPERATURE`` only, so a row under another name (``gemm_logits_tau`` for ``gemm_logits_mcnt_tau``)
    would drop out silently -- the hidden valid-row count would never find its temperature form and the device
    temperature tests would skip."""
    cake_support = cake_package("cake.cake_lm_head_nondet_train", "cake_support")
    one_pass = {
        cake_backend.ONE_PASS_COMBINE_STAGE,
        cake_backend.ONE_PASS_DW_STAGE,
        *(s for forms in cake_backend.ONE_PASS_FWD_STAGES.values() for s in forms),
    }
    known = (
        set(cake_jit.STAGES_WITH_TEMPERATURE)
        | set(cake_support.SUPPORT_STAGES)
        | set(cake_support.SUPPORT_DENSE_STAGES)
        | one_pass
    )
    for name, record in cake_jit.MODULES.items():
        unknown = set(record.get("stages", ())) - known
        assert not unknown, (name, sorted(unknown))


def test_registered_stages_with_temperature(monkeypatch):
    """A call without a temperature reads the stage tables without the temperature forms: ``STAGES``,
    ``GEMM_STAGES``, ``ROW_STAGES`` and ``STAGE_TENSORS`` carry none, and
    :func:`cake_jit.registered_stages` lists a record's stages without them.
    :func:`cake_jit.registered_stages_with_temperature` lists every registered stage in the launch order of
    ``STAGES_WITH_TEMPERATURE``, with the same declared-versus-carried check."""
    assert (
        tuple(
            s
            for s in cake_jit.STAGES_WITH_TEMPERATURE
            if s.startswith("gemm_") and s not in TAU_STAGES
        )
        == cake_jit.GEMM_STAGES
    )
    tables = (
        cake_jit.STAGES
        + cake_jit.GEMM_STAGES
        + cake_jit.ROW_STAGES
        + tuple(cake_backend.STAGE_TENSORS)
    )
    assert not any(s.endswith("_tau") for s in tables)
    with_tau = cake_jit.registered_stages_with_temperature
    for name in cake_jit.MODULES:
        stages = with_tau(name)
        assert cake_jit.registered_stages(name) == tuple(
            s for s in stages if s not in TAU_STAGES
        )
        assert list(stages) == [
            s for s in cake_jit.STAGES_WITH_TEMPERATURE if s in stages
        ]
    name = "cake_lm_head_loss_fake"
    record = _fake_record(_ROW_GRAD_PLAN)
    tau = dict(record, stages=["row_grad", "row_finalize_tau"])
    tau["row_finalize_tau"] = record["row_grad"]
    monkeypatch.setitem(cake_jit.MODULES, name, tau)
    assert cake_jit.registered_stages(name) == ("row_grad",)
    assert with_tau(name) == ("row_finalize_tau", "row_grad")
    monkeypatch.setitem(
        cake_jit.MODULES, name, dict(record, stages=["row_grad", "row_finalize"])
    )
    with pytest.raises(ValueError, match="declares stages"):
        with_tau(name)


def test_temperature_stage_values_and_runner():
    """A runner with a temperature binds it as ``tensors["temperature"]`` (the chunk loop's rows: ``[T]``
    uncompacted, the compact ``[T_v]`` rows when compacted, like ``lse``) and every logits GEMM / row kernel role
    reads it (the reference path keeps the stage names); the dX / dW GEMMs never do; its steps give the bits of the
    autograd entry; a temperature of another shape or for the loss entry is refused."""
    T, C = 37, 16
    inp = _host_inputs(T, ignore_frac=0.2, temperature="hetero")
    kw = dict(chunk_size=C, entry="logprob", backend="reference", fuse_dw_cast=False)
    runner = cake_backend.prepare_lm_head_loss_temperature(
        inp.X,
        inp.W,
        inp.labels,
        temperature=inp.temperature,
        **kw,
    )
    plan, t = runner.plan, runner.tensors
    assert plan.problem.temperature and not plan.temperature
    assert t["temperature"] is inp.temperature
    tau_values = cake_backend.stage_values_temperature
    for index in range(plan.num_chunks):
        for stage in ("gemm_logits", "gemm_logits_nostats", "row_finalize", "row_grad"):
            v = tau_values(stage, t, plan, index)
            assert v["temperature"] is t["temperature"], (stage, index)
            # the stage values of a call without a temperature read none
            assert "temperature" not in cake_backend.stage_values(stage, t, plan, index)
        for stage in ("gemm_dx", "gemm_dw_acc"):
            assert tau_values(stage, t, plan, index).get("temperature") is None
    plain = cake_backend.prepare_lm_head_loss(inp.X, inp.W, inp.labels, **kw)
    assert not plain.plan.problem.temperature
    for stage in ("gemm_logits", "row_finalize", "row_grad"):
        assert (
            cake_backend.stage_values(stage, plain.tensors, plain.plan, 0).get(
                "temperature"
            )
            is None
        )
    row_index = inp.valid.nonzero().reshape(-1)
    assert 0 < row_index.numel() < T
    tau_c = inp.temperature.index_select(0, row_index).contiguous()
    compact = cake_backend.prepare_lm_head_loss_temperature(
        inp.X,
        inp.W,
        inp.labels,
        compact_rows=row_index,
        temperature=tau_c,
        **kw,
    )
    assert compact.plan.compact and compact.tensors["temperature"] is tau_c
    compact.forward()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    eager = _run(inp, C, entry="logprob", compact_rows=True, fuse_dw_cast=False)
    assert torch.equal(compact.scatter(compact.logp), eager["logp"])
    for bad in (inp.temperature, tau_c[:-1], tau_c.double()):
        with pytest.raises(ValueError, match="temperature"):
            cake_backend.prepare_lm_head_loss_temperature(
                inp.X,
                inp.W,
                inp.labels,
                compact_rows=row_index,
                temperature=bad,
                **kw,
            )
    with pytest.raises(ValueError, match="temperature"):
        cake_backend.prepare_lm_head_loss_temperature(
            inp.X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=C,
            backend="reference",
            temperature=inp.temperature,
        )


def test_temperature_logprob_arm_switches():
    """A temperature runs both logits GEMM arms: an explicit ``logprob_lbl=False`` / ``logprob_dz=False`` with a
    temperature is a ``ValueError`` on every backend (the default ``None`` and ``True`` are accepted); without a
    temperature the switches keep their meaning."""
    inp = _host_inputs(37, ignore_frac=0.2, temperature="hetero")
    for kw in (dict(logprob_lbl=False), dict(logprob_dz=False)):
        name = next(iter(kw))
        with pytest.raises(ValueError, match=name):
            _run(inp, 16, entry="logprob", **kw)
        with pytest.raises(ValueError, match=name):
            cake_backend.prepare_lm_head_loss_temperature(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                entry="logprob",
                backend="reference",
                temperature=inp.temperature,
                **kw,
            )
        _run(dataclasses.replace(inp, temperature=None), 16, entry="logprob", **kw)
    on = _run(inp, 16, entry="logprob", logprob_lbl=True, logprob_dz=True)
    _assert_same_bits(on, _run(inp, 16, entry="logprob"), _LOGPROB_KEYS)


def test_temperature_dx_slices_fit_the_k_steps():
    """The temperature plans of both modules take the fitted dX slice counts of the plan without a temperature (V =
    256 has four K steps, V = 512 eight: every slice holds a K step), every pin whose slices all hold a K step is kept
    and a pin of 3 slices over V = 256 is rejected."""
    g = cake_backend.Geometry.from_record(None)
    kw = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
        logprob_dz=True,
        logprob_lbl=True,
    )
    for module in (cake_backend, opt_in):
        for V in (256, 512):
            problem = _problem(1000, 6144, V, 4096, entry="logprob")
            plain = module.make_plan(problem, **kw)
            tau = module.make_plan_temperature(
                cake_backend._with_temperature(problem), **kw
            )
            assert tau.temperature and tau.dx_slices == plain.dx_slices, (
                module.__name__,
                V,
                tau.dx_slices,
                plain.dx_slices,
            )
            assert all(_holds_a_k_step(s, V // 64) for s in _slices_per_chunk(tau))
        tau_problem = cake_backend._with_temperature(
            _problem(1000, 6144, 256, 4096, entry="logprob")
        )
        for k in (1, 2, 4):
            pinned = module.make_plan_temperature(
                tau_problem, gemm_tuning={"dx": {"k_slices": k}}, **kw
            )
            assert pinned.dx_slices == (k,), (module.__name__, k)
        with pytest.raises(ValueError, match="K steps"):
            module.make_plan_temperature(
                tau_problem, gemm_tuning={"dx": {"k_slices": 3}}, **kw
            )


def test_temperature_refuses_keep_logits():
    """``memory_mode="keep_logits"`` takes no temperature: the opt-in module's temperature entry rejects a scalar
    other than 1.0 and a per-token tensor (ignored rows included) with a ``ValueError`` before any launch; a scalar
    whose FP32 value is 1.0 is the call without a temperature (on the reference backend the same error as the call
    without one, which runs the kept form on the ``cake`` backend only)."""
    inp = _host_inputs(37, ignore_frac=0.2, temperature="hetero")
    for temperature in (0.5, 2, inp.temperature, torch.ones_like(inp.temperature)):
        for backend in ("cake", "reference"):
            with pytest.raises(ValueError, match="takes no temperature"):
                opt_in.cake_lm_head_nondet_train_logprob_temperature(
                    inp.X,
                    inp.W,
                    inp.labels,
                    chunk_size=16,
                    backend=backend,
                    memory_mode="keep_logits",
                    temperature=temperature,
                )
    with pytest.raises(ValueError) as plain:
        opt_in.cake_lm_head_nondet_train_logprob(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=16,
            backend="reference",
            memory_mode="keep_logits",
        )
    for one in (1.0, 1, 1.0 + 2.0**-30):
        with pytest.raises(ValueError) as got:
            opt_in.cake_lm_head_nondet_train_logprob_temperature(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=16,
                backend="reference",
                memory_mode="keep_logits",
                temperature=one,
            )
        assert str(got.value) == str(plain.value), one
    with pytest.raises(ValueError, match="takes no temperature"):
        _run(inp, 16, entry="logprob", memory_mode="keep_logits")


def test_temperature_needs_the_registered_forms():
    """On a registered program a call with a temperature needs the arm's temperature form: a record without it is a
    ``NotImplementedError`` naming the missing stage (never a silent fallback to the stages without the
    temperature), an arm switched off a ``ValueError``; a call without a temperature keeps today's rule; the
    reference path (no record) runs no arm."""
    record = {"arch": "sm_100a"}
    stages = tuple(s for s in cake_jit.STAGES if s not in TAU_STAGES)
    arm = cake_backend._logprob_arm_temperature
    for stage in (cake_backend.LOGPROB_DZ_STAGE, cake_backend.LOGPROB_LBL_STAGE):
        tau_stage = stage + "_tau"
        assert arm(record, stages, stage, True) is True
        assert arm(record, stages, stage, False) is False
        with pytest.raises(NotImplementedError, match=tau_stage):
            arm(record, stages, stage, True, temperature=True)
        assert arm(record, stages + (tau_stage,), stage, True, temperature=True) is True
        with pytest.raises(ValueError, match=stage):
            arm(record, stages + (tau_stage,), stage, False, temperature=True)
        assert arm(None, stages, stage, True, temperature=True) is False


def test_temperature_binding_keys_carry_the_flag():
    """The binding keys of the log-probability entry carry whether the call has a temperature -- a flag, never the
    temperature tensor (every tensor of a remembered launch is slotted per call)."""
    kw = _valid_call(T=8)
    X, W, labels = kw["X"], kw["W"], kw["labels"]
    common = dict(
        objective="ce",
        loss_div=None,
        infer_logp=None,
        loss_weights=None,
        chunk_size=4096,
        need_dx=False,
        need_dw=False,
        grad_weight_dtype=torch.bfloat16,
        entry="logprob",
    )
    tau_key = cake_backend.temperature_binding_key
    fwd = cake_backend.forward_binding_key(X, W, labels, **common)
    assert fwd == cake_backend.forward_binding_key(X, W, labels, **common)
    assert tau_key(fwd) != fwd and tau_key(fwd) == tau_key(fwd)
    assert not any(isinstance(v, torch.Tensor) for v in tau_key(fwd))
    lse, dlogp = torch.zeros(8), torch.zeros(8)
    bkw = dict(chunk_size=4096, need_dx=True, need_dw=True)
    bwd = cake_backend.logprob_backward_binding_key(X, W, labels, lse, dlogp, **bkw)
    assert tau_key(bwd) != bwd and tau_key(bwd) != tau_key(fwd)
    # the default path's keys take no temperature (the flag is the temperature path's)
    assert (
        "temperature"
        not in inspect.signature(cake_backend.forward_binding_key).parameters
    )
    assert (
        "temperature"
        not in inspect.signature(cake_backend.logprob_backward_binding_key).parameters
    )


def test_bind_stage_refuses_a_temperature_without_its_operand(monkeypatch):
    """A call with a temperature fails closed on a temperature a stage cannot read: before its launches are bound
    (:func:`cake_backend._check_temperature_operands`), a ``temperature`` host value for a stage whose argument plan
    has no temperature operand is a ``ValueError`` before any module loads (host steps are skipped); ``None`` passes;
    a stage whose plan has the operand passes and :func:`bind_stage` binds it as a tensor slot."""
    name = "cake_lm_head_loss_fake"
    monkeypatch.setitem(cake_jit.MODULES, name, _fake_record(_ROW_GRAD_PLAN))
    loads: list = []

    def load(n, s):
        loads.append(s)
        return SimpleNamespace(run=lambda *a: None)

    monkeypatch.setattr(cake_backend, "load_cake_lm_head_loss_module", load)
    check = cake_backend._check_temperature_operands
    key = ("row_grad", 0)
    values = _row_grad_values()
    values["temperature"] = torch.ones(4)
    with pytest.raises(ValueError, match="temperature"):
        check(name, (("gather_rows", 0), key), {key: values})
    assert not loads
    values["temperature"] = None
    check(name, (key,), {key: values})
    assert cake_backend.bind_stage(name, "row_grad", values, (4, 1, 1)).slots == (
        (0, "z"),
        (1, "labels"),
        (2, "lse"),
        (3, "d"),
    )
    monkeypatch.setitem(
        cake_jit.MODULES,
        name,
        _fake_record(
            _ROW_GRAD_PLAN[:4] + [["buffer", "temperature"]] + _ROW_GRAD_PLAN[4:]
        ),
    )
    values["temperature"] = torch.ones(4)
    check(name, (key,), {key: values})
    launch = cake_backend.bind_stage(name, "row_grad", values, (4, 1, 1))
    assert (4, "temperature") in launch.slots
    assert launch.arguments[4] is values["temperature"]


def test_temperature_memory_report():
    """``memory_report_temperature(temperature=True)`` counts the compact FP32 temperature the forward saves (4 bytes per loop
    row) and, on the hidden valid-row count path, the ``[T]`` buffer and its transient int64 clamped index; the
    default report is unchanged; the loss entry has no temperature."""
    T, T_v, C = 16231, 15419, 4096
    base = cake_backend.memory_report(
        T, DEFAULT_H, DEFAULT_V, C, entry="logprob", valid_rows=T_v
    )
    tau = cake_backend.memory_report_temperature(
        T, DEFAULT_H, DEFAULT_V, C, entry="logprob", valid_rows=T_v, temperature=True
    )
    assert "temperature_rows" not in base["accumulators"]
    assert tau["accumulators"]["temperature_rows"] == 4 * T_v
    assert tau["accumulator_bytes"] - base["accumulator_bytes"] == 4 * T_v
    assert tau["temporary"] == base["temporary"]
    hidden = cake_backend.memory_report_temperature(
        T,
        DEFAULT_H,
        DEFAULT_V,
        C,
        entry="logprob",
        valid_rows=T_v,
        hidden_count=True,
        temperature=True,
    )
    assert hidden["accumulators"]["temperature_rows"] == 4 * T
    assert hidden["temporary"]["temperature_index"] == 8 * T
    with pytest.raises(ValueError, match="temperature"):
        cake_backend.memory_report_temperature(
            T, DEFAULT_H, DEFAULT_V, C, temperature=True
        )


# --- device layer: the generated temperature forms


def _require_tau_program():
    """Skip unless the registered program carries the five temperature forms (``TAU_STAGES``)."""
    _require_program(entry="logprob")
    name, _ = cake_backend.record_for(CUDA)
    missing = [
        s
        for s in TAU_STAGES
        if s not in cake_jit.registered_stages_with_temperature(name)
    ]
    if missing:
        pytest.skip(f"the registered program has no temperature forms {missing}")
    return name


def _spy_bound_stages(monkeypatch) -> list:
    """Record the stage of every launch bound from now on (the chunk loop, the hidden count's prologue and the eager
    launches all bind through ``_bind_all``)."""
    bound: list = []

    def spy_of(real):
        def spy(record, module_name, keys, values, device, geometry):
            bound.extend(k[0] for k in keys)
            return real(record, module_name, keys, values, device, geometry)

        return spy

    # the opt-in entry points bind through the opt-in module's own ``_bind_all``
    for be in (cake_backend, opt_in):
        monkeypatch.setattr(be, "_bind_all", spy_of(be._bind_all))
    return bound


def _assert_temperature_plan(bound, *, hidden, case):
    """The launches of a temperature call that bound: the label-column, ``dz`` (counted under the zero-gradient row
    skip, or not) and row-finalize temperature forms, the device-count form exactly when the hidden valid-row count
    ran, and none of the stages they replace."""
    stages = set(bound)
    need = {"gemm_logits_lbl_tau", "row_finalize_tau"}
    assert need <= stages, (case, sorted(stages))
    assert stages & {"gemm_logits_dz_tau", "gemm_logits_dz_dcnt_tau"}, (
        case,
        sorted(stages),
    )
    assert ("gemm_logits_mcnt_tau" in stages) == bool(hidden), (case, sorted(stages))
    assert not stages & _UNTAU_STAGES, (case, sorted(stages & _UNTAU_STAGES))


def _expect_hidden(inp, C, kw) -> bool:
    """Does a compacted-by-default autograd call with the switches ``kw`` run the hidden valid-row count on this
    device and toolchain (``cake_jit.toolchain_runs_hidden_count``: not sm_103a with CUDA 13.0)?  A
    ``memory_mode="budget"`` call never does: the hidden count's chunk-0 buffers take ``min(chunk, T)`` rows before
    the count is known."""
    return bool(
        kw.get("compact_rows", True) is not False
        and kw.get("hidden_count", True) is not False
        and kw.get("memory_mode") != "budget"
        and kw.get("num_valid_rows") is None
        and -(-inp.T // C) >= cake_backend.HIDDEN_COUNT_MIN_CHUNKS
        and cake_jit.toolchain_runs_hidden_count(cake_backend.arch_for(CUDA))
    )


# (name, switches): every switch whose values must give the bits of the defaults
_TAU_SWITCH_MATRIX: tuple[tuple[str, dict], ...] = (
    ("defaults", {}),
    ("uncompacted", dict(compact_rows=False)),
    ("host_count", dict(hidden_count=False)),
    ("given_count", dict(num_valid_rows=True)),
    ("budget", dict(memory_mode="budget")),
    ("unfused_dw_cast", dict(fuse_dw_cast=False)),
    ("no_dx_finalize", dict(dx_finalize=False)),
    ("no_dw_stream", dict(dw_stream=False)),
    ("dw_stream", dict(dw_stream=True)),
)


def _tau_switches(inp, switches) -> dict:
    kw = dict(switches)
    if kw.get("num_valid_rows") is True:
        kw["num_valid_rows"] = int(inp.valid.sum())
    return kw


def _tau_inputs(T, *, W, ignore_frac=0.05, temperature="hetero", seed_offset=7637):
    return make_inputs(
        T,
        seed=SEED + seed_offset + T,
        device=CUDA,
        W=W,
        ignore_frac=ignore_frac,
        temperature=temperature,
    )


def test_device_temperature_one_is_bitwise_the_default(glm_weight, monkeypatch):
    """``temperature=1.0`` and a ones tensor give the bits (integer views) of the call without a temperature over
    the switch matrix -- compacted / uncompacted, hidden / host / caller's count, fast / budget, fused / unfused dW
    cast, dX finalize, dW side stream -- with the binding cache on and off, at two and four chunks, one dX slice pinned
    (``_TAU_ONE``); with the default dX slices ``logp`` / ``dW`` keep the bits and ``dX`` is held by
    :func:`_assert_tau_close`.  ``1.0`` binds no temperature form and the ones tensor binds the temperature forms (the
    device-count one exactly when the hidden valid-row count runs) and none of the stages they replace, so no cell
    compares a path with itself."""
    _require_tau_program()
    bound = _spy_bound_stages(monkeypatch)
    for T in (4097, 16231):
        inp = _device_inputs(T, W=glm_weight)
        ones = torch.ones(T, dtype=torch.float32, device=CUDA)
        for name, switches in _TAU_SWITCH_MATRIX:
            kw = dict(_tau_switches(inp, switches), **_TAU_ONE)
            for cache_on in (True, False):
                case = (T, name, cache_on)
                with _cache(cache_on, kw) as cache:
                    cache.clear()
                    bound.clear()
                    base = _run(inp, 4096, entry="logprob", backend="cake", **kw)
                    assert not any(s.endswith("_tau") for s in bound), case
                    bound.clear()
                    one = _run(
                        inp,
                        4096,
                        entry="logprob",
                        backend="cake",
                        temperature=1.0,
                        **kw,
                    )
                    assert not any(s.endswith("_tau") for s in bound), case
                    bound.clear()
                    tensor = _run(
                        inp,
                        4096,
                        entry="logprob",
                        backend="cake",
                        temperature=ones,
                        **kw,
                    )
                    _assert_temperature_plan(
                        bound, hidden=_expect_hidden(inp, 4096, kw), case=case
                    )
                _assert_same_bits(base, one, _LOGPROB_KEYS)
                _assert_same_bits(base, tensor, _LOGPROB_KEYS)
        sliced = _run(inp, 4096, entry="logprob", backend="cake")
        _assert_tau_close(
            sliced,
            _run(inp, 4096, entry="logprob", backend="cake", temperature=ones),
            (T, "default slices"),
        )


@pytest.mark.parametrize("T", [4097, 16231])
def test_device_temperature_scalar_is_the_constant_tensor(glm_weight, monkeypatch, T):
    """A scalar temperature gives the bits of the constant FP32 tensor of its value through the temperature forms
    (two chunks; four chunks with the hidden valid-row count; one dX slice pinned, ``_TAU_ONE``) and, with the default
    dX slices, passes the accuracy gates of its references."""
    _require_tau_program()
    bound = _spy_bound_stages(monkeypatch)
    inp = _device_inputs(T, W=glm_weight)
    for s in (0.5, 0.7, 1.3):
        with _cache(False):
            bound.clear()
            scalar = _run(
                inp, 4096, entry="logprob", backend="cake", temperature=s, **_TAU_ONE
            )
            _assert_temperature_plan(
                bound, hidden=_expect_hidden(inp, 4096, {}), case=(T, s)
            )
            tensor = _run(
                inp,
                4096,
                entry="logprob",
                backend="cake",
                temperature=torch.full((T,), s, dtype=torch.float32, device=CUDA),
                **_TAU_ONE,
            )
        _assert_same_bits(scalar, tensor, _LOGPROB_KEYS)
        if T == 4097:
            sliced = _run(inp, 4096, entry="logprob", backend="cake", temperature=s)
            _assert_tau_close(sliced, scalar, (T, s))
            _check_against_references(
                sliced, dataclasses.replace(inp, temperature=s), entry="logprob"
            )


@pytest.mark.parametrize(
    "T, C, ignore",
    [
        (1, 4096, 0.0),
        (4095, 4096, 0.05),
        (4097, 4096, 0.05),
        (4097, 2048, 0.33),
        (16231, 4096, 0.05),
        (16231, 8192, 0.33),
    ],
)
def test_device_temperature_heterogeneous_matches_the_references(
    logprob_chunk_loop, glm_weight, record_property, T, C, ignore
):
    """A heterogeneous per-token temperature on the generated program (the default dX slices), ``REPEATS`` runs on
    the same inputs (fresh allocator state each time, 0xFF-poisoned): every run passes the gates against the FP64
    oracle and the unchunked PyTorch path (both with the temperature: 1.05x its error plus the floor; the production
    ceiling, a temperature-1 figure, only where every temperature is at least 1) with zeros on the ignored rows;
    ``logp`` / ``dW`` never change and are the deterministic package's bits (its temperature entry on the same
    inputs; ``logp`` only where the short tail folds); the worst errors, the run-to-run ``dX`` spread and the worst max |ND - det| and BF16 flip rate of ``dX``
    are recorded (junit properties)."""
    _require_tau_program()
    inp = _tau_inputs(T, W=glm_weight, ignore_frac=ignore)
    ceiling = bool(inp.temperature.min() >= 1.0)
    oracle = reference_fp64(inp, entry="logprob")
    b0 = error_report(reference_unchunked(inp, entry="logprob"), oracle, inp.labels)
    det = _run(inp, C, entry="logprob", backend="cake", package="det")
    first, worst, spread, versus = None, {}, 0.0, []
    for _ in range(REPEATS):
        _poison_allocator(1 << 30)
        run = _run(inp, C, entry="logprob", backend="cake")
        _check_dtypes(run, inp)
        _compare_with_det(run, det, record=versus, folded=_folds(inp, C))
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
        if ceiling:
            for key, measured in PRODUCTION_B0.items():
                if key in errors:
                    assert errors[key] <= 2.0 * measured, key
        if first is None:
            first = run
        else:
            for key in ("loss", "logp", "dW"):
                assert _same_bits(run[key], first[key]), key
            spread = max(
                spread, float((run["dX"].float() - first["dX"].float()).abs().max())
            )
    record_property("temperature_errors_worst", repr(worst))
    record_property("temperature_unchunked_errors", repr(b0))
    record_property(
        "temperature_dX_vs_det",
        repr(
            dict(
                run_to_run_max=spread,
                max_abs=max(v["dX_max_abs_vs_det"] for v in versus),
                flip_rate=max(v["dX_flip_rate"] for v in versus),
                max_ulp=max(v["dX_max_ulp"] for v in versus),
            )
        ),
    )


def test_device_temperature_switches_are_bitwise(
    logprob_chunk_loop, glm_weight, monkeypatch
):
    """A heterogeneous temperature over the switch matrix, one dX slice pinned (``_TAU_ONE``): every switch gives the
    bits (integer views) of the defaults with the binding cache on (first call and remembered binding) and off, each
    cell through the temperature forms; compacted vs uncompacted: ``logp`` bitwise, the first chunk's valid ``dX`` rows
    bitwise, ``dW`` at the precision level; a row permutation of ``(X, labels, temperature, dlogp)`` permutes ``logp``
    and the ``dX`` rows bitwise (row-local arithmetic) -- a temperature read at the wrong row breaks each of these.
    With the default dX slices the defaults keep ``logp`` / ``dW`` and hold ``dX`` (:func:`_assert_tau_close`)."""
    _require_tau_program()
    bound = _spy_bound_stages(monkeypatch)
    one = dict(entry="logprob", backend="cake", **_TAU_ONE)
    for T in (4097, 16231):
        inp = _tau_inputs(T, W=glm_weight)
        with _cache(False):
            bound.clear()
            ref = _run(inp, 4096, **one)
            _assert_temperature_plan(
                bound, hidden=_expect_hidden(inp, 4096, {}), case=(T, "defaults")
            )
        _assert_tau_close(
            _run(inp, 4096, entry="logprob", backend="cake"), ref, (T, "sliced")
        )
        for name, switches in _TAU_SWITCH_MATRIX[1:]:
            if name == "uncompacted":
                continue  # its own expectations below
            kw = _tau_switches(inp, switches)
            for cache_on in (True, False):
                case = (T, name, cache_on)
                with _cache(cache_on, kw) as cache:
                    cache.clear()
                    bound.clear()
                    runs = [_run(inp, 4096, **one, **kw)]
                    _assert_temperature_plan(
                        bound, hidden=_expect_hidden(inp, 4096, kw), case=case
                    )
                    if cache_on:
                        runs.append(_run(inp, 4096, **one, **kw))
                for run in runs:
                    _assert_same_bits(ref, run, _LOGPROB_KEYS)
        with _cache(True) as cache:
            cache.clear()
            bound.clear()
            plain = _run(inp, 4096, compact_rows=False, **one)
            _assert_temperature_plan(bound, hidden=False, case=(T, "uncompacted"))
            comp = _run(inp, 4096, **one)
            plans = {
                b.plan.compact: b.plan
                for key, b in cache._bindings.items()
                if key[0] == "bwd"
            }
        assert set(plans) == {True, False}, plans
        assert torch.equal(_bits(plain["logp"]), _bits(ref["logp"]))
        _assert_same_bits(comp, ref, _LOGPROB_KEYS)
        assert torch.all(plain["dX"][~inp.valid] == 0)
        assert rel_l2(plain["dW"].float(), ref["dW"].float()) <= GATE_TINY["dW_rel_l2"]
        rows = inp.valid[:4096]
        assert torch.equal(
            _bits(plain["dX"][:4096][rows]), _bits(ref["dX"][:4096][rows])
        )
    # row permutation: one chunk, and two full chunks (8192 valid rows of 8623; the hidden valid-row count)
    for T in (4000, 8623):
        inp = _tau_inputs(T, W=glm_weight)
        gen = torch.Generator(device=CUDA)
        gen.manual_seed(SEED + T)
        perm = torch.randperm(T, generator=gen, device=CUDA)
        moved = dataclasses.replace(
            inp,
            X=inp.X[perm].contiguous(),
            labels=inp.labels[perm].contiguous(),
            dlogp=inp.dlogp[perm].contiguous(),
            temperature=inp.temperature[perm].contiguous(),
        )
        a = _run(inp, 4096, **one)
        b = _run(moved, 4096, **one)
        assert torch.equal(_bits(b["logp"]), _bits(a["logp"][perm])), T
        rows = moved.valid
        assert torch.equal(_bits(b["dX"][rows]), _bits(a["dX"][perm][rows])), T


def test_device_temperature_budget_forms_are_bitwise(glm_weight, monkeypatch):
    """Every budget form of the log-probability backward with a heterogeneous temperature gives the fast form's
    bits (integer views; one dX slice pinned, ``_TAU_ONE``), with and without ``dX``; the vocabulary-block recompute
    binds the ``dz`` temperature form and none of the stages the temperature forms replace."""
    _require_tau_program()
    step = _require_recompute()
    for T, C, train_x in ((1500, 4096, True), (8700, 4096, True), (8700, 4096, False)):
        inp = _tau_inputs(T, W=glm_weight, seed_offset=7638)
        _poison_allocator()
        fast = _run(
            inp, C, entry="logprob", backend="cake", train_x=train_x, **_TAU_ONE
        )
        for form in _budget_forms(step):
            calls: list = []
            _force_budget_plan(monkeypatch, form, calls)
            monkeypatch.setattr(opt_in, "_budget_plan_temperature", opt_in._budget_plan)
            bound = _spy_bound_stages(monkeypatch)
            _poison_allocator()
            with _cache(False, ("memory_mode",)):
                budget = _run(
                    inp,
                    C,
                    entry="logprob",
                    backend="cake",
                    train_x=train_x,
                    memory_mode="budget",
                    **_TAU_ONE,
                )
            assert calls == ["logprob"], (T, C, train_x, form)
            _assert_same_bits(fast, budget, _LOGPROB_KEYS)
            assert "gemm_logits_dz_tau" in bound, (T, form)
            assert not set(bound) & _UNTAU_STAGES, (T, form, sorted(set(bound)))
            monkeypatch.undo()


def test_device_temperature_dw_side_stream_is_bitwise(glm_weight, monkeypatch):
    """Four compacted chunks with a heterogeneous temperature: the dW side stream (forked behind each chunk's ``dz``
    temperature form) gives the single-stream bits (integer views; one dX slice pinned, ``_TAU_ONE``) in ten runs,
    through a fresh binding and the remembered one; the side-stream calls enter two streams."""
    _require_tau_program()
    T, C = 16231, 4096
    inp = _tau_inputs(T, W=glm_weight)
    streams: list = []
    real = cake_backend._ffi_stream_context

    def spy(index):
        streams.append(torch.cuda.current_stream(index).cuda_stream)
        return real(index)

    monkeypatch.setattr(cake_backend, "_ffi_stream_context", spy)
    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    off = _run(inp, C, dw_stream=False, **kw)
    for i in range(10):
        with _cache(i % 2 == 0):
            streams.clear()
            on = _run(inp, C, dw_stream=True, **kw)
            assert len(set(streams)) == 2, i
        _assert_same_bits(off, on, _LOGPROB_KEYS)
    _assert_same_bits(off, _run(inp, C, **kw), _LOGPROB_KEYS)


def test_device_temperature_frozen_inputs(glm_weight):
    """Frozen ``X``, frozen ``W`` or both with a heterogeneous temperature: the one-sided gradients equal the
    two-sided ones (one dX slice pinned, ``_TAU_ONE``), ``logp`` is unchanged."""
    _require_tau_program()
    inp = _tau_inputs(4097, W=glm_weight)
    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    both = _run(inp, 4096, **kw)
    x_only = _run(inp, 4096, train_w=False, **kw)
    w_only = _run(inp, 4096, train_x=False, **kw)
    neither = _run(inp, 4096, train_x=False, train_w=False, **kw)
    assert x_only["dW"] is None and w_only["dX"] is None
    assert neither["dX"] is None and neither["dW"] is None
    assert torch.equal(_bits(x_only["dX"]), _bits(both["dX"]))
    assert torch.equal(_bits(w_only["dW"]), _bits(both["dW"]))
    for r in (x_only, w_only, neither):
        assert torch.equal(_bits(r["logp"]), _bits(both["logp"]))


def test_device_temperature_upstream_gradients(glm_weight):
    """Arbitrary per-row upstream gradients with a heterogeneous temperature on the generated program: random
    (nonzero on ignored rows too) and large ones within the gates of their references (the default dX slices), zeros
    give zero gradients (values; the default slices), the negated upstream the negated gradients and a retained graph
    repeats the backward bitwise (one dX slice pinned, ``_TAU_ONE``)."""
    _require_tau_program()
    T = 4097
    inp = _tau_inputs(T, W=glm_weight)
    kw = dict(entry="logprob", backend="cake")
    base = _run(inp, 4096, **kw, **_TAU_ONE)
    zeros = _run(
        dataclasses.replace(inp, dlogp=torch.zeros_like(inp.dlogp)), 4096, **kw
    )
    assert torch.all(zeros["dX"] == 0) and torch.all(zeros["dW"] == 0)
    negated = _run(dataclasses.replace(inp, dlogp=-inp.dlogp), 4096, **kw, **_TAU_ONE)
    assert torch.equal(negated["dX"], -base["dX"])
    assert torch.equal(negated["dW"], -base["dW"])
    gen = torch.Generator(device=CUDA)
    gen.manual_seed(SEED + 7639)
    for g in (torch.randn(T, generator=gen, device=CUDA), inp.dlogp * 1e4):
        g_inp = dataclasses.replace(inp, dlogp=g.contiguous())
        _check_against_references(_run(g_inp, 4096, **kw), g_inp, entry="logprob")
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
        X,
        W,
        inp.labels,
        chunk_size=4096,
        backend="cake",
        temperature=inp.temperature,
        **_TAU_ONE,
    )
    first = torch.autograd.grad(logp, (X, W), inp.dlogp, retain_graph=True)
    second = torch.autograd.grad(logp, (X, W), inp.dlogp)
    for a, b, ref in zip(first, second, (base["dX"], base["dW"]), strict=True):
        assert torch.equal(_bits(a), _bits(b)) and torch.equal(_bits(a), _bits(ref))


_FRESH_PROCESS_TAU_SCRIPT = """
import sys, torch
sys.path.insert(0, {root!r})
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.cake.test_helpers.cake_package import cake_package
cake_backend = cake_package("cake.cake_lm_head_nondet_train", "cake_backend")
W = make_inputs(1, seed={seed_w}, device="cuda", ignore_frac=0.0).W  # the module fixture's weight
inp = make_inputs({T}, seed={seed}, device="cuda", W=W, ignore_frac=0.05, temperature="hetero")
X = inp.X.detach().requires_grad_(True)
W = inp.W.detach().requires_grad_(True)
logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
    X, W, inp.labels, chunk_size={C}, backend="cake", temperature=inp.temperature,
    gemm_tuning={tuning!r},
)
dX, dW = torch.autograd.grad(logp, (X, W), inp.dlogp)
torch.cuda.synchronize()
torch.save(dict(logp=logp.detach().cpu(), dX=dX.cpu(), dW=dW.cpu()), {out!r})
"""


def test_device_temperature_fresh_thread_process_and_changing_t(
    logprob_chunk_loop, glm_weight, tmp_path
):
    """With a heterogeneous temperature and one dX slice pinned (``_TAU_ONE``): the backward from a fresh thread and
    the whole step in a fresh process give the main thread's bits; calls whose ``T`` changes between them (two chunks,
    one, four) repeat bitwise; with the default dX slices they repeat within :func:`_assert_tau_close`."""
    _require_tau_program()
    T, C = 4097, 4096
    inp = _tau_inputs(T, W=glm_weight)

    def grad_path():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
            X,
            W,
            inp.labels,
            chunk_size=C,
            backend="cake",
            temperature=inp.temperature,
            **_TAU_ONE,
        )
        dX, dW = torch.autograd.grad(logp, (X, W), inp.dlogp)
        torch.cuda.synchronize()
        return dict(logp=logp.detach(), dX=dX, dW=dW)

    main = grad_path()
    _check_against_references(main, inp, entry="logprob")
    outcome = {}

    def worker():
        try:
            outcome["result"] = grad_path()
        except BaseException as exc:  # surfaced by the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="fresh-temperature-thread")
    thread.start()
    thread.join()
    assert "error" not in outcome, repr(outcome.get("error"))
    _assert_same_bits(outcome["result"], main, _LOGPROB_KEYS)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = tmp_path / "fresh_process_temperature.pt"
    script = _FRESH_PROCESS_TAU_SCRIPT.format(
        root=root,
        T=T,
        seed=SEED + 7637 + T,
        seed_w=SEED,
        C=C,
        tuning=_TAU_ONE["gemm_tuning"],
        out=str(out),
    )
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"fresh process failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )
    fresh = torch.load(out)
    _assert_same_bits({k: v.to(CUDA) for k, v in fresh.items()}, main, _LOGPROB_KEYS)
    inputs = [_tau_inputs(n, W=glm_weight) for n in (4095, 4097, 1500, 16231, 4095)]
    kw = dict(entry="logprob", backend="cake")
    first = [_run(i, 4096, **kw, **_TAU_ONE) for i in inputs]
    again = [_run(i, 4096, **kw, **_TAU_ONE) for i in inputs]
    for a, b in zip(first, again, strict=True):
        _assert_same_bits(a, b, _LOGPROB_KEYS)
    _assert_same_bits(first[0], first[-1], _LOGPROB_KEYS)
    sliced = [_run(i, 4096, **kw) for i in inputs]
    for i, (a, b) in enumerate(zip(sliced, first, strict=True)):
        _assert_tau_close(a, b, ("sliced vs one slice", inputs[i].T))
    _assert_tau_close(sliced[0], sliced[-1], "sliced, T changed and back")


def test_device_temperature_ignored_rows(glm_weight):
    """Ignored rows (``-100`` and other negative labels) with a heterogeneous temperature: ``logp`` and the ``dX``
    rows are exactly zero, and the temperatures of ignored rows reach no output -- extreme valid values there
    (``1e-30``, ``3e38``) give the same bits (one dX slice pinned, ``_TAU_ONE``) -- compacted, uncompacted and with the
    hidden valid-row count; the default dX slices keep the zeros."""
    _require_tau_program()
    for T, compact in ((4097, True), (4097, False), (16231, True)):
        inp = _tau_inputs(T, W=glm_weight, ignore_frac=0.33)
        labels = inp.labels.clone()
        ignored = (~inp.valid).nonzero().reshape(-1)
        labels[ignored[:4]] = torch.tensor([-1, -7, -2, -(2**40)], device=CUDA)
        inp = dataclasses.replace(inp, labels=labels)
        kw = dict(entry="logprob", backend="cake", compact_rows=compact)
        sliced = _run(inp, 4096, **kw)
        assert torch.all(sliced["logp"][~inp.valid] == 0), (T, compact)
        assert torch.all(sliced["dX"][~inp.valid] == 0), (T, compact)
        kw.update(_TAU_ONE)
        base = _run(inp, 4096, **kw)
        assert torch.all(base["dX"][~inp.valid] == 0), (T, compact)
        tau = inp.temperature.clone()
        tau[ignored[0::2]] = 1e-30
        tau[ignored[1::2]] = 3e38
        other = _run(dataclasses.replace(inp, temperature=tau), 4096, **kw)
        _assert_same_bits(base, other, _LOGPROB_KEYS)


def test_device_temperature_does_not_synchronize(logprob_backward_form, glm_weight):
    """With a temperature (a tensor, validated by a device assertion, and a scalar) the log-probability backward runs
    without a host synchronization from a remembered binding (compacted and uncompacted, one to three chunks), and
    the forward with the caller's count runs without one."""
    _require_tau_program()
    for T, C, compact in ((1500, 4096, True), (6000, 4096, False), (8700, 4096, True)):
        inp = _tau_inputs(T, W=glm_weight)
        for form in (inp.temperature, 0.7):
            X = inp.X.detach().requires_grad_(True)
            W = inp.W.detach().requires_grad_(True)
            for i in range(2):
                logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
                    X,
                    W,
                    inp.labels,
                    chunk_size=C,
                    backend="cake",
                    compact_rows=compact,
                    temperature=form,
                )
                torch.cuda.synchronize()
                if i:
                    torch.cuda.set_sync_debug_mode("error")
                try:
                    logp.backward(inp.dlogp)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                torch.cuda.synchronize()
                assert torch.all(X.grad[~inp.valid] == 0)
                X.grad = W.grad = None
    if opt_in._given_count_readback(CUDA):
        return  # this toolchain checks the caller's count by a readback
    inp = _tau_inputs(8700, W=glm_weight)
    n = int(inp.valid.sum())
    for form in (inp.temperature, 0.7):
        _run(
            inp,
            4096,
            entry="logprob",
            backend="cake",
            num_valid_rows=n,
            temperature=form,
        )
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            opt_in.cake_lm_head_nondet_train_logprob_temperature(
                inp.X,
                inp.W,
                inp.labels,
                chunk_size=4096,
                backend="cake",
                num_valid_rows=n,
                temperature=form,
            )
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()


@pytest.mark.parametrize("T", [4097, 16231])
def test_device_temperature_step_peak(glm_weight, T):
    """The step peak (forward through backward, outputs and gradients included) with a heterogeneous temperature
    exceeds the peak without one by at most the temperature's own buffers -- the compact ``[T]`` FP32 vector, the
    hidden count's transient int64 clamped index, the validation's bool temporaries -- plus allocator rounding, in
    the fast and the budget memory mode (no ``[T, V]`` buffer)."""
    _require_tau_program()
    _require_recompute()
    inp = _tau_inputs(T, W=glm_weight)
    plain = dataclasses.replace(inp, temperature=None)
    allowance = 4 * T + 8 * T + 3 * T + (16 << 10)
    for kw in ({}, dict(memory_mode="budget")):
        peaks = {}
        for name, case in (("plain", plain), ("tau", inp)):
            extra = (
                {} if case.temperature is None else dict(temperature=case.temperature)
            )
            for _ in range(2):  # the second call launches from the remembered binding
                _, peak, _ = _budget_step(case, "logprob", **extra, **kw)
            peaks[name] = peak
        assert peaks["tau"] - peaks["plain"] <= allowance, (T, kw, peaks, allowance)


_TAU_ASSERT_SCRIPT = """
import sys, torch
sys.path.insert(0, {root!r})
from tests.experimental.test_helpers.cake_lm_head_loss_reference import make_weight
from tests.cake.test_helpers.cake_package import cake_package
cake_backend = cake_package("cake.cake_lm_head_nondet_train", "cake_backend")
H, V = {H}, {V}
labels = torch.tensor([0, -100, 3, 1, -100, 2], device="cuda")
T = int(labels.numel())
X = torch.randn(T, H, device="cuda").to(torch.bfloat16)
W = make_weight(V, H, seed=1, device="cuda")
good = torch.full((T,), 0.7, device="cuda")
cake_backend.cake_lm_head_nondet_train_logprob_temperature(X, W, labels, chunk_size=4, backend={backend!r}, temperature=good)
torch.cuda.synchronize()
print("RIGHT", flush=True)
bad = good.clone()
bad[{row}] = {value}
cake_backend.cake_lm_head_nondet_train_logprob_temperature(X, W, labels, chunk_size=4, backend={backend!r}, temperature=bad)
torch.cuda.synchronize()
print("WRONG_PASSED", flush=True)
"""


def test_device_temperature_assertion_stops_the_stream():
    """An invalid CUDA temperature without ``check_temperature`` stops the stream with a device-side assertion (run in
    a fresh process: the assertion ends the CUDA context) -- zero, negative, NaN and infinite entries at valid rows and
    zero / NaN at ignored rows (strict on every row), on the reference backend and on the generated program; the
    valid temperature passes first."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    # labels [0, -100, 3, 1, -100, 2]: rows 1 and 4 ignored
    cases = [
        ("reference", H_HOST, V_HOST, 0, "0.0"),
        ("reference", H_HOST, V_HOST, 3, "-0.5"),
        ("reference", H_HOST, V_HOST, 2, "float('nan')"),
        ("reference", H_HOST, V_HOST, 5, "float('inf')"),
        ("reference", H_HOST, V_HOST, 1, "0.0"),
        ("reference", H_HOST, V_HOST, 4, "float('nan')"),
    ]
    if _device_supported() and cake_backend.generated_program_available(
        CUDA, entry="logprob"
    ):
        name, _ = cake_backend.record_for(CUDA)
        if all(
            s in cake_jit.registered_stages_with_temperature(name) for s in TAU_STAGES
        ):
            cases += [
                ("cake", DEFAULT_H, DEFAULT_V, 0, "0.0"),
                ("cake", DEFAULT_H, DEFAULT_V, 4, "float('nan')"),
            ]
    for backend, H, V, row, value in cases:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _TAU_ASSERT_SCRIPT.format(
                    root=root, H=H, V=V, backend=backend, row=row, value=value
                ),
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=root,
            timeout=900,
        )
        report = f"{(backend, row, value)} {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        assert "RIGHT" in proc.stdout, report
        assert "WRONG_PASSED" not in proc.stdout and proc.returncode != 0, report
        assert "assert" in proc.stderr.lower(), report


@pytest.mark.parametrize("T", [4097, 16231])
def test_device_temperature_binding_cache_rebinds_it(glm_weight, T):
    """The binding key carries whether a call has a temperature, not its storage or values: calls with different
    temperature tensors, scalars and an equal tensor in other storage launch from one remembered binding and give the
    bits of fresh bindings (cache off; one dX slice pinned, ``_TAU_ONE``); different temperatures give different
    results."""
    _require_tau_program()
    inp = _device_inputs(T, W=glm_weight)
    tau1 = make_temperature(T, "hetero", seed=SEED + 1, device=CUDA)
    tau2 = make_temperature(T, "hetero", seed=SEED + 2, device=CUDA)
    forms = (tau1, tau2, 0.7, 1.3, tau1.clone())
    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    with _cache(True) as cache:
        cache.clear()
        hits = cache.hits
        hot = [_run(inp, 4096, temperature=f, **kw) for f in forms]
        assert cache.hits - hits >= len(forms) - 1, (cache.hits, hits)
    with _cache(False):
        cold = [_run(inp, 4096, temperature=f, **kw) for f in forms]
    for h, c in zip(hot, cold, strict=True):
        _assert_same_bits(h, c, _LOGPROB_KEYS)
    assert not torch.equal(hot[0]["logp"], hot[1]["logp"])
    _assert_same_bits(hot[0], hot[4], _LOGPROB_KEYS)


def test_device_temperature_needs_the_registered_forms(glm_weight, monkeypatch):
    """A registered program without one of the temperature forms the call needs fails closed: without the
    label-column, ``dz`` or row-finalize temperature form a call with a temperature is a ``NotImplementedError``
    naming the missing stage (fast and budget memory modes; never a silent run of the stages without the
    temperature), while the call without a temperature still runs; without the device-count temperature form the
    call takes the host valid-row count and gives the bits of the hidden-count call (one dX slice pinned,
    ``_TAU_ONE``)."""
    _require_tau_program()
    real = cake_backend.registered_stages_with_temperature

    def registered_without(stage):
        # both modules read their own ``registered_stages_with_temperature`` (the stage lists of a call with a
        # temperature; the budget memory mode plans in the opt-in module)
        for be in (cake_backend, opt_in):
            monkeypatch.setattr(
                be,
                "registered_stages_with_temperature",
                lambda n, stage=stage: tuple(s for s in real(n) if s != stage),
            )

    T = 16231
    inp = _device_inputs(T, W=glm_weight)
    tau = make_temperature(T, "hetero", seed=SEED + 11, device=CUDA)
    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    bound = _spy_bound_stages(monkeypatch)
    with _cache(False), _cache(False, ("memory_mode",)):
        full = _run(inp, 4096, temperature=tau, **kw)
        _assert_temperature_plan(
            bound, hidden=_expect_hidden(inp, 4096, {}), case="registered"
        )
        for drop in ("gemm_logits_dz_tau", "gemm_logits_lbl_tau", "row_finalize_tau"):
            registered_without(drop)
            for mode in ("fast", "budget"):
                bound.clear()
                with pytest.raises(NotImplementedError, match=drop):
                    _run(inp, 4096, temperature=tau, memory_mode=mode, **kw)
                assert not set(bound) & _UNTAU_STAGES, (drop, mode, sorted(set(bound)))
            torch.cuda.synchronize()
            _run(inp, 4096, **kw)  # the call without a temperature is unaffected
        registered_without("gemm_logits_mcnt_tau")
        bound.clear()
        host = _run(inp, 4096, temperature=tau, **kw)
        _assert_temperature_plan(bound, hidden=False, case="no mcnt_tau")
    _assert_same_bits(full, host, _LOGPROB_KEYS)


@pytest.mark.parametrize("T,H", [(1000, 6144), (4097, 6144)])
def test_device_vocab_256_dx_slices_temperature(T, H):
    """A call with a temperature plans the dX slices of the call without one (V = 256 has four K steps: every slice
    holds a K step): the default plan and every pin whose slices all hold a K step match the references with a
    per-token temperature, and a pin of 3 slices is rejected."""
    _require_tau_program()
    W = make_inputs(1, seed=SEED, H=H, V=256, device=CUDA, ignore_frac=0.0).W
    inp = make_inputs(
        T, seed=SEED + 7637 + T, H=H, V=256, device=CUDA, W=W, temperature="hetero"
    )
    for k in (None, 1, 2, 4):
        tuning = None if k is None else {"dx": {"k_slices": k}}
        result = _run(inp, 4096, entry="logprob", backend="cake", gemm_tuning=tuning)
        _check_dtypes(result, inp)
        _check_against_references(result, inp, entry="logprob")
    with pytest.raises(ValueError, match="K steps"):
        _run(
            inp,
            4096,
            entry="logprob",
            backend="cake",
            gemm_tuning={"dx": {"k_slices": 3}},
        )
    plan = cake_backend.prepare_lm_head_loss_temperature(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=4096,
        entry="logprob",
        temperature=inp.temperature,
    ).plan
    plain = cake_backend.prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, chunk_size=4096, entry="logprob"
    ).plan
    assert plan.temperature and not plain.temperature
    assert _slices_per_chunk(plan) == _slices_per_chunk(plain), (plan, plain)
    assert all(_holds_a_k_step(s, 4) for s in _slices_per_chunk(plan)), plan.dx_slices


@pytest.mark.parametrize("V", [256, 512, 154880])
def test_device_budget_plan_temperature_is_the_plain_budget_plan(V):
    """The budget form's memory plan of a call with a temperature (``_budget_plan_temperature``) is the plan of the
    call without one on the device's registered program (the K-sliced dX reduction holds no slab: the slice counts do
    not enter the model)."""
    _require_tau_program()
    rows = 1000
    kw = dict(
        num_rows=rows,
        rows=rows,
        hidden=6144,
        vocab=V,
        chunk=4096,
        compact=False,
        need_dx=True,
        fuse_dw_cast=True,
        buffer_rows=rows,
        x_copy=False,
        dx_finalize=False,
        budget_rows=rows,
    )
    plain = opt_in._budget_plan("logprob", CUDA, **kw)
    tau = opt_in._budget_plan_temperature("logprob", CUDA, **kw)
    assert tau == plain, (V, tau, plain)


def test_device_keep_logits_temperature_one_is_the_kept_call(glm_weight, monkeypatch):
    """``memory_mode="keep_logits"`` with a temperature whose FP32 value is 1.0 is the kept call without a
    temperature: the same bits (integer views; one dX slice pinned, ``_TAU_ONE``) and no temperature form bound; a temperature other than 1.0 is a
    ``ValueError``."""
    _require_tau_program()
    bound = _spy_bound_stages(monkeypatch)
    inp = _device_inputs(4097, W=glm_weight)
    base = _run(
        inp,
        4096,
        entry="logprob",
        backend="cake",
        memory_mode="keep_logits",
        **_TAU_ONE,
    )
    assert not any(s.endswith("_tau") for s in bound)
    bound.clear()
    one = _run(
        inp,
        4096,
        entry="logprob",
        backend="cake",
        memory_mode="keep_logits",
        temperature=1.0,
        **_TAU_ONE,
    )
    assert not any(s.endswith("_tau") for s in bound)
    _assert_same_bits(base, one, _LOGPROB_KEYS)
    with pytest.raises(ValueError, match="takes no temperature"):
        _run(
            inp,
            4096,
            entry="logprob",
            backend="cake",
            memory_mode="keep_logits",
            temperature=torch.full((inp.T,), 0.7, device=CUDA),
        )


@pytest.mark.parametrize("module", ["backend", "opt_in"])
@pytest.mark.parametrize(
    "kind", ["selective_recompute", "selective_save_mm", "selective_save_all"]
)
def test_temperature_selective_checkpoint_runs_outside_the_dispatch_modes(
    monkeypatch, kind, module
):
    """The temperature entry under selective activation checkpointing runs its forward with every torch dispatch
    mode popped in both passes (the checkpointed forward and the recomputation), as the other entries of the package, and gives
    the uncheckpointed step's ``logp`` / ``dX`` / ``dW`` bits (a per-token temperature).  ``opt_in``: the opt-in
    module's autograd function (the entry with a count).  Under the gradient placement ``"auto"`` the main module's
    recomputation takes the checkpointed call's row results (no forward); ``"forward"`` runs it in both passes."""
    inp = _host_inputs(65, ignore_frac=0.1, temperature="hetero")
    mod = cake_backend if module == "backend" else opt_in
    kw = (
        {} if module == "backend" else dict(num_valid_rows=int((inp.labels >= 0).sum()))
    )
    real = mod.forward_logprob_temperature
    stacks = []

    def spy(*args, **kwargs):
        stacks.append(torch._C._len_torch_dispatch_stack())
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "forward_logprob_temperature", spy)
    logps = []

    def head(a, b):
        logp = mod.cake_lm_head_nondet_train_logprob_temperature(
            a,
            b,
            inp.labels,
            chunk_size=32,
            backend="reference",
            temperature=inp.temperature,
            **kw,
        )
        logps.append(logp.detach())
        return (logp * inp.dlogp).sum()

    def step(k):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        _checkpointed(k, head, X, W).backward()
        return logps[0], X.grad, W.grad

    plain = step("plain")
    assert stacks == [0], stacks
    for placement in ("auto", "forward"):
        stacks.clear()
        logps.clear()
        with _placement(placement):
            run = step(kind)
        taken = module == "backend" and placement == "auto"
        assert stacks == ([0] if taken else [0, 0]), (placement, stacks)
        for name, a, b in zip(("logp", "dX", "dW"), run, plain, strict=True):
            assert torch.equal(a, b), (placement, name)
    gc.collect()
    assert len(cake_backend._CHECKPOINT_CALLS) == 0
    assert len(cake_backend._LOGPROB_CHECKPOINT_CALLS) == 0


def _tau_region_entry(entry, inp, X, W):
    """The scalar a temperature checkpoint test differentiates: the temperature entry on ``X`` / ``W`` (a per-token
    temperature; ``opt_in_count``: the opt-in entry with a count, ``opt_in_budget``: its budget form; one dX slice
    pinned, ``_TAU_ONE``)."""
    if entry == "backend":
        logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
            X,
            W,
            inp.labels,
            chunk_size=4096,
            temperature=inp.temperature,
            **_TAU_ONE,
        )
    else:
        kw = (
            dict(num_valid_rows=int((inp.labels >= 0).sum()))
            if entry == "opt_in_count"
            else dict(memory_mode="budget")
        )
        logp = opt_in.cake_lm_head_nondet_train_logprob_temperature(
            X,
            W,
            inp.labels,
            chunk_size=4096,
            temperature=inp.temperature,
            **kw,
            **_TAU_ONE,
        )
    return (logp * inp.dlogp).sum()


@pytest.mark.parametrize("entry", ["backend", "opt_in_count", "opt_in_budget"])
def test_device_checkpointed_temperature_step_is_bitwise(glm_weight, entry):
    """The temperature entry under a non-reentrant checkpoint, selective activation checkpointing and a dispatch mode
    (around the checkpointed forward; around the whole step) gives the uncheckpointed step's ``dX`` / ``dW`` bits."""
    _require_tau_program()
    inp = _tau_inputs(12289, W=glm_weight)

    def head(a, b):
        return _tau_region_entry(entry, inp, a, b)

    def step(kind):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        if kind == "dispatch_mode_step":
            with _PassThroughMode():
                _checkpointed("nonreentrant", head, X, W).backward()
        else:
            _checkpointed(kind, head, X, W).backward()
        torch.cuda.synchronize()
        return X.grad, W.grad

    step("plain")  # binds
    plain = step("plain")
    for kind in (
        "nonreentrant",
        "selective_recompute",
        "selective_save_mm",
        "selective_save_all",
        "dispatch_mode",
        "dispatch_mode_step",
    ):
        run = step(kind)
        assert _same_bits(run[0], plain[0]), kind
        assert _same_bits(run[1], plain[1]), kind


@pytest.mark.parametrize("kind", ["hold", "user_mode", "nested"])
@pytest.mark.parametrize("entry", ["backend", "opt_in_count", "opt_in_budget"])
def test_device_selective_checkpoint_region_input_temperature(glm_weight, entry, kind):
    """:func:`test_device_selective_checkpoint_region_input` for the temperature entry: its input formed inside the
    selective-checkpoint region (``X = Hin @ P``), memory held between the forward and the backward (a binding miss in
    the recomputation only), a user dispatch mode around the step (``user_mode``) or a non-reentrant checkpoint nested
    inside the selective one (``nested``).  ``logp`` sum / ``dHin`` / ``dP`` / ``dW`` are the uncheckpointed step's
    bits."""
    _require_tau_program()
    inp = _tau_inputs(12289, W=glm_weight)
    H = inp.X.shape[1]
    g = torch.Generator(device="cpu").manual_seed(SEED + 2)
    Hin0 = (torch.randn(inp.X.shape, generator=g) * 0.5).to(torch.bfloat16).cuda()
    P0 = (torch.randn(H, H, generator=g) / H**0.5).to(torch.bfloat16).cuda()

    def region(h, p, w):
        return _tau_region_entry(entry, inp, torch.mm(h, p), w)

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
                fn = region
                if kind == "nested":

                    def fn(a, b, c):
                        return torch.utils.checkpoint.checkpoint(
                            region, a, b, c, use_reentrant=False
                        )

                out = torch.utils.checkpoint.checkpoint(
                    fn, h, p, w, use_reentrant=False, context_fn=_selective("recompute")
                )
            held = [torch.empty_like(Hin0) for _ in range(2)] if selective else []
            out.backward()
        torch.cuda.synchronize()
        del held
        return [out.detach(), h.grad, p.grad, w.grad]

    step(False)  # binds
    plain = step(False)
    for _ in range(2):
        step(False)
        run = step(True)
        for name, a, b in zip(("logp", "dHin", "dP", "dW"), run, plain, strict=True):
            assert _same_bits(a, b), name


def _distinct_tau(T: int) -> torch.Tensor:
    """A per-token FP32 temperature with a distinct value on every row (log-spaced over ``[0.5, 2]``, shuffled), so a
    temperature row read in the wrong row order changes the result (a constant temperature would hide it)."""
    gen = torch.Generator(device="cpu").manual_seed(SEED + 9011 + int(T))
    values = torch.logspace(
        math.log10(0.5), math.log10(2.0), int(T), dtype=torch.float64
    ).float()
    tau = values[torch.randperm(int(T), generator=gen)].contiguous().to(CUDA)
    assert int(torch.unique(tau).numel()) == int(T)
    return tau


def _counted_partition(monkeypatch) -> list:
    """Record the rows of every :func:`cake_backend.zero_dlogp_partition` call from now on (the skip ran)."""
    partition = cake_backend.zero_dlogp_partition
    calls: list = []

    def counted(*args, **kwargs):
        calls.append(int(args[2].numel()))
        return partition(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "zero_dlogp_partition", counted)
    return calls


TAU_ZERO_DLOGP_CASES = [
    (12289, "none", "distinct"),
    (12289, ("random", 0.166), "distinct"),
    (12289, ("random", 0.5), "distinct"),
    (12289, "all", "distinct"),
    (12289, ("group", 0.5), "scalar"),
    (8117, ("random", 0.166), "scalar"),
    # two chunks, the second of about 650 valid rows: the TMA cast's counted form (the shipped chunks)
    (5000, ("random", 0.5), "distinct"),
    # 4370 valid rows: the short-tail fold (one counted chunk)
    (4600, ("random", 0.166), "distinct"),
]

# one-chunk calls (5 % ignored rows): T 2048 = 1946 valid rows (all live, all zero, z 0.166 / 0.5), T 1158 = 1100,
# T 4096 = 3891, T 64 = 61
TAU_ZERO_DLOGP_ONE_CHUNK_CASES = [
    (2048, "none", "distinct"),
    (2048, "all", "distinct"),
    (2048, ("random", 0.166), "distinct"),
    (2048, ("random", 0.5), "scalar"),
    (1158, ("random", 0.166), "distinct"),
    (4096, ("random", 0.166), "distinct"),
    (64, ("random", 0.5), "scalar"),
]


@pytest.mark.parametrize("dz_form", ("counted", "mixed"))
@pytest.mark.parametrize(
    "T, kind, form", TAU_ZERO_DLOGP_CASES + TAU_ZERO_DLOGP_ONE_CHUNK_CASES, ids=str
)
def test_device_temperature_zero_dlogp_rows_skip(
    logprob_chunk_loop, glm_weight, monkeypatch, request, T, kind, form, dz_form
):
    """The zero-gradient row skip with a temperature (a distinct per-token temperature on every row, or the scalar
    0.7; ignored rows; zero rows ``+0`` and ``-0``) against the same calls without the skip
    (``set_zero_dlogp_min_rows(None)``), the deterministic package's temperature entry and the FP64 oracle.  Every
    call with the skip runs the partition (binding and remembered binding) and binds the counted weight-gradient
    GEMMs; its backward plan runs the ``dz`` arm's counted temperature form ``gemm_logits_dz_dcnt_tau``
    (``dz_form="counted"``, the default, where the record registers it) or the uncounted ``gemm_logits_dz_tau``
    (``"mixed"``: ``set_zero_dlogp_temperature_dz_counted(False)``, which binds no counted form), never
    ``gemm_logits_dz_dcnt`` (the forward's runner, prepared before ``dlogp`` is known, binds the uncounted form in
    both: bound, not launched by the backward under the skip).  ``logp`` keeps
    its bits; the zero rows' and the ignored rows' ``dX`` rows are exact ``+0``; with one dX slice pinned
    (``_TAU_ONE``) a live row's ``dX`` -- the same product of its ``dz`` row -- keeps the bits of the call without the
    skip (the ``dz`` of the live rows is unchanged: the temperature rows follow the skip's row order).  ``dW`` and the
    default-slice ``dX`` stay within the ND budget floors of the call without the skip and of the deterministic
    package (``dW`` the same bits when no row is zero) and pass the gates against the FP64 oracle; every row zero
    gives zero gradients.  A distinct temperature also keeps its bits with extreme values on the ignored rows.  T 5000
    runs the shipped chunks (``shipped_chunks``: its two-chunk cast forms); T 4600 the short-tail fold."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    if T == 5000:
        request.getfixturevalue("shipped_chunks")
    # the counted path of the default Function: a call of at most uncompacted_max_rows() rows would run uncompacted
    # (no row skip), one of at most SMALL_T_KEEP_MAX_ROWS rows the kept logits
    monkeypatch.setitem(cake_backend._UNCOMPACTED_MAX_ROWS, "rows", 0)
    monkeypatch.setitem(cake_backend._SMALL_T_KEEP, "on", False)
    base = _tau_inputs(T, W=glm_weight)
    zero = _zero_dlogp_set(base, kind)
    tau = _distinct_tau(T) if form == "distinct" else 0.7
    inp = dataclasses.replace(_with_zero_dlogp(base, zero), temperature=tau)
    valid = inp.labels >= 0
    live = valid & ~zero
    rows = int(valid.sum())
    case = (T, kind, form, dz_form)
    module_name, _ = cake_backend.record_for(CUDA)
    counted_dz = dz_form == "counted" and (
        cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE
        in cake_jit.registered_stages_with_temperature(module_name)
    )
    dz_stage = "gemm_logits_dz_dcnt_tau" if counted_dz else "gemm_logits_dz_tau"
    calls = _counted_partition(monkeypatch)
    bound = _spy_bound_stages(monkeypatch)
    kw = dict(entry="logprob", backend="cake")
    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(0), _tau_dz_counted(dz_form == "counted"):
        one = _run(inp, 4096, **kw, **_TAU_ONE)
        again = _run(inp, 4096, **kw, **_TAU_ONE)  # the remembered binding
        sliced = _run(inp, 4096, **kw)
    assert calls == [rows] * 3, (case, calls)
    _assert_temperature_plan(bound, hidden=_expect_hidden(inp, 4096, {}), case=case)
    assert any(s.startswith("gemm_dw") and s.endswith("_dcnt") for s in bound), case
    # the backward binds its plan's dz form; the forward's runner (no skip yet) binds the uncounted one in both forms,
    # so the bound set shows the counted form's presence / absence and the plans below which form the backward runs
    bound_set = sorted(set(bound))
    assert dz_stage in bound, (case, bound_set)
    assert "gemm_logits_dz_dcnt" not in bound, (case, bound_set)
    assert counted_dz or "gemm_logits_dz_dcnt_tau" not in bound, (case, bound_set)
    skip_plans = _logprob_bwd_plans()
    assert len(skip_plans) == 2, case
    for plan, keys in skip_plans:
        assert plan.temperature and plan.dcnt, case
        assert plan.counted_dz == counted_dz, case
        assert {plan.recompute_stage(i) for i in range(plan.num_chunks)} == {dz_stage}
        # the backward's launch sequence: one dz launch per chunk, every one in the plan's form
        dz_run = [k[0] for k in keys if k[0].startswith("gemm_logits_dz")]
        assert dz_run == [dz_stage] * plan.num_chunks, (case, dz_run)
    if T == 5000:
        module_name, record = cake_backend.record_for(CUDA)
        tma = cake_backend.dw_cast_tma_rule(2, rows - 4096, record["arch"])
        counted_tma = cake_backend.DW_CAST_TMA_DCNT_STAGE in cake_jit.registered_stages(
            module_name
        )
        for plan, _keys in skip_plans:
            assert plan.num_chunks == 2 and plan.chunks[-1][1] == rows - 4096
            assert plan.live_cast == _lc_registered(module_name), case
            assert plan.dw_cast_stage == _lc_form(
                cake_backend.DW_CAST_TMA_DCNT_STAGE
                if tma and counted_tma
                else "gemm_dw_cast_bf16_dcnt",
                plan,
            ), case
    calls.clear()
    bound.clear()
    with _zero_dlogp_floor(None):
        one_off = _run(inp, 4096, **kw, **_TAU_ONE)
        sliced_off = _run(inp, 4096, **kw)
    assert not calls and not any(s.endswith("_dcnt") for s in bound), case
    det = _run(inp, 4096, **kw, package="det")
    for run in (one, again, sliced):
        assert _same_bits(run["logp"], one_off["logp"]), case
        assert _same_bits(run["logp"], det["logp"]), case
        assert not bool(_bits(run["dX"][zero | ~valid]).any()), case
    for run in (one, again):
        assert _same_bits(run["dX"][live], one_off["dX"][live]), case
    assert _same_bits(again["dW"], one["dW"]), case
    if not bool(live.any()):
        for run in (one, sliced):
            assert not bool(run["dW"].float().any()), case
        return
    if not bool(zero.any()):
        # no zero row: the partition keeps the forward's order, the counted forms run every row
        for run, off in ((one, one_off), (sliced, sliced_off), (sliced, det)):
            assert _same_bits(run["dW"], off["dW"]), case
    dw_rel = {}
    for name, run, ref in (
        ("one", one, one_off),
        ("sliced", sliced, sliced_off),
        ("det", sliced, det),
    ):
        dw_rel[name] = float(rel_l2(run["dW"].float(), ref["dW"].float()))
        assert dw_rel[name] <= GATE_TINY["dW_rel_l2"], (case, name, dw_rel[name])
    _assert_tau_close(sliced, sliced_off, (case, "skip off"), keys=("logp", "dX"))
    _assert_tau_close(sliced, det, (case, "det"), keys=("logp", "dX"))
    oracle = reference_fp64(inp, entry="logprob")
    b0 = error_report(reference_unchunked(inp, entry="logprob"), oracle, inp.labels)
    for run in (one, sliced):
        errors = error_report(run, oracle, inp.labels)
        assert not errors["nan"] and errors["logp_ignored_zero"], case
        assert errors["dX_ignored_zero"], case
        for key, tiny in GATE_TINY.items():
            if key in errors:
                assert errors[key] <= GATE_MARGIN * b0[key] + tiny, (case, key)
    if form == "distinct" and bool((~valid).any()):
        # the ignored rows' temperatures reach no output (the skip gathers the valid rows only)
        extreme = tau.clone()
        ignored = (~valid).nonzero().reshape(-1)
        extreme[ignored[0::2]] = 1e-30
        extreme[ignored[1::2]] = 3e38
        with _zero_dlogp_floor(0), _tau_dz_counted(dz_form == "counted"):
            other = _run(
                dataclasses.replace(inp, temperature=extreme), 4096, **kw, **_TAU_ONE
            )
        _assert_same_bits(one, other, _LOGPROB_KEYS)
    print(
        f"\ntemperature zero-dlogp skip {case} zero {int(zero.sum())}/{rows}: dW rel-L2 vs skip off "
        f"{dw_rel['one']:.2e} (one slice) / {dw_rel['sliced']:.2e}, vs det {dw_rel['det']:.2e}"
    )


@pytest.mark.parametrize(
    "T, kind, form",
    [
        (12289, ("random", 0.166), "distinct"),
        (12289, ("random", 0.5), "distinct"),
        (12289, "all", "distinct"),
        (8117, ("random", 0.5), "scalar"),
        # two chunks, the second of about 650 valid rows (the TMA cast's counted form)
        (5000, ("random", 0.5), "distinct"),
    ],
    ids=str,
)
def test_device_temperature_counted_dz_is_the_mixed_plans_bits(
    logprob_chunk_loop, glm_weight, monkeypatch, T, kind, form
):
    """Under the zero-gradient row skip with a temperature, the counted ``dz`` temperature form
    (``gemm_logits_dz_dcnt_tau``) leaves out only work whose result no later launch reads: ``logp``, ``dX`` and ``dW``
    (one dX slice pinned: binding and remembered binding) are the mixed plan's bits
    (``set_zero_dlogp_temperature_dz_counted(False)``: ``gemm_logits_dz_tau`` over every row) -- the same row order,
    the same counted dX / dW forms, the same ``dz`` on every row they read; with the default slices ``logp`` and
    ``dW`` are its bits and ``dX`` -- whose K-sliced reduction adds its slices in the order they finish, so it may
    differ from run to run by FP32 rounding -- stays within the dX budget floor (:func:`_assert_tau_close`)."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    module_name, _ = cake_backend.record_for(CUDA)
    if (
        cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE
        not in cake_jit.registered_stages_with_temperature(module_name)
    ):
        pytest.skip("this device's record registers no counted dz temperature form")
    base = _tau_inputs(T, W=glm_weight)
    zero = _zero_dlogp_set(base, kind)
    tau = _distinct_tau(T) if form == "distinct" else 0.7
    inp = dataclasses.replace(_with_zero_dlogp(base, zero), temperature=tau)
    case = (T, kind, form)
    bound = _spy_bound_stages(monkeypatch)
    kw = dict(entry="logprob", backend="cake")
    runs = {}
    for counted in (True, False):
        cake_backend.BINDING_CACHE.clear()
        bound.clear()
        with _zero_dlogp_floor(0), _tau_dz_counted(counted):
            runs[counted] = [
                _run(inp, 4096, **kw, **_TAU_ONE),
                _run(inp, 4096, **kw, **_TAU_ONE),  # the remembered binding
                _run(inp, 4096, **kw),
            ]
        dz_stage = "gemm_logits_dz_dcnt_tau" if counted else "gemm_logits_dz_tau"
        assert dz_stage in bound, (case, counted, sorted(set(bound)))
        plans = _logprob_bwd_plans()
        assert len(plans) == 2, (case, counted)
        # the backward's launch sequence: one dz launch per chunk, in this plan's form
        for plan, keys in plans:
            dz_run = [k[0] for k in keys if k[0].startswith("gemm_logits_dz")]
            assert dz_run == [dz_stage] * plan.num_chunks, (case, counted, dz_run)
    names = ("one", "again", "sliced")
    for name, a, b in zip(names, runs[True], runs[False], strict=True):
        if name == "sliced":
            _assert_tau_close(a, b, (case, name), keys=_LOGPROB_KEYS)
        else:
            _assert_same_bits(a, b, _LOGPROB_KEYS)


def test_device_temperature_zero_dlogp_skip_follows_the_row_order(
    glm_weight, monkeypatch
):
    """Negative control of the temperature rows' order under the skip: with a distinct per-token temperature and half
    the valid rows zero, a skip whose temperature rows are gathered in the forward's ascending order instead of the
    skip's row order (patched here) changes the live rows' ``dX`` (one dX slice pinned), which the correct order keeps
    at the bits of the call without the skip -- the check of :func:`test_device_temperature_zero_dlogp_rows_skip`
    detects a wrong permutation."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    T = 8117
    base = _tau_inputs(T, W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.5))
    inp = dataclasses.replace(
        _with_zero_dlogp(base, zero), temperature=_distinct_tau(T)
    )
    live = (inp.labels >= 0) & ~zero
    kw = dict(entry="logprob", backend="cake", **_TAU_ONE)
    cake_backend.BINDING_CACHE.clear()
    with _zero_dlogp_floor(None):
        off = _run(inp, 4096, **kw)
    with _zero_dlogp_floor(0):
        right = _run(inp, 4096, **kw)
    assert _same_bits(right["dX"][live], off["dX"][live])
    real = cake_backend._temperature_rows
    reordered: list = []

    def ascending(form, index, rows, device):
        if form is not None and form.kind == "tensor" and index is not None:
            ordered = index.sort().values
            if not torch.equal(ordered, index):
                reordered.append(int(index.numel()))
            index = ordered
        return real(form, index, rows, device)

    monkeypatch.setattr(cake_backend, "_temperature_rows", ascending)
    with _zero_dlogp_floor(0):
        wrong = _run(inp, 4096, **kw)
    assert reordered, "the skip gathered no temperature rows in its own order"
    assert not _same_bits(wrong["dX"][live], off["dX"][live])


def test_device_temperature_zero_dlogp_skip_adds_no_synchronization(
    logprob_backward_form, glm_weight
):
    """The backward with the skip and a per-token temperature (the partition, the temperature gather in its row order,
    the mixed plan) runs under PyTorch's synchronization debug mode ``"error"``, binding (first call) and from the
    remembered binding (second call)."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    base = _tau_inputs(8117, W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = _with_zero_dlogp(base, zero)
    tau = _distinct_tau(8117)
    cake_backend.BINDING_CACHE.clear()

    def backward_under_error_mode():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
            X, W, inp.labels, chunk_size=4096, backend="cake", temperature=tau
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


def test_device_temperature_zero_dlogp_backward_cuda_graph(
    logprob_backward_form, glm_weight
):
    """The log-probability backward with a per-token temperature under the skip, captured in a CUDA graph (after a
    warm-up that binds it on the capture stream) as the autograd Function calls it (the forward's temperature rows and
    its resolved temperature): replays with other ``dlogp`` values -- other zero rows and live counts per chunk, none
    and every row zero -- written into the captured input give the eager backward's results on those values (``dW``
    within the FP32 reduction order, the zero rows' ``dX`` exact ``+0``)."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    base = _tau_inputs(12289, W=glm_weight)
    tau = _distinct_tau(12289)
    kw = dict(chunk_size=4096)
    fr = cake_backend.forward_logprob_temperature(
        base.X, base.W, base.labels, temperature=tau, **kw
    )
    form = cake_backend._resolve_temperature(
        tau, int(base.labels.numel()), base.X.device
    )
    torch.cuda.synchronize()
    assert fr.row_index is not None and fr.temperature_rows is not None
    dlogp = base.dlogp.clone()

    def backward(d):
        return cake_backend.backward_logprob_temperature(
            base.X,
            base.W,
            base.labels,
            fr.lse,
            d,
            row_index=fr.row_index,
            temperature=form,
            temperature_rows=fr.temperature_rows,
            **kw,
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


def test_device_checkpointed_temperature_zero_dlogp_step_is_bitwise(
    glm_weight, monkeypatch
):
    """The temperature entry under the zero-gradient row skip (a distinct per-token temperature, a third of the valid
    rows with a zero upstream gradient, one dX slice pinned) inside a non-reentrant checkpoint, selective activation
    checkpointing and a dispatch mode (around the checkpointed forward; around the whole step) gives the
    uncheckpointed step's ``dX`` / ``dW`` bits; every step's backward runs the partition once."""
    _require_zero_dlogp_skip()
    _require_tau_program()
    T = 12289
    base = _tau_inputs(T, W=glm_weight)
    zero = _zero_dlogp_set(base, ("random", 0.3))
    inp = dataclasses.replace(
        _with_zero_dlogp(base, zero), temperature=_distinct_tau(T)
    )
    valid = inp.labels >= 0
    calls = _counted_partition(monkeypatch)

    def head(a, b):
        return _tau_region_entry("backend", inp, a, b)

    def step(kind):
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        if kind == "dispatch_mode_step":
            with _PassThroughMode():
                _checkpointed("nonreentrant", head, X, W).backward()
        else:
            _checkpointed(kind, head, X, W).backward()
        torch.cuda.synchronize()
        return X.grad, W.grad

    kinds = (
        "nonreentrant",
        "selective_recompute",
        "selective_save_mm",
        "dispatch_mode",
        "dispatch_mode_step",
    )
    with _zero_dlogp_floor(0):
        step("plain")  # binds
        plain = step("plain")
        calls.clear()
        for kind in kinds:
            run = step(kind)
            assert _same_bits(run[0], plain[0]), kind
            assert _same_bits(run[1], plain[1]), kind
    assert calls == [int(valid.sum())] * len(kinds), calls
    assert not bool(_bits(plain[0][zero | ~valid]).any())


def test_temperature_entries_run_their_function():
    """The temperature entries of both modules run :class:`CakeLMHeadNondetLogprobTemperatureFunction` (they take no
    ``deterministic``: :func:`test_no_entry_takes_a_mode_keyword`)."""
    inp = _host_inputs(8, device=CPU, temperature="hetero")
    X = inp.X.detach().requires_grad_(True)
    for mod in (cake_backend, opt_in):
        logp = mod.cake_lm_head_nondet_train_logprob_temperature(
            X,
            inp.W,
            inp.labels,
            chunk_size=4,
            backend="reference",
            temperature=inp.temperature,
        )
        assert (
            logp.grad_fn._forward_cls.__name__
            == "CakeLMHeadNondetLogprobTemperatureFunction"
        ), mod.__name__


@pytest.mark.parametrize("form", ["scalar", "hetero"])
@pytest.mark.parametrize("compact", [True, False])
def test_reference_temperature_matches_the_deterministic_reference(form, compact):
    """With a temperature the reference backend is the deterministic package's temperature entry bit for bit
    (``logp`` / ``dX`` / ``dW``) and passes the gates against the references."""
    inp = _host_inputs(37, ignore_frac=0.2, temperature=form)
    nd = _run(inp, 16, entry="logprob", compact_rows=compact)
    det = _run(inp, 16, entry="logprob", compact_rows=compact, package="det")
    for key in ("logp", "dX", "dW"):
        assert _same_bits(nd[key], det[key]), key
    _check_against_references(nd, inp, entry="logprob")


def test_temperature_stage_names_round_trip():
    """The temperature forms of the log-probability entry (the logits GEMM arms, the device-count logits GEMM with
    the statistics and ``row_finalize``): suffix ``_tau`` last (after ``_dcnt``, which only the ``dz`` arm's counted
    form has), every form parses back to its base and knobs through :func:`cake_backend.stage_form`, a base without a
    temperature form, a counted form of another base and a misplaced suffix are refused; every stage name of
    ``STAGES_WITH_TEMPERATURE`` round-trips."""
    form = cake_backend.stage_form
    assert form("gemm_logits_lbl", temperature=True) == "gemm_logits_lbl_tau"
    assert form("gemm_logits_dz", temperature=True) == "gemm_logits_dz_tau"
    assert (
        form("gemm_logits_dz", dcnt=True, temperature=True)
        == cake_backend.LOGPROB_DZ_DCNT_TAU_STAGE
        == "gemm_logits_dz_dcnt_tau"
    )
    assert cake_backend.parse_stage("gemm_logits_dz_dcnt_tau")[1]["dcnt"] is True
    assert form("gemm_logits", count=True, temperature=True) == "gemm_logits_mcnt_tau"
    assert form("row_finalize", temperature=True) == "row_finalize_tau"
    assert cake_backend.parse_stage("gemm_logits_mcnt_tau") == (
        "gemm_logits",
        dict(
            count=True,
            tile_n=None,
            stages=None,
            split=False,
            swab=False,
            blocked=False,
            epi_store=None,
            live_cast=False,
            dcnt=False,
            temperature=True,
        ),
    )
    assert (
        tuple(cake_backend.parse_stage("gemm_dx_red_dcnt")[1])
        == cake_backend.STAGE_KNOBS
    )
    for stage in TAU_STAGES:
        base = cake_backend.base_stage(stage)
        suffix = next(
            x for x in ("_mcnt_tau", "_dcnt_tau", "_tau") if stage.endswith(x)
        )
        assert stage == base + suffix
        assert cake_backend.parse_stage(stage)[1]["temperature"] is True
    for stage in cake_jit.STAGES_WITH_TEMPERATURE:
        base, knobs = cake_backend.parse_stage(stage)
        assert form(base, **knobs) == stage
        assert knobs["temperature"] == stage.endswith("_tau"), stage
    for bad in (
        # the temperature form of gemm_logits is the device-count one only
        dict(base="gemm_logits"),
        dict(base="gemm_logits_nostats"),
        dict(base="gemm_logits_nostats", count=True),
        dict(base="gemm_logits_lbl", dcnt=True),
        dict(base="gemm_logits", count=True, dcnt=True),
        dict(base="row_grad"),
        dict(base="loss_reduce"),
        dict(base="gemm_dx"),
        dict(base="gemm_dx_red"),
        dict(base="gemm_dw_acc"),
        dict(base="gather_rows_bf16"),
    ):
        with pytest.raises(ValueError):
            form(bad.pop("base"), temperature=True, **bad)
    for bad_name in (
        "gemm_logits_tau",
        "gemm_logits_nostats_tau",
        "gemm_logits_tau_mcnt",
        "gemm_logits_dz_tau_tau",
        "gemm_logits_dz_tau_dcnt",
        "gemm_logits_dz_dcnt_tau_tau",
        "gemm_logits_lbl_dcnt_tau",
        "gemm_logits_mcnt_dcnt_tau",
        "row_grad_tau",
        "gemm_dx_tau",
        "gemm_dx_red_tau",
        "row_finalize_tau_tma",
    ):
        with pytest.raises(ValueError):
            cake_backend.parse_stage(bad_name)


def _require_temperature_forms():
    """Skip unless the registered program carries the temperature forms and runs the hidden valid-row count."""
    _require_tau_program()
    if not cake_jit.toolchain_runs_hidden_count(cake_backend.arch_for(CUDA)):
        pytest.skip("the hidden valid-row count does not run with this toolchain")


def _logprob_temperature_step(inp, chunk):
    """One log-probability step of this package's temperature entry (one dX slice pinned, ``_TAU_ONE``)."""
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    logp = cake_backend.cake_lm_head_nondet_train_logprob_temperature(
        X,
        W,
        inp.labels,
        chunk_size=chunk,
        temperature=inp.temperature,
        **_TAU_ONE,
    )
    logp.backward(inp.dlogp)
    return dict(logp=logp.detach(), dX=X.grad, dW=W.grad)


def _run_two_threads(first, second):
    """Run ``first`` and ``second`` on the threads named ``first`` / ``second``, each on its own stream of the device;
    re-raise the first error."""
    errors = []

    def wrap(fn):
        def run():
            try:
                with torch.cuda.device(CUDA), torch.cuda.stream(torch.cuda.Stream()):
                    fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised in the test thread
                errors.append(exc)

        return run

    threads = [
        threading.Thread(target=wrap(fn), name=fn.__name__) for fn in (first, second)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT_S)
    assert not any(t.is_alive() for t in threads), "a thread did not finish"
    torch.cuda.synchronize()
    if errors:
        raise errors[0]


@pytest.mark.parametrize(
    "T, chunk", [(4097, 4096), (1000, 256)]
)  # the host count (two chunks), the hidden count (four chunks)
def test_device_logprob_temperature_steps_hit_one_backward_binding(
    glm_weight, T, chunk
):
    """Training steps of the log-probability entry with a per-token temperature reuse one forward and one backward
    binding (the temperature path keys ``lse`` and ``dlogp`` by layout, as the path without one): no step after the
    first misses, every step allocates the same number of blocks, and the bits are the uncached call's (one dX slice
    pinned)."""
    _require_temperature_forms()
    inp = make_inputs(T, seed=SEED + T, device=CUDA, W=glm_weight, temperature="hetero")
    cache = cake_backend.BINDING_CACHE
    cache.clear()
    out = _logprob_temperature_step(inp, chunk)  # binds (misses once per direction)
    out = _logprob_temperature_step(inp, chunk)
    torch.cuda.synchronize()
    misses, allocations = cache.misses, []
    for _ in range(6):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = _logprob_temperature_step(inp, chunk)
        torch.cuda.synchronize()
        allocations.append(
            torch.cuda.memory_stats()["allocation.all.allocated"] - before
        )
    assert cache.misses == misses, (cache.misses - misses, allocations)
    assert len(set(allocations)) == 1, allocations
    with _cache(False):
        fresh = _logprob_temperature_step(inp, chunk)
    for key, value in fresh.items():
        assert torch.equal(_bits(out[key]), _bits(value)), key


def test_device_temperature_concurrent_threads(glm_weight, monkeypatch):
    """Log-probability steps with a per-token temperature and ignored rows on two threads, each on its own stream
    (four chunks: the hidden count with the temperature's device-count form and the dW side stream run).  The first
    thread's call waits between its count's copy and its read until the second thread's whole step returned.  Both
    give the single-thread bits (one dX slice pinned; one shared count cell would hand the first call the second's
    count)."""
    _require_temperature_forms()
    C, T = 256, 1000
    inputs = {
        name: make_inputs(
            T,
            seed=seed,
            device=CUDA,
            W=glm_weight,
            ignore_frac=ignore,
            temperature="hetero",
        )
        for name, seed, ignore in (
            ("first", SEED + 15, 0.05),
            ("second", SEED + 16, 0.40),
        )
    }
    counts = {name: int(i.labels.ge(0).sum()) for name, i in inputs.items()}
    assert counts["second"] < counts["first"]  # the other count never indexes past
    ref = {
        name: {
            k: _bits(v).clone() for k, v in _logprob_temperature_step(inp, C).items()
        }
        for name, inp in inputs.items()
    }
    torch.cuda.synchronize()
    begun, read = threading.Barrier(2), threading.Barrier(2)
    finish = cake_backend._HiddenCount.finish
    paused = []

    def finish_after_the_other_step(self):
        if threading.current_thread().name == "first" and not paused:
            paused.append(True)
            begun.wait(TIMEOUT_S)
            read.wait(TIMEOUT_S)
        return finish(self)

    monkeypatch.setattr(
        cake_backend._HiddenCount, "finish", finish_after_the_other_step
    )
    got = {}

    def first():
        got["first"] = _logprob_temperature_step(inputs["first"], C)

    def second():
        begun.wait(TIMEOUT_S)
        got["second"] = _logprob_temperature_step(inputs["second"], C)
        torch.cuda.current_stream().synchronize()
        read.wait(TIMEOUT_S)

    _run_two_threads(first, second)
    assert paused == [True]
    wrong = [
        (name, key)
        for name in ref
        for key in ref[name]
        if not torch.equal(_bits(got[name][key]), ref[name][key])
    ]
    assert not wrong, wrong


# ---------------------------------------------------------------------------
# Vocabulary-major log-probability backward (multi-chunk calls; the internal rule ``_set_logprob_vocab_major``)
# ---------------------------------------------------------------------------


def test_vocab_major_mode_and_block_rows():
    """The mode setter normalises like ``dw_stream`` (``None`` / ``"auto"``, ``True`` / ``"1"``, ``False`` / ``"0"``) and
    refuses anything else; a block holds at most one chunk's logits (``rows * block <= chunk * V``, at least one
    256-row granule), its rows are a multiple of the granule, the blocks cover the vocabulary and are balanced (the
    last block is at most one granule per block shorter than the others)."""
    assert cake_backend._logprob_vocab_major() == "auto"
    with _vocab_major(False):
        assert cake_backend._logprob_vocab_major() == "0"
        assert cake_backend._set_logprob_vocab_major(True) == "0"
        assert cake_backend._logprob_vocab_major() == "1"
        assert cake_backend._set_logprob_vocab_major(None) == "1"
        assert cake_backend._logprob_vocab_major() == "auto"
        with pytest.raises(ValueError):
            cake_backend._set_logprob_vocab_major("yes")
    assert cake_backend._logprob_vocab_major() == "auto"
    for rows, V, C in (
        (15420, 154880, 4096),
        (4370, 154880, 4096),
        (61780, 154880, 4096),
        (31130, 248320, 4096),
        (65535, 512, 64),
        (5000, 32768, 4096),
    ):
        vb = cake_backend._logprob_vm_block_rows(rows, V, C)
        n = -(-V // vb)
        assert vb % 256 == 0 and vb >= 256, (rows, V, C, vb)
        assert rows * vb <= C * V or vb == 256, (rows, V, C, vb)
        assert (n - 1) * vb < V <= n * vb, (rows, V, C, vb)
        assert n * vb - V < 256 * n, (rows, V, C, vb)
    assert cake_backend._logprob_vm_block_rows(15420, 154880, 4096) == 38912


@pytest.mark.parametrize("tau", [None, "scalar", "hetero"])
@pytest.mark.parametrize("dead", [0.0, 0.5])
def test_device_vocab_major_logprob_backward(glm_weight, monkeypatch, tau, dead):
    """A log-probability call of 2 .. 4 chunks takes the vocabulary-major backward (mode ``"auto"``; ``"0"`` runs the
    chunk loop; a call of more than :data:`cake_backend.VM_AUTO_MAX_CHUNKS` chunks keeps the chunk loop under
    ``"auto"`` and takes the form under ``"1"``): ``logp`` is the chunk loop's bit for bit (the forward is unchanged), ``dX`` and ``dW`` repeat bit for bit
    (three runs), pass the suite's unchanged accuracy gates against the FP64 oracle like the chunk loop's, the rows of a
    zero ``dlogp`` (the zero-gradient row skip at ``dead`` 0.5) get exact-zero ``dX`` rows, a non-reentrant
    checkpointed step and a frozen ``X`` give the plain step's ``dX`` / ``dW`` bits, and the kept logits (no
    temperature) keep their chunk loop: the default call's ``logp`` bits, ``dX`` / ``dW`` to the same gates -- with no
    temperature, a scalar and a per-token one."""
    _require_program(entry="logprob")
    calls = []
    real = cake_backend._backward_logprob_vm

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_backward_logprob_vm", counted)
    assert cake_backend.VM_AUTO_MAX_CHUNKS == 4
    for T, C in ((6000, 4096), (7000, 2048)):
        inp = make_inputs(
            T,
            seed=SEED + 4100 + T,
            device=CUDA,
            W=glm_weight,
            ignore_frac=0.05,
            temperature=tau,
        )
        if dead:
            gen = torch.Generator().manual_seed(SEED + T)
            zero = (torch.rand(T, generator=gen) < dead).to(CUDA)
            inp = dataclasses.replace(
                inp, dlogp=torch.where(zero, torch.zeros_like(inp.dlogp), inp.dlogp)
            )
        case = (T, C, tau, dead)
        with _vocab_major("0"):
            calls.clear()
            loop = _run(inp, C, entry="logprob", backend="cake")
            assert not calls, case
        calls.clear()
        runs = [_run(inp, C, entry="logprob", backend="cake") for _ in range(3)]
        assert len(calls) == 3, case
        vm = runs[0]
        assert _same_bits(vm["logp"], loop["logp"]), case
        for other in runs[1:]:
            assert _same_bits(vm["dX"], other["dX"]), case
            assert _same_bits(vm["dW"], other["dW"]), case
        _check_against_references(vm, inp, entry="logprob")
        dead_rows = inp.dlogp == 0
        assert torch.all(vm["dX"][dead_rows] == 0), case
        # a non-reentrant checkpointed step and a frozen X: the same backward, the same bits
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        kw = dict(chunk_size=C, backend="cake")
        fn = cake_backend.cake_lm_head_nondet_train_logprob
        if inp.temperature is not None:
            kw["temperature"] = inp.temperature
            fn = cake_backend.cake_lm_head_nondet_train_logprob_temperature
        L = torch.utils.checkpoint.checkpoint(
            lambda a, b: (fn(a, b, inp.labels, **kw) * inp.dlogp).sum(),
            X,
            W,
            use_reentrant=False,
        )
        dX, dW = torch.autograd.grad(L, (X, W))
        assert _same_bits(dX, vm["dX"]) and _same_bits(dW, vm["dW"]), case
        frozen = _run(inp, C, entry="logprob", backend="cake", train_x=False)
        assert frozen["dX"] is None and _same_bits(frozen["dW"], vm["dW"]), case
        if tau is None:
            # the kept logits keep their own chunk loop: the default call's logp bits; dX / dW round differently
            # (another summation order) and pass the same unchanged gates
            calls.clear()
            kept = _run(inp, C, entry="logprob", backend="cake", keep=True)
            assert not calls, case
            assert _same_bits(kept["logp"], vm["logp"]), case
            _check_against_references(kept, inp, entry="logprob")
            dead_rows = inp.dlogp == 0
            assert torch.all(kept["dX"][dead_rows] == 0), case
    # five chunks (8265 valid rows over 2048-row chunks): the auto rule keeps the chunk loop, mode "1" runs the form
    inp = make_inputs(
        8700,
        seed=SEED + 4100 + 8700,
        device=CUDA,
        W=glm_weight,
        ignore_frac=0.05,
        temperature=tau,
    )
    assert int((inp.labels >= 0).sum()) > 4 * 2048
    calls.clear()
    auto = _run(inp, 2048, entry="logprob", backend="cake")
    assert not calls
    with _vocab_major("1"):
        forced = [_run(inp, 2048, entry="logprob", backend="cake") for _ in range(2)]
    assert len(calls) == 2
    assert _same_bits(forced[0]["logp"], auto["logp"])
    assert _same_bits(forced[0]["dX"], forced[1]["dX"])
    assert _same_bits(forced[0]["dW"], forced[1]["dW"])
    _check_against_references(auto, inp, entry="logprob")
    _check_against_references(forced[0], inp, entry="logprob")
    # two chunks whose second chunk holds at least half a chunk (about 7600 valid rows over 4096-row chunks): the auto
    # rule keeps the chunk loop
    inp = make_inputs(
        8000,
        seed=SEED + 4100 + 8000,
        device=CUDA,
        W=glm_weight,
        ignore_frac=0.05,
        temperature=tau,
    )
    assert 4096 + 2048 <= int((inp.labels >= 0).sum()) <= 2 * 4096
    calls.clear()
    two = _run(inp, 4096, entry="logprob", backend="cake")
    assert not calls
    _check_against_references(two, inp, entry="logprob")


# ---------------------------------------------------------------------------
# Small calls: staged rows without a valid-row count at T <= uncompacted_max_rows(), the small-call kept logits (on by
# default), the small-call padding
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _uncompacted_max_rows(rows: int):
    previous = cake_backend.set_uncompacted_max_rows(rows)
    cake_backend.BINDING_CACHE.clear()
    try:
        yield
    finally:
        cake_backend.set_uncompacted_max_rows(previous)
        cake_backend.BINDING_CACHE.clear()


@contextlib.contextmanager
def _small_t_keep(on: bool):
    previous = cake_backend.set_small_t_keep(on)
    try:
        yield
    finally:
        cake_backend.set_small_t_keep(previous)


@contextlib.contextmanager
def _small_pad_rows(rows: int):
    previous = cake_backend.set_small_pad_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_small_pad_rows(previous)


@contextlib.contextmanager
def _shipped_zero_weight_floor():
    previous = cake_backend.set_zero_weight_min_rows(
        cake_backend.ZERO_WEIGHT_MIN_ROWS_DEFAULT
    )
    try:
        yield
    finally:
        cake_backend.set_zero_weight_min_rows(previous)


_ONE_SLICE = {"dx": {"k_slices": 1}}


def test_small_call_switches_validate_and_restore():
    """The small-call switches return their previous values, reject bad values, and default to the shipped values
    (256 rows, the small-call kept logits on, padding to 16 rows -- 256 rows for the log-probability entries on
    sm_107a); a padding row count set with :func:`set_small_pad_rows` applies to every entry until ``None`` restores
    the defaults."""
    meta = torch.device("meta")
    assert (
        cake_backend.uncompacted_max_rows()
        == cake_backend.UNCOMPACTED_MAX_ROWS_DEFAULT
        == 256
    )
    assert cake_backend.small_t_keep() is True
    assert cake_backend.small_pad_rows() == cake_backend.SMALL_PAD_ROWS_DEFAULT == 16
    assert cake_backend.small_pad_rows("logprob", meta) == 16
    assert cake_backend.SMALL_PAD_ROWS_LOGPROB_BY_ARCH == {"sm_107a": 256}
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            cake_backend.set_uncompacted_max_rows(bad)
    for bad in (1, None, "on"):
        with pytest.raises(ValueError):
            cake_backend.set_small_t_keep(bad)
    for bad in (-1, 1.5, True, "32"):
        with pytest.raises(ValueError):
            cake_backend.set_small_pad_rows(bad)
    with pytest.raises(ValueError):
        cake_backend.small_pad_rows("temperature")
    assert cake_backend.set_uncompacted_max_rows(7) == 256
    assert cake_backend.set_uncompacted_max_rows(256) == 7
    assert cake_backend.set_small_t_keep(False) is True
    assert cake_backend.set_small_t_keep(True) is False
    assert cake_backend.set_small_pad_rows(0) is None
    assert cake_backend.small_pad_rows() == cake_backend.small_pad_rows("logprob") == 0
    assert cake_backend.set_small_pad_rows(32) == 0
    assert cake_backend.small_pad_rows("logprob", meta) == 32
    assert cake_backend.set_small_pad_rows(None) == 32
    assert cake_backend.small_pad_rows() == 16


def test_small_call_rule_is_host_only(monkeypatch):
    """The small-call rule reads host values alone (meta tensors and attribute stand-ins here): a 2-D BF16 CUDA ``X``
    with int64 ``[T]`` labels on its device, ``backend="cake"``, compacted, ``0 < T <= min(uncompacted_max_rows(),
    chunk_size)`` stages ``T`` rows, or the entry's ``small_pad_rows(...)`` when ``T`` is below it and the padded count
    fits (the log-probability entries on sm_107a: 256 rows, the loss entry 16); a policy call at the zero-weight floor,
    a per-row operand that is not an FP32 ``[T]`` tensor on that device or requires a gradient, and every other call
    stage nothing."""
    meta = torch.device("meta")

    def X(T, **kw):
        return SimpleNamespace(
            **dict(
                dict(
                    ndim=2,
                    dtype=torch.bfloat16,
                    is_cuda=True,
                    device=meta,
                    shape=(T, 8),
                ),
                **kw,
            )
        )

    def lab(T, **kw):
        return SimpleNamespace(
            **dict(dict(ndim=1, dtype=torch.int64, device=meta, shape=(T,)), **kw)
        )

    def row(T, dtype=torch.float32):
        return torch.empty((T,), dtype=dtype, device=meta)

    rule = cake_backend._small_call_rows
    with _small_pad_rows(0):
        assert rule(X(32), lab(32), "cake", 4096, None) == 32
        assert rule(X(1), lab(1), "cake", 4096, True) == 1
        assert rule(X(256), lab(256), "cake", 4096, None) == 256
        assert rule(X(257), lab(257), "cake", 4096, None) == 0
        assert rule(X(0), lab(0), "cake", 4096, None) == 0
        assert rule(X(32), lab(32), "cake", 16, None) == 0
        assert rule(X(32), lab(32), "reference", 4096, None) == 0
        assert rule(X(32), lab(32), "cake", 4096, False) == 0
        assert rule(X(32, is_cuda=False), lab(32), "cake", 4096, None) == 0
        assert rule(X(32, dtype=torch.float32), lab(32), "cake", 4096, None) == 0
        assert rule(X(32, ndim=3), lab(32), "cake", 4096, None) == 0
        assert rule(X(32), lab(31), "cake", 4096, None) == 0
        assert rule(X(32), lab(32, dtype=torch.int32), "cake", 4096, None) == 0
        assert rule(X(32), lab(32, device=torch.device("cpu")), "cake", 4096, None) == 0
        assert rule(X(32), lab(32), "cake", "4096x", None) == 0
        with _uncompacted_max_rows(0):
            assert rule(X(1), lab(1), "cake", 4096, None) == 0
        policy = dict(objective="policy", per_row=(row(32), row(32)))
        with _shipped_zero_weight_floor():
            assert rule(X(32), lab(32), "cake", 4096, None, **policy) == 32
            for bad in (row(31), row(32, torch.float16), row(32).requires_grad_(True)):
                per_row = (row(32), bad)
                assert (
                    rule(
                        X(32),
                        lab(32),
                        "cake",
                        4096,
                        None,
                        objective="policy",
                        per_row=per_row,
                    )
                    == 0
                )
        previous = cake_backend.set_zero_weight_min_rows(32)
        try:
            assert rule(X(32), lab(32), "cake", 4096, None, **policy) == 0
            assert rule(X(32), lab(32), "cake", 4096, None, per_row=(row(32),)) == 32
        finally:
            cake_backend.set_zero_weight_min_rows(previous)
        assert rule(X(32), lab(32), "cake", 4096, None, per_row=(2.0, None)) == 32
    with _small_pad_rows(16):
        assert rule(X(1), lab(1), "cake", 4096, None) == 16
        assert rule(X(15), lab(15), "cake", 4096, None) == 16
        assert rule(X(16), lab(16), "cake", 4096, None) == 16
        assert rule(X(17), lab(17), "cake", 4096, None) == 17
        assert rule(X(1), lab(1), "cake", 8, None) == 1
        with _uncompacted_max_rows(8):
            assert rule(X(1), lab(1), "cake", 4096, None) == 1
        previous = cake_backend.set_zero_weight_min_rows(16)
        try:
            assert (
                rule(
                    X(1),
                    lab(1),
                    "cake",
                    4096,
                    None,
                    objective="policy",
                    per_row=(row(1), row(1)),
                )
                == 1
            )
        finally:
            cake_backend.set_zero_weight_min_rows(previous)
    # the shipped defaults per entry and architecture (the architecture read from X's device)
    with _small_pad_rows(None):
        assert rule(X(1), lab(1), "cake", 4096, None) == 16
        assert rule(X(1), lab(1), "cake", 4096, None, logprob=True) == 16
        monkeypatch.setattr(cake_backend, "arch_for", lambda device=None: "sm_107a")
        assert rule(X(1), lab(1), "cake", 4096, None) == 16
        assert rule(X(1), lab(1), "cake", 4096, None, logprob=True) == 256
        assert rule(X(255), lab(255), "cake", 4096, None, logprob=True) == 256
        assert rule(X(16), lab(16), "cake", 4096, None, logprob=True) == 256
        assert rule(X(1), lab(1), "cake", 128, None, logprob=True) == 1
        with _uncompacted_max_rows(128):
            assert rule(X(1), lab(1), "cake", 4096, None, logprob=True) == 1
        tau = (row(1),)
        assert rule(X(1), lab(1), "cake", 4096, None, per_row=tau, logprob=True) == 256
        monkeypatch.setattr(cake_backend, "arch_for", lambda device=None: "sm_100a")
        assert rule(X(1), lab(1), "cake", 4096, None, logprob=True) == 16


@pytest.mark.parametrize("rows", [5, 16])
def test_device_small_call_stage_is_one_block(rows):
    """The staged inputs of a small call are views of one allocation: ``X`` with its ignored rows zero (an infinite
    and a NaN row here) and zero padded rows, labels padded with ``-100``, ``infer_logp`` / ``loss_weights`` zero on the
    ignored and padded rows, a temperature padded with ones; the valid rows keep their bits.  Unpadded (``rows == T``)
    the labels and the temperature come back as is.  The backward hands back ``dX``'s first ``T`` rows."""
    _require_program(entry="loss")
    T, H = 5, 64
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    X = torch.randn((T, H), generator=gen).to(CUDA, torch.bfloat16)
    X[1] = float("inf")
    X[3] = float("nan")
    X.requires_grad_(True)
    labels = torch.tensor([3, -100, 7, -100, 0], dtype=torch.int64, device=CUDA)
    infer = torch.tensor([1.0, float("nan"), 2.0, float("inf"), 3.0], device=CUDA)
    weights = torch.tensor([0.5, float("inf"), -1.0, float("nan"), 2.0], device=CUDA)
    tau = torch.tensor([0.5, 2.0, 1.5, 0.25, 4.0], device=CUDA)
    xs, lab, inf, wts, tau_s = cake_backend._SmallCallStage.apply(
        X, labels, rows, infer, weights, tau
    )
    valid = labels >= 0
    staged = [xs, inf, wts] + ([lab, tau_s] if rows > T else [])
    base = xs.untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr() == base for t in staged)
    assert xs.shape == (rows, H) and xs.is_contiguous() and xs.requires_grad
    assert torch.equal(_bits(xs[:T][valid]), _bits(X.detach()[valid]))
    assert not bool(_bits(xs[:T][~valid]).any()) and not bool(_bits(xs[T:]).any())
    for got, src in ((inf, infer), (wts, weights)):
        assert got.shape == (rows,)
        assert torch.equal(_bits(got[:T][valid]), _bits(src[valid]))
        assert not bool(_bits(got[:T][~valid]).any()) and not bool(_bits(got[T:]).any())
    if rows > T:
        assert lab.tolist() == labels.tolist() + [-100] * (rows - T)
        assert torch.equal(_bits(tau_s[:T]), _bits(tau)) and bool(
            (tau_s[T:] == 1.0).all()
        )
    else:
        assert (
            lab.data_ptr() == labels.data_ptr() and tau_s.data_ptr() == tau.data_ptr()
        )
    assert not (lab.requires_grad or inf.requires_grad or wts.requires_grad)
    upstream = torch.randn((rows, H), generator=gen).to(CUDA, torch.bfloat16)
    (dx,) = torch.autograd.grad(xs, X, grad_outputs=upstream)
    assert torch.equal(_bits(dx), _bits(upstream[:T]))
    scalar = cake_backend._SmallCallStage.apply(X, labels, rows, None, None, 0.7)
    assert scalar[2] is None and scalar[3] is None and scalar[4] == 0.7


@pytest.mark.parametrize("T", [1, 32, 256])
@pytest.mark.parametrize(
    "entry, objective", [("loss", "ce"), ("loss", "policy"), ("logprob", "ce")]
)
def test_device_small_call_all_valid_is_the_counted_call(
    glm_weight, T, entry, objective
):
    """Every row valid: a call of ``T <= uncompacted_max_rows()`` runs the counted call's launches and gives its bits
    (``loss`` / ``logp`` / ``dW``; ``dX`` with one dX slice pinned) -- policy at the shipped zero-weight floor, small-call
    padding off (it changes the launches of a call under its row count), the default log-probability Function (the
    small-call kept logits off)."""
    _require_program(entry=entry)
    inp = _device_inputs(T, objective, W=glm_weight, ignore_frac=0.0)
    kw = dict(backend="cake", entry=entry, gemm_tuning=_ONE_SLICE)
    with _shipped_zero_weight_floor(), _small_pad_rows(0), _small_t_keep(False):
        with _uncompacted_max_rows(0), _launch_record() as counted_keys:
            counted = _run(inp, 4096, **kw)
        with _uncompacted_max_rows(256), _launch_record() as small_keys:
            small = _run(inp, 4096, **kw)
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(small[key], counted[key]), key
    assert small_keys == counted_keys


def _small_call_step(X, W, labels, entry, *, loss_div=None, dlogp=None, **switches):
    """Forward + ``torch.autograd.grad`` of a public backend entry without any host synchronization of its own."""
    if entry == "loss":
        loss, logp = cake_backend.cake_lm_head_nondet_train_loss(
            X,
            W,
            labels,
            objective="ce",
            loss_div=loss_div,
            chunk_size=4096,
            return_logp=True,
            backend="cake",
            **switches,
        )
        dx, dw = torch.autograd.grad(loss, (X, W))
        return loss, logp, dx, dw
    logp = cake_backend.cake_lm_head_nondet_train_logprob(
        X, W, labels, chunk_size=4096, backend="cake", **switches
    )
    dx, dw = torch.autograd.grad(logp, (X, W), grad_outputs=dlogp)
    return None, logp, dx, dw


@pytest.mark.parametrize("entry", ["loss", "logprob"])
@pytest.mark.parametrize("ignore_frac", [0.0, 0.25, 1.0])
def test_device_small_call_has_no_host_synchronization(glm_weight, entry, ignore_frac):
    """A remembered small call (T = 32: forward and backward of the autograd entry, the log-probability entry on its
    shipped small-call kept logits) synchronizes nothing: PyTorch's synchronization debug mode at ``"error"`` raises
    nothing, with ignored rows and with every row ignored too; the counted call (``uncompacted_max_rows`` 0) raises
    there (its valid-row readback), so the check sees the readback."""
    _require_program(entry=entry)
    inp = _device_inputs(32, "ce", W=glm_weight, ignore_frac=ignore_frac)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def step():
        return _small_call_step(
            X, W, inp.labels, entry, loss_div=inp.loss_div, dlogp=inp.dlogp
        )

    for rows, syncs in ((256, False), (0, True)):
        with _uncompacted_max_rows(rows):
            for _ in range(2):
                step()
            torch.cuda.synchronize()
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            raised = None
            try:
                step()
            except RuntimeError as exc:
                raised = exc
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            torch.cuda.synchronize()
        assert (raised is not None) == syncs, (rows, raised)


@pytest.mark.parametrize("T", [32, 256])
@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_small_call_with_ignored_rows_meets_the_gates(glm_weight, T, entry):
    """Ignored rows in a small call (staged, uncompacted, no count): the deterministic suite's gates against the FP64
    oracle with exact zeros on the ignored rows, and the counted call's ``logp`` bits on every row."""
    _require_program(entry=entry)
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.25)
    with _uncompacted_max_rows(256):
        small = _run(inp, 4096, backend="cake", entry=entry)
    _check_against_references(small, inp, entry=entry, ceiling=True)
    with _uncompacted_max_rows(0):
        counted = _run(inp, 4096, backend="cake", entry=entry)
    assert _same_bits(small["logp"], counted["logp"])


def _non_finite_ignored_rows(inp, value):
    """``inp`` with ``value`` in every ignored row of ``X``, of the policy operands and of the incoming ``dlogp``."""
    ignored = inp.labels < 0
    fields = {}
    for name in ("X", "infer_logp", "loss_weights", "dlogp"):
        t = getattr(inp, name)
        if t is None:
            continue
        t = t.detach().clone()
        t[ignored] = value
        fields[name] = t
    return dataclasses.replace(inp, **fields)


@pytest.mark.parametrize("value", ["inf", "nan"])
@pytest.mark.parametrize("T", [5, 32, 256])
@pytest.mark.parametrize(
    "entry, objective, form",
    [
        ("loss", "ce", None),
        ("loss", "policy", None),
        ("logprob", "ce", None),
        ("logprob", "ce", "nokeep"),
        ("logprob", "ce", "temperature"),
    ],
)
def test_device_small_call_non_finite_ignored_rows_reach_no_output(
    glm_weight, value, T, entry, objective, form
):
    """A small call whose ignored rows hold ``inf`` / NaN -- in ``X``, in the policy operands and in the incoming
    ``dlogp`` -- gives the bits of the same call with zero ``X`` rows there (``loss`` / ``logp`` / ``dW`` and ``dX`` with
    one dX slice pinned), every output finite and the ignored rows' ``dX`` exact zeros: the staged rows keep such a
    value out of every kernel, as the compacted call does (it never reads an ignored row).  Covers the loss entry (CE,
    policy), the log-probability entry on its small-call kept logits and on the default Function, and a per-token
    temperature; ``T`` = 5 runs padded."""
    _require_program(entry=entry)
    if form == "temperature":
        _require_tau_program()
        inp = _tau_inputs(T, W=glm_weight, ignore_frac=0.25)
    else:
        inp = _device_inputs(T, objective, W=glm_weight, ignore_frac=0.25)
    ignored = inp.labels < 0
    assert bool(ignored.any()) and bool((~ignored).any())
    clean = dataclasses.replace(
        inp, X=torch.where(ignored[:, None], torch.zeros_like(inp.X), inp.X)
    )
    dirty = _non_finite_ignored_rows(inp, float(value))
    kw = dict(backend="cake", entry=entry, gemm_tuning=_ONE_SLICE)
    with _shipped_zero_weight_floor(), _small_t_keep(form != "nokeep"):
        ref = _run(clean, 4096, **kw)
        got = _run(dirty, 4096, **kw)
    for key in ("loss", "logp", "dX", "dW"):
        assert bool(torch.isfinite(got[key].float()).all()), key
        assert _same_bits(got[key], ref[key]), key
    assert not bool(got["dX"][ignored].float().any())


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_small_call_every_row_ignored(glm_weight, entry):
    """Every row of a small call ignored: no early return (nothing is counted); ``loss`` 0, ``logp`` 0 and zero
    gradients, as the counted call's early return gives."""
    _require_program(entry=entry)
    inp = _device_inputs(32, "ce", W=glm_weight, ignore_frac=1.0)
    for rows in (256, 0):
        with _uncompacted_max_rows(rows):
            got = _run(inp, 4096, backend="cake", entry=entry)
        assert float(got["loss"]) == 0.0, rows
        assert not bool(got["logp"].any()), rows
        assert not bool(got["dX"].float().any()) and not bool(
            got["dW"].float().any()
        ), rows
        assert bool(
            torch.isfinite(got["dX"].float()).all()
            and torch.isfinite(got["dW"].float()).all()
        ), rows


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_small_call_cuda_graph(glm_weight, entry):
    """A small autograd call (T = 32 with ignored rows: no count; the log-probability entry on its small-call kept
    logits) and its backward captured in one CUDA graph replay for other ``X`` / ``labels`` / ``dlogp`` values written
    into the captured inputs and give the eager call's bits (``loss`` / ``logp`` / ``dW``; ``dX`` with one dX slice
    pinned)."""
    _require_program(entry=entry)
    inps = [
        _device_inputs(32, "ce", W=glm_weight, ignore_frac=0.25, seed=SEED + k)
        for k in range(3)
    ]
    W = glm_weight.detach().requires_grad_(True)
    X = inps[0].X.detach().clone().requires_grad_(True)
    labels, dlogp = inps[0].labels.clone(), inps[0].dlogp.clone()
    loss_div = inps[0].loss_div

    def step(x, lab, d):
        return _small_call_step(
            x, W, lab, entry, loss_div=loss_div, dlogp=d, gemm_tuning=_ONE_SLICE
        )

    with _uncompacted_max_rows(256):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step(X, labels, dlogp)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step(X, labels, dlogp)
        for src in inps[1:] + inps[:1]:
            with torch.no_grad():
                X.copy_(src.X)
            labels.copy_(src.labels)
            dlogp.copy_(src.dlogp)
            graph.replay()
            torch.cuda.synchronize()
            got = [None if t is None else t.clone() for t in captured]
            x_e = src.X.detach().clone().requires_grad_(True)
            want = step(x_e, src.labels.clone(), src.dlogp.clone())
            torch.cuda.synchronize()
            for name, g, w in zip(("loss", "logp", "dX", "dW"), got, want, strict=True):
                assert _same_bits(g, w), name


@pytest.mark.parametrize("T", [1, 32, 256, 257])
def test_device_small_t_keep_routes_small_logprob_calls(glm_weight, T):
    """The small-call kept logits (on by default): a log-probability call of at most ``SMALL_T_KEEP_MAX_ROWS`` rows
    where a backward can follow runs the kept-logits Function and gives the default Function's bits (``logp`` / ``dW``;
    ``dX`` with one dX slice pinned); 257 rows, and a call under ``torch.no_grad``, run the default Function.
    Small-call padding off: a padded call returns a slice of the kept call's output."""
    _require_program(entry="logprob")
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
    kw = dict(backend="cake", entry="logprob", gemm_tuning=_ONE_SLICE)
    with _small_pad_rows(0):
        with _small_t_keep(False):
            base = _run(inp, 4096, **kw)
        X = inp.X.detach().requires_grad_(True)
        logp = cake_backend.cake_lm_head_nondet_train_logprob(
            X,
            inp.W,
            inp.labels,
            chunk_size=4096,
            backend="cake",
            gemm_tuning=_ONE_SLICE,
        )
        kept = type(logp.grad_fn).__name__.startswith(
            "CakeLMHeadNondetLogprobKeepFunction"
        )
        del logp
        with torch.no_grad():
            nog = cake_backend.cake_lm_head_nondet_train_logprob(
                inp.X, inp.W, inp.labels, chunk_size=4096, backend="cake"
            )
        got = _run(inp, 4096, **kw)
    assert kept == (T <= cake_backend.SMALL_T_KEEP_MAX_ROWS), T
    assert nog.grad_fn is None and _same_bits(nog, base["logp"])
    for key in ("logp", "dX", "dW"):
        assert _same_bits(got[key], base[key]), key


@pytest.mark.parametrize("T", [1, 5, 31])
@pytest.mark.parametrize(
    "entry, objective, form",
    [
        ("loss", "ce", None),
        ("loss", "policy", None),
        ("logprob", "ce", None),
        ("logprob", "ce", "nokeep"),
        ("logprob", "ce", "temperature"),
    ],
)
def test_device_small_pad_is_the_unpadded_call(glm_weight, T, entry, objective, form):
    """A call padded to 32 rows (``set_small_pad_rows(32)``) returns the unpadded small call's bits and shapes:
    ``loss`` / ``logp`` / ``dW`` and ``dX`` (one dX slice pinned), with ignored rows among the ``T`` rows; the
    log-probability entry on its small-call kept logits and on the default Function, and with a temperature."""
    _require_program(entry=entry)
    inp = _device_inputs(
        T, objective, W=glm_weight, ignore_frac=0.0 if T == 1 else 0.25
    )
    kw = dict(backend="cake", entry=entry, gemm_tuning=_ONE_SLICE)
    if form == "temperature":
        _require_tau_program()
        gen = torch.Generator(device=CUDA).manual_seed(SEED + T)
        kw["temperature"] = (
            torch.rand(T, device=CUDA, generator=gen, dtype=torch.float32) + 0.5
        )
    with _shipped_zero_weight_floor(), _small_t_keep(form != "nokeep"):
        with _small_pad_rows(0):
            ref = _run(inp, 4096, **kw)
        with _small_pad_rows(32):
            padded = _run(inp, 4096, **kw)
    for key in ("loss", "logp", "dX", "dW"):
        assert padded[key].shape == ref[key].shape, key
        assert _same_bits(padded[key], ref[key]), key


@pytest.mark.parametrize("entry", ["loss", "logprob", "temperature"])
def test_device_small_call_pads_to_the_entry_default(glm_weight, entry, monkeypatch):
    """With the shipped defaults a one-row call stages ``small_pad_rows(entry, device)`` rows: 16 for the loss entry,
    the device architecture's :data:`SMALL_PAD_ROWS_LOGPROB_BY_ARCH` entry (else 16) for the log-probability entries
    (plain and with a temperature); the per-row output keeps one row.  The one-pass cross-entropy is off
    (``one_pass_rows`` empty): by default a T = 1 loss call at ``ONE_PASS_HIDDEN`` takes it and stages nothing."""
    if entry == "temperature":
        _require_tau_program()
        inp = _tau_inputs(1, W=glm_weight, ignore_frac=0.0)
    else:
        _require_program(entry=entry)
        inp = _device_inputs(1, "ce", W=glm_weight, ignore_frac=0.0)
    kind = "loss" if entry == "loss" else "logprob"
    expected = cake_backend.small_pad_rows(kind, inp.X.device)
    if kind == "loss":
        assert expected == cake_backend.SMALL_PAD_ROWS_DEFAULT
    else:
        arch = cake_backend.arch_for(inp.X.device)
        table = cake_backend.SMALL_PAD_ROWS_LOGPROB_BY_ARCH
        assert expected == table.get(arch, cake_backend.SMALL_PAD_ROWS_DEFAULT)
    seen = []
    apply = cake_backend._SmallCallStage.apply

    def spy(X, labels, rows, *rest):
        seen.append(int(rows))
        return apply(X, labels, rows, *rest)

    monkeypatch.setattr(cake_backend._SmallCallStage, "apply", spy)
    with _small_pad_rows(None), _one_pass_rows(()):
        out = _run(inp, 4096, entry=kind, backend="cake")
    assert seen == [expected], seen
    assert tuple(out["logp"].shape) == (1,)


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_device_small_pad_has_no_host_synchronization(glm_weight, entry):
    """A padded small call (T = 1 padded to 16, forward and backward) synchronizes nothing (PyTorch's
    synchronization debug mode at ``"error"`` raises nothing).  The one-pass cross-entropy is off (``one_pass_rows``
    empty), so the loss call is staged and padded too."""
    _require_program(entry=entry)
    inp = _device_inputs(1, "ce", W=glm_weight, ignore_frac=0.0)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)

    def step():
        return _small_call_step(
            X, W, inp.labels, entry, loss_div=inp.loss_div, dlogp=inp.dlogp
        )

    with _small_pad_rows(16), _one_pass_rows(()):
        for _ in range(2):
            step()
        torch.cuda.synchronize()
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            loss, logp, dx, dw = step()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
        torch.cuda.synchronize()
    assert logp.shape == (1,) and dx.shape == X.shape and dw.shape == W.shape


@pytest.mark.parametrize("T", [1, 32, 256])
@pytest.mark.parametrize("form", ["loss", "logprob", "nokeep", "temperature"])
def test_device_small_call_repeats_its_bindings(glm_weight, T, form):
    """A repeated small call (T = 1 padded to 16, T = 32 / 256 unpadded; the loss entry, the log-probability entry on its
    small-call kept logits and on the default Function, a per-token temperature) meets the bindings the previous call
    remembered: the second and the third call add binding hits and no miss -- every staged tensor of a call is a view
    of one allocation, which the caching allocator hands back to the next call of the same shape.  The one-pass
    cross-entropy is off (``one_pass_rows`` empty): a T = 1 loss call takes the small-call path, not the one-pass
    programs."""
    entry = "loss" if form == "loss" else "logprob"
    _require_program(entry=entry)
    if form == "temperature":
        _require_tau_program()
        inp = _tau_inputs(T, W=glm_weight, ignore_frac=0.0 if T == 1 else 0.25)
    else:
        inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0 if T == 1 else 0.25)
    cache = cake_backend.BINDING_CACHE
    seen = []
    with _small_t_keep(form != "nokeep"), _one_pass_rows(()):
        cache.clear()
        for _ in range(3):
            hits, misses = cache.hits, cache.misses
            _run(inp, 4096, entry=entry, backend="cake")
            seen.append((cache.hits - hits, cache.misses - misses))
    assert seen[0][1] > 0, seen  # the first call binds
    for added_hits, added_misses in seen[1:]:
        assert added_misses == 0 and added_hits > 0, seen


# ---------------------------------------------------------------------------
# One-pass cross-entropy (one_pass_rows): a loss entry call of T in one_pass_rows() rows, H = ONE_PASS_HIDDEN, one
# chunk, runs the one-pass forward, merge and backward
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _one_pass_rows(rows):
    previous = cake_backend.set_one_pass_rows(rows)
    try:
        yield
    finally:
        cake_backend.set_one_pass_rows(previous)


def _require_one_pass(T: int):
    _require_program(entry="loss")
    if T not in cake_backend._one_pass_record(torch.cuda.current_device())[2]:
        pytest.skip(f"this device's record registers no one-pass programs of {T} rows")


@contextlib.contextmanager
def _one_pass_launches():
    """The one-pass stages the calls of the block launch, in order."""
    launched: list = []
    original = cake_backend._one_pass_launch

    def recording(module_name, record, stage, values, device, index):
        launched.append(stage)
        return original(module_name, record, stage, values, device, index)

    cake_backend._one_pass_launch = recording
    try:
        yield launched
    finally:
        cake_backend._one_pass_launch = original


def _one_pass_stages(T: int, *, need_dx: bool = True, need_dw: bool = True) -> list:
    fwd, fwd_nodx = cake_backend.ONE_PASS_FWD_STAGES[T]
    stages = [fwd if need_dx else fwd_nodx, cake_backend.ONE_PASS_COMBINE_STAGE]
    return stages + [cake_backend.ONE_PASS_DW_STAGE]


def test_one_pass_switch_validates_and_restores():
    """``set_one_pass_rows`` takes row counts with a one-pass forward and returns the previous set; the default is one
    row."""
    assert (
        cake_backend.one_pass_rows()
        == cake_backend.ONE_PASS_ROWS_DEFAULT
        == frozenset({1})
    )
    for bad in (1, True, "1", [3], [1.0], [0], [True]):
        with pytest.raises(ValueError):
            cake_backend.set_one_pass_rows(bad)
    assert cake_backend.one_pass_rows() == frozenset({1})
    assert cake_backend.set_one_pass_rows(()) == frozenset({1})
    assert cake_backend.set_one_pass_rows({1, 2}) == frozenset()
    assert cake_backend.set_one_pass_rows(
        cake_backend.ONE_PASS_ROWS_DEFAULT
    ) == frozenset({1, 2})


def test_device_one_pass_rule_is_host_only(glm_weight):
    """``_one_pass_call`` decides from host values alone (no synchronization): a cross-entropy call of a row count in
    ``one_pass_rows()`` with ``H = ONE_PASS_HIDDEN``, one chunk, 16-byte aligned operands, ``backend="cake"``, compacted
    rows and no explicit GEMM knobs; any other call keeps the chunked path."""
    _require_one_pass(1)
    inp = _device_inputs(1, "ce", W=glm_weight, ignore_frac=0.0)
    two = _device_inputs(2, "ce", W=glm_weight, ignore_frac=0.0)
    five = _device_inputs(5, "ce", W=glm_weight, ignore_frac=0.0)
    H = int(inp.X.shape[1])
    odd = torch.empty((H + 8,), dtype=torch.bfloat16, device=CUDA)[1 : H + 1].view(1, H)
    narrow = torch.empty((256, H - 256), dtype=torch.bfloat16, device=CUDA)
    kw = dict(
        objective="ce",
        infer_logp=None,
        loss_weights=None,
        chunk_size=4096,
        backend="cake",
        compact_rows=None,
        gemm_tuning=None,
    )
    call = cake_backend._one_pass_call
    host = (inp.X.cpu(), glm_weight.cpu(), inp.labels.cpu())
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        assert call(inp.X, glm_weight, inp.labels, **kw)
        assert call(
            inp.X,
            glm_weight,
            inp.labels,
            **{**kw, "compact_rows": True, "gemm_tuning": {}},
        )
        for change in (
            {"backend": "reference"},
            {"compact_rows": False},
            {"gemm_tuning": {"dx": {"k_slices": 1}}},
            {"objective": "policy"},
            {"infer_logp": inp.loss_div},
            {"chunk_size": 0},
        ):
            assert not call(inp.X, glm_weight, inp.labels, **{**kw, **change}), change
        assert not call(
            two.X, glm_weight, two.labels, **kw
        )  # 2 rows: not in the default set
        assert not call(five.X, glm_weight, five.labels, **kw)
        assert not call(odd, glm_weight, inp.labels, **kw)  # an X base off 16 bytes
        assert not call(
            inp.X[:, : H - 256], narrow, inp.labels, **kw
        )  # another hidden size
        assert not call(*host, **kw)
        with _one_pass_rows(()):
            assert not call(inp.X, glm_weight, inp.labels, **kw)
        with _one_pass_rows({1, 2}):
            registered = cake_backend._one_pass_record(torch.cuda.current_device())[2]
            assert call(two.X, glm_weight, two.labels, **kw) == (2 in registered)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.parametrize("T", [1, 2])
@pytest.mark.parametrize("train", ["both", "x", "w"])
def test_device_one_pass_meets_the_gates(glm_weight, T, train):
    """A one-pass call (T rows; X and W trainable, X alone, W alone) launches the forward (without the head-row sums
    when X is frozen), the merge and the backward, passes the deterministic suite's gates against the FP64 oracle,
    and repeats its bits on a second call (no reduction whose order the launch leaves open)."""
    _require_one_pass(T)
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)
    kw = dict(backend="cake", train_x=train != "w", train_w=train != "x")
    with _one_pass_rows({T}), _one_pass_launches() as launched:
        got = _run(inp, 4096, **kw)
        again = _run(inp, 4096, **kw)
    stages = _one_pass_stages(T, need_dx=train != "w", need_dw=train != "x")
    assert launched == stages + stages
    _check_against_references(got, inp, ceiling=True)
    for key in ("loss", "logp", "dX", "dW"):
        assert _same_bits(again[key], got[key]), key


def _padded_rows(X: torch.Tensor, pad: int = 64) -> torch.Tensor:
    """``X`` as a view of a ``[T, H + pad]`` buffer (row pitch ``H + pad``, NaN in the padding)."""
    T, H = X.shape
    buf = torch.full((T, H + pad), float("nan"), dtype=X.dtype, device=X.device)
    view = buf[:, :H]
    view.copy_(X)
    return view


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_device_one_pass_ignored_rows_reach_no_output(glm_weight, value):
    """An ignored row of a one-pass call holding ``inf`` / NaN in ``X`` (rows at a padded pitch) gives the bits of the
    same call with a zero ``X`` row there -- ``loss`` / ``logp`` / ``dX`` / ``dW``, every output finite, the ignored
    row's ``logp`` and ``dX`` exact zeros -- and the clean call meets the gates against the FP64 oracle."""
    _require_one_pass(2)
    inp = make_inputs(2, seed=SEED + 2, device=CUDA, W=glm_weight, ignore_frac=0.5)
    ignored = inp.labels < 0
    assert int(ignored.sum()) == 1
    zeroed = torch.where(ignored[:, None], torch.zeros_like(inp.X), inp.X)
    clean = dataclasses.replace(inp, X=_padded_rows(zeroed))
    dirty = dataclasses.replace(
        inp, X=_padded_rows(_non_finite_ignored_rows(inp, float(value)).X)
    )
    assert clean.X.stride(0) == dirty.X.stride(0) == inp.X.shape[1] + 64
    with _one_pass_rows({2}), _one_pass_launches() as launched:
        ref = _run(clean, 4096, backend="cake")
        got = _run(dirty, 4096, backend="cake")
    assert launched == _one_pass_stages(2) * 2
    for key in ("loss", "logp", "dX", "dW"):
        assert bool(torch.isfinite(got[key].float()).all()), key
        assert _same_bits(got[key], ref[key]), key
    assert not bool(got["dX"][ignored].float().any())
    assert not bool(got["logp"][ignored].any())
    _check_against_references(ref, clean, ceiling=True)


def test_device_one_pass_every_row_ignored(glm_weight):
    """A one-pass call whose rows are all ignored (NaN in X): zero ``loss`` / ``logp`` / ``dX`` / ``dW``, through the
    same three launches (the forward streams no W)."""
    _require_one_pass(1)
    inp = _non_finite_ignored_rows(
        _device_inputs(1, "ce", W=glm_weight, ignore_frac=1.0), float("nan")
    )
    with _one_pass_launches() as launched:
        got = _run(inp, 4096, backend="cake")
    assert launched == _one_pass_stages(1)
    for key in ("loss", "logp", "dX", "dW"):
        assert (
            torch.count_nonzero(got[key]) == 0
            and not torch.isnan(got[key].float()).any()
        ), key


@pytest.mark.parametrize("T", [1, 2])
def test_device_one_pass_checkpoint_replay(glm_weight, T):
    """Under ``torch.utils.checkpoint`` (non-reentrant, gradient placement "auto") the checkpointed forward runs the
    forward without the head-row sums and the merge, the recomputation the full forward and the merge, the backward
    once: the uncheckpointed call's bits."""
    _require_one_pass(T)
    inp = _device_inputs(T, "ce", W=glm_weight, ignore_frac=0.0)

    def head(X, W):
        return cake_backend.cake_lm_head_nondet_train_loss(
            X, W, inp.labels, loss_div=inp.loss_div, chunk_size=4096
        )

    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    with _one_pass_rows({T}):
        with _one_pass_launches() as launched:
            loss = torch.utils.checkpoint.checkpoint(head, X, W, use_reentrant=False)
            loss.backward()
        plain = _run(inp, 4096, backend="cake")
    fwd, fwd_nodx = cake_backend.ONE_PASS_FWD_STAGES[T]
    combine, dw = cake_backend.ONE_PASS_COMBINE_STAGE, cake_backend.ONE_PASS_DW_STAGE
    assert launched == [fwd_nodx, combine, fwd, combine, dw]
    assert _same_bits(loss.detach(), plain["loss"])
    assert _same_bits(X.grad, plain["dX"]) and _same_bits(W.grad, plain["dW"])


def test_device_one_pass_gradient_accumulation_and_retained_graph(glm_weight):
    """Two backward passes into the same ``W`` (the second call with a frozen X: the backward writes ``dW`` alone):
    ``W.grad`` is the autograd engine's sum of the two incoming gradients, each the single call's bits; a retained
    graph's second backward repeats the first's bits."""
    _require_one_pass(1)
    a = _device_inputs(1, "ce", W=glm_weight, ignore_frac=0.0)
    b = _device_inputs(1, "ce", W=glm_weight, seed=SEED + 7, ignore_frac=0.0)
    X = a.X.detach().requires_grad_(True)
    W = glm_weight.detach().requires_grad_(True)
    seen_w = []
    W.register_hook(lambda g: seen_w.append(g.detach().clone()))
    for inp in (a, b):
        loss = cake_backend.cake_lm_head_nondet_train_loss(
            X if inp is a else inp.X.detach(),
            W,
            inp.labels,
            loss_div=inp.loss_div,
            chunk_size=4096,
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
    assert _same_bits(X.grad, single_a["dX"])
    X2 = a.X.detach().requires_grad_(True)
    W2 = glm_weight.detach().requires_grad_(True)
    loss = cake_backend.cake_lm_head_nondet_train_loss(
        X2, W2, a.labels, loss_div=a.loss_div, chunk_size=4096
    )
    dx1, dw1 = torch.autograd.grad(loss, (X2, W2), retain_graph=True)
    dx2, dw2 = torch.autograd.grad(loss, (X2, W2))
    assert _same_bits(dx1, dx2) and _same_bits(dw1, dw2)


@pytest.mark.parametrize("ignore_frac", [0.0, 1.0])
def test_device_one_pass_has_no_host_synchronization(glm_weight, ignore_frac):
    """A one-pass call (forward and backward of the autograd entry, every row ignored too) synchronizes nothing:
    PyTorch's synchronization debug mode at ``"error"`` raises nothing."""
    _require_one_pass(1)
    inp = _device_inputs(1, "ce", W=glm_weight, ignore_frac=ignore_frac)
    X = inp.X.detach().requires_grad_(True)
    W = inp.W.detach().requires_grad_(True)
    for _ in range(2):
        _small_call_step(X, W, inp.labels, "loss", loss_div=inp.loss_div)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with _one_pass_launches() as launched:
            _small_call_step(X, W, inp.labels, "loss", loss_div=inp.loss_div)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()
    assert launched == _one_pass_stages(1)


def test_device_one_pass_cuda_graph(glm_weight):
    """A one-pass step (forward and ``torch.autograd.grad``) captured into a CUDA graph replays the eager step's bits,
    and the eager bits of new inputs written into the captured ``X`` and ``labels``."""
    _require_one_pass(1)
    inp = _device_inputs(1, "ce", W=glm_weight, ignore_frac=0.0)
    new = _device_inputs(1, "ce", W=glm_weight, seed=SEED + 11, ignore_frac=0.0)
    X = inp.X.detach().clone().requires_grad_(True)
    labels = inp.labels.clone()
    W = inp.W.detach().requires_grad_(True)

    def step():
        loss = cake_backend.cake_lm_head_nondet_train_loss(
            X, W, labels, loss_div=inp.loss_div, chunk_size=4096
        )
        dx, dw = torch.autograd.grad(loss, (X, W))
        return loss.detach(), dx, dw

    def clone(out):
        return tuple(t.clone() for t in out)

    eager = clone(step())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with _one_pass_launches() as launched, torch.cuda.graph(graph):
        captured = step()
    assert launched == _one_pass_stages(1)
    graph.replay()
    torch.cuda.synchronize()
    assert all(_same_bits(g, e) for g, e in zip(captured, eager, strict=True))
    with torch.no_grad():
        X.copy_(new.X)
    labels.copy_(new.labels)
    want = clone(step())
    graph.replay()
    torch.cuda.synchronize()
    assert all(_same_bits(g, w) for g, w in zip(captured, want, strict=True))
