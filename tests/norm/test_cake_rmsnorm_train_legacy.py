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

Compatibility of ``flashinfer.cake_rmsnorm_train`` with the module of that name
before the non-deterministic package replaced it.

Before, ``flashinfer.cake_rmsnorm_train`` was the deterministic facade itself
(``sys.modules`` aliased it to ``flashinfer.cake_rmsnorm_det_train``) and the
package ``flashinfer.cake_rmsnorm_train_backend`` resolved every attribute on the
deterministic backend package.  Code written against them keeps working when every
public attribute they had -- the deterministic names of ``__all__`` and the names
from before the deterministic rename -- stays reachable on the new module and
package, accepts every argument the old object accepted, and, called as before
(``deterministic`` left at its default ``True``), returns the deterministic
package's results bit for bit.  :data:`PREVIOUS_PUBLIC` and
:data:`PREVIOUS_SIGNATURES` record that surface as it was (FlashInfer main
``18fb0bfcc6d7``, the deterministic facade and backend of #8045); the device-free
tests check reachability and signatures, the device tests the results.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch

import flashinfer.cake_rmsnorm_det_train as det_facade
import flashinfer.cake_rmsnorm_det_train_backend as det_backend

NAMESPACES = {
    "facade": "flashinfer.cake_rmsnorm_train",
    "backend": "flashinfer.cake_rmsnorm_train_backend",
}
# Every public attribute of the previous ``flashinfer.cake_rmsnorm_train`` (the
# deterministic facade): its ``__all__`` and the names from before the
# deterministic rename (the same objects as their deterministic names); the
# previous backend package resolved the same names on the deterministic backend.
PREVIOUS_ALL = (
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormDetFunction",
    "cake_rmsnorm_det",
    "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_det_train_forward",
    "is_cake_rmsnorm_det_train_supported",
    "route_key",
)
# previous name -> the deterministic object it was
PREVIOUS_ALIASES = {
    "CakeRMSNormFunction": "CakeRMSNormDetFunction",
    "cake_rmsnorm": "cake_rmsnorm_det",
    "cake_rmsnorm_train_backward": "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_train_backward_workspace": "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes": "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward": "cake_rmsnorm_det_train_forward",
    "is_cake_rmsnorm_train_supported": "is_cake_rmsnorm_det_train_supported",
}
PREVIOUS_PUBLIC = tuple(sorted({*PREVIOUS_ALL, *PREVIOUS_ALIASES}))
# The deterministic object each previous name was.
DET_OBJECT = {name: PREVIOUS_ALIASES.get(name, name) for name in PREVIOUS_PUBLIC}
# Names only the deterministic package defines: on the new module they are the
# deterministic objects themselves (the other previous names are this package's
# objects, deterministic by default).
DET_NAMED = tuple(
    name for name in PREVIOUS_ALL if name == DET_OBJECT[name] and "det" in name.lower()
)
# (parameter, kind, default) of every previous callable (``inspect`` kinds; ``None``
# default: no default), by its deterministic name; the previous facade's three
# decorated entry points and the backend's helpers.
_P, _K = inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY
_NO = inspect.Parameter.empty
PREVIOUS_SIGNATURES = {
    "cake_rmsnorm_det_train_forward": (
        ("x", _P, _NO),
        ("w", _P, _NO),
        ("eps", _P, _NO),
        ("residual", _K, None),
        ("out", _K, None),
        ("rstd", _K, None),
        ("residual_out", _K, None),
    ),
    "cake_rmsnorm_det_train_backward": (
        ("g", _P, _NO),
        ("x", _P, _NO),
        ("w", _P, _NO),
        ("rstd", _P, _NO),
        ("deterministic", _K, True),
        ("g_residual", _K, None),
        ("workspace", _K, None),
        ("dx", _K, None),
        ("dw", _K, None),
    ),
    "cake_rmsnorm_det": (
        ("x", _P, _NO),
        ("w", _P, _NO),
        ("eps", _P, 1e-6),
        ("residual", _P, None),
        ("deterministic", _P, True),
    ),
    "cake_rmsnorm_det_train_backward_workspace_bytes": (
        ("rows", _P, _NO),
        ("hidden", _P, _NO),
        ("device", _P, _NO),
        ("residual", _K, False),
    ),
    "cake_rmsnorm_det_train_backward_workspace": (
        ("rows", _P, _NO),
        ("hidden", _P, _NO),
        ("device", _P, _NO),
        ("residual", _K, False),
    ),
    "is_cake_rmsnorm_det_train_supported": (
        ("device", _P, _NO),
        ("hidden", _P, None),
    ),
    "route_key": (
        ("hidden", _P, _NO),
        ("kind", _P, _NO),
        ("residual", _P, _NO),
    ),
}
SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# Widths the deterministic package exports (BF16 weights), with their eps.
DET_WIDTHS = {6144: 1e-5, 4096: 1e-5, 2048: 1e-6, 1536: 1e-6, 512: 1e-6}
SEED = 8066


def _namespace(name: str):
    return importlib.import_module(NAMESPACES[name])


def _previous_object(name: str):
    """The object the previous module bound to ``name`` (the deterministic one)."""
    return getattr(det_facade, DET_OBJECT[name])


# --------------------------------------------------------------------------- reachability and signatures (no device)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", PREVIOUS_PUBLIC)
@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_previous_public_attribute_is_reachable(namespace: str, name: str) -> None:
    """Every public attribute of the previous module / package resolves on the
    new one (``getattr``), as it did before; a deterministic-named attribute
    resolves to the deterministic object itself (lazy
    re-exports of the deterministic sibling)."""
    module = _namespace(namespace)
    assert hasattr(module, name), (
        f"{NAMESPACES[namespace]}.{name} was public before the non-deterministic "
        f"package (the deterministic {DET_OBJECT[name]}) and is not reachable now"
    )
    if name in DET_NAMED:
        sibling = det_facade if namespace == "facade" else det_backend
        assert getattr(module, name) is getattr(sibling, name), (
            f"{NAMESPACES[namespace]}.{name} is not {sibling.__name__}.{name}"
        )
    print(f"LEGACY reachable {namespace} {name}")


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", PREVIOUS_PUBLIC)
@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_previous_callable_accepts_its_previous_arguments(
    namespace: str, name: str
) -> None:
    """A callable keeps every previous parameter with its kind and default; a
    parameter it adds is keyword-only with a default, so every previous call
    form binds unchanged.  Constants keep their previous values."""
    module = _namespace(namespace)
    if not hasattr(module, name):
        pytest.fail(f"{NAMESPACES[namespace]}.{name} is not reachable")
    obj = getattr(module, name)
    previous = _previous_object(name)
    if not callable(previous):
        assert obj == previous, (name, obj, previous)
        return
    if inspect.isclass(previous):
        assert inspect.isclass(obj) and issubclass(obj, torch.autograd.Function)
        old = inspect.signature(previous.forward)
        new = inspect.signature(obj.forward)
        assert list(old.parameters)[:4] == list(new.parameters)[:4], (old, new)
        return
    old_params = PREVIOUS_SIGNATURES[DET_OBJECT[name]]
    new_params = inspect.signature(obj).parameters
    for param, kind, default in old_params:
        assert param in new_params, f"{name}: parameter {param} is gone"
        got = new_params[param]
        assert got.kind == kind, f"{name}: {param} is {got.kind}, was {kind}"
        assert got.default == default or (got.default is _NO and default is _NO), (
            f"{name}: {param} defaults to {got.default!r}, was {default!r}"
        )
    previous_names = [p for p, _k, _d in old_params]
    positional = [
        p
        for p, v in new_params.items()
        if v.kind in (inspect.Parameter.POSITIONAL_ONLY, _P)
    ]
    assert positional[: len([p for p, k, _ in old_params if k == _P])] == [
        p for p, k, _ in old_params if k == _P
    ], f"{name}: positional order {positional}"
    for param, value in new_params.items():
        if param in previous_names:
            continue
        assert value.default is not _NO, f"{name}: new parameter {param} has no default"
        if param == "deterministic":
            assert value.default is True, (
                f"{name}: deterministic defaults to {value.default}"
            )


# --------------------------------------------------------------------------- deterministic results (device)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and det_facade.is_cake_rmsnorm_det_train_supported(_device())
    )


requires_device = pytest.mark.skipif(
    not _supported(),
    reason="the deterministic Cake RMSNorm package needs an SM100 / SM103 / SM107 device",
)


@pytest.fixture
def no_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _inputs(rows: int, hidden: int, residual: bool, key: int = 0):
    generator = torch.Generator(device=_device()).manual_seed(
        (SEED * 1_000_003 + rows * 31 + hidden * 7 + int(residual) + key) % (2**31 - 1)
    )

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=_device(), generator=generator
        ).to(torch.bfloat16)

    x = randn(rows, hidden)
    u = randn(rows, hidden) if residual else None
    g = randn(rows, hidden)
    g_h = randn(rows, hidden) if residual else None
    w = (
        1
        + 0.1
        * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
    ).to(torch.bfloat16)
    return x, u, g, g_h, w


def _det_cases():
    for hidden in DET_WIDTHS:
        for rows in (1, 257, 16385):
            yield pytest.param(hidden, False, rows, id=f"h{hidden}-T{rows}")
    for rows in (7, 4096):
        yield pytest.param(6144, True, rows, id=f"res-h6144-T{rows}")


def _skip_unless_det_serves(hidden: int, residual: bool) -> None:
    if not det_backend.cake_jit.route_applies(
        device_capability=torch.cuda.get_device_capability(_device()),
        hidden=hidden,
        kind="bwd",
        residual=residual,
    ):
        pytest.skip(
            f"the deterministic package exports no hidden={hidden} residual={residual} route here"
        )


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


# names a device case could not reach (the reachability test fails for them)
_MISSING: set[str] = set()


def _reachable(module, name: str):
    """The attribute, or ``None`` (recorded in ``_MISSING``) when the namespace
    lacks it: the reachability test reports that; the device test checks the
    results of every name it can reach."""
    if not hasattr(module, name):
        _MISSING.add(name)
        return None
    return getattr(module, name)


@requires_device
@pytest.mark.parametrize(("hidden", "residual", "rows"), list(_det_cases()))
@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_previous_entry_points_return_the_det_bits(
    namespace: str, hidden: int, residual: bool, rows: int, no_determinism
) -> None:
    """Called as before (``deterministic`` at its default), every previous entry
    point of the namespace -- the previous names and the deterministic names --
    returns exactly the deterministic package's results: forward ``(y, rstd[,
    h_new])``, backward ``(dx, dw)`` with and without a workspace from the
    namespace's workspace helper (sized as the deterministic package sizes it),
    the autograd function and wrapper (outputs and gradients), the support
    query, the route key and the constants.  A name the namespace lacks is
    left to :func:`test_previous_public_attribute_is_reachable` (listed as
    unreachable); the previous names of the rename must all be checked here."""
    _skip_unless_det_serves(hidden, residual)
    module = _namespace(namespace)
    eps = DET_WIDTHS[hidden]
    x, u, g, g_h, w = _inputs(rows, hidden, residual, key=3)
    device = _device()
    ref_fwd = det_facade.cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
    normalized, rstd = (ref_fwd[2] if residual else x), ref_fwd[1]
    ref_bwd = det_facade.cake_rmsnorm_det_train_backward(
        g, normalized, w, rstd, g_residual=g_h
    )
    ref_bytes = det_facade.cake_rmsnorm_det_train_backward_workspace_bytes(
        rows, hidden, device, residual=residual
    )
    checked = []
    for old, det_name in (
        ("cake_rmsnorm_train_forward", "cake_rmsnorm_det_train_forward"),
        ("cake_rmsnorm_train_backward", "cake_rmsnorm_det_train_backward"),
    ):
        for name in (old, det_name):
            fn = _reachable(module, name)
            if fn is None:
                continue
            if "forward" in name:
                got = fn(x, w, eps, residual=u)
                assert _equal(tuple(got), tuple(ref_fwd)), (
                    f"{namespace}.{name}: forward bits"
                )
            else:
                got = fn(g, normalized, w, rstd, g_residual=g_h)
                assert _equal(tuple(got), tuple(ref_bwd)), (
                    f"{namespace}.{name}: backward bits"
                )
            checked.append(name)
    for name in (
        "cake_rmsnorm_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
    ):
        helper = _reachable(module, name)
        if helper is None:
            continue
        size = helper(rows, hidden, device, residual=residual)
        assert size == ref_bytes, f"{namespace}.{name}: {size} bytes, det {ref_bytes}"
        checked.append(name)
    for name in (
        "cake_rmsnorm_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace",
    ):
        helper = _reachable(module, name)
        if helper is None:
            continue
        ws = helper(rows, hidden, device, residual=residual)
        assert ws.dtype == torch.uint8 and ws.numel() == ref_bytes and not ws.any()
        backward = module.cake_rmsnorm_train_backward
        got = backward(g, normalized, w, rstd, g_residual=g_h, workspace=ws)
        assert _equal(tuple(got), tuple(ref_bwd)), (
            f"{namespace}.{name}: backward bits on its workspace"
        )
        checked.append(name)
    for name in (
        "is_cake_rmsnorm_train_supported",
        "is_cake_rmsnorm_det_train_supported",
    ):
        query = _reachable(module, name)
        if query is None:
            continue
        for width in (None, hidden, 1000, 256, 5120):
            assert query(
                device, width
            ) == det_facade.is_cake_rmsnorm_det_train_supported(device, width), (
                f"{namespace}.{name}({width})"
            )
        checked.append(name)
    for name in ("route_key",):
        key = _reachable(module, name)
        if key is None:
            continue
        for kind in ("fwd", "bwd"):
            assert key(hidden, kind, residual) == det_facade.route_key(
                hidden, kind, residual
            ), f"{namespace}.{name}"
        checked.append(name)
    for name in ("ARCHES", "SUPPORTED_COMPUTE_CAPABILITIES"):
        value = _reachable(module, name)
        if value is None:
            continue
        assert value == getattr(det_facade, name), name
        checked.append(name)

    # autograd: the wrapper and the Function, outputs and gradients
    def autograd(apply):
        xa = x.clone().requires_grad_(True)
        wa = w.clone().requires_grad_(True)
        ua = u.clone().requires_grad_(True) if residual else None
        out = apply(xa, wa, ua)
        outs = out if residual else (out,)
        torch.autograd.backward(outs, (g, g_h) if residual else (g,))
        return (
            tuple(o.detach() for o in outs),
            xa.grad,
            wa.grad,
            ua.grad if residual else None,
        )

    ref_auto = autograd(
        lambda a, b, c: det_facade.cake_rmsnorm_det(a, b, eps, residual=c)
    )
    for name in ("cake_rmsnorm", "cake_rmsnorm_det"):
        fn = _reachable(module, name)
        if fn is None:
            continue
        got = autograd(lambda a, b, c, fn=fn: fn(a, b, eps, residual=c))
        assert _equal(got[0], ref_auto[0]), f"{namespace}.{name}: outputs"
        for index, label in ((1, "dx"), (2, "dw"), (3, "du")):
            if ref_auto[index] is not None:
                assert _equal(got[index], ref_auto[index]), (
                    f"{namespace}.{name}: {label}"
                )
        checked.append(name)
    for name in ("CakeRMSNormFunction", "CakeRMSNormDetFunction"):
        cls = _reachable(module, name)
        if cls is None:
            continue
        got = autograd(lambda a, b, c, cls=cls: cls.apply(a, b, eps, c))
        assert _equal(got[0], ref_auto[0]), f"{namespace}.{name}: outputs"
        for index, label in ((1, "dx"), (2, "dw"), (3, "du")):
            if ref_auto[index] is not None:
                assert _equal(got[index], ref_auto[index]), (
                    f"{namespace}.{name}: {label}"
                )
        checked.append(name)
    missing = sorted(set(PREVIOUS_PUBLIC) - set(checked))
    assert set(missing) <= _MISSING, missing  # every name either checked or unreachable
    assert set(PREVIOUS_ALIASES) <= set(checked), sorted(
        set(PREVIOUS_ALIASES) - set(checked)
    )
    print(
        f"LEGACY det_bits {namespace} h={hidden} res={residual} T={rows} "
        f"checked={len(set(checked))} unreachable={missing}"
    )
