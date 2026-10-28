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

Backend of the Cake single-GPU norm kernels of the Muse Spark family as one
self-contained package (the host bodies, the ``cake_jit`` registry and loader
glue and the generated CUDA sources under ``csrc/``): the gated residual with
the next RMSNorm fused behind
``flashinfer.norm.gated_residual_next_rmsnorm(..., backend="cake")``
(:func:`cake_gated_residual_next_rmsnorm`; hidden size 8192), the gated
residual RMSNorm behind ``flashinfer.norm.gated_residual_rmsnorm(...,
backend="cake")`` (:func:`cake_gated_residual_rmsnorm`; hidden size 8192) and
the per-head RMSNorm times the sigmoid gate behind
``flashinfer.norm.rmsnorm_sigmoid_gate(..., backend="cake")``
(:func:`cake_rmsnorm_sigmoid_gate`; head dimension 64, in place); SM100 /
SM103 / SM107.  The stable, ``@flashinfer_api``-decorated entry points and the
user-facing documentation live in :mod:`flashinfer.norm`; import it.

The two-norm kernels ship in three variants of their optional inputs:
``alpha`` a tensor or a scalar, ``post_weight`` present or not
(``alpha_tensor_pw``, ``alpha_scalar_nopw``, ``alpha_scalar_pw``).  A call with
a tensor ``alpha`` and no ``post_weight`` runs the ``alpha_tensor_pw`` programs
with a zero ``post_weight`` vector and ``post_gain_center = 1.0``: ``branch *
(0 + 1)`` is ``branch`` exactly, so the result is the one a dedicated program
would give.  The zero vector is allocated once per device on first use (never
inside CUDA graph capture: :func:`warm_up` prepares a device before a capture).

The gated residual RMSNorm ships per mode (``weight`` absent: ``mode0``;
``weight`` applied before the norm: ``mode1``) in a plain and a programmatic
dependent launch build (``_pdl``); the sigmoid gate ships one BF16 route whose
programs are selected on the head count and the token count together.  Each
route's windows may leave gaps where the kernel lane measured the FlashInfer
kernel ahead: the ``is_cake_*_supported`` predicates answer per call so that
the facade's ``"auto"`` keeps the CuTe DSL kernel there.
"""

from __future__ import annotations

import bisect
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, cast

import torch
import tvm_ffi

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    HIDDEN_SIZE,
    OPS,
    VARIANTS,
    arch_for_capability,
    route_key,
)

GATED_RESIDUAL_NEXT_RMSNORM = "gated_residual_next_rmsnorm"
GATED_RESIDUAL_RMSNORM = "gated_residual_rmsnorm"
RMSNORM_SIGMOID_GATE = "rmsnorm_sigmoid_gate"
#: Every program of the package reaches rows and vectors in accesses of up to 256 bits (the two-norm kernels on every
#: operand, the gated residual RMSNorm on its FP32 planes, the sigmoid gate's 16-head programs on their rows): the
#: package serves 32-byte aligned storage alone, so that no 256-bit access meets a view aligned to 16 bytes only.
ALIGNMENT_BYTES = 32
#: The alignment of the gated residual RMSNorm's and the sigmoid gate's rows and vectors: the package's one alignment.
ROW_ALIGNMENT_BYTES = ALIGNMENT_BYTES
#: The sigmoid gate's token stride is a multiple of this many BF16 elements: every token's row starts aligned.
GATE_STRIDE_ELEMENTS = ROW_ALIGNMENT_BYTES // 2
#: The head dimension of the sigmoid gate's rows.
HEAD_DIM = 64

_zero_post_weights: dict[int, torch.Tensor] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


_CAPABILITIES: dict[tuple[int, int], tuple[Any, tuple[int, int]]] = {}


def _capability(device: torch.device) -> tuple[int, int]:
    """``torch.cuda.get_device_capability(device)``, asked once per device and per probe function.  The entry keeps
    the probe it came from, so a test that replaces ``torch.cuda.get_device_capability`` gets an entry of its own and
    the probe's id cannot be reused while the entry lives."""
    probe = torch.cuda.get_device_capability
    key = (_device_index(device), id(probe))
    entry = _CAPABILITIES.get(key)
    if entry is None or entry[0] is not probe:
        major, minor = probe(device)
        entry = _CAPABILITIES[key] = (probe, (int(major), int(minor)))
    return entry[1]


def _arch(device: torch.device) -> Optional[str]:
    """The exported architecture of ``device``, or ``None`` for a device the package has no build for."""
    if device.type != "cuda":
        return None
    return arch_for_capability(_capability(device))


# ----------------------------------------------------------------------------- admit index
#
# ``backend="auto"`` asks ``is_cake_*_supported(..., rows=)`` (``heads=`` / ``tokens=`` for the sigmoid gate) on every
# call.  The loader's ``admits`` answers one record at a time -- a Python loop over the route's records per call, growing
# with the records before the one that admits the row count.  The index below folds each route's windows, per
# architecture and (for the sigmoid gate) per head count, into one sorted list of disjoint spans ``above < rows <=
# at_most`` -- at import for every route of the registry, on first use for any other (arch, route) a call asks (the
# variant functions spell finitely many) -- so a call answers with one bisection.  It admits exactly the row counts the
# loader's scan admits: tests/norm/test_cake_muse_spark_norm_launch_plans.py compares the two over every row count up to
# 20000 of every route.  A test that replaces registry records or route lists resets the index with
# ``monkeypatch.setattr(pkg, "_ADMIT_INDEX", {})`` after the replacement.

_UNBOUNDED = float("inf")
_Spans = tuple[list[int], list[float]]
_RouteIndex = tuple[bool, bool, list[int], list[float], dict[int, _Spans]]


def _spans(windows: Iterable[Sequence[Any]]) -> _Spans:
    """``(aboves, at_mosts)`` of the disjoint spans ``above < rows <= at_most`` that the ``(above, at_most)`` windows
    cover (``at_most`` ``None`` = unbounded), sorted by ``above``: a window holding no row count is dropped, touching
    or overlapping windows are merged, so the span holding ``rows`` is the last one whose ``above`` lies below it."""
    merged: list[list[float]] = []
    for above, at_most in sorted(
        (int(above), _UNBOUNDED if at_most is None else int(at_most))
        for above, at_most in windows
    ):
        if at_most <= above:
            continue
        if merged and above <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], at_most)
        else:
            merged.append([above, at_most])
    return [int(above) for above, _ in merged], [at_most for _, at_most in merged]


def _route_index(arch: str, route: str) -> _RouteIndex:
    """``(exported, always, aboves, at_mosts, by_heads)`` of ``route`` on ``arch``: ``exported`` when the architecture
    ships a record of the route (the loader's ``route_available``); ``always`` when a record of the route has no
    selection rule; the spans of the route's ``rows_between`` windows; the spans of its ``tokens_heads_between``
    windows per head count.  A rule's ``routes`` spell the windows another route gives the record (a PDL route
    launching the plain build): the index of that route reads them in place of ``windows``."""
    always = False
    row_windows: list[Sequence[Any]] = []
    head_windows: dict[int, list[Sequence[Any]]] = {}
    for name in _loader.route_modules(arch, route):
        rule = _loader.MODULES[name].get("select_rule")
        if rule is None:
            always = True
            continue
        kind = rule.get("kind")
        if kind not in _loader.SELECT_KINDS:
            raise ValueError(f"unknown Cake norm select rule {kind!r}")
        windows = rule.get("routes", {}).get(route, rule["windows"]).get(arch, ())
        if kind == _loader.SELECT_KIND:
            row_windows.extend(windows)
        else:
            for count, above, at_most in windows:
                head_windows.setdefault(int(count), []).append((above, at_most))
    aboves, at_mosts = _spans(row_windows)
    return (
        _loader.route_available(arch, route),
        always,
        aboves,
        at_mosts,
        {count: _spans(w) for count, w in head_windows.items()},
    )


_ADMIT_INDEX: dict[tuple[str, str], _RouteIndex] = {
    (arch, route): _route_index(arch, route)
    for arch, routes in _loader.ROUTES.items()
    for route in routes
}


def _admits(
    arch: str, route: str, rows: Optional[int], heads: Optional[int] = None
) -> bool:
    """Whether ``arch`` ships a program of ``route`` and, with ``rows``, one of the route's windows admits that row
    count; for a route selecting on the head count as well (the sigmoid gate), with ``heads``, whether one of its
    windows names that head count and, with ``rows`` too, holds the token count.  The loader's record-by-record
    ``admits`` folded into one bisection of the admit index."""
    entry = _ADMIT_INDEX.get((arch, route))
    if entry is None:
        entry = _ADMIT_INDEX[(arch, route)] = _route_index(arch, route)
    exported, always, aboves, at_mosts, by_heads = entry
    if not exported:
        return False
    if always:
        return True
    if heads is not None:
        spans = by_heads.get(int(heads))
        if spans is None:
            return False
        aboves, at_mosts = spans
    if rows is None:
        return True
    rows = int(rows)
    i = bisect.bisect_left(aboves, rows)
    return i > 0 and rows <= at_mosts[i - 1]


def gated_residual_next_rmsnorm_variant(
    tensor_alpha: bool, has_post_weight: bool
) -> str:
    """The variant a call maps onto (``alpha_tensor_pw``, ``alpha_scalar_nopw``, ``alpha_scalar_pw``,
    ``alpha_tensor_nopw``)."""
    alpha = "alpha_tensor" if tensor_alpha else "alpha_scalar"
    return f"{alpha}_{'pw' if has_post_weight else 'nopw'}"


def gated_residual_next_rmsnorm_route(tensor_alpha: bool, has_post_weight: bool) -> str:
    """The route a call launches from: its variant's, except that a tensor ``alpha`` without ``post_weight``
    runs the ``alpha_tensor_pw`` route with a zero ``post_weight`` (module docstring)."""
    variant = gated_residual_next_rmsnorm_variant(tensor_alpha, has_post_weight)
    if variant == "alpha_tensor_nopw":
        variant = "alpha_tensor_pw"
    return route_key(GATED_RESIDUAL_NEXT_RMSNORM, variant)


def is_cake_gated_residual_next_rmsnorm_supported(
    device: torch.device,
    *,
    tensor_alpha: bool = True,
    has_post_weight: bool = True,
    rows: Optional[int] = None,
) -> bool:
    """Whether this package ships a program for ``device`` and the call's variant (tensor or scalar ``alpha``,
    with or without ``post_weight``) -- and, with ``rows``, one whose window admits that row count: a route's
    windows may leave gaps where the kernel lane measured the FlashInfer kernel ahead, and no program is shipped
    for those row counts.  The tensor contract is :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    return _admits(
        arch, gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight), rows
    )


def zero_post_weight(device: torch.device) -> torch.Tensor:
    """The zero BF16 ``post_weight`` vector of ``device`` (one per device, allocated on first use)."""
    index = _device_index(device)
    found = _zero_post_weights.get(index)
    if found is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cake_gated_residual_next_rmsnorm: the zero post_weight vector of device "
                f"cuda:{index} is allocated on first use, which cannot happen inside CUDA graph "
                "capture; call the function once (or warm_up) on this device before capturing"
            )
        found = torch.zeros(
            HIDDEN_SIZE, dtype=torch.bfloat16, device=torch.device("cuda", index)
        )
        _zero_post_weights[index] = found
    return found


def warm_up(
    device: torch.device, *, tensor_alpha: bool = True, has_post_weight: bool = True
) -> None:
    """Load the libraries of the call's route on ``device``, read their co-residency and allocate the zero
    ``post_weight`` vector where the call needs one, so that a later call can run inside CUDA graph capture."""
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_next_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    route = gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    index = _device_index(device)
    for name in _loader.route_modules(arch, route):
        _loader.load(name, arch)
        if _loader.MODULES[name]["grid_rule"]["kind"] == "persistent_rows":
            _loader.cluster_capacity(name, arch, index)
    if tensor_alpha and not has_post_weight:
        zero_post_weight(device)


def _check(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: {name} must be a contiguous {dtype} tensor of shape "
            f"{tuple(shape)} on {device} with {ALIGNMENT_BYTES}-byte aligned storage"
        )


def cake_gated_residual_next_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    next_weight: torch.Tensor,
    alpha: Optional[torch.Tensor],
    beta: torch.Tensor,
    residual_out: torch.Tensor,
    ffn_input: torch.Tensor,
    *,
    post_eps: float,
    next_eps: float,
    next_gain_center: float,
    post_weight: Optional[torch.Tensor] = None,
    post_gain_center: float = 0.0,
    alpha_scalar: Optional[float] = None,
) -> None:
    """``residual_out = alpha * residual + beta * RMSNorm(branch * (post_weight + post_gain_center))`` in FP32
    and ``ffn_input = bf16(RMSNorm(residual_out) * (next_weight + next_gain_center))``, one launch of the
    exported program whose window admits the row count.

    ``alpha`` is the FP32 gate vector, or ``None`` with ``alpha_scalar`` the scalar gate (``alpha`` is then
    unread); ``post_weight`` ``None`` drops the first multiply.  The outputs may alias ``residual`` and
    ``branch``.  An empty batch launches nothing.
    """
    device = branch.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_next_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if branch.ndim != 2 or branch.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: branch must be [rows, {HIDDEN_SIZE}], got {tuple(branch.shape)}"
        )
    rows = int(branch.shape[0])
    matrix = (rows, HIDDEN_SIZE)
    vector = (HIDDEN_SIZE,)
    _check(branch, "branch", shape=matrix, dtype=torch.bfloat16, device=device)
    _check(residual, "residual", shape=matrix, dtype=torch.float32, device=device)
    _check(
        next_weight, "next_weight", shape=vector, dtype=torch.bfloat16, device=device
    )
    _check(beta, "beta", shape=vector, dtype=torch.float32, device=device)
    _check(
        residual_out, "residual_out", shape=matrix, dtype=torch.float32, device=device
    )
    _check(ffn_input, "ffn_input", shape=matrix, dtype=torch.bfloat16, device=device)
    tensor_alpha = alpha_scalar is None
    if tensor_alpha:
        if alpha is None:
            raise ValueError(
                "cake_gated_residual_next_rmsnorm: pass the alpha tensor or alpha_scalar"
            )
        _check(alpha, "alpha", shape=vector, dtype=torch.float32, device=device)
        alpha_value = 1.0
    else:
        # The scalar programs never read the alpha slot; beta keeps the tensor ABI.
        alpha = beta
        alpha_value = float(alpha_scalar)  # type: ignore[arg-type]
    has_post_weight = post_weight is not None
    if post_weight is not None:
        _check(
            post_weight,
            "post_weight",
            shape=vector,
            dtype=torch.bfloat16,
            device=device,
        )
        gain = float(post_gain_center)
    elif tensor_alpha:
        # No dedicated program: the post-weight route with a zero vector and gain 1.0 is exact.
        post_weight = zero_post_weight(device)
        gain = 1.0
    else:
        # The no-post-weight programs never read the slot; next_weight keeps the tensor ABI.
        post_weight = next_weight
        gain = 0.0
    route = gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: no exported program of route {route!r} on {arch}"
        )
    if rows == 0:
        return
    if not is_cake_gated_residual_next_rmsnorm_supported(
        device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight, rows=rows
    ):
        # a declared gap of the route: the facade's "auto" runs the CuTe DSL kernel there
        raise ValueError(
            f"cake_gated_residual_next_rmsnorm: no exported program of route {route!r} on {arch} admits "
            f"{rows} rows"
        )
    name = _loader.selected_module(arch, route, rows)
    values = {
        "branch": branch,
        "residual": residual,
        "alpha": alpha,
        "beta": beta,
        "next_weight": next_weight,
        "post_weight": post_weight,
        "residual_out": residual_out,
        "ffn_input": ffn_input,
        "rows": rows,
        "post_eps": float(post_eps),
        "next_eps": float(next_eps),
        "post_gain_center": gain,
        "next_gain_center": float(next_gain_center),
        "alpha_value": alpha_value,
    }
    # The loader binds the current torch stream of the *current* device; make that the
    # tensors' device so a call made while another device is current keeps the order
    # of the stream its inputs were produced on.
    with torch.cuda.device(device):
        _loader.run_module(name, values, rows=rows, device_index=_device_index(device))


def gated_residual_rmsnorm_variant(weight_before_norm: bool, enable_pdl: bool) -> str:
    """The variant a gated residual RMSNorm call maps onto: ``mode0`` (no ``weight``) or ``mode1`` (``weight +
    gain_center`` applied to ``branch`` before its norm), with ``_pdl`` for a programmatic dependent launch."""
    return f"mode{int(bool(weight_before_norm))}{'_pdl' if enable_pdl else ''}"


def gated_residual_rmsnorm_route(weight_before_norm: bool, enable_pdl: bool) -> str:
    """The route a gated residual RMSNorm call launches from (its variant's)."""
    return route_key(
        GATED_RESIDUAL_RMSNORM,
        gated_residual_rmsnorm_variant(weight_before_norm, enable_pdl),
    )


def is_cake_gated_residual_rmsnorm_supported(
    device: torch.device,
    *,
    weight_before_norm: bool = False,
    enable_pdl: bool = False,
    rows: Optional[int] = None,
) -> bool:
    """Whether this package ships a program for ``device`` and the call's variant (``weight`` applied before the norm
    or absent, programmatic dependent launch or not) -- and, with ``rows``, one whose window admits that row count
    (a route's windows may leave gaps where the kernel lane measured the FlashInfer kernel ahead; a window of a PDL
    route may name the plain build of its program, launched without the attribute).  The tensor contract is
    :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    return _admits(
        arch, gated_residual_rmsnorm_route(weight_before_norm, enable_pdl), rows
    )


def _check_row(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % ROW_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_gated_residual_rmsnorm: {name} must be a contiguous {dtype} tensor of shape "
            f"{tuple(shape)} on {device} with {ROW_ALIGNMENT_BYTES}-byte aligned storage"
        )


# ----------------------------------------------------------------------------- launch plans
#
# A public call resolves to one record of its route (the loader's ``selected_module``), one launch grid and one
# positional argument order of the record's tvm-ffi entry.  None of that depends on the tensors' storage, so a plan
# keyed by the call signature (device, row / token / head counts, strides, variant) is prepared once and a repeated
# call only checks its tensors' contract and launches: no device-capability query, no window scan, no attribute
# lookup of the module, no torch stream object (the raw current-stream handle of the tensors' device drives the
# tvm-ffi stream context; the binding itself guards the device).  The slow path keeps every check and message.

_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_STREAM_CONTEXT = getattr(tvm_ffi, "StreamContext", None)
_FFI_DEVICES: dict[int, Any] = {}
_LAUNCH_PLANS: dict[tuple[Any, ...], "_LaunchPlan"] = {}
_VECTOR = (HIDDEN_SIZE,)


def _ffi_device(index: int) -> Any:
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return device


class _LaunchPlan:
    """One record bound to one call signature: the loaded entry, its grid and its argument order, resolved once."""

    __slots__ = (
        "name",
        "rows",
        "device_index",
        "entry",
        "grid",
        "order",
        "ffi_device",
        "constants",
        "pack",
    )

    def __init__(
        self, name: str, arch: str, device_index: int, values: dict[str, Any], rows: int
    ) -> None:
        record = _loader.MODULES[name]
        self.name, self.rows, self.device_index = name, rows, device_index
        self.grid = _loader.launch_grid(
            record, rows, device_index=device_index, values=values
        )
        _loader.launch_args(
            record, values, self.grid
        )  # the plan's values name every argument
        grid_values = {
            "grid_x": self.grid[0],
            "grid_y": self.grid[1],
            "grid_z": self.grid[2],
        }
        logical = record["logical"]
        self.order = tuple(
            grid_values[key] if kind == "grid" else logical[key]
            for kind, key in record["arg_plan"]
        )
        self.entry = getattr(_loader.load(name, arch), record["ffi_entry"])
        self.ffi_device = _ffi_device(device_index)
        # ``pack`` turns one positional row -- the call's values in the order ``values`` lists them, followed by
        # ``constants`` -- into the entry's argument tuple without a dict or a comprehension per call.
        names = tuple(values)
        constants: list[Any] = []
        indices: list[int] = []
        for source in self.order:
            if type(source) is str:
                indices.append(names.index(source))
            else:
                indices.append(len(names) + len(constants))
                constants.append(source)
        self.constants = tuple(constants)
        if len(indices) == 1:
            only = indices[0]

            def pack(row: tuple[Any, ...], index: int = only) -> tuple[Any, ...]:
                return (row[index],)

            self.pack: Callable[[tuple[Any, ...]], tuple[Any, ...]] = pack
        else:
            self.pack = cast(
                Callable[[tuple[Any, ...]], tuple[Any, ...]],
                operator.itemgetter(*indices),
            )

    def args(self, values: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            values[source] if type(source) is str else source for source in self.order
        )


def _launch(plan: _LaunchPlan, args: tuple[Any, ...]) -> None:
    """Call the plan's entry on the current torch stream of its device (a CUDA-graph capture stream included)."""
    if _RAW_STREAM is None or _STREAM_CONTEXT is None:
        with torch.cuda.device(plan.device_index), tvm_ffi.use_torch_stream():
            plan.entry(*args)
        return
    with _STREAM_CONTEXT(plan.ffi_device, _RAW_STREAM(plan.device_index)):
        plan.entry(*args)


def clear_launch_plans() -> None:
    """Forget the prepared launch plans (after the registry or the routes change under test)."""
    _LAUNCH_PLANS.clear()


def _prepare_gated_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    *,
    enable_pdl: bool,
) -> Optional[_LaunchPlan]:
    """The full contract check of one call and its launch plan (``None`` for an empty batch, which launches nothing)."""
    device = branch.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_gated_residual_rmsnorm supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if branch.ndim != 2 or branch.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"cake_gated_residual_rmsnorm: branch must be [rows, {HIDDEN_SIZE}], got {tuple(branch.shape)}"
        )
    rows = int(branch.shape[0])
    matrix = (rows, HIDDEN_SIZE)
    _check_row(branch, "branch", shape=matrix, dtype=torch.bfloat16, device=device)
    _check_row(residual, "residual", shape=matrix, dtype=torch.float32, device=device)
    _check_row(alpha, "alpha", shape=_VECTOR, dtype=torch.float32, device=device)
    _check_row(beta, "beta", shape=_VECTOR, dtype=torch.float32, device=device)
    _check_row(out, "out", shape=matrix, dtype=torch.float32, device=device)
    weight_before_norm = weight is not None
    if weight is not None:
        _check_row(weight, "weight", shape=_VECTOR, dtype=torch.bfloat16, device=device)
    route = gated_residual_rmsnorm_route(weight_before_norm, enable_pdl)
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_gated_residual_rmsnorm: no exported program of route {route!r} on {arch}"
        )
    if rows == 0:
        return None
    if not is_cake_gated_residual_rmsnorm_supported(
        device, weight_before_norm=weight_before_norm, enable_pdl=enable_pdl, rows=rows
    ):
        # a declared gap of the route: the facade's "auto" runs the CuTe DSL kernel there
        raise ValueError(
            f"cake_gated_residual_rmsnorm: no exported program of route {route!r} on {arch} admits "
            f"{rows} rows"
        )
    name = _loader.selected_module(arch, route, rows)
    values = {
        "branch": branch,
        "residual": residual,
        # The mode-0 programs never read the weight slot; branch itself keeps the BF16 tensor ABI.
        "weight": weight if weight is not None else branch,
        "alpha": alpha,
        "beta": beta,
        "out": out,
        "rows": rows,
        "eps": 0.0,
        "gain_center": 0.0,
    }
    key = (
        GATED_RESIDUAL_RMSNORM,
        branch.get_device(),
        rows,
        weight_before_norm,
        bool(enable_pdl),
    )
    plan = _LAUNCH_PLANS[key] = _LaunchPlan(
        name, arch, _device_index(device), values, rows
    )
    return plan


def cake_gated_residual_rmsnorm(
    branch: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    alpha: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
    gain_center: float = 0.0,
    enable_pdl: bool = False,
) -> None:
    """``out = alpha * residual + beta * RMSNorm(branch * (weight + gain_center))`` in FP32 (``weight`` ``None``
    drops the multiply), one launch of the exported program of the call's variant whose window admits the row
    count; ``enable_pdl`` selects the programmatic dependent launch build.  ``out`` may alias ``residual``.  An
    empty batch launches nothing.
    """
    shape = branch.shape
    index = branch.get_device()
    weight_before_norm = weight is not None
    plan = None
    if len(shape) == 2:
        plan = _LAUNCH_PLANS.get(
            (
                GATED_RESIDUAL_RMSNORM,
                index,
                shape[0],
                weight_before_norm,
                bool(enable_pdl),
            )
        )
    # The plan's key fixes the device, the row count, the variant and the launch mode; the rest of the contract
    # is one flat check per call (dtypes, shapes, same device, contiguous, 32-byte aligned storage).
    f32, bf16, align = torch.float32, torch.bfloat16, ROW_ALIGNMENT_BYTES
    if plan is None or not (
        shape[1] == HIDDEN_SIZE
        and branch.dtype is bf16
        and residual.dtype is f32
        and out.dtype is f32
        and alpha.dtype is f32
        and beta.dtype is f32
        and residual.shape == shape
        and out.shape == shape
        and alpha.shape == _VECTOR
        and beta.shape == _VECTOR
        and residual.get_device() == index
        and out.get_device() == index
        and alpha.get_device() == index
        and beta.get_device() == index
        and branch.is_contiguous()
        and residual.is_contiguous()
        and out.is_contiguous()
        and alpha.is_contiguous()
        and beta.is_contiguous()
        and (
            branch.data_ptr()
            | residual.data_ptr()
            | out.data_ptr()
            | alpha.data_ptr()
            | beta.data_ptr()
        )
        % align
        == 0
        and (
            weight is None
            or (
                weight.dtype is bf16
                and weight.shape == _VECTOR
                and weight.get_device() == index
                and weight.is_contiguous()
                and weight.data_ptr() % align == 0
            )
        )
    ):
        plan = _prepare_gated_residual_rmsnorm(
            branch, residual, weight, alpha, beta, out, enable_pdl=enable_pdl
        )
        if plan is None:
            return
    _launch(
        plan,
        plan.pack(
            (
                branch,
                residual,
                weight if weight is not None else branch,
                alpha,
                beta,
                out,
                plan.rows,
                float(eps),
                float(gain_center),
            )
            + plan.constants
        ),
    )


def rmsnorm_sigmoid_gate_variant(enable_pdl: bool = False) -> str:
    """The variant a sigmoid gate call maps onto: ``bf16`` (BF16 rows of 64), with ``_pdl`` for a programmatic
    dependent launch."""
    return f"bf16{'_pdl' if enable_pdl else ''}"


def rmsnorm_sigmoid_gate_route(enable_pdl: bool = False) -> str:
    """The route a sigmoid gate call launches from (its variant's).  A window of the ``_pdl`` route names the PDL
    build of a program, its late-signal PDL build or its plain build (launched without the attribute where the PDL
    build measured slower); the route covers every token count the plain route covers but for its declared gaps,
    where the facade keeps the CuTe DSL kernel under PDL."""
    return route_key(RMSNORM_SIGMOID_GATE, rmsnorm_sigmoid_gate_variant(enable_pdl))


def is_cake_rmsnorm_sigmoid_gate_supported(
    device: torch.device,
    *,
    heads: Optional[int] = None,
    tokens: Optional[int] = None,
    enable_pdl: bool = False,
) -> bool:
    """Whether this package ships a program for ``device`` and the call's launch form (plain, or the ``_pdl`` route
    under ``enable_pdl``) -- and, with ``heads``, one whose windows name that head count (a 2-D ``(rows, 64)`` call
    is one head), with ``tokens`` too, one whose window admits that token count.  The tensor contract is
    :mod:`flashinfer.norm`'s."""
    arch = _arch(device)
    if arch is None:
        return False
    route = rmsnorm_sigmoid_gate_route(enable_pdl)
    return _admits(arch, route, None if heads is None else tokens, heads)


def _gate_layout(tensor: torch.Tensor, name: str, device: torch.device) -> int:
    """The token stride of ``tensor`` after the layout checks of the contract: BF16 on the call's device, 2-D
    ``(tokens, 64)`` or 3-D ``(tokens, heads, 64)``, unit stride along the row, packed heads, a token stride that is a
    multiple of :data:`GATE_STRIDE_ELEMENTS` (16: a 32-byte multiple) and 32-byte aligned storage (the 16-head programs
    of sm_107a read and write 256-bit vectors; the package admits one alignment for every program)."""
    if tensor.dtype != torch.bfloat16 or tensor.device != device:
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must be a bfloat16 tensor on {device}"
        )
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must be [tokens, {HEAD_DIM}] or [tokens, heads, {HEAD_DIM}], "
            f"got {tuple(tensor.shape)}"
        )
    if (
        tensor.stride(-1) != 1
        or (tensor.ndim == 3 and tensor.stride(1) != HEAD_DIM)
        or tensor.stride(0) % GATE_STRIDE_ELEMENTS != 0
        or tensor.data_ptr() % ROW_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: {name} must hold contiguous rows of {HEAD_DIM} with packed heads, a token "
            f"stride that is a multiple of {GATE_STRIDE_ELEMENTS} elements and {ROW_ALIGNMENT_BYTES}-byte aligned "
            "storage"
        )
    return int(tensor.stride(0))


def _prepare_rmsnorm_sigmoid_gate(
    input: torch.Tensor, gate: torch.Tensor, *, enable_pdl: bool
) -> Optional[_LaunchPlan]:
    """The full contract check of one call and its launch plan (``None`` for an empty batch, which launches nothing)."""
    device = input.device
    arch = _arch(device)
    if arch is None:
        raise ValueError(
            "cake_rmsnorm_sigmoid_gate supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {device}"
        )
    if gate.shape != input.shape or gate.device != device:
        raise ValueError(
            "cake_rmsnorm_sigmoid_gate: gate must have the shape and device of input, got "
            f"{tuple(gate.shape)} on {gate.device} for {tuple(input.shape)} on {device}"
        )
    input_stride = _gate_layout(input, "input", device)
    gate_stride = _gate_layout(gate, "gate", device)
    tokens = int(input.shape[0])
    heads = int(input.shape[1]) if input.ndim == 3 else 1
    route = rmsnorm_sigmoid_gate_route(enable_pdl)
    if not _loader.route_available(arch, route):
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: no exported program of route {route!r} on {arch}"
        )
    if tokens == 0:
        return None
    if not is_cake_rmsnorm_sigmoid_gate_supported(
        device, heads=heads, tokens=tokens, enable_pdl=enable_pdl
    ):
        # a head count or a token count outside the route's windows: the facade's "auto" runs the CuTe DSL kernel
        raise ValueError(
            f"cake_rmsnorm_sigmoid_gate: no exported program of route {route!r} on {arch} admits "
            f"{tokens} tokens of {heads} heads"
        )
    name = _loader.selected_module(arch, route, tokens, heads)
    values = {
        "input": input,
        "gate": gate,
        "tokens": tokens,
        "heads": heads,
        "input_stride": input_stride,
        "gate_stride": gate_stride,
        "eps": 0.0,
    }
    key = (
        RMSNORM_SIGMOID_GATE,
        device.index,
        input.ndim,
        tokens,
        heads,
        input_stride,
        gate_stride,
        bool(enable_pdl),
    )
    plan = _LAUNCH_PLANS[key] = _LaunchPlan(
        name, arch, _device_index(device), values, tokens
    )
    return plan


def cake_rmsnorm_sigmoid_gate(
    input: torch.Tensor, gate: torch.Tensor, *, eps: float, enable_pdl: bool = False
) -> None:
    """``input = bf16(RMSNorm(input) * sigmoid(gate))`` per row of 64, in place, one launch of the exported program
    whose window admits the head count and the token count.  ``input`` and ``gate`` are ``(tokens, 64)`` or
    ``(tokens, heads, 64)`` with the same shape; their token strides may differ (``gate`` is typically a view of a
    wider projection).  ``enable_pdl`` selects the programmatic dependent launch route: its window names the PDL
    build of a program or its plain build (which launches without the attribute); a window naming no program, like a
    head count the routes do not name, is the facade's CuTe DSL branch (``ValueError`` here).  An empty batch
    launches nothing.
    """
    device = input.device
    shape = input.shape
    ndim = len(shape)
    plan = None
    if ndim in (2, 3):
        heads = shape[1] if ndim == 3 else 1
        input_stride, gate_stride = input.stride(0), gate.stride(0)
        plan = _LAUNCH_PLANS.get(
            (
                RMSNORM_SIGMOID_GATE,
                device.index,
                ndim,
                shape[0],
                heads,
                input_stride,
                gate_stride,
                bool(enable_pdl),
            )
        )
    bf16, align = torch.bfloat16, ROW_ALIGNMENT_BYTES
    if plan is None or not (
        shape[-1] == HEAD_DIM
        and gate.shape == shape
        and input.dtype is bf16
        and gate.dtype is bf16
        and gate.device == device
        and input.stride(-1) == 1
        and gate.stride(-1) == 1
        and (ndim == 2 or (input.stride(1) == HEAD_DIM and gate.stride(1) == HEAD_DIM))
        and input.data_ptr() % align == 0
        and gate.data_ptr() % align == 0
    ):
        plan = _prepare_rmsnorm_sigmoid_gate(input, gate, enable_pdl=enable_pdl)
        if plan is None:
            return
        heads = shape[1] if ndim == 3 else 1
        input_stride, gate_stride = input.stride(0), gate.stride(0)
    _launch(
        plan,
        plan.pack(
            (input, gate, plan.rows, heads, input_stride, gate_stride, float(eps))
            + plan.constants
        ),
    )


__all__ = [
    "ALIGNMENT_BYTES",
    "GATE_STRIDE_ELEMENTS",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GATED_RESIDUAL_NEXT_RMSNORM",
    "GATED_RESIDUAL_RMSNORM",
    "HEAD_DIM",
    "HIDDEN_SIZE",
    "OPS",
    "RMSNORM_SIGMOID_GATE",
    "ROW_ALIGNMENT_BYTES",
    "VARIANTS",
    "cake_gated_residual_next_rmsnorm",
    "cake_gated_residual_rmsnorm",
    "cake_rmsnorm_sigmoid_gate",
    "clear_launch_plans",
    "gated_residual_next_rmsnorm_route",
    "gated_residual_next_rmsnorm_variant",
    "gated_residual_rmsnorm_route",
    "gated_residual_rmsnorm_variant",
    "is_cake_gated_residual_next_rmsnorm_supported",
    "is_cake_gated_residual_rmsnorm_supported",
    "is_cake_rmsnorm_sigmoid_gate_supported",
    "rmsnorm_sigmoid_gate_route",
    "rmsnorm_sigmoid_gate_variant",
    "warm_up",
    "zero_post_weight",
]
