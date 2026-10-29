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

Backend of :mod:`flashinfer.cake_mhc_train`: Cake four-stream mHC training
forward / backward (hidden 4096, 4 streams; SM100 / SM103 / SM107) as one
self-contained package (host API bodies, the ``cake_jit`` loader and the
generated CUDA sources under ``csrc/``).  The stable, ``@flashinfer_api``-decorated
entry points and the full user-facing documentation live in
:mod:`flashinfer.cake_mhc_train`; import that module.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    GSTATS_COLS,
    HC_EPS,
    HIDDEN,
    HIDDEN_TOTAL,
    MAPPING_ROWS,
    RMS_EPS,
    SINKHORN_ITERS,
    SINKHORN_ITERS_MAX,
    STATS_COLS,
    STREAMS,
    arch_for_capability,
    route_applies,
    route_key,
    route_modules,
    selected_modules,
    supported_hidden_sizes,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The Sinkhorn iteration counts the delivered programs compile in: their chain is
# unrolled for the count of record, the default, which costs nothing at run time.
# Every other count from 1 to SINKHORN_ITERS_MAX runs too: the coefficient pass and
# the token-owner pass then launch their run-time-count programs, which take the
# count as a kernel argument (the other kernels never read it).  The epsilons are
# launch arguments of every program and never select one.
SHIPPED_SINKHORN_ITERS: tuple[int, ...] = (SINKHORN_ITERS,)
# Every kernel tensor is read with 256-bit vector loads or described to the
# tensor memory accelerator: contiguous storage, 32-byte aligned base.
TENSOR_ALIGNMENT_BYTES = 32
WORKSPACE_ALIGNMENT_BYTES = 256
# Bound of the per-token-count caches below (layouts, default-workspace views,
# launch plans); ``cake_jit.host_plan`` keeps the same bound.
_CACHE_ENTRIES = 4096
# The call paths of the per-call PDL argument (``cake_jit.PDL_PATHS``): the forward
# prepare, the backward from saved statistics, the backward after a replayed forward
# prefix (the autograd nodes know which; the public backward entry points are the
# backward path).
_PATH_FORWARD, _PATH_BACKWARD, _PATH_CHECKPOINT = _loader.PDL_PATHS
_BF16 = torch.bfloat16
_F32 = torch.float32
_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "uint32": torch.uint32,
}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


# The exported architecture of each CUDA device, resolved once per device
# index (the capability of a device never changes within a process).
_DEVICE_ARCH: dict[int, str] = {}


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Cake mHC training requires CUDA tensors, got {device}")
    index = _device_index(device)
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        capability = tuple(torch.cuda.get_device_capability(index))
        arch = arch_for_capability(capability)
        if arch is None:
            raise ValueError(
                f"Cake mHC training supports compute capability {sorted(ARCH_BY_CAPABILITY)} only, got {capability}"
            )
        _DEVICE_ARCH[index] = arch
    return arch


def is_cake_mhc_train_supported(
    device: torch.device, hidden: Optional[int] = None
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has every exported route."""

    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    if hidden is None:
        return bool(supported_hidden_sizes(arch))
    return int(hidden) in supported_hidden_sizes(arch)


# ----------------------------------------------------------------------------
# Validation


def _validate_tensor(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A kernel operand: exact shape, dtype and device, contiguous, 32-byte aligned storage.

    No layout is materialized silently: a view the kernels cannot read in place
    is a ``ValueError`` naming the argument (the caller decides whether to copy).
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        return tensor
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (strides {tuple(tensor.stride())} for shape {shape})"
        )
    if tensor.data_ptr() % TENSOR_ALIGNMENT_BYTES:
        raise ValueError(
            f"{name} must start at a {TENSOR_ALIGNMENT_BYTES}-byte aligned address"
        )
    return tensor


def _rows_of(X: Any) -> tuple[int, torch.device]:
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if X.ndim != 3 or tuple(X.shape[1:]) != (STREAMS, HIDDEN):
        raise ValueError(
            f"X must be a [T, {STREAMS}, {HIDDEN}] tensor, got {tuple(X.shape)}"
        )
    return int(X.shape[0]), X.device


def _check_constants(rms_eps: float, hc_eps: float, sinkhorn_iters: int) -> None:
    """Validate the three model constants of a call.

    ``rms_eps`` and ``hc_eps`` are launch arguments of the exported kernels (every
    mapping-pass and token-owner program takes them as kernel parameters), so any
    finite positive value runs on the compiled programs without a rebuild; the
    defaults are the model's ``1e-5`` / ``1e-6``.  ``sinkhorn_iters`` is an integer
    from 1 to :data:`SINKHORN_ITERS_MAX`: the counts of
    :data:`SHIPPED_SINKHORN_ITERS` (the default ``20``) run the compiled programs,
    every other count the run-time-count programs of the coefficient and
    token-owner passes; a count above :data:`SINKHORN_ITERS_MAX` is beyond what
    those programs were built for (``NotImplementedError``).
    """

    if not (math.isfinite(float(rms_eps)) and float(rms_eps) > 0.0):
        raise ValueError(f"rms_eps must be a finite positive float, got {rms_eps!r}")
    if not (math.isfinite(float(hc_eps)) and float(hc_eps) > 0.0):
        raise ValueError(f"hc_eps must be a finite positive float, got {hc_eps!r}")
    if isinstance(sinkhorn_iters, bool):
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        )
    try:
        iters = int(sinkhorn_iters)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        ) from None
    if iters != sinkhorn_iters or iters < 1:
        raise ValueError(
            f"sinkhorn_iters must be a positive integer, got {sinkhorn_iters!r}"
        )
    if iters > SINKHORN_ITERS_MAX:
        raise NotImplementedError(
            f"the exported Cake mHC training programs run sinkhorn_iters from 1 to {SINKHORN_ITERS_MAX} "
            f"(the run-time-count programs are built for at most {SINKHORN_ITERS_MAX}); got sinkhorn_iters={sinkhorn_iters!r}"
        )


def _output(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """A caller-provided output (validated like an input) or a fresh contiguous buffer."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return _validate_tensor(tensor, name, shape=shape, device=device, dtype=dtype)


# The (architecture, route kind) pairs the registry has already been checked
# for; the check itself is a pure function of the registry.
_ROUTES_CHECKED: set[tuple[str, str]] = set()


def _check_route(device: torch.device, arch: str, kind: str) -> None:
    if (arch, kind) in _ROUTES_CHECKED:
        return
    if not route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device)),
        hidden=HIDDEN,
        kind=kind,
    ):
        raise ValueError(
            f"no exported Cake mHC training route {route_key(HIDDEN, kind)!r} on {arch}; "
            f"supported hidden sizes: {supported_hidden_sizes(arch)}"
        )
    _ROUTES_CHECKED.add((arch, kind))


def _parameters(
    fn: Any, scale: Any, base: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = _validate_tensor(
        fn, "fn", shape=(MAPPING_ROWS, HIDDEN_TOTAL), device=device, dtype=_F32
    )
    scale = _validate_tensor(scale, "scale", shape=(3,), device=device, dtype=_F32)
    base = _validate_tensor(
        base, "base", shape=(MAPPING_ROWS,), device=device, dtype=_F32
    )
    return fn, scale, base


# ----------------------------------------------------------------------------
# Workspaces


def _workspace_bytes(kind: str, rows: int, device: torch.device) -> int:
    device = torch.device(device)
    rows = int(rows)
    if rows < 0:
        raise ValueError("T must be non-negative")
    arch = _device_arch(device)
    return int(
        _loader.workspace_bytes_upto(
            kind, rows, _loader.sm_count(_device_index(device)), arch
        )
    )


def cake_mhc_train_forward_workspace_bytes(T: int, device: torch.device) -> int:
    """Bytes of the caller-owned forward workspace for up to ``T`` tokens on ``device``.

    The forward routes (``forward_prepare`` and the ``forward_stats`` replay) hold
    one 256-byte ticket region (first, at the same offset for every token count),
    the BF16 pieces of ``fn`` in the mapping operand layout and the FP32
    per-K-unit partials of the mapping pass.  The size is the maximum over every
    token count up to ``T``, so a buffer sized for the largest token count
    serves every smaller one (and CUDA Graph replays at a smaller ``T``).  The
    ticket words must be zero on first use (allocate with ``torch.zeros`` or
    :func:`cake_mhc_train_forward_workspace`); the kernels leave them zero.
    ``T == 0`` needs no workspace (nothing is launched).
    """

    return _workspace_bytes("fwd", T, device)


def cake_mhc_train_backward_workspace_bytes(T: int, device: torch.device) -> int:
    """Bytes of the caller-owned backward workspace for up to ``T`` tokens on ``device``.

    The ``backward_finish`` route holds one 256-byte ticket region (first, at
    the same offset for every token count), the transposed BF16 pieces of
    ``fn``, the per-token records of the token-owner pass (``dz'`` pieces and the
    per-token scalars, padded to a multiple of 128 rows), the per-CTA parameter
    partials and the per-chunk FP32 partials of ``dfn``.  The size is the
    maximum over every token count up to ``T`` (see
    :func:`cake_mhc_train_forward_workspace_bytes`).
    """

    return _workspace_bytes("bwd", T, device)


def cake_mhc_train_forward_workspace(T: int, device: torch.device) -> torch.Tensor:
    """Allocate a zero-initialized forward workspace (see ``..._forward_workspace_bytes``)."""

    nbytes = cake_mhc_train_forward_workspace_bytes(T, device)
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_mhc_train_backward_workspace(T: int, device: torch.device) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace (see ``..._backward_workspace_bytes``)."""

    nbytes = cake_mhc_train_backward_workspace_bytes(T, device)
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_mhc_train_reset_workspace(workspace: torch.Tensor) -> None:
    """Return a workspace to its first-use state after an interrupted launch.

    The kernels' unit tickets self-reset at the end of every launch; a launch
    that did not complete (an asynchronous error, a cancelled stream capture)
    can leave them non-zero.  Zeroing the whole buffer restores the contract for
    every token count the buffer serves; no other region needs a value.
    """

    if (
        not isinstance(workspace, torch.Tensor)
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
    ):
        raise ValueError("workspace must be a 1-D uint8 tensor")
    workspace.zero_()


# The byte layout of one (workspace kind, token count, SM count, architecture)
# -- region dtypes, shapes, contiguous strides and element offsets -- is
# computed once;
# the region views of a caller workspace are then one ``as_strided`` per
# region on a typed view of the buffer.
_LAYOUTS: dict[tuple[str, int, int, str], dict[str, Any]] = {}


def _layout(kind: str, rows: int, sm_count: int, arch: str) -> dict[str, Any]:
    key = (kind, rows, sm_count, arch)
    found = _LAYOUTS.get(key)
    if found is None:
        regions = _loader.workspace_regions(kind, rows, sm_count, arch)
        plan = []
        for name, region in regions["regions"].items():
            dtype = _TORCH_DTYPES[region["dtype"]]
            shape = tuple(int(dim) for dim in region["shape"])
            strides = []
            step = 1
            for dim in reversed(shape):
                strides.append(step)
                step *= dim
            element_offset = region["offset"] // dtype.itemsize
            plan.append((name, dtype, shape, tuple(reversed(strides)), element_offset))
        found = {"total_bytes": int(regions["total_bytes"]), "plan": tuple(plan)}
        if len(_LAYOUTS) >= _CACHE_ENTRIES:
            _LAYOUTS.clear()
        _LAYOUTS[key] = found
    return found


def _workspace_views(
    workspace: Any, kind: str, *, rows: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """The region views of ``workspace`` for exactly ``rows`` tokens (validated against the layout)."""

    layout = _layout(
        kind, rows, _loader.sm_count(_device_index(device)), _device_arch(device)
    )
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    if (
        workspace.device != device
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
    ):
        raise ValueError(
            f"workspace must be a 1-D uint8 CUDA tensor on {device}, got "
            f"{workspace.dtype} {tuple(workspace.shape)} on {workspace.device}"
        )
    if (
        not workspace.is_contiguous()
        or workspace.data_ptr() % WORKSPACE_ALIGNMENT_BYTES != 0
    ):
        raise ValueError(
            f"workspace must be contiguous and {WORKSPACE_ALIGNMENT_BYTES}-byte aligned"
        )
    total = layout["total_bytes"]
    if workspace.numel() < total:
        helper = (
            "cake_mhc_train_forward_workspace_bytes"
            if kind == "fwd"
            else "cake_mhc_train_backward_workspace_bytes"
        )
        raise ValueError(
            f"workspace holds {workspace.numel()} bytes; T={rows} needs {total} ({helper})"
        )
    base = workspace[:total]  # a multiple of 256 bytes: every typed view divides it
    typed: dict[torch.dtype, torch.Tensor] = {}
    views: dict[str, torch.Tensor] = {}
    for name, dtype, shape, strides, offset in layout["plan"]:
        flat = typed.get(dtype)
        if flat is None:
            flat = typed[dtype] = base.view(dtype)
        views[name] = flat.as_strided(shape, strides, flat.storage_offset() + offset)
    if kind == "bwd":
        views["dzp_rows"] = views["dzp"][:rows]
    return views


# Default workspaces of the convenience path (``workspace=None``): one buffer
# per (device, kind, power-of-two token bucket), allocated on first use, so a
# caller that passes no workspace allocates at most log2(T_max) buffers per
# device while the caller-held path never allocates.
_DEFAULT_WORKSPACES: dict[tuple[int, str, int], torch.Tensor] = {}


def _default_workspace(kind: str, rows: int, device: torch.device) -> torch.Tensor:
    bucket = 1 << (max(int(rows), 1) - 1).bit_length()
    key = (_device_index(device), kind, bucket)
    found = _DEFAULT_WORKSPACES.get(key)
    if found is None:
        found = torch.zeros(
            _workspace_bytes(kind, bucket, device), dtype=torch.uint8, device=device
        )
        _DEFAULT_WORKSPACES[key] = found
    return found


# The region views of the default workspaces, per (device, kind, token
# count): those buffers live for the whole process, so their views can too.
_DEFAULT_VIEWS: dict[tuple[int, str, int], dict[str, torch.Tensor]] = {}


def _workspace_for(
    workspace: Optional[torch.Tensor], kind: str, rows: int, device: torch.device
) -> dict[str, torch.Tensor]:
    if workspace is None:
        key = (_device_index(device), kind, rows)
        views = _DEFAULT_VIEWS.get(key)
        if views is None:
            views = _workspace_views(
                _default_workspace(kind, rows, device), kind, rows=rows, device=device
            )
            if len(_DEFAULT_VIEWS) >= _CACHE_ENTRIES:
                _DEFAULT_VIEWS.clear()
            _DEFAULT_VIEWS[key] = views
        return views
    return _workspace_views(workspace, kind, rows=rows, device=device)


# ----------------------------------------------------------------------------
# Launch plans
#
# The modules one public call launches, their grids and their argument order
# are pure functions of (architecture, route, token count, device); the plan
# below resolves them once per such key and keeps the bound FFI entry of every
# module, so a call fills the tensor slots of each argument template and
# launches.  The route runs on the current torch stream of the tensors'
# device, set once for the whole route.

_TVM_FFI: Any = None
_TVM_DEVICES: dict[int, Any] = {}


def _tvm_ffi() -> Any:
    global _TVM_FFI
    if _TVM_FFI is None:
        import tvm_ffi

        _TVM_FFI = tvm_ffi
    return _TVM_FFI


def _tvm_device(device_index: int) -> Any:
    found = _TVM_DEVICES.get(device_index)
    if found is None:
        found = _TVM_DEVICES[device_index] = _tvm_ffi().device("cuda", device_index)
    return found


class _Launch:
    """One module launch of a route plan: the bound FFI entry, the argument
    template (grid and record-fixed scalars filled in) and the template
    positions a call fills from its logical values."""

    __slots__ = ("run", "template", "slots")

    def __init__(
        self, run: Any, template: list[Any], slots: tuple[tuple[int, str], ...]
    ) -> None:
        self.run, self.template, self.slots = run, template, slots


_PLANS: dict[tuple[str, str, int, int, str, bool], tuple[_Launch, ...]] = {}


def _route_plan(
    arch: str,
    kind: str,
    rows: int,
    device_index: int,
    path: str,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> tuple[_Launch, ...]:
    """The launches of one route for ``rows`` tokens on ``device_index`` along ``path``
    (one of ``cake_jit.PDL_PATHS``: it fixes the per-call PDL argument of the records
    whose binding takes one, :func:`cake_jit.enable_pdl`).  A ``sinkhorn_iters``
    other than the count of record selects the run-time-count records; the count
    itself is a launch value, so one plan serves every such count."""

    run_time = int(sinkhorn_iters) != SINKHORN_ITERS
    key = (arch, kind, rows, device_index, path, run_time)
    plan = _PLANS.get(key)
    if plan is None:
        launches = []
        sm_count = _loader.sm_count(device_index)
        for name in selected_modules(
            arch,
            HIDDEN,
            kind,
            rows=rows,
            sm_count=sm_count,
            sinkhorn_iters=sinkhorn_iters,
        ):
            record = _loader.MODULES[name]
            logical = record["logical"]
            fixed = _loader.RECORD_SCALARS.get(name, {})
            grid_x, grid_y, grid_z = _loader.launch_grid(
                record, rows, device_index=device_index
            )
            grid_values = {"grid_x": grid_x, "grid_y": grid_y, "grid_z": grid_z}
            template: list[Any] = []
            slots: list[tuple[int, str]] = []
            for position, (arg_kind, arg_key) in enumerate(record["arg_plan"]):
                if arg_kind == "grid":
                    template.append(grid_values[arg_key])
                    continue
                if arg_kind == "launch":
                    template.append(_loader.enable_pdl(name, rows, arch, path))
                    continue
                logical_name = logical[arg_key]
                if logical_name in fixed:
                    template.append(fixed[logical_name])
                    continue
                template.append(None)
                slots.append((position, logical_name))
            module = _loader.load(name, arch)
            launches.append(
                _Launch(getattr(module, record["ffi_entry"]), template, tuple(slots))
            )
        if len(_PLANS) >= _CACHE_ENTRIES:
            _PLANS.clear()
        plan = _PLANS[key] = tuple(launches)
    return plan


def _run_route(
    arch: str,
    kind: str,
    values: dict[str, Any],
    *,
    rows: int,
    device: torch.device,
    path: str,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> None:
    device_index = _device_index(device)
    plan = _route_plan(arch, kind, rows, device_index, path, sinkhorn_iters)
    stream = torch.cuda.current_stream(device_index).cuda_stream
    with _tvm_ffi().use_raw_stream(_tvm_device(device_index), stream):
        for launch in plan:
            args = launch.template.copy()
            try:
                for position, logical_name in launch.slots:
                    args[position] = values[logical_name]
            except KeyError as missing:
                raise ValueError(
                    f"the {route_key(HIDDEN, kind)!r} route needs the logical value {missing.args[0]!r}"
                ) from None
            launch.run(*args)


# ----------------------------------------------------------------------------
# Entry points


def _forward_values(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    ws: dict[str, torch.Tensor],
    *,
    rows: int,
    device: torch.device,
    rms_eps: float = RMS_EPS,
    hc_eps: float = HC_EPS,
    sinkhorn_iters: int = SINKHORN_ITERS,
) -> dict[str, Any]:
    plan = _loader.host_plan(
        rows, _loader.sm_count(_device_index(device)), _device_arch(device)
    )
    # The fn split writes the forward operand layout only (``want_t = 0`` is the
    # record's own scalar); its transposed-layout pointer is bound to the same
    # region, exactly like the source launcher binds an unused parameter.
    return {
        "fn": fn,
        "fnp": ws["fnp"],
        "fnTp": ws["fnp"],
        "x": X,
        "x_flat": X.view(rows, HIDDEN_TOTAL),
        "spart": ws["spart"],
        "qpart": ws["qpart"],
        "ticket": ws["ticket"],
        "scale": scale,
        "base": base,
        "stats": stats,
        "rows": rows,
        "m_tiles": plan["m_tiles"],
        "log2_pk": plan["log2_pk"],
        "p_k": plan["p_k"],
        "t_pad": plan["t_pad"],
        "rms_eps": float(rms_eps),
        "hc_eps": float(hc_eps),
        "sinkhorn_iters": int(sinkhorn_iters),
    }


def cake_mhc_train_forward_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    out: Optional[torch.Tensor] = None,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Forward mapping, coefficients and collapse: ``U`` and the per-token statistics.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams (``T`` is a runtime
        scalar; ``T == 0`` returns empty outputs without launching).
    fn : torch.Tensor
        FP32 ``[24, 16384]`` contiguous mapping matrix.
    scale : torch.Tensor
        FP32 ``[3]`` group scales of the pre / post / Sinkhorn logits.
    base : torch.Tensor
        FP32 ``[24]`` logit offsets.
    rms_eps : float
        Normalization epsilon (default ``1e-5``): a launch argument of the exported
        kernels, so any finite positive value runs on the same compiled programs.
    hc_eps : float
        Coefficient epsilon (default ``1e-6``): a launch argument like ``rms_eps``.
    sinkhorn_iters : int
        Sinkhorn iteration count, ``1 .. SINKHORN_ITERS_MAX`` (64; default ``20``).
        The default runs the compiled programs, whose chain is unrolled for it;
        every other count runs the run-time-count programs of the coefficient and
        token-owner passes.  A larger count raises ``NotImplementedError``.
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4096]`` contiguous ``U``.
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_forward_workspace`;
        a per-device default buffer is used when omitted.

    Returns
    -------
    ``(U, stats)``: BF16 ``U[t, d] = BF16(sum_i pre[t, i] X[t, i, d])`` and the
    FP32 ``[T, 64]`` per-token record ``r | z[24] | pre[4] | post[4] | C[16] | pad[15]``
    (256 bytes per token) that ``forward_finish``, ``backward_prepare`` and
    ``backward_finish`` consume.  No ``[T, 16384]`` or ``[T, 4, 4096]`` FP32
    temporary is produced.

    The launches run on the current torch stream of the tensors' device.
    """

    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    fn, scale, base = _parameters(fn, scale, base, device)
    _check_route(device, arch, "fwd_prepare")
    U = _output(out, "out", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    stats = _output(
        stats_out, "stats_out", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    if rows == 0:
        return U, stats
    ws = _workspace_for(workspace, "fwd", rows, device)
    values = _forward_values(
        X,
        fn,
        scale,
        base,
        stats,
        ws,
        rows=rows,
        device=device,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    values["u"] = U
    _run_route(
        arch,
        "fwd_prepare",
        values,
        rows=rows,
        device=device,
        path=_PATH_FORWARD,
        sinkhorn_iters=sinkhorn_iters,
    )
    return U, stats


def cake_mhc_train_forward_stats(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    stats_out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Recompute the per-token statistics of ``forward_prepare`` (checkpoint replay).

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_train_forward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 64]`` contiguous statistics record.
    workspace : torch.Tensor, optional
        Caller-owned forward workspace (:func:`cake_mhc_train_forward_workspace`).

    Returns
    -------
    FP32 ``[T, 64]`` ``stats``, bitwise identical to the record
    :func:`cake_mhc_train_forward_prepare` returns for the same inputs on the same
    device: the route is the strict prefix of the forward (fn split, mapping
    pass, coefficients) under the same work partition, without the ``U`` write.

    The launches run on the current torch stream of the tensors' device.
    """

    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    fn, scale, base = _parameters(fn, scale, base, device)
    _check_route(device, arch, "fwd_recompute")
    stats = _output(
        stats_out, "stats_out", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    if rows == 0:
        return stats
    ws = _workspace_for(workspace, "fwd", rows, device)
    values = _forward_values(
        X,
        fn,
        scale,
        base,
        stats,
        ws,
        rows=rows,
        device=device,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    _run_route(
        arch,
        "fwd_recompute",
        values,
        rows=rows,
        device=device,
        path=_PATH_CHECKPOINT,
        sinkhorn_iters=sinkhorn_iters,
    )
    return stats


def cake_mhc_train_forward_finish(
    X: torch.Tensor,
    B: torch.Tensor,
    stats: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Residual mixing: ``Y[t, j, d] = BF16(post[t, j] B[t, d] + sum_i C[t, i, j] X[t, i, d])``.

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams (the ``forward_prepare`` input).
    B : torch.Tensor
        BF16 ``[T, 4096]`` contiguous output of the external branch applied to ``U``.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record returned by :func:`cake_mhc_train_forward_prepare`
        (or :func:`cake_mhc_train_forward_stats`).
    out : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, 4096]`` contiguous ``Y``.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, 4096]``; the FP32 sum is rounded once.

    The launches run on the current torch stream of the tensors' device.
    """

    rows, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    B = _validate_tensor(B, "B", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "fwd_finish")
    Y = _output(out, "out", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16)
    if rows == 0:
        return Y
    values = {"x": X, "b": B, "stats": stats, "y": Y, "rows": rows}
    _run_route(arch, "fwd_finish", values, rows=rows, device=device, path=_PATH_FORWARD)
    return Y


def cake_mhc_train_backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor] = None,
    grad_stats_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Branch-output gradient and the per-token gradient statistics (before the branch backward).

    Parameters
    ----------
    dY : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous gradient of ``Y``.
    B : torch.Tensor
        BF16 ``[T, 4096]`` contiguous branch output of the forward.
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams of the forward.
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward.
    dB : torch.Tensor, optional
        Preallocated BF16 ``[T, 4096]`` contiguous output.
    grad_stats_out : torch.Tensor, optional
        Preallocated FP32 ``[T, 32]`` contiguous output.

    Returns
    -------
    ``(dB, grad_stats)``: ``dB[t, d] = BF16(sum_j post[t, j] dY[t, j, d])`` (one
    rounding of the full FP32 sum) and the FP32 ``[T, 32]`` record
    ``dpost[4] | dC[16] | zeros[12]`` (``dpost[t, j] = sum_d dY B``,
    ``dC[t, i, j] = sum_d X[t, i, d] dY[t, j, d]``) that ``backward_finish`` consumes.

    The launches run on the current torch stream of the tensors' device.
    """

    return _backward_prepare(
        dY, B, X, stats, dB=dB, grad_stats_out=grad_stats_out, path=_PATH_BACKWARD
    )


def _backward_prepare(
    dY: torch.Tensor,
    B: torch.Tensor,
    X: torch.Tensor,
    stats: torch.Tensor,
    *,
    dB: Optional[torch.Tensor],
    grad_stats_out: Optional[torch.Tensor],
    path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`cake_mhc_train_backward_prepare` along ``path`` (``_PATH_BACKWARD`` from
    saved statistics, ``_PATH_CHECKPOINT`` after the forward prefix was replayed: the
    autograd nodes know which): the path fixes the per-call PDL argument of the
    backward prepare's records (:func:`cake_jit.enable_pdl`)."""

    rows, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    dY = _validate_tensor(
        dY, "dY", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    B = _validate_tensor(B, "B", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "bwd_prepare")
    dB = _output(dB, "dB", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    gstats = _output(
        grad_stats_out,
        "grad_stats_out",
        shape=(rows, GSTATS_COLS),
        device=device,
        dtype=_F32,
    )
    if rows == 0:
        return dB, gstats
    values = {
        "dy": dY,
        "b": B,
        "x": X,
        "stats": stats,
        "db": dB,
        "gstats": gstats,
        "rows": rows,
    }
    _run_route(arch, "bwd_prepare", values, rows=rows, device=device, path=path)
    return dB, gstats


def cake_mhc_train_backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    accumulate: bool = False,
    workspace: Optional[torch.Tensor] = None,
    dX: Optional[torch.Tensor] = None,
    dfn: Optional[torch.Tensor] = None,
    dscale: Optional[torch.Tensor] = None,
    dbase: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Input gradient and parameter gradients (after the branch backward produced ``dU``).

    Parameters
    ----------
    X : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous residual streams of the forward.
    dY : torch.Tensor
        BF16 ``[T, 4, 4096]`` contiguous gradient of ``Y``.
    dU : torch.Tensor
        BF16 ``[T, 4096]`` contiguous gradient of ``U`` from the branch backward.
    B : torch.Tensor, optional
        BF16 ``[T, 4096]`` branch output of the forward; accepted for the
        prepare / finish symmetry and validated when given, not read (its
        contribution is already in ``grad_stats``).
    fn, scale, base : torch.Tensor
        The parameters of the forward (FP32 ``[24, 16384]``, ``[3]``, ``[24]``).
    stats : torch.Tensor
        FP32 ``[T, 64]`` record of the forward (saved or recomputed with
        :func:`cake_mhc_train_forward_stats`; both give bitwise the same gradients).
    grad_stats : torch.Tensor
        FP32 ``[T, 32]`` record returned by :func:`cake_mhc_train_backward_prepare`.
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    accumulate : bool
        ``False`` (default): ``dfn``, ``dscale`` and ``dbase`` are written.
        ``True``: the caller passes all three and the fixed-order sums are added
        to their current values (one read-modify-write per element by its single
        owner, so four consecutive accumulations are reproducible).
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from :func:`cake_mhc_train_backward_workspace`;
        a per-device default buffer is used when omitted.
    dX : torch.Tensor, optional
        Preallocated BF16 ``[T, 4, 4096]`` contiguous output.
    dfn, dscale, dbase : torch.Tensor, optional
        Preallocated FP32 ``[24, 16384]``, ``[3]``, ``[24]`` contiguous outputs
        (required with ``accumulate=True``).

    Returns
    -------
    ``(dX, dfn, dscale, dbase)``: the BF16 ``dX`` sums the mapping, collapse and
    residual-mixing contributions in FP32 and rounds once; the FP32 parameter
    gradients are fixed-order reductions, bitwise reproducible for a given token
    count on a given device.  With ``T == 0`` nothing is launched: ``dX`` is
    empty and the parameter gradients are zeroed (``accumulate=False``) or left
    untouched (``accumulate=True``).

    The launches run on the current torch stream of the tensors' device.
    """

    return _backward_finish(
        X,
        dY,
        dU,
        B,
        fn,
        scale,
        base,
        stats,
        grad_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        accumulate=accumulate,
        workspace=workspace,
        dX=dX,
        dfn=dfn,
        dscale=dscale,
        dbase=dbase,
        path=_PATH_BACKWARD,
    )


def _backward_finish(
    X: torch.Tensor,
    dY: torch.Tensor,
    dU: torch.Tensor,
    B: Optional[torch.Tensor],
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    stats: torch.Tensor,
    grad_stats: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    accumulate: bool,
    workspace: Optional[torch.Tensor],
    dX: Optional[torch.Tensor],
    dfn: Optional[torch.Tensor],
    dscale: Optional[torch.Tensor],
    dbase: Optional[torch.Tensor],
    path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`cake_mhc_train_backward_finish` along ``path`` (``_PATH_BACKWARD`` from
    saved statistics, ``_PATH_CHECKPOINT`` after the forward prefix was replayed: the
    autograd nodes know which): the path fixes the per-call PDL argument of the
    column-slab and reduction records (:func:`cake_jit.enable_pdl`)."""

    _check_constants(rms_eps, hc_eps, sinkhorn_iters)
    rows, device = _rows_of(X)
    arch = _device_arch(device)
    X = _validate_tensor(
        X, "X", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    dY = _validate_tensor(
        dY, "dY", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16
    )
    dU = _validate_tensor(dU, "dU", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    if B is not None:
        _validate_tensor(B, "B", shape=(rows, HIDDEN), device=device, dtype=_BF16)
    fn, scale, base = _parameters(fn, scale, base, device)
    stats = _validate_tensor(
        stats, "stats", shape=(rows, STATS_COLS), device=device, dtype=_F32
    )
    grad_stats = _validate_tensor(
        grad_stats, "grad_stats", shape=(rows, GSTATS_COLS), device=device, dtype=_F32
    )
    _check_route(device, arch, "bwd_finish")
    if accumulate and (dfn is None or dscale is None or dbase is None):
        raise ValueError(
            "accumulate=True needs the dfn, dscale and dbase buffers to accumulate into"
        )
    dX = _output(dX, "dX", shape=(rows, STREAMS, HIDDEN), device=device, dtype=_BF16)
    dfn = _output(
        dfn, "dfn", shape=(MAPPING_ROWS, HIDDEN_TOTAL), device=device, dtype=_F32
    )
    dscale = _output(dscale, "dscale", shape=(3,), device=device, dtype=_F32)
    dbase = _output(dbase, "dbase", shape=(MAPPING_ROWS,), device=device, dtype=_F32)
    if rows == 0:
        if not accumulate:
            dfn.zero_()
            dscale.zero_()
            dbase.zero_()
        return dX, dfn, dscale, dbase
    ws = _workspace_for(workspace, "bwd", rows, device)
    plan = _loader.host_plan(
        rows, _loader.sm_count(_device_index(device)), _device_arch(device)
    )
    # The fn split writes the transposed operand layout only (``want_t = 1`` is
    # the record's own scalar); its forward-layout pointer is bound to the same
    # region.  The column-slab pass describes the ``[T, 4, 4096]`` tensors as
    # ``[T, 16384]`` and reads the first ``T`` rows of the padded ``dz'`` record;
    # its dynamic-schedule form draws its work units from the workspace's
    # self-resetting ticket word.
    values: dict[str, Any] = {
        "fn": fn,
        "fnTp": ws["fnTp"],
        "fnp": ws["fnTp"],
        "x": X,
        "du": dU,
        "stats": stats,
        "gstats": grad_stats,
        "scale": scale,
        "base": base,
        "dzp": ws["dzp"],
        "tokp": ws["tokp"],
        "param_partial": ws["param_partial"],
        "rows": rows,
        "t_pad": plan["t_pad"],
        "rows_per_cta": plan["rows_per_cta"],
        "hc_eps": float(hc_eps),
        "sinkhorn_iters": int(sinkhorn_iters),
        "x_flat": X.view(rows, HIDDEN_TOTAL),
        "dy_flat": dY.view(rows, HIDDEN_TOTAL),
        "dzp_rows": ws["dzp_rows"],
        "dx_flat": dX.view(rows, HIDDEN_TOTAL),
        "dfn_part": ws["dfn_part"],
        "ticket": ws["ticket"],
        "parts": plan["parts"],
        "rows_per_chunk": plan["rows_per_chunk"],
        "dfn": dfn,
        "dscale": dscale,
        "dbase": dbase,
        "chunks": plan["chunks"],
        "accumulate": 1 if accumulate else 0,
    }
    _run_route(
        arch,
        "bwd_finish",
        values,
        rows=rows,
        device=device,
        path=path,
        sinkhorn_iters=sinkhorn_iters,
    )
    return dX, dfn, dscale, dbase


# ----------------------------------------------------------------------------
# Autograd


class _MHCTrainState:
    """What the prepare and finish autograd nodes of one layer application share.

    The parameters (detached), the constants, the workspaces and, transiently,
    ``stats``: between the prepare and the finish forward.  Nothing the forward
    produced stays on the state once the finish forward has run: ``stats``
    travels through ``save_for_backward`` like the inputs, so an activation
    checkpoint (non-reentrant ``torch.utils.checkpoint``) keeps no tensor of the
    layer alive.

    The finish node's backward hands ``dY``, the gradient statistics and, under
    ``save_stats=False``, the statistics it replayed to the prepare node's
    backward on autograd edges (:class:`_MHCTrainHandle`), so the autograd engine
    releases them once the prepare node's backward has run, however long the
    graph stays alive.  ``dY``, ``grad_stats`` and ``stats`` carry them on the
    state only from a finish node without those edges whose prepare node is
    ``recorded`` (a finish forward rerun by a reentrant checkpoint region that
    starts after the prepare); the prepare node's backward empties them.
    """

    __slots__ = (
        "fn",
        "scale",
        "base",
        "stats",
        "grad_stats",
        "dY",
        "recorded",
        "save_stats",
        "rms_eps",
        "hc_eps",
        "sinkhorn_iters",
        "workspace",
        "backward_workspace",
    )

    def __init__(
        self,
        *,
        save_stats: bool = True,
        rms_eps: float = 1e-5,
        hc_eps: float = 1e-6,
        sinkhorn_iters: int = 20,
        workspace: Optional[torch.Tensor] = None,
        backward_workspace: Optional[torch.Tensor] = None,
    ) -> None:
        self.fn = self.scale = self.base = None
        self.stats = self.grad_stats = self.dY = None
        self.recorded = False
        self.save_stats = bool(save_stats)
        self.rms_eps, self.hc_eps, self.sinkhorn_iters = rms_eps, hc_eps, sinkhorn_iters
        self.workspace, self.backward_workspace = workspace, backward_workspace

    def replay(self, X: torch.Tensor) -> torch.Tensor:
        """The bitwise-identical replay of the forward prefix's ``stats``."""

        return cake_mhc_train_forward_stats(
            X,
            self.fn,
            self.scale,
            self.base,
            rms_eps=self.rms_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.sinkhorn_iters,
            workspace=self.workspace,
        )

    def take(
        self,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``(dY, grad_stats, stats)`` as a finish node without edges left them; the state lets go of them."""

        held = self.dY, self.grad_stats, self.stats
        self.dY = self.grad_stats = self.stats = None
        return held


class _MHCTrainHandle:
    """What :func:`cake_mhc_train_prepare` returns beside ``U``: the shared state and the prepare node's edges.

    The edges are extra outputs of the prepare node: one-element, zero-stride
    tensors shaped like what the finish node's backward hands the prepare node's
    backward (``dY``, the gradient statistics and, under ``save_stats=False``,
    the replayed statistics).  The finish forward takes them off the handle as
    inputs, so that hand-over runs on autograd edges.  The state never holds
    them: the prepare node's context holds the state, and the edges hold the
    prepare node.
    """

    __slots__ = ("state", "edges")

    def __init__(self, state: _MHCTrainState, edges: tuple[torch.Tensor, ...]) -> None:
        self.state, self.edges = state, edges


# The edge stand-ins per (device, token count, save_stats).  No kernel reads or
# writes an edge, so the edges of every prepare call alias these one-element
# buffers; each call takes its own tensors (``detach``), so the edges of two
# connections are distinct autograd outputs, and allocates no memory.
_EDGES: dict[tuple[torch.device, int, bool], tuple[torch.Tensor, ...]] = {}


def _edges(
    rows: int, device: torch.device, save_stats: bool
) -> tuple[torch.Tensor, ...]:
    """One-element, zero-stride stand-ins of BF16 ``dY`` ``[T, 4, 4096]``, the FP32
    gradient statistics ``[T, 32]`` and, under ``save_stats=False``, the FP32
    statistics ``[T, 64]``: the prepare node's edge outputs."""

    key = (device, rows, save_stats)
    edges = _EDGES.get(key)
    if edges is None:
        shapes = [((rows, STREAMS, HIDDEN), _BF16), ((rows, GSTATS_COLS), _F32)]
        if not save_stats:
            shapes.append(((rows, STATS_COLS), _F32))
        edges = tuple(
            torch.empty_strided(shape, (0,) * len(shape), dtype=dtype, device=device)
            for shape, dtype in shapes
        )
        if len(_EDGES) >= _CACHE_ENTRIES:
            _EDGES.clear()
        _EDGES[key] = edges
    if save_stats:
        return edges[0].detach(), edges[1].detach()
    return edges[0].detach(), edges[1].detach(), edges[2].detach()


def _as_bf16(
    grad: Optional[torch.Tensor], *, shape: tuple[int, ...], device: torch.device
) -> torch.Tensor:
    if grad is None:
        return torch.zeros(shape, dtype=_BF16, device=device)
    return grad.contiguous().to(_BF16)


class CakeMHCTrainPrepareFunction(torch.autograd.Function):
    """Autograd node of ``forward_prepare``: ``forward(X, fn, scale, base, state) -> (U, *edges)``.

    Its backward is ``backward_finish``: it receives ``dU`` from the external
    branch and, as the gradients of the edges (outputs the finish node takes as
    inputs), ``dY`` and the gradient statistics from the finish node's backward;
    it returns the single-cast ``dX`` (mapping, collapse and residual-mixing
    contributions summed in FP32) and the FP32 parameter gradients; the finish
    node returns no gradient for ``X``.
    """

    @staticmethod
    def forward(ctx, X, fn, scale, base, state):
        U, stats = cake_mhc_train_forward_prepare(
            X,
            fn,
            scale,
            base,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            workspace=state.workspace,
        )
        state.fn, state.scale, state.base = fn.detach(), scale.detach(), base.detach()
        state.stats = stats  # read and released by the finish forward
        if state.save_stats:
            ctx.save_for_backward(X, stats)
        else:
            ctx.save_for_backward(X)
        ctx.state = state
        # An output the backward did not reach arrives as ``None``, not as zeros.
        ctx.set_materialize_grads(False)
        return (U,) + _edges(X.shape[0], X.device, state.save_stats)

    @staticmethod
    def backward(ctx, dU, dY, grad_stats, stats=None):
        X, *saved = ctx.saved_tensors
        state = ctx.state
        held = state.take()
        if dY is None:
            # No edge gradient: a finish node rerun by a reentrant checkpoint left
            # them on the state, or the finish node's backward never ran (``Y``
            # did not reach the loss: the residual-mixing path then carries no
            # gradient).
            dY, grad_stats, stats = held
        rows = int(X.shape[0])
        if saved:
            stats = saved[0]
        elif stats is None:
            stats = state.replay(X)
        if dY is None:
            dY = torch.zeros_like(X)
        if grad_stats is None:
            grad_stats = torch.zeros((rows, GSTATS_COLS), dtype=_F32, device=X.device)
        dX, dfn, dscale, dbase = _backward_finish(
            X,
            dY,
            _as_bf16(dU, shape=(rows, HIDDEN), device=X.device),
            None,  # ``B``: validated when given, not read
            state.fn,
            state.scale,
            state.base,
            stats,
            grad_stats,
            rms_eps=state.rms_eps,
            hc_eps=state.hc_eps,
            sinkhorn_iters=state.sinkhorn_iters,
            accumulate=False,
            workspace=state.backward_workspace,
            dX=None,
            dfn=None,
            dscale=None,
            dbase=None,
            path=_PATH_BACKWARD if saved else _PATH_CHECKPOINT,
        )
        return dX, dfn, dscale, dbase, None


class CakeMHCTrainFinishFunction(torch.autograd.Function):
    """Autograd node of ``forward_finish``: ``forward(X, B, state, *edges) -> Y``.

    Its backward is ``backward_prepare``: it returns ``dB`` for the external
    branch and hands ``dY`` and the gradient statistics (under
    ``save_stats=False`` also the statistics it replayed) to the prepare node as
    the gradients of ``edges``, the prepare node's edge outputs, so the autograd
    engine releases them once the prepare node's backward has run.  When a
    checkpoint region that starts after the prepare reruns this forward, the
    statistics the state already released are replayed bitwise from ``X``; a
    reentrant region reruns it without the edges, and its backward leaves them
    on the state for the prepare node instead.
    """

    @staticmethod
    def forward(ctx, X, B, state, *edges):
        stats = state.stats
        if stats is None:
            if state.fn is None:
                raise RuntimeError(
                    "cake_mhc_train_finish needs the state of a preceding cake_mhc_train_prepare"
                )
            # Replay: a checkpoint region that starts after the prepare reruns the
            # finish on a state that already released its statistics; recompute
            # them bitwise from ``X``.
            stats = state.replay(X)
        Y = cake_mhc_train_forward_finish(X, B, stats)
        if state.save_stats:
            ctx.save_for_backward(X, B, stats)
        else:
            ctx.save_for_backward(X, B)
        state.stats = None  # nothing of the forward stays on the state
        ctx.state = state
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, B, *saved = ctx.saved_tensors
        state = ctx.state
        stats = saved[0] if saved else state.replay(X)
        dY = _as_bf16(dY, shape=(X.shape[0], STREAMS, HIDDEN), device=X.device)
        dB, grad_stats = _backward_prepare(
            dY,
            B,
            X,
            stats,
            dB=None,
            grad_stats_out=None,
            path=_PATH_BACKWARD if saved else _PATH_CHECKPOINT,
        )
        handed = (dY, grad_stats) if saved else (dY, grad_stats, stats)
        edges = len(ctx.needs_input_grad) - 3
        if edges and ctx.needs_input_grad[3]:
            return (None, dB, None, *handed)
        if state.recorded:
            # No edge reaches the prepare node (a finish rerun by a reentrant
            # checkpoint): its backward takes them off the state.
            state.dY, state.grad_stats = dY, grad_stats
            if not saved:
                state.stats = stats
        return (None, dB, None) + (None,) * edges


def cake_mhc_train_prepare(
    X: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    sinkhorn_iters: int = 20,
    save_stats: bool = True,
    workspace: Optional[torch.Tensor] = None,
    backward_workspace: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, _MHCTrainHandle]:
    """Autograd-enabled ``forward_prepare``: ``(U, handle)``; pass ``handle`` to :func:`cake_mhc_train_finish`.

    Parameters
    ----------
    X, fn, scale, base : torch.Tensor
        As in :func:`cake_mhc_train_forward_prepare` (``X`` and the parameters may require gradients).
    rms_eps, hc_eps, sinkhorn_iters : float, float, int
        As in :func:`cake_mhc_train_forward_prepare` (the epsilons are launch
        arguments; the iteration count is ``1 .. SINKHORN_ITERS_MAX``, default ``20``).
    save_stats : bool
        ``True`` (default): the per-token statistics are saved for the backward
        through the autograd saved tensors (an activation checkpoint frees and
        recomputes them with the inputs).  ``False``: they are dropped after the
        forward and recomputed from ``X`` in the backward (checkpoint replay);
        the gradients are bitwise the same.
    workspace, backward_workspace : torch.Tensor, optional
        Caller-owned forward / backward workspaces; per-device defaults when omitted.

    Returns
    -------
    ``(U, handle)``: BF16 ``U`` and the handle of this prepare node for one
    :func:`cake_mhc_train_finish`.
    """

    state = _MHCTrainState(
        save_stats=save_stats,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        workspace=workspace,
        backward_workspace=backward_workspace,
    )
    out = CakeMHCTrainPrepareFunction.apply(X, fn, scale, base, state)
    U = out[0]
    # ``U`` requires grad exactly when the prepare node was recorded.
    state.recorded = U.requires_grad
    return U, _MHCTrainHandle(state, out[1:] if state.recorded else ())


def cake_mhc_train_finish(
    X: torch.Tensor, B: torch.Tensor, handle: _MHCTrainHandle
) -> torch.Tensor:
    """Autograd-enabled ``forward_finish``: ``Y`` from ``X``, the branch output ``B`` and the prepare ``handle``.

    Parameters
    ----------
    X : torch.Tensor
        The BF16 ``[T, 4, 4096]`` tensor passed to :func:`cake_mhc_train_prepare`.
    B : torch.Tensor
        BF16 ``[T, 4096]`` output of the external branch applied to ``U`` (may require gradients).
    handle : object
        The handle returned by :func:`cake_mhc_train_prepare`, for one finish.  A
        reentrant checkpoint region (``use_reentrant=True``) holds both calls or
        starts after the prepare; it does not end between them.

    Returns
    -------
    BF16 ``Y`` of shape ``[T, 4, 4096]``.
    """

    if not isinstance(handle, _MHCTrainHandle):
        raise TypeError("handle must be the handle returned by cake_mhc_train_prepare")
    # The forward takes the edges off the handle: the handle then holds nothing of
    # the prepare node, and a checkpoint rerun of this call runs without them.
    edges, handle.edges = handle.edges, ()
    return CakeMHCTrainFinishFunction.apply(X, B, handle.state, *edges)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "TENSOR_ALIGNMENT_BYTES",
    "WORKSPACE_ALIGNMENT_BYTES",
    "CakeMHCTrainFinishFunction",
    "CakeMHCTrainPrepareFunction",
    "cake_mhc_train_backward_finish",
    "cake_mhc_train_backward_prepare",
    "cake_mhc_train_backward_workspace",
    "cake_mhc_train_backward_workspace_bytes",
    "cake_mhc_train_finish",
    "cake_mhc_train_forward_finish",
    "cake_mhc_train_forward_prepare",
    "cake_mhc_train_forward_stats",
    "cake_mhc_train_forward_workspace",
    "cake_mhc_train_forward_workspace_bytes",
    "cake_mhc_train_prepare",
    "cake_mhc_train_reset_workspace",
    "is_cake_mhc_train_supported",
    "route_key",
    "route_modules",
]
