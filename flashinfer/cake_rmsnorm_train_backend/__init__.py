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

Backend of :mod:`flashinfer.cake_rmsnorm_train`: Cake fused BF16 RMSNorm training
forward / backward (SM100 / SM103 / SM107) with a deterministic default and an
opt-in non-deterministic weight gradient, as one self-contained package (host
API bodies, the ``cake_jit`` loader and the generated CUDA sources under
``csrc/``).

Every entry point takes ``deterministic`` (default ``True``).  ``True`` runs the
deterministic package :mod:`flashinfer.cake_rmsnorm_det_train_backend` with the
same arguments, so the default call returns that package's bits.
``deterministic=False`` runs this package's kernels: the forward programs of
the deterministic package (rendered here; the forward has no cross-row
reduction) and, per token-count window, the backward form the registry lists
for it: in the windows that ship it, one kernel whose CTAs add their ``dw``
sums into FP64 workspace sums in the order they finish, so ``dw`` is not
guaranteed bitwise reproducible; in the others, the deterministic package's
backward pair (main kernel and fixed-order ``dw`` reduce) rendered here.  The stable,
``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in :mod:`flashinfer.cake_rmsnorm_train`; import that module.

Vendoring: this package imports and runs ``deterministic=False`` on its own.
``deterministic=True`` (the default) needs the deterministic package as its
sibling: vendor ``cake_rmsnorm_det_train_backend`` next to this package under
the same parent package (it is imported as ``..cake_rmsnorm_det_train_backend``
on the first deterministic call); without it a deterministic call raises
``RuntimeError`` naming the missing package.
"""

from __future__ import annotations

import functools
import math
import textwrap
import warnings
from typing import Optional

import torch

from . import cake_jit as _loader
from . import cake_windows as _windows
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    BACKWARD_ALIGNMENT_BYTES,
    FORWARD_ALIGNMENT_BYTES,
    FP32_WEIGHT_ALIGNMENT_BYTES,
    arch_for_capability,
    route_applies,
    route_key,
    selected_modules,
    supported_hidden_sizes,
    weight_dtypes,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)

# The deterministic entry point each entry point of this package delegates to.
_DET_NAMES = {
    "cake_rmsnorm_train_forward": "cake_rmsnorm_det_train_forward",
    "cake_rmsnorm_train_backward": "cake_rmsnorm_det_train_backward",
    "cake_rmsnorm_train_backward_workspace": "cake_rmsnorm_det_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes": "cake_rmsnorm_det_train_backward_workspace_bytes",
    "cake_rmsnorm": "cake_rmsnorm_det",
}


# What the warn_only fallback of each entry point runs (see ``_fallback``).
_FALLBACK_RUNS = {
    "cake_rmsnorm_train_forward": "this package's forward programs (the forward "
    "has no cross-row reduction; they are deterministic)",
    "cake_rmsnorm_train_backward": "the deterministic package's backward with a "
    "workspace of its own (the workspace argument is not used)",
    "cake_rmsnorm_train_backward_workspace": "the deterministic package's workspace "
    "helper",
    "cake_rmsnorm_train_backward_workspace_bytes": "the deterministic package's "
    "workspace helper",
    "cake_rmsnorm": "this package's forward programs and the deterministic "
    "package's backward",
}

# The modes of a call (``_resolve_deterministic``): the deterministic package,
# this package's kernels, the warn_only fallback of ``deterministic=False``.
_DET, _NONDET, _FALLBACK = "det", "nondet", "fallback"


def _det_package():
    """The deterministic package the ``deterministic=True`` calls run.

    It is imported on first use, so this package imports on its own (a
    consumer may vendor it without the deterministic package); a deterministic
    call then raises ``RuntimeError`` naming the missing package.
    """

    try:
        from .. import cake_rmsnorm_det_train_backend as det
    except ImportError as exc:
        raise RuntimeError(
            "deterministic=True runs the deterministic Cake RMSNorm training "
            "package (flashinfer.cake_rmsnorm_det_train_backend), which cannot be "
            f"imported here ({exc}); install it, or call with deterministic=False "
            "to run this package's kernels"
        ) from exc
    return det


# The deterministic names of the previous ``flashinfer.cake_rmsnorm_train_backend``
# (before this package it resolved every attribute on the deterministic backend).
# Each resolves on first access to the deterministic package's object of that name
# (:func:`__getattr__`; not in ``__all__``), so this package still imports without
# the deterministic package.
_PREVIOUS_DET_NAMES = frozenset(
    (
        "CakeRMSNormDetFunction",
        "cake_rmsnorm_det",
        "cake_rmsnorm_det_train_backward",
        "cake_rmsnorm_det_train_backward_workspace",
        "cake_rmsnorm_det_train_backward_workspace_bytes",
        "cake_rmsnorm_det_train_forward",
        "is_cake_rmsnorm_det_train_supported",
    )
)


def __getattr__(name: str):
    """A deterministic name of the previous package (:data:`_PREVIOUS_DET_NAMES`): the
    deterministic package's object of that name."""
    if name in _PREVIOUS_DET_NAMES:
        try:
            det = _det_package()
        except RuntimeError as exc:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}: {exc}"
            ) from exc
        return getattr(det, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _check_flag(deterministic: bool, api: str) -> None:
    if not isinstance(deterministic, bool):
        raise TypeError(
            f"flashinfer.{api}: deterministic must be a bool, got "
            f"{type(deterministic).__name__}"
        )


def _refusal(api: str) -> str:
    return (
        f"flashinfer.{api}(deterministic=False) sums the weight gradient over the "
        "CTAs in a scheduling-dependent order, but "
        "torch.use_deterministic_algorithms(True) is in effect; call it with "
        f"deterministic=True or use flashinfer.{_DET_NAMES[api]}"
    )


def _resolve_deterministic(deterministic: bool, api: str) -> str:
    """The mode of a call of ``api``: ``_DET``, ``_NONDET`` or ``_FALLBACK``.

    ``deterministic`` must be a ``bool`` (``TypeError`` otherwise).  ``True``
    runs the deterministic package.  ``False`` runs this package's kernels
    unless ``torch.use_deterministic_algorithms(True)`` is in effect: then the
    call raises ``RuntimeError``, or, with ``warn_only=True``, takes the
    fallback (``_fallback``).  A call that asked for determinism never runs the
    non-deterministic kernels.
    """

    _check_flag(deterministic, api)
    if deterministic:
        return _DET
    if not torch.are_deterministic_algorithms_enabled():
        return _NONDET
    if torch.is_deterministic_algorithms_warn_only_enabled():
        return _FALLBACK
    raise RuntimeError(_refusal(api))


def _fallback(api: str, gap: Optional[str] = None) -> None:
    """The ``warn_only`` fallback of ``deterministic=False``: one ``UserWarning``
    naming what runs instead (``_FALLBACK_RUNS``), or, when the deterministic
    package cannot serve the call (``gap``: the feature it lacks), one
    ``RuntimeError`` and no warning: such a call is never run with
    non-deterministic kernels in silence nor refused with an unrelated error."""

    if gap is not None:
        raise RuntimeError(
            f"{_refusal(api)}; with warn_only=True the call falls back to the "
            f"deterministic package, which does not support {gap}: only "
            "deterministic=False serves this call, with "
            "torch.use_deterministic_algorithms off"
        )
    warnings.warn(
        f"{_refusal(api)}; running {_FALLBACK_RUNS[api]}", UserWarning, stacklevel=3
    )


def _det_gap(
    kind: str, x: torch.Tensor, w: torch.Tensor, residual: bool
) -> Optional[str]:
    """What the deterministic package lacks to serve the ``kind`` (``"fwd"`` /
    ``"bwd"``) call of these inputs, or ``None``: an input of more than two
    dimensions, a weight other than BF16, or a width / residual route it does
    not export on the device."""

    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        return None  # the deterministic package reports the argument error
    if x.ndim > 2:
        return f"inputs of more than two dimensions (x of shape {tuple(x.shape)})"
    if w.dtype != torch.bfloat16:
        return f"{w.dtype} weights"
    if x.device.type != "cuda" or x.ndim != 2:
        return None
    capability = _device_capability(x.device)
    hidden = int(x.shape[1])
    if not _det_package().cake_jit.route_applies(
        device_capability=capability, hidden=hidden, kind=kind, residual=residual
    ):
        word = "forward" if kind == "fwd" else "backward"
        return (
            f"the {word} of hidden={hidden} (residual={residual}) on "
            f"{arch_for_capability(capability)}"
        )
    return None


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _capability(device_index: int) -> tuple[int, int]:
    """The compute capability of a CUDA device (fixed for the process)."""

    major, minor = torch.cuda.get_device_capability(device_index)
    return int(major), int(minor)


def _device_capability(device: torch.device) -> tuple[int, int]:
    return _capability(_device_index(device))


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"Cake RMSNorm training requires CUDA tensors, got {device}")
    capability = _device_capability(device)
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            "Cake RMSNorm training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got {capability}"
        )
    return arch


def is_cake_rmsnorm_train_supported(
    device: torch.device, hidden: Optional[int] = None, *, deterministic: bool = True
) -> bool:
    """Whether ``device`` (and optionally ``hidden``) has an exported route.

    ``deterministic=True`` answers for the deterministic package
    (:func:`flashinfer.cake_rmsnorm_det_train.is_cake_rmsnorm_det_train_supported`),
    ``False`` for this package's non-deterministic routes.  The answer does not
    depend on ``torch.use_deterministic_algorithms``.
    """

    _check_flag(deterministic, "is_cake_rmsnorm_train_supported")
    if deterministic:
        return _det_package().is_cake_rmsnorm_det_train_supported(device, hidden)
    device = torch.device(device)
    if device.type != "cuda":
        return False
    arch = arch_for_capability(_device_capability(device))
    if arch is None:
        return False
    if hidden is None:
        return bool(supported_hidden_sizes(arch))
    return int(hidden) in supported_hidden_sizes(arch)


def _validate_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _row_view(
    tensor: torch.Tensor, name: str, *, copy: bool = False
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """``(the [rows, H] view of an input, its leading shape)``.

    A ``[T, H]`` tensor is its own view.  A tensor of more dimensions,
    ``[..., H]``, collapses its leading dimensions into rows when its feature
    dimension is contiguous (``stride(-1) == 1``) and the leading dimensions
    are contiguous with one another in row-major order --
    ``stride(i) == stride(i + 1) * size(i + 1)`` for every pair of leading
    dimensions of size other than one -- so that consecutive rows sit one
    stride apart (a ``[T, heads, H]`` slice of a wider ``[T, heads, W]``
    buffer, strides ``(heads * W, W, 1)``, qualifies).  The view keeps that
    row stride, so the alignment rules of the kernel inputs apply to it
    unchanged.  Any other layout raises ``ValueError`` naming the rule (the
    kernels index rows by one stride and write contiguous outputs), or, with
    ``copy`` (the backward's gradients, which autograd hands over in whatever
    layout the consumer produced), is copied once into a contiguous tensor.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must be a [T, H] tensor (or [..., H] with more leading "
            f"dimensions), got shape {tuple(tensor.shape)}"
        )
    if tensor.ndim == 2:
        return tensor, (int(tensor.shape[0]),)
    lead = tuple(int(size) for size in tensor.shape[:-1])
    hidden = int(tensor.shape[-1])
    rows = math.prod(lead)
    if rows == 0 or hidden == 0:
        return tensor.reshape(rows, hidden), lead
    dims = [i for i in range(tensor.ndim - 1) if tensor.shape[i] != 1]
    collapsible = (tensor.stride(-1) == 1 or hidden == 1) and all(
        tensor.stride(outer) == tensor.stride(inner) * tensor.shape[inner]
        for outer, inner in zip(dims, dims[1:], strict=False)
    )
    if not collapsible and copy:
        return tensor.contiguous().view(rows, hidden), lead
    if not collapsible:
        raise ValueError(
            f"{name} of shape {tuple(tensor.shape)} with strides "
            f"{tuple(tensor.stride())} cannot be viewed as [{rows}, {hidden}] "
            "rows: the feature dimension must be contiguous (stride(-1) == 1) "
            "and the leading dimensions contiguous with one another in "
            "row-major order (stride(i) == stride(i + 1) * size(i + 1)); "
            "reshape or copy the tensor into such a layout"
        )
    pitch = tensor.stride(dims[-1]) if dims else hidden
    view = tensor.as_strided((rows, hidden), (pitch, 1), tensor.storage_offset())
    return view, lead


def _companion_view(
    tensor: torch.Tensor,
    name: str,
    *,
    lead: tuple[int, ...],
    hidden: int,
    copy: bool = False,
) -> torch.Tensor:
    """The ``[rows, H]`` view of an input that takes the first input's shape
    (``residual``; the backward's ``g``, ``x`` companions and ``g_residual``)."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected = (*lead, hidden)
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
        )
    return _row_view(tensor, name, copy=copy)[0]


def _rows_view(
    tensor: torch.Tensor, *, lead: tuple[int, ...], rows: int
) -> torch.Tensor:
    """``rstd`` as ``[rows]``: the leading shape of an N-D call reshaped, anything else as is."""

    if (
        isinstance(tensor, torch.Tensor)
        and len(lead) > 1
        and tuple(tensor.shape) == tuple(lead)
    ):
        return tensor.reshape(rows)
    return tensor


def _materialize(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    """A contiguous copy whose storage satisfies ``alignment`` (fresh allocations are 256-byte aligned)."""

    out = tensor.contiguous()
    if out.data_ptr() % alignment:
        out = out.clone(memory_format=torch.contiguous_format)
    return out


def _kernel_input(
    tensor: torch.Tensor,
    name: str,
    *,
    rows: int,
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
) -> torch.Tensor:
    """Validate a ``[rows, hidden]`` input of any layout and return what the kernel can read.

    The kernels read a row-strided tensor in place when the feature dimension is
    contiguous, the storage base is ``alignment``-byte aligned and the row stride
    in bytes is a multiple of ``alignment`` (``stride(0) == 0`` of an expanded
    gradient included).  Any other layout -- odd column offset, odd row stride,
    transposed storage -- is materialized once as a contiguous copy.  A single row
    never dereferences its row stride and is re-viewed with the canonical stride.
    """

    _validate_tensor(tensor, name, shape=(rows, hidden), device=device, dtype=dtype)
    if rows == 0 or hidden == 0:
        return tensor
    feature_contiguous = tensor.stride(1) == 1 or hidden == 1
    if feature_contiguous and tensor.data_ptr() % alignment == 0:
        if rows == 1:
            return (
                tensor
                if tensor.stride(0) == hidden
                else tensor.as_strided((1, hidden), (hidden, 1))
            )
        row_stride = tensor.stride(0)
        if (row_stride * tensor.element_size()) % alignment == 0 and (
            row_stride >= hidden or row_stride == 0
        ):
            return tensor
    return _materialize(tensor, alignment)


def _contiguous_input(
    tensor: torch.Tensor,
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
) -> torch.Tensor:
    """Validate a small input (``w``, ``rstd``); contiguous and aligned in place, else a copy."""

    _validate_tensor(tensor, name, shape=shape, device=device, dtype=dtype)
    if tensor.numel() == 0 or (
        tensor.is_contiguous() and tensor.data_ptr() % alignment == 0
    ):
        return tensor
    return _materialize(tensor, alignment)


# The weight dtypes the host accepts and their route tokens (``cake_jit.WEIGHT_DTYPES``).
_WEIGHT_DTYPES = {torch.bfloat16: "bf16", torch.float32: "f32"}


def _weight(
    w: torch.Tensor, *, hidden: int, device: torch.device, alignment: int
) -> tuple[torch.Tensor, str]:
    """``(the weight the kernel reads, its route token)``: a BF16 weight in place
    when contiguous and ``alignment``-byte aligned, an FP32 weight in place when
    contiguous and ``FP32_WEIGHT_ALIGNMENT_BYTES`` aligned (the FP32-weight
    programs load it with 32-byte vectors), else a copy; another dtype is refused."""

    if not isinstance(w, torch.Tensor):
        raise TypeError("w must be a torch.Tensor")
    token = _WEIGHT_DTYPES.get(w.dtype)
    if token is None:
        raise ValueError(f"w must be one of {tuple(_WEIGHT_DTYPES)}, got {w.dtype}")
    if token == "f32":
        alignment = FP32_WEIGHT_ALIGNMENT_BYTES
    kernel_w = _contiguous_input(
        w, "w", shape=(hidden,), device=device, dtype=w.dtype, alignment=alignment
    )
    return kernel_w, token


def _route_error(
    kind: str, hidden: int, residual: bool, w: torch.Tensor, arch: str
) -> ValueError:
    """The error of a call no exported route serves: the hidden sizes with routes
    on ``arch`` and the weight dtypes this route ships for."""

    word = "forward" if kind == "fwd" else "backward"
    shipped = weight_dtypes(arch, hidden, kind, residual)
    dtypes = tuple(dtype for dtype, token in _WEIGHT_DTYPES.items() if token in shipped)
    return ValueError(
        f"no exported non-deterministic Cake RMSNorm training {word} for "
        f"hidden={hidden} (residual={residual}, w={w.dtype}) on {arch}; supported "
        f"hidden sizes: {supported_hidden_sizes(arch)}; weight dtypes exported for "
        f"this route: {dtypes if dtypes else 'none'}"
    )


def _output_buffer(
    tensor: Optional[torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    alignment: int,
    lead: Optional[tuple[int, ...]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """``(kernel buffer, the tensor returned to the caller, user buffer to copy back into or None)``.

    A caller-provided output of any layout is honoured: the kernel writes a
    contiguous aligned buffer directly when the caller's tensor is one, otherwise
    into a temporary that is copied back into the caller's view after the launch.
    ``lead`` (an N-D call) is the leading shape the caller's buffer and the
    returned tensor have in place of the ``shape[0]`` rows of the kernel buffer.
    """

    user_shape = shape if lead is None else (*lead, *shape[1:])
    if tensor is None:
        kernel = torch.empty(shape, dtype=dtype, device=device)
        return kernel, (kernel if lead is None else kernel.view(user_shape)), None
    _validate_tensor(tensor, name, shape=user_shape, device=device, dtype=dtype)
    if tensor.numel() == 0 or (
        tensor.is_contiguous() and tensor.data_ptr() % alignment == 0
    ):
        return (tensor if lead is None else tensor.reshape(shape)), tensor, None
    return torch.empty(shape, dtype=dtype, device=device), tensor, tensor


def _copy_back(pairs: list[tuple[torch.Tensor, Optional[torch.Tensor]]]) -> None:
    for kernel_buffer, user_buffer in pairs:
        if user_buffer is not None:
            user_buffer.copy_(kernel_buffer.reshape(user_buffer.shape))


# Entries of each host-path memo (``_backward_plan``, ``_workspace_bytes_of``):
# one per (device, width, residual flag, weight dtype, token count) a process
# calls with; a training run repeats a handful.
_PLAN_CACHE_SIZE = 256


def _registry_token() -> tuple:
    """The registry objects a memoised host plan reads (part of its key): a
    plan computed against a replaced registry table or SM-count function is
    never served for another."""

    return (
        id(_loader.MODULES),
        id(_loader.ROUTES),
        _loader.sm_count,
        _loader.SUMS_BYTES,
        _loader.PAIR_ROWS_OFFSET,
    )


def _backward_owners(arch: str, hidden: int, residual: bool, rows: int) -> list[dict]:
    """The workspace-owning backward records (single kernels and pair main
    kernels) of the width's backward routes on ``arch`` -- both weight dtypes --
    whose window admits a token count of at most ``rows``."""

    owners = []
    for w_dtype in weight_dtypes(arch, hidden, "bwd", residual):
        for name in _loader.route_modules(arch, hidden, "bwd", residual, w_dtype):
            record = _loader.MODULES[name]
            rule = record["select_rule"]
            if record["workspace_rule"] is None:
                continue
            if rule is None or int(rule["above"]) < rows:
                owners.append(record)
    return owners


def _capacity_record(
    owners: list[dict], rows: int, device_index: int
) -> Optional[dict]:
    """The pair main record of ``owners`` with the largest partial capacity for
    ``rows`` tokens (its capacity sizes the partial rows of every pair record of
    the routes), or ``None`` when no window runs a deterministic pair."""

    pairs = [
        record
        for record in owners
        if record["workspace_rule"]["kind"] == "chunk_partials"
    ]
    if not pairs:
        return None
    return max(
        pairs,
        key=lambda record: (
            _loader.partial_capacity(record, rows, device_index=device_index),
            int(record["grid_rule"]["ctas_per_sm"]),
        ),
    )


def _workspace_bytes(
    rows: int, hidden: int, device: torch.device, residual: bool
) -> int:
    """Bytes of this package's backward workspace of width ``hidden`` for up to
    ``rows`` tokens on ``device``: the largest layout of the records the
    family's backward routes (``hidden``, ``residual``, both weight dtypes) run
    for those token counts."""

    arch = _device_arch(device)
    hidden, rows = int(hidden), max(int(rows), 1)
    if hidden <= 0:
        raise ValueError("hidden must be positive")
    return _workspace_bytes_of(
        _device_index(device), arch, hidden, bool(residual), rows, _registry_token()
    )


@functools.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _workspace_bytes_of(
    device_index: int,
    arch: str,
    hidden: int,
    residual: bool,
    rows: int,
    registry: tuple,
) -> int:
    """:func:`_workspace_bytes` for one device, memoised (registry constants and
    the device's SM count only; an error is raised again on every call)."""

    if not route_applies(
        device_capability=_capability(device_index),
        hidden=hidden,
        kind="bwd",
        residual=residual,
    ):
        raise ValueError(
            "no exported non-deterministic Cake RMSNorm training backward for "
            f"hidden={hidden} (residual={residual}) on {arch}; supported hidden "
            f"sizes: {supported_hidden_sizes(arch)}"
        )
    owners = _backward_owners(arch, hidden, residual, rows)
    capacity = _capacity_record(owners, rows, device_index)
    return max(
        int(
            _loader.workspace_layout(
                record,
                rows,
                device_index=device_index,
                capacity_record=capacity,
            )["total_bytes"]
        )
        for record in owners
    )


def cake_rmsnorm_train_backward_workspace_bytes(
    rows: int,
    hidden: int,
    device: torch.device,
    *,
    residual: bool = False,
    deterministic: bool = True,
) -> int:
    """Bytes of the caller-owned backward workspace of width ``hidden`` on ``device`` for up to ``rows`` tokens.

    ``deterministic`` is resolved as the backward resolves it (see
    :func:`cake_rmsnorm_train_backward`), so the buffer fits the kernels the
    backward call with the same flag runs.  Under
    ``torch.use_deterministic_algorithms(True, warn_only=True)``,
    ``deterministic=False`` warns and returns the deterministic package's size;
    the backward call of that fallback does not use a caller's workspace (it
    passes none to the deterministic package, which allocates its own).  The deterministic package's
    workspace holds its chunk-claim counters and one FP32 ``dw`` partial row
    per chunk and grows with ``rows``
    (:func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward_workspace_bytes`).
    The non-deterministic workspace has one layout for every call
    (``cake_jit.workspace_layout``, offsets fixed by the package): the
    deterministic pairs' ``uint32`` counters (``cake_jit.COUNTERS_BYTES``), the
    single kernel's counters (``cake_jit.SINGLE_COUNTERS_BYTES`` at
    ``cake_jit.SINGLE_COUNTERS_OFFSET``), its FP64 column sums
    (``cake_jit.SUMS_BYTES`` at ``cake_jit.SUMS_OFFSET``, the largest
    ``workspace_bytes`` of the package's single kernels) and, when a window up
    to ``rows`` of the width runs the deterministic pair, that pair's FP32
    partial rows at ``cake_jit.PAIR_ROWS_OFFSET``, which grow with ``rows``.
    The size returned is the largest layout of the records the width's
    backward routes (``hidden``, ``residual``, both weight dtypes) run for up
    to ``rows`` tokens.  The buffer must be zero on first use (allocate with
    ``torch.zeros`` or use :func:`cake_rmsnorm_train_backward_workspace`);
    after every launch the kernels leave the counters at rest and the sums at
    zero, so with no host write one buffer serves any sequence of
    ``deterministic=False`` backward calls -- hidden sizes, residual flags,
    weight dtypes, token counts, windows and CUDA Graph replays -- whose
    sizes it covers.  It serves calls of one mode: do not pass it to a
    ``deterministic=True`` call (that package's workspace has another layout).
    Zeroing it again between calls is harmless.  Backward calls that may run
    at the same time (on different streams) need one buffer each.
    """

    api = "cake_rmsnorm_train_backward_workspace_bytes"
    mode = _resolve_deterministic(deterministic, api)
    if mode == _FALLBACK:
        _fallback(api)
    if mode != _NONDET:
        return _det_package().cake_rmsnorm_det_train_backward_workspace_bytes(
            rows, hidden, device, residual=residual
        )
    device = torch.device(device)
    if int(rows) < 0:
        raise ValueError("rows must be non-negative")
    return _workspace_bytes(rows, hidden, device, residual)


def cake_rmsnorm_train_backward_workspace(
    rows: int,
    hidden: int,
    device: torch.device,
    *,
    residual: bool = False,
    deterministic: bool = True,
) -> torch.Tensor:
    """Allocate a zero-initialized backward workspace (see ``..._workspace_bytes``,
    including its ``warn_only`` fallback: a warning and the deterministic
    package's workspace, which that fallback's backward does not use)."""

    api = "cake_rmsnorm_train_backward_workspace"
    mode = _resolve_deterministic(deterministic, api)
    if mode == _FALLBACK:
        _fallback(api)
    if mode != _NONDET:
        return _det_package().cake_rmsnorm_det_train_backward_workspace(
            rows, hidden, device, residual=residual
        )
    nbytes = cake_rmsnorm_train_backward_workspace_bytes(
        rows, hidden, device, residual=residual, deterministic=False
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def _workspace_views(
    workspace: torch.Tensor,
    record: dict,
    *,
    layout: dict[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(the region bound to ``partial``, the counters, the layout)`` of the
    launched workspace owner ``record``: the FP64 column sums of a single
    kernel (its workspace rule's ``sum_bytes``, the kernel's
    ``workspace_bytes``), the FP32 ``[n_chunks, hidden]`` partial rows of a pair
    main kernel (``capacity``: the routes' pair record of the largest
    capacity; ``layout``: its ``cake_jit.workspace_layout``).  The region is
    checked against the kernel's requirement before the launch."""

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
    if not workspace.is_contiguous() or workspace.data_ptr() % 256 != 0:
        raise ValueError("workspace must be contiguous and 256-byte aligned")
    if workspace.numel() < layout["total_bytes"]:
        raise ValueError(
            f"workspace holds {workspace.numel()} bytes; this call needs "
            f"{layout['total_bytes']} "
            "(cake_rmsnorm_train_backward_workspace_bytes(..., deterministic=False))"
        )
    begin = layout["partial_offset"]
    hidden = int(record["hidden"])
    if record["workspace_rule"]["kind"] == "claim_sum_f64":
        partial = workspace[begin : begin + layout["partial_bytes"]].view(torch.float64)
        required = int(record["workspace_rule"]["sum_bytes"]) // 8
    else:
        partial = workspace[begin : begin + 4 * layout["n_chunks"] * hidden].view(
            torch.float32
        )
        partial = partial.view(layout["n_chunks"], hidden)
        required = int(layout["n_chunks"]) * hidden
    if partial.numel() < required:
        raise RuntimeError(
            f"module kernel_{record['kernel']} needs {required} elements of "
            f"{partial.dtype} partials; the workspace region holds {partial.numel()}"
        )
    begin = layout["counters_offset"]
    counters = workspace[begin : begin + 4 * max(layout["counters"], 1)].view(
        torch.uint32
    )
    return partial, counters


@functools.lru_cache(maxsize=_PLAN_CACHE_SIZE)
def _backward_plan(
    device_index: int,
    arch: str,
    hidden: int,
    residual: bool,
    w_dtype: str,
    rows: int,
    registry: tuple,
) -> tuple[tuple[str, ...], dict, dict[str, int]]:
    """``(module names, workspace owner record, its workspace layout)`` of the
    backward call of ``rows`` tokens: the window of ``rows`` runs one workspace
    owner, a single kernel or the main kernel of a deterministic pair followed
    by its dw reduce record.  Memoised per device, width, residual flag, weight
    dtype and token count: it reads only registry constants and the device's SM
    count (no tensor, pointer or stream); an error is raised again on every
    call.  The layout is shared: callers read it only."""

    names = tuple(
        selected_modules(arch, hidden, "bwd", residual, rows=rows, w_dtype=w_dtype)
    )
    owners = [
        _loader.MODULES[name]
        for name in names
        if _loader.MODULES[name]["workspace_rule"] is not None
    ]
    if len(owners) != 1:
        raise RuntimeError(
            f"the exported backward route for hidden={hidden} launches {len(owners)} "
            f"workspace-owning modules for {rows} rows; expected one"
        )
    record = owners[0]
    capacity = _capacity_record(
        _backward_owners(arch, hidden, residual, rows), rows, device_index
    )
    layout = _loader.workspace_layout(
        record, rows, device_index=device_index, capacity_record=capacity
    )
    return names, record, layout


def _forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    rstd: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """The forward of this package's programs (``deterministic=False``)."""

    x, lead = _row_view(x, "x")
    nd = lead if len(lead) > 1 else None
    device = x.device
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    arch = _device_arch(device)
    has_residual = residual is not None
    align = FORWARD_ALIGNMENT_BYTES
    x = _kernel_input(
        x,
        "x",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    w, w_dtype = _weight(w, hidden=hidden, device=device, alignment=align)
    if has_residual:
        residual = _kernel_input(
            _companion_view(residual, "residual", lead=lead, hidden=hidden),
            "residual",
            rows=rows,
            hidden=hidden,
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
        )
    if not route_applies(
        device_capability=_device_capability(device),
        hidden=hidden,
        kind="fwd",
        residual=has_residual,
        w_dtype=w_dtype,
    ):
        raise _route_error("fwd", hidden, has_residual, w, arch)
    out, y_result, out_user = _output_buffer(
        out,
        "out",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
        lead=nd,
    )
    rstd, rstd_result, rstd_user = _output_buffer(
        rstd,
        "rstd",
        shape=(rows,),
        device=device,
        dtype=torch.float32,
        alignment=BACKWARD_ALIGNMENT_BYTES,
        lead=nd,
    )
    residual_out_user = None
    results: tuple[torch.Tensor, ...] = (y_result, rstd_result)
    if has_residual:
        residual_out, h_new_result, residual_out_user = _output_buffer(
            residual_out,
            "residual_out",
            shape=(rows, hidden),
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
            lead=nd,
        )
        results = (y_result, rstd_result, h_new_result)
    copy_backs = [(out, out_user), (rstd, rstd_user), (residual_out, residual_out_user)]
    if rows == 0:
        return results
    # The plain kernel keeps the residual parameters in its signature and never
    # reads them: bind them to ``x`` / ``y`` / 0 exactly like the source launcher.
    values: dict[str, object] = {
        "x": x,
        "w": w,
        "y": out,
        "r": rstd,
        "rows": rows,
        "eps": float(eps),
        "x_stride": int(x.stride(0)),
        "u": residual if has_residual else x,
        "h_new": residual_out if has_residual else out,
        "u_stride": int(residual.stride(0)) if has_residual else 0,
    }
    device_index = _device_index(device)
    for name in selected_modules(
        arch, hidden, "fwd", has_residual, rows=rows, w_dtype=w_dtype
    ):
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return results


def cake_rmsnorm_train_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    *,
    residual: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    rstd: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Fused BF16 RMSNorm forward with FP32 statistics.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input of any layout.  Row-strided views with a contiguous
        feature dimension, 32-byte aligned storage and a row stride that is a
        multiple of 32 bytes are read in place; other layouts (odd column offset,
        odd row stride, transposed storage) are materialized once.  With
        ``deterministic=False`` a tensor of more dimensions, ``[..., H]``, is
        normalized row by row when its feature dimension is contiguous and its
        leading dimensions are contiguous with one another in row-major order
        (``stride(i) == stride(i + 1) * size(i + 1)``; a ``[T, heads, H]`` slice
        of a wider ``[T, heads, W]`` buffer qualifies): the leading dimensions
        collapse into ``prod(...)`` rows, the outputs take the input's leading
        shape (``rstd`` is ``[...]``), ``w`` is shared by every row and the
        backward's ``dw`` sums them all; any other N-D layout raises
        ``ValueError`` naming the rule.  With ``residual`` this is the residual
        stream ``h`` and the normalized value is ``h_new = BF16(h + residual)``.
    w : torch.Tensor
        BF16 ``[H]`` weight, read in place when contiguous and 32-byte aligned,
        else copied once.  With ``deterministic=False`` an FP32 ``[H]`` weight --
        the effective weight ``1 + w`` a zero-centred model computes in FP32 --
        is accepted on the routes that ship the FP32-weight programs
        (``cake_jit.weight_dtypes``), else ``ValueError``.
    eps : float
        Variance epsilon (FP32 inside the kernel).
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization (same
        layout rules as ``x``).  Enables the fused residual-add route.
    out, rstd, residual_out : torch.Tensor, optional
        Preallocated outputs: BF16 ``y`` of ``x``'s shape, FP32 reciprocal RMS of
        its leading shape (``[T]``) and (residual only) BF16 ``h_new`` of ``x``'s
        shape.  Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.
    deterministic : bool
        ``True`` (default) runs the deterministic package's forward
        (:func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_forward`)
        with these arguments.  ``False`` runs this package's forward programs, the
        deterministic package's programs rendered here (the forward has no
        cross-row reduction, so both are deterministic); it is refused while
        ``torch.use_deterministic_algorithms(True)`` is in effect (with
        ``warn_only=True``: a warning, then this package's forward programs,
        which serve the inputs the deterministic package does not).  A value
        other than a ``bool`` raises ``TypeError``.

    Returns
    -------
    ``(y, rstd)`` or ``(y, rstd, h_new)`` when ``residual`` is given.
    ``rstd`` is the FP32 per-row ``1 / sqrt(mean(h_new^2) + eps)`` the backward
    needs; no other ``[T, H]`` temporary is produced.
    """

    mode = _resolve_deterministic(deterministic, "cake_rmsnorm_train_forward")
    if mode == _FALLBACK:
        _fallback("cake_rmsnorm_train_forward")
    if mode == _DET:
        return _det_package().cake_rmsnorm_det_train_forward(
            x, w, eps, residual=residual, out=out, rstd=rstd, residual_out=residual_out
        )
    return _forward(
        x, w, eps, residual=residual, out=out, rstd=rstd, residual_out=residual_out
    )


def _backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    g_residual: Optional[torch.Tensor],
    workspace: Optional[torch.Tensor],
    dx: Optional[torch.Tensor],
    dw: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward of this package's kernel (``deterministic=False``)."""

    x, lead = _row_view(x, "x")
    nd = lead if len(lead) > 1 else None
    device = x.device
    rows, hidden = int(x.shape[0]), int(x.shape[1])
    arch = _device_arch(device)
    has_residual = g_residual is not None
    align = BACKWARD_ALIGNMENT_BYTES
    x = _kernel_input(
        x,
        "x",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    w, w_dtype = _weight(w, hidden=hidden, device=device, alignment=align)
    rstd = _contiguous_input(
        _rows_view(rstd, lead=lead, rows=rows),
        "rstd",
        shape=(rows,),
        device=device,
        dtype=torch.float32,
        alignment=4,
    )
    g = _kernel_input(
        _companion_view(g, "g", lead=lead, hidden=hidden, copy=True),
        "g",
        rows=rows,
        hidden=hidden,
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
    )
    if has_residual:
        g_residual = _kernel_input(
            _companion_view(
                g_residual, "g_residual", lead=lead, hidden=hidden, copy=True
            ),
            "g_residual",
            rows=rows,
            hidden=hidden,
            device=device,
            dtype=torch.bfloat16,
            alignment=align,
        )
    if not route_applies(
        device_capability=_device_capability(device),
        hidden=hidden,
        kind="bwd",
        residual=has_residual,
        w_dtype=w_dtype,
    ):
        raise _route_error("bwd", hidden, has_residual, w, arch)
    dx, dx_result, dx_user = _output_buffer(
        dx,
        "dx",
        shape=(rows, hidden),
        device=device,
        dtype=torch.bfloat16,
        alignment=align,
        lead=nd,
    )
    dw, dw_result, dw_user = _output_buffer(
        dw, "dw", shape=(hidden,), device=device, dtype=torch.float32, alignment=align
    )
    copy_backs = [(dx, dx_user), (dw, dw_user)]
    if rows == 0:
        dw_result.zero_()
        return dx_result, dw_result
    device_index = _device_index(device)
    names, record, layout = _backward_plan(
        device_index, arch, hidden, has_residual, w_dtype, rows, _registry_token()
    )
    if workspace is None:
        workspace = torch.zeros(
            _workspace_bytes(rows, hidden, device, has_residual),
            dtype=torch.uint8,
            device=device,
        )
    partial, counters = _workspace_views(
        workspace, record, layout=layout, device=device
    )
    # The plain kernels keep the residual-gradient parameters in their signature
    # and never read them: bind them to ``g`` exactly like the source launcher.
    # The single kernel's last CTA, or the pair's dw reduce, stores all of ``dw``,
    # so a caller's ``dw`` buffer may hold anything.
    g_h = g_residual if has_residual else g
    values: dict[str, object] = {
        "g": g,
        "x": x,
        "w": w,
        "r": rstd,
        "dx": dx,
        "dw": dw,
        "partial": partial,
        "counters": counters,
        "g_h": g_h,
        "rows": rows,
        "g_stride": int(g.stride(0)),
        "x_stride": int(x.stride(0)),
        "g_h_stride": int(g_h.stride(0)),
        "n_chunks": int(layout["n_chunks"]),
        "slots": int(layout["slots"]),
    }
    for name in names:
        _loader.run_module(name, values, rows=rows, device_index=device_index)
    _copy_back(copy_backs)
    return dx_result, dw_result


def cake_rmsnorm_train_backward(
    g: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    rstd: torch.Tensor,
    *,
    deterministic: bool = True,
    g_residual: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dw: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fused BF16 RMSNorm backward: ``dx`` and the FP32 weight gradient ``dw``.

    Parameters
    ----------
    g : torch.Tensor
        BF16 ``[T, H]`` gradient of ``y`` of any layout.  Views with a contiguous
        feature dimension, 16-byte aligned storage and a row stride that is a
        multiple of 16 bytes (expanded or sliced autograd gradients included) are
        read in place; other layouts are materialized once.  With
        ``deterministic=False`` an N-D gradient, ``[..., H]``, takes ``x``'s
        shape and collapses its leading dimensions into rows under the forward's
        rule.
    x : torch.Tensor
        The BF16 tensor that was normalized (``x`` of the plain forward,
        ``h_new`` of the residual forward); same layout rules as ``g``.
    w : torch.Tensor
        The forward's weight: BF16 ``[H]`` (read in place when contiguous and
        16-byte aligned), or, with ``deterministic=False`` on the FP32-weight
        routes, FP32 ``[H]`` (in place when contiguous and 32-byte aligned);
        any other layout is copied once.
    rstd : torch.Tensor
        FP32 ``[T]`` reciprocal RMS returned by the forward (of the leading
        shape ``[...]`` for an N-D call, or flat ``[prod(...)]``).
    deterministic : bool
        ``True`` (default) runs the deterministic package's backward
        (:func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det_train_backward`):
        ``dw`` through a fixed row-chunk partition and a fixed-order reduction,
        bitwise reproducible for a given token count on a given architecture.
        ``False`` runs, per window of ``T``, the form the notes below list
        for it: in the windows that ship it, one kernel that writes ``dx`` and
        adds the ``dw`` sums of its CTAs into FP64 sums in the workspace, which
        the CTA that finishes last rounds into ``dw`` (``dx`` is bitwise
        reproducible and within one BF16 ulp of max(|dx|, rms(dx)) of the
        deterministic kernel's ``dx``, not always bitwise equal to it; the
        order of the FP64 additions
        depends on scheduling, so ``dw`` is not guaranteed bitwise
        reproducible); in the others, the deterministic package's backward pair
        rendered into this package.
        ``False`` is refused with ``RuntimeError`` while
        ``torch.use_deterministic_algorithms(True)`` is in effect; with
        ``warn_only=True`` it warns and runs the deterministic backward with a
        workspace of its own (``workspace`` is not used), or, when the
        deterministic package cannot serve the call (N-D inputs, an FP32
        weight, a width or residual route it does not export), raises
        ``RuntimeError`` naming the missing feature.  A value other than a
        ``bool`` raises ``TypeError``.
    g_residual : torch.Tensor, optional
        BF16 gradient of ``g``'s shape flowing directly through ``h_new``
        (residual forward), any layout.  It is added once so that the returned
        ``dx`` is the gradient of both residual-add inputs.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_rmsnorm_train_backward_workspace` called with the same
        ``deterministic`` flag, at least as large as that helper returns for
        this call (with ``deterministic=False`` one buffer serves calls of any
        hidden size, residual flag and weight dtype it is large enough for; see
        :func:`cake_rmsnorm_train_backward_workspace_bytes`).  The call issues
        no memset, copy or synchronization on it.  Under the ``warn_only``
        fallback of ``deterministic=False`` it is not used: the deterministic
        backward runs with a workspace of its own.  When omitted, the call
        allocates and zero-fills one: a caching-allocator call, one fill kernel
        (about 3 us on B200) and about 35 us of host time per call.  One buffer
        per backward that may run concurrently (on another stream).
    dx, dw : torch.Tensor, optional
        Preallocated outputs: BF16 ``dx`` of ``x``'s shape and FP32 ``[H]``.
        Contiguous aligned buffers are written directly; any other layout
        receives a copy after the launch.  Their previous contents are ignored.

    Returns
    -------
    ``(dx, dw)`` with ``dw`` in FP32; the caller decides whether to cast it.
    ``T == 0`` launches nothing and returns an empty ``dx`` and a zero ``dw``.
    """

    mode = _resolve_deterministic(deterministic, "cake_rmsnorm_train_backward")
    if mode == _FALLBACK:
        _fallback(
            "cake_rmsnorm_train_backward",
            _det_gap("bwd", x, w, g_residual is not None),
        )
        # one workspace serves calls of one mode: the caller's buffer is laid
        # out for this package's kernels, the deterministic package gets none
        workspace = None
    if mode != _NONDET:
        return _det_package().cake_rmsnorm_det_train_backward(
            g,
            x,
            w,
            rstd,
            deterministic=True,
            g_residual=g_residual,
            workspace=workspace,
            dx=dx,
            dw=dw,
        )
    return _backward(
        g,
        x,
        w,
        rstd,
        g_residual=g_residual,
        workspace=workspace,
        dx=dx,
        dw=dw,
    )


class CakeRMSNormFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x_or_h_new, w, rstd)`` for the backward.

    ``forward(x, w, eps, residual=None, deterministic=True)`` returns ``y`` or
    ``(y, h_new)``.  ``deterministic`` selects the package of both directions
    (see :func:`cake_rmsnorm_train_backward`); the forward resolves it once
    (refusing ``False`` under ``torch.use_deterministic_algorithms(True)``) and
    the backward resolves the same flag again when it runs.  In residual mode
    the gradient of ``x`` and of ``residual`` is the same tensor
    (``dx + g_h_new``).  The weight gradient is accumulated in FP32 by the
    kernel and returned in ``w.dtype``.
    """

    @staticmethod
    def forward(ctx, x, w, eps, residual=None, deterministic=True):
        mode = _resolve_deterministic(deterministic, "cake_rmsnorm")
        if mode == _FALLBACK:
            # the backward of this call runs the deterministic package: refuse
            # now when it cannot serve it, before any work
            _fallback("cake_rmsnorm", _det_gap("bwd", x, w, residual is not None))
        if mode == _DET:
            outputs = _det_package().cake_rmsnorm_det_train_forward(
                x, w, eps, residual=residual
            )
        else:
            outputs = _forward(
                x, w, eps, residual=residual, out=None, rstd=None, residual_out=None
            )
        y, rstd = outputs[0], outputs[1]
        normalized = outputs[2] if residual is not None else x
        ctx.save_for_backward(normalized, w, rstd)
        ctx.deterministic = deterministic
        ctx.has_residual = residual is not None
        if residual is not None:
            return y, outputs[2]
        return y

    @staticmethod
    def backward(ctx, grad_y, *grad_rest):
        normalized, w, rstd = ctx.saved_tensors
        grad_h = None
        if ctx.has_residual and grad_rest and grad_rest[0] is not None:
            grad_h = grad_rest[0].to(torch.bfloat16)
        if grad_y is None:
            grad_y = torch.zeros_like(normalized)
        dx, dw = cake_rmsnorm_train_backward(
            grad_y.to(torch.bfloat16),
            normalized,
            w,
            rstd,
            deterministic=ctx.deterministic,
            g_residual=grad_h,
        )
        dw = dw if w.dtype == torch.float32 else dw.to(w.dtype)
        return dx, dw, None, (dx if ctx.has_residual else None), None


def cake_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
    deterministic: bool = True,
):
    """Autograd-enabled fused BF16 RMSNorm (optionally with a fused residual add).

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[T, H]`` input on a supported CUDA device (with
        ``deterministic=False`` also ``[..., H]`` under the layout rule of
        :func:`cake_rmsnorm_train_forward`).
    w : torch.Tensor
        BF16 ``[H]`` RMSNorm weight (with ``deterministic=False`` also FP32 on
        the routes that ship the FP32-weight programs); the weight gradient is
        returned in ``w.dtype``.
    eps : float
        Variance epsilon evaluated in FP32 by the kernel.
    residual : torch.Tensor, optional
        BF16 update of ``x``'s shape added to ``x`` before normalization.  When
        provided, the normalized input ``h_new`` is returned with the output.
    deterministic : bool
        ``True`` (default): the deterministic package in both directions, the
        bits of :func:`flashinfer.cake_rmsnorm_det_train.cake_rmsnorm_det`.
        ``False``: this package's kernels, whose weight gradient is summed over
        the CTAs in a scheduling-dependent order and is not guaranteed bitwise
        reproducible; refused while
        ``torch.use_deterministic_algorithms(True)`` is in effect (with
        ``warn_only=True``: a warning, then this package's forward programs and
        the deterministic package's backward, or one ``RuntimeError`` when that
        backward cannot serve the call).

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        BF16 ``y`` of ``x``'s shape, or ``(y, h_new)`` when ``residual`` is
        given.  Backward launches the kernels of
        :func:`cake_rmsnorm_train_backward` under the same ``deterministic``.
    """

    return CakeRMSNormFunction.apply(x, w, eps, residual, deterministic)


def _with_program_selection(doc: Optional[str], kind: str) -> str:
    """``doc`` with the "Program selection by token count" notes of this package's routes of ``kind`` appended.

    The notes are the exporter's rendering of the registry (``cake_windows``):
    the row-count windows of every route of the ``deterministic=False``
    programs, so the documented thresholds are the registry's.
    ``cake_jit.program_windows`` returns the windows.
    """

    section = textwrap.indent(_windows.SECTIONS[kind].strip("\n"), "    ")
    return f"{(doc or '').rstrip()}\n\n    Notes\n    -----\n{section}\n    "


cake_rmsnorm_train_forward.__doc__ = _with_program_selection(
    cake_rmsnorm_train_forward.__doc__, "fwd"
)
cake_rmsnorm_train_backward.__doc__ = _with_program_selection(
    cake_rmsnorm_train_backward.__doc__, "bwd"
)


__all__ = [
    "ARCHES",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeRMSNormFunction",
    "cake_rmsnorm",
    "cake_rmsnorm_train_backward",
    "cake_rmsnorm_train_backward_workspace",
    "cake_rmsnorm_train_backward_workspace_bytes",
    "cake_rmsnorm_train_forward",
    "is_cake_rmsnorm_train_supported",
    "route_key",
]
