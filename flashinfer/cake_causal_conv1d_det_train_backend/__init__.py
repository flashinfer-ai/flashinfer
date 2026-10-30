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

Backend of :mod:`flashinfer.cake_causal_conv1d_train`: the Cake packed width-4
depthwise causal convolution + SiLU training forward / backward (SM100 / SM103 /
SM107) as one self-contained package (host API bodies, the ``cake_jit`` registry,
the pinned ``cake_build`` loader and the generated CUDA sources under ``csrc/``).
The stable, ``@flashinfer_api``-decorated entry points and the full user-facing
documentation live in :mod:`flashinfer.cake_causal_conv1d_train`; import that
module.

Host design (the knobs are keywords; no environment variable):

- Taps ``weight[C, 4]`` are FP32 or BF16; an optional per-channel ``bias[C]``
  is FP32 or BF16, independently of the taps.  The kernels widen BF16 to FP32
  on load and compute in FP32; the bias is added LAST, after the four taps
  (FLA's order).  ``dweight`` / ``dbias`` default to the parameter's dtype and
  may be preallocated as FP32 or BF16.
- ``silu_grad_preactivation``: ``"bf16"`` (default, the shipped policy: SiLU'
  at the BF16-rounded preactivation, the boundary FLA's backward uses) or
  ``"fp32"`` (SiLU' at the unrounded FP32 preactivation and an FP64 final
  reduction of ``dweight``: a weight gradient within ~2e-7 rel-L2 of exact).
- A call's arguments select its route (``cake_jit.forward_route`` /
  ``backward_route``): the default routes run the shipped programs unchanged;
  the other routes' programs read the ``flags`` bit mask (``cake_jit.FLAG_*``,
  derived here from the actual tensors: taps, the ``dweight`` out tensor, the
  bias, the ``dbias`` out tensor) for the BF16 / FP32 load and store forms.
  An architecture that lacks a route refuses the call with a ``ValueError``
  naming the route.
- The backward workspace is sized for every backward program of the device's
  architecture (the bias programs' partials carry a fifth, dbias, slab), so one
  buffer serves every route, token count and CUDA Graph.
- The forward (the dynamic token-chunk programs) takes its token chunks from
  int32 counters, two per channel tile, in a forward workspace that is zero
  before its first launch and that every launch leaves zero: a caller-owned
  buffer (``workspace=``, from :func:`cake_causal_conv1d_train_forward_workspace`)
  or, when omitted, a zeroed buffer this module keeps per (device, current
  stream), grown as needed and never zeroed again; inside a CUDA Graph capture
  an omitted workspace is a zeroed buffer of the call's own (its zeroing is
  captured with the launch), so a graph never shares counters with eager
  calls or with another graph.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from . import cake_jit as _loader
from .cake_jit import (
    ARCH_BY_CAPABILITY,
    ARCHES,
    CHANNEL_MULTIPLE,
    FLAG_B_BF16,
    FLAG_DB_BF16,
    FLAG_DW_BF16,
    FLAG_W_BF16,
    INT32_MAX,
    REQUIRED_KINDS,
    WIDTH,
    arch_for_capability,
    backward_route,
    counter_words,
    forward_route,
    forward_workspace_words,
    launch_grid,
    partial_bytes,
    partial_rows,
    plan,
    route_applies,
    route_modules,
    run_module,
    run_modules,
    selected_modules,
    workspace_layout,
    workspace_owners,
)

SUPPORTED_COMPUTE_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    sorted(ARCH_BY_CAPABILITY)
)
# The dtypes of the taps, the bias and their gradients (FP32 or BF16, any combination).
PARAMETER_DTYPES = (torch.float32, torch.bfloat16)
# The two SiLU'-evaluation policies of the weight gradient (``silu_grad_preactivation``).
SILU_GRAD_PREACTIVATIONS = ("bf16", "fp32")
# The forward workspaces of the calls that pass none: (device index, CUDA stream
# handle) -> a zeroed int32 buffer, grown as needed.  The launches of one stream
# run in order and each leaves its counters zero, so the stream's buffer serves
# every later forward call on it.
_FORWARD_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _arch_of_index(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device index, queried once per process."""

    return arch_for_capability(tuple(torch.cuda.get_device_capability(device_index)))


def _device_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(
            f"Cake causal conv1d training requires CUDA tensors, got {device}"
        )
    arch = _arch_of_index(_device_index(device))
    if arch is None:
        raise ValueError(
            "Cake causal conv1d training supports compute capability "
            f"{sorted(ARCH_BY_CAPABILITY)} only, got "
            f"{tuple(torch.cuda.get_device_capability(device))}"
        )
    return arch


def is_cake_causal_conv1d_train_supported(device: torch.device) -> bool:
    """Whether ``device`` has the exported routes every architecture ships (``fwd`` and ``bwd``).

    The optional routes (bias, BF16 taps / gradients, FP32 SiLU') are checked
    per call: a call that needs one the architecture lacks raises a
    ``ValueError`` naming it.
    """

    device = torch.device(device)
    if device.type != "cuda":
        return False
    capability = tuple(torch.cuda.get_device_capability(device))
    return all(
        route_applies(device_capability=capability, kind=kind)
        for kind in REQUIRED_KINDS
    )


def _require_tensor(tensor: object, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    return tensor


def _check_placement(
    tensor: torch.Tensor,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype | tuple[torch.dtype, ...],
) -> None:
    """On ``device``, of one of the ``dtype``\\ s, contiguous (``ValueError`` otherwise)."""

    dtypes = (dtype,) if isinstance(dtype, torch.dtype) else tuple(dtype)
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.dtype not in dtypes:
        wanted = " or ".join(str(d) for d in dtypes)
        raise ValueError(f"{name} must be {wanted}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _tokens(
    tensor: object,
    name: str,
    *,
    device: torch.device,
    shape: Optional[tuple[int, ...]] = None,
) -> tuple[int, int, torch.Tensor]:
    """Validate a BF16 ``[S, C]`` or ``[1, S, C]`` token tensor; return ``(S, C, [S, C] view)``.

    ``shape`` pins the exact shape another token tensor of the call has (``dy``
    against ``x``).  Both kernels read the packed rows as 16-byte vectors of eight
    channels, so ``C`` is a multiple of eight; ``S`` and ``C`` are 32-bit kernel
    scalars (``S x C`` itself may exceed 2^31: the kernels address with 64-bit
    offsets).
    """

    t = _require_tensor(tensor, name)
    if t.ndim == 3 and t.shape[0] == 1:
        S, C = int(t.shape[1]), int(t.shape[2])
    elif t.ndim == 2:
        S, C = int(t.shape[0]), int(t.shape[1])
    else:
        raise ValueError(
            f"{name} must be a [S, C] or [1, S, C] tensor, got shape {tuple(t.shape)}"
        )
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(t.shape)}")
    _check_placement(t, name, device=device, dtype=torch.bfloat16)
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"{name} must have a channel count C that is a positive multiple of "
            f"{CHANNEL_MULTIPLE}, got C={C}"
        )
    if S > INT32_MAX or C > INT32_MAX:
        raise ValueError(
            f"{name}: S={S} and C={C} must each fit a 32-bit kernel scalar"
        )
    return S, C, t.view(S, C)


def _weight(
    weight: object,
    C: int,
    *,
    device: torch.device,
    name: str = "weight",
    dtypes: tuple[torch.dtype, ...] = (torch.float32,),
) -> torch.Tensor:
    """Validate the ``[C, 4]`` taps of one of ``dtypes`` (the training API passes FP32 and BF16; the KCP path
    keeps the default, FP32 only)."""

    w = _require_tensor(weight, name)
    if w.ndim != 2 or int(w.shape[1]) != WIDTH:
        raise ValueError(
            f"{name} must be a [C, {WIDTH}] tensor of a width-{WIDTH} convolution "
            f"(tap {WIDTH - 1} multiplies the current token), got shape {tuple(w.shape)}"
        )
    if int(w.shape[0]) != C:
        raise ValueError(
            f"{name} must have one row per channel ([{C}, {WIDTH}]), got shape {tuple(w.shape)}"
        )
    _check_placement(w, name, device=device, dtype=dtypes)
    return w


def _bias(bias: object, C: int, *, device: torch.device) -> torch.Tensor:
    """Validate the per-channel ``[C]`` bias (FP32 or BF16, independently of the taps)."""

    b = _require_tensor(bias, "bias")
    if b.ndim != 1 or int(b.shape[0]) != C:
        raise ValueError(
            f"bias must be a [C] tensor with one value per channel ([{C}]), got shape {tuple(b.shape)}"
        )
    _check_placement(b, "bias", device=device, dtype=PARAMETER_DTYPES)
    return b


def _cu_seqlens(
    cu_seqlens: object, *, device: torch.device
) -> tuple[torch.Tensor, int]:
    """Validate the ``int32 [num_docs + 1]`` device prefix sums; return ``(cu_seqlens, num_docs)``.

    The host reads the tensor's metadata only: no element is copied to the host
    and no stream is synchronized, so the call is CUDA Graph capturable.
    """

    cu = _require_tensor(cu_seqlens, "cu_seqlens")
    if cu.ndim != 1:
        raise ValueError(
            f"cu_seqlens must be a 1-D [num_docs + 1] tensor, got shape {tuple(cu.shape)}"
        )
    _check_placement(cu, "cu_seqlens", device=device, dtype=torch.int32)
    num_docs = int(cu.numel()) - 1
    if num_docs < 1:
        raise ValueError(
            "cu_seqlens must hold at least two prefix sums (one document), got "
            f"{int(cu.numel())} elements"
        )
    return cu, num_docs


def _output(
    tensor: Optional[object],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    dtypes: Optional[tuple[torch.dtype, ...]] = None,
) -> torch.Tensor:
    """A caller-provided output (exact shape, contiguous, of ``dtype`` or of one of ``dtypes`` when given) or a
    fresh ``dtype`` one."""

    if tensor is None:
        return torch.empty(shape, dtype=dtype, device=device)
    out = _require_tensor(tensor, name)
    if tuple(out.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(out.shape)}"
        )
    _check_placement(out, name, device=device, dtype=dtypes or dtype)
    return out


def _silu_grad_preactivation(value: object) -> str:
    if value not in SILU_GRAD_PREACTIVATIONS:
        raise ValueError(
            f"silu_grad_preactivation must be one of {SILU_GRAD_PREACTIVATIONS}, got {value!r}"
        )
    return str(value)


def _scalars(scalars: dict[str, int]) -> dict[str, int]:
    """The launch scalars :func:`cake_jit.plan` produced (``flags`` is set beside them by the caller)."""

    return {key: int(scalars[key]) for key in _loader.PLAN_SCALARS}


def _flags(
    w: torch.Tensor,
    dw: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    db: Optional[torch.Tensor] = None,
) -> int:
    """The ``flags`` bit mask of a ``dtype_flags`` program from the ACTUAL tensors of the call."""

    flags = 0
    for tensor, bit in (
        (w, FLAG_W_BF16),
        (dw, FLAG_DW_BF16),
        (b, FLAG_B_BF16),
        (db, FLAG_DB_BF16),
    ):
        if tensor is not None and tensor.dtype == torch.bfloat16:
            flags |= bit
    return flags


def _route(arch: str, kind: str, *, needs: str, C: int, S: int) -> tuple[str, ...]:
    """The programs ``kind`` launches on ``arch`` for the call (``ValueError`` naming the route when the
    architecture's export lacks it)."""

    if kind not in _loader.ROUTES.get(arch, {}):
        raise ValueError(
            f"{needs} needs the {kind!r} route, which this package does not export for "
            f"{arch}; its routes: {sorted(_loader.ROUTES.get(arch, {}))}"
        )
    return selected_modules(arch, kind, C=C, S=S)


def _launch(
    record: dict, S: int, C: int, num_docs: int, *, device_index: int
) -> tuple[dict[str, int], tuple[int, int, int]]:
    """The kernel scalars and the grid of one launch of ``record`` (the host plan of ``cake_jit``).

    CTA ``b`` owns channel tile ``b % n_ch_tiles``, so every channel tile needs
    a CTA of the launched program's grid on the device: a ``C`` beyond ``grid
    x threads x vec`` channels is refused rather than left partly unwritten.
    The caller launches this grid (``run_module`` / ``run_modules`` ``grid``),
    so a call plans it once.
    """

    scalars = plan(
        S,
        C,
        num_docs,
        geometry=record["grid_rule"]["geometry"],
        role=record["role"],
    )
    grid = launch_grid(record, device_index=device_index, C=C, S=S)
    if scalars["n_ch_tiles"] > grid[0]:
        raise ValueError(
            f"C={C} spans {scalars['n_ch_tiles']} channel tiles but the launch grid on this "
            f"device is {grid[0]} CTAs; every channel tile needs a CTA"
        )
    return scalars, grid


def cake_causal_conv1d_train_backward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Bytes of the caller-owned backward workspace for ``C`` channels on ``device``.

    The workspace holds the FP32 ``[rows, C, 4]`` ``dweight`` partials (and, for
    the bias routes, a ``[rows, C]`` dbias slab after them) of the backward's
    main program, one row per ``n_ch_tiles`` CTAs of a program's launch grid
    (``rows = ceil(grid / n_ch_tiles)``, ``grid = grid_mult x ctas_per_sm x SM
    count``: the bytes grow with the program's over-decomposition factor
    ``grid_mult``; a call whose grid drops CTAs to whole channel-tile ranks,
    ``cake_jit.main_grid``, writes fewer rows), sized by the largest such
    buffer over every backward program of every backward route of the
    architecture: the call's arguments select the route and the channel count
    the program, every program writes the leading rows of its own grid inside
    this buffer, and its reduce
    program sums exactly those.  So one buffer serves every route (the bias
    routes' fifth slab makes it 25 % larger than the dweight partials alone,
    for every caller) and, as it depends on ``C`` and the device only, every
    token count; the main program fully writes the rows the reduce program
    sums, so the buffer needs no zeroing between launches or CUDA Graph
    replays.
    """

    device = torch.device(device)
    S, C = int(S), int(C)
    if S < 0:
        raise ValueError("S must be non-negative")
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"C must be a positive multiple of {CHANNEL_MULTIPLE}, got {C}"
        )
    arch = _device_arch(device)
    device_index = _device_index(device)
    return max(
        int(workspace_layout(owner, C, device_index=device_index)["total_bytes"])
        for owner in workspace_owners(arch)
    )


def cake_causal_conv1d_train_backward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a backward workspace (see ``..._workspace_bytes``; no zeroing is needed)."""

    nbytes = cake_causal_conv1d_train_backward_workspace_bytes(S, C, device)
    return torch.empty(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_causal_conv1d_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device
) -> int:
    """Bytes of the caller-owned forward workspace for ``C`` channels on ``device``.

    The forward's programs take their token chunks from int32 counters, two
    per channel tile (the next-chunk counter and the done-CTA counter of the
    tile), sized by the most over every forward program of every forward route
    of the architecture (``cake_jit.forward_workspace_words``), so one buffer
    serves every route; the size depends on ``C`` only (``S`` is checked and
    otherwise unused, as for the backward helper).  The buffer must be zero
    before its first launch; every launch leaves it zero, so one buffer serves
    every later forward call of one stream, CUDA Graph replays included,
    without zeroing.  Two launches that may run at once (on two streams, or
    two graphs replayed concurrently) need two buffers.
    """

    device = torch.device(device)
    S, C = int(S), int(C)
    if S < 0:
        raise ValueError("S must be non-negative")
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"C must be a positive multiple of {CHANNEL_MULTIPLE}, got {C}"
        )
    return 4 * forward_workspace_words(_device_arch(device), C)


def cake_causal_conv1d_train_forward_workspace(
    S: int, C: int, device: torch.device
) -> torch.Tensor:
    """Allocate a zeroed int32 forward workspace (see ``..._workspace_bytes``); keep it for every later forward
    call of one stream."""

    nbytes = cake_causal_conv1d_train_forward_workspace_bytes(S, C, device)
    return torch.zeros(nbytes // 4, dtype=torch.int32, device=torch.device(device))


def _forward_work(
    workspace: object, words: int, *, device: torch.device, device_index: int
) -> Optional[torch.Tensor]:
    """The ``work`` counters of one forward launch needing ``words`` int32 words.

    A caller-owned ``workspace`` (1-D int32, contiguous, on ``device``, at least
    ``words`` words, zero before its first launch) is used as is.  Without one:
    inside a CUDA Graph capture a fresh zeroed buffer (its zeroing is captured
    with the launch, so the graph shares no counters with eager calls or other
    graphs); else the zeroed buffer this module keeps for the device and the
    current stream, replaced by a larger zeroed one when it is too small (the
    stream-ordered allocator frees the old one after the launches queued on
    the stream).  ``None`` when the program takes no counters.
    """

    if workspace is not None:
        ws = _require_tensor(workspace, "workspace")
        if ws.ndim != 1:
            raise ValueError(
                f"workspace must be a 1-D int32 CUDA tensor, got shape {tuple(ws.shape)}"
            )
        _check_placement(ws, "workspace", device=device, dtype=torch.int32)
        if ws.numel() < words:
            raise ValueError(
                f"workspace holds {ws.numel()} int32 words; this call needs at least "
                f"{words}; allocate cake_causal_conv1d_train_forward_workspace(S, C, device)"
            )
        return ws
    if words == 0:
        return None
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(words, dtype=torch.int32, device=device)
    key = (device_index, int(torch.cuda.current_stream(device_index).cuda_stream))
    held = _FORWARD_WORKSPACES.get(key)
    if held is None or held.numel() < words:
        held = torch.zeros(words, dtype=torch.int32, device=device)
        _FORWARD_WORKSPACES[key] = held
    return held


def _partial_view(
    workspace: object, record: dict, *, C: int, rows: int, device: torch.device
) -> torch.Tensor:
    """The FP32 ``[rows, C, taps]`` capacity view of this launch's partial rows inside ``workspace``.

    ``taps`` is the record's (4, or 5 for a bias program whose kernel lays the
    ``[rows, C, 4]`` tap slab out first and the ``[rows, C]`` dbias slab after
    it); the view only sizes the kernel's buffer.
    """

    ws = _require_tensor(workspace, "workspace")
    if ws.ndim != 1:
        raise ValueError(
            f"workspace must be a 1-D uint8 CUDA tensor, got shape {tuple(ws.shape)}"
        )
    _check_placement(ws, "workspace", device=device, dtype=torch.uint8)
    if ws.data_ptr() % 16 != 0:
        raise ValueError("workspace must be 16-byte aligned")
    nbytes = partial_bytes(record, C, rows)
    if ws.numel() < nbytes:
        raise ValueError(
            f"workspace holds {ws.numel()} bytes; this call needs at least "
            f"{nbytes}; allocate "
            "cake_causal_conv1d_train_backward_workspace_bytes(S, C, device) bytes"
        )
    taps = int(record["workspace_rule"]["taps"])
    return ws[:nbytes].view(torch.float32).view(rows, C, taps)


def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations of every
        document; ``C`` is a multiple of 8.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the
        current token, ``weight[c, 0]`` the token three positions back.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` contiguous device prefix sums of the document
        lengths (``cu_seqlens[0] == 0``, ``cu_seqlens[-1] == S``; a repeated
        value is an empty document).  The host reads its metadata only.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias (its dtype is
        independent of ``weight``'s), added after the four taps.
    out : torch.Tensor, optional
        Preallocated BF16 output of ``x``'s shape, contiguous.
    workspace : torch.Tensor, optional
        Caller-owned int32 buffer from
        :func:`cake_causal_conv1d_train_forward_workspace` (the token-chunk
        counters, zero before its first launch; every launch leaves it zero,
        so one buffer serves every later call of one stream).  Two launches
        that may run at once need two buffers.  When omitted the package
        uses a zeroed buffer it keeps per (device, current stream), or a
        fresh zeroed one inside a CUDA Graph capture.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c] + bias[c]))`` with zero history before each document's
        first token.  ``S == 0`` returns the empty output without launching.
    """

    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device, dtypes=PARAMETER_DTYPES)
    b = None if bias is None else _bias(bias, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    if S == 0:
        return y
    kind = forward_route(bias=b is not None, bf16_taps=w.dtype == torch.bfloat16)
    names = _route(arch, kind, needs="this forward (bias / BF16 taps)", C=C, S=S)
    device_index = _device_index(device)
    scalars, grid = _launch(
        _loader.MODULES[names[0]], S, C, num_docs, device_index=device_index
    )
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "y": y.view(S, C),
        **_scalars(scalars),
        "flags": _flags(w, b=b),
    }
    if b is not None:
        values["b"] = b
    words = max(
        counter_words(_loader.MODULES[name], scalars["n_ch_tiles"]) for name in names
    )
    work = _forward_work(workspace, words, device=device, device_index=device_index)
    if work is not None:
        values["work"] = work
    for name in names:
        run_module(
            name,
            values,
            device_index=device_index,
            grid=grid if name == names[0] else None,
        )
    return y


def cake_causal_conv1d_train_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dy: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    silu_grad_preactivation: str = "bf16",
    workspace: Optional[torch.Tensor] = None,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    dbias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward.

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        Where the SiLU derivative of the weight gradient is evaluated:
        ``"bf16"`` at the BF16-rounded recomputed preactivation (the shipped
        policy and FLA's boundary; ``dweight`` within ~3e-5 rel-L2 of exact),
        ``"fp32"`` at the unrounded FP32 preactivation with an FP64 final
        reduction (~2e-7).  ``dx`` is within one BF16 ulp of exact either way.
    workspace : torch.Tensor, optional
        Caller-owned ``uint8`` buffer from
        :func:`cake_causal_conv1d_train_backward_workspace`; allocated when
        omitted.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted), contiguous.
        ``dbias`` needs ``bias``.

    Returns
    -------
    ``(dx, dweight)``, or ``(dx, dweight, dbias)`` when ``bias`` is given: BF16
    ``dx`` of ``x``'s shape and the ``[C, 4]`` weight gradient (and ``[C]`` bias
    gradient), each accumulated in a fixed order, so they are bitwise
    reproducible for a given ``(S, C, device)`` and route.  ``S == 0`` returns
    the empty ``dx`` and zero gradients without launching.

    The call's arguments select the route (``cake_jit.backward_route``:
    ``bias``, the policy, BF16 among ``weight`` / ``dweight``), and within a
    route the channel count selects the program through the registry's
    host-side rule (``cake_jit.selected_modules``; the data is never read), so a
    given ``(S, C, device)`` and route always run the same program.  The
    ``workspace`` is sized for every program of every route
    (:func:`cake_causal_conv1d_train_backward_workspace_bytes`); the selected
    program uses the leading rows of its own grid.

    A call is two ordinary launches on the current stream: the selected
    program writes ``dx`` and one FP32 partial row of the gradients per CTA,
    then the route's reduce program, launched programmatically dependent on
    it (the launch latency overlaps the main kernel's tail; the dependency is
    a stream-order edge, so the pair captures into a CUDA Graph), sums the
    rows per channel in a fixed order into ``dweight`` (and ``dbias``) in the
    requested dtypes.  Neither launch is cooperative, so the backward keeps
    its speed while kernels of other streams (NCCL collectives) are resident:
    a cooperative grid could not start until they finished.  Build every
    library of the device's architecture (``cake_jit.specs_for_arch``) before
    capturing a CUDA Graph: the capture holds the programs selected for its
    shape and route, and no JIT build may happen inside it.
    """

    policy = _silu_grad_preactivation(silu_grad_preactivation)
    x = _require_tensor(x, "x")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x", device=device)
    w = _weight(weight, C, device=device, dtypes=PARAMETER_DTYPES)
    b = None if bias is None else _bias(bias, C, device=device)
    cu, num_docs = _cu_seqlens(cu_seqlens, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _output(
        dweight,
        "dweight",
        shape=(C, WIDTH),
        dtype=w.dtype,
        device=device,
        dtypes=PARAMETER_DTYPES,
    )
    if b is None:
        if dbias is not None:
            raise ValueError(
                "dbias is the bias gradient's output; pass bias to produce it"
            )
        db_out = None
    else:
        db_out = _output(
            dbias,
            "dbias",
            shape=(C,),
            dtype=b.dtype,
            device=device,
            dtypes=PARAMETER_DTYPES,
        )
    if S == 0:
        dw_out.zero_()
        if db_out is None:
            return dx_out, dw_out
        db_out.zero_()
        return dx_out, dw_out, db_out
    kind = backward_route(
        bias=b is not None,
        fp32_grad=policy == "fp32",
        bf16_taps_or_dweight=torch.bfloat16 in (w.dtype, dw_out.dtype),
    )
    # Within the route the program is a pure function of the architecture and C
    # (S never enters): its own grid sizes the partial rows this call writes
    # inside the workspace.
    names = _route(
        arch,
        kind,
        needs=f"this backward (bias / BF16 taps or dweight / silu_grad_preactivation={policy!r})",
        C=C,
        S=S,
    )
    record = _loader.MODULES[names[0]]
    if record["workspace_rule"] is None:
        raise RuntimeError(
            f"the backward program {names[0]} selected for C={C} on {arch} owns no "
            "partial workspace"
        )
    device_index = _device_index(device)
    scalars, grid = _launch(record, S, C, num_docs, device_index=device_index)
    grid_x = grid[0]
    if workspace is None:
        workspace = cake_causal_conv1d_train_backward_workspace(S, C, device)
    rows = partial_rows(grid_x, scalars["n_ch_tiles"])
    partial = _partial_view(workspace, record, C=C, rows=rows, device=device)
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "partial": partial,
        **_scalars(scalars),
        "flags": _flags(w, dw_out, b, db_out),
        "main_ctas": grid_x,
    }
    if b is not None:
        values["b"] = b
        values["db"] = db_out
    # The main program (an ordinary launch) writes dx and the partial rows; the
    # route's reduce program, programmatically dependent on it through its
    # binding's launch attribute, sums into dweight (and dbias) on the same
    # stream the rows of each channel tile that a grid of main_ctas CTAs wrote.
    # Both are issued back to back in one stream context, so the reduce is
    # queued while the main program runs.
    run_modules(
        ((names[0], grid), (record["reduce"], None)),
        values,
        device_index=device_index,
    )
    if db_out is None:
        return dx_out, dw_out
    return dx_out, dw_out, db_out


class CakeCausalConv1dFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x, weight, cu_seqlens, bias)`` for the backward.

    ``forward(x, weight, cu_seqlens, bias=None, silu_grad_preactivation="bf16")``
    returns ``y``; the backward runs the fused ``dx`` + deterministic
    ``dweight`` (+ ``dbias``) pair of launches of :func:`cake_causal_conv1d_train_backward`
    on the incoming gradient (materialized as a contiguous BF16 tensor when
    autograd hands over another layout or dtype) and returns the gradients of
    ``weight`` and ``bias`` in the parameters' dtypes.
    """

    @staticmethod
    def forward(ctx, x, weight, cu_seqlens, bias=None, silu_grad_preactivation="bf16"):
        y = cake_causal_conv1d_train_forward(x, weight, cu_seqlens, bias=bias)
        ctx.save_for_backward(x, weight, cu_seqlens, bias)
        ctx.silu_grad_preactivation = _silu_grad_preactivation(silu_grad_preactivation)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, weight, cu_seqlens, bias = ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x)
        dy = grad_y.to(torch.bfloat16).contiguous()
        grads = cake_causal_conv1d_train_backward(
            x,
            weight,
            cu_seqlens,
            dy,
            bias=bias,
            silu_grad_preactivation=ctx.silu_grad_preactivation,
        )
        dbias = grads[2] if bias is not None else None
        return grads[0], grads[1], None, dbias, None


def cake_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU.

    Parameters
    ----------
    x : torch.Tensor
        BF16 ``[S, C]`` or ``[1, S, C]`` contiguous packed activations on a
        supported CUDA device.
    weight : torch.Tensor
        FP32 or BF16 ``[C, 4]`` contiguous taps.
    cu_seqlens : torch.Tensor
        ``int32 [num_docs + 1]`` device prefix sums of the document lengths.
    bias : torch.Tensor, optional
        FP32 or BF16 ``[C]`` contiguous per-channel bias.
    silu_grad_preactivation : {"bf16", "fp32"}
        The weight gradient's SiLU' policy (see
        :func:`cake_causal_conv1d_train_backward`).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns BF16 ``dx`` and the ``dweight``
        (and ``dbias``) in the parameters' dtypes.
    """

    return CakeCausalConv1dFunction.apply(
        x, weight, cu_seqlens, bias, silu_grad_preactivation
    )


__all__ = [
    "ARCHES",
    "PARAMETER_DTYPES",
    "SILU_GRAD_PREACTIVATIONS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "CakeCausalConv1dFunction",
    "cake_causal_conv1d",
    "cake_causal_conv1d_train_backward",
    "cake_causal_conv1d_train_backward_workspace",
    "cake_causal_conv1d_train_backward_workspace_bytes",
    "cake_causal_conv1d_train_forward",
    "cake_causal_conv1d_train_forward_workspace",
    "cake_causal_conv1d_train_forward_workspace_bytes",
    "is_cake_causal_conv1d_train_supported",
]
