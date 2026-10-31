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

Non-deterministic backend of the Cake packed width-4 depthwise causal
convolution + SiLU training forward / backward (SM100 / SM103 / SM107), one
self-contained package (host API bodies, the ``cake_jit`` registry, the pinned
``cake_build`` loader and the generated CUDA sources under ``csrc/``).  It
imports nothing from other flashinfer packages.  The dispatching entry points
(``deterministic=True`` -> the deterministic package
``flashinfer.cake_causal_conv1d_det_train``) and the user-facing documentation
live in :mod:`flashinfer.experimental.cake.cake_causal_conv1d_train`.

Numerical contract against the deterministic package: the forward runs this
package's copies of the det forward programs, so ``y`` is the det ``y`` bit
for bit.  The backward is ONE ordinary launch of a program built from the det
backward body of the same route and channel rule: every per-thread and
per-CTA FP32 step is the det program's and ``dx`` is the det ``dx`` bit for
bit; only the cross-CTA reduction of ``dweight`` / ``dbias`` differs.  Each
CTA adds its CTA row into an accumulator with unordered global reductions --
FP32 where the det reduce program sums in FP32 (the ``bwd``, ``bwd_dt`` and
``bwd_bias`` routes), FP64 where it sums in FP64 (``bwd_fp32g`` and
``bwd_bias_fp32g``, the ``silu_grad_preactivation="fp32"`` routes, whose
gradient is then rounded once from FP64) -- and the last CTA of each channel
tile stores the gradients and zeroes its accumulators.  The order of that
cross-CTA sum depends on the CTAs' timing, so ``dweight`` / ``dbias`` are not
bitwise reproducible run to run; that is the only relaxed property (the
accumulation dtypes, rounding boundaries and accuracy budgets are the det
package's).  No reduce program, no host synchronization, no cooperative
launch, no grid barrier.

Numerical-mode keyword: every entry point takes ``deterministic: bool = True``
and raises a ``ValueError`` unless it is ``False`` -- this package has no
deterministic programs, so the call must opt in explicitly (the facade routes
``deterministic=True`` to the deterministic package).  With
``torch.use_deterministic_algorithms(True)`` the entry points that launch the
unordered reduction (the backward, hence the autograd backward) call
``torch._C._alert_not_deterministic`` (a ``RuntimeError``, or a warning under
``warn_only=True``) and never fall back.  The forward is deterministic (the
det programs' bits) and does not alert.

Host design (the knobs are keywords; no environment variable):

- Taps ``weight[C, 4]`` are FP32 or BF16; an optional per-channel ``bias[C]``
  is FP32 or BF16, independently of the taps.  ``dweight`` / ``dbias`` default
  to the parameter's dtype and may be preallocated as FP32 or BF16 (the
  ``flags`` bit mask tells the kernels which).
- ``silu_grad_preactivation``: ``"bf16"`` (default) or ``"fp32"`` (SiLU' at
  the unrounded FP32 preactivation; FP64 accumulators).
- A call's arguments select its route (``cake_jit.forward_route`` /
  ``backward_route``) and the channel count the route's program, the det
  package's rules.
- The backward workspace holds the accumulators ``acc`` and the per channel
  tile ticket counters ``work`` (``cake_jit.accumulator_layout``): zero before
  the first launch and left zero by every launch.  A caller-owned buffer
  (``workspace=``, from :func:`cake_causal_conv1d_train_backward_workspace`,
  which returns it zeroed) serves every later backward call of one stream;
  when omitted the package keeps one zeroed buffer per (device, current
  stream), grown as needed and never zeroed again, and inside a CUDA Graph
  capture an omitted workspace is a fresh zeroed buffer of the call's own.
  Its size depends on ``C`` (and the route), not on ``S``.  Two backward
  launches that may run at once (two streams, two graphs replayed
  concurrently) need two buffers: never share one between them.
- The forward workspace (int32 token-chunk counters) follows the det
  package's rules.
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
    accumulator_layout,
    arch_for_capability,
    backward_route,
    backward_workspace_bytes,
    counter_words,
    forward_route,
    forward_workspace_words,
    launch_grid,
    plan,
    route_applies,
    run_module,
    selected_modules,
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
# The backward workspaces of the calls that pass none (single-rank and KCP):
# (device index, CUDA stream handle) -> a zeroed uint8 buffer of accumulators and
# ticket counters, grown as needed.  Every launch leaves its accumulators and
# counters zero, so the stream's buffer serves every later backward call on it.
_BACKWARD_WORKSPACES: dict[tuple[int, int], torch.Tensor] = {}
# The torch dtype of each accumulator / counter dtype name of a workspace rule.
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "uint32": torch.uint32,
}


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


def _require_nondeterministic(deterministic: object, entry: str) -> None:
    """Refuse every numerical mode but ``deterministic=False``: this package has no deterministic programs."""

    if not isinstance(deterministic, bool) or deterministic:
        raise ValueError(
            f"{entry} is the non-deterministic implementation (unordered dweight / dbias reductions) and needs "
            f"deterministic=False, got deterministic={deterministic!r}; the deterministic programs are "
            "flashinfer.cake_causal_conv1d_det_train (or the facade "
            "flashinfer.experimental.cake.cake_causal_conv1d_train with deterministic=True)"
        )


def _alert_nondeterministic(entry: str) -> None:
    """``torch._C._alert_not_deterministic(entry)`` when torch's deterministic algorithms are enabled (a
    ``RuntimeError``, or a warning under ``warn_only=True``); nothing otherwise.  Called by the entry points that
    launch the unordered reduction, before they launch."""

    if torch.are_deterministic_algorithms_enabled():
        torch._C._alert_not_deterministic(entry)


def is_cake_causal_conv1d_train_supported(device: torch.device) -> bool:
    """Whether ``device`` has the exported routes every architecture ships (``fwd`` and ``bwd``) and the package
    is rendered (its ``build.json`` exists).

    The optional routes (bias, BF16 taps / gradients, FP32 SiLU') are checked
    per call: a call that needs one the architecture lacks raises a
    ``ValueError`` naming it.
    """

    device = torch.device(device)
    if device.type != "cuda" or not _loader.is_rendered():
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
    a CTA of the launched program's grid on the device (for the backward also
    so that every tile has a last CTA that stores its gradients): a ``C``
    beyond ``grid x threads x vec`` channels is refused rather than left partly
    unwritten.  The caller launches this grid, so a call plans it once.
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


def _check_shape_args(S: int, C: int) -> tuple[int, int]:
    S, C = int(S), int(C)
    if S < 0:
        raise ValueError("S must be non-negative")
    if C <= 0 or C % CHANNEL_MULTIPLE:
        raise ValueError(
            f"C must be a positive multiple of {CHANNEL_MULTIPLE}, got {C}"
        )
    return S, C


def cake_causal_conv1d_train_backward_workspace_bytes(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned backward workspace for ``C`` channels on ``device`` (``deterministic=False``
    required).

    The workspace holds the backward's accumulators and per channel tile ticket
    counters (``cake_jit.accumulator_layout``): ``taps x C`` FP32 words (FP64
    on the ``_fp32g`` routes; ``taps`` 5 on the bias routes) and one uint32
    counter per channel tile, sized by the largest over every backward
    program of every backward route of the architecture
    (``cake_jit.backward_workspace_bytes``), so one buffer serves every route.
    It depends on ``C`` and the architecture only (``S`` is checked and
    otherwise unused): one buffer serves every token count.
    """

    _require_nondeterministic(
        deterministic, "cake_causal_conv1d_train_backward_workspace_bytes"
    )
    device = torch.device(device)
    S, C = _check_shape_args(S, C)
    return backward_workspace_bytes(_device_arch(device), C)


def cake_causal_conv1d_train_backward_workspace(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """Allocate a ZEROED ``uint8`` backward workspace (see ``..._workspace_bytes``; ``deterministic=False``
    required).

    Every launch leaves it zero, so one buffer serves every later backward call
    of one stream, CUDA Graph replays included.  Two launches that may run at
    once need two buffers.
    """

    _require_nondeterministic(
        deterministic, "cake_causal_conv1d_train_backward_workspace"
    )
    nbytes = cake_causal_conv1d_train_backward_workspace_bytes(
        S, C, device, deterministic=False
    )
    return torch.zeros(nbytes, dtype=torch.uint8, device=torch.device(device))


def cake_causal_conv1d_train_forward_workspace_bytes(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> int:
    """Bytes of the caller-owned forward workspace for ``C`` channels on ``device`` (``deterministic=False``
    required).

    The forward's programs take their token chunks from int32 counters, two
    per channel tile, sized by the most over every forward program of every
    forward route of the architecture (``cake_jit.forward_workspace_words``):
    the det package's rule over the same programs.  The buffer must be zero
    before its first launch; every launch leaves it zero.  Two launches that
    may run at once need two buffers.
    """

    _require_nondeterministic(
        deterministic, "cake_causal_conv1d_train_forward_workspace_bytes"
    )
    device = torch.device(device)
    S, C = _check_shape_args(S, C)
    return 4 * forward_workspace_words(_device_arch(device), C)


def cake_causal_conv1d_train_forward_workspace(
    S: int, C: int, device: torch.device, *, deterministic: bool = True
) -> torch.Tensor:
    """Allocate a zeroed int32 forward workspace (see ``..._workspace_bytes``; ``deterministic=False`` required);
    keep it for every later forward call of one stream."""

    _require_nondeterministic(
        deterministic, "cake_causal_conv1d_train_forward_workspace"
    )
    nbytes = cake_causal_conv1d_train_forward_workspace_bytes(
        S, C, device, deterministic=False
    )
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
                f"{words}; allocate cake_causal_conv1d_train_forward_workspace(S, C, device, "
                "deterministic=False)"
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


def _backward_accumulators(
    workspace: object,
    layout: dict,
    *,
    device: torch.device,
    device_index: int,
    entry: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(acc, work)`` views of one backward launch with the byte ``layout`` of
    :func:`cake_jit.accumulator_layout` (shared by the single-rank and the KCP backward).

    A caller-owned ``workspace`` (1-D uint8, contiguous, 16-byte aligned, on
    ``device``, at least ``layout["total_bytes"]`` bytes, ZERO before its first
    launch: allocate it with :func:`cake_causal_conv1d_train_backward_workspace`)
    is used as is.  Without one: inside a CUDA Graph capture a fresh zeroed
    buffer (its zeroing is captured with the launch, so the graph shares no
    accumulators with eager calls or other graphs); else the zeroed buffer this
    module keeps for the device and the current stream, replaced by a larger
    zeroed one when it is too small.  Every launch leaves the bytes it uses
    zero, so a zeroed buffer stays zero whatever ``C`` or route the next call
    has.  ``acc`` is a ``float32`` / ``float64`` view of ``layout["acc_elems"]``
    words at offset 0, ``work`` a ``uint32`` view of ``layout["work_words"]``
    counters at ``layout["work_offset"]``.
    """

    nbytes = int(layout["total_bytes"])
    if workspace is not None:
        ws = _require_tensor(workspace, "workspace")
        if ws.ndim != 1:
            raise ValueError(
                f"workspace must be a 1-D uint8 CUDA tensor, got shape {tuple(ws.shape)}"
            )
        _check_placement(ws, "workspace", device=device, dtype=torch.uint8)
        if ws.data_ptr() % _loader.WORKSPACE_ALIGN != 0:
            raise ValueError(
                f"workspace must be {_loader.WORKSPACE_ALIGN}-byte aligned"
            )
        if ws.numel() < nbytes:
            raise ValueError(
                f"workspace holds {ws.numel()} bytes; {entry} needs at least {nbytes}; allocate "
                "cake_causal_conv1d_train_backward_workspace(S, C, device, deterministic=False) "
                "(a zeroed buffer)"
            )
    elif torch.cuda.is_current_stream_capturing():
        ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    else:
        key = (
            device_index,
            int(torch.cuda.current_stream(device_index).cuda_stream),
        )
        ws = _BACKWARD_WORKSPACES.get(key)
        if ws is None or ws.numel() < nbytes:
            ws = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            _BACKWARD_WORKSPACES[key] = ws
    acc_end = int(layout["acc_bytes"])
    work_offset = int(layout["work_offset"])
    work_end = work_offset + 4 * int(layout["work_words"])
    acc = ws[:acc_end].view(_TORCH_DTYPES[layout["acc_dtype"]])
    work = ws[work_offset:work_end].view(_TORCH_DTYPES[layout["work_dtype"]])
    return acc, work


def cake_causal_conv1d_train_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    r"""Packed width-4 depthwise causal convolution + SiLU, training forward (``deterministic=False`` required).

    The forward has no cross-CTA reduction: this package's programs are copies
    of the det forward programs, so ``y`` is bitwise the det
    ``cake_causal_conv1d_det_train_forward``'s (the forward does not alert
    under torch's deterministic mode).

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
    deterministic : bool
        Must be ``False`` (``ValueError`` otherwise): the numerical mode of this
        package.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape: ``y[t, c] = bf16(silu(sum_k weight[c, k] *
        x[t - 3 + k, c] + bias[c]))`` with zero history before each document's
        first token.  ``S == 0`` returns the empty output without launching.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_train_forward")
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
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    r"""Packed width-4 depthwise causal convolution + SiLU, training backward, ONE ordinary launch with unordered
    cross-CTA ``dweight`` / ``dbias`` reductions (``deterministic=False`` required).

    Parameters
    ----------
    x, weight, cu_seqlens, bias
        The forward's inputs (same contracts as
        :func:`cake_causal_conv1d_train_forward`); nothing else is saved.
    dy : torch.Tensor
        BF16 gradient of ``y``, ``x``'s shape, contiguous.
    silu_grad_preactivation : {"bf16", "fp32"}
        Where the SiLU derivative of the weight gradient is evaluated:
        ``"bf16"`` at the BF16-rounded recomputed preactivation (FP32
        accumulators), ``"fp32"`` at the unrounded FP32 preactivation with FP64
        accumulators (the det FP64 reduction's accuracy).  ``dx`` is within one
        BF16 ulp of exact either way, and bitwise the det package's.
    workspace : torch.Tensor, optional
        Caller-owned ZEROED ``uint8`` buffer from
        :func:`cake_causal_conv1d_train_backward_workspace` (the accumulators and
        ticket counters; every launch leaves it zero, so one buffer serves every
        later call of one stream; never share one between two launches that may
        run at once).  When omitted the package uses a zeroed buffer it keeps per
        (device, current stream), or a fresh zeroed one inside a CUDA Graph
        capture.
    dx, dweight, dbias : torch.Tensor, optional
        Preallocated outputs: BF16 of ``x``'s shape, ``[C, 4]`` and ``[C]``
        (FP32 or BF16 each; the parameter's dtype when omitted), contiguous.
        ``dbias`` needs ``bias``.  Every element is overwritten.
    deterministic : bool
        Must be ``False`` (``ValueError`` otherwise).  Under
        ``torch.use_deterministic_algorithms(True)`` the call raises (or warns
        with ``warn_only=True``) through ``torch._C._alert_not_deterministic``.

    Returns
    -------
    ``(dx, dweight)``, or ``(dx, dweight, dbias)`` when ``bias`` is given: BF16
    ``dx`` of ``x``'s shape (the det program's bits) and the ``[C, 4]`` weight
    gradient (and ``[C]`` bias gradient), accumulated across CTAs in an
    unordered order in FP32 (FP64 under ``"fp32"``) and rounded once into the
    requested dtype.  ``S == 0`` returns the empty ``dx`` and zero gradients
    without launching.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_train_backward")
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
    # Every argument is valid: a call under torch's deterministic mode is refused (or warned about) here, before
    # anything is allocated or launched.
    _alert_nondeterministic("cake_causal_conv1d_train_backward")
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
    names = _route(
        arch,
        kind,
        needs=f"this backward (bias / BF16 taps or dweight / silu_grad_preactivation={policy!r})",
        C=C,
        S=S,
    )
    record = _loader.MODULES[names[0]]
    device_index = _device_index(device)
    scalars, grid = _launch(record, S, C, num_docs, device_index=device_index)
    layout = accumulator_layout(record, C, scalars["n_ch_tiles"])
    acc, work = _backward_accumulators(
        workspace,
        layout,
        device=device,
        device_index=device_index,
        entry="cake_causal_conv1d_train_backward",
    )
    values: dict[str, object] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "acc": acc,
        "work": work,
        **_scalars(scalars),
        "flags": _flags(w, dw_out, b, db_out),
    }
    if b is not None:
        values["b"] = b
        values["db"] = db_out
    run_module(names[0], values, device_index=device_index, grid=grid)
    if db_out is None:
        return dx_out, dw_out
    return dx_out, dw_out, db_out


class CakeCausalConv1dFunction(torch.autograd.Function):
    """Autograd wrapper saving only ``(x, weight, cu_seqlens, bias)`` for the backward.

    ``forward(x, weight, cu_seqlens, bias=None, silu_grad_preactivation="bf16", deterministic=True)`` returns
    ``y`` and refuses every ``deterministic`` but ``False`` (``ValueError``); the backward runs the one-launch
    ``dx`` + unordered ``dweight`` (+ ``dbias``) program of :func:`cake_causal_conv1d_train_backward` on the
    incoming gradient (materialized as a contiguous BF16 tensor when autograd hands over another layout or
    dtype) and returns the gradients of ``weight`` and ``bias`` in the parameters' dtypes.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        cu_seqlens,
        bias=None,
        silu_grad_preactivation="bf16",
        deterministic=True,
    ):
        _require_nondeterministic(deterministic, "CakeCausalConv1dFunction")
        y = cake_causal_conv1d_train_forward(
            x, weight, cu_seqlens, bias=bias, deterministic=False
        )
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
            deterministic=False,
        )
        dbias = grads[2] if bias is not None else None
        return grads[0], grads[1], None, dbias, None, None


def cake_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    silu_grad_preactivation: str = "bf16",
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled packed width-4 depthwise causal convolution + SiLU, non-deterministic backward
    (``deterministic=False`` required).

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
    deterministic : bool
        Must be ``False`` (``ValueError`` otherwise).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x``'s shape.  The backward saves only ``x``, ``weight``,
        ``cu_seqlens`` and ``bias`` and returns BF16 ``dx`` and the ``dweight``
        (and ``dbias``) in the parameters' dtypes.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d")
    return CakeCausalConv1dFunction.apply(
        x, weight, cu_seqlens, bias, silu_grad_preactivation, False
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
