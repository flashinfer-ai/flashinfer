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

Context-parallel (KCP) halo forward / backward of the Cake packed width-4
depthwise causal convolution + SiLU, non-deterministic package: the rank's
fragments of the packed documents are convolved with the three document tokens
before each continuing fragment, received from the ranks that hold them by one
``all_to_all_single`` per direction, and the backward sends the first rows'
derivative back to the predecessor fragments, which fold it into their last
rows' ``dx`` before the one BF16 rounding.  ``cake_causal_conv1d_kcp(x_local,
weight, ctx, deterministic=False)`` is the autograd entry; the dispatching
facade is :mod:`flashinfer.cake_causal_conv1d_train_kcp`.  This module imports
only ``os``, torch, torch.distributed, tvm_ffi and its own package.

The host flow mirrors the deterministic package's
(``flashinfer.cake_causal_conv1d_det_train_backend.kcp``): the same context
contract (the route reads ``group``, ``world_size``, ``cu_seqlens_local``,
``cu_seqlens_local_cpu``, ``halo``, ``halo_tables[name]``, ``n_frag``,
``T_r``, ``conv_kernel_size`` and ``wait_ready()``; the numerical mode is
never stored in the context), FP32 taps only, no bias, the legacy channel tile
rule ``cake_jit.plan(..., legacy=True)``, the same forward and head programs
(this package's copies of the det programs) and the same exchanges, recompute
handle, graph replay and multi-tensor calls.  The one difference is the main
backward program (registry ``kcp_jit``, kind ``kcp_bwd``): ONE ORDINARY launch
(no cooperative attribute, no grid barrier, so a kernel resident on another
stream never holds it back) whose CTAs add their FP32 ``dweight`` CTA rows into
the zeroed FP32 accumulators ``acc`` (``[C, 4]``) with unordered global
reductions; the last CTA of each channel tile (a ticket on its ``work``
counter) stores the rank-local FP32 ``dweight`` partial and zeroes the tile's
accumulators and counter.  ``acc`` / ``work`` are the package's cached
backward workspace of the device and current stream (zero before and after
every launch; a fresh zeroed buffer inside a CUDA Graph capture).

Numerics: ``y`` and ``hist`` are the det KCP forward's bits; ``dx`` is the det
KCP backward's bits (same per-thread FP32 steps, same tail fold, one BF16
rounding); ``dweight`` is the rank's FP32 partial, within the det program's
budget of an FP64 oracle (``5e-2 + 1e-4 |dw|``) but not bitwise reproducible
run to run (the cross-CTA sum is unordered; the only relaxed property).  The
cross-rank reduction of ``dweight`` stays the framework's.

Numerical-mode keyword: every public entry takes ``deterministic: bool = True``
and raises a ``ValueError`` unless it is ``False``; with
``torch.use_deterministic_algorithms(True)`` the backward entries (hence the
autograd backward) call ``torch._C._alert_not_deterministic`` and never fall
back.

Shipped host path (this revision;
``HOST_TRIM_DEFAULT = 1``): the context-derived host work of the eager halo
call -- the validated ``cu`` / ``hist_index`` / row-offset tables and halo
splits, the plan scalars, the grids, the record names and the loaded modules
with their positional argument plans -- is computed once per context and
reused, and the kernels launch through a raw-stream FFI context of the current
stream (no ``torch.cuda.Stream`` object per launch).  ``_HOST_CACHE`` is keyed
on the context object's identity (the entry pins the context) and, inside it,
on ``(kind, S, C, device, cu_seqlens_local_cpu)`` -- host integers of the
context and the shapes of the call, never a tensor's contents; it holds at most
``HOST_CACHE_MAX = 64`` contexts (the oldest dropped) and
:func:`host_cache_clear` empties it; ``_FFI_DEVICES`` caches the FFI device per
device index.  The operand checks, the gather, the exchange, its wait and the
kernel launches run as before with the same arguments on the same stream, so
no bit changes; a replayed graph (``graph="replay"``) contains no host work.
``_set_test_knobs({"host_trim": 0})`` (with ``CAKE_KCP_TEST_KNOBS=1``) selects
the previous host path for the A/B record.

Checkpoint recompute without the halo exchange (FlashInfer issues #7815 /
#7818): with ``recompute_handle=`` (:class:`RecomputeHandleKcp`, or any object
with an ``entries`` dict) the original forward exchanges as always and stores
its history pool ``hist`` (the saved set's own tensor, key ``"conv"``) in the
handle; torch's recompute of that forward inside the backward binds it and
launches the same forward kernel on it -- no gather, no collective, no wait --
so ``y`` and the saved set ``(x_local, weight, hist)`` are bitwise those of the
exchanging recompute.  The backward is unchanged (it still exchanges the head
rows).  The forward exchange is a collective: every rank of ``ctx.group`` passes
a handle for the same call or none, created at the same checkpoint nesting
level (no check is made; a mismatch hangs or corrupts data).

Several convolutions of one layer (FlashInfer issue #7815): ``cake_causal_conv1d_multi_kcp(xs, weights, ctx,
deterministic=False)`` runs
the convolutions of ``n`` tensors that share the context (a layer's q, k and v) with one halo exchange per direction
instead of ``n``: the tensors' send rows travel side by side in one ``all_to_all_single`` and are copied out into
each tensor's own contiguous pool after the wait, and each tensor then runs the single call's kernels with the
single call's arguments, so outputs and ``dx`` are bitwise one single call per tensor (``dweight`` is each
tensor's own unordered partial).
One recompute handle serves the whole call (its ``"conv"`` entry holds the ``n`` pools).  The multi path always
uses the trimmed host path's per-context cache (the ``host_trim`` test knob selects the single call's path only).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import tvm_ffi

from . import (
    _alert_nondeterministic,
    _backward_accumulators,
    _cu_seqlens,
    _device_arch,
    _device_index,
    _output,
    _require_nondeterministic,
    _require_tensor,
    _scalars,
    _tokens,
    _weight,
)
from . import kcp_jit as _loader
from .cake_jit import WIDTH, accumulator_layout, launch_grid, plan
from .kcp_jit import load, route_modules, run_module, selected_modules

HALO = WIDTH - 1  # history rows, tail rows and head rows per fragment
HEAD_SLOTS = HALO  # send slots per head row (a head token is consumed by at most three predecessor fragments)
KCP_FWD = "kcp_fwd"
KCP_BWD = "kcp_bwd"
KCP_BWD_HEAD = "kcp_bwd_head"
TEST_KNOBS_ENV = "CAKE_KCP_TEST_KNOBS"  # the KCP route's private test-hook gate (the variable the KDA host module reads too)
GRAPH_MODES = (
    "off",
    "replay",
)  # graph replay: keyed CUDA-graph capture / replay of the halo calls; public keyword graph= since an earlier revision
GRAPH_DEFAULT = (
    "off"  # shipped: the eager path; the public keyword graph="replay" opts in per call
)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _table(
    ctx: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """One of the context's halo tables, checked against the kernels' ABI shape."""

    table = _require_tensor(ctx.halo_tables[name], f"ctx.halo_tables[{name!r}]")
    if tuple(table.shape) != shape or table.dtype != dtype:
        raise ValueError(
            f"ctx.halo_tables[{name!r}] must be {dtype} {shape}, got {table.dtype} {tuple(table.shape)}"
        )
    if table.device != device or not table.is_contiguous():
        raise ValueError(
            f"ctx.halo_tables[{name!r}] must be a contiguous tensor on {device}"
        )
    return table


def _pool(
    rows: int, C: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A fresh row pool; a pool without rows still holds one row so that the kernel has a valid pointer (no table
    entry names it)."""

    return torch.empty((max(rows, 1), C), dtype=dtype, device=device)


def _exchange(
    send: torch.Tensor,
    *,
    recv_rows: int,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    ctx: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    """One ``all_to_all_single`` of ``send`` (rows in destination-rank order) into a fresh ``[recv_rows, C]`` tensor,
    issued asynchronously on the current stream; with one rank the send buffer is the received pool (no collective).
    Returns ``(recv, work)``; the caller waits ``work`` at the first consumer."""

    if int(ctx.world_size) == 1:
        return send, None
    recv = torch.empty((recv_rows, send.shape[1]), dtype=send.dtype, device=send.device)
    work = dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=ctx.group,
        async_op=True,
    )
    return recv, work


def _launch_with_grid(
    name: str, values: dict[str, Any], *, grid: tuple[int, int, int], device_index: int
) -> None:
    """Launch record ``name`` with logical ``values`` and an explicit grid (the head program's grid depends on the
    fragment and channel counts, not on the device); otherwise :func:`cake_jit.run_module`'s argument plan."""

    record = _loader.MODULES[name]
    logical = record["logical"]
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, _loader.device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


def _wait_ready(ctx: Any) -> None:
    """``ctx.wait_ready()`` of a context that has one (a context prepared on another stream orders its table uploads
    on the current stream; a duck-typed context without it needs nothing)."""
    wait_ready = getattr(ctx, "wait_ready", None)
    if wait_ready is not None:
        wait_ready()


def _context(ctx: Any, *, device: torch.device, T_r: int) -> tuple[torch.Tensor, int]:
    """The context's local ``cu_seqlens`` (int32 device ``[n_frag + 1]``) and fragment count, checked against the
    tokens the call received."""

    if int(ctx.T_r) != T_r:
        raise ValueError(
            f"x_local holds {T_r} tokens but ctx was built for T_r={int(ctx.T_r)}"
        )
    if int(ctx.conv_kernel_size) != WIDTH:
        raise ValueError(
            f"ctx was built for a width-{int(ctx.conv_kernel_size)} convolution; this kernel is width {WIDTH}"
        )
    if int(ctx.n_frag) == 0:
        # A rank without fragments (T_r == 0): nothing is launched, the collectives get empty splits.
        if T_r != 0:
            raise ValueError(f"ctx has no fragments but x_local holds {T_r} tokens")
        return _require_tensor(ctx.cu_seqlens_local, "ctx.cu_seqlens_local"), 0
    cu, n_frag = _cu_seqlens(ctx.cu_seqlens_local, device=device)
    if n_frag != int(ctx.n_frag):
        raise ValueError(
            f"ctx.cu_seqlens_local names {n_frag} fragments but ctx.n_frag is {int(ctx.n_frag)}"
        )
    return cu, n_frag


# ----------------------------------------------------------------------------------------------------------------
# Checkpoint recompute without the halo exchange: the recompute handle (``recompute_handle=``).
#
# The phase comes from ``torch._C._current_graph_task_id()`` (checked once: present and returning an int; without
# it a call with a handle raises): -1 outside the autograd engine (an original forward), the running graph task
# inside a backward (torch's checkpoint recompute, reentrant or not).  Rules of a call with a handle:
#   * no "conv" entry -> an original forward (also one inside an outer checkpoint's recompute, nested regions):
#     gather + exchange as always, then store ``hist`` (no copy) and its metadata;
#   * an entry inside the backward -> the recompute: bind the stored ``hist``, validate it against the call, issue
#     no collective; the entry stays (``retain_graph`` / a second backward replays it again);
#   * an entry outside the backward -> raise (a handle reused across micro-steps or for a second convolution).
# Graph replay (keyword or knob) and a forward inside a CUDA-graph capture are refused with a handle, in the
# autograd Function itself (``_handle_refusals``), so ``CakeCausalConv1dFunctionKcp.apply`` refuses like the wrapper.
# The ranks are not checked against each other: the forward exchange is a collective, so every rank of the group
# must take the same branch (all with a handle at the same nesting level, or all without).

HANDLE_KEY = "conv"  # this package's entry in a recompute handle

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: Optional[bool] = None  # None = not checked yet


class RecomputeHandleKcp:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward, so the recompute runs without cross-rank communication.

    Create one handle per convolution call and micro-step, outside the outermost checkpoint region, and pass it as an
    argument of the checkpointed function, so the recompute sees the same object::

        def layer(x, weight, handle):
            return cake_causal_conv1d_kcp(x, weight, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, w, RecomputeHandleKcp(), use_reentrant=False)

    ``entries`` maps an operation key to that operation's entry: ``"conv"`` is this package's (the forward's history
    pool ``hist``, the very tensor the saved set holds); other packages that accept a handle with an ``entries`` dict
    use their own keys, so the handle of a convolution may also serve the recurrence of the same layer.  A handle
    created inside a checkpoint region is a new, empty one in that region's recompute, which then exchanges again
    (correct, not communication-free).  The entries live as long as the handle; :meth:`nbytes` is the retained
    size.

    The forward halo exchange is a collective.  Every rank of ``ctx.group`` must pass a handle for the same call, or
    no rank passes one, and every rank creates it at the same checkpoint nesting level.  A mismatch pairs one rank's
    recompute exchange with another rank's backward exchange: the ranks hang or exchange wrong rows.  No check is
    made.  Never share one handle between two convolution calls (also not in nested regions).  Never call
    :meth:`clear` on a handle, or reuse it, while a backward that needs its entry is pending."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def clear(self) -> None:
        """Drop every entry (the retained tensors go with the last reference)."""
        self.entries.clear()

    def nbytes(self) -> int:
        """Bytes of the distinct tensors the entries hold."""
        seen: dict[int, int] = {}
        for entry in self.entries.values():
            for t in entry.values():
                for x in t if isinstance(t, (tuple, list)) else (t,):
                    if isinstance(x, torch.Tensor):
                        seen[id(x)] = x.numel() * x.element_size()
        return sum(seen.values())


def _kcp_in_backward() -> bool:
    """Whether this thread runs inside the autograd engine's backward pass (torch's checkpoint recompute runs there,
    an original forward does not).  The one place the recompute handle reads the phase.  Raises without the torch
    binding (checked once)."""
    global _GRAPH_TASK_ID_USABLE
    if _GRAPH_TASK_ID_USABLE is None:
        ok = False
        if _GRAPH_TASK_ID_BINDING is not None:
            try:
                ok = isinstance(_GRAPH_TASK_ID_BINDING(), int)
            except Exception:  # noqa: BLE001 - an unusable binding disables the option
                ok = False
        _GRAPH_TASK_ID_USABLE = ok
    if not _GRAPH_TASK_ID_USABLE:
        raise RuntimeError(
            "recompute_handle needs torch._C._current_graph_task_id to tell the recompute from the original "
            "forward; this torch does not provide it"
        )
    return int(_GRAPH_TASK_ID_BINDING()) != -1


def _handle_refusals(graph: Optional[str], handle: Any) -> None:
    """The refusals of a call with a handle, shared by the wrapper and the autograd Function: graph replay (keyword
    or test knob), a forward inside a CUDA-graph capture, a handle without an ``entries`` dict."""
    mode = _graph_mode(graph)
    if mode != "off":
        raise ValueError(
            f"recompute_handle needs graph='off' (keyword and test knob), got {mode!r}"
        )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise ValueError(
            "recompute_handle is refused inside a CUDA-graph capture (a replay would overwrite the stored rows)"
        )
    if not isinstance(getattr(handle, "entries", None), dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (RecomputeHandleKcp), got {type(handle).__name__}"
        )


def _handle_lookup(handle: Any) -> Optional[dict[str, Any]]:
    """The handle's ``"conv"`` entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (it exchanges and stores one).  An entry found outside the backward raises."""
    entries = getattr(handle, "entries", None)
    if not isinstance(entries, dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (RecomputeHandleKcp), got {type(handle).__name__}"
        )
    in_backward = _kcp_in_backward()
    entry = entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'conv' entry outside the backward -- a handle serves one "
            "convolution call of one micro-step; create a new RecomputeHandleKcp per call and micro-step"
        )
    return entry


def _handle_hist(
    entry: dict[str, Any], x_local: torch.Tensor, ctx: Any
) -> torch.Tensor:
    """The stored ``hist`` of ``entry``, checked against this recompute: the same context, token count and channel
    count, and the stored shape / dtype / device (BF16, ``x_local``'s device)."""
    hist = entry.get("hist")
    if not isinstance(hist, torch.Tensor):
        raise ValueError("recompute_handle: the 'conv' entry holds no hist tensor")
    if entry.get("ctx") is not ctx:
        raise ValueError(
            "recompute_handle: the 'conv' entry belongs to another KcpContext"
        )
    S = int(x_local.shape[-2]) if x_local.ndim >= 2 else -1
    C = int(x_local.shape[-1]) if x_local.ndim >= 1 else -1
    if entry.get("key") != (S, C):
        raise ValueError(
            f"recompute_handle: the 'conv' entry was stored for (T_r, C) = {entry.get('key')}, "
            f"this recompute has {(S, C)}"
        )
    if (
        tuple(hist.shape) != entry.get("shape")
        or hist.dtype != entry.get("dtype")
        or hist.device != entry.get("device")
        or hist.dtype != torch.bfloat16
        or hist.device != x_local.device
        or hist.ndim != 2
        or int(hist.shape[1]) != C
    ):
        raise ValueError(
            f"recompute_handle: the stored hist ({hist.dtype} {tuple(hist.shape)} on {hist.device}) does not match "
            f"its entry ({entry.get('dtype')} {entry.get('shape')} on {entry.get('device')}) or this call "
            f"(BF16 [n_hist_rows, {C}] on {x_local.device})"
        )
    return hist


# ----------------------------------------------------------------------------------------------------------------
# CUDA-graph replay of a call and the test knobs.
#
# The public keyword ``graph="replay"`` of :func:`cake_causal_conv1d_kcp` and of the forward / backward below (shipped
# ``GRAPH_DEFAULT = "off"``) -- or, for the A/B record, the private ``_set_test_knobs({"graph": "replay"})`` (accepted
# only with ``CAKE_KCP_TEST_KNOBS=1``; a keyword contradicting an active knob is a ``ValueError``) -- makes
# :func:`cake_causal_conv1d_forward_kcp` / :func:`cake_causal_conv1d_backward_kcp` run their first call of a
# validity key (kind, the context object, its world size, and the shape / stride / dtype / data pointer of every
# operand and caller-owned output) eagerly, capture the second call into a ``torch.cuda.CUDAGraph`` (replayed once)
# and replay the graph on every later call of the key: the halo gather, the ``all_to_all_single``, its wait and the
# kernels with the same arguments in the same order (``y`` / ``hist`` / ``dx`` bitwise the eager result; ``dweight``
# within the eager call's budget, summed in an unordered cross-CTA order) and no host work between them.  Any
# other key runs eagerly (silent fallback); a rank without tokens is never captured.  A replayed call returns the
# graph's static tensors (detached): the next replay of the key overwrites them, so a framework must keep its inputs
# pointer-stable and consume the outputs before the next call of the same layer.  This module keeps its own cache
# (no import of the KDA host module); ``GRAPH_STATS`` counts the events and ``LAST_CALL[kind]`` says how the last
# forward / backward ran (keyword or knob).  Every captured graph keeps its fresh tensors in a private pool (one per
# cached key; at most GRAPH_CACHE_MAX keys, the oldest dropped first); ``graph_cache_clear()`` releases them and must run
# before ``torch.distributed.destroy_process_group()`` (a graph holding the NCCL collective blocks ``ncclCommDestroy``).

# Trimmed host path; shipped default:
# 1 = the trimmed host path of the eager halo call (same launches, bitwise; the module docstring's "Shipped host path"),
# 0 = the previous host path, selectable through _set_test_knobs for the A/B record.
HOST_TRIMS = (0, 1)
HOST_TRIM_DEFAULT = (
    1  # shipped; 0 = the previous path, via _set_test_knobs({"host_trim": 0}) only
)
_TEST_KNOBS: dict[str, Any] = {"graph": "off", "host_trim": HOST_TRIM_DEFAULT}
GRAPH_STATS: dict[str, int] = {"eager": 0, "capture": 0, "replay": 0, "fallback": 0}
GRAPH_CACHE_MAX = 64
LAST_CALL: dict[str, dict[str, Any]] = {}

# -- the trimmed host path (host_trim=1): the per-context host state of the eager halo calls.  ``_HOST_CACHE[id(ctx)]`` holds the
# context (pinned, so its id cannot be reused) and one entry per ``(kind, S, C, device, cu_seqlens_local_cpu)``: the
# validated context tables and tensors (``cu``, ``hist_row_off``, ...), the plan scalars, the record names, the loaded
# modules with their positional argument plans and grids.  Everything in an entry derives from host integers of the
# context and the shapes of the call; no tensor content is read.  At most HOST_CACHE_MAX contexts (the oldest dropped).
HOST_CACHE_MAX = 64
_HOST_CACHE: dict[int, tuple[Any, dict[tuple, dict[str, Any]]]] = {}
_FFI_DEVICES: dict[int, Any] = {}


def _host_cache_entry(ctx: Any, key: tuple) -> dict[str, Any]:
    """The (possibly empty, to be filled) host-cache entry of ``ctx`` under ``key``."""
    hit = _HOST_CACHE.get(id(ctx))
    if hit is None or hit[0] is not ctx:
        if len(_HOST_CACHE) >= HOST_CACHE_MAX:
            del _HOST_CACHE[next(iter(_HOST_CACHE))]
        hit = _HOST_CACHE[id(ctx)] = (ctx, {})
    entry = hit[1].get(key)
    if entry is None:
        entry = hit[1][key] = {}
    return entry


def host_cache_clear() -> int:
    """Drop the trimmed host path's per-context cache; returns the number of contexts dropped."""
    n = len(_HOST_CACHE)
    _HOST_CACHE.clear()
    return n


def _ffi_device(device_index: int) -> Any:
    dev = _FFI_DEVICES.get(int(device_index))
    if dev is None:
        dev = _FFI_DEVICES[int(device_index)] = tvm_ffi.device(
            f"cuda:{int(device_index)}"
        )
    return dev


def _raw_stream(device_index: int) -> int:
    get = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if get is not None:
        return int(get(int(device_index)))
    return int(torch.cuda.current_stream(int(device_index)).cuda_stream)


def _launch_plan(
    name: str, grid: tuple[int, int, int], device_index: int
) -> tuple[Any, str, tuple[tuple[str | None, int], ...]]:
    """``(module, ffi entry, plan)`` of record ``name`` for the trimmed launch: ``plan`` lists ``(None, grid value)``
    per grid argument and ``(logical name, 0)`` per kernel parameter, in ``run`` order (the same arguments
    :func:`run_module` / :func:`_launch_with_grid` build per call)."""
    record = _loader.MODULES[name]
    logical = record["logical"]
    grid_values = {
        "grid_x": int(grid[0]),
        "grid_y": int(grid[1]),
        "grid_z": int(grid[2]),
    }
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, grid_values[key]))
        else:
            if key not in logical:
                raise ValueError(
                    f"module {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], 0))
    module = load(name, _loader.device_arch(device_index))
    return module, record["ffi_entry"], tuple(plan)


def _launch_trim(
    info: tuple[Any, str, tuple[tuple[str | None, int], ...]],
    values: dict[str, Any],
    device_index: int,
) -> None:
    """Launch a cached :func:`_launch_plan` with the per-call ``values`` on the device's current torch stream (raw-stream
    FFI context: the same stream ``tvm_ffi.use_torch_stream`` would set, without building a torch.cuda.Stream)."""
    module, entry, plan = info
    args = []
    for logical_name, grid_value in plan:
        if logical_name is None:
            args.append(grid_value)
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"the launch needs logical argument {logical_name!r}"
                ) from None
    with tvm_ffi.use_raw_stream(_ffi_device(device_index), _raw_stream(device_index)):
        getattr(module, entry)(*args)


def _set_test_knobs(knobs: Optional[dict]) -> dict:
    """Set the module's private test knobs (``graph``: ``"off"`` shipped | ``"replay"``; ``host_trim``: ``1`` shipped =
    the trimmed host path of the eager call | ``0`` the previous host path); a knob
    the dict does not name keeps its shipped default; ``None`` restores the defaults.  Refused without
    ``CAKE_KCP_TEST_KNOBS=1`` in the environment.  Returns the active knobs."""
    if knobs is None:
        _TEST_KNOBS["graph"] = "off"
        _TEST_KNOBS["host_trim"] = HOST_TRIM_DEFAULT
        return dict(_TEST_KNOBS)
    if os.environ.get(TEST_KNOBS_ENV) != "1":
        raise ValueError(
            f"_set_test_knobs is a private test hook: it is accepted only with {TEST_KNOBS_ENV}=1 in the environment"
        )
    if not isinstance(knobs, dict):
        raise TypeError(f"_set_test_knobs expects a dict, got {type(knobs).__name__}")
    unknown = sorted(set(knobs) - {"graph", "host_trim"})
    if unknown:
        raise ValueError(
            f"_set_test_knobs: unknown knob(s) {unknown}; known: ['graph', 'host_trim']"
        )
    graph = str(knobs.get("graph", "off"))
    if graph not in GRAPH_MODES:
        raise ValueError(
            f"_set_test_knobs: graph must be one of {GRAPH_MODES}, got {graph!r}"
        )
    host_trim = knobs.get("host_trim", HOST_TRIM_DEFAULT)
    if isinstance(host_trim, bool) or host_trim not in HOST_TRIMS:
        raise ValueError(
            f"_set_test_knobs: host_trim must be one of {HOST_TRIMS}, got {host_trim!r}"
        )
    _TEST_KNOBS["graph"] = graph
    _TEST_KNOBS["host_trim"] = int(host_trim)
    return dict(_TEST_KNOBS)


def _graph_mode(graph: Optional[str]) -> str:
    """The effective graph mode of a call: the public keyword when given, else the module's test knob (shipped
    ``"off"``); validated against ``GRAPH_MODES``; a keyword contradicting an active knob is a ``ValueError``."""
    knob = _TEST_KNOBS["graph"]
    if graph is None:
        return knob
    mode = str(graph)
    if mode not in GRAPH_MODES:
        raise ValueError(f"graph must be one of {GRAPH_MODES}, got {graph!r}")
    if knob != "off" and mode != knob:
        raise ValueError(
            f"graph given twice with different values: keyword {mode!r}, _set_test_knobs {knob!r}"
        )
    return mode


class _GraphEntry:
    """One validity key: its graph (after the capture) and the static results."""

    __slots__ = ("key", "pins", "seen", "graph", "result", "disabled", "digest")

    def __init__(self, key: tuple, pins: tuple) -> None:
        self.key, self.pins = (
            key,
            pins,
        )  # the pins keep the keyed objects alive so their ids cannot be reused
        self.seen = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.result: Optional[tuple] = None
        self.disabled: Optional[str] = None
        self.digest = f"{key[0]}:{hash(key) & 0xFFFFFFFFFFFF:012x}"


_GRAPHS: dict[tuple, _GraphEntry] = {}


def _sig(t: Any) -> tuple:
    return (
        (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), t.device.index)
        if isinstance(t, torch.Tensor)
        else ("none",)
    )


def _graph_entry(
    kind: str, ctx: Any, tensors: tuple, mode: Optional[str] = None
) -> Optional[_GraphEntry]:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured (mode off, no tokens);
    ``mode`` = the effective graph mode (:func:`_graph_mode`), ``None`` = the knob."""
    if (
        (_TEST_KNOBS["graph"] if mode is None else mode) == "off"
        or int(getattr(ctx, "T_r", 0)) == 0
        or int(getattr(ctx, "n_frag", 0)) == 0
    ):
        return None
    key = (kind, id(ctx), int(ctx.world_size), tuple(_sig(t) for t in tensors))
    e = _GRAPHS.get(key)
    if e is None:
        while len(_GRAPHS) >= GRAPH_CACHE_MAX:
            del _GRAPHS[next(iter(_GRAPHS))]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx,))
    return e


def graph_cache_clear() -> int:
    """Drop every cached graph with its static buffers and private memory pool; returns the number of entries dropped.

    A graph that captured a NCCL collective (the halo ``all_to_all_single`` here) must be destroyed before the process group
    it used is destroyed (``ncclCommDestroy`` waits for graph-captured work; ``torch.distributed.destroy_process_group()``
    hangs while such a graph is alive).  Call this before destroying the process group when ``graph="replay"`` (keyword or test knob) was used, and
    whenever the operands of a key are gone for good to release its pool.  The knob and ``GRAPH_STATS`` are left as they are."""
    n = len(_GRAPHS)
    _GRAPHS.clear()
    return n


def _graph_run(
    kind: str, entry: Optional[_GraphEntry], body: Any, mode: Optional[str] = None
) -> tuple:
    """Run ``body()`` under the graph mode (keyword or knob): replay the key's graph, or capture this call (the key's second), or run eagerly
    (the key's first call, a key whose capture failed, or a call that is never captured).  Returns the results."""
    if entry is not None and entry.graph is not None:
        entry.graph.replay()
        GRAPH_STATS["replay"] += 1
        LAST_CALL[kind] = {
            "graph": "replay",
            "graph_event": "replay",
            "graph_key": entry.digest,
            "graph_fallback": None,
        }
        return entry.result
    if entry is not None and entry.seen > 0 and entry.disabled is None:
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                result = body()
        except Exception as exc:  # noqa: BLE001 - the failure is recorded; the call runs eagerly below
            entry.disabled = f"{type(exc).__name__}: {str(exc)[:200]}"
            GRAPH_STATS["fallback"] += 1
        else:
            g.replay()
            entry.graph, entry.result = g, result
            entry.seen += 1
            GRAPH_STATS["capture"] += 1
            LAST_CALL[kind] = {
                "graph": "replay",
                "graph_event": "capture",
                "graph_key": entry.digest,
                "graph_fallback": None,
            }
            return result
    result = body()
    GRAPH_STATS["eager"] += 1
    if entry is not None:
        entry.seen += 1
    LAST_CALL[kind] = {
        "graph": _TEST_KNOBS["graph"] if mode is None else mode,
        "graph_event": "eager",
        "graph_key": None if entry is None else entry.digest,
        "graph_fallback": None if entry is None else entry.disabled,
    }
    return result


def cake_causal_conv1d_forward_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP forward: ``y`` of the rank's fragments, and the history pool ``hist`` it used (saved for the backward).
    ``deterministic`` must be ``False`` (``ValueError`` otherwise); ``y`` and ``hist`` are the det KCP forward's bits.

    ``x_local`` BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous, the rank's fragments back to back in the order of
    ``ctx.cu_seqlens_local``; ``weight`` FP32 ``[C, 4]``; ``ctx`` a ``KcpContext`` (its halo plan ``ctx.halo``, the
    device tables ``ctx.halo_tables`` and the process group ``ctx.group``).  Every rank of the group calls this
    function once per layer, in the same order: the forward halo exchange is a collective (a rank without tokens
    joins it with empty splits and returns empty outputs).  ``hist`` is a fresh ``[n_hist_rows, C]`` BF16 tensor
    (``[1, C]`` when the rank receives no row).  ``graph`` (``None`` = the shipped ``GRAPH_DEFAULT`` ``"off"`` |
    ``"replay"``): keyed CUDA-graph capture / replay of this call (the graph-replay option; see the section comment
    above: pointer-stable operands, private pools, eager fallback, :func:`graph_cache_clear`).
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_forward_kcp")
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_forward(x_local, weight, ctx, out=out)
    entry = _graph_entry("fwd", ctx, (x_local, weight, out), mode)
    y, hist = _graph_run(
        "fwd", entry, lambda: _kcp_forward(x_local, weight, ctx, out=out), mode
    )
    return (y if out is not None else y.detach()), hist


def _kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    _hist: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_forward_kcp`.  ``_hist``:
    the history pool a recompute handle stored (checked by :func:`_handle_hist`): no gather, no exchange, no wait;
    the kernel launch is the same."""

    x = _require_tensor(x_local, "x_local")
    device = x.device
    # tables of a context prepared on another stream: order them on this call's stream, on every call (the trimmed
    # path's host cache is not per stream; after the first call on a stream this is one dict lookup)
    _wait_ready(ctx)
    if _TEST_KNOBS["host_trim"] == 1:
        return _kcp_forward_trim(x, weight, ctx, out=out, device=device, _hist=_hist)
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    cu, n_frag = _context(ctx, device=device, T_r=S)
    halo = ctx.halo
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    if _hist is None:
        hist_index = _require_tensor(
            ctx.halo_tables["hist_index"], "ctx.halo_tables['hist_index']"
        )
        send = (
            x_flat.index_select(0, hist_index)
            if hist_index.numel()
            else x_flat.new_empty((0, C))
        )
        hist, work = _exchange(
            send,
            recv_rows=int(halo.n_hist_rows),
            send_splits=tuple(halo.fwd_send_rows),
            recv_splits=tuple(halo.fwd_recv_rows),
            ctx=ctx,
        )
    else:
        hist, work = _hist, None
    if S == 0:
        if work is not None:
            work.wait()
        return y, hist
    hist_row_off = _table(
        ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    (name,) = route_modules(arch, KCP_FWD)
    device_index = _device_index(device)
    record = _loader.MODULES[name]
    scalars = plan(S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True)
    if scalars["n_ch_tiles"] > launch_grid(record, device_index=device_index)[0]:
        raise ValueError(
            f"C={C} spans more channel tiles than the forward's grid on this device holds"
        )
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "y": y.view(S, C),
        "hist": hist
        if hist.shape[0]
        else _pool(0, C, dtype=torch.bfloat16, device=device),
        "hist_row_off": hist_row_off,
        **_scalars(scalars),
    }
    if work is not None:
        work.wait()
    run_module(name, values, device_index=device_index)
    return y, hist


def _fwd_host(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The trimmed forward's operand checks, output and host-cache entry: ``(S, C, x_flat, w, y, entry)``.  The entry
    is cached per ``(ctx, S, C, device, cu_seqlens_local_cpu)``: the validated ``cu`` / ``hist_index`` /
    ``hist_row_off`` tables, the halo splits, the record, its plan scalars and grid, the loaded module with its
    argument plan."""

    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    e = _host_cache_entry(
        ctx, ("fwd", S, C, str(device), tuple(ctx.cu_seqlens_local_cpu))
    )
    if not e:
        arch = _device_arch(device)
        cu, n_frag = _context(ctx, device=device, T_r=S)
        halo = ctx.halo
        e.update(
            cu=cu,
            n_frag=n_frag,
            hist_index=_require_tensor(
                ctx.halo_tables["hist_index"], "ctx.halo_tables['hist_index']"
            ),
            recv_rows=int(halo.n_hist_rows),
            send_splits=tuple(halo.fwd_send_rows),
            recv_splits=tuple(halo.fwd_recv_rows),
        )
        if S != 0:
            e["hist_row_off"] = _table(
                ctx,
                "hist_row_off",
                shape=(n_frag, HALO),
                dtype=torch.int32,
                device=device,
            )
            (name,) = route_modules(arch, KCP_FWD)
            device_index = _device_index(device)
            record = _loader.MODULES[name]
            scalars = plan(
                S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
            )
            grid = launch_grid(record, device_index=device_index)
            if scalars["n_ch_tiles"] > grid[0]:
                raise ValueError(
                    f"C={C} spans more channel tiles than the forward's grid on this device holds"
                )
            e["scalars"] = _scalars(scalars)
            e["device_index"] = device_index
            e["launch"] = _launch_plan(name, grid, device_index)
    return S, C, x_flat, w, y, e


def _fwd_gather(x_flat: torch.Tensor, C: int, e: dict[str, Any]) -> torch.Tensor:
    """The forward's send rows: the context's ``hist_index`` rows of ``x_flat`` in destination-rank order."""
    hist_index = e["hist_index"]
    return (
        x_flat.index_select(0, hist_index)
        if hist_index.numel()
        else x_flat.new_empty((0, C))
    )


def _fwd_launch(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    hist: torch.Tensor,
    work: Optional[Any],
    *,
    S: int,
    C: int,
    device: torch.device,
) -> None:
    """Wait ``work`` (when given) and launch the forward kernel of a call with ``S > 0`` on ``hist``."""
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": e["cu"],
        "y": y.view(S, C),
        "hist": hist
        if hist.shape[0]
        else _pool(0, C, dtype=torch.bfloat16, device=device),
        "hist_row_off": e["hist_row_off"],
        **e["scalars"],
    }
    if work is not None:
        work.wait()
    _launch_trim(e["launch"], values, e["device_index"])


def _kcp_forward_trim(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor],
    device: torch.device,
    _hist: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trimmed host path: :func:`_kcp_forward` with the context-derived host work cached per ``(ctx, S, C, device,
    cu_seqlens_local_cpu)`` (:func:`_fwd_host`).  The operand checks, the gather, the exchange, the wait and the
    kernel launch are the untrimmed call's, in the same order on the same stream."""

    S, C, x_flat, w, y, e = _fwd_host(x, weight, ctx, out=out, device=device)
    if _hist is None:
        hist, work = _exchange(
            _fwd_gather(x_flat, C, e),
            recv_rows=e["recv_rows"],
            send_splits=e["send_splits"],
            recv_splits=e["recv_splits"],
            ctx=ctx,
        )
    else:
        hist, work = _hist, None
    if S == 0:
        if work is not None:
            work.wait()
        return y, hist
    _fwd_launch(e, x_flat, w, y, hist, work, S=S, C=C, device=device)
    return y, hist


def cake_causal_conv1d_backward_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP backward: ``dx`` of the rank's fragments (the halo contributions of the successor fragments folded in
    before the one BF16 rounding; the det KCP backward's bits) and the rank's FP32 ``dweight`` partial (summed across
    CTAs in an unordered order: not bitwise reproducible run to run; its sum over the ranks is the framework's).

    ``hist`` is the forward's pool.  The head program writes the derivative rows of every continuing fragment's first
    tokens into the backward exchange's send layout, the exchange delivers them to the predecessors as ``dpre_tail``,
    and the main program folds them in one ordinary launch.  Every rank calls this function once per layer, in the
    same order.  ``graph`` as in the forward (the backward's key holds the forward's ``hist`` pointer, so its capture
    follows the forward's by one call).  ``deterministic`` must be ``False`` (``ValueError`` otherwise); under
    ``torch.use_deterministic_algorithms(True)`` the call raises (or warns with ``warn_only=True``) through
    ``torch._C._alert_not_deterministic`` before it launches or exchanges anything.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_backward_kcp")
    _alert_nondeterministic("cake_causal_conv1d_backward_kcp")
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_backward(x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight)
    entry = _graph_entry("bwd", ctx, (x_local, weight, dy, hist, dx, dweight), mode)
    dx_out, dw_out = _graph_run(
        "bwd",
        entry,
        lambda: _kcp_backward(x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight),
        mode,
    )
    return (dx_out if dx is not None else dx_out.detach()), (
        dw_out if dweight is not None else dw_out.detach()
    )


def _kcp_backward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_backward_kcp`."""

    x = _require_tensor(x_local, "x_local")
    device = x.device
    # as in the forward: every call, before the host-cache lookup of the trimmed path
    _wait_ready(ctx)
    if _TEST_KNOBS["host_trim"] == 1:
        return _kcp_backward_trim(
            x, weight, ctx, dy, hist, dx=dx, dweight=dweight, device=device
        )
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    cu, n_frag = _context(ctx, device=device, T_r=S)
    halo = ctx.halo
    hist_t = _require_tensor(hist, "hist")
    if (
        hist_t.ndim != 2
        or int(hist_t.shape[1]) != C
        or hist_t.dtype != torch.bfloat16
        or not hist_t.is_contiguous()
    ):
        raise ValueError(
            f"hist must be the forward's contiguous BF16 [n_hist_rows, {C}] pool, got {hist_t.dtype} {tuple(hist_t.shape)}"
        )
    if hist_t.shape[0] == 0:
        # A valid pointer for the kernels; no table entry names a row of it.
        hist_t = _pool(0, C, dtype=torch.bfloat16, device=device)
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _output(
        dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
    )
    device_index = _device_index(device)
    n_head = int(halo.n_head_rows)
    dpre_head = _pool(n_head, C, dtype=torch.float32, device=device)
    if S > 0 and n_head > 0:
        hist_row_off = _table(
            ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
        )
        head_row_off = _table(
            ctx,
            "head_row_off",
            shape=(n_frag, HALO, HEAD_SLOTS),
            dtype=torch.int32,
            device=device,
        )
        (head_name,) = route_modules(arch, KCP_BWD_HEAD)
        geometry = _loader.MODULES[head_name]["grid_rule"]["geometry"]
        channels_per_cta = int(geometry["threads"]) * int(geometry["vec"])
        grid = (max(1, n_frag * _ceil_div(C, channels_per_cta)), 1, 1)
        _launch_with_grid(
            head_name,
            {
                "x": x_flat,
                "w": w,
                "cu": cu,
                "dy": dy_flat,
                "hist": hist_t,
                "hist_row_off": hist_row_off,
                "dpre_head": dpre_head,
                "head_row_off": head_row_off,
                "S": S,
                "C": C,
                "num_docs": n_frag,
            },
            grid=grid,
            device_index=device_index,
        )
    dpre_tail, work = _exchange(
        dpre_head[:n_head],
        recv_rows=int(halo.n_tail_rows),
        send_splits=tuple(halo.bwd_send_rows),
        recv_splits=tuple(halo.bwd_recv_rows),
        ctx=ctx,
    )
    if S == 0:
        if work is not None:
            work.wait()
        dw_out.zero_()
        return dx_out, dw_out
    hist_row_off = _table(
        ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    tail_row_off = _table(
        ctx, "tail_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    names = selected_modules(arch, KCP_BWD, C=C, S=S)
    record = _loader.MODULES[names[0]]
    scalars = plan(S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True)
    grid_x = launch_grid(record, device_index=device_index)[0]
    if scalars["n_ch_tiles"] > grid_x:
        raise ValueError(
            f"C={C} spans more channel tiles than the backward's grid on this device holds"
        )
    acc, acc_work = _backward_accumulators(
        None,
        accumulator_layout(record, C, scalars["n_ch_tiles"]),
        device=device,
        device_index=device_index,
        entry="cake_causal_conv1d_backward_kcp",
    )
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "acc": acc,
        "work": acc_work,
        "hist": hist_t,
        "hist_row_off": hist_row_off,
        "dpre_tail": dpre_tail
        if dpre_tail.shape[0]
        else _pool(0, C, dtype=torch.float32, device=device),
        "tail_row_off": tail_row_off,
        **_scalars(scalars),
    }
    if work is not None:
        work.wait()
    for name in names:
        run_module(name, values, device_index=device_index)
    return dx_out, dw_out


def _bwd_host(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor],
    dweight: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[Any, ...]:
    """The trimmed backward's operand checks, outputs and host-cache entry: ``(S, C, x_flat, w, dy_flat, hist_t,
    dx_out, dw_out, entry)``.  The entry is cached per ``(ctx, S, C, device, cu_seqlens_local_cpu)``: the tables, the
    head grid, the backward records, plan scalars, the accumulator layout, the loaded modules with their argument
    plans."""

    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    hist_t = _require_tensor(hist, "hist")
    if (
        hist_t.ndim != 2
        or int(hist_t.shape[1]) != C
        or hist_t.dtype != torch.bfloat16
        or not hist_t.is_contiguous()
    ):
        raise ValueError(
            f"hist must be the forward's contiguous BF16 [n_hist_rows, {C}] pool, got {hist_t.dtype} {tuple(hist_t.shape)}"
        )
    if hist_t.shape[0] == 0:
        hist_t = _pool(0, C, dtype=torch.bfloat16, device=device)
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _output(
        dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
    )
    e = _host_cache_entry(
        ctx, ("bwd", S, C, str(device), tuple(ctx.cu_seqlens_local_cpu))
    )
    if not e:
        arch = _device_arch(device)
        cu, n_frag = _context(ctx, device=device, T_r=S)
        halo = ctx.halo
        device_index = _device_index(device)
        n_head = int(halo.n_head_rows)
        e.update(
            cu=cu,
            n_frag=n_frag,
            device_index=device_index,
            n_head=n_head,
            tail_rows=int(halo.n_tail_rows),
            send_splits=tuple(halo.bwd_send_rows),
            recv_splits=tuple(halo.bwd_recv_rows),
            head=None,
        )
        if S > 0 and n_head > 0:
            hist_row_off = _table(
                ctx,
                "hist_row_off",
                shape=(n_frag, HALO),
                dtype=torch.int32,
                device=device,
            )
            head_row_off = _table(
                ctx,
                "head_row_off",
                shape=(n_frag, HALO, HEAD_SLOTS),
                dtype=torch.int32,
                device=device,
            )
            (head_name,) = route_modules(arch, KCP_BWD_HEAD)
            geometry = _loader.MODULES[head_name]["grid_rule"]["geometry"]
            channels_per_cta = int(geometry["threads"]) * int(geometry["vec"])
            grid = (max(1, n_frag * _ceil_div(C, channels_per_cta)), 1, 1)
            e["head"] = (
                _launch_plan(head_name, grid, device_index),
                hist_row_off,
                head_row_off,
            )
        if S != 0:
            e["hist_row_off"] = _table(
                ctx,
                "hist_row_off",
                shape=(n_frag, HALO),
                dtype=torch.int32,
                device=device,
            )
            e["tail_row_off"] = _table(
                ctx,
                "tail_row_off",
                shape=(n_frag, HALO),
                dtype=torch.int32,
                device=device,
            )
            names = selected_modules(arch, KCP_BWD, C=C, S=S)
            record = _loader.MODULES[names[0]]
            scalars = plan(
                S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
            )
            grid_x = launch_grid(record, device_index=device_index)[0]
            if scalars["n_ch_tiles"] > grid_x:
                raise ValueError(
                    f"C={C} spans more channel tiles than the backward's grid on this device holds"
                )
            e["layout"] = accumulator_layout(record, C, scalars["n_ch_tiles"])
            e["scalars"] = _scalars(scalars)
            e["launches"] = tuple(
                _launch_plan(
                    name,
                    launch_grid(_loader.MODULES[name], device_index=device_index),
                    device_index,
                )
                for name in names
            )
    return S, C, x_flat, w, dy_flat, hist_t, dx_out, dw_out, e


def _bwd_head(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    *,
    S: int,
    C: int,
    device: torch.device,
) -> torch.Tensor:
    """The head program of the backward (when the rank has head rows): the fresh FP32 ``dpre_head`` pool (at least one
    row) whose first ``n_head`` rows are the backward exchange's send rows."""
    dpre_head = _pool(e["n_head"], C, dtype=torch.float32, device=device)
    if e["head"] is not None:
        info, hist_row_off, head_row_off = e["head"]
        _launch_trim(
            info,
            {
                "x": x_flat,
                "w": w,
                "cu": e["cu"],
                "dy": dy_flat,
                "hist": hist_t,
                "hist_row_off": hist_row_off,
                "dpre_head": dpre_head,
                "head_row_off": head_row_off,
                "S": S,
                "C": C,
                "num_docs": e["n_frag"],
            },
            e["device_index"],
        )
    return dpre_head


def _bwd_main(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    dx_out: torch.Tensor,
    dw_out: torch.Tensor,
    dpre_tail: torch.Tensor,
    work: Optional[Any],
    *,
    S: int,
    C: int,
    device: torch.device,
) -> None:
    """Wait ``work`` (when given) and launch the main backward program of a call with ``S > 0`` (``dx`` with the
    received ``dpre_tail`` rows folded in, and the rank's FP32 ``dweight`` partial, summed across CTAs in the
    zeroed accumulators of the device's cached backward workspace)."""
    acc, acc_work = _backward_accumulators(
        None,
        e["layout"],
        device=device,
        device_index=e["device_index"],
        entry="cake_causal_conv1d_backward_kcp",
    )
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": e["cu"],
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "acc": acc,
        "work": acc_work,
        "hist": hist_t,
        "hist_row_off": e["hist_row_off"],
        "dpre_tail": dpre_tail
        if dpre_tail.shape[0]
        else _pool(0, C, dtype=torch.float32, device=device),
        "tail_row_off": e["tail_row_off"],
        **e["scalars"],
    }
    if work is not None:
        work.wait()
    for info in e["launches"]:
        _launch_trim(info, values, e["device_index"])


def _kcp_backward_trim(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor],
    dweight: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trimmed host path: :func:`_kcp_backward` with the context-derived host work cached (:func:`_bwd_host`); the
    operand checks, allocations, exchange, wait and launches are the untrimmed call's."""

    S, C, x_flat, w, dy_flat, hist_t, dx_out, dw_out, e = _bwd_host(
        x, weight, ctx, dy, hist, dx=dx, dweight=dweight, device=device
    )
    dpre_head = _bwd_head(e, x_flat, w, dy_flat, hist_t, S=S, C=C, device=device)
    dpre_tail, work = _exchange(
        dpre_head[: e["n_head"]],
        recv_rows=e["tail_rows"],
        send_splits=e["send_splits"],
        recv_splits=e["recv_splits"],
        ctx=ctx,
    )
    if S == 0:
        if work is not None:
            work.wait()
        dw_out.zero_()
        return dx_out, dw_out
    _bwd_main(
        e,
        x_flat,
        w,
        dy_flat,
        hist_t,
        dx_out,
        dw_out,
        dpre_tail,
        work,
        S=S,
        C=C,
        device=device,
    )
    return dx_out, dw_out


class CakeCausalConv1dFunctionKcp(torch.autograd.Function):
    """Autograd wrapper: ``apply(x_local, weight, ctx, graph, handle, deterministic)`` (``deterministic`` must be
    ``False``); the forward saves ``(x_local, weight)`` and the history pool ``hist`` (no second exchange in the
    backward); the backward runs the head program, the backward halo exchange and the main backward program.
    ``handle``: a recompute handle (see :class:`RecomputeHandleKcp`): the original forward stores ``hist`` in it, the
    recompute binds it and exchanges nothing; the saved set is the same three tensors either way.  A handle is
    refused with graph replay (keyword or knob) and inside a CUDA-graph capture."""

    @staticmethod
    def forward(
        fn_ctx, x_local, weight, ctx, graph=None, handle=None, deterministic=True
    ):
        _require_nondeterministic(deterministic, "CakeCausalConv1dFunctionKcp")
        if handle is None:
            y, hist = cake_causal_conv1d_forward_kcp(
                x_local, weight, ctx, graph=graph, deterministic=False
            )
        else:
            _handle_refusals(graph, handle)
            entry = _handle_lookup(handle)
            if entry is None:
                y, hist = _kcp_forward(x_local, weight, ctx)
                handle.entries[HANDLE_KEY] = {
                    "hist": hist,
                    "shape": tuple(hist.shape),
                    "dtype": hist.dtype,
                    "device": hist.device,
                    "ctx": ctx,
                    "key": (int(x_local.shape[-2]), int(x_local.shape[-1])),
                }
                LAST_CALL["recompute_handle"] = {"fwd": "capture"}
            else:
                hist = _handle_hist(entry, x_local, ctx)
                y, hist = _kcp_forward(x_local, weight, ctx, _hist=hist)
                LAST_CALL["recompute_handle"] = {"fwd": "replay"}
        fn_ctx.save_for_backward(x_local, weight, hist)
        fn_ctx.kcp = ctx
        fn_ctx.graph = graph  # the backward runs under the same graph mode
        return y

    @staticmethod
    def backward(fn_ctx, grad_y):
        x_local, weight, hist = fn_ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x_local)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_backward_kcp(
            x_local,
            weight,
            fn_ctx.kcp,
            dy,
            hist,
            graph=fn_ctx.graph,
            deterministic=False,
        )
        return dx, dweight, None, None, None, None


def cake_causal_conv1d_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    deterministic: bool = True,
) -> torch.Tensor:
    """Autograd-enabled KCP packed width-4 depthwise causal convolution + SiLU of the rank's fragments, with the
    non-deterministic main backward program (``deterministic=False`` required).

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous activations of the rank's fragments, back to back in the
        order of ``ctx.cu_seqlens_local``.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current token.
    ctx : KcpContext
        The rank's context (``flashinfer.experimental.cake_kda_train.build_kcp_context``): fragment table, halo
        plan, process group.  Every rank of the group calls this function once per layer, in the same order.
    graph : str, optional
        ``None`` (the shipped default ``GRAPH_DEFAULT`` = ``"off"``, the eager path) or ``"replay"``: keyed CUDA-graph
        capture / replay of this call.  The first call of a validity key (this
        context, its world size, and the shape / stride / dtype / data pointer of ``x_local`` and ``weight``) runs
        eagerly, the second is captured, every later one replays the graph (bitwise the eager result).  Requirements:
        pointer-stable operands and the same ``KcpContext`` per layer, outputs consumed before the next call of the
        same layer (a replay overwrites the graph's static tensors); any other key runs eagerly (``GRAPH_STATS``,
        ``LAST_CALL``).  Each cached key pins a private memory pool (at most ``GRAPH_CACHE_MAX`` keys); call
        :func:`graph_cache_clear` before ``torch.distributed.destroy_process_group()``.
    recompute_handle : RecomputeHandleKcp, optional
        ``None`` (default): unchanged.  A handle (any object with an ``entries`` dict): the checkpoint recompute of
        this call exchanges nothing -- the original forward stores its history pool under ``"conv"`` and torch's
        recompute inside the backward binds it (bitwise the exchanging recompute); the backward still exchanges.
        One handle per convolution call and micro-step, created outside the outermost checkpoint region; a forward
        outside the backward that finds an entry raises.  Refused with ``graph="replay"`` (keyword or knob) and
        inside a CUDA-graph capture.  Retains ``hist`` (``recompute_handle.nbytes()``) while the handle lives.
        The forward halo exchange is a collective: every rank of ``ctx.group`` passes a handle for the same call or
        none, created at the same checkpoint nesting level.  A mismatch pairs one rank's recompute exchange with
        another rank's backward exchange and hangs or corrupts data; no check is made.  Never share one handle
        between two convolution calls (also not in nested regions), and never ``clear()`` or reuse a handle while a
        backward that needs it is pending.
    deterministic : bool
        Must be ``False`` (``ValueError`` otherwise): the numerical mode of this package.  Under
        ``torch.use_deterministic_algorithms(True)`` the backward raises (or warns with ``warn_only=True``).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape, with the three document tokens before every continuing fragment as
        history (zero history at a document's start).  The backward returns BF16 ``dx`` (the successor fragments'
        contributions folded in before the one rounding; the det KCP backward's bits) and the rank's FP32
        ``dweight`` partial (unordered cross-CTA sum).
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_kcp")
    _graph_mode(graph)  # validate on the caller's frame
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    # Always six inputs: the backward returns six gradients.
    return CakeCausalConv1dFunctionKcp.apply(
        x_local, weight, ctx, graph, recompute_handle, False
    )


# ----------------------------------------------------------------------------------------------------------------
# Several convolutions of one layer with one halo exchange per direction (FlashInfer issue #7815): the q / k / v
# convolutions of a layer share the context, so their history rows (forward) and head rows (backward) travel in ONE
# ``all_to_all_single`` instead of one per tensor.  The send rows of the tensors are packed side by side
# (``[n_rows, sum C_i]``, the context's splits unchanged), and after the wait each tensor's rows are copied out into
# a fresh contiguous pool -- the forward's ``hist`` ``[n_hist_rows, C_i]`` (the saved tensor) and the backward's
# ``dpre_tail`` ``[n_tail_rows, C_i]`` -- before that tensor's kernels launch with the single call's arguments.  The
# exchange and the unpacking are copies, and each tensor keeps its own kernels and accumulator launch, so every output
# and ``dx`` is bitwise that of one single call per tensor (``dweight`` is each tensor's own unordered partial).
# One rank (no collective): the per-tensor send rows are the pools, exactly as in the single call.


def _multi_operands(
    xs: Any, weights: Any, outs: Any, name: str = "xs"
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[Optional[torch.Tensor]]]:
    """``xs`` / ``weights`` (/ ``outs``) as lists of one length >= 1, on one device."""
    if isinstance(xs, torch.Tensor) or isinstance(weights, torch.Tensor):
        raise TypeError(
            f"{name} and weights must be sequences of tensors (one per convolution), not a tensor"
        )
    xs, weights = list(xs), list(weights)
    if not xs or len(xs) != len(weights):
        raise ValueError(
            f"{name} and weights must hold the same number (>= 1) of tensors, got {len(xs)} and {len(weights)}"
        )
    if outs is None:
        outs = [None] * len(xs)
    else:
        outs = list(outs)
        if len(outs) != len(xs):
            raise ValueError(
                f"outs must hold one tensor per convolution ({len(xs)}), got {len(outs)}"
            )
    x0 = _require_tensor(xs[0], f"{name}[0]")
    for i, x in enumerate(xs):
        if _require_tensor(x, f"{name}[{i}]").device != x0.device:
            raise ValueError(
                f"{name}[{i}] is on {x.device}, {name}[0] on {x0.device}: one device"
            )
    return xs, weights, outs


def _unpack(recv: torch.Tensor, widths: list[int]) -> list[torch.Tensor]:
    """The tensors' contiguous column blocks of a packed receive pool (one copy each)."""
    blocks, off = [], 0
    for C in widths:
        blocks.append(recv[:, off : off + C].contiguous())
        off += C
    return blocks


def cake_causal_conv1d_multi_forward_kcp(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    outs: Optional[Any] = None,
    graph: Optional[str] = None,
    deterministic: bool = True,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP forward of several convolutions of one layer with one halo exchange: ``(ys, hists)``, bitwise
    ``cake_causal_conv1d_forward_kcp(xs[i], weights[i], ctx, deterministic=False)`` per tensor
    (``deterministic`` must be ``False``).

    ``xs`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous tensors of the rank's fragments (the channel counts
    may differ); ``weights`` FP32 ``[C_i, 4]``; ``outs`` optional preallocated outputs.  Every rank of ``ctx.group``
    calls this function once per layer, in the same order, with the same number of tensors and the same channel
    counts: the one forward exchange is a collective.  ``hists[i]`` is a fresh contiguous ``[n_hist_rows, C_i]``
    BF16 pool (the saved tensor of tensor ``i``).  ``graph`` as in :func:`cake_causal_conv1d_forward_kcp` (the
    validity key holds every operand).
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_multi_forward_kcp")
    xs, weights, outs = _multi_operands(xs, weights, outs)
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_forward_multi(xs, weights, ctx, outs=outs)
    entry = _graph_entry("fwd_multi", ctx, (*xs, *weights, *outs), mode)
    ys, hists = _graph_run(
        "fwd_multi",
        entry,
        lambda: _kcp_forward_multi(xs, weights, ctx, outs=outs),
        mode,
    )
    ys = tuple(
        y if o is not None else y.detach() for y, o in zip(ys, outs, strict=True)
    )
    return ys, hists


def _kcp_forward_multi(
    xs: list[torch.Tensor],
    weights: list[torch.Tensor],
    ctx: Any,
    *,
    outs: list[Optional[torch.Tensor]],
    _hists: Optional[tuple[torch.Tensor, ...]] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The multi forward's host flow (eagerly or inside a capture): per-tensor operand checks, the packed gather, one
    exchange, one wait, the per-tensor unpack copies and the single call's forward kernel per tensor.  ``_hists``:
    the pools a recompute handle stored (checked by :func:`_handle_hists`): no gather, no exchange, no wait."""

    device = _require_tensor(xs[0], "xs[0]").device
    _wait_ready(ctx)
    calls = [
        _fwd_host(x, w, ctx, out=o, device=device)
        for x, w, o in zip(xs, weights, outs, strict=True)
    ]
    S = calls[0][0]
    widths = [c[1] for c in calls]
    work = None
    if _hists is not None:
        hists = list(_hists)
    elif int(ctx.world_size) == 1:
        # one rank: each tensor's send rows are its pool (the single call's identity exchange)
        hists = [_fwd_gather(c[2], c[1], c[5]) for c in calls]
    else:
        e0 = calls[0][5]
        sends = [_fwd_gather(c[2], c[1], c[5]) for c in calls]
        send = torch.cat(sends, dim=1) if len(sends) > 1 else sends[0]
        recv, work = _exchange(
            send,
            recv_rows=e0["recv_rows"],
            send_splits=e0["send_splits"],
            recv_splits=e0["recv_splits"],
            ctx=ctx,
        )
        if work is not None:
            work.wait()
        hists = _unpack(recv, widths) if len(calls) > 1 else [recv]
    ys = tuple(c[4] for c in calls)
    if S == 0:
        return ys, tuple(hists)
    for (S_i, C, x_flat, w, y, e), hist in zip(calls, hists, strict=True):
        _fwd_launch(e, x_flat, w, y, hist, None, S=S_i, C=C, device=device)
    return ys, tuple(hists)


def cake_causal_conv1d_multi_backward_kcp(
    xs: Any,
    weights: Any,
    ctx: Any,
    dys: Any,
    hists: Any,
    *,
    dxs: Optional[Any] = None,
    dweights: Optional[Any] = None,
    graph: Optional[str] = None,
    deterministic: bool = True,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP backward of several convolutions of one layer with one halo exchange: ``(dxs, dweights)``,
    ``cake_causal_conv1d_backward_kcp(xs[i], weights[i], ctx, dys[i], hists[i], deterministic=False)`` per tensor
    (``dxs`` bitwise; each ``dweights[i]`` an unordered FP32 partial of the same budget).

    The head programs of every tensor run first, their send rows travel packed in one exchange, and after the wait
    each tensor's received rows are copied out and folded by that tensor's main backward program (one ordinary
    launch per tensor on the stream's accumulator workspace).  ``hists`` are the forward's pools.  Every rank calls
    this function once per layer, in the same order.  ``graph`` as in :func:`cake_causal_conv1d_backward_kcp`.
    ``deterministic`` must be ``False``; under torch's deterministic mode the call alerts like the single backward.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_multi_backward_kcp")
    _alert_nondeterministic("cake_causal_conv1d_multi_backward_kcp")
    xs, weights, _ = _multi_operands(xs, weights, None)
    n = len(xs)
    dys, hists = list(dys), list(hists)
    dxs = [None] * n if dxs is None else list(dxs)
    dweights = [None] * n if dweights is None else list(dweights)
    if not (len(dys) == len(hists) == len(dxs) == len(dweights) == n):
        raise ValueError(
            f"dys, hists, dxs and dweights must hold one entry per convolution ({n})"
        )
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_backward_multi(xs, weights, ctx, dys, hists, dxs, dweights)
    entry = _graph_entry(
        "bwd_multi", ctx, (*xs, *weights, *dys, *hists, *dxs, *dweights), mode
    )
    dx_out, dw_out = _graph_run(
        "bwd_multi",
        entry,
        lambda: _kcp_backward_multi(xs, weights, ctx, dys, hists, dxs, dweights),
        mode,
    )
    return (
        tuple(
            d if o is not None else d.detach() for d, o in zip(dx_out, dxs, strict=True)
        ),
        tuple(
            d if o is not None else d.detach()
            for d, o in zip(dw_out, dweights, strict=True)
        ),
    )


def _kcp_backward_multi(
    xs: list[torch.Tensor],
    weights: list[torch.Tensor],
    ctx: Any,
    dys: list[torch.Tensor],
    hists: list[torch.Tensor],
    dxs: list[Optional[torch.Tensor]],
    dweights: list[Optional[torch.Tensor]],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The multi backward's host flow (eagerly or inside a capture): see
    :func:`cake_causal_conv1d_multi_backward_kcp`."""

    device = _require_tensor(xs[0], "xs[0]").device
    _wait_ready(ctx)
    calls = [
        _bwd_host(x, w, ctx, dy, h, dx=dx, dweight=dw, device=device)
        for x, w, dy, h, dx, dw in zip(
            xs, weights, dys, hists, dxs, dweights, strict=True
        )
    ]
    heads = [
        _bwd_head(c[8], c[2], c[3], c[4], c[5], S=c[0], C=c[1], device=device)
        for c in calls
    ]
    e0 = calls[0][8]
    n_head = e0["n_head"]
    work = None
    if int(ctx.world_size) == 1:
        tails = [h[:n_head] for h in heads]
    else:
        send = (
            torch.cat([h[:n_head] for h in heads], dim=1)
            if len(heads) > 1
            else heads[0][:n_head]
        )
        recv, work = _exchange(
            send,
            recv_rows=e0["tail_rows"],
            send_splits=e0["send_splits"],
            recv_splits=e0["recv_splits"],
            ctx=ctx,
        )
        if work is not None:
            work.wait()
        tails = _unpack(recv, [c[1] for c in calls]) if len(calls) > 1 else [recv]
    dx_out = tuple(c[6] for c in calls)
    dw_out = tuple(c[7] for c in calls)
    if calls[0][0] == 0:
        for dw in dw_out:
            dw.zero_()
        return dx_out, dw_out
    for (S, C, x_flat, w, dy_flat, hist_t, dx, dw, e), tail in zip(
        calls, tails, strict=True
    ):
        _bwd_main(
            e, x_flat, w, dy_flat, hist_t, dx, dw, tail, None, S=S, C=C, device=device
        )
    return dx_out, dw_out


def _handle_hists(
    entry: dict[str, Any], xs: list[torch.Tensor], ctx: Any
) -> tuple[torch.Tensor, ...]:
    """The stored pools of a multi call's ``"conv"`` entry, each checked like :func:`_handle_hist` against its tensor
    of this recompute (an entry of a single call, or of another tensor count, raises)."""
    pools, keys, shapes = entry.get("hists"), entry.get("keys"), entry.get("shapes")
    n = len(xs)
    if not all(isinstance(t, tuple) and len(t) == n for t in (pools, keys, shapes)):
        raise ValueError(
            f"recompute_handle: the 'conv' entry holds no {n}-tensor multi-call pools (an entry of "
            f"cake_causal_conv1d_kcp or of another tensor count); one handle serves one call"
        )
    if entry.get("ctx") is not ctx:
        raise ValueError(
            "recompute_handle: the 'conv' entry belongs to another KcpContext"
        )
    return tuple(
        _handle_hist(
            {
                "hist": hist,
                "ctx": ctx,
                "key": key,
                "shape": shape,
                "dtype": entry.get("dtype"),
                "device": entry.get("device"),
            },
            x,
            ctx,
        )
        for hist, x, key, shape in zip(pools, xs, keys, shapes, strict=True)
    )


class CakeCausalConv1dMultiFunctionKcp(torch.autograd.Function):
    """Autograd wrapper of several convolutions with one halo exchange per direction: ``apply(ctx, graph, handle,
    deterministic, n, *xs, *weights)`` (``deterministic`` must be ``False``) returns the ``n`` outputs; the forward saves ``(*xs, *weights, *hists)``; the backward runs
    :func:`cake_causal_conv1d_multi_backward_kcp`.  ``handle``: one recompute handle for the call, whose ``"conv"``
    entry holds the ``n`` pools (the saved set's own tensors): the recompute binds them and exchanges nothing.
    Refused with graph replay (keyword or knob) and inside a CUDA-graph capture, like the single call."""

    @staticmethod
    def forward(fn_ctx, ctx, graph, handle, deterministic, n, *tensors):
        _require_nondeterministic(deterministic, "CakeCausalConv1dMultiFunctionKcp")
        xs, weights = list(tensors[:n]), list(tensors[n:])
        if handle is None:
            ys, hists = cake_causal_conv1d_multi_forward_kcp(
                xs, weights, ctx, graph=graph, deterministic=False
            )
        else:
            _handle_refusals(graph, handle)
            entry = _handle_lookup(handle)
            outs = [None] * n
            if entry is None:
                ys, hists = _kcp_forward_multi(xs, weights, ctx, outs=outs)
                handle.entries[HANDLE_KEY] = {
                    "hists": hists,
                    "shapes": tuple(tuple(h.shape) for h in hists),
                    "dtype": torch.bfloat16,
                    "device": hists[0].device,
                    "ctx": ctx,
                    "keys": tuple((int(x.shape[-2]), int(x.shape[-1])) for x in xs),
                }
                LAST_CALL["recompute_handle"] = {"fwd": "capture", "n": n}
            else:
                hists = _handle_hists(entry, xs, ctx)
                ys, hists = _kcp_forward_multi(
                    xs, weights, ctx, outs=outs, _hists=hists
                )
                LAST_CALL["recompute_handle"] = {"fwd": "replay", "n": n}
        fn_ctx.save_for_backward(*xs, *weights, *hists)
        fn_ctx.kcp = ctx
        fn_ctx.graph = graph
        fn_ctx.n = n
        return tuple(ys)

    @staticmethod
    def backward(fn_ctx, *grad_ys):
        n = fn_ctx.n
        saved = fn_ctx.saved_tensors
        xs, weights, hists = saved[:n], saved[n : 2 * n], saved[2 * n :]
        dys = [
            (torch.zeros_like(x) if g is None else g).to(torch.bfloat16).contiguous()
            for g, x in zip(grad_ys, xs, strict=True)
        ]
        dxs, dweights = cake_causal_conv1d_multi_backward_kcp(
            xs, weights, fn_ctx.kcp, dys, hists, graph=fn_ctx.graph, deterministic=False
        )
        return (None, None, None, None, None, *dxs, *dweights)


def cake_causal_conv1d_multi_kcp(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    deterministic: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled KCP convolutions of several tensors of one layer (for example its q, k and v) with one halo
    exchange per direction (``deterministic=False`` required).

    ``tuple(cake_causal_conv1d_kcp(x, w, ctx, deterministic=False) for x, w in zip(xs, weights))`` -- outputs and
    ``dx`` bitwise, each ``dweight`` that tensor's own unordered FP32 partial -- with one collective in the forward
    and one in the backward instead of one per tensor: fewer global synchronization points per layer.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous activations of the rank's fragments, in the
        order of ``ctx.cu_seqlens_local``; the channel counts may differ.
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : KcpContext
        The rank's context, as in :func:`cake_causal_conv1d_kcp`.  Every rank of the group calls this function once
        per layer, in the same order, with the same tensor count and channel counts.  The ranks are not compared
        (that would cost a collective): a different count, channel count or order hangs the exchange or silently
        corrupts the history rows, as a mismatch of the single entry's per-rank contract does.
    graph : str, optional
        As in :func:`cake_causal_conv1d_kcp` (keyed CUDA-graph capture / replay; the key holds every operand).
    recompute_handle : RecomputeHandleKcp, optional
        ``None`` (default): unchanged.  One handle for the whole call: the original forward stores the ``n`` history
        pools under ``"conv"``, torch's recompute inside the backward binds them and exchanges nothing; the backward
        still exchanges once.  Same rules as :func:`cake_causal_conv1d_kcp` (one handle per call and micro-step,
        created outside the outermost checkpoint region; refused with ``graph="replay"`` and inside a CUDA-graph
        capture; every rank passes one or none).  The handle may also carry the ``"kda"`` entry of the layer's
        recurrence; a handle's ``"conv"`` entry serves either one multi call or one single call, never both.
    deterministic : bool
        Must be ``False`` (``ValueError`` otherwise).

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    _require_nondeterministic(deterministic, "cake_causal_conv1d_multi_kcp")
    _graph_mode(graph)
    xs, weights, _ = _multi_operands(xs, weights, None)
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    return CakeCausalConv1dMultiFunctionKcp.apply(
        ctx, graph, recompute_handle, False, len(xs), *xs, *weights
    )


__all__ = [
    "GRAPH_MODES",
    "GRAPH_DEFAULT",
    "GRAPH_STATS",
    "HANDLE_KEY",
    "RecomputeHandleKcp",
    "LAST_CALL",
    "graph_cache_clear",
    "CakeCausalConv1dFunctionKcp",
    "cake_causal_conv1d_kcp",
    "cake_causal_conv1d_backward_kcp",
    "cake_causal_conv1d_forward_kcp",
    "CakeCausalConv1dMultiFunctionKcp",
    "cake_causal_conv1d_multi_kcp",
    "cake_causal_conv1d_multi_backward_kcp",
    "cake_causal_conv1d_multi_forward_kcp",
]
