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
depthwise causal convolution + SiLU (FlashInfer issue #6919): the rank's
fragments of the packed documents are convolved with the three document tokens
before each continuing fragment, received from the ranks that hold them by one
``all_to_all_single`` per direction, and the backward sends the first rows'
derivative back to the predecessor fragments, which fold it into their last
rows' ``dx`` before the one BF16 rounding.  ``cake_causal_conv1d_kcp(x_local,
weight, ctx)`` is the autograd entry; the stable, ``@flashinfer_api``-decorated
facade is :mod:`flashinfer.cake_causal_conv1d_det_train`.  This module imports
only ``os``, torch, torch.distributed, tvm_ffi and its own package.

Registry contract (``kcp_jit``, the KCP records beside the zero-history
registry ``cake_jit``, whose tables stay untouched): kinds ``kcp_fwd`` (one
record per arch: co-resident grid, 4 CTAs per SM, 128 threads x 8 channels),
``kcp_bwd`` (the fallback record and the ``_st8`` record selected by
``channels_at_least`` = the zero-history rule; ``workspace_rule``
``partial_rows`` with 4 taps) and ``kcp_bwd_head`` (one record; its grid is
``max(1, n_frag x ceil(C / (threads x vec)))`` computed here and launched
through the record's ``arg_plan`` / ``logical`` / ``ffi_entry`` and ``load``);
logical tensor names are ``cake_jit.LOGICAL_TENSORS`` plus ``hist``,
``hist_row_off``, ``dpre_tail``, ``tail_row_off``, ``dpre_head``,
``head_row_off`` (``x_map`` / ``dy_map`` map to ``x`` / ``dy``); the scalars are
``cake_jit.PLAN_SCALARS`` (the KCP programs take no ``flags``).  The KCP path
keeps the package's pre-round-6 contract: FP32 taps only (``_weight``'s default
dtype set), no bias, and the legacy channel tile rule ``cake_jit.plan(...,
legacy=True)`` (``ct_vec = min(C / 8, 128)``; its kernels pick the grouped path
from ``ct_vec`` on the device and were validated and measured under that rule),
so C = 1280 still plans (128, 1, 2) here while the training routes tile it as
(32, 4, 5).

Numerics against the zero-history kernels of this package: the forward runs
the zero-history forward's instructions with history rows, so ``y`` is bitwise
equal to the no-CP ``y`` of the same documents.  The backward takes the SiLU
derivative on the FP32 preactivation (the issue's requirement), whereas the
shipped zero-history backward differentiates the BF16-rounded preactivation,
so ``dx`` is not the no-CP ``dx`` bit for bit: the two differ by that
rounding's effect on ``dpre`` plus one BF16 output rounding.  Measured on
GB300 (sm_103a) over the single-GPU test rows (``S x C`` up to 12000 x 8192,
random BF16 inputs, FP32 taps of scale 0.5): max ``|dx - dx_noCP|`` 0.03125
(one BF16 ulp at magnitude 4-8), 0 elements beyond ``1e-2 + 1e-2 |dx_noCP|``,
and the same maximum between the two policies' FP32 oracles with no kernel
involved; each kernel stays within one BF16 ulp of its own oracle.  ``dweight``
is the rank's fixed-order FP32 partial; it is reproducible across launches and
within ``5e-2 + 1e-4 |dw|`` of an FP64 oracle, not bitwise the no-CP partial.

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
#7818): with ``recompute_handle=`` (:class:`KcpRecomputeHandle`, or any object
with an ``entries`` dict) the original forward exchanges as always and stores
its history pool ``hist`` (the saved set's own tensor, key ``"conv"``) in the
handle; torch's recompute of that forward inside the backward binds it and
launches the same forward kernel on it -- no gather, no collective, no wait --
so ``y`` and the saved set ``(x_local, weight, hist)`` are bitwise those of the
exchanging recompute.  The backward is unchanged (it still exchanges the head
rows).  The forward exchange is a collective: every rank of ``ctx.group`` passes
a handle for the same call or none, created at the same checkpoint nesting
level (no check is made; a mismatch hangs or corrupts data).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.distributed as dist
import tvm_ffi

from . import (
    _cu_seqlens,
    _device_arch,
    _device_index,
    _output,
    _require_tensor,
    _scalars,
    _tokens,
    _weight,
)
from . import kcp_jit as _loader
from .cake_jit import WIDTH, launch_grid, partial_rows, plan
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
# autograd Function itself (``_handle_refusals``), so ``CakeCausalConv1dKcpFunction.apply`` refuses like the wrapper.
# The ranks are not checked against each other: the forward exchange is a collective, so every rank of the group
# must take the same branch (all with a handle at the same nesting level, or all without).

HANDLE_KEY = "conv"  # this package's entry in a recompute handle

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: Optional[bool] = None  # None = not checked yet


class KcpRecomputeHandle:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward, so the recompute runs without cross-rank communication.

    Create one handle per convolution call and micro-step, outside the outermost checkpoint region, and pass it as an
    argument of the checkpointed function, so the recompute sees the same object::

        def layer(x, weight, handle):
            return cake_causal_conv1d_kcp(x, weight, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, w, KcpRecomputeHandle(), use_reentrant=False)

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
            f"recompute_handle must carry an 'entries' dict (KcpRecomputeHandle), got {type(handle).__name__}"
        )


def _handle_lookup(handle: Any) -> Optional[dict[str, Any]]:
    """The handle's ``"conv"`` entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (it exchanges and stores one).  An entry found outside the backward raises."""
    entries = getattr(handle, "entries", None)
    if not isinstance(entries, dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (KcpRecomputeHandle), got {type(handle).__name__}"
        )
    in_backward = _kcp_in_backward()
    entry = entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'conv' entry outside the backward -- a handle serves one "
            "convolution call of one micro-step; create a new KcpRecomputeHandle per call and micro-step"
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
# :func:`cake_causal_conv1d_kcp_forward` / :func:`cake_causal_conv1d_kcp_backward` run their first call of a
# validity key (kind, the context object, its world size, and the shape / stride / dtype / data pointer of every
# operand and caller-owned output) eagerly, capture the second call into a ``torch.cuda.CUDAGraph`` (replayed once)
# and replay the graph on every later call of the key: the halo gather, the ``all_to_all_single``, its wait and the
# kernels with the same arguments in the same order (bitwise the eager result) and no host work between them.  Any
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


def cake_causal_conv1d_kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP forward: ``y`` of the rank's fragments, and the history pool ``hist`` it used (saved for the backward).

    ``x_local`` BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous, the rank's fragments back to back in the order of
    ``ctx.cu_seqlens_local``; ``weight`` FP32 ``[C, 4]``; ``ctx`` a ``KcpContext`` (its halo plan ``ctx.halo``, the
    device tables ``ctx.halo_tables`` and the process group ``ctx.group``).  Every rank of the group calls this
    function once per layer, in the same order: the forward halo exchange is a collective (a rank without tokens
    joins it with empty splits and returns empty outputs).  ``hist`` is a fresh ``[n_hist_rows, C]`` BF16 tensor
    (``[1, C]`` when the rank receives no row).  ``graph`` (``None`` = the shipped ``GRAPH_DEFAULT`` ``"off"`` |
    ``"replay"``): keyed CUDA-graph capture / replay of this call (the graph-replay option; see the section comment
    above: pointer-stable operands, private pools, eager fallback, :func:`graph_cache_clear`).
    """

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
    """The forward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_kcp_forward`.  ``_hist``:
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
    cu_seqlens_local_cpu)``: the validated ``cu`` / ``hist_index`` / ``hist_row_off`` tables, the halo splits, the
    record, its plan scalars and grid, the loaded module with its argument plan.  The operand checks, the gather, the
    exchange, the wait and the kernel launch are the untrimmed call's, in the same order on the same stream."""

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
    if _hist is None:
        hist_index = e["hist_index"]
        send = (
            x_flat.index_select(0, hist_index)
            if hist_index.numel()
            else x_flat.new_empty((0, C))
        )
        hist, work = _exchange(
            send,
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
    return y, hist


def cake_causal_conv1d_kcp_backward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP backward: ``dx`` of the rank's fragments (the halo contributions of the successor fragments folded in
    before the one BF16 rounding) and the rank's FP32 ``dweight`` partial (fixed order; its sum over the ranks is the
    framework's).

    ``hist`` is the forward's pool.  The head program writes the derivative rows of every continuing fragment's first
    tokens into the backward exchange's send layout, the exchange delivers them to the predecessors as ``dpre_tail``,
    and the main program folds them.  Every rank calls this function once per layer, in the same order.  ``graph`` as
    in the forward (the backward's key holds the forward's ``hist`` pointer, so its capture follows the forward's by
    one call).
    """

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
    """The backward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_kcp_backward`."""

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
    rows = partial_rows(grid_x, scalars["n_ch_tiles"])
    partial = torch.empty((rows, C, WIDTH), dtype=torch.float32, device=device)
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "partial": partial,
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
    """Trimmed host path: :func:`_kcp_backward` with the context-derived host work cached per ``(ctx, S, C, device,
    cu_seqlens_local_cpu)`` (tables, the head grid, the backward records, plan scalars, partial rows, the loaded modules
    with their argument plans); the operand checks, allocations, exchange, wait and launches are the untrimmed call's."""

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
            e["rows"] = partial_rows(grid_x, scalars["n_ch_tiles"])
            e["scalars"] = _scalars(scalars)
            e["launches"] = tuple(
                _launch_plan(
                    name,
                    launch_grid(_loader.MODULES[name], device_index=device_index),
                    device_index,
                )
                for name in names
            )
    n_head = e["n_head"]
    device_index = e["device_index"]
    dpre_head = _pool(n_head, C, dtype=torch.float32, device=device)
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
            device_index,
        )
    dpre_tail, work = _exchange(
        dpre_head[:n_head],
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
    partial = torch.empty((e["rows"], C, WIDTH), dtype=torch.float32, device=device)
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": e["cu"],
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "partial": partial,
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
        _launch_trim(info, values, device_index)
    return dx_out, dw_out


class CakeCausalConv1dKcpFunction(torch.autograd.Function):
    """Autograd wrapper: the forward saves ``(x_local, weight)`` and the history pool ``hist`` (no second exchange in
    the backward); the backward runs the head program, the backward halo exchange and the main backward program.
    ``handle``: a recompute handle (see :class:`KcpRecomputeHandle`): the original forward stores ``hist`` in it, the
    recompute binds it and exchanges nothing; the saved set is the same three tensors either way.  A handle is
    refused with graph replay (keyword or knob) and inside a CUDA-graph capture."""

    @staticmethod
    def forward(fn_ctx, x_local, weight, ctx, graph=None, handle=None):
        if handle is None:
            y, hist = cake_causal_conv1d_kcp_forward(x_local, weight, ctx, graph=graph)
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
        dx, dweight = cake_causal_conv1d_kcp_backward(
            x_local, weight, fn_ctx.kcp, dy, hist, graph=fn_ctx.graph
        )
        return dx, dweight, None, None, None


def cake_causal_conv1d_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
) -> torch.Tensor:
    """Autograd-enabled KCP packed width-4 depthwise causal convolution + SiLU of the rank's fragments.

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous activations of the rank's fragments, back to back in the
        order of ``ctx.cu_seqlens_local``.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current token.
    ctx : KcpContext
        The rank's context (``flashinfer.experimental.cake_kda_det_train.build_kcp_context``): fragment table, halo
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
    recompute_handle : KcpRecomputeHandle, optional
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

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape, with the three document tokens before every continuing fragment as
        history (zero history at a document's start).  The backward returns BF16 ``dx`` (the successor fragments'
        contributions folded in before the one rounding) and the rank's FP32 ``dweight`` partial.
    """

    _graph_mode(graph)  # validate on the caller's frame
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    # Always five inputs: the backward returns five gradients.
    return CakeCausalConv1dKcpFunction.apply(
        x_local, weight, ctx, graph, recompute_handle
    )


__all__ = [
    "GRAPH_MODES",
    "GRAPH_DEFAULT",
    "GRAPH_STATS",
    "HANDLE_KEY",
    "KcpRecomputeHandle",
    "LAST_CALL",
    "graph_cache_clear",
    "CakeCausalConv1dKcpFunction",
    "cake_causal_conv1d_kcp",
    "cake_causal_conv1d_kcp_backward",
    "cake_causal_conv1d_kcp_forward",
]
