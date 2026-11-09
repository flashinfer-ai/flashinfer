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
rows' ``dx`` before the one BF16 rounding.  ``cake_causal_conv1d_det_kcp(x_local,
weight, ctx)`` is the autograd entry; the stable, ``@flashinfer_api``-decorated
facade is :mod:`flashinfer.cake_causal_conv1d_det_train_kcp`.  This module imports
only torch, torch.distributed, tvm_ffi and its own package, and reads no environment variable: every switch is a
keyword argument of a call.

Registry contract (``kcp_jit``, the KCP records beside the zero-history
registry ``cake_jit``, whose tables stay untouched): kinds ``kcp_fwd`` (one
record per arch: co-resident grid, 4 CTAs per SM, 128 threads x 8 channels),
``kcp_bwd`` and ``kcp_bwd_split`` (each: the fallback record and the ``_st8``
record selected by ``channels_at_least`` = the zero-history rule;
``workspace_rule`` ``partial_rows`` with 4 taps) and ``kcp_bwd_head`` (one record; its grid is
``max(1, n_frag x ceil(C / (threads x vec)))`` computed here and launched
through the record's ``arg_plan`` / ``logical`` / ``ffi_entry`` and ``load``), ``kcp_bwd_halo`` (one record: the
overlapped backward's tail-row program, launched on the head program's grid) and ``kcp_fwd_halo`` (one record: the
two-launch forward's history-row program, launched on the head program's grid);
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
:func:`cake_causal_conv1d_det_train_kcp_host_cache_clear` empties it; ``_FFI_DEVICES`` caches the FFI device per
device index.  The operand checks, the gather, the exchange, its wait and the
kernel launches run as before with the same arguments on the same stream, so
no bit changes; a replayed graph (``graph="replay"``) contains no host work.
The private test hook ``_set_test_knobs({"host_trim": 0})`` selects the previous
host path for the A/B record.

Two launch forms of the main backward program (FlashInfer issue #8365), one per
call through the ``cooperative`` argument of the backward and autograd entries
(``None`` = :data:`COOPERATIVE_DEFAULT`, ``False``: the ordinary launch).
``cooperative=False`` runs the ``kcp_bwd_split`` record: the token loop as an
ordinary launch ``ctas_per_sm x (SM count - reserve_sms)`` CTAs wide
(:func:`_launch_grid`), writing per-CTA partial rows, then the record's ``reduce``
program (the zero-history package's fixed-order reduction, programmatically
dependent on the main program) on the same stream.  It starts beside one
small-carveout kernel of another stream per reserved SM; more such kernels
delay some CTAs, never the whole launch.  ``cooperative=True`` runs the
``kcp_bwd`` record: one cooperative launch whose grid barrier precedes the
fixed-order ``dweight`` reduction; a cooperative grid starts only when all its
CTAs fit at once, so it waits for every kernel of another stream that occupies
an SM.  ``y`` and ``dx`` are bitwise equal in both forms.  ``dweight`` is
reproducible in each form (a pure function of the device type and the call's
shapes) and within the same FP64-oracle bound, but the two forms sum different
partial rows, so their ``dweight`` bits differ.  The form is fixed per call and
changes no collective; every rank of a group passes the same ``cooperative``
value (one launch form per layer of the group, so that the ranks' ``dweight``
partials come from one reduction order).

Overlapped backward (``cooperative=False`` with an exchange in flight, i.e. at more than one rank): the
``kcp_bwd_split`` program does not wait for the backward halo exchange.  The host issues the exchange of the head
rows, launches the ``kcp_bwd_split`` program and its reduce program with every ``tail_row_off`` entry ``-1`` (no
received row is read; a fragment's last rows are written as at a document end), then waits for the exchange and
launches the ``kcp_bwd_halo`` program, which writes ``dx`` of the last ``min(3, len)`` rows of every fragment with a
tail row again: it recomputes their ``dpre`` from ``x``, the history rows and ``dy`` with the main program's
instructions and folds the received rows with the main program's FMAs in its order, before the one BF16 rounding.  So
``dx`` is bitwise the waiting launch's, and ``dweight`` keeps its bits: the received rows enter ``dx`` only, and the
partial rows and their reduction are the same.  The exchange, its wait and the collectives are unchanged; the call
launches one more program when the rank receives tail rows.  On sm_103a and sm_107a the multi entry overlaps its
first tensor this way and launches the other tensors' programs after the wait, so it too adds one program, not one
per tensor; on sm_100a it waits first (``MULTI_BWD_OVERLAP_ARCHS``).
``cooperative=True`` waits before its launch as before
(a cooperative grid waits for the communication kernel anyway).  The private test knob ``bwd_overlap`` (``1``
shipped, ``0`` = wait before the main program) selects the previous order for the A/B record.

Two-launch forward: the forward can run as a main program that reads no received row and the ``kcp_fwd_halo``
history-row program.  The main program writes every row, a continuing fragment's first rows as at a document start:
either the zero-history registry's single-rank ``fwd`` route program (``cake_jit.selected_modules(arch, "fwd", C=C,
S=S)``, the program the single-rank forward runs) on the rank's fragments, each a document of its own (``cu =
ctx.cu_seqlens_local``, ``num_docs = n_frag``; the record's full co-resident grid and token-chunk counters as in the
single-rank forward), or, in the overlapped form only, the ``kcp_fwd`` program with every ``hist_row_off`` entry
``-1``.  The history-row program then writes ``y`` of the first ``min(3, len)`` rows of every fragment with a history
row again from the received pool, with the ``kcp_fwd`` program's instructions, so ``y`` is bitwise the one-launch
forward's; it runs only when the rank receives history rows.  With an exchange in flight (more than one rank) the
overlapped form launches the main program before the exchange's wait and the history-row program after it.  The
waiting form launches after the wait: the single-rank main program and the history-row program, or, with the
``kcp_fwd`` program as main, the one-launch forward (the ``kcp_fwd`` program with the context's table, no history-row
program).  ``FWD_MAIN_ARCHS`` names the architectures whose forward runs the single-rank main program (else the
``kcp_fwd`` program), ``FWD_OVERLAP_ARCHS`` those whose forward overlaps; ``MULTI_FWD_MAIN_ARCHS`` and
``MULTI_FWD_OVERLAP_ARCHS`` are the multi entry's, which overlaps its first tensor only (the other tensors take the
waiting form after the wait).  The sets come from measurements against the one-launch forward (their comment): every
architecture's single call runs the single-rank main program after the wait, and so does the multi entry on sm_100a
and sm_103a; on sm_107a the multi entry overlaps.  The exchange, its wait and the collectives are the same in every
form.  The private test knobs ``fwd_main`` and
``fwd_overlap`` (``None`` shipped = the architecture sets, ``0`` = off, ``1`` = on) select the forms for the A/B
record, and ``LAST_CALL["fwd"]`` / ``LAST_CALL["fwd_multi"]`` record the form a call took (:func:`_fwd_record`).

The ``kcp_fwd`` program launches ``ctas_per_sm x (SM count - FWD_RESERVE_SMS)``
CTAs when they hold every channel tile of ``C``, else the full co-resident grid
(:func:`_forward_grid`).  Beside one small-carveout kernel of another stream
that occupies a single SM, every CTA of the shorter grid can then start at
once; kernels on more SMs still make some CTAs wait for a later wave.  The
single-rank main program launches its record's full co-resident grid.  ``y``
does not depend on the grid.

Checkpoint recompute without the halo exchange (FlashInfer issues #7815 /
#7818): with ``recompute_handle=`` (:class:`CakeCausalConv1dDetKcpRecomputeHandle`, or any object
with an ``entries`` dict) the original forward exchanges as always and stores
its history pool ``hist`` (the saved set's own tensor, key ``"conv"``) in the
handle; torch's recompute of that forward inside the backward binds it and
launches the same forward programs on it (the two-launch forward in its waiting form) -- no gather, no collective,
no wait --
so ``y`` and the saved set ``(x_local, weight, hist)`` are bitwise those of the
exchanging recompute.  The backward is unchanged (it still exchanges the head
rows).  The forward exchange is a collective: every rank of ``ctx.group`` passes
a handle for the same call or none, created at the same checkpoint nesting
level (no check is made; a mismatch hangs or corrupts data).

Several convolutions of one layer (FlashInfer issue #7815): ``cake_causal_conv1d_det_kcp_multi(xs, weights, ctx)`` runs
the convolutions of ``n`` tensors that share the context (a layer's q, k and v) with one halo exchange per direction
instead of ``n``: the tensors' send rows travel side by side in one ``all_to_all_single`` and are copied out into
each tensor's own contiguous pool after the wait, and each tensor then runs the single call's kernels with the
single call's arguments.  In the overlapped backward (sm_103a and sm_107a, ``MULTI_BWD_OVERLAP_ARCHS``) only the
first tensor takes the overlapped form (its main programs before the wait, its tail-row program after it); the other
tensors run the waiting form's main programs after the wait, whose ``dx`` and ``dweight`` are bitwise the overlapped
form's.  So outputs and gradients (``dweight`` included) are bitwise one single call per tensor.
One recompute handle serves the whole call (its ``"conv"`` entry holds the ``n`` pools).  The multi path always
uses the trimmed host path's per-context cache (the ``host_trim`` test knob selects the single call's path only).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import tvm_ffi

from . import (
    _cu_seqlens,
    _device_arch,
    _device_index,
    _forward_work,
    _output,
    _require_tensor,
    _scalars,
    _tokens,
    _weight,
)
from . import cake_jit as _zero_history
from . import kcp_jit as _loader
from .cake_jit import WIDTH, partial_rows, plan
from .cake_jit import launch_grid as _record_launch_grid
from .kcp_jit import load, route_modules, run_module, selected_modules

HALO = WIDTH - 1  # history rows, tail rows and head rows per fragment
HEAD_SLOTS = HALO  # send slots per head row (a head token is consumed by at most three predecessor fragments)
KCP_FWD = "kcp_fwd"
KCP_BWD = "kcp_bwd"
KCP_BWD_SPLIT = "kcp_bwd_split"
KCP_BWD_HEAD = "kcp_bwd_head"
KCP_BWD_HALO = "kcp_bwd_halo"
KCP_FWD_HALO = "kcp_fwd_halo"
# The zero-history registry's forward route of FP32 taps and no bias (the KCP contract): the single-rank main program
# of the two-launch forward (the module docstring).
FWD_MAIN_ROUTE = "fwd"
# The main backward's launch form when a call passes ``cooperative=None``: ``False`` = the ``kcp_bwd_split`` record and
# its reduce program, which starts beside kernels resident on other streams; ``True`` = the cooperative ``kcp_bwd``
# record, which waits for them (FlashInfer issue #8365).
COOPERATIVE_DEFAULT = False
# SMs of co-resident CTAs the forward (``kcp_fwd``) leaves free.  The forward record carries no ``reserve_sms`` (the
# non-deterministic package's KCP forward runs the same record), so the rule is this host's (:func:`_launch_grid`): with
# the shorter grid, no forward CTA waits for a second wave beside one small-carveout kernel of another stream resident
# on an SM (a communication kernel).  A ``C`` whose channel tiles need the reserved SM launches the full grid
# (:func:`_forward_grid`) and gives up that property.  ``y`` is formed per token in a fixed FMA order, so it does not
# depend on the grid.  The single-rank main program of the two-launch forward (``FWD_MAIN_ARCHS``) is the
# zero-history registry's record and launches its full co-resident grid, as the single-rank forward does; the
# history-row program launches the head program's grid.
FWD_RESERVE_SMS = 1
GRAPH_MODES = (
    "off",
    "replay",
)  # graph replay: keyed CUDA-graph capture / replay of the halo calls (keyword graph= or the test knob)
GRAPH_DEFAULT = (
    "off"  # shipped: the eager path; the public keyword graph="replay" opts in per call
)


def _launch_grid(
    record: dict[str, Any],
    *,
    device_index: int,
    C: int | None = None,
    S: int | None = None,
) -> tuple[int, int, int]:
    """The record's launch grid (:func:`cake_jit.launch_grid`) less its reserved SMs of co-resident CTAs: the record's
    ``grid_rule.reserve_sms``, else :data:`FWD_RESERVE_SMS` for the forward record, else none.

    A record with ``reserve_sms = 1`` launches ``ctas_per_sm x (SM count - 1)`` CTAs: one small-carveout kernel of
    another stream resident on an SM keeps the whole SM from hosting a CTA of this program (its shared memory needs the
    SM's large carveout), so a grid one SM short of the co-resident count still starts every CTA at once beside it
    (FlashInfer issue #8365).  The grid is a pure function of the device and the record, so the partial rows and the
    fixed-order ``dweight`` sum are the same on every call on one device type (bits differ from a full-grid launch).
    More such kernels than reserved SMs make the displaced CTAs start after the first ones finish: the backward then
    takes at most about twice its time, independent of how long the other kernels run.
    """

    grid = _record_launch_grid(record, device_index=device_index, C=C, S=S)
    reserve = int(
        record["grid_rule"].get(
            "reserve_sms", FWD_RESERVE_SMS if record.get("role") == KCP_FWD else 0
        )
    )
    if reserve:
        reserved = reserve * int(record["grid_rule"]["ctas_per_sm"])
        if grid[0] <= reserved:
            raise ValueError(
                f"{record['kernel_symbol']}: a grid of {grid[0]} CTAs cannot reserve {reserve} SM(s)"
            )
        grid = (grid[0] - reserved, grid[1], grid[2])
    return grid


def _forward_grid(
    record: dict[str, Any], *, device_index: int, C: int, n_ch_tiles: int
) -> tuple[int, int, int]:
    """The forward's launch grid: :func:`_launch_grid` (one SM short, :data:`FWD_RESERVE_SMS`) when that grid holds
    all ``n_ch_tiles`` channel tiles, else the record's full co-resident grid, which every ``C`` it holds launched with
    before the reservation (also on a device too small to give up an SM); a ``ValueError`` when not even the full grid
    holds them.  ``y`` does not depend on the grid."""

    full = _record_launch_grid(record, device_index=device_index)
    if n_ch_tiles > full[0]:
        raise ValueError(
            f"C={C} spans more channel tiles than the forward's grid on this device holds"
        )
    reserved = FWD_RESERVE_SMS * int(record["grid_rule"]["ctas_per_sm"])
    if full[0] - reserved >= max(n_ch_tiles, 1):
        return _launch_grid(record, device_index=device_index)
    return full


def _cooperative(cooperative: Optional[bool]) -> bool:
    """The launch form of one call's main backward program: ``cooperative`` itself, or :data:`COOPERATIVE_DEFAULT`
    for ``None`` (a ``TypeError`` for anything but ``None`` or a bool)."""

    if cooperative is None:
        return COOPERATIVE_DEFAULT
    if not isinstance(cooperative, bool):
        raise TypeError(
            f"cooperative must be None or a bool, got {type(cooperative).__name__}"
        )
    return cooperative


def _bwd_kind(cooperative: bool) -> str:
    """The backward route of a launch form: ``kcp_bwd`` (cooperative) or ``kcp_bwd_split``."""

    return KCP_BWD if cooperative else KCP_BWD_SPLIT


def _overlapped(cooperative: bool, work: Optional[Any]) -> bool:
    """Whether a backward call with tokens runs the overlapped backward (the module docstring): the ``kcp_bwd_split``
    form with an exchange in flight (``work``, more than one rank) and the shipped ``bwd_overlap`` knob."""

    return not cooperative and work is not None and _TEST_KNOBS["bwd_overlap"] == 1


def _bwd_variant(cooperative: bool) -> Any:
    """The backward's graph-key variant: the launch form, with the ``bwd_overlap`` knob when it is not shipped."""

    knob = _TEST_KNOBS["bwd_overlap"]
    if knob == BWD_OVERLAP_DEFAULT:
        return cooperative
    return (cooperative, ("bwd_overlap", knob))


def _fwd_form(arch: str, *, multi: bool = False) -> tuple[bool, bool]:
    """``(sr, overlap)`` of the forward on ``arch`` (the module docstring's "Two-launch forward"): whether its main
    program is the single-rank ``fwd`` route program (else the ``kcp_fwd`` program), and whether it launches the main
    program before the exchange's wait when an exchange is in flight.  Each is its test knob (``fwd_main`` /
    ``fwd_overlap``) when the knob is ``0`` or ``1``, else (``None``, shipped) whether ``arch`` is in the entry's set:
    ``FWD_MAIN_ARCHS`` / ``FWD_OVERLAP_ARCHS``, or ``MULTI_FWD_MAIN_ARCHS`` / ``MULTI_FWD_OVERLAP_ARCHS`` for the multi
    entry (``multi``)."""

    main, overlap = _TEST_KNOBS["fwd_main"], _TEST_KNOBS["fwd_overlap"]
    if main is None:
        main = arch in (MULTI_FWD_MAIN_ARCHS if multi else FWD_MAIN_ARCHS)
    if overlap is None:
        overlap = arch in (MULTI_FWD_OVERLAP_ARCHS if multi else FWD_OVERLAP_ARCHS)
    return bool(main), bool(overlap)


def _fwd_variant(x: Any, *, multi: bool = False) -> tuple[bool, bool]:
    """The forward's graph-key variant: :func:`_fwd_form` on the device of ``x`` (the effective forms, so a changed
    knob or architecture set is another key)."""

    return _fwd_form(_device_arch(_require_tensor(x, "x_local").device), multi=multi)


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
# autograd Function itself (``_handle_refusals``), so ``CakeCausalConv1dDetKcpFunction.apply`` refuses like the wrapper.
# The ranks are not checked against each other: the forward exchange is a collective, so every rank of the group
# must take the same branch (all with a handle at the same nesting level, or all without).

HANDLE_KEY = "conv"  # this package's entry in a recompute handle

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: Optional[bool] = None  # None = not checked yet


class CakeCausalConv1dDetKcpRecomputeHandle:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward, so the recompute runs without cross-rank communication.

    Create one handle per convolution call and micro-step, outside the outermost checkpoint region, and pass it as an
    argument of the checkpointed function, so the recompute sees the same object::

        def layer(x, weight, handle):
            return cake_causal_conv1d_det_kcp(x, weight, ctx, recompute_handle=handle)

        y = torch.utils.checkpoint.checkpoint(layer, x, w, CakeCausalConv1dDetKcpRecomputeHandle(), use_reentrant=False)

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
            f"recompute_handle must carry an 'entries' dict (CakeCausalConv1dDetKcpRecomputeHandle), got {type(handle).__name__}"
        )


def _handle_lookup(handle: Any) -> Optional[dict[str, Any]]:
    """The handle's ``"conv"`` entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (it exchanges and stores one).  An entry found outside the backward raises."""
    entries = getattr(handle, "entries", None)
    if not isinstance(entries, dict):
        raise TypeError(
            f"recompute_handle must carry an 'entries' dict (CakeCausalConv1dDetKcpRecomputeHandle), got {type(handle).__name__}"
        )
    in_backward = _kcp_in_backward()
    entry = entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'conv' entry outside the backward -- a handle serves one "
            "convolution call of one micro-step; create a new CakeCausalConv1dDetKcpRecomputeHandle per call and micro-step"
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
            "recompute_handle: the 'conv' entry belongs to another KCP context"
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
# The public keyword ``graph="replay"`` of :func:`cake_causal_conv1d_det_kcp` and of the forward / backward below (shipped
# ``GRAPH_DEFAULT = "off"``) -- or, for the A/B record, the private test hook ``_set_test_knobs({"graph": "replay"})``
# (a keyword contradicting an active knob is a ``ValueError``) -- makes
# :func:`cake_causal_conv1d_det_train_kcp_forward` / :func:`cake_causal_conv1d_det_train_kcp_backward` run their first call of a
# validity key (kind, the context object, its world size, and the shape / stride / dtype / data pointer of every
# operand and caller-owned output) eagerly, capture the second call into a ``torch.cuda.CUDAGraph`` (replayed once)
# and replay the graph on every later call of the key: the halo gather, the ``all_to_all_single``, its wait and the
# kernels with the same arguments in the same order (bitwise the eager result) and no host work between them.  Any
# other key runs eagerly (silent fallback); a rank without tokens is never captured.  A replayed call returns the
# graph's static tensors (detached): the next replay of the key overwrites them, so a framework must keep its inputs
# pointer-stable and consume the outputs before the next call of the same layer.  This module keeps its own cache
# (no import of the KDA host module); ``GRAPH_STATS`` counts the events and ``LAST_CALL[kind]`` says how the last
# forward / backward ran (keyword or knob).  Every captured graph keeps its fresh tensors in a private pool (one per
# cached key; at most GRAPH_CACHE_MAX keys, the oldest dropped first); ``cake_causal_conv1d_det_train_kcp_graph_cache_clear()`` releases them and must run
# before ``torch.distributed.destroy_process_group()`` (a graph holding the NCCL collective blocks ``ncclCommDestroy``).

# Trimmed host path; shipped default:
# 1 = the trimmed host path of the eager halo call (same launches, bitwise; the module docstring's "Shipped host path"),
# 0 = the previous host path, selectable through _set_test_knobs for the A/B record.
HOST_TRIMS = (0, 1)
HOST_TRIM_DEFAULT = (
    1  # shipped; 0 = the previous path, via _set_test_knobs({"host_trim": 0}) only
)
# The overlapped backward (the module docstring's "Overlapped backward"): 1 = shipped, the split program before the
# exchange's wait and the tail-row program after it; 0 = the wait before the main program, via _set_test_knobs only.
BWD_OVERLAPS = (0, 1)
BWD_OVERLAP_DEFAULT = 1
# The architectures whose multi entry takes the overlapped backward (its first tensor's main programs before the
# exchange's wait).  On sm_100a the overlapped multi backward measured no robust gain over the waiting one (B200,
# 3 x C = 8192 at CP4 and CP8: -0.9% to +0.8% per shape; CP4 zigzag 0.5% slower with only the knob changed), so its
# multi entry waits first.  The single call overlaps on every architecture.
MULTI_BWD_OVERLAP_ARCHS = frozenset(("sm_103a", "sm_107a"))
# The two-launch forward (the module docstring's "Two-launch forward"): the knob values of ``fwd_main`` (the main
# program: 1 = the single-rank ``fwd`` route program, 0 = the ``kcp_fwd`` program) and ``fwd_overlap`` (1 = the main
# program before the exchange's wait, 0 = after it); ``None``, shipped, = the architecture sets below.
FWD_MAINS = (None, 0, 1)
FWD_MAIN_DEFAULT = None
FWD_OVERLAPS = (None, 0, 1)
FWD_OVERLAP_DEFAULT = None
# The architectures whose single call runs the single-rank main program (``FWD_MAIN_ARCHS``) and launches its main
# program before the exchange's wait (``FWD_OVERLAP_ARCHS``), and the multi entry's (``MULTI_*``).  Each entry takes,
# per architecture, the form faster than the one-launch forward on every forward, forward + backward and checkpointed
# step row, the simpler form of a tie (launch-to-launch time with the split backward; CP4, and CP8 on B200; contiguous,
# zigzag and per-document zigzag layouts; one tensor of C = 10240, three of C = 8192; % = speedup - 1):
# - the single call waits and runs the single-rank main program on every architecture: forward 11.6-15.3% /
#   6.6-8.1% / 8.0-9.4%, forward + backward 4.6-6.4% / 2.5-2.8% / 3.3-3.7%, checkpointed step 6.7-8.5% / 3.5-3.8% /
#   3.7-4.7% (B200 / GB300 / R200).  Overlapped, its forward gains more, but beside the overlapped backward the
#   contiguous row loses (forward + backward 0.881x / 0.887x on B200 / GB300, checkpointed step 0.943x-0.957x on all
#   three);
# - the multi entry waits on sm_100a and sm_103a (forward 7.5-10.0% / 3.5-4.5%, forward + backward 3.0-4.3% /
#   1.6-2.0%, checkpointed step 4.1-5.7% / 2.3-2.8%; overlapped, within -0.2% to +1.2% of that on every row, a tie that
#   goes to the waiting form) and overlaps on sm_107a (forward 2.8-3.0%, forward + backward 1.1-1.6%, checkpointed
#   step 1.6-1.8%; waiting reads -0.9% to +0.5% there).
FWD_MAIN_ARCHS = frozenset(("sm_100a", "sm_103a", "sm_107a"))
FWD_OVERLAP_ARCHS: frozenset[str] = frozenset()
MULTI_FWD_MAIN_ARCHS = frozenset(("sm_100a", "sm_103a", "sm_107a"))
MULTI_FWD_OVERLAP_ARCHS = frozenset(("sm_107a",))
_TEST_KNOBS: dict[str, Any] = {
    "graph": "off",
    "host_trim": HOST_TRIM_DEFAULT,
    "bwd_overlap": BWD_OVERLAP_DEFAULT,
    "fwd_main": FWD_MAIN_DEFAULT,
    "fwd_overlap": FWD_OVERLAP_DEFAULT,
}
GRAPH_STATS: dict[str, int] = {"eager": 0, "capture": 0, "replay": 0, "fallback": 0}
GRAPH_CACHE_MAX = 64
LAST_CALL: dict[str, dict[str, Any]] = {}

# -- the trimmed host path (host_trim=1): the per-context host state of the eager halo calls.  ``_HOST_CACHE[id(ctx)]`` holds the
# context (pinned, so its id cannot be reused) and one entry per ``(kind, S, C, device, cu_seqlens_local_cpu)`` and
# launch form (the backward's ``cooperative``, the forward's effective ``(sr, overlap)`` of :func:`_fwd_form`): the
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


def cake_causal_conv1d_det_train_kcp_host_cache_clear() -> int:
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


def _reduce_plan(
    record: dict[str, Any], C: int, grid_x: int, device_index: int
) -> Optional[tuple[Any, str, tuple[tuple[str | None, int], ...]]]:
    """The cached launch of the backward record's ``reduce`` program (a record of the zero-history registry, FlashInfer
    issue #8365), or ``None`` for a record without one: ``ceil(C / channels_per_cta)`` CTAs summing the ``grid_x`` main
    CTAs' partial rows in the fixed order; its binding launches it programmatically dependent on the main program."""
    name = record.get("reduce")
    if name is None:
        return None
    reduce_record = _zero_history.MODULES[name]
    grid = _zero_history.launch_grid(reduce_record, device_index=device_index, C=C)
    return _zero_history_plan(name, grid, device_index)


def _zero_history_plan(
    name: str, grid: tuple[int, int, int], device_index: int
) -> tuple[Any, str, tuple[tuple[str | None, int], ...]]:
    """:func:`_launch_plan` of record ``name`` of the zero-history registry (``cake_jit``: the backward's reduce
    programs and the two-launch forward's single-rank main program), loaded through that registry."""
    record = _zero_history.MODULES[name]
    logical = record["logical"]
    grid_values = {
        "grid_x": int(grid[0]),
        "grid_y": int(grid[1]),
        "grid_z": int(grid[2]),
    }
    plan_: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan_.append((None, grid_values[key]))
        else:
            plan_.append((logical[key], 0))
    module = _zero_history.load(name, _loader.device_arch(device_index))
    return module, record["ffi_entry"], tuple(plan_)


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
    the trimmed host path of the eager call | ``0`` the previous host path; ``bwd_overlap``: ``1`` shipped = the
    overlapped backward | ``0`` the wait before the main program; ``fwd_main`` / ``fwd_overlap``: ``None`` shipped =
    the architecture sets of the two-launch forward | ``0`` off | ``1`` on, :func:`_fwd_form`); a knob
    the dict does not name keeps its shipped default; ``None`` restores the defaults.  A private test hook, called by
    the tests only (no environment variable opens or selects it).  Returns the active knobs."""
    if knobs is None:
        _TEST_KNOBS["graph"] = "off"
        _TEST_KNOBS["host_trim"] = HOST_TRIM_DEFAULT
        _TEST_KNOBS["bwd_overlap"] = BWD_OVERLAP_DEFAULT
        _TEST_KNOBS["fwd_main"] = FWD_MAIN_DEFAULT
        _TEST_KNOBS["fwd_overlap"] = FWD_OVERLAP_DEFAULT
        return dict(_TEST_KNOBS)
    if not isinstance(knobs, dict):
        raise TypeError(f"_set_test_knobs expects a dict, got {type(knobs).__name__}")
    unknown = sorted(
        set(knobs) - {"graph", "host_trim", "bwd_overlap", "fwd_main", "fwd_overlap"}
    )
    if unknown:
        raise ValueError(
            f"_set_test_knobs: unknown knob(s) {unknown}; known: ['bwd_overlap', 'fwd_main', 'fwd_overlap', 'graph', "
            "'host_trim']"
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
    bwd_overlap = knobs.get("bwd_overlap", BWD_OVERLAP_DEFAULT)
    if isinstance(bwd_overlap, bool) or bwd_overlap not in BWD_OVERLAPS:
        raise ValueError(
            f"_set_test_knobs: bwd_overlap must be one of {BWD_OVERLAPS}, got {bwd_overlap!r}"
        )
    fwd_main = knobs.get("fwd_main", FWD_MAIN_DEFAULT)
    if isinstance(fwd_main, bool) or fwd_main not in FWD_MAINS:
        raise ValueError(
            f"_set_test_knobs: fwd_main must be one of {FWD_MAINS}, got {fwd_main!r}"
        )
    fwd_overlap = knobs.get("fwd_overlap", FWD_OVERLAP_DEFAULT)
    if isinstance(fwd_overlap, bool) or fwd_overlap not in FWD_OVERLAPS:
        raise ValueError(
            f"_set_test_knobs: fwd_overlap must be one of {FWD_OVERLAPS}, got {fwd_overlap!r}"
        )
    _TEST_KNOBS["graph"] = graph
    _TEST_KNOBS["host_trim"] = int(host_trim)
    _TEST_KNOBS["bwd_overlap"] = int(bwd_overlap)
    _TEST_KNOBS["fwd_main"] = None if fwd_main is None else int(fwd_main)
    _TEST_KNOBS["fwd_overlap"] = None if fwd_overlap is None else int(fwd_overlap)
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

    __slots__ = (
        "key",
        "pins",
        "seen",
        "graph",
        "result",
        "record",
        "disabled",
        "digest",
    )

    def __init__(self, key: tuple, pins: tuple) -> None:
        self.key, self.pins = (
            key,
            pins,
        )  # the pins keep the keyed objects alive so their ids cannot be reused
        self.seen = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.result: Optional[tuple] = None
        # what the captured call recorded in LAST_CALL[kind] (the forward's launch form), repeated by every replay
        self.record: dict[str, Any] = {}
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
    kind: str,
    ctx: Any,
    tensors: tuple,
    mode: Optional[str] = None,
    variant: Any = None,
) -> Optional[_GraphEntry]:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured (mode off, no tokens);
    ``mode`` = the effective graph mode (:func:`_graph_mode`), ``None`` = the knob; ``variant`` = a host value of the
    call that changes the captured work (the backward's launch form, the forward's :func:`_fwd_form`), part of the
    key."""
    if (
        (_TEST_KNOBS["graph"] if mode is None else mode) == "off"
        or int(getattr(ctx, "T_r", 0)) == 0
        or int(getattr(ctx, "n_frag", 0)) == 0
    ):
        return None
    key = (
        kind,
        id(ctx),
        int(ctx.world_size),
        tuple(_sig(t) for t in tensors),
        variant,
    )
    e = _GRAPHS.get(key)
    if e is None:
        while len(_GRAPHS) >= GRAPH_CACHE_MAX:
            del _GRAPHS[next(iter(_GRAPHS))]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx,))
    return e


def cake_causal_conv1d_det_train_kcp_graph_cache_clear() -> int:
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
    (the key's first call, a key whose capture failed, or a call that is never captured).  Returns the results.
    ``LAST_CALL[kind]`` holds what the body recorded (the forward's launch form; a replay repeats the captured call's)
    and the graph event."""
    if entry is not None and entry.graph is not None:
        entry.graph.replay()
        GRAPH_STATS["replay"] += 1
        LAST_CALL[kind] = {
            **entry.record,
            "graph": "replay",
            "graph_event": "replay",
            "graph_key": entry.digest,
            "graph_fallback": None,
        }
        return entry.result
    LAST_CALL.pop(kind, None)  # the body records afresh
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
            entry.record = LAST_CALL.pop(kind, {})
            entry.seen += 1
            GRAPH_STATS["capture"] += 1
            LAST_CALL[kind] = {
                **entry.record,
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
        **LAST_CALL.get(kind, {}),
        "graph": _TEST_KNOBS["graph"] if mode is None else mode,
        "graph_event": "eager",
        "graph_key": None if entry is None else entry.digest,
        "graph_fallback": None if entry is None else entry.disabled,
    }
    return result


def cake_causal_conv1d_det_train_kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP forward: ``y`` of the rank's fragments, and the history pool ``hist`` it used (saved for the backward).

    ``x_local`` BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous, the rank's fragments back to back in the order of
    ``ctx.cu_seqlens_local``; ``weight`` FP32 ``[C, 4]``; ``ctx`` a KCP context (its halo plan ``ctx.halo``, the
    device tables ``ctx.halo_tables`` and the process group ``ctx.group``).  Every rank of the group calls this
    function once per layer, in the same order: the forward halo exchange is a collective (a rank without tokens
    joins it with empty splits and returns empty outputs).  ``hist`` is a fresh ``[n_hist_rows, C]`` BF16 tensor
    (``[0, C]`` when the rank receives no row; the program then reads a private one-row placeholder that is not
    returned).  ``graph`` (``None`` = the shipped ``GRAPH_DEFAULT`` ``"off"`` |
    ``"replay"``): keyed CUDA-graph capture / replay of this call (the graph-replay option; see the section comment
    above: pointer-stable operands, private pools, eager fallback, :func:`cake_causal_conv1d_det_train_kcp_graph_cache_clear`).
    """

    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_forward(x_local, weight, ctx, out=out)
    entry = _graph_entry(
        "fwd", ctx, (x_local, weight, out), mode, variant=_fwd_variant(x_local)
    )
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
    """The forward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_det_train_kcp_forward`.  ``_hist``:
    the history pool a recompute handle stored (checked by :func:`_handle_hist`): no gather, no exchange, no wait;
    the programs are the same, launched in the waiting form."""

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
        _fwd_record("fwd", None, False, False)
        return y, hist
    hist_row_off = _table(
        ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    device_index = _device_index(device)
    sr, overlap = _fwd_form(arch)
    overlap = overlap and work is not None
    values: dict[str, Any] = {"x": x_flat, "w": w, "cu": cu, "y": y.view(S, C)}
    if sr:
        # the single-rank fwd route program on the fragments (each a document of its own): the single-rank forward's
        # plan, full co-resident grid and token-chunk counters
        (name,) = _zero_history.selected_modules(arch, FWD_MAIN_ROUTE, C=C, S=S)
        record = _zero_history.MODULES[name]
        scalars = plan(
            S,
            C,
            n_frag,
            geometry=record["grid_rule"]["geometry"],
            role=record["role"],
            route=_zero_history.record_route(record, arch),
            arch=arch,
        )
        grid = _record_launch_grid(record, device_index=device_index, C=C, S=S)
        if scalars["n_ch_tiles"] > grid[0]:
            raise ValueError(
                f"C={C} spans more channel tiles than the forward's grid on this device holds"
            )
        counters = _forward_work(
            None,
            _zero_history.counter_words(record, scalars["n_ch_tiles"]),
            device=device,
            device_index=device_index,
        )
        if counters is not None:
            values["work"] = counters
    else:
        (name,) = route_modules(arch, KCP_FWD)
        record = _loader.MODULES[name]
        scalars = plan(
            S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
        )
        grid = _forward_grid(
            record, device_index=device_index, C=C, n_ch_tiles=scalars["n_ch_tiles"]
        )
        if overlap:
            # the overlapped forward: the main program reads no received row while the exchange is in flight
            # (``x`` stands for the pool: a valid pointer no -1 entry names)
            values["hist"] = x_flat
            values["hist_row_off"] = torch.full(
                (n_frag, HALO), -1, dtype=torch.int32, device=device
            )
        else:
            values["hist"] = (
                hist
                if hist.shape[0]
                else _pool(0, C, dtype=torch.bfloat16, device=device)
            )
            values["hist_row_off"] = hist_row_off
    values.update(_scalars(scalars))
    if work is not None and not overlap:
        work.wait()
    # the grid of :func:`_forward_grid` (the kcp_fwd program) or the record's (the single-rank program), as on the
    # trimmed host path
    if sr:
        _launch_trim(_zero_history_plan(name, grid, device_index), values, device_index)
    else:
        run_module(name, values, device_index=device_index, grid=grid)
    if overlap:
        work.wait()
    repair = (sr or overlap) and int(halo.n_hist_rows) > 0
    if repair:
        (halo_name,) = route_modules(arch, KCP_FWD_HALO)
        _launch_with_grid(
            halo_name,
            {
                "x": x_flat,
                "w": w,
                "cu": cu,
                "y": y.view(S, C),
                "hist": hist,
                "hist_row_off": hist_row_off,
                "C": C,
                "num_docs": n_frag,
            },
            grid=_loader.head_grid(_loader.MODULES[halo_name], n_frag=n_frag, C=C),
            device_index=device_index,
        )
    _fwd_record("fwd", sr, overlap, repair)
    return y, hist


def _fwd_host(
    x: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor],
    device: torch.device,
    multi: bool = False,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """The trimmed forward's operand checks, output and host-cache entry: ``(S, C, x_flat, w, y, entry)``.  The entry
    is cached per ``(ctx, S, C, device, cu_seqlens_local_cpu, sr, overlap)`` (:func:`_fwd_form` of the device's
    architecture, of the multi entry's sets with ``multi``; both kept in the entry): the validated ``cu`` /
    ``hist_index`` / ``hist_row_off`` tables, the halo splits, the main program's record, its plan scalars, grid and
    counter words, the loaded module with its argument plan."""

    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    arch = _device_arch(device)
    sr, overlap = _fwd_form(arch, multi=multi)
    e = _host_cache_entry(
        ctx,
        ("fwd", S, C, str(device), tuple(ctx.cu_seqlens_local_cpu), sr, overlap),
    )
    if not e:
        cu, n_frag = _context(ctx, device=device, T_r=S)
        halo = ctx.halo
        # filled into the entry at the end, so a call that raises below (C too wide for the grid) leaves it empty
        fresh: dict[str, Any] = dict(
            cu=cu,
            n_frag=n_frag,
            hist_index=_require_tensor(
                ctx.halo_tables["hist_index"], "ctx.halo_tables['hist_index']"
            ),
            recv_rows=int(halo.n_hist_rows),
            send_splits=tuple(halo.fwd_send_rows),
            recv_splits=tuple(halo.fwd_recv_rows),
            sr=sr,
            overlap=overlap,
        )
        if S != 0:
            fresh["hist_row_off"] = _table(
                ctx,
                "hist_row_off",
                shape=(n_frag, HALO),
                dtype=torch.int32,
                device=device,
            )
            device_index = _device_index(device)
            if sr:
                # the single-rank fwd route program: the single-rank forward's plan, full co-resident grid and counters
                (name,) = _zero_history.selected_modules(arch, FWD_MAIN_ROUTE, C=C, S=S)
                record = _zero_history.MODULES[name]
                scalars = plan(
                    S,
                    C,
                    n_frag,
                    geometry=record["grid_rule"]["geometry"],
                    role=record["role"],
                    route=_zero_history.record_route(record, arch),
                    arch=arch,
                )
                grid = _record_launch_grid(record, device_index=device_index, C=C, S=S)
                if scalars["n_ch_tiles"] > grid[0]:
                    raise ValueError(
                        f"C={C} spans more channel tiles than the forward's grid on this device holds"
                    )
                fresh["words"] = _zero_history.counter_words(
                    record, scalars["n_ch_tiles"]
                )
                fresh["launch"] = _zero_history_plan(name, grid, device_index)
            else:
                (name,) = route_modules(arch, KCP_FWD)
                record = _loader.MODULES[name]
                scalars = plan(
                    S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
                )
                grid = _forward_grid(
                    record,
                    device_index=device_index,
                    C=C,
                    n_ch_tiles=scalars["n_ch_tiles"],
                )
                fresh["words"] = 0
                fresh["launch"] = _launch_plan(name, grid, device_index)
            fresh["scalars"] = _scalars(scalars)
            fresh["device_index"] = device_index
        e.update(fresh)
    return S, C, x_flat, w, y, e


def _fwd_gather(x_flat: torch.Tensor, C: int, e: dict[str, Any]) -> torch.Tensor:
    """The forward's send rows: the context's ``hist_index`` rows of ``x_flat`` in destination-rank order."""
    hist_index = e["hist_index"]
    return (
        x_flat.index_select(0, hist_index)
        if hist_index.numel()
        else x_flat.new_empty((0, C))
    )


def _fwd_main(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    hist: Optional[torch.Tensor],
    counters: Optional[torch.Tensor],
    *,
    S: int,
    C: int,
    device: torch.device,
    deferred: bool = False,
) -> None:
    """Launch the main forward program of a call with ``S > 0`` (no wait).  The single-rank ``fwd`` route program
    (``e["sr"]``) reads no history row: it writes every fragment as a document of its own, taking its token chunks
    from ``counters``.  The ``kcp_fwd`` program reads ``hist`` through the context's ``hist_row_off``; ``deferred``
    (the overlapped forward's main launch, before the exchange's wait): every ``hist_row_off`` entry ``-1`` and ``x``
    as the (unread) pool, so it reads no received row and writes a continuing fragment's first rows as at a document
    start (:func:`_fwd_repair` writes them again after the wait); ``hist`` is then not used."""
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": e["cu"],
        "y": y.view(S, C),
        **e["scalars"],
    }
    if e["sr"]:
        if counters is not None:
            values["work"] = counters
    elif deferred:
        values["hist"] = x_flat
        values["hist_row_off"] = _no_hist(e, device)
    else:
        values["hist"] = (
            hist if hist.shape[0] else _pool(0, C, dtype=torch.bfloat16, device=device)
        )
        values["hist_row_off"] = e["hist_row_off"]
    _launch_trim(e["launch"], values, e["device_index"])


def _fwd_repair(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    hist: torch.Tensor,
    *,
    S: int,
    C: int,
    device: torch.device,
) -> bool:
    """The two-launch forward's history-row program (route ``kcp_fwd_halo``), after the main program and the
    exchange's wait: ``y`` of the first ``min(3, len)`` rows of every fragment with a history row, written again from
    the received ``hist`` with the ``kcp_fwd`` program's instructions (nothing when the rank receives no history
    row).  Its launch plan is made at the entry's first such call, then reused.  Returns whether it ran."""
    if e["recv_rows"] == 0:
        return False
    info = e.get("repair_launch")
    if info is None:
        (name,) = route_modules(_device_arch(device), KCP_FWD_HALO)
        grid = _loader.head_grid(_loader.MODULES[name], n_frag=e["n_frag"], C=C)
        info = e["repair_launch"] = _launch_plan(name, grid, e["device_index"])
    _launch_trim(
        info,
        {
            "x": x_flat,
            "w": w,
            "cu": e["cu"],
            "y": y.view(S, C),
            "hist": hist,
            "hist_row_off": e["hist_row_off"],
            "C": C,
            "num_docs": e["n_frag"],
        },
        e["device_index"],
    )
    return True


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
    counters: Optional[torch.Tensor] = None,
) -> bool:
    """The waiting form: wait ``work`` (when given) and launch the forward of a call with ``S > 0`` on ``hist``: its
    main program (:func:`_fwd_main`) and, after the single-rank main program, the history-row program
    (:func:`_fwd_repair`).  Returns whether the history-row program ran."""
    if work is not None:
        work.wait()
    _fwd_main(e, x_flat, w, y, hist, counters, S=S, C=C, device=device)
    return e["sr"] and _fwd_repair(e, x_flat, w, y, hist, S=S, C=C, device=device)


def _fwd_record(kind: str, sr: Optional[bool], pair: bool, repair: bool) -> None:
    """Record the launch form of the forward call ``kind`` (``"fwd"`` / ``"fwd_multi"``) in ``LAST_CALL[kind]``:
    ``main`` -- ``"sr"`` (the single-rank ``fwd`` route program), ``"f"`` (the ``kcp_fwd`` program) or ``None`` (a rank
    without tokens: no launch); ``form`` -- ``"pair"`` (a main program before the exchange's wait) or ``"wait"``;
    ``repair`` -- whether a history-row program ran.  :func:`_graph_run` adds the graph event."""
    LAST_CALL[kind] = {
        "main": None if sr is None else ("sr" if sr else "f"),
        "form": "pair" if pair else "wait",
        "repair": bool(repair),
    }


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
    cu_seqlens_local_cpu, sr, overlap)`` (:func:`_fwd_host`).  The operand checks, the gather, the exchange, the wait
    and the launches are the untrimmed call's, in the same order on the same stream."""

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
        _fwd_record("fwd", None, False, False)
        return y, hist
    counters = (
        _forward_work(None, e["words"], device=device, device_index=e["device_index"])
        if e["words"]
        else None
    )
    if work is not None and e["overlap"]:
        # the overlapped forward: the main program reads no received row while the exchange is in flight
        _fwd_main(
            e, x_flat, w, y, None, counters, S=S, C=C, device=device, deferred=True
        )
        work.wait()
        repair = _fwd_repair(e, x_flat, w, y, hist, S=S, C=C, device=device)
        _fwd_record("fwd", e["sr"], True, repair)
    else:
        repair = _fwd_launch(
            e, x_flat, w, y, hist, work, S=S, C=C, device=device, counters=counters
        )
        _fwd_record("fwd", e["sr"], False, repair)
    return y, hist


def cake_causal_conv1d_det_train_kcp_backward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
    cooperative: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP backward: ``dx`` of the rank's fragments (the halo contributions of the successor fragments folded in
    before the one BF16 rounding) and the rank's FP32 ``dweight`` partial (fixed order; its sum over the ranks is the
    framework's).

    ``hist`` is the forward's pool.  The head program writes the derivative rows of every continuing fragment's first
    tokens into the backward exchange's send layout, the exchange delivers them to the predecessors as ``dpre_tail``,
    and the main program folds them (``cooperative=False`` at more than one rank: the main program runs while the
    exchange is in flight and the tail-row program folds them after the wait, bitwise the same ``dx`` and
    ``dweight``; the module docstring's "Overlapped backward").  Every rank calls this function once per layer, in
    the same order.  ``graph`` as
    in the forward (the backward's key holds the forward's ``hist`` pointer, so its capture follows the forward's by
    one call).  ``cooperative``: the main program's launch form (``None`` = :data:`COOPERATIVE_DEFAULT`, ``False``;
    ``False`` the ordinary launch and the reduce program, ``True`` the cooperative launch; see the module docstring);
    ``dx`` is bitwise the same in both, ``dweight`` is reproducible in each but differs in its bits between them.
    Every rank of the group passes the same ``cooperative``.
    """

    coop = _cooperative(cooperative)
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_backward(
            x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight, cooperative=coop
        )
    entry = _graph_entry(
        "bwd",
        ctx,
        (x_local, weight, dy, hist, dx, dweight),
        mode,
        variant=_bwd_variant(coop),
    )
    dx_out, dw_out = _graph_run(
        "bwd",
        entry,
        lambda: _kcp_backward(
            x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight, cooperative=coop
        ),
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
    cooperative: bool = COOPERATIVE_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward's host flow (eagerly or inside a capture): see :func:`cake_causal_conv1d_det_train_kcp_backward`."""

    x = _require_tensor(x_local, "x_local")
    device = x.device
    # as in the forward: every call, before the host-cache lookup of the trimmed path
    _wait_ready(ctx)
    if _TEST_KNOBS["host_trim"] == 1:
        return _kcp_backward_trim(
            x,
            weight,
            ctx,
            dy,
            hist,
            dx=dx,
            dweight=dweight,
            device=device,
            cooperative=cooperative,
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
    names = selected_modules(arch, _bwd_kind(cooperative), C=C, S=S)
    record = _loader.MODULES[names[0]]
    scalars = plan(S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True)
    grid_x = _launch_grid(record, device_index=device_index)[0]
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
    overlap = _overlapped(cooperative, work)
    if overlap:
        # the overlapped backward: the main programs read no received row while the exchange is in flight
        values["tail_row_off"] = torch.full(
            (n_frag, HALO), -1, dtype=torch.int32, device=device
        )
    elif work is not None:
        work.wait()
    for name in names:
        # the grid of :func:`_launch_grid` (less the record's reserved SMs): the one the partial rows are sized for
        run_module(
            name,
            values,
            device_index=device_index,
            grid=_launch_grid(_loader.MODULES[name], device_index=device_index),
        )
    reduce = _reduce_plan(record, C, grid_x, device_index)
    if reduce is not None:
        _launch_trim(reduce, {**values, "main_ctas": grid_x}, device_index)
    if overlap:
        work.wait()
        if int(halo.n_tail_rows) > 0:
            (halo_name,) = route_modules(arch, KCP_BWD_HALO)
            _launch_with_grid(
                halo_name,
                {**values, "tail_row_off": tail_row_off},
                grid=_loader.head_grid(_loader.MODULES[halo_name], n_frag=n_frag, C=C),
                device_index=device_index,
            )
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
    cooperative: bool = COOPERATIVE_DEFAULT,
) -> tuple[Any, ...]:
    """The trimmed backward's operand checks, outputs and host-cache entry: ``(S, C, x_flat, w, dy_flat, hist_t,
    dx_out, dw_out, entry)``.  The entry is cached per ``(ctx, S, C, device, cu_seqlens_local_cpu, cooperative)``: the
    tables, the head grid, the backward records of the launch form, plan scalars, partial rows, the loaded modules with
    their argument plans."""

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
        ctx,
        ("bwd", S, C, str(device), tuple(ctx.cu_seqlens_local_cpu), cooperative),
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
            names = selected_modules(arch, _bwd_kind(cooperative), C=C, S=S)
            record = _loader.MODULES[names[0]]
            scalars = plan(
                S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
            )
            grid_x = _launch_grid(record, device_index=device_index)[0]
            if scalars["n_ch_tiles"] > grid_x:
                raise ValueError(
                    f"C={C} spans more channel tiles than the backward's grid on this device holds"
                )
            e["rows"] = partial_rows(grid_x, scalars["n_ch_tiles"])
            e["scalars"] = {**_scalars(scalars), "main_ctas": grid_x}
            e["reduce"] = _reduce_plan(record, C, grid_x, device_index)
            e["launches"] = tuple(
                _launch_plan(
                    name,
                    _launch_grid(_loader.MODULES[name], device_index=device_index),
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
    deferred: bool = False,
) -> None:
    """Wait ``work`` (when given) and launch the main backward programs of a call with ``S > 0`` (``dx`` with the
    received ``dpre_tail`` rows folded in, and the rank's fixed-order FP32 ``dweight`` partial).  ``deferred``: the
    overlapped backward's main launch (:func:`_overlapped`): no wait and every ``tail_row_off`` entry ``-1``, so the
    programs read no received row and write a fragment's last rows as at a document end (:func:`_bwd_tail` writes
    them again after the wait); ``dpre_tail`` is then only a valid pointer."""
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
        "tail_row_off": _no_tail(e, device) if deferred else e["tail_row_off"],
        **e["scalars"],
    }
    if work is not None and not deferred:
        work.wait()
    for info in e["launches"]:
        _launch_trim(info, values, e["device_index"])
    if e["reduce"] is not None:
        _launch_trim(e["reduce"], values, e["device_index"])


# The all ``-1`` ``tail_row_off`` rows of the overlapped backward and ``hist_row_off`` rows of the overlapped forward's
# ``kcp_fwd`` main program on the trimmed host path, per device index: grow-only buffers, none ever freed, shared by
# both (each host-cache entry names its rows under its own key).  A captured graph keeps the pointer of the rows it
# read, and the host cache may drop the entry that named them (:func:`cake_causal_conv1d_det_train_kcp_host_cache_clear`,
# ``HOST_CACHE_MAX``); unlike the context's tables, nothing else holds them.  Every context and stream of the device
# reads them (the host cache is not per stream either), so a new buffer is filled on the device's private
# ``_NO_TAIL_STREAMS`` stream and the host waits for that fill before any call names the buffer: no program on any
# stream reads it unfilled.
_NO_TAIL: dict[int, list[torch.Tensor]] = {}
_NO_TAIL_STREAMS: dict[int, torch.cuda.Stream] = {}


def _no_rows(e: dict[str, Any], device: torch.device, key: str) -> torch.Tensor:
    """An all ``-1`` int32 ``[n_frag, 3]`` row-offset table of the host-cache entry, named by ``e[key]``: the first
    ``n_frag`` rows of the device's largest ``_NO_TAIL`` buffer (a new one, at least twice the last, when that is too
    small, filled before this returns), named by the entry at its first overlapped call, then reused.  During a graph
    capture that finds no buffer large enough (the eager first call of a graph key made one) the call's own table,
    filled on the capturing stream as in the untrimmed path."""
    table = e.get(key)
    if table is None:
        n = e["n_frag"]
        idx = e["device_index"]
        bufs = _NO_TAIL.setdefault(idx, [])
        if not bufs or bufs[-1].shape[0] < n:
            if torch.cuda.is_current_stream_capturing():
                return torch.full((n, HALO), -1, dtype=torch.int32, device=device)
            rows = max(n, 2 * bufs[-1].shape[0] if bufs else 64)
            side = _NO_TAIL_STREAMS.get(idx)
            if side is None:
                side = _NO_TAIL_STREAMS[idx] = torch.cuda.Stream(device=device)
            with torch.cuda.stream(side):
                buf = torch.full((rows, HALO), -1, dtype=torch.int32, device=device)
            side.synchronize()
            bufs.append(buf)
        table = e[key] = bufs[-1][:n]
    return table


def _no_tail(e: dict[str, Any], device: torch.device) -> torch.Tensor:
    """The overlapped backward's ``tail_row_off`` of the host-cache entry (:func:`_no_rows`, key ``"no_tail"``)."""
    return _no_rows(e, device, "no_tail")


def _no_hist(e: dict[str, Any], device: torch.device) -> torch.Tensor:
    """The overlapped forward's ``hist_row_off`` of the ``kcp_fwd`` main program (:func:`_no_rows`, key
    ``"no_hist"``)."""
    return _no_rows(e, device, "no_hist")


def _bwd_tail(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    dx_out: torch.Tensor,
    dpre_tail: torch.Tensor,
    *,
    S: int,
    C: int,
    device: torch.device,
) -> None:
    """The overlapped backward's tail-row program, after the exchange's wait: ``dx`` of the last ``min(3, len)`` rows
    of every fragment with a tail row, written again with the received ``dpre_tail`` rows folded in (nothing when the
    rank receives no tail row).  Its launch plan is made at the entry's first overlapped call, then reused."""
    if e["tail_rows"] == 0:
        return
    info = e.get("tail_launch")
    if info is None:
        (name,) = route_modules(_device_arch(device), KCP_BWD_HALO)
        grid = _loader.head_grid(_loader.MODULES[name], n_frag=e["n_frag"], C=C)
        info = e["tail_launch"] = _launch_plan(name, grid, e["device_index"])
    _launch_trim(
        info,
        {
            "x": x_flat,
            "w": w,
            "cu": e["cu"],
            "dy": dy_flat,
            "dx": dx_out.view(S, C),
            "hist": hist_t,
            "hist_row_off": e["hist_row_off"],
            "dpre_tail": dpre_tail,
            "tail_row_off": e["tail_row_off"],
            "C": C,
            "num_docs": e["n_frag"],
        },
        e["device_index"],
    )


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
    cooperative: bool = COOPERATIVE_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trimmed host path: :func:`_kcp_backward` with the context-derived host work cached (:func:`_bwd_host`); the
    operand checks, allocations, exchange, wait and launches are the untrimmed call's."""

    S, C, x_flat, w, dy_flat, hist_t, dx_out, dw_out, e = _bwd_host(
        x,
        weight,
        ctx,
        dy,
        hist,
        dx=dx,
        dweight=dweight,
        device=device,
        cooperative=cooperative,
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
    overlap = _overlapped(cooperative, work)
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
        deferred=overlap,
    )
    if overlap:
        work.wait()
        _bwd_tail(
            e, x_flat, w, dy_flat, hist_t, dx_out, dpre_tail, S=S, C=C, device=device
        )
    return dx_out, dw_out


class CakeCausalConv1dDetKcpFunction(torch.autograd.Function):
    """Autograd wrapper: the forward saves ``(x_local, weight)`` and the history pool ``hist`` (no second exchange in
    the backward); the backward runs the head program, the backward halo exchange and the main backward program.
    ``handle``: a recompute handle (see :class:`CakeCausalConv1dDetKcpRecomputeHandle`): the original forward stores ``hist`` in it, the
    recompute binds it and exchanges nothing; the saved set is the same three tensors either way.  A handle is
    refused with graph replay (keyword or knob) and inside a CUDA-graph capture.  ``cooperative``: the main backward
    program's launch form (a bool, see :func:`cake_causal_conv1d_det_train_kcp_backward`)."""

    @staticmethod
    def forward(
        fn_ctx, x_local, weight, ctx, graph=None, handle=None, cooperative=None
    ):
        if handle is None:
            y, hist = cake_causal_conv1d_det_train_kcp_forward(
                x_local, weight, ctx, graph=graph
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
        fn_ctx.cooperative = cooperative
        return y

    @staticmethod
    def backward(fn_ctx, grad_y):
        x_local, weight, hist = fn_ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x_local)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_det_train_kcp_backward(
            x_local,
            weight,
            fn_ctx.kcp,
            dy,
            hist,
            graph=fn_ctx.graph,
            cooperative=fn_ctx.cooperative,
        )
        return dx, dweight, None, None, None, None


def cake_causal_conv1d_det_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
) -> torch.Tensor:
    """Autograd-enabled KCP packed width-4 depthwise causal convolution + SiLU of the rank's fragments.

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous activations of the rank's fragments, back to back in the
        order of ``ctx.cu_seqlens_local``.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current token.
    ctx : KCP context
        The rank's KCP context (built by the calling family's KCP context builder): fragment table, halo
        plan, process group.  Every rank of the group calls this function once per layer, in the same order.
    graph : str, optional
        ``None`` (the shipped default ``GRAPH_DEFAULT`` = ``"off"``, the eager path) or ``"replay"``: keyed CUDA-graph
        capture / replay of this call.  The first call of a validity key (this
        context, its world size, and the shape / stride / dtype / data pointer of ``x_local`` and ``weight``) runs
        eagerly, the second is captured, every later one replays the graph (bitwise the eager result).  Requirements:
        pointer-stable operands and the same KCP context per layer, outputs consumed before the next call of the
        same layer (a replay overwrites the graph's static tensors); any other key runs eagerly (``GRAPH_STATS``,
        ``LAST_CALL``).  Each cached key pins a private memory pool (at most ``GRAPH_CACHE_MAX`` keys); call
        :func:`cake_causal_conv1d_det_train_kcp_graph_cache_clear` before ``torch.distributed.destroy_process_group()``.
    recompute_handle : CakeCausalConv1dDetKcpRecomputeHandle, optional
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
    cooperative : bool, optional
        The launch form of the backward's main program (the forward is the same either way).  ``None`` (default):
        :data:`COOPERATIVE_DEFAULT`, ``False``.  ``False``: an ordinary launch one SM short of co-resident plus a
        fixed-order reduce program on the same stream; it starts beside a small kernel of another stream.
        ``True``: one cooperative launch; it starts only when every CTA fits at once, so it waits for any kernel of
        another stream that occupies an SM.  ``dx`` is bitwise the same in both forms.  ``dweight`` is reproducible
        and within the same FP64-oracle bound in each form, but its bits differ between the forms (each form sums
        its own partial rows), so keep one form for runs whose ``dweight`` bits must match.  Every rank of
        ``ctx.group`` passes the same value (one launch form per layer of the group; the form changes no
        collective, so a mismatch does not hang, but the ranks' ``dweight`` partials then mix two reduction
        orders).

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape, with the three document tokens before every continuing fragment as
        history (zero history at a document's start).  The backward returns BF16 ``dx`` (the successor fragments'
        contributions folded in before the one rounding) and the rank's FP32 ``dweight`` partial.
    """

    _graph_mode(graph)  # validate on the caller's frame
    coop = _cooperative(cooperative)
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    # Always six inputs: the backward returns six gradients.
    return CakeCausalConv1dDetKcpFunction.apply(
        x_local, weight, ctx, graph, recompute_handle, coop
    )


# ----------------------------------------------------------------------------------------------------------------
# Several convolutions of one layer with one halo exchange per direction (FlashInfer issue #7815): the q / k / v
# convolutions of a layer share the context, so their history rows (forward) and head rows (backward) travel in ONE
# ``all_to_all_single`` instead of one per tensor.  The send rows of the tensors are packed side by side
# (``[n_rows, sum C_i]``, the context's splits unchanged), and after the wait each tensor's rows are copied out into
# a fresh contiguous pool -- the forward's ``hist`` ``[n_hist_rows, C_i]`` (the saved tensor) and the backward's
# ``dpre_tail`` ``[n_tail_rows, C_i]`` -- before that tensor's kernels launch with the single call's arguments.  The
# exchange and the unpacking are copies, and each tensor keeps its own kernels, partial workspace and fixed
# reduction order, so every output and gradient (``dweight`` included) is bitwise that of one single call per tensor.
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


def cake_causal_conv1d_det_train_kcp_multi_forward(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    outs: Optional[Any] = None,
    graph: Optional[str] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP forward of several convolutions of one layer with one halo exchange: ``(ys, hists)``, bitwise
    ``cake_causal_conv1d_det_train_kcp_forward(xs[i], weights[i], ctx)`` per tensor.

    ``xs`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous tensors of the rank's fragments (the channel counts
    may differ); ``weights`` FP32 ``[C_i, 4]``; ``outs`` optional preallocated outputs.  Every rank of ``ctx.group``
    calls this function once per layer, in the same order, with the same number of tensors and the same channel
    counts: the one forward exchange is a collective.  ``hists[i]`` is a fresh contiguous ``[n_hist_rows, C_i]``
    BF16 pool (the saved tensor of tensor ``i``).  ``graph`` as in :func:`cake_causal_conv1d_det_train_kcp_forward` (the
    validity key holds every operand).
    """

    xs, weights, outs = _multi_operands(xs, weights, outs)
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_forward_multi(xs, weights, ctx, outs=outs)
    entry = _graph_entry(
        "fwd_multi",
        ctx,
        (*xs, *weights, *outs),
        mode,
        variant=_fwd_variant(xs[0], multi=True),
    )
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
    exchange, one wait, the per-tensor unpack copies and the single call's forward programs per tensor in the waiting
    form; in the overlapped forward (``MULTI_FWD_OVERLAP_ARCHS`` or the ``fwd_overlap`` knob, an exchange in flight)
    the first tensor's main program runs before the wait and its history-row program after it.  ``_hists``: the pools
    a recompute handle stored (checked by :func:`_handle_hists`): no gather, no exchange, no wait."""

    device = _require_tensor(xs[0], "xs[0]").device
    _wait_ready(ctx)
    calls = [
        _fwd_host(x, w, ctx, out=o, device=device, multi=True)
        for x, w, o in zip(xs, weights, outs, strict=True)
    ]
    S = calls[0][0]
    widths = [c[1] for c in calls]
    # one counter buffer for every tensor's single-rank main program (launched in stream order), sized once
    counters = (
        _forward_work(
            None,
            max(c[5]["words"] for c in calls),
            device=device,
            device_index=calls[0][5]["device_index"],
        )
        if S > 0
        else None
    )
    work = None
    overlap = False
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
        overlap = S > 0 and work is not None and e0["overlap"]
        if overlap:
            # the first tensor's main program while the exchange is in flight; the others wait, so that the call adds
            # the history-row program of one tensor, not of every tensor
            S0, C0, x0, w0, y0, _ = calls[0]
            _fwd_main(
                e0, x0, w0, y0, None, counters, S=S0, C=C0, device=device, deferred=True
            )
        if work is not None:
            work.wait()
        hists = _unpack(recv, widths) if len(calls) > 1 else [recv]
    ys = tuple(c[4] for c in calls)
    if S == 0:
        _fwd_record("fwd_multi", None, False, False)
        return ys, tuple(hists)
    repair = False
    for i, ((S_i, C, x_flat, w, y, e), hist) in enumerate(
        zip(calls, hists, strict=True)
    ):
        if overlap and i == 0:
            ran = _fwd_repair(e, x_flat, w, y, hist, S=S_i, C=C, device=device)
        else:
            ran = _fwd_launch(
                e,
                x_flat,
                w,
                y,
                hist,
                None,
                S=S_i,
                C=C,
                device=device,
                counters=counters,
            )
        repair = repair or ran
    _fwd_record("fwd_multi", calls[0][5]["sr"], overlap, repair)
    return ys, tuple(hists)


def cake_causal_conv1d_det_train_kcp_multi_backward(
    xs: Any,
    weights: Any,
    ctx: Any,
    dys: Any,
    hists: Any,
    *,
    dxs: Optional[Any] = None,
    dweights: Optional[Any] = None,
    graph: Optional[str] = None,
    cooperative: Optional[bool] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP backward of several convolutions of one layer with one halo exchange: ``(dxs, dweights)``, bitwise
    ``cake_causal_conv1d_det_train_kcp_backward(xs[i], weights[i], ctx, dys[i], hists[i])`` per tensor.

    The head programs of every tensor run first, their send rows travel packed in one exchange, and after the wait
    each tensor's received rows are copied out and folded by that tensor's main backward programs (own partial
    workspace, fixed order); in the overlapped backward (``cooperative=False`` at more than one rank) the first
    tensor's main programs run while the exchange is in flight and its tail-row program folds its copied-out rows
    after the wait, before the other tensors' main programs (one tail-row program per call, as in the single
    entry).  ``hists`` are the forward's pools.  Every rank calls this function once per layer, in
    the same order.  ``graph`` and ``cooperative`` (one launch form for every tensor) as in
    :func:`cake_causal_conv1d_det_train_kcp_backward`.
    """

    xs, weights, _ = _multi_operands(xs, weights, None)
    n = len(xs)
    dys, hists = list(dys), list(hists)
    dxs = [None] * n if dxs is None else list(dxs)
    dweights = [None] * n if dweights is None else list(dweights)
    if not (len(dys) == len(hists) == len(dxs) == len(dweights) == n):
        raise ValueError(
            f"dys, hists, dxs and dweights must hold one entry per convolution ({n})"
        )
    coop = _cooperative(cooperative)
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_backward_multi(
            xs, weights, ctx, dys, hists, dxs, dweights, cooperative=coop
        )
    entry = _graph_entry(
        "bwd_multi",
        ctx,
        (*xs, *weights, *dys, *hists, *dxs, *dweights),
        mode,
        variant=_bwd_variant(coop),
    )
    dx_out, dw_out = _graph_run(
        "bwd_multi",
        entry,
        lambda: _kcp_backward_multi(
            xs, weights, ctx, dys, hists, dxs, dweights, cooperative=coop
        ),
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
    cooperative: bool = COOPERATIVE_DEFAULT,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The multi backward's host flow (eagerly or inside a capture): see
    :func:`cake_causal_conv1d_det_train_kcp_multi_backward`."""

    device = _require_tensor(xs[0], "xs[0]").device
    _wait_ready(ctx)
    calls = [
        _bwd_host(
            x, w, ctx, dy, h, dx=dx, dweight=dw, device=device, cooperative=cooperative
        )
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
    recv = None
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
    dx_out = tuple(c[6] for c in calls)
    dw_out = tuple(c[7] for c in calls)
    if calls[0][0] == 0:
        if work is not None:
            work.wait()
        for dw in dw_out:
            dw.zero_()
        return dx_out, dw_out
    overlap = (
        _overlapped(cooperative, work)
        and _device_arch(device) in MULTI_BWD_OVERLAP_ARCHS
    )
    if overlap:
        # the first tensor's main programs while the exchange is in flight (its head pool: a valid, unread
        # dpre_tail); the others wait, so that the call adds one tail-row program, not one per tensor
        S, C, x_flat, w, dy_flat, hist_t, dx, dw, e = calls[0]
        _bwd_main(
            e,
            x_flat,
            w,
            dy_flat,
            hist_t,
            dx,
            dw,
            heads[0],
            None,
            S=S,
            C=C,
            device=device,
            deferred=True,
        )
    if work is not None:
        work.wait()
    if recv is not None:
        tails = _unpack(recv, [c[1] for c in calls]) if len(calls) > 1 else [recv]
    for i, ((S, C, x_flat, w, dy_flat, hist_t, dx, dw, e), tail) in enumerate(
        zip(calls, tails, strict=True)
    ):
        if overlap and i == 0:
            _bwd_tail(e, x_flat, w, dy_flat, hist_t, dx, tail, S=S, C=C, device=device)
        else:
            _bwd_main(
                e,
                x_flat,
                w,
                dy_flat,
                hist_t,
                dx,
                dw,
                tail,
                None,
                S=S,
                C=C,
                device=device,
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
            f"cake_causal_conv1d_det_kcp or of another tensor count); one handle serves one call"
        )
    if entry.get("ctx") is not ctx:
        raise ValueError(
            "recompute_handle: the 'conv' entry belongs to another KCP context"
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


class CakeCausalConv1dDetKcpMultiFunction(torch.autograd.Function):
    """Autograd wrapper of several convolutions with one halo exchange per direction: ``apply(ctx, graph, handle,
    n, *xs, *weights[, cooperative])`` returns the ``n`` outputs; the forward saves ``(*xs, *weights, *hists)``; the
    backward runs :func:`cake_causal_conv1d_det_train_kcp_multi_backward` (the optional trailing ``cooperative``: its launch
    form, ``None`` or a bool).  ``handle``: one recompute handle for the call, whose ``"conv"``
    entry holds the ``n`` pools (the saved set's own tensors): the recompute binds them and exchanges nothing.
    Refused with graph replay (keyword or knob) and inside a CUDA-graph capture, like the single call."""

    @staticmethod
    def forward(fn_ctx, ctx, graph, handle, n, *tensors):
        if len(tensors) not in (2 * n, 2 * n + 1):
            raise ValueError(
                f"expected {n} inputs, {n} weights and an optional cooperative, got {len(tensors)} values"
            )
        cooperative = tensors[2 * n] if len(tensors) == 2 * n + 1 else None
        xs, weights = list(tensors[:n]), list(tensors[n : 2 * n])
        if handle is None:
            ys, hists = cake_causal_conv1d_det_train_kcp_multi_forward(
                xs, weights, ctx, graph=graph
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
        fn_ctx.cooperative = cooperative
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
        dxs, dweights = cake_causal_conv1d_det_train_kcp_multi_backward(
            xs,
            weights,
            fn_ctx.kcp,
            dys,
            hists,
            graph=fn_ctx.graph,
            cooperative=fn_ctx.cooperative,
        )
        # one gradient per input; a trailing ``cooperative`` gets ``None`` (autograd accepts the extra ``None``)
        return (None, None, None, None, *dxs, *dweights, None)


def cake_causal_conv1d_det_kcp_multi(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled KCP convolutions of several tensors of one layer (for example its q, k and v) with one halo
    exchange per direction.

    Bitwise ``tuple(cake_causal_conv1d_det_kcp(x, w, ctx) for x, w in zip(xs, weights))`` -- outputs and every gradient,
    ``dweight`` included (each tensor keeps its own kernels and fixed-order partial) -- with one collective in the
    forward and one in the backward instead of one per tensor: fewer global synchronization points per layer.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous activations of the rank's fragments, in the
        order of ``ctx.cu_seqlens_local``; the channel counts may differ.
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : KCP context
        The rank's context, as in :func:`cake_causal_conv1d_det_kcp`.  Every rank of the group calls this function once
        per layer, in the same order, with the same tensor count and channel counts.  The ranks are not compared
        (that would cost a collective): a different count, channel count or order hangs the exchange or silently
        corrupts the history rows, as a mismatch of the single entry's per-rank contract does.
    graph : str, optional
        As in :func:`cake_causal_conv1d_det_kcp` (keyed CUDA-graph capture / replay; the key holds every operand).
    recompute_handle : CakeCausalConv1dDetKcpRecomputeHandle, optional
        ``None`` (default): unchanged.  One handle for the whole call: the original forward stores the ``n`` history
        pools under ``"conv"``, torch's recompute inside the backward binds them and exchanges nothing; the backward
        still exchanges once.  Same rules as :func:`cake_causal_conv1d_det_kcp` (one handle per call and micro-step,
        created outside the outermost checkpoint region; refused with ``graph="replay"`` and inside a CUDA-graph
        capture; every rank passes one or none).  The handle may also carry the ``"kda"`` entry of the layer's
        recurrence; a handle's ``"conv"`` entry serves either one multi call or one single call, never both.
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_det_kcp` (default ``False``, the same value on every rank); one launch form
        for every tensor of the call.  The result is bitwise the single calls with the same ``cooperative``.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    _graph_mode(graph)
    coop = _cooperative(cooperative)
    xs, weights, _ = _multi_operands(xs, weights, None)
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    return CakeCausalConv1dDetKcpMultiFunction.apply(
        ctx, graph, recompute_handle, len(xs), *xs, *weights, coop
    )


__all__ = [
    "COOPERATIVE_DEFAULT",
    "GRAPH_MODES",
    "GRAPH_DEFAULT",
    "GRAPH_STATS",
    "HANDLE_KEY",
    "CakeCausalConv1dDetKcpRecomputeHandle",
    "LAST_CALL",
    "cake_causal_conv1d_det_train_kcp_graph_cache_clear",
    "cake_causal_conv1d_det_train_kcp_host_cache_clear",
    "CakeCausalConv1dDetKcpFunction",
    "cake_causal_conv1d_det_kcp",
    "cake_causal_conv1d_det_train_kcp_backward",
    "cake_causal_conv1d_det_train_kcp_forward",
    "CakeCausalConv1dDetKcpMultiFunction",
    "cake_causal_conv1d_det_kcp_multi",
    "cake_causal_conv1d_det_train_kcp_multi_backward",
    "cake_causal_conv1d_det_train_kcp_multi_forward",
]
