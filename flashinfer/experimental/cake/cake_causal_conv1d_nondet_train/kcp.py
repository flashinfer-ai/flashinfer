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
rows' ``dx`` before the one BF16 rounding.
``cake_causal_conv1d_nondet_kcp(x_local, weight, ctx)`` is the autograd entry;
its stable entry points are the non-deterministic package's KCP facade
:mod:`flashinfer.cake_causal_conv1d_nondet_train_kcp`.  This module imports
only ``hashlib``, ``weakref``, torch, torch.distributed, tvm_ffi and its own package.

The host flow mirrors the deterministic package's
(``flashinfer.cake_causal_conv1d_det_train_kcp``): the same context
contract (the route reads ``group``, ``world_size``, ``cu_seqlens_local``,
``cu_seqlens_local_cpu``, ``halo``, ``halo_tables[name]``, ``n_frag``,
``T_r``, ``conv_kernel_size`` and ``wait_ready()``; the numerical mode is
never stored in the context), FP32 taps only, no bias, the legacy channel tile
rule ``cake_jit.plan(..., legacy=True)``, the same forward and head programs
(this package's copies of the det programs: the KCP forward, its history-row
program and the head program), the same two-launch forward (below) and the
same exchanges, recompute handle, graph replay and multi-tensor calls.  The
difference is the backward after the head program, which overlaps the halo
exchange (:func:`_bwd_overlap`):

1. the exchange is issued;
2. the main program (registry ``kcp_jit``, kind ``kcp_bwd_dyn``) runs before
   the exchange is waited, with every ``tail_row_off`` entry ``-1`` (no tail
   row arrived).  Its CTAs take units of tokens from lanes and, through Cluster
   Launch Control, the lanes of CTAs that have not started, so a kernel resident
   on another stream leaves no work waiting for the SMs it holds; their FP32
   ``dweight`` runs are added into four FP32 accumulator copies in the order the
   runs end, summed once per channel tile;
3. after the wait, when the rank receives tail rows, the tail-row program
   (kind ``kcp_bwd_halo``) writes ``dx`` of the last three rows of each
   fragment that has them, from the received rows.

With one rank every launch is on the caller's stream: the main program writes
every row (its ``skip_row_off`` table all ``-1``) and the tail-row program
writes the tail rows again after it.  With more than one rank the head
program(s), the multi call's concatenation of the send rows, the exchange
issue, the wait and the tail-row program(s) run on the device's high-priority
side stream (:func:`_side_stream`), forked from the caller's stream, and the
main program(s) launch on the caller's stream at once (they read no head row)
with the context's ``tail_row_off`` as their ``skip_row_off``: they leave the
last rows of every fragment with a tail row to the tail-row programs, so the
two write disjoint rows and the tail-row programs run as soon as the exchange
arrives, beside the main programs, instead of after them.  The caller's stream
joins the side stream after the main programs (also when a launch raises).
The buffers the side stream fills are allocated on the caller's stream before
the fork, so the join orders their reuse (no ``record_stream``).  After the
wait, the tail-row programs of a multi call whose tensors share one width read
their rows in place in the received pool and run concurrently
(:func:`_halos_in_place`).  The same programs read the same inputs and every
``dx`` row is written by the program that writes it with one rank, so no bit
changes.

The plan of the main program is :func:`kcp_jit.dyn_plan` (units of ``M``
tokens, ``P`` lanes per 2048-channel tile, one CTA per SM).  A call that
reserves SMs (``bwd_reserve_sms >= 1``), has more than
:data:`kcp_jit.DYN_MAX_CHANNELS` channels or no token runs the one-program
backward (kind ``kcp_bwd``) after the wait instead: one ordinary launch whose
CTAs add their FP32 ``dweight`` CTA rows into the zeroed FP32 accumulators
``acc`` (``[C, 4]``) with unordered global reductions; the last CTA of each
channel tile (a ticket on its ``work`` counter) stores the rank-local FP32
``dweight`` partial and zeroes the tile's accumulators and counter.  Both
programs' workspaces are the package's cached backward workspace of the device
and current stream (zero before and after every launch; inside a CUDA Graph
capture with a nonzero capture id one zeroed buffer per capture and stream,
shared by the capture's backward calls on that stream, with another fill node
when a call needs a larger buffer, and kept alive after its graph is freed
until a later capture on that stream replaces it; with capture id 0, also
without ``cuda.bindings``, each captured call zeroes a buffer of its own).

Two-launch forward (the deterministic KCP package's forward, FlashInfer #9225): the forward can run as a main
program that reads no received row and the ``kcp_fwd_halo`` history-row program.  The main program writes every row,
a continuing fragment's first rows as at a document start: either this package's zero-history registry's single-rank
``fwd`` route program (``cake_jit.selected_modules(arch, "fwd", C=C, S=S)``, the program the single-rank forward
runs) on the rank's fragments, each a document of its own (``cu = ctx.cu_seqlens_local``, ``num_docs = n_frag``; the
record's full co-resident grid and token-chunk counters as in the single-rank forward), or, in the overlapped form
only, the ``kcp_fwd`` program with every ``hist_row_off`` entry ``-1``.  The history-row program then writes ``y`` of
the first ``min(3, len)`` rows of every fragment with a history row again from the received pool, with the
``kcp_fwd`` program's instructions, so ``y`` is bitwise the one-launch forward's; it runs only when the rank receives
history rows.  With an exchange in flight (more than one rank) the overlapped form launches the main program before
the exchange's wait and the history-row program after it.  The waiting form launches after the wait: the single-rank
main program and the history-row program, or, with the ``kcp_fwd`` program as main, the one-launch forward (the
``kcp_fwd`` program with the context's table, no history-row program).  ``FWD_MAIN_ARCHS`` names the architectures
whose forward runs the single-rank main program (else the ``kcp_fwd`` program), ``FWD_OVERLAP_ARCHS`` those whose
forward overlaps; ``MULTI_FWD_MAIN_ARCHS`` and ``MULTI_FWD_OVERLAP_ARCHS`` are the multi entry's, which overlaps its
first tensor only (the other tensors take the waiting form after the wait).  The sets are the deterministic
package's (the same programs; their comment): every architecture's single call runs the single-rank main program
after the wait, and so does the multi entry on sm_100a and sm_103a; on sm_107a the multi entry overlaps.  The
exchange, its wait and the collectives are the same in every form.  The private test knobs ``fwd_main`` and
``fwd_overlap`` (``None`` shipped = the architecture sets, ``0`` = off, ``1`` = on) select the forms for the A/B
record, and ``LAST_CALL["fwd"]`` / ``LAST_CALL["fwd_multi"]`` record the form a call took (:func:`_fwd_record`).

Numerics: ``y`` and ``hist`` are the det KCP forward's bits; ``dx`` is the det
KCP backward's bits on both backward forms (same per-thread FP32 steps, same
tail fold, one BF16 rounding); ``dweight`` is the rank's FP32 partial, within
the det program's budget of an FP64 oracle (``5e-2 + 1e-4 |dw|``) but not
bitwise reproducible run to run (the cross-CTA sum is unordered; the only
relaxed property).  The
cross-rank reduction of ``dweight`` stays the framework's.

Numerical mode: the package is chosen by its module name and no entry takes a
mode keyword (the deterministic KCP package is
``flashinfer.cake_causal_conv1d_det_train_kcp``).  With
``torch.use_deterministic_algorithms(True)`` the backward entries (hence the
autograd backward) raise PyTorch's non-determinism alert (a ``RuntimeError``,
or a ``UserWarning`` with ``warn_only=True``, after which they run), reading
the setting when the backward runs, and never fall back.  The forward entries
do not alert.

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
:func:`cake_causal_conv1d_nondet_train_kcp_host_cache_clear` empties it; ``_FFI_DEVICES`` caches the FFI device per
device index.  The operand checks, the gather, the exchange, its wait and the
kernel launches run as before with the same arguments, so no bit changes; a
replayed graph (``graph="replay"``) contains no host work.  The trimmed path
and the multi path issue an overlapped backward's head programs and exchange
from the side stream (above); the previous host path keeps every launch on the
caller's stream.  ``_set_test_knobs({"host_trim": 0})`` (a private test hook;
the backend reads no environment variable) selects the previous host path for
the A/B record.

Backward grid beside a concurrent kernel (FlashInfer issue #8365): the main
backward launches :func:`_bwd_grid`, ``ctas_per_sm x SM count`` CTAs less
``ctas_per_sm`` when the last channel-tile rank of that grid holds at least
``ctas_per_sm`` CTAs.  A kernel on another stream that keeps one SM's shared
memory leaves room for ``ctas_per_sm`` fewer CTAs; with the full grid the top
CTAs, each carrying a whole token stream, then run in a second wave.  The
program derives its token streams and ticket count from ``gridDim``, so ``dx``
keeps its bits and ``dweight`` sums the same terms in other partial sums.

Opt-in SM reserve of the main backward (keyword ``bwd_reserve_sms``, default
``None`` = ``RESERVE_DEFAULT`` = 0, the grid above): ``bwd_reserve_sms=K >= 1``
launches the single-wave grid less ``K x ctas_per_sm`` CTAs, whatever the last
channel-tile rank holds, so ``K`` SMs held by kernels of other streams leave the
backward one wave.  It changes the launch grid only (same programs, ``dx``
bitwise, ``dweight`` within the same budget); the host cache and graph keys hold
``K``.  A multi-wave grid (``grid_mult`` > 1) launches unchanged.  On sm_107a
``K >= 1`` raises ``ValueError`` (:data:`RESERVE_UNSUPPORTED`): there the reserved
grid costs up to 1.055x the shipped grid unloaded (C 1024, 62585 tokens).

Checkpoint recompute without the halo exchange (FlashInfer issues #7815 /
#7818): with ``recompute_handle=``
(:class:`CakeCausalConv1dNondetKcpRecomputeHandle`, or any object with an
``entries`` dict) the original forward exchanges as always and stores
its history pool ``hist`` (the saved set's own tensor, key ``"conv"``) in the
handle; torch's recompute of that forward inside the backward binds it and
launches the same forward programs on it (the two-launch forward in its waiting form) -- no gather, no collective,
no wait --
so ``y`` and the saved set ``(x_local, weight, hist)`` are bitwise those of the
exchanging recompute.  The backward is unchanged (it still exchanges the head
rows).  The forward exchange is a collective: every rank of ``ctx.group`` passes
a handle for the same call or none, created at the same checkpoint nesting
level (no check is made; a mismatch hangs or corrupts data).

Several convolutions of one layer (FlashInfer issue #7815): ``cake_causal_conv1d_nondet_kcp_multi(xs, weights, ctx)``
runs the convolutions of ``n`` tensors that share the context (a layer's q, k and v) with one halo exchange per
direction instead of ``n``: the tensors' send rows travel side by side in one ``all_to_all_single``; after the wait,
when the backward runs the overlapped pair with more than one rank and all the tensors share one width, the tail-row
programs read their rows in place in the received pool, and otherwise each tensor's rows are copied out into its own
contiguous pool; each tensor then runs the single call's kernels with the single call's
arguments (the tail-row program with its rows' offsets in the pool), so outputs and ``dx`` are bitwise one single
call per tensor (``dweight`` is each tensor's own unordered partial).
One recompute handle serves the whole call (its ``"conv"`` entry holds the ``n`` pools).  The multi path always
uses the trimmed host path's per-context cache (the ``host_trim`` test knob selects the single call's path only).
"""

from __future__ import annotations

import hashlib
import weakref
from typing import Any, Optional

import torch
import torch.distributed as dist
import tvm_ffi

from . import (
    _accumulator_layout,
    _alert_nondeterministic,
    _backward_accumulators,
    _cu_seqlens,
    _device_arch,
    _device_index,
    _dweight_aligned,
    _forward_work,
    _output,
    _require_tensor,
    _scalars,
    _tokens,
    _weight,
)
from . import cake_jit as _zero_history
from . import kcp_jit as _loader
from .cake_jit import WIDTH, launch_grid, plan
from .kcp_jit import (
    DYN_TILE_WORDS,
    dyn_applies,
    dyn_plan,
    dyn_workspace_layout,
    head_grid,
    load,
    route_modules,
    run_module,
    selected_modules,
)

HALO = WIDTH - 1  # history rows, tail rows and head rows per fragment
HEAD_SLOTS = HALO  # send slots per head row (a head token is consumed by at most three predecessor fragments)
KCP_FWD = "kcp_fwd"
KCP_BWD = "kcp_bwd"
KCP_BWD_HEAD = "kcp_bwd_head"
KCP_BWD_DYN = "kcp_bwd_dyn"
KCP_BWD_HALO = "kcp_bwd_halo"
KCP_FWD_HALO = "kcp_fwd_halo"
# The zero-history registry's forward route of FP32 taps and no bias (the KCP contract): the single-rank main program
# of the two-launch forward (the module docstring).
FWD_MAIN_ROUTE = "fwd"
# SMs of co-resident CTAs the ``kcp_fwd`` program (the deterministic KCP forward program) leaves free, per
# architecture (:func:`_forward_grid`).  sm_107a launches the deterministic KCP forward's grid, one SM short: unloaded it
# measured about 0.3% faster there than the full co-resident grid (R200, 212 SMs, world 4, 16384 tokens x 8192
# channels x 3 tensors per rank; within noise at 32768 x 10240).  sm_100a and sm_103a keep the full grid, which
# measured faster than the reserved one there.  The single-rank main program of the two-launch forward
# (``FWD_MAIN_ARCHS``) is the zero-history registry's record and launches its full co-resident grid, as the
# single-rank forward does; the history-row program launches the head program's grid.  ``y`` does not depend on the
# grid.
FWD_RESERVE_SMS = {"sm_107a": 1}
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
    agree: tuple[str, tuple[int, ...], bool],
    recv: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[Any]]:
    """One ``all_to_all_single`` of ``send`` (rows in destination-rank order) into ``recv`` (a ``[recv_rows, C]``
    tensor; ``None`` = a fresh one), issued asynchronously on the current stream; with one rank the send buffer is
    the received pool (no collective).
    ``agree`` = ``(phase, channels, multi)`` of the call: the first exchange of a (context, phase) runs the
    rank-agreement check first (:func:`_agree`).  Returns ``(recv, work)``; the caller waits ``work`` at the first
    consumer."""

    if int(ctx.world_size) == 1:
        return send, None
    _agree(ctx, *agree, device=send.device)
    if recv is None:
        recv = torch.empty(
            (recv_rows, send.shape[1]), dtype=send.dtype, device=send.device
        )
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
# Rank agreement and one mode per collective (FlashInfer issue #8946): every rank of a call must pass a context built
# from the same partition and call the same entry (this package's nondet entry, single or multi) with the same
# channel counts.  The first eager exchange of each (context, phase) -- phase "fwd" or "bwd" -- all-gathers one int64[8]
# fingerprint per rank over ``ctx.group`` and compares them on the host (the call's only collective beyond the halo
# exchange and its only host sync, once per context and phase); every rank raises the same ``ValueError`` naming the
# ranks and fields that differ.  Later calls, recompute with a handle (no exchange) and graph replays do nothing.  A
# first call inside a CUDA-graph capture raises: run one eager call first.  One rank: no check.  The layout is shared
# with the deterministic KCP package's protocol (field 1 = mode: 1 det, 2 nondet).
#   [0] protocol 0x4B43504650310001; [1] mode (2); [2] phase (1 fwd, 2 bwd); [3] (entry << 32) | n_tensors (entry 1
#   single, 2 multi); [4] channels (C, or for multi the low 63 bits of sha256(",".join(C_i))); [5] (world_size << 8) |
#   conv width; [6] low 63 bits of sha256 of the partition (world size, layout, cu_seqlens, every fragment's doc, off,
#   length, rank, slot); [7] launch form (fwd 0, bwd 1).
AGREE_PROTOCOL = 0x4B43504650310001
AGREE_MODE = 2  # nondet
AGREE_FIELDS = (
    "protocol",
    "mode",
    "phase",
    "entry/tensors",
    "channels",
    "world/width",
    "partition",
    "form",
)
_AGREE_PHASES = {"fwd": 1, "bwd": 2}
_AGREED: "weakref.WeakKeyDictionary[Any, set]" = weakref.WeakKeyDictionary()
_AGREED_PINNED: dict[
    int, tuple[Any, set]
] = {}  # contexts that are not weak-referenceable (pinned)


def _low63(digest: bytes) -> int:
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def _partition_hash(part: Any) -> int:
    text = (
        f"{int(part.world_size)}|{part.layout}|"
        + ",".join(str(int(c)) for c in part.cu_seqlens)
        + "|"
        + ";".join(
            f"{f.doc},{f.off},{f.length},{f.rank},{f.slot}"
            for rank_frags in part.fragments
            for f in rank_frags
        )
    )
    return _low63(hashlib.sha256(text.encode("utf-8")).digest())


def _fingerprint(
    ctx: Any, phase: str, channels: tuple[int, ...], multi: bool
) -> list[int]:
    part = getattr(ctx, "partition", None)
    if part is None:
        raise TypeError(
            "ctx has no `partition`: a multi-rank call needs the context of the KCP plan builder "
            "(the rank-agreement check hashes its partition)"
        )
    if multi:
        width = _low63(
            hashlib.sha256(
                ",".join(str(int(c)) for c in channels).encode("utf-8")
            ).digest()
        )
    else:
        width = int(channels[0])
    return [
        AGREE_PROTOCOL,
        AGREE_MODE,
        _AGREE_PHASES[phase],
        ((2 if multi else 1) << 32) | len(channels),
        width,
        (int(ctx.world_size) << 8) | int(ctx.conv_kernel_size),
        _partition_hash(part),
        0 if phase == "fwd" else 1,
    ]


def _agreed_phases(ctx: Any) -> set:
    try:
        return _AGREED.setdefault(ctx, set())
    except TypeError:
        return _AGREED_PINNED.setdefault(id(ctx), (ctx, set()))[1]


def _agree(
    ctx: Any, phase: str, channels: tuple[int, ...], multi: bool, device: torch.device
) -> None:
    """The rank-agreement check of a (context, phase): see the block comment above."""

    world = int(ctx.world_size)
    if world == 1:
        return
    done = _agreed_phases(ctx)
    if phase in done:
        return
    if torch.cuda.is_current_stream_capturing():
        raise ValueError(
            f"the first {phase} call of a KCP context ran inside a CUDA-graph capture: the rank-agreement check of "
            "#8946 needs one eager call per context and phase before a capture"
        )
    mine = torch.tensor(
        _fingerprint(ctx, phase, channels, multi), dtype=torch.int64, device=device
    )
    every = torch.empty(world * len(AGREE_FIELDS), dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(every, mine, group=ctx.group)
    rows = every.view(world, len(AGREE_FIELDS)).cpu().tolist()
    bad = [
        (
            rank,
            [
                AGREE_FIELDS[i]
                for i, (a, b) in enumerate(zip(row, rows[0], strict=True))
                if a != b
            ],
        )
        for rank, row in enumerate(rows)
        if row != rows[0]
    ]
    if bad:
        detail = "; ".join(
            f"rank {rank} differs from rank 0 in {', '.join(fields)}"
            for rank, fields in bad
        )
        raise ValueError(
            f"KCP rank agreement failed on the first {phase} call of this context (#8946: every rank passes the "
            f"same partition and calls the same entry, mode and channel counts): {detail}"
        )
    done.add(phase)


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
# autograd Function itself (``_handle_refusals``), so ``CakeCausalConv1dNondetKcpFunction.apply`` refuses like the
# wrapper.
# The ranks are not checked against each other: the forward exchange is a collective, so every rank of the group
# must take the same branch (all with a handle at the same nesting level, or all without).

HANDLE_KEY = "conv"  # this package's entry in a recompute handle

_GRAPH_TASK_ID_BINDING = getattr(torch._C, "_current_graph_task_id", None)
_GRAPH_TASK_ID_USABLE: Optional[bool] = None  # None = not checked yet


class CakeCausalConv1dNondetKcpRecomputeHandle:
    """The boundary tensors of one checkpointed call, carried from its original forward to torch's recompute of that
    forward, so the recompute runs without cross-rank communication.

    Create one handle per convolution call and micro-step, outside the outermost checkpoint region, and pass it as an
    argument of the checkpointed function, so the recompute sees the same object::

        def layer(x, weight, handle):
            return cake_causal_conv1d_nondet_kcp(x, weight, ctx, recompute_handle=handle)

        handle = CakeCausalConv1dNondetKcpRecomputeHandle()
        y = torch.utils.checkpoint.checkpoint(layer, x, w, handle, use_reentrant=False)

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
            "recompute_handle must carry an 'entries' dict (CakeCausalConv1dNondetKcpRecomputeHandle), got "
            f"{type(handle).__name__}"
        )


def _handle_lookup(handle: Any) -> Optional[dict[str, Any]]:
    """The handle's ``"conv"`` entry when this forward is the recompute of the forward that stored it, ``None`` for an
    original forward (it exchanges and stores one).  An entry found outside the backward raises."""
    entries = getattr(handle, "entries", None)
    if not isinstance(entries, dict):
        raise TypeError(
            "recompute_handle must carry an 'entries' dict (CakeCausalConv1dNondetKcpRecomputeHandle), got "
            f"{type(handle).__name__}"
        )
    in_backward = _kcp_in_backward()
    entry = entries.get(HANDLE_KEY)
    if entry is None:
        return None
    if not in_backward:
        raise RuntimeError(
            "recompute_handle: the handle already holds a 'conv' entry outside the backward -- a handle serves one "
            "convolution call of one micro-step; create a new CakeCausalConv1dNondetKcpRecomputeHandle per call and "
            "micro-step"
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
            "recompute_handle: the 'conv' entry belongs to another CakeKDADetKcpContext"
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
# The public keyword ``graph="replay"`` of :func:`cake_causal_conv1d_nondet_kcp` and of the forward / backward below
# (shipped ``GRAPH_DEFAULT = "off"``) -- or, for the A/B record, the private ``_set_test_knobs({"graph": "replay"})``
# (a private test hook; a keyword contradicting an active knob is a ``ValueError``) -- makes
# :func:`cake_causal_conv1d_nondet_train_kcp_forward` / :func:`cake_causal_conv1d_nondet_train_kcp_backward` run their
# first call of a validity key (kind, the context object, its world size, and the shape / stride / dtype / data pointer
# of every operand and caller-owned output) eagerly, capture the second call into a ``torch.cuda.CUDAGraph`` (replayed
# once) and replay the graph on every later call of the key: the halo gather, the ``all_to_all_single``, its wait and
# the kernels with the same arguments in the same order (``y`` / ``hist`` / ``dx`` bitwise the eager result; ``dweight``
# within the eager call's budget, summed in an unordered cross-CTA order) and no host work between them.  Any other key
# runs eagerly (silent fallback); a rank without tokens is never captured.  A replayed call returns the graph's static
# tensors (detached): the next replay of the key overwrites them, so a framework must keep its inputs pointer-stable and
# consume the outputs before the next call of the same layer.  This module keeps its own cache (no import of the KDA
# host module); ``GRAPH_STATS`` counts the events and ``LAST_CALL[kind]`` says how the last forward / backward ran
# (keyword or knob).  Every captured graph keeps its fresh tensors in a private pool (one per cached key; at most
# GRAPH_CACHE_MAX keys, the oldest dropped first); ``cake_causal_conv1d_nondet_train_kcp_graph_cache_clear()`` releases them and must run before
# ``torch.distributed.destroy_process_group()`` (a graph holding the NCCL collective blocks ``ncclCommDestroy``).

# Trimmed host path; shipped default:
# 1 = the trimmed host path of the eager halo call (same launches, bitwise; the module docstring's "Shipped host path"),
# 0 = the previous host path, selectable through _set_test_knobs for the A/B record.
HOST_TRIMS = (0, 1)
HOST_TRIM_DEFAULT = (
    1  # shipped; 0 = the previous path, via _set_test_knobs({"host_trim": 0}) only
)
# The backward's route: ``"overlap"`` shipped (:func:`_overlapped`) | ``"single"`` the ``kcp_bwd`` program after the
# exchange's wait for every call, via ``_set_test_knobs({"bwd_route": "single"})`` only.
BWD_ROUTES = ("overlap", "single")
BWD_ROUTE_DEFAULT = "overlap"
# The two-launch forward (the module docstring's "Two-launch forward"): the knob values of ``fwd_main`` (the main
# program: 1 = the single-rank ``fwd`` route program, 0 = the ``kcp_fwd`` program) and ``fwd_overlap`` (1 = the main
# program before the exchange's wait, 0 = after it); ``None``, shipped, = the architecture sets below.
FWD_MAINS = (None, 0, 1)
FWD_MAIN_DEFAULT = None
FWD_OVERLAPS = (None, 0, 1)
FWD_OVERLAP_DEFAULT = None
# The architectures whose single call runs the single-rank main program (``FWD_MAIN_ARCHS``) and launches its main
# program before the exchange's wait (``FWD_OVERLAP_ARCHS``), and the multi entry's (``MULTI_*``).  The deterministic
# KCP package's sets (FlashInfer #9225): this package runs the same forward programs (the zero-history registry's
# ``fwd`` route records carry the deterministic records' build identity digits; the ``kcp_fwd`` and ``kcp_fwd_halo``
# programs are copies of the deterministic ones), so each entry takes the form the deterministic package measured
# faster than the one-launch forward on every forward, forward + backward and checkpointed step row of B200, GB300 and
# R200: the single call waits and runs the single-rank main program on every architecture; the multi entry waits on
# sm_100a and sm_103a and overlaps its first tensor on sm_107a.
FWD_MAIN_ARCHS = frozenset(("sm_100a", "sm_103a", "sm_107a"))
FWD_OVERLAP_ARCHS: frozenset[str] = frozenset()
MULTI_FWD_MAIN_ARCHS = frozenset(("sm_100a", "sm_103a", "sm_107a"))
MULTI_FWD_OVERLAP_ARCHS = frozenset(("sm_107a",))
_TEST_KNOBS: dict[str, Any] = {
    "graph": "off",
    "host_trim": HOST_TRIM_DEFAULT,
    "bwd_route": BWD_ROUTE_DEFAULT,
    "fwd_main": FWD_MAIN_DEFAULT,
    "fwd_overlap": FWD_OVERLAP_DEFAULT,
}
# The opt-in SM reserve of the main backward (keyword ``bwd_reserve_sms``): 0 = the grid of :func:`_bwd_grid`.
RESERVE_DEFAULT = 0
GRAPH_STATS: dict[str, int] = {"eager": 0, "capture": 0, "replay": 0, "fallback": 0}
GRAPH_CACHE_MAX = 64
LAST_CALL: dict[str, dict[str, Any]] = {}

# -- the trimmed host path (host_trim=1): the per-context host state of the eager halo calls.  ``_HOST_CACHE[id(ctx)]`` holds the
# context (pinned, so its id cannot be reused) and one entry per ``(kind, S, C, device, cu_seqlens_local_cpu)`` (the
# forward's also per effective ``(sr, overlap)`` of :func:`_fwd_form`): the
# validated context tables and tensors (``cu``, ``hist_row_off``, ...), the plan scalars, the record names, the loaded
# modules with their positional argument plans and grids.  Everything in an entry derives from host integers of the
# context and the shapes of the call; no tensor content is read.  At most HOST_CACHE_MAX contexts (the oldest dropped).
HOST_CACHE_MAX = 64
_HOST_CACHE: dict[int, tuple[Any, dict[tuple, dict[str, Any]]]] = {}
_FFI_DEVICES: dict[int, Any] = {}
# The overlapped backward's extra streams per device: slot 0 the high-priority side stream of the head programs and
# the exchange issue (:func:`_side_stream`), slots 1.. the multi call's tail-row programs (:func:`_halos_in_place`).
# ``_BWD_STREAMS = False`` keeps every launch on the caller's stream and copies a multi call's received rows (the
# previous order, for the A/B record).
_BWD_STREAMS = True
_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


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


def cake_causal_conv1d_nondet_train_kcp_host_cache_clear() -> int:
    """Drop the trimmed host path's per-context cache; returns the number of contexts dropped."""
    n = len(_HOST_CACHE)
    _HOST_CACHE.clear()
    return n


def _side_stream(
    ctx: Any, device: torch.device, overlapped: bool
) -> Optional[torch.cuda.Stream]:
    """The stream of an overlapped backward's head programs and exchange issue: the device's high-priority side
    stream when the call exchanges (``world_size > 1``) and runs the overlapped pair, else ``None`` (the current
    stream).  The caller allocates the buffers the side stream fills on the current stream, then forks
    (:func:`_fork`)."""
    if not (_BWD_STREAMS and overlapped and int(ctx.world_size) > 1):
        return None
    return _device_stream(device, 0)


def _fork(side: Optional[torch.cuda.Stream], device: torch.device) -> None:
    """The side stream waits for the current stream (after the side stream's buffers were allocated on it)."""
    if side is not None:
        side.wait_stream(torch.cuda.current_stream(device))


def _device_stream(device: torch.device, slot: int) -> torch.cuda.Stream:
    """The device's extra stream ``slot`` (created once): slot 0 at high priority, the others at the default."""
    key = (_device_index(device), slot)
    stream = _STREAMS.get(key)
    if stream is None:
        stream = _STREAMS[key] = torch.cuda.Stream(
            device=device, priority=-1 if slot == 0 else 0
        )
    return stream


def _rejoin(side: Optional[torch.cuda.Stream], device: torch.device) -> None:
    """The current stream waits for the side stream, so that a call raising after the fork (e.g. the rank-agreement
    check inside a capture) leaves no unjoined stream behind."""
    if side is not None:
        torch.cuda.current_stream(device).wait_stream(side)


def _join_side(side: Optional[torch.cuda.Stream], device: torch.device) -> None:
    """After the main program(s), and the exchange's wait and the tail-row program(s) queued on the side stream: the
    current stream waits for the side stream (inside a capture this joins it).  The head pools, the send rows and the
    received pool were allocated on the current stream before the fork and live until after this join, so the
    allocator reuses them only after the side stream's and the collective's work (no ``record_stream``, whose blocks
    the allocator frees only at a later allocation)."""
    if side is not None:
        torch.cuda.current_stream(device).wait_stream(side)


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


def _bwd_grid(
    record: dict[str, Any],
    n_ch_tiles: int,
    device_index: int,
    reserve: int = RESERVE_DEFAULT,
) -> tuple[int, int, int]:
    """The main backward program's launch grid (issue #8365): :func:`launch_grid` of ``record`` on the device, less
    ``ctas_per_sm`` CTAs when that grid is a single wave (``grid_mult`` 1) whose last channel-tile rank holds at
    least ``ctas_per_sm`` CTAs (``grid % n_ch_tiles >= ctas_per_sm``), so the grid fits on one SM fewer; any other
    grid launches unchanged.  The trimmed grid keeps every channel tile's CTAs but one rank's worth at most.

    ``reserve`` (``bwd_reserve_sms``) ``K >= 1``: a single-wave grid less ``K x ctas_per_sm`` CTAs (``ValueError``
    when fewer CTAs than channel tiles remain); a multi-wave grid launches unchanged."""
    grid = launch_grid(record, device_index=device_index)
    rule = record["grid_rule"]
    k = int(rule["ctas_per_sm"])
    single_wave = int(rule["grid_mult"][_loader.device_arch(device_index)]) == 1
    if reserve and single_wave:
        reserved = grid[0] - int(reserve) * k
        if reserved < max(1, int(n_ch_tiles)):
            raise ValueError(
                f"bwd_reserve_sms={reserve} leaves {reserved} CTAs of the backward grid {grid[0]}; "
                f"{int(n_ch_tiles)} channel tiles need at least {max(1, int(n_ch_tiles))}"
            )
        return (reserved, grid[1], grid[2])
    if single_wave and grid[0] % int(n_ch_tiles) >= k:
        return (grid[0] - k, grid[1], grid[2])
    return grid


def _overlapped(C: int, S: int, reserve: int) -> bool:
    """Whether a backward call runs the overlapped pair: ``kcp_bwd_dyn`` launched before the halo exchange's wait
    with every ``tail_row_off`` entry ``-1``, then ``kcp_bwd_halo`` (the rows the exchanged tail rows reach) after
    it.  The shipped route of every call with tokens and at most ``DYN_MAX_CHANNELS`` channels; ``bwd_reserve_sms
    >= 1`` keeps the ``kcp_bwd`` program's reserved grid, so it and a wider ``C`` run that program after the wait
    (as does the ``bwd_route`` test knob ``"single"``)."""
    return _TEST_KNOBS["bwd_route"] == "overlap" and not reserve and dyn_applies(C, S)


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
    """The forward's graph-key value: :func:`_fwd_form` on the device of ``x`` (the effective form, so a changed knob
    or architecture set is another key)."""

    return _fwd_form(_device_arch(_require_tensor(x, "x_local").device), multi=multi)


def _dyn_host(
    arch: str,
    *,
    S: int,
    C: int,
    n_frag: int,
    tail_rows: int,
    device: torch.device,
    device_index: int,
) -> dict[str, Any]:
    """The overlapped backward's host state of a call: the main program's launch plan over :func:`dyn_plan`'s
    ``n_ctas`` CTAs, its work-plan scalars, accumulator layout and tile-word count, the all ``-1`` table it runs with as
    ``tail_row_off`` (and as ``skip_row_off`` when it writes every row), and the tail-row program's launch plan over
    :func:`head_grid` (``None`` when the rank receives no tail row: the main program then writes every row as the
    single program does)."""
    (main,) = route_modules(arch, KCP_BWD_DYN)
    p = dyn_plan(S, C, _loader.sm_count(device_index))
    halo = None
    if tail_rows > 0:
        (halo_name,) = route_modules(arch, KCP_BWD_HALO)
        grid = head_grid(_loader.MODULES[halo_name], n_frag=n_frag, C=C)
        halo = _launch_plan(halo_name, grid, device_index)
    return {
        "main": _launch_plan(main, (p["n_ctas"], 1, 1), device_index),
        "layout": dyn_workspace_layout(_loader.MODULES[main], C, p["n_tiles"], p["L"]),
        "tile_words": DYN_TILE_WORDS * p["n_tiles"],
        "scalars": {key: p[key] for key in ("M", "NL", "P", "L", "n_ctas")},
        "no_tail": torch.full((n_frag, HALO), -1, dtype=torch.int32, device=device),
        "halo": halo,
    }


def _launch_dyn(
    d: dict[str, Any],
    *,
    x_flat: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    hist_row_off: torch.Tensor,
    dx2: torch.Tensor,
    dw_out: torch.Tensor,
    spare: torch.Tensor,
    skip: torch.Tensor,
    S: int,
    C: int,
    n_frag: int,
    device: torch.device,
    device_index: int,
) -> None:
    """The overlapped backward's main program (:func:`_dyn_host` state ``d``), launched while the halo exchange is in
    flight: ``dx`` of every row but the last ``min(3, len)`` rows of each fragment with an entry ``>= 0`` in ``skip``
    (``skip_row_off``: ``d["no_tail"]`` writes every row; the context's ``tail_row_off`` leaves the rows the tail-row
    program writes), a fragment's last rows without the successors' tail rows, as at a document end, and the rank's
    FP32 ``dweight`` partial, summed in the zeroed accumulator copies and words of the device's cached backward
    workspace.  ``spare``: an FP32 ``[*, C]`` tensor passed as ``dpre_tail``, which no ``-1`` entry reads."""
    acc, words = _backward_accumulators(
        None,
        d["layout"],
        device=device,
        device_index=device_index,
        entry="cake_causal_conv1d_nondet_train_kcp_backward",
    )
    k = d["tile_words"]
    lanes = d["scalars"]["L"]
    _launch_trim(
        d["main"],
        {
            "x": x_flat,
            "w": w,
            "cu": cu,
            "dy": dy_flat,
            "dx": dx2,
            "dw": dw_out,
            "acc": acc,
            "tile_tk": words[:k],
            "claims": words[k : k + lanes],
            "departs": words[k + lanes : k + lanes + 1],
            "hist": hist_t,
            "hist_row_off": hist_row_off,
            "dpre_tail": spare,
            "tail_row_off": d["no_tail"],
            "skip_row_off": skip,
            "S": S,
            "C": C,
            "num_docs": n_frag,
            **d["scalars"],
        },
        device_index,
    )


def _launch_halo(
    d: dict[str, Any],
    *,
    x_flat: torch.Tensor,
    w: torch.Tensor,
    cu: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    hist_row_off: torch.Tensor,
    tail_row_off: torch.Tensor,
    dx2: torch.Tensor,
    dpre_tail: torch.Tensor,
    C: int,
    n_frag: int,
    device_index: int,
) -> None:
    """The overlapped backward's tail-row program, after the exchange's wait: ``dx`` of the last ``min(3, len)`` rows
    of every fragment with a tail row, written again with the received ``dpre_tail`` rows folded in (nothing when the
    rank receives no tail row)."""
    if d["halo"] is None:
        return
    _launch_trim(
        d["halo"],
        {
            "x": x_flat,
            "w": w,
            "cu": cu,
            "dy": dy_flat,
            "dx": dx2,
            "hist": hist_t,
            "hist_row_off": hist_row_off,
            "dpre_tail": dpre_tail,
            "tail_row_off": tail_row_off,
            "C": C,
            "num_docs": n_frag,
        },
        device_index,
    )


def _ordinary_launch(cooperative: Any, entry: str) -> None:
    """Validate ``cooperative`` of a call: ``None`` or ``False`` (the main backward is one ordinary launch);
    ``ValueError`` for ``True``, ``TypeError`` for a non-``bool``."""
    if cooperative is None:
        return
    if not isinstance(cooperative, bool):
        raise TypeError(
            f"{entry}: cooperative must be None or a bool, got {type(cooperative).__name__}"
        )
    if cooperative:
        raise ValueError(
            f"{entry}: cooperative=True applies to the deterministic KCP package "
            "flashinfer.cake_causal_conv1d_det_train_kcp only; the non-deterministic main backward is one ordinary "
            "launch (pass cooperative=None or False)"
        )


def _reserve(value: Any, entry: str) -> int:
    """``bwd_reserve_sms`` of a call: ``None`` -> ``RESERVE_DEFAULT``; else a non-negative ``int`` (``TypeError`` for a
    ``bool`` or a non-``int``, ``ValueError`` below zero)."""
    if value is None:
        return RESERVE_DEFAULT
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{entry}: bwd_reserve_sms must be an int or None, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{entry}: bwd_reserve_sms must be >= 0, got {value}")
    return int(value)


# Architectures where ``bwd_reserve_sms >= 1`` is refused, with the reason.
RESERVE_UNSUPPORTED = {
    "sm_107a": (
        "the reserved backward grid costs up to 1.055x the shipped grid unloaded there "
        "(C=1024, 62585 tokens), above the 1.040 limit of this opt-in"
    ),
}


def _reserve_fits(reserve: int, x: Any, entry: str) -> None:
    """``ValueError`` when ``reserve >= 1`` on an architecture of :data:`RESERVE_UNSUPPORTED`, or when ``reserve``
    SMs are not fewer than the SMs of ``x``'s device (checked at the call, before the forward)."""
    if reserve and isinstance(x, torch.Tensor) and x.is_cuda:
        arch = _device_arch(x.device)
        if arch in RESERVE_UNSUPPORTED:
            raise ValueError(
                f"{entry}: bwd_reserve_sms={reserve} is not supported on {arch}: "
                f"{RESERVE_UNSUPPORTED[arch]}; pass bwd_reserve_sms=None"
            )
        sms = int(torch.cuda.get_device_properties(x.device).multi_processor_count)
        if reserve >= sms:
            raise ValueError(
                f"{entry}: bwd_reserve_sms={reserve} must be smaller than the device's {sms} SMs"
            )


def _set_test_knobs(knobs: Optional[dict]) -> dict:
    """Set the module's private test knobs (``graph``: ``"off"`` shipped | ``"replay"``; ``host_trim``: ``1`` shipped =
    the trimmed host path of the eager call | ``0`` the previous host path; ``bwd_route``: ``"overlap"`` shipped |
    ``"single"`` the one backward program after the exchange's wait; ``fwd_main`` / ``fwd_overlap``: ``None`` shipped
    = the architecture sets of the two-launch forward | ``0`` off | ``1`` on, :func:`_fwd_form`); a knob
    the dict does not name keeps its shipped default; ``None`` restores the defaults.  A private test hook: tests call
    it directly and restore the defaults afterwards; the backend reads no environment variable.  Returns the active
    knobs."""
    if knobs is None:
        _TEST_KNOBS["graph"] = "off"
        _TEST_KNOBS["host_trim"] = HOST_TRIM_DEFAULT
        _TEST_KNOBS["bwd_route"] = BWD_ROUTE_DEFAULT
        _TEST_KNOBS["fwd_main"] = FWD_MAIN_DEFAULT
        _TEST_KNOBS["fwd_overlap"] = FWD_OVERLAP_DEFAULT
        return dict(_TEST_KNOBS)
    if not isinstance(knobs, dict):
        raise TypeError(f"_set_test_knobs expects a dict, got {type(knobs).__name__}")
    unknown = sorted(
        set(knobs) - {"graph", "host_trim", "bwd_route", "fwd_main", "fwd_overlap"}
    )
    if unknown:
        raise ValueError(
            f"_set_test_knobs: unknown knob(s) {unknown}; known: ['bwd_route', 'fwd_main', 'fwd_overlap', 'graph', "
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
    bwd_route = knobs.get("bwd_route", BWD_ROUTE_DEFAULT)
    if bwd_route not in BWD_ROUTES:
        raise ValueError(
            f"_set_test_knobs: bwd_route must be one of {BWD_ROUTES}, got {bwd_route!r}"
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
    _TEST_KNOBS["bwd_route"] = str(bwd_route)
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
    extra: tuple = (),
) -> Optional[_GraphEntry]:
    """The cache entry of this call's validity key, or ``None`` when the call is never captured (mode off, no tokens);
    ``mode`` = the effective graph mode (:func:`_graph_mode`), ``None`` = the knob; ``extra`` = further host values
    of the key (the backward's ``bwd_reserve_sms`` when not the default, the forward's :func:`_fwd_form`; a backward
    key also holds the ``bwd_route`` test knob when not the default)."""
    if (
        (_TEST_KNOBS["graph"] if mode is None else mode) == "off"
        or int(getattr(ctx, "T_r", 0)) == 0
        or int(getattr(ctx, "n_frag", 0)) == 0
    ):
        return None
    key: tuple = (kind, id(ctx), int(ctx.world_size), tuple(_sig(t) for t in tensors))
    if extra:
        key += tuple(extra)
    if kind.startswith("bwd") and _TEST_KNOBS["bwd_route"] != BWD_ROUTE_DEFAULT:
        key += (("bwd_route", _TEST_KNOBS["bwd_route"]),)
    e = _GRAPHS.get(key)
    if e is None:
        while len(_GRAPHS) >= GRAPH_CACHE_MAX:
            del _GRAPHS[next(iter(_GRAPHS))]
        e = _GRAPHS[key] = _GraphEntry(key, (ctx,))
    return e


def cake_causal_conv1d_nondet_train_kcp_graph_cache_clear() -> int:
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
            # the graph holds the addresses of the context's cached host-path tensors (tail tables): keep them alive
            # past a host-cache clear or eviction (pins[0] is the context)
            entry.pins += (_HOST_CACHE.get(id(entry.pins[0])),)
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


def cake_causal_conv1d_nondet_train_kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP forward: ``y`` of the rank's fragments, and the history pool ``hist`` it used (saved for the backward).
    ``y`` and ``hist`` are the det KCP forward's bits (no alert under torch's deterministic mode).

    ``x_local`` BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous, the rank's fragments back to back in the order of
    ``ctx.cu_seqlens_local``; ``weight`` FP32 ``[C, 4]``; ``ctx`` a ``CakeKDADetKcpContext`` (its halo plan ``ctx.halo``, the
    device tables ``ctx.halo_tables`` and the process group ``ctx.group``).  Every rank of the group calls this
    function once per layer, in the same order: the forward halo exchange is a collective (a rank without tokens
    joins it with empty splits and returns empty outputs).  ``hist`` is a fresh ``[n_hist_rows, C]`` BF16 tensor
    (``[0, C]`` when the rank receives no row; the program then reads a private one-row placeholder that is not
    returned).  ``graph`` (``None`` = the shipped ``GRAPH_DEFAULT`` ``"off"`` |
    ``"replay"``): keyed CUDA-graph capture / replay of this call (the graph-replay option; see the section comment
    above: pointer-stable operands, private pools, eager fallback, :func:`cake_causal_conv1d_nondet_train_kcp_graph_cache_clear`).
    """

    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_forward(x_local, weight, ctx, out=out)
    entry = _graph_entry(
        "fwd",
        ctx,
        (x_local, weight, out),
        mode,
        extra=(("fwd_form", _fwd_variant(x_local)),),
    )
    y, hist = _graph_run(
        "fwd", entry, lambda: _kcp_forward(x_local, weight, ctx, out=out), mode
    )
    return (y if out is not None else y.detach()), hist


def _forward_grid(
    record: dict[str, Any], *, arch: str, device_index: int, C: int, n_ch_tiles: int
) -> tuple[int, int, int]:
    """The ``kcp_fwd`` program's launch grid: the record's co-resident grid (:func:`launch_grid`) less
    :data:`FWD_RESERVE_SMS` SMs of CTAs for ``arch`` when that shorter grid still holds all ``n_ch_tiles`` channel
    tiles, else the full grid; a ``ValueError`` when not even the full grid holds them.  ``y`` does not depend on the
    grid."""

    full = launch_grid(record, device_index=device_index)
    if n_ch_tiles > full[0]:
        raise ValueError(
            f"C={C} spans more channel tiles than the forward's grid on this device holds"
        )
    reserved = FWD_RESERVE_SMS.get(arch, 0) * int(record["grid_rule"]["ctas_per_sm"])
    if reserved and full[0] - reserved >= max(n_ch_tiles, 1):
        return (full[0] - reserved, full[1], full[2])
    return full


def _sr_main(
    arch: str, *, S: int, C: int, n_frag: int, device_index: int
) -> tuple[str, dict[str, int], tuple[int, int, int], int]:
    """The two-launch forward's single-rank main program of a call: ``(record name, plan scalars, grid, counter
    words)`` of the zero-history registry's ``fwd`` route record ``cake_jit.selected_modules`` picks for ``C`` and
    ``S`` -- the single-rank forward's plan (its role, route and architecture) over the rank's ``n_frag`` fragments as
    documents, its record's full co-resident grid and its token-chunk counter words; a ``ValueError`` when ``C`` spans
    more channel tiles than that grid holds."""
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
    grid = launch_grid(record, device_index=device_index, C=C, S=S)
    if scalars["n_ch_tiles"] > grid[0]:
        raise ValueError(
            f"C={C} spans more channel tiles than the forward's grid on this device holds"
        )
    return (
        name,
        scalars,
        grid,
        _zero_history.counter_words(record, scalars["n_ch_tiles"]),
    )


def _kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
    _hist: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward's host flow (eagerly or inside a capture): see
    :func:`cake_causal_conv1d_nondet_train_kcp_forward`.  ``_hist``: the history pool a recompute handle stored
    (checked by :func:`_handle_hist`): no gather, no exchange, no wait; the programs are the same, launched in the
    waiting form."""

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
            agree=("fwd", (C,), False),
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
        name, scalars, grid, words = _sr_main(
            arch, S=S, C=C, n_frag=n_frag, device_index=device_index
        )
        counters = _forward_work(None, words, device=device, device_index=device_index)
        if counters is not None:
            values["work"] = counters
    else:
        (name,) = route_modules(arch, KCP_FWD)
        record = _loader.MODULES[name]
        scalars = plan(
            S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
        )
        grid = _forward_grid(
            record,
            arch=arch,
            device_index=device_index,
            C=C,
            n_ch_tiles=scalars["n_ch_tiles"],
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
            grid=head_grid(_loader.MODULES[halo_name], n_frag=n_frag, C=C),
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
                name, scalars, grid, words = _sr_main(
                    arch, S=S, C=C, n_frag=n_frag, device_index=device_index
                )
                fresh["words"] = words
                fresh["launch"] = _zero_history_plan(name, grid, device_index)
            else:
                (name,) = route_modules(arch, KCP_FWD)
                record = _loader.MODULES[name]
                scalars = plan(
                    S, C, n_frag, geometry=record["grid_rule"]["geometry"], legacy=True
                )
                grid = _forward_grid(
                    record,
                    arch=arch,
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


def _zero_history_plan(
    name: str, grid: tuple[int, int, int], device_index: int
) -> tuple[Any, str, tuple[tuple[str | None, int], ...]]:
    """:func:`_launch_plan` of record ``name`` of the zero-history registry (``cake_jit``: the two-launch forward's
    single-rank main program), loaded through that registry."""
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


def _no_hist(e: dict[str, Any], device: torch.device) -> torch.Tensor:
    """The overlapped forward's all ``-1`` ``hist_row_off`` of the ``kcp_fwd`` main program (int32 ``[n_frag, 3]``),
    made at the host-cache entry's first overlapped ``kcp_fwd`` call, then reused (a captured graph pins the entry, so
    the table outlives a host-cache clear)."""
    table = e.get("no_hist")
    if table is None:
        table = e["no_hist"] = torch.full(
            (e["n_frag"], HALO), -1, dtype=torch.int32, device=device
        )
    return table


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
        grid = head_grid(_loader.MODULES[name], n_frag=e["n_frag"], C=C)
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
            agree=("fwd", (C,), False),
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


def cake_causal_conv1d_nondet_train_kcp_backward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
    graph: Optional[str] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP backward: ``dx`` of the rank's fragments (the halo contributions of the successor fragments folded in
    before the one BF16 rounding; the det KCP backward's bits) and the rank's FP32 ``dweight`` partial (summed across
    CTAs in an unordered order: not bitwise reproducible run to run; its sum over the ranks is the framework's).

    ``hist`` is the forward's pool.  The head program writes the derivative rows of every continuing fragment's first
    tokens into the backward exchange's send layout, the exchange delivers them to the predecessors as ``dpre_tail``,
    and the main program folds them in one ordinary launch.  Every rank calls this function once per layer, in the
    same order.  ``graph`` as in the forward (the backward's key holds the forward's ``hist`` pointer, so its capture
    follows the forward's by one call).  Under ``torch.use_deterministic_algorithms(True)`` (read when the call
    runs) the call raises a ``RuntimeError`` (a ``UserWarning``, then it runs, with ``warn_only=True``) before it
    launches or exchanges anything.  Preallocated outputs: ``dx`` BF16 of ``x_local``'s shape and ``dweight`` FP32
    ``[C, 4]``, contiguous; ``dweight`` must be 16-byte aligned (one 16-byte store per channel; ``ValueError``
    otherwise).  ``bwd_reserve_sms`` (``None`` = ``RESERVE_DEFAULT``):
    the main backward's grid gives up that many SMs (:func:`_bwd_grid`); the graph key holds it.  ``ValueError`` for
    ``K >= 1`` on sm_107a (:data:`RESERVE_UNSUPPORTED`) or at the device's SM count or above.
    """

    if bwd_reserve_sms is not None:
        return _backward_kcp_given(
            x_local,
            weight,
            ctx,
            dy,
            hist,
            dx=dx,
            dweight=dweight,
            graph=graph,
            bwd_reserve_sms=bwd_reserve_sms,
        )
    _alert_nondeterministic("cake_causal_conv1d_nondet_train_kcp_backward")
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


def _backward_kcp_given(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor],
    dweight: Optional[torch.Tensor],
    graph: Optional[str],
    bwd_reserve_sms: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`cake_causal_conv1d_nondet_train_kcp_backward` with ``bwd_reserve_sms`` given: validated before anything
    runs, then the backward on the grid of that reserve (0: the shipped grid)."""
    reserve = _reserve(bwd_reserve_sms, "cake_causal_conv1d_nondet_train_kcp_backward")
    _reserve_fits(reserve, x_local, "cake_causal_conv1d_nondet_train_kcp_backward")
    _alert_nondeterministic("cake_causal_conv1d_nondet_train_kcp_backward")
    mode = _graph_mode(graph)
    if mode == "off":
        return _kcp_backward(
            x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight, reserve=reserve
        )
    entry = _graph_entry(
        "bwd",
        ctx,
        (x_local, weight, dy, hist, dx, dweight),
        mode,
        extra=(("reserve", reserve),) if reserve else (),
    )
    dx_out, dw_out = _graph_run(
        "bwd",
        entry,
        lambda: _kcp_backward(
            x_local, weight, ctx, dy, hist, dx=dx, dweight=dweight, reserve=reserve
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
    reserve: int = RESERVE_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The backward's host flow (eagerly or inside a capture): see
    :func:`cake_causal_conv1d_nondet_train_kcp_backward`."""

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
            reserve=reserve,
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
    dw_out = _dweight_aligned(
        _output(
            dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
        )
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
        agree=("bwd", (C,), False),
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
    if _overlapped(C, S, reserve):
        d = _dyn_host(
            arch,
            S=S,
            C=C,
            n_frag=n_frag,
            tail_rows=int(halo.n_tail_rows),
            device=device,
            device_index=device_index,
        )
        dx2 = dx_out.view(S, C)
        common = {
            "x_flat": x_flat,
            "w": w,
            "cu": cu,
            "dy_flat": dy_flat,
            "hist_t": hist_t,
            "hist_row_off": hist_row_off,
            "dx2": dx2,
            "C": C,
            "n_frag": n_frag,
            "device_index": device_index,
        }
        _launch_dyn(
            d,
            dw_out=dw_out,
            spare=dpre_head,
            skip=d["no_tail"],
            S=S,
            device=device,
            **common,
        )
        if work is not None:
            work.wait()
        _launch_halo(d, tail_row_off=tail_row_off, dpre_tail=dpre_tail, **common)
        return dx_out, dw_out
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
        _accumulator_layout(record, C, scalars["n_ch_tiles"]),
        device=device,
        device_index=device_index,
        entry="cake_causal_conv1d_nondet_train_kcp_backward",
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
        run_module(
            name,
            values,
            device_index=device_index,
            grid=_bwd_grid(
                _loader.MODULES[name], scalars["n_ch_tiles"], device_index, reserve
            ),
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
    reserve: int = RESERVE_DEFAULT,
) -> tuple[Any, ...]:
    """The trimmed backward's operand checks, outputs and host-cache entry: ``(S, C, x_flat, w, dy_flat, hist_t,
    dx_out, dw_out, entry)``.  The entry is cached per ``(ctx, S, C, device, cu_seqlens_local_cpu, reserve)`` (and
    the ``bwd_route`` test knob when not the default): the tables, the head grid, and either the overlapped
    backward's state (``entry["dyn"]``, :func:`_dyn_host`) or the backward records, plan scalars, the accumulator
    layout, the loaded modules with their argument plans (grids for ``reserve``)."""

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
    dw_out = _dweight_aligned(
        _output(
            dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
        )
    )
    key: tuple = ("bwd", S, C, str(device), tuple(ctx.cu_seqlens_local_cpu))
    if reserve:
        key += (reserve,)
    if _TEST_KNOBS["bwd_route"] != BWD_ROUTE_DEFAULT:
        key += (_TEST_KNOBS["bwd_route"],)
    e = _host_cache_entry(ctx, key)
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
            dyn=None,
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
        if S != 0 and _overlapped(C, S, reserve):
            e["dyn"] = _dyn_host(
                arch,
                S=S,
                C=C,
                n_frag=n_frag,
                tail_rows=e["tail_rows"],
                device=device,
                device_index=device_index,
            )
        elif S != 0:
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
            e["layout"] = _accumulator_layout(record, C, scalars["n_ch_tiles"])
            e["scalars"] = _scalars(scalars)
            e["launches"] = tuple(
                _launch_plan(
                    name,
                    _bwd_grid(
                        _loader.MODULES[name],
                        scalars["n_ch_tiles"],
                        device_index,
                        reserve,
                    ),
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
    dpre_head: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The head program of the backward (when the rank has head rows): the FP32 ``dpre_head`` pool (at least one
    row; ``None`` = a fresh one) whose first ``n_head`` rows are the backward exchange's send rows."""
    if dpre_head is None:
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
        entry="cake_causal_conv1d_nondet_train_kcp_backward",
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


def _bwd_args(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    dx_out: torch.Tensor,
    *,
    S: int,
    C: int,
) -> dict[str, Any]:
    """The keyword arguments :func:`_launch_dyn` and :func:`_launch_halo` share, from a trimmed host-cache entry."""
    return {
        "x_flat": x_flat,
        "w": w,
        "cu": e["cu"],
        "dy_flat": dy_flat,
        "hist_t": hist_t,
        "hist_row_off": e["hist_row_off"],
        "dx2": dx_out.view(S, C),
        "C": C,
        "n_frag": e["n_frag"],
        "device_index": e["device_index"],
    }


def _dyn_skip(e: dict[str, Any], side: Optional[torch.cuda.Stream]) -> torch.Tensor:
    """The main program's ``skip_row_off`` for a trimmed host-cache entry ``e``: the context's ``tail_row_off`` when
    the tail-row program runs on the side stream (``side``) beside it, so it leaves the rows that program writes;
    else the all ``-1`` table (every row written, the tail-row program after it on the same stream)."""
    if side is not None and e["dyn"]["halo"] is not None:
        return e["tail_row_off"]
    return e["dyn"]["no_tail"]


def _bwd_overlap(
    e: dict[str, Any],
    x_flat: torch.Tensor,
    w: torch.Tensor,
    dy_flat: torch.Tensor,
    hist_t: torch.Tensor,
    dx_out: torch.Tensor,
    dw_out: torch.Tensor,
    dpre_head: torch.Tensor,
    dpre_tail: torch.Tensor,
    work: Optional[Any],
    *,
    S: int,
    C: int,
    device: torch.device,
    side: Optional[torch.cuda.Stream] = None,
) -> None:
    """The overlapped backward of a call with ``S > 0``: the main program while ``work`` (the exchange) is in flight.
    With ``side`` (the stream of the head program and the exchange issue) the main program leaves the tail-row
    program's rows (:func:`_dyn_skip`), and the side stream waits for the exchange and runs the tail-row program
    beside it; the current stream then joins the side stream, also when a launch raises.  Without (``None``) the main
    program writes every row, the current stream waits, then the tail-row program writes the tail rows again."""
    args = _bwd_args(e, x_flat, w, dy_flat, hist_t, dx_out, S=S, C=C)
    try:
        _launch_dyn(
            e["dyn"],
            dw_out=dw_out,
            spare=dpre_head,
            skip=_dyn_skip(e, side),
            S=S,
            device=device,
            **args,
        )
        if side is not None:
            with torch.cuda.stream(side):
                if work is not None:
                    work.wait()
                _launch_halo(
                    e["dyn"],
                    tail_row_off=e["tail_row_off"],
                    dpre_tail=dpre_tail,
                    **args,
                )
    finally:
        _join_side(side, device)
    if side is not None:
        return
    if work is not None:
        work.wait()
    _launch_halo(e["dyn"], tail_row_off=e["tail_row_off"], dpre_tail=dpre_tail, **args)


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
    reserve: int = RESERVE_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trimmed host path: :func:`_kcp_backward` with the context-derived host work cached (:func:`_bwd_host`); the
    operand checks, allocations, exchange, wait and launches are the untrimmed call's, except that an overlapped call
    with more than one rank issues the head program and the exchange from the side stream (:func:`_side_stream`)."""

    S, C, x_flat, w, dy_flat, hist_t, dx_out, dw_out, e = _bwd_host(
        x,
        weight,
        ctx,
        dy,
        hist,
        dx=dx,
        dweight=dweight,
        device=device,
        reserve=reserve,
    )
    side = _side_stream(ctx, device, S > 0 and e["dyn"] is not None)
    pool = recv = None
    if side is not None:
        pool = _pool(e["n_head"], C, dtype=torch.float32, device=device)
        recv = torch.empty((e["tail_rows"], C), dtype=torch.float32, device=device)
    _fork(side, device)
    try:
        with torch.cuda.stream(side):
            dpre_head = _bwd_head(
                e, x_flat, w, dy_flat, hist_t, S=S, C=C, device=device, dpre_head=pool
            )
            dpre_tail, work = _exchange(
                dpre_head[: e["n_head"]],
                recv_rows=e["tail_rows"],
                send_splits=e["send_splits"],
                recv_splits=e["recv_splits"],
                ctx=ctx,
                agree=("bwd", (C,), False),
                recv=recv,
            )
    except BaseException:
        _rejoin(side, device)
        raise
    if S == 0:
        if work is not None:
            work.wait()
        dw_out.zero_()
        return dx_out, dw_out
    if e["dyn"] is not None:
        _bwd_overlap(
            e,
            x_flat,
            w,
            dy_flat,
            hist_t,
            dx_out,
            dw_out,
            dpre_head,
            dpre_tail,
            work,
            S=S,
            C=C,
            device=device,
            side=side,
        )
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


class CakeCausalConv1dNondetKcpFunction(torch.autograd.Function):
    """Autograd wrapper: ``apply(x_local, weight, ctx, graph, handle)``; the forward saves ``(x_local, weight)`` and
    the history pool ``hist`` (no second exchange in the backward); the backward runs the head program, the backward
    halo exchange and the main backward program (it alerts under torch's deterministic mode).  ``handle``: a
    recompute handle (see :class:`CakeCausalConv1dNondetKcpRecomputeHandle`): the original forward stores ``hist`` in
    it, the recompute binds it and exchanges nothing; the saved set is the same three tensors either way.  A handle
    is refused with graph replay (keyword or knob) and inside a CUDA-graph capture."""

    @staticmethod
    def forward(fn_ctx, x_local, weight, ctx, graph=None, handle=None):
        if handle is None:
            y, hist = cake_causal_conv1d_nondet_train_kcp_forward(
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
        return y

    @staticmethod
    def backward(fn_ctx, grad_y):
        x_local, weight, hist = fn_ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x_local)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_nondet_train_kcp_backward(
            x_local,
            weight,
            fn_ctx.kcp,
            dy,
            hist,
            graph=fn_ctx.graph,
        )
        return dx, dweight, None, None, None


class _CakeCausalConv1dReserveFunctionKcp(CakeCausalConv1dNondetKcpFunction):
    """:class:`CakeCausalConv1dNondetKcpFunction` with the opt-in SM reserve of the main backward:
    ``apply(x_local, weight, ctx, graph, handle, reserve)``, ``reserve`` the caller's validated
    ``bwd_reserve_sms >= 1`` (one more input, one more ``None`` gradient)."""

    @staticmethod
    def forward(fn_ctx, x_local, weight, ctx, graph, handle, reserve):
        fn_ctx.reserve = reserve
        return CakeCausalConv1dNondetKcpFunction.forward(
            fn_ctx, x_local, weight, ctx, graph, handle
        )

    @staticmethod
    def backward(fn_ctx, grad_y):
        x_local, weight, hist = fn_ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x_local)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_nondet_train_kcp_backward(
            x_local,
            weight,
            fn_ctx.kcp,
            dy,
            hist,
            graph=fn_ctx.graph,
            bwd_reserve_sms=fn_ctx.reserve,
        )
        return dx, dweight, None, None, None, None


def cake_causal_conv1d_nondet_kcp(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> torch.Tensor:
    """Autograd-enabled KCP packed width-4 depthwise causal convolution + SiLU of the rank's fragments, with the
    non-deterministic main backward program.  Under ``torch.use_deterministic_algorithms(True)`` (read when the
    backward runs) the backward raises a ``RuntimeError`` before it exchanges (a ``UserWarning``, then it runs, with
    ``warn_only=True``); it never falls back to the deterministic KCP package.

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous activations of the rank's fragments, back to back in the
        order of ``ctx.cu_seqlens_local``.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current token.
    ctx : CakeKDADetKcpContext
        The rank's context (``flashinfer.cake_kda_det_train_kcp.cake_kda_det_train_kcp_build_context``): fragment table, halo
        plan, process group.  Every rank of the group calls this function once per layer, in the same order.
    graph : str, optional
        ``None`` (the shipped default ``GRAPH_DEFAULT`` = ``"off"``, the eager path) or ``"replay"``: keyed CUDA-graph
        capture / replay of this call.  The first call of a validity key (this
        context, its world size, and the shape / stride / dtype / data pointer of ``x_local`` and ``weight``) runs
        eagerly, the second is captured, every later one replays the graph (bitwise the eager result).  Requirements:
        pointer-stable operands and the same ``CakeKDADetKcpContext`` per layer, outputs consumed before the next call of the
        same layer (a replay overwrites the graph's static tensors); any other key runs eagerly (``GRAPH_STATS``,
        ``LAST_CALL``).  Each cached key pins a private memory pool (at most ``GRAPH_CACHE_MAX`` keys); call
        :func:`cake_causal_conv1d_nondet_train_kcp_graph_cache_clear` before ``torch.distributed.destroy_process_group()``.
    recompute_handle : CakeCausalConv1dNondetKcpRecomputeHandle, optional
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
        The deterministic package's launch-form keyword, accepted so both packages take one parameter list.
        The main backward here is always one ordinary launch, so ``None`` (default) and
        ``False`` mean the same; ``True`` raises ``ValueError`` and a non-``bool`` ``TypeError``.
    bwd_reserve_sms : int, optional
        ``None`` (default) = ``RESERVE_DEFAULT`` = 0: the backward grid of :func:`_bwd_grid`.  ``K >= 1``: the main
        backward launches its single-wave grid less ``K x ctas_per_sm`` CTAs, so it stays one wave while kernels of
        other streams hold up to ``K`` SMs; it costs the unloaded backward the work of ``K`` SMs.  ``y``, ``dx`` and
        the exchanges are unchanged; ``dweight`` stays within the budget.  ``TypeError`` for a ``bool`` or a
        non-``int``; ``ValueError`` below zero, at the device's SM count or above, or ``K >= 1`` on sm_107a
        (:data:`RESERVE_UNSUPPORTED`: the reserved grid is outside the opt-in's unloaded-cost limit there).  The
        ranks of a group may differ.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape, with the three document tokens before every continuing fragment as
        history (zero history at a document's start).  The backward returns BF16 ``dx`` (the successor fragments'
        contributions folded in before the one rounding; the det KCP backward's bits) and the rank's FP32
        ``dweight`` partial (unordered cross-CTA sum).
    """

    if cooperative is not None or bwd_reserve_sms is not None:
        return _kcp_given(
            x_local,
            weight,
            ctx,
            graph=graph,
            recompute_handle=recompute_handle,
            cooperative=cooperative,
            bwd_reserve_sms=bwd_reserve_sms,
        )
    _graph_mode(graph)  # validate on the caller's frame
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    # Always five inputs: the backward returns five gradients.
    return CakeCausalConv1dNondetKcpFunction.apply(
        x_local, weight, ctx, graph, recompute_handle
    )


def _kcp_given(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    graph: Optional[str],
    recompute_handle: Optional[Any],
    cooperative: Any,
    bwd_reserve_sms: Any,
) -> torch.Tensor:
    """:func:`cake_causal_conv1d_nondet_kcp` with ``cooperative`` or ``bwd_reserve_sms`` given: both validated before
    anything runs; a reserve of 1 or more runs :class:`_CakeCausalConv1dReserveFunctionKcp`."""
    _ordinary_launch(cooperative, "cake_causal_conv1d_nondet_kcp")
    reserve = _reserve(bwd_reserve_sms, "cake_causal_conv1d_nondet_kcp")
    _reserve_fits(reserve, x_local, "cake_causal_conv1d_nondet_kcp")
    _graph_mode(graph)  # validate on the caller's frame
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    if reserve:
        return _CakeCausalConv1dReserveFunctionKcp.apply(
            x_local, weight, ctx, graph, recompute_handle, reserve
        )
    return CakeCausalConv1dNondetKcpFunction.apply(
        x_local, weight, ctx, graph, recompute_handle
    )


# ----------------------------------------------------------------------------------------------------------------
# Several convolutions of one layer with one halo exchange per direction (FlashInfer issue #7815): the q / k / v
# convolutions of a layer share the context, so their history rows (forward) and head rows (backward) travel in ONE
# ``all_to_all_single`` instead of one per tensor.  The send rows of the tensors are packed side by side
# (``[n_rows, sum C_i]``, the context's splits unchanged), and after the wait each tensor's rows are copied out into
# a fresh contiguous pool -- the forward's ``hist`` ``[n_hist_rows, C_i]`` (the saved tensor) and the backward's
# ``dpre_tail`` ``[n_tail_rows, C_i]`` -- before that tensor's kernels launch with the single call's arguments; an
# overlapped backward whose tensors share one width lets each tail-row program read its rows in place in the received
# pool instead (:func:`_halos_in_place`).  The exchange and the unpacking move the same values, and each tensor keeps
# its own kernels and accumulator launch, so every output and ``dx`` is bitwise that of one single call per tensor
# (``dweight`` is each tensor's own unordered partial).
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


def _halos_in_place(
    calls: list[tuple], recv: torch.Tensor, device: torch.device
) -> None:
    """The tail-row programs of an overlapped multi call whose ``n`` tensors share one width ``C``, after the wait.
    Row ``r`` of tensor ``i`` sits at element ``(r * n + i) * C`` of the packed receive pool, so tensor ``i`` reads
    the pool from element ``i * C`` with its tail-row table scaled by ``n`` (no copy; the program addresses
    ``dpre_tail + row * C`` and needs a contiguous pool).  The scaled tables are made on the current stream once per
    host-cache entry and calling stream, so a table is read only on the stream that allocated it and on streams
    forked from it and joined back before its next launch: when the entry is dropped, the allocator reuses the
    table only after those reads (inside a CUDA-graph capture they are made for the call only: a captured table
    holds its values only after a replay).  Then the first program runs on the current stream and the others on
    the device's streams 1 .. n - 1, forked from it and joined back (the pool, allocated on the caller's stream
    before the side stream's fork, lives until the caller's stream joins the side stream)."""
    n = len(calls)
    capturing = torch.cuda.is_current_stream_capturing()
    current = torch.cuda.current_stream(device)
    tables = []
    for call in calls:
        e = call[8]
        key = f"tail_row_off_x{n}@{current.cuda_stream:x}"
        table = e.get(key)
        if table is None:
            t = e["tail_row_off"]
            table = torch.where(t >= 0, t * n, t)
            if not capturing:
                e[key] = table
        tables.append(table)
    streams = [current] + [_device_stream(device, i) for i in range(1, n)]
    for stream in streams[1:]:
        stream.wait_stream(current)
    flat = recv.view(-1)
    for i, ((S, C, x_flat, w, dy_flat, hist_t, dx, _dw, e), table, stream) in enumerate(
        zip(calls, tables, streams, strict=True)
    ):
        with torch.cuda.stream(stream):
            args = _bwd_args(e, x_flat, w, dy_flat, hist_t, dx, S=S, C=C)
            _launch_halo(e["dyn"], tail_row_off=table, dpre_tail=flat[i * C :], **args)
    for stream in streams[1:]:
        current.wait_stream(stream)


def cake_causal_conv1d_nondet_train_kcp_multi_forward(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    outs: Optional[Any] = None,
    graph: Optional[str] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP forward of several convolutions of one layer with one halo exchange: ``(ys, hists)``, bitwise
    ``cake_causal_conv1d_nondet_train_kcp_forward(xs[i], weights[i], ctx)`` per tensor.

    ``xs`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous tensors of the rank's fragments (the channel counts
    may differ); ``weights`` FP32 ``[C_i, 4]``; ``outs`` optional preallocated outputs.  Every rank of ``ctx.group``
    calls this function once per layer, in the same order, with the same number of tensors and the same channel
    counts: the one forward exchange is a collective.  ``hists[i]`` is a fresh contiguous ``[n_hist_rows, C_i]``
    BF16 pool (the saved tensor of tensor ``i``).  ``graph`` as in
    :func:`cake_causal_conv1d_nondet_train_kcp_forward` (the validity key holds every operand).
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
        extra=(("fwd_form", _fwd_variant(xs[0], multi=True)),),
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
            agree=("fwd", tuple(c[1] for c in calls), True),
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


def cake_causal_conv1d_nondet_train_kcp_multi_backward(
    xs: Any,
    weights: Any,
    ctx: Any,
    dys: Any,
    hists: Any,
    *,
    dxs: Optional[Any] = None,
    dweights: Optional[Any] = None,
    graph: Optional[str] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    r"""The KCP backward of several convolutions of one layer with one halo exchange: ``(dxs, dweights)``,
    ``cake_causal_conv1d_nondet_train_kcp_backward(xs[i], weights[i], ctx, dys[i], hists[i])`` per tensor (``dxs``
    bitwise; each ``dweights[i]`` an unordered FP32 partial of the same budget).

    The head programs of every tensor run first, their send rows travel packed in one exchange, and each tensor's
    received rows are folded into its ``dx`` by that tensor's programs: on the overlapped route its main program runs
    while the exchange is in flight and its tail-row program after the wait (reading the rows in place when every
    tensor has the same width, else from a copy); otherwise its one-program backward runs after the wait (one ordinary
    launch per tensor on the stream's accumulator workspace).  ``hists`` are the forward's pools.  Every rank calls
    this function once per layer, in the same order.  ``graph`` as in
    :func:`cake_causal_conv1d_nondet_train_kcp_backward`.  Under torch's deterministic mode the call alerts like the
    single backward.  ``dxs`` / ``dweights`` entries follow the single backward's output rules (``dweight`` 16-byte
    aligned).  ``bwd_reserve_sms`` as in :func:`cake_causal_conv1d_nondet_train_kcp_backward`, for every tensor's main
    backward.
    """

    if bwd_reserve_sms is not None:
        return _multi_backward_kcp_given(
            xs,
            weights,
            ctx,
            dys,
            hists,
            dxs=dxs,
            dweights=dweights,
            graph=graph,
            bwd_reserve_sms=bwd_reserve_sms,
        )
    _alert_nondeterministic("cake_causal_conv1d_nondet_train_kcp_multi_backward")
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


def _multi_backward_kcp_given(
    xs: Any,
    weights: Any,
    ctx: Any,
    dys: Any,
    hists: Any,
    *,
    dxs: Any,
    dweights: Any,
    graph: Optional[str],
    bwd_reserve_sms: Any,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """:func:`cake_causal_conv1d_nondet_train_kcp_multi_backward` with ``bwd_reserve_sms`` given: validated before
    anything runs, then every tensor's main backward on the grid of that reserve (0: the shipped grid)."""
    reserve = _reserve(
        bwd_reserve_sms, "cake_causal_conv1d_nondet_train_kcp_multi_backward"
    )
    _reserve_fits(
        reserve,
        xs[0] if isinstance(xs, (list, tuple)) and xs else None,
        "cake_causal_conv1d_nondet_train_kcp_multi_backward",
    )
    _alert_nondeterministic("cake_causal_conv1d_nondet_train_kcp_multi_backward")
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
        return _kcp_backward_multi(
            xs, weights, ctx, dys, hists, dxs, dweights, reserve=reserve
        )
    entry = _graph_entry(
        "bwd_multi",
        ctx,
        (*xs, *weights, *dys, *hists, *dxs, *dweights),
        mode,
        extra=(("reserve", reserve),) if reserve else (),
    )
    dx_out, dw_out = _graph_run(
        "bwd_multi",
        entry,
        lambda: _kcp_backward_multi(
            xs, weights, ctx, dys, hists, dxs, dweights, reserve=reserve
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
    reserve: int = RESERVE_DEFAULT,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """The multi backward's host flow (eagerly or inside a capture): see
    :func:`cake_causal_conv1d_nondet_train_kcp_multi_backward`."""

    device = _require_tensor(xs[0], "xs[0]").device
    _wait_ready(ctx)
    calls = [
        _bwd_host(x, w, ctx, dy, h, dx=dx, dweight=dw, device=device, reserve=reserve)
        for x, w, dy, h, dx, dw in zip(
            xs, weights, dys, hists, dxs, dweights, strict=True
        )
    ]
    side = _side_stream(
        ctx, device, calls[0][0] > 0 and all(c[8]["dyn"] is not None for c in calls)
    )
    e0 = calls[0][8]
    n_head = e0["n_head"]
    work = None
    recv = None
    tails: Optional[list[torch.Tensor]] = None
    pools: list[Optional[torch.Tensor]] = [None] * len(calls)
    send = None
    if side is not None:
        widths = [c[1] for c in calls]
        pools = [
            _pool(c[8]["n_head"], c[1], dtype=torch.float32, device=device)
            for c in calls
        ]
        if len(calls) > 1:
            send = torch.empty(
                (n_head, sum(widths)), dtype=torch.float32, device=device
            )
        recv = torch.empty(
            (e0["tail_rows"], sum(widths)), dtype=torch.float32, device=device
        )
    _fork(side, device)
    try:
        with torch.cuda.stream(side):
            heads = [
                _bwd_head(
                    c[8],
                    c[2],
                    c[3],
                    c[4],
                    c[5],
                    S=c[0],
                    C=c[1],
                    device=device,
                    dpre_head=pool,
                )
                for c, pool in zip(calls, pools, strict=True)
            ]
            if int(ctx.world_size) == 1:
                tails = [h[:n_head] for h in heads]
            else:
                if len(heads) == 1:
                    send = heads[0][:n_head]
                else:
                    send = torch.cat([h[:n_head] for h in heads], dim=1, out=send)
                recv, work = _exchange(
                    send,
                    recv_rows=e0["tail_rows"],
                    send_splits=e0["send_splits"],
                    recv_splits=e0["recv_splits"],
                    ctx=ctx,
                    agree=("bwd", tuple(c[1] for c in calls), True),
                    recv=recv,
                )
    except BaseException:
        _rejoin(side, device)
        raise
    dx_out = tuple(c[6] for c in calls)
    dw_out = tuple(c[7] for c in calls)
    if calls[0][0] == 0:
        if work is not None:
            work.wait()
        for dw in dw_out:
            dw.zero_()
        return dx_out, dw_out
    # The overlapped tensors' main programs run while the exchange is in flight; every other launch follows the wait,
    # on the side stream when the call has one (the main programs then leave the tail-row programs' rows).
    try:
        for (S, C, x_flat, w, dy_flat, hist_t, dx, dw, e), head in zip(
            calls, heads, strict=True
        ):
            if e["dyn"] is not None:
                args = _bwd_args(e, x_flat, w, dy_flat, hist_t, dx, S=S, C=C)
                _launch_dyn(
                    e["dyn"],
                    dw_out=dw,
                    spare=head,
                    skip=_dyn_skip(e, side),
                    S=S,
                    device=device,
                    **args,
                )
        if side is not None:
            in_place = len(calls) > 1 and len({c[1] for c in calls}) == 1
            with torch.cuda.stream(side):
                if work is not None:
                    work.wait()
                _multi_after_wait(calls, recv, tails, device, in_place=in_place)
    finally:
        _join_side(side, device)
    if side is not None:
        return dx_out, dw_out
    if work is not None:
        work.wait()
    _multi_after_wait(calls, recv, tails, device, in_place=False)
    return dx_out, dw_out


def _multi_after_wait(
    calls: list[tuple],
    recv: Optional[torch.Tensor],
    tails: Optional[list[torch.Tensor]],
    device: torch.device,
    *,
    in_place: bool,
) -> None:
    """The launches of a multi backward after the exchange's wait, on the current stream: with ``in_place`` (more than
    one tensor of one width, on the side stream) the tail-row programs read the received pool in place
    (:func:`_halos_in_place`); otherwise each tensor's tail rows (the received pool ``recv`` cut per tensor, or
    ``tails`` with one rank) go to its tail-row program, or to its one-program backward when it is not overlapped."""
    if in_place:
        _halos_in_place(calls, recv, device)
        return
    if recv is not None:
        tails = _unpack(recv, [c[1] for c in calls]) if len(calls) > 1 else [recv]
    for (S, C, x_flat, w, dy_flat, hist_t, dx, dw, e), tail in zip(
        calls, tails, strict=True
    ):
        if e["dyn"] is not None:
            args = _bwd_args(e, x_flat, w, dy_flat, hist_t, dx, S=S, C=C)
            _launch_halo(
                e["dyn"], tail_row_off=e["tail_row_off"], dpre_tail=tail, **args
            )
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
            f"cake_causal_conv1d_nondet_kcp or of another tensor count); one handle serves one call"
        )
    if entry.get("ctx") is not ctx:
        raise ValueError(
            "recompute_handle: the 'conv' entry belongs to another CakeKDADetKcpContext"
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


class CakeCausalConv1dNondetKcpMultiFunction(torch.autograd.Function):
    """Autograd wrapper of several convolutions with one halo exchange per direction: ``apply(ctx, graph, handle,
    n, *xs, *weights)`` returns the ``n`` outputs; the forward saves ``(*xs, *weights, *hists)``; the backward runs
    :func:`cake_causal_conv1d_nondet_train_kcp_multi_backward` (it alerts under torch's deterministic mode).
    ``handle``: one recompute handle for the call, whose ``"conv"`` entry holds the ``n`` pools (the saved set's own
    tensors): the recompute binds them and exchanges nothing.  Refused with graph replay (keyword or knob) and
    inside a CUDA-graph capture, like the single call."""

    @staticmethod
    def forward(fn_ctx, ctx, graph, handle, n, *tensors):
        xs, weights = list(tensors[:n]), list(tensors[n:])
        if handle is None:
            ys, hists = cake_causal_conv1d_nondet_train_kcp_multi_forward(
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
        dxs, dweights = cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs, weights, fn_ctx.kcp, dys, hists, graph=fn_ctx.graph
        )
        return (None, None, None, None, *dxs, *dweights)


class _CakeCausalConv1dReserveMultiFunctionKcp(CakeCausalConv1dNondetKcpMultiFunction):
    """:class:`CakeCausalConv1dNondetKcpMultiFunction` with the opt-in SM reserve of every main backward:
    ``apply(reserve, ctx, graph, handle, n, *xs, *weights)``, ``reserve`` the caller's validated
    ``bwd_reserve_sms >= 1`` (one more input, one more ``None`` gradient)."""

    @staticmethod
    def forward(fn_ctx, reserve, ctx, graph, handle, n, *tensors):
        fn_ctx.reserve = reserve
        return CakeCausalConv1dNondetKcpMultiFunction.forward(
            fn_ctx, ctx, graph, handle, n, *tensors
        )

    @staticmethod
    def backward(fn_ctx, *grad_ys):
        n = fn_ctx.n
        saved = fn_ctx.saved_tensors
        xs, weights, hists = saved[:n], saved[n : 2 * n], saved[2 * n :]
        dys = [
            (torch.zeros_like(x) if g is None else g).to(torch.bfloat16).contiguous()
            for g, x in zip(grad_ys, xs, strict=True)
        ]
        dxs, dweights = cake_causal_conv1d_nondet_train_kcp_multi_backward(
            xs,
            weights,
            fn_ctx.kcp,
            dys,
            hists,
            graph=fn_ctx.graph,
            bwd_reserve_sms=fn_ctx.reserve,
        )
        return (None, None, None, None, None, *dxs, *dweights)


def cake_causal_conv1d_nondet_kcp_multi(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    graph: Optional[str] = None,
    recompute_handle: Optional[Any] = None,
    cooperative: Optional[bool] = None,
    bwd_reserve_sms: Optional[int] = None,
) -> tuple[torch.Tensor, ...]:
    """Autograd-enabled KCP convolutions of several tensors of one layer (for example its q, k and v) with one halo
    exchange per direction.

    ``tuple(cake_causal_conv1d_nondet_kcp(x, w, ctx) for x, w in zip(xs, weights))`` -- outputs and ``dx`` bitwise,
    each ``dweight`` that tensor's own unordered FP32 partial -- with one collective in the forward and one in the
    backward instead of one per tensor: fewer global synchronization points per layer.  Under torch's deterministic
    mode the backward alerts like the single call's.

    Parameters
    ----------
    xs : sequence of torch.Tensor
        ``n >= 1`` BF16 ``[T_r, C_i]`` (or ``[1, T_r, C_i]``) contiguous activations of the rank's fragments, in the
        order of ``ctx.cu_seqlens_local``; the channel counts may differ.
    weights : sequence of torch.Tensor
        FP32 ``[C_i, 4]`` contiguous taps, one per tensor.
    ctx : CakeKDADetKcpContext
        The rank's context, as in :func:`cake_causal_conv1d_nondet_kcp`.  Every rank of the group calls this function
        once per layer, in the same order, with the same tensor count and channel counts.  The ranks are not compared
        (that would cost a collective): a different count, channel count or order hangs the exchange or silently
        corrupts the history rows, as a mismatch of the single entry's per-rank contract does.
    graph : str, optional
        As in :func:`cake_causal_conv1d_nondet_kcp` (keyed CUDA-graph capture / replay; the key holds every operand).
    recompute_handle : CakeCausalConv1dNondetKcpRecomputeHandle, optional
        ``None`` (default): unchanged.  One handle for the whole call: the original forward stores the ``n`` history
        pools under ``"conv"``, torch's recompute inside the backward binds them and exchanges nothing; the backward
        still exchanges once.  Same rules as :func:`cake_causal_conv1d_nondet_kcp` (one handle per call and
        micro-step, created outside the outermost checkpoint region; refused with ``graph="replay"`` and inside a
        CUDA-graph capture; every rank passes one or none).  The handle may also carry the ``"kda"`` entry of the
        layer's recurrence; a handle's ``"conv"`` entry serves either one multi call or one single call, never both.
    cooperative : bool, optional
        As in :func:`cake_causal_conv1d_nondet_kcp` (``None`` or ``False``).
    bwd_reserve_sms : int, optional
        As in :func:`cake_causal_conv1d_nondet_kcp`, for every tensor's main backward.

    Returns
    -------
    tuple of torch.Tensor
        The ``n`` BF16 outputs, each of its input's shape.
    """

    if cooperative is not None or bwd_reserve_sms is not None:
        return _multi_kcp_given(
            xs,
            weights,
            ctx,
            graph=graph,
            recompute_handle=recompute_handle,
            cooperative=cooperative,
            bwd_reserve_sms=bwd_reserve_sms,
        )
    _graph_mode(graph)
    xs, weights, _ = _multi_operands(xs, weights, None)
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    return CakeCausalConv1dNondetKcpMultiFunction.apply(
        ctx, graph, recompute_handle, len(xs), *xs, *weights
    )


def _multi_kcp_given(
    xs: Any,
    weights: Any,
    ctx: Any,
    *,
    graph: Optional[str],
    recompute_handle: Optional[Any],
    cooperative: Any,
    bwd_reserve_sms: Any,
) -> tuple[torch.Tensor, ...]:
    """:func:`cake_causal_conv1d_nondet_kcp_multi` with ``cooperative`` or ``bwd_reserve_sms`` given: both validated
    before anything runs; a reserve of 1 or more runs :class:`_CakeCausalConv1dReserveMultiFunctionKcp`."""
    _ordinary_launch(cooperative, "cake_causal_conv1d_nondet_kcp_multi")
    reserve = _reserve(bwd_reserve_sms, "cake_causal_conv1d_nondet_kcp_multi")
    _graph_mode(graph)
    xs, weights, _ = _multi_operands(xs, weights, None)
    _reserve_fits(reserve, xs[0], "cake_causal_conv1d_nondet_kcp_multi")
    if recompute_handle is not None:
        _handle_refusals(graph, recompute_handle)
    if reserve:
        return _CakeCausalConv1dReserveMultiFunctionKcp.apply(
            reserve, ctx, graph, recompute_handle, len(xs), *xs, *weights
        )
    return CakeCausalConv1dNondetKcpMultiFunction.apply(
        ctx, graph, recompute_handle, len(xs), *xs, *weights
    )


__all__ = [
    "GRAPH_MODES",
    "GRAPH_DEFAULT",
    "GRAPH_STATS",
    "HANDLE_KEY",
    "RESERVE_DEFAULT",
    "RESERVE_UNSUPPORTED",
    "CakeCausalConv1dNondetKcpRecomputeHandle",
    "LAST_CALL",
    "cake_causal_conv1d_nondet_train_kcp_graph_cache_clear",
    "CakeCausalConv1dNondetKcpFunction",
    "cake_causal_conv1d_nondet_kcp",
    "cake_causal_conv1d_nondet_train_kcp_backward",
    "cake_causal_conv1d_nondet_train_kcp_forward",
    "CakeCausalConv1dNondetKcpMultiFunction",
    "cake_causal_conv1d_nondet_kcp_multi",
    "cake_causal_conv1d_nondet_train_kcp_multi_backward",
    "cake_causal_conv1d_nondet_train_kcp_multi_forward",
]
