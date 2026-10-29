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
"""

# Experimental Cake backend: the Kimi-K3 TP12 fused LatentMoE communication
# tail for SM100 / SM103 twelve-rank multi-node NVLink domains (GB200 /
# GB300 NVL72; flashinfer-ai/flashinfer#4542, tracker #4254).
#
# Per tensor-parallel rank ``r`` of twelve, ``M`` tokens, BF16::
#
#     latent = KimiRMSNorm(sum_r routed_partial_r)                  [M, 3584]
#     out    = BF16(latent @ up_weight.T + sum_r shared_partial_r)  [M, 7168]
#
# ``out`` is replicated and bitwise identical on every rank.  The fused tail
# runs on the current stream (CUDA-Graph capturable) as one generated launch
# per rank for ``M <= K23_MAX_TOKENS`` and as two generated launches, K1 and
# the tail, with cuBLAS between them above:
#
# * ``K1``  Lamport all-reduce of ``routed_partial`` fused with KimiRMSNorm ->
#   the normalised latent ``y`` on every rank (one-shot for ``M <= 16``, the
#   token-sliced two-shot form above).  Below ``K3_PERSIST_MIN_TOKENS`` the
#   early-shared-scatter (ESS) forms ``k1_oneshot_ess`` and
#   ``k1_twoshot_ess:grouped`` also scatter this rank's ``shared_partial``
#   columns into the K3 workspace slots of their owner ranks, so the tail
#   kernel of that range has no scatter stage of its own;
# * the up-projection of ``y`` with this rank's contiguous 640 / 512-row slice
#   of ``up_weight`` (``7168 = 8 x 640 + 4 x 512``: 7168 is not divisible by
#   12): fused into the tail for ``M <= K23_MAX_TOKENS``, cuBLAS
#   (``torch.mm``, BF16 slice) above;
# * the tail  owner reduce of the scattered ``shared_partial`` columns, add of
#   the up-projection slice, one BF16 rounding, multicast all-gather into the
#   caller-owned ``out``, by ``ROUTE_TABLE``: ``k3_ess:grouped`` (one CTA per
#   token and column half) for ``8 < M < 320``; the persistent reduce mode
#   ``k3_reduce:grouped`` (``min(M, SM count)`` CTAs per column half over the
#   slots the ESS K1 filled) for ``320 <= M < K3_PERSIST_MIN_TOKENS`` (961);
#   from there on the persistent token pipeline (``min(M, SM count)`` CTAs per
#   column half, own scatter of ``shared_partial``) with ``cp.async.bulk``
#   reduce-scatter pushes, ``k3_persist_bulk:pinned``.
#
# For ``M <= K23_MAX_TOKENS`` one program, ``k123:c<capacity>``, runs both
# stages as two CTA roles of one grid ``(M + cols / K23_ROWS, 1, 1)``: the
# first ``M`` CTAs run the one-shot ESS K1, one token each, and the other
# CTAs run the K23 tail, one CTA per ``K23_ROWS`` output columns (the fp32
# slice GEMM in the K2-stream summation order fused with the owner reduce,
# add and all-gather).  The token CTAs count in on the K3 flags record after
# their ``y`` stores and the K23 CTAs wait for ``M`` arrivals before they
# read ``y``, so the tail needs no second launch.  The four-token program
# serves ``M <= 4``, the eight-token program ``5 <= M <= 8``; the rank is a
# launch argument and its column width only sizes the grid.  Its CTAs wait
# for CTAs of the same grid with other indices (the low tail CTAs gather the
# columns the owner CTAs of every rank write, and each owner reduce polls
# the slots that tail CTAs of other indices scatter), so the grid is
# co-resident: it launches cooperatively (with PDL), and the driver either
# dispatches all of its at most 88 CTAs, one per SM, at once or refuses the
# launch.
#
# Rows ``M <= 4`` and ``M > 8`` are bitwise identical to the round-4 / round-5
# routes (the K23 role keeps the numerics of the round-4 K2-stream + fp32-add
# K3 pair; the ESS and the persistent tails reduce in the same order);
# ``5 <= M <= 8`` (round 6) round once instead of twice -- a strict error
# reduction against the fp64 reference.  The one-shot / K23 thresholds are
# pinned by the round-5 / round-6 paired A/Bs on GB200 and GB300 NVL72, the
# ESS / persistent boundary by a paired A/B of the two tail routes at 640 to
# 1056 tokens on both.
#
# Concurrency and provisioning.  A workspace is collective state, like an
# NCCL communicator: the kernels rotate its Lamport epoch on every call, so
# the calls that share it run one at a time and in the same order on every
# rank.  A launch from another stream than the previous launch on the
# workspace is ordered after it with an event wait (outside CUDA Graph
# capture); concurrent graph replays and several host threads are not
# ordered and must not share a workspace.  The one-launch program
# (``M <= 8``) needs its whole grid resident at once on every rank:
# ``M + 80`` SMs on the 640-column ranks and ``M + 64`` on the 512-column
# ranks.  Under a green context or MPS partition with fewer SMs the launch
# raises a RuntimeError (and leaves no CUDA error behind) instead of
# spinning, and a CUDA Graph capture on such a partition fails, so give
# every rank at least 88 SMs.  Do not overlap two ``M <= 8`` calls, or one
# and another cooperative peer-waiting kernel, on one GPU, even with
# separate workspaces: the two grids do not fit together, and an admission
# order that differs between ranks deadlocks the group.  A co-runner that
# never frees the grid's SMs on one rank stalls every rank; the kernels'
# waits are unbounded and the caller's watchdog is the deadline.  The
# persistent tails map token ``t`` to CTA ``t mod P`` on every rank, so
# ``P`` is the group's smallest SM count.
#
# The three Lamport workspaces are FlashInfer's own MNNVL all-reduce
# workspaces (``MNNVLAllReduceFusionWorkspace``: three rotating symmetric
# buffers initialised to negative zero, multicast mapping, nine-word flags),
# created once per process group by :class:`KimiK3Tp12TailWorkspace`.
# The host runtime here mirrors the Cake TP12 tail module
# (``kimi_k3_tp12_tail.py``); the generated sources live in ``csrc/``.

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    kernel_module_name,
    load_cake_kimi_k3_tp12_tail_module,
    route_available,
)

HIDDEN = 7168
LATENT = 3584
WORLD_SIZE = 12
RMS_EPS = 1.0e-5
ELEMENT_BYTES = 2  # BF16 payload elements
#: Output-column widths per rank (rank order), 128-column granular.
PARTITION = (640,) * 8 + (512,) * 4
#: One-shot K1 up to this many tokens, two-shot above.
ONESHOT_MAX_TOKENS = 16
#: Grouped owner poll-reduce loads up to this many tokens (the ESS range), pinned above.
GROUPED_MAX_TOKENS = 960
#: One 16-byte packet per thread covers one 3584-wide row (K1) or one 3584-wide half row (K3).
THREADS = 448
K3_GRID_Y = 2
#: The one-launch program (``k123:c<capacity>``: the one-shot ESS K1 and the fused K23 tail -- this rank's fp32
#: up-projection slice GEMM in the K2-stream summation order fused with the owner reduce, add, BF16 rounding and
#: multicast all-gather; no ``torch.mm`` -- as two CTA roles of one grid) up to this many tokens; K1, cuBLAS
#: (``torch.mm`` on the contiguous weight-row slice, BF16) and a separate tail kernel above.
K23_MAX_TOKENS = 8
#: K23 accumulator capacities compiled side by side (round 6): ``M`` runs on the smallest capacity ``>= M``, so
#: ``M <= 4`` keeps the round-5 four-token module bit for bit and ``5 <= M <= 8`` takes the eight-token module.  The
#: eight-token rows round once (fp32 slice + shared sum -> BF16) where the round-5 chain rounded twice (cuBLAS BF16 slice,
#: then the add): paired against an fp64 reference their mean / max absolute error is 0.78x / 0.64-0.93x of the round-5
#: chain on every correctness shape (Cake design doc, round 6).
K23_LADDER = (4, 8)
#: Up-projection weight rows (= output columns) per K23 CTA of the one-launch program: ``cols / K23_ROWS`` K23 CTAs
#: after the ``M`` token CTAs, 80 on a 640-column rank, 64 on a 512-column rank.
K23_ROWS = 8
#: The tail runs the persistent token pipeline (``k3_persist*:*``, grid ``(min(M, SM count), 2, 1)``: CTA ``b`` walks
#: tokens ``b, b + P, ...`` scattering token ``t_k`` while reducing ``t_{k-1}`` and gathering ``t_{k-2}``) from this many
#: tokens on; below it an early-shared-scatter K1 (``k1_*_ess``) scatters ``shared_partial`` and the one-CTA-per-token
#: K3-ESS form (``k3_ess:grouped``, grid ``(M, 2, 1)``) or, by ``ROUTE_TABLE``, the persistent tail's reduce mode
#: (``k3_reduce:grouped``) reduces, adds and gathers.  In a paired A/B on GB200 and GB300
#: NVL72 the ESS route is 1.6-7.2 us faster than the persistent one from 640 through 960 tokens and ties or loses from
#: 1023 tokens on.
K3_PERSIST_MIN_TOKENS = GROUPED_MAX_TOKENS + 1
#: The persistent tail pushes its reduce-scatter stage with ``cp.async.bulk`` (``k3_persist_bulk:pinned``) from this many
#: tokens on, i.e. over the whole pinned poll-schedule range.  It equals ``K3_PERSIST_MIN_TOKENS``: the plain persistent
#: tail ``k3_persist:grouped`` and the plain grouped two-shot K1 ``k1_twoshot:grouped`` stay registered, but no token
#: count selects them.
BULK_MIN_TOKENS = K3_PERSIST_MIN_TOKENS
#: Tail route for ``M > ONESHOT_MAX_TOKENS``: ``(first token count, form)`` rows in increasing first-count order; ``M``
#: takes the last row whose first count is ``<= M``.  Forms:
#:
#: * ``"ess"``: ESS two-shot K1 -> cuBLAS -> K3-ESS (``k3_ess:grouped``, one CTA per token and column half);
#: * ``"reduce"``: ESS two-shot K1 -> cuBLAS -> the persistent tail's reduce mode (``k3_reduce:grouped``, grid
#:   ``(min(M, SM count), 2, 1)``: CTA ``b`` reduces token ``t_k`` of tokens ``b, b + P, ...`` over the slots the ESS K1
#:   filled while it gathers the multicast row of ``t_{k-2}``);
#: * ``"persist"``: plain two-shot K1 -> cuBLAS -> the persistent K3 with its own scatter and ``cp.async.bulk`` pushes
#:   (``k3_persist_bulk:pinned``).
#:
#: Every form writes the same packets into the same slots and reduces them in the same order, so ``out`` is bitwise
#: identical across forms.  The ESS K1 forms serve the grouped poll-schedule range only (``M <= GROUPED_MAX_TOKENS``).
#: In a paired A/B on GB300 NVL72 the reduce form ties K3-ESS at 320 and 321 tokens and is 0.6-4.9 us faster from 383
#: through 960 tokens.
ROUTE_TABLE: tuple[tuple[int, str], ...] = (
    (ONESHOT_MAX_TOKENS + 1, "ess"),
    (320, "reduce"),
    (K3_PERSIST_MIN_TOKENS, "persist"),
)
DEFAULT_MAX_TOKENS = 4096
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


def col_begin(partition: tuple[int, ...] = PARTITION) -> tuple[int, ...]:
    """Prefix sums of the column partition: 13 boundaries, the last one 7168."""
    if len(partition) != WORLD_SIZE or sum(partition) != HIDDEN:
        raise ValueError(
            f"partition must have {WORLD_SIZE} widths summing to {HIDDEN}, got {partition}"
        )
    bounds = [0]
    for width in partition:
        bounds.append(bounds[-1] + width)
    return tuple(bounds)


COL_BEGIN = col_begin(PARTITION)


def poll_schedule_for(num_tokens: int) -> str:
    """Owner poll-reduce schedule of the two-shot K1 and of the tail for ``num_tokens``."""
    return "grouped" if num_tokens <= GROUPED_MAX_TOKENS else "pinned"


def route_form_for(num_tokens: int) -> str:
    """The ``ROUTE_TABLE`` form of ``num_tokens > ONESHOT_MAX_TOKENS``."""
    if num_tokens <= ONESHOT_MAX_TOKENS:
        raise ValueError(
            f"num_tokens={num_tokens} <= {ONESHOT_MAX_TOKENS} runs the one-shot routes (no route-table row)"
        )
    form = ROUTE_TABLE[0][1]
    for first, row_form in ROUTE_TABLE:
        if first <= num_tokens:
            form = row_form
    return form


def k1_kernel_key(num_tokens: int, rank: int) -> str:
    """The K1 kernel key: the one-launch program ``k123:c<capacity>`` (K1 and the fused K23 tail in one grid) for
    ``num_tokens <= K23_MAX_TOKENS``, the one-shot early-shared-scatter (ESS) form up to ``ONESHOT_MAX_TOKENS``, then
    the two-shot ESS form for the ``"ess"`` / ``"reduce"`` routes and the plain two-shot form for ``"persist"``."""
    if num_tokens <= K23_MAX_TOKENS:
        return f"k123:c{k23_capacity_for(num_tokens)}"
    if num_tokens <= ONESHOT_MAX_TOKENS:
        return "k1_oneshot_ess"
    if route_form_for(num_tokens) in ("ess", "reduce"):
        return f"k1_twoshot_ess:{poll_schedule_for(num_tokens)}"
    return f"k1_twoshot:{poll_schedule_for(num_tokens)}"


def up_proj_form_for(num_tokens: int) -> str:
    """``"k23"`` (slice GEMM fused into the K23 role of the one-launch program) for ``num_tokens <= K23_MAX_TOKENS``,
    else ``"cublas"``."""
    return "k23" if num_tokens <= K23_MAX_TOKENS else "cublas"


def k23_capacity_for(num_tokens: int) -> int:
    """The K23 accumulator capacity serving ``num_tokens`` (``<= K23_MAX_TOKENS``): the smallest ladder rung ``>= M``."""
    if not 1 <= num_tokens <= K23_MAX_TOKENS:
        raise ValueError(
            f"num_tokens must be in [1, {K23_MAX_TOKENS}] for the one-launch program, got {num_tokens}"
        )
    return next(c for c in K23_LADDER if c >= num_tokens)


def k3_form_for(num_tokens: int) -> str:
    """Tail form for ``num_tokens``: ``"k123"`` (the K23 role of the one-launch program, ``M <= K23_MAX_TOKENS``: no
    tail launch of its own), ``"ess"`` (one CTA per token and column half after an ESS K1: the one-shot rows and the
    ``"ess"`` route), ``"reduce"`` (the persistent reduce mode after an ESS K1: the ``"reduce"`` route), ``"persist"``
    (persistent token pipeline with its own scatter, ``M < BULK_MIN_TOKENS``: an empty range while the two thresholds
    are equal) or ``"persist_bulk"`` (the same with ``cp.async.bulk`` pushes) for the ``"persist"`` route."""
    if num_tokens <= K23_MAX_TOKENS:
        return "k123"
    if num_tokens <= ONESHOT_MAX_TOKENS:
        return "ess"
    form = route_form_for(num_tokens)
    if form in ("ess", "reduce"):
        return form
    if num_tokens < BULK_MIN_TOKENS:
        return "persist"
    return "persist_bulk"


def _tail_launch_form(num_tokens: int) -> str:
    form = k3_form_for(num_tokens)
    if form == "k123":
        raise ValueError(
            f"num_tokens={num_tokens} <= {K23_MAX_TOKENS} runs the one-launch program: no tail launch of its own"
        )
    return form


def k3_kernel_key(num_tokens: int, rank: int) -> str:
    """The tail kernel key for ``num_tokens > K23_MAX_TOKENS`` (below, the one-launch program has no tail launch)."""
    form = _tail_launch_form(num_tokens)
    return f"k3_{form}:{poll_schedule_for(num_tokens)}"


def k123_grid(num_tokens: int, rank: int) -> tuple[int, int, int]:
    """Launch grid of the one-launch program for ``num_tokens <= K23_MAX_TOKENS`` on ``rank``: the ``num_tokens``
    token CTAs, then one K23 CTA per ``K23_ROWS`` output columns of this rank."""
    k23_capacity_for(num_tokens)
    return (num_tokens + PARTITION[rank] // K23_ROWS, 1, 1)


def k3_grid(num_tokens: int, sm_count: int, rank: int) -> tuple[int, int, int]:
    """Launch grid of the tail kernel for ``num_tokens > K23_MAX_TOKENS`` on ``rank`` and a device with ``sm_count``
    SMs."""
    if not isinstance(sm_count, int) or sm_count <= 0:
        raise ValueError(f"sm_count must be a positive integer, got {sm_count!r}")
    form = _tail_launch_form(num_tokens)
    if form == "ess":
        # one CTA per token and column half
        return (num_tokens, K3_GRID_Y, 1)
    # persistent pipeline: one CTA per SM and column half, at most one per token
    return (min(num_tokens, sm_count), K3_GRID_Y, 1)


def route_kernel_keys(num_tokens: int, rank: int) -> tuple[str, ...]:
    """The generated kernels the runtime launches for ``num_tokens`` on ``rank``, in launch order: the one-launch
    program ``(k123:c<capacity>,)`` for ``num_tokens <= K23_MAX_TOKENS``, else ``(K1, tail)`` with cuBLAS between
    them."""
    if num_tokens <= K23_MAX_TOKENS:
        return (k1_kernel_key(num_tokens, rank),)
    return k1_kernel_key(num_tokens, rank), k3_kernel_key(num_tokens, rank)


def workspace_buffer_bytes(max_tokens: int) -> dict[str, int]:
    """Bytes of one Lamport buffer (of three) per workspace for up to ``max_tokens`` tokens."""
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {max_tokens!r}")
    oneshot_tokens = min(ONESHOT_MAX_TOKENS, max_tokens)
    twoshot_tokens = math.ceil(max_tokens / WORLD_SIZE) * WORLD_SIZE
    return {
        # every rank receives the other ranks' packets of every token
        "k1_oneshot": oneshot_tokens * LATENT * WORLD_SIZE * ELEMENT_BYTES,
        # scatter stage (ceil(M / 12) x 12 x 3584) + broadcast stage (M x 3584)
        "k1_twoshot": 2 * twoshot_tokens * LATENT * ELEMENT_BYTES,
        # scatter stage (M x 12 x n_max) + broadcast stage (M x 7168 <= M x 12 x n_max)
        "k3": 2 * max_tokens * WORLD_SIZE * max(PARTITION) * ELEMENT_BYTES,
    }


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 TP12 tail programs require compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def group_sm_count(comm_backend: Any, local_sm_count: int) -> int:
    """The smallest SM count of the group (collective: every rank must call it).

    The persistent tails map token ``t`` to CTA ``t mod P`` and their peer waits pair CTAs of equal index across the
    ranks, so ``P`` must be equal on every rank.
    """
    if not isinstance(local_sm_count, int) or local_sm_count <= 0:
        raise ValueError(
            f"local_sm_count must be a positive integer, got {local_sm_count!r}"
        )
    counts = [int(count) for count in comm_backend.allgather(local_sm_count)]
    if not counts or min(counts) <= 0:
        raise ValueError(f"the group reported SM counts {counts}")
    return min(counts)


def generated_program_available(device: torch.device) -> bool:
    """True when every kernel of the route is registered for ``device``'s architecture."""
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        return False
    from .cake_jit import required_kernel_keys

    return route_available(arch, required_kernel_keys())


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class KimiK3Tp12TailWorkspace:
    """Caller-owned state of one twelve-rank process group.

    Three FlashInfer MNNVL Lamport workspaces (one-shot K1, two-shot K1, K3),
    the per-workspace unicast pointer tables, the normalised-latent and the
    up-projection slice buffers for up to ``max_tokens`` tokens.  Create it
    once (collective: every rank of ``group`` must call it), reuse it for every
    token count up to ``max_tokens``, and :meth:`destroy` it before the
    process group is destroyed.

    The workspace is collective state, like an NCCL communicator: the calls
    that share it run one at a time and in the same order on every rank.  A
    call from another stream than the previous call waits for it
    (:meth:`order_after_previous_stream`; not under CUDA Graph capture).
    ``sm_count`` is the smallest SM count of the group.
    """

    def __init__(
        self,
        *,
        rank: int,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        group: Any = None,
        device: Optional[torch.device] = None,
        comm_backend: Any = None,
    ) -> None:
        import torch.distributed as dist

        from ...comm.comm_backend import TorchDistBackend
        from ...comm.mapping import Mapping
        from ...comm.trtllm_mnnvl_ar import MNNVLAllReduceFusionWorkspace

        if not isinstance(rank, int) or not 0 <= rank < WORLD_SIZE:
            raise ValueError(
                f"rank must be an integer in [0, {WORLD_SIZE}), got {rank!r}"
            )
        if comm_backend is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "KimiK3Tp12TailWorkspace needs an initialised torch.distributed "
                    "process group (or an explicit comm_backend)"
                )
            comm_backend = TorchDistBackend(group)
        world = int(comm_backend.Get_size())
        if world != WORLD_SIZE:
            raise ValueError(
                f"the Kimi-K3 TP12 tail needs a {WORLD_SIZE}-rank group, got {world}"
            )
        if int(comm_backend.Get_rank()) != rank:
            raise ValueError(
                f"rank {rank} does not match the process group rank {comm_backend.Get_rank()}"
            )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index is not None:
            torch.cuda.set_device(device.index)
        self.device = torch.device("cuda", torch.cuda.current_device())
        # persistent K3 grid: one CTA per SM and column half (the Cake runtime's default k3p_ctas), the group's
        # smallest SM count, so the token -> CTA map is the same on every rank
        props = torch.cuda.get_device_properties(self.device)
        self.sm_count = group_sm_count(comm_backend, int(props.multi_processor_count))
        self.rank = rank
        self.world_size = WORLD_SIZE
        self.max_tokens = int(max_tokens)
        self.arch = _device_arch(self.device)
        self.my_col_begin = COL_BEGIN[rank]
        self.my_cols = PARTITION[rank]
        mapping = Mapping(world_size=WORLD_SIZE, rank=rank, tp_size=WORLD_SIZE)
        self.buffers: dict[str, MNNVLAllReduceFusionWorkspace] = {}
        self.peer_ptrs: dict[str, torch.Tensor] = {}
        self._destroyed = False
        #: the stream of the previous launch on this workspace (outside capture)
        self._last_stream: Optional[torch.cuda.Stream] = None
        try:
            for name, size in workspace_buffer_bytes(self.max_tokens).items():
                ws = MNNVLAllReduceFusionWorkspace(
                    mapping, buffer_size_in_bytes=size, comm_backend=comm_backend
                )
                self.buffers[name] = ws
                self.peer_ptrs[name] = torch.tensor(
                    [int(p) for p in ws.ptrs], dtype=torch.int64, device=self.device
                )
            self.y = torch.empty(
                (self.max_tokens, LATENT), dtype=torch.bfloat16, device=self.device
            )
            self.gemm = torch.empty(
                (self.max_tokens, self.my_cols),
                dtype=torch.bfloat16,
                device=self.device,
            )
        except Exception:
            self.destroy()
            raise
        torch.cuda.synchronize(self.device)
        comm_backend.barrier()

    def destroy(self) -> None:
        """Release the symmetric buffers (collective teardown order is the caller's)."""
        if self._destroyed:
            return
        self._destroyed = True
        for ws in list(self.buffers.values()):
            ws.destroy()
        self.buffers.clear()
        self.peer_ptrs.clear()

    def order_after_previous_stream(self) -> None:
        """Order a launch from another stream than the previous launch on this workspace after it.

        The current stream waits on an event recorded on the previous stream, so the calls run one at a time; one
        stream costs a stream compare.  Skipped under CUDA Graph capture: the caller orders graph replays.
        """
        if torch.cuda.is_current_stream_capturing():
            return
        current = torch.cuda.current_stream(self.device)
        previous = self._last_stream
        if previous is not None and previous != current:
            event = torch.cuda.Event()
            event.record(previous)
            current.wait_event(event)
        self._last_stream = current

    def flags(self, name: str) -> torch.Tensor:
        return self.buffers[name].buffer_flags

    def multicast_ptr(self, name: str) -> int:
        return int(self.buffers[name].mc_ptr)

    def local_unicast_ptr(self, name: str) -> int:
        return int(self.buffers[name].uc_ptr_local)


# ---------------------------------------------------------------------------
# Launch binding
# ---------------------------------------------------------------------------


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entry."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_tp12_tail_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class _Launch:
    stage: str
    key: str
    module: str
    kwargs: dict[str, Any] = field(repr=False)
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


@dataclass(frozen=True)
class KimiK3Tp12TailRunner:
    """The prepared launch sequence of one fused tail call on one rank.

    ``launch()`` submits the one-launch program alone for
    ``M <= K23_MAX_TOKENS`` (``k3`` is None); above, K1 (the
    early-shared-scatter form below ``K3_PERSIST_MIN_TOKENS``), cuBLAS for
    the up-projection slice (``cublas``) and the tail kernel -- K3-ESS or the
    persistent reduce mode (by ``ROUTE_TABLE``) below
    ``K3_PERSIST_MIN_TOKENS``, the persistent K3 with ``cp.async.bulk``
    pushes from there on -- in
    order on the current torch stream with no CUDA allocation and no host
    synchronisation; the kernels read every operand on device at launch, so
    the runner (or a CUDA Graph capturing it) replays for new values written
    into the same buffers.  Every rank of the group must launch the same
    token count; prepare a new runner when a shape or a tensor binding
    changes.  A launch from another stream than the previous launch on the
    workspace first waits for it (outside capture).
    """

    num_tokens: int
    rank: int
    arch: str
    k1: _Launch
    #: the tail launch; None for the one-launch program (``M <= K23_MAX_TOKENS``)
    k3: Optional[_Launch]
    y: torch.Tensor
    gemm: torch.Tensor
    up_weight_slice: torch.Tensor
    out: torch.Tensor
    #: cuBLAS ``torch.mm`` between K1 and the tail (``M > K23_MAX_TOKENS``); False = the one-launch program
    cublas: bool
    #: the workspace the launches use; orders a launch from another stream after the previous one
    workspace: Optional[KimiK3Tp12TailWorkspace] = field(
        default=None, repr=False, compare=False
    )

    @property
    def kernel_keys(self) -> tuple[str, ...]:
        """Generated-kernel keys in launch order: ``(k123,)`` for the one-launch program, else ``(K1, tail)``."""
        if self.k3 is None:
            return (self.k1.key,)
        return (self.k1.key, self.k3.key)

    @property
    def module_names(self) -> tuple[str, ...]:
        if self.k3 is None:
            return (self.k1.module,)
        return (self.k1.module, self.k3.module)

    @property
    def launch_count(self) -> int:
        """Launches per call: one for the one-launch program (``M <= K23_MAX_TOKENS``), else K1, cuBLAS and the
        tail."""
        if self.k3 is None:
            return 1
        return 3 if self.cublas else 2

    def launch(self) -> torch.Tensor:
        if self.workspace is not None:
            self.workspace.order_after_previous_stream()
        with tvm_ffi.use_torch_stream():
            if self.k3 is None:
                _launch_one_launch_program(self.k1, self.num_tokens, self.rank)
                return self.out
            self.k1()
            if self.cublas:
                torch.mm(self.y, self.up_weight_slice.t(), out=self.gemm)
            if self.k3 is not None:
                self.k3()
        return self.out

    __call__ = launch


def _clear_cuda_last_error() -> None:
    """Reset the CUDA runtime's last error (``cudaGetLastError``).

    The generated bindings and torch share the process's ``libcudart``, so the call goes to that library, loaded
    already (``RTLD_NOLOAD``); ``cuda.bindings.runtime`` keeps a runtime state of its own and does not reset it.
    """
    major = (torch.version.cuda or "").split(".")[0]
    try:
        cudart = ctypes.CDLL(f"libcudart.so.{major}", mode=os.RTLD_NOLOAD | os.RTLD_NOW)
    except OSError:
        return
    cudart.cudaGetLastError.restype = ctypes.c_int
    cudart.cudaGetLastError()


def _launch_one_launch_program(launch: _Launch, num_tokens: int, rank: int) -> None:
    """Launch the one-launch program; a refused cooperative launch raises a RuntimeError and leaves no CUDA error.

    The driver refuses the cooperative launch when the grid cannot be resident at once, for example on a green
    context or MPS partition with fewer SMs than the grid.  Nothing runs, but the refusal stays as the runtime's last
    error, which the next launch check that reads it (torch's, for example) would raise; it is cleared here.
    """
    try:
        launch()
    except Exception as err:
        if "cooperative launch" not in str(err):
            raise
        _clear_cuda_last_error()
        raise RuntimeError(
            f"kimi_k3_tp12_tail: the one-launch program ({launch.key}, {num_tokens} tokens) needs "
            f"{k123_grid(num_tokens, rank)[0]} SMs resident at once on rank {rank}; the current stream's SMs cannot "
            "hold its grid, so the cooperative launch was refused and nothing ran"
        ) from None


def _check(
    t: torch.Tensor,
    shape: tuple[int, ...],
    name: str,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(t.shape) != tuple(shape) or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {tuple(shape)}, "
            f"got {tuple(t.shape)} {t.dtype}"
        )


def _k1_launch_kwargs(
    k1_key: str,
    *,
    M: int,
    rank: int,
    workspace: Any,
    routed_partial: Any,
    shared_partial: Any,
    y: Any,
    norm_weight: Any,
    up_weight_slice: Any = None,
    out: Any = None,
) -> tuple[str, dict[str, Any]]:
    """The K1 launch arguments for ``k1_key`` (exact-key dispatch; ``name`` is the workspace buffer set).

    ``k123:c<capacity>`` (``M <= K23_MAX_TOKENS``) is the one-launch program: it also takes the K23 tail's operands
    (``up_weight_slice``, ``out``) and its grid holds the K23 CTAs after the token CTAs.  ``k1_oneshot_ess``
    (``M <= ONESHOT_MAX_TOKENS``) carries no poll-schedule suffix; the two-shot keys do.
    """
    if k1_key.startswith("k123:"):
        # one-shot ESS K1 in the first M CTAs, the fused K23 tail in the rest: both workspaces, the weight slice, out
        name = "k1_oneshot"
        k23_ctas = workspace.my_cols // K23_ROWS
        k1_kwargs: dict[str, Any] = dict(
            routed=routed_partial,
            shared=shared_partial,
            y=y,
            gamma=norm_weight,
            w_slice=up_weight_slice,
            out=out,
            mcast_ptr=workspace.multicast_ptr(name),
            local_unicast_ptr=workspace.local_unicast_ptr(name),
            buffer_flags=workspace.flags(name),
            k3_peer_ptrs=workspace.peer_ptrs["k3"],
            k3_mcast_ptr=workspace.multicast_ptr("k3"),
            k3_flags=workspace.flags("k3"),
            num_tokens=M,
            rank=rank,
            my_col_begin=workspace.my_col_begin,
            k23_ctas=k23_ctas,
            epsilon=float(RMS_EPS),
            grid=(M + k23_ctas, 1, 1),
        )
    elif k1_key == "k1_oneshot_ess":
        # early shared scatter: K1 also scatters shared_partial into the "k3" workspace slots
        name = "k1_oneshot"
        k1_kwargs = dict(
            routed=routed_partial,
            shared=shared_partial,
            y_out=y,
            gamma=norm_weight,
            mcast_ptr=workspace.multicast_ptr(name),
            local_unicast_ptr=workspace.local_unicast_ptr(name),
            buffer_flags=workspace.flags(name),
            k3_peer_ptrs=workspace.peer_ptrs["k3"],
            k3_mcast_ptr=workspace.multicast_ptr("k3"),
            k3_flags=workspace.flags("k3"),
            num_tokens=M,
            rank=rank,
            epsilon=float(RMS_EPS),
            grid=(M, 1, 1),
        )
    elif k1_key.startswith("k1_twoshot_ess:"):
        name = "k1_twoshot"
        k1_kwargs = dict(
            routed=routed_partial,
            shared=shared_partial,
            y_out=y,
            gamma=norm_weight,
            peer_ptrs=workspace.peer_ptrs[name],
            mcast_ptr=workspace.multicast_ptr(name),
            buffer_flags=workspace.flags(name),
            k3_peer_ptrs=workspace.peer_ptrs["k3"],
            k3_mcast_ptr=workspace.multicast_ptr("k3"),
            k3_flags=workspace.flags("k3"),
            num_tokens=M,
            rank=rank,
            epsilon=float(RMS_EPS),
            grid=(M, 1, 1),
        )
    else:
        # plain two-shot K1 (M >= K3_PERSIST_MIN_TOKENS): the persistent tail scatters shared_partial itself
        name = "k1_twoshot"
        k1_kwargs = dict(
            routed=routed_partial,
            y_out=y,
            gamma=norm_weight,
            peer_ptrs=workspace.peer_ptrs[name],
            mcast_ptr=workspace.multicast_ptr(name),
            buffer_flags=workspace.flags(name),
            num_tokens=M,
            rank=rank,
            epsilon=float(RMS_EPS),
            grid=(M, 1, 1),
        )
    return name, k1_kwargs


def _k3_launch_kwargs(
    k3_key: str,
    *,
    M: int,
    rank: int,
    workspace: Any,
    grid: tuple[int, int, int],
    shared_partial: Any,
    gemm: Any,
    out: Any,
) -> dict[str, Any]:
    """The tail launch arguments for ``k3_key`` (``k3_ess:<schedule>`` or the persistent K3)."""
    if k3_key.startswith("k3_ess:"):
        # K3-ESS: no shared operand, the ESS K1 already scattered it
        k3_kwargs: dict[str, Any] = dict(
            gemm_slice=gemm,
            out=out,
            peer_ptrs=workspace.peer_ptrs["k3"],
            mcast_ptr=workspace.multicast_ptr("k3"),
            buffer_flags=workspace.flags("k3"),
            num_tokens=M,
            rank=rank,
            my_col_begin=workspace.my_col_begin,
            my_cols=workspace.my_cols,
            gemm_plane_stride=M * workspace.my_cols,
            num_gemm_splits=1,
            grid=grid,
        )
    else:
        # persistent K3 (cp.async.bulk pushes; the unselected plain form and the reduce mode take the same
        # arguments): scatters shared_partial itself, or (reduce mode) reads the slots the ESS K1 filled
        k3_kwargs = dict(
            shared=shared_partial,
            gemm_slice=gemm,
            out=out,
            peer_ptrs=workspace.peer_ptrs["k3"],
            mcast_ptr=workspace.multicast_ptr("k3"),
            buffer_flags=workspace.flags("k3"),
            num_tokens=M,
            rank=rank,
            my_col_begin=workspace.my_col_begin,
            my_cols=workspace.my_cols,
            gemm_plane_stride=M * workspace.my_cols,
            num_gemm_splits=1,
            grid=grid,
        )
    return k3_kwargs


def prepare_kimi_k3_tp12_tail(
    routed_partial: torch.Tensor,
    shared_partial: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    workspace: KimiK3Tp12TailWorkspace,
) -> KimiK3Tp12TailRunner:
    """Validate the operands, select the route for ``M`` and bind the launches.

    The JIT modules are built and loaded here; prepare outside CUDA Graph
    capture.  ``up_weight`` is the replicated ``[7168, 3584]`` BF16 weight;
    the rank's slice is a contiguous row-slice view (no copy).
    """
    if not isinstance(workspace, KimiK3Tp12TailWorkspace):
        raise TypeError("workspace must be a KimiK3Tp12TailWorkspace")
    if workspace._destroyed:
        raise RuntimeError("the workspace has been destroyed")
    M = int(routed_partial.shape[0]) if routed_partial.dim() == 2 else -1
    if M <= 0:
        raise ValueError("routed_partial must be a [M, 3584] tensor with M >= 1")
    if workspace.max_tokens < M:
        raise ValueError(
            f"M={M} exceeds the workspace capacity max_tokens={workspace.max_tokens}"
        )
    _check(routed_partial, (M, LATENT), "routed_partial")
    _check(shared_partial, (M, HIDDEN), "shared_partial")
    _check(norm_weight, (LATENT,), "norm_weight")
    _check(up_weight, (HIDDEN, LATENT), "up_weight")
    _check(out, (M, HIDDEN), "out")
    tensors = dict(
        routed_partial=routed_partial,
        shared_partial=shared_partial,
        norm_weight=norm_weight,
        up_weight=up_weight,
        out=out,
    )
    devices = {t.device for t in tensors.values()}
    if len(devices) != 1 or next(iter(devices)) != workspace.device:
        raise ValueError(
            f"every operand must live on the workspace device {workspace.device}, "
            f"got {sorted(map(str, devices))}"
        )
    arch = workspace.arch
    rank = workspace.rank
    keys = route_kernel_keys(M, rank)
    k1_key = keys[0]
    k1_module = kernel_module_name(arch, k1_key)
    cublas = up_proj_form_for(M) == "cublas"
    y = workspace.y[:M]
    # cuBLAS's BF16 up-projection slice (M > K23_MAX_TOKENS); the one-launch program needs no GEMM buffer
    gemm = workspace.gemm[:M]
    up_weight_slice = up_weight[
        workspace.my_col_begin : workspace.my_col_begin + workspace.my_cols
    ]
    _, k1_kwargs = _k1_launch_kwargs(
        k1_key,
        M=M,
        rank=rank,
        workspace=workspace,
        routed_partial=routed_partial,
        shared_partial=shared_partial,
        y=y,
        norm_weight=norm_weight,
        up_weight_slice=up_weight_slice,
        out=out,
    )
    k1_entry, k1_args = _bind(k1_module, k1_kwargs)
    k1 = _Launch("k1", k1_key, k1_module, k1_kwargs, k1_entry, k1_args)
    if len(keys) == 1:
        return KimiK3Tp12TailRunner(
            num_tokens=M,
            rank=rank,
            arch=arch,
            k1=k1,
            k3=None,
            y=y,
            gemm=gemm,
            up_weight_slice=up_weight_slice,
            out=out,
            cublas=cublas,
            workspace=workspace,
        )
    k3_key = keys[1]
    k3_module = kernel_module_name(arch, k3_key)
    grid = k3_grid(M, workspace.sm_count, rank)
    k3_kwargs = _k3_launch_kwargs(
        k3_key,
        M=M,
        rank=rank,
        workspace=workspace,
        grid=grid,
        shared_partial=shared_partial,
        gemm=gemm,
        out=out,
    )
    k3_entry, k3_args = _bind(k3_module, k3_kwargs)
    return KimiK3Tp12TailRunner(
        num_tokens=M,
        rank=rank,
        arch=arch,
        k1=k1,
        k3=_Launch("k3", k3_key, k3_module, k3_kwargs, k3_entry, k3_args),
        y=y,
        gemm=gemm,
        up_weight_slice=up_weight_slice,
        out=out,
        cublas=cublas,
        workspace=workspace,
    )


__all__ = [
    "BULK_MIN_TOKENS",
    "COL_BEGIN",
    "DEFAULT_MAX_TOKENS",
    "GROUPED_MAX_TOKENS",
    "K23_LADDER",
    "K23_MAX_TOKENS",
    "K23_ROWS",
    "K3_PERSIST_MIN_TOKENS",
    "HIDDEN",
    "KERNELS",
    "KimiK3Tp12TailRunner",
    "KimiK3Tp12TailWorkspace",
    "LATENT",
    "MODULES",
    "ONESHOT_MAX_TOKENS",
    "PARTITION",
    "RMS_EPS",
    "ROUTE_TABLE",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "THREADS",
    "WORLD_SIZE",
    "col_begin",
    "generated_program_available",
    "group_sm_count",
    "k1_kernel_key",
    "k123_grid",
    "k3_form_for",
    "k3_grid",
    "k23_capacity_for",
    "k3_kernel_key",
    "poll_schedule_for",
    "prepare_kimi_k3_tp12_tail",
    "route_form_for",
    "route_kernel_keys",
    "up_proj_form_for",
    "workspace_buffer_bytes",
]
