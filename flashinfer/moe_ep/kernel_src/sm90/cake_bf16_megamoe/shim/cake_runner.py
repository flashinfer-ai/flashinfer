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

from __future__ import annotations

import atexit
import contextlib
import os
import weakref
from enum import Enum

import torch

from ...push_style_megamoe.shim.protocol import (
    _CAPSULE_KEEPALIVE,
    Sm90PushCombine,
    Sm90PushPayload,
    Sm90PushPipe,
    _record_stage,
    _run_guarded_phase,
)
from .cake_gemm import (
    GroupedGemm,
    create_grouped_gemm,
    output_row_capacity,
    validate_grouped_gemm_geometry,
)
from .cake_jit import (
    gen_sm90_cake_bf16_combine_per_route_module,
    gen_sm90_cake_bf16_combine_prereduced_module,
    gen_sm90_cake_bf16_combine_tail_module,
    gen_sm90_cake_bf16_combine_tail_prereduced_module,
    gen_sm90_cake_bf16_compact_module,
    gen_sm90_cake_bf16_dispatch_grid_module,
    gen_sm90_cake_bf16_dispatch_module,
)
from .cake_weights import GATE_UP_GROUP, Sm90CakeBf16Weights

__all__ = [
    "COMBINE_WIRES",
    "COMBINE_WIRE_PER_SHAPE_MAX_TOKENS",
    "COMBINE_WIRE_ENV",
    "COMPACT_FORM_ENV",
    "COPY_FORMS",
    "DIRECT_PLACE_DEFAULTS",
    "DIRECT_PLACE_ENV",
    "DIRECT_PLACE_MIN_ROUTE_ELEMS",
    "DIRECT_PLACE_MIN_TOKENS_ENV",
    "FC2_EARLY_MAX_TOKENS",
    "FC2_EARLY_MAX_TOKENS_ENV",
    "FC2_EARLY_WINDOWS",
    "FC2_EARLY_WINDOWS_ENV",
    "FUSED_BEGIN_ENV",
    "FUSED_COMBINE_TAIL_ENV",
    "FUSED_DISPATCH_ENV",
    "FUSED_DISPATCH_MAX_ROUTES",
    "FUSED_DISPATCH_MAX_TOKENS",
    "PDL_EDGES_ALL",
    "PDL_EDGE_COMPACT",
    "PDL_EDGE_FC1",
    "PDL_EDGE_FC2",
    "PDL_EDGE_PUBLISH",
    "PDL_EDGE_TAIL",
    "PDL_ENV",
    "PDL_LATE_ENV",
    "PER_ROUTE_PUBLISH_PDL_ENV",
    "PREFIX_FORM_ENV",
    "FUSED_DISPATCH_MAX_TOKENS_ENV",
    "GRID_DISPATCH_MIN_TOKENS_ENV",
    "PUBLISH_FORM_ENV",
    "Sm90CakeBf16MoERunner",
]

# Fast-path kernel selection.  Each toggle defaults to the fused kernel; "0" /
# "false" / "off" / "no" selects the vendored kernels for that phase.  Every
# combination is bit-identical and interoperates with peers on any setting.
#
# Receiver-side combine tail: one kernel (wait for every source, fp32 top-k
# reduce, ack) and no per-round combine-inbox fill, vs. the vendored
# wait_combine -> reduce -> ack kernels with the fill.
FUSED_COMBINE_TAIL_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_TAIL"
# Sender-side dispatch: one cooperative launch (count + reserve + store_publish)
# vs. the vendored three kernels.  Used for dedup dispatch with
# T <= the fused-dispatch token bound (default FUSED_DISPATCH_MAX_TOKENS, at most
# FUSED_DISPATCH_MAX_TOKENS_LIMIT; FUSED_DISPATCH_MAX_TOKENS_ENV or the runner's
# ``fused_dispatch_max_tokens``) and T * top_k <= FUSED_DISPATCH_MAX_ROUTES_LIMIT;
# larger rounds and non-dedup pipes always take the vendored kernels.  Rounds
# with T * top_k <= FUSED_DISPATCH_MAX_ROUTES launch the kernel's small-route
# instance, larger ones its large-route instance.
FUSED_DISPATCH_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH"
FUSED_DISPATCH_MAX_TOKENS = 512
FUSED_DISPATCH_MAX_ROUTES = 1024
FUSED_DISPATCH_MAX_TOKENS_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH_MAX_TOKENS"
FUSED_DISPATCH_MAX_TOKENS_LIMIT = 512
FUSED_DISPATCH_MAX_ROUTES_LIMIT = 4096
# Sender-side dispatch of large dedup rounds: one cooperative launch over the
# co-resident blocks (count + reserve + store_publish, plus the round opening
# and the receiver prefix at the fused_begin level) vs. the vendored kernels.
# Rounds with T >= the bound (runner ``grid_dispatch_min_tokens`` or this
# environment variable; default GRID_DISPATCH_MIN_TOKENS, the rounds above the
# default fused-dispatch bound) take it, ahead of the fused small-T path; 0
# disables it.  Bit-identical outputs; interoperates with peers on any setting.
GRID_DISPATCH_MIN_TOKENS_ENV = "FLASHINFER_SM90_CAKE_BF16_GRID_DISPATCH_MIN_TOKENS"
GRID_DISPATCH_MIN_TOKENS = FUSED_DISPATCH_MAX_TOKENS + 1
_GRID_DISPATCH_MAX_SMEM_INTS = (
    12 * 1024
)  # 2 * E * ep + 1 + 2 * ep ints of dynamic smem (48 KB)
# Combine wire format (NOT interoperable: every rank of a pipe must agree, checked
# at construction).  "prereduced": the expert rank pre-reduces the routes of a
# token that landed on it in fp32 (ascending k), rounds once to bf16 and sends
# one row per (token, source rank); the owner sums the <= ep_size rows in
# ascending rank order.  "per_route": one bf16(y_k * w_k) row per route, summed
# over k by the owner.  Both deterministic; outputs differ by rounding only.
COMBINE_WIRE_ENV = "FLASHINFER_SM90_CAKE_BF16_COMBINE_WIRE"
COMBINE_WIRE_PREREDUCED = "prereduced"
# "prereduced_hilo": as "prereduced", but a group of >= 2 routes carries its fp32
# partial as two bf16 rows (hi + residual) so no group partial is rounded to bf16;
# single-route groups are bf16(w * y) exactly as the per-route wire.
COMBINE_WIRE_PREREDUCED_HILO = "prereduced_hilo"
COMBINE_WIRE_PER_ROUTE = "per_route"
COMBINE_WIRES = (
    COMBINE_WIRE_PREREDUCED,
    COMBINE_WIRE_PREREDUCED_HILO,
    COMBINE_WIRE_PER_ROUTE,
)
_PREREDUCED_WIRES = (COMBINE_WIRE_PREREDUCED, COMBINE_WIRE_PREREDUCED_HILO)
# Per-shape default (combine_wire=None, environment unset): rounds whose token
# capacity is at or below this bound run the per-route wire (R0 numerics; the
# pre-reduced grouping is not repaid at the protocol's fixed-cost floor), larger
# capacities run the pre-reduced wire.  The capacity is a pipe-creation constant
# identical on every rank, so the selection is rank-consistent by construction.
COMBINE_WIRE_PER_SHAPE_MAX_TOKENS = 8
# Programmatic dependent launch of this package's kernels (compact, FC1, FC2,
# combine publish, combine tail): each one is scheduled while its predecessor
# finishes and waits for it (griddepcontrol.wait) before touching memory.
# The value is on/off or a mask of PDL_EDGE_* bits (the kernels launched with
# the PDL attribute).  With the per-route wire a PDL-launched publish runs this
# package's twin of the vendored kernel (same bytes on the wire, self-resetting
# row counters).
PDL_ENV = "FLASHINFER_SM90_CAKE_BF16_PDL"
PDL_EDGE_COMPACT = 1
PDL_EDGE_FC1 = 2
PDL_EDGE_FC2 = 4
PDL_EDGE_PUBLISH = 8
PDL_EDGE_TAIL = 16
PDL_EDGES_ALL = 31
# Default edges: FC1 and FC2 (each GEMM is launched as a programmatic dependent
# of the kernel before it).
PDL_EDGES_DEFAULT = PDL_EDGE_FC1 | PDL_EDGE_FC2
# 1 (default) = this package's small kernels let their PDL dependents be
# scheduled only when they exit instead of right after their own wait (the
# early form lets the dependent's blocks occupy the SMs while the producer runs).
PDL_LATE_ENV = "FLASHINFER_SM90_CAKE_BF16_PDL_LATE"
# 1 (default) = with the per-route wire (token capacity at or below
# COMBINE_WIRE_PER_SHAPE_MAX_TOKENS by default) the combine publish is launched
# with the PDL attribute as if PDL_EDGE_PUBLISH were in the mask (it then runs
# this package's per-route twin); pipes on the pre-reduced wires are unaffected.
PER_ROUTE_PUBLISH_PDL_ENV = "FLASHINFER_SM90_CAKE_BF16_PER_ROUTE_PUBLISH_PDL"
# Round opening folded into the small-round fused dispatch (requires the fused
# tail and the fused dispatch path): 1 = the tag bump and the ack wait run in
# the dispatch launch, 2 (default) = also the receiver-side prefix (wait_prefix).
FUSED_BEGIN_ENV = "FLASHINFER_SM90_CAKE_BF16_FUSED_BEGIN"
FUSED_BEGIN_LEVELS = (0, 1, 2)
_FUSED_PREFIX_MAX_KEYS = (
    1024  # E * ep + 2 * ep bound of the fused prefix (dispatch smem)
)
# Kernel forms of the compact (inbox rows -> expert-major activation matrix) and of
# the pre-reduced combine publish: 0 = the one-block-per-row compact and the
# route-serial publish, 1 = the latency-pipelined compact (several rows in flight,
# loads hoisted before stores) and the load-hoisted publish (default).  Both
# forms write the same bytes, so ranks may run different forms.
COMPACT_FORM_ENV = "FLASHINFER_SM90_CAKE_BF16_COMPACT_FORM"
PUBLISH_FORM_ENV = "FLASHINFER_SM90_CAKE_BF16_PUBLISH_FORM"
COPY_FORMS = (0, 1)
# FC2 early start in the decode window: rounds with 0 < local tokens <= this bound
# launch FC1 variants that release per-m-tile readiness counters and an FC2 that
# claims its tiles dynamically as they become ready (no griddepcontrol.wait), so
# FC2 tiles fill the SMs that FC1's tail wave leaves free.  Same tiles, same
# instruction stream per tile, same bits; larger rounds run the regular pair.
# 0 disables it.  FC1 and FC2 are rank-local, so ranks may differ per round.
FC2_EARLY_MAX_TOKENS_ENV = "FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_MAX_TOKENS"
FC2_EARLY_MAX_TOKENS = 128
# FC2 early start above the decode window: rounds whose local token count lies in
# the window (lo, hi] of the pipe's shape also run the early-start pair.  Keyed
# by host-known values only: (hidden size, intermediate size, total experts,
# top_k, EP size).  The windows hold the measured shapes and token ranges where
# the early-start pair was faster than the regular pair on H100; other shapes and
# rounds keep the decode window alone (there the early FC2 tiles slowed FC1 by
# more than they saved, or made no difference; the (7168, 2048, 256, 8) shape has
# no window at EP 2 because its gain there was below 0.3 % in the latest
# measurement).  Same tiles, same bits.  0 disables the windows (the decode window
# is unaffected).  Rank-local, so ranks may differ.
FC2_EARLY_WINDOWS_ENV = "FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_WINDOWS"
FC2_EARLY_WINDOWS: dict[tuple[int, int, int, int, int], tuple[int, int]] = {
    # (hidden, intermediate, total experts, top_k, ep): (lo, hi)
    (7168, 2048, 256, 8, 8): (128, 4096),
    (7168, 2048, 256, 8, 4): (128, 1024),
    (7168, 3072, 384, 6, 8): (512, 1024),
}
# Receiver-side prefix inside the single-launch dispatches (fused_begin 2): 1
# (default) = the small-round fused dispatch and the grid dispatch compute the
# integer prefix sums over the E x ep (expert, source) keys with the whole block,
# 0 = one thread walks the keys (the vendored wait_prefix body) in every
# dispatch.  The 129..512-token fused dispatch walks the keys with one thread in
# both settings.  Both forms write the same values, so ranks may differ.
PREFIX_FORM_ENV = "FLASHINFER_SM90_CAKE_BF16_PREFIX_FORM"
# Direct placement in grid-dispatch rounds: the activation matrix of every rank is a
# symmetric (peer-writable) buffer; the sender stores the first route of a token to
# a rank (its carrier) straight into that rank's matrix at the row the compact would
# have copied it to, and the compact copies only the later routes of the token to
# the same rank (from the carrier's row).  A round is direct only when every rank
# dispatches it this way (decided on the device, the same on every rank); other
# rounds use the inbox path.  Requested with 1 (``direct_place=True``; by default
# on for the EP sizes in DIRECT_PLACE_DEFAULTS, off for the others); effective only
# when every rank requests it and its pipe allows it (dedup dispatch, the grid
# dispatch on, compact form 1, at most 1024 (expert, rank) keys, a token capacity of
# at least both round bounds below), and then for grid-dispatch rounds with at least
# DIRECT_PLACE_MIN_TOKENS_ENV (``direct_place_min_tokens``; default at the EP sizes
# in DIRECT_PLACE_DEFAULTS the larger of the EP size's bound and the routed-element
# floor below, else the grid-dispatch bound) local tokens.  A pipe that cannot use
# it keeps the regular activation matrix.
# Bit-identical outputs.
DIRECT_PLACE_ENV = "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE"
DIRECT_PLACE_MIN_TOKENS_ENV = "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE_MIN_TOKENS"
# EP size -> smallest local token count of a direct round, for the EP sizes where
# direct placement was measured faster than the inbox path on H100 (on by default
# there).  Other EP sizes (EP 2: no measured gain) keep the inbox path unless
# direct placement is requested.
DIRECT_PLACE_DEFAULTS: dict[int, int] = {8: 513, 4: 513}
# Routed-element floor of the default bound: a direct round adds a count exchange of
# a fixed cost and saves compact copies in proportion to the round's routed
# elements (local tokens * top_k * hidden).  On H100 the rounds with at least 44M
# routed elements gained clearly and the round with 25M did not, so by default a
# direct round needs ceil(DIRECT_PLACE_MIN_ROUTE_ELEMS / (top_k * hidden)) local
# tokens (586 at hidden 7168 / top-k 8, 781 at 7168 / 6, 1366 at 4096 / 6).
DIRECT_PLACE_MIN_ROUTE_ELEMS = 1 << 25
_DIRECT_PLACE_MAX_KEYS = 1024  # E * ep: the pipelined compact's segment tables
_DIRECT_PLACE_MAX_ROWS = 1 << 30  # the SlotMeta row field (bits 0..29)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def _fused_combine_tail_default() -> bool:
    return _env_flag(FUSED_COMBINE_TAIL_ENV)


def _fused_dispatch_default() -> bool:
    return _env_flag(FUSED_DISPATCH_ENV)


def _fused_dispatch_max_tokens_default() -> int:
    raw = os.environ.get(FUSED_DISPATCH_MAX_TOKENS_ENV)
    return FUSED_DISPATCH_MAX_TOKENS if raw is None else int(raw.strip())


def _grid_dispatch_min_tokens_default() -> int:
    raw = os.environ.get(GRID_DISPATCH_MIN_TOKENS_ENV)
    return GRID_DISPATCH_MIN_TOKENS if raw is None else int(raw.strip())


def _pdl_mask(value: bool | int | str) -> int:
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in ("0", "false", "off", "no"):
            return 0
        if raw in ("1", "true", "on", "yes"):
            return PDL_EDGES_ALL
        value = int(raw)
    if isinstance(value, bool):
        return PDL_EDGES_ALL if value else 0
    mask = int(value)
    if not 0 <= mask <= PDL_EDGES_ALL:
        raise ValueError(
            f"pdl must be a bool or an edge mask in [0, {PDL_EDGES_ALL}], got {value!r}"
        )
    return mask


def _pdl_default() -> int:
    raw = os.environ.get(PDL_ENV)
    return PDL_EDGES_DEFAULT if raw is None else _pdl_mask(raw)


def _per_route_publish_pdl_default() -> bool:
    raw = os.environ.get(PER_ROUTE_PUBLISH_PDL_ENV, "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "")


def _pdl_late_default() -> bool:
    return os.environ.get(PDL_LATE_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _fused_begin_default() -> int:
    raw = os.environ.get(FUSED_BEGIN_ENV, "2").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return 0
    if raw in ("1", "2"):
        return int(raw)
    raise ValueError(f"{FUSED_BEGIN_ENV} must be one of 0, 1, 2, got {raw!r}")


def _form_default(env: str) -> int:
    raw = os.environ.get(env, "1").strip().lower()
    if raw in ("0", "false", "off", "no"):
        return 0
    if raw in ("1", "true", "on", "yes"):
        return 1
    raise ValueError(f"{env} must be 0 or 1, got {raw!r}")


def _form(value: int | None, env: str, name: str) -> int:
    form = _form_default(env) if value is None else int(value)
    if form not in COPY_FORMS:
        raise ValueError(f"{name} must be one of {COPY_FORMS}, got {value!r}")
    return form


def _fc2_early_max_tokens_default() -> int:
    raw = os.environ.get(FC2_EARLY_MAX_TOKENS_ENV)
    return FC2_EARLY_MAX_TOKENS if raw is None else int(raw.strip())


def _direct_place_default(ep: int) -> bool:
    raw = os.environ.get(DIRECT_PLACE_ENV)
    if raw is None:
        return ep in DIRECT_PLACE_DEFAULTS
    raw = raw.strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{DIRECT_PLACE_ENV} must be 0 or 1, got {raw!r}")
    return raw == "1"


def _direct_place_min_tokens_default(ep: int, top_k: int, hidden: int) -> int:
    raw = os.environ.get(DIRECT_PLACE_MIN_TOKENS_ENV)
    if raw is not None:
        return int(raw.strip())
    if ep in DIRECT_PLACE_DEFAULTS:
        floor = -(-DIRECT_PLACE_MIN_ROUTE_ELEMS // (top_k * hidden))  # ceil
        return max(DIRECT_PLACE_DEFAULTS[ep], floor)
    return GRID_DISPATCH_MIN_TOKENS


def _align256(nbytes: int) -> int:
    return (nbytes + 255) // 256 * 256


# Symmetric a1 views carry C++ deleters that must not run during interpreter
# finalization (see protocol.py): atexit drops them early, the buffer itself is left
# to the process exit since a peer may still map it.
_LIVE_RUNNERS: list = []


def _drain_live_runners() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    for ref in _LIVE_RUNNERS:
        runner = ref()
        if runner is not None:
            runner._release_direct_views()
    _LIVE_RUNNERS.clear()


atexit.register(_drain_live_runners)


def _fc2_early_windows_default() -> bool:
    raw = os.environ.get(FC2_EARLY_WINDOWS_ENV, "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "")


def fc2_early_window_for(
    hidden_size: int, intermediate_size: int, num_experts: int, top_k: int, ep: int
) -> tuple[int, int] | None:
    """The FC2 early-start window (lo, hi] above the decode window for a pipe
    shape (``num_experts`` = total experts over the EP group), or ``None``."""
    return FC2_EARLY_WINDOWS.get(
        (
            int(hidden_size),
            int(intermediate_size),
            int(num_experts),
            int(top_k),
            int(ep),
        )
    )


def fc2_early_round(
    num_tokens: int, max_tokens: int, window: tuple[int, int] | None
) -> bool:
    """True when a round of ``num_tokens`` local tokens runs the FC2 early-start
    pair: the decode window ``0 < T <= max_tokens`` or the shape window
    ``lo < T <= hi``."""
    if 0 < num_tokens <= max_tokens:
        return True
    return window is not None and window[0] < num_tokens <= window[1]


def _combine_wire_default(token_capacity: int) -> str:
    """Default combine wire: the environment override when set, otherwise the
    per-shape default (``per_route`` when ``token_capacity`` is at or below
    ``COMBINE_WIRE_PER_SHAPE_MAX_TOKENS``, ``prereduced`` above)."""
    raw = os.environ.get(COMBINE_WIRE_ENV)
    if raw is not None:
        value = raw.strip().lower()
        if value not in COMBINE_WIRES:
            raise ValueError(
                f"{COMBINE_WIRE_ENV} must be one of {COMBINE_WIRES}, got {value!r}"
            )
        return value
    if int(token_capacity) <= COMBINE_WIRE_PER_SHAPE_MAX_TOKENS:
        return COMBINE_WIRE_PER_ROUTE
    return COMBINE_WIRE_PREREDUCED


class _RunnerState(Enum):
    IDLE = "idle"
    STAGED = "staged"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


class Sm90CakeBf16MoERunner:
    """Native BF16 SM90 MegaMoE forward over a :class:`Sm90PushPipe`.

    Round structure (all on the caller's stream):

    ``begin_round`` (round-tag bump, wait for the peers' acks) -> ``dispatch``
    (bf16 payload, P2P push; one cooperative kernel for small rounds, the
    vendored count / reserve / store_publish kernels otherwise) | ``wait_prefix``
    -> ``compact_bf16`` (gather inbox rows expert-major) -> FC1 (Cake WGMMA,
    fused SwiGLU, bf16 intermediate) -> FC2 (Cake WGMMA, bf16 out) ->
    ``combine`` -> ``combine_tail``.

    Combine wire (``combine_wire``; ``None`` = ``COMBINE_WIRE_ENV`` when set,
    otherwise the per-shape default: ``per_route`` for a token capacity at or
    below ``COMBINE_WIRE_PER_SHAPE_MAX_TOKENS``, ``prereduced`` above):

    * ``"prereduced"``: ``combine`` groups the received rows by (owner, token),
      pre-reduces each group in fp32 (``fmaf`` in ascending route order), rounds
      once to bf16 and pushes one row per (token, source rank) into the owner's
      inbox slot ``k_min`` of the group; ``combine_tail`` (one kernel) waits for
      every source, sums the group rows in ascending source-rank order in fp32,
      rounds once and acks.  Requires the fused tail.
    * ``"per_route"``: the vendored ``combine_publish`` (one
      ``bf16(fp32(y_k) * w_k)`` row per route) and either the fused tail (fp32
      sum over the unmasked top-k slots in route order) or, with
      ``fused_combine_tail=False``, the vendored ``wait_combine`` -> ``reduce``
      -> ``ack`` kernels with the per-round combine-inbox fill.

    Every rank of a pipe must use the same ``combine_wire`` (verified with an
    allgather at construction).  Both wires are deterministic; the two outputs
    differ by rounding only (per_route rounds every route and the sum,
    prereduced rounds each group partial and the sum).  Within one wire the
    fused / vendored kernel choices (``fused_combine_tail``, ``fused_dispatch``,
    default from ``FUSED_COMBINE_TAIL_ENV`` / ``FUSED_DISPATCH_ENV``) are
    bit-identical and a peer rank may use any of them.

    Direct placement (``direct_place``; ``None`` = ``DIRECT_PLACE_ENV``, default
    on for the EP sizes in ``DIRECT_PLACE_DEFAULTS``): grid-dispatch rounds store
    each token's row for a rank straight into
    that rank's activation matrix (a symmetric buffer), so the compact copies only
    the later routes of a token to the same rank.  It is effective only when every
    rank of the pipe requests it and can run it; a round is direct only when every
    rank dispatches it through the grid dispatch with at least
    ``direct_place_min_tokens`` tokens (``None`` = ``DIRECT_PLACE_MIN_TOKENS_ENV``,
    default at the EP sizes in ``DIRECT_PLACE_DEFAULTS`` the larger of the listed
    bound and ``ceil(DIRECT_PLACE_MIN_ROUTE_ELEMS / (top_k * hidden))``), otherwise
    the round takes the inbox path.
    Bit-identical outputs.

    Precision contract: bf16 operands into both GEMMs, fp32 accumulation, bf16
    intermediate, bf16 dispatch payload and combine wire, bf16 output.  No FP8
    anywhere.  The pipe must be configured with ``payload_dtype=BF16`` and
    ``combine_dtype=BF16``.
    """

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: Sm90CakeBf16Weights,
        *,
        gemm: GroupedGemm | None = None,
        clamp: float | None = None,
        fused_combine_tail: bool | None = None,
        fused_dispatch: bool | None = None,
        combine_wire: str | None = None,
        pdl: bool | int | None = None,
        pdl_late: bool | None = None,
        fused_begin: int | None = None,
        fused_dispatch_max_tokens: int | None = None,
        grid_dispatch_min_tokens: int | None = None,
        compact_form: int | None = None,
        publish_form: int | None = None,
        per_route_publish_pdl: bool | None = None,
        fc2_early_max_tokens: int | None = None,
        prefix_form: int | None = None,
        direct_place: bool | None = None,
        direct_place_min_tokens: int | None = None,
        fc2_early_windows: bool | None = None,
    ) -> None:
        self.pipe = pipe
        self._state = _RunnerState.IDLE
        self._staged_tokens: int | None = None
        self._staged_topk_ids: torch.Tensor | None = None
        self._staged_stream: torch.cuda.Stream | None = None
        self._staged_stream_capturing = False
        self._caller_ready_event: torch.cuda.Event | None = None
        self._round_event: torch.cuda.Event | None = None
        self._round_stream_id: int | None = None
        self.record_stages = False
        self._clamp = clamp
        self._fused_tail = (
            _fused_combine_tail_default()
            if fused_combine_tail is None
            else bool(fused_combine_tail)
        )
        self._fused_dispatch = (
            _fused_dispatch_default()
            if fused_dispatch is None
            else bool(fused_dispatch)
        )
        wire = (
            _combine_wire_default(pipe.token_capacity)
            if combine_wire is None
            else str(combine_wire)
        )
        if wire not in COMBINE_WIRES:
            raise ValueError(
                f"combine_wire must be one of {COMBINE_WIRES}, got {combine_wire!r}"
            )
        if wire in _PREREDUCED_WIRES and not self._fused_tail:
            raise ValueError(
                "combine_wire='prereduced' requires the fused combine tail "
                f"(fused_combine_tail=True / {FUSED_COMBINE_TAIL_ENV}=1)"
            )
        self._combine_wire = wire
        self._pdl = _pdl_default() if pdl is None else _pdl_mask(pdl)
        self._pdl_late = _pdl_late_default() if pdl_late is None else bool(pdl_late)
        level = _fused_begin_default() if fused_begin is None else int(fused_begin)
        if level not in FUSED_BEGIN_LEVELS:
            raise ValueError(
                f"fused_begin must be one of {FUSED_BEGIN_LEVELS}, got {fused_begin!r}"
            )
        # the grid dispatch takes the round opening (and the prefix) at the same level
        self._grid_begin = (
            level if (self._fused_tail and pipe.config.dedup_dispatch) else 0
        )
        if not (
            self._fused_tail and self._fused_dispatch and pipe.config.dedup_dispatch
        ):
            level = 0  # the round opening is fused into the fused dispatch launch only
        if level == 2 and pipe.E * pipe.ep + 2 * pipe.ep > _FUSED_PREFIX_MAX_KEYS:
            level = 1
        self._fused_begin = level
        max_tokens = (
            _fused_dispatch_max_tokens_default()
            if fused_dispatch_max_tokens is None
            else int(fused_dispatch_max_tokens)
        )
        if not 1 <= max_tokens <= FUSED_DISPATCH_MAX_TOKENS_LIMIT:
            raise ValueError(
                f"fused_dispatch_max_tokens must be in [1, {FUSED_DISPATCH_MAX_TOKENS_LIMIT}], "
                f"got {max_tokens!r}"
            )
        self._fused_dispatch_max_tokens = max_tokens
        grid_min = (
            _grid_dispatch_min_tokens_default()
            if grid_dispatch_min_tokens is None
            else int(grid_dispatch_min_tokens)
        )
        if grid_min < 0:
            raise ValueError(
                f"grid_dispatch_min_tokens must be >= 0, got {grid_dispatch_min_tokens!r}"
            )
        if not pipe.config.dedup_dispatch or (
            2 * pipe.E * pipe.ep + 1 + 2 * pipe.ep > _GRID_DISPATCH_MAX_SMEM_INTS
        ):
            grid_min = 0  # dedup payload mode only; the key tables must fit in smem
        self._grid_dispatch_min_tokens = grid_min
        self._compact_form = _form(compact_form, COMPACT_FORM_ENV, "compact_form")
        self._publish_form = _form(publish_form, PUBLISH_FORM_ENV, "publish_form")
        early_max = (
            _fc2_early_max_tokens_default()
            if fc2_early_max_tokens is None
            else int(fc2_early_max_tokens)
        )
        if early_max < 0:
            raise ValueError(
                f"fc2_early_max_tokens must be >= 0, got {fc2_early_max_tokens!r}"
            )
        self._fc2_early_max_tokens = early_max  # effective value fixed after the JIT
        self._fc2_early_windows = (
            _fc2_early_windows_default()
            if fc2_early_windows is None
            else bool(fc2_early_windows)
        )
        self._fc2_early_window: tuple[int, int] | None = None  # fixed after the JIT
        self._gemm_sched: torch.Tensor | None = None
        self._prefix_form = _form(prefix_form, PREFIX_FORM_ENV, "prefix_form")
        if direct_place is None:
            requested = _direct_place_default(pipe.ep)
        elif direct_place in (0, 1):  # bool or 0 / 1
            requested = bool(direct_place)
        else:
            raise ValueError(
                f"direct_place must be a bool or None, got {direct_place!r}"
            )
        direct_min = (
            _direct_place_min_tokens_default(pipe.ep, pipe.K, pipe.H)
            if direct_place_min_tokens is None
            else int(direct_place_min_tokens)
        )
        if direct_min < 0:
            raise ValueError(
                f"direct_place_min_tokens must be >= 0, got {direct_place_min_tokens!r}"
            )
        self._direct_min = direct_min
        # this rank can run it; effective (self._direct) = every rank can (init handshake)
        self._direct_local = (
            requested
            and pipe.config.dedup_dispatch
            and self._grid_dispatch_min_tokens > 0
            and self._compact_form == 1
            and pipe.E * pipe.ep <= _DIRECT_PLACE_MAX_KEYS
            and output_row_capacity(pipe.m_cap) < _DIRECT_PLACE_MAX_ROWS
            # a pipe none of whose rounds can be placed directly keeps the regular a1
            and pipe.token_capacity >= max(direct_min, self._grid_dispatch_min_tokens)
        )
        self._direct = False
        # symmetric a1 + count all-gather area (direct placement)
        self._a1_symm: object | None = None
        self._a1_peers = 0
        self._ag_offset = 0
        self._ag_view: torch.Tensor | None = None
        self._grid_a1base: torch.Tensor | None = None
        self._staged_direct = False
        self._last_round_direct = False
        self._per_route_publish_pdl = (
            _per_route_publish_pdl_default()
            if per_route_publish_pdl is None
            else bool(per_route_publish_pdl)
        )
        # effective edge mask of this pipe's launches (``pdl`` reports the requested one)
        self._pdl_edges = self._pdl
        if self._per_route_publish_pdl and wire == COMBINE_WIRE_PER_ROUTE:
            self._pdl_edges |= PDL_EDGE_PUBLISH
        # the per-route wire publishes through this package's twin under PDL
        self._own_per_route = (
            bool(self._pdl_edges & PDL_EDGE_PUBLISH)
            and self._fused_tail
            and wire == COMBINE_WIRE_PER_ROUTE
        )
        self._staged_prefix_done = False
        self.weights: Sm90CakeBf16Weights | None = None

        def _local_init():
            if pipe.config.payload_dtype != Sm90PushPayload.BF16:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake requires Sm90PushConfig(payload_dtype=BF16); "
                    f"got {pipe.config.payload_dtype!r}"
                )
            if pipe.config.combine_dtype != Sm90PushCombine.BF16:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake requires Sm90PushConfig(combine_dtype=BF16); "
                    f"got {pipe.config.combine_dtype!r}"
                )
            if pipe.config.fuse_fc1_epilogue:
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake implements its own fused FC1 epilogue; "
                    "Sm90PushConfig.fuse_fc1_epilogue must be False"
                )
            self._check_weights(weights)
            validate_grouped_gemm_geometry(
                hidden_size=pipe.H,
                intermediate_size=weights.intermediate_size,
                gate_up_group=GATE_UP_GROUP,
                row_capacity=output_row_capacity(pipe.m_cap),
            )
            self.I = weights.intermediate_size
            self.weights = weights
            self._init_buffers()
            return None

        _run_guarded_phase(pipe._comm, pipe.rank, "weights+buffers", _local_init)

        def _wire_handshake():
            # the combine wire is a cross-rank contract (slot keying + reduce order)
            wires = list(pipe._comm.allgather(self._combine_wire))
            if any(w != self._combine_wire for w in wires):
                raise ValueError(
                    "sm90_bf16_bf16_bf16_push_cake: combine_wire must match on every EP rank; "
                    f"got {wires}"
                )
            return None

        _run_guarded_phase(
            pipe._comm, pipe.rank, "cake-bf16-combine-wire", _wire_handshake
        )

        # direct placement needs every rank: a rank that cannot run it disables it for
        # the pipe (no error, ranks may request different switches)
        direct_flags = _run_guarded_phase(
            pipe._comm, pipe.rank, "cake-bf16-direct-place", lambda: self._direct_local
        )
        self._direct = all(direct_flags)
        _run_guarded_phase(
            pipe._comm, pipe.rank, "cake-bf16-direct-a1", self._init_direct_a1
        )
        if self._direct:
            _LIVE_RUNNERS.append(weakref.ref(self))

        def _jit():
            self.compact_module = gen_sm90_cake_bf16_compact_module().build_and_load()
            if self._combine_wire in _PREREDUCED_WIRES:
                self.publish_module = (
                    gen_sm90_cake_bf16_combine_prereduced_module().build_and_load()
                )
                self.tail_module = (
                    gen_sm90_cake_bf16_combine_tail_prereduced_module().build_and_load()
                )
            else:
                self.publish_module = (
                    gen_sm90_cake_bf16_combine_per_route_module().build_and_load()
                    if self._own_per_route
                    else None
                )
                self.tail_module = (
                    gen_sm90_cake_bf16_combine_tail_module().build_and_load()
                    if self._fused_tail
                    else None
                )
            self.dispatch_module = (
                gen_sm90_cake_bf16_dispatch_module().build_and_load()
                if self._fused_dispatch
                else None
            )
            self.dispatch_grid_module = (
                gen_sm90_cake_bf16_dispatch_grid_module().build_and_load()
                if self._grid_dispatch_min_tokens > 0
                else None
            )
            self.gemm = gemm if gemm is not None else create_grouped_gemm(clamp=clamp)
            early_ok = (
                pipe.E <= 1023  # the FC2 tile id carries the expert in 10 bits
                and min(pipe.H, self.I) >= 256  # >= 4 k-stages per FC2 tile
                and getattr(self.gemm, "supports_early_start", False)
            )
            if not early_ok:
                self._fc2_early_max_tokens = 0
            if early_ok and self._fc2_early_windows:
                self._fc2_early_window = fc2_early_window_for(
                    pipe.H, self.I, pipe.E * pipe.ep, pipe.K, pipe.ep
                )
            if self._fc2_early_max_tokens > 0 or self._fc2_early_window is not None:
                # FC2 tile ticket, FC2 exit ticket, FC1 go word, FC1 CTA count, then one
                # FC1 readiness counter per m-tile: zeroed once here and kept zeroed by
                # FC2's last CTA every round (no per-round memset, graph-safe)
                self._gemm_sched = torch.zeros(
                    16 + self.m_buf // 128 + pipe.E,
                    dtype=torch.int32,
                    device=pipe.device,
                )
            return None

        _run_guarded_phase(pipe._comm, pipe.rank, "cake-bf16-jit", _jit)

    @property
    def fused_combine_tail(self) -> bool:
        """True when the receiver-side combine tail runs as the single fused kernel."""
        return self._fused_tail

    @property
    def fused_dispatch(self) -> bool:
        """True when small dedup rounds dispatch through the single cooperative kernel."""
        return self._fused_dispatch

    @property
    def combine_wire(self) -> str:
        """``"prereduced"`` or ``"per_route"`` (see the class docstring)."""
        return self._combine_wire

    @property
    def pdl(self) -> int:
        """Mask of PDL_EDGE_* bits: the kernels launched as programmatic dependent launches."""
        return self._pdl

    @property
    def per_route_publish_pdl(self) -> bool:
        """True when a per-route-wire pipe launches its publish as a PDL secondary (``PER_ROUTE_PUBLISH_PDL_ENV``)."""
        return self._per_route_publish_pdl

    @property
    def pdl_late(self) -> bool:
        """True when the small kernels let their PDL dependents start only at exit."""
        return self._pdl_late

    def _pdl_flag(self, edge: int) -> int:
        # bit 0: launch with the PDL attribute; bit 1: no early trigger
        return (1 if self._pdl_edges & edge else 0) | (2 if self._pdl_late else 0)

    @property
    def fused_dispatch_max_tokens(self) -> int:
        """Largest round (tokens) that takes the single-launch fused dispatch."""
        return self._fused_dispatch_max_tokens

    @property
    def grid_dispatch_min_tokens(self) -> int:
        """Smallest round (tokens) that takes the grid-strided single-launch dispatch (0 = off)."""
        return self._grid_dispatch_min_tokens

    @property
    def compact_form(self) -> int:
        """Compact kernel form (0 or 1; see ``COMPACT_FORM_ENV``)."""
        return self._compact_form

    @property
    def publish_form(self) -> int:
        """Pre-reduced publish kernel form (0 or 1; see ``PUBLISH_FORM_ENV``)."""
        return self._publish_form

    @property
    def fc2_early_max_tokens(self) -> int:
        """Largest round (local tokens) that starts FC2 in FC1's tail wave (0 = off)."""
        return self._fc2_early_max_tokens

    @property
    def fc2_early_windows(self) -> bool:
        """True when the per-shape FC2 early-start windows are enabled."""
        return self._fc2_early_windows

    @property
    def fc2_early_window(self) -> tuple[int, int] | None:
        """This pipe's FC2 early-start window (lo, hi] above the decode window, or None."""
        return self._fc2_early_window

    @property
    def prefix_form(self) -> int:
        """Receiver prefix form of the single-launch dispatches (0 or 1; see ``PREFIX_FORM_ENV``)."""
        return self._prefix_form

    @property
    def direct_place(self) -> bool:
        """True when grid-dispatch rounds may place carrier rows directly (every rank opted in)."""
        return self._direct

    @property
    def direct_place_min_tokens(self) -> int:
        """Smallest round (local tokens) that takes the direct-placement grid dispatch."""
        return self._direct_min

    def _direct_flag(self) -> int:
        """Debug: 1 when the last round placed its rows directly on this rank, else 0 (syncs)."""
        if not self._last_round_direct or self._grid_a1base is None:
            return 0
        return int(self._grid_a1base[-1].item())

    @property
    def fused_begin(self) -> int:
        """Round-opening fusion level of small rounds (0, 1 or 2; see ``FUSED_BEGIN_ENV``)."""
        return self._fused_begin

    def _use_fused_dispatch(self, num_tokens: int) -> bool:
        pipe = self.pipe
        return (
            self._fused_dispatch
            and pipe.config.dedup_dispatch
            and num_tokens <= self._fused_dispatch_max_tokens
            and num_tokens * pipe.K <= FUSED_DISPATCH_MAX_ROUTES_LIMIT
        )

    def _use_grid_dispatch(self, num_tokens: int) -> bool:
        return 0 < self._grid_dispatch_min_tokens <= num_tokens

    # ------------------------------------------------------------------ setup
    def _check_weights(self, weights: Sm90CakeBf16Weights) -> None:
        pipe = self.pipe
        if not isinstance(weights, Sm90CakeBf16Weights):
            raise ValueError(
                f"weights must be Sm90CakeBf16Weights (make_sm90_cake_bf16_weights), got {type(weights).__name__}"
            )
        if weights.num_local_experts != pipe.E or weights.hidden_size != pipe.H:
            raise ValueError(
                f"weights describe E={weights.num_local_experts}, H={weights.hidden_size}; "
                f"the pipe has E={pipe.E}, H={pipe.H}"
            )
        if weights.device != pipe.device:
            raise ValueError(f"weights must be on {pipe.device}, got {weights.device}")

    def _init_buffers(self) -> None:
        pipe = self.pipe
        H = pipe.H
        self.m_buf = output_row_capacity(pipe.m_cap)
        dv = pipe.device
        # direct placement: a1 is the symmetric buffer of _init_direct_a1 (or, when a peer
        # cannot run it, a regular tensor allocated there)
        self.a1: torch.Tensor | None = (
            None
            if self._direct_local
            else torch.empty(self.m_buf, H, dtype=torch.bfloat16, device=dv)
        )
        self.meta = torch.empty(self.m_buf, 4, dtype=torch.int32, device=dv)
        self.row_expert = torch.empty(self.m_buf, dtype=torch.int32, device=dv)
        self.h2 = torch.empty(self.m_buf, self.I, dtype=torch.bfloat16, device=dv)
        self.y = torch.empty(self.m_buf, H, dtype=torch.bfloat16, device=dv)
        # fused tail: grid-completion counter (the last block resets it to zero)
        self._tail_blocks_done = torch.zeros(1, dtype=torch.int32, device=dv)
        # fused dispatch: per-destination meta / payload bases shared across blocks
        self._dispatch_bases = torch.zeros(2 * pipe.ep, dtype=torch.int32, device=dv)
        if self._grid_dispatch_min_tokens > 0:
            # grid dispatch: dense carrier-route list, block-local item index, item
            # counter (zero here; the kernel resets it at the end of every round)
            routes = pipe.token_capacity * pipe.K
            self._grid_items = torch.empty(routes, dtype=torch.int32, device=dv)
            self._grid_iloc = torch.empty(routes, dtype=torch.int32, device=dv)
            self._grid_item_ctr = torch.zeros(1, dtype=torch.int32, device=dv)
        if self._combine_wire in _PREREDUCED_WIRES:
            # (owner rank, token) group worklist of the pre-reduced publish
            nslots = pipe.ep * pipe.token_capacity
            self._grp_cnt = torch.zeros(nslots, dtype=torch.int32, device=dv)
            self._grp_rows = torch.zeros(nslots * pipe.K, dtype=torch.int32, device=dv)
            self._grp_list = torch.zeros(nslots, dtype=torch.int32, device=dv)
            self._n_groups = torch.zeros(1, dtype=torch.int32, device=dv)
            self._groups_per_src = torch.zeros(pipe.ep, dtype=torch.int32, device=dv)
            # grid-completion counter of the pre-reduced publish (its last block resets
            # the worklist scratch for the next round: no per-round memsets)
            self._pub_blocks_done = torch.zeros(1, dtype=torch.int32, device=dv)
            pipe._cdone_local.zero_()  # the publish kernel keeps it zeroed from here on
        else:  # per-route wire: compact builds no worklist; placeholders for the binding
            if self._own_per_route:
                # grid-completion counter of the per-route twin publish, which keeps
                # the per-source row counters zeroed from here on (no per-round memset)
                self._pub_blocks_done = torch.zeros(1, dtype=torch.int32, device=dv)
                pipe._cdone_local.zero_()
            self._grp_cnt = self._grp_rows = self._grp_list = torch.zeros(
                1, dtype=torch.int32, device=dv
            )
            self._n_groups = self._groups_per_src = torch.zeros(
                1, dtype=torch.int32, device=dv
            )

    def _init_direct_a1(self) -> None:
        """Guarded phase after the direct-placement handshake (every rank runs it)."""
        pipe = self.pipe
        if not self._direct:
            if self.a1 is None:
                self.a1 = torch.empty(
                    self.m_buf, pipe.H, dtype=torch.bfloat16, device=pipe.device
                )
            return None
        from ......comm import pack_strided_memory
        from ......comm.mnnvl import SymmDeviceMemory

        # [a1 rows | one flag per source | one per-key count row per source]
        nkeys = pipe.E * pipe.ep
        a1_bytes = self.m_buf * pipe.H * 2
        self._ag_offset = _align256(a1_bytes)
        ag_bytes = _align256(8 * pipe.ep) + pipe.ep * ((nkeys + 3) // 4 * 4) * 4
        symm = SymmDeviceMemory(
            buf_size=self._ag_offset + ag_bytes,
            group_size=pipe.ep,
            group_rank=pipe.rank,
            device_idx=pipe.device_index,
            comm_backend_for_handle_transfer=pipe._comm,
            enable_multicast=False,
            allocate_signal_pads=False,
        )
        self._a1_symm = symm
        self._a1_peers = symm.get_buffer_ptrs_dev()
        base = symm.get_unicast_ptr(pipe.rank)

        def view(offset: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
            t = pack_strided_memory(
                base + offset, nbytes, nbytes, 1, dtype, pipe.device_index
            )
            wrapper = getattr(t, "_capsule_wrapper", None)
            if wrapper is not None:  # see protocol._CAPSULE_KEEPALIVE
                _CAPSULE_KEEPALIVE.append(wrapper)
            t = t.reshape(-1)
            return (
                t if t.dtype == dtype else t.view(dtype)
            )  # dlpack labels bf16 as fp16

        self.a1 = view(0, a1_bytes, torch.bfloat16).reshape(self.m_buf, pipe.H)
        self._ag_view = view(self._ag_offset, ag_bytes, torch.uint8)
        self._ag_view.zero_()  # tag 0 never matches a round (the first round's tag is 1)
        self._grid_a1base = torch.zeros(
            nkeys + 1, dtype=torch.int32, device=pipe.device
        )
        torch.cuda.synchronize(pipe.device)
        return None

    def _release_direct_views(self) -> None:
        """Drop the dlpack views of the symmetric a1 (destroy / interpreter exit)."""
        if self._a1_symm is not None:
            self.a1 = None
            self._ag_view = None

    def bind_weights(self, weights: Sm90CakeBf16Weights) -> None:
        """Swap the expert weights between rounds (same geometry, same device)."""
        self._require_usable()
        if self._state == _RunnerState.STAGED:
            raise RuntimeError("cannot rebind weights while a round is staged")
        self._check_weights(weights)
        if weights.intermediate_size != self.I:
            raise ValueError(
                f"weights intermediate_size {weights.intermediate_size} does not match the runner's {self.I}"
            )
        self.weights = weights

    # ------------------------------------------------------------- lifecycle
    @property
    def state(self) -> str:
        return self._state.value

    def _require_usable(self) -> None:
        if self._state == _RunnerState.DESTROYED:
            raise RuntimeError("sm90_cake_bf16 runner has been destroyed")
        if self._state == _RunnerState.POISONED:
            raise RuntimeError(
                "sm90_cake_bf16 runner is poisoned by an earlier mid-round failure; "
                "destroy it and build a new layer"
            )

    def _poison(self) -> None:
        if self._state in (_RunnerState.POISONED, _RunnerState.DESTROYED):
            return
        self._state = _RunnerState.POISONED
        self._staged_direct = False
        self._staged_tokens = None
        self._staged_topk_ids = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        with contextlib.suppress(Exception):
            self.pipe.proto_abort()

    def abort(self) -> None:
        """Poison the active pipe without entering a cross-rank barrier."""
        self._poison()

    def destroy(self) -> None:
        if self._state == _RunnerState.DESTROYED:
            return
        self._state = _RunnerState.DESTROYED
        gemm = getattr(self, "gemm", None)
        if gemm is not None:
            with contextlib.suppress(Exception):
                gemm.destroy()
        if self._a1_symm is not None:  # quiesce, then drop the views before the buffer
            torch.cuda.synchronize(self.pipe.device)
            self._release_direct_views()
            self._a1_symm = None
        self.pipe.destroy()

    # ---------------------------------------------------------------- rounds
    def _validate_round_inputs(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> int:
        pipe = self.pipe
        if x.ndim != 2 or x.shape[1] != pipe.H:
            raise ValueError(f"x must be [num_tokens, {pipe.H}], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise ValueError(
                f"x must be bf16 (native BF16 dispatch payload), got {x.dtype}"
            )
        num_tokens = int(x.shape[0])
        if num_tokens > pipe.token_capacity:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds token_capacity {pipe.token_capacity}"
            )
        if tuple(topk_ids.shape) != (num_tokens, pipe.K) or tuple(
            topk_weights.shape
        ) != (num_tokens, pipe.K):
            raise ValueError(
                f"topk_ids/topk_weights must be [{num_tokens}, {pipe.K}], got "
                f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError(f"topk_ids must be int32, got {topk_ids.dtype}")
        if topk_weights.dtype != torch.float32:
            raise ValueError(f"topk_weights must be float32, got {topk_weights.dtype}")
        for name, t in (
            ("x", x),
            ("topk_ids", topk_ids),
            ("topk_weights", topk_weights),
        ):
            if t.device != pipe.device:
                raise ValueError(f"{name} must be on {pipe.device}, got {t.device}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        return num_tokens

    def _validate_output(self, output: torch.Tensor, num_tokens: int) -> None:
        pipe = self.pipe
        if tuple(output.shape) != (num_tokens, pipe.H):
            raise ValueError(
                f"output must be [{num_tokens}, {pipe.H}], got {tuple(output.shape)}"
            )
        if output.dtype != pipe.out_dtype:
            raise ValueError(
                f"output dtype must be {pipe.out_dtype}, got {output.dtype}"
            )
        if output.device != pipe.device or not output.is_contiguous():
            raise ValueError("output must be a contiguous tensor on the pipe's device")

    @staticmethod
    def _current_stream() -> tuple[torch.cuda.Stream, bool]:
        stream = torch.cuda.current_stream()
        return stream, torch.cuda.is_current_stream_capturing()

    def _begin_round(self, *, fused: bool = False) -> None:
        """Open the round: round-tag bump and ack wait, in the pipe's order.

        With the fused tail this mirrors ``Sm90PushPipe.proto_begin_round``
        minus the combine-inbox fill: the senders rewrite every unmasked slot of
        a token below ``num_tokens`` each round and the fused tail skips masked
        slots instead of reading zeroed ones, so the fill has no observable
        effect on the output.
        """
        pipe = self.pipe
        if not self._fused_tail:
            pipe.proto_begin_round()
            return
        if pipe._destroyed:
            raise RuntimeError("sm90_push pipe has been destroyed")
        if pipe._poisoned:
            raise RuntimeError(
                "sm90_push pipe is poisoned by an earlier mid-round failure"
            )
        if pipe._round_open:
            raise RuntimeError(
                "sm90_push pipe: previous round was never acked; "
                "the pipe runs ONE round at a time"
            )
        pipe._round_open = True
        if fused:  # the fused dispatch bumps the tag and waits for the acks itself
            return
        pipe.module.sm90_push_bump_tag(pipe._round)
        pipe.module.sm90_push_wait_acks(*pipe._layout_args(), pipe._round)

    def stage_inputs(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> None:
        """Validate one round, then dispatch its bf16 payload and routing tensors."""
        self._require_usable()
        if self._state == _RunnerState.STAGED:
            raise RuntimeError(
                "stage_inputs called twice for one round; call compute first"
            )
        num_tokens = self._validate_round_inputs(x, topk_ids, topk_weights)
        stream, capturing = self._current_stream()
        stream_id = int(stream.cuda_stream)
        if (
            not capturing
            and self._round_event is not None
            and self._round_stream_id != stream_id
            and not self._round_event.query()
        ):
            # The previous round was issued on another stream and may still be
            # running: order this round after it on the device.  The pipe runs
            # one round at a time; this edge enforces it without requiring the
            # host to have synchronised (a warm-up loop that hops streams, as the
            # graph-capture recipes do, would otherwise fail spuriously).  Inside
            # a capture the caller must have joined the streams beforehand.
            stream.wait_event(self._round_event)
        nv = self.record_stages
        use_grid = self._use_grid_dispatch(num_tokens)
        use_fused = not use_grid and self._use_fused_dispatch(num_tokens)
        direct = use_grid and self._direct and num_tokens >= self._direct_min
        level = (
            self._grid_begin if use_grid else (self._fused_begin if use_fused else 0)
        )
        try:
            with _record_stage("begin_round", nv):
                self._begin_round(fused=level > 0)
            with _record_stage("dispatch", nv):
                if use_grid:
                    pipe = self.pipe
                    grid_args = (
                        x,
                        topk_ids,
                        topk_weights,
                        *pipe._layout_args(),
                        pipe._round,
                        pipe._lc,
                        pipe._loff,
                        pipe._ploff,
                        pipe._keybase,
                        self._dispatch_bases,
                        self._grid_items,
                        self._grid_iloc,
                        self._grid_item_ctr,
                        1 if level >= 1 else 0,
                        1 if level == 2 else 0,
                        pipe._rows_per_src,
                        pipe._offsets,
                        pipe._seg_src_base,
                        pipe._seg_out_base,
                        pipe._pad_base,
                        pipe._m_dev,
                        pipe._p_dev,
                        pipe._next_row,
                        pipe.m_cap,
                        self._prefix_form,
                    )
                    if direct:
                        self.dispatch_grid_module.sm90_cake_dispatch_grid_direct_bf16(
                            *grid_args,
                            self._grid_a1base,
                            self._a1_peers,
                            self._ag_offset,
                        )
                    else:
                        self.dispatch_grid_module.sm90_cake_dispatch_grid_bf16(
                            *grid_args
                        )
                elif use_fused:
                    pipe = self.pipe
                    self.dispatch_module.sm90_cake_dispatch_fused_bf16(
                        x,
                        topk_ids,
                        topk_weights,
                        *pipe._layout_args(),
                        pipe._round,
                        self._dispatch_bases,
                        1 if level >= 1 else 0,
                        1 if level == 2 else 0,
                        pipe._rows_per_src,
                        pipe._offsets,
                        pipe._seg_src_base,
                        pipe._seg_out_base,
                        pipe._pad_base,
                        pipe._m_dev,
                        pipe._p_dev,
                        pipe._next_row,
                        pipe.m_cap,
                        self._prefix_form,
                    )
                else:
                    self.pipe.proto_dispatch(x, topk_ids, topk_weights)
        except Exception:
            self._poison()
            raise
        self._staged_tokens = num_tokens
        self._staged_direct = direct
        self._last_round_direct = direct
        self._staged_prefix_done = level == 2
        self._staged_topk_ids = topk_ids  # the fused tail skips masked (-1) slots
        self._staged_stream = stream
        self._staged_stream_capturing = capturing
        self._state = _RunnerState.STAGED

    def compute(self, *, output: torch.Tensor) -> torch.Tensor:
        """Finish the staged round and reduce directly into ``output``."""
        self._require_usable()
        if self._state != _RunnerState.STAGED:
            raise RuntimeError("compute requires a preceding stage_inputs call")
        pipe = self.pipe
        weights = self.weights
        num_tokens = self._staged_tokens
        topk_ids = self._staged_topk_ids
        staged_stream = self._staged_stream
        staged_capturing = self._staged_stream_capturing
        assert (
            num_tokens is not None
            and topk_ids is not None
            and staged_stream is not None
            and weights is not None
        )
        try:
            self._validate_output(output, num_tokens)
        except Exception:
            self._poison()
            raise
        staged_stream_id = int(staged_stream.cuda_stream)
        caller_stream, _caller_capturing = self._current_stream()
        streams_differ = int(caller_stream.cuda_stream) != staged_stream_id
        if streams_differ:
            if self._caller_ready_event is None:
                self._caller_ready_event = torch.cuda.Event()
            self._caller_ready_event.record(caller_stream)
            staged_stream.wait_event(self._caller_ready_event)
        stream_context = (
            contextlib.nullcontext()
            if not streams_differ
            else torch.cuda.stream(staged_stream)
        )
        nv = self.record_stages
        try:
            with stream_context:
                if not self._staged_prefix_done:  # else run by the fused dispatch
                    with _record_stage("wait_prefix", nv):
                        pipe.proto_wait_prefix()
                with _record_stage("compact_bf16", nv):
                    build_groups = 1 if self._combine_wire in _PREREDUCED_WIRES else 0
                    if self._staged_direct:  # the carriers are in a1 already
                        self.compact_module.sm90_cake_compact_direct_bf16(
                            self.a1,
                            self.meta,
                            self.row_expert,
                            *pipe._layout_args(),
                            pipe._seg_src_base,
                            pipe._seg_out_base,
                            pipe._m_dev,
                            self._grp_cnt,
                            self._grp_rows,
                            self._grp_list,
                            self._n_groups,
                            self._groups_per_src,
                            pipe._round,
                            build_groups,
                            self._pdl_flag(PDL_EDGE_COMPACT),
                        )
                    else:
                        self.compact_module.sm90_cake_compact_bf16(
                            self.a1,
                            self.meta,
                            self.row_expert,
                            *pipe._layout_args(),
                            pipe._seg_src_base,
                            pipe._seg_out_base,
                            pipe._m_dev,
                            pipe._next_row,
                            self._grp_cnt,
                            self._grp_rows,
                            self._grp_list,
                            self._n_groups,
                            self._groups_per_src,
                            pipe._round,
                            build_groups,
                            self._pdl_flag(PDL_EDGE_COMPACT),
                            self._compact_form,
                        )
                # FC2 early start in the decode window or the shape's window above
                # it (rank-local: no cross-rank protocol)
                sched = (
                    self._gemm_sched
                    if fc2_early_round(
                        num_tokens, self._fc2_early_max_tokens, self._fc2_early_window
                    )
                    else None
                )
                with _record_stage("fc1", nv):
                    if sched is not None:
                        self.gemm.fc1(
                            self.a1,
                            weights.w13,
                            pipe._offsets,
                            self.h2,
                            pdl=bool(self._pdl & PDL_EDGE_FC1),
                            sched=sched,
                        )
                    else:
                        self.gemm.fc1(
                            self.a1,
                            weights.w13,
                            pipe._offsets,
                            self.h2,
                            pdl=bool(self._pdl & PDL_EDGE_FC1),
                        )
                with _record_stage("fc2", nv):
                    if sched is not None:
                        self.gemm.fc2(
                            self.h2,
                            weights.w2,
                            pipe._offsets,
                            self.y,
                            pdl=bool(self._pdl & PDL_EDGE_FC2),
                            sched=sched,
                        )
                    else:
                        self.gemm.fc2(
                            self.h2,
                            weights.w2,
                            pipe._offsets,
                            self.y,
                            pdl=bool(self._pdl & PDL_EDGE_FC2),
                        )
                if self._combine_wire in _PREREDUCED_WIRES:
                    split = (
                        1 if self._combine_wire == COMBINE_WIRE_PREREDUCED_HILO else 0
                    )
                    with _record_stage("combine", nv):
                        self.publish_module.sm90_cake_combine_prereduced_bf16(
                            self.y,
                            self.meta,
                            *pipe._layout_args(),
                            pipe._m_dev,
                            self._grp_cnt,
                            self._grp_rows,
                            self._grp_list,
                            self._n_groups,
                            self._groups_per_src,
                            pipe._cdone_local,
                            pipe._round,
                            self._pub_blocks_done,
                            split,
                            self._pdl_flag(PDL_EDGE_PUBLISH),
                            self._publish_form,
                        )
                    with _record_stage("combine_tail", nv):
                        self.tail_module.sm90_cake_combine_tail_prereduced_bf16(
                            output,
                            topk_ids,
                            *pipe._layout_args(),
                            pipe._round,
                            pipe._lc,
                            pipe._done,
                            self._tail_blocks_done,
                            num_tokens,
                            split,
                            self._pdl_flag(PDL_EDGE_TAIL),
                        )
                    pipe._round_open = False  # the fused tail performed the ack
                elif self._fused_tail:
                    with _record_stage("combine", nv):
                        if self._own_per_route:
                            self.publish_module.sm90_cake_combine_per_route_bf16(
                                self.y,
                                self.meta,
                                *pipe._layout_args(),
                                pipe._m_dev,
                                pipe._rows_per_src,
                                pipe._cdone_local,
                                pipe._round,
                                self._pub_blocks_done,
                                self._pdl_flag(PDL_EDGE_PUBLISH),
                            )
                        else:
                            pipe.proto_combine(self.y, self.meta)
                    with _record_stage("combine_tail", nv):
                        self.tail_module.sm90_cake_combine_tail_bf16(
                            output,
                            topk_ids,
                            *pipe._layout_args(),
                            pipe._round,
                            pipe._lc,
                            pipe._done,
                            self._tail_blocks_done,
                            num_tokens,
                            self._pdl_flag(PDL_EDGE_TAIL),
                        )
                    pipe._round_open = False  # the fused tail performed the ack
                else:
                    with _record_stage("combine", nv):
                        pipe.proto_combine(self.y, self.meta)
                    with _record_stage("wait_combine", nv):
                        pipe.proto_wait_combine()
                    with _record_stage("reduce", nv):
                        pipe.proto_reduce(output, num_tokens)
                    with _record_stage("ack", nv):
                        pipe.proto_ack()
        except Exception:
            self._poison()
            raise
        self._state = _RunnerState.IDLE
        self._staged_tokens = None
        self._staged_direct = False
        self._staged_prefix_done = False
        self._staged_topk_ids = None
        self._staged_stream = None
        self._staged_stream_capturing = False
        if not staged_capturing or streams_differ:
            if self._round_event is None:
                self._round_event = torch.cuda.Event()
            self._round_event.record(staged_stream)
            self._round_stream_id = staged_stream_id
            if streams_differ:
                caller_stream.wait_event(self._round_event)
        return output

    def forward(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Submit a complete round."""
        num_tokens = self._validate_round_inputs(x, topk_ids, topk_weights)
        self._validate_output(output, num_tokens)
        self.stage_inputs(x, topk_ids, topk_weights)
        return self.compute(output=output)
