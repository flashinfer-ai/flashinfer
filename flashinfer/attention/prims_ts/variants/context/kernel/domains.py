# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Work domains of the context kernel: dense, causal and variable-window tasks.

``select_fmha_domain_policy`` picks which domain task each warp role runs for a
configuration; the domain tasks bound every task's K/V loop to the visible
tiles of its work tile.
"""

import cutlass.cute as cute
import cutlass
from collections.abc import Callable
from cutlass import Int32
from cutlass.experimental.task_scheduling.task import Task
from dataclasses import dataclass
from .config import FmhaConfig
from .device.windows import (
    bottom_right_window_max_tiles,
    bottom_right_window_tile_start,
    variable_window_cta_min_start,
)
from typing import Any


DomainPolicyValue = int | bool | Int32 | type[Task] | cute.Tensor | cute.Pointer
DomainKwargs = dict[str, DomainPolicyValue]


class CausalDomainTask(Task):
    """Task with causal and optional left-window masking domain.

    For causal FMHA, each Q tile only attends to K tiles where k <= q.
    The domain (K-loop iteration count) varies per tile based on the
    tile's sequence position.  Head-paired sliding-window schedules reuse
    the same domain calculation after subtracting skipped left-window KV
    tiles.
    """

    def __init__(
        self,
        num_kv_tiles: int | Int32,
        tile_size_q: int | Int32,
        tile_size_kv: int | Int32,
        q_offset: int | Int32 = 0,
        seq_idx: int = 0,
        batch_idx: int | None = None,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        seq_lens_kv: cute.Pointer | None = None,
        runtime_kv_tile_multiple: int = 1,
        reverse_seq_tiles: int | Int32 | None = None,
        offset: int = 0,
        window_size_left: int = 0,
        packed_window: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize causal-domain parameters.

        Args:
            num_kv_tiles: Total number of K/V tiles in the sequence.
            tile_size_q: Number of Q rows covered by one CTA tile.
            tile_size_kv: Number of K/V rows covered by one K-loop iteration.
            q_offset: Causal row-index shift for S_q < S_kv.
            seq_idx: Index of the sequence coordinate in ``tile_coord``.
            batch_idx: Index of the request coordinate in ``tile_coord`` for
                packed causal launches.
            cum_seqlen_q: Per-run packed-Q cumulative offsets.
            cum_seqlen_k: Per-run packed-K/V cumulative offsets for contiguous K/V.
            seq_lens_kv: Per-run K/V lengths for paged K/V.
            runtime_kv_tile_multiple: Round a request-local K/V tile count up
                to this multiple. Query-paired zero-offset causal scheduling
                uses two to retain its synthetic peer-0 tail slot.
            reverse_seq_tiles: Number of Q sequence work tiles when causal
                balancing reverses launch order. ``None`` keeps natural order.
            offset: Loop-count decrement after the causal/window tile count is
                computed. Use 0 for N, 1 for N-1, and 2 for N-2 domains.
            window_size_left: Left sliding-window size in tokens. Zero disables
                left-window domain trimming.
            packed_window: Use an offset-independent maximum window span for a
                packed-ragged batch; runtime loads still use each request's offset.
            **kwargs: Remaining ``Task`` constructor arguments, including an
                optional captured ``schedule``.
        """
        init_causal_domain_state(
            self,
            num_kv_tiles=num_kv_tiles,
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            seq_lens_kv=seq_lens_kv,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            reverse_seq_tiles=reverse_seq_tiles,
            offset=offset,
            window_size_left=window_size_left,
            packed_window=packed_window,
            kwargs=kwargs,
        )

    def get_domain(self, tile_coord: cute.Coord) -> int | Int32:
        """Return the per-work-tile K-loop iteration count.

        Folds to a Python ``int`` when the tile coordinate and shape fields are
        static, and defers to ``cute.math`` for runtime coordinates.
        """
        seq_coord = tile_coord[self.seq_idx]
        if cutlass.const_expr(self._reverse_seq_tiles is not None):
            seq_coord = self._reverse_seq_tiles - seq_coord - 1
        q_offset = self._q_offset
        num_kv_tiles = self.num_kv_tiles
        runtime_q_tile_active = None
        if cutlass.const_expr(self._cum_seqlen_q is not None):
            assert self.batch_idx is not None
            batch_coord = Int32(tile_coord[self.batch_idx])
            q_begin = Int32(self._cum_seqlen_q[batch_coord])
            seqlen_q = Int32(self._cum_seqlen_q[batch_coord + Int32(1)]) - q_begin
            if cutlass.const_expr(self.seq_lens_kv is not None):
                seqlen_k = Int32(self.seq_lens_kv[batch_coord])
            else:
                assert self.cum_seqlen_k is not None
                k_begin = Int32(self.cum_seqlen_k[batch_coord])
                seqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - k_begin
            runtime_q_tile_active = Int32(seq_coord * self.tile_size_q < seqlen_q)
            q_offset = seqlen_k - seqlen_q
            num_kv_tiles = cute.ceil_div(seqlen_k, self.tile_size_kv)
            if cutlass.const_expr(self._runtime_kv_tile_multiple > 1):
                num_kv_tiles = (
                    cute.ceil_div(num_kv_tiles, self._runtime_kv_tile_multiple)
                    * self._runtime_kv_tile_multiple
                )
        if self._packed_window:
            max_window_tiles = bottom_right_window_max_tiles(
                q_tile_m=self.tile_size_q,
                kv_tile_n=self.tile_size_kv,
                window_size_left=self._window_size_left,
            )
            result = domain_min(num_kv_tiles, max_window_tiles)
        else:
            max_q_row = q_offset + seq_coord * self.tile_size_q + self.tile_size_q - 1
            causal_n = max_q_row // self.tile_size_kv + 1
            # Softmax assumes there is at least one Q-tile-width of K work.
            result = domain_max(
                self.tile_size_q // self.tile_size_kv,
                domain_min(num_kv_tiles, causal_n),
            )
        if self._window_size_left > 0 and not self._packed_window:
            result -= bottom_right_window_tile_start(
                seq_coord=seq_coord,
                q_tile_m=self.tile_size_q,
                kv_tile_n=self.tile_size_kv,
                q_offset=q_offset,
                window_size_left=self._window_size_left,
            )
        if self.offset:
            result = result - self.offset
        if cutlass.const_expr(runtime_q_tile_active is not None):
            # The outer grid is sized by the planned maximum Q length, so a
            # mixed packed request can have trailing work-tile slots with no Q
            # rows. Retain the smallest legal N/N-1/N-2 pipeline domains for
            # those slots instead of traversing K/V according to a large
            # bottom-right offset. Resource-level ragged extents suppress the
            # dummy Q/O traffic; the minimum domains preserve task handoffs.
            minimum_domain = max(self.tile_size_q // self.tile_size_kv - self.offset, 0)
            result = (
                runtime_q_tile_active * result
                + (Int32(1) - runtime_q_tile_active) * minimum_domain
            )
        return result


class CausalSoftmaxDomainTask(CausalDomainTask):
    """Softmax task with causal domain selection."""

    def __init__(
        self,
        num_kv_tiles: int | Int32,
        tile_size_q: int | Int32,
        tile_size_kv: int | Int32,
        q_offset: int | Int32 = 0,
        seq_idx: int = 0,
        batch_idx: int | None = None,
        cum_seqlen_q: cute.Tensor | None = None,
        cum_seqlen_k: cute.Tensor | None = None,
        seq_lens_kv: cute.Pointer | None = None,
        runtime_kv_tile_multiple: int = 1,
        reverse_seq_tiles: int | Int32 | None = None,
        offset: int = 0,
        window_size_left: int = 0,
        packed_window: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize softmax causal-domain parameters.

        Parameters
        ----------
        num_kv_tiles : int or Int32
            Total number of K/V tiles in the sequence.
        tile_size_q : int or Int32
            Number of Q rows covered by one CTA tile.
        tile_size_kv : int or Int32
            Number of K/V rows covered by one K-loop iteration.
        q_offset : int or Int32
            Causal row-index shift for S_q < S_kv.
        seq_idx : int
            Index of the sequence coordinate in ``tile_coord``.
        batch_idx : int or None
            Index of the request coordinate for packed causal launches.
        cum_seqlen_q, cum_seqlen_k : cute.Tensor or None
            Per-run cumulative offsets used to derive request-local causal
            shifts for contiguous packed input.
        seq_lens_kv : cute.Pointer or None
            Per-run K/V lengths used for paged input.
        runtime_kv_tile_multiple : int
            Request-local K/V tile-count alignment for paired-tail scheduling.
        reverse_seq_tiles : int or Int32 or None
            Number of Q sequence work tiles for reversed causal-balanced order.
        offset : int
            Loop-count decrement after the causal/window tile count is
            computed.
        window_size_left : int
            Left sliding-window size in tokens. Zero disables left-window
            domain trimming.
        packed_window : bool
            Whether to use an offset-independent packed-ragged window span.
        **kwargs : Any
            Remaining ``Task`` constructor arguments.
        """
        init_causal_domain_state(
            self,
            num_kv_tiles=num_kv_tiles,
            tile_size_q=tile_size_q,
            tile_size_kv=tile_size_kv,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            seq_lens_kv=seq_lens_kv,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            reverse_seq_tiles=reverse_seq_tiles,
            offset=offset,
            window_size_left=window_size_left,
            packed_window=packed_window,
            kwargs=kwargs,
        )


@dataclass(frozen=True)
class FmhaDomainPolicy:
    """Domain and mask policy selected for one FMHA launch flavor."""

    domain_n_kwargs: DomainKwargs
    domain_n_minus_1_kwargs: DomainKwargs
    softmax0_domain_kwargs: DomainKwargs
    softmax1_domain_kwargs: DomainKwargs


class PackedDenseDomainTask(Task):
    """Request-local dense KV extent with one masked tile for empty rows."""

    def __init__(
        self,
        num_kv_tiles: int | Int32,
        tile_size_kv: int,
        batch_idx: int,
        cum_seqlen_k: cute.Tensor,
        offset: int = 0,
        **kwargs: Any,
    ) -> None:
        """Retain a static validation bound when available and clip runtime work."""
        static_domain = (
            max(num_kv_tiles - offset, 0) if isinstance(num_kv_tiles, int) else None
        )
        init_task_with_domain(self, kwargs, static_domain)
        self.num_kv_tiles = num_kv_tiles
        self.tile_size_kv = tile_size_kv
        self.batch_idx = batch_idx
        self.cum_seqlen_k = cum_seqlen_k
        self.offset = offset

    def get_domain(self, tile_coord: cute.Coord) -> int | Int32:
        """Use the same live KV bound for each task's N or N-1 domain."""
        batch_coord = Int32(tile_coord[self.batch_idx])
        k_begin = Int32(self.cum_seqlen_k[batch_coord])
        seqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)]) - k_begin
        runtime_tiles = cute.ceil_div(seqlen_k, self.tile_size_kv)
        # Keep the paired N/N-1 pipeline domain valid even for an empty request.
        domain = domain_min(self.num_kv_tiles, domain_max(1, runtime_tiles))
        return domain - self.offset


class VariableWindowDomainTask(Task):
    """Task whose K-loop domain comes from packed-Q window endpoints."""

    def __init__(
        self,
        variable_window_token_starts: cute.Tensor,
        variable_window_token_ends: cute.Tensor,
        variable_window_cta_starts: cute.Tensor,
        num_kv_tiles: int | Int32,
        q_stride: int | Int32,
        tile_size_q: int,
        tile_size_kv: int,
        seq_idx: int,
        batch_idx: int,
        offset: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a variable-window task domain.

        Args:
            variable_window_token_starts: Flattened inclusive K-token start for
                every Q row, laid out with ``q_stride`` rows per batch.
            variable_window_token_ends: Flattened inclusive K-token end for
                every Q row, using the same layout as the start tensor.
            variable_window_cta_starts: Precomputed minimum K-token start for
                each Q CTA, used as the common K/V load origin.
            num_kv_tiles: Planned maximum number of K/V tiles.
            q_stride: Number of Q rows per batch in the flattened bound tensors.
            tile_size_q: Number of Q rows covered by one CTA.
            tile_size_kv: Number of K/V tokens covered by one loop tile.
            seq_idx: Index of the Q-sequence coordinate in ``tile_coord``.
            batch_idx: Index of the batch coordinate in ``tile_coord``.
            offset: Domain-count decrement. Zero selects N; one selects N-1.
            **kwargs: Remaining ``Task`` arguments, including the required
                captured ``schedule``.
        """
        if kwargs.get("schedule") is None:
            raise ValueError("VariableWindow domain tasks require a captured schedule")
        super().__init__(**kwargs)
        self.variable_window_token_starts = variable_window_token_starts
        self.variable_window_token_ends = variable_window_token_ends
        self.variable_window_cta_starts = variable_window_cta_starts
        self.num_kv_tiles = num_kv_tiles
        self.q_stride = q_stride
        self.tile_size_q = tile_size_q
        self.tile_size_kv = tile_size_kv
        self.seq_idx = seq_idx
        self.batch_idx = batch_idx
        self.offset = offset

    def get_domain(self, tile_coord: cute.Coord) -> Int32:
        """Return the number of K tiles intersecting this Q CTA's bounds."""
        seq_coord = Int32(tile_coord[self.seq_idx])
        batch_coord = Int32(tile_coord[self.batch_idx])
        first_local_q = seq_coord * self.tile_size_q
        last_local_q = cute.math.min(
            first_local_q + self.tile_size_q - Int32(1),
            self.q_stride - Int32(1),
        )
        packed_q_base = batch_coord * self.q_stride
        first_k = variable_window_cta_min_start(
            self.variable_window_cta_starts,
            batch_coord=batch_coord,
            seq_coord=seq_coord,
            q_stride=self.q_stride,
            tile_size_q=self.tile_size_q,
        )
        last_k = Int32(self.variable_window_token_ends[packed_q_base + last_local_q])
        first_k_tile = first_k // self.tile_size_kv
        last_k_tile = (last_k + self.tile_size_kv) // self.tile_size_kv
        return last_k_tile - first_k_tile - self.offset


def as_i32(x: int | Int32) -> Int32:
    return Int32(x) if isinstance(x, int) else x


def causal_domain_kwargs(
    *,
    num_kv_tiles: int | Int32,
    tile_size_q: int,
    tile_size_kv: int,
    q_offset: int | Int32,
    seq_idx: int,
    batch_idx: int | None = None,
    cum_seqlen_q: cute.Tensor | None = None,
    cum_seqlen_k: cute.Tensor | None = None,
    seq_lens_kv: cute.Pointer | None = None,
    runtime_kv_tile_multiple: int = 1,
    offset: int,
    reverse_seq_tiles: int | Int32 | None = None,
    window_size_left: int | None = None,
    packed_window: bool = False,
) -> DomainKwargs:
    """Build kwargs for a causal/window-aware task domain.

    ``offset`` is the loop-domain decrement applied after the causal/window
    tile count is computed. The schedule uses it to derive N, N-1, and N-2
    loop domains from the same causal formula without changing the Q/K tile
    coordinates or the S_q < S_kv ``q_offset`` mask shift.
    """
    result: DomainKwargs = {
        "task_class": CausalDomainTask,
        "num_kv_tiles": num_kv_tiles,
        "tile_size_q": tile_size_q,
        "tile_size_kv": tile_size_kv,
        "q_offset": q_offset,
        "seq_idx": seq_idx,
        "offset": offset,
    }
    if reverse_seq_tiles is not None:
        result["reverse_seq_tiles"] = reverse_seq_tiles
    if cum_seqlen_q is not None:
        if batch_idx is None or (cum_seqlen_k is None and seq_lens_kv is None):
            raise ValueError(
                "runtime causal domains require batch_idx, cumulative Q offsets, "
                "and cumulative K offsets or paged K/V lengths"
            )
        result["batch_idx"] = batch_idx
        result["cum_seqlen_q"] = cum_seqlen_q
        if cum_seqlen_k is not None:
            result["cum_seqlen_k"] = cum_seqlen_k
        if seq_lens_kv is not None:
            result["seq_lens_kv"] = seq_lens_kv
        if runtime_kv_tile_multiple > 1:
            result["runtime_kv_tile_multiple"] = runtime_kv_tile_multiple
    if window_size_left is not None:
        result["window_size_left"] = window_size_left
    if packed_window:
        result["packed_window"] = True
    return result


def domain_max(a: int | Int32, b: int | Int32) -> int | Int32:
    """``max`` counterpart of :func:`domain_min`."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return cute.math.max(as_i32(a), as_i32(b))


def domain_min(a: int | Int32, b: int | Int32) -> int | Int32:
    """``min`` that stays a Python ``int`` for static inputs and defers to
    ``cute.math.min`` otherwise."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return cute.math.min(as_i32(a), as_i32(b))


def init_causal_domain_state(
    task: Task,
    *,
    num_kv_tiles: int | Int32,
    tile_size_q: int | Int32,
    tile_size_kv: int | Int32,
    q_offset: int | Int32,
    seq_idx: int,
    batch_idx: int | None,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
    seq_lens_kv: cute.Pointer | None,
    runtime_kv_tile_multiple: int,
    reverse_seq_tiles: int | Int32 | None,
    offset: int,
    window_size_left: int,
    packed_window: bool,
    kwargs: dict[str, Any],
    task_init: Callable[..., None] = Task.__init__,
) -> None:
    """Initialize causal-domain fields and the static validation domain."""
    static_domain = None
    if isinstance(num_kv_tiles, int):
        assert isinstance(tile_size_q, int) and isinstance(tile_size_kv, int)
        if packed_window:
            static_domain = min(
                num_kv_tiles,
                bottom_right_window_max_tiles(
                    q_tile_m=tile_size_q,
                    kv_tile_n=tile_size_kv,
                    window_size_left=window_size_left,
                ),
            )
        else:
            seq_coord = 0
            max_q_row = q_offset + seq_coord * tile_size_q + tile_size_q - 1
            causal_n = max_q_row // tile_size_kv + 1
            static_domain = max(
                tile_size_q // tile_size_kv, min(num_kv_tiles, causal_n)
            )
        if window_size_left > 0 and not packed_window:
            static_domain -= bottom_right_window_tile_start(
                seq_coord=0,
                q_tile_m=tile_size_q,
                kv_tile_n=tile_size_kv,
                q_offset=q_offset,
                window_size_left=window_size_left,
            )
        if offset:
            static_domain -= offset
        static_domain = max(static_domain, 0)
    init_task_with_domain(task, kwargs, static_domain, task_init=task_init)
    task.num_kv_tiles = num_kv_tiles
    task.tile_size_q = tile_size_q
    task.tile_size_kv = tile_size_kv
    task._q_offset = q_offset
    # Index of the sequence coordinate in tile_coord.
    task.seq_idx = seq_idx
    # Mixed packed causal launches derive a request-local safe K-loop extent
    # from live metadata. Fixed and uniform-packed launches leave these fields
    # unset and retain the static plan-time domain calculation above.
    task.batch_idx = batch_idx
    task._cum_seqlen_q = cum_seqlen_q
    task.cum_seqlen_k = cum_seqlen_k
    task.seq_lens_kv = seq_lens_kv
    task._runtime_kv_tile_multiple = runtime_kv_tile_multiple
    task._reverse_seq_tiles = reverse_seq_tiles
    # Offset adjusts the domain count.
    task.offset = offset
    task._window_size_left = window_size_left
    task._packed_window = packed_window


def init_task_with_domain(
    task: Task,
    kwargs: dict[str, Any],
    domain: int | None,
    task_init: Callable[..., None] = Task.__init__,
) -> None:
    """Initialize a captured Task with its static validation loop domain."""
    schedule = kwargs.get("schedule")
    if schedule is None:
        raise ValueError("Domain tasks require a captured schedule.")
    if domain is not None:
        schedule.loop_end = domain
    task_init(task, **kwargs)


def select_fmha_domain_policy(
    cfg: FmhaConfig,
    *,
    num_kv_tiles: int | Int32,
    q_offset: int | Int32,
    cum_seqlen_q: cute.Tensor | None,
    cum_seqlen_k: cute.Tensor | None,
    seq_lens_kv: cute.Pointer | None = None,
    variable_window_token_starts: cute.Tensor | None = None,
    variable_window_token_ends: cute.Tensor | None = None,
    variable_window_cta_starts: cute.Tensor | None = None,
    variable_window_q_stride: int | Int32 = 0,
) -> FmhaDomainPolicy:
    """Select loop domains and softmax masks for the configured FMHA mode."""
    seq_idx = cfg.work_tile_coord_indices[0]
    batch_idx = cfg.work_tile_coord_indices[2]
    reverse_seq_tiles = (
        cfg.num_seq_tiles
        if cfg.uses_causal_reversed_head_batch_seq_tile_order
        else None
    )
    if cfg.has_variable_window:
        if (
            variable_window_token_starts is None
            or variable_window_token_ends is None
            or variable_window_cta_starts is None
        ):
            raise ValueError("VariableWindow domain requires start and end tensors")
        base_kwargs: DomainKwargs = {
            "task_class": VariableWindowDomainTask,
            "variable_window_token_starts": variable_window_token_starts,
            "variable_window_token_ends": variable_window_token_ends,
            "variable_window_cta_starts": variable_window_cta_starts,
            "num_kv_tiles": num_kv_tiles,
            "q_stride": variable_window_q_stride,
            "tile_size_q": cfg.cta_tiler[0],
            "tile_size_kv": cfg.kv_tile_n,
            "seq_idx": seq_idx,
            "batch_idx": batch_idx,
        }
        domain_n_kwargs = {**base_kwargs, "offset": 0}
        domain_n_minus_1_kwargs = {**base_kwargs, "offset": 1}
        return FmhaDomainPolicy(
            domain_n_kwargs=domain_n_kwargs,
            domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
            softmax0_domain_kwargs=domain_n_kwargs,
            softmax1_domain_kwargs=domain_n_kwargs,
        )
    # Head-paired causal/window: both peers use the same Q sequence tile from
    # adjacent Q heads, so both softmax tasks share the same causal tail domain.
    if cfg.head_paired and cfg.is_causal:
        causal_n = causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            tile_size_q=cfg.q_tile_m,
            tile_size_kv=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            offset=0,
            reverse_seq_tiles=reverse_seq_tiles,
            window_size_left=cfg.window_size_left,
            packed_window=cfg.has_varlen,
        )
        causal_n_minus_1 = causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            tile_size_q=cfg.q_tile_m,
            tile_size_kv=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            offset=1,
            reverse_seq_tiles=reverse_seq_tiles,
            window_size_left=cfg.window_size_left,
            packed_window=cfg.has_varlen,
        )
        return FmhaDomainPolicy(
            domain_n_kwargs=causal_n,
            domain_n_minus_1_kwargs=causal_n_minus_1,
            softmax0_domain_kwargs={
                **causal_n_minus_1,
                "task_class": CausalSoftmaxDomainTask,
            },
            softmax1_domain_kwargs={
                **causal_n_minus_1,
                "task_class": CausalSoftmaxDomainTask,
            },
        )
    # Head-paired dense: no causal/window trimming is needed, so both peers use
    # the full static K/V domain and the same non-tail softmax mask settings.
    if cfg.head_paired:
        domain_n_kwargs: DomainKwargs = {"domain": num_kv_tiles}
        domain_n_minus_1_kwargs: DomainKwargs = {"domain": num_kv_tiles - 1}
        return FmhaDomainPolicy(
            domain_n_kwargs=domain_n_kwargs,
            domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
            softmax0_domain_kwargs=domain_n_kwargs,
            softmax1_domain_kwargs=domain_n_kwargs,
        )
    # Query-paired causal: peer0 and peer1 cover consecutive Q sequence tiles,
    # so peer0 may need one fewer K/V loop tile than peer1.
    if cfg.is_causal:
        if cfg.causal_single_kv_tile:
            # The whole fixed K/V extent is one tile.  Static N=1/N-1=0
            # domains balance the existing head/tail protocol without the
            # generic synthetic peer0 tail iteration.
            domain_n_kwargs: DomainKwargs = {"domain": 1}
            domain_n_minus_1_kwargs: DomainKwargs = {"domain": 0}
            return FmhaDomainPolicy(
                domain_n_kwargs=domain_n_kwargs,
                domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
                softmax0_domain_kwargs=domain_n_minus_1_kwargs,
                softmax1_domain_kwargs=domain_n_minus_1_kwargs,
            )
        runtime_kv_tile_multiple = (
            (cfg.cta_tiler[0] + cfg.kv_tile_n - 1) // cfg.kv_tile_n
            if cfg.skip_causal_invalid_peer0
            else 1
        )
        causal_n = causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            tile_size_q=cfg.cta_tiler[0],
            tile_size_kv=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            seq_lens_kv=seq_lens_kv,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=0,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        causal_n_minus_1 = causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            tile_size_q=cfg.cta_tiler[0],
            tile_size_kv=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            seq_lens_kv=seq_lens_kv,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=1,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        causal_n_minus_2 = causal_domain_kwargs(
            num_kv_tiles=num_kv_tiles,
            tile_size_q=cfg.cta_tiler[0],
            tile_size_kv=cfg.kv_tile_n,
            q_offset=q_offset,
            seq_idx=seq_idx,
            batch_idx=batch_idx,
            cum_seqlen_q=cum_seqlen_q,
            cum_seqlen_k=cum_seqlen_k,
            seq_lens_kv=seq_lens_kv,
            runtime_kv_tile_multiple=runtime_kv_tile_multiple,
            offset=2,
            reverse_seq_tiles=reverse_seq_tiles,
        )
        # Query-paired causal maps peer0 one Q tile before peer1.
        # Whether peer0 has an extra invalid SP tile depends on the QK MMA
        # tile geometry: if peer0 spans fewer K/V tiles than the paired CTA,
        # Softmax0 uses the N-2 domain and consumes the remaining extra tile in
        # TAIL.
        # Otherwise it uses the same N-1 domain as peer1. S_q < S_kv keeps N-1
        # and lets the shifted causal mask handle the right edge.
        mma0_has_invalid_tail = cfg.skip_causal_invalid_peer0
        mma0_domain = causal_n_minus_2 if mma0_has_invalid_tail else causal_n_minus_1
        mma1_domain = causal_n_minus_1
        softmax0_domain = {
            **mma0_domain,
            "task_class": CausalSoftmaxDomainTask,
        }
        softmax1_domain = {
            **mma1_domain,
            "task_class": CausalSoftmaxDomainTask,
        }
        return FmhaDomainPolicy(
            domain_n_kwargs=causal_n,
            domain_n_minus_1_kwargs=causal_n_minus_1,
            softmax0_domain_kwargs=softmax0_domain,
            softmax1_domain_kwargs=softmax1_domain,
        )
    if uses_packed_dense_runtime_kv_bounds(cfg):
        if cum_seqlen_q is None or cum_seqlen_k is None:
            raise ValueError("Packed dense domains require runtime Q and KV offsets")
        dense_kwargs: DomainKwargs = {
            "task_class": PackedDenseDomainTask,
            "num_kv_tiles": num_kv_tiles,
            "tile_size_kv": cfg.kv_tile_n,
            "batch_idx": batch_idx,
            "cum_seqlen_k": cum_seqlen_k,
        }
        domain_n_kwargs = {**dense_kwargs, "offset": 0}
        domain_n_minus_1_kwargs = {**dense_kwargs, "offset": 1}
        return FmhaDomainPolicy(
            domain_n_kwargs=domain_n_kwargs,
            domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
            softmax0_domain_kwargs=domain_n_kwargs,
            softmax1_domain_kwargs=domain_n_kwargs,
        )
    # Dense query-paired fallback: no head-pairing or causal mask, so both
    # peers traverse the full static K/V domain.
    domain_n_kwargs = {"domain": num_kv_tiles}
    domain_n_minus_1_kwargs = {"domain": num_kv_tiles - 1}
    # Query-paired softmax handles the partial last K/V tile after its loop:
    # N-1 unmasked iterations, then a masked tail. The single-QKV schedules
    # mask the last tile inside the loop instead, so they keep the full N domain.
    softmax_uses_dense_k_tail = (
        not cfg.single_qkv_instance
        and not cfg.is_causal
        and not cfg.has_varlen
        and cfg.fixed_dense_k_tail > 0
    )
    softmax_domain_kwargs = (
        domain_n_minus_1_kwargs if softmax_uses_dense_k_tail else domain_n_kwargs
    )
    return FmhaDomainPolicy(
        domain_n_kwargs=domain_n_kwargs,
        domain_n_minus_1_kwargs=domain_n_minus_1_kwargs,
        softmax0_domain_kwargs=softmax_domain_kwargs,
        softmax1_domain_kwargs=softmax_domain_kwargs,
    )


def uses_packed_dense_runtime_kv_bounds(cfg: FmhaConfig) -> bool:
    """Bound ordinary query-paired dense context by each live packed KV extent."""
    # All task roles share the same paired N/N-1 protocol and int32 offsets.
    # Uniform packed plans omit live offsets; specialized recipes and other
    # topologies retain their own domains and tail handling.
    # These are the public API's paired geometries; symmetric D192/D256 use
    # one instance, and the public context API does not admit D64.
    return (
        not cfg.is_causal
        and not cfg.has_variable_window
        and cfg.window_size_left == 0
        and not cfg.head_paired
        and not cfg.use_paged_kv
        and cfg.has_varlen
        and not cfg.has_uniform_varlen
        and cfg.packed_dense_k_mask
        and cfg.num_qkv_instances == 2
        and not cfg.two_cta_umma
        and cfg.logical_head_dim_qk in (128, 192)
        and cfg.pv_mma_tiler[1] == 128
        and cfg.vc_attention is None
        and not cfg.use_sage_attention
        and not cfg.use_mxfp8_qk
        and cfg.q_dtype == cfg.k_dtype == cfg.v_dtype
        and cfg.q_dtype in (cutlass.Float16, cutlass.BFloat16, cutlass.Float8E4M3FN)
        and cfg.o_dtype in (cutlass.Float16, cutlass.BFloat16)
    )


def writes_empty_dense_lse(cfg: FmhaConfig) -> bool:
    """Publish neutral O/LSE for empty ordinary contiguous dense requests."""
    return (
        cfg.lse_mode != "none"
        and not cfg.is_causal
        and not cfg.has_variable_window
        and cfg.window_size_left == 0
        and not cfg.use_paged_kv
        and cfg.vc_attention is None
        and not cfg.use_sage_attention
        and not cfg.use_mxfp8_qk
    )
