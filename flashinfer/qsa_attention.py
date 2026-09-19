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

import math
import numbers
from typing import NamedTuple, Optional, Tuple

import torch

from .qsa_output_gate import qsa_output_gate
from .sparse import BlockSparseAttentionWrapper
from .sparse_route import qsa_route_from_logical
from .topk import WORKSPACE_ALIGNMENT


#: What a cache's bytes mean. A ``uint8`` tensor is a view of raw memory and
#: says nothing on its own -- raw FP8 and packed NVFP4 both arrive as one -- so
#: the caller names the format and the checks below follow from the name.
#:
#: ``dense``     values at their own dtype, no scales at all.
#: ``fp8_e4m3``  e4m3 values, one host scale per tensor, no scale planes.
#: ``nvfp4``     packed e2m1 values with an e4m3 scale plane each, and a host
#:               scale per tensor on top of them.
KV_CACHE_FORMATS = ("dense", "fp8_e4m3", "nvfp4")


#: The rung a decode step lands on. Decode batches are small and the padding is
#: masked off, so one rung well under a chunk keeps them off the wide plan.
_DECODE_BUCKET = 128
#: What the rungs above it step by. A caller's captured shapes tend to come in
#: multiples of this, so each gets a rung of its own.
_ROW_GRANULARITY = 1024
#: What the sizing query is asked with. The plan's size does not depend on it,
#: and the real one is not known until there is a cache.
_SIZING_PAGE_SIZE = 16


def row_buckets(max_rows: int, max_plans: int = 16) -> Tuple[int, ...]:
    """Row counts to keep a plan for, for a caller that may send ``max_rows``.

    A step pads up to one of these and the padding rows are masked off, so a
    batch whose size moves never replans -- and a batch past the widest rung
    has no plan at all, so the widest rung is ``max_rows``.

    How many rungs below it is a measured question, and the answer is "as many
    as fit", because the route and the mask are one shared allocation: a rung
    costs about 33 KiB of plan arena and nothing else. Measured at 24 query
    heads, head_dim 256, route 2051, nvfp4, ms per step:

        rows          128    2048    3072    5120    6144    8192
        (128, 8192)  0.73   34.25   34.02      --      --   34.97
        powers of 2  0.55    8.71   17.24   34.44   34.64   34.96
        this ladder  0.72    8.70   13.04   21.79   26.16   34.94

    Persistent memory across all three is 386.6-387.2 MiB: flat, because the
    buffers that scale with rows are shared.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if max_plans < 1:
        raise ValueError(f"max_plans must be positive, got {max_plans}")
    decode = min(_DECODE_BUCKET, max_rows)
    rungs = {decode, max_rows}
    rungs.update(range(_ROW_GRANULARITY, max_rows, _ROW_GRANULARITY))
    if len(rungs) > max_plans:
        # Thin geometrically rather than dropping the bottom: what a rung is
        # worth is the padding it saves, and padding is a ratio, so rungs
        # evenly spaced on a log scale bound it evenly. Dropping the low ones
        # would leave a decode batch padded to a chunk.
        steps = max_plans - 1
        rungs = {decode, max_rows}
        if steps > 0:
            ratio = (max_rows / decode) ** (1.0 / steps)
            rungs.update(round(decode * ratio**step) for step in range(1, steps))
    return tuple(sorted(rungs))


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _check_geometry(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    route_width: int,
    kv_cache_format: str,
    kv_data_type: torch.dtype,
    kv_layout: str,
) -> None:
    """Everything about a shape that can be judged without a cache.

    Asked in one place because two callers ask it: the constructor, which has
    no cache yet, and the sizing query, which never builds one. A geometry this
    refuses has no workspace size either.
    """
    if route_width < 1:
        raise ValueError(f"route_width must be positive, got {route_width}")
    if kv_layout not in ("NHD", "HND"):
        raise ValueError(f"kv_layout must be NHD or HND, got {kv_layout!r}")
    if num_qo_heads < 1 or num_kv_heads < 1 or head_dim < 1:
        raise ValueError("heads and head_dim have to be positive")
    if num_qo_heads % num_kv_heads:
        raise ValueError(
            "every KV head serves a whole number of query heads, got "
            f"{num_qo_heads} over {num_kv_heads}"
        )
    if kv_cache_format not in KV_CACHE_FORMATS:
        raise ValueError(
            f"kv_cache_format must be one of {KV_CACHE_FORMATS}, got "
            f"{kv_cache_format!r}"
        )
    # The format says what the bytes are; the dtype has to agree with it
    # rather than stand in for it.
    if kv_cache_format == "nvfp4":
        if kv_data_type is not torch.uint8:
            raise ValueError(
                f"a packed NVFP4 cache is handed over as uint8, got {kv_data_type}"
            )
        # One e4m3 scale per sixteen values, two values per byte: a head the
        # groups do not divide has no layout.
        if head_dim % 16:
            raise ValueError(
                "NVFP4 packs one scale per sixteen values, so head_dim has to "
                f"be a multiple of sixteen, got {head_dim}"
            )
    if kv_cache_format == "fp8_e4m3" and kv_data_type is not torch.float8_e4m3fn:
        raise ValueError(
            f"an FP8 cache is handed over as float8_e4m3fn, got {kv_data_type}; "
            "a uint8 view of the same bytes is a zero-copy .view() away and "
            "keeps the format in the type"
        )
    if kv_cache_format == "dense" and kv_data_type in (
        torch.uint8,
        torch.float8_e4m3fn,
    ):
        raise ValueError(f"{kv_data_type} is not a dense cache; name the format it is")


def _plan_bytes_for(
    *,
    buckets,
    device: torch.device,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    route_width: int,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    o_data_type: torch.dtype,
    backend: str,
) -> Tuple[int, Tuple[int, ...]]:
    """Float workspace bytes, and the integer region each bucket's plan needs.

    Asked of the planner rather than of a wrapper: building one to size its own
    workspace costs eight megabytes of device memory and as much pinned host
    memory, and this is called before there is a workspace at all.
    """
    float_bytes = 0
    sizes = []
    for rows in buckets:
        # The planner reads this on the host and builds nothing.
        indptr = torch.arange(
            0,
            (rows + 1) * route_width,
            route_width,
            dtype=torch.int32,
            device="cpu",
        )
        rows_float, rows_int = BlockSparseAttentionWrapper.query_workspace_size(
            device,
            indptr,
            rows,
            1,  # R
            1,  # C
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            use_custom_mask=True,
            # The planner lays out one entry per route element and the page
            # size only says how those elements are addressed, so the answer
            # does not move with it. Asserted in the suite rather than assumed.
            kv_cache_page_size=_SIZING_PAGE_SIZE,
            backend=backend,
        )
        float_bytes = max(float_bytes, int(rows_float))
        sizes.append(_round_up(int(rows_int), WORKSPACE_ALIGNMENT))
    return _round_up(float_bytes, WORKSPACE_ALIGNMENT), tuple(sizes)


class _Persistent(NamedTuple):
    """What has to survive between calls, and where it sits in its buffer.

    The plans keep byte offsets into the integer arena and read them back when
    they run; the row pointers are read by every plan that was built against
    them. Neither is rewritten by a call, so neither can live in memory another
    consumer reuses in the meantime.
    """

    arena: Tuple[int, int]
    indptr: Tuple[Tuple[int, int], ...]
    total: int


class _Transient(NamedTuple):
    """What a call rewrites before it reads, and where it sits in its buffer.

    All of it is scratch: the route and the mask are written for every row the
    plan covers, the padded output is written by the kernel, the padded query's
    live rows are copied in, and the float workspace is the split-k algorithm's
    own. A caller may hand over the same bytes it gives everything else.
    """

    float_workspace: Tuple[int, int]
    padded_q: Tuple[int, int]
    padded_out: Tuple[int, int]
    route: Tuple[int, int]
    mask: Tuple[int, int]
    total: int


def _walk():
    """A cursor that hands out aligned spans and remembers where it got to."""
    offset = 0

    def take(nbytes):
        nonlocal offset
        begin = offset
        offset += _round_up(nbytes, WORKSPACE_ALIGNMENT)
        return (begin, nbytes)

    def total():
        return offset

    return take, total


def _check_buffer(buffer: torch.Tensor, what: str) -> None:
    if buffer.dtype != torch.uint8:
        raise ValueError(f"{what} is raw bytes, got {buffer.dtype}")
    if not buffer.is_cuda:
        raise ValueError(f"{what} has to be on a CUDA device")
    if not buffer.is_contiguous():
        raise ValueError(f"{what} has to be contiguous")
    if buffer.data_ptr() % WORKSPACE_ALIGNMENT:
        raise ValueError(f"{what} has to be {WORKSPACE_ALIGNMENT}-byte aligned")


def _cut(buffer: torch.Tensor, span, dtype: torch.dtype, shape):
    begin, size = span
    view = buffer[begin : begin + size].view(dtype)
    return view if shape is None else view.view(shape)


def _persistent_layout(*, buckets, plan_bytes) -> _Persistent:
    """Cut the buffer this object keeps for as long as it exists."""
    take, total = _walk()
    arena = take(sum(plan_bytes))
    indptr = tuple(take((rows + 1) * 4) for rows in buckets)
    return _Persistent(arena=arena, indptr=indptr, total=total())


def _transient_layout(
    *,
    buckets,
    float_bytes: int,
    num_qo_heads: int,
    head_dim: int,
    route_width: int,
    mask_bytes: int,
    q_data_type: torch.dtype,
    o_data_type: torch.dtype,
) -> _Transient:
    """Cut the buffer a call may share with everything else the step runs."""
    take, total = _walk()
    widest = buckets[-1]
    float_workspace = take(float_bytes)
    padded_q = take(widest * num_qo_heads * head_dim * q_data_type.itemsize)
    padded_out = take(widest * num_qo_heads * head_dim * o_data_type.itemsize)
    route = take(widest * route_width * 4)
    mask = take(widest * mask_bytes)
    return _Transient(
        float_workspace=float_workspace,
        padded_q=padded_q,
        padded_out=padded_out,
        route=route,
        mask=mask,
        total=total(),
    )


class QSAAttention:
    r"""Sparse attention over a paged cache, from a logical token route.

    What a caller hands over is the route a selector produced -- logical token
    indices, ``-1`` where a position has no token -- and what comes back is the
    gated attention output. Everything between is here: the route is mapped
    through the block table into physical slots and a validity mask, the plan
    for the row count is already built, and the gate is folded in by a kernel
    rather than by a chain of elementwise ops.

    **Nothing is allocated by :meth:`run`.** The integer workspace the planner
    writes into is a region the caller provides once, cut into a slice per row
    bucket that no other plan touches; the route, the mask, the padded query and
    the padded output are the instance's own, made when it is built.

    **The plans are all built up front.** A row count is rounded up to one of
    ``row_buckets`` and the plan for that bucket already exists, so a step whose
    batch changes size never plans, never allocates, and never has to be told
    that a CUDA graph capture is in progress.

    Parameters
    ----------
    float_workspace_buffer : torch.Tensor
        The split-k scratch, as :class:`BlockSparseAttentionWrapper` takes it.
        Shared by every bucket: they run one after another inside a step.
    max_rows : int
        The widest batch the caller may send. The plan ladder is derived
        from it; see :func:`row_buckets`.
    num_qo_heads, num_kv_heads, head_dim : int
        The attention's shape.
    num_slots : int
        How many KV entries the cache holds, counting every page.
    page_size : int
        KV entries per page. The route names entries, so the wrapper divides
        each index back into a page and an offset.
    route_width : int
        Columns the route carries. Indices only -- a route with a trailing count
        column is a different width and is refused.
    q_data_type, kv_data_type, o_data_type : torch.dtype
        The query's, the cache's and the output's dtypes. An NVFP4 cache is
        ``uint8`` here and its scales arrive separately; the dtype alone is
        never read as a format.
    kv_layout : str
        ``NHD`` or ``HND``, as the cache is laid out.
    backend : str
        Which block-sparse backend to plan for. A paged route is served by
        ``fa2`` alone, so ``auto`` resolves there.
    max_plans : int
        Kept for symmetry with wrappers that evict; here the plan set is the
        bucket set, so this only bounds how many buckets may be asked for.
    """

    def __init__(
        self,
        persistent: torch.Tensor,
        *,
        max_rows: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        route_width: int,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        o_data_type: torch.dtype,
        kv_cache_format: str,
        kv_layout: str = "NHD",
        backend: str = "auto",
        max_plans: int = 16,
    ) -> None:
        buckets = list(row_buckets(int(max_rows), max_plans))
        _check_geometry(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            route_width=route_width,
            kv_cache_format=kv_cache_format,
            kv_data_type=kv_data_type,
            kv_layout=kv_layout,
        )
        _check_buffer(persistent, "the persistent workspace")

        device = persistent.device
        self.device = device
        self.row_buckets = tuple(buckets)
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.route_width = route_width
        self.q_data_type = q_data_type
        self.kv_data_type = kv_data_type
        self.o_data_type = o_data_type
        self.kv_layout = kv_layout
        self.kv_cache_format = kv_cache_format
        self.backend = backend
        self.mask_bytes = -(-route_width // 8)
        # Set by plan_cache, which is the only thing that needs a cache. The
        # page size is the cache's too: how many slots a page holds is decided
        # when the cache is allocated, which is after this is sized.
        self.num_slots: Optional[int] = None
        self.page_size: Optional[int] = None
        self.pages: Optional[int] = None

        self._float_bytes, self._plan_bytes = _plan_bytes_for(
            buckets=buckets,
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            route_width=route_width,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            backend=backend,
        )
        regions = _persistent_layout(buckets=buckets, plan_bytes=self._plan_bytes)
        if persistent.numel() < regions.total:
            raise ValueError(
                f"this geometry needs {regions.total} persistent bytes, got "
                f"{persistent.numel()}"
            )

        self._persistent = persistent
        self._arena = _cut(persistent, regions.arena, torch.uint8, None)
        self._indptr = {}
        for rows, span in zip(buckets, regions.indptr, strict=True):
            view = _cut(persistent, span, torch.int32, None)
            torch.arange(0, (rows + 1) * route_width, route_width, out=view)
            self._indptr[rows] = view

        self._transient: Optional[torch.Tensor] = None
        self._float_workspace: Optional[torch.Tensor] = None
        self._padded_q: Optional[torch.Tensor] = None
        self._padded_out: Optional[torch.Tensor] = None
        self._route_base: Optional[torch.Tensor] = None
        self._mask_base: Optional[torch.Tensor] = None
        self._route: dict = {}
        self._mask: dict = {}
        self._wrappers: dict = {}
        self._slices: list = []
        self._staging: Optional[torch.Tensor] = None
        self._frozen = False

    def bind_transient_workspace(self, transient: torch.Tensor) -> None:
        """Take the scratch a call rewrites before it reads.

        Separate from the buffer above because the two have different
        lifetimes, and a caller's scratch is scratch: the memory handed over
        here may be the same memory every other consumer of the step reuses,
        and between two calls anything at all may have been written to it. What
        may never live here is a plan, because a plan is read back.

        Bound once the caller's scratch has stopped moving, which for a
        workspace that grows by reallocating means after it is locked.
        """
        _check_buffer(transient, "the transient workspace")
        regions = self._transient_regions()
        if transient.numel() < regions.total:
            raise ValueError(
                f"this geometry needs {regions.total} transient bytes, got "
                f"{transient.numel()}"
            )
        buckets = self.row_buckets
        widest = buckets[-1]
        self._transient = transient
        self._float_workspace = _cut(
            transient, regions.float_workspace, torch.uint8, None
        )
        self._padded_q = _cut(
            transient,
            regions.padded_q,
            self.q_data_type,
            (widest, self.num_qo_heads, self.head_dim),
        )
        self._padded_out = _cut(
            transient,
            regions.padded_out,
            self.o_data_type,
            (widest, self.num_qo_heads, self.head_dim),
        )
        self._route_base = _cut(
            transient, regions.route, torch.int32, (widest, self.route_width)
        )
        self._mask_base = _cut(transient, regions.mask, torch.uint8, None)
        self._route = {rows: self._route_base[:rows] for rows in buckets}
        self._mask = {
            rows: self._mask_base[: rows * self.mask_bytes] for rows in buckets
        }

    def _transient_regions(self) -> _Transient:
        return _transient_layout(
            buckets=self.row_buckets,
            float_bytes=self._float_bytes,
            num_qo_heads=self.num_qo_heads,
            head_dim=self.head_dim,
            route_width=self.route_width,
            mask_bytes=self.mask_bytes,
            q_data_type=self.q_data_type,
            o_data_type=self.o_data_type,
        )

    # -- workspace ---------------------------------------------------------

    @staticmethod
    def workspace_bytes(
        *,
        device: torch.device,
        max_rows: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        route_width: int,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        o_data_type: torch.dtype,
        kv_cache_format: str,
        kv_layout: str = "NHD",
        backend: str = "auto",
        max_plans: int = 16,
    ) -> Tuple[int, int]:
        """What this geometry runs out of, as two numbers and no allocation.

        Two, because the bytes have two lifetimes. The first are the plans and
        the row pointers they read: those are written once and read back on
        every call, so they need memory nobody else writes. The second are the
        padded query and output, the route, its mask and the float workspace:
        every call rewrites them before it reads them, so they can come out of
        whatever scratch the caller shares between its consumers.

        ``num_slots`` and ``page_size`` are absent from both: they are what a
        cache has, and they change the plan rather than the room it needs.

        Returns
        -------
        Tuple[int, int]
            Persistent bytes, then transient bytes.
        """
        _check_geometry(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            route_width=route_width,
            kv_cache_format=kv_cache_format,
            kv_data_type=kv_data_type,
            kv_layout=kv_layout,
        )
        buckets = row_buckets(int(max_rows), max_plans)
        float_bytes, plan_bytes = _plan_bytes_for(
            buckets=buckets,
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            route_width=route_width,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            backend=backend,
        )
        persistent = _persistent_layout(buckets=buckets, plan_bytes=plan_bytes)
        transient = _transient_layout(
            buckets=buckets,
            float_bytes=float_bytes,
            num_qo_heads=num_qo_heads,
            head_dim=head_dim,
            route_width=route_width,
            mask_bytes=-(-route_width // 8),
            q_data_type=q_data_type,
            o_data_type=o_data_type,
        )
        return persistent.total, transient.total

    # -- planning ----------------------------------------------------------

    def plan_cache(self, num_slots: int, page_size: int) -> None:
        """Build every bucket's plan for a cache of this many slots.

        Separate from construction because the two are answered at different
        times: how much room this needs is a property of the geometry and is
        asked while the caller's workspace can still grow, and what the plans
        are depends on the cache, which does not exist until later. A caller
        that binds a minimal cache for a memory profile and then the real one
        calls this twice; the workspace is the same both times and nothing is
        allocated by the second call.

        Building the plans here rather than on first use is what keeps
        :meth:`run` free of planning: a capture never reaches a plan that does
        not exist. Every plan is built before any of them is published, so a
        failure leaves the attention unplanned rather than half planned.

        Asking for the cache it is already planned for does nothing, which is
        what makes a shared runtime work: every layer of a rank binds the same
        cache and calls this, and only the first of them does anything. It
        stays a no-op after a run for the same reason -- nothing about the
        schedule changes, so there is nothing for a capture to lose.

        A *different* cache after a run is refused. The plans keep byte offsets
        into the arena and a graph replays them, so replacing one under a
        capture that already holds it would be reading a schedule that is no
        longer there.
        """
        if self._transient is None:
            raise RuntimeError(
                "bind_transient_workspace() has to be called before planning: "
                "the plans run out of the float workspace it carries"
            )
        if (
            self._wrappers
            and self.num_slots == num_slots
            and self.page_size == page_size
        ):
            return
        if self._frozen:
            raise RuntimeError(
                f"this attention has run against {self.num_slots} slots: "
                f"replanning for {num_slots} would move the schedule out from "
                "under plans a graph may still replay"
            )
        if num_slots < 1:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}")
        if num_slots % page_size:
            raise ValueError(
                "the cache holds whole pages, so num_slots has to be a "
                f"multiple of page_size, got {num_slots} and {page_size}"
            )
        self.page_size = page_size

        ranges = []
        offset = 0
        for rows, size in zip(self.row_buckets, self._plan_bytes, strict=True):
            ranges.append((rows, offset, offset + size))
            offset += size

        # One staging buffer for every plan. The planner copies through it and
        # is done with it before it returns, and the plans below are built one
        # after another, so they share it rather than taking eight megabytes of
        # pinned host memory each. Host memory, so it is not the caller's
        # workspace and not device memory anyone counted.
        if self._staging is None:
            self._staging = torch.empty(
                (max(self._plan_bytes),),
                dtype=torch.uint8,
                pin_memory=True,
                device="cpu",
            )

        # The planner reads the route and the mask, and both live in the
        # caller's shared scratch, which at this point still holds whatever the
        # last consumer of that buffer left there. A route element is loaded
        # into a uint32_t, so a leftover negative is rejected and a leftover
        # in-range value plans against a pattern no step will use. The schedule
        # comes from the row pointers, which are fixed, so an all-zero route
        # and an empty mask plan exactly the same thing and depend on nothing.
        self._route_base.zero_()
        self._mask_base.zero_()

        # Nothing is published until every plan is built: a failure half way
        # through would otherwise leave an object that looks planned and is not.
        wrappers = {}
        for rows, start, end in ranges:
            wrapper = BlockSparseAttentionWrapper(
                self._float_workspace,
                backend=self.backend,
                kv_layout=self.kv_layout,
                int_workspace_buffer=self._arena[start:end],
                pin_memory_int_workspace_buffer=self._staging[: end - start],
            )
            wrapper.plan(
                self._indptr[rows],
                self._route[rows].view(-1),
                rows,
                num_slots,
                1,
                1,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                packed_mask=self._mask[rows],
                mask=None,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_data_type,
                o_data_type=self.o_data_type,
                kv_cache_page_size=page_size,
            )
            wrappers[rows] = wrapper

        self.num_slots = num_slots
        self.pages = num_slots // page_size
        self._slices = ranges
        # The previous set, if there was one, is dropped here: its plans lived
        # in the same arena these just wrote over, so nothing may still be
        # holding them -- which is what the refusal above is for.
        self._wrappers = wrappers

    def bucket_for(self, rows: int) -> int:
        """The plan a batch of this many rows runs on."""
        for bucket in self.row_buckets:
            if rows <= bucket:
                return bucket
        raise ValueError(
            f"{rows} rows is past the largest bucket {self.row_buckets[-1]}"
        )

    # -- execution ---------------------------------------------------------

    # -- validation --------------------------------------------------------

    def _check_tensor(self, tensor, name, shape, dtype, packed=True):
        """Shape, dtype, device and layout, before anything is launched.

        ``packed`` is for the cache planes, which are not required to be
        contiguous -- only unit-stride in their innermost dimension. A packed
        NVFP4 cache is commonly laid out as one allocation per slot holding
        ``[fp4 data | e4m3 block scales]``, which is what this library's own
        writer takes and what leaves both planes strided; the block-sparse
        route reads page, token and head strides from the tensors. Demanding
        contiguity here would refuse a cache the writer just filled.
        """
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"{name} must be {tuple(shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}, got {tensor.device}")
        if packed:
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        elif tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must be contiguous in its innermost dimension, got "
                f"stride {tensor.stride(-1)}"
            )

    def _checked_scale(self, value, name):
        """A global scale is a real, finite, positive host number.

        Not a tensor: reading one from the device on every call is a
        synchronisation a graph capture refuses. Not a bool, not a NaN, not
        zero and not negative -- each of those reaches the kernel as a
        multiplier and produces something that looks like an answer.
        """
        if isinstance(value, torch.Tensor):
            raise TypeError(
                f"{name} has to be a host float: reading it from the device on "
                "every call is a synchronisation a graph capture refuses"
            )
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite, got {number}")
        if number <= 0.0:
            raise ValueError(f"{name} must be positive, got {number}")
        return number

    def _cache_shape(self, last):
        """A cache of this layout, with ``last`` values per entry."""
        if self.kv_layout == "NHD":
            return (self.pages, self.page_size, self.num_kv_heads, last)
        return (self.pages, self.num_kv_heads, self.page_size, last)

    def _check_call(
        self,
        rows,
        q,
        k_data,
        v_data,
        route,
        block_table,
        token_to_req,
        output_gate,
        k_sf,
        v_sf,
        k_scale,
        v_scale,
    ):
        """Everything the kernels assume, checked before the first of them runs.

        A cache of the wrong shape, a scale plane that is too short, a global
        scale left out -- each of those reaches a kernel as a pointer it reads
        past the end of, or as a multiplier of one where the caller meant
        something else, and none of them is a mistake the kernel can report.
        """
        self._check_tensor(
            q, "q", (rows, self.num_qo_heads, self.head_dim), self.q_data_type
        )
        if route.ndim == 2 and route.size(1) == self.route_width + 1:
            raise ValueError(
                f"route must be [{rows}, {self.route_width}]: the selection's "
                "trailing count column is not a route column, and reading it as "
                "one would send the kernel to whatever slot the count names"
            )
        self._check_tensor(route, "route", (rows, self.route_width), torch.int32)
        if token_to_req.ndim != 1:
            raise ValueError(f"token_to_req is one axis, got {token_to_req.ndim}")
        if token_to_req.size(0) < rows:
            raise ValueError(
                f"token_to_req must cover {rows} rows, got {token_to_req.size(0)}"
            )
        self._check_tensor(
            token_to_req, "token_to_req", (token_to_req.size(0),), torch.int32
        )
        if block_table.ndim != 2:
            raise ValueError(
                f"block_table is [requests, pages], got {tuple(block_table.shape)}"
            )
        self._check_tensor(
            block_table, "block_table", tuple(block_table.shape), torch.int32
        )

        # The gate arrives split by head or flat over them; both are the same
        # values, and both have to cover the batch.
        if output_gate.ndim == 3:
            gate_shape = (rows, self.num_qo_heads, self.head_dim)
        elif output_gate.ndim == 2:
            gate_shape = (rows, self.num_qo_heads * self.head_dim)
        else:
            raise ValueError(
                "output_gate is [rows, heads, dim] or [rows, heads * dim], got "
                f"{output_gate.ndim} axes"
            )
        if tuple(output_gate.shape) != gate_shape:
            raise ValueError(
                f"output_gate must be {gate_shape}, got {tuple(output_gate.shape)}"
            )
        if output_gate.dtype != self.o_data_type:
            raise ValueError(
                f"output_gate must be {self.o_data_type}, got {output_gate.dtype}"
            )
        if output_gate.device != self.device:
            raise ValueError(
                f"output_gate must be on {self.device}, got {output_gate.device}"
            )

        # The cache, by the format that was named when this was built. A packed
        # NVFP4 entry is half a byte per value; everything else is one value.
        entry = self.head_dim // 2 if self.kv_cache_format == "nvfp4" else self.head_dim
        for name, tensor in (("k_data", k_data), ("v_data", v_data)):
            self._check_tensor(
                tensor, name, self._cache_shape(entry), self.kv_data_type, packed=False
            )

        planes = (k_sf, v_sf)
        if self.kv_cache_format == "nvfp4":
            if any(plane is None for plane in planes):
                raise ValueError(
                    "an nvfp4 cache carries a scale plane for K and one for V"
                )
            plane_shape = self._cache_shape(self.head_dim // 16)
            for name, plane in (("k_sf", k_sf), ("v_sf", v_sf)):
                self._check_tensor(
                    plane, name, plane_shape, torch.float8_e4m3fn, packed=False
                )
        elif any(plane is not None for plane in planes):
            raise ValueError(
                f"a {self.kv_cache_format} cache has no scale planes; k_sf and "
                "v_sf belong to nvfp4"
            )

        checked: dict = {}
        if self.kv_cache_format == "dense":
            for name, value in (("k_scale", k_scale), ("v_scale", v_scale)):
                if value is not None:
                    raise ValueError(
                        f"a dense cache has no global scale to apply, got {name}"
                    )
        else:
            for name, value in (("k_scale", k_scale), ("v_scale", v_scale)):
                if value is None:
                    raise ValueError(
                        f"a {self.kv_cache_format} cache is stored relative to "
                        f"{name}; leaving it out reads the values as if it were "
                        "one"
                    )
                checked[name] = self._checked_scale(value, name)
        return checked

    def run(
        self,
        q: torch.Tensor,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        *,
        route: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor,
        k_sf: Optional[torch.Tensor] = None,
        v_sf: Optional[torch.Tensor] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend over the route, gate the result, and write it to ``out``.

        Parameters
        ----------
        q : torch.Tensor
            Queries, ``[rows, num_qo_heads, head_dim]``.
        k_data, v_data : torch.Tensor
            The cache as it is allocated. For a packed NVFP4 cache these are the
            data planes and ``k_sf``/``v_sf`` are the scale planes; the split is
            the caller's, because the caller is the one that laid the cache out.
        route : torch.Tensor
            Logical token route, ``[rows, route_width]`` int32, ``-1`` where a
            position has no token. Indices only.
        block_table : torch.Tensor
            Logical page to physical page per request.
        token_to_req : torch.Tensor
            Request each row belongs to.
        k_sf, v_sf : Optional[torch.Tensor]
            The NVFP4 block-scale planes, when the cache is packed.
        k_scale, v_scale : Optional[float]
            The cache's global scales, as **host floats**. A tensor would be
            read back from the device on every call, which a graph capture
            refuses.
        output_gate : torch.Tensor
            The gate to fold in, ``[rows, num_qo_heads * head_dim]`` or
            ``[rows, num_qo_heads, head_dim]``. Left unchanged. **Required**:
            the model this serves gates its attention output, and a route that
            served it ungated once already is what this API exists to prevent.
            Sparse attention without a gate is
            :class:`BlockSparseAttentionWrapper`.
        out : Optional[torch.Tensor]
            Where to write, ``[rows, num_qo_heads, head_dim]``. Allocated when
            omitted, which a captured step should not do.

        Returns
        -------
        torch.Tensor
            ``out``.
        """
        if self._transient is None:
            raise RuntimeError(
                "bind_transient_workspace() has to be called before the first "
                "run: the route, the mask and the padded buffers live in it"
            )
        if not self._wrappers:
            raise RuntimeError(
                "plan_cache() has to be called before the first run: a plan is "
                "for a cache of a particular size and there is none yet"
            )
        # Every later call sees this, so the plans this run is about to use are
        # the plans a capture of it will keep replaying.
        self._frozen = True
        rows = q.size(0)
        bucket = self.bucket_for(rows)
        checked_scales = self._check_call(
            rows,
            q,
            k_data,
            v_data,
            route,
            block_table,
            token_to_req,
            output_gate,
            k_sf,
            v_sf,
            k_scale,
            v_scale,
        )
        if out is None:
            out = torch.empty(
                (rows, self.num_qo_heads, self.head_dim),
                dtype=self.o_data_type,
                device=self.device,
            )
        else:
            self._check_tensor(
                out, "out", (rows, self.num_qo_heads, self.head_dim), self.o_data_type
            )

        # The route for this step, mapped into the plan's own buffers. Rows past
        # the batch are padding and come out fully masked.
        qsa_route_from_logical(
            route,
            token_to_req,
            block_table,
            self._route[bucket],
            self._mask[bucket],
            rows,
            self.page_size,
            self.num_slots,
        )

        padded_q = self._padded_q[:bucket]
        if rows < bucket:
            padded_q[:rows].copy_(q)
            query = padded_q
        else:
            query = q
        attention = self._padded_out[:bucket]

        extra = {}
        if k_sf is not None:
            extra["kv_cache_sf"] = (k_sf, v_sf)
        # The numbers the wrapper is given are the ones that were checked, not
        # whatever object the caller passed in.
        extra.update(checked_scales)
        self._wrappers[bucket].run(query, k_data, v_data, out=attention, **extra)

        gate = (
            output_gate
            if output_gate.ndim == 3
            else output_gate.unflatten(1, (self.num_qo_heads, self.head_dim))
        )
        # The gate kernel reads the padded attention output and writes the
        # caller's rows, so the padding never has to be copied away separately.
        qsa_output_gate(attention, gate, out=out)
        return out
