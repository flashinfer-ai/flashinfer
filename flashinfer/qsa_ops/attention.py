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

from ..api_logging import flashinfer_api
from ..trace.templates.qsa import qsa_attention_run_trace
from .output_gate import qsa_output_gate
from ..sparse import BlockSparseAttentionWrapper
from .route import qsa_route_from_logical
from ._workspace import check_buffer, check_tensor, cut, walk
from ..topk import WORKSPACE_ALIGNMENT
from ..utils import round_up


#: What a cache's bytes mean; a ``uint8`` cache says nothing on its own.
#: ``dense``     values at their own dtype, no scales at all.
#: ``fp8_e4m3``  e4m3 values, one host scale per tensor, no scale planes.
#: ``nvfp4``     packed e2m1 values with an e4m3 scale plane each, and a host
#:               scale per tensor on top of them.
KV_CACHE_FORMATS = ("dense", "fp8_e4m3", "nvfp4")


#: What the rungs above the powers of two step by. A caller's captured shapes
#: tend to come in multiples of this, so each gets a rung of its own.
_ROW_GRANULARITY = 1024
#: What the sizing query is asked with. The plan's size does not depend on it,
#: and the real one is not known until there is a cache.
_SIZING_PAGE_SIZE = 16


def row_buckets(max_rows: int, max_plans: int = 16) -> Tuple[int, ...]:
    """Row counts to keep a plan for, for a caller that may send ``max_rows``.

    A step pads up to one of these and the top rung is ``max_rows``. The rungs
    are the powers of two and the multiples of the granularity above them,
    which keep a batch under twice its size unless ``max_plans`` thins the
    ladder. A padding row is masked off but costs what a real row does.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")
    if max_plans < 1:
        raise ValueError(f"max_plans must be positive, got {max_plans}")
    rungs = {max_rows}
    rungs.update(1 << k for k in range(max_rows.bit_length()) if 1 << k < max_rows)
    rungs.update(range(_ROW_GRANULARITY, max_rows, _ROW_GRANULARITY))
    if len(rungs) > max_plans:
        # Thin geometrically: padding is a ratio, so log-spaced rungs bound it evenly.
        steps = max_plans - 1
        rungs = {max_rows}
        if steps > 0:
            ratio = max_rows ** (1.0 / steps)
            rungs.update(round(ratio**step) for step in range(steps))
    return tuple(sorted(rungs))


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
    """Shape checks that need no cache, shared by the constructor and the sizing query."""
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
    """Float workspace bytes and each bucket's integer region, asked of the planner
    without building a wrapper (which would allocate its own workspace)."""
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
            # The planner lays out one entry per route element; the page size
            # only addresses them, so the sizes do not move with it.
            kv_cache_page_size=_SIZING_PAGE_SIZE,
            backend=backend,
        )
        float_bytes = max(float_bytes, int(rows_float))
        sizes.append(round_up(int(rows_int), WORKSPACE_ALIGNMENT))
    return round_up(float_bytes, WORKSPACE_ALIGNMENT), tuple(sizes)


class _Persistent(NamedTuple):
    """Bytes read back by later calls -- the plan arena and the row pointers -- so
    never in memory another consumer reuses. A call rewrites its bucket's row
    pointers ahead of the plan that reads them, to give padding rows no entries."""

    arena: Tuple[int, int]
    indptr: Tuple[Tuple[int, int], ...]
    total: int


class _Transient(NamedTuple):
    """Bytes a call rewrites before it reads; may be the caller's shared scratch."""

    float_workspace: Tuple[int, int]
    padded_q: Tuple[int, int]
    padded_out: Tuple[int, int]
    route: Tuple[int, int]
    mask: Tuple[int, int]
    total: int


def _persistent_layout(*, buckets, plan_bytes) -> _Persistent:
    """Cut the buffer this object keeps for as long as it exists."""
    take, total = walk()
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
    take, total = walk()
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

    The route (logical token indices, ``-1`` where a position has no token) is
    mapped through the block table into physical slots and a mask, attended
    with the plan already built for its row bucket, and gated by a kernel.
    Given ``out``, :meth:`run` allocates nothing: the plans and their row
    pointers live in the persistent buffer, the scratch in the transient one
    from :meth:`bind_transient_workspace`. Every ``row_buckets`` plan is built
    up front, so a batch that changes size never plans under a capture.

    Parameters
    ----------
    persistent : torch.Tensor
        The ``uint8`` buffer the plans and their row pointers live in, at least
        the persistent size :meth:`workspace_bytes` reports. Kept for the life
        of this object.
    max_rows : int
        The widest batch the caller may send. The plan ladder is derived
        from it; see :func:`row_buckets`.
    num_qo_heads, num_kv_heads, head_dim : int
        The attention's shape.
    route_width : int
        Columns the route carries. Indices only -- a route with a trailing count
        column is a different width and is refused.
    q_data_type, kv_data_type, o_data_type : torch.dtype
        The query's, the cache's and the output's dtypes. An NVFP4 cache is
        ``uint8`` here and its scales arrive separately.
    kv_cache_format : str
        ``dense``, ``fp8_e4m3`` or ``nvfp4``: what the cache's bytes mean. The
        dtype alone is never read as a format.
    kv_layout : str
        ``NHD`` or ``HND``, as the cache is laid out.
    backend : str
        Which block-sparse backend to plan for. A paged route is served by
        ``fa2`` alone, so ``auto`` resolves there.
    max_plans : int
        Bounds how many rungs the plan ladder may have.
    """

    @flashinfer_api
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
        """See :class:`QSAAttention`."""
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
        check_buffer(persistent, "the persistent workspace")

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
        # Set by plan_cache: slots and page size are the cache's, which does not exist yet.
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

        self._arena = cut(persistent, regions.arena, torch.uint8, None)
        self._indptr = {}
        for rows, span in zip(buckets, regions.indptr, strict=True):
            view = cut(persistent, span, torch.int32, None)
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
        self._staging: Optional[torch.Tensor] = None
        self._frozen = False

    @flashinfer_api
    def bind_transient_workspace(self, transient: torch.Tensor) -> None:
        """Take the scratch a call rewrites before it reads.

        It may be the memory every other consumer of the step reuses, so
        nothing read back later -- a plan -- may live here. Bind it once the
        caller's scratch has stopped moving.

        Parameters
        ----------
        transient : torch.Tensor
            ``uint8``, at least the transient size :meth:`workspace_bytes`
            reports, aligned like ``persistent``.
        """
        check_buffer(transient, "the transient workspace")
        regions = self._transient_regions()
        if transient.numel() < regions.total:
            raise ValueError(
                f"this geometry needs {regions.total} transient bytes, got "
                f"{transient.numel()}"
            )
        buckets = self.row_buckets
        widest = buckets[-1]
        self._transient = transient
        self._float_workspace = cut(
            transient, regions.float_workspace, torch.uint8, None
        )
        self._padded_q = cut(
            transient,
            regions.padded_q,
            self.q_data_type,
            (widest, self.num_qo_heads, self.head_dim),
        )
        self._padded_out = cut(
            transient,
            regions.padded_out,
            self.o_data_type,
            (widest, self.num_qo_heads, self.head_dim),
        )
        self._route_base = cut(
            transient, regions.route, torch.int32, (widest, self.route_width)
        )
        self._mask_base = cut(transient, regions.mask, torch.uint8, None)
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
    @flashinfer_api
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
        """Persistent and transient bytes for this geometry, allocating nothing.

        Persistent bytes are read back on later calls and need memory nobody
        else writes; transient bytes are rewritten before they are read. The
        cache's ``num_slots`` and ``page_size`` change the plan, not the room.

        Parameters
        ----------
        device : torch.device
            The device the workspaces will be on.
        max_rows, num_qo_heads, num_kv_heads, head_dim, route_width : int
            As for :class:`QSAAttention`.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            As for :class:`QSAAttention`.
        kv_cache_format, kv_layout, backend : str
            As for :class:`QSAAttention`.
        max_plans : int
            As for :class:`QSAAttention`.

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

    @flashinfer_api
    def plan_cache(self, num_slots: int, page_size: int) -> None:
        """Build every bucket's plan for a cache of this many slots.

        Separate from construction because the cache does not exist when the
        workspace is sized. Every plan is built before any is published, so a
        failure leaves the attention unplanned. Planning for the cache it
        already has is a no-op, so every layer of a rank may call it; a
        different cache after a run is refused, because a captured graph
        replays the plans' byte offsets into the arena.

        Parameters
        ----------
        num_slots : int
            KV entries the cache holds, a whole number of pages.
        page_size : int
            KV entries per page.
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
        ranges = []
        offset = 0
        for rows, size in zip(self.row_buckets, self._plan_bytes, strict=True):
            ranges.append((rows, offset, offset + size))
            offset += size

        # One pinned staging buffer for every plan: the planner is done with it
        # before it returns, and the plans are built one after another.
        if self._staging is None:
            self._staging = torch.empty(
                (max(self._plan_bytes),),
                dtype=torch.uint8,
                pin_memory=True,
                device="cpu",
            )

        # The planner reads the route and the mask out of the caller's dirty
        # scratch. The schedule comes from the row pointers alone, so an
        # all-zero route and an empty mask plan the same thing as any step.
        self._route_base.zero_()
        self._mask_base.zero_()

        # Nothing is published until every plan is built: a failure half way
        # through would otherwise leave an object that looks planned and is not.
        wrappers = {}
        for rows, start, end in ranges:
            # The planner splits by these lengths: every row gets the full width.
            torch.arange(
                0,
                (rows + 1) * self.route_width,
                self.route_width,
                out=self._indptr[rows],
            )
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
        self.page_size = page_size
        self.pages = num_slots // page_size
        # The previous plans lived in the arena these just overwrote.
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
        """Shape, dtype, device and layout, before anything is launched. ``packed``
        planes need only unit stride innermost: a packed NVFP4 cache keeps data
        and block scales strided in one allocation, and the kernel reads strides."""
        check_tensor(
            tensor,
            name,
            shape=shape,
            dtype=dtype,
            device=self.device,
            contiguous=packed,
            innermost=not packed,
        )

    def _checked_scale(self, value, name):
        """A real, finite, positive host number: a device tensor would synchronise
        under capture, and a bool, NaN, zero or negative reaches the kernel as a multiplier."""
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
        token_to_request,
        output_gate,
        k_sf,
        v_sf,
        k_scale,
        v_scale,
    ):
        """Everything the kernels assume, checked before the first of them launches:
        a short plane or a missing scale is read past the end or multiplied by one."""
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
        if token_to_request.ndim != 1:
            raise ValueError(
                f"token_to_request is one axis, got {token_to_request.ndim}"
            )
        if token_to_request.size(0) < rows:
            raise ValueError(
                f"token_to_request must cover {rows} rows, got {token_to_request.size(0)}"
            )
        self._check_tensor(
            token_to_request,
            "token_to_request",
            (token_to_request.size(0),),
            torch.int32,
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
        if output_gate.ndim not in (2, 3):
            raise ValueError(
                "output_gate is [rows, heads, dim] or [rows, heads * dim], got "
                f"{output_gate.ndim} axes"
            )
        gate_shape = (rows, self.num_qo_heads, self.head_dim)
        if output_gate.ndim == 2:
            gate_shape = (rows, self.num_qo_heads * self.head_dim)
        check_tensor(
            output_gate,
            "output_gate",
            shape=gate_shape,
            dtype=self.o_data_type,
            device=self.device,
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

    @flashinfer_api(trace=qsa_attention_run_trace)
    def run(
        self,
        q: torch.Tensor,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        *,
        route: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
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
            position has no token. Indices only. A row that names no token its
            request's pages map has nothing to attend to, and its output is
            undefined.
        block_table : torch.Tensor
            Logical page to physical page per request.
        token_to_request : torch.Tensor
            Request each row belongs to.
        k_sf, v_sf : Optional[torch.Tensor]
            The NVFP4 block-scale planes, when the cache is packed.
        k_scale, v_scale : Optional[float]
            The cache's global scales, as **host floats**. A tensor would be
            read back from the device on every call, which a graph capture
            refuses.
        output_gate : torch.Tensor
            The gate to fold in, ``[rows, num_qo_heads * head_dim]`` or
            ``[rows, num_qo_heads, head_dim]``. Left unchanged. Required: the
            model this serves gates its attention output; ungated sparse
            attention is :class:`BlockSparseAttentionWrapper`.
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
            token_to_request,
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

        # Rows past the batch are padding: fully masked, with row pointers that
        # give them no entries.
        qsa_route_from_logical(
            route,
            token_to_request,
            block_table,
            self._route[bucket],
            self._mask[bucket],
            rows,
            self.page_size,
            self.num_slots,
            out_indptr=self._indptr[bucket],
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
