"""Caller-owned memory for the top-k kernels.

Every kernel writes a few words of status per row; the streaming kernel's slab merge also
publishes candidates to a global slab and counts arrivals per row; inputs whose rows are not
16-byte aligned are copied once into a padded arena.  By default the entry points keep these
buffers in per-(device, stream, shape) caches, so a call allocates nothing after its first
launch and two streams never share a buffer.  A caller who wants to own the memory instead (an
engine that preallocates everything, or bounds the library's footprint) sizes one byte tensor
with :func:`workspace_bytes` and passes it as ``workspace``; the entry point carves the buffers
out of it in the layout below and allocates nothing.

Layout: ``[status | counters | slab | order | arena]``, each region starting on a 256-byte
boundary (``order``: the row permutation of the streaming kernel's prepare launch, present for
configurations with ``row_order``).
The arrival counters must be zero when a launch starts.  The cached counters are zeroed once
and every launch leaves them zero (the last arriver resets its row); a caller-owned workspace
may hold anything, so the entry point zeroes the counter region before each launch (one small
memset on the stream, only for slab-merge configurations).

A workspace is bound to one stream at a time: launches on one stream are ordered, so one
workspace per stream is safe; two streams sharing one workspace race on the counters and slab
(the same is true of every buffer a kernel writes, including the outputs).  A CUDA graph
captures the workspace it was given, so replays of one graph must not overlap either.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "WorkspaceLayout",
    "Workspace",
    "workspace_layout",
    "workspace_bytes",
    "carve",
]

REGION_ALIGN = 256  # bytes; every region starts on a boundary at least as strict as the kernels' 16-byte loads
WORD = 4  # int32 status, counter and slab words


def _round(nbytes: int) -> int:
    return -(-nbytes // REGION_ALIGN) * REGION_ALIGN


@dataclass(frozen=True)
class WorkspaceLayout:
    """Region sizes (in int32 words, arena in bytes) and their byte offsets in the workspace."""

    status_words: int
    counter_words: int  # rows for a slab-merge configuration, 0 otherwise
    slab_words: (
        int  # rows x slab_words_per_row for a slab-merge configuration, 0 otherwise
    )
    arena_bytes: int  # the padded copy of a misaligned input, 0 when read in place
    order_words: int = 0  # rows for a streaming configuration with row_order (the prepare launch's permutation), 0 otherwise

    @property
    def status_offset(self) -> int:
        return 0

    @property
    def counter_offset(self) -> int:
        return _round(self.status_words * WORD)

    @property
    def slab_offset(self) -> int:
        return self.counter_offset + _round(self.counter_words * WORD)

    @property
    def order_offset(self) -> int:
        return self.slab_offset + _round(self.slab_words * WORD)

    @property
    def arena_offset(self) -> int:
        return self.order_offset + _round(self.order_words * WORD)

    @property
    def total_bytes(self) -> int:
        return self.arena_offset + _round(self.arena_bytes)


@dataclass(frozen=True)
class Workspace:
    """The int32 views carved out of a caller's workspace, and the byte arena (or None)."""

    status: torch.Tensor
    counters: torch.Tensor | None
    slab: torch.Tensor | None
    arena: torch.Tensor | None
    order: torch.Tensor | None = None


def workspace_layout(kind: str, config, rows: int, arena_bytes: int) -> WorkspaceLayout:
    """The layout for ``rows`` rows under kernel ``kind`` ("register", "register_cluster",
    "streaming") with its configuration; ``arena_bytes`` from :func:`..kernels.layout.arena_bytes`."""
    from ..kernels import census_split, register_cluster, register_resident, streaming

    if kind == "register":
        return WorkspaceLayout(rows * register_resident.STATUS_WORDS, 0, 0, arena_bytes)
    if kind == "register_cluster":
        return WorkspaceLayout(rows * register_cluster.STATUS_WORDS, 0, 0, arena_bytes)
    if kind == "census_split":
        assert isinstance(config, census_split.CensusSplitConfig)
        return WorkspaceLayout(
            rows * census_split.STATUS_WORDS,
            2 * rows,
            rows * census_split.slab_words_per_row(config.splits, config.tie_slab),
            arena_bytes,
        )
    assert kind == "streaming" and isinstance(config, streaming.StreamingConfig), kind
    counters = slab = 0
    if config.merge == "slab" and config.splits > 1:
        counters = rows
        slab = rows * streaming.slab_words_per_row(config.splits, config.stage)
    return WorkspaceLayout(
        rows * streaming.STATUS_WORDS,
        counters,
        slab,
        arena_bytes,
        rows if config.row_order else 0,
    )


def workspace_bytes(x: torch.Tensor, k: int) -> int:
    """Bytes of workspace :func:`..dispatch.router.topk` needs for ``x`` (rows, N; its layout
    counts) at this k on this device.  Zero rows need none.  The size depends on the row count
    through the router's choice, so a caller running several batch sizes takes the maximum over
    them (the query is host-only and cheap)."""
    from ...dispatch.device import device_facts

    from ..kernels.layout import arena_bytes, check_layout
    from .router import choose

    check_layout(x)
    rows, n = x.shape
    if rows == 0:
        return 0
    kind, config = choose(device_facts(x.device), x.dtype, k, n, rows)
    return workspace_layout(kind, config, rows, arena_bytes(x)).total_bytes


def carve(
    workspace: torch.Tensor, layout: WorkspaceLayout, device: torch.device
) -> Workspace:
    """Views into ``workspace`` for ``layout``.  The workspace must be a contiguous CUDA tensor on
    ``device`` of at least ``layout.total_bytes`` bytes with a 256-byte-aligned base (a fresh
    allocation, or a slice of one at a multiple of 256 bytes); its dtype is immaterial."""
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace must be a torch.Tensor")
    if not workspace.is_cuda or (
        device.index is not None and workspace.device.index != device.index
    ):
        raise ValueError(f"workspace must live on {device}, got {workspace.device}")
    if not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous")
    nbytes = workspace.numel() * workspace.element_size()
    if nbytes < layout.total_bytes:
        raise ValueError(
            f"workspace holds {nbytes} bytes, {layout.total_bytes} needed (workspace_bytes())"
        )
    if workspace.data_ptr() % REGION_ALIGN:
        raise ValueError(f"workspace base must be {REGION_ALIGN}-byte aligned")
    raw = workspace.view(-1).view(torch.uint8)

    def words(offset: int, count: int) -> torch.Tensor | None:
        if count == 0:
            return None
        return raw[offset : offset + count * WORD].view(torch.int32)

    status = words(layout.status_offset, layout.status_words)
    assert status is not None
    arena = None
    if layout.arena_bytes:
        arena = raw[layout.arena_offset : layout.arena_offset + layout.arena_bytes]
    return Workspace(
        status,
        words(layout.counter_offset, layout.counter_words),
        words(layout.slab_offset, layout.slab_words),
        arena,
        words(layout.order_offset, layout.order_words),
    )
