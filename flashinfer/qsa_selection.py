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
from typing import Optional

import torch

from .sparse_route import expand_block_route
from .sparse_scores import sparse_paged_scores
from .topk import (
    WORKSPACE_ALIGNMENT,
    PreparedTopKRaggedTransform,
    TopKTieBreak,
)


#: What each dtype a region is cut in costs per element. A constant, so the
#: cutting never has to make a tensor to ask.
_ITEM_BYTES = {torch.float32: 4, torch.int32: 4}

#: The shapes the scorer has kernels for.
_SCORER_HEAD_DIMS = (64, 128, 192, 256)
_MAX_SCORER_HEADS = 16

#: What a query and the cache it is scored against may be.
_SCORE_DTYPES = (torch.float16, torch.bfloat16)


#: What ``query_positions`` may arrive in. A position is bounded by the
#: context length, so both hold one; the kernels narrow and validate.
_POSITION_DTYPES = (torch.int32, torch.int64)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def selection_columns(max_model_len: int, compress_ratio: int, capacity: int) -> int:
    """How many compressed columns a row can ever score against.

    The context bounds it; what the cache can address bounds it again. Rounded
    up to 64 so the score rows stay cooperative with the top-k's alignment.
    """
    columns = -(-max_model_len // compress_ratio)
    return min(max(64, -(-columns // 64) * 64), capacity)


def selection_route_width(token_topk: int, compress_ratio: int) -> int:
    """Columns the expansion writes: every selected block, plus the tail."""
    return (token_topk // compress_ratio) * compress_ratio + compress_ratio - 1


class QSASelection:
    r"""Score a compressed cache, take the top blocks, and expand them to a route.

    The three steps a block-granular selector runs -- score, select, expand --
    with the scratch between them owned by the caller. Nothing is allocated
    inside :meth:`run`, so the whole selection can be captured into a CUDA graph
    without leaving buffers in the graph's private pool, and two selections can
    run at once by handing each its own slice of an arena.

    The score buffer is the large one, and it is what bounds a chunk: a batch
    wider or taller than the budget is scored a few rows at a time, with the
    top-k taken per chunk, and the expansion run once over the whole batch at
    the end.

    Parameters
    ----------
    max_rows : int
        The most query rows a call will bring.
    max_columns : int
        The most compressed columns any row will score against. A caller sizes
        this from its context length: ``ceil(max_seq_len / compress_ratio)``,
        rounded up, and clamped to what the page table can address.
    compress_ratio : int
        Tokens per compressed block.
    token_topk : int
        Tokens the selection keeps per query. It has to be a whole number of
        blocks: ``token_topk % compress_ratio == 0``.
    num_heads, head_dim : int
        The query's shape.
    device : torch.device
        Where everything lives.
    score_budget_bytes : int
        How much of the workspace the scores may take. It decides the chunk, and
        with it how many passes a batch takes. A budget below one row's width is
        rounded up to that row: something has to fit.
    deterministic : bool
        Whether the selection has to repeat its output exactly. Defaults to
        ``True``: a selection that reorders between replays of the same graph is
        a selection whose attention output moves for no reason.
    tie_break : int
        Which index wins when two blocks score the same. Defaults to preferring
        the smaller index, which is the earlier block.
    dsa_graph_safe : bool
        Whether the top-k has to be one that runs under graph capture. Defaults
        to ``True``.
    """

    def __init__(
        self,
        *,
        max_rows: int,
        max_columns: int,
        compress_ratio: int,
        token_topk: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        score_budget_bytes: int = 256 * 1024 * 1024,
        deterministic: bool = True,
        tie_break: int = TopKTieBreak.SMALL,
        dsa_graph_safe: bool = True,
    ) -> None:
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        if token_topk < 1:
            raise ValueError(f"token_topk must be positive, got {token_topk}")
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim have to be positive")
        # What the scorer has an instantiation for. Its mma tile fixes both: a
        # head dimension it was built with, and one n-tile of query heads. A
        # shape outside this is one the library cannot build, not one it is
        # merely slower at, so it is refused while planning.
        if num_heads > _MAX_SCORER_HEADS:
            raise ValueError(
                f"the scorer serves at most {_MAX_SCORER_HEADS} query heads, "
                f"got {num_heads}"
            )
        if head_dim not in _SCORER_HEAD_DIMS:
            raise ValueError(
                f"the scorer serves head dimensions {sorted(_SCORER_HEAD_DIMS)}, "
                f"got {head_dim}"
            )
        if score_budget_bytes < 1:
            raise ValueError(
                f"score_budget_bytes must be positive, got {score_budget_bytes}"
            )
        if token_topk % compress_ratio:
            raise ValueError(
                "token_topk has to be a whole number of blocks, got "
                f"{token_topk} for a ratio of {compress_ratio}"
            )
        if max_rows < 1 or max_columns < 1:
            raise ValueError("max_rows and max_columns have to be positive")

        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self.max_rows = max_rows
        self.max_columns = max_columns
        self.compress_ratio = compress_ratio
        self.token_topk = token_topk
        self.block_topk = token_topk // compress_ratio
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        # The route the expansion writes: every selected block's tokens, plus
        # the tail of the block the query sits in.
        self.route_width = self.block_topk * compress_ratio + compress_ratio - 1
        self._bound_workspace: Optional[torch.Tensor] = None

        # Rows one pass covers. Never zero: a width past the whole budget still
        # has to score its row, and the reservation below has to hold it.
        per_row = max_columns * 4
        self.rows_per_chunk = max(1, min(max_rows, score_budget_bytes // per_row))

        self._topk = PreparedTopKRaggedTransform(
            num_rows=self.rows_per_chunk,
            max_len=max_columns,
            k=self.block_topk,
            dtype=torch.float32,
            device=device,
            deterministic=deterministic,
            tie_break=tie_break,
            dsa_graph_safe=dsa_graph_safe,
        )
        self.backend = self._topk.backend

        # The workspace, as offsets fixed here so run() only takes views.
        scores_bytes = _round_up(
            self.rows_per_chunk * max_columns * 4, WORKSPACE_ALIGNMENT
        )
        visible_bytes = _round_up(self.rows_per_chunk * 4, WORKSPACE_ALIGNMENT)
        # The top-k writes a full chunk's worth of rows every pass, so the
        # block buffer is padded to whole chunks; the rows past the batch are
        # written and never read.
        self.padded_rows = _round_up(max_rows, self.rows_per_chunk)
        blocks_bytes = _round_up(
            self.padded_rows * self.block_topk * 4, WORKSPACE_ALIGNMENT
        )
        offsets_bytes = _round_up(self.rows_per_chunk * 4, WORKSPACE_ALIGNMENT)
        self._scores_at = 0
        self._visible_at = scores_bytes
        self._blocks_at = self._visible_at + visible_bytes
        self._offsets_at = self._blocks_at + blocks_bytes
        self._topk_at = self._offsets_at + offsets_bytes
        self._topk_bytes = self._measure_topk_workspace()
        self._total_bytes = self._topk_at + _round_up(
            self._topk_bytes, WORKSPACE_ALIGNMENT
        )

    def _measure_topk_workspace(self) -> int:
        """Ask the top-k how much scratch it needs, once, while planning.

        Both backends answer from the shape they were prepared for, so nothing
        is allocated here at all -- which is the point: a selection that exists
        to keep its memory in a caller's arena should not reserve a score buffer
        just to say how big the arena has to be.
        """
        return int(self._topk.workspace_size())

    def bind_workspace(self, workspace: torch.Tensor) -> None:
        """Keep the scratch this selection runs out of.

        The caller reserves it -- it is the caller's memory budget -- and hands
        it over once, here, rather than on every call. ``run`` still takes one
        per call for a caller that would rather pass it.
        """
        if workspace.dtype != torch.uint8:
            raise ValueError(f"the workspace must be uint8, got {workspace.dtype}")
        if workspace.device != self.device:
            raise ValueError(
                f"the workspace must be on {self.device}, got {workspace.device}"
            )
        needed = self.workspace_size()
        if workspace.numel() < needed:
            raise ValueError(
                f"the workspace holds {workspace.numel()} bytes and this "
                f"selection needs {needed}"
            )
        self._bound_workspace = workspace[:needed]

    @staticmethod
    def plan_workspace_size(
        *,
        device: torch.device,
        max_rows: int,
        max_columns: int,
        compress_ratio: int,
        token_topk: int,
        num_heads: int,
        head_dim: int,
    ) -> int:
        """Bytes :meth:`run` will need, without allocating any of them.

        A caller reserves this before the workspace it draws on is locked, and
        before there is a cache to run against. Preparing the object is itself
        allocation-free -- every offset is arithmetic on the geometry -- so the
        answer is the object's own, asked and dropped.
        """
        return QSASelection(
            max_rows=max_rows,
            max_columns=max_columns,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        ).workspace_size()

    def workspace_size(self) -> int:
        """Bytes :meth:`run` needs, for the shape this was prepared for."""
        return self._total_bytes

    def _views(self, workspace: torch.Tensor, rows: int, columns: int):
        def region(offset: int, count: int, dtype: torch.dtype):
            # The element size is a constant of the dtype, taken at plan time:
            # making a tensor to ask would be an allocation per call.
            itemsize = _ITEM_BYTES[dtype]
            flat = workspace[offset : offset + count * itemsize]
            return flat.view(dtype)

        scores = region(self._scores_at, self.rows_per_chunk * columns, torch.float32)
        visible = region(self._visible_at, self.rows_per_chunk, torch.int32)
        blocks = region(
            self._blocks_at, self.padded_rows * self.block_topk, torch.int32
        )
        offsets = region(self._offsets_at, self.rows_per_chunk, torch.int32)
        topk = workspace[self._topk_at : self._topk_at + self._topk_bytes]
        return (
            scores.view(self.rows_per_chunk, columns),
            visible,
            blocks.view(self.padded_rows, self.block_topk),
            offsets,
            topk,
        )

    def run(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        page_table: torch.Tensor,
        token_to_req: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        out_route: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select for one batch, writing the route into ``out_route``.

        Parameters
        ----------
        q : torch.Tensor
            Queries, ``[rows, num_heads, head_dim]``.
        k_compressed : torch.Tensor
            The compressed key cache, ``[pages, page_size, head_dim]``.
        page_table : torch.Tensor
            Logical page to physical page per request, int32.
        token_to_req : torch.Tensor
            Request each row belongs to, ``[rows]``, int32.
        query_positions : torch.Tensor
            Position of each query inside its request, ``[rows]``, int32.
        sequence_lengths : torch.Tensor
            KV length per request, int32.
        out_route : torch.Tensor
            Receives the logical token route, ``[rows, route_width]``, int32,
            ``-1`` where a position has no token. **No trailing count column**:
            the route is indices and nothing else.
        workspace : torch.Tensor
            :meth:`workspace_size` bytes of uint8 on this selection's device,
            contiguous and aligned. It is this call's alone for its duration.

        Returns
        -------
        torch.Tensor
            ``out_route``.
        """
        if workspace is None:
            workspace = self._bound_workspace
            if workspace is None:
                raise ValueError(
                    "this selection has no workspace: pass one to run() or "
                    "hand one over once with bind_workspace()"
                )
        rows = q.size(0)
        # The width is the plan's: the top-k below was prepared for exactly this
        # many columns, and a narrower call would have to prepare its own.
        columns = self.max_columns
        if rows > self.max_rows:
            raise ValueError(
                f"this selection was prepared for {self.max_rows} rows, got {rows}"
            )
        if tuple(q.shape[1:]) != (self.num_heads, self.head_dim):
            raise ValueError(
                f"q must be [rows, {self.num_heads}, {self.head_dim}], got "
                f"{tuple(q.shape)}"
            )
        if q.dtype not in _SCORE_DTYPES:
            raise ValueError(f"q must be one of {_SCORE_DTYPES}, got {q.dtype}")
        if q.dtype != k_compressed.dtype:
            raise ValueError(
                "q and the compressed cache share one dtype, got "
                f"{q.dtype} and {k_compressed.dtype}"
            )
        if k_compressed.ndim != 3 or k_compressed.size(2) != self.head_dim:
            raise ValueError(
                "k_compressed must be [pages, page_size, head_dim] with "
                f"head_dim {self.head_dim}, got {tuple(k_compressed.shape)}"
            )
        for name, tensor, shape in (
            ("token_to_req", token_to_req, (rows,)),
            ("query_positions", query_positions, (rows,)),
        ):
            if tensor.shape[0] < shape[0]:
                raise ValueError(
                    f"{name} must cover {shape[0]} rows, got {tensor.shape[0]}"
                )
        if token_to_req.dtype != page_table.dtype:
            raise ValueError(
                "token_to_req must carry the block table's dtype "
                f"{page_table.dtype}, got {token_to_req.dtype}"
            )
        # Positions are not an index into the cache -- a position is a number
        # bounded by the context length -- so they carry their own type. A
        # caller that builds them beside a slot mapping keeps them in int64,
        # and converting on the way in would allocate once per call. The
        # kernels read them in whichever of the two they arrive in and narrow
        # inside, where a value that does not fit becomes a masked-off row
        # rather than a wrapped one.
        if query_positions.dtype not in _POSITION_DTYPES:
            raise ValueError(
                f"query_positions must be one of {_POSITION_DTYPES}, got "
                f"{query_positions.dtype}"
            )
        if sequence_lengths.dtype != page_table.dtype:
            raise ValueError(
                "sequence_lengths must carry the block table's dtype "
                f"{page_table.dtype}, got {sequence_lengths.dtype}"
            )
        # The visible-block counts this cuts out of the workspace are int32, and
        # the scorer requires them to match the block table, so the whole index
        # side is int32 here rather than failing inside the kernel on a dtype it
        # cannot report usefully.
        if page_table.dtype != torch.int32:
            raise ValueError(f"page_table must be int32, got {page_table.dtype}")
        if page_table.ndim != 2:
            raise ValueError(
                f"page_table must be [requests, pages], got {tuple(page_table.shape)}"
            )
        for name, tensor in (
            ("token_to_req", token_to_req),
            ("query_positions", query_positions),
            ("sequence_lengths", sequence_lengths),
        ):
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be one axis, got {tensor.ndim}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if not page_table.is_contiguous():
            raise ValueError("page_table must be contiguous")
        if not out_route.is_contiguous():
            raise ValueError("out_route must be contiguous")
        for name, tensor in (
            ("q", q),
            ("k_compressed", k_compressed),
            ("page_table", page_table),
            ("token_to_req", token_to_req),
            ("query_positions", query_positions),
            ("sequence_lengths", sequence_lengths),
            ("out_route", out_route),
        ):
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} must be on {self.device}, got {tensor.device}"
                )
        if tuple(out_route.shape) != (rows, self.route_width):
            raise ValueError(
                f"out_route must be [{rows}, {self.route_width}], got "
                f"{tuple(out_route.shape)}"
            )
        if out_route.dtype != torch.int32:
            raise ValueError(f"out_route must be int32, got {out_route.dtype}")
        if workspace.dtype != torch.uint8 or not workspace.is_contiguous():
            raise ValueError("workspace must be contiguous uint8")
        if workspace.device != self.device:
            raise ValueError(
                f"workspace must be on {self.device}, got {workspace.device}"
            )
        if workspace.numel() < self._total_bytes:
            raise ValueError(
                f"workspace needs {self._total_bytes} bytes, got {workspace.numel()}"
            )
        if workspace.data_ptr() % WORKSPACE_ALIGNMENT:
            raise ValueError(f"workspace must be {WORKSPACE_ALIGNMENT}-byte aligned")
        if rows == 0:
            return out_route

        scores, visible, blocks, offsets, topk_workspace = self._views(
            workspace, rows, columns
        )
        offsets.zero_()
        # The radix top-k reads its counters before it writes them on the first
        # round. Rather than make that the caller's problem -- a contract they
        # would meet by accident with torch.zeros and break with torch.empty --
        # the region is cleared here. It is a memset of a megabyte inside a
        # kernel launch, and it is graph-safe.
        topk_workspace.zero_()

        for start in range(0, rows, self.rows_per_chunk):
            end = min(start + self.rows_per_chunk, rows)
            chunk = end - start
            chunk_scores = scores[:chunk]
            chunk_visible = visible[:chunk]
            sparse_paged_scores(
                q[start:end],
                k_compressed,
                page_table,
                token_to_req[start:end],
                query_positions[start:end],
                sequence_lengths,
                self.compress_ratio,
                math.sqrt(self.head_dim),
                num_columns=columns,
                logits=chunk_scores,
                visible_blocks=chunk_visible,
            )
            if chunk < self.rows_per_chunk:
                # A short tail still runs a full chunk: the rows past the batch
                # score nothing, so their selection is empty and unread.
                chunk_visible = visible[chunk:]
                chunk_visible.zero_()
            self._topk.run(
                scores,
                offsets,
                visible,
                out=blocks[start : start + self.rows_per_chunk],
                workspace=topk_workspace,
            )

        expand_block_route(
            blocks[:rows],
            query_positions[:rows],
            sequence_lengths,
            token_to_req[:rows],
            self.compress_ratio,
            out=out_route,
        )
        return out_route
