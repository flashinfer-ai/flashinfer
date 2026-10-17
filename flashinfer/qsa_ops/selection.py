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

from ..api_logging import flashinfer_api
from ..trace.templates.qsa import qsa_selection_run_trace
from .route import qsa_expand_block_route
from .scores import qsa_paged_scores
from ._workspace import check_buffer, check_tensor, cut
from ..topk import (
    WORKSPACE_ALIGNMENT,
    TopKTieBreak,
    resolve_ragged_transform_backend,
    top_k_ragged_transform,
    _ragged_transform_workspace_size,
)
from ..utils import round_up


#: What each dtype a region is cut in costs per element. A constant, so the
#: cutting never has to make a tensor to ask.

#: The shapes the scorer has kernels for.
_SCORER_HEAD_DIMS = (64, 128, 192, 256)
_MAX_SCORER_HEADS = 16

#: What a query and the cache it is scored against may be.
_SCORE_DTYPES = (torch.float16, torch.bfloat16)


#: What ``query_positions`` may arrive in. A position is bounded by the
#: context length, so both hold one; the kernels narrow and validate.
_POSITION_DTYPES = (torch.int32, torch.int64)


def selection_columns(max_model_len: int, compress_ratio: int, capacity: int) -> int:
    """Compressed columns a row can score against: bounded by the context and the
    block table, rounded up to 64 for the top-k's alignment."""
    columns = -(-max_model_len // compress_ratio)
    return min(max(64, -(-columns // 64) * 64), capacity)


def selection_route_width(token_topk: int, compress_ratio: int) -> int:
    """Columns the expansion writes: every selected block, plus the tail."""
    return (token_topk // compress_ratio) * compress_ratio + compress_ratio - 1


class QSASelection:
    r"""Score a compressed cache, take the top blocks, and expand them to a route.

    Score, select and expand, with the scratch between them owned by the
    caller: :meth:`run` allocates nothing, so it captures into a CUDA graph and
    two selections run at once from slices of one arena. A batch past the
    score budget is scored a chunk of rows at a time, the top-k taken per
    chunk, and expanded once at the end.

    Parameters
    ----------
    max_rows : int
        The most query rows a call will bring.
    max_columns : int
        The most compressed columns any row will score against. A caller sizes
        this from its context length: ``ceil(max_seq_len / compress_ratio)``,
        rounded up, and clamped to what the block table can address.
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

    @flashinfer_api
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
        """See :class:`QSASelection`."""
        if compress_ratio < 1:
            raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
        if token_topk < 1:
            raise ValueError(f"token_topk must be positive, got {token_topk}")
        if num_heads < 1 or head_dim < 1:
            raise ValueError("num_heads and head_dim have to be positive")
        # The scorer's mma tile fixes the head dimensions and the query-head
        # count it is instantiated for; anything else cannot be built.
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

        self.deterministic = deterministic
        self.tie_break = tie_break
        self.dsa_graph_safe = dsa_graph_safe
        self.backend = resolve_ragged_transform_backend(
            num_rows=self.rows_per_chunk,
            max_len=max_columns,
            k=self.block_topk,
            dtype=torch.float32,
            device=device,
            deterministic=deterministic,
            tie_break=tie_break,
            dsa_graph_safe=dsa_graph_safe,
            use_row_starts=False,
        )

        # The workspace, as offsets fixed here so run() only takes views.
        scores_bytes = round_up(
            self.rows_per_chunk * max_columns * 4, WORKSPACE_ALIGNMENT
        )
        visible_bytes = round_up(self.rows_per_chunk * 4, WORKSPACE_ALIGNMENT)
        # The top-k writes whole chunks, so the block buffer is padded to them.
        self.padded_rows = round_up(max_rows, self.rows_per_chunk)
        blocks_bytes = round_up(
            self.padded_rows * self.block_topk * 4, WORKSPACE_ALIGNMENT
        )
        offsets_bytes = round_up(self.rows_per_chunk * 4, WORKSPACE_ALIGNMENT)
        self._scores_at = 0
        self._visible_at = scores_bytes
        self._blocks_at = self._visible_at + visible_bytes
        self._offsets_at = self._blocks_at + blocks_bytes
        self._topk_at = self._offsets_at + offsets_bytes
        self._topk_bytes = self._measure_topk_workspace()
        self._total_bytes = self._topk_at + round_up(
            self._topk_bytes, WORKSPACE_ALIGNMENT
        )

    def _measure_topk_workspace(self) -> int:
        return _ragged_transform_workspace_size(
            self.rows_per_chunk,
            self.max_columns,
            self.block_topk,
            torch.float32,
            self.device,
            backend=self.backend,
            tie_break=self.tie_break,
        )

    @flashinfer_api
    def bind_workspace(self, workspace: torch.Tensor) -> None:
        """Keep the scratch this selection runs out of; ``run`` still takes one per call.

        Parameters
        ----------
        workspace : torch.Tensor
            ``uint8`` on this selection's device, at least
            :meth:`workspace_size` bytes.
        """
        check_buffer(workspace, "the workspace")
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
    @flashinfer_api
    def plan_workspace_size(
        *,
        device: torch.device,
        max_rows: int,
        max_columns: int,
        compress_ratio: int,
        token_topk: int,
        num_heads: int,
        head_dim: int,
        score_budget_bytes: int = 256 * 1024 * 1024,
        deterministic: bool = True,
        tie_break: int = TopKTieBreak.SMALL,
        dsa_graph_safe: bool = True,
    ) -> int:
        """Bytes :meth:`run` will need, without allocating any of them.

        Parameters
        ----------
        device : torch.device
            The device the workspace will be on.
        max_rows, max_columns, compress_ratio, token_topk, num_heads, head_dim : int
            As for :class:`QSASelection`.
        score_budget_bytes, deterministic, tie_break, dsa_graph_safe
            As for :class:`QSASelection`. These change the size, so a caller
            that passes them to the constructor has to pass them here too.
        """
        return QSASelection(
            max_rows=max_rows,
            max_columns=max_columns,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
            score_budget_bytes=score_budget_bytes,
            deterministic=deterministic,
            tie_break=tie_break,
            dsa_graph_safe=dsa_graph_safe,
        ).workspace_size()

    @flashinfer_api
    def workspace_size(self) -> int:
        """Bytes :meth:`run` needs, for the shape this was prepared for."""
        return self._total_bytes

    def _views(self, workspace: torch.Tensor, rows: int, columns: int):
        scores = cut(
            workspace,
            (self._scores_at, self.rows_per_chunk * columns * 4),
            torch.float32,
            None,
        )
        visible = cut(
            workspace, (self._visible_at, self.rows_per_chunk * 4), torch.int32, None
        )
        blocks = cut(
            workspace,
            (self._blocks_at, self.padded_rows * self.block_topk * 4),
            torch.int32,
            None,
        )
        offsets = cut(
            workspace, (self._offsets_at, self.rows_per_chunk * 4), torch.int32, None
        )
        topk = workspace[self._topk_at : self._topk_at + self._topk_bytes]
        return (
            scores.view(self.rows_per_chunk, columns),
            visible,
            blocks.view(self.padded_rows, self.block_topk),
            offsets,
            topk,
        )

    @flashinfer_api(trace=qsa_selection_run_trace)
    def run(
        self,
        q: torch.Tensor,
        k_compressed: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: torch.Tensor,
        query_positions: torch.Tensor,
        seq_lens: torch.Tensor,
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
        block_table : torch.Tensor
            Logical page to physical page per request, int32.
        token_to_request : torch.Tensor
            Request each row belongs to, ``[rows]``, int32.
        query_positions : torch.Tensor
            Position of each query inside its request, ``[rows]``, int32.
        seq_lens : torch.Tensor
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
        if q.dtype not in _SCORE_DTYPES:
            raise ValueError(f"q must be one of {_SCORE_DTYPES}, got {q.dtype}")
        check_tensor(
            q, "q", shape=(rows, self.num_heads, self.head_dim), device=self.device
        )
        if k_compressed.ndim != 3 or k_compressed.size(2) != self.head_dim:
            raise ValueError(
                "k_compressed must be [pages, page_size, head_dim] with "
                f"head_dim {self.head_dim}, got {tuple(k_compressed.shape)}"
            )
        check_tensor(k_compressed, "k_compressed", dtype=q.dtype, device=self.device)
        if block_table.ndim != 2:
            raise ValueError(
                f"block_table must be [requests, pages], got {tuple(block_table.shape)}"
            )
        # The visible-block counts are int32 and the scorer requires the block
        # table and the per-request arrays to match, so the index side is int32.
        # Positions may arrive as int64 beside a slot mapping; converting would
        # allocate per call, so the kernels take either and narrow inside, where
        # a value that does not fit becomes a masked-off row, not a wrapped one.
        if query_positions.dtype not in _POSITION_DTYPES:
            raise ValueError(
                f"query_positions must be one of {_POSITION_DTYPES}, got "
                f"{query_positions.dtype}"
            )
        for name, tensor, dtype in (
            ("block_table", block_table, torch.int32),
            ("token_to_request", token_to_request, torch.int32),
            ("query_positions", query_positions, query_positions.dtype),
            ("seq_lens", seq_lens, torch.int32),
        ):
            if name != "block_table" and tensor.ndim != 1:
                raise ValueError(f"{name} must be one axis, got {tensor.ndim}")
            check_tensor(tensor, name, dtype=dtype, device=self.device, contiguous=True)
        for name, tensor in (
            ("token_to_request", token_to_request),
            ("query_positions", query_positions),
        ):
            if tensor.shape[0] < rows:
                raise ValueError(
                    f"{name} must cover {rows} rows, got {tensor.shape[0]}"
                )
        check_tensor(
            out_route,
            "out_route",
            shape=(rows, self.route_width),
            dtype=torch.int32,
            device=self.device,
            contiguous=True,
        )
        check_buffer(workspace, "workspace")
        if workspace.device != self.device:
            raise ValueError(
                f"workspace must be on {self.device}, got {workspace.device}"
            )
        if workspace.numel() < self._total_bytes:
            raise ValueError(
                f"workspace needs {self._total_bytes} bytes, got {workspace.numel()}"
            )
        if rows == 0:
            return out_route

        scores, visible, blocks, offsets, topk_workspace = self._views(
            workspace, rows, columns
        )
        offsets.zero_()
        # The radix top-k reads its counters before it writes them on the first
        # round, so its region is cleared here rather than by the caller.
        topk_workspace.zero_()

        for start in range(0, rows, self.rows_per_chunk):
            end = min(start + self.rows_per_chunk, rows)
            chunk = end - start
            chunk_scores = scores[:chunk]
            chunk_visible = visible[:chunk]
            qsa_paged_scores(
                q[start:end],
                k_compressed,
                block_table,
                token_to_request[start:end],
                query_positions[start:end],
                seq_lens,
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
            top_k_ragged_transform(
                scores,
                offsets,
                visible,
                self.block_topk,
                self.deterministic,
                self.tie_break,
                self.dsa_graph_safe,
                out=blocks[start : start + self.rows_per_chunk],
                workspace=topk_workspace,
                backend=self.backend,
            )

        qsa_expand_block_route(
            blocks[:rows],
            query_positions[:rows],
            seq_lens,
            token_to_request[:rows],
            self.compress_ratio,
            out=out_route,
        )
        return out_route
