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

The per-layer facade of the Cake MiniMax-M3 ICP indexer: one object per rank,
built once from the serving limits, that runs the decode chain (scorer ->
Top16 select and publish into the owners' receive rows -> owner merge) and the
prefill / mixed chain (per window device plan; per query chunk scorer and
Top16 select into a carrier, or with a caller pool the fused list scorer and
list-to-carrier merge into the same carrier; the caller's one exchange per
layer; merge).

The caller owns every buffer that crosses ranks or outlives a call: the
receive storage and its counters (``comm``, laid out by
:func:`~.cake_backend.comm_layout`), the decode score rows, the prefill score
scratch or list pool, the carriers, and the outputs.  The facade owns only
the prefill plan storage of its query windows, on the layouts where its
decode runs the fused chain (:func:`~.cake_backend.fused_decode_min_batch`)
that chain's pool, and on the layouts with short prefill windows
(:func:`~.cake_backend.short_prefill_max_tokens`) their score scratch, all
allocated once here.  No method allocates,
synchronizes or reads a device value; every method is CUDA Graph safe once
:meth:`MSAIcpIndexerPlan.warmup` has loaded the programs.  ``warmup`` must run
on every rank before the first exchange (a kernel loaded lazily at its first
launch can stall behind a peer's spinning merge).
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Callable, Hashable, Optional, Sequence

import torch

from . import cake_backend as cb
from . import cake_jit as cj

# Bounds of one prefill query chunk (vLLM's chunking: the score scratch holds
# at most ``_SCRATCH_BYTES`` over five bytes per score element).
_CHUNK_MAX = 8192
_SLOTS = range(cb.NUM_SLOT_SETS)
_CHUNK_MIN = 128
_SCRATCH_BYTES = 256 * 1024 * 1024
_COMM_FIELDS = (
    "recv",
    "tagged_recv",
    "recv_bases",
    "tagged_recv_bases",
    "publish_generations",
    "expected_generations",
    "errors",
)


def _comm_print(comm: Any) -> tuple:
    """Per comm field: the tensor object, its base address, shape, strides
    and device, as ``check_comm`` validated them."""
    return tuple(
        (t, t.data_ptr(), t.shape, t.stride(), t.device)
        for t in (getattr(comm, field) for field in _COMM_FIELDS)
    )


def _same_comm(comm: Any, prints: tuple) -> bool:
    """Whether every comm field is still the validated tensor at the same
    address, shape, strides and device (host metadata only: no allocation,
    no synchronization, safe under CUDA Graph capture)."""
    for field, (t0, ptr, shape, stride, device) in zip(
        _COMM_FIELDS, prints, strict=True
    ):
        t = getattr(comm, field, None)
        if (
            t is not t0
            or t.data_ptr() != ptr
            or t.shape != shape
            or t.stride() != stride
            or t.device != device
        ):
            return False
    return True


def prefill_score_columns(max_seq_len: int, page_size: int) -> int:
    """Row pitch of the prefill score scratch: the scorer's columns rounded up
    to a multiple of 16."""
    return (cb.index_score_columns(max_seq_len, page_size) + 15) // 16 * 16


def prefill_window_tokens(max_num_batched_tokens: int) -> int:
    """Query tokens per prefill window of the fused path: the plan storage's
    capacity, ``min(8192, max_num_batched_tokens)``.  Its list pool does not
    grow with the history, so unlike :func:`prefill_chunk_tokens` the window
    does not shrink for long sequences."""
    if max_num_batched_tokens < 1:
        raise ValueError("max_num_batched_tokens must be positive")
    return min(_CHUNK_MAX, max_num_batched_tokens)


def prefill_chunk_tokens(max_seq_len: int, page_size: int, num_heads: int) -> int:
    """Query tokens per prefill chunk for a step whose longest sequence is
    ``max_seq_len``: ``min(8192, max(128, 256 MiB // (5 H cols) // 128 * 128))``."""
    cols = prefill_score_columns(max_seq_len, page_size)
    return min(
        _CHUNK_MAX,
        max(_CHUNK_MIN, _SCRATCH_BYTES // (num_heads * cols * 5) // 128 * 128),
    )


class MSAIcpIndexerPlan:
    """The ICP indexer of one rank; see :func:`prepare_msa_icp_indexer`."""

    def __init__(
        self,
        *,
        world_size: int,
        rank: int,
        num_index_heads: int,
        page_size: int,
        init_blocks: int,
        local_blocks: int,
        max_decode_tokens: int,
        max_decode_query_len: int,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        num_sparse_layers: int,
        device: torch.device,
    ) -> None:
        cb._world(world_size, rank)
        cb._heads(num_index_heads)
        if page_size not in cb.SUPPORTED_PAGE_SIZES:
            raise ValueError("page_size must be 128 or 256")
        cb._forced(init_blocks, local_blocks)
        if (
            max_decode_query_len < 1
            or num_index_heads * max_decode_query_len > cb.MAX_DECODE_ROWS
        ):
            raise ValueError(
                f"need max_decode_query_len >= 1 and num_index_heads * max_decode_query_len <= {cb.MAX_DECODE_ROWS}"
            )
        if (
            min(max_decode_tokens, max_num_batched_tokens, max_num_seqs, max_model_len)
            < 1
        ):
            raise ValueError("token, sequence and length limits must be positive")
        if num_sparse_layers < 1:
            raise ValueError("num_sparse_layers must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("device must be a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.arch = cb.cake_jit.device_arch(device.index)
        self.world_size = world_size
        self.rank = rank
        self.num_index_heads = num_index_heads
        self.heads_per_rank = num_index_heads // world_size
        self.page_size = page_size
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.max_decode_tokens = max_decode_tokens
        self.max_decode_query_len = max_decode_query_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.num_sparse_layers = num_sparse_layers
        # The decode chain rule: from ``fused_decode_min_batch`` requests on,
        # ``decode`` without a caller pool runs the fused chain in
        # ``_fused_pool`` (``None``: always the score plane).
        self.fused_decode_min_batch = cb.fused_decode_min_batch(
            self.arch, page_size, world_size
        )
        self._fused_pool = (
            torch.empty(
                cb.decode_pool_bytes(
                    max_decode_tokens=max_decode_tokens, num_heads=num_index_heads
                ),
                dtype=torch.uint8,
                device=device,
            )
            if self.fused_decode_min_batch is not None
            else None
        )
        # The prefill chain rule: whether the caller's default on this layout
        # is the fused path (list pool of ``prefill_pool_shape()``, windows of
        # ``prefill_window_tokens()``); ``prefill_local`` runs the fused path
        # exactly when it is given a pool.
        self.fused_prefill = cb.fused_prefill(self.arch, page_size, world_size)
        self.index_scale, self.index_offset = cb.index_layout(
            page_size, world_size, rank
        )
        self.comm_layout = cb.comm_layout(
            world_size, self.heads_per_rank, max_decode_tokens
        )
        # The caller's optional prefill publish storage (the prefill exchange
        # as a peer publish; :class:`MSAIcpPrefillPublish`).
        self.prefill_publish_layout = cb.prefill_publish_layout(
            world_size, self.heads_per_rank, max_num_batched_tokens
        )
        self.decode_score_columns = cb.decode_score_columns(max_model_len, page_size)
        # The smallest chunk any step can use (the longest sequence) bounds the
        # number of query windows of one step.
        self.prefill_score_columns = prefill_score_columns(max_model_len, page_size)
        # Short prefill windows: with a pool, a window of at most
        # ``prefill_short_tokens`` query tokens runs the score rows and the
        # Top16 select in this scratch (``[prefill_short_tokens, H, pitch]``
        # float32 at the model's pitch) instead of the fused path.
        short = min(
            cb.short_prefill_max_tokens(self.arch, page_size, world_size),
            max_num_batched_tokens,
        )
        if self.prefill_score_columns >= cb.MAX_SELECT_COLUMNS:
            short = 0
        self.prefill_short_tokens = short
        self._short_scratch = (
            torch.empty(
                (short, num_index_heads, self.prefill_score_columns),
                dtype=torch.float32,
                device=device,
            )
            if short
            else None
        )
        chunk = prefill_chunk_tokens(max_model_len, page_size, num_index_heads)
        self.max_prefill_windows = (max_num_batched_tokens + chunk - 1) // chunk
        storage = cb.index_score_plan_storage(
            num_qo_heads=num_index_heads,
            total_q=min(max_num_batched_tokens, _CHUNK_MAX),
            batch=max_num_seqs,
            device=device,
        )
        self.work_ranges = torch.empty(
            (self.max_prefill_windows, storage["work_range_numel"]),
            dtype=torch.int64,
            device=device,
        )
        self.work_info = torch.empty(
            (self.max_prefill_windows, storage["work_info_numel"]),
            dtype=torch.int64,
            device=device,
        )
        self.plan_workspace = torch.empty(
            storage["workspace_bytes"], dtype=torch.uint8, device=device
        )
        # The prefill merge reads exchanged rows without waiting, so it never
        # counts a timeout; its counter is the facade's own.
        self.prefill_errors = torch.zeros(1, dtype=torch.int32, device=device)
        # Comms whose receive storage and counters check_comm validated (by
        # id, holding a reference so an id is never reused): decode validates
        # a comm once, then checks only its per-call tensors.
        self._checked_comms: dict[int, tuple[Any, tuple]] = {}
        # (comm id, slot, tagged) -> that slot's receive rows and merge
        # counters (views into the checked storage, made once).
        self._slot_views: dict[
            tuple[int, int, bool], tuple[torch.Tensor, Optional[torch.Tensor]]
        ] = {}

    # ----------------------------------------------------------------- shapes

    def decode_scores_shape(self) -> tuple[int, int, int]:
        """Decode score rows ``[max_decode_tokens, H, columns]`` (float32)."""
        return (self.max_decode_tokens, self.num_index_heads, self.decode_score_columns)

    def prefill_scratch_shape(
        self, max_seq_len: Optional[int] = None
    ) -> tuple[int, int, int]:
        """Prefill score scratch ``[chunk tokens, H, pitch]`` (float32) for a
        step whose longest sequence is ``max_seq_len`` (default: the model limit)."""
        seq = self.max_model_len if max_seq_len is None else max_seq_len
        chunk = prefill_chunk_tokens(seq, self.page_size, self.num_index_heads)
        return (
            min(chunk, self.max_num_batched_tokens),
            self.num_index_heads,
            prefill_score_columns(seq, self.page_size),
        )

    def prefill_window_tokens(self) -> int:
        """Query tokens per window of the fused prefill path (the plan
        storage's capacity, independent of the history length)."""
        return prefill_window_tokens(self.max_num_batched_tokens)

    def prefill_pool_shape(self) -> tuple[int]:
        """The fused prefill path's list pool ``[bytes]`` (uint8, 16-byte
        aligned; never zeroed) for any window of this facade's limits:
        ``prefill_window_tokens()`` tokens in up to ``max_num_seqs``
        segments, scored by one CTA per SM of the device."""
        return (
            cb.prefill_pool_bytes(
                total_q=self.prefill_window_tokens(),
                batch=self.max_num_seqs,
                device=self.device,
                num_heads=self.num_index_heads,
            ),
        )

    def prefill_carrier_shape(self) -> tuple[int, int, int, int]:
        """Send and receive carriers ``[2, H, max_num_batched_tokens, 33]`` (int32)."""
        return (2, self.num_index_heads, self.max_num_batched_tokens, cb.ROW_WORDS)

    # ----------------------------------------------------------------- setup

    def warmup(self) -> None:
        """Load (build on first use) every program of this device's
        architecture with its kernel resident in the device's context.  Run on
        every rank before the first decode / prefill exchange and before CUDA
        Graph capture."""
        cb.warmup(self.device)

    def check_comm(self, comm: Any) -> None:
        """Validate the caller's receive storage views and counters against
        :attr:`comm_layout` (attributes ``recv``, ``tagged_recv``,
        ``recv_bases``, ``tagged_recv_bases``, ``publish_generations``,
        ``expected_generations``, ``errors``)."""
        layout = self.comm_layout
        want = {
            "recv": (torch.int32, layout["legacy_shape"]),
            "tagged_recv": (torch.int64, layout["tagged_shape"]),
            "recv_bases": (torch.int64, (self.world_size,)),
            "tagged_recv_bases": (torch.int64, (self.world_size,)),
            "publish_generations": (torch.int32, layout["generations_shape"]),
            "expected_generations": (torch.int32, layout["expected_shape"]),
        }
        for field in _COMM_FIELDS:
            t = getattr(comm, field, None)
            if not isinstance(t, torch.Tensor) or t.device != self.device:
                raise ValueError(f"comm.{field} must be a tensor on {self.device}")
            if field == "errors":
                cb._errors(t)
                continue
            dtype, shape = want[field]
            if (
                t.dtype != dtype
                or tuple(t.shape) != tuple(shape)
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"comm.{field} must be contiguous {dtype} {list(shape)}"
                )
        if (
            comm.tagged_recv.data_ptr() - comm.recv.data_ptr()
            != layout["tagged_offset_bytes"]
        ):
            raise ValueError(
                "comm.tagged_recv must follow comm.recv in one receive storage"
            )
        self._remember_comm(comm)

    def _remember_comm(self, comm: Any) -> None:
        """Record a validated comm (its fields' identity and metadata) and
        drop the slot views made from an earlier version of it."""
        self._checked_comms[id(comm)] = (comm, _comm_print(comm))
        for key in [key for key in self._slot_views if key[0] == id(comm)]:
            del self._slot_views[key]

    def _ensure_comm(self, comm: Any) -> None:
        """Validate ``comm`` unless every field is still the tensor (address,
        shape, strides, device) ``check_comm`` validated; a swapped or
        re-allocated field re-runs ``check_comm`` (which raises on a bad one)."""
        entry = self._checked_comms.get(id(comm))
        if entry is None or entry[0] is not comm or not _same_comm(comm, entry[1]):
            self.check_comm(comm)

    def check_prefill_publish(self, publish: "MSAIcpPrefillPublish") -> None:
        """Validate the caller's prefill publish storage against
        :attr:`prefill_publish_layout`."""
        layout = self.prefill_publish_layout
        want = {
            "rows": (torch.int64, layout["rows_shape"]),
            "generations": (torch.int32, layout["generations_shape"]),
            "expected": (torch.int32, layout["expected_shape"]),
        }
        peers = tuple(getattr(publish, "peer_rows", ()))
        if len(peers) != self.world_size:
            raise ValueError(
                f"publish.peer_rows must hold {self.world_size} tensors (one per rank)"
            )
        checks = [(f, getattr(publish, f, None), *w) for f, w in want.items()]
        checks += [
            (f"peer_rows[{p}]", t, torch.int64, layout["rows_shape"])
            for p, t in enumerate(peers)
        ]
        for field, t, dtype, shape in checks:
            if (
                not isinstance(t, torch.Tensor)
                or t.device != self.device
                or t.dtype != dtype
                or tuple(t.shape) != tuple(shape)
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"publish.{field} must be contiguous {dtype} {list(shape)} on {self.device}"
                )

    def prefill_window_fused(
        self, tokens: int, window_plan: cb.MSAIcpScorePlan, pool: Optional[torch.Tensor]
    ) -> bool:
        """Whether :meth:`prefill_local` runs the fused path (the list scorer
        and the list merge) for a window of ``tokens`` query tokens: with a
        pool, unless the window is short and its plan's columns fit the
        model's pitch."""
        return pool is not None and (
            tokens > self.prefill_short_tokens
            or (
                (window_plan.columns + 1) // 2
                if self.page_size == 256
                else window_plan.columns
            )
            > self.prefill_score_columns
        )

    # ----------------------------------------------------------------- decode

    def decode(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_valid_pages: torch.Tensor,
        scores: torch.Tensor,
        output: torch.Tensor,
        *,
        comm: Any,
        slot: int,
        tagged: bool,
        decode_query_len: int,
        enable_pdl: bool = True,
        pool: Optional[torch.Tensor] = None,
    ) -> None:
        """One layer's decode chain: score this rank's K shard into
        ``scores[:T]``, publish each (token, head) row's local Top16 into the
        owner's ``slot`` receive rows, then merge the rows of the heads this
        rank owns into ``output`` (int32 ``[T, hpr, 16]``, ascending global
        B128 ids, ``-1`` tail).

        The default chain is the score plane and the select publish.  With
        ``pool`` (uint8 scratch of ``decode_pool_bytes`` bytes) the fused
        chain runs instead: the scorer keeps per-split Top16 lists in
        ``pool`` and the list publisher merges them (``scores`` is unused);
        its receive rows are the default chain's.  Without ``pool``, a batch
        of :attr:`fused_decode_min_batch` or more requests (compute
        capability 10.7 at P128 TP2 and TP4, from 8 requests) also runs the fused
        chain, in the facade's own pool; ``scores`` is then unused.

        With ``enable_pdl`` (the default) the default chain's score plane on
        compute capability 10.0 and 10.3 at 64 or 128 local K rows per page,
        and both chains' scorers on 10.7 at 32 or 64, read ``seq_lens`` and
        ``block_table`` and start loading the history ``index_k`` pages
        (those ending at or before ``seq_len - decode_query_len``) before
        they wait on the preceding kernel.  The kernel launched just before
        ``decode`` must not write them and must either be launched without
        programmatic dependent launch or wait on its own predecessor before
        it triggers; ``enable_pdl=False`` removes the requirement.  With
        ``tagged`` rows at world size 4 the merge also reads ``num_valid_pages``, its receive
        rows and the slot set's merge counters before it waits (it waits
        before it exits): no kernel of the step may write ``num_valid_pages``,
        and a kernel launched without programmatic dependent launch must run
        between two decodes on one slot set, so the merge that last wrote the
        counters has completed (vLLM writes ``num_valid_pages`` in its
        metadata builder and launches its decode K writer that way).  vLLM
        runs tagged rows on decode-only steps; legacy rows wait first.

        ``tagged`` selects the generation-tagged rows; it needs at least two
        sparse layers between uses of one slot set (``num_sparse_layers >= 2``
        and a rotation like vLLM's).  Legacy rows (``tagged=False``) are safe
        at any distance.  The first decode with a ``comm`` runs
        :meth:`check_comm` on it; later ones compare each comm field with the
        validated tensor (object, address, shape, strides, device) and re-run
        :meth:`check_comm` when one changed, then check the per-call tensors.
        """
        if tagged and self.num_sparse_layers < 2:
            raise ValueError(
                "tagged rows need at least two sparse layers between slot reuses"
            )
        if slot not in _SLOTS:
            raise ValueError("slot must be 0, 1 or 2")
        self._ensure_comm(comm)
        t = index_q.shape[0]
        if (
            pool is None
            and self.fused_decode_min_batch is not None
            and seq_lens.ndim == 1
            and seq_lens.shape[0] >= self.fused_decode_min_batch
        ):
            pool = self._fused_pool
        if pool is not None:
            cb.decode_score_publish(
                index_q,
                index_k,
                block_table,
                seq_lens,
                num_valid_pages,
                pool,
                comm.tagged_recv_bases if tagged else comm.recv_bases,
                comm.errors,
                rank=self.rank,
                world_size=self.world_size,
                page_size=self.page_size,
                decode_query_len=decode_query_len,
                slot=slot,
                capacity=self.max_decode_tokens,
                force_begin_blocks=self.init_blocks,
                force_end_blocks=self.local_blocks,
                generations=comm.publish_generations if tagged else None,
                enable_pdl=enable_pdl,
            )
        else:
            self._decode_plane_publish(
                index_q,
                index_k,
                block_table,
                seq_lens,
                num_valid_pages,
                scores[:t],
                comm=comm,
                slot=slot,
                tagged=tagged,
                decode_query_len=decode_query_len,
                enable_pdl=enable_pdl,
            )
        views = self._slot_views.get((id(comm), slot, tagged))
        if views is None:
            views = (
                (comm.tagged_recv if tagged else comm.recv)[slot],
                comm.expected_generations[slot] if tagged else None,
            )
            self._slot_views[(id(comm), slot, tagged)] = views
        cb.merge_candidates(
            views[0],
            num_valid_pages,
            output,
            comm.errors,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            enable_pdl=enable_pdl,
            expected_generations=views[1],
            _trusted=True,
            _poll_first=self.world_size in cb.POLL_FIRST_WORLD_SIZES,
        )

    def _decode_plane_publish(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_valid_pages: torch.Tensor,
        scores: torch.Tensor,
        *,
        comm: Any,
        slot: int,
        tagged: bool,
        decode_query_len: int,
        enable_pdl: bool,
    ) -> None:
        """The default chain's first two kernels: score plane, select publish."""
        cb.decode_index_score_sharded(
            index_q,
            index_k,
            block_table,
            seq_lens,
            scores,
            rank=self.rank,
            world_size=self.world_size,
            page_size=self.page_size,
            decode_query_len=decode_query_len,
            max_decode_query_len=self.max_decode_query_len,
            enable_pdl=enable_pdl,
        )
        cb.topk_select_publish(
            scores,
            num_valid_pages,
            comm.tagged_recv_bases if tagged else comm.recv_bases,
            comm.errors,
            slot=slot,
            rank=self.rank,
            world_size=self.world_size,
            heads_per_rank=self.heads_per_rank,
            capacity=self.max_decode_tokens,
            force_begin_blocks=self.init_blocks,
            force_end_blocks=self.local_blocks,
            enable_pdl=enable_pdl,
            generations=comm.publish_generations if tagged else None,
            index_scale=self.index_scale,
            index_offset=self.index_offset,
            _trusted=True,
        )

    # ----------------------------------------------------------------- prefill

    def prefill_window_plan(
        self,
        window: int,
        qo_segment_lens: torch.Tensor,
        kv_segment_lens: torch.Tensor,
        qo_segment_offsets: torch.Tensor,
        kv_segment_offsets: torch.Tensor,
        qo_offset: torch.Tensor,
        *,
        total_q: int,
        max_query_len: int,
        max_seq_len: int,
    ) -> cb.MSAIcpScorePlan:
        """The device plan of query window ``window`` (in the facade's plan
        storage; one plan per window, reused by every layer of the step)."""
        if not 0 <= window < self.max_prefill_windows:
            raise ValueError(f"window must be in [0, {self.max_prefill_windows})")
        return cb.index_score_sharded_plan_device(
            qo_segment_lens,
            kv_segment_lens,
            qo_segment_offsets,
            kv_segment_offsets,
            qo_offset,
            self.num_index_heads,
            total_q=total_q,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            packed_work_range=self.work_ranges[window],
            packed_work_info=self.work_info[window],
            workspace=self.plan_workspace,
        )

    def prefill_local(
        self,
        query: torch.Tensor,
        index_k: torch.Tensor,
        block_table: torch.Tensor,
        window_plan: cb.MSAIcpScorePlan,
        score_scratch: Optional[torch.Tensor],
        num_valid_pages: torch.Tensor,
        carrier_view: Optional[torch.Tensor],
        *,
        pool: Optional[torch.Tensor] = None,
        publish: Optional[dict[str, Any]] = None,
    ) -> None:
        """Score one query chunk into ``score_scratch[:T]`` and write its
        per-(head, token) Top16 rows into ``carrier_view`` (int32 ``[H, T, 33]``,
        typically ``send[:, start:end]``).

        With ``pool`` (uint8 of ``prefill_pool_shape()``) the fused path runs
        instead: the list scorer keeps each (CTA, plan item) cell's sorted
        Top16 lists in ``pool`` and the list-to-carrier merge writes the same
        carrier rows; ``score_scratch`` is unused and may be ``None``.  One
        pool serves every window, chunk and layer (the list scorer writes
        every list its merge reads, so the pool is never zeroed).  A chunk of
        at most :attr:`prefill_short_tokens` tokens (whose plan's columns fit
        the model's pitch) runs the score rows and the Top16 select in the
        facade's own scratch even with a pool: the same carrier rows, decided
        on the host from the chunk's size.

        ``publish`` (the fused path only; :meth:`MSAIcpPrefillStep` passes
        it): instead of ``carrier_view``, the rows go to the owners' receive
        rows of a :class:`MSAIcpPrefillPublish` (keys ``peer_rows``,
        ``generations``, ``slot``, ``t_base``, ``capacity`` of
        :func:`~.cake_backend.prefill_score_select`)."""
        t = query.shape[0]
        fused = self.prefill_window_fused(t, window_plan, pool)
        if publish is not None and not fused:
            raise ValueError(
                "a prefill publish needs the fused path (a pool and a long window)"
            )
        if fused:
            cb.prefill_score_select(
                query,
                index_k,
                block_table,
                window_plan,
                num_valid_pages,
                pool,
                carrier_view,
                rank=self.rank,
                tp=self.world_size,
                compact=self.page_size == 256,
                force_begin_blocks=self.init_blocks,
                force_end_blocks=self.local_blocks,
                publish=publish,
            )
            return
        if pool is not None:
            score_scratch = self._short_scratch
        if score_scratch is None:
            raise ValueError(
                "prefill_local needs score_scratch unless it is given a pool"
            )
        scores = score_scratch[:t]
        cb.index_score_sharded(
            query,
            index_k,
            block_table,
            window_plan,
            rank=self.rank,
            max_score=scores,
            initialize=False,
            tp=self.world_size,
            compact=self.page_size == 256,
        )
        cb.topk_select_prefill(
            scores,
            num_valid_pages,
            carrier_view,
            force_begin_blocks=self.init_blocks,
            force_end_blocks=self.local_blocks,
            index_scale=self.index_scale,
            index_offset=self.index_offset,
        )

    def prefill_merge(
        self,
        received: torch.Tensor,
        num_valid_pages: torch.Tensor,
        output: torch.Tensor,
        expected: Optional[torch.Tensor] = None,
    ) -> None:
        """Merge the exchanged carrier rows ``received`` (int32
        ``[W, hpr, T, 33]``) of the heads this rank owns into ``output``; with
        ``expected`` (int32 ``[hpr, capacity]``, one slot of a
        :class:`MSAIcpPrefillPublish`) ``received`` is that slot's published
        tagged rows (int64 ``[W, hpr, capacity, 32]``), waited for by
        generation."""
        cb.merge_candidates(
            received,
            num_valid_pages,
            output,
            self.prefill_errors,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            prefill=True,
            expected_generations=expected,
        )

    # ----------------------------------------------------------------- step prepare

    def prepare_prefill_step(
        self,
        windows: Sequence[tuple[int, int, torch.Tensor, cb.MSAIcpScorePlan]],
        score_scratch: Optional[torch.Tensor],
        num_valid_pages: torch.Tensor,
        send: torch.Tensor,
        received: torch.Tensor,
        output: torch.Tensor,
        *,
        debug: bool = False,
        publish: Optional["MSAIcpPrefillPublish"] = None,
        **local_kwargs: Any,
    ) -> "MSAIcpPrefillStep":
        """The prefill side of one step, prepared once for all its layers.

        ``windows`` lists ``(start, end, block_table, window_plan)`` per query
        window of the step (``window_plan`` from :meth:`prefill_window_plan`);
        ``num_valid_pages`` (``[T]``), ``send`` (``[H, T, 33]``),
        ``received`` (``[W, hpr, T, 33]``) and ``output`` (``[T, hpr, 16]``)
        cover the ``T`` prefill tokens of the step.  Per layer,
        :meth:`MSAIcpPrefillStep.local` runs what ``prefill_local(index_q[start:end],
        index_k, block_table, window_plan, score_scratch,
        num_valid_pages[start:end], send[:, start:end], **local_kwargs)``
        runs for every window, and :meth:`MSAIcpPrefillStep.merge` what
        ``prefill_merge(received, num_valid_pages, output)`` runs: the same
        launches with the same arguments, checked once per step.  See
        :class:`MSAIcpPrefillStep`.

        With ``publish`` (a :class:`MSAIcpPrefillPublish` this indexer
        validates) and a ``pool``, a step whose every window takes the fused
        path publishes its rows to the owners instead
        (:attr:`MSAIcpPrefillStep.publishes`): per layer ``local(...,
        slot=)`` and ``merge(slot=)`` and no caller exchange; any other step
        keeps ``send -> received`` (decided here, on the host)."""
        if publish is not None:
            self.check_prefill_publish(publish)
            if output.shape[0] > self.max_num_batched_tokens:
                raise ValueError(
                    "a published step holds at most max_num_batched_tokens tokens"
                )
            pool = local_kwargs.get("pool")
            if not all(
                self.prefill_window_fused(int(end) - int(start), plan, pool)
                for start, end, _, plan in windows
            ):
                publish = None
        return MSAIcpPrefillStep(
            self,
            windows,
            score_scratch,
            num_valid_pages,
            send,
            received,
            output,
            debug=debug,
            local_kwargs=local_kwargs,
            publish=publish,
        )

    def prepare_decode_step(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_valid_pages: torch.Tensor,
        scores: torch.Tensor,
        output: torch.Tensor,
        *,
        comm: Any,
        tagged: bool,
        decode_query_len: int,
        enable_pdl: bool = True,
        pool: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> "MSAIcpDecodeStep":
        """The decode side of one step, prepared once for all its layers:
        :meth:`MSAIcpDecodeStep.run` ``(index_q, index_k, slot=slot)`` runs
        what :meth:`decode` runs with these arguments, checked once per step
        and slot.  See :class:`MSAIcpDecodeStep`."""
        return MSAIcpDecodeStep(
            self,
            block_table,
            seq_lens,
            num_valid_pages,
            scores,
            output,
            comm=comm,
            tagged=tagged,
            decode_query_len=decode_query_len,
            enable_pdl=enable_pdl,
            pool=pool,
            debug=debug,
        )

    def layer_graphs(
        self, *, capture_after: int = 1, max_keys: int = 4
    ) -> "MSAIcpLayerGraphs":
        """A cache of per-layer CUDA graphs of prepared steps, replayed across
        steps whose launches are the same (see :class:`MSAIcpLayerGraphs`).
        One per rank, kept for the life of the caller's buffers."""
        return MSAIcpLayerGraphs(self, capture_after=capture_after, max_keys=max_keys)


# --------------------------------------------------------------------------- step prepare


def _signature(t: torch.Tensor) -> tuple:
    """The tensor metadata a prepared launch depends on, besides the address."""
    return (t.shape, t.stride(), t.dtype, t.device)


def _same_arg(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.data_ptr() == b.data_ptr()
            and _signature(a) == _signature(b)
        )
    return type(a) is type(b) and a == b


def _check_same_launches(
    want: Sequence[cj.RecordedLaunch], got: Sequence[cj.RecordedLaunch], what: str
) -> None:
    """``got`` (the per-call path's launches) equals ``want`` (the prepared
    launches with this layer's q / k): same programs, entries and arguments."""
    if len(want) != len(got):
        raise RuntimeError(
            f"{what}: the per-call path makes {len(got)} launches, the prepared step {len(want)}"
        )
    for i, (w, g) in enumerate(zip(want, got, strict=True)):
        if w.name != g.name or w.entry is not g.entry or w.device != g.device:
            raise RuntimeError(
                f"{what}: launch {i} is {g.name}, the prepared step has {w.name}"
            )
        for j, (kind_key, a, b) in enumerate(zip(w.plan, w.args, g.args, strict=True)):
            if not _same_arg(a, b):
                raise RuntimeError(
                    f"{what}: launch {i} ({w.name}) argument {j} {kind_key} differs "
                    "from the per-call path"
                )


class _PreparedLaunch:
    """A recorded launch whose ``q`` / ``k`` arguments a layer replaces: ``q``
    with rows ``rows`` (``None``: all) of the layer's uint8 query view, ``k``
    with the layer's uint8 K view."""

    __slots__ = ("call", "q_pos", "k_pos", "rows")

    def __init__(self, call: cj.RecordedLaunch, rows: Optional[tuple[int, int]]):
        self.call = call
        self.q_pos = call.position("q")
        self.k_pos = call.position("k")
        self.rows = rows

    def bind(self, q8: torch.Tensor, k8: torch.Tensor) -> None:
        args = self.call.args
        if self.q_pos >= 0:
            args[self.q_pos] = (
                q8 if self.rows is None else q8[self.rows[0] : self.rows[1]]
            )
        if self.k_pos >= 0:
            args[self.k_pos] = k8


class _StepBase:
    """Shared per-layer path of a prepared step.

    The first call records the launches the facade's per-call path makes (all
    of its host checks run; nothing launches while recording) and launches
    them.  Every later call checks that ``index_q`` / ``index_k`` have the
    shape, strides, dtype and device of the first call's, rebinds the two
    tensors in the recorded argument lists and launches them, with one FFI
    stream switch per call.  Every other argument of a prepared launch is the
    same object as in the first call: the step's buffers must stay allocated
    and keep their address, shape and strides for the step (the caller's
    persistent buffers do).

    ``debug=True`` runs the full per-call path for every layer instead (all
    host checks), compares its launches with the prepared ones (program,
    entry and every argument: tensor address, shape, strides, dtype and
    device, scalar value) and raises ``RuntimeError`` on a difference before
    launching anything.
    """

    def __init__(self, indexer: MSAIcpIndexerPlan, debug: bool) -> None:
        self.indexer = indexer
        self.debug = bool(debug)
        self.device_index = indexer.device.index

    def _record(self, fn: Any, *args: Any, **kwargs: Any) -> list[cj.RecordedLaunch]:
        with cj.record_launches() as calls:
            fn(*args, **kwargs)
        return calls

    def _check_layer(
        self, sig: tuple, index_q: torch.Tensor, index_k: torch.Tensor
    ) -> None:
        if (_signature(index_q), _signature(index_k)) != sig:
            raise ValueError(
                "index_q / index_k must have the shape, strides, dtype and device of "
                "the step's first layer"
            )

    def _replay(
        self,
        prepared: Sequence[_PreparedLaunch],
        index_q: torch.Tensor,
        index_k: torch.Tensor,
    ) -> list[cj.RecordedLaunch]:
        q8 = index_q.view(torch.uint8)
        k8 = index_k.view(torch.uint8)
        for p in prepared:
            p.bind(q8, k8)
        return [p.call for p in prepared]


@dataclasses.dataclass(eq=False)
class MSAIcpPrefillPublish:
    """The caller's prefill publish storage (laid out by
    :func:`~.cake_backend.prefill_publish_layout`; all zeroed once on every
    rank, never reset): ``rows`` this rank's local view of its symmetric
    receive rows (int64 ``[3, W, hpr, capacity, 32]``), ``peer_rows`` every
    rank's rows as a tensor on this device (``W`` of them in rank order: the
    symmetric allocation's peer views, entry ``rank`` this rank's own rows),
    ``generations`` the publisher's counters (int32 ``[3, H, capacity]``) and
    ``expected`` the merge's (int32 ``[3, hpr, capacity]``).  The caller keeps the symmetric
    allocation alive while any step uses it."""

    rows: torch.Tensor
    peer_rows: tuple[torch.Tensor, ...]
    generations: torch.Tensor
    expected: torch.Tensor


class MSAIcpPrefillStep(_StepBase):
    """The prefill side of one step (:meth:`MSAIcpIndexerPlan.prepare_prefill_step`).

    Per layer: :meth:`local` (score and local Top16 of every window into
    ``send``), the caller's one exchange ``send -> received``, then
    :meth:`merge`.  A step that :attr:`publishes` writes every window's rows
    into the owners' receive rows of the layer's slot instead
    (``local(..., slot=)``) and merges them (``merge(slot=)``): the layer's
    one exchange without the caller's.  The launches and their arguments are exactly those of the
    facade's ``prefill_local`` per window and ``prefill_merge``; only the host
    work differs (see :class:`_StepBase`).  CUDA Graph safe like the facade.
    """

    def __init__(
        self,
        indexer: MSAIcpIndexerPlan,
        windows: Sequence[tuple[int, int, torch.Tensor, cb.MSAIcpScorePlan]],
        score_scratch: Optional[torch.Tensor],
        num_valid_pages: torch.Tensor,
        send: torch.Tensor,
        received: torch.Tensor,
        output: torch.Tensor,
        *,
        debug: bool,
        local_kwargs: dict,
        publish: Optional[MSAIcpPrefillPublish] = None,
    ) -> None:
        super().__init__(indexer, debug)
        self.windows = tuple(
            (int(start), int(end), block_table, plan)
            for start, end, block_table, plan in windows
        )
        if not self.windows:
            raise ValueError("a prefill step needs at least one window")
        for (s0, e0, _, _), (s1, _, _, _) in zip(
            self.windows[:-1], self.windows[1:], strict=True
        ):
            if not s0 < e0 <= s1:
                raise ValueError("prefill windows must be ordered and disjoint")
        if self.windows[0][0] < 0 or self.windows[-1][1] > send.shape[1]:
            raise ValueError("prefill windows must lie inside the step's tokens")
        self.score_scratch = score_scratch
        self.num_valid_pages = num_valid_pages
        self.send = send
        self.received = received
        self.output = output
        self.local_kwargs = dict(local_kwargs)
        self.publish = publish
        # Prepared window launches and merge launches, by slot (``None``: the
        # carrier exchange, whose launches do not depend on the slot).
        self._local: dict[Optional[int], list[_PreparedLaunch]] = {}
        self._sig: tuple = ()
        self._merge: dict[Optional[int], list[cj.RecordedLaunch]] = {}

    @property
    def publishes(self) -> bool:
        """Whether the step publishes its rows (no caller exchange)."""
        return self.publish is not None

    def _slot(self, slot: Optional[int]) -> Optional[int]:
        if not self.publishes:
            return None
        if slot not in _SLOTS:
            raise ValueError(
                "a publishing prefill step needs the layer's slot (0, 1 or 2)"
            )
        return int(slot)

    def _per_call_local(
        self, index_q: torch.Tensor, index_k: torch.Tensor, slot: Optional[int]
    ):
        """(window rows, launches) of the facade's per-window prefill_local."""
        out = []
        pub = self.publish
        for start, end, block_table, plan in self.windows:
            extra = (
                {}
                if pub is None
                else {
                    "publish": {
                        "peer_rows": pub.peer_rows,
                        "generations": pub.generations,
                        "slot": slot,
                        "t_base": start,
                        "capacity": pub.rows.shape[3],
                    }
                }
            )
            calls = self._record(
                self.indexer.prefill_local,
                index_q[start:end],
                index_k,
                block_table,
                plan,
                self.score_scratch,
                self.num_valid_pages[start:end],
                None if pub is not None else self.send[:, start:end],
                **self.local_kwargs,
                **extra,
            )
            out.append(((start, end), calls))
        return out

    def _prepare_local(
        self, index_q: torch.Tensor, index_k: torch.Tensor, slot: Optional[int]
    ) -> list[_PreparedLaunch]:
        full = (0, index_q.shape[0])
        prepared = []
        for rows, calls in self._per_call_local(index_q, index_k, slot):
            for call in calls:
                prepared.append(_PreparedLaunch(call, None if rows == full else rows))
        self._local[slot] = prepared
        self._sig = (_signature(index_q), _signature(index_k))
        return prepared

    def local(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        *,
        slot: Optional[int] = None,
    ) -> None:
        """Score and select every window of this layer: ``index_q`` holds the
        step's ``T`` prefill query tokens ``[T, H, 128]``, ``index_k`` this
        layer's K shard; ``slot`` the layer's slot set (a publishing step)."""
        slot = self._slot(slot)
        prepared = self._local.get(slot)
        if prepared is None:
            if self._local:
                self._check_layer(self._sig, index_q, index_k)
            prepared = self._prepare_local(index_q, index_k, slot)
            cj.launch_recorded([p.call for p in prepared], self.device_index)
            return
        self._check_layer(self._sig, index_q, index_k)
        calls = self._replay(prepared, index_q, index_k)
        if self.debug:
            got = [
                c for _, cs in self._per_call_local(index_q, index_k, slot) for c in cs
            ]
            _check_same_launches(calls, got, "prefill step local")
        cj.launch_recorded(calls, self.device_index)

    def _merge_calls(self, slot: Optional[int]) -> list[cj.RecordedLaunch]:
        pub = self.publish
        if pub is None:
            return self._record(
                self.indexer.prefill_merge,
                self.received,
                self.num_valid_pages,
                self.output,
            )
        return self._record(
            self.indexer.prefill_merge,
            pub.rows[slot],
            self.num_valid_pages,
            self.output,
            pub.expected[slot],
        )

    def merge(self, *, slot: Optional[int] = None) -> None:
        """Merge the exchanged ``received`` rows of this layer into ``output``
        (a publishing step: the rows published into slot ``slot``)."""
        slot = self._slot(slot)
        calls = self._merge.get(slot)
        if calls is None or self.debug:
            got = self._merge_calls(slot)
            if calls is None:
                calls = self._merge[slot] = got
            else:
                _check_same_launches(calls, got, "prefill step merge")
        cj.launch_recorded(calls, self.device_index)

    def record(
        self,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        *,
        slot: Optional[int] = None,
    ) -> list[cj.RecordedLaunch]:
        """Prepare the step without launching (the first layer's host checks)
        and return its launches: every window's, then the merge's."""
        slot = self._slot(slot)
        if slot not in self._local:
            if self._local:
                self._check_layer(self._sig, index_q, index_k)
            self._prepare_local(index_q, index_k, slot)
        else:
            self._check_layer(self._sig, index_q, index_k)
        if slot not in self._merge:
            self._merge[slot] = self._merge_calls(slot)
        return [p.call for p in self._local[slot]] + list(self._merge[slot])


class MSAIcpDecodeStep(_StepBase):
    """The decode side of one step (:meth:`MSAIcpIndexerPlan.prepare_decode_step`).

    :meth:`run` per layer launches exactly what :meth:`MSAIcpIndexerPlan.decode`
    launches with the step's arguments and that layer's ``index_q``,
    ``index_k`` and ``slot``.  The first layer that uses a slot records its
    launches (all host checks); later layers of that slot only rebind
    ``index_q`` / ``index_k`` (see :class:`_StepBase`).  CUDA Graph safe like
    the facade.
    """

    def __init__(
        self,
        indexer: MSAIcpIndexerPlan,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        num_valid_pages: torch.Tensor,
        scores: torch.Tensor,
        output: torch.Tensor,
        *,
        comm: Any,
        tagged: bool,
        decode_query_len: int,
        enable_pdl: bool,
        pool: Optional[torch.Tensor],
        debug: bool,
    ) -> None:
        super().__init__(indexer, debug)
        self.args = (block_table, seq_lens, num_valid_pages, scores, output)
        self.kwargs = dict(
            comm=comm,
            tagged=tagged,
            decode_query_len=decode_query_len,
            enable_pdl=enable_pdl,
            pool=pool,
        )
        self._slots: dict[int, list[_PreparedLaunch]] = {}
        self._sig: tuple = ()

    def _per_call(self, index_q: torch.Tensor, index_k: torch.Tensor, slot: int):
        return self._record(
            self.indexer.decode, index_q, index_k, *self.args, slot=slot, **self.kwargs
        )

    def run(self, index_q: torch.Tensor, index_k: torch.Tensor, *, slot: int) -> None:
        """One layer's decode chain: ``index_q`` ``[T, H, 128]``, this layer's
        K shard ``index_k`` and receive slot set ``slot``."""
        prepared = self._slots.get(slot)
        if prepared is None:
            calls = self._per_call(index_q, index_k, slot)
            if not self._sig:
                self._sig = (_signature(index_q), _signature(index_k))
            else:
                self._check_layer(self._sig, index_q, index_k)
            self._slots[slot] = [_PreparedLaunch(c, None) for c in calls]
            cj.launch_recorded(calls, self.device_index)
            return
        self._check_layer(self._sig, index_q, index_k)
        calls = self._replay(prepared, index_q, index_k)
        if self.debug:
            _check_same_launches(
                calls, self._per_call(index_q, index_k, slot), "decode step"
            )
        cj.launch_recorded(calls, self.device_index)

    def record(
        self, index_q: torch.Tensor, index_k: torch.Tensor, *, slot: int
    ) -> list[cj.RecordedLaunch]:
        """Prepare slot ``slot`` without launching (the facade's host checks
        on its first use) and return its launches."""
        prepared = self._slots.get(slot)
        if prepared is None:
            calls = self._per_call(index_q, index_k, slot)
            if not self._sig:
                self._sig = (_signature(index_q), _signature(index_k))
            else:
                self._check_layer(self._sig, index_q, index_k)
            prepared = self._slots[slot] = [_PreparedLaunch(c, None) for c in calls]
        else:
            self._check_layer(self._sig, index_q, index_k)
        return [p.call for p in prepared]


# --------------------------------------------------------------------------- layer graphs

# Step keys whose run count MSAIcpLayerGraphs remembers (most recent first out).
_SEEN_KEYS = 64


def _arg_key(a: Any) -> Any:
    """What a recorded launch depends on in argument ``a``: a tensor's address
    and metadata, a scalar's value."""
    if isinstance(a, torch.Tensor):
        return (a.data_ptr(), tuple(a.shape), tuple(a.stride()), a.dtype)
    return (type(a).__name__, a)


def _launch_key(calls: Sequence[cj.RecordedLaunch]) -> tuple:
    """The launches ``calls`` as a hashable key, except their ``q`` and ``k``
    arguments (a layer's own, bound at capture)."""
    out = []
    for c in calls:
        skip = {c.position("q"), c.position("k")}
        out.append(
            (
                c.name,
                c.device,
                tuple(_arg_key(a) for i, a in enumerate(c.args) if i not in skip),
            )
        )
    return tuple(out)


class MSAIcpLayerGraphs:
    """Per-layer CUDA graphs of prepared steps, replayed across steps
    (:meth:`MSAIcpIndexerPlan.layer_graphs`).

    A step's layer runs the decode chain of a :class:`MSAIcpDecodeStep`, the
    windows of a :class:`MSAIcpPrefillStep`, the caller's exchange and the
    prefill merge (each part optional).  Its graph key is every argument of
    every launch of that layer (tensor address, shape, strides and dtype;
    scalar value) except the query, plus the layer's K shard, its slot, the
    query's shape and strides and the caller's ``exchange_key``.  The first
    ``capture_after`` steps of a key run eagerly; the next one captures each
    layer into its own graph (the exchange inside: still one per layer) and
    replays it; every later step of the key only replays.  A replay copies the
    layer's query into the key's staging buffer, then launches the graph:
    about 10 microseconds of host time per layer.

    Device data (plans, valid-page counts, block tables, sequence lengths, K
    contents, receive rows and their counters) may change between steps:
    the graphs read it where the launches do.  The caller's buffers must keep
    their address, shape and strides while a key's graphs live (persistent
    buffers do); a buffer that moves gives a new key.  A window plan's
    ``max_seq_len`` and the score scratch pitch are launch arguments: build
    the plans with the model limit and the scratch with
    :meth:`MSAIcpIndexerPlan.prefill_scratch_shape` to keep one key over
    steps of different lengths.  At most
    ``max_keys`` keys keep graphs, least recently used first out.  Each rank
    decides alone; every path makes the same launches and one exchange per
    layer, so ranks never disagree on the communication.
    """

    def __init__(
        self, indexer: MSAIcpIndexerPlan, *, capture_after: int, max_keys: int
    ) -> None:
        if capture_after < 0 or max_keys < 1:
            raise ValueError("need capture_after >= 0 and max_keys >= 1")
        self.indexer = indexer
        self.capture_after = int(capture_after)
        self.max_keys = int(max_keys)
        self.device_index = indexer.device.index
        # step key -> {layer key -> graph}; LRU order; how often each step key ran.
        self._graphs: collections.OrderedDict[tuple, dict[tuple, Any]] = (
            collections.OrderedDict()
        )
        # How often each recent step key ran (the last _SEEN_KEYS keys).
        self._seen: collections.OrderedDict[tuple, int] = collections.OrderedDict()
        # query (shape, strides, dtype) -> staging buffer the graphs read.
        self._staging: dict[tuple, torch.Tensor] = {}
        self._pool: Any = None
        self._stream: Optional[torch.cuda.Stream] = None
        self.stats: collections.Counter[str] = collections.Counter()

    def begin_step(
        self,
        *,
        prefill: Optional[MSAIcpPrefillStep] = None,
        decode: Optional[MSAIcpDecodeStep] = None,
        exchange: Optional[Callable[[], None]] = None,
        exchange_key: Hashable = None,
        decode_tokens: int = 0,
    ) -> "MSAIcpGraphStep":
        """One step: ``prefill`` / ``decode`` the step objects of this step
        (from this indexer), ``exchange`` the caller's one exchange per layer
        (``send -> received`` of ``prefill``; required with ``prefill``) and
        ``exchange_key`` what identifies it across steps (its group, say).
        A layer's query holds the ``decode_tokens`` decode rows first, then
        the prefill rows."""
        if prefill is None and decode is None:
            raise ValueError("a graph step needs a prefill or a decode step")
        for st in (prefill, decode):
            if st is not None and st.indexer is not self.indexer:
                raise ValueError("the step objects must come from this indexer")
        if (prefill is None) != (exchange is None):
            raise ValueError("exchange goes with a prefill step, and only with one")
        if decode is None and decode_tokens:
            raise ValueError("decode_tokens needs a decode step")
        return MSAIcpGraphStep(
            self, prefill, decode, exchange, exchange_key, int(decode_tokens)
        )

    def clear(self) -> None:
        """Drop every graph and staging buffer (after the caller's buffers move
        for good, or to free the graphs)."""
        self._graphs.clear()
        self._seen.clear()
        self._staging.clear()
        self._pool = None

    def _staging_for(self, index_q: torch.Tensor) -> torch.Tensor:
        key = (tuple(index_q.shape), tuple(index_q.stride()), index_q.dtype)
        buf = self._staging.get(key)
        if buf is None:
            buf = torch.empty_strided(
                index_q.shape,
                index_q.stride(),
                dtype=index_q.dtype,
                device=index_q.device,
            )
            self._staging[key] = buf
        return buf

    def _capture(self, body: Callable[[], None]) -> Any:
        """``body``'s launches captured on a side stream into a new graph."""
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device_index)
        if self._pool is None:
            # One private memory pool shared by the live graphs; torch frees
            # it with the last of them, so a new one follows a full eviction.
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        current = torch.cuda.current_stream(self.device_index)
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            graph.capture_begin(pool=self._pool, capture_error_mode="thread_local")
            try:
                body()
            finally:
                graph.capture_end()
        current.wait_stream(self._stream)
        return graph

    def _graphs_of(self, step_key: tuple) -> Optional[dict[tuple, Any]]:
        graphs = self._graphs.get(step_key)
        if graphs is not None:
            self._graphs.move_to_end(step_key)
        return graphs

    def _new_key(self, step_key: tuple) -> dict[tuple, Any]:
        while len(self._graphs) >= self.max_keys:
            self._graphs.popitem(last=False)
            self.stats["evicted_keys"] += 1
        if not any(self._graphs.values()):
            self._pool = None
        graphs = self._graphs[step_key] = {}
        return graphs


class MSAIcpGraphStep:
    """One step of :class:`MSAIcpLayerGraphs`
    (:meth:`MSAIcpLayerGraphs.begin_step`)."""

    def __init__(
        self,
        cache: MSAIcpLayerGraphs,
        prefill: Optional[MSAIcpPrefillStep],
        decode: Optional[MSAIcpDecodeStep],
        exchange: Optional[Callable[[], None]],
        exchange_key: Hashable,
        decode_tokens: int,
    ) -> None:
        self.cache = cache
        self.prefill = prefill
        self.decode = decode
        self.exchange = exchange
        self.exchange_key = exchange_key
        self.decode_tokens = decode_tokens
        self._step_key: Optional[tuple] = None
        self._mode: Optional[str] = None  # "eager", "capture" or "replay"

    def _parts(self, index_q: torch.Tensor):
        nd = self.decode_tokens
        q_d = index_q[:nd] if self.decode is not None else None
        q_p = index_q[nd:] if self.prefill is not None else None
        return q_d, q_p

    def _layer_body(
        self, index_q: torch.Tensor, index_k: torch.Tensor, slot: int
    ) -> None:
        q_d, q_p = self._parts(index_q)
        if self.decode is not None:
            self.decode.run(q_d, index_k, slot=slot)
        if self.prefill is not None:
            if self.prefill.publishes:
                self.prefill.local(q_p, index_k, slot=slot)
                self.prefill.merge(slot=slot)
            else:
                self.prefill.local(q_p, index_k)
                self.exchange()
                self.prefill.merge()

    def run(
        self, layer: int, index_q: torch.Tensor, index_k: torch.Tensor, *, slot: int = 0
    ) -> None:
        """Layer ``layer`` of the step: ``index_q`` ``[T, H, 128]`` (the decode
        rows, then the prefill rows), ``index_k`` the layer's K shard,
        ``slot`` its decode receive slot set."""
        cache = self.cache
        if self._step_key is None:
            # The step's launches, recorded once (no launch): the prefill
            # windows and merge, and the decode chain of every slot set.
            q_d, q_p = self._parts(index_q)
            prefill_key = (
                (
                    tuple(
                        _launch_key(self.prefill.record(q_p, index_k, slot=s))
                        for s in _SLOTS
                    )
                    if self.prefill.publishes
                    else _launch_key(self.prefill.record(q_p, index_k))
                )
                if self.prefill is not None
                else None
            )
            decode_key = (
                tuple(
                    _launch_key(self.decode.record(q_d, index_k, slot=s))
                    for s in _SLOTS
                )
                if self.decode is not None
                else None
            )
            self._step_key = (
                prefill_key,
                decode_key,
                self.exchange_key,
                self.decode_tokens,
                (tuple(index_q.shape), tuple(index_q.stride()), index_q.dtype),
            )
            graphs = cache._graphs_of(self._step_key)
            seen = cache._seen.pop(self._step_key, 0)
            cache._seen[self._step_key] = seen + 1
            while len(cache._seen) > _SEEN_KEYS:
                cache._seen.popitem(last=False)
            if graphs is not None:
                self._mode = "replay"
            elif seen >= cache.capture_after:
                cache._new_key(self._step_key)
                self._mode = "capture"
            else:
                self._mode = "eager"
        if self._mode == "eager":
            self._layer_body(index_q, index_k, slot)
            cache.stats["eager_layers"] += 1
            return
        layer_key = (layer, slot, _arg_key(index_k))
        graphs = cache._graphs.get(self._step_key)
        if graphs is None:
            # The key was evicted within the step (more keys than max_keys in flight).
            graphs = cache._new_key(self._step_key)
        staging = cache._staging_for(index_q)
        staging.copy_(index_q)
        graph = graphs.get(layer_key)
        if graph is None:
            graph = cache._capture(lambda: self._layer_body(staging, index_k, slot))
            graphs[layer_key] = graph
            cache.stats["captured_layers"] += 1
        else:
            cache.stats["replayed_layers"] += 1
        graph.replay()
