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
Top16 select into a carrier; the caller's one exchange per layer; merge).

The caller owns every buffer that crosses ranks or outlives a call: the
receive storage and its counters (``comm``, laid out by
:func:`~.cake_backend.comm_layout`), the decode score rows, the prefill score
scratch and carriers, and the outputs.  The facade owns only the prefill plan
storage of its query windows, allocated once here.  No method allocates,
synchronizes or reads a device value; every method is CUDA Graph safe once
:meth:`MSAIcpIndexerPlan.warmup` has loaded the programs.  ``warmup`` must run
on every rank before the first exchange (a kernel loaded lazily at its first
launch can stall behind a peer's spinning merge).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from . import cake_backend as cb

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
        self.index_scale, self.index_offset = cb.index_layout(
            page_size, world_size, rank
        )
        self.comm_layout = cb.comm_layout(
            world_size, self.heads_per_rank, max_decode_tokens
        )
        self.decode_score_columns = cb.decode_score_columns(max_model_len, page_size)
        # The smallest chunk any step can use (the longest sequence) bounds the
        # number of query windows of one step.
        self.prefill_score_columns = prefill_score_columns(max_model_len, page_size)
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
        its receive rows are the default chain's.

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
        score_scratch: torch.Tensor,
        num_valid_pages: torch.Tensor,
        carrier_view: torch.Tensor,
    ) -> None:
        """Score one query chunk into ``score_scratch[:T]`` and write its
        per-(head, token) Top16 rows into ``carrier_view`` (int32 ``[H, T, 33]``,
        typically ``send[:, start:end]``)."""
        t = query.shape[0]
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
    ) -> None:
        """Merge the exchanged carrier rows ``received`` (int32
        ``[W, hpr, T, 33]``) of the heads this rank owns into ``output``."""
        cb.merge_candidates(
            received,
            num_valid_pages,
            output,
            self.prefill_errors,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            prefill=True,
        )
