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
storage of its query windows and, on the layouts where its decode runs the
fused chain (:func:`~.cake_backend.fused_decode_min_batch`), that chain's
pool, both allocated once here.  No method allocates,
synchronizes or reads a device value; every method is CUDA Graph safe once
:meth:`MSAIcpIndexerPlan.warmup` has loaded the programs.  ``warmup`` must run
on every rank before the first exchange (a kernel loaded lazily at its first
launch can stall behind a peer's spinning merge).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

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
        it triggers; ``enable_pdl=False`` removes the requirement.

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
        :class:`MSAIcpPrefillStep`."""
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


class MSAIcpPrefillStep(_StepBase):
    """The prefill side of one step (:meth:`MSAIcpIndexerPlan.prepare_prefill_step`).

    Per layer: :meth:`local` (score and local Top16 of every window into
    ``send``), the caller's one exchange ``send -> received``, then
    :meth:`merge`.  The launches and their arguments are exactly those of the
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
        self._local: Optional[list[_PreparedLaunch]] = None
        self._sig: tuple = ()
        self._merge: Optional[list[cj.RecordedLaunch]] = None

    def _per_call_local(self, index_q: torch.Tensor, index_k: torch.Tensor):
        """(window rows, launches) of the facade's per-window prefill_local."""
        out = []
        for start, end, block_table, plan in self.windows:
            calls = self._record(
                self.indexer.prefill_local,
                index_q[start:end],
                index_k,
                block_table,
                plan,
                self.score_scratch,
                self.num_valid_pages[start:end],
                self.send[:, start:end],
                **self.local_kwargs,
            )
            out.append(((start, end), calls))
        return out

    def _prepare_local(
        self, index_q: torch.Tensor, index_k: torch.Tensor
    ) -> list[_PreparedLaunch]:
        full = (0, index_q.shape[0])
        prepared = []
        for rows, calls in self._per_call_local(index_q, index_k):
            for call in calls:
                prepared.append(_PreparedLaunch(call, None if rows == full else rows))
        self._local = prepared
        self._sig = (_signature(index_q), _signature(index_k))
        return prepared

    def local(self, index_q: torch.Tensor, index_k: torch.Tensor) -> None:
        """Score and select every window of this layer: ``index_q`` holds the
        step's ``T`` prefill query tokens ``[T, H, 128]``, ``index_k`` this
        layer's K shard."""
        if self._local is None:
            prepared = self._prepare_local(index_q, index_k)
            cj.launch_recorded([p.call for p in prepared], self.device_index)
            return
        self._check_layer(self._sig, index_q, index_k)
        calls = self._replay(self._local, index_q, index_k)
        if self.debug:
            got = [c for _, cs in self._per_call_local(index_q, index_k) for c in cs]
            _check_same_launches(calls, got, "prefill step local")
        cj.launch_recorded(calls, self.device_index)

    def merge(self) -> None:
        """Merge the exchanged ``received`` rows of this layer into ``output``."""
        calls = self._merge
        if calls is None or self.debug:
            got = self._record(
                self.indexer.prefill_merge,
                self.received,
                self.num_valid_pages,
                self.output,
            )
            if calls is None:
                calls = self._merge = got
            else:
                _check_same_launches(calls, got, "prefill step merge")
        cj.launch_recorded(calls, self.device_index)


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
