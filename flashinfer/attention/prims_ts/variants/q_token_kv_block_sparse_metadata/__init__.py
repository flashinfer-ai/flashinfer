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

"""CUDA-graph-safe compact QToken-KvBlock-Sparse-Attention metadata construction.

The public attention kernel consumes dense sparse-page indices and, for
grouped routes, a parallel packed-membership table. Q1 rows map their compact
selected logical blocks directly to encoded physical
subpage locators. Q2--Q8 rows additionally union the selected blocks of
every adjacent query and write per-page query-membership masks into a separate
dense table, packed four 8-bit masks per Int32 word. The public API names the
sparse block size explicitly: 4, 8, 16, 32, 64, or 128 tokens.
Patterns can be shared across KV heads or supplied independently for each head.

Construction uses one CUDA C++ CTA per route. Q1 maps selected and tail blocks
directly. Q2--Q8 union at most ``G * (topk + 1)`` selected/tail IDs using a
shared byte/bit map when its model-context footprint fits without adding CTA
waves, or a bounded radix-sort fallback. No global-cache-sized scratch is
required. On SM90 and newer, each metadata CTA releases the attention grid
through programmatic dependent launch (PDL) after its last output store; the
consumer wait covers metadata completion.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, Optional

import torch
from flashinfer.api_logging import flashinfer_experimental_api

from ...common.scales import GlobalScale, normalize_scale, resolve_scale, scale_types
from .metadata import (
    PlanConfig,
    SparsePlan,
    describe_tensor,
    flatten_fixed_groups,
    get_workspace_layout,
    get_workspace_layout_from_tensors,
    prepare_metadata_plan,
    tensor_abi_key,
    validate_attention_workspace,
    validate_metadata_inputs,
    validate_metadata_storage_page_size,
    validate_paged_kv_cache,
    validate_plan_tensor,
    validate_qo_indptr_tensor,
    validate_shape_parameters,
    validate_sparse_block_size,
)

if TYPE_CHECKING:
    from ...common.paged_kv import PagedKVScaleFactors


class QTokenKvBlockSparsePagedTSWrapper:
    """Plan QToken-KvBlock-Sparse-Attention capacity and run live routes.

    :meth:`plan` fixes query grouping, K/V capacity, dtypes, and the single
    caller-owned workspace. It must run outside CUDA Graph capture. The first
    eager :meth:`run` binds the live tensor ABI, compiles the selected PrimTS
    kernels, and initializes split-KV scratch. Later runs rebuild compact
    routes from the current indexer output and are CUDA-graph capturable.

    One wrapper revision owns mutable route and split-KV workspace. Runs must
    therefore be ordered on one stream or externally synchronized; unordered
    concurrent runs require distinct wrappers.

    Output and workspace must be disjoint from each other and from all live
    inputs, including the original indexer and request metadata. Storage
    overlap is not checked; this precondition also applies after rebinding
    inputs and across CUDA-graph replays.
    """

    def __init__(self) -> None:
        self.config: Optional[PlanConfig] = None
        self.prepared_plan: Optional[SparsePlan] = None
        self.prepared_key: Optional[tuple[object, ...]] = None
        self.default_out: Optional[torch.Tensor] = None
        self.plan_lock = threading.Lock()

    def plan(
        self,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_block_size: int,
        page_size: int,
        block_topk: int,
        max_seq_len_kv: int,
        *,
        device: torch.device | str | int,
        workspace_buffer: torch.Tensor,
        group_size: int = 1,
        packed_query: bool = False,
        split_kv: bool = True,
        share_pattern_across_kv_heads: bool = True,
        count_current_block_in_topk: bool = True,
        balanced_scheduler: Optional[bool] = None,
        mask_type: Literal["causal"] = "causal",
        q_dtype: torch.dtype = torch.float16,
        kv_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        sm_scale: Optional[GlobalScale] = None,
        v_scale: GlobalScale = 1.0,
    ) -> None:
        """Publish a capacity-only QToken-KvBlock-Sparse-Attention plan.

        ``batch_size`` is the number of query routes: ``len(qo_indptr)-1`` for
        packed Q, or ``B * Nq`` for fixed ``[B, Nq, G, Hq, D]`` Q. The planned
        ``group_size`` is the maximum packed route length or fixed group size
        ``G``, defaulting to one (no grouping), not the request sequence length.
        ``kv_block_size`` is the indexer's sparse K/V atom and currently
        supports 4/8/16/32/64/128 tokens; ``page_size`` is the physical cache
        page. Blocks may cross cache pages. The loader uses fragments of size
        gcd(kv_block_size, page_size), resolving every fragment independently.

        ``share_pattern_across_kv_heads=True`` preserves the existing shared
        top-k ``[T,K]`` input. False requires ``[T,Hkv,K]`` and prepares one
        union per group and KV head. Q, K, V and output layouts are unchanged.

        ``count_current_block_in_topk=True`` selects completed blocks and
        appends only an incomplete current block. False excludes the current
        block from the top-k budget and always appends its visible tokens,
        including the entire block when the query is its last token.

        For example, with block size 8, top-k budget 2, and query position 23,
        True selects two IDs from blocks 0/1/2 and appends nothing; IDs [0, 1]
        attend to 16 tokens. False selects blocks [0, 1] and appends complete
        current block 2 outside the budget, attending to 24 tokens. At position
        22 both modes append the seven visible tokens of partial block 2.

        ``split_kv=True`` permits automatic useful split fanout; False forces
        one KV partition and omits split scratch. Pass False for prefill and
        True for decode, independently of ``packed_query``. Workspace sizing
        must use the same value. Change it only by replanning outside capture.

        ``balanced_scheduler`` controls longest-first scheduling of grouped
        packed routes on the persistent (CLC) scheduler: the metadata kernel
        orders routes by their K/V union and SMs take the longest first, so
        long routes do not extend the last wave. ``None`` (default) enables it
        when the routes span more than one wave of SMs and can read more than
        one KV128 tile (``max_seq_len_kv`` > 128); routes of up to four tiles
        (``max_seq_len_kv`` <= 512) also need three or more waves. True
        enables it on every persistent launch; False keeps route order. Launches without the
        persistent scheduler (split-KV, single-wave static grids) ignore it.
        Routes are ranked by whole KV128 tiles of their union; routes of 127
        or more tiles (16K+ tokens) share the longest rank, so their order
        among themselves, and which of them fall into a reordered final-wave
        tail, is arbitrary.

        ``sm_scale`` and ``v_scale`` independently accept host floats or CUDA
        FP32 tensors of shape () or (1,). They establish the compiled argument kinds
        and default run values. Softmax scales are in natural units. Device
        values must remain finite and positive and may change during replay.
        """

        from ...common.tensors import resolve_cuda_device, validate_runtime_device
        from ..decode.validation import validate_head_dim, validate_head_geometry
        from ...common.attention_args import validate_positive_int

        batch_size = validate_positive_int(batch_size, "batch_size")
        group_size = validate_positive_int(group_size, "group_size")
        block_topk = validate_positive_int(block_topk, "block_topk")
        max_seq_len_kv = validate_positive_int(max_seq_len_kv, "max_seq_len_kv")
        page_size = validate_positive_int(page_size, "page_size")
        head_dim = validate_head_dim(head_dim)
        validate_head_geometry(num_qo_heads, num_kv_heads)
        kv_block_size = validate_sparse_block_size(kv_block_size)
        validate_metadata_storage_page_size(page_size)
        validate_shape_parameters(batch_size * group_size, block_topk, group_size)
        if mask_type != "causal":
            raise ValueError(
                "QToken-KvBlock-Sparse-Attention requires mask_type='causal'"
            )
        if not isinstance(packed_query, bool):
            raise TypeError("packed_query must be a bool")
        for dtype, name in (
            (q_dtype, "q_dtype"),
            (kv_dtype, "kv_dtype"),
            (out_dtype, "out_dtype"),
        ):
            if dtype is not None and not isinstance(dtype, torch.dtype):
                raise TypeError(f"{name} must be a torch.dtype")
        if kv_dtype is None:
            kv_dtype = q_dtype
        if out_dtype is None:
            out_dtype = q_dtype
        planned_device, device_index = resolve_cuda_device(device)
        validate_runtime_device(planned_device)
        sm_scale = normalize_scale(
            head_dim**-0.5 if sm_scale is None else sm_scale,
            "sm_scale",
            planned_device,
        )
        v_scale = normalize_scale(v_scale, "v_scale", planned_device)
        if not isinstance(workspace_buffer, torch.Tensor):
            raise TypeError("workspace_buffer must be a torch.Tensor")
        if workspace_buffer.device != planned_device:
            raise ValueError("workspace_buffer must be on the planned device")

        max_num_storage_pages = (max_seq_len_kv + page_size - 1) // page_size
        layout = get_workspace_layout(
            batch_size * group_size,
            block_topk,
            max_num_storage_pages,
            page_size,
            group_size,
            max_seq_len_kv=max_seq_len_kv,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            out_dtype=out_dtype,
            device=planned_device,
            num_query_groups=batch_size,
            packed_query=packed_query,
            sparse_block_size=kv_block_size,
            split_kv=split_kv,
            share_pattern_across_kv_heads=share_pattern_across_kv_heads,
            count_current_block_in_topk=count_current_block_in_topk,
        )
        validate_attention_workspace(workspace_buffer, layout.total_bytes)
        with torch.cuda.device(device_index):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "QToken-KvBlock-Sparse-Attention planning is unsupported during "
                    "CUDA Graph capture"
                )
        if balanced_scheduler is not None and not isinstance(balanced_scheduler, bool):
            raise TypeError("balanced_scheduler must be None or a bool")
        candidate = PlanConfig(
            batch_size=batch_size,
            group_size=group_size,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_block_size=kv_block_size,
            page_size=page_size,
            block_topk=block_topk,
            max_seq_len_kv=max_seq_len_kv,
            device=planned_device,
            workspace_buffer=workspace_buffer,
            packed_query=packed_query,
            split_kv=split_kv,
            share_pattern_across_kv_heads=share_pattern_across_kv_heads,
            count_current_block_in_topk=count_current_block_in_topk,
            balanced_scheduler=balanced_scheduler,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            out_dtype=out_dtype,
            sm_scale=sm_scale,
            v_scale=v_scale,
        )
        with self.plan_lock:
            self.config = candidate
            self.prepared_plan = None
            self.prepared_key = None
            self.default_out = None

    def require_config(self) -> PlanConfig:
        config = self.config
        if config is None:
            raise RuntimeError("plan() must be called before run()")
        return config

    @flashinfer_experimental_api
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        block_table: torch.Tensor,
        indexer_block_ids: torch.Tensor,
        token_to_request: Optional[torch.Tensor] = None,
        query_positions: Optional[torch.Tensor] = None,
        *,
        seq_lens: Optional[torch.Tensor] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        sm_scale: Optional[GlobalScale] = None,
        v_scale: Optional[GlobalScale] = None,
        out: Optional[torch.Tensor] = None,
        validate: bool = True,
        kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    ) -> torch.Tensor:
        """Build live routes and launch the current QToken attention plan.

        Packed NVFP4 K/V (``torch.uint8``, E2M1 with the even element in the
        low nibble) takes one of two layouts:

        - ``[pages, Hkv, page_size, D/2]`` with
          ``kv_scale_factors=(K_SF, V_SF)``, the decode
          ``PagedKVScaleFactors``: E4M3 tensors shaped
          ``[pages, Hkv, page_size, D/16]``, K token-major and V in TRT-LLM's
          four-token interleaved order. Like the caches, they may be strided
          HND views.
        - ``[pages, Hkv, page_size, D/2 + D/16]`` without
          ``kv_scale_factors``: each token row holds its payload followed by
          its own D/16 E4M3 block scales.

        Q's dtype selects the MMA: BF16 Q runs exact BF16 operands (E2M1 x E4M3
        products are exact in BF16), FP8 Q runs FP8 MMA on K/V rounded to E4M3.
        The output is BF16 either way.

        The kernel transforms NVFP4 K/V once per KV tile of every route that
        loads it. Grouped routes overlap, so long packed query batches
        transform each selected token many times (about 90 times at 4K queries
        over a 64K context). With BF16 Q, a framework can instead convert each
        request's visible NVFP4 pages to a BF16 paged cache once per call and
        run this wrapper with BF16 K/V (``kv_dtype=torch.bfloat16`` and that
        cache's block table). Results are unchanged, since E2M1 x E4M3 products
        are exact in BF16. This does not apply to FP8 Q, which rounds K/V to
        E4M3 and requires FP8 K/V otherwise. It needs no Prims-TS support; the
        trade-off is:

        - It pays only with TileQ128 groups (``group_size * Hq / Hkv > 64``,
          e.g. G32 for 4 Q heads per KV head, G16 for 8), whose transform is
          on the critical path, and with many queries. On GB300 (D256, top-k
          512 blocks of 4 tokens, 0-64K prefixes) it measured 1.02-1.37x faster
          than the best NVFP4 launch for 1K-4K packed queries (more with more
          queries and longer prefixes), 0.84-1.11x at 512, and 0.63-1.0x
          (never faster) at 128-256.
        - Its cost grows with visible tokens x KV heads, not with the queries
          (about 26 us for 64K tokens x 2 KV heads, near HBM bandwidth), and
          the BF16 copy takes about 3.6x the NVFP4 cache's memory. Overlap it
          with independent work (e.g. the indexer's top-k) and skip it for long
          contexts, where it can cost more than it saves (not measured past
          64K).
        - Converting only the blocks the indexer selected did not pay: marking
          them costs more than it saves, and long query batches select half or
          more of the visible blocks anyway.

        Packed Q is ``[total_q, Hq, D]`` with ``qo_indptr``. Fixed Q is
        ``[B, Nq, G, Hq, D]`` without ``qo_indptr``. ``indexer_block_ids`` is
        Int32 ``[total_q, block_topk]`` for shared patterns or
        ``[total_q, Hkv, block_topk]`` for independent patterns, with a
        contiguous last dimension. It never uses the BSR name ``block_indices``.
        K/V are separate HND caches shaped ``[num_pages, Hkv, page_size, D]``.

        Request-aligned fixed Q may omit both per-token metadata tensors and
        supply ``seq_lens`` instead. For each request, all ``Nq * G`` queries
        are consecutive and end at its KV length. Request IDs and positions
        are derived inside the kernels; nonpositive lengths mark inactive
        requests. Packed Q and arbitrary mappings retain explicit metadata.

        For every valid query at zero-based position ``p``, let ``bound`` be
        ``(p + int(count_current_block_in_topk)) // kv_block_size``. The first
        ``min(block_topk, bound)`` indexer entries must be distinct block IDs
        in ``[0, bound)``.
        Their order is unrestricted; later entries are ignored. In particular,
        Q1 does not compact missing entries or arbitrary ``-1`` holes inside
        this required prefix. The appended current block is derived from
        ``query_positions`` and must not also appear in the selected prefix
        when ``count_current_block_in_topk=False``.

        The KV-cache writer must zero-fill unused token slots in allocated
        pages before attention. TMA loads complete fragments, including the
        partial causal tail; score masking does not sanitize NaN/Inf padding
        in V. Attention does not clear cache padding itself.

        Run once eagerly after :meth:`plan`; capture only subsequent calls with
        stable tensor storage. A changed packed extent or retained K/V/offset
        storage triggers a new preparation outside capture. ``sm_scale`` and
        ``v_scale`` are live launch scalars, so frameworks may update Q/K and V
        dequantization scales without rebuilding the capacity plan. Each scale
        independently accepts a host float or a contiguous CUDA FP32 tensor
        of shape () or (1,) on the planned device. Run overrides must preserve the
        respective plan-time type; omitted operands inherit the plan defaults.
        Tensor values must remain finite and positive, without host readback.

        Parameters
        ----------
        q : torch.Tensor
            Packed ``[total_q, Hq, D]`` or fixed ``[B, Nq, G, Hq, D]`` query,
            with the planned device, dtype and head geometry.
            Packed queries use a TensorMap bounded by the physical tensor;
            partial final routes do not require extra tail storage.
        paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
            Separate HND K/V caches shaped
            ``[num_pages, Hkv, page_size, D]``.
        block_table : torch.Tensor
            Dense CUDA Int32 physical-page table
            ``[num_requests, max_storage_pages]``.
        indexer_block_ids : torch.Tensor
            CUDA Int32 logical block IDs ``[T, block_topk]`` when the plan
            shares patterns, otherwise ``[T, Hkv, block_topk]``. The last
            dimension must be contiguous; each row supplies the distinct
            completed-block prefix above. The causal tail is added internally.
        token_to_request : torch.Tensor, optional
            CUDA Int32 request ID for each flattened query token. Omit with
            ``query_positions`` when supplying fixed-Q ``seq_lens``.
        query_positions : torch.Tensor, optional
            CUDA Int32 or Int64 zero-based causal position for each query token.
        seq_lens : torch.Tensor, optional
            Contiguous CUDA Int32 or Int64 KV lengths, including all queries,
            shaped ``[B]`` for request-aligned fixed Q. Mutually exclusive
            with explicit per-token metadata. Active lengths must be at least
            ``Nq * G`` and at most the planned model length.
        qo_indptr : torch.Tensor, optional
            CUDA Int32 packed-route offsets. Required for packed Q and omitted
            for fixed Q; each route must stay within one request and contain
            at most the planned ``group_size`` tokens.
        sm_scale : float or torch.Tensor, optional
            Softmax scale. Omitted values inherit ``plan()``'s scale, which
            defaults to ``head_dim**-0.5``. Include Q/K dequantization scales
            here when using FP8 inputs.
        v_scale : float or torch.Tensor, optional
            Value-cache dequantization scale applied to the output,
            inheriting ``plan()``'s scale when omitted. The plan defaults to one.
        out : torch.Tensor, optional
            Caller-owned output with Q's logical shape and the planned output
            dtype. When omitted, the wrapper retains and reuses its output.
            Must not overlap any input or workspace storage; this is not checked.

        validate : bool, optional
            Validate tensor metadata and prepare changed bindings (default True).
            Set False only after a successful validated run with the same plan.
            All tensors must preserve their validated shapes, strides, devices,
            dtypes and alignment; K/V, kv_scale_factors and qo_indptr must also
            retain their storage (unchecked runs reuse the validated bindings).
            Contents may change, and Q, indexer, table, per-token metadata and out
            may use replacement storage with the same ABI. Scale overrides are
            still checked. Revalidate after changing bindings or calling plan().

        Returns
        -------
        torch.Tensor
            Attention output in Q's logical layout; returns ``out`` when given.
        """

        config = self.require_config()
        sm_scale = resolve_scale(sm_scale, config.sm_scale, "sm_scale", config.device)
        v_scale = resolve_scale(v_scale, config.v_scale, "v_scale", config.device)
        if not validate:
            if self.prepared_plan is None:
                raise RuntimeError(
                    "run with validate=True after plan() before validate=False"
                )
            if out is None:
                out = self.default_out
                if out is None:
                    raise RuntimeError(
                        "supply out or first run with validate=True and out=None"
                    )
                validate_plan_tensor(out, "out", self.prepared_plan.out)
            return self.prepared_plan.run_unchecked(
                q,
                indexer_block_ids,
                block_table,
                token_to_request,
                seq_lens if seq_lens is not None else query_positions,
                out=out,
                sm_scale=sm_scale,
                v_scale=v_scale,
            )
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        if q.device != config.device or q.dtype != config.q_dtype:
            raise ValueError("q must use the planned device and q_dtype")
        if config.packed_query:
            if q.ndim != 3 or qo_indptr is None:
                raise ValueError("packed Q requires [total_q,Hq,D] and qo_indptr")
            validate_qo_indptr_tensor(qo_indptr, expected_device=config.device)
            if qo_indptr.numel() != config.batch_size + 1:
                raise ValueError("qo_indptr length must match the planned route count")
            if q.shape[0] <= 0 or q.shape[0] > config.batch_size * config.group_size:
                raise ValueError("packed q exceeds the planned query-token capacity")
        else:
            if qo_indptr is not None:
                raise ValueError("fixed Q does not accept qo_indptr")
            if (
                q.ndim != 5
                or q.shape[0] * q.shape[1] != config.batch_size
                or q.shape[2] != config.group_size
            ):
                raise ValueError(
                    "fixed q must be [B,Nq,G,Hq,D] with B*Nq and G matching plan"
                )
        if tuple(q.shape[-2:]) != (config.num_qo_heads, config.head_dim):
            raise ValueError("q must use the planned query-head and head dimensions")
        num_query_tokens = (
            int(q.shape[0])
            if config.packed_query
            else (config.batch_size * config.group_size)
        )
        expected_ids_shape = (
            (num_query_tokens, config.block_topk)
            if config.share_pattern_across_kv_heads
            else (num_query_tokens, config.num_kv_heads, config.block_topk)
        )
        if (
            not isinstance(indexer_block_ids, torch.Tensor)
            or indexer_block_ids.shape != expected_ids_shape
        ):
            raise ValueError(f"indexer_block_ids must have shape {expected_ids_shape}")
        fixed_query_len = 0
        if seq_lens is not None:
            if (
                config.packed_query
                or token_to_request is not None
                or query_positions is not None
            ):
                raise ValueError(
                    "seq_lens requires fixed Q without explicit token metadata"
                )
            fixed_query_len = int(q.shape[1] * q.shape[2])
            query_positions = seq_lens
        for tensor, name in (
            (token_to_request, "token_to_request"),
            (query_positions, "query_positions"),
            (block_table, "block_table"),
        ):
            if name == "token_to_request" and fixed_query_len:
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if token_to_request is not None and token_to_request.shape != (
            num_query_tokens,
        ):
            raise ValueError("token_to_request must have one entry per query token")
        assert query_positions is not None
        metadata_rows = int(q.shape[0]) if fixed_query_len else num_query_tokens
        if query_positions.shape != (metadata_rows,):
            raise ValueError(
                "expected one KV length per request or one position per query"
            )

        k_cache, v_cache = validate_paged_kv_cache(paged_kv_cache)
        # Packed NVFP4 stores two E2M1 values per byte. Without separate scale
        # tensors, each row also carries the token's D/16 E4M3 block scales.
        cache_head_dim = config.head_dim
        if config.kv_dtype == torch.uint8:
            cache_head_dim = config.head_dim // 2 + (
                config.head_dim // 16 if kv_scale_factors is None else 0
            )
        if (
            k_cache.device != config.device
            or k_cache.dtype != config.kv_dtype
            or tuple(k_cache.shape[1:])
            != (config.num_kv_heads, config.page_size, cache_head_dim)
        ):
            raise ValueError("paged_kv_cache does not match the planned HND cache ABI")
        required_columns = (
            config.max_seq_len_kv + config.page_size - 1
        ) // config.page_size
        if (
            block_table.ndim != 2
            or block_table.dtype != torch.int32
            or block_table.device != config.device
            or block_table.shape[1] < required_columns
        ):
            raise ValueError(
                "block_table must be CUDA Int32 [requests, pages] with planned capacity"
            )
        if out is None:
            if self.default_out is None or self.default_out.shape != q.shape:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "run once eagerly before capture when out is not supplied"
                    )
                self.default_out = torch.empty_like(q, dtype=config.out_dtype)
            out = self.default_out
        elif not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        elif (
            out.shape != q.shape
            or out.device != config.device
            or out.dtype != config.out_dtype
        ):
            raise ValueError("out must match q shape and the planned output dtype")

        prepared_key = (
            fixed_query_len,
            scale_types(sm_scale, v_scale),
            tensor_abi_key(q),
            tensor_abi_key(indexer_block_ids),
            tensor_abi_key(block_table),
            tensor_abi_key(token_to_request),
            tensor_abi_key(query_positions),
            tensor_abi_key(out),
            tensor_abi_key(k_cache, include_storage=True),
            tensor_abi_key(v_cache, include_storage=True),
            tensor_abi_key(qo_indptr, include_storage=True),
            *(
                tensor_abi_key(scale, include_storage=True)
                for scale in (kv_scale_factors or ())
            ),
        )
        if self.prepared_plan is None or prepared_key != self.prepared_key:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "run once eagerly after changing the QToken tensor ABI"
                )
            with self.plan_lock:
                if self.prepared_plan is None or prepared_key != self.prepared_key:
                    self.prepared_plan = prepare_q_token_kv_block_sparse_attention(
                        q,
                        paged_kv_cache,
                        indexer_block_ids,
                        block_table,
                        token_to_request,
                        query_positions,
                        config.workspace_buffer,
                        out=out,
                        max_seq_len_kv=config.max_seq_len_kv,
                        bmm1_scale=config.sm_scale,
                        bmm2_scale=config.v_scale,
                        qo_indptr=qo_indptr,
                        group_size=config.group_size,
                        sparse_block_size=config.kv_block_size,
                        split_kv=config.split_kv,
                        share_pattern_across_kv_heads=config.share_pattern_across_kv_heads,
                        count_current_block_in_topk=config.count_current_block_in_topk,
                        fixed_query_len=fixed_query_len,
                        kv_scale_factors=kv_scale_factors,
                        balanced_scheduler=config.balanced_scheduler,
                    )
                    self.prepared_key = prepared_key
        assert self.prepared_plan is not None
        return self.prepared_plan.run(
            q,
            indexer_block_ids,
            block_table,
            token_to_request,
            query_positions,
            out=out,
            sm_scale=sm_scale,
            v_scale=v_scale,
        )


@flashinfer_experimental_api
def get_q_token_kv_block_sparse_workspace_size(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_topk: int,
    max_seq_len_kv: int,
    out_dtype: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    group_size: int = 1,
    kv_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
) -> int:
    """Return bytes for QToken-KvBlock-Sparse-Attention workspace storage.

    Packed queries use ``[num_query_tokens, num_qo_heads, head_dim]`` and
    Int32 ``qo_indptr`` partitions their rows into request-safe routes of
    length at most ``group_size``. Fixed queries use
    ``[batch, num_query_groups, group_size, num_qo_heads, head_dim]`` and do
    not require query-offset metadata. The first two fixed axes are flattened
    into the attention route axis without copying tensor storage.

    The dense QToken-KvBlock-Sparse-Attention page indices, grouped membership words, and compact sequence
    lengths are owned by the workspace. These persistent metadata outputs and
    the attention scratch occupy disjoint regions, and callers do not allocate
    any metadata output separately.
    The attention region contains split-KV partials, statistics, and counters
    only when the resolved policy uses split-KV; direct prefill retains only
    small uniform-ABI placeholders.
    ``kv_block_size`` supports 4/8/16/32/64/128 tokens. Shared patterns use
    one metadata row per query group; independent patterns use one per group
    and KV head. Storage-page crossings increase the number of physical
    fragments, not the number of logical candidates sorted.

    Parameters
    ----------
    q : torch.Tensor
        Packed ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied, or
        contiguous fixed ``[B, Nq, G, Hq, D]`` query otherwise.
    k_cache : torch.Tensor
        HND key cache shaped ``[num_pages, Hkv, storage_page_size, D]``. The
        matching value cache does not affect workspace sizing.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]``.
    block_topk : int
        Number of selected logical sparse blocks supplied for each query row.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It may be smaller than the reserved block-table
        capacity and must not change across replay of a prepared CUDA graph.
    out_dtype : torch.dtype, optional
        Output dtype used to size attention partials. It defaults to the query
        dtype.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative offsets for packed query routes.
    group_size : int, optional
        Maximum query tokens per group, default one (no grouping). For fixed
        Q, this must match its ``G`` dimension; packed final groups may be shorter.
    kv_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    split_kv : bool
        Allow automatic split-KV selection (default True), independently of
        query layout. Use False for prefill and True for decode. Pass the same
        value to attention planning; False omits split-KV scratch.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    count_current_block_in_topk : bool
        True (default) counts a completed current block in the selection budget.
        False excludes it from selected IDs and always appends its visible tokens.
        Match this setting in workspace sizing and planning; False can require
        a larger workspace.

    Returns
    -------
    int
        Required unified workspace size in bytes, including persistent QToken-KvBlock-Sparse-Attention
        metadata outputs and a disjoint attention-scratch region.
    """

    return get_workspace_layout_from_tensors(
        q,
        k_cache,
        block_table,
        block_topk,
        max_seq_len_kv=max_seq_len_kv,
        out_dtype=out_dtype,
        qo_indptr=qo_indptr,
        group_size=group_size,
        sparse_block_size=kv_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
    ).total_bytes


def prepare_q_token_kv_block_sparse_attention(
    query: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: Optional[torch.Tensor],
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    out: torch.Tensor,
    max_seq_len_kv: int,
    bmm1_scale: Optional[GlobalScale] = None,
    bmm2_scale: GlobalScale = 1.0,
    qo_indptr: Optional[torch.Tensor] = None,
    group_size: int = 1,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
    fixed_query_len: int = 0,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    balanced_scheduler: Optional[bool] = None,
) -> SparsePlan:
    """Prepare one graph-stable metadata-plus-attention QToken-KvBlock-Sparse-Attention launch.

    Packed Q/output use ``[num_query_tokens, Hq, D]`` and describe Q1--Q8
    routes with ``qo_indptr`` and ``group_size``. A request's final route
    may be shorter than the maximum group size. Fixed Q/output use
    ``[B, Nq, G, Hq, D]`` and omit ``qo_indptr``; ``B * Nq`` becomes the
    internal attention route count through a zero-copy view.

    Frameworks call :meth:`metadata.SparsePlan.run` with current inputs that preserve
    the prepared geometry. The plan owns its dense page-index table, grouped
    membership words, and compact sequence lengths inside ``workspace_buffer``
    and resolves metadata geometry and the PrimTS attention launch only once.
    Packed ``qo_indptr`` must be an
    Int32 device copy of CPU-validated route offsets, such as those from
    :func:`make_q_token_kv_block_sparse_qo_indptr`. Preparation checks only the
    structural tensor contract and does not materialize values on the host.

    Call ``run`` once outside CUDA graph capture to compile and initialize the
    plan, then capture with all semantic inputs, output, and workspace storage
    kept at stable addresses.

    G1 reads the raw indexer inputs directly inside attention and does not
    materialize the workspace's page-index or sequence-length views. The
    workspace size remains a conservative capacity, shared with grouped routes.

    ``sparse_block_size`` defaults to four and selects the semantic block
    specialization in {4, 8, 16, 32, 64, 128}.

    Parameters
    ----------
    query : torch.Tensor
        Packed ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied, or
        contiguous fixed ``[B, Nq, G, Hq, D]`` query otherwise.
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        Separate HND key and value caches, each shaped
        ``[num_pages, Hkv, storage_page_size, D]`` with matching shape, dtype,
        and device.
    block_indices : torch.Tensor
        CUDA Int32 logical sparse-block IDs shaped ``[T, block_topk]`` when
        shared, otherwise ``[T, Hkv, block_topk]``, with a contiguous last dimension.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]``.
    token_to_request : torch.Tensor
        Contiguous CUDA Int32 request index for each flattened query row.
    query_positions : torch.Tensor
        Contiguous CUDA Int32 or Int64 absolute position for each flattened
        query row.
    workspace_buffer : torch.Tensor
        Caller-owned contiguous CUDA byte workspace with at least the size
        returned by :func:`get_q_token_kv_block_sparse_workspace_size`. Its storage must
        remain stable and disjoint from all inputs and ``out``.
    out : torch.Tensor
        Caller-owned output with the same logical shape as ``query``.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It must fit within each dense block-table row and
        remain unchanged across CUDA graph replay.
    bmm1_scale : float or torch.Tensor, optional
        Scale applied to QK scores. It defaults to ``head_dim**-0.5``.
    bmm2_scale : float or torch.Tensor
        Scale applied to the attention output.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative route offsets selecting packed-query
        mode.
    group_size : int, optional
        Maximum query tokens per group, default one (no grouping). For fixed
        Q, this must match its ``G`` dimension; packed final groups may be shorter.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    metadata.SparsePlan
        Reusable prepared metadata-plus-attention plan. Call
        :meth:`metadata.SparsePlan.run` once before CUDA graph capture.
    """

    sparse_block_size = validate_sparse_block_size(sparse_block_size)
    k_cache, v_cache = validate_paged_kv_cache(paged_kv_cache)
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a caller-owned torch.Tensor")

    packed_query = qo_indptr is not None
    fixed_query_group_size = None
    if packed_query:
        if query.ndim != 3:
            raise ValueError(
                "packed QToken-KvBlock-Sparse-Attention query must have shape [total_q,Hq,D]"
            )
        num_query_tokens = int(query.shape[0])
        validate_qo_indptr_tensor(
            qo_indptr,
            expected_device=query.device,
        )
        attention_query = query
        attention_out = out
    elif query.ndim == 5:
        if out.shape != query.shape:
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention output must have the same shape as query"
            )
        if not query.is_contiguous() or not out.is_contiguous():
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention query and output must be contiguous"
            )
        if group_size != query.shape[2]:
            raise ValueError("fixed group_size must match query.shape[2]")
        num_query_tokens = int(query.shape[0] * query.shape[1]) * group_size
        attention_query = flatten_fixed_groups(query, group_size)
        attention_out = flatten_fixed_groups(out, group_size)
        fixed_query_group_size = group_size
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,G,Hq,D] without qo_indptr"
        )

    validate_metadata_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        int(k_cache.shape[2]),
        max_seq_len_kv,
        sparse_block_size,
        share_pattern_across_kv_heads,
        count_current_block_in_topk,
        fixed_query_len,
    )
    if not share_pattern_across_kv_heads and block_indices.shape[1] != k_cache.shape[1]:
        raise ValueError("block_indices pattern head count must match the KV cache")
    if block_indices.shape[0] != num_query_tokens:
        raise ValueError("block_indices must have one row per flattened query token")

    layout = get_workspace_layout_from_tensors(
        query,
        k_cache,
        block_table,
        int(block_indices.shape[-1]),
        max_seq_len_kv=max_seq_len_kv,
        out_dtype=out.dtype,
        qo_indptr=qo_indptr,
        group_size=group_size,
        sparse_block_size=sparse_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
    )
    views = layout.bind(workspace_buffer)
    from ..decode.plan import prepare_batch_decode_plan

    scale_qk = normalize_scale(
        query.shape[-1] ** -0.5 if bmm1_scale is None else bmm1_scale,
        "bmm1_scale",
        query.device,
    )
    scale_v = normalize_scale(bmm2_scale, "bmm2_scale", query.device)
    # Generic FMHA sees one query group as a route, so its route-local
    # sequence-length parameters receive G, not the full request length.
    attention_plan, _ = prepare_batch_decode_plan(
        attention_query,
        paged_kv_cache,
        views.attention_workspace_buffer,
        views.q_token_kv_block_sparse_page_indices,
        views.seq_lens,
        layout.max_seq_len,
        out=attention_out,
        seq_len_q=group_size,
        qo_indptr=qo_indptr,
        max_q_len=group_size if packed_query else None,
        out_dtype=out.dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=math.gcd(sparse_block_size, int(k_cache.shape[2])),
        kv_scale_factors=kv_scale_factors,
        direct_q1_sparse_block_size=sparse_block_size,
        query_major_memberships=layout.query_major_memberships,
        use_q_token_kv_block_sparse_route=True,
        use_pdl=True,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
        q_token_kv_block_sparse_page_memberships=(
            views.q_token_kv_block_sparse_page_memberships if group_size > 1 else None
        ),
        direct_q1_inputs=(
            (block_indices, block_table, token_to_request, query_positions)
            if group_size == 1
            else ()
        ),
        direct_q1_max_model_len=max_seq_len_kv if group_size == 1 else None,
        direct_q1_fixed_query_len=fixed_query_len,
        allow_head_aware_locators=group_size > 1,
        softmax_scale=scale_qk,
        output_scale=scale_v,
        balanced_scheduler=balanced_scheduler,
    )
    # Preparation only binds shapes and storage; attention does not read these
    # outputs until metadata runs. Use its resolved format for the producer.
    metadata_plan = prepare_metadata_plan(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        views.q_token_kv_block_sparse_page_indices,
        views.q_token_kv_block_sparse_page_memberships,
        views.seq_lens,
        group_size=group_size,
        storage_page_size=int(k_cache.shape[2]),
        max_seq_len_kv=max_seq_len_kv,
        sparse_block_size=sparse_block_size,
        qo_indptr=qo_indptr,
        query_major_memberships=layout.query_major_memberships,
        locator_head_multiplier=attention_plan.flat_locator_heads,
        count_current_block_in_topk=count_current_block_in_topk,
        fixed_query_len=fixed_query_len,
    )
    if attention_plan.uses_route_order:
        if views.route_order is None:
            raise AssertionError("persistent grouped routes need a route-order region")
        # When the requests' K/V fits in half of L2, every route is ordered
        # longest first. Beyond that, only the last two SM waves are reordered:
        # the others keep index order, so routes running at the same time stay
        # close in their request and keep reusing its K/V in L2.
        properties = torch.cuda.get_device_properties(query.device)
        # Bytes per token over all KV heads: K/V (with their rows' scales when
        # interleaved) plus separate NVFP4 scale tensors; pages lead each.
        kv_bytes = (
            int(block_table.shape[0])
            * max_seq_len_kv
            * sum(
                tensor[0].numel() * tensor.element_size()
                for tensor in (k_cache, v_cache, *(kv_scale_factors or ()))
            )
            // int(k_cache.shape[2])
        )
        route_order_tail = (
            0
            if 2 * kv_bytes <= properties.L2_cache_size
            else 2 * properties.multi_processor_count
        )
        # The counter is plan-owned so that no other plan sharing the workspace
        # can disturb it; it is zero between launches (the last metadata CTA
        # rearms it).
        metadata_plan = replace(
            metadata_plan,
            route_order=views.route_order,
            route_order_counter=torch.zeros(1, dtype=torch.int32, device=query.device),
            route_order_tail=route_order_tail,
        )
    return SparsePlan(
        metadata_plan=metadata_plan,
        bmm1_scale=scale_qk,
        bmm2_scale=scale_v,
        attention_plan=attention_plan,
        query=describe_tensor(query),
        block_indices=describe_tensor(block_indices),
        block_table=describe_tensor(block_table),
        token_to_request=(
            describe_tensor(token_to_request) if token_to_request is not None else None
        ),
        query_positions=describe_tensor(query_positions),
        out=describe_tensor(out),
        fixed_query_group_size=fixed_query_group_size,
    )


@flashinfer_experimental_api
def q_token_kv_block_sparse_attention_with_paged_kv_cache(
    q: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_table: torch.Tensor,
    indexer_block_ids: torch.Tensor,
    token_to_request: Optional[torch.Tensor],
    query_positions: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
    *,
    max_seq_len_kv: int,
    group_size: int = 1,
    kv_block_size: int = 4,
    mask_type: Literal["causal"] = "causal",
    sm_scale: Optional[GlobalScale] = None,
    v_scale: Optional[GlobalScale] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
    seq_lens: Optional[torch.Tensor] = None,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
) -> torch.Tensor:
    """Plan and run one QToken-KvBlock-Sparse-Attention launch.

    This eager convenience API creates a temporary
    :class:`QTokenKvBlockSparsePagedTSWrapper`, plans capacity, and launches
    once. It is not CUDA-graph capturable. Repeated launches should retain a
    wrapper, call :meth:`QTokenKvBlockSparsePagedTSWrapper.plan` outside
    capture, run once eagerly, and capture only :meth:`~QTokenKvBlockSparsePagedTSWrapper.run`.

    Packed Q is ``[total_q, Hq, D]`` with ``qo_indptr``; ``group_size``
    is its maximum route length. Fixed Q is ``[B, Nq, G, Hq, D]`` without
    ``qo_indptr``, and ``group_size`` must match ``G``. The default is one
    (no grouping) for both layouts. The indexer
    output is named ``indexer_block_ids`` to distinguish it from BSR
    ``block_indices``. ``kv_block_size`` is the semantic indexer atom
    (4/8/16/32/64/128), while the physical cache extent is inferred from
    ``paged_kv_cache``.

    Parameters
    ----------
    q : torch.Tensor
        Packed ``[total_q, Hq, D]`` or fixed ``[B, Nq, G, Hq, D]`` query.
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        Separate HND K/V cache tensors shaped
        ``[num_pages, Hkv, page_size, D]`` (packed NVFP4: last dimension
        ``D/2`` or ``D/2 + D/16``, see ``kv_scale_factors``). The cache writer must zero-fill
        unused token slots in allocated pages, as required by
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run`.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table
        ``[num_requests, max_storage_pages]``.
    indexer_block_ids : torch.Tensor
        CUDA Int32 indexer-selected logical K/V blocks ``[T, block_topk]``
        when shared, otherwise ``[T, Hkv, block_topk]``, with a contiguous last
        dimension. Each valid row provides the completed-block prefix described by
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run`; Q1 does not compact
        missing entries inside that prefix. Tail tokens are derived separately.
    token_to_request : torch.Tensor or None
        CUDA Int32 request ID for each flattened query token. Pass None
        with ``query_positions`` when using request-aligned fixed-Q lengths.
    query_positions : torch.Tensor or None
        CUDA Int32 or Int64 causal position for each query token.
    seq_lens : torch.Tensor, optional
        Per-request fixed-Q KV lengths, as described by
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run`.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace sized by
        :func:`get_q_token_kv_block_sparse_workspace_size`.
    max_seq_len_kv : int
        Static maximum logical K/V length in tokens.
    group_size : int, optional
        Maximum query tokens per group, default one (no grouping). Must match
        ``G`` for fixed Q; packed final groups may contain fewer tokens.
    kv_block_size : int
        Semantic sparse K/V block size: 4/8/16/32/64/128, default four.
    mask_type : {"causal"}
        QToken-KvBlock-Sparse-Attention currently supports causal masking only.
    sm_scale : float or torch.Tensor, optional
        Softmax scale, defaulting to ``head_dim**-0.5``.
    v_scale : float or torch.Tensor, optional
        Value-cache dequantization scale applied to the attention output,
        defaulting to one. Frameworks using an FP8 K/V cache should pass the
        cache's live V scale.
    out : torch.Tensor, optional
        Caller-owned output with the same logical shape as ``q``.
        Must be disjoint from all inputs, including indexer/request metadata,
        and from ``workspace_buffer``. Storage overlap is not checked.
    out_dtype : torch.dtype, optional
        Output dtype, defaulting to ``out.dtype`` or ``q.dtype``.
    qo_indptr : torch.Tensor, optional
        CUDA Int32 packed-route offsets.
    split_kv : bool
        Allow automatic split-KV selection (default True), independently of
        query layout. Pass False for prefill and True for decode, matching
        the value used to size ``workspace_buffer``.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.
    count_current_block_in_topk : bool
        True (default) counts a completed current block in the selection budget.
        False excludes it from selected IDs and always appends its visible tokens.
        Match this setting in workspace sizing and planning.
    kv_scale_factors : PagedKVScaleFactors, optional
        ``(K_SF, V_SF)`` block scales of a packed NVFP4 cache without
        interleaved scales, as in paged decode; see
        :meth:`QTokenKvBlockSparsePagedTSWrapper.run` for both packed NVFP4
        layouts and the MMA precision each Q dtype selects.

    Returns
    -------
    torch.Tensor
        Attention output in the same logical layout as ``q``.
    """

    k_cache, v_cache = validate_paged_kv_cache(paged_kv_cache)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    pattern_rank = 2 if share_pattern_across_kv_heads else 3
    if (
        not isinstance(indexer_block_ids, torch.Tensor)
        or indexer_block_ids.ndim != pattern_rank
    ):
        raise ValueError(
            "indexer_block_ids must be [T,K] when shared or [T,Hkv,K] otherwise"
        )
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out_dtype is None:
        out_dtype = out.dtype if out is not None else q.dtype
    if qo_indptr is None:
        if q.ndim != 5:
            raise ValueError(
                "fixed q must be [B,Nq,G,Hq,D], or supply qo_indptr for packed q"
            )
        if group_size != q.shape[2]:
            raise ValueError("fixed group_size must match q.shape[2]")
        batch_size = int(q.shape[0] * q.shape[1])
        packed_query = False
    else:
        if q.ndim != 3:
            raise ValueError("packed q must have shape [total_q,Hq,D]")
        batch_size = int(qo_indptr.numel()) - 1
        packed_query = True

    wrapper = QTokenKvBlockSparsePagedTSWrapper()
    wrapper.plan(
        batch_size,
        int(q.shape[-2]),
        int(k_cache.shape[1]),
        int(q.shape[-1]),
        kv_block_size,
        int(k_cache.shape[2]),
        int(indexer_block_ids.shape[-1]),
        max_seq_len_kv,
        device=q.device,
        workspace_buffer=workspace_buffer,
        group_size=group_size,
        packed_query=packed_query,
        mask_type=mask_type,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=out_dtype,
        sm_scale=sm_scale,
        v_scale=1.0 if v_scale is None else v_scale,
    )
    return wrapper.run(
        q,
        (k_cache, v_cache),
        block_table,
        indexer_block_ids,
        token_to_request,
        query_positions,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        sm_scale=sm_scale,
        v_scale=v_scale,
        out=out,
        kv_scale_factors=kv_scale_factors,
    )


__all__ = [
    "QTokenKvBlockSparsePagedTSWrapper",
    "get_q_token_kv_block_sparse_workspace_size",
    "q_token_kv_block_sparse_attention_with_paged_kv_cache",
]
